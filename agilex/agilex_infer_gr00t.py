# -- coding: UTF-8
"""GR00T-N1.7 version of agilex_infer.py.

Drop-in replacement of ``agilex_infer.py`` that talks to a GR00T inference
server (started with ``gr00t/eval/run_gr00t_server.py``) instead of an openpi
WebSocket server. The GR00T server exposes a ZeroMQ REQ/REP endpoint with
msgpack-encoded payloads; we implement a minimal client inline so this script
does not need the ``gr00t`` package installed in the openpi venv (only
``pyzmq`` and ``msgpack`` are required).

Assumes the server was trained with ``examples/Agilex/agilex_config.py``:
    state  = [left_arm_joint_position(6), left_gripper(1),
              right_arm_joint_position(6), right_gripper(1)]   # 14
    action = same layout, horizon 16
"""

import argparse
import io
import os
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque
from typing import Any

import msgpack
import numpy as np
import rospy
import zmq

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agilex_utils import (
    InferenceDataRecorder,
    build_observation,
    check_keyboard_input,
    get_config,
    get_inference_observation,
    get_rollout_observation,
    handle_interactive_mode,
    process_action,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ros_operator import RosOperator

observation_window = None
observation_window_lock = threading.Lock()
shutdown_event = threading.Event()


# ---------------------------------------------------------------------------
# GR00T ZeroMQ client (minimal reimplementation of gr00t.policy.server_client)
# ---------------------------------------------------------------------------

# State/action split for the Agilex embodiment (matches agilex_config.py and
# the modality.json shipped with the training dataset).
STATE_SPLITS: tuple[tuple[str, int, int], ...] = (
    ("left_arm_joint_position", 0, 6),
    ("left_gripper", 6, 7),
    ("right_arm_joint_position", 7, 13),
    ("right_gripper", 13, 14),
)
LANGUAGE_KEY = "annotation.language.task"
VIDEO_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _encode(obj: Any):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj: Any):
    if isinstance(obj, dict) and obj.get("__ndarray_class__"):
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _pack(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode)


def _unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode)


class Gr00tClient:
    """Thin ZeroMQ client that maps agilex payloads to GR00T's nested format."""

    def __init__(self, host: str, port: int, timeout_ms: int = 30_000) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()
        self._connect()

    def _connect(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        try:
            self.socket.send(_pack(request))
            reply = self.socket.recv()
        except zmq.error.Again:
            self.socket.close(linger=0)
            self._connect()
            raise
        response = _unpack(reply)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    # ---- payload shaping ------------------------------------------------

    @staticmethod
    def _as_video_batch(img: np.ndarray) -> np.ndarray:
        """Return (1, 1, H, W, 3) uint8, matching Gr00tPolicy.check_observation."""
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr[None, None, ...]

    @staticmethod
    def _split_state(state_14: np.ndarray) -> dict[str, np.ndarray]:
        state_14 = np.asarray(state_14, dtype=np.float32).reshape(-1)
        if state_14.shape[0] != 14:
            raise ValueError(f"Expected 14-dim state, got {state_14.shape}")
        out: dict[str, np.ndarray] = {}
        for name, lo, hi in STATE_SPLITS:
            out[name] = state_14[lo:hi].astype(np.float32)[None, None, :]  # (1, 1, D)
        return out

    @staticmethod
    def _merge_action(action: dict[str, np.ndarray]) -> np.ndarray:
        """Reassemble per-key action dict into (T, 14) matching the ctrl payload."""
        parts = []
        for name, _lo, _hi in STATE_SPLITS:
            if name not in action:
                raise KeyError(f"Action response missing key '{name}'. Got {list(action)}")
            arr = np.asarray(action[name])
            if arr.ndim == 3:  # (B, T, D) — drop batch
                arr = arr[0]
            parts.append(arr.astype(np.float32))  # (T, D)
        return np.concatenate(parts, axis=-1)  # (T, 14)

    def _build_observation(self, payload) -> dict:
        obs: dict[str, Any] = {
            "video": {
                VIDEO_KEYS[0]: self._as_video_batch(payload["top"]),
                VIDEO_KEYS[1]: self._as_video_batch(payload["left"]),
                VIDEO_KEYS[2]: self._as_video_batch(payload["right"]),
            },
            "state": self._split_state(payload["state"]),
            "language": {LANGUAGE_KEY: [[payload["instruction"]]]},
        }
        return obs

    # ---- public API (mirrors OpenpiClient) ------------------------------

    def predict_action(self, payload) -> np.ndarray:
        observation = self._build_observation(payload)
        response = self._call("get_action", {"observation": observation, "options": None})
        # Server returns [action_dict, info_dict] (tuple serialised as list).
        action_dict = response[0] if isinstance(response, (list, tuple)) else response
        return self._merge_action(action_dict)

    def warmup(self) -> None:
        dummy = {
            "top": np.zeros((480, 640, 3), dtype=np.uint8),
            "left": np.zeros((480, 640, 3), dtype=np.uint8),
            "right": np.zeros((480, 640, 3), dtype=np.uint8),
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "warmup",
        }
        _ = self.predict_action(dummy)

    def reset(self) -> None:
        self._call("reset", {"options": None})


# ---------------------------------------------------------------------------
# ROS / inference loop (identical structure to agilex_infer.py)
# ---------------------------------------------------------------------------

def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def reset_observation_window():
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = deque(maxlen=2)
            observation_window.append(
                {
                    "qpos": None,
                    "images": {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
                    "eef_pose": None,
                }
            )

    observation = get_inference_observation(args, config, ros_operator)
    if observation is None:
        return False

    with observation_window_lock:
        observation_window.append(observation)
    return True


def inference_fn_sync(args, config, policy, ros_operator):
    if not update_observation_window(args, config, ros_operator):
        return None

    start_time = time.perf_counter()

    with observation_window_lock:
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint":
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            state = observation_window[-1]["eef_pose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    payload = {
        "top": image_arrs[0],
        "left": image_arrs[1],
        "right": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state,
    }

    actions = policy.predict_action(payload)
    print(f"Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms")
    return actions


def model_inference(args, config, ros_operator):
    policy = Gr00tClient(host=args.host, port=args.port)

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    left0 = config["left0"]
    right0 = config["right0"]

    ros_operator.follower_arm_publish_continuous(left0, right0)

    print("Warmup the GR00T server...")
    policy.warmup()
    print("Server warmed up")

    input("Press enter to continue")
    task_time = time.time()
    ros_operator.follower_arm_publish_continuous(left0, right0)
    recorder = InferenceDataRecorder(args, config, shutdown_event=shutdown_event)

    try:
        while not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()
            try:
                policy.reset()
            except Exception as exc:  # reset is best-effort
                print(f"[warn] policy.reset failed: {exc}")

            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            episode_closed = False

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        recorder.save_episode()
                        episode_closed = True
                        ros_operator.follower_arm_publish_continuous(left0, right0)
                        input("Press enter to continue")
                        task_time = time.time()
                        break
                    elif result == "quit":
                        recorder.save_episode()
                        return

                if t % chunk_size == 0:
                    action_buffer = inference_fn_sync(args, config, policy, ros_operator)
                    if action_buffer is None:
                        break
                    assert action_buffer is not None, "Sync inference returned None"
                    assert action_buffer.shape[0] >= chunk_size, (
                        f"Action chunk length {action_buffer.shape[0]} is smaller than {chunk_size}"
                    )

                act = action_buffer[t % chunk_size]
                observation_to_save = (
                    get_rollout_observation(args, config, ros_operator) if recorder.enabled else None
                )
                if recorder.enabled and observation_to_save is None:
                    break

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    ros_operator.follower_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    ros_operator.follower_arm_pose_publish(left_action, right_action)

                recorder.add_step(observation_to_save, action_to_save)
                t += 1
                print("Published Step", t)
                rate.sleep()

            if not episode_closed:
                recorder.save_episode()
            if shutdown_event.is_set():
                return
    finally:
        ros_operator.follower_arm_publish_continuous(left0, right0)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic", type=str, default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic", type=str, default="/camera_r/depth/image_raw")
    parser.add_argument("--leader_arm_left_topic", type=str, default="/leader/joint_left")
    parser.add_argument("--leader_arm_right_topic", type=str, default="/leader/joint_right")
    parser.add_argument("--follower_arm_left_topic", type=str, default="/follower/joint_left")
    parser.add_argument("--follower_arm_right_topic", type=str, default="/follower/joint_right")
    parser.add_argument("--pos_cmd_left_topic", type=str, default="/follower/pos_cmd_left")
    parser.add_argument("--pos_cmd_right_topic", type=str, default="/follower/pos_cmd_right")
    parser.add_argument("--follower_arm_left_pose_topic", type=str, default="/follower/end_pose_euler_left")
    parser.add_argument("--follower_arm_right_pose_topic", type=str, default="/follower/end_pose_euler_right")
    parser.add_argument("--publish_rate", type=int, default=30)
    # GR00T agilex action horizon is 16 (action delta_indices = range(16)).
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument(
        "--arm_steps_length", type=float, nargs=7,
        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2],
    )
    parser.add_argument("--use_depth_image", action="store_true", default=False)
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="/home/sail/data_rollout")
    parser.add_argument("--ctrl_type", type=str, choices=["joint", "eef"], default="joint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--task", type=str, required=True)
    # Kept for CLI compatibility with agilex_infer.py; ignored here.
    parser.add_argument("--model", type=str, default="gr00t")
    return parser.parse_args()


def main():
    args = get_arguments()
    ros_operator = RosOperator(args, mode="inference")
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
