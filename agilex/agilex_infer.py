# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import os
import queue
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque

import h5py
import numpy as np
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import check_keyboard_input, get_config, handle_interactive_mode, process_action

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ros_operator import RosOperator, get_ros_observation

observation_window = None
observation_window_lock = threading.Lock()

shutdown_event = threading.Event()


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
    """Reset observation window for a new episode."""
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = deque(maxlen=2)

            # Append the first dummy image
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

    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose = (
        get_ros_observation(args, ros_operator)
    )

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)),
        axis=0,
    )

    eef_pose = ros_operator.build_puppet_arm_pose(
        puppet_arm_left_pose,
        puppet_arm_right_pose,
        puppet_arm_left,
        puppet_arm_right,
    )

    with observation_window_lock:
        observation_window.append(
            {
                "qpos": qpos,
                "images": {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_left,
                    config["camera_names"][2]: img_right,
                },
                "eef_pose": eef_pose,
            }
        )


def inference_fn_sync(args, config, policy, ros_operator):
    update_observation_window(args, config, ros_operator)

    start_time = time.perf_counter()

    with observation_window_lock:
        # fetch images in sequence [front, left, right]
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint":
            # state: Abs Joint 14dim
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            # state: Abs EEF 14dim
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


def _next_episode_index(dataset_dir):
    """Pick the next episode_<idx>.hdf5 index given existing files in dataset_dir."""
    if not os.path.exists(dataset_dir):
        return 0
    indices = []
    for fname in os.listdir(dataset_dir):
        if fname.startswith("episode_") and fname.endswith(".hdf5"):
            try:
                indices.append(int(fname.replace("episode_", "").replace(".hdf5", "")))
            except ValueError:
                continue
    return max(indices) + 1 if indices else 0


def _save_rollout_hdf5(
    dataset_path,
    camera_names,
    frames_by_cam,
    qpos_list,
    eef_list,
    action_list,
    meta_attrs,
):
    """Write one rollout episode in the aloha-style HDF5 layout used by collect_data_new.py."""
    data_size = len(action_list)
    if data_size == 0:
        print(f"[Rollout] No frames to save for {dataset_path}; skipping")
        return

    t0 = time.time()
    with h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        for k, v in meta_attrs.items():
            root.attrs[k] = v

        obs = root.create_group("observations")
        image_grp = obs.create_group("images")
        for cam_name in camera_names:
            frames = frames_by_cam[cam_name]
            assert len(frames) == data_size, (
                f"cam {cam_name}: {len(frames)} frames vs {data_size} actions"
            )
            h, w = frames[0].shape[:2]
            ds = image_grp.create_dataset(
                cam_name,
                (data_size, h, w, 3),
                dtype="uint8",
                chunks=(1, h, w, 3),
            )
            ds[...] = np.stack(frames).astype(np.uint8)

        obs.create_dataset("qpos", (data_size, 14))[...] = np.stack(qpos_list)
        obs.create_dataset("eef_pose", (data_size, 14))[...] = np.stack(eef_list)
        root.create_dataset("action", (data_size, 14))[...] = np.stack(action_list)

    print(f"\033[32m[Rollout] Saved {data_size} steps in {time.time() - t0:.1f}s -> {dataset_path}\033[0m")


class AsyncRolloutRecorder:
    """Off-hot-path rollout recorder.

    Main thread calls `record_step` with image *references* (no copy) plus
    small state/action arrays. A background worker drains the queue, buffers
    the episode, and writes the HDF5 file on `end_episode` without blocking
    the publish loop. `record_step` never blocks: if the queue fills up it
    drops the frame rather than stall inference.
    """

    _SHUTDOWN = object()

    def __init__(self, camera_names, rollout_dir, start_idx, queue_size=4000):
        self._camera_names = list(camera_names)
        self._rollout_dir = rollout_dir
        self._episode_idx = start_idx
        self._queue = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def record_step(self, imgs, qpos, eef, action):
        try:
            self._queue.put_nowait(("step", imgs, qpos, eef, action))
        except queue.Full:
            self._dropped += 1

    def end_episode(self, meta_attrs):
        self._queue.put(("end", meta_attrs))

    def close(self):
        self._queue.put((self._SHUTDOWN, None))
        self._worker.join()

    def _run(self):
        # Stream per-frame to HDF5 so memory stays O(1) instead of growing
        # with episode length. Accumulating GB of frame refs caused ROS
        # callback allocations / GC pauses to disturb the publish loop.
        h5_file = None
        dataset_path = None
        img_ds = None
        qpos_ds = None
        eef_ds = None
        action_ds = None
        step_count = 0
        t_open = 0.0

        def _open_episode(first_imgs):
            nonlocal h5_file, dataset_path, img_ds, qpos_ds, eef_ds, action_ds, step_count, t_open
            idx = self._episode_idx
            dataset_path = os.path.join(self._rollout_dir, f"episode_{idx}.hdf5")
            h5_file = h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2)
            h5_file.attrs["sim"] = False
            h5_file.attrs["compress"] = False
            obs = h5_file.create_group("observations")
            image_grp = obs.create_group("images")
            img_ds = {}
            for c in self._camera_names:
                h, w = first_imgs[c].shape[:2]
                img_ds[c] = image_grp.create_dataset(
                    c, shape=(0, h, w, 3), maxshape=(None, h, w, 3),
                    dtype="uint8", chunks=(1, h, w, 3),
                )
            qpos_ds = obs.create_dataset(
                "qpos", shape=(0, 14), maxshape=(None, 14),
                dtype="float32", chunks=(1, 14),
            )
            eef_ds = obs.create_dataset(
                "eef_pose", shape=(0, 14), maxshape=(None, 14),
                dtype="float32", chunks=(1, 14),
            )
            action_ds = h5_file.create_dataset(
                "action", shape=(0, 14), maxshape=(None, 14),
                dtype="float32", chunks=(1, 14),
            )
            step_count = 0
            t_open = time.time()

        def _close_episode(meta):
            nonlocal h5_file, dataset_path, img_ds, qpos_ds, eef_ds, action_ds, step_count
            if h5_file is None:
                return
            for k, v in meta.items():
                h5_file.attrs[k] = v
            h5_file.close()
            print(f"\033[32m[Rollout] Saved {step_count} steps in {time.time() - t_open:.1f}s -> {dataset_path}\033[0m")
            self._episode_idx += 1
            h5_file = None
            dataset_path = None
            img_ds = qpos_ds = eef_ds = action_ds = None
            step_count = 0

        while True:
            item = self._queue.get()
            kind = item[0]
            if kind is self._SHUTDOWN:
                if h5_file is not None:
                    h5_file.close()
                break
            if kind == "step":
                _, imgs, qpos, eef, action = item
                try:
                    if h5_file is None:
                        _open_episode(imgs)
                    n = step_count + 1
                    for c in self._camera_names:
                        img_ds[c].resize((n, *img_ds[c].shape[1:]))
                        img_ds[c][step_count] = imgs[c]
                    qpos_ds.resize((n, 14))
                    qpos_ds[step_count] = qpos
                    eef_ds.resize((n, 14))
                    eef_ds[step_count] = eef
                    action_ds.resize((n, 14))
                    action_ds[step_count] = action
                    step_count = n
                except Exception as e:
                    print(f"\033[31m[Rollout] Streaming write failed: {e}\033[0m")
                # Drop refs so GC can reclaim the image memory this iteration.
                item = None
                imgs = None
            elif kind == "end":
                _, meta = item
                try:
                    _close_episode(meta)
                except Exception as e:
                    print(f"\033[31m[Rollout] Close failed: {e}\033[0m")
                if self._dropped > 0:
                    print(f"\033[33m[Rollout] Dropped {self._dropped} frames (queue full)\033[0m")
                    self._dropped = 0


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    if args.model == "openpi":
        policy = OpenpiClient(host=args.host, port=args.port)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    left0 = config["left0"]
    right0 = config["right0"]

    ros_operator.puppet_arm_publish_continuous(left0, right0)

    print("Warmup the server...")
    policy.warmup()
    print("Server warmed up")

    instruction = config["language_instruction"]
    camera_names = list(config["camera_names"])

    recorder = None
    if args.save_rollout:
        rollout_dir = os.path.join(args.rollout_out_path, args.task)
        os.makedirs(rollout_dir, exist_ok=True)
        start_idx = _next_episode_index(rollout_dir)
        recorder = AsyncRolloutRecorder(camera_names, rollout_dir, start_idx)
        print(f"[Rollout] Async recorder writing HDF5 episodes to {rollout_dir} (starting at episode_{start_idx})")

    input("Press enter to continue")
    task_time = time.time()
    ros_operator.puppet_arm_publish_continuous(left0, right0)

    try:
        user_quit = False
        # Inference loop
        while not rospy.is_shutdown() and not user_quit:
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()
            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            recorded_steps = 0
            user_stopped = False

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                # Check for keyboard input (space to enter interactive mode)
                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        ros_operator.puppet_arm_publish_continuous(left0, right0)
                        user_stopped = True
                        break
                    elif result == "quit":
                        user_stopped = True
                        user_quit = True
                        break
                    # 'continue' just resumes the loop

                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference (also refreshes observation_window)
                    action_buffer = inference_fn_sync(args, config, policy, ros_operator)
                    assert action_buffer is not None, "Sync inference returned None"
                    assert action_buffer.shape[0] >= chunk_size, (
                        f"Action chunk length {action_buffer.shape[0]} is smaller than {chunk_size}"
                    )

                act = action_buffer[t % chunk_size]

                if recorder is not None:
                    # Cheap: grab refs under lock, copy only small state/action
                    # arrays. Images stay as refs — the worker thread owns them
                    # once ROS publishes new frames into fresh arrays.
                    with observation_window_lock:
                        cur = observation_window[-1] if observation_window else None
                        if cur is not None:
                            imgs_ref = {cam: cur["images"][cam] for cam in camera_names}
                            qpos_ref = cur["qpos"]
                            eef_ref = cur["eef_pose"]
                        else:
                            imgs_ref = None
                    if (imgs_ref is not None
                            and all(v is not None for v in imgs_ref.values())
                            and qpos_ref is not None and eef_ref is not None):
                        recorder.record_step(
                            imgs_ref,
                            np.asarray(qpos_ref, dtype=np.float32),
                            np.asarray(eef_ref, dtype=np.float32),
                            np.asarray(act, dtype=np.float32).copy(),
                        )
                        recorded_steps += 1

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.puppet_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.puppet_arm_pose_publish(left_action, right_action)

                t += 1
                print("Published Step", t)
                rate.sleep()

            if recorder is not None and recorded_steps > 0:
                outcome = "quit" if user_quit else ("stopped" if user_stopped else "done")
                meta_attrs = {
                    "task": config["task"],
                    "instruction": instruction,
                    "ctrl_type": args.ctrl_type,
                    "publish_rate": args.publish_rate,
                    "chunk_size": chunk_size,
                    "outcome": outcome,
                    "camera_names": camera_names,
                }
                recorder.end_episode(meta_attrs)

            if user_quit:
                break
            if user_stopped:
                input("Press enter to continue")
                task_time = time.time()
    finally:
        ros_operator.puppet_arm_publish_continuous(left0, right0)
        if recorder is not None:
            recorder.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--master_arm_left_topic",
        action="store",
        type=str,
        help="master_arm_left_topic",
        default="/master/joint_left",
        required=False,
    )
    parser.add_argument(
        "--master_arm_right_topic",
        action="store",
        type=str,
        help="master_arm_right_topic",
        default="/master/joint_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_topic",
        action="store",
        type=str,
        help="puppet_arm_left_topic",
        default="/puppet/joint_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_topic",
        action="store",
        type=str,
        help="puppet_arm_right_topic",
        default="/puppet/joint_right",
        required=False,
    )
    parser.add_argument(
        "--pos_cmd_left_topic",
        action="store",
        type=str,
        help="pos_cmd_left_topic",
        default="/puppet/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--pos_cmd_right_topic",
        action="store",
        type=str,
        help="pos_cmd_right_topic",
        default="/puppet/pos_cmd_right",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_left_pose_topic",
        action="store",
        type=str,
        default="/puppet/end_pose_euler_left",
        required=False,
    )
    parser.add_argument(
        "--puppet_arm_right_pose_topic",
        action="store",
        type=str,
        default="/puppet/end_pose_euler_right",
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        nargs=7,
        help="The maximum change allowed for each joint per timestep (7 values)",
        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2],
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint", "eef"],
        help="Control type for the robot arm",
        default="joint",
    )
    parser.add_argument(
        "--host",
        action="store",
        type=str,
        help="Websocket server host",
        default="0.0.0.0",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Websocket server port",
        default=8000,
        required=False,
    )
    parser.add_argument(
        "--task",
        action="store",
        type=str,
        help="Task name",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openpi"],
        help="Model to use",
        default="openpi",
        required=False,
    )
    parser.add_argument(
        "--save_rollout",
        action="store_true",
        help="Save inference rollouts as aloha-style HDF5 episodes",
        default=False,
    )
    parser.add_argument(
        "--rollout_out_path",
        type=str,
        help="Output root for rollout HDF5 files (actual dir = <rollout_out_path>/<task>)",
        default="data/agilex/rollouts_hdf5",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args, mode="inference")
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    # Set terminal to raw mode for non-blocking keyboard input
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
