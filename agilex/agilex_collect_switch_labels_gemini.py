"""
Agilex real-robot data collection for switch head training with Gemini labeling.

Runs the policy on the agilex robot while asynchronously querying Gemini for
dense value scores.  At each replanning boundary the rescue logic is evaluated
and the observation + binary switch_label is saved to disk.

The output .npz files are directly consumable by:
  - train_switch_head_gemini.py train  (standalone DINOv2 switch head)
  - train_switch_head_gemini.py export (inject into LeRobot dataset for pi05)

Differences from agilex_infer_with_gemini_rescue_dreamdojo.py:
  - Does NOT actually perform rescue (no DreamDojo, no Gemini selection)
  - Instead records switch_label = 1.0 where rescue WOULD have triggered
  - Saves per-replan-step .npz files with images, state, actions, switch_label

Usage:
  python agilex_collect_switch_labels_gemini.py \
      --host 10.0.0.1 --port 8000 \
      --task towel \
      --output_dir data/agilex_switch_labels
"""

import argparse
import collections
import concurrent.futures
import json
import logging
import os
import pathlib
import signal
import sys
import tempfile
import termios
import threading
import time
import tty

import imageio
import numpy as np
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from rosoperator import RosOperator, get_ros_observation
from rotation import abs_6d_2_abs_euler, quat_2_euler

from google import genai
from google.genai import types
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Constants (match agilex_infer_with_gemini_rescue_dreamdojo.py)
# ---------------------------------------------------------------------------
ROLLOUT_FPS = 10

GEMINI_QUERY_INTERVAL_FRAMES = 40   # 4s at 10fps
GEMINI_HISTORY_FRAMES = 200         # ~20s context window
GEMINI_VALUE_MODEL = "gemini-3.1-flash-lite-preview"

RESCUE_SCORE_ABSOLUTE = 0.30
RESCUE_SCORE_DROP = 0.20

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
observation_window = None
observation_window_lock = threading.Lock()

shutdown_event = threading.Event()
inference_paused = threading.Event()
inference_paused.clear()

inference_stamp = 0


# ---------------------------------------------------------------------------
# Gemini value scoring (identical to agilex_infer_with_gemini_rescue_dreamdojo)
# ---------------------------------------------------------------------------

class ValueEvaluation(BaseModel):
    reasoning: str
    score: float
    status: str


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
    client = _get_gemini_client()
    tmp_path = None
    video_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        imageio.mimwrite(tmp_path, [np.asarray(x) for x in frames], fps=ROLLOUT_FPS)

        video_file = client.files.upload(file=tmp_path)
        file_info = client.files.get(name=video_file.name)
        while file_info.state.name == "PROCESSING":
            time.sleep(2)
            file_info = client.files.get(name=video_file.name)
        if file_info.state.name == "FAILED":
            return {"step": step_idx, "error": "Video processing failed"}

        prompt = (
            f'You are a top-tier robot action evaluation expert responsible for constructing a '
            f'Dense Value Function for an RL model. '
            f'The robot is performing the task: "{task_description}". '
            f'Based on the provided video sequence (including the past history), please '
            f'evaluate the robot\'s state **over the most recent 2s** and provide a **Value Score** '
            f'between **0.00** and **1.00**.\n'
            f'IMPORTANT: Focus on the **final frames** of the video to judge the current state. '
            f'Do NOT give a high score just because the robot appeared to be on the right track earlier.\n'
            f'Rigorous Scoring Scale:\n'
            f'- 0.00 - 0.20 (Disengaged/Failure State): The robot is not in contact with the target '
            f'object, is moving in the wrong direction, has knocked the object away, or the object '
            f'has slipped out of the gripper.\n'
            f'- 0.20 - 0.40 (Approach State): The robot\'s end-effector is moving correctly toward '
            f'the target object, but has not yet made contact.\n'
            f'- 0.40 - 0.60 (Initial Interaction State): The gripper is touching or closing on the '
            f'object, but the object is NOT yet securely grasped or lifted.\n'
            f'- 0.60 - 0.80 (Critical Execution State): The object is securely grasped and being '
            f'lifted, but has not yet reached the goal height or position.\n'
            f'- 0.80 - 1.00 (Completion State): The task goal is fully achieved — for pick tasks, '
            f'the object is clearly lifted off the surface and stably held in the gripper.\n'
            f'Common failure patterns to watch for:\n'
            f'- Gripper closes but misses the object → score 0.10-0.20\n'
            f'- Object touched but not grasped (slides away) → score 0.20-0.30\n'
            f'- Object grasped but slips during lift → score 0.30-0.40\n'
            f'- Robot arm moving aimlessly or oscillating → score 0.05-0.15\n'
            f'Output strictly in **JSON array format**. Include reasoning, score (two decimal places) '
            f'and status. Example: [{{"reasoning": "...", "score": 0.35, "status": "Approach State"}}]'
        )

        response = client.models.generate_content(
            model=GEMINI_VALUE_MODEL,
            contents=[prompt, "\n[Current Video]:", video_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[ValueEvaluation],
                temperature=0.0,
            ),
        )
        result = json.loads(response.text)

        if result:
            score = result[0].get("score")
            if score is not None:
                with lock:
                    score_history.append((step_idx, float(score)))
                logging.info(
                    f"[Gemini Value] frame={step_idx} score={score:.2f} "
                    f"status={result[0].get('status')}"
                )

        return {"step": step_idx, "result": result}

    except Exception as e:
        logging.error(f"[Gemini Value] frame={step_idx} error: {e}")
        return {"step": step_idx, "error": str(e)}
    finally:
        if video_file is not None:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _check_rescue_needed(score_history: list, lock: threading.Lock) -> bool:
    with lock:
        if not score_history:
            return False
        sorted_scores = sorted(score_history, key=lambda x: x[0])

    latest_frame, latest_score = sorted_scores[-1]

    if latest_score <= RESCUE_SCORE_ABSOLUTE:
        logging.info(f"[Rescue] Would trigger: score {latest_score:.2f} <= {RESCUE_SCORE_ABSOLUTE}")
        return True

    prev_score = None
    for frame_idx, score in reversed(sorted_scores[:-1]):
        if latest_frame - frame_idx >= GEMINI_QUERY_INTERVAL_FRAMES:
            prev_score = score
            break
    if prev_score is not None and (latest_score - prev_score) <= -RESCUE_SCORE_DROP:
        logging.info(
            f"[Rescue] Would trigger: score dropped {prev_score:.2f} -> {latest_score:.2f} "
            f"(drop={prev_score - latest_score:.2f} >= {RESCUE_SCORE_DROP})"
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Signal handling
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


# ---------------------------------------------------------------------------
# Observation window management (from agilex_infer.py)
# ---------------------------------------------------------------------------

def reset_observation_window():
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = collections.deque(maxlen=2)
            observation_window.append({
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
                "endpose": None,
            })

    img_front, img_left, img_right, follower_arm_left, follower_arm_right, endpose_left, endpose_right = (
        get_ros_observation(args, ros_operator)
    )

    qpos = np.concatenate(
        (np.array(follower_arm_left.position), np.array(follower_arm_right.position)),
        axis=0,
    )

    left_pos = endpose_left.pose.position
    left_rpy = quat_2_euler(endpose_left.pose.orientation)
    left_gripper = follower_arm_left.position[-1]
    endpose_left_arr = np.array([
        left_pos.x, left_pos.y, left_pos.z,
        left_rpy[0], left_rpy[1], left_rpy[2], left_gripper,
    ])

    right_pos = endpose_right.pose.position
    right_rpy = quat_2_euler(endpose_right.pose.orientation)
    right_gripper = follower_arm_right.position[-1]
    endpose_right_arr = np.array([
        right_pos.x, right_pos.y, right_pos.z,
        right_rpy[0], right_rpy[1], right_rpy[2], right_gripper,
    ])

    endpose = np.concatenate((endpose_left_arr, endpose_right_arr), axis=0)

    with observation_window_lock:
        observation_window.append({
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_right,
                config["camera_names"][2]: img_left,
            },
            "endpose": endpose,
        })


# ---------------------------------------------------------------------------
# StreamActionBuffer (from agilex_infer_with_gemini_rescue_dreamdojo.py)
# ---------------------------------------------------------------------------

class StreamActionBuffer:

    def __init__(self, delay, exec_horizon, state_dim):
        self.delay = delay
        self.exec_horizon = exec_horizon
        self.lock = threading.Lock()

        self.cur_chunk = np.zeros((exec_horizon, state_dim))
        self.next_chunk = np.zeros((exec_horizon, state_dim))
        self.cur_len = 0
        self.next_len = 0
        self.cur_step = 0
        self.cur_stamp = 0

    def reset(self):
        with self.lock:
            self.cur_len = 0
            self.next_len = 0
            self.cur_step = 0
            self.cur_stamp = 0

    def should_launch_inference(self):
        if self.exec_horizon > self.delay:
            return self.cur_step == (self.exec_horizon - self.delay - 1)
        else:
            return self.cur_step == 0

    def integrate_first_chunk(self, actions_chunk: np.ndarray):
        with self.lock:
            assert self.cur_len == 0
            assert self.cur_stamp == 0
            assert actions_chunk.shape[0] == self.exec_horizon
            self.cur_chunk[:actions_chunk.shape[0]] = actions_chunk
            self.cur_len = self.exec_horizon

    def integrate_new_chunk(self, actions_chunk: np.ndarray):
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or empty when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            assert L == self.exec_horizon
            if self.cur_len == 0:
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                self.cur_chunk[:L] = actions_chunk
                self.cur_len = L
            else:
                self.next_chunk[:L] = actions_chunk
                self.next_len = L

    def integrate_new_chunk_streaming(self, actions_chunk: np.ndarray, stamp: int):
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or empty when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            if self.cur_len == 0:
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                safe_L = min(L, self.exec_horizon)
                self.cur_chunk[:safe_L] = actions_chunk[:safe_L]
                self.cur_len = safe_L
            else:
                if self.cur_stamp == stamp:
                    remaining = self.exec_horizon - self.cur_len
                    if remaining > 0:
                        safe_L = min(L, remaining)
                        self.cur_chunk[self.cur_len:self.cur_len + safe_L] = actions_chunk[:safe_L]
                        self.cur_len += safe_L
                else:
                    remaining = self.exec_horizon - self.next_len
                    if remaining > 0:
                        safe_L = min(L, remaining)
                        self.next_chunk[self.next_len:self.next_len + safe_L] = actions_chunk[:safe_L]
                        self.next_len += safe_L

    def get_next_action(self):
        with self.lock:
            if self.cur_step >= self.cur_len:
                return None
            action = self.cur_chunk[self.cur_step]
            self.cur_step += 1
            if self.cur_step == self.exec_horizon:
                self.cur_chunk, self.next_chunk = self.next_chunk, self.cur_chunk
                self.cur_len = self.next_len
                self.next_len = 0
                self.cur_step = 0
                self.cur_stamp += 1
            return action


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def inference_fn_sync(args, config, policy, ros_operator):
    global inference_stamp

    update_observation_window(args, config, ros_operator)
    start_time = time.perf_counter()

    with observation_window_lock:
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]
        if args.ctrl_type in ("joint", "ee6d"):
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            state = observation_window[-1]["endpose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    payload = {
        "top": image_arrs[0], "right": image_arrs[1], "left": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state, "action_prefix": None, "delay": None,
    }

    if args.streaming:
        actions = policy.predict_action_streaming(payload)
    else:
        actions = policy.predict_action(payload)
    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Sync   {inference_stamp:2d}] Model inference: {elapsed:.1f}ms")
    inference_stamp += 1
    return actions


def inference_fn_async(args, config, policy, ros_operator, action_buffer):
    global inference_stamp

    while not rospy.is_shutdown():
        try:
            inference_paused.wait()
            print(f"[Async  {inference_stamp:2d}] Start inference")

            d = config["delay"]
            s = config["exec_horizon"]

            with action_buffer.lock:
                if action_buffer.cur_chunk is None or config["mode"] == "naive":
                    action_prefix = None
                    if config["mode"] == "rtc":
                        rospy.logwarn("RTC mode: action_prefix is None")
                else:
                    action_prefix = action_buffer.cur_chunk[(s - d):s].copy()
                    assert action_prefix.shape[0] == d

            update_observation_window(args, config, ros_operator)
            start_time = time.perf_counter()

            with observation_window_lock:
                image_arrs = [
                    observation_window[-1]["images"][config["camera_names"][0]],
                    observation_window[-1]["images"][config["camera_names"][1]],
                    observation_window[-1]["images"][config["camera_names"][2]],
                ]
                if args.ctrl_type in ("joint", "ee6d"):
                    state = observation_window[-1]["qpos"]
                elif args.ctrl_type == "eef":
                    state = observation_window[-1]["endpose"]
                else:
                    raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

            payload = {
                "top": image_arrs[0], "right": image_arrs[1], "left": image_arrs[2],
                "instruction": config["language_instruction"],
                "state": state, "action_prefix": action_prefix, "delay": np.array(d),
            }

            from functools import partial
            on_actions_ready = partial(
                action_buffer.integrate_new_chunk_streaming, stamp=inference_stamp
            )

            if args.streaming:
                policy.predict_action_streaming(payload, on_actions_ready=on_actions_ready)
            else:
                actions = policy.predict_action(payload)
                if actions is not None and len(actions) > 0:
                    action_buffer.integrate_new_chunk(actions[d:s + d])
                else:
                    print("actions is None or len(actions) == 0")

            elapsed = (time.perf_counter() - start_time) * 1000
            print(f"[Async  {inference_stamp:2d}] Inference: {elapsed:.1f}ms")
            inference_stamp += 1
            inference_paused.clear()

        except Exception as e:
            rospy.logwarn(f"[inference_fn_async] {e}")
            time.sleep(0.1)


def start_inference_thread(args, config, policy, ros_operator, action_buffer):
    t = threading.Thread(
        target=inference_fn_async,
        args=(args, config, policy, ros_operator, action_buffer),
    )
    t.daemon = True
    t.start()


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def model_inference(args, config, ros_operator):
    global inference_stamp

    policy = OpenpiClient(host=args.host, port=args.port)

    max_publish_step = config["episode_len"]
    left0 = config["left0"]
    right0 = config["right0"]
    task_description = config["language_instruction"]
    print(config)

    ros_operator.follower_arm_publish_continuous(left0, right0)
    print("Warmup the server...")
    policy.warmup(rtc=(args.mode == "rtc"), streaming=args.streaming)
    print("Server warmed up")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_buffer = StreamActionBuffer(
        delay=config["delay"],
        exec_horizon=config["exec_horizon"],
        state_dim=config["state_dim"],
    )

    start_inference_thread(args, config, policy, ros_operator, action_buffer)

    # Track global sample index across episodes (resume if existing samples found)
    import glob as _glob
    existing = _glob.glob(str(output_dir / "sample_*.npz"))
    global_sample_idx = len(existing)
    if global_sample_idx > 0:
        logging.info(f"Resuming from sample index {global_sample_idx}")

    all_episode_meta = []

    try:
        episode_idx = 0
        while not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()
            action_buffer.reset()
            inference_paused.clear()
            inference_stamp = 0

            input("Press enter to start episode")
            task_time = time.time()
            ros_operator.follower_arm_publish_continuous(left0, right0)

            # Gemini value tracking
            score_history: list = []
            score_lock = threading.Lock()
            gemini_futures: list = []
            gemini_all_results: list = []
            gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

            # Collected frames for Gemini video and final recording
            collected_frames: list = []       # front camera (for Gemini)
            collected_right: list = []        # right camera history (for video clips)
            collected_left: list = []         # left camera history (for video clips)
            frame_counter = 0

            # Per-replan-step records
            replan_records: list = []

            task_segment = task_description.replace(" ", "_")
            rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_running"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            # First synchronous inference
            actions = inference_fn_sync(args, config, policy, ros_operator)
            action_buffer.integrate_first_chunk(actions[:config["exec_horizon"]])

            last_valid_act = None
            user_stopped = False

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                print(
                    f"[Step {t:4d}] cur_step={action_buffer.cur_step:3d} | "
                    f"cur_chunk={action_buffer.cur_len:3d} | next_chunk={action_buffer.next_len:3d} | "
                    f"cur_stamp={action_buffer.cur_stamp:3d}"
                )

                # Keyboard handling
                key = check_keyboard_input()
                if key == " ":
                    inference_paused.clear()
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        ros_operator.follower_arm_publish_continuous(left0, right0)
                        user_stopped = True
                        break
                    elif result == "quit":
                        user_stopped = True
                        gemini_executor.shutdown(wait=False)
                        return

                # Capture current frame (all cameras)
                with observation_window_lock:
                    if observation_window is not None and len(observation_window) > 0:
                        imgs = observation_window[-1]["images"]
                        front_img = imgs.get(config["camera_names"][0])
                        right_img = imgs.get(config["camera_names"][1])
                        left_img = imgs.get(config["camera_names"][2])
                        if front_img is not None:
                            collected_frames.append(np.asarray(front_img).copy())
                            collected_right.append(
                                np.asarray(right_img).copy() if right_img is not None
                                else np.zeros_like(np.asarray(front_img))
                            )
                            collected_left.append(
                                np.asarray(left_img).copy() if left_img is not None
                                else np.zeros_like(np.asarray(front_img))
                            )
                            frame_counter += 1

                # Periodic Gemini value query
                if frame_counter > 0 and frame_counter % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                    clip = list(collected_frames[-GEMINI_HISTORY_FRAMES:])
                    future = gemini_executor.submit(
                        _query_gemini_value,
                        clip, task_description, frame_counter,
                        score_history, score_lock,
                    )
                    gemini_futures.append(future)
                    logging.info(f"[Gemini] Submitted value query at frame {frame_counter}")

                # ---- Record at replan boundaries ----
                if action_buffer.cur_step == 0 and frame_counter > 0:
                    rescue = _check_rescue_needed(score_history, score_lock)

                    # Snapshot current observation for saving
                    with observation_window_lock:
                        image_arrs = [
                            observation_window[-1]["images"][config["camera_names"][0]],
                            observation_window[-1]["images"][config["camera_names"][1]],
                            observation_window[-1]["images"][config["camera_names"][2]],
                        ]
                        if args.ctrl_type in ("joint", "ee6d"):
                            state = observation_window[-1]["qpos"]
                        elif args.ctrl_type == "eef":
                            state = observation_window[-1]["endpose"]
                        else:
                            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

                    # Current action chunk from the buffer
                    with action_buffer.lock:
                        action_chunk = action_buffer.cur_chunk[:action_buffer.cur_len].copy()

                    # Extract video clips of recent clip_len frames per camera
                    clip_len = args.clip_len
                    top_clip = list(collected_frames[-clip_len:])
                    right_clip = list(collected_right[-clip_len:])
                    left_clip = list(collected_left[-clip_len:])
                    if len(top_clip) < clip_len:
                        pad_n = clip_len - len(top_clip)
                        top_clip = [top_clip[0]] * pad_n + top_clip
                        right_clip = [right_clip[0]] * pad_n + right_clip
                        left_clip = [left_clip[0]] * pad_n + left_clip

                    replan_records.append({
                        "frame_idx": frame_counter,
                        # Single-frame (for image-input model)
                        "top": np.asarray(image_arrs[0]).copy() if image_arrs[0] is not None else None,
                        "right": np.asarray(image_arrs[1]).copy() if image_arrs[1] is not None else None,
                        "left": np.asarray(image_arrs[2]).copy() if image_arrs[2] is not None else None,
                        # Video clips (for video-input model)
                        "top_clip": np.stack(top_clip),       # (T, H, W, 3) uint8
                        "right_clip": np.stack(right_clip),   # (T, H, W, 3) uint8
                        "left_clip": np.stack(left_clip),     # (T, H, W, 3) uint8
                        "state": np.array(state, dtype=np.float32).copy() if state is not None else None,
                        "actions": np.array(action_chunk, dtype=np.float32),
                        "rescue": rescue,
                    })

                    if rescue:
                        logging.info(
                            f"[Collect] frame={frame_counter} switch_label=1.0 (rescue would trigger)"
                        )

                # Normal action execution (NO rescue — just keep running the policy)
                if action_buffer.should_launch_inference() and not inference_paused.is_set():
                    inference_paused.set()
                    time.sleep(0.001)

                act = action_buffer.get_next_action()

                if act is None:
                    rospy.logwarn(f"[Step {t:4d}] act is None")
                    if last_valid_act is not None:
                        act = last_valid_act
                    else:
                        rate.sleep()
                        continue

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.follower_arm_publish(left_action, right_action)
                elif args.ctrl_type == "ee6d":
                    act = abs_6d_2_abs_euler(act)
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.endpose_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.endpose_publish(left_action, right_action)

                if args.use_robot_base:
                    vel_action = act[14:16]
                    ros_operator.robot_base_publish(vel_action)

                t += 1
                last_valid_act = act
                rate.sleep()

            # ---- Episode finished: wait for Gemini queries ----
            gemini_executor.shutdown(wait=True)
            for future in gemini_futures:
                try:
                    gemini_all_results.append(future.result())
                except Exception as e:
                    gemini_all_results.append({"error": str(e)})

            # ---- Post-process: re-label with complete score history ----
            sorted_scores = sorted(score_history, key=lambda x: x[0])

            for rec in replan_records:
                frame = rec["frame_idx"]
                scores_up_to = [(f, s) for f, s in sorted_scores if f <= frame]
                if not scores_up_to:
                    rec["switch_label"] = 0.0
                    continue

                latest_f, latest_s = scores_up_to[-1]
                should_rescue = False

                if latest_s <= RESCUE_SCORE_ABSOLUTE:
                    should_rescue = True

                if not should_rescue:
                    prev_s = None
                    for f, s in reversed(scores_up_to[:-1]):
                        if latest_f - f >= GEMINI_QUERY_INTERVAL_FRAMES:
                            prev_s = s
                            break
                    if prev_s is not None and (latest_s - prev_s) <= -RESCUE_SCORE_DROP:
                        should_rescue = True

                rec["switch_label"] = 1.0 if should_rescue else 0.0

            # ---- Save per-step .npz files ----
            n_rescue = sum(1 for r in replan_records if r.get("switch_label", 0) > 0.5)
            n_normal = len(replan_records) - n_rescue

            for rec in replan_records:
                if rec["top"] is None or rec["state"] is None:
                    continue

                save_path = output_dir / f"sample_{global_sample_idx:07d}.npz"
                save_dict = {
                    # Single-frame images (for image-input model)
                    "top": rec["top"],                                  # (H, W, 3) uint8
                    "right": rec["right"],                              # (H, W, 3) uint8
                    "left": rec["left"],                                # (H, W, 3) uint8
                    # Video clips (for video-input model)
                    "top_clip": rec["top_clip"],                        # (T, H, W, 3) uint8
                    "right_clip": rec["right_clip"],                    # (T, H, W, 3) uint8
                    "left_clip": rec["left_clip"],                      # (T, H, W, 3) uint8
                    "state": rec["state"],                              # (state_dim,) float32
                    "actions": rec["actions"],                          # (exec_horizon, state_dim) float32
                    "switch_label": np.float32(rec["switch_label"]),    # 0.0 or 1.0
                    "clip_len": np.int32(args.clip_len),
                    "prompt": np.array(task_description),
                    "task": np.array(args.task),
                    "episode_idx": np.int32(episode_idx),
                    "frame_idx": np.int32(rec["frame_idx"]),
                }
                # Filter out None values
                save_dict = {k: v for k, v in save_dict.items() if v is not None}
                np.savez_compressed(save_path, **save_dict)
                global_sample_idx += 1

            logging.info(
                f"Episode {episode_idx} done: steps={len(replan_records)} "
                f"rescue={n_rescue} normal={n_normal} | "
                f"total_samples={global_sample_idx}"
            )

            # Save episode metadata
            episode_meta = {
                "episode_idx": episode_idx,
                "task": args.task,
                "task_description": task_description,
                "stopped_by_user": user_stopped,
                "num_replan_steps": len(replan_records),
                "num_rescue": n_rescue,
                "num_normal": n_normal,
                "gemini_scores": [(int(f), float(s)) for f, s in sorted_scores],
            }
            all_episode_meta.append(episode_meta)

            # Rename rollout dir
            suffix = "stopped" if user_stopped else "done"
            final_rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            rollout_dir.rename(final_rollout_dir)
            rollout_dir = final_rollout_dir

            # Save Gemini results
            _write_gemini_results(gemini_all_results, rollout_dir, task_description, suffix)

            # Save episode video
            if collected_frames:
                imageio.mimwrite(
                    str(rollout_dir / "complete_video.mp4"),
                    [np.asarray(x) for x in collected_frames],
                    fps=ROLLOUT_FPS,
                )

            episode_idx += 1
            ros_operator.follower_arm_publish_continuous(left0, right0)

    finally:
        ros_operator.follower_arm_publish_continuous(left0, right0)

        # Save collection summary
        meta_path = output_dir / "collection_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "total_samples": global_sample_idx,
                "episodes": all_episode_meta,
            }, f, indent=2)
        logging.info(f"Collection metadata saved to {meta_path}")


def _write_gemini_results(gemini_results: list, rollout_dir: pathlib.Path,
                          task_description: str, suffix: str):
    out_path = rollout_dir / "gemini_values.txt"
    with open(out_path, "w") as f:
        f.write(f"Task: {task_description}\n")
        f.write(f"Outcome: {suffix}\n")
        f.write("=" * 60 + "\n\n")
        for entry in sorted(gemini_results, key=lambda x: x.get("step", 0)):
            step = entry.get("step", "?")
            ts = f"{step / ROLLOUT_FPS:.1f}s" if isinstance(step, int) else "?"
            f.write(f"[Frame {step} / ~{ts}]\n")
            if "error" in entry:
                f.write(f"  ERROR: {entry['error']}\n")
            else:
                for item in entry.get("result", []):
                    f.write(f"  Status : {item.get('status', '')}\n")
                    f.write(f"  Score  : {item.get('score', '')}\n")
                    f.write(f"  Reason : {item.get('reasoning', '')}\n")
            f.write("\n")
    logging.info(f"[Gemini] Results written to {out_path}")

    json_path = rollout_dir / "gemini_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": task_description,
            "outcome": suffix,
            "value_evaluations": sorted(gemini_results, key=lambda x: x.get("step", 0)),
        }, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Agilex switch label collection with Gemini scoring"
    )
    # --- ROS topics ---
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic", type=str, default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic", type=str, default="/camera_r/depth/image_raw")
    parser.add_argument("--follower_arm_left_cmd_topic", type=str, default="/leader/joint_left")
    parser.add_argument("--follower_arm_right_cmd_topic", type=str, default="/leader/joint_right")
    parser.add_argument("--follower_arm_left_topic", type=str, default="/follower/joint_left")
    parser.add_argument("--follower_arm_right_topic", type=str, default="/follower/joint_right")
    parser.add_argument("--endpose_left_cmd_topic", type=str, default="/follower/pos_cmd_left")
    parser.add_argument("--endpose_right_cmd_topic", type=str, default="/follower/pos_cmd_right")
    parser.add_argument("--endpose_left_topic", type=str, default="/follower/end_pose_left")
    parser.add_argument("--endpose_right_topic", type=str, default="/follower/end_pose_right")
    parser.add_argument("--robot_base_topic", type=str, default="/odom_raw")
    parser.add_argument("--robot_base_cmd_topic", type=str, default="/cmd_vel")
    parser.add_argument("--use_robot_base", action="store_true", default=False)
    # --- Inference params ---
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs=7,
                        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    parser.add_argument("--use_depth_image", action="store_true", default=False)
    parser.add_argument("--ctrl_type", type=str, choices=["joint", "eef", "ee6d"], default="joint")
    parser.add_argument("--host", type=str, default="10.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, required=True,
                        choices=["towel", "rubbish", "tissue", "beverage",
                                 "play_pp", "pick_pp", "react"])
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--exec_horizon", type=int, default=25)
    parser.add_argument("--mode", type=str, choices=["naive", "rtc"], default="rtc")
    parser.add_argument("--streaming", action="store_true")
    # --- Data collection params ---
    parser.add_argument("--output_dir", type=str, default="data/agilex_switch_labels",
                        help="Directory to save .npz files with switch labels")
    parser.add_argument("--clip_len", type=int, default=20,
                        help="Number of recent frames per camera for video clips (default 20 = 2s)")

    return parser.parse_args()


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
