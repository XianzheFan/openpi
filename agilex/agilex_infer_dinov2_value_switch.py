"""
Agilex robot inference (SYNC): Standalone DINOv2 Switch Head + SDE + DreamDojo + DINOv2 Value Expert.

Synchronous counterpart to agilex_infer_async_dinov2_value_switch.py.

Differences from the async version:
  - No inference thread, no StreamActionBuffer, no RTC/streaming.
  - Re-plan every `chunk_size` steps with a blocking policy call.
  - At each replan boundary, optionally query the switch head. If switch fires,
    run SDE + DreamDojo + value expert to select the best action chunk.
  - Otherwise, run a fresh ODE policy inference.

Switch source:
  - Primary (--switch_head_ckpt): standalone DINOv2SwitchHead trained by
    train_switch_head_robometer.py on Robometer progress/success labels.
  - Fallback (no --switch_head_ckpt): pi05-integrated 'switch' output from
    the ODE policy server.

Usage:
  # 1. ODE policy server
  python scripts/serve_policy.py policy:checkpoint \\
      --policy.config pi05_libero --policy.dir <ode_ckpt> --port 8000

  # 2. SDE policy server
  python scripts/serve_policy.py policy:checkpoint \\
      --policy.config pi05_sde_libero --policy.dir <sde_ckpt> --port 8001

  # 3. DreamDojo servers (one per candidate, ports 8020..8024)

  # 4. Run sync inference with standalone switch head
  python agilex_infer_sync_dinov2_value_switch.py \\
      --task towel --host 10.0.0.1 --port 8000 \\
      --sde_host 10.0.0.1 --sde_port 8001 \\
      --switch_head_ckpt checkpoints/switch_head_robometer/best_model.pt \\
      --value_expert_ckpt checkpoints/dinov2_value_expert/best_model.pt \\
      --switch_threshold 0.5 --num_sde_samples 5 --dd_base_port 8020
"""

import argparse
import base64
import collections
import concurrent.futures
import json
import logging
import os
import pathlib
import shutil
import signal
import sys
import termios
import threading
import time
import tty

import cv2
import imageio
import numpy as np
import requests
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from ros_operator import RosOperator, get_ros_observation
from train_dinov2_value_expert import DINOv2ValueExpert


ROLLOUT_FPS = 10

observation_window = None
observation_window_lock = threading.Lock()

shutdown_event = threading.Event()


class StandaloneSwitchHead:
    """Wraps the standalone DINOv2SwitchHead trained by train_switch_head_robometer.py."""

    def __init__(self, checkpoint_path: str, dinov2_model: str = "dinov2_vitb14",
                 hidden_dim: int = 256, state_dim: int = 14, num_cameras: int = 3,
                 device: str | None = None):
        from train_switch_head_robometer import DINOv2SwitchHead

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DINOv2SwitchHead(
            dinov2_model=dinov2_model,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_cameras=num_cameras,
            freeze_backbone=True,
        )
        sd = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    def _frame_to_tensor(self, frame_hwc: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(frame_hwc)).permute(2, 0, 1).float() / 255.0

    @torch.no_grad()
    def predict(self, top, right, left, state_np: np.ndarray) -> float:
        """Each camera arg is either a single (H,W,3) uint8 frame or a list of such (clip)."""
        images = []
        for cam in (top, right, left):
            if isinstance(cam, list):
                t = torch.stack([self._frame_to_tensor(f) for f in cam]).unsqueeze(0)
            else:
                t = self._frame_to_tensor(cam).unsqueeze(0)
            images.append(t.to(self.device))
        state_t = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        prob = self.model.predict_switch_prob(images, state_t)
        return float(prob.item())


class SwitchClipBuffer:
    """Rolling buffer of the last clip_len frames per camera for the switch head."""

    def __init__(self, clip_len: int):
        self.clip_len = clip_len
        self.top = collections.deque(maxlen=clip_len)
        self.right = collections.deque(maxlen=clip_len)
        self.left = collections.deque(maxlen=clip_len)

    def update(self, top, right, left):
        if top is None or right is None or left is None:
            return
        self.top.append(np.asarray(top).copy())
        self.right.append(np.asarray(right).copy())
        self.left.append(np.asarray(left).copy())

    def ready(self) -> bool:
        return len(self.top) > 0

    def clips(self) -> tuple[list, list, list]:
        def _pad(dq):
            frames = list(dq)
            if len(frames) < self.clip_len:
                frames = [frames[0]] * (self.clip_len - len(frames)) + frames
            return frames
        return _pad(self.top), _pad(self.right), _pad(self.left)

    def reset(self):
        self.top.clear()
        self.right.clear()
        self.left.clear()


class ValueExpertScorer:
    """Wraps a trained DINOv2ValueExpert for sliding-window video scoring."""

    def __init__(self, checkpoint_path: str, num_clip_frames: int = 4,
                 dinov2_model: str = "dinov2_vitb14", attn_heads: int = 8,
                 attn_layers: int = 2, hidden_dim: int = 512,
                 device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_clip_frames = num_clip_frames

        self.model = DINOv2ValueExpert(
            num_clip_frames=num_clip_frames,
            dinov2_model=dinov2_model,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            hidden_dim=hidden_dim,
            freeze_backbone=True,
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _img_to_tensor(self, img_hwc_uint8: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float().to(self.device) / 255.0

    @torch.no_grad()
    def score_video(self, video_frames: list[np.ndarray]) -> np.ndarray:
        L = len(video_frames)
        if L < self.num_clip_frames:
            video_frames = list(video_frames) + [video_frames[-1]] * (self.num_clip_frames - L)
            L = len(video_frames)

        video_t = torch.stack([self._img_to_tensor(f) for f in video_frames])
        video_t = video_t.unsqueeze(0)

        values = self.model.score_video(video_t,
                                        window_size=self.num_clip_frames, stride=1)
        return values.squeeze(0).cpu().numpy()

    def aggregate_video_score(self, per_window_values: np.ndarray) -> float:
        """Average values from start up to the first dip (lower = closer to subtask completion)."""
        if len(per_window_values) == 0:
            return float("inf")

        min_idx = 0
        min_val = per_window_values[0]
        for i in range(1, len(per_window_values)):
            if per_window_values[i] < min_val:
                min_val = per_window_values[i]
                min_idx = i
            elif per_window_values[i] > min_val + 0.05:
                break

        return float(np.mean(per_window_values[: min_idx + 1]))


def _dreamdojo_generate(host: str, port: int, frame_np: np.ndarray, actions: np.ndarray,
                        save_name: str, task_description: str = "") -> str | None:
    url = f"http://{host}:{port}/generate"

    h, w = frame_np.shape[:2]
    frame_bytes = base64.b64encode(frame_np.tobytes()).decode()

    payload = {
        "frame": frame_bytes,
        "frame_height": h,
        "frame_width": w,
        "actions": actions.tolist(),
        "save_name": save_name,
        "prompt": task_description,
    }
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"[DreamDojo port={port}] generation failed: {e}")
        return None


def _select_best_action_with_value_expert(
    obs_snapshot: dict,
    sde_policy: "OpenpiClient",
    value_scorer: ValueExpertScorer,
    exec_horizon: int,
    task_description: str,
    step_save_dir: pathlib.Path,
    dd_host: str,
    dd_base_port: int,
    num_samples: int = 5,
) -> tuple[np.ndarray, dict]:
    """
    1. Sample N action chunks from SDE policy
    2. Generate N candidate future videos via DreamDojo (parallel)
    3. Score each video with DINOv2 value expert (sliding 4-frame windows)
    4. Pick candidate with lowest aggregated score (closest to subtask completion)

    Returns: (best_actions_np, selection_record)
    """
    step_save_dir.mkdir(parents=True, exist_ok=True)

    action_chunks = [sde_policy.predict_action(obs_snapshot) for _ in range(num_samples)]

    frame_img = obs_snapshot["top"]
    save_prefix = step_save_dir.name

    tasks = [
        {
            "host": dd_host,
            "port": dd_base_port + i,
            "actions": np.array(action_chunks[i][:exec_horizon], dtype=np.float32),
            "save_name": f"{save_prefix}/chunk_{i}",
        }
        for i in range(num_samples)
    ]

    logging.info(f"[Switch] Launching {num_samples} parallel DreamDojo generation requests...")

    def _submit(t):
        return _dreamdojo_generate(t["host"], t["port"], frame_img, t["actions"], t["save_name"], task_description)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        save_paths = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            save_paths[idx] = fut.result()

    valid = [(i, save_paths[i]) for i in range(num_samples)
             if save_paths.get(i) and os.path.exists(save_paths[i])]

    if not valid:
        logging.warning("[Switch] All DreamDojo generations failed; using chunk 0.")
        return np.asarray(action_chunks[0][:exec_horizon]), {
            "num_candidates": 0, "scores": [], "best_idx": 0,
            "error": "All DreamDojo generations failed",
        }

    local_valid = []
    for orig_i, orig_path in valid:
        dst = step_save_dir / f"output_{orig_i}.mp4"
        try:
            shutil.copy2(orig_path, dst)
            local_valid.append((orig_i, str(dst)))
        except Exception as e:
            logging.warning(f"[Switch] Could not copy {orig_path} -> {dst}: {e}")
            local_valid.append((orig_i, orig_path))

    candidate_scores = {}
    per_window_details = {}

    for orig_i, video_path in local_valid:
        try:
            frames = imageio.mimread(video_path)
            frames = [np.asarray(f) for f in frames]
        except Exception as e:
            logging.error(f"[ValueExpert] Failed to read {video_path}: {e}")
            continue

        per_window = value_scorer.score_video(video_frames=frames)
        agg_score = value_scorer.aggregate_video_score(per_window)
        candidate_scores[orig_i] = agg_score
        per_window_details[orig_i] = per_window.tolist()

        logging.info(
            f"[ValueExpert] Candidate {orig_i}: agg_score={agg_score:.4f} "
            f"(windows={len(per_window)}, min={per_window.min():.4f})"
        )

    if not candidate_scores:
        logging.warning("[Switch] All video scoring failed; using first valid chunk.")
        best_idx = local_valid[0][0]
    else:
        best_idx = min(candidate_scores, key=candidate_scores.get)

    selection_record = {
        "num_candidates": len(local_valid),
        "candidate_scores": {str(k): v for k, v in candidate_scores.items()},
        "per_window_details": {str(k): v for k, v in per_window_details.items()},
        "best_idx": int(best_idx),
        "best_score": candidate_scores.get(best_idx, float("inf")),
    }

    logging.info(
        f"[Switch] Selected candidate {best_idx} "
        f"(score={candidate_scores.get(best_idx, 'N/A')})"
    )

    return np.asarray(action_chunks[best_idx][:exec_horizon]), selection_record


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        import rospy
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
            observation_window = collections.deque(maxlen=2)
            observation_window.append({
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                    config["camera_names"][2]: None,
                },
                "eef_pose": None,
            })

    img_front, img_left, img_right, follower_arm_left, follower_arm_right, follower_arm_left_pose, follower_arm_right_pose = (
        get_ros_observation(args, ros_operator)
    )

    qpos = np.concatenate(
        (np.array(follower_arm_left.position), np.array(follower_arm_right.position)), axis=0,
    )

    eef_pose = ros_operator.build_follower_arm_pose(
        follower_arm_left_pose,
        follower_arm_right_pose,
        follower_arm_left,
        follower_arm_right,
    )

    with observation_window_lock:
        observation_window.append({
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_right,
                config["camera_names"][2]: img_left,
            },
            "eef_pose": eef_pose,
        })


def _get_obs_snapshot(args, config):
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

    return {
        "top": image_arrs[0], "right": image_arrs[1], "left": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state, "action_prefix": None, "delay": None,
    }


def inference_fn_sync(args, config, policy, ros_operator):
    """Synchronous blocking policy call. Returns action chunk (np.ndarray)."""
    update_observation_window(args, config, ros_operator)
    start_time = time.perf_counter()

    payload = _get_obs_snapshot(args, config)
    actions = policy.predict_action(payload)

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Sync] Model inference: {elapsed:.1f}ms")
    return np.asarray(actions)


def model_inference(args, config, ros_operator):
    import rospy

    # In --combined mode, one server handles both ODE and SDE on args.host:args.port
    # (see scripts/serve_combined_policy.py). Clients tag each request with "mode"
    # so the server can dispatch. Otherwise we talk to two independent servers as before.
    if args.combined:
        sde_host = args.sde_host or args.host
        sde_port = args.sde_port if args.sde_host else args.port
        policy = OpenpiClient(host=args.host, port=args.port, mode="ode")
        sde_policy = OpenpiClient(host=sde_host, port=sde_port, mode="sde")
        logging.info(f"Combined policy server: {args.host}:{args.port} (ODE+SDE share weights)")
    else:
        # ODE policy (with auxiliary switch head)
        policy = OpenpiClient(host=args.host, port=args.port)

        # SDE policy (queried when switch head triggers)
        sde_policy = None
        if args.sde_host:
            sde_policy = OpenpiClient(host=args.sde_host, port=args.sde_port)
            logging.info(f"SDE policy connected to {args.sde_host}:{args.sde_port}")

    # DINOv2 value expert
    value_scorer = None
    if args.value_expert_ckpt and os.path.exists(args.value_expert_ckpt):
        value_scorer = ValueExpertScorer(
            checkpoint_path=args.value_expert_ckpt,
            num_clip_frames=args.num_clip_frames,
            dinov2_model=args.dinov2_model,
            attn_heads=args.attn_heads,
            attn_layers=args.attn_layers,
            hidden_dim=args.value_hidden_dim,
        )
        logging.info(f"DINOv2 value expert loaded from {args.value_expert_ckpt}")
    elif args.value_expert_ckpt:
        logging.warning(f"Value expert checkpoint not found: {args.value_expert_ckpt}")

    # Standalone DINOv2 switch head.
    standalone_switch_head = None
    if args.switch_head_ckpt and os.path.exists(args.switch_head_ckpt):
        standalone_switch_head = StandaloneSwitchHead(
            checkpoint_path=args.switch_head_ckpt,
            dinov2_model=args.switch_head_dinov2_model,
            hidden_dim=args.switch_head_hidden_dim,
            state_dim=14,
            num_cameras=3,
        )
        logging.info(
            f"Standalone switch head loaded from {args.switch_head_ckpt} "
            f"(use_clip={args.switch_head_use_clip}, clip_len={args.switch_head_clip_len})"
        )
    elif args.switch_head_ckpt:
        logging.warning(f"Switch head checkpoint not found: {args.switch_head_ckpt}")

    can_switch = sde_policy is not None and value_scorer is not None

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]
    left0 = config["left0"]
    right0 = config["right0"]
    task_description = config["language_instruction"]
    print(config)

    ros_operator.follower_arm_publish_continuous(left0, right0)
    print("Warmup servers...")
    policy.warmup()
    if sde_policy:
        sde_policy.warmup(rtc=False, streaming=False)
    print("Servers warmed up")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(args.video_out_path) / f"run_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output dir for this run: {output_dir}")

    try:
        episode_idx = 0
        while not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()

            input("Press enter to start episode")
            task_time = time.time()
            ros_operator.follower_arm_publish_continuous(left0, right0)

            # Tracking
            switch_log: list = []
            value_selections: list = []
            collected_frames: list = []
            frame_counter = 0

            clip_buffer = (
                SwitchClipBuffer(args.switch_head_clip_len)
                if standalone_switch_head is not None and args.switch_head_use_clip
                else None
            )

            task_segment = task_description.replace(" ", "_")
            rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_running"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            user_stopped = False

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                # Keyboard handling
                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        ros_operator.follower_arm_publish_continuous(left0, right0)
                        user_stopped = True
                        break
                    elif result == "quit":
                        user_stopped = True
                        return

                # Capture current frame for rollout + clip buffer.
                # Pull straight from ROS camera queues so recording runs at
                # publish_rate, not at chunk_size cadence.
                front_msg = (
                    ros_operator.front_image_queue[-1]
                    if len(ros_operator.front_image_queue) > 0 else None
                )
                if front_msg is not None:
                    front_img = ros_operator.bridge.imgmsg_to_cv2(front_msg, "passthrough")
                    collected_frames.append(np.asarray(front_img).copy())
                    frame_counter += 1
                    if clip_buffer is not None:
                        right_msg = (
                            ros_operator.right_image_queue[-1]
                            if len(ros_operator.right_image_queue) > 0 else None
                        )
                        left_msg = (
                            ros_operator.left_image_queue[-1]
                            if len(ros_operator.left_image_queue) > 0 else None
                        )
                        if right_msg is not None and left_msg is not None:
                            right_img = ros_operator.bridge.imgmsg_to_cv2(right_msg, "passthrough")
                            left_img = ros_operator.bridge.imgmsg_to_cv2(left_msg, "passthrough")
                            clip_buffer.update(front_img, right_img, left_img)

                # Replan at every chunk boundary
                if t % chunk_size == 0:
                    update_observation_window(args, config, ros_operator)
                    obs_snapshot = _get_obs_snapshot(args, config)

                    # Re-capture clip buffer with fresh frame
                    with observation_window_lock:
                        cur_imgs = observation_window[-1]["images"]
                        front_img = cur_imgs.get(config["camera_names"][0])
                        right_img = cur_imgs.get(config["camera_names"][1])
                        left_img = cur_imgs.get(config["camera_names"][2])
                    if clip_buffer is not None and front_img is not None:
                        clip_buffer.update(front_img, right_img, left_img)

                    triggered = False
                    if can_switch and frame_counter > 0:
                        cur_switch_prob = None
                        if standalone_switch_head is not None:
                            with observation_window_lock:
                                switch_state = observation_window[-1]["qpos"]
                            if clip_buffer is not None and clip_buffer.ready():
                                top_c, right_c, left_c = clip_buffer.clips()
                                cur_switch_prob = standalone_switch_head.predict(
                                    top_c, right_c, left_c, switch_state,
                                )
                            else:
                                cur_switch_prob = standalone_switch_head.predict(
                                    obs_snapshot["top"], obs_snapshot["right"],
                                    obs_snapshot["left"], switch_state,
                                )
                        else:
                            ode_result = policy.predict_action(obs_snapshot)
                            if isinstance(ode_result, dict):
                                cur_switch_prob = ode_result.get("switch", None)

                        if cur_switch_prob is not None and cur_switch_prob > args.switch_threshold:
                            logging.info(
                                f"[Switch] Triggered at frame {frame_counter} "
                                f"(prob={cur_switch_prob:.3f} > {args.switch_threshold})"
                            )

                            print(f"\n\033[93m>>> [System] SDE Policy triggered (prob: {cur_switch_prob:.3f})!\033[0m", flush=True)
                            print(f"\033[93m>>> [System] Waiting for DreamDojo to generate {args.num_sde_samples} candidate videos and Value Expert to score them. Please wait... <<<\033[0m\n", flush=True)

                            switch_log.append({
                                "frame": frame_counter,
                                "switch_prob": float(cur_switch_prob),
                            })

                            step_save_dir = rollout_dir / "switch_steps" / f"frame{frame_counter}"
                            best_actions, sel_record = _select_best_action_with_value_expert(
                                obs_snapshot=obs_snapshot,
                                sde_policy=sde_policy,
                                value_scorer=value_scorer,
                                exec_horizon=chunk_size,
                                task_description=task_description,
                                step_save_dir=step_save_dir,
                                dd_host=args.dd_host,
                                dd_base_port=args.dd_base_port,
                                num_samples=args.num_sde_samples,
                            )

                            best_score = sel_record.get('best_score', 'N/A')
                            if isinstance(best_score, float):
                                best_score_str = f"{best_score:.4f}"
                            else:
                                best_score_str = str(best_score)

                            print(f"\n\033[92m>>> [System] Evaluation complete! Selected optimal action sequence (Candidate ID: {sel_record.get('best_idx')}, Score: {best_score_str})\033[0m", flush=True)
                            print(f"\033[92m>>> [System] Injecting actions into buffer. Robot resuming execution! <<<\033[0m\n", flush=True)
                            
                            sel_record["frame"] = frame_counter
                            sel_record["switch_prob"] = float(cur_switch_prob)
                            value_selections.append(sel_record)

                            L = min(best_actions.shape[0], chunk_size)
                            action_buffer = np.zeros_like(action_buffer)
                            action_buffer[:L] = best_actions[:L]
                            logging.info(f"[Switch] Injected {L} SDE actions")
                            triggered = True

                    if not triggered:
                        actions = inference_fn_sync(args, config, policy, ros_operator)
                        assert actions is not None, "Sync inference returned None"
                        assert actions.shape[0] >= chunk_size, (
                            f"Action chunk length {actions.shape[0]} is smaller than {chunk_size}"
                        )
                        action_buffer = actions[:chunk_size]

                act = action_buffer[t % chunk_size]

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.follower_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.follower_arm_pose_publish(left_action, right_action)

                t += 1
                print(f"[Step {t:4d}] Published")
                rate.sleep()

            suffix = "stopped" if user_stopped else "done"
            base_name = f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            final_dir = output_dir / base_name
            dedup_idx = 1
            while final_dir.exists():
                final_dir = output_dir / f"{base_name}_{dedup_idx}"
                dedup_idx += 1
            shutil.move(str(rollout_dir), str(final_dir))

            _write_results(final_dir, task_description, suffix, switch_log, value_selections)

            if collected_frames:
                annotated = _annotate_rescue_frames(
                    collected_frames, switch_log, window=chunk_size,
                )
                imageio.mimwrite(
                    str(final_dir / "complete_video.mp4"),
                    annotated,
                    fps=ROLLOUT_FPS,
                )

            logging.info(f"Episode {episode_idx} finished: {suffix}")
            logging.info(f"Switch activations: {len(switch_log)}")
            episode_idx += 1
            ros_operator.follower_arm_publish_continuous(left0, right0)

    finally:
        ros_operator.follower_arm_publish_continuous(left0, right0)


def _annotate_rescue_frames(frames: list, switch_log: list, window: int) -> list:
    """Return a copy of ``frames`` with a RESCUE banner drawn during switch windows.

    Each entry in ``switch_log`` has a 1-based ``frame`` index (== ``frame_counter``
    at trigger time). The banner persists for ``window`` frames (== chunk_size,
    the SDE-action injection horizon), or until the next switch trigger.
    """
    if not switch_log:
        return [np.asarray(f) for f in frames]

    events = sorted(switch_log, key=lambda e: int(e["frame"]))
    out = []
    ev_idx = 0
    active_until = -1
    active_prob = 0.0
    for i, frm in enumerate(frames):
        frame_no = i + 1
        while ev_idx < len(events) and int(events[ev_idx]["frame"]) <= frame_no:
            active_until = int(events[ev_idx]["frame"]) + window - 1
            active_prob = float(events[ev_idx]["switch_prob"])
            ev_idx += 1
        img = np.ascontiguousarray(np.asarray(frm))
        if frame_no <= active_until:
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
            txt = f"RESCUE p={active_prob:.2f}"
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)
        out.append(img)
    return out


def _write_results(rollout_dir: pathlib.Path, task_description: str, suffix: str,
                   switch_log: list, value_selections: list):
    json_path = rollout_dir / "switch_value_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": task_description,
            "outcome": suffix,
            "switch_activations": switch_log,
            "value_expert_selections": value_selections,
        }, f, indent=2)
    logging.info(f"Results written to {json_path}")

    txt_path = rollout_dir / "switch_value_results.txt"
    with open(txt_path, "w") as f:
        f.write(f"Task: {task_description}\nOutcome: {suffix}\n")
        f.write(f"Switch activations: {len(switch_log)}\n{'='*60}\n\n")
        for e in switch_log:
            f.write(f"  Frame {e['frame']}: switch_prob={e['switch_prob']:.3f}\n")
        f.write(f"\nValue expert selections: {len(value_selections)}\n{'='*60}\n\n")
        for s in value_selections:
            f.write(
                f"  Frame {s['frame']}: best_idx={s['best_idx']} "
                f"best_score={s.get('best_score', 'N/A')}\n"
                f"    scores={s.get('candidate_scores', {})}\n"
            )


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Agilex SYNC inference: Switch Head + DreamDojo + DINOv2 Value Expert"
    )
    # ---- ROS topics ----
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
    # ---- Inference ----
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs=7,
                        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    parser.add_argument("--use_depth_image", action="store_true", default=False)
    parser.add_argument("--ctrl_type", type=str, choices=["joint", "eef"], default="joint")
    parser.add_argument("--host", type=str, default="10.0.0.1",
                        help="ODE policy server (with switch head)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, default="data/agilex/output")
    # ---- SDE policy ----
    parser.add_argument("--sde_host", type=str, default=None,
                        help="SDE policy server host (required for switch mode)")
    parser.add_argument("--sde_port", type=int, default=8001)
    parser.add_argument("--combined", action="store_true", default=False,
                        help="Single combined ODE+SDE server (scripts/serve_combined_policy.py). "
                             "SDE queries are sent to --host:--port with mode='sde' unless --sde_host is set.")
    # ---- Switch head ----
    parser.add_argument("--switch_threshold", type=float, default=0.5,
                        help="Prob threshold for switching ODE → SDE")
    parser.add_argument("--switch_head_ckpt", type=str, default=None,
                        help="Standalone DINOv2SwitchHead checkpoint (.pt). If not set, "
                             "falls back to the pi05-integrated 'switch' output.")
    parser.add_argument("--switch_head_dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--switch_head_hidden_dim", type=int, default=256)
    parser.add_argument("--switch_head_clip_len", type=int, default=20,
                        help="Rolling clip length (frames per cam) — match training clip_len")
    parser.add_argument("--switch_head_use_clip", action="store_true", default=True,
                        help="Feed the standalone head multi-frame clips (training default)")
    parser.add_argument("--switch_head_no_clip", dest="switch_head_use_clip",
                        action="store_false",
                        help="Use single-frame input instead of clips")
    # ---- DreamDojo ----
    parser.add_argument("--dd_host", type=str, default="127.0.0.1")
    parser.add_argument("--dd_base_port", type=int, default=8020,
                        help="Base port for DreamDojo servers (one per candidate)")
    parser.add_argument("--num_sde_samples", type=int, default=4,
                        help="Number of SDE action candidates")
    # ---- DINOv2 Value Expert ----
    parser.add_argument("--value_expert_ckpt", type=str, default=None)
    parser.add_argument("--num_clip_frames", type=int, default=4)
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument("--value_hidden_dim", type=int, default=512)

    return parser.parse_args()


def main():
    args = get_arguments()

    import rospy  # noqa: F401
    ros_operator = RosOperator(args, mode="inference")
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)
    logging.basicConfig(level=logging.INFO)

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
