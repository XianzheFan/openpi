"""
Agilex robot inference (SYNC): DINOv2 Dual-Head (progress + success) +
threshold-based rescue + SDE + DreamDojo + DINOv2 Value Expert.

Unlike agilex_infer_dinov2_value_switch.py (which uses a binary switch-head
classifier), this version:

  * Loads the dual-head regressor trained by `train_switch_head_dual.py`
    (DINOv2 × 3-cam + proprio → [progress_logit, success_logit]).
  * Every `rescue_check_interval_sec` (= replan_interval=3 frames @ 3 FPS = 1s)
    runs the dual head and applies the same 4-condition rescue criterion that
    the Robometer labeler uses in training (see `progress_to_switch_labels`
    in train_switch_head_robometer.py):

      1. progress < (progress_threshold + time_fraction × 0.3) AND
         time_fraction > 0.15
      2. progress dropped by ≥ progress_drop compared to the window 2 samples
         back (= replan_interval × 2 = 6 frames = 2s)
      3. success < success_threshold AND time_fraction > 0.1,
         guarded by: skip if progress rose by ≥ progress_rising within the
         last 3 frames (= 1 sample back = 1s)
      4. (default off) expected_progress_rate > 0 AND progress < expected × 0.5
         AND time_fraction > 0.3

  * On rescue trigger, runs the SDE-policy + DreamDojo + DINOv2 Value Expert
    selection pipeline (shared with the switch-head version) and overwrites
    the current action chunk.

Usage
-----
  # 1. ODE policy server
  python scripts/serve_policy.py policy:checkpoint \\
      --policy.config pi05_libero --policy.dir <ode_ckpt> --port 8000

  # 2. SDE policy server
  python scripts/serve_policy.py policy:checkpoint \\
      --policy.config pi05_sde_libero --policy.dir <sde_ckpt> --port 8001

  # 3. DreamDojo servers (one per candidate, ports 8020..8024)

  # 4. Run sync inference with dual-head + threshold rescue
  python agilex_infer_dinov2_dual_threshold.py \\
      --task towel --host 10.0.0.1 --port 8000 \\
      --sde_host 10.0.0.1 --sde_port 8001 \\
      --dual_head_ckpt checkpoints/switch_head_dual/best_model.pt \\
      --value_expert_ckpt checkpoints/dinov2_value_expert/best_model.pt \\
      --num_sde_samples 5 --dd_base_port 8020 \\
      --rescue_check_interval_sec 1.0 \\
      --progress_threshold 0.25 --progress_drop 0.04 --success_threshold 0.6
"""

import argparse
import collections
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
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import check_keyboard_input, get_config, handle_interactive_mode, process_action
from ros_operator import RosOperator, get_ros_observation
from agilex_infer_dinov2_value_switch import (
    ROLLOUT_FPS,
    SwitchClipBuffer,
    ValueExpertScorer,
    _select_best_action_with_value_expert,
)


observation_window = None
observation_window_lock = threading.Lock()
shutdown_event = threading.Event()


class StandaloneDualHead:
    """Wraps DINOv2DualHead from train_switch_head_dual.py.

    predict(...) returns (progress, success) ∈ [0, 1]².
    """

    def __init__(
        self,
        checkpoint_path: str,
        dinov2_model: str = "dinov2_vitb14",
        hidden_dim: int = 256,
        state_dim: int = 14,
        num_cameras: int = 3,
        device: str | None = None,
    ):
        from train_switch_head_dual import DINOv2DualHead

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DINOv2DualHead(
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
    def predict(self, top, right, left, state_np: np.ndarray) -> tuple[float, float]:
        """Each camera arg is either a single (H, W, 3) uint8 frame or a list of frames (clip)."""
        images = []
        for cam in (top, right, left):
            if isinstance(cam, list):
                t = torch.stack([self._frame_to_tensor(f) for f in cam]).unsqueeze(0)
            else:
                t = self._frame_to_tensor(cam).unsqueeze(0)
            images.append(t.to(self.device))
        state_t = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        probs = self.model.predict(images, state_t)  # (1, 2) sigmoid
        return float(probs[0, 0].item()), float(probs[0, 1].item())


class RescueThresholdChecker:
    """Mirrors `progress_to_switch_labels` (train_switch_head_robometer.py) at
    inference time.

    A "tick" here represents one dual-head sample, taken every
    `rescue_check_interval_sec` seconds (= replan_interval=3 frames @ 3 FPS = 1s
    in training). Each tick appends (progress, success, time_fraction) to
    history. `add_and_check` applies the 4-condition rescue rule.

    `prev_p` uses the tick 2 samples back (= replan_interval × 2 = 6 frames
    @ 3 FPS = 2s), matching training.

    The condition-3 "is_rising" guard uses the tick 1 sample back (= 3 frames
    @ 3 FPS = 1s).
    """

    def __init__(
        self,
        progress_threshold: float = 0.25,
        progress_drop: float = 0.04,
        success_threshold: float = 0.6,
        progress_rising: float = 0.05,
        expected_progress_rate: float = 0.0,
    ):
        self.progress_threshold = progress_threshold
        self.progress_drop = progress_drop
        self.success_threshold = success_threshold
        self.progress_rising = progress_rising
        self.expected_progress_rate = expected_progress_rate
        self.history: list[tuple[float, float, float]] = []

    def reset(self):
        self.history.clear()

    def add_and_check(
        self, progress: float, success: float, time_fraction: float,
    ) -> tuple[bool, list[str], dict]:
        self.history.append((float(progress), float(success), float(time_fraction)))

        p = float(progress)
        s = float(success)
        prev_p = self.history[-3][0] if len(self.history) >= 3 else p
        recent_p = self.history[-2][0] if len(self.history) >= 2 else p
        is_rising = (p - recent_p) >= self.progress_rising

        adaptive_threshold = self.progress_threshold + time_fraction * 0.3
        reasons: list[str] = []

        if p < adaptive_threshold and time_fraction > 0.15:
            reasons.append("cond1_progress_low")

        if (prev_p - p) >= self.progress_drop:
            reasons.append("cond2_progress_drop")

        if s < self.success_threshold and time_fraction > 0.1 and not is_rising:
            reasons.append("cond3_success_low")

        if self.expected_progress_rate > 0:
            expected_p = time_fraction * self.expected_progress_rate
            if p < expected_p * 0.5 and time_fraction > 0.3:
                reasons.append("cond4_behind_expected")

        info = {
            "progress": p,
            "success": s,
            "time_fraction": time_fraction,
            "prev_p_2s": prev_p,
            "recent_p_1s": recent_p,
            "is_rising": is_rising,
            "adaptive_threshold": adaptive_threshold,
        }
        return bool(reasons), reasons, info


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

    img_front, img_left, img_right, puppet_arm_left, puppet_arm_right, puppet_arm_left_pose, puppet_arm_right_pose = (
        get_ros_observation(args, ros_operator)
    )

    qpos = np.concatenate(
        (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0,
    )

    eef_pose = ros_operator.build_puppet_arm_pose(
        puppet_arm_left_pose,
        puppet_arm_right_pose,
        puppet_arm_left,
        puppet_arm_right,
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
    update_observation_window(args, config, ros_operator)
    start_time = time.perf_counter()

    payload = _get_obs_snapshot(args, config)
    actions = policy.predict_action(payload)

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Sync] Model inference: {elapsed:.1f}ms")
    return np.asarray(actions)


def model_inference(args, config, ros_operator):
    import rospy

    if args.combined:
        sde_host = args.sde_host or args.host
        sde_port = args.sde_port if args.sde_host else args.port
        policy = OpenpiClient(host=args.host, port=args.port, mode="ode")
        sde_policy = OpenpiClient(host=sde_host, port=sde_port, mode="sde")
        logging.info(f"Combined policy server: {args.host}:{args.port} (ODE+SDE share weights)")
    else:
        policy = OpenpiClient(host=args.host, port=args.port)
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

    # Dual head (progress + success)
    if not args.dual_head_ckpt or not os.path.exists(args.dual_head_ckpt):
        raise FileNotFoundError(
            f"--dual_head_ckpt is required and must exist (got: {args.dual_head_ckpt})"
        )
    dual_head = StandaloneDualHead(
        checkpoint_path=args.dual_head_ckpt,
        dinov2_model=args.dual_head_dinov2_model,
        hidden_dim=args.dual_head_hidden_dim,
        state_dim=14,
        num_cameras=3,
    )
    logging.info(
        f"Dual head loaded from {args.dual_head_ckpt} "
        f"(use_clip={args.dual_head_use_clip}, clip_len={args.dual_head_clip_len})"
    )

    can_switch = sde_policy is not None and value_scorer is not None
    if not can_switch:
        logging.warning(
            "SDE policy or value expert missing; dual head will still report "
            "scores but rescue actions will not be executed."
        )

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]
    left0 = config["left0"]
    right0 = config["right0"]
    task_description = config["language_instruction"]
    print(config)

    rescue_check_stride = max(1, int(round(args.publish_rate * args.rescue_check_interval_sec)))
    rescue_cooldown_steps = max(rescue_check_stride, int(round(chunk_size * args.rescue_cooldown_frac)))
    logging.info(
        f"[Rescue] check every {rescue_check_stride} publish steps "
        f"(~{args.rescue_check_interval_sec:.2f}s @ {args.publish_rate} Hz), "
        f"cooldown {rescue_cooldown_steps} steps."
    )

    ros_operator.puppet_arm_publish_continuous(left0, right0)
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
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()

            input("Press enter to start episode")
            task_time = time.time()
            ros_operator.puppet_arm_publish_continuous(left0, right0)

            rescue_log: list = []
            dual_head_log: list = []
            value_selections: list = []
            collected_frames: list = []
            frame_counter = 0

            clip_buffer = (
                SwitchClipBuffer(args.dual_head_clip_len)
                if args.dual_head_use_clip else None
            )

            checker = RescueThresholdChecker(
                progress_threshold=args.progress_threshold,
                progress_drop=args.progress_drop,
                success_threshold=args.success_threshold,
                progress_rising=args.progress_rising,
                expected_progress_rate=args.expected_progress_rate,
            )

            task_segment = task_description.replace(" ", "_")
            rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_running"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            action_buffer_base_t = -chunk_size  # forces ODE replan on first step
            last_rescue_check_t = -rescue_check_stride
            last_rescue_trigger_t = -10 ** 9
            last_published_action: np.ndarray | None = None
            user_stopped = False
            t = 0

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                key = check_keyboard_input()
                if key == " ":
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        ros_operator.puppet_arm_publish_continuous(left0, right0)
                        user_stopped = True
                        break
                    elif result == "quit":
                        user_stopped = True
                        return

                # Capture current frame for rollout video + clip buffer.
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

                # ----- Rescue check at fixed interval (independent of chunk boundary) -----
                rescue_triggered = False
                if (
                    can_switch
                    and frame_counter > 0
                    and (t - last_rescue_check_t) >= rescue_check_stride
                    and (t - last_rescue_trigger_t) >= rescue_cooldown_steps
                ):
                    last_rescue_check_t = t
                    update_observation_window(args, config, ros_operator)
                    obs_snapshot = _get_obs_snapshot(args, config)

                    with observation_window_lock:
                        cur_imgs = observation_window[-1]["images"]
                        front_img_now = cur_imgs.get(config["camera_names"][0])
                        right_img_now = cur_imgs.get(config["camera_names"][1])
                        left_img_now = cur_imgs.get(config["camera_names"][2])
                        switch_state = observation_window[-1]["qpos"]
                    if clip_buffer is not None and front_img_now is not None:
                        clip_buffer.update(front_img_now, right_img_now, left_img_now)

                    if clip_buffer is not None and clip_buffer.ready():
                        top_c, right_c, left_c = clip_buffer.clips()
                        progress_p, success_p = dual_head.predict(
                            top_c, right_c, left_c, switch_state,
                        )
                    else:
                        progress_p, success_p = dual_head.predict(
                            obs_snapshot["top"], obs_snapshot["right"],
                            obs_snapshot["left"], switch_state,
                        )

                    time_fraction = t / max(max_publish_step - 1, 1)
                    triggered, reasons, info = checker.add_and_check(
                        progress_p, success_p, time_fraction,
                    )
                    dual_head_log.append({
                        "t": t,
                        "frame": frame_counter,
                        "progress": progress_p,
                        "success": success_p,
                        "time_fraction": time_fraction,
                        "triggered": triggered,
                        "reasons": reasons,
                        "prev_p_2s": info["prev_p_2s"],
                        "recent_p_1s": info["recent_p_1s"],
                        "is_rising": info["is_rising"],
                        "adaptive_threshold": info["adaptive_threshold"],
                    })

                    logging.info(
                        f"[DualHead t={t:4d} f={frame_counter:4d}] "
                        f"p={progress_p:.3f} s={success_p:.3f} "
                        f"tf={time_fraction:.2f} "
                        f"adapt={info['adaptive_threshold']:.3f} "
                        f"rising={info['is_rising']} "
                        f"{'RESCUE' if triggered else 'ok'}"
                        f"{' ' + ','.join(reasons) if triggered else ''}"
                    )

                    if triggered:
                        step_save_dir = rollout_dir / "rescue_steps" / f"t{t}_f{frame_counter}"
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
                        sel_record["t"] = t
                        sel_record["frame"] = frame_counter
                        sel_record["progress"] = progress_p
                        sel_record["success"] = success_p
                        sel_record["reasons"] = reasons
                        value_selections.append(sel_record)

                        rescue_log.append({
                            "t": t,
                            "frame": frame_counter,
                            "progress": progress_p,
                            "success": success_p,
                            "reasons": reasons,
                        })

                        L = min(best_actions.shape[0], chunk_size)
                        action_buffer = np.zeros_like(action_buffer)
                        best_actions = np.asarray(best_actions, dtype=action_buffer.dtype)
                        K = int(max(0, args.rescue_blend_steps))
                        if K > 0 and last_published_action is not None and L > 0:
                            offset = last_published_action.astype(action_buffer.dtype) - best_actions[0]
                            ramp = np.maximum(0.0, 1.0 - np.arange(L, dtype=np.float64) / float(K))
                            action_buffer[:L] = best_actions[:L] + offset[None, :] * ramp[:, None]
                            max_jump = float(np.max(np.abs(action_buffer[0] - last_published_action)))
                            logging.info(
                                f"[Rescue] Blended first {min(K, L)} steps "
                                f"(max |Δ| step0 -> last = {max_jump:.4f})"
                            )
                        else:
                            action_buffer[:L] = best_actions[:L]
                        action_buffer_base_t = t
                        last_rescue_trigger_t = t
                        rescue_triggered = True
                        logging.info(f"[Rescue] Injected {L} SDE actions at t={t}")

                # ----- ODE replan when action buffer exhausted (and no rescue just fired) -----
                if not rescue_triggered and (t - action_buffer_base_t) >= chunk_size:
                    actions = inference_fn_sync(args, config, policy, ros_operator)
                    assert actions is not None, "Sync inference returned None"
                    assert actions.shape[0] >= chunk_size, (
                        f"Action chunk length {actions.shape[0]} is smaller than {chunk_size}"
                    )
                    action_buffer = actions[:chunk_size]
                    action_buffer_base_t = t

                # ----- Execute action from buffer -----
                idx_in_chunk = t - action_buffer_base_t
                idx_in_chunk = max(0, min(idx_in_chunk, chunk_size - 1))
                act = action_buffer[idx_in_chunk]

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.puppet_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.puppet_arm_pose_publish(left_action, right_action)
                last_published_action = np.asarray(act).copy()

                t += 1
                print(f"[Step {t:4d}] Published  (buf_idx={idx_in_chunk}/{chunk_size})")
                rate.sleep()

            suffix = "stopped" if user_stopped else "done"
            base_name = f"rollout_{task_segment}_ep{episode_idx}_{suffix}"
            final_dir = output_dir / base_name
            dedup_idx = 1
            while final_dir.exists():
                final_dir = output_dir / f"{base_name}_{dedup_idx}"
                dedup_idx += 1
            shutil.move(str(rollout_dir), str(final_dir))

            _write_results(
                final_dir, task_description, suffix,
                rescue_log, dual_head_log, value_selections,
            )

            if collected_frames:
                annotated = _annotate_rescue_frames(
                    collected_frames, rescue_log, window=chunk_size,
                )
                imageio.mimwrite(
                    str(final_dir / "complete_video.mp4"),
                    annotated,
                    fps=ROLLOUT_FPS,
                )

            logging.info(f"Episode {episode_idx} finished: {suffix}")
            logging.info(f"Rescue activations: {len(rescue_log)}")
            episode_idx += 1
            ros_operator.puppet_arm_publish_continuous(left0, right0)

    finally:
        ros_operator.puppet_arm_publish_continuous(left0, right0)


def _annotate_rescue_frames(frames: list, rescue_log: list, window: int) -> list:
    """Return a copy of ``frames`` with a RESCUE banner drawn during rescue windows.

    Each entry in ``rescue_log`` has a 1-based ``frame`` index and dual-head
    fields ``progress`` and ``success``. The banner persists for ``window``
    frames (== chunk_size, the SDE-action injection horizon), or until the
    next rescue trigger.
    """
    if not rescue_log:
        return [np.asarray(f) for f in frames]

    events = sorted(rescue_log, key=lambda e: int(e["frame"]))
    out = []
    ev_idx = 0
    active_until = -1
    active_p = 0.0
    active_s = 0.0
    active_reasons: list = []
    for i, frm in enumerate(frames):
        frame_no = i + 1
        while ev_idx < len(events) and int(events[ev_idx]["frame"]) <= frame_no:
            active_until = int(events[ev_idx]["frame"]) + window - 1
            active_p = float(events[ev_idx].get("progress", 0.0))
            active_s = float(events[ev_idx].get("success", 0.0))
            active_reasons = list(events[ev_idx].get("reasons", []))
            ev_idx += 1
        img = np.ascontiguousarray(np.asarray(frm))
        if frame_no <= active_until:
            h, w = img.shape[:2]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
            txt = f"RESCUE p={active_p:.2f} s={active_s:.2f}"
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)
            if active_reasons:
                sub = ",".join(active_reasons)
                cv2.putText(img, sub, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, sub, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 1, cv2.LINE_AA)
        out.append(img)
    return out


def _write_results(
    rollout_dir: pathlib.Path,
    task_description: str,
    suffix: str,
    rescue_log: list,
    dual_head_log: list,
    value_selections: list,
):
    json_path = rollout_dir / "dual_threshold_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "task": task_description,
            "outcome": suffix,
            "rescue_activations": rescue_log,
            "dual_head_trace": dual_head_log,
            "value_expert_selections": value_selections,
        }, f, indent=2)
    logging.info(f"Results written to {json_path}")

    txt_path = rollout_dir / "dual_threshold_results.txt"
    with open(txt_path, "w") as f:
        f.write(f"Task: {task_description}\nOutcome: {suffix}\n")
        f.write(f"Rescue activations: {len(rescue_log)}\n{'='*60}\n\n")
        for e in rescue_log:
            f.write(
                f"  t={e['t']} frame={e['frame']}  "
                f"p={e['progress']:.3f} s={e['success']:.3f}  "
                f"reasons={','.join(e['reasons'])}\n"
            )
        f.write(f"\nDual-head samples: {len(dual_head_log)}\n{'='*60}\n\n")
        for e in dual_head_log:
            f.write(
                f"  t={e['t']:4d} f={e['frame']:4d}  "
                f"p={e['progress']:.3f} s={e['success']:.3f} tf={e['time_fraction']:.2f}  "
                f"adapt={e['adaptive_threshold']:.3f}  rising={e['is_rising']}  "
                f"{'RESCUE' if e['triggered'] else 'ok'}"
                f"{' ' + ','.join(e['reasons']) if e['triggered'] else ''}\n"
            )
        f.write(f"\nValue expert selections: {len(value_selections)}\n{'='*60}\n\n")
        for s in value_selections:
            f.write(
                f"  t={s.get('t')} frame={s.get('frame')}  best_idx={s['best_idx']}  "
                f"best_score={s.get('best_score', 'N/A')}\n"
                f"    scores={s.get('candidate_scores', {})}\n"
            )


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Agilex SYNC inference: Dual Head (progress+success) + "
                    "threshold rescue + DreamDojo + DINOv2 Value Expert"
    )
    # ---- ROS topics ----
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic", type=str, default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic", type=str, default="/camera_r/depth/image_raw")
    parser.add_argument("--master_arm_left_topic", type=str, default="/master/joint_left")
    parser.add_argument("--master_arm_right_topic", type=str, default="/master/joint_right")
    parser.add_argument("--puppet_arm_left_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--puppet_arm_right_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--pos_cmd_left_topic", type=str, default="/puppet/pos_cmd_left")
    parser.add_argument("--pos_cmd_right_topic", type=str, default="/puppet/pos_cmd_right")
    parser.add_argument("--puppet_arm_left_pose_topic", type=str, default="/puppet/end_pose_euler_left")
    parser.add_argument("--puppet_arm_right_pose_topic", type=str, default="/puppet/end_pose_euler_right")
    # ---- Inference ----
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs=7,
                        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    parser.add_argument("--use_depth_image", action="store_true", default=False)
    parser.add_argument("--ctrl_type", type=str, choices=["joint", "eef"], default="joint")
    parser.add_argument("--host", type=str, default="10.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, default="data/agilex/output")
    # ---- SDE policy ----
    parser.add_argument("--sde_host", type=str, default=None)
    parser.add_argument("--sde_port", type=int, default=8001)
    parser.add_argument("--combined", action="store_true", default=False)
    # ---- Dual head ----
    parser.add_argument("--dual_head_ckpt", type=str, required=True,
                        help="DINOv2DualHead checkpoint (.pt) from train_switch_head_dual.py")
    parser.add_argument("--dual_head_dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--dual_head_hidden_dim", type=int, default=256)
    parser.add_argument("--dual_head_clip_len", type=int, default=20)
    parser.add_argument("--dual_head_use_clip", action="store_true", default=True)
    parser.add_argument("--dual_head_no_clip", dest="dual_head_use_clip", action="store_false")
    # ---- Rescue threshold ----
    parser.add_argument("--rescue_check_interval_sec", type=float, default=1.0,
                        help="Seconds between dual-head rescue checks "
                             "(replan_interval=3 frames @ 3 FPS = 1s in training)")
    parser.add_argument("--rescue_cooldown_frac", type=float, default=1.0,
                        help="After a rescue fires, skip new rescue checks for "
                             "chunk_size × this many publish steps (default 1 chunk).")
    parser.add_argument("--rescue_blend_steps", type=int, default=5,
                        help="Linearly blend the first K SDE actions toward the "
                             "last published action so the injection is C0 "
                             "continuous. 0 disables blending.")
    parser.add_argument("--progress_threshold", type=float, default=0.25,
                        help="Base progress threshold; adaptive = base + tf × 0.3")
    parser.add_argument("--progress_drop", type=float, default=0.04,
                        help="Trigger if progress drops ≥ this vs. 2s-ago tick")
    parser.add_argument("--success_threshold", type=float, default=0.6,
                        help="Trigger if success < this (guarded by rising progress)")
    parser.add_argument("--progress_rising", type=float, default=0.05,
                        help="Suppress cond-3 if progress rose ≥ this vs. 1s-ago tick")
    parser.add_argument("--expected_progress_rate", type=float, default=0.0,
                        help="If > 0, enable cond-4 (below expected linear trajectory)")
    # ---- DreamDojo ----
    parser.add_argument("--dd_host", type=str, default="127.0.0.1")
    parser.add_argument("--dd_base_port", type=int, default=8020)
    parser.add_argument("--num_sde_samples", type=int, default=4)
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
