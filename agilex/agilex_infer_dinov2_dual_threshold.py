"""
Agilex robot inference (SYNC): DINOv2 Dual-Head (progress + success) +
threshold-based rescue + SDE + DreamDojo.

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

  * On rescue trigger, runs the SDE-policy + DreamDojo selection pipeline
    (DreamDojo server scores each candidate) and overwrites the current
    action chunk.

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
      --num_sde_samples 5 --dd_base_port 8020 \\
      --rescue_check_interval_sec 1.0 \\
      --progress_threshold 0.25 --progress_drop 0.04 --success_threshold 0.6
"""

import argparse
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
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from agilex_utils import (
    drain_keyboard_events,
    get_config,
    handle_interactive_mode,
    process_action,
)
from ros_operator import RosOperator, get_ros_observation
from agilex_infer_dinov2_value_switch import (
    ROLLOUT_FPS,
    SwitchClipBuffer,
    _dreamdojo_generate,
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
        completion_guard: float = 1.01,
    ):
        self.progress_threshold = progress_threshold
        self.progress_drop = progress_drop
        self.success_threshold = success_threshold
        self.progress_rising = progress_rising
        self.expected_progress_rate = expected_progress_rate
        self.completion_guard = completion_guard
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
        info = {
            "progress": p,
            "success": s,
            "time_fraction": time_fraction,
            "prev_p_2s": prev_p,
            "recent_p_1s": recent_p,
            "is_rising": is_rising,
            "adaptive_threshold": adaptive_threshold,
            "completion_guard": self.completion_guard,
        }

        # Completion guard: if the model thinks the task is basically finished,
        # skip all rescue conditions. This prevents end-of-episode false
        # triggers where the head's prediction noise causes cond2/cond3 to
        # fire after the task is already done.
        if max(p, s) >= self.completion_guard:
            info["completion_guarded"] = True
            return False, [], info

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

        return bool(reasons), reasons, info


DAGGER_KEYS = {
    # arrow keys (may not work on all terminals — DECCKM application mode etc.)
    "UP", "DOWN", "LEFT", "RIGHT",
    # letter fallbacks: IJKL drives x/y, u/d drives z, r releases
    "i", "k", "j", "l",
    "u", "d",
    "r",
}


class DaggerController:
    """Tracks human teleop overrides on the active arm's EEF position.

    On the first DAgger keypress of an episode (or the first one after an 'r'
    release), we snapshot the current eef_pose as ``base_eef``. Subsequent
    arrow / u / d presses accumulate a (dx, dy, dz) offset, which we add to
    the active arm's xyz at publish time. The non-active arm is held at its
    snapshot pose so the policy doesn't fight the human.
    """

    def __init__(self, step_xyz: float, arm: str):
        self.step_xyz = float(step_xyz)
        self.arm = arm  # 'right' or 'left'
        self.active = False
        self.base_eef: np.ndarray | None = None
        self.offset = np.zeros(3, dtype=np.float64)
        self.events_since_start = 0

    def reset(self):
        self.active = False
        self.base_eef = None
        self.offset = np.zeros(3, dtype=np.float64)
        self.events_since_start = 0

    def begin(self, base_eef: np.ndarray):
        self.active = True
        self.base_eef = np.asarray(base_eef, dtype=np.float64).copy()
        self.offset = np.zeros(3, dtype=np.float64)
        self.events_since_start = 0

    def apply(self, key: str) -> bool:
        """Update offset based on key. Returns True if the key was consumed.

        Layout:
          i / k  → x +/-   (arrow ↑ / ↓ also map here)
          j / l  → y +/-   (arrow ← / → also map here)
          u / d  → z +/-
          r      → release
        """
        if key in ("UP", "i"):
            self.offset[0] += self.step_xyz
        elif key in ("DOWN", "k"):
            self.offset[0] -= self.step_xyz
        elif key in ("LEFT", "j"):
            self.offset[1] += self.step_xyz
        elif key in ("RIGHT", "l"):
            self.offset[1] -= self.step_xyz
        elif key == "u":
            self.offset[2] += self.step_xyz
        elif key == "d":
            self.offset[2] -= self.step_xyz
        elif key == "r":
            self.reset()
            return True
        else:
            return False
        self.events_since_start += 1
        return True

    def target_eef(self) -> np.ndarray:
        if self.base_eef is None:
            raise RuntimeError("DaggerController.target_eef() called before begin()")
        target = self.base_eef.copy()
        slc = slice(7, 10) if self.arm == "right" else slice(0, 3)
        target[slc] = self.base_eef[slc] + self.offset
        return target


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
    update_observation_window(args, config, ros_operator)
    start_time = time.perf_counter()

    payload = _get_obs_snapshot(args, config)
    actions = policy.predict_action(payload)

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"[Sync] Model inference: {elapsed:.1f}ms")
    return np.asarray(actions)


def _select_best_action_with_prefix(
    obs_snapshot: dict,
    sde_policy: "OpenpiClient",
    exec_horizon: int,
    task_description: str,
    step_save_dir: pathlib.Path,
    dd_host: str,
    dd_base_port: int,
    num_samples: int,
    prefix_action: "np.ndarray | None",
    skip_steps: int,
    dd_action_stride: int = 4,
    dd_action_chunk_in: int = 13,
) -> tuple[np.ndarray, dict]:
    """Sends action candidates to DreamDojo server, which generates the video 
    AND scores it in-memory, returning the score over the network.
    """
    step_save_dir.mkdir(parents=True, exist_ok=True)

    raw_chunks = [np.asarray(sde_policy.predict_action(obs_snapshot)) for _ in range(num_samples)]
    skip = int(max(0, skip_steps))
    action_chunks: list[np.ndarray] = []
    for ch in raw_chunks:
        ch = np.array(ch, copy=True)
        if prefix_action is not None and skip > 0:
            k = min(skip, ch.shape[0])
            ch[:k] = np.asarray(prefix_action, dtype=ch.dtype)
        action_chunks.append(ch)

    frame_img       = obs_snapshot["top"]
    frame_left_img  = obs_snapshot.get("left")
    frame_right_img = obs_snapshot.get("right")
    # Server-side save path is rooted at its own --save-dir; pass a short
    # relative prefix so the saved video lands under
    # "<server save_dir>/rollout_<task>_ep<N>_running/rescue_steps/t<t>_f<frame>/
    #  chunk_<i>_s<server_id>_epN.mp4"
    # That keeps the rollout / episode / step info but avoids mirroring the
    # absolute client-side path. step_save_dir is built as
    # "<output_dir>/rollout_..._ep<N>_running/rescue_steps/t<t>_f<frame>"
    # so the trailing 3 path parts encode everything we want to keep.
    tail_parts = step_save_dir.parts[-3:]
    save_prefix = "/".join(tail_parts) if tail_parts else step_save_dir.name
    # Subsample 30Hz pi05 chunks down to DreamDojo training rate
    # (new_agilex_3view: timestep_interval=4 over 30fps native = 7.5fps).
    # Send `dd_action_chunk_in` (default 13 = model_action_chunk+1) entries
    # spaced by `dd_action_stride` so DreamDojo's grouped-delta computation
    # sees in-distribution motion magnitudes.
    s = max(1, int(dd_action_stride))
    n_in = max(1, int(dd_action_chunk_in))
    base_seed = int(time.time() * 1e6) % (2**31)
    tasks = [
        {
            "host": dd_host,
            "port": dd_base_port + i,
            "actions": np.asarray(
                action_chunks[i][:exec_horizon][::s][:n_in], dtype=np.float32
            ),
            "save_name": f"{save_prefix}/chunk_{i}",
            "seed": base_seed + i,
        }
        for i in range(num_samples)
    ]

    logging.info(
        f"[Dual] Launching {num_samples} DreamDojo gens+scores "
        f"(prefix_skip={skip}, horizon={exec_horizon}, "
        f"dd_stride={s}, dd_n_in={n_in}, "
        f"sent_actions={tasks[0]['actions'].shape[0] if tasks else 0})..."
    )

    for i, ch in enumerate(action_chunks):
        raw = np.asarray(ch[:exec_horizon], dtype=np.float32)
        sub = np.asarray(tasks[i]["actions"], dtype=np.float32)
        if raw.shape[0] >= 2:
            raw_d = np.linalg.norm(np.diff(raw, axis=0), axis=1)
            raw_stats = (
                f"mean={raw_d.mean():.4f} max={raw_d.max():.4f} "
                f"min={raw_d.min():.4f} sum={raw_d.sum():.4f}"
            )
        else:
            raw_stats = "n<2"
        if sub.shape[0] >= 2:
            sub_d = np.linalg.norm(np.diff(sub, axis=0), axis=1)
            sub_stats = (
                f"mean={sub_d.mean():.4f} max={sub_d.max():.4f} "
                f"min={sub_d.min():.4f} sum={sub_d.sum():.4f}"
            )
        else:
            sub_stats = "n<2"
        logging.info(
            f"[Dual][cand={i}] raw_chunk(N={raw.shape[0]}) Δ-norm: {raw_stats} | "
            f"sent_to_dd(N={sub.shape[0]}, stride={s}) Δ-norm: {sub_stats}"
        )

    def _submit(task):
        return _dreamdojo_generate(
            task["host"], task["port"], frame_img,
            task["actions"], task["save_name"], task_description,
            frame_left_np=frame_left_img,
            frame_right_np=frame_right_img,
            seed=task["seed"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_samples)) as ex:
        futures = {ex.submit(_submit, t): i for i, t in enumerate(tasks)}
        results_dict: dict[int, dict] = {}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            try:
                results_dict[idx] = fut.result()
            except Exception as e:
                logging.error(f"[Dual] Request {idx} failed: {e}")
                results_dict[idx] = None

    candidate_scores: dict[int, float] = {}
    local_valid: list[tuple[int, str]] = []

    for orig_i in range(num_samples):
        res = results_dict.get(orig_i)
        if res and res.get("score") is not None:
            score = float(res["score"])
            candidate_scores[orig_i] = score
            logging.info(f"[ServerScore] Candidate {orig_i} evaluated by server: score={score:.4f}")
            orig_path = res.get("save_path")
            if orig_path and os.path.exists(orig_path):
                dst = step_save_dir / f"output_{orig_i}.mp4"
                try:
                    shutil.copy2(orig_path, dst)
                    local_valid.append((orig_i, str(dst)))
                except Exception as e:
                    logging.warning(f"[Dual] Could not copy {orig_path} -> {dst}: {e}")
                    local_valid.append((orig_i, orig_path))
        else:
            logging.warning(f"[Dual] Candidate {orig_i} failed or returned no score.")

    if not candidate_scores:
        logging.warning("[Dual] All video generations/scoring failed; using chunk 0.")
        best_idx = 0
    else:
        best_idx = min(candidate_scores, key=candidate_scores.get)

    selection_record = {
        "num_candidates": len(candidate_scores),
        "candidate_scores": {str(k): v for k, v in candidate_scores.items()},
        "best_idx": int(best_idx),
        "best_score": candidate_scores.get(best_idx, float("inf")),
        "prefix_skip_steps": skip,
    }
    # Stash the full SDE candidate set so the caller can save all of them
    # (not just the winning one). Each entry is the action chunk truncated
    # to ``exec_horizon`` so it lines up with what DreamDojo actually saw.
    selection_record["all_candidates"] = [
        np.asarray(ch[:exec_horizon], dtype=np.float32) for ch in action_chunks
    ]
    selection_record["all_candidate_scores"] = [
        candidate_scores.get(i, None) for i in range(num_samples)
    ]
    logging.info(
        f"[Dual] Selected candidate {best_idx} "
        f"(score={candidate_scores.get(best_idx, 'N/A')}, prefix_skip={skip})"
    )

    return np.asarray(action_chunks[best_idx][:exec_horizon]), selection_record


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

    can_switch = sde_policy is not None
    if not can_switch:
        logging.warning(
            "SDE policy missing; dual head will still report scores but "
            "rescue actions will not be executed."
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
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()

            input("Press enter to start episode")
            task_time = time.time()
            ros_operator.follower_arm_publish_continuous(left0, right0)

            rescue_log: list = []
            dual_head_log: list = []
            value_selections: list = []
            collected_frames: list = []
            collected_left_frames: list = []
            collected_right_frames: list = []
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
                completion_guard=args.rescue_completion_guard,
            )

            task_segment = task_description.replace(" ", "_")
            rollout_dir = output_dir / f"rollout_{task_segment}_ep{episode_idx}_running"
            rollout_dir.mkdir(parents=True, exist_ok=True)

            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            action_buffer_base_t = -chunk_size  # forces ODE replan on first step
            action_buffer_source = "ode"
            last_rescue_check_t = -rescue_check_stride
            last_rescue_trigger_t = -10 ** 9
            rescue_submitted_count = 0
            last_published_action: np.ndarray | None = None
            latest_dual_progress: float = 0.0
            latest_dual_success: float = 0.0
            latest_dual_reasons: list = []
            latest_dual_tf: float = 0.0
            latest_dual_valid: bool = False
            rescue_active_until_frame: int = -1
            user_stopped = False
            t = 0

            dagger = (
                DaggerController(args.dagger_step_xyz, args.dagger_arm)
                if args.dagger_mode else None
            )
            dagger_log: list = []
            sde_chunk_log: list = []  # (t_start, source, chunk array)

            # ---- Rescue execution mode:
            #      * dagger_mode=True  → ASYNC (worker thread). Publish loop
            #        stays responsive so the operator can take over with
            #        keys while DreamDojo is computing.
            #      * dagger_mode=False → SYNC (blocking call). Arm pauses
            #        while DreamDojo scores the candidates, then injects on
            #        the same iteration. Default.
            rescue_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rescue",
            )
            pending_rescue: dict | None = None
            # Counter for chained manual rescues: when 's' is pressed and
            # --manual_rescue_repeat > 1, we auto-retrigger another rescue
            # right after the previous chunk finishes playing.
            manual_rescue_remaining: int = 0
            # Hold window after pressing 's': keeps the arm at the action
            # that was being published when the rescue was submitted, so the
            # operator has a few seconds to react and (optionally) take over
            # with DAgger keys before the SDE chunk gets injected.
            s_hold_until_t = -1
            s_hold_action: np.ndarray | None = None
            s_hold_steps = max(0, int(round(args.s_hold_seconds * args.publish_rate)))
            # Extra publish-step gap inserted between chained manual rescues
            # so the arm has time to settle and the camera sees the result of
            # the previous chunk before the next one is computed.
            manual_chain_gap_steps = max(
                0, int(round(args.manual_rescue_chain_pause_sec * args.publish_rate))
            )

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                manual_rescue_pressed = False
                dagger_keys: list = []
                events = drain_keyboard_events()
                for key in events:
                    if key == " ":
                        result = handle_interactive_mode(task_time)
                        if result == "reset":
                            ros_operator.follower_arm_publish_continuous(left0, right0)
                            user_stopped = True
                            break
                        elif result == "quit":
                            user_stopped = True
                            return
                    elif key == "s":
                        manual_rescue_pressed = True
                        # The current press counts as the first rescue; queue
                        # up the rest (if --manual_rescue_repeat > 1) so they
                        # auto-fire after each previous chunk plays out.
                        manual_rescue_remaining = max(0, int(args.manual_rescue_repeat) - 1)
                        if manual_rescue_remaining > 0:
                            print(
                                f"\n\033[93m>>> [System] Key 's' detected! Manually triggering "
                                f"SDE + DreamDojo rescue mechanism! "
                                f"(will chain {manual_rescue_remaining} more rescue(s) after this one)\033[0m",
                                flush=True,
                            )
                        else:
                            print("\n\033[93m>>> [System] Key 's' detected! Manually triggering SDE + DreamDojo rescue mechanism!\033[0m", flush=True)
                    elif args.dagger_mode and key in DAGGER_KEYS:
                        dagger_keys.append(key)
                if user_stopped:
                    break

                # Auto-chain follow-up manual rescues: fire only once the
                # previous rescue chunk has fully played out (plus optional
                # observation-settle gap) and there is no in-flight DreamDojo
                # job. Each follow-up uses a fresh obs_snapshot captured
                # later in this same loop iteration.
                if (
                    not manual_rescue_pressed
                    and manual_rescue_remaining > 0
                    and pending_rescue is None
                    and frame_counter > rescue_active_until_frame + manual_chain_gap_steps
                ):
                    manual_rescue_pressed = True
                    manual_rescue_remaining -= 1
                    print(
                        f"\n\033[93m>>> [System] Auto-chaining manual rescue "
                        f"({manual_rescue_remaining} more after this)\033[0m",
                        flush=True,
                    )

                # Capture current frame for rollout video + clip buffer.
                front_msg = (
                    ros_operator.front_image_queue[-1]
                    if len(ros_operator.front_image_queue) > 0 else None
                )
                if front_msg is not None:
                    front_img = ros_operator.bridge.imgmsg_to_cv2(front_msg, "passthrough")
                    right_msg = (
                        ros_operator.right_image_queue[-1]
                        if len(ros_operator.right_image_queue) > 0 else None
                    )
                    left_msg = (
                        ros_operator.left_image_queue[-1]
                        if len(ros_operator.left_image_queue) > 0 else None
                    )
                    right_img = (
                        ros_operator.bridge.imgmsg_to_cv2(right_msg, "passthrough")
                        if right_msg is not None else None
                    )
                    left_img = (
                        ros_operator.bridge.imgmsg_to_cv2(left_msg, "passthrough")
                        if left_msg is not None else None
                    )
                    frame_counter += 1
                    rescue_active_now = frame_counter <= rescue_active_until_frame
                    hud_frame = np.ascontiguousarray(np.asarray(front_img).copy())
                    if args.show_live_window:
                        try:
                            cv2.imshow("dualhead", hud_frame[..., ::-1])
                            cv2.waitKey(1)
                        except Exception as e:
                            logging.warning(f"[HUD] cv2.imshow failed: {e}")
                    collected_frames.append(hud_frame)
                    h, w = hud_frame.shape[:2]
                    if left_img is not None:
                        collected_left_frames.append(
                            np.ascontiguousarray(np.asarray(left_img).copy())
                        )
                    else:
                        collected_left_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                    if right_img is not None:
                        collected_right_frames.append(
                            np.ascontiguousarray(np.asarray(right_img).copy())
                        )
                    else:
                        collected_right_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
                    if clip_buffer is not None and right_img is not None and left_img is not None:
                        clip_buffer.update(front_img, right_img, left_img)

                # ----- Rescue check at fixed interval (independent of chunk boundary) -----
                # While the human is in DAgger override, skip rescue entirely:
                # we don't want to bother dual-head / DreamDojo while the
                # operator is steering the arm. We also skip while a previous
                # rescue is still in-flight on the worker thread — only one
                # outstanding submission at a time.
                rescue_triggered = False
                dagger_engaged = dagger is not None and dagger.active
                rescue_in_flight = (
                    pending_rescue is not None
                    and not pending_rescue["future"].done()
                )
                rescue_cap_reached = (
                    args.max_rescues_per_episode > 0
                    and rescue_submitted_count >= args.max_rescues_per_episode
                )
                should_check_auto = (
                    can_switch
                    and not dagger_engaged
                    and not rescue_in_flight
                    and not rescue_cap_reached
                    and frame_counter > 0
                    and (t - last_rescue_check_t) >= rescue_check_stride
                    and (t - last_rescue_trigger_t) >= rescue_cooldown_steps
                )
                if rescue_cap_reached and manual_rescue_pressed:
                    print(
                        f"\n\033[93m>>> [Rescue] cap reached "
                        f"({rescue_submitted_count}/{args.max_rescues_per_episode}) — "
                        f"'s' ignored.\033[0m",
                        flush=True,
                    )
                    manual_rescue_pressed = False
                if dagger_engaged and manual_rescue_pressed:
                    print(
                        "\n\033[96m>>> [DAgger] 's' ignored — release with 'r' "
                        "to re-enable rescue.\033[0m",
                        flush=True,
                    )
                    manual_rescue_pressed = False
                if rescue_in_flight and manual_rescue_pressed:
                    print(
                        "\n\033[93m>>> [Rescue] DreamDojo still running from a "
                        "previous trigger — 's' ignored.\033[0m",
                        flush=True,
                    )
                    manual_rescue_pressed = False

                if can_switch and (should_check_auto or manual_rescue_pressed):
                    if should_check_auto:
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

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _dh_t0 = time.perf_counter()
                    if clip_buffer is not None and clip_buffer.ready():
                        top_c, right_c, left_c = clip_buffer.clips()
                        progress_p, success_p = dual_head.predict(
                            top_c, right_c, left_c, switch_state,
                        )
                        _dh_mode = f"clip(N={len(top_c)})"
                    else:
                        progress_p, success_p = dual_head.predict(
                            obs_snapshot["top"], obs_snapshot["right"],
                            obs_snapshot["left"], switch_state,
                        )
                        _dh_mode = "single"
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _dh_ms = (time.perf_counter() - _dh_t0) * 1000.0
                    print(
                        f"[DualHead] inference time: {_dh_ms:.2f} ms ({_dh_mode})",
                        flush=True,
                    )

                    time_fraction = t / max(max_publish_step - 1, 1)
                    auto_triggered, reasons, info = checker.add_and_check(
                        progress_p, success_p, time_fraction,
                    )

                    triggered = auto_triggered or manual_rescue_pressed
                    if manual_rescue_pressed:
                        reasons.append("manual_keyboard_override")

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

                    latest_dual_progress = float(progress_p)
                    latest_dual_success = float(success_p)
                    latest_dual_tf = float(time_fraction)
                    latest_dual_reasons = list(reasons)
                    latest_dual_valid = True

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
                        reasons_str = ",".join(reasons)
                        print(f"\n\033[93m>>> [System] Rescue Triggered (Progress: {progress_p:.3f}, Success: {success_p:.3f})!\033[0m", flush=True)
                        print(f"\033[93m>>> [System] Reasons: {reasons_str}\033[0m", flush=True)
                        if args.dagger_mode:
                            print(f"\033[93m>>> [System] DreamDojo running ASYNC ({args.num_sde_samples} candidates). Arm stays responsive — chunk will inject when ready.\033[0m\n", flush=True)
                            future = rescue_executor.submit(
                                _select_best_action_with_prefix,
                                obs_snapshot=obs_snapshot,
                                sde_policy=sde_policy,
                                exec_horizon=chunk_size,
                                task_description=task_description,
                                step_save_dir=step_save_dir,
                                dd_host=args.dd_host,
                                dd_base_port=args.dd_base_port,
                                num_samples=args.num_sde_samples,
                                prefix_action=last_published_action,
                                skip_steps=args.rescue_skip_sde_steps,
                                dd_action_stride=args.dd_action_stride,
                                dd_action_chunk_in=args.dd_action_chunk_in,
                            )
                        else:
                            print(f"\033[93m>>> [System] DreamDojo running SYNC ({args.num_sde_samples} candidates). Arm pausing until candidates scored...\033[0m\n", flush=True)
                            future = concurrent.futures.Future()
                            try:
                                _sync_result = _select_best_action_with_prefix(
                                    obs_snapshot=obs_snapshot,
                                    sde_policy=sde_policy,
                                    exec_horizon=chunk_size,
                                    task_description=task_description,
                                    step_save_dir=step_save_dir,
                                    dd_host=args.dd_host,
                                    dd_base_port=args.dd_base_port,
                                    num_samples=args.num_sde_samples,
                                    prefix_action=last_published_action,
                                    skip_steps=args.rescue_skip_sde_steps,
                                    dd_action_stride=args.dd_action_stride,
                                    dd_action_chunk_in=args.dd_action_chunk_in,
                                )
                                future.set_result(_sync_result)
                            except Exception as _sync_err:
                                future.set_exception(_sync_err)
                        pending_rescue = {
                            "future": future,
                            "t_submit": t,
                            "frame_submit": frame_counter,
                            "progress": float(progress_p),
                            "success": float(success_p),
                            "reasons": list(reasons),
                        }
                        # Cooldown is gated from submission time so we don't
                        # double-submit while DreamDojo is still working.
                        last_rescue_trigger_t = t
                        rescue_submitted_count += 1
                        if args.max_rescues_per_episode > 0:
                            print(
                                f"\033[93m>>> [Rescue] submitted "
                                f"{rescue_submitted_count}/{args.max_rescues_per_episode} "
                                f"for this episode.\033[0m",
                                flush=True,
                            )

                        # s-hold only matters in dagger_mode (async). It
                        # freezes the arm so the operator has time to take
                        # over with keys while DreamDojo is computing. In
                        # sync mode the arm already paused during the call
                        # so there's nothing to hold.
                        if (
                            args.dagger_mode
                            and manual_rescue_pressed
                            and s_hold_steps > 0
                            and last_published_action is not None
                        ):
                            s_hold_until_t = t + s_hold_steps
                            s_hold_action = np.asarray(last_published_action).copy()
                            print(
                                f"\033[93m>>> [System] Arm holding for "
                                f"~{args.s_hold_seconds:.1f}s — press i/k/j/l/u/d "
                                f"now to take over.\033[0m",
                                flush=True,
                            )

                # ----- Drain a completed background rescue (sync injection) -----
                in_s_hold = t < s_hold_until_t and not (dagger is not None and dagger.active)
                if (
                    pending_rescue is not None
                    and pending_rescue["future"].done()
                    and not (dagger is not None and dagger.active)
                    and not in_s_hold
                ):
                    pr = pending_rescue
                    pending_rescue = None
                    try:
                        best_actions, sel_record = pr["future"].result()
                    except Exception as e:
                        logging.error(f"[Rescue] async DreamDojo failed: {e}")
                        best_actions, sel_record = None, None

                    if best_actions is not None and sel_record is not None:
                        best_score = sel_record.get('best_score', 'N/A')
                        if isinstance(best_score, float):
                            best_score_str = f"{best_score:.4f}"
                        else:
                            best_score_str = str(best_score)
                        wait_steps = t - pr["t_submit"]
                        print(
                            f"\n\033[92m>>> [System] DreamDojo done after "
                            f"{wait_steps} publish steps "
                            f"(best_idx={sel_record.get('best_idx')}, "
                            f"score={best_score_str}). Injecting now.\033[0m\n",
                            flush=True,
                        )

                        sel_record["t"] = int(t)
                        sel_record["frame"] = int(frame_counter)
                        sel_record["t_submit"] = int(pr["t_submit"])
                        sel_record["frame_submit"] = int(pr["frame_submit"])
                        sel_record["progress"] = pr["progress"]
                        sel_record["success"] = pr["success"]
                        sel_record["reasons"] = pr["reasons"]
                        json_safe_sel = {
                            k: v for k, v in sel_record.items()
                            if k not in ("all_candidates",)
                        }
                        value_selections.append(json_safe_sel)

                        rescue_log.append({
                            "t": int(t),
                            "frame": int(frame_counter),
                            "t_submit": int(pr["t_submit"]),
                            "progress": pr["progress"],
                            "success": pr["success"],
                            "reasons": pr["reasons"],
                        })

                        best_actions = np.asarray(best_actions)
                        L = min(best_actions.shape[0], chunk_size)
                        action_buffer = np.zeros_like(action_buffer)
                        best_actions = best_actions.astype(action_buffer.dtype, copy=False)
                        K = int(max(0, args.rescue_blend_steps))
                        # Blending only makes sense when the most recent
                        # publish was a joint/eef action of the same shape;
                        # during/after DAgger that may be an EEF pose with
                        # different semantics, so guard the shape match.
                        can_blend = (
                            K > 0
                            and last_published_action is not None
                            and L > 0
                            and last_published_action.shape == best_actions[0].shape
                        )
                        if can_blend:
                            offset = (
                                last_published_action.astype(action_buffer.dtype)
                                - best_actions[0]
                            )
                            ramp = np.maximum(
                                0.0, 1.0 - np.arange(L, dtype=np.float64) / float(K)
                            )
                            action_buffer[:L] = best_actions[:L] + offset[None, :] * ramp[:, None]
                            max_jump = float(np.max(np.abs(action_buffer[0] - last_published_action)))
                            logging.info(
                                f"[Rescue] Blended first {min(K, L)} steps "
                                f"(max |Δ| step0 -> last = {max_jump:.4f})"
                            )
                        else:
                            action_buffer[:L] = best_actions[:L]
                        action_buffer_base_t = t
                        action_buffer_source = "sde"
                        rescue_active_until_frame = frame_counter + chunk_size - 1
                        sde_chunk_log.append({
                            "t_start": int(t),
                            "frame_start": int(frame_counter),
                            "actions": np.asarray(action_buffer[:L]).copy(),
                            "best_idx": int(sel_record.get("best_idx", -1)),
                            "best_score": sel_record.get("best_score", None),
                            "all_candidates": [
                                np.asarray(c, dtype=np.float32)
                                for c in sel_record.get("all_candidates", [])
                            ],
                            "all_candidate_scores": list(
                                sel_record.get("all_candidate_scores", [])
                            ),
                            "reasons": list(pr["reasons"]),
                            "progress": float(pr["progress"]),
                            "success": float(pr["success"]),
                        })
                        rescue_triggered = True
                        logging.info(
                            f"[Rescue] Injected {L} SDE actions at t={t} "
                            f"(submitted at t={pr['t_submit']})"
                        )

                # ----- ODE replan when action buffer exhausted (and no rescue just fired) -----
                # Skip the replan during the s-hold window so the hold pose
                # isn't immediately overwritten by a fresh ODE chunk.
                if (
                    not rescue_triggered
                    and not in_s_hold
                    and (t - action_buffer_base_t) >= chunk_size
                ):
                    actions = inference_fn_sync(args, config, policy, ros_operator)
                    assert actions is not None, "Sync inference returned None"
                    assert actions.shape[0] >= chunk_size, (
                        f"Action chunk length {actions.shape[0]} is smaller than {chunk_size}"
                    )
                    action_buffer = actions[:chunk_size]
                    action_buffer_base_t = t
                    action_buffer_source = "ode"

                # ----- Optional DAgger: drive the EEF from the keyboard -----
                dagger_override_used = False
                if dagger is not None:
                    if dagger_keys and not dagger.active:
                        # Snapshot a fresh observation so the user starts from
                        # the live arm pose, not a stale rescue-time snapshot.
                        update_observation_window(args, config, ros_operator)
                        with observation_window_lock:
                            base_eef = np.asarray(
                                observation_window[-1]["eef_pose"], dtype=np.float64
                            ).copy()
                        dagger.begin(base_eef)
                        # If the operator started DAgger inside an s-hold
                        # window, cancel the hold so the arm doesn't snap
                        # back to the pre-DAgger snapshot after release.
                        s_hold_until_t = -1
                        s_hold_action = None
                        print(
                            f"\n\033[96m>>> [DAgger] Manual EEF override engaged on '{dagger.arm}' arm. "
                            f"Keys: i/k = x+/x-, j/l = y+/y-, u/d = z+/z- "
                            f"(arrow keys also work if terminal supports them); "
                            f"'r' to release.\033[0m",
                            flush=True,
                        )
                    consumed_release = False
                    for k in dagger_keys:
                        if not dagger.active:
                            # 'r' before begin() is a no-op; arrows fall through
                            continue
                        if k == "r":
                            consumed_release = True
                        dagger.apply(k)
                        print(
                            f"\033[96m[DAgger] key={k} offset="
                            f"({dagger.offset[0]:+.3f}, {dagger.offset[1]:+.3f}, "
                            f"{dagger.offset[2]:+.3f})\033[0m",
                            flush=True,
                        )
                    if consumed_release:
                        print(
                            "\n\033[96m>>> [DAgger] Released. Resuming policy actions.\033[0m",
                            flush=True,
                        )

                # ----- Execute action from buffer (or DAgger override) -----
                # Re-evaluate the hold flag in case DAgger started this iter,
                # in which case we want the user to drive instead of holding.
                in_s_hold = (
                    t < s_hold_until_t
                    and not (dagger is not None and dagger.active)
                )
                idx_in_chunk = t - action_buffer_base_t
                idx_in_chunk = max(0, min(idx_in_chunk, chunk_size - 1))
                if in_s_hold and s_hold_action is not None:
                    act = np.asarray(s_hold_action).copy()
                else:
                    act = action_buffer[idx_in_chunk]

                if dagger is not None and dagger.active:
                    target_eef = dagger.target_eef()
                    left_action, right_action = process_action(config["task"], target_eef)
                    ros_operator.follower_arm_pose_publish(left_action, right_action)
                    last_published_action = np.asarray(target_eef).copy()
                    dagger_override_used = True
                    dagger_log.append({
                        "t": int(t),
                        "frame": int(frame_counter),
                        "policy_action": np.asarray(act, dtype=np.float64).tolist(),
                        "policy_source": action_buffer_source,
                        "executed_eef": target_eef.tolist(),
                        "offset": dagger.offset.tolist(),
                        "arm": dagger.arm,
                    })
                elif args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.follower_arm_publish(left_action, right_action)
                    last_published_action = np.asarray(act).copy()
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    ros_operator.follower_arm_pose_publish(left_action, right_action)
                    last_published_action = np.asarray(act).copy()

                t += 1
                if dagger_override_used:
                    tag = "DAGGER"
                elif in_s_hold:
                    tag = "S_HOLD"
                else:
                    tag = action_buffer_source.upper()
                print(f"[Step {t:4d}] Published [{tag}] (buf_idx={idx_in_chunk}/{chunk_size})")
                rate.sleep()

            # Drop any in-flight rescue without blocking — DAgger may have
            # ended the episode while DreamDojo was still running.
            if pending_rescue is not None:
                pending_rescue["future"].cancel()
                pending_rescue = None
            rescue_executor.shutdown(wait=False, cancel_futures=True)

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

            if args.dagger_mode and (dagger_log or sde_chunk_log):
                _save_dagger_dataset(
                    final_dir, task_description, suffix,
                    dagger_log, sde_chunk_log,
                    human_score=float(args.dagger_human_score),
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
            if collected_left_frames:
                annotated_left = _annotate_rescue_frames(
                    collected_left_frames, rescue_log, window=chunk_size,
                )
                imageio.mimwrite(
                    str(final_dir / "complete_video_left.mp4"),
                    annotated_left,
                    fps=ROLLOUT_FPS,
                )
            if collected_right_frames:
                annotated_right = _annotate_rescue_frames(
                    collected_right_frames, rescue_log, window=chunk_size,
                )
                imageio.mimwrite(
                    str(final_dir / "complete_video_right.mp4"),
                    annotated_right,
                    fps=ROLLOUT_FPS,
                )

            logging.info(f"Episode {episode_idx} finished: {suffix}")
            logging.info(f"Rescue activations: {len(rescue_log)}")
            episode_idx += 1
            ros_operator.follower_arm_publish_continuous(left0, right0)

    finally:
        ros_operator.follower_arm_publish_continuous(left0, right0)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


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
    active_reasons: list = []
    for i, frm in enumerate(frames):
        frame_no = i + 1
        while ev_idx < len(events) and int(events[ev_idx]["frame"]) <= frame_no:
            active_until = int(events[ev_idx]["frame"]) + window - 1
            active_reasons = list(events[ev_idx].get("reasons", []))
            ev_idx += 1

        img = np.ascontiguousarray(np.asarray(frm))
        if frame_no <= active_until:
            h, w = img.shape[:2]

            is_manual = "manual_keyboard_override" in active_reasons
            color = (114, 255, 193) if is_manual else (0, 0, 255)
            prefix = "CRITICAL PHASE" if is_manual else "RESCUE"

            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 4)

            cv2.putText(img, prefix, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, prefix, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color, 2, cv2.LINE_AA)

            display_reasons = [r for r in active_reasons if r != "manual_keyboard_override"]
            if display_reasons:
                sub = ",".join(display_reasons)
                cv2.putText(img, sub, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, sub, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            color, 1, cv2.LINE_AA)
        out.append(img)
    return out


def _save_dagger_dataset(
    rollout_dir: pathlib.Path,
    task_description: str,
    suffix: str,
    dagger_log: list,
    sde_chunk_log: list,
    human_score: float = 0.0,
):
    """Persist DAgger pairs (policy vs human action) and SDE rescue chunks.

    The two artifacts are written into ``rollout_dir/dagger/``:
      * ``pairs.npz`` — per-step DAgger overrides. Arrays are aligned by index
        and indicate (policy_action, executed_eef, offset, t, frame).
      * ``sde_chunks.npz`` — every rescue-time SDE chunk that was injected
        into the buffer. Useful for DreamDojo / value-expert finetuning.
      * ``meta.json`` — a small JSON sidecar with the task language, episode
        outcome, and per-event indices so loaders can match images with
        actions.
    """
    out_dir = rollout_dir / "dagger"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dagger_log:
        n = len(dagger_log)
        np.savez_compressed(
            out_dir / "pairs.npz",
            t=np.asarray([e["t"] for e in dagger_log], dtype=np.int64),
            frame=np.asarray([e["frame"] for e in dagger_log], dtype=np.int64),
            policy_action=np.asarray(
                [e["policy_action"] for e in dagger_log], dtype=np.float32
            ),
            executed_eef=np.asarray(
                [e["executed_eef"] for e in dagger_log], dtype=np.float32
            ),
            offset=np.asarray(
                [e["offset"] for e in dagger_log], dtype=np.float32
            ),
            # Pseudo value-expert label for every human-teleop step. Lower is
            # better in the existing expert's convention (value = 1 - progress);
            # default 0.0 marks the human as "best".
            human_score=np.full(n, float(human_score), dtype=np.float32),
        )

    if sde_chunk_log:
        # For each rescue event we save every SDE candidate + its DreamDojo
        # value-expert score, plus the chunk that was actually executed.
        # Variable-length / per-event ragged data → object arrays.
        chosen = np.empty(len(sde_chunk_log), dtype=object)
        all_cands = np.empty(len(sde_chunk_log), dtype=object)
        all_scores = np.empty(len(sde_chunk_log), dtype=object)
        for i, e in enumerate(sde_chunk_log):
            chosen[i] = np.asarray(e["actions"], dtype=np.float32)
            cand_arr = np.asarray(e.get("all_candidates", []), dtype=np.float32)
            all_cands[i] = cand_arr  # shape (num_samples, T, action_dim)
            scores = e.get("all_candidate_scores", [])
            all_scores[i] = np.asarray(
                [np.nan if s is None else float(s) for s in scores],
                dtype=np.float32,
            )
        best_scores = np.asarray(
            [
                np.nan if e.get("best_score") in (None, float("inf"))
                else float(e["best_score"])
                for e in sde_chunk_log
            ],
            dtype=np.float32,
        )
        np.savez_compressed(
            out_dir / "sde_chunks.npz",
            chosen_actions=chosen,
            all_candidate_actions=all_cands,
            all_candidate_scores=all_scores,
            best_idx=np.asarray(
                [e["best_idx"] for e in sde_chunk_log], dtype=np.int64,
            ),
            best_score=best_scores,
            t_start=np.asarray([e["t_start"] for e in sde_chunk_log], dtype=np.int64),
            frame_start=np.asarray(
                [e["frame_start"] for e in sde_chunk_log], dtype=np.int64,
            ),
            progress=np.asarray(
                [e.get("progress", np.nan) for e in sde_chunk_log], dtype=np.float32,
            ),
            success=np.asarray(
                [e.get("success", np.nan) for e in sde_chunk_log], dtype=np.float32,
            ),
        )

    meta = {
        "task": task_description,
        "outcome": suffix,
        "num_dagger_steps": len(dagger_log),
        "num_sde_chunks": len(sde_chunk_log),
        "human_score": float(human_score),
        "score_convention": "value = 1 - mean(progress); LOWER is better",
        "dagger_arms": sorted({e["arm"] for e in dagger_log}) if dagger_log else [],
        "policy_sources": sorted(
            {e.get("policy_source", "?") for e in dagger_log}
        ) if dagger_log else [],
        "dagger_event_indices": [
            {"t": e["t"], "frame": e["frame"]} for e in dagger_log
        ],
        "sde_event_indices": [
            {"t_start": e["t_start"], "frame_start": e["frame_start"]}
            for e in sde_chunk_log
        ],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(
        f"[DAgger] Saved {len(dagger_log)} pairs and {len(sde_chunk_log)} "
        f"SDE chunks to {out_dir}"
    )


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
    parser.add_argument("--max_rescues_per_episode", type=int, default=0,
                        help="Hard cap on rescue submissions per episode. 0 = unlimited "
                             "(default). Set to 1 to fire DreamDojo at most once per "
                             "episode; further triggers (auto or manual) are silently ignored.")
    parser.add_argument("--rescue_blend_steps", type=int, default=5,
                        help="Linearly blend the first K SDE actions toward the "
                             "last published action so the injection is C0 "
                             "continuous. 0 disables blending.")
    parser.add_argument("--rescue_skip_sde_steps", type=int, default=0,
                        help="Hold the last published action for the first N "
                             "steps of every SDE candidate chunk (both for "
                             "DreamDojo scoring and for execution) to suppress "
                             "the 'rescue swings home first' artifact. 0 keeps "
                             "the raw SDE chunk head.")
    parser.add_argument("--dd_action_stride", type=int, default=4,
                        help="Stride applied to the pi05 action chunk before "
                             "sending to DreamDojo. Default 4 matches the "
                             "new_agilex_3view training (timestep_interval=4 "
                             "over 30fps native = 7.5fps). Set to 1 to send "
                             "raw pi05 actions (training-distribution mismatch).")
    parser.add_argument("--dd_action_chunk_in", type=int, default=13,
                        help="Number of strided actions to send to DreamDojo "
                             "per candidate. Default 13 = num_action_per_chunk "
                             "+ 1 expected by the model's grouped-delta loop.")
    parser.add_argument("--show_live_window", action="store_true", default=False,
                        help="Open a cv2 window that shows the rollout frame "
                             "with the latest dual-head p/s scores overlaid "
                             "in real time. Requires a display ($DISPLAY).")
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
    parser.add_argument("--rescue_completion_guard", type=float, default=0.85,
                        help="If max(progress, success) >= this value at check "
                             "time, suppress all rescue conditions. Prevents "
                             "end-of-episode false triggers once the head "
                             "thinks the task is basically done. Set to 1.01 "
                             "to disable the guard.")
    # ---- DreamDojo ----
    parser.add_argument("--dd_host", type=str, default="127.0.0.1")
    parser.add_argument("--dd_base_port", type=int, default=8020)
    parser.add_argument("--num_sde_samples", type=int, default=4)
    # ---- DAgger keyboard teleop ----
    parser.add_argument("--dagger_mode", action="store_true", default=False,
                        help="Enable keyboard teleop overrides for the active "
                             "arm. Letter keys (preferred — work on every "
                             "terminal): i/k = x+/x-, j/l = y+/y-, u/d = z+/z-. "
                             "Arrow keys (←→↑↓) are also accepted when the "
                             "terminal sends standard ANSI sequences. Press "
                             "'r' to release back to the policy. SDE chunks "
                             "and per-step (policy_action, executed_eef) "
                             "pairs are saved under <rollout>/dagger/ for "
                             "later finetuning.")
    parser.add_argument("--dagger_arm", type=str, choices=["left", "right"],
                        default="right",
                        help="Which arm the DAgger keys steer (default: right)")
    parser.add_argument("--dagger_step_xyz", type=float, default=0.01,
                        help="Per-keypress translation delta in metres "
                             "applied to the active arm's x/y/z target.")
    parser.add_argument("--dagger_human_score", type=float, default=0.0,
                        help="Pseudo value-expert label assigned to every "
                             "human-teleoperated step. The expert is trained "
                             "with `value = 1 - mean(progress)`, so LOWER is "
                             "better; 0.0 marks the human action as 'best' "
                             "(default). Pass 1.0 if you want to flip the "
                             "convention.")
    parser.add_argument("--s_hold_seconds", type=float, default=3.0,
                        help="When 's' is pressed, freeze the arm at the "
                             "currently published action for this many "
                             "seconds before any SDE chunk is allowed to "
                             "inject. Gives the operator time to take over "
                             "with DAgger keys. Auto rescues do not freeze. "
                             "Set to 0 to disable.")
    parser.add_argument("--manual_rescue_repeat", type=int, default=2,
                        help="Number of consecutive rescue chunks to launch "
                             "from a single 's' keypress. Each follow-up "
                             "rescue is queued automatically right after the "
                             "previous chunk finishes playing, using a fresh "
                             "observation snapshot. 1 = legacy single-shot.")
    parser.add_argument("--manual_rescue_chain_pause_sec", type=float, default=0.0,
                        help="Wait this many seconds AFTER the previous "
                             "rescue chunk finishes publishing before "
                             "launching the next chained manual rescue. "
                             "Useful when the arm needs a beat to settle so "
                             "the next obs_snapshot reflects steady state. "
                             "0 = chain immediately (legacy).")

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
