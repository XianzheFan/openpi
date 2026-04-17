"""
Two-phase pipeline for training a switch head on Agilex LeRobot v2.1 data
with pre-computed Robometer reward scores.

Phase 1 – pack:
  Read a LeRobot v2.1 dataset that already has Robometer scores under
  <data_dir>/rewards/ (episode_XXXXXX_progress.npy / _success.npy).
  Extract video frames, load states/actions from parquet, convert
  progress/success scores into binary switch labels, and save each
  replan step as an .npz file ready for training.

Phase 2 – train:
  Train a standalone DINOv2-based binary classifier on the packed data.

Usage
-----
  # 1) Pack existing Robometer scores into training .npz files
  python train_switch_head_robometer.py pack \
      --data_dir data/pnp_cup_0415 \
      --output_dir data/agilex_switch_labels

  # 2) Train standalone switch head
  python train_switch_head_robometer.py train \
      --data_dir data/agilex_switch_labels \
      --output_dir checkpoints/switch_head_robometer \
      --epochs 30

  # 2b) Multi-GPU training via torchrun
  torchrun --nproc_per_node=8 train_switch_head_robometer.py train \
      --data_dir data/agilex_switch_labels \
      --output_dir checkpoints/switch_head_robometer \
      --batch_size 8 --epochs 30
"""

import argparse
import glob
import json
import logging
import os
import pathlib
import random
import time

import av
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_FPS = 30

# Rescue trigger thresholds based on Robometer progress scores
# Progress score is 0→1 (0 = start, 1 = task complete)
RESCUE_PROGRESS_ABSOLUTE = 0.25     # rescue if progress stalls below this
RESCUE_PROGRESS_DROP = 0.05         # rescue if progress drops by this much
RESCUE_SUCCESS_THRESHOLD = 0.5      # rescue if success probability below this
RESCUE_PROGRESS_RISING = 0.02       # skip rescue if progress rose by this much recently

CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
CAMERA_TO_KEY = {
    "cam_high": "top",
    "cam_left_wrist": "left",
    "cam_right_wrist": "right",
}


# ============================================================================
#  Video frame extraction
# ============================================================================

def extract_frames_pyav(
    video_path: str, fps: float = 3.0, max_frames: int = 512,
) -> np.ndarray:
    """Extract frames from video using PyAV (supports AV1). Returns uint8 (T, H, W, C)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate)
    frame_interval = max(1, int(round(video_fps / fps)))

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % frame_interval == 0:
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    container.close()

    if not frames:
        return np.array([], dtype=np.uint8)
    return np.stack(frames, axis=0)


# ============================================================================
#  LeRobot v2.1 dataset helpers
# ============================================================================

def load_dataset_info(data_dir: pathlib.Path) -> dict:
    with open(data_dir / "meta" / "info.json") as f:
        return json.load(f)


def load_episodes(data_dir: pathlib.Path) -> list[dict]:
    episodes = []
    with open(data_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_task(data_dir: pathlib.Path) -> str:
    with open(data_dir / "meta" / "tasks.jsonl") as f:
        first_line = f.readline().strip()
        return json.loads(first_line)["task"]


def get_video_path(data_dir: pathlib.Path, episode_index: int, camera: str) -> pathlib.Path:
    video_key = f"observation.images.{camera}"
    return (
        data_dir / "videos" / "chunk-000" / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def load_episode_parquet(data_dir: pathlib.Path, episode_index: int):
    """Load parquet data for an episode. Returns (states, actions, frame_indices)."""
    import pandas as pd
    pq_path = data_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(pq_path)
    states = np.stack(df["observation.state"].values).astype(np.float32)
    actions = np.stack(df["action"].values).astype(np.float32)
    frame_indices = df["frame_index"].values.astype(np.int64)
    return states, actions, frame_indices


def progress_to_switch_labels(
    progress: np.ndarray,
    success: np.ndarray,
    replan_interval: int,
    progress_threshold: float = RESCUE_PROGRESS_ABSOLUTE,
    progress_drop: float = RESCUE_PROGRESS_DROP,
    success_threshold: float = RESCUE_SUCCESS_THRESHOLD,
    expected_progress_rate: float = 0.0,
) -> list[dict]:
    """
    Convert Robometer progress scores to switch labels at replan boundaries.

    A rescue is triggered when:
      1. Progress score is below progress_threshold (stalled/failed), OR
      2. Progress dropped by >= progress_drop compared to a recent window, OR
      3. Success probability is below success_threshold (guarded), OR
      4. Progress is far below expected linear trajectory (only if expected_progress_rate > 0).

    Guardrail on condition 3 only: Robometer's success probability lags behind
    actual progress (stays near 0 until the task is nearly complete). If progress
    is actively rising (>= RESCUE_PROGRESS_RISING over the lookback window), low
    success is not a reliable failure signal — suppress. Conditions 1, 2, 4 are
    real failure signals and are NOT suppressed.

    Returns list of dicts with frame_idx, switch_label, progress_score, success_prob.
    """
    T = len(progress)
    records = []

    for frame_i in range(0, T, replan_interval):
        p = float(progress[frame_i])
        s = float(success[frame_i]) if frame_i < len(success) else 0.5

        # Compute progress trend over a short lookback window
        lookback = max(1, replan_interval * 2)
        prev_idx = max(0, frame_i - lookback)
        prev_p = float(progress[prev_idx]) if prev_idx < len(progress) else p
        is_rising = (p - prev_p) >= RESCUE_PROGRESS_RISING

        should_rescue = False
        time_fraction = frame_i / max(T - 1, 1)

        # Condition 1: absolute low progress at this stage
        adaptive_threshold = progress_threshold + time_fraction * 0.3
        if p < adaptive_threshold and time_fraction > 0.15:
            should_rescue = True

        # Condition 2: progress drop (hard fail)
        if (prev_p - p) >= progress_drop:
            should_rescue = True

        # Condition 3: low success probability — SKIP if progress is rising
        if s < success_threshold and time_fraction > 0.1 and not is_rising:
            should_rescue = True

        # Condition 4: below expected linear progress
        if expected_progress_rate > 0:
            expected_p = time_fraction * expected_progress_rate
            if p < expected_p * 0.5 and time_fraction > 0.3:
                should_rescue = True

        records.append({
            "frame_idx": frame_i,
            "switch_label": 1.0 if should_rescue else 0.0,
            "progress_score": p,
            "success_prob": s,
        })

    return records


# ============================================================================
#  Phase 1: Pack pre-computed Robometer scores into training .npz files
# ============================================================================

def pack(args):
    """Pack pre-computed Robometer scores + video frames + parquet into .npz files."""
    data_dir = pathlib.Path(args.data_dir)
    rewards_dir = pathlib.Path(args.rewards_dir) if args.rewards_dir else data_dir / "rewards"
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(data_dir)
    episodes = load_episodes(data_dir)
    task = load_task(data_dir)
    total_episodes = info["total_episodes"]
    dataset_fps = info["fps"]

    # Discover which episodes have reward files
    available_eps = set()
    for f in rewards_dir.glob("episode_*_progress.npy"):
        ep_idx = int(f.stem.split("_")[1])
        available_eps.add(ep_idx)
    available_eps = sorted(available_eps)

    if args.episodes:
        ep_start, ep_end = map(int, args.episodes.split(":"))
        available_eps = [e for e in available_eps if ep_start <= e < ep_end]

    logging.info(f"Dataset: {data_dir}")
    logging.info(f"Rewards: {rewards_dir}")
    logging.info(f"Task: {task}")
    logging.info(f"Episodes with rewards: {len(available_eps)} / {total_episodes}")
    logging.info(f"Extract FPS: {args.extract_fps} (from {dataset_fps}fps video)")

    all_episode_meta = []
    stats = {"total_episodes": 0, "total_rescue_steps": 0, "total_normal_steps": 0}
    t_start = time.time()

    for ep_idx in available_eps:
        task_segment = task.replace(" ", "_")[:50]

        # Check if already packed
        existing = list(output_dir.glob(f"rollout_{task_segment}_ep{ep_idx}_done"))
        if existing:
            logging.info(f"  Skip: episode {ep_idx} (already packed)")
            continue

        # Load pre-computed Robometer scores
        progress_path = rewards_dir / f"episode_{ep_idx:06d}_progress.npy"
        success_path = rewards_dir / f"episode_{ep_idx:06d}_success.npy"
        progress_array = np.load(str(progress_path)).astype(np.float32)
        success_array = np.load(str(success_path)).astype(np.float32)

        # Extract frames from all 3 cameras
        camera_frames = {}
        for cam in CAMERA_NAMES:
            video_path = get_video_path(data_dir, ep_idx, cam)
            if not video_path.exists():
                logging.warning(f"  Episode {ep_idx}: {cam} video not found, skipping")
                break
            frames = extract_frames_pyav(
                str(video_path), fps=args.extract_fps, max_frames=args.max_frames,
            )
            if frames.size == 0:
                logging.warning(f"  Episode {ep_idx}: failed to extract {cam} frames")
                break
            camera_frames[cam] = frames

        if len(camera_frames) != len(CAMERA_NAMES):
            continue

        # T = min of all sources (cameras + reward arrays)
        T = min(
            min(f.shape[0] for f in camera_frames.values()),
            len(progress_array),
            len(success_array),
        )
        for cam in camera_frames:
            camera_frames[cam] = camera_frames[cam][:T]
        progress_array = progress_array[:T]
        success_array = success_array[:T]

        # Load state and action from parquet
        states, actions, frame_indices = load_episode_parquet(data_dir, ep_idx)
        state_interval = max(1, int(round(dataset_fps / args.extract_fps)))
        sampled_states = states[::state_interval][:T]
        sampled_actions = actions[::state_interval][:T]

        # Convert progress/success to switch labels at replan boundaries
        switch_records = progress_to_switch_labels(
            progress_array, success_array,
            replan_interval=args.replan_interval,
            progress_threshold=args.progress_threshold,
            progress_drop=args.progress_drop,
            success_threshold=args.success_threshold,
        )

        # Build full replan records with images
        replan_records = []
        for rec in switch_records:
            frame_i = rec["frame_idx"]
            if frame_i >= T:
                break

            clip_len = args.clip_len
            clips = {}
            for cam in CAMERA_NAMES:
                cam_key = CAMERA_TO_KEY[cam]
                cam_clip = list(camera_frames[cam][max(0, frame_i - clip_len + 1):frame_i + 1])
                if len(cam_clip) < clip_len:
                    pad_n = clip_len - len(cam_clip)
                    cam_clip = [cam_clip[0]] * pad_n + cam_clip
                clips[cam_key] = np.stack(cam_clip)

            action_end = min(frame_i + args.replan_interval, T)
            action_chunk = sampled_actions[frame_i:action_end]

            replan_records.append({
                "frame_idx": frame_i,
                "top": camera_frames["cam_high"][frame_i].copy(),
                "right": camera_frames["cam_right_wrist"][frame_i].copy(),
                "left": camera_frames["cam_left_wrist"][frame_i].copy(),
                "top_clip": clips["top"],
                "right_clip": clips["right"],
                "left_clip": clips["left"],
                "state": sampled_states[frame_i].copy() if frame_i < len(sampled_states) else np.zeros(14, dtype=np.float32),
                "actions": action_chunk,
                "switch_label": rec["switch_label"],
                "progress_score": rec["progress_score"],
                "success_prob": rec["success_prob"],
            })

        # Label shifting
        if args.label_shift_steps > 0:
            for i in range(len(replan_records)):
                if replan_records[i]["switch_label"] > 0.5:
                    for j in range(max(0, i - args.label_shift_steps), i):
                        replan_records[j]["switch_label"] = 1.0

        # Save .npz files
        n_rescue = sum(1 for r in replan_records if r["switch_label"] > 0.5)
        n_normal = len(replan_records) - n_rescue

        rollout_dir = output_dir / f"rollout_{task_segment}_ep{ep_idx}_done"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        for step_i, rec in enumerate(replan_records):
            save_path = rollout_dir / f"step_{step_i:04d}.npz"
            np.savez_compressed(
                save_path,
                top=rec["top"],
                right=rec["right"],
                left=rec["left"],
                top_clip=rec["top_clip"],
                right_clip=rec["right_clip"],
                left_clip=rec["left_clip"],
                state=rec["state"],
                actions=rec["actions"],
                switch_label=np.float32(rec["switch_label"]),
                progress_score=np.float32(rec["progress_score"]),
                success_prob=np.float32(rec["success_prob"]),
                clip_len=np.int32(args.clip_len),
                prompt=np.array(task),
                task=np.array(task),
                episode_idx=np.int32(ep_idx),
                frame_idx=np.int32(rec["frame_idx"]),
            )

        stats["total_episodes"] += 1
        stats["total_rescue_steps"] += n_rescue
        stats["total_normal_steps"] += n_normal

        episode_meta = {
            "episode_idx": ep_idx,
            "task": task,
            "num_steps": len(replan_records),
            "num_rescue": n_rescue,
            "progress_range": [
                float(progress_array.min()), float(progress_array.max()),
            ],
            "success_mean": float(success_array.mean()),
        }
        all_episode_meta.append(episode_meta)

        elapsed = time.time() - t_start
        eps_done = len(all_episode_meta)
        eps_per_sec = eps_done / elapsed
        remaining = (len(available_eps) - eps_done) / eps_per_sec if eps_per_sec > 0 else 0

        logging.info(
            f"  Episode {ep_idx}: steps={len(replan_records)} "
            f"rescue={n_rescue} normal={n_normal} | "
            f"progress=[{progress_array.min():.3f}, {progress_array.max():.3f}] "
            f"success_mean={success_array.mean():.3f} | ETA {remaining:.0f}s"
        )

    # Save metadata
    meta_path = output_dir / "collection_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"stats": stats, "episodes": all_episode_meta}, f, indent=2)

    total_samples = stats["total_rescue_steps"] + stats["total_normal_steps"]
    logging.info(f"\nPacking complete.")
    logging.info(f"  Total samples: {total_samples}")
    logging.info(f"  Rescue steps : {stats['total_rescue_steps']}")
    logging.info(f"  Normal steps : {stats['total_normal_steps']}")
    logging.info(f"  Saved to     : {output_dir}")


# ============================================================================
#  Phase 2: Training (DINOv2 switch head, 3-camera agilex)
# ============================================================================

class DINOv2SwitchHead(nn.Module):
    """
    DINOv2-based binary classifier for switch/intervention prediction.
    Supports 3-camera agilex data (top + right + left).
    """

    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        hidden_dim: int = 256,
        state_dim: int = 14,
        num_cameras: int = 3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras

        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.feature_dim = self.backbone.embed_dim
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        input_dim = num_cameras * self.feature_dim + state_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_image(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 5:
            B, T, C, H, W = img.shape
            img_flat = img.reshape(B * T, C, H, W)
        else:
            img_flat = img
            B = img.shape[0]
            T = 1

        img_flat = F.interpolate(img_flat, size=(224, 224), mode="bilinear", align_corners=False)
        img_flat = (img_flat - self.img_mean) / self.img_std

        if self.freeze_backbone:
            with torch.no_grad():
                feat_flat = self.backbone(img_flat)
        else:
            feat_flat = self.backbone(img_flat)

        if T > 1:
            feat = feat_flat.view(B, T, -1)
            return feat.mean(dim=1)
        return feat_flat

    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        feats = [self._encode_image(img) for img in images]
        combined = torch.cat(feats + [state], dim=-1)
        return self.classifier(combined).squeeze(-1)

    def predict_switch_prob(
        self, images: list[torch.Tensor], state: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(images, state))


class AgilexSwitchLabelDataset(Dataset):
    """
    Loads .npz files with 3-camera agilex data (top, right, left).
    Supports both Gemini-labeled and Robometer-labeled data.
    """

    IMAGE_KEYS = ["top", "right", "left"]

    def __init__(
        self,
        data_dir: str,
        use_clip: bool = True,
        use_soft_label: bool = False,
        files: list | None = None,
    ):
        if files is not None:
            self.files = sorted(files)
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, "rollout_*", "step_*.npz")))
            if not self.files:
                self.files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        logging.info(f"Found {len(self.files)} samples in {data_dir}")

        first = np.load(self.files[0], allow_pickle=True)
        self.state_dim = first["state"].shape[0]

        self.use_clip = use_clip
        clip_keys = [k for k in ["top_clip", "right_clip", "left_clip"] if k in first]
        self.has_clips = len(clip_keys) == 3
        if self.use_clip and not self.has_clips:
            logging.warning("  Clip keys not found, falling back to single frames")
            self.use_clip = False

        # Soft label: use progress_score (from Robometer) or gemini_score
        self.use_soft_label = False
        self.soft_label_key = None
        if use_soft_label:
            if "progress_score" in first:
                self.use_soft_label = True
                self.soft_label_key = "progress_score"
            elif "gemini_score" in first:
                self.use_soft_label = True
                self.soft_label_key = "gemini_score"
            else:
                logging.warning("  --use_soft_label requested but no soft score found")

        logging.info(
            f"  Agilex 3-camera data, state_dim={self.state_dim}, "
            f"clip={self.use_clip}, soft_label={self.use_soft_label}"
            + (f" (key={self.soft_label_key})" if self.soft_label_key else "")
        )

        # Count class balance
        labels = []
        valid_files = []
        for f in self.files:
            try:
                d = np.load(f)
                labels.append(float(d["switch_label"]))
                valid_files.append(f)
            except Exception:
                logging.warning(f"  Skipping corrupt file: {f}")
        self.files = valid_files
        self._labels = labels
        n_pos = sum(1 for l in labels if l > 0.5)
        n_neg = len(labels) - n_pos
        logging.info(f"  Class balance: {n_pos} rescue (pos) / {n_neg} normal (neg)")

    def __len__(self):
        return len(self.files)

    @property
    def pos_weight(self) -> float:
        n_pos = sum(1 for l in self._labels if l > 0.5)
        n_neg = len(self._labels) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        images = []
        for key in self.IMAGE_KEYS:
            if self.use_clip and f"{key}_clip" in data:
                clip = torch.from_numpy(
                    data[f"{key}_clip"].copy()
                ).permute(0, 3, 1, 2).float() / 255.0
                images.append(clip)
            else:
                img = torch.from_numpy(
                    data[key].copy()
                ).permute(2, 0, 1).float() / 255.0
                images.append(img)

        state = torch.from_numpy(data["state"].astype(np.float32))

        if self.use_soft_label and self.soft_label_key in data:
            score = float(data[self.soft_label_key])
            if self.soft_label_key == "progress_score":
                # Progress 0→1 maps to rescue probability 1→0
                label = torch.tensor(1.0 - score, dtype=torch.float32)
            else:
                # Gemini score: same mapping
                label = torch.tensor(1.0 - score, dtype=torch.float32)
        else:
            label = torch.tensor(float(data["switch_label"]), dtype=torch.float32)

        return images, state, label


def _split_train_val_by_rollout(
    data_dir: str,
    val_ratio: float,
    seed: int,
    use_clip: bool,
    use_soft_label: bool,
):
    """Split a packed dataset into train/val by rollout folder (no frame leakage)."""
    files = sorted(glob.glob(os.path.join(data_dir, "rollout_*", "step_*.npz")))
    if files:
        rollouts: dict[str, list[str]] = {}
        for f in files:
            rid = os.path.basename(os.path.dirname(f))
            rollouts.setdefault(rid, []).append(f)
        rollout_ids = sorted(rollouts.keys())
        rng = random.Random(seed)
        shuffled = rollout_ids[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        if n_val >= len(shuffled):
            raise ValueError(f"val_ratio={val_ratio} leaves no train rollouts")
        val_ids = set(shuffled[:n_val])
        train_files = [f for r in rollout_ids if r not in val_ids for f in rollouts[r]]
        val_files = [f for r in rollout_ids if r in val_ids for f in rollouts[r]]
        logging.info(
            f"Split by rollout (seed={seed}): "
            f"{len(rollout_ids) - n_val} train / {n_val} val rollouts "
            f"({len(train_files)} / {len(val_files)} steps)"
        )
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        rng = random.Random(seed)
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        if n_val >= len(shuffled):
            raise ValueError(f"val_ratio={val_ratio} leaves no train samples")
        val_files = sorted(shuffled[:n_val])
        train_files = sorted(shuffled[n_val:])
        logging.warning(
            "No rollout_*/step_*.npz found — splitting flat sample_*.npz by index "
            "(risk of frame leakage if samples come from same episode)."
        )
    train_ds = AgilexSwitchLabelDataset(
        data_dir, use_clip=use_clip, use_soft_label=use_soft_label, files=train_files,
    )
    val_ds = AgilexSwitchLabelDataset(
        data_dir, use_clip=use_clip, use_soft_label=use_soft_label, files=val_files,
    )
    return train_ds, val_ds


def _collate_switch(batch):
    images_list, states, labels = zip(*batch)
    num_cameras = len(images_list[0])
    batched_images = [torch.stack([img[c] for img in images_list]) for c in range(num_cameras)]
    return batched_images, torch.stack(states), torch.stack(labels)


def _setup_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    """Phase 2: Train the standalone switch head."""
    rank, local_rank, world_size = _setup_distributed()
    distributed = world_size > 1
    is_main = rank == 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        logging.info(f"Using device: {device}  (world_size={world_size})")

    if args.val_dir:
        train_dataset = AgilexSwitchLabelDataset(
            args.data_dir, use_clip=args.use_clip, use_soft_label=args.use_soft_label,
        )
        val_dataset = AgilexSwitchLabelDataset(
            args.val_dir, use_clip=args.use_clip, use_soft_label=args.use_soft_label,
        )
    elif args.val_ratio > 0:
        train_dataset, val_dataset = _split_train_val_by_rollout(
            args.data_dir,
            val_ratio=args.val_ratio,
            seed=args.val_split_seed,
            use_clip=args.use_clip,
            use_soft_label=args.use_soft_label,
        )
    else:
        train_dataset = AgilexSwitchLabelDataset(
            args.data_dir, use_clip=args.use_clip, use_soft_label=args.use_soft_label,
        )
        val_dataset = None

    state_dim = train_dataset.state_dim
    num_cameras = 3
    if is_main:
        logging.info(f"Training with {num_cameras}-camera agilex data, state_dim={state_dim}")

    model = DINOv2SwitchHead(
        dinov2_model=args.dinov2_model,
        hidden_dim=args.hidden_dim,
        state_dim=state_dim,
        num_cameras=num_cameras,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)
    raw_model = model.module if distributed else model

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate_switch,
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed and val_dataset else None
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=_collate_switch,
        )
        if val_dataset
        else None
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_soft_label = args.use_soft_label and train_dataset.use_soft_label
    if use_soft_label:
        if is_main:
            logging.info("Using MSE loss for soft-label regression")
        criterion = nn.MSELoss()
    else:
        pw = torch.tensor([train_dataset.pos_weight], device=device)
        if is_main:
            logging.info(f"BCE pos_weight: {pw.item():.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    output_dir = pathlib.Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    best_val_metric = -float("inf")

    use_wandb = is_main and wandb is not None and args.wandb_project
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "dinov2_model": args.dinov2_model,
                "hidden_dim": args.hidden_dim,
                "freeze_backbone": args.freeze_backbone,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "use_clip": args.use_clip,
                "use_soft_label": args.use_soft_label,
                "world_size": world_size,
                "data_dir": args.data_dir,
                "num_samples": len(train_dataset),
                "num_cameras": num_cameras,
                "state_dim": state_dim,
                "labeler": "robometer",
            },
        )

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        if args.freeze_backbone:
            backbone = raw_model.backbone
            backbone.eval()

        total_loss, num_batches = 0.0, 0
        train_correct, train_total = 0, 0

        for images, state, label in train_loader:
            images = [img.to(device) for img in images]
            state = state.to(device)
            label = label.to(device)

            logit = model(images, state)
            if use_soft_label:
                loss = criterion(torch.sigmoid(logit), label)
            else:
                loss = criterion(logit, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pred = (torch.sigmoid(logit) > 0.5).float()
            label_hard = (label > 0.5).float()
            train_correct += (pred == label_hard).sum().item()
            train_total += label.numel()

        avg_train_loss = total_loss / max(num_batches, 1)
        train_acc = train_correct / max(train_total, 1)
        scheduler.step()

        val_str = "N/A"
        if val_loader is not None:
            model.eval()
            vl, vn = 0.0, 0
            val_correct, val_total = 0, 0
            val_tp, val_fp, val_fn = 0, 0, 0

            with torch.no_grad():
                for images, state, label in val_loader:
                    images = [img.to(device) for img in images]
                    state = state.to(device)
                    label = label.to(device)

                    logit = model(images, state)
                    if use_soft_label:
                        vl += criterion(torch.sigmoid(logit), label).item()
                    else:
                        vl += criterion(logit, label).item()
                    vn += 1

                    pred = (torch.sigmoid(logit) > 0.5).float()
                    label_hard = (label > 0.5).float()
                    val_correct += (pred == label_hard).sum().item()
                    val_total += label.numel()
                    val_tp += ((pred == 1) & (label_hard == 1)).sum().item()
                    val_fp += ((pred == 1) & (label_hard == 0)).sum().item()
                    val_fn += ((pred == 0) & (label_hard == 1)).sum().item()

            avg_val_loss = vl / max(vn, 1)
            val_acc = val_correct / max(val_total, 1)
            val_precision = val_tp / max(val_tp + val_fp, 1)
            val_recall = val_tp / max(val_tp + val_fn, 1)
            val_f1 = 2 * val_precision * val_recall / max(val_precision + val_recall, 1e-8)

            val_str = (
                f"loss={avg_val_loss:.4f} acc={val_acc:.3f} "
                f"P={val_precision:.3f} R={val_recall:.3f} F1={val_f1:.3f}"
            )

            if is_main and val_f1 > best_val_metric:
                best_val_metric = val_f1
                torch.save(raw_model.state_dict(), output_dir / "best_model.pt")
                logging.info(f"  -> Saved best model (F1={val_f1:.4f})")

        if is_main:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f}  "
                f"val=[{val_str}]  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/acc": train_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            if val_loader is not None:
                log_dict.update({
                    "val/loss": avg_val_loss,
                    "val/acc": val_acc,
                    "val/precision": val_precision,
                    "val/recall": val_recall,
                    "val/f1": val_f1,
                })
            wandb.log(log_dict, step=epoch + 1)

        if is_main and (epoch + 1) % args.save_every == 0:
            torch.save(raw_model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    if is_main:
        torch.save(raw_model.state_dict(), output_dir / "model_final.pt")
        logging.info(f"Training complete. Models saved to {output_dir}")

    if use_wandb:
        wandb.finish()
    _cleanup_distributed()


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train switch head for Agilex with pre-computed Robometer scores (pack → train)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- pack ----
    p_pack = subparsers.add_parser(
        "pack", help="Pack pre-computed Robometer scores into training .npz files"
    )
    p_pack.add_argument("--data_dir", type=str, required=True,
                        help="Path to LeRobot v2.1 dataset (e.g. data/pnp_cup_0415)")
    p_pack.add_argument("--rewards_dir", type=str, default=None,
                        help="Path to Robometer rewards (default: <data_dir>/rewards)")
    p_pack.add_argument("--output_dir", type=str,
                        default="data/agilex_switch_labels")
    p_pack.add_argument("--extract_fps", type=float, default=3.0,
                        help="FPS for frame extraction from video")
    p_pack.add_argument("--max_frames", type=int, default=512)
    p_pack.add_argument("--replan_interval", type=int, default=3,
                        help="Interval (in extracted frames) between replan steps")
    p_pack.add_argument("--clip_len", type=int, default=20,
                        help="Number of recent frames per camera for video clips")
    p_pack.add_argument("--label_shift_steps", type=int, default=0,
                        help="Shift rescue labels backward for anticipation")
    p_pack.add_argument("--episodes", type=str, default=None,
                        help="Episode range, e.g. '0:50' (default: all with rewards)")
    p_pack.add_argument("--progress_threshold", type=float,
                        default=RESCUE_PROGRESS_ABSOLUTE,
                        help="Base progress threshold for rescue trigger")
    p_pack.add_argument("--progress_drop", type=float,
                        default=RESCUE_PROGRESS_DROP,
                        help="Progress drop threshold for rescue trigger")
    p_pack.add_argument("--success_threshold", type=float,
                        default=RESCUE_SUCCESS_THRESHOLD,
                        help="Success probability threshold for rescue trigger")

    # ---- train ----
    p_train = subparsers.add_parser(
        "train", help="Train DINOv2-based switch head on labeled data"
    )
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--val_dir", type=str, default=None)
    p_train.add_argument("--val_ratio", type=float, default=0.0,
                         help="Hold out this fraction of rollouts as val if --val_dir not set")
    p_train.add_argument("--val_split_seed", type=int, default=42)
    p_train.add_argument("--output_dir", type=str,
                         default="checkpoints/switch_head_robometer")
    p_train.add_argument(
        "--dinov2_model", type=str, default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--freeze_backbone", action="store_true", default=True)
    p_train.add_argument("--no_freeze_backbone", dest="freeze_backbone",
                         action="store_false")
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--save_every", type=int, default=5)
    p_train.add_argument("--use_clip", action="store_true", default=True)
    p_train.add_argument("--no_clip", dest="use_clip", action="store_false")
    p_train.add_argument("--use_soft_label", action="store_true", default=False,
                         help="Regress Robometer progress scores instead of binary labels")
    p_train.add_argument("--wandb_project", type=str, default=None)
    p_train.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "pack":
        pack(args)
    elif args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
