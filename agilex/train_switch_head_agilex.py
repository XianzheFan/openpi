"""
Two-phase pipeline for training a switch head on Agilex LeRobot v2.1 data with Gemini labels.

Phase 1 – label:
  Read a LeRobot v2.1 dataset (3 cameras: cam_high, cam_left_wrist, cam_right_wrist),
  extract video frames, query Gemini for dense value scores, and produce binary
  switch labels.  Each replan step is saved as an .npz file.

Phase 2 – train:
  Train a standalone DINOv2-based binary classifier on the labeled data.

Phase 3 – export:
  Inject switch_label into the LeRobot v2 dataset for integrated training.

Usage
-----
  # 1) Label existing dataset with Gemini (needs GOOGLE_API_KEY set)
  python train_switch_head_agilex.py label \
      --data_dir data/pnp_cup_0415 \
      --camera cam_high \
      --output_dir data/agilex_switch_labels \
      --replan_interval 10

  # 2) Train standalone switch head (single GPU)
  python train_switch_head_agilex.py train \
      --data_dir data/agilex_switch_labels \
      --output_dir checkpoints/switch_head_agilex \
      --epochs 30

  # 2b) Multi-GPU training via torchrun
  torchrun --nproc_per_node=8 train_switch_head_agilex.py train \
      --data_dir data/agilex_switch_labels \
      --output_dir checkpoints/switch_head_agilex \
      --batch_size 8 --epochs 30

  # 3) Export labels into LeRobot v2 dataset for integrated training
  python train_switch_head_agilex.py export \
      --data_dir data/agilex_switch_labels \
      --lerobot_dataset data/pnp_cup_0415 \
      --output_dir data/pnp_cup_0415_with_switch
"""

import argparse
import glob
import json
import logging
import os
import pathlib
import tempfile
import threading
import time
import concurrent.futures

import av
import imageio
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

GEMINI_QUERY_INTERVAL_FRAMES = 20  # query every 20 extracted frames
GEMINI_HISTORY_FRAMES = 60         # context window for Gemini (~20s at 3fps)
GEMINI_VALUE_MODEL = "gemini-3.1-flash-lite-preview"

RESCUE_SCORE_ABSOLUTE = 0.40
RESCUE_SCORE_DROP = 0.15

CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
# Mapping from camera names to .npz keys (3-camera agilex format)
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
#  LeRobot dataset helpers
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


# ============================================================================
#  Gemini value scoring
# ============================================================================

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(http_options={"api_version": "v1alpha"})
    return _gemini_client


def _query_gemini_value(frames: list, task_description: str, step_idx: int,
                        score_history: list, lock: threading.Lock) -> dict:
    """Query Gemini for a value score on a video clip. Thread-safe."""
    from google import genai
    from google.genai import types
    from pydantic import BaseModel

    class ValueEvaluation(BaseModel):
        reasoning: str
        score: float
        status: str

    client = _get_gemini_client()
    tmp_path = None
    video_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        imageio.mimwrite(tmp_path, [np.asarray(x) for x in frames], fps=10)

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


# ============================================================================
#  Phase 1: Label existing Agilex LeRobot dataset with Gemini
# ============================================================================

def label(args):
    """Label an existing Agilex LeRobot dataset with Gemini value scores."""
    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_dataset_info(data_dir)
    episodes = load_episodes(data_dir)
    task = load_task(data_dir)
    total_episodes = info["total_episodes"]
    dataset_fps = info["fps"]

    logging.info(f"Dataset: {data_dir}")
    logging.info(f"Task: {task}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Camera: {args.camera}")
    logging.info(f"Extract FPS: {args.extract_fps} (from {dataset_fps}fps video)")

    # Determine episode range
    if args.episodes:
        ep_start, ep_end = map(int, args.episodes.split(":"))
    else:
        ep_start, ep_end = 0, total_episodes

    all_episode_meta = []
    global_sample_idx = 0
    stats = {"total_episodes": 0, "total_rescue_steps": 0, "total_normal_steps": 0}

    for ep_idx in range(ep_start, ep_end):
        ep_info = episodes[ep_idx]

        # Check if already labeled
        existing = list(output_dir.glob(f"rollout_*_ep{ep_idx}_*"))
        if any(d.name.endswith("_done") for d in existing):
            logging.info(f"  Skip: episode {ep_idx} (already labeled)")
            continue

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
        else:
            # All cameras loaded successfully
            pass

        if len(camera_frames) != len(CAMERA_NAMES):
            continue

        T = min(f.shape[0] for f in camera_frames.values())
        # Trim all cameras to same length
        for cam in camera_frames:
            camera_frames[cam] = camera_frames[cam][:T]

        # Load state and action from parquet
        states, actions, frame_indices = load_episode_parquet(data_dir, ep_idx)

        # Subsample state/action to match extracted frames
        state_interval = max(1, int(round(dataset_fps / args.extract_fps)))
        sampled_states = states[::state_interval][:T]
        sampled_actions = actions[::state_interval][:T]
        sampled_frame_indices = frame_indices[::state_interval][:T]

        logging.info(
            f"  Episode {ep_idx}: {T} frames, state_dim={sampled_states.shape[1]}"
        )

        # Use the primary camera for Gemini queries
        primary_frames = list(camera_frames[args.camera])

        # Async Gemini scoring
        score_history = []
        score_lock = threading.Lock()
        gemini_futures = []
        gemini_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        for frame_i in range(T):
            if (frame_i + 1) % GEMINI_QUERY_INTERVAL_FRAMES == 0:
                clip = primary_frames[max(0, frame_i + 1 - GEMINI_HISTORY_FRAMES):frame_i + 1]
                future = gemini_executor.submit(
                    _query_gemini_value,
                    clip, task, frame_i + 1,
                    score_history, score_lock,
                )
                gemini_futures.append(future)

        gemini_executor.shutdown(wait=True)

        # Build replan records at regular intervals
        replan_records = []
        replan_interval = args.replan_interval
        sorted_scores = sorted(score_history, key=lambda x: x[0])

        for frame_i in range(0, T, replan_interval):
            # Get action chunk starting from this frame
            action_end = min(frame_i + replan_interval, T)
            action_chunk = sampled_actions[frame_i:action_end]

            # Video clips for each camera
            clip_len = args.clip_len
            clips = {}
            for cam in CAMERA_NAMES:
                cam_key = CAMERA_TO_KEY[cam]
                cam_clip = list(camera_frames[cam][max(0, frame_i - clip_len + 1):frame_i + 1])
                if len(cam_clip) < clip_len:
                    pad_n = clip_len - len(cam_clip)
                    cam_clip = [cam_clip[0]] * pad_n + cam_clip
                clips[cam_key] = np.stack(cam_clip)

            # Determine rescue label from Gemini scores
            scores_up_to = [(f, s) for f, s in sorted_scores if f <= frame_i + 1]
            gemini_score = 0.5  # default
            should_rescue = False

            if scores_up_to:
                latest_f, latest_s = scores_up_to[-1]
                gemini_score = latest_s
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
                "switch_label": 1.0 if should_rescue else 0.0,
                "gemini_score": gemini_score,
            })

        # Label shifting: propagate rescue labels backward
        if args.label_shift_steps > 0:
            for i in range(len(replan_records)):
                if replan_records[i]["switch_label"] > 0.5:
                    for j in range(max(0, i - args.label_shift_steps), i):
                        replan_records[j]["switch_label"] = 1.0

        # Save .npz files
        n_rescue = sum(1 for r in replan_records if r["switch_label"] > 0.5)
        n_normal = len(replan_records) - n_rescue

        task_segment = task.replace(" ", "_")[:50]
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
                gemini_score=np.float32(rec["gemini_score"]),
                clip_len=np.int32(args.clip_len),
                prompt=np.array(task),
                task=np.array(task),
                episode_idx=np.int32(ep_idx),
                frame_idx=np.int32(rec["frame_idx"]),
            )
            global_sample_idx += 1

        stats["total_episodes"] += 1
        stats["total_rescue_steps"] += n_rescue
        stats["total_normal_steps"] += n_normal

        episode_meta = {
            "episode_idx": ep_idx,
            "task": task,
            "num_steps": len(replan_records),
            "num_rescue": n_rescue,
            "gemini_scores": [(int(f), float(s)) for f, s in sorted_scores],
        }
        all_episode_meta.append(episode_meta)

        logging.info(
            f"  -> Episode {ep_idx}: steps={len(replan_records)} "
            f"rescue={n_rescue} normal={n_normal}"
        )

    # Save metadata
    meta_path = output_dir / "collection_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"stats": stats, "episodes": all_episode_meta}, f, indent=2)

    total_samples = stats["total_rescue_steps"] + stats["total_normal_steps"]
    logging.info(f"\nLabeling complete.")
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
    """

    IMAGE_KEYS = ["top", "right", "left"]

    def __init__(self, data_dir: str, use_clip: bool = True, use_soft_label: bool = False):
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
            logging.warning("  --use_clip requested but clip keys not found, falling back to single frames")
            self.use_clip = False

        self.use_soft_label = use_soft_label and "gemini_score" in first
        if use_soft_label and not self.use_soft_label:
            logging.warning("  --use_soft_label requested but 'gemini_score' not found")

        logging.info(
            f"  Agilex 3-camera data, state_dim={self.state_dim}, "
            f"clip={self.use_clip}, soft_label={self.use_soft_label}"
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

        if self.use_soft_label and "gemini_score" in data:
            label = torch.tensor(1.0 - float(data["gemini_score"]), dtype=torch.float32)
        else:
            label = torch.tensor(float(data["switch_label"]), dtype=torch.float32)

        return images, state, label


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

    train_dataset = AgilexSwitchLabelDataset(
        args.data_dir, use_clip=args.use_clip, use_soft_label=args.use_soft_label,
    )
    val_dataset = (
        AgilexSwitchLabelDataset(
            args.val_dir, use_clip=args.use_clip, use_soft_label=args.use_soft_label,
        )
        if args.val_dir
        else None
    )

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
#  Phase 3: Export – inject switch_label into LeRobot v2 dataset
# ============================================================================

def export_for_training(args):
    """Inject switch_label into a local LeRobot v2 dataset."""
    import shutil
    import pandas as pd

    data_dir = pathlib.Path(args.data_dir)
    src_dataset = pathlib.Path(args.lerobot_dataset)
    output_dir = pathlib.Path(args.output_dir)

    # Load collected switch labels
    label_files = sorted(glob.glob(str(data_dir / "rollout_*" / "step_*.npz")))
    if not label_files:
        label_files = sorted(glob.glob(str(data_dir / "sample_*.npz")))
    if not label_files:
        logging.error(f"No .npz files found in {data_dir}")
        return

    logging.info(f"Loading {len(label_files)} label files...")
    label_lookup: dict[tuple[int, int], float] = {}
    for f in label_files:
        d = np.load(f, allow_pickle=True)
        ep = int(d["episode_idx"])
        fr = int(d["frame_idx"])
        label_lookup[(ep, fr)] = float(d["switch_label"])

    n_pos = sum(1 for v in label_lookup.values() if v > 0.5)
    n_neg = len(label_lookup) - n_pos
    logging.info(f"Loaded {len(label_lookup)} labels ({n_pos} rescue / {n_neg} normal)")

    # Copy source dataset
    if output_dir.exists():
        logging.warning(f"Output dir {output_dir} exists, overwriting parquet files in-place")
    else:
        logging.info(f"Copying {src_dataset} -> {output_dir} ...")
        shutil.copytree(src_dataset, output_dir)

    # Add switch_label column
    parquet_files = sorted(glob.glob(str(output_dir / "data" / "chunk-*" / "*.parquet")))
    if not parquet_files:
        logging.error(f"No parquet files found in {output_dir / 'data'}")
        return

    logging.info(f"Processing {len(parquet_files)} parquet files...")
    total_frames, matched_frames = 0, 0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        total_frames += len(df)

        labels = []
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            fr = int(row["frame_index"])
            key = (ep, fr)
            if key in label_lookup:
                labels.append(label_lookup[key])
                matched_frames += 1
            else:
                ep_labels = [(f, l) for (e, f), l in label_lookup.items() if e == ep]
                if ep_labels:
                    closest_fr, closest_label = min(ep_labels, key=lambda x: abs(x[0] - fr))
                    labels.append(closest_label)
                else:
                    labels.append(0.0)

        df["switch_label"] = np.array(labels, dtype=np.float32)
        df.to_parquet(pq_path)

    logging.info(f"Processed {total_frames} frames, {matched_frames} exact matches")

    # Update info.json
    info_path = output_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if "switch_label" not in info.get("features", {}):
        info["features"]["switch_label"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        logging.info("Updated meta/info.json with switch_label feature")

    logging.info(f"Done. Augmented dataset saved to {output_dir}")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train switch head for Agilex robot with Gemini labels (label → train → export)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- label ----
    p_label = subparsers.add_parser(
        "label", help="Label existing Agilex LeRobot dataset with Gemini value scores"
    )
    p_label.add_argument("--data_dir", type=str, required=True,
                         help="Path to LeRobot v2 dataset (e.g. data/pnp_cup_0415)")
    p_label.add_argument("--camera", type=str, default="cam_high",
                         choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
                         help="Primary camera for Gemini queries")
    p_label.add_argument("--output_dir", type=str, default="data/agilex_switch_labels")
    p_label.add_argument("--extract_fps", type=float, default=3.0,
                         help="FPS for frame extraction from video")
    p_label.add_argument("--max_frames", type=int, default=512)
    p_label.add_argument("--replan_interval", type=int, default=10,
                         help="Interval (in extracted frames) between replan steps")
    p_label.add_argument("--clip_len", type=int, default=20,
                         help="Number of recent frames per camera for video clips")
    p_label.add_argument("--label_shift_steps", type=int, default=0,
                         help="Shift rescue labels backward for anticipation")
    p_label.add_argument("--episodes", type=str, default=None,
                         help="Episode range, e.g. '0:50' (default: all)")

    # ---- train ----
    p_train = subparsers.add_parser(
        "train", help="Train DINOv2-based switch head on labeled data"
    )
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--val_dir", type=str, default=None)
    p_train.add_argument("--output_dir", type=str, default=None,
                         help="Checkpoint dir. If unset, defaults to checkpoints/switch_head_agilex/<run_tag>, "
                              "where run_tag is --wandb_run_name or the basename of --data_dir.")
    p_train.add_argument(
        "--dinov2_model", type=str, default="dinov2_vitb14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--freeze_backbone", action="store_true", default=True)
    p_train.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--num_workers", type=int, default=4)
    p_train.add_argument("--save_every", type=int, default=5)
    p_train.add_argument("--use_clip", action="store_true", default=True)
    p_train.add_argument("--no_clip", dest="use_clip", action="store_false")
    p_train.add_argument("--use_soft_label", action="store_true", default=False)
    p_train.add_argument("--wandb_project", type=str, default=None)
    p_train.add_argument("--wandb_run_name", type=str, default=None)

    # ---- export ----
    p_export = subparsers.add_parser(
        "export", help="Inject switch_label into LeRobot v2 dataset"
    )
    p_export.add_argument("--data_dir", type=str, required=True)
    p_export.add_argument("--lerobot_dataset", type=str, required=True)
    p_export.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "label":
        label(args)
    elif args.command == "train":
        if args.output_dir is None:
            run_tag = args.wandb_run_name or pathlib.Path(args.data_dir).name
            args.output_dir = str(pathlib.Path("checkpoints/switch_head_agilex") / run_tag)
        train(args)
    elif args.command == "export":
        export_for_training(args)


if __name__ == "__main__":
    main()
