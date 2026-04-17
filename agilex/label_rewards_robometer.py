#!/usr/bin/env python3
"""
Batch label frame-level rewards for pnp_cup_0415 dataset using Robometer.

Iterates over all episodes, extracts frames from video, runs Robometer inference
to get per-frame progress scores and success probabilities, and saves results.

Usage:
  python label_rewards_robometer.py \
    --data-dir /home/zhiqil/workspace/fxz/openpi/data/pnp_cup_0415 \
    --model-path robometer/Robometer-4B \
    --camera cam_high \
    --fps 3 \
    --max-frames 512

Output structure (saved under <data-dir>/rewards/):
  rewards/
    episode_000000_progress.npy    # (T,) float32, progress 0->1
    episode_000000_success.npy     # (T,) float32, success probability
    episode_000000_plot.png        # visualization
    ...
    summary.json                   # aggregated statistics
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_frames_pyav(
    video_path: str, fps: float = 1.0, max_frames: int = 512
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


def load_dataset_info(data_dir: Path) -> dict:
    """Load dataset metadata."""
    info_path = data_dir / "meta" / "info.json"
    with open(info_path) as f:
        return json.load(f)


def load_episodes(data_dir: Path) -> list[dict]:
    """Load per-episode metadata from episodes.jsonl."""
    episodes = []
    with open(data_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def load_task(data_dir: Path) -> str:
    """Load the task instruction."""
    with open(data_dir / "meta" / "tasks.jsonl") as f:
        first_line = f.readline().strip()
        return json.loads(first_line)["task"]


def get_video_path(data_dir: Path, episode_index: int, camera: str) -> Path:
    """Construct path to video file for a given episode and camera."""
    video_key = f"observation.images.{camera}"
    return (
        data_dir
        / "videos"
        / "chunk-000"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Label frame-level rewards for robot episodes using Robometer."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/zhiqil/workspace/fxz/openpi/data/pnp_cup_0415",
        help="Path to the LeRobot-format dataset directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="robometer/Robometer-4B",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="cam_high",
        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        help="Which camera view to use for reward labeling (default: cam_high)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="FPS for frame extraction from 30fps video (default: 3.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=512,
        help="Maximum frames to extract per episode (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for rewards (default: <data-dir>/rewards)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save progress/success plots per episode (default: True)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of episodes to batch together (default: 1)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Episode range to process, e.g. '0:10' or '5:20' (default: all)",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary success in plots (default: 0.5)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "rewards"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_plots = args.save_plots and not args.no_plots

    # Load dataset metadata
    info = load_dataset_info(data_dir)
    episodes = load_episodes(data_dir)
    task = load_task(data_dir)
    total_episodes = info["total_episodes"]

    print(f"Dataset: {data_dir}")
    print(f"Task: {task}")
    print(f"Total episodes: {total_episodes}")
    print(f"Camera: {args.camera}")
    print(f"Model: {args.model_path}")
    print(f"FPS sampling: {args.fps} (from {info['fps']}fps video)")
    print(f"Output: {output_dir}")
    print()

    # Determine episode range
    if args.episodes:
        start, end = args.episodes.split(":")
        ep_start, ep_end = int(start), int(end)
    else:
        ep_start, ep_end = 0, total_episodes

    # Load model once
    print("Loading Robometer model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=args.model_path,
        device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    # Get loss config for discrete mode detection
    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    print(f"Model loaded. Discrete mode: {is_discrete}")
    print()

    # Process episodes
    summary = {
        "model_path": args.model_path,
        "camera": args.camera,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "task": task,
        "episodes": {},
    }

    t_start = time.time()
    for ep_idx in range(ep_start, ep_end):
        ep_info = episodes[ep_idx]
        video_path = get_video_path(data_dir, ep_idx, args.camera)

        if not video_path.exists():
            print(f"[WARN] Episode {ep_idx}: video not found at {video_path}, skipping")
            continue

        # Extract frames using PyAV (AV1 codec support)
        frames = extract_frames_pyav(
            str(video_path), fps=args.fps, max_frames=args.max_frames
        )
        if frames is None or frames.size == 0:
            print(f"[WARN] Episode {ep_idx}: failed to extract frames, skipping")
            continue

        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        T = frames.shape[0]

        # Build trajectory and run inference
        traj = Trajectory(
            frames=frames,
            frames_shape=tuple(frames.shape),
            task=task,
            id=str(ep_idx),
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = batch_collator([progress_sample])

        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(device)

        with torch.no_grad():
            results = compute_batch_outputs(
                reward_model,
                tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=is_discrete,
                num_bins=num_bins,
            )

        # Extract predictions
        progress_pred = results.get("progress_pred", [])
        progress_array = (
            np.array(progress_pred[0], dtype=np.float32)
            if progress_pred and len(progress_pred) > 0
            else np.array([], dtype=np.float32)
        )

        outputs_success = results.get("outputs_success", {})
        success_probs = (
            outputs_success.get("success_probs", []) if outputs_success else []
        )
        success_array = (
            np.array(success_probs[0], dtype=np.float32)
            if success_probs and len(success_probs) > 0
            else np.array([], dtype=np.float32)
        )

        # Save per-episode results
        ep_name = f"episode_{ep_idx:06d}"
        np.save(str(output_dir / f"{ep_name}_progress.npy"), progress_array)
        np.save(str(output_dir / f"{ep_name}_success.npy"), success_array)

        # Save plot
        if save_plots and progress_array.size > 0:
            show_success = (
                success_array.size > 0 and success_array.size == progress_array.size
            )
            success_binary = (
                (success_array > args.success_threshold).astype(np.int32)
                if show_success
                else None
            )
            fig = create_combined_progress_success_plot(
                progress_pred=progress_array,
                num_frames=T,
                success_binary=success_binary,
                success_probs=success_array if show_success else None,
                success_labels=None,
                title=f"Episode {ep_idx} — {task}",
            )
            fig.savefig(str(output_dir / f"{ep_name}_plot.png"), dpi=150)
            plt.close(fig)

        # Record summary
        ep_summary = {
            "episode_length": ep_info["length"],
            "extracted_frames": T,
            "progress_min": float(progress_array.min()) if progress_array.size else None,
            "progress_max": float(progress_array.max()) if progress_array.size else None,
            "progress_mean": float(progress_array.mean()) if progress_array.size else None,
            "success_mean": float(success_array.mean()) if success_array.size else None,
        }
        summary["episodes"][ep_idx] = ep_summary

        # Free intermediate CUDA tensors to reduce fragmentation
        del results, batch, progress_inputs
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        eps_per_sec = (ep_idx - ep_start + 1) / elapsed
        remaining = (ep_end - ep_idx - 1) / eps_per_sec if eps_per_sec > 0 else 0
        print(
            f"Episode {ep_idx:3d}/{ep_end} | "
            f"frames={T:3d} | "
            f"progress=[{ep_summary['progress_min']:.3f}, {ep_summary['progress_max']:.3f}] "
            f"mean={ep_summary['progress_mean']:.3f} | "
            f"success_mean={ep_summary['success_mean']:.3f} | "
            f"ETA {remaining:.0f}s"
        )

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed_total = time.time() - t_start
    n_processed = ep_end - ep_start
    print()
    print(f"Done! Processed {n_processed} episodes in {elapsed_total:.1f}s")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
