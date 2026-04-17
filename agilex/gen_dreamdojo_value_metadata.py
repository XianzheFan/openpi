#!/usr/bin/env python3
"""
Run Robometer on DreamDojo-generated (stitched 3-view) videos and emit the
`--dreamdojo_metadata` JSON consumed by
`train_dinov2_value_expert.py pack --source dreamdojo`.

Since Robometer was trained on single-camera `cam_high` views, we crop a
single column out of each stitched frame before feeding it to Robometer
(default: column 0, which is cam_high under view_layout=top_left_right).

Output JSON schema:
  {
    "entries": [
      {
        "video": "/abs/path/to/video.mp4",
        "progress": [0.12, 0.15, 0.18, ...],   # len == decoded frames
        "success": [0.01, 0.02, ...],          # same length (optional)
        "task": "pick up the yellow cup and place it on the white plate",
        "source_episode": 5,                    # optional (-1 if unknown)
        "source_frame": 42                      # optional (-1 if unknown)
      },
      ...
    ]
  }

Usage
-----
  # Simple: glob DreamDojo outputs under a directory
  python gen_dreamdojo_value_metadata.py \
      --videos_dir ~/workspace/fxz/DreamDojo/outputs_pnp_cup_new_agilex_3view \
      --videos_glob '**/chunk_*.mp4' \
      --task "pick up the yellow cup and place it on the white plate" \
      --output ../data/dreamdojo_progress_0416.json

  # With a source map (to stamp source_episode / source_frame on each entry)
  python gen_dreamdojo_value_metadata.py \
      --videos_dir .../outputs_pnp_cup_new_agilex_3view \
      --source_map .../source_map.json \
      --task "pick up the yellow cup and place it on the white plate" \
      --output ../data/dreamdojo_progress_0416.json

source_map.json format:
  {"chunk_0.mp4": {"source_episode": 5, "source_frame": 42}, ...}
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import json
import time
from pathlib import Path

import av
import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def extract_frames_pyav(
    video_path: str, fps: float = 3.0, max_frames: int = 512,
) -> np.ndarray:
    """Decode video with PyAV. Returns uint8 (T, H, W, 3)."""
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


def crop_stitched_column(
    frames: np.ndarray, view_index: int, num_views: int,
) -> np.ndarray:
    """Crop the view_index-th horizontal strip of (T, H, W, 3). Assumes widths are equal."""
    if num_views <= 1:
        return frames
    T, H, W, C = frames.shape
    strip_w = W // num_views
    start = view_index * strip_w
    end = start + strip_w
    return frames[:, :, start:end, :]


def discover_videos(args) -> list[Path]:
    videos_dir = Path(args.videos_dir).expanduser().resolve() if args.videos_dir else None
    if videos_dir is not None:
        pattern = args.videos_glob or "**/*.mp4"
        matches = sorted(videos_dir.glob(pattern))
    elif args.videos_glob:
        matches = sorted(Path(p).resolve() for p in glob.glob(args.videos_glob, recursive=True))
    else:
        raise ValueError("Provide --videos_dir or --videos_glob (or both).")
    if not matches:
        raise FileNotFoundError(
            f"No videos matched "
            f"videos_dir={args.videos_dir} videos_glob={args.videos_glob}"
        )
    return matches


def load_source_map(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return data


def main():
    ap = argparse.ArgumentParser(
        description="Run Robometer on stitched 3-view DreamDojo videos to produce "
                    "a --dreamdojo_metadata JSON for train_dinov2_value_expert.py."
    )
    ap.add_argument("--videos_dir", type=str, default=None,
                    help="Root directory to search for DreamDojo videos")
    ap.add_argument("--videos_glob", type=str, default=None,
                    help="Glob pattern relative to --videos_dir (or absolute if --videos_dir omitted)")
    ap.add_argument("--source_map", type=str, default=None,
                    help="Optional JSON mapping {video_basename: {source_episode, source_frame}}")
    ap.add_argument("--task", type=str, required=True,
                    help="Task instruction string (must match training)")
    ap.add_argument("--output", type=str, required=True,
                    help="Destination metadata JSON path")

    ap.add_argument("--model_path", type=str, default="robometer/Robometer-4B")
    ap.add_argument("--fps", type=float, default=3.0,
                    help="Extract FPS. Must match the extract_fps you'll pass to "
                         "`train_dinov2_value_expert.py pack --extract_fps`.")
    ap.add_argument("--max_frames", type=int, default=512)

    ap.add_argument("--num_views", type=int, default=3,
                    help="Number of horizontally stitched views per frame (1 = no crop)")
    ap.add_argument("--view_index", type=int, default=0,
                    help="Which stitched column to feed Robometer. For the default "
                         "view_layout=top_left_right, 0=cam_high, 1=cam_left_wrist, 2=cam_right_wrist.")
    ap.add_argument("--save_success", action="store_true", default=True,
                    help="Also record Robometer success probabilities in the JSON")
    args = ap.parse_args()

    videos = discover_videos(args)
    src_map = load_source_map(args.source_map)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(videos)} videos")
    print(f"Task: {args.task}")
    print(f"Robometer model: {args.model_path}")
    print(f"Extract FPS: {args.fps}, view_index={args.view_index}/{args.num_views}")
    print(f"Output: {out_path}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Robometer on {device}...")
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=args.model_path, device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )
    print(f"Model ready. Discrete mode: {is_discrete}\n")

    entries = []
    t_start = time.time()
    for i, vp in enumerate(videos):
        vp_str = str(vp)
        basename = vp.name

        frames = extract_frames_pyav(vp_str, fps=args.fps, max_frames=args.max_frames)
        if frames.size == 0:
            print(f"[WARN] {basename}: failed to decode, skip")
            continue
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        cropped = crop_stitched_column(frames, args.view_index, args.num_views)
        T = cropped.shape[0]

        traj = Trajectory(
            frames=cropped,
            frames_shape=tuple(cropped.shape),
            task=args.task,
            id=basename,
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )
        sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = batch_collator([sample])
        prog_inputs = batch["progress_inputs"]
        for k, v in prog_inputs.items():
            if hasattr(v, "to"):
                prog_inputs[k] = v.to(device)

        with torch.no_grad():
            results = compute_batch_outputs(
                reward_model, tokenizer, prog_inputs,
                sample_type="progress",
                is_discrete_mode=is_discrete, num_bins=num_bins,
            )

        progress_pred = results.get("progress_pred", [])
        progress_arr = (
            np.array(progress_pred[0], dtype=np.float32)
            if progress_pred and len(progress_pred) > 0
            else np.array([], dtype=np.float32)
        )
        outputs_success = results.get("outputs_success", {})
        success_probs = (
            outputs_success.get("success_probs", []) if outputs_success else []
        )
        success_arr = (
            np.array(success_probs[0], dtype=np.float32)
            if success_probs and len(success_probs) > 0
            else np.array([], dtype=np.float32)
        )

        if progress_arr.size != T:
            print(f"[WARN] {basename}: got {progress_arr.size} progress vs {T} frames; "
                  f"using min length")

        src = src_map.get(basename, {})
        entry = {
            "video": vp_str,
            "progress": progress_arr.tolist(),
            "task": args.task,
            "source_episode": int(src.get("source_episode", -1)),
            "source_frame": int(src.get("source_frame", -1)),
        }
        if args.save_success and success_arr.size > 0:
            entry["success"] = success_arr.tolist()
        entries.append(entry)

        del results, batch, prog_inputs
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        eps = (i + 1) / elapsed
        eta = (len(videos) - i - 1) / eps if eps > 0 else 0
        print(
            f"[{i+1:4d}/{len(videos)}] {basename} | frames={T:3d} | "
            f"progress=[{progress_arr.min():.3f}, {progress_arr.max():.3f}] "
            f"mean={progress_arr.mean():.3f}"
            + (f" | success_mean={success_arr.mean():.3f}" if success_arr.size else "")
            + f" | ETA {eta:.0f}s"
        )

    with open(out_path, "w") as f:
        json.dump({"entries": entries}, f, indent=2)

    print(f"\nDone. Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
