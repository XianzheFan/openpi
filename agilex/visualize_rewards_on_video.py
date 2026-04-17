#!/usr/bin/env python3
"""
Overlay frame-level Robometer rewards onto the original episode videos.

Reads per-episode progress/success .npy files (sampled at lower fps) and the
original 30fps MP4, interpolates rewards to every frame, then renders a new
video with:
  - progress bar (color-coded green→yellow→red)
  - numeric progress & success scores
  - a trailing sparkline chart of recent progress history

Usage:
  python visualize_rewards_on_video.py                        # all episodes
  python visualize_rewards_on_video.py --episodes 0:5         # first 5
  python visualize_rewards_on_video.py --episode-id 42        # single episode
  python visualize_rewards_on_video.py --camera cam_left_wrist  # different cam
  python visualize_rewards_on_video.py --labels-dir data/agilex_switch_labels  # with rescue markers
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import cv2
import numpy as np


def interpolate_rewards(rewards: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate reward array from sampled length to target video length."""
    if rewards.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if rewards.size == target_len:
        return rewards
    x_src = np.linspace(0, 1, len(rewards))
    x_dst = np.linspace(0, 1, target_len)
    return np.interp(x_dst, x_src, rewards).astype(np.float32)


def progress_color(value: float) -> tuple[int, int, int]:
    """Map progress 0->1 to BGR color: red(0) -> yellow(0.5) -> green(1)."""
    v = max(0.0, min(1.0, value))
    if v < 0.5:
        r = 255
        g = int(255 * (v / 0.5))
    else:
        r = int(255 * ((1.0 - v) / 0.5))
        g = 255
    return (0, g, r)  # BGR


def load_rescue_info(labels_dir: Path, ep_idx: int, extract_fps: float, video_fps: float, total_frames: int) -> tuple[set[int], list[tuple[int, int, bool]]]:
    """Load rescue info from packed .npz files. Returns (rescue_frame_set, timeline_markers)."""
    pattern = str(labels_dir / f"rollout_*_ep{ep_idx}_done" / "step_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        return set(), []

    frame_interval = max(1, int(round(video_fps / extract_fps)))
    rescue_frames = set()
    markers = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        fidx = int(d["frame_idx"])
        is_rescue = float(d["switch_label"]) > 0.5
        orig_start = fidx * frame_interval
        orig_end = min((fidx + 10) * frame_interval, total_frames)
        markers.append((orig_start, orig_end, is_rescue))
        if is_rescue:
            for fr in range(orig_start, orig_end):
                rescue_frames.add(fr)
    return rescue_frames, markers


def draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    progress: float,
    success: float | None,
    progress_history: np.ndarray,
) -> np.ndarray:
    """Draw reward overlay on a single frame. Modifies frame in-place."""
    h, w = frame.shape[:2]

    # Semi-transparent dark band at the top
    band_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, band_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # -- Text info --
    color = progress_color(progress)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Progress score
    cv2.putText(
        frame,
        f"Progress: {progress:.3f}",
        (12, 28),
        font, 0.7, color, 2, cv2.LINE_AA,
    )

    # Success score
    if success is not None:
        s_color = (0, 220, 0) if success > 0.5 else (0, 0, 220)
        cv2.putText(
            frame,
            f"Success: {success:.3f}",
            (12, 56),
            font, 0.7, s_color, 2, cv2.LINE_AA,
        )

    # Frame counter
    cv2.putText(
        frame,
        f"Frame {frame_idx}/{total_frames}",
        (w - 200, 28),
        font, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # -- Progress bar (bottom) --
    bar_h = 14
    bar_y = h - bar_h - 8
    bar_x0, bar_x1 = 10, w - 10
    bar_w = bar_x1 - bar_x0

    # Background
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (60, 60, 60), -1)
    # Filled portion
    fill_w = int(bar_w * max(0.0, min(1.0, progress)))
    if fill_w > 0:
        cv2.rectangle(
            frame,
            (bar_x0, bar_y),
            (bar_x0 + fill_w, bar_y + bar_h),
            color, -1,
        )
    # Border
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (180, 180, 180), 1)

    # -- Sparkline chart (top-right) --
    chart_w, chart_h = 180, 45
    chart_x0 = w - chart_w - 12
    chart_y0 = band_h + 6

    # Chart background
    chart_overlay = frame.copy()
    cv2.rectangle(
        chart_overlay,
        (chart_x0, chart_y0),
        (chart_x0 + chart_w, chart_y0 + chart_h),
        (0, 0, 0), -1,
    )
    cv2.addWeighted(chart_overlay, 0.5, frame, 0.5, 0, frame)

    # Draw sparkline from progress_history
    hist = progress_history
    n = len(hist)
    if n >= 2:
        pts = []
        for i, val in enumerate(hist):
            px = chart_x0 + int(i * (chart_w - 1) / (n - 1))
            py = chart_y0 + chart_h - 1 - int(max(0.0, min(1.0, val)) * (chart_h - 2))
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (0, 255, 100), 1, cv2.LINE_AA)

    # Chart border
    cv2.rectangle(
        frame,
        (chart_x0, chart_y0),
        (chart_x0 + chart_w, chart_y0 + chart_h),
        (100, 100, 100), 1,
    )

    return frame


def render_episode(
    video_path: Path,
    progress_path: Path,
    success_path: Path | None,
    output_path: Path,
    fps_out: float | None = None,
    rescue_frames: set[int] | None = None,
    rescue_markers: list[tuple[int, int, bool]] | None = None,
):
    """Render a single episode video with reward overlay."""
    # Read video with PyAV (supports AV1)
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total_frames = stream.frames
    video_fps = float(stream.average_rate) or 30.0
    w, h = stream.width, stream.height

    if fps_out is None:
        fps_out = video_fps

    # Load rewards and interpolate to match video frame count
    progress_raw = np.load(str(progress_path))
    progress = interpolate_rewards(progress_raw, total_frames)

    success = None
    if success_path and success_path.exists():
        success_raw = np.load(str(success_path))
        if success_raw.size > 0:
            success = interpolate_rewards(success_raw, total_frames)

    # Write output video with cv2 (H.264 compatible output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (w, h))

    sparkline_window = 90  # ~3 seconds of history at 30fps

    for fi, frame in enumerate(container.decode(video=0)):
        # PyAV gives RGB, cv2 needs BGR
        img = frame.to_ndarray(format="bgr24")

        # Build recent progress history for sparkline
        start = max(0, fi - sparkline_window + 1)
        history = progress[start : fi + 1]

        s_val = float(success[fi]) if success is not None else None

        draw_overlay(
            img,
            frame_idx=fi,
            total_frames=total_frames,
            progress=float(progress[fi]),
            success=s_val,
            progress_history=history,
        )

        # --- Rescue overlay ---
        if rescue_frames is not None:
            is_rescue = fi in rescue_frames
            font = cv2.FONT_HERSHEY_SIMPLEX
            if is_rescue:
                cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
                label = "RESCUE"
                (tw, th), _ = cv2.getTextSize(label, font, 1.4, 3)
                bx, by = 12, 80
                cv2.rectangle(img, (bx - 4, by - 4), (bx + tw + 8, by + th + 12), (0, 0, 180), -1)
                cv2.putText(img, label, (bx, by + th + 2), font, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

            # Timeline bar at bottom showing rescue segments
            if rescue_markers:
                bar_x0, bar_x1 = 10, w - 10
                bar_w = bar_x1 - bar_x0
                tl_h = 10
                tl_y = h - tl_h - 8
                cv2.rectangle(img, (bar_x0, tl_y), (bar_x1, tl_y + tl_h), (80, 80, 80), -1)
                for m_start, m_end, m_rescue in rescue_markers:
                    x0 = bar_x0 + int(bar_w * m_start / max(total_frames, 1))
                    x1 = bar_x0 + int(bar_w * m_end / max(total_frames, 1))
                    color = (0, 0, 255) if m_rescue else (0, 160, 0)
                    cv2.rectangle(img, (x0, tl_y), (max(x1, x0 + 2), tl_y + tl_h), color, -1)
                pos_x = bar_x0 + int(bar_w * fi / max(total_frames - 1, 1))
                cv2.line(img, (pos_x, tl_y - 2), (pos_x, tl_y + tl_h + 2), (255, 255, 255), 2)
                cv2.rectangle(img, (bar_x0, tl_y), (bar_x1, tl_y + tl_h), (180, 180, 180), 1)

        writer.write(img)

    container.close()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Robometer rewards on episode videos."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/zhiqil/workspace/fxz/openpi/data/pnp_cup_0415",
    )
    parser.add_argument(
        "--rewards-dir",
        type=str,
        default=None,
        help="Directory with reward .npy files (default: <data-dir>/rewards)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="cam_high",
        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <data-dir>/rewards_vis)",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Episode range, e.g. '0:10' (default: all available)",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Single episode index to visualize",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video fps (default: same as source)",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Directory with packed .npz labels to overlay rescue markers",
    )
    parser.add_argument(
        "--extract-fps",
        type=float,
        default=3.0,
        help="FPS used when extracting frames for labeling (default: 3.0)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rewards_dir = Path(args.rewards_dir) if args.rewards_dir else data_dir / "rewards"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "rewards_vis"
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(data_dir / "meta" / "info.json") as f:
        info = json.load(f)
    total_episodes = info["total_episodes"]

    # Determine which episodes to render
    if args.episode_id is not None:
        ep_list = [args.episode_id]
    elif args.episodes:
        s, e = args.episodes.split(":")
        ep_list = list(range(int(s), int(e)))
    else:
        # Auto-detect from available reward files
        ep_list = []
        for i in range(total_episodes):
            if (rewards_dir / f"episode_{i:06d}_progress.npy").exists():
                ep_list.append(i)

    if not ep_list:
        print("No reward files found. Run label_rewards_robometer.py first.")
        return

    print(f"Rendering {len(ep_list)} episodes with reward overlay")
    print(f"  Camera: {args.camera}")
    print(f"  Rewards: {rewards_dir}")
    if labels_dir:
        print(f"  Labels:  {labels_dir} (rescue markers enabled)")
    print(f"  Output:  {output_dir}")
    print()

    video_key = f"observation.images.{args.camera}"

    for idx, ep_idx in enumerate(ep_list):
        ep_name = f"episode_{ep_idx:06d}"
        video_path = data_dir / "videos" / "chunk-000" / video_key / f"{ep_name}.mp4"
        progress_path = rewards_dir / f"{ep_name}_progress.npy"
        success_path = rewards_dir / f"{ep_name}_success.npy"
        out_path = output_dir / f"{ep_name}_reward_vis.mp4"

        if not video_path.exists():
            print(f"  [{ep_idx}] Video not found: {video_path}, skipping")
            continue
        if not progress_path.exists():
            print(f"  [{ep_idx}] Progress file not found: {progress_path}, skipping")
            continue

        # Load rescue info if labels dir provided
        r_frames, r_markers = None, None
        if labels_dir:
            # Need video fps to map frame indices
            c = av.open(str(video_path))
            s = c.streams.video[0]
            vid_fps = float(s.average_rate) or 30.0
            vid_total = s.frames
            c.close()
            r_frames, r_markers = load_rescue_info(
                labels_dir, ep_idx, args.extract_fps, vid_fps, vid_total,
            )

        ok = render_episode(
            video_path=video_path,
            progress_path=progress_path,
            success_path=success_path if success_path.exists() else None,
            output_path=out_path,
            fps_out=args.fps,
            rescue_frames=r_frames,
            rescue_markers=r_markers,
        )

        n_info = ""
        if r_frames is not None and r_markers:
            n_rescue = sum(1 for _, _, r in r_markers if r)
            n_info = f" rescue={n_rescue}/{len(r_markers)}"
        status = "OK" if ok else "FAIL"
        print(f"  [{idx + 1}/{len(ep_list)}] Episode {ep_idx}: {status}{n_info} -> {out_path.name}")

    print()
    print(f"Done! Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
