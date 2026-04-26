#!/usr/bin/env python3
"""Per-episode truncation tool for a LeRobot v2.1 dataset.

Two-step human-in-the-loop workflow:

  Step 1 - generate the plan template:
      python scripts/truncate_lerobot_dataset.py init \
          --src  data/spray_cap_merged \
          --plan data/spray_cap_merged_truncate_plan.csv

  Step 2 - edit the CSV by hand. Columns:
      episode_index      original index in the source dataset
      original_seconds   length of the source episode in seconds (read-only hint)
      original_frames    length of the source episode in frames (read-only hint)
      keep_seconds       what you fill in:
                           empty / -1   keep the full episode
                           0            drop the episode entirely
                           > 0          truncate to this many seconds
                                        (clamped at original_seconds)

  Step 3 - apply:
      python scripts/truncate_lerobot_dataset.py apply \
          --src  data/spray_cap_merged \
          --plan data/spray_cap_merged_truncate_plan.csv \
          --dst  data/spray_cap_merged_trunc

The new dataset reassigns episode_index sequentially (0..N-1) so LeRobot loaders
keep working. Aggregate stats files (stats.json, relative_stats.json) are copied
verbatim - re-run scripts/compute_norm_stats.py if you need exact numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REWARD_STRIDE = 10  # rewards/episode_*.npy contain ceil(parquet_len / 10) entries.

# Map ffmpeg encoder name -> codec id stored in info.json's video metadata.
ENCODER_CODEC = {
    "libopenh264": "h264",
    "libx264": "h264",
    "h264_nvenc": "h264",
    "h264_vaapi": "h264",
    "libaom-av1": "av1",
    "libsvtav1": "av1",
    "av1_nvenc": "av1",
    "av1_vaapi": "av1",
    "libx265": "hevc",
    "hevc_nvenc": "hevc",
    "hevc_vaapi": "hevc",
}


@dataclass
class PlanRow:
    episode_index: int
    original_frames: int
    keep_frames: int  # -1 means keep all; 0 means drop


def load_info(src: Path) -> dict:
    with (src / "meta" / "info.json").open() as f:
        return json.load(f)


def load_episodes(src: Path) -> list[dict]:
    rows = []
    with (src / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def video_keys(info: dict) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def episode_parquet_path(root: Path, info: dict, episode_index: int) -> Path:
    chunk = episode_index // info["chunks_size"]
    return root / info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_index
    )


def episode_video_path(root: Path, info: dict, video_key: str, episode_index: int) -> Path:
    chunk = episode_index // info["chunks_size"]
    return root / info["video_path"].format(
        episode_chunk=chunk, video_key=video_key, episode_index=episode_index
    )


def episode_reward_paths(root: Path, episode_index: int) -> tuple[Path, Path, Path]:
    base = root / "rewards" / f"episode_{episode_index:06d}"
    return (
        base.with_name(base.name + "_progress.npy"),
        base.with_name(base.name + "_success.npy"),
        base.with_name(base.name + "_plot.png"),
    )


# --------------------------------------------------------------------------- init


def cmd_init(args: argparse.Namespace) -> None:
    src = Path(args.src).resolve()
    plan_path = Path(args.plan).resolve()

    info = load_info(src)
    episodes = load_episodes(src)
    fps = info["fps"]

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "episode_index",
                "original_seconds",
                "original_frames",
                "keep_seconds",
            ]
        )
        for ep in episodes:
            w.writerow(
                [
                    ep["episode_index"],
                    f"{ep['length'] / fps:.3f}",
                    ep["length"],
                    -1,
                ]
            )

    print(f"[init] Wrote template with {len(episodes)} rows -> {plan_path}")
    print(
        "[init] Edit the keep_seconds column: -1 keeps the full episode, 0 drops it,\n"
        "       any positive number truncates that episode to that many seconds."
    )


# --------------------------------------------------------------------------- apply


def parse_keep(keep_seconds_str: str, fps: int, original_frames: int) -> int:
    """Return frame count to keep. -1 means full, 0 means drop."""
    s = (keep_seconds_str or "").strip()
    if s == "" or s == "-1":
        return original_frames
    val = float(s)
    if val < 0:
        return original_frames
    if val == 0:
        return 0
    keep = int(round(val * fps))
    return min(max(keep, 1), original_frames)


def read_plan(plan_path: Path, episodes: list[dict], fps: int) -> list[PlanRow]:
    by_idx = {ep["episode_index"]: ep for ep in episodes}
    seen: set[int] = set()
    rows: list[PlanRow] = []
    with plan_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            ep_idx = int(r["episode_index"])
            if ep_idx not in by_idx:
                raise ValueError(f"Plan references unknown episode_index={ep_idx}")
            if ep_idx in seen:
                raise ValueError(f"Plan has duplicate episode_index={ep_idx}")
            seen.add(ep_idx)
            orig_frames = by_idx[ep_idx]["length"]
            keep_frames = parse_keep(r.get("keep_seconds", ""), fps, orig_frames)
            rows.append(
                PlanRow(
                    episode_index=ep_idx,
                    original_frames=orig_frames,
                    keep_frames=keep_frames,
                )
            )
    rows.sort(key=lambda r: r.episode_index)
    return rows


def truncate_video(
    src_video: Path,
    dst_video: Path,
    keep_frames: int,
    fps: int,
    encoder: str,
) -> None:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-frames:v",
        str(keep_frames),
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-an",
        str(dst_video),
    ]
    subprocess.run(cmd, check=True)


def truncate_parquet(
    src_parquet: Path,
    dst_parquet: Path,
    keep_frames: int,
    new_episode_index: int,
    new_global_index_start: int,
) -> pd.DataFrame:
    df = pd.read_parquet(src_parquet)
    df = df.iloc[:keep_frames].copy().reset_index(drop=True)
    df["episode_index"] = np.int64(new_episode_index)
    # frame_index is already 0..keep_frames-1 in the source slice, but be defensive.
    df["frame_index"] = np.arange(keep_frames, dtype=np.int64)
    df["index"] = np.arange(
        new_global_index_start,
        new_global_index_start + keep_frames,
        dtype=np.int64,
    )
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False)
    return df


def truncate_reward(src_path: Path, dst_path: Path, keep_reward_len: int) -> None:
    if not src_path.exists():
        return
    arr = np.load(src_path)
    arr = arr[:keep_reward_len]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_path, arr)


def recompute_scalar_stats(df: pd.DataFrame, feature: str, info_feat: dict) -> dict:
    """Per-episode min/max/mean/std/count for non-image features."""
    col = df[feature].to_numpy()
    if col.dtype == object:
        # Stored as list of arrays - stack into 2-D.
        col = np.stack(col)
    col = np.asarray(col)
    if col.ndim == 1:
        col = col[:, None]
    axis = 0
    out = {
        "min": col.min(axis=axis).astype(np.float64).tolist(),
        "max": col.max(axis=axis).astype(np.float64).tolist(),
        "mean": col.mean(axis=axis).astype(np.float64).tolist(),
        "std": col.std(axis=axis).astype(np.float64).tolist(),
        "count": [int(col.shape[0])],
    }
    # Match the original 1-D shape if the feature was 1-D (timestamp, frame_index, ...).
    shape = info_feat.get("shape", [])
    if shape == [1]:
        for k in ("min", "max", "mean", "std"):
            out[k] = [float(out[k][0])]
    return out


def scale_image_stats(orig: dict, ratio: float) -> dict:
    """Keep min/max/mean/std unchanged; scale count proportionally to new length."""
    out = {k: orig[k] for k in ("min", "max", "mean", "std")}
    old_count = orig.get("count", [0])
    if isinstance(old_count, list) and old_count:
        out["count"] = [max(int(round(old_count[0] * ratio)), 1)]
    else:
        out["count"] = old_count
    return out


def cmd_apply(args: argparse.Namespace) -> None:
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    plan_path = Path(args.plan).resolve()

    if dst.exists():
        if not args.overwrite:
            sys.exit(
                f"[apply] Destination {dst} already exists. "
                f"Pass --overwrite to wipe and rebuild it."
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    info = load_info(src)
    episodes = load_episodes(src)
    fps = info["fps"]
    plan = read_plan(plan_path, episodes, fps)

    keepers = [r for r in plan if r.keep_frames > 0]
    drops = [r for r in plan if r.keep_frames == 0]
    truncs = [r for r in keepers if r.keep_frames < r.original_frames]
    print(
        f"[apply] {len(plan)} episodes in plan: "
        f"{len(keepers)} keep, {len(drops)} drop, "
        f"{len(truncs)} truncated, "
        f"{len(keepers) - len(truncs)} kept full."
    )
    if not keepers:
        sys.exit("[apply] Nothing to keep - aborting.")

    # Load original per-episode stats for image-stat passthrough.
    src_episodes_stats: dict[int, dict] = {}
    src_stats_path = src / "meta" / "episodes_stats.jsonl"
    if src_stats_path.exists():
        with src_stats_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                src_episodes_stats[row["episode_index"]] = row["stats"]

    vkeys = video_keys(info)
    image_features = set(vkeys)
    feature_specs = info["features"]
    scalar_features = [
        k
        for k, v in feature_specs.items()
        if v.get("dtype") in {"float32", "float64", "int64", "int32"}
    ]

    new_episodes_meta: list[dict] = []
    new_episodes_stats: list[dict] = []
    cumulative_index = 0

    for new_idx, row in enumerate(keepers):
        keep = row.keep_frames
        old_idx = row.episode_index
        print(
            f"[apply] ep {old_idx:>3} -> {new_idx:>3} : "
            f"{row.original_frames} -> {keep} frames "
            f"({keep / fps:.2f}s)"
        )

        # Parquet
        src_pq = episode_parquet_path(src, info, old_idx)
        dst_pq = episode_parquet_path(dst, info, new_idx)
        df = truncate_parquet(src_pq, dst_pq, keep, new_idx, cumulative_index)

        # Videos
        for vk in vkeys:
            src_v = episode_video_path(src, info, vk, old_idx)
            dst_v = episode_video_path(dst, info, vk, new_idx)
            truncate_video(src_v, dst_v, keep, fps, args.video_codec)

        # Rewards
        keep_rewards = math.ceil(keep / REWARD_STRIDE)
        src_p, src_s, src_plot = episode_reward_paths(src, old_idx)
        dst_p, dst_s, dst_plot = episode_reward_paths(dst, new_idx)
        truncate_reward(src_p, dst_p, keep_rewards)
        truncate_reward(src_s, dst_s, keep_rewards)
        if src_plot.exists():
            dst_plot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_plot, dst_plot)

        # Episode meta
        src_ep_meta = next(e for e in episodes if e["episode_index"] == old_idx)
        new_episodes_meta.append(
            {
                "episode_index": new_idx,
                "tasks": src_ep_meta.get("tasks", []),
                "length": keep,
            }
        )

        # Episode stats: scalars recomputed, image stats inherited (count rescaled).
        ratio = keep / row.original_frames if row.original_frames else 1.0
        src_ep_stats = src_episodes_stats.get(old_idx, {})
        ep_stats: dict = {}
        for feat, spec in feature_specs.items():
            if feat in image_features:
                if feat in src_ep_stats:
                    ep_stats[feat] = scale_image_stats(src_ep_stats[feat], ratio)
            elif feat in scalar_features:
                ep_stats[feat] = recompute_scalar_stats(df, feat, spec)
        new_episodes_stats.append({"episode_index": new_idx, "stats": ep_stats})

        cumulative_index += keep

    # ---------------- meta files ----------------
    meta_dst = dst / "meta"
    meta_dst.mkdir(exist_ok=True)

    new_total_eps = len(new_episodes_meta)
    new_total_frames = cumulative_index
    new_total_chunks = (new_total_eps + info["chunks_size"] - 1) // info["chunks_size"]
    new_total_videos = new_total_eps * len(vkeys)

    # info.json
    new_info = dict(info)
    new_info["total_episodes"] = new_total_eps
    new_info["total_frames"] = new_total_frames
    new_info["total_chunks"] = new_total_chunks
    new_info["total_videos"] = new_total_videos
    new_codec = ENCODER_CODEC.get(args.video_codec, args.video_codec)
    for vk in vkeys:
        feat = new_info["features"][vk]
        if "video_info" in feat:
            feat["video_info"]["video.codec"] = new_codec
        if "info" in feat:
            feat["info"]["video.codec"] = new_codec
    # Preserve the original train/val split ratio.
    if "splits" in info and "train" in info["splits"]:
        try:
            train_str = info["splits"]["train"]  # e.g. "0:90"
            train_end = int(train_str.split(":")[1])
            old_total = info["total_episodes"]
            ratio = train_end / old_total if old_total else 1.0
            new_train_end = max(1, min(new_total_eps - 1, int(round(new_total_eps * ratio))))
            new_info["splits"] = {
                "train": f"0:{new_train_end}",
                "val": f"{new_train_end}:{new_total_eps}",
            }
        except (KeyError, ValueError):
            pass
    with (meta_dst / "info.json").open("w") as f:
        json.dump(new_info, f, indent=4)

    # episodes.jsonl
    with (meta_dst / "episodes.jsonl").open("w") as f:
        for row in new_episodes_meta:
            f.write(json.dumps(row) + "\n")

    # episodes_stats.jsonl
    with (meta_dst / "episodes_stats.jsonl").open("w") as f:
        for row in new_episodes_stats:
            f.write(json.dumps(row) + "\n")

    # Copy unchanged meta files.
    for name in ("tasks.jsonl", "modality.json", "stats.json", "relative_stats.json"):
        p = src / "meta" / name
        if p.exists():
            shutil.copy2(p, meta_dst / name)

    print(
        f"[apply] Wrote new dataset to {dst}\n"
        f"        episodes: {new_total_eps}  frames: {new_total_frames}  "
        f"videos: {new_total_videos}\n"
        f"[apply] Note: stats.json / relative_stats.json were copied verbatim. "
        f"Re-run scripts/compute_norm_stats.py if you need exact aggregate stats."
    )


# --------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Write a CSV plan template the human edits.")
    p_init.add_argument("--src", required=True, help="Source LeRobot dataset root.")
    p_init.add_argument("--plan", required=True, help="Output CSV path.")
    p_init.set_defaults(func=cmd_init)

    p_apply = sub.add_parser("apply", help="Apply a filled-in plan to build a new dataset.")
    p_apply.add_argument("--src", required=True, help="Source LeRobot dataset root.")
    p_apply.add_argument("--plan", required=True, help="Filled-in CSV path.")
    p_apply.add_argument("--dst", required=True, help="Destination LeRobot dataset root (must not exist unless --overwrite).")
    p_apply.add_argument("--overwrite", action="store_true", help="Wipe the destination if it already exists.")
    p_apply.add_argument(
        "--video-codec",
        default="libopenh264",
        help="ffmpeg encoder for the truncated videos (default: libopenh264 — fast CPU h264). "
             "Use libaom-av1 to keep AV1 fidelity at the cost of speed. info.json's "
             "codec field is updated to match.",
    )
    p_apply.set_defaults(func=cmd_apply)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
