#!/usr/bin/env python3
"""
Offline VLM (gemini-3.1-flash-lite-preview) failure prediction on plug_merged.

Compares against agilex_infer_dinov2_value_switch.py (DreamDojo + Value Expert).

Protocol:
  - Per ep, t_crit = first step with switch_label == 1.0
  - At step t_crit - delta (delta in {2, 5, 10}), load {top,right,left}_clip
    (each (20, 480, 640, 3) uint8), horizontally concat -> (20, 480, 1920, 3),
    write a temp 2s mp4 at 10fps, send to Gemini
  - Prompt asks for fail_prob ∈ [0, 1]; temperature=0
  - GT: rewards/episode_XXXXXX_success.npy[-1] < 0.5 -> failure

Usage:
  python agilex/offline_vlm_predict.py
  python agilex/offline_vlm_predict.py --eval_only
"""

import argparse
import concurrent.futures
import json
import logging
import os
import pathlib
import re
import tempfile
import threading
import time

import imageio
import numpy as np
from PIL import Image
from pydantic import BaseModel

from google import genai
from google.genai import types


ROLLOUT_FPS = 10
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_TASK_TEXT = "Plug the charger into the socket."

DEFAULT_LABELS_DIR = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "zhiqil/workspace/fxz/openpi/data/agilex_switch_labels_plug_merged"
)
DEFAULT_REWARDS_DIR = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "zhiqil/workspace/fxz/openpi/data/plug_merged/rewards"
)
DEFAULT_OUTPUT_DIR = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "zhiqil/workspace/fxz/openpi/data/plug_merged/vlm_predictions"
)


class FailureEvaluation(BaseModel):
    reasoning: str
    fail_prob: float


_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            _client = genai.Client(http_options={"api_version": "v1alpha"})
    return _client


def _find_t_crit(ep_dir: pathlib.Path) -> int | None:
    steps = sorted(ep_dir.glob("step_*.npz"))
    for i, p in enumerate(steps):
        with np.load(p, allow_pickle=True) as d:
            if float(d["switch_label"]) >= 0.5:
                return i
    return None


def _resize_clip_bilinear(clip: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize (T, H, W, 3) uint8 to (T, h, w, 3) uint8 via PIL bilinear."""
    T = clip.shape[0]
    out = np.empty((T, h, w, 3), dtype=np.uint8)
    for i in range(T):
        im = Image.fromarray(clip[i])
        im = im.resize((w, h), Image.BILINEAR)
        out[i] = np.asarray(im)
    return out


def _load_and_stitch_clip(step_path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Return (20, 480, 640, 3) uint8 2x2 grid and original frame_idx.
    Each quadrant is the cam clip resized to (240, 320). Canvas matches the
    native single-view resolution (480, 640).
        upper-left  = top   (cam_high)
        upper-right = left  (cam_left_wrist)
        lower-left  = right (cam_right_wrist)
        lower-right = black
    """
    with np.load(step_path, allow_pickle=True) as d:
        top = d["top_clip"]        # (T, 480, 640, 3)
        right = d["right_clip"]
        left = d["left_clip"]
        frame_idx = int(d["frame_idx"])

    T, H, W, C = top.shape
    h, w = H // 2, W // 2          # (240, 320)
    top_s = _resize_clip_bilinear(top, h, w)
    left_s = _resize_clip_bilinear(left, h, w)
    right_s = _resize_clip_bilinear(right, h, w)

    canvas = np.zeros((T, H, W, C), dtype=np.uint8)
    canvas[:, :h, :w] = top_s       # upper-left
    canvas[:, :h, w:] = left_s      # upper-right
    canvas[:, h:, :w] = right_s     # lower-left
    # lower-right stays zeros
    return canvas, frame_idx


def _write_clip_mp4(frames: np.ndarray, path: str) -> None:
    imageio.mimwrite(path, [np.asarray(f) for f in frames],
                     fps=ROLLOUT_FPS, codec="libx264")


def _query_gemini(video_path: str, task_text: str) -> dict:
    client = _get_client()
    video_file = None
    try:
        video_file = client.files.upload(file=video_path)
        info = client.files.get(name=video_file.name)
        while info.state.name == "PROCESSING":
            time.sleep(2)
            info = client.files.get(name=video_file.name)
        if info.state.name == "FAILED":
            return {"error": "video_processing_failed"}

        prompt = (
            f'The robot is performing the task: "{task_text}". '
            f'Given this 2s video of the robot, output the probability that the '
            f'overall task will FAIL to complete. Consider whether the robot '
            f'appears on track toward completion, is oscillating, or has '
            f'knocked / dropped the target.\n'
            f'Return JSON strictly as: '
            f'[{{"reasoning": "...", "fail_prob": 0.00-1.00}}].'
        )

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, "\n[Video]:", video_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[FailureEvaluation],
                temperature=0.0,
            ),
        )
        parsed = json.loads(resp.text)
        if not parsed:
            return {"error": "empty_response"}
        first = parsed[0]
        if "fail_prob" not in first:
            return {"error": "missing_fail_prob", "raw": first}
        return {
            "reasoning": first.get("reasoning", ""),
            "fail_prob": float(first["fail_prob"]),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if video_file is not None:
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass


def _process_one(ep_idx: int, delta: int, ep_dir: pathlib.Path,
                 out_dir: pathlib.Path, overwrite: bool,
                 task_text: str = DEFAULT_TASK_TEXT) -> dict:
    out_path = out_dir / f"ep{ep_idx:03d}_delta{delta}.json"
    if out_path.exists() and not overwrite:
        try:
            cached = json.loads(out_path.read_text())
            if isinstance(cached.get("result"), dict) and "fail_prob" in cached["result"]:
                return {"ep": ep_idx, "delta": delta, "skipped": "cached",
                        "fail_prob": cached["result"]["fail_prob"]}
        except Exception:
            pass

    t_crit = _find_t_crit(ep_dir)
    if t_crit is None:
        rec = {"ep": ep_idx, "delta": delta, "skipped": "no_t_crit"}
        out_path.write_text(json.dumps(rec, indent=2))
        return rec

    t_pre = t_crit - delta
    if t_pre < 0:
        rec = {"ep": ep_idx, "delta": delta, "t_crit": t_crit, "t_pre": t_pre,
               "skipped": "t_pre_negative"}
        out_path.write_text(json.dumps(rec, indent=2))
        return rec

    step_path = ep_dir / f"step_{t_pre:04d}.npz"
    if not step_path.exists():
        rec = {"ep": ep_idx, "delta": delta, "t_crit": t_crit, "t_pre": t_pre,
               "skipped": "step_missing"}
        out_path.write_text(json.dumps(rec, indent=2))
        return rec

    frames, frame_idx = _load_and_stitch_clip(step_path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        _write_clip_mp4(frames, tmp_path)
        result = _query_gemini(tmp_path, task_text)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    rec = {
        "ep": ep_idx,
        "delta": delta,
        "t_crit": t_crit,
        "t_pre": t_pre,
        "frame_idx": frame_idx,
        "result": result,
    }
    out_path.write_text(json.dumps(rec, indent=2))
    return rec


def _discover_episodes(labels_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    eps = []
    for d in labels_dir.glob("rollout_*_ep*_done"):
        m = re.search(r"_ep(\d+)_done$", d.name)
        if m:
            eps.append((int(m.group(1)), d))
    eps.sort()
    return eps


def _evaluate(out_dir: pathlib.Path, rewards_dir: pathlib.Path,
              ep_list: list[tuple[int, pathlib.Path]], deltas: list[int]) -> None:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        logging.error("sklearn not installed; cannot compute AUROC/AP.")
        return

    logging.info("=== Evaluation (GT: final success < 0.5 -> failure) ===")
    for delta in deltas:
        y_true, y_score, skipped = [], [], 0
        for ep_idx, _ in ep_list:
            p = out_dir / f"ep{ep_idx:03d}_delta{delta}.json"
            if not p.exists():
                continue
            rec = json.loads(p.read_text())
            res = rec.get("result")
            if not isinstance(res, dict) or "fail_prob" not in res:
                skipped += 1
                continue
            s_path = rewards_dir / f"episode_{ep_idx:06d}_success.npy"
            if not s_path.exists():
                continue
            succ_final = float(np.load(s_path)[-1])
            y_true.append(1 if succ_final < 0.5 else 0)
            y_score.append(float(res["fail_prob"]))

        y_true = np.array(y_true)
        y_score = np.array(y_score)
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            logging.info(f"Δ={delta}: n={len(y_true)} fail={n_pos} succ={n_neg} "
                         f"(skipped_pred={skipped}) — degenerate")
            continue
        auroc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        logging.info(f"Δ={delta}: n={len(y_true)} fail={n_pos} succ={n_neg} "
                     f"(skipped_pred={skipped})  AUROC={auroc:.3f}  AP={ap:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default=DEFAULT_LABELS_DIR)
    ap.add_argument("--rewards_dir", default=DEFAULT_REWARDS_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--task_text", default=DEFAULT_TASK_TEXT,
                    help="Task description sent to the VLM prompt.")
    ap.add_argument("--deltas", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--limit_eps", type=int, default=0,
                    help="If >0, only process the first N episodes (for smoke test).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    labels_dir = pathlib.Path(args.labels_dir)
    out_dir = pathlib.Path(args.output_dir)
    rewards_dir = pathlib.Path(args.rewards_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_list = _discover_episodes(labels_dir)
    if args.limit_eps > 0:
        ep_list = ep_list[: args.limit_eps]
    logging.info(f"Found {len(ep_list)} episodes under {labels_dir}")

    if not args.eval_only:
        jobs = [(ep_idx, delta, ep_dir)
                for ep_idx, ep_dir in ep_list for delta in args.deltas]
        logging.info(f"Submitting {len(jobs)} jobs with {args.num_workers} workers...")
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(_process_one, ep, d, ed, out_dir, args.overwrite,
                          args.task_text): (ep, d)
                for ep, d, ed in jobs
            }
            for fut in concurrent.futures.as_completed(futures):
                ep, d = futures[fut]
                done += 1
                try:
                    r = fut.result()
                except Exception as e:
                    logging.error(f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d} crashed: {e}")
                    continue
                res = r.get("result", {})
                if isinstance(res, dict) and "fail_prob" in res:
                    logging.info(
                        f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d}  "
                        f"t_crit={r.get('t_crit')} t_pre={r.get('t_pre')}  "
                        f"fail_prob={res['fail_prob']:.3f}"
                    )
                elif r.get("skipped"):
                    logging.info(
                        f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d}  skip={r['skipped']}"
                    )
                elif isinstance(res, dict) and "error" in res:
                    logging.warning(
                        f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d}  error={res['error']}"
                    )

    _evaluate(out_dir, rewards_dir, ep_list, args.deltas)


if __name__ == "__main__":
    main()
