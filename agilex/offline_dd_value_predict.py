#!/usr/bin/env python3
"""
Offline DreamDojo + DINOv2 Value Expert failure prediction.

Apples-to-apples counterpart of agilex/offline_vlm_predict.py: at each branch
point `t_crit - delta`, produce a scalar "episode will fail" score so we can
run the same AUROC/AP evaluation against GT (rewards/episode_XXXXXX_success.npy[-1]
< 0.5). Output JSON is intentionally shaped like offline_vlm_predict.py's so
the two can be compared side by side.

Two modes for sourcing the action chunk(s) fed to DreamDojo:

  (A) cached-actions (default, no policy server needed)
      Uses `actions` straight from step_XXXX.npz. num_samples forced to 1.
      Signal: "DD imagines the realized future; VE scores how good it looks."
      Simplest path to get numbers.

  (B) policy-sampled (--policy_host ...)
      Queries an openpi policy server at `t_pre` for num_samples action chunks
      (SDE recommended for diversity). Mirrors the online rescue pipeline.

Per candidate: DD -> mp4, VE -> per-window scores -> agg_score
(lower = closer to subtask completion, i.e. MORE likely to succeed).

Aggregation across candidates (higher = more failure-likely):
  - agg_best  = min_i agg_score_i   (if even the best candidate looks bad...)
  - agg_mean  = mean_i agg_score_i  (average quality)

Both are stored and evaluated; AUROC/AP only cares about order, so we pass
raw scores (no sigmoid) to sklearn.

Usage
-----
Prerequisite: one DreamDojo server on port 8020, VE ckpt, packed rollout dir.

  # (A) cached-actions mode (pnp_cup_0415)
  python agilex/offline_dd_value_predict.py \
      --labels_dir  data/agilex_switch_labels_pnp_cup_0415 \
      --rewards_dir data/pnp_cup_0415/rewards \
      --output_dir  data/pnp_cup_0415/dd_value_predictions \
      --task_text   "Pick up the paper cup and put it into the cup sleeve." \
      --dd_host 127.0.0.1 --dd_port 8020 \
      --value_expert_ckpt agilex/checkpoints/dinov2_value_expert/best_model.pt

  # Eval only (no DD/VE calls)
  python agilex/offline_dd_value_predict.py --eval_only \
      --labels_dir ... --rewards_dir ... --output_dir ...

  # (B) policy-sampled mode (requires ODE/SDE server up)
  python agilex/offline_dd_value_predict.py \
      ... \
      --policy_host 127.0.0.1 --policy_port 8000 --policy_mode sde \
      --num_samples 5
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import threading
import time

import imageio
import numpy as np
import requests
import torch

from train_dinov2_value_expert import DINOv2ValueExpert


DEFAULT_TASK_TEXT = "Pick up the paper cup and put it into the cup sleeve."
DEFAULT_LABELS_DIR = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "zhiqil/workspace/fxz/openpi/data/agilex_switch_labels_pnp_cup_0415"
)
DEFAULT_REWARDS_DIR = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "zhiqil/workspace/fxz/openpi/data/pnp_cup_0415/rewards"
)
DEFAULT_OUTPUT_DIR = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "zhiqil/workspace/fxz/openpi/data/pnp_cup_0415/dd_value_predictions"
)
DEFAULT_VALUE_CKPT = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/"
    "zhiqil/workspace/fxz/openpi/agilex/checkpoints/dinov2_value_expert/best_model.pt"
)


class ValueExpertScorer:
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
        self.model.to(self.device).eval()

    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) / 255.0

    @torch.no_grad()
    def score_video(self, frames: list[np.ndarray]) -> np.ndarray:
        L = len(frames)
        if L < self.num_clip_frames:
            frames = list(frames) + [frames[-1]] * (self.num_clip_frames - L)
        video_t = torch.stack([self._img_to_tensor(f) for f in frames]).unsqueeze(0)
        values = self.model.score_video(video_t,
                                        window_size=self.num_clip_frames, stride=1)
        return values.squeeze(0).cpu().numpy()

    @staticmethod
    def aggregate(per_window: np.ndarray) -> float:
        """Average values from start up to the first dip (matches online script)."""
        if len(per_window) == 0:
            return float("inf")
        min_idx, min_val = 0, per_window[0]
        for i in range(1, len(per_window)):
            if per_window[i] < min_val:
                min_val, min_idx = per_window[i], i
            elif per_window[i] > min_val + 0.05:
                break
        return float(np.mean(per_window[: min_idx + 1]))


_scorer: ValueExpertScorer | None = None
_scorer_lock = threading.Lock()


def _get_scorer(args) -> ValueExpertScorer:
    global _scorer
    with _scorer_lock:
        if _scorer is None:
            _scorer = ValueExpertScorer(
                checkpoint_path=args.value_expert_ckpt,
                num_clip_frames=args.num_clip_frames,
                dinov2_model=args.dinov2_model,
                attn_heads=args.attn_heads,
                attn_layers=args.attn_layers,
                hidden_dim=args.value_hidden_dim,
            )
            logging.info(f"Value expert loaded: {args.value_expert_ckpt}")
    return _scorer


_policy = None
_policy_lock = threading.Lock()


def _get_policy(args):
    global _policy
    if not args.policy_host:
        return None
    with _policy_lock:
        if _policy is None:
            from clients import OpenpiClient
            _policy = OpenpiClient(host=args.policy_host, port=args.policy_port,
                                   mode=args.policy_mode or None)
            logging.info(
                f"Policy server: {args.policy_host}:{args.policy_port} "
                f"(mode={args.policy_mode or 'default'})"
            )
    return _policy


def _dd_generate(host: str, port: int, frame: np.ndarray, actions: np.ndarray,
                 save_name: str, prompt: str, timeout: float = 600.0) -> str | None:
    url = f"http://{host}:{port}/generate"
    h, w = frame.shape[:2]
    payload = {
        "frame": base64.b64encode(frame.tobytes()).decode(),
        "frame_height": h,
        "frame_width": w,
        "actions": actions.tolist(),
        "save_name": save_name,
        "prompt": prompt,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["save_path"]
    except Exception as e:
        logging.error(f"[DD {host}:{port}] {save_name}: {e}")
        return None


def _find_t_crit(ep_dir: pathlib.Path) -> int | None:
    for i, p in enumerate(sorted(ep_dir.glob("step_*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            if float(d["switch_label"]) >= 0.5:
                return i
    return None


def _load_step(step_path: pathlib.Path) -> dict:
    with np.load(step_path, allow_pickle=True) as d:
        return {
            "top": d["top"].copy(),
            "left": d["left"].copy(),
            "right": d["right"].copy(),
            "state": d["state"].copy(),
            "actions": d["actions"].copy(),
            "frame_idx": int(d["frame_idx"]),
        }


def _get_action_chunks(args, obs: dict, task_text: str) -> list[np.ndarray]:
    policy = _get_policy(args)
    if policy is None:
        return [np.asarray(obs["actions"], dtype=np.float32)]
    payload = {
        "top": obs["top"], "left": obs["left"], "right": obs["right"],
        "state": obs["state"], "instruction": task_text,
        "action_prefix": None, "delay": None,
    }
    return [np.asarray(policy.predict_action(payload), dtype=np.float32)
            for _ in range(args.num_samples)]


def _process_one(args, ep_idx: int, delta: int, ep_dir: pathlib.Path,
                 out_dir: pathlib.Path) -> dict:
    out_path = out_dir / f"ep{ep_idx:03d}_delta{delta}.json"
    if out_path.exists() and not args.overwrite:
        try:
            cached = json.loads(out_path.read_text())
            if isinstance(cached.get("result"), dict) and "agg_best" in cached["result"]:
                return {"ep": ep_idx, "delta": delta, "skipped": "cached",
                        "agg_best": cached["result"]["agg_best"]}
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

    obs = _load_step(step_path)
    chunks = _get_action_chunks(args, obs, args.task_text)

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"dd_ep{ep_idx}_d{delta}_"))
    try:
        save_paths: list[str | None] = [None] * len(chunks)
        def _submit(i: int) -> tuple[int, str | None]:
            return i, _dd_generate(
                args.dd_host, args.dd_port, obs["top"], chunks[i],
                save_name=f"{tmp_dir.name}/cand_{i}",
                prompt=args.task_text,
                timeout=args.dd_timeout,
            )
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, args.dd_parallel)) as ex:
            for fut in concurrent.futures.as_completed(
                    [ex.submit(_submit, i) for i in range(len(chunks))]):
                i, p = fut.result()
                save_paths[i] = p

        scorer = _get_scorer(args)
        per_cand_scores: list[float] = []
        per_cand_windows: list[list[float]] = []
        videos_kept: list[str] = []
        for i, sp in enumerate(save_paths):
            if not sp or not os.path.exists(sp):
                continue
            dst = tmp_dir / f"cand_{i}.mp4"
            try:
                shutil.copy2(sp, dst)
                videos_kept.append(str(dst))
            except Exception:
                videos_kept.append(sp)
            try:
                frames = [np.asarray(f) for f in imageio.mimread(videos_kept[-1])]
            except Exception as e:
                logging.error(f"ep{ep_idx:03d} Δ={delta} cand {i}: read fail {e}")
                continue
            per_window = scorer.score_video(frames)
            per_cand_scores.append(scorer.aggregate(per_window))
            per_cand_windows.append(per_window.tolist())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not per_cand_scores:
        rec = {"ep": ep_idx, "delta": delta, "t_crit": t_crit, "t_pre": t_pre,
               "result": {"error": "no_valid_candidates"}}
        out_path.write_text(json.dumps(rec, indent=2))
        return rec

    arr = np.asarray(per_cand_scores, dtype=np.float32)
    result = {
        "num_candidates": int(len(arr)),
        "scores": arr.tolist(),
        "per_window": per_cand_windows,
        "agg_best": float(arr.min()),
        "agg_mean": float(arr.mean()),
    }
    rec = {
        "ep": ep_idx, "delta": delta, "t_crit": t_crit, "t_pre": t_pre,
        "frame_idx": obs["frame_idx"],
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
        logging.error("sklearn not installed.")
        return

    logging.info("=== Evaluation (GT: final success < 0.5 -> failure) ===")
    for delta in deltas:
        for key in ("agg_best", "agg_mean"):
            y_true, y_score = [], []
            for ep_idx, _ in ep_list:
                p = out_dir / f"ep{ep_idx:03d}_delta{delta}.json"
                if not p.exists():
                    continue
                rec = json.loads(p.read_text())
                res = rec.get("result") or {}
                if key not in res:
                    continue
                s_path = rewards_dir / f"episode_{ep_idx:06d}_success.npy"
                if not s_path.exists():
                    continue
                y_true.append(1 if float(np.load(s_path)[-1]) < 0.5 else 0)
                y_score.append(float(res[key]))
            yt = np.asarray(y_true)
            ys = np.asarray(y_score)
            n_pos, n_neg = int(yt.sum()), int(len(yt) - yt.sum())
            if n_pos == 0 or n_neg == 0:
                logging.info(
                    f"Δ={delta} {key}: n={len(yt)} fail={n_pos} succ={n_neg} — degenerate")
                continue
            auroc = roc_auc_score(yt, ys)
            ap = average_precision_score(yt, ys)
            logging.info(
                f"Δ={delta} {key}: n={len(yt)} fail={n_pos} succ={n_neg}  "
                f"AUROC={auroc:.3f}  AP={ap:.3f}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default=DEFAULT_LABELS_DIR)
    ap.add_argument("--rewards_dir", default=DEFAULT_REWARDS_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--task_text", default=DEFAULT_TASK_TEXT)
    ap.add_argument("--deltas", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--limit_eps", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--eval_only", action="store_true")

    # DreamDojo
    ap.add_argument("--dd_host", default="127.0.0.1")
    ap.add_argument("--dd_port", type=int, default=8020)
    ap.add_argument("--dd_timeout", type=float, default=600.0)
    ap.add_argument("--dd_parallel", type=int, default=1,
                    help="Concurrent DD requests per step (use >1 only if you have multiple DD servers; currently points at a single port).")

    # Value Expert
    ap.add_argument("--value_expert_ckpt", default=DEFAULT_VALUE_CKPT)
    ap.add_argument("--num_clip_frames", type=int, default=4)
    ap.add_argument("--dinov2_model", default="dinov2_vitb14")
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--attn_layers", type=int, default=2)
    ap.add_argument("--value_hidden_dim", type=int, default=512)

    # Optional policy server (mode B)
    ap.add_argument("--policy_host", default=None,
                    help="If set, query policy server for action chunks instead of using cached npz actions.")
    ap.add_argument("--policy_port", type=int, default=8000)
    ap.add_argument("--policy_mode", default=None, choices=[None, "ode", "sde"],
                    help="Routed mode for serve_combined_policy.py; omit if single-mode server.")
    ap.add_argument("--num_samples", type=int, default=1,
                    help="Number of action-chunk samples per step (>1 only meaningful if policy is stochastic, e.g. SDE).")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.policy_host and args.num_samples != 1:
        logging.warning("cached-actions mode: forcing --num_samples 1")
        args.num_samples = 1

    labels_dir = pathlib.Path(args.labels_dir)
    out_dir = pathlib.Path(args.output_dir)
    rewards_dir = pathlib.Path(args.rewards_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_list = _discover_episodes(labels_dir)
    if args.limit_eps > 0:
        ep_list = ep_list[: args.limit_eps]
    logging.info(f"Found {len(ep_list)} episodes under {labels_dir}")

    if not args.eval_only:
        jobs = [(ep, d, ed) for ep, ed in ep_list for d in args.deltas]
        logging.info(f"Submitting {len(jobs)} jobs (sequential per step, "
                     f"{args.num_samples} candidate(s) each)")
        done = 0
        t0 = time.perf_counter()
        for ep, d, ed in jobs:
            done += 1
            try:
                r = _process_one(args, ep, d, ed, out_dir)
            except Exception as e:
                logging.error(f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d} crashed: {e}")
                continue
            res = r.get("result", {})
            if isinstance(res, dict) and "agg_best" in res:
                logging.info(
                    f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d}  "
                    f"t_crit={r.get('t_crit')} t_pre={r.get('t_pre')}  "
                    f"best={res['agg_best']:.3f}  mean={res['agg_mean']:.3f}  "
                    f"n={res['num_candidates']}"
                )
            elif r.get("skipped"):
                logging.info(f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d}  "
                             f"skip={r['skipped']}")
            elif isinstance(res, dict) and "error" in res:
                logging.warning(f"[{done}/{len(jobs)}] ep{ep:03d} Δ={d}  "
                                f"error={res['error']}")
        logging.info(f"Done {len(jobs)} jobs in {(time.perf_counter()-t0)/60:.1f} min")

    _evaluate(out_dir, rewards_dir, ep_list, args.deltas)


if __name__ == "__main__":
    main()
