"""
Inference helpers for the temporal-attention switch head v2.

Two entry points:

  1. `StandaloneSwitchHeadV2` — drop-in replacement for
     `StandaloneSwitchHead` in `agilex_infer_dinov2_value_switch.py`. Exposes
     `.predict(top, right, left, state) -> (progress, success)` where each
     camera arg is a single (H,W,3) uint8 frame or a list of such (clip).

  2. `python infer_switch_head_v2.py --ckpt ... --data_dir ...` — offline
     evaluation: runs the model on every packed .npz step under a data dir,
     computes per-sample (progress, success) predictions, and writes
     per-rollout plots + aggregate metrics into an output directory.

The packed format is the same as the training data (produced by
`train_switch_head_robometer.py pack`). No re-packing required.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pathlib

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_switch_head_v2 import DINOv2TemporalSwitchHead


# ---------------------------------------------------------------------------
#  Live-inference wrapper (importable from the policy server)
# ---------------------------------------------------------------------------

class StandaloneSwitchHeadV2:
    """Wraps `DINOv2TemporalSwitchHead` for online per-step scoring."""

    def __init__(
        self,
        checkpoint_path: str,
        dinov2_model: str = "dinov2_vitb14",
        state_dim: int = 14,
        num_cameras: int = 3,
        max_clip_frames: int = 20,
        attn_dim: int = 384,
        attn_heads: int = 4,
        attn_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = DINOv2TemporalSwitchHead(
            dinov2_model=dinov2_model,
            state_dim=state_dim,
            num_cameras=num_cameras,
            max_clip_frames=max_clip_frames,
            attn_dim=attn_dim,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=True,
        )
        sd = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    def _frame_to_tensor(self, frame_hwc: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(frame_hwc)).permute(2, 0, 1).float() / 255.0

    @torch.no_grad()
    def predict(self, top, right, left, state_np: np.ndarray) -> tuple[float, float]:
        """
        Returns (progress, success) ∈ [0,1].
        Each camera arg is a single (H,W,3) uint8 frame or a list of such (clip).
        """
        images = []
        for cam in (top, right, left):
            if isinstance(cam, list):
                t = torch.stack([self._frame_to_tensor(f) for f in cam]).unsqueeze(0)
            else:
                t = self._frame_to_tensor(cam).unsqueeze(0)
            images.append(t.to(self.device))
        state_t = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        probs = self.model.predict(images, state_t)
        return float(probs[0, 0].item()), float(probs[0, 1].item())

    @torch.no_grad()
    def predict_batch(
        self, images_tuple: tuple, state_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched version: images_tuple = (top_list, right_list, left_list) of np arrays."""
        cams = []
        for cam_list in images_tuple:
            ts = [self._frame_to_tensor(f) for f in cam_list]
            cams.append(torch.stack(ts).unsqueeze(0).to(self.device))
        state_t = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        probs = self.model.predict(cams, state_t).cpu().numpy()
        return probs[:, 0], probs[:, 1]


# ---------------------------------------------------------------------------
#  Offline evaluation over a packed dataset
# ---------------------------------------------------------------------------

def _list_rollouts(data_dir: str) -> dict[str, list[str]]:
    files = sorted(glob.glob(os.path.join(data_dir, "rollout_*", "step_*.npz")))
    rollouts: dict[str, list[str]] = {}
    for f in files:
        rid = os.path.basename(os.path.dirname(f))
        rollouts.setdefault(rid, []).append(f)
    for rid in rollouts:
        rollouts[rid].sort()
    return rollouts


def _load_step(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    clips = {}
    for key in ("top", "right", "left"):
        ck = f"{key}_clip"
        if ck in d:
            clips[key] = d[ck]  # (T, H, W, 3) uint8
        else:
            clips[key] = d[key][None]
    state = d["state"].astype(np.float32)
    gt_p = float(d["progress_score"]) if "progress_score" in d else float("nan")
    gt_s = float(d["success_prob"]) if "success_prob" in d else float("nan")
    frame_idx = int(d["frame_idx"]) if "frame_idx" in d else -1
    return clips, state, gt_p, gt_s, frame_idx


def _frames_to_tensor(frames_np: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(frames_np.copy()).permute(0, 3, 1, 2).float() / 255.0
    return t.to(device).unsqueeze(0)  # (1, T, 3, H, W)


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = StandaloneSwitchHeadV2(
        checkpoint_path=args.ckpt,
        dinov2_model=args.dinov2_model,
        max_clip_frames=args.max_clip_frames,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        hidden_dim=args.hidden_dim,
        device=str(device),
    )

    rollouts = _list_rollouts(args.data_dir)
    if not rollouts:
        raise FileNotFoundError(f"No rollout_*/step_*.npz under {args.data_dir}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_plots:
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    summary = {
        "ckpt": args.ckpt,
        "data_dir": args.data_dir,
        "num_rollouts": len(rollouts),
        "rollouts": {},
    }
    all_abs_p = []
    all_abs_s = []

    for rid, files in rollouts.items():
        pred_p, pred_s, gt_p_list, gt_s_list, frame_idxs = [], [], [], [], []
        for f in files:
            clips, state, gt_p, gt_s, fidx = _load_step(f)
            cam_tensors = [
                _frames_to_tensor(clips["top"], device),
                _frames_to_tensor(clips["right"], device),
                _frames_to_tensor(clips["left"], device),
            ]
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            probs = scorer.model.predict(cam_tensors, state_t).cpu().numpy()[0]
            pred_p.append(float(probs[0]))
            pred_s.append(float(probs[1]))
            gt_p_list.append(gt_p)
            gt_s_list.append(gt_s)
            frame_idxs.append(fidx)

        pred_p = np.asarray(pred_p)
        pred_s = np.asarray(pred_s)
        gt_p_arr = np.asarray(gt_p_list)
        gt_s_arr = np.asarray(gt_s_list)

        abs_p = np.abs(pred_p - gt_p_arr)
        abs_s = np.abs(pred_s - gt_s_arr)
        all_abs_p.append(abs_p)
        all_abs_s.append(abs_s)

        summary["rollouts"][rid] = {
            "num_steps": len(files),
            "progress_mae": float(np.mean(abs_p)),
            "success_mae": float(np.mean(abs_s)),
            "pred_progress_mean": float(np.mean(pred_p)),
            "pred_success_mean": float(np.mean(pred_s)),
            "gt_progress_mean": float(np.mean(gt_p_arr)),
            "gt_success_mean": float(np.mean(gt_s_arr)),
        }

        if args.save_csv:
            csv_path = out_dir / f"{rid}.csv"
            with open(csv_path, "w") as fp:
                fp.write("step,frame_idx,gt_progress,pred_progress,gt_success,pred_success\n")
                for i in range(len(files)):
                    fp.write(
                        f"{i},{frame_idxs[i]},{gt_p_arr[i]:.5f},{pred_p[i]:.5f},"
                        f"{gt_s_arr[i]:.5f},{pred_s[i]:.5f}\n"
                    )

        if args.save_plots:
            x = np.arange(len(files))
            fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            axes[0].plot(x, gt_p_arr, label="teacher progress", color="C0")
            axes[0].plot(x, pred_p, label="v2 progress", color="C1", linestyle="--")
            axes[0].set_ylabel("progress")
            axes[0].set_ylim(-0.05, 1.05)
            axes[0].legend(loc="best", fontsize=8)
            axes[1].plot(x, gt_s_arr, label="teacher success", color="C0")
            axes[1].plot(x, pred_s, label="v2 success", color="C1", linestyle="--")
            axes[1].set_ylabel("success")
            axes[1].set_xlabel("packed step index")
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].legend(loc="best", fontsize=8)
            fig.suptitle(
                f"{rid} | p_mae={np.mean(abs_p):.3f}  s_mae={np.mean(abs_s):.3f}",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(out_dir / "plots" / f"{rid}.png", dpi=120)
            plt.close(fig)

        logging.info(
            f"{rid}: steps={len(files):3d}  "
            f"p_mae={np.mean(abs_p):.4f}  s_mae={np.mean(abs_s):.4f}"
        )

    all_abs_p = np.concatenate(all_abs_p) if all_abs_p else np.zeros(0)
    all_abs_s = np.concatenate(all_abs_s) if all_abs_s else np.zeros(0)
    summary["aggregate"] = {
        "total_steps": int(all_abs_p.size),
        "progress_mae": float(np.mean(all_abs_p)) if all_abs_p.size else None,
        "success_mae": float(np.mean(all_abs_s)) if all_abs_s.size else None,
        "progress_mae_std": float(np.std(all_abs_p)) if all_abs_p.size else None,
        "success_mae_std": float(np.std(all_abs_s)) if all_abs_s.size else None,
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    logging.info(
        f"[aggregate] rollouts={len(rollouts)}  steps={all_abs_p.size}  "
        f"p_mae={summary['aggregate']['progress_mae']:.4f}  "
        f"s_mae={summary['aggregate']['success_mae']:.4f}"
    )
    logging.info(f"Wrote summary → {out_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline evaluation for switch-head v2 (progress + success soft labels)"
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Packed dataset dir containing rollout_*/step_*.npz")
    parser.add_argument("--output_dir", type=str, default="eval_switch_head_v2")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--max_clip_frames", type=int, default=20)
    parser.add_argument("--attn_dim", type=int, default=384)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--no_plots", dest="save_plots", action="store_false")
    parser.add_argument("--save_csv", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate(args)


if __name__ == "__main__":
    main()
