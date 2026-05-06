"""
Visualize the trained Dream Trigger on a validation rollout.

For each packed step (step_*.npz) of the chosen rollout we run the trained
f_phi(O, s) and plot:
  * p_t           — model output (sigmoid logit)
  * y_tilde_t     — packed soft label sigmoid((t - t_crit)/beta)
  * gamma         — dashed horizontal trigger threshold
  * t_crit        — vertical dashed line (read from the .npz)

Usage
-----
    python visualize_dream_trigger.py \\
        --data_dir ../data/dream_trigger_screw_0426 \\
        --ckpt ../checkpoints/dream_trigger/dt_screw_0426/model_final.pt \\
        --val_ratio 0.1 --val_split_seed 42 \\
        --gamma 0.5 \\
        --output ../checkpoints/dream_trigger/dt_screw_0426/val_curves
"""

import argparse
import glob
import logging
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_switch_head_dual import _split_by_rollout
from train_dream_trigger import DreamTrigger


def _load_step(path: str):
    d = np.load(path, allow_pickle=True)
    images = []
    for k in ("top_clip", "right_clip", "left_clip"):
        clip = torch.from_numpy(d[k].copy()).permute(0, 3, 1, 2).float() / 255.0
        images.append(clip)
    state = torch.from_numpy(d["state"].astype(np.float32))
    return {
        "images": images,
        "state": state,
        "y_tilde": float(d["y_tilde"]),
        "t_crit": float(d["t_crit"]),
        "frame_idx": int(d["frame_idx"]),
    }


def _rollout_to_episode(rollout_id: str) -> str:
    # rollout_<task>_ep<N>_done -> "ep<N>"
    if "_ep" in rollout_id:
        return "ep" + rollout_id.split("_ep")[1].split("_")[0]
    return rollout_id


def visualize_rollout(model, rollout_dir: str, device, gamma: float, out_path: pathlib.Path):
    files = sorted(glob.glob(os.path.join(rollout_dir, "step_*.npz")))
    if not files:
        logging.warning(f"  no steps in {rollout_dir}")
        return None

    frame_idx, y_tilde, p_t = [], [], []
    t_crit = None

    model.eval()
    with torch.no_grad():
        for f in files:
            step = _load_step(f)
            images = [img.unsqueeze(0).to(device) for img in step["images"]]
            state = step["state"].unsqueeze(0).to(device)
            p = model.predict_prob(images, state)
            p_t.append(float(p[0].item()))
            y_tilde.append(step["y_tilde"])
            frame_idx.append(step["frame_idx"])
            if t_crit is None:
                t_crit = step["t_crit"]

    frame_idx = np.asarray(frame_idx)
    y_tilde = np.asarray(y_tilde)
    p_t = np.asarray(p_t)

    rid = pathlib.Path(rollout_dir).name
    short = _rollout_to_episode(rid)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frame_idx, y_tilde, color="#4c72b0", lw=2, label=r"$\tilde{y}_t$ (soft label)")
    ax.plot(frame_idx, p_t,    color="#dd8452", lw=2, label=r"$p_t$ (model)")
    ax.axhline(gamma, color="black", ls="--", lw=1, alpha=0.5, label=fr"$\gamma={gamma}$")
    if t_crit is not None:
        ax.axvline(t_crit, color="red", ls="--", lw=1, alpha=0.6,
                   label=fr"$t_{{crit}}={t_crit:.1f}$")
    ax.set_xlabel("frame index (extracted FPS)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Dream Trigger | {short}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    mae = float(np.mean(np.abs(p_t - y_tilde)))
    return {
        "rollout": rid,
        "n_steps": len(files),
        "t_crit": float(t_crit) if t_crit is not None else None,
        "p_mean": float(p_t.mean()),
        "y_mean": float(y_tilde.mean()),
        "mae": mae,
        "out": str(out_path),
    }


def main():
    p = argparse.ArgumentParser(
        description="Plot Dream Trigger p_t / y_tilde curves on val rollouts."
    )
    p.add_argument("--data_dir", required=True,
                   help="Packed dream-trigger dataset (rollout_*/step_*.npz)")
    p.add_argument("--ckpt", required=True, help="Trained DreamTrigger .pt")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_split_seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--output", required=True,
                   help="Output dir for PNG curves (one per rollout)")
    p.add_argument("--rollout", default=None,
                   help="Visualize only this rollout id (e.g. 'rollout_..._ep7_done'). "
                        "Default: all val rollouts.")
    p.add_argument("--dinov2_model", default="dinov2_vitb14",
                   choices=["dinov2_vits14", "dinov2_vitb14",
                            "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--max_clip_frames", type=int, default=20)
    p.add_argument("--attn_dim", type=int, default=384)
    p.add_argument("--attn_heads", type=int, default=4)
    p.add_argument("--attn_layers", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=256)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_files = _split_by_rollout(args.data_dir, args.val_ratio, args.val_split_seed)
    val_rollouts = sorted({pathlib.Path(f).parent.as_posix() for f in val_files})
    logging.info(f"val rollouts: {len(val_rollouts)}")
    for r in val_rollouts:
        logging.info(f"  {pathlib.Path(r).name}")

    if args.rollout is not None:
        val_rollouts = [r for r in val_rollouts if pathlib.Path(r).name == args.rollout]
        if not val_rollouts:
            raise ValueError(f"--rollout {args.rollout} not in val split")

    first = np.load(sorted(glob.glob(os.path.join(val_rollouts[0], "step_*.npz")))[0])
    state_dim = first["state"].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DreamTrigger(
        dinov2_model=args.dinov2_model,
        state_dim=state_dim,
        num_cameras=3,
        max_clip_frames=args.max_clip_frames,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        hidden_dim=args.hidden_dim,
        dropout=0.0,
        freeze_backbone=True,
    ).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    logging.info(f"Loaded {args.ckpt}")

    summaries = []
    for rd in val_rollouts:
        rid = pathlib.Path(rd).name
        out_path = output_dir / f"{rid}.png"
        logging.info(f"plotting {rid} -> {out_path}")
        s = visualize_rollout(model, rd, device, args.gamma, out_path)
        if s is not None:
            summaries.append(s)

    if summaries:
        logging.info(f"\nWrote {len(summaries)} plots to {output_dir}:")
        for s in summaries:
            logging.info(
                f"  {s['rollout']:55s}  steps={s['n_steps']:3d}  "
                f"t_crit={s['t_crit']:.1f}  "
                f"y_mean={s['y_mean']:.3f}  p_mean={s['p_mean']:.3f}  "
                f"MAE={s['mae']:.3f}"
            )


if __name__ == "__main__":
    main()
