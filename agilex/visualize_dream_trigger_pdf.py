"""
Build a multi-page PDF where each page = 4 right_wrist frames (top row) +
the p_t / y_tilde curve (bottom row) for one validation rollout.

Frames are sampled at quartiles of the rollout's step list and each pick is
marked on the curve as a vertical tick.

Font: prefers Nunito (drop a .ttf into --font_dir or pass --font_path).
Falls back to DejaVu Sans. Writes PDF with `pdf.fonttype = 42` so glyphs are
embedded as TrueType (editable in Illustrator / Inkscape — what people
usually mean by "Type 1-like editable PDF").

Usage
-----
    python visualize_dream_trigger_pdf.py \\
        --data_dir ../data/dream_trigger_screw_0426 \\
        --ckpt ../checkpoints/dream_trigger/dt_screw_0426/model_final.pt \\
        --val_ratio 0.1 --val_split_seed 42 \\
        --gamma 0.5 \\
        --font_dir dream_trigger/fonts \\
        --output dream_trigger/dt_screw_0426/val_curves.pdf
"""

import argparse
import glob
import logging
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from train_switch_head_dual import _split_by_rollout
from train_dream_trigger import DreamTrigger


# Colors
KEY_BORDER_COLOR = "#c1ff72"  # key-frame border
TCRIT_LINE_COLOR = "#A4E288"  # t_crit vertical line
DT_COLOR = "#5170FF"          # dream trigger p_t curve
SOFT_COLOR = "#CB6CE6"        # soft label y_tilde curve

# Rounded-corner radius (in pixel units of the image)
ROUNDING_RADIUS = 30


def _brighten(img: np.ndarray, factor: float = 1.35, bias: float = 28.0) -> np.ndarray:
    out = img.astype(np.float32) * factor + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def _show_rounded_image(ax, img, edge_color=None, edge_width=0.0,
                        rounding=ROUNDING_RADIUS):
    """Draw img with rounded corners; if edge_width<=0, no border is drawn."""
    H, W = img.shape[:2]
    ax.patch.set_alpha(0.0)  # transparent so overlapping axes show through
    im = ax.imshow(img, interpolation="none", resample=False, aspect="auto")
    has_border = edge_color is not None and edge_width > 0
    bbox = FancyBboxPatch(
        (0, 0), W, H,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        fc="none",
        ec=edge_color if has_border else "none",
        lw=edge_width if has_border else 0.0,
        transform=ax.transData, zorder=10,
    )
    ax.add_patch(bbox)
    im.set_clip_path(bbox)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


def _setup_font(font_dir: str | None, font_path: str | None):
    """Register Nunito if available; return the family name to use."""
    candidates = []
    if font_path:
        candidates.append(pathlib.Path(font_path))
    if font_dir:
        for p in pathlib.Path(font_dir).glob("Nunito*.ttf"):
            candidates.append(p)
        for p in pathlib.Path(font_dir).glob("Nunito*.otf"):
            candidates.append(p)

    family = None
    for p in candidates:
        if not p.exists():
            continue
        fm.fontManager.addfont(str(p))
        try:
            family = fm.FontProperties(fname=str(p)).get_name()
            logging.info(f"Registered font {family} from {p}")
        except Exception as e:
            logging.warning(f"Could not register {p}: {e}")

    if family is None:
        logging.warning("Nunito not found; falling back to DejaVu Sans.")
        family = "DejaVu Sans"

    matplotlib.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family, "DejaVu Sans"],
        "pdf.fonttype": 42,   # embed TrueType (editable, paper-friendly)
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    return family


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
        "top_frame": _brighten(d["top_clip"][-1].copy()),
        "left_frame": _brighten(d["left_clip"][-1].copy()),
        "right_frame": _brighten(d["right_clip"][-1].copy()),
    }


def _rollout_short(rid: str) -> str:
    if "_ep" in rid:
        return "ep" + rid.split("_ep")[1].split("_")[0]
    return rid


def render_rollout_page(model, rollout_dir: str, device, gamma: float, family: str):
    files = sorted(glob.glob(os.path.join(rollout_dir, "step_*.npz")))
    if not files:
        return None

    frame_idx, y_tilde, p_t = [], [], []
    top_frames, left_frames, right_frames = [], [], []
    t_crit = None

    model.eval()
    with torch.no_grad():
        for f in files:
            step = _load_step(f)
            images = [img.unsqueeze(0).to(device) for img in step["images"]]
            state = step["state"].unsqueeze(0).to(device)
            p = float(model.predict_prob(images, state)[0].item())
            frame_idx.append(step["frame_idx"])
            y_tilde.append(step["y_tilde"])
            p_t.append(p)
            top_frames.append(step["top_frame"])
            left_frames.append(step["left_frame"])
            right_frames.append(step["right_frame"])
            if t_crit is None:
                t_crit = step["t_crit"]

    n = len(files)
    frame_idx = np.asarray(frame_idx)
    y_tilde = np.asarray(y_tilde)
    p_t = np.asarray(p_t)

    # Pick 4 anchor steps: early, pre-crit, at-crit (key frame), late.
    crit_step = int(np.argmin(np.abs(frame_idx - t_crit))) if t_crit is not None else n // 2
    early = int(round(n * 0.10))
    early = min(early, max(0, crit_step - 2))
    late = int(round(n * 0.90))
    late = max(late, min(n - 1, crit_step + 2))
    pre = (early + crit_step) // 2
    pick_pos = [
        max(0, min(n - 1, early)),
        max(0, min(n - 1, pre)),
        max(0, min(n - 1, crit_step)),
        max(0, min(n - 1, late)),
    ]

    rid = pathlib.Path(rollout_dir).name
    short = _rollout_short(rid)
    mae = float(np.mean(np.abs(p_t - y_tilde)))

    fig = plt.figure(figsize=(22, 5.0))

    # ---- Layout (figure-normalized coords) ----
    # All 12 images on a single horizontal row, grouped into 4 trios of
    # (main, left wrist, right wrist). Adjacent images within a trio overlap
    # 1/5 of width; left wrist sits on the bottom layer.
    margin_l = 0.045
    margin_r = 0.025
    margin_t = 0.085     # room for trio labels at the top
    margin_b = 0.065
    curve_h = 0.50
    curve_gap = 0.012

    inner_w = 1.0 - margin_l - margin_r
    n_trios = 4
    trio_gap = 0.010
    # Each trio horizontally: main + left + right with 1/5 overlap.
    # trio span = w + 2*(w - 0.2w) = 2.6 * w.
    # Total: n_trios * 2.6w + (n_trios-1) * trio_gap = inner_w
    img_w = (inner_w - (n_trios - 1) * trio_gap) / (n_trios * 2.6)
    overlap_x = img_w * 0.20
    trio_w = 2.6 * img_w

    row_h = 1.0 - margin_t - margin_b - curve_h - curve_gap
    row_bot = margin_b + curve_h + curve_gap

    # Stacking order (bottom -> top): main, left wrist, right wrist.
    cam_views = [
        ("main view", top_frames, 1),
        ("left wrist view", left_frames, 2),
        ("right wrist view", right_frames, 3),
    ]
    trio_titles = ["early", "pre-crit", r"key frame ($t_{crit}$)", "late"]

    for trio_idx, idx in enumerate(pick_pos):
        trio_left = margin_l + trio_idx * (trio_w + trio_gap)
        is_key = (trio_idx == 2)

        for cam_idx, (cam_name, frames_list, zord) in enumerate(cam_views):
            x = trio_left + cam_idx * (img_w - overlap_x)
            ax = fig.add_axes([x, row_bot, img_w, row_h], zorder=zord)
            if is_key:
                _show_rounded_image(ax, frames_list[idx],
                                    edge_color=KEY_BORDER_COLOR,
                                    edge_width=4.0,
                                    rounding=ROUNDING_RADIUS)
            else:
                _show_rounded_image(ax, frames_list[idx],
                                    edge_color=None, edge_width=0.0,
                                    rounding=ROUNDING_RADIUS)
            # Camera-name overlay only on the first trio.
            if trio_idx == 0:
                ax.text(
                    0.04, 0.96, cam_name,
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=12, color="white",
                    bbox=dict(boxstyle="round,pad=0.22", fc="#000000aa",
                              ec="none"),
                )

        # Trio label hugging the image top
        fig.text(
            trio_left + trio_w / 2,
            row_bot + row_h + 0.005,
            f"{trio_titles[trio_idx]}    "
            f"f={frame_idx[idx]}, p={p_t[idx]:.2f}, ỹ={y_tilde[idx]:.2f}",
            ha="center", va="bottom",
            fontsize=14, color="#222",
        )

    ax_curve = fig.add_axes([margin_l, margin_b, inner_w, curve_h], zorder=2)
    ax_curve.plot(frame_idx, y_tilde, color=SOFT_COLOR, lw=2.6,
                  label=r"$\tilde{y}_t$ (soft label)")
    ax_curve.plot(frame_idx, p_t, color=DT_COLOR, lw=2.6,
                  label=r"$p_t$ (Dream Trigger)")
    ax_curve.axhline(gamma, color="black", ls="--", lw=1.2, alpha=0.55,
                     label=fr"$\gamma={gamma}$")
    if t_crit is not None:
        ax_curve.axvline(t_crit, color=TCRIT_LINE_COLOR, ls="--", lw=2.0, alpha=0.95,
                         label=r"$t_{crit}$")

    ax_curve.set_ylabel("probability", fontsize=13)
    ax_curve.tick_params(axis="both", labelsize=11)
    ax_curve.set_ylim(-0.02, 1.02)
    # x-axis title moved inside the plot (top-center)
    ax_curve.text(
        0.5, 0.985,
        "frame index (extracted FPS)",
        transform=ax_curve.transAxes,
        ha="center", va="top",
        fontsize=15, color="#444",
    )
    leg = ax_curve.legend(
        loc="lower right", fontsize=14,
        frameon=True, fancybox=True, framealpha=1.0,
        edgecolor="#bbb", facecolor="white",
    )
    leg.get_frame().set_linewidth(0.6)
    ax_curve.grid(alpha=0.25)
    ax_curve.text(
        0.012, 0.965,
        fr"MAE($p_t$, $\tilde{{y}}_t$)$={mae:.3f}$",
        transform=ax_curve.transAxes,
        ha="left", va="top",
        fontsize=13, color="#111",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.6),
    )

    return fig, {"rollout": rid, "n_steps": n, "mae": mae,
                 "t_crit": float(t_crit) if t_crit is not None else None,
                 "y_mean": float(y_tilde.mean()), "p_mean": float(p_t.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_split_seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--output", required=True, help="Output PDF path")
    p.add_argument("--font_dir", default="dream_trigger/fonts",
                   help="Directory containing Nunito*.ttf / .otf (optional)")
    p.add_argument("--font_path", default=None,
                   help="Direct path to Nunito ttf/otf (optional)")
    p.add_argument("--rollout", default=None,
                   help="Render only this rollout id (otherwise all val rollouts)")
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

    family = _setup_font(args.font_dir, args.font_path)

    out_dir = pathlib.Path(args.output)
    if out_dir.suffix.lower() == ".pdf":
        out_dir = out_dir.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_files = _split_by_rollout(args.data_dir, args.val_ratio, args.val_split_seed)
    val_rollouts = sorted({pathlib.Path(f).parent.as_posix() for f in val_files})
    if args.rollout is not None:
        val_rollouts = [r for r in val_rollouts if pathlib.Path(r).name == args.rollout]
        if not val_rollouts:
            raise ValueError(f"--rollout {args.rollout} not in val split")
    logging.info(f"val rollouts: {len(val_rollouts)} | font={family}")

    first = np.load(sorted(glob.glob(os.path.join(val_rollouts[0], "step_*.npz")))[0])
    state_dim = first["state"].shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DreamTrigger(
        dinov2_model=args.dinov2_model,
        state_dim=state_dim, num_cameras=3,
        max_clip_frames=args.max_clip_frames,
        attn_dim=args.attn_dim, attn_heads=args.attn_heads,
        attn_layers=args.attn_layers, hidden_dim=args.hidden_dim,
        dropout=0.0, freeze_backbone=True,
    ).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    logging.info(f"Loaded {args.ckpt}")

    summaries = []
    for rd in val_rollouts:
        rid = pathlib.Path(rd).name
        logging.info(f"rendering {rid}")
        res = render_rollout_page(model, rd, device, args.gamma, family)
        if res is None:
            continue
        fig, summary = res
        ep_short = _rollout_short(rid)  # "ep10"
        out_path = out_dir / f"{ep_short}_steps{summary['n_steps']}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=200,
                    metadata={"Title": f"Dream Trigger | {rid}",
                              "Subject": "main / left wrist / right wrist + p_t curve"})
        plt.close(fig)
        summary["out"] = str(out_path)
        summaries.append(summary)

    logging.info(f"\nWrote {len(summaries)} per-rollout PDFs to {out_dir}")
    for s in summaries:
        logging.info(
            f"  {s['rollout']:55s}  steps={s['n_steps']:3d}  "
            f"t_crit={s['t_crit']:.1f}  MAE={s['mae']:.3f}"
        )


if __name__ == "__main__":
    main()
