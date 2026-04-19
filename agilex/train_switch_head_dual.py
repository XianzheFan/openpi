"""
Train a dual-head regressor that predicts Robometer progress and success jointly.

The backbone/feature path is identical to `train_switch_head_robometer.py`
(DINOv2 × 3 cameras + proprio state); only the final head becomes a 2-D output
and the loss is BCE-with-logits on both targets (progress, success).

At inference you can combine the two scores however you like, e.g.:
    rescue_prob = 1 − 0.5·(sigmoid(logit_p) + sigmoid(logit_s))
without re-training.

Usage
-----
    python train_switch_head_dual.py \
        --data_dir ../data/agilex_switch_labels_pnp_cup_0415 \
        --val_ratio 0.1 \
        --wandb_project switch_head \
        --wandb_run_name switch_head_dual_0418
"""

import argparse
import glob
import logging
import os
import pathlib

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

from train_switch_head_robometer import (
    _collate_switch,
    _setup_distributed,
    _cleanup_distributed,
)


class AgilexDualLabelDataset(Dataset):
    """Returns (images, state, [progress, success]) per packed .npz step."""

    IMAGE_KEYS = ["top", "right", "left"]

    def __init__(
        self,
        data_dir: str,
        use_clip: bool = True,
        files: list | None = None,
    ):
        if files is not None:
            self.files = sorted(files)
        else:
            self.files = sorted(glob.glob(os.path.join(data_dir, "rollout_*", "step_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        first = np.load(self.files[0], allow_pickle=True)
        if "progress_score" not in first or "success_prob" not in first:
            raise KeyError(
                "Packed .npz must contain 'progress_score' and 'success_prob'. "
                "Re-pack with train_switch_head_robometer.py pack ..."
            )
        self.state_dim = first["state"].shape[0]
        self.use_clip = use_clip and all(f"{k}_clip" in first for k in self.IMAGE_KEYS)

        valid = []
        p_vals, s_vals = [], []
        for f in self.files:
            try:
                d = np.load(f)
                p_vals.append(float(d["progress_score"]))
                s_vals.append(float(d["success_prob"]))
                valid.append(f)
            except Exception:
                logging.warning(f"  Skipping corrupt file: {f}")
        self.files = valid
        logging.info(
            f"  Dual-head dataset: {len(valid)} samples | "
            f"progress μ={np.mean(p_vals):.3f} σ={np.std(p_vals):.3f} | "
            f"success μ={np.mean(s_vals):.3f} σ={np.std(s_vals):.3f} | "
            f"clip={self.use_clip}"
        )

    def __len__(self):
        return len(self.files)

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
        label = torch.tensor(
            [float(data["progress_score"]), float(data["success_prob"])],
            dtype=torch.float32,
        )
        return images, state, label


class DINOv2DualHead(nn.Module):
    """DINOv2 × 3-camera feature extractor with two regression heads (progress, success)."""

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
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Linear(hidden_dim, 2)  # [progress_logit, success_logit]

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
            return feat_flat.view(B, T, -1).mean(dim=1)
        return feat_flat

    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        feats = [self._encode_image(img) for img in images]
        combined = torch.cat(feats + [state], dim=-1)
        return self.head(self.trunk(combined))  # (B, 2) logits

    def predict(self, images, state):
        """Returns (progress_hat, success_hat) in [0,1]."""
        return torch.sigmoid(self.forward(images, state))


def _split_by_rollout(data_dir: str, val_ratio: float, seed: int):
    import random
    files = sorted(glob.glob(os.path.join(data_dir, "rollout_*", "step_*.npz")))
    rollouts: dict[str, list[str]] = {}
    for f in files:
        rid = os.path.basename(os.path.dirname(f))
        rollouts.setdefault(rid, []).append(f)
    rollout_ids = sorted(rollouts.keys())
    rng = random.Random(seed)
    shuffled = rollout_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_ids = set(shuffled[:n_val])
    train_files = [f for r in rollout_ids if r not in val_ids for f in rollouts[r]]
    val_files = [f for r in rollout_ids if r in val_ids for f in rollouts[r]]
    logging.info(
        f"Split by rollout (seed={seed}): "
        f"{len(rollout_ids) - n_val} train / {n_val} val rollouts "
        f"({len(train_files)} / {len(val_files)} steps)"
    )
    return train_files, val_files


def train(args):
    rank, local_rank, world_size = _setup_distributed()
    distributed = world_size > 1
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        logging.info(f"Using device: {device}  (world_size={world_size})")

    if args.val_ratio > 0:
        train_files, val_files = _split_by_rollout(args.data_dir, args.val_ratio, args.val_split_seed)
        train_ds = AgilexDualLabelDataset(args.data_dir, use_clip=args.use_clip, files=train_files)
        val_ds = AgilexDualLabelDataset(args.data_dir, use_clip=args.use_clip, files=val_files)
    else:
        train_ds = AgilexDualLabelDataset(args.data_dir, use_clip=args.use_clip)
        val_ds = None

    state_dim = train_ds.state_dim
    model = DINOv2DualHead(
        dinov2_model=args.dinov2_model,
        hidden_dim=args.hidden_dim,
        state_dim=state_dim,
        num_cameras=3,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    raw_model = model.module if distributed else model

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate_switch,
    )
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed and val_ds else None
    val_loader = (
        DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_switch,
        )
        if val_ds else None
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()  # targets are already in [0,1]

    output_dir = pathlib.Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    use_wandb = is_main and wandb is not None and args.wandb_project
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args) | {"world_size": world_size, "mode": "dual_head"},
        )

    best_val = float("inf")
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        if args.freeze_backbone:
            raw_model.backbone.eval()

        total_loss = total_p = total_s = 0.0
        nb = 0
        for images, state, label in train_loader:
            images = [img.to(device) for img in images]
            state = state.to(device)
            label = label.to(device)  # (B, 2)

            logits = model(images, state)  # (B, 2)
            loss_p = criterion(logits[:, 0], label[:, 0])
            loss_s = criterion(logits[:, 1], label[:, 1])
            loss = args.progress_weight * loss_p + args.success_weight * loss_s

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_p += loss_p.item()
            total_s += loss_s.item()
            nb += 1

        avg_loss = total_loss / max(nb, 1)
        avg_p = total_p / max(nb, 1)
        avg_s = total_s / max(nb, 1)
        scheduler.step()

        val_msg = "N/A"
        if val_loader is not None:
            model.eval()
            vl = vp = vs = 0.0
            vn = 0
            mae_p = mae_s = 0.0
            with torch.no_grad():
                for images, state, label in val_loader:
                    images = [img.to(device) for img in images]
                    state = state.to(device)
                    label = label.to(device)
                    logits = model(images, state)
                    lp = criterion(logits[:, 0], label[:, 0])
                    ls = criterion(logits[:, 1], label[:, 1])
                    vp += lp.item()
                    vs += ls.item()
                    vl += (args.progress_weight * lp + args.success_weight * ls).item()
                    pred = torch.sigmoid(logits)
                    mae_p += (pred[:, 0] - label[:, 0]).abs().mean().item()
                    mae_s += (pred[:, 1] - label[:, 1]).abs().mean().item()
                    vn += 1
            avg_vl = vl / max(vn, 1)
            val_msg = (
                f"loss={avg_vl:.4f} "
                f"p_bce={vp/max(vn,1):.4f} s_bce={vs/max(vn,1):.4f} "
                f"p_mae={mae_p/max(vn,1):.4f} s_mae={mae_s/max(vn,1):.4f}"
            )
            if is_main and avg_vl < best_val:
                best_val = avg_vl
                torch.save(raw_model.state_dict(), output_dir / "best_model.pt")

        if is_main:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={avg_loss:.4f} (p={avg_p:.4f} s={avg_s:.4f})  "
                f"val=[{val_msg}]  lr={scheduler.get_last_lr()[0]:.2e}"
            )
        if use_wandb:
            log = {
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/progress_bce": avg_p,
                "train/success_bce": avg_s,
                "lr": scheduler.get_last_lr()[0],
            }
            if val_loader is not None:
                log.update({
                    "val/loss": avg_vl,
                    "val/progress_bce": vp / max(vn, 1),
                    "val/success_bce": vs / max(vn, 1),
                    "val/progress_mae": mae_p / max(vn, 1),
                    "val/success_mae": mae_s / max(vn, 1),
                })
            wandb.log(log, step=epoch + 1)

        if is_main and (epoch + 1) % args.save_every == 0:
            torch.save(raw_model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    if is_main:
        torch.save(raw_model.state_dict(), output_dir / "model_final.pt")
    if use_wandb:
        wandb.finish()
    _cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Dual-head regressor (progress + success)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="checkpoints/switch_head_dual")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--use_clip", action="store_true", default=True)
    parser.add_argument("--no_clip", dest="use_clip", action="store_false")
    parser.add_argument("--progress_weight", type=float, default=1.0)
    parser.add_argument("--success_weight", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(args)


if __name__ == "__main__":
    main()
