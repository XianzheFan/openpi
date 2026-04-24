"""
Switch-head v2: temporal-attention variant of the dual-head (progress + success)
soft-label regressor.

Differences vs. `train_switch_head_dual.py`:
  * DINOv2 per-frame CLS tokens across the 3 cameras and 20-frame clip are
    aggregated by a small Temporal Transformer (learnable CLS pooling) instead
    of a plain `mean(dim=1)` over frames.
  * Per-camera and per-time position embeddings preserve view/order.
  * Output still = 2 soft-label logits: (progress, success). Loss is BCE with
    the packed teacher targets `progress_score`, `success_prob`.

Data format is identical to existing packed switch-label datasets, so no
re-packing is required:
    <data_dir>/rollout_*/step_*.npz   with {top,right,left}_clip + state +
                                      progress_score + success_prob

Usage
-----
    python train_switch_head_v2.py \
        --data_dir ../data/agilex_switch_labels_plug_merged \
        --val_ratio 0.1 \
        --wandb_project switch_head --wandb_run_name switch_head_v2_0419
"""

import argparse
import logging
import pathlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
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
from train_switch_head_dual import AgilexDualLabelDataset, _split_by_rollout


class DINOv2TemporalSwitchHead(nn.Module):
    """
    3-camera × T-frame DINOv2 features → Temporal Transformer → (progress, success) logits.

    Token layout per sample (B, 1 + C*T, D):
        [CLS] [cam0_t0, cam0_t1, ..., cam0_{T-1}, cam1_t0, ..., cam2_{T-1}]

    Both a per-camera and a per-time embedding are added; the learnable CLS
    output is concatenated with proprio state and fed to a small MLP trunk.
    """

    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        state_dim: int = 14,
        num_cameras: int = 3,
        max_clip_frames: int = 20,
        attn_dim: int = 384,
        attn_heads: int = 4,
        attn_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.max_clip_frames = max_clip_frames

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

        self.feat_proj = nn.Linear(self.feature_dim, attn_dim)

        self.cam_emb = nn.Parameter(torch.zeros(num_cameras, attn_dim))
        self.time_emb = nn.Parameter(torch.zeros(max_clip_frames, attn_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
        nn.init.trunc_normal_(self.cam_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=attn_heads,
            dim_feedforward=attn_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)

        trunk_in = attn_dim + state_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(trunk_in),
            nn.Linear(trunk_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 2)  # (progress_logit, success_logit)

    def _encode_cam_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (B, T, 3, H, W) or (B, 3, H, W) — returns (B, T, feature_dim).
        """
        if clip.dim() == 4:
            clip = clip.unsqueeze(1)
        B, T, C, H, W = clip.shape
        flat = clip.reshape(B * T, C, H, W)
        flat = F.interpolate(flat, size=(224, 224), mode="bilinear", align_corners=False)
        flat = (flat - self.img_mean) / self.img_std
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(flat)
        else:
            feat = self.backbone(flat)
        return feat.view(B, T, -1)

    def forward(self, images: list[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """
        images: list length num_cameras of tensors (B, T, 3, H, W) or (B, 3, H, W)
        state:  (B, state_dim)
        returns: (B, 2) logits — [progress, success]
        """
        cam_tokens = []
        for c, img in enumerate(images):
            feat = self._encode_cam_clip(img)           # (B, T, feature_dim)
            feat = self.feat_proj(feat)                 # (B, T, attn_dim)
            T = feat.shape[1]
            if T > self.max_clip_frames:
                raise ValueError(
                    f"clip length {T} exceeds max_clip_frames {self.max_clip_frames}"
                )
            feat = feat + self.time_emb[:T].unsqueeze(0)
            feat = feat + self.cam_emb[c].view(1, 1, -1)
            cam_tokens.append(feat)

        tokens = torch.cat(cam_tokens, dim=1)           # (B, C*T, attn_dim)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, attn_dim)
        seq = torch.cat([cls, tokens], dim=1)           # (B, 1 + C*T, attn_dim)
        seq = self.temporal_attn(seq)
        pooled = seq[:, 0]

        combined = torch.cat([pooled, state], dim=-1)
        return self.head(self.trunk(combined))

    def predict(self, images, state):
        """Returns (progress, success) ∈ [0,1]."""
        return torch.sigmoid(self.forward(images, state))


def train(args):
    rank, local_rank, world_size = _setup_distributed()
    distributed = world_size > 1
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main:
        logging.info(f"Using device: {device}  (world_size={world_size})")

    if args.val_ratio > 0:
        train_files, val_files = _split_by_rollout(
            args.data_dir, args.val_ratio, args.val_split_seed
        )
        train_ds = AgilexDualLabelDataset(args.data_dir, use_clip=True, files=train_files)
        val_ds = AgilexDualLabelDataset(args.data_dir, use_clip=True, files=val_files)
    else:
        train_ds = AgilexDualLabelDataset(args.data_dir, use_clip=True)
        val_ds = None

    state_dim = train_ds.state_dim
    model = DINOv2TemporalSwitchHead(
        dinov2_model=args.dinov2_model,
        state_dim=state_dim,
        num_cameras=3,
        max_clip_frames=args.max_clip_frames,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
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
        logging.info(
            f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

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
            config=vars(args) | {"world_size": world_size, "mode": "dual_head_v2"},
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
            label = label.to(device)  # (B, 2) = [progress, success]

            logits = model(images, state)
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
        avg_vl = vp = vs = mae_p = mae_s = vn = 0.0
        if val_loader is not None:
            model.eval()
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
                    vl_step = args.progress_weight * lp + args.success_weight * ls
                    avg_vl += vl_step.item()
                    pred = torch.sigmoid(logits)
                    mae_p += (pred[:, 0] - label[:, 0]).abs().mean().item()
                    mae_s += (pred[:, 1] - label[:, 1]).abs().mean().item()
                    vn += 1
            vn = max(vn, 1)
            avg_vl /= vn
            val_msg = (
                f"loss={avg_vl:.4f} "
                f"p_bce={vp/vn:.4f} s_bce={vs/vn:.4f} "
                f"p_mae={mae_p/vn:.4f} s_mae={mae_s/vn:.4f}"
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
                    "val/progress_bce": vp / vn,
                    "val/success_bce": vs / vn,
                    "val/progress_mae": mae_p / vn,
                    "val/success_mae": mae_s / vn,
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
    parser = argparse.ArgumentParser(
        description="Temporal-attention switch head v2 (soft-label progress + success)"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Checkpoint dir. If unset, defaults to checkpoints/switch_head_v2/<run_tag>, "
                             "where run_tag is --wandb_run_name or the basename of --data_dir.")
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--max_clip_frames", type=int, default=20)
    parser.add_argument("--attn_dim", type=int, default=384)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--progress_weight", type=float, default=1.0)
    parser.add_argument("--success_weight", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        run_tag = args.wandb_run_name or pathlib.Path(args.data_dir).name
        args.output_dir = str(pathlib.Path("checkpoints/switch_head_v2") / run_tag)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(args)


if __name__ == "__main__":
    main()
