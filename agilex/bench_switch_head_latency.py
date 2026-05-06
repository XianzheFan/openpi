"""Benchmark the DINOv2 switch-head inference latency (single or dual head).

Mirrors the call site in agilex_infer_dinov2_value_switch / dual_threshold:
  head.predict(top_clip, right_clip, left_clip, state)
where each clip is a list of `clip_len` (H, W, 3) uint8 frames.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class StandaloneSwitchHead:
    """Wraps either DINOv2SwitchHead (single) or DINOv2DualHead (progress+success)."""

    def __init__(self, checkpoint_path, dinov2_model="dinov2_vitb14",
                 hidden_dim=256, state_dim=14, num_cameras=3, device=None,
                 dual=False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dual = dual
        if dual:
            from train_switch_head_dual import DINOv2DualHead
            self.model = DINOv2DualHead(
                dinov2_model=dinov2_model, hidden_dim=hidden_dim,
                state_dim=state_dim, num_cameras=num_cameras, freeze_backbone=True,
            )
        else:
            from train_switch_head_robometer import DINOv2SwitchHead
            self.model = DINOv2SwitchHead(
                dinov2_model=dinov2_model, hidden_dim=hidden_dim,
                state_dim=state_dim, num_cameras=num_cameras, freeze_backbone=True,
            )
        sd = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    def _frame_to_tensor(self, frame_hwc):
        return torch.from_numpy(np.asarray(frame_hwc)).permute(2, 0, 1).float() / 255.0

    @torch.no_grad()
    def predict(self, top, right, left, state_np):
        images = []
        for cam in (top, right, left):
            if isinstance(cam, list):
                t = torch.stack([self._frame_to_tensor(f) for f in cam]).unsqueeze(0)
            else:
                t = self._frame_to_tensor(cam).unsqueeze(0)
            images.append(t.to(self.device))
        state_t = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0).to(self.device)
        if self.dual:
            return self.model.predict(images, state_t)
        return self.model.predict_switch_prob(images, state_t)


def run(ckpt: str, clip_len: int, h: int, w: int, warmup: int, iters: int,
        use_clip: bool, dinov2_model: str, dual: bool):
    head = StandaloneSwitchHead(
        checkpoint_path=ckpt,
        dinov2_model=dinov2_model,
        hidden_dim=256,
        state_dim=14,
        num_cameras=3,
        dual=dual,
    )

    rng = np.random.default_rng(0)
    def _frame():
        return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    if use_clip:
        top = [_frame() for _ in range(clip_len)]
        right = [_frame() for _ in range(clip_len)]
        left = [_frame() for _ in range(clip_len)]
    else:
        top, right, left = _frame(), _frame(), _frame()
    state = rng.standard_normal(14).astype(np.float32)

    for _ in range(warmup):
        head.predict(top, right, left, state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        head.predict(top, right, left, state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.asarray(times_ms)
    mode = f"clip_len={clip_len}" if use_clip else "single-frame"
    head_kind = "dual" if dual else "single"
    print(f"\n=== Switch head latency ({head_kind}, {mode}, {dinov2_model}, {h}x{w}) ===")
    print(f"  iters       : {iters} (warmup {warmup})")
    print(f"  mean        : {arr.mean():.2f} ms")
    print(f"  median      : {np.median(arr):.2f} ms")
    print(f"  p90 / p99   : {np.percentile(arr, 90):.2f} / {np.percentile(arr, 99):.2f} ms")
    print(f"  min / max   : {arr.min():.2f} / {arr.max():.2f} ms")
    print(f"  device      : {head.device}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--switch_head_ckpt", type=str,
                   default="agilex/checkpoints/switch_head_robometer/best_model.pt")
    p.add_argument("--dual", action="store_true",
                   help="Benchmark DINOv2DualHead (progress+success) instead of single-head.")
    p.add_argument("--dinov2_model", type=str, default="dinov2_vitb14")
    p.add_argument("--clip_len", type=int, default=20)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--also_single_frame", action="store_true",
                   help="Also benchmark the no-clip fallback path.")
    args = p.parse_args()

    run(args.switch_head_ckpt, args.clip_len, args.height, args.width,
        args.warmup, args.iters, use_clip=True, dinov2_model=args.dinov2_model,
        dual=args.dual)

    if args.also_single_frame:
        run(args.switch_head_ckpt, args.clip_len, args.height, args.width,
            args.warmup, args.iters, use_clip=False, dinov2_model=args.dinov2_model,
            dual=args.dual)


if __name__ == "__main__":
    main()
