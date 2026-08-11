import argparse
from pathlib import Path

import torch

from model.backbones.mvp.backbone import MVPBackbone
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a forward-only smoke test of the MVP DPT backbone."
    )
    parser.add_argument("--config", default="configs/mvp/inference.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=960)
    return parser.parse_args()


def make_smoke_input(batch_size, num_views, height, width, device):
    images = torch.rand(batch_size, num_views, 3, height, width, device=device)

    intrinsics = torch.empty(batch_size, num_views, 4, device=device)
    focal = float(max(height, width))
    intrinsics[..., 0] = focal
    intrinsics[..., 1] = focal
    intrinsics[..., 2] = width / 2.0
    intrinsics[..., 3] = height / 2.0

    c2w = torch.eye(4, device=device).view(1, 1, 4, 4)
    c2w = c2w.repeat(batch_size, num_views, 1, 1)
    return images, intrinsics, c2w


def main():
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config.inference.ckpt_path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"MVP checkpoint does not exist: {checkpoint_path}. "
            "Pass --checkpoint or set inference.ckpt_path."
        )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    group_size = config.model.group_size
    if args.num_views % group_size != 0:
        raise ValueError(
            f"num_views must be divisible by group_size={group_size}, "
            f"got {args.num_views}"
        )
    merge_factor = config.model.patch_size * 4
    if args.height % merge_factor or args.width % merge_factor:
        raise ValueError(
            f"height and width must be divisible by {merge_factor}, "
            f"got {args.height}x{args.width}"
        )

    backbone = MVPBackbone(config, freeze=True)
    load_result = backbone.load_checkpoint(str(checkpoint_path))
    backbone = backbone.to(device).eval()
    images, intrinsics, c2w = make_smoke_input(
        args.batch_size,
        args.num_views,
        args.height,
        args.width,
        device,
    )

    amp_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=amp_enabled,
        dtype=torch.bfloat16 if amp_enabled else torch.float32,
    ):
        evidence = backbone(images, intrinsics, c2w)

    print(f"checkpoint: {checkpoint_path}")
    print(f"load_result: {load_result}")
    print(f"device: {device}")
    print(f"feature: {tuple(evidence.feature.shape)} {evidence.feature.dtype}")
    print(f"center_ray: {tuple(evidence.center_ray.shape)} {evidence.center_ray.dtype}")
    print(f"grid_size: {evidence.grid_size}")
    print(f"feature_finite: {torch.isfinite(evidence.feature).all().item()}")
    print(f"center_ray_finite: {torch.isfinite(evidence.center_ray).all().item()}")


if __name__ == "__main__":
    main()
