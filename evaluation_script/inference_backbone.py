import argparse
import importlib
from pathlib import Path

import torch

from model.backbones.mvp.backbone import MVPBackbone
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MVP DPT evidence from one dataset scene."
    )
    parser.add_argument("--config", default="configs/mvp/inference.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_dataset_input(config, sample_index, device):
    dataset_path = config.inference.get(
        "dataset_name", "data.mvp_dataset.Dataset"
    )
    module_name, class_name = dataset_path.rsplit(".", 1)
    dataset_class = importlib.import_module(module_name).__dict__[class_name]
    dataset = dataset_class(config)
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"sample_index must be in [0, {len(dataset) - 1}], got {sample_index}"
        )

    sample = dataset[sample_index]
    num_input_views = config.data.num_input_frames
    images = sample["image"][:num_input_views].unsqueeze(0).to(device)
    intrinsics = sample["fxfycxcy"][:num_input_views].unsqueeze(0).to(device)
    c2w = sample["c2w"][:num_input_views].unsqueeze(0).to(device)
    return sample, images, intrinsics, c2w


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
    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32

    sample, images, intrinsics, c2w = load_dataset_input(
        config, args.sample_index, device
    )
    num_views = images.shape[1]
    if num_views % config.model.group_size != 0:
        raise ValueError(
            f"num_input_frames must be divisible by group_size={config.model.group_size}, "
            f"got {num_views}"
        )

    backbone = MVPBackbone(config, freeze=True)
    load_result = backbone.load_checkpoint(str(checkpoint_path))
    backbone = backbone.to(device).eval()

    amp_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }[config.inference.amp_dtype]
    amp_enabled = config.inference.use_amp and device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=amp_enabled,
        dtype=amp_dtype,
    ):
        evidence = backbone(images, intrinsics, c2w)

    print(f"checkpoint: {checkpoint_path}")
    print(f"load_result: {load_result}")
    print(f"device: {device}")
    print(f"scene: {sample['scene_name']}")
    print(f"input_image: {tuple(images.shape)} {images.dtype}")
    print(f"feature: {tuple(evidence.feature.shape)} {evidence.feature.dtype}")
    print(f"center_ray: {tuple(evidence.center_ray.shape)} {evidence.center_ray.dtype}")
    print(f"grid_size: {evidence.grid_size}")
    print(f"feature_finite: {torch.isfinite(evidence.feature).all().item()}")
    print(f"center_ray_finite: {torch.isfinite(evidence.center_ray).all().item()}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scene_name": sample["scene_name"],
                "feature": evidence.feature.cpu(),
                "center_ray": evidence.center_ray.cpu(),
                "grid_size": evidence.grid_size,
                "input_intrinsics": evidence.input_intrinsics.cpu(),
                "input_c2w": evidence.input_c2w.cpu(),
            },
            output_path,
        )
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
