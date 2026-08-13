import importlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data.batch import make_camera_batch
from evaluation_script.metrics import compute_psnr, export_results, summarize_evaluation
from utils.runtime import init_config


def save_context_images(input_data, output_dir, uid):
    scene_name = input_data["scene_name"][0]
    input_dir = Path(output_dir) / f"{uid:06d}_{scene_name}" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for view_index, image in enumerate(input_data["image"][0]):
        save_image(image, input_dir / f"{view_index}.png")


def main():
    config = init_config()
    os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32

    dataset_module, dataset_name = config.inference.dataset_name.rsplit(".", 1)
    dataset_class = importlib.import_module(dataset_module).__dict__[dataset_name]
    dataset = dataset_class(config)
    num_workers = config.inference.num_workers
    loader_kwargs = {
        "batch_size": config.inference.batch_size_per_gpu,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.inference.get("prefetch_factor", 2)
    data_loader = DataLoader(dataset, **loader_kwargs)

    model_module, model_name = config.model.class_name.rsplit(".", 1)
    model_class = importlib.import_module(model_module).__dict__[model_name]
    model = model_class(config).to(device)
    load_result = model.load_ckpt(config.inference.ckpt_path)
    if load_result is None:
        raise RuntimeError(f"Failed to load checkpoint: {config.inference.ckpt_path}")
    if model.group_size != config.data.num_context_views:
        raise ValueError(
            f"Direct MVP inference requires group_size == num_context_views, got "
            f"{model.group_size} and {config.data.num_context_views}"
        )

    amp_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }[config.inference.amp_dtype]
    output_dir = config.inference.out_dir
    max_batches = config.inference.max_batches
    compute_metrics = config.inference.get("compute_metrics", True)
    print(f"checkpoint: {config.inference.ckpt_path}")
    print(f"device: {device}")
    print(f"context views: {config.data.num_context_views}")
    print(f"output: {output_dir}")

    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=config.inference.use_amp and device.type == "cuda",
        dtype=amp_dtype,
    ):
        for batch_index, raw_batch in enumerate(data_loader):
            if batch_index >= max_batches:
                break
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            input_data = make_camera_batch(batch, "input")
            target_data = make_camera_batch(batch, "target")
            result = model(input_data, target_data)
            export_results(
                result,
                output_dir,
                compute_metrics=compute_metrics,
                uid=batch_index + 1,
            )
            save_context_images(input_data, output_dir, batch_index + 1)
            batch_size, num_targets = result.render.shape[:2]
            psnr = compute_psnr(
                target_data["image"].reshape(
                    batch_size * num_targets,
                    *target_data["image"].shape[2:],
                ),
                result.render.reshape(
                    batch_size * num_targets,
                    *result.render.shape[2:],
                ),
            ).mean()
            print(
                f"batch {batch_index + 1}/{max_batches}: "
                f"{input_data['scene_name'][0]} psnr={psnr.item():.4f}"
            )

    if compute_metrics:
        summarize_evaluation(output_dir)


if __name__ == "__main__":
    main()
