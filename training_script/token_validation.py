import torch
import wandb

from data.batch import make_camera_batch
from training_script.wandb_visualization import make_rendering_view


@torch.no_grad()
def run_validation(
    model,
    raw_batch,
    config,
    device,
    amp_dtype,
    global_step,
    wandb_enabled,
):
    from evaluation_script.metrics import compute_lpips, compute_psnr, compute_ssim

    was_training = model.training
    model.eval()
    try:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        input_data = make_camera_batch(batch, "input")
        target_data = make_camera_batch(batch, "target")
        with torch.autocast(
            device_type=device.type,
            enabled=config.training.use_amp and device.type == "cuda",
            dtype=amp_dtype,
        ):
            output = model(
                input_data,
                target_data,
                global_step=global_step,
            )

        batch_size, views = output.render.shape[:2]
        target = target_data["image"].reshape(
            batch_size * views,
            *target_data["image"].shape[2:],
        ).float()
        prediction = output.render.reshape(
            batch_size * views,
            *output.render.shape[2:],
        ).float()
        metrics = {
            "val/psnr": compute_psnr(target, prediction).mean().item(),
            "val/lpips": compute_lpips(target, prediction).mean().item(),
            "val/ssim": compute_ssim(target, prediction).mean().item(),
        }
        scene_name = target_data.get("scene_name", "")
        if isinstance(scene_name, (list, tuple)):
            scene_name = scene_name[0]
        context_indices = input_data["index"][0].flatten().tolist()

        if wandb_enabled:
            metrics["val/rendering_views"] = wandb.Image(
                make_rendering_view(input_data, target_data, output.render),
                caption=(
                    f"{scene_name}; columns: context | target | prediction | error"
                ),
            )
            wandb.log(metrics, step=global_step)

        print(
            f"validation step={global_step} "
            f"scene={scene_name} context={context_indices} "
            f"psnr={metrics['val/psnr']:.3f} "
            f"lpips={metrics['val/lpips']:.4f} "
            f"ssim={metrics['val/ssim']:.4f}"
        )
        return metrics
    finally:
        model.train(was_training)
