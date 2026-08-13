import importlib
import os
import time

import torch
import wandb

from data import get_train_data_loader
from data.batch import make_camera_batch
from training_script.training_utils import (
    auto_resume_job,
    configure_3d_token_training_stage,
    create_lr_scheduler,
    create_optimizer,
    find_checkpoints,
)
from training_script.wandb_visualization import (
    make_rendering_view,
    make_xyz_projection_view,
)
from utils.runtime import init_config, init_wandb_and_backup


def decoder_state_dict(model):
    excluded_prefixes = ["loss_computer."]
    if getattr(model.backbone, "freeze", False):
        excluded_prefixes.append("backbone.")
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith(tuple(excluded_prefixes))
    }


def _wandb_metrics(output, update_step, learning_rate, grad_norm, elapsed):
    metrics = {
        "loss/total": output.loss_metrics.loss.detach().float().item(),
        "info/global_step": update_step,
        "train/learning_rate": learning_rate,
        "train/gradient_norm": float(grad_norm),
        "train/iteration_time": elapsed,
        "train/low_pass_filter": output.low_pass_filter,
        "train/num_initial_tokens": output.decoder_output.z_initial.shape[1],
        "train/num_final_tokens": (
            output.decoder_output.z_final.shape[1]
            if output.decoder_output.z_final is not None
            else output.decoder_output.z_initial.shape[1]
        ),
    }
    for name, value in output.loss_metrics.items():
        if name == "loss":
            continue
        namespace = "train" if "psnr" in name else "loss"
        metrics[f"{namespace}/{name}"] = value.detach().float().item()
    return metrics


def _wandb_visualizations(config, output, input_data, target_data, update_step):
    visualization_config = config.get("visualization", {})
    rendering_view = make_rendering_view(
        input_data,
        target_data,
        output.render,
    )
    xyz_view = make_xyz_projection_view(
        output.gaussians,
        sh_degree=config.model.gaussians.sh_degree,
        near_plane=config.model.gaussians.near_plane,
        far_plane=config.model.gaussians.far_plane,
        low_pass_filter=output.low_pass_filter,
        background_color=config.model.gaussians.background_color,
        resolution=visualization_config.get("projection_resolution", 256),
        margin=visualization_config.get("projection_margin", 0.1),
        fov_degrees=visualization_config.get("projection_fov_degrees", 10.0),
    )
    scene_name = target_data.get("scene_name", "")
    if isinstance(scene_name, (list, tuple)):
        scene_name = scene_name[0]
    wandb.log(
        {
            "train/rendering_views": wandb.Image(
                rendering_view,
                caption=(
                    f"{scene_name}; columns: context | target | prediction | error"
                ),
            ),
            "train/xyz_views": wandb.Image(
                xyz_view,
                caption=f"{scene_name}; projections: YZ | ZX | XY",
            ),
        },
        step=update_step,
    )


def run_training(required_stage):
    config = init_config()
    training_stage = config.training.stage
    if training_stage != required_stage:
        raise ValueError(
            f"This entrypoint requires training.stage={required_stage}, "
            f"got {training_stage}"
        )

    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
    wandb_enabled = init_wandb_and_backup(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
    torch.backends.cudnn.allow_tf32 = config.training.use_tf32
    amp_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }[config.training.amp_dtype]

    data_loader = get_train_data_loader(
        config,
        num_workers=config.training.num_workers,
        shuffle=True,
        drop_last=True,
        pin_mem=True,
    )
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    model_class = importlib.import_module(module_name).__dict__[class_name]
    model = model_class(config).to(device)

    resume_path = config.training.get("resume_ckpt", config.training.checkpoint_dir)
    if required_stage == "full" and not find_checkpoints(resume_path):
        initialization_checkpoint = config.training.initialization_checkpoint
        load_status = model.load_ckpt(initialization_checkpoint)
        print(
            f"Loaded initialization checkpoint: {initialization_checkpoint}; "
            f"status: {load_status}"
        )

    trainable_names = configure_3d_token_training_stage(model, training_stage)
    print(
        f"3D-token training stage: {training_stage}; "
        f"trainable tensors: {len(trainable_names)}"
    )
    optimizer, _, _ = create_optimizer(
        model,
        config.training.weight_decay,
        config.training.lr,
        (config.training.beta1, config.training.beta2),
    )
    scheduler = create_lr_scheduler(
        optimizer,
        config.training.train_steps,
        config.training.warmup,
        config.training.get("scheduler_type", "cosine"),
    )
    optimizer, scheduler, step, update_step = auto_resume_job(
        resume_path,
        model,
        optimizer,
        scheduler,
        config.training.get("reset_training_state", False),
    )

    use_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    grad_accumulation = config.training.grad_accum_steps
    gradient_clip_norm = config.training.get("gradient_clip_norm", 0.5)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logged_step_zero_visualization = update_step > 0

    while update_step < config.training.train_steps:
        for raw_batch in data_loader:
            if update_step >= config.training.train_steps:
                break
            started = time.time()
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
                    global_step=update_step,
                )
                loss = output.loss_metrics.loss / grad_accumulation

            if (
                wandb_enabled
                and update_step == 0
                and not logged_step_zero_visualization
            ):
                _wandb_visualizations(
                    config,
                    output,
                    input_data,
                    target_data,
                    update_step=0,
                )
                logged_step_zero_visualization = True

            scaler.scale(loss).backward()
            step += 1
            should_update = step % grad_accumulation == 0
            if not should_update:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                gradient_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_step += 1
            elapsed = time.time() - started

            if wandb_enabled and update_step % config.training.wandb_log_every == 0:
                wandb.log(
                    _wandb_metrics(
                        output,
                        update_step,
                        optimizer.param_groups[0]["lr"],
                        grad_norm,
                        elapsed,
                    ),
                    step=update_step,
                )
            if (
                wandb_enabled
                and update_step % config.training.image_log_every == 0
            ):
                _wandb_visualizations(
                    config,
                    output,
                    input_data,
                    target_data,
                    update_step,
                )

            if update_step % config.training.print_every == 0:
                print(
                    f"step={update_step} loss={output.loss_metrics.loss.item():.6f} "
                    f"psnr={output.loss_metrics.get('final_psnr', output.loss_metrics.get('psnr')).item():.3f} "
                    f"grad_norm={float(grad_norm):.3f} "
                    f"low_pass={output.low_pass_filter:.3f} time={elapsed:.2f}s"
                )

            if update_step % config.training.checkpoint_every == 0:
                os.makedirs(config.training.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    config.training.checkpoint_dir,
                    f"ckpt_{update_step:08}.pt",
                )
                torch.save(
                    {
                        "model": decoder_state_dict(model),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                        "fwdbwd_pass_step": step,
                        "param_update_step": update_step,
                        "training_stage": training_stage,
                        "backbone_checkpoint": model.backbone_checkpoint_path,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint: {checkpoint_path}")

    if wandb_enabled:
        wandb.finish()
