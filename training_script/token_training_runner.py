import importlib
import os
import time

import torch
import wandb

from data import get_train_data_loader, get_val_data_loader
from data.batch import concatenate_camera_batches, make_camera_batch
from data.step_tracker import StepTracker
from training_script.training_utils import (
    auto_resume_job,
    configure_3d_token_training_stage,
    create_lr_scheduler,
    create_optimizer,
    find_checkpoints,
)
from training_script.token_validation import run_validation
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


def _loss_is_finite(loss):
    return loss is not None and bool(torch.isfinite(loss.detach().float()).all())


def _report_nonfinite_skip(
    kind,
    batch,
    update_step,
    skip_count,
    extra="",
):
    scene_name = batch.get("scene_name", "unknown")
    if isinstance(scene_name, (list, tuple)):
        scene_name = scene_name[0]
    suffix = f"; {extra}" if extra else ""
    print(
        f"Skipped non-finite {kind} at update_step={update_step}; "
        f"scene={scene_name}; total_skips={skip_count}{suffix}"
    )


def _param_grad_group(name):
    name = name.removeprefix("module.")
    if name == "initializer.query_bank":
        return "query_bank"
    if name.startswith("initializer.stack.layers.0."):
        return "init_cross"
    if name.startswith("initializer.stack.layers.1."):
        return "init_slot"
    if name.startswith("gaussian_head."):
        return "gaussian_head"
    if name.startswith("evidence_adapter."):
        return "adapter"
    if name.startswith("backbone."):
        return "backbone"
    return "other"


def _finite_abs_max(tensor, exp=False):
    value = tensor.detach().float()
    if exp:
        value = value.exp()
    finite = value[torch.isfinite(value)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.abs().amax().cpu())


def _geometry_health(output, camera_data, near_plane):
    xyz = output.gaussians.xyz.detach().float()
    xyz_finite = torch.isfinite(xyz)
    metrics = {
        "health/xyz_abs_max": _finite_abs_max(xyz),
        "health/xyz_finite_ratio": float(xyz_finite.float().mean().cpu()),
    }
    try:
        w2c = torch.linalg.inv(camera_data["c2w"].detach().float())
        camera_xyz = torch.einsum(
            "bvij,bnj->bvni", w2c[..., :3, :3], xyz
        ) + w2c[..., :3, 3].unsqueeze(-2)
        camera_z = camera_xyz[..., 2]
        finite = torch.isfinite(camera_z)
        finite_z = camera_z[finite]
        metrics["health/camera_z_finite_ratio"] = float(
            finite.float().mean().cpu()
        )
        metrics["health/camera_z_abs_min"] = (
            float(finite_z.abs().amin().cpu())
            if finite_z.numel() > 0
            else float("nan")
        )
        positive_z = finite_z[finite_z > 0]
        metrics["health/camera_z_positive_min"] = (
            float(positive_z.amin().cpu())
            if positive_z.numel() > 0
            else float("nan")
        )
        metrics["health/behind_camera_ratio"] = (
            float((finite_z <= 0).float().mean().cpu())
            if finite_z.numel() > 0
            else float("nan")
        )
        metrics["health/near_plane_violation_ratio"] = (
            float(
                ((finite_z > 0) & (finite_z < near_plane))
                .float()
                .mean()
                .cpu()
            )
            if finite_z.numel() > 0
            else float("nan")
        )
        metrics["health/renderer_invalid_ratio"] = (
            float((finite_z < near_plane).float().mean().cpu())
            if finite_z.numel() > 0
            else float("nan")
        )
    except RuntimeError:
        metrics.update(
            {
                "health/camera_z_finite_ratio": 0.0,
                "health/camera_z_abs_min": float("nan"),
                "health/camera_z_positive_min": float("nan"),
                "health/behind_camera_ratio": float("nan"),
                "health/near_plane_violation_ratio": float("nan"),
                "health/renderer_invalid_ratio": float("nan"),
            }
        )
    return metrics


def _xyz_grad_metrics(xyz, grad_scale=1.0):
    grad = xyz.grad
    if grad is None:
        return {
            "grad/xyz_norm": float("nan"),
            "grad/xyz_abs_max": float("nan"),
        }
    grad = grad.detach().float() / float(grad_scale)
    return {
        "grad/xyz_norm": float(torch.linalg.vector_norm(grad).cpu()),
        "grad/xyz_abs_max": _finite_abs_max(grad),
    }


def _module_grad_metrics(model):
    grouped_norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _param_grad_group(name)
        norm = torch.linalg.vector_norm(parameter.grad.detach().float())
        grouped_norms.setdefault(group, []).append(norm)
    metrics = {}
    for group, norms in grouped_norms.items():
        metrics[f"grad/{group}"] = float(
            torch.linalg.vector_norm(torch.stack(norms)).cpu()
        )
    return metrics


def _head_parameter_metrics(model):
    head = model.gaussian_head
    return {
        "health/head_proj_weight_norm": float(
            torch.linalg.vector_norm(head.proj.weight.detach().float()).cpu()
        ),
        "health/head_proj_weight_abs_max": _finite_abs_max(head.proj.weight),
        "health/head_proj_bias_abs_max": _finite_abs_max(head.proj.bias),
        "health/head_norm_weight_abs_max": _finite_abs_max(head.norm.weight),
        "health/head_norm_bias_abs_max": _finite_abs_max(head.norm.bias),
    }


def _optimizer_lr_metrics(optimizer):
    return {
        f"train/learning_rate_{group['group_name']}": group["lr"]
        for group in optimizer.param_groups
    }


def _optimizer_group_lr(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("group_name") == group_name:
            return group["lr"]
    raise KeyError(f"Optimizer group '{group_name}' was not found")


def _nonfinite_grad_debug(model, output, update_step, grad_norm):
    groups = {
        "query_bank": False,
        "init_cross": False,
        "init_slot": False,
        "gaussian_head": False,
        "adapter": False,
        "backbone": False,
        "other": False,
    }
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or torch.isfinite(parameter.grad).all():
            continue
        group = _param_grad_group(name)
        groups[group] = True
        if len(names) < 8:
            names.append(name.removeprefix("module."))
    gaussians = output.gaussians
    fired = [group for group, hit in groups.items() if hit]
    return {
        "groups": fired,
        "group_flags": groups,
        "names": names,
        "grad_norm": (
            float(grad_norm)
            if torch.isfinite(grad_norm)
            else "nonfinite"
        ),
        "xyz_abs_max": _finite_abs_max(gaussians.xyz),
        "scale_max": _finite_abs_max(gaussians.scale, exp=True),
        "opacity_max": _finite_abs_max(gaussians.opacity),
        "update_step": int(update_step),
    }


def _wandb_metrics(
    output,
    update_step,
    learning_rate,
    grad_norm,
    elapsed,
    nonfinite_loss_skips,
    nonfinite_gradient_skips,
    optimizer,
    health_metrics,
    gradient_metrics,
):
    metrics = {
        "loss/total": output.loss_metrics.loss.detach().float().item(),
        "info/global_step": update_step,
        "train/learning_rate": learning_rate,
        "train/gradient_norm": float(grad_norm),
        "train/iteration_time": elapsed,
        "train/skipped_nonfinite_loss": nonfinite_loss_skips,
        "train/skipped_nonfinite_grad": nonfinite_gradient_skips,
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
    metrics.update(_optimizer_lr_metrics(optimizer))
    metrics.update(health_metrics)
    metrics.update(gradient_metrics)
    return metrics


def _wandb_visualizations(config, output, input_data, target_data, update_step):
    visualization_config = config.get("visualization", {})
    target_prediction = output.render[:, : target_data["image"].shape[1]]
    rendering_view = make_rendering_view(
        input_data,
        target_data,
        target_prediction,
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

    step_tracker = StepTracker()
    data_loader = get_train_data_loader(
        config,
        num_workers=config.training.num_workers,
        shuffle=True,
        drop_last=True,
        pin_mem=True,
        step_tracker=step_tracker,
    )
    validation_config = config.get("validation", {})
    validation_enabled = validation_config.get("enabled", False)
    val_data_loader = (
        get_val_data_loader(config, step_tracker=step_tracker, pin_mem=True)
        if validation_enabled
        else None
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
        backbone_lr_multiplier=config.training.get(
            "backbone_lr_multiplier", 0.01
        ),
        query_lr_multiplier=config.training.get("query_lr_multiplier", 0.01),
        gaussian_head_lr_multiplier=config.training.get(
            "gaussian_head_lr_multiplier", 1.0
        ),
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
    step_tracker.set_step(update_step)

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
    accumulation_step = 0
    nonfinite_loss_skips = 0
    nonfinite_gradient_skips = 0
    near_plane = float(config.model.gaussians.near_plane)

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
            supervision_data = (
                concatenate_camera_batches(target_data, input_data)
                if config.training.get("context_view_loss", False)
                else target_data
            )
            with torch.autocast(
                device_type=device.type,
                enabled=config.training.use_amp and device.type == "cuda",
                dtype=amp_dtype,
            ):
                output = model(
                    input_data,
                    supervision_data,
                    global_step=update_step,
                )
                loss = output.loss_metrics.loss / grad_accumulation

            if not _loss_is_finite(output.loss_metrics.loss):
                nonfinite_loss_skips += 1
                if wandb_enabled:
                    wandb.log(
                        {
                            "health/nonfinite_loss_event": 1,
                            "train/skipped_nonfinite_loss": nonfinite_loss_skips,
                            **_geometry_health(
                                output,
                                supervision_data,
                                near_plane,
                            ),
                        },
                        step=update_step,
                    )
                _report_nonfinite_skip(
                    "loss",
                    batch,
                    update_step,
                    nonfinite_loss_skips,
                )
                continue

            if output.gaussians.xyz.requires_grad:
                output.gaussians.xyz.retain_grad()

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
            accumulation_step += 1
            should_update = accumulation_step == grad_accumulation
            if not should_update:
                continue

            scaler.unscale_(optimizer)
            xyz_grad_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
            should_log_update = (
                wandb_enabled
                and (update_step + 1) % config.training.wandb_log_every == 0
            )
            gradient_metrics = (
                {
                    **_xyz_grad_metrics(
                        output.gaussians.xyz,
                        grad_scale=xyz_grad_scale,
                    ),
                    **_module_grad_metrics(model),
                }
                if should_log_update
                else {}
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                gradient_clip_norm,
            )
            if not bool(torch.isfinite(grad_norm)):
                nonfinite_gradient_skips += 1
                debug_payload = _nonfinite_grad_debug(
                    model, output, update_step, grad_norm
                )
                if wandb_enabled:
                    wandb.log(
                        {
                            "health/nonfinite_grad_event": 1,
                            "train/skipped_nonfinite_grad": (
                                nonfinite_gradient_skips
                            ),
                            **_geometry_health(
                                output,
                                supervision_data,
                                near_plane,
                            ),
                            **_xyz_grad_metrics(
                                output.gaussians.xyz,
                                grad_scale=xyz_grad_scale,
                            ),
                            **{
                                f"health/nonfinite_grad_{group}": float(hit)
                                for group, hit in debug_payload[
                                    "group_flags"
                                ].items()
                            },
                        },
                        step=update_step,
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                accumulation_step = 0
                _report_nonfinite_skip(
                    "gradient",
                    batch,
                    update_step,
                    nonfinite_gradient_skips,
                    extra=(
                        f"groups={','.join(debug_payload['groups']) or 'none'}; "
                        f"names={','.join(debug_payload['names'])}; "
                        f"xyz_abs_max={debug_payload['xyz_abs_max']}; "
                        f"scale_max={debug_payload['scale_max']}"
                    ),
                )
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulation_step = 0
            scheduler.step()
            update_step += 1
            step_tracker.set_step(update_step)
            elapsed = time.time() - started

            if wandb_enabled and update_step % config.training.wandb_log_every == 0:
                health_metrics = {
                    **_geometry_health(output, supervision_data, near_plane),
                    **_head_parameter_metrics(model),
                }
                wandb.log(
                    _wandb_metrics(
                        output,
                        update_step,
                        _optimizer_group_lr(optimizer, "decoder"),
                        grad_norm,
                        elapsed,
                        nonfinite_loss_skips,
                        nonfinite_gradient_skips,
                        optimizer,
                        health_metrics,
                        gradient_metrics,
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

            if (
                validation_enabled
                and update_step % validation_config.every == 0
            ):
                run_validation(
                    model,
                    next(iter(val_data_loader)),
                    config,
                    device,
                    amp_dtype,
                    update_step,
                    wandb_enabled,
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
