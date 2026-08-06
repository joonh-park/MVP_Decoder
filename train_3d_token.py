import importlib
import os
import time

import torch
import wandb

from data import get_train_data_loader
from setup import init_config, init_wandb_and_backup
from training_3d_token_utils import make_camera_batch
from training_utils import auto_resume_job, create_lr_scheduler, create_optimizer


def decoder_state_dict(model):
    """Do not duplicate the immutable MVP checkpoint in every decoder checkpoint."""

    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.")
    }


config = init_config()
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
init_wandb_and_backup(config)

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
resume_path = config.training.get("resume_ckpt", config.training.checkpoint_dir)
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
model.train()
optimizer.zero_grad(set_to_none=True)

while step < config.training.train_steps * grad_accumulation:
    for raw_batch in data_loader:
        if step >= config.training.train_steps * grad_accumulation:
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
            output = model(input_data, target_data)
            loss = output.loss_metrics.loss / grad_accumulation

        scaler.scale(loss).backward()
        step += 1
        should_update = step % grad_accumulation == 0
        if should_update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_step += 1

        if step % config.training.wandb_log_every == 0:
            metrics = {
                f"train/{name}": value.detach().float().item()
                for name, value in output.loss_metrics.items()
            }
            metrics.update(
                {
                    "step": step,
                    "update_step": update_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "iteration_time": time.time() - started,
                    "num_initial_tokens": output.decoder_output.z_initial.shape[1],
                    "num_final_tokens": (
                        output.decoder_output.z_final.shape[1]
                        if output.decoder_output.z_final is not None
                        else output.decoder_output.z_initial.shape[1]
                    ),
                }
            )
            wandb.log(metrics, step=step)

        if step % config.training.print_every == 0:
            print(
                f"step={step} update={update_step} "
                f"loss={output.loss_metrics.loss.item():.6f} "
                f"time={time.time() - started:.2f}s"
            )

        if should_update and update_step % config.training.checkpoint_every == 0:
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                config.training.checkpoint_dir,
                f"ckpt_{step:016}.pt",
            )
            torch.save(
                {
                    "model": decoder_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "fwdbwd_pass_step": step,
                    "param_update_step": update_step,
                    "backbone_checkpoint": config.model.backbone.checkpoint_path,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")
