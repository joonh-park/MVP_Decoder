import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def configure_3d_token_training_stage(model, stage):
    """Select exactly which 3D-token modules are optimized in each stage."""

    valid_stages = {"init", "full"}
    if stage not in valid_stages:
        raise ValueError(
            f"Unknown 3D-token training stage '{stage}'; "
            f"expected one of {sorted(valid_stages)}"
        )

    if stage == "init":
        if model.split_enabled or model.refinement_enabled:
            raise ValueError(
                "Init training requires split.enabled=false and refinement.layers=[]"
            )
        trainable_prefixes = (
            "evidence_adapter.",
            "initializer.",
            "gaussian_head.",
        )
        train_backbone = not getattr(model.backbone, "freeze", True)
        for name, parameter in model.named_parameters():
            trainable = name.startswith(trainable_prefixes) or (
                train_backbone and name.startswith("backbone.")
            )
            parameter.requires_grad_(trainable)
    else:
        if not (model.split_enabled or model.refinement_enabled):
            raise ValueError(
                "Full training requires split or refinement to be enabled"
            )
        train_backbone = not getattr(model.backbone, "freeze", True)
        for name, parameter in model.named_parameters():
            trainable = not name.startswith("loss_computer.") and (
                train_backbone or not name.startswith("backbone.")
            )
            parameter.requires_grad_(trainable)

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise ValueError(f"Training stage '{stage}' selected no trainable parameters")
    return trainable_names


def create_optimizer(
    model,
    weight_decay,
    learning_rate,
    betas,
    backbone_lr_multiplier=1.0,
    query_lr_multiplier=1.0,
    gaussian_head_lr_multiplier=1.0,
):
    if backbone_lr_multiplier <= 0:
        raise ValueError("backbone_lr_multiplier must be positive")
    if query_lr_multiplier <= 0:
        raise ValueError("query_lr_multiplier must be positive")
    if gaussian_head_lr_multiplier <= 0:
        raise ValueError("gaussian_head_lr_multiplier must be positive")
    # start with all of the candidate parameters
    all_param_dict = {name: param for name, param in model.named_parameters()}
    # filter out those that do not require grad
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    grouped_params = {
        (group_name, use_decay): []
        for group_name in ("decoder", "gaussian_head", "query_bank", "backbone")
        for use_decay in (True, False)
    }
    for name, param in optimized_param_dict.items():
        normalized_name = name.removeprefix("module.")
        if normalized_name.startswith("backbone."):
            group_name = "backbone"
        elif normalized_name.startswith("gaussian_head."):
            group_name = "gaussian_head"
        elif normalized_name == "initializer.query_bank":
            group_name = "query_bank"
        else:
            group_name = "decoder"
        use_decay = not (
            param.dim() == 1 or getattr(param, '_no_weight_decay', False)
        )
        grouped_params[(group_name, use_decay)].append(param)

    optim_groups = []
    lr_multipliers = {
        "decoder": 1.0,
        "gaussian_head": gaussian_head_lr_multiplier,
        "query_bank": query_lr_multiplier,
        "backbone": backbone_lr_multiplier,
    }
    for (group_name, use_decay), params in grouped_params.items():
        if not params:
            continue
        group_lr = learning_rate * lr_multipliers[group_name]
        optim_groups.append(
            {
                'params': params,
                'weight_decay': weight_decay if use_decay else 0.0,
                'lr': group_lr,
                'group_name': group_name,
            }
        )
    # use fused AdamW optimizer by default. 
    # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,fused=True)
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    
    # Print Model Information
    if dist.is_initialized():
        if dist.get_rank() == 0:    
            def get_module_name(name):
                parts = name.split('.')
                if len(parts) > 2 and parts[0] == 'module':
                    return parts[1] + '.' + parts[2]
                return parts[0]  # Fallback to first part if no 'module.' prefix
            print(
                f'Optimizer: AdamW, learning rate: {learning_rate}, '
                f'backbone multiplier: {backbone_lr_multiplier}, '
                f'query multiplier: {query_lr_multiplier}, '
                f'gaussian head multiplier: {gaussian_head_lr_multiplier}, '
                f'weight decay: {weight_decay}, betas: {betas}'
            )
            # Number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in optimized_param_dict.values())
            optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
            frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))
            
            print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')        
            print(f'Optimized parameters: {optim_module_names}')
            print(f'Frozen parameters: {frozen_module_names}')
            
    return optimizer, optimized_param_dict, all_param_dict

def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths



def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights
    if isinstance(model, DDP):
        status = model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        status = model.load_state_dict(checkpoint['model'], strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")

    # resume training state
    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(
                "Failed to load optimizer and lr_scheduler from "
                f"{ckpt_path}; model weights remain loaded, but optimizer, "
                "scheduler, training steps, warmup, and LPF schedule restart "
                "from step 0"
            )
    
    return optimizer, lr_scheduler, forward_pass_step, param_update_step
