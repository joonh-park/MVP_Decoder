import importlib

import torch

from data import get_train_data_loader
from data.batch import make_camera_batch
from evaluation_script.metrics import compute_psnr
from utils.runtime import init_config


config = init_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = get_train_data_loader(
    config,
    num_workers=config.training.num_workers,
    shuffle=False,
    drop_last=False,
    pin_mem=True,
)
module_name, class_name = config.model.class_name.rsplit(".", 1)
model_class = importlib.import_module(module_name).__dict__[class_name]
model = model_class(config).to(device)
model.load_ckpt(config.evaluation.checkpoint_path)
model.eval()

psnr_values = []
with torch.no_grad(), torch.autocast(
    device_type=device.type,
    enabled=config.training.use_amp and device.type == "cuda",
    dtype={"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        config.training.amp_dtype, torch.float32
    ),
):
    for batch_index, raw_batch in enumerate(data_loader):
        if batch_index >= config.evaluation.max_batches:
            break
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        input_data = make_camera_batch(batch, "input")
        target_data = make_camera_batch(batch, "target")
        output = model(input_data, target_data)
        batch_size, views = output.render.shape[:2]
        psnr_values.append(
            compute_psnr(
                target_data["image"].reshape(batch_size * views, *target_data["image"].shape[2:]),
                output.render.reshape(batch_size * views, *output.render.shape[2:]),
            )
        )

print(f"PSNR: {torch.cat(psnr_values).mean().item():.4f}")
