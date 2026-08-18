import torch


def make_camera_batch(batch, prefix):
    camera_batch = {
        "image": batch[f"{prefix}_image"],
        "fxfycxcy": batch[f"{prefix}_fxfycxcy"],
        "c2w": batch[f"{prefix}_c2w"],
        "index": batch[f"{prefix}_indices"],
    }
    if "scene_name" in batch:
        camera_batch["scene_name"] = batch["scene_name"]
    return camera_batch


def concatenate_camera_batches(*camera_batches):
    if not camera_batches:
        raise ValueError("At least one camera batch is required")
    combined = {
        key: torch.cat(
            tuple(camera_batch[key] for camera_batch in camera_batches),
            dim=1,
        )
        for key in ("image", "fxfycxcy", "c2w", "index")
    }
    if "scene_name" in camera_batches[0]:
        combined["scene_name"] = camera_batches[0]["scene_name"]
    return combined
