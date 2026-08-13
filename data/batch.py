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
