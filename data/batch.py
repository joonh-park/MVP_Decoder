def make_camera_batch(batch, prefix):
    return {
        "image": batch[f"{prefix}_image"],
        "fxfycxcy": batch[f"{prefix}_fxfycxcy"],
        "c2w": batch[f"{prefix}_c2w"],
        "index": batch[f"{prefix}_indices"],
    }
