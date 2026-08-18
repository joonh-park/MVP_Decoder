from io import BytesIO
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms.functional import pil_to_tensor

from data.view_sampler import BoundedViewSampler


class RE10KDataset(IterableDataset):
    """C3G-compatible RE10K chunk reader with MVP camera tensors."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root = Path(config.data.root)
        self.stage = config.data.get("stage", "train")
        data_stage = "test" if self.stage == "val" else self.stage
        self.chunk_paths = sorted((self.root / data_stage).glob("*.torch"))
        if not self.chunk_paths:
            raise FileNotFoundError(
                f"No RE10K .torch chunks found in {self.root / data_stage}"
            )
        self.image_size = tuple(config.data.image_size)
        self.original_image_size = tuple(config.data.original_image_size)
        self.num_context_views = config.data.num_context_views
        self.num_target_views = config.data.num_target_views
        self.min_context_gap = config.data.min_context_gap
        self.max_context_gap = config.data.max_context_gap
        self.min_target_distance = config.data.get("min_target_distance", 0)
        self.baseline_min = config.data.get("baseline_min", 1e-3)
        self.baseline_max = config.data.get("baseline_max", 1e10)
        self.max_fov = config.data.get("max_fov", 100.0)
        self.make_baseline_one = config.data.get("make_baseline_one", True)
        self.relative_pose = config.data.get("relative_pose", True)
        self.pose_normalization = config.data.get("pose_normalization", "re10k")
        self.augment = config.data.get("augment", True)
        self.skip_bad_shape = config.data.get("skip_bad_shape", True)
        self.seed = config.data.get("seed", 42)
        self.step_tracker = None
        self.view_sampler = BoundedViewSampler(
            config.data,
            stage=self.stage,
            step_tracker=self.step_tracker,
        )
        if self.num_context_views < 2:
            raise ValueError("RE10K requires at least 2 context views")
        if self.pose_normalization not in {"re10k", "mvp"}:
            raise ValueError(
                "pose_normalization must be 're10k' or 'mvp', got "
                f"{self.pose_normalization!r}"
            )

    @staticmethod
    def _decode_image(encoded):
        image = Image.open(BytesIO(encoded.numpy().tobytes())).convert("RGB")
        return pil_to_tensor(image).float() / 255.0

    @staticmethod
    def _convert_cameras(cameras):
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
        intrinsics[:, 0, 0] = cameras[:, 0]
        intrinsics[:, 1, 1] = cameras[:, 1]
        intrinsics[:, 0, 2] = cameras[:, 2]
        intrinsics[:, 1, 2] = cameras[:, 3]
        w2c = torch.eye(4, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
        w2c[:, :3] = cameras[:, 6:].reshape(-1, 3, 4)
        return torch.linalg.inv(w2c), intrinsics

    def _fov_is_valid(self, intrinsics):
        if not torch.isfinite(intrinsics).all():
            return False
        fx = intrinsics[:, 0, 0].clamp_min(1e-8)
        fy = intrinsics[:, 1, 1].clamp_min(1e-8)
        fov_x = torch.rad2deg(2 * torch.atan(0.5 / fx))
        fov_y = torch.rad2deg(2 * torch.atan(0.5 / fy))
        return bool(torch.maximum(fov_x, fov_y).max() <= self.max_fov)

    def set_step_tracker(self, step_tracker):
        self.step_tracker = step_tracker
        self.view_sampler.step_tracker = step_tracker

    def _sample_indices(self, num_frames, generator):
        return self.view_sampler.sample(num_frames, generator)

    @staticmethod
    def _normalize_poses_for_mvp(c2w, context_indices):
        """Match the camera normalization used by the original MVP dataset."""

        context_c2w = c2w[context_indices]
        position_avg = context_c2w[:, :3, 3].mean(dim=0)
        forward_avg = F.normalize(context_c2w[:, :3, 2].mean(dim=0), dim=0)
        down_avg = context_c2w[:, :3, 1].mean(dim=0)
        down_avg = F.normalize(
            down_avg - down_avg.dot(forward_avg) * forward_avg,
            dim=0,
        )
        right_avg = torch.cross(down_avg, forward_avg, dim=0)
        average_pose = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
        average_pose[:3] = torch.stack(
            (right_avg, down_avg, forward_avg, position_avg),
            dim=1,
        )
        normalized = torch.linalg.inv(average_pose) @ c2w
        scene_extent = normalized[context_indices, :3, 3].abs().max().clamp_min(1e-8)
        normalized[:, :3, 3] /= scene_extent
        return normalized

    def _resize_and_crop(self, images, intrinsics):
        output_h, output_w = self.image_size
        input_h, input_w = images.shape[-2:]
        scale = max(output_h / input_h, output_w / input_w)
        scaled_h = round(input_h * scale)
        scaled_w = round(input_w * scale)
        images = F.interpolate(
            images,
            size=(scaled_h, scaled_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        top = (scaled_h - output_h) // 2
        left = (scaled_w - output_w) // 2
        images = images[..., top : top + output_h, left : left + output_w]

        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] *= scaled_w / output_w
        intrinsics[:, 1, 1] *= scaled_h / output_h
        return images, intrinsics

    def _make_example(self, example, generator):
        scene_name = example.get("key", "unknown")
        cameras = example.get("cameras")
        if (
            not isinstance(cameras, torch.Tensor)
            or cameras.ndim != 2
            or cameras.shape[-1] != 18
        ):
            return None
        try:
            c2w, intrinsics = self._convert_cameras(cameras.float())
        except RuntimeError:
            return None
        if not self._fov_is_valid(intrinsics):
            return None
        sampled = self._sample_indices(c2w.shape[0], generator)
        if sampled is None:
            return None
        context_indices, target_indices = sampled
        all_indices = torch.cat((context_indices, target_indices))
        try:
            images = torch.stack(
                [self._decode_image(example["images"][int(index)]) for index in all_indices]
            )
        except (IndexError, OSError):
            return None
        if self.skip_bad_shape and images.shape[-2:] != self.original_image_size:
            return None

        context_c2w = c2w[context_indices]
        baseline = (context_c2w[0, :3, 3] - context_c2w[-1, :3, 3]).norm()
        if (
            not torch.isfinite(baseline)
            or not self.baseline_min <= baseline <= self.baseline_max
        ):
            return None
        try:
            if self.pose_normalization == "mvp":
                c2w = self._normalize_poses_for_mvp(c2w, context_indices)
            else:
                if self.make_baseline_one:
                    c2w = c2w.clone()
                    c2w[:, :3, 3] /= baseline
                if self.relative_pose:
                    c2w = torch.linalg.inv(c2w[context_indices[0]]) @ c2w
        except RuntimeError:
            return None

        selected_intrinsics = intrinsics[all_indices]
        images, selected_intrinsics = self._resize_and_crop(
            images, selected_intrinsics
        )
        output_h, output_w = self.image_size
        fxfycxcy = torch.stack(
            (
                selected_intrinsics[:, 0, 0] * output_w,
                selected_intrinsics[:, 1, 1] * output_h,
                selected_intrinsics[:, 0, 2] * output_w,
                selected_intrinsics[:, 1, 2] * output_h,
            ),
            dim=-1,
        )

        selected_c2w = c2w[all_indices]
        if (
            self.stage == "train"
            and self.augment
            and bool(torch.rand((), generator=generator) >= 0.5)
        ):
            images = images.flip(-1)
            reflection = torch.eye(4)
            reflection[0, 0] = -1
            selected_c2w = reflection @ selected_c2w @ reflection

        if not (
            torch.isfinite(images).all()
            and torch.isfinite(fxfycxcy).all()
            and torch.isfinite(selected_c2w).all()
            and bool((fxfycxcy[:, :2] > 0).all())
        ):
            return None

        context_count = self.num_context_views
        return {
            "input_image": images[:context_count],
            "input_fxfycxcy": fxfycxcy[:context_count],
            "input_c2w": selected_c2w[:context_count],
            "input_indices": context_indices[:, None],
            "target_image": images[context_count:],
            "target_fxfycxcy": fxfycxcy[context_count:],
            "target_c2w": selected_c2w[context_count:],
            "target_indices": target_indices[:, None],
            "scene_name": scene_name,
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        generator = torch.Generator().manual_seed(self.seed + shard_id)

        chunks = self.chunk_paths[shard_id::num_shards]
        if not chunks:
            return
        while True:
            chunk_order = torch.randperm(len(chunks), generator=generator).tolist()
            for chunk_index in chunk_order:
                chunk = torch.load(
                    chunks[chunk_index], map_location="cpu", weights_only=True
                )
                example_order = torch.randperm(
                    len(chunk), generator=generator
                ).tolist()
                for example_index in example_order:
                    result = self._make_example(chunk[example_index], generator)
                    if result is not None:
                        yield result
