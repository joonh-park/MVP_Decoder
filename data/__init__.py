import importlib
import random
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from data.validation_wrapper import ValidationWrapper


def _seed_worker(_worker_id):
    worker_seed = int(torch.utils.data.get_worker_info().seed) % (2**32 - 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _single_batch_size(value, stage):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{stage} iterable datasets require one batch size")
        return value[0]
    return value


class DynamicBatchDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        """
        Handle batch of indices from DynamicBatchedMultiFeatureRandomSampler.

        Args:
            batch_indices: List of tuples like [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]

        Returns:
            List of samples from the underlying dataset
        """
        if isinstance(batch_indices, (list, tuple)) and len(batch_indices) > 0:
            # If it's a batch (list of tuples), process each item
            if isinstance(batch_indices[0], (list, tuple)):
                return [self.dataset[idx] for idx in batch_indices]
            else:
                # Single tuple, call dataset directly
                return self.dataset[batch_indices]
        else:
            # Fallback for single index
            return self.dataset[batch_indices]

    def __len__(self):
        return len(self.dataset)


def get_train_data_loader(
    config,
    num_workers,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    step_tracker=None,
):
    dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
    module_name, class_name = dataset_name.rsplit(".", 1)
    dataset_class = importlib.import_module(module_name).__dict__[class_name]
    dataset_config = deepcopy(config)
    loader_seed = config.training.get(
        "data_loader_seed", dataset_config.data.get("seed", 42)
    )
    dataset_config.data.seed = loader_seed
    dataset = dataset_class(dataset_config)
    if step_tracker is not None and hasattr(dataset, "set_step_tracker"):
        dataset.set_step_tracker(step_tracker)

    if isinstance(dataset, IterableDataset):
        batch_size = _single_batch_size(
            config.training.batch_size_per_gpu,
            "Train",
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        generator = torch.Generator().manual_seed(loader_seed + rank)
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_mem,
            "drop_last": drop_last,
            "persistent_workers": num_workers > 0,
            "generator": generator,
            "worker_init_fn": _seed_worker,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = config.training.get(
                "prefetch_factor", 2
            )
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    batch_sampler = dataset.make_sampler(
        batch_size_per_gpu=config.training.batch_size_per_gpu,
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        use_dynamic_sampler=True,
    )

    warpped_dataset = DynamicBatchDatasetWrapper(dataset)

    data_loader = torch.utils.data.DataLoader(
        warpped_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return data_loader


def get_val_data_loader(config, step_tracker=None, pin_mem=True):
    validation_config = config.validation
    dataset_name = validation_config.get(
        "dataset_name",
        config.training.get("dataset_name", "data.dataset.Dataset"),
    )
    module_name, class_name = dataset_name.rsplit(".", 1)
    dataset_class = importlib.import_module(module_name).__dict__[class_name]

    dataset_config = deepcopy(config)
    dataset_config.data.stage = validation_config.get("data_stage", "val")
    dataset_config.data.augment = False
    dataset_config.data.seed = validation_config.seed
    dataset = dataset_class(dataset_config)
    if step_tracker is not None and hasattr(dataset, "set_step_tracker"):
        dataset.set_step_tracker(step_tracker)
    if not isinstance(dataset, IterableDataset):
        raise TypeError("Periodic validation currently requires an IterableDataset")

    num_workers = validation_config.num_workers
    rank = dist.get_rank() if dist.is_initialized() else 0
    generator = torch.Generator().manual_seed(validation_config.seed + rank)
    loader_kwargs = {
        "batch_size": validation_config.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_mem,
        "drop_last": False,
        "persistent_workers": num_workers > 0,
        "generator": generator,
        "worker_init_fn": _seed_worker,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = validation_config.get(
            "prefetch_factor", 2
        )
    return torch.utils.data.DataLoader(
        ValidationWrapper(dataset, length=1),
        **loader_kwargs,
    )
