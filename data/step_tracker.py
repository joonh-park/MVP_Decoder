from multiprocessing import RLock

import torch
from torch.multiprocessing import Manager


class StepTracker:
    """Share the optimizer step with persistent data-loader workers."""

    def __init__(self):
        self.lock: RLock = Manager().RLock()
        self.step = torch.tensor(0, dtype=torch.int64).share_memory_()

    def set_step(self, step: int) -> None:
        with self.lock:
            self.step.fill_(step)

    def get_step(self) -> int:
        with self.lock:
            return int(self.step.item())
