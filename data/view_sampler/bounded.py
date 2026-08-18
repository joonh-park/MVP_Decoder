import torch


class BoundedViewSampler:
    """C3G-style bounded context/target sampler with gap warmup."""

    def __init__(self, config, stage, step_tracker=None):
        self.stage = stage
        self.step_tracker = step_tracker
        self.num_context_views = config.num_context_views
        self.num_target_views = config.num_target_views
        self.min_context_gap = config.min_context_gap
        self.max_context_gap = config.max_context_gap
        self.initial_min_context_gap = config.get(
            "initial_min_context_gap", self.min_context_gap
        )
        self.initial_max_context_gap = config.get(
            "initial_max_context_gap", self.max_context_gap
        )
        self.warmup_steps = config.get("context_gap_warmup_steps", 0)
        self.min_target_distance = config.get("min_target_distance", 0)

    @property
    def global_step(self):
        return 0 if self.step_tracker is None else self.step_tracker.get_step()

    def _schedule(self, initial, final):
        if self.warmup_steps <= 0:
            return final
        fraction = self.global_step / self.warmup_steps
        return min(initial + int((final - initial) * fraction), final)

    def context_gap_bounds(self, num_frames):
        if self.stage in {"train", "val"}:
            minimum_gap = self._schedule(
                self.initial_min_context_gap,
                self.min_context_gap,
            )
            maximum_gap = self._schedule(
                self.initial_max_context_gap,
                self.max_context_gap,
            )
        else:
            minimum_gap = self.min_context_gap
            maximum_gap = self.max_context_gap

        maximum_gap = min(maximum_gap, num_frames - 1)
        minimum_gap = max(
            2 * self.min_target_distance,
            minimum_gap,
            self.num_context_views - 1,
        )
        return minimum_gap, maximum_gap

    def sample(self, num_frames, generator=None):
        minimum_gap, maximum_gap = self.context_gap_bounds(num_frames)
        if maximum_gap < minimum_gap:
            return None

        gap = int(
            torch.randint(
                minimum_gap,
                maximum_gap + 1,
                (),
                generator=generator,
            )
        )
        left = int(
            torch.randint(0, num_frames - gap, (), generator=generator)
        )
        right = left + gap

        if self.num_context_views > 2:
            interior = torch.randperm(gap - 1, generator=generator)[
                : self.num_context_views - 2
            ]
            context = torch.cat(
                (
                    torch.tensor([left]),
                    interior.add(left + 1),
                    torch.tensor([right]),
                )
            ).sort().values
        else:
            context = torch.tensor([left, right])

        targets = torch.randint(
            left + self.min_target_distance,
            right + 1 - self.min_target_distance,
            (self.num_target_views,),
            generator=generator,
        )
        return context, targets
