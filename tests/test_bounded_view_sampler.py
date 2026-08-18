from easydict import EasyDict as edict

from data.view_sampler import BoundedViewSampler


class _StepTracker:
    def __init__(self, step=0):
        self.step = step

    def get_step(self):
        return self.step


def _config():
    return edict(
        num_context_views=2,
        num_target_views=4,
        min_context_gap=45,
        max_context_gap=90,
        initial_min_context_gap=25,
        initial_max_context_gap=25,
        context_gap_warmup_steps=37500,
        min_target_distance=0,
    )


def test_bounded_sampler_warms_context_gap_like_c3g():
    tracker = _StepTracker()
    sampler = BoundedViewSampler(_config(), "train", tracker)

    assert sampler.context_gap_bounds(100) == (25, 25)

    tracker.step = 18750
    assert sampler.context_gap_bounds(100) == (35, 57)

    tracker.step = 37500
    assert sampler.context_gap_bounds(100) == (45, 90)


def test_validation_uses_the_same_gap_schedule():
    tracker = _StepTracker(37500)
    sampler = BoundedViewSampler(_config(), "val", tracker)

    assert sampler.context_gap_bounds(100) == (45, 90)
