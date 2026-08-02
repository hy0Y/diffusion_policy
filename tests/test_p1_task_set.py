from pathlib import Path

import pytest

from diffusion_policy.dataset.p1_mvp_sampler import EpisodeBundle, GroupSampleIndex
from diffusion_policy.dataset.p1_task_set import P1TaskEntry, P1TaskSet, sample_task_set_epoch


class FakeSampler:
    def __init__(self, episode_ids):
        self.episode_ids = tuple(episode_ids)

    def sample_epoch(self, *, epoch):
        batches = tuple((EpisodeBundle(episode_id=episode_id, groups=(GroupSampleIndex(episode_id, 0, "natural"),)),) for episode_id in self.episode_ids)
        return type("Schedule", (), {"batches": batches})()


def task_set(*names):
    return P1TaskSet(tuple(P1TaskEntry(name, Path("/raw") / name, Path("/split"), Path("/ann") / name, Path("/cache") / name) for name in names))


def test_task_set_schedule_is_explicit_deterministic_and_task_homogeneous():
    selected = task_set("A", "B")
    samplers = {"A": FakeSampler([1, 2]), "B": FakeSampler([10])}
    left = sample_task_set_epoch(samplers, task_set=selected, epoch=3, seed=42)
    right = sample_task_set_epoch(samplers, task_set=selected, epoch=3, seed=42)
    assert left == right
    assert set(left.task_ids_by_batch()) == {"A", "B"}
    seen = {(item.task_id, item.bundle.episode_id) for batch in left.batches for item in batch}
    assert seen == {("A", 1), ("A", 2), ("B", 10)}


def test_task_set_rejects_duplicate_or_mismatched_entries():
    with pytest.raises(ValueError, match="duplicate"):
        task_set("A", "A")
    with pytest.raises(ValueError, match="match"):
        sample_task_set_epoch({"A": FakeSampler([1])}, task_set=task_set("A", "B"), epoch=0, seed=0)
