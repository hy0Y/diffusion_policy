"""Task-set scaffolding that preserves the single-task P1 MVP path.

This module intentionally does not discover task directories. Callers must provide
an explicit registry, so a run has a reproducible selected task set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from diffusion_policy.dataset.p1_mvp_sampler import EpisodeBundle, EpisodeBundleSampler


@dataclass(frozen=True)
class P1TaskEntry:
    """Immutable per-task data contract for a selected target/composite task."""

    task_id: str
    dataset_path: Path
    split_artifact: Path
    annotation_root: Path
    cache_root: Path

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")


@dataclass(frozen=True)
class P1TaskSet:
    """An explicit ordered task set; singleton, subset, and full set share it."""

    entries: tuple[P1TaskEntry, ...]

    def __post_init__(self) -> None:
        ids = self.task_ids
        if not ids:
            raise ValueError("task set must select at least one task")
        if len(ids) != len(set(ids)):
            raise ValueError("task set contains duplicate task_id values")

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(entry.task_id for entry in self.entries)

    def by_id(self) -> Mapping[str, P1TaskEntry]:
        return {entry.task_id: entry for entry in self.entries}


@dataclass(frozen=True)
class TaskEpisodeBundle:
    """A pre-existing episode bundle with an explicit task namespace."""

    task_id: str
    bundle: EpisodeBundle


@dataclass(frozen=True)
class TaskSetEpochSchedule:
    """M_b=1 schedule: every physical batch belongs to exactly one task."""

    epoch: int
    batches: tuple[tuple[TaskEpisodeBundle, ...], ...]

    def task_ids_by_batch(self) -> tuple[str, ...]:
        result = []
        for batch in self.batches:
            task_ids = {item.task_id for item in batch}
            if len(task_ids) != 1:
                raise AssertionError("M_b=1 batch contains multiple tasks")
            result.append(next(iter(task_ids)))
        return tuple(result)


def sample_task_set_epoch(
        samplers: Mapping[str, EpisodeBundleSampler],
        *,
        task_set: P1TaskSet,
        epoch: int,
        seed: int,
    ) -> TaskSetEpochSchedule:
    """Compose existing per-task samplers without changing their quota logic.

    Each selected task contributes every one of its own episode bundles exactly
    once. Batches remain task-homogeneous (M_b=1); their order is shuffled only
    after each task sampler has formed its normal physical batches.
    """
    expected = set(task_set.task_ids)
    if set(samplers) != expected:
        raise ValueError("samplers must match the explicit task set exactly")
    task_batches: list[tuple[TaskEpisodeBundle, ...]] = []
    for task_id in task_set.task_ids:
        schedule = samplers[task_id].sample_epoch(epoch=epoch)
        for batch in schedule.batches:
            task_batches.append(tuple(
                TaskEpisodeBundle(task_id=task_id, bundle=bundle)
                for bundle in batch
            ))
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0x7A5E7]))
    order = rng.permutation(len(task_batches))
    shuffled = tuple(task_batches[int(index)] for index in order)
    result = TaskSetEpochSchedule(epoch=epoch, batches=shuffled)
    result.task_ids_by_batch()
    return result
