"""Deterministic episode-complete sampler for the GetToastedBread P1 MVP."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Sampler

from diffusion_policy.dataset.p1_mvp_group_dataset import (
    GROUP_UNION,
    GroupSampleIndex,
)
from diffusion_policy.dataset.p1_mvp_contract import resolve_p1_mvp_split


NaturalDonePolicy = Literal["exclude_terminal", "include_terminal"]
CATEGORY_QUOTA = {
    "boundary_centered": 2,
    "near_negative": 2,
    "interior": 2,
    "natural": 1,
    "episode_start": 1,
}


@dataclass(frozen=True)
class EpisodePools:
    episode_id: int
    boundary: np.ndarray
    terminal_boundary: np.ndarray
    near: np.ndarray
    interior: np.ndarray
    interior_segment: np.ndarray
    natural_nonterminal: np.ndarray
    natural_with_terminal: np.ndarray
    natural_touches_done: np.ndarray
    episode_start: np.ndarray


@dataclass(frozen=True)
class EpisodeBundle:
    episode_id: int
    groups: tuple[GroupSampleIndex, ...]


@dataclass(frozen=True)
class EpochSchedule:
    epoch: int
    batches: tuple[tuple[EpisodeBundle, ...], ...]

    @property
    def episode_count(self) -> int:
        return sum(len(batch) for batch in self.batches)

    @property
    def group_count(self) -> int:
        return sum(len(bundle.groups) for batch in self.batches for bundle in batch)

    @property
    def window_count(self) -> int:
        return self.group_count * 4


@dataclass(frozen=True)
class RankEpochSchedule:
    """One DDP rank's fixed-width view of a global episode-complete epoch."""

    epoch: int
    rank: int
    world_size: int
    batches: tuple[tuple[EpisodeBundle, ...], ...]
    valid_episode_mask: tuple[tuple[bool, ...], ...]

    @property
    def real_episode_count(self) -> int:
        return sum(sum(mask) for mask in self.valid_episode_mask)

    @property
    def dummy_episode_count(self) -> int:
        return sum(len(mask) - sum(mask) for mask in self.valid_episode_mask)


def intervals_overlap(left: int, right: int, union: int = GROUP_UNION) -> bool:
    return not (left + union - 1 < right or right + union - 1 < left)


def _available(starts: np.ndarray, selected: Iterable[int]) -> np.ndarray:
    occupied = tuple(selected)
    if not occupied:
        return starts
    return np.asarray(
        [
            int(start)
            for start in starts.tolist()
            if all(not intervals_overlap(int(start), used) for used in occupied)
        ],
        dtype=np.int64,
    )


class CandidateStore:
    def __init__(self, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as archive:
            self._arrays = {name: archive[name].copy() for name in archive.files}
        self.episode_ids = tuple(
            int(value) for value in np.unique(self._arrays["episode_start_episode"])
        )

    def _starts(self, prefix: str, episode_id: int) -> np.ndarray:
        mask = self._arrays[f"{prefix}_episode"] == episode_id
        return self._arrays[f"{prefix}_start"][mask]

    def episode(self, episode_id: int) -> EpisodePools:
        if episode_id not in self.episode_ids:
            raise KeyError(f"episode {episode_id} is not in the candidate artifact")
        interior_mask = self._arrays["interior_episode"] == episode_id
        return EpisodePools(
            episode_id=episode_id,
            boundary=self._starts("boundary", episode_id),
            terminal_boundary=self._arrays["boundary_start"][
                self._arrays["boundary_terminal"].astype(bool)
                & (self._arrays["boundary_episode"] == episode_id)
            ],
            near=self._starts("near", episode_id),
            interior=self._arrays["interior_start"][interior_mask],
            interior_segment=self._arrays["interior_segment"][interior_mask],
            natural_nonterminal=self._starts("natural_nonterminal", episode_id),
            natural_with_terminal=self._starts("natural_with_terminal", episode_id),
            natural_touches_done=self._arrays["natural_with_terminal_start"][
                self._arrays["natural_with_terminal_touches_done"].astype(bool)
                & (self._arrays["natural_with_terminal_episode"] == episode_id)
            ],
            episode_start=self._starts("episode_start", episode_id),
        )


class AnnotationCandidateStore:
    """Build the same candidate pools directly from a locked split annotation.

    The audited NPZ remains the canonical train artifact. This read-only store is
    primarily for validation/test episodes, which are deliberately absent from
    that train-only artifact.
    """

    def __init__(
            self,
            split_artifact: str | Path,
            *,
            split: str,
            task: str = "GetToastedBread",
            annotation_root: str | Path | None = None,
            cache_root: str | Path | None = None,
            canonical_sha256: str | None = None,
        ):
        contract = resolve_p1_mvp_split(
            split_artifact,
            split=split,
            task=task,
            annotation_root=annotation_root,
            cache_root=cache_root,
            canonical_sha256=canonical_sha256,
        )
        self.episode_ids = contract.episode_ids
        self.annotation_root = contract.annotation_root

    @staticmethod
    def _segments(values: np.ndarray) -> list[tuple[int, int, int]]:
        cuts = np.flatnonzero(values[1:] != values[:-1]) + 1
        starts = np.concatenate((np.asarray([0]), cuts))
        stops = np.concatenate((cuts, np.asarray([values.size])))
        return [
            (int(start), int(stop - 1), int(values[start]))
            for start, stop in zip(starts, stops)
        ]

    @lru_cache(maxsize=512)
    def episode(self, episode_id: int) -> EpisodePools:
        if episode_id not in self.episode_ids:
            raise KeyError(f"episode {episode_id} is not in this split")
        path = self.annotation_root / f"episode_{episode_id:06d}.parquet"
        table = pq.read_table(
            path,
            columns=[
                "frame_index",
                "subtask_idx",
                "stage_text",
                "boundary_any_hard",
                "valid_mask",
            ],
        ).to_pydict()
        frame = np.asarray(table["frame_index"], dtype=np.int64)
        subtask = np.asarray(table["subtask_idx"], dtype=np.int64)
        stage = np.asarray(table["stage_text"], dtype=str)
        boundary_mask = np.asarray(table["boundary_any_hard"], dtype=bool)
        valid = np.asarray(table["valid_mask"], dtype=bool)
        if not np.array_equal(frame, np.arange(frame.size)) or not np.all(valid):
            raise ValueError(f"episode {episode_id} violates the P1 frame/valid contract")

        all_starts = np.arange(
            max(0, frame.size - GROUP_UNION + 1), dtype=np.int64)
        boundary_frames = np.flatnonzero(boundary_mask).astype(np.int64)
        centered: list[int] = []
        centered_terminal: list[int] = []
        near: list[int] = []
        for start in all_starts.tolist():
            stop = start + GROUP_UNION - 1
            inside = boundary_frames[
                (boundary_frames >= start) & (boundary_frames <= stop)]
            if inside.size:
                if inside.size == 1 and start + 11 <= int(inside[0]) <= start + 13:
                    centered.append(start)
                    if stage[int(inside[0])] == "done":
                        centered_terminal.append(start)
                continue
            future = boundary_frames[boundary_frames > stop]
            if future.size and start + 25 <= int(future[0]) <= start + 32:
                if stage[int(future[0])] != "done":
                    near.append(start)

        interior: list[int] = []
        interior_segment: list[int] = []
        for segment_number, (lo, hi, _) in enumerate(self._segments(subtask)):
            stage_values = np.unique(stage[lo : hi + 1])
            if stage_values.size != 1:
                raise ValueError("one subtask segment maps to multiple stages")
            last_start = hi - GROUP_UNION + 1
            if last_start >= lo and str(stage_values[0]) != "done":
                starts = np.arange(lo, last_start + 1, dtype=np.int64)
                interior.extend(starts.tolist())
                interior_segment.extend([segment_number] * int(starts.size))

        natural_with_terminal = all_starts
        natural_touches_done = np.asarray([
            start for start in all_starts.tolist()
            if np.any(stage[start : start + GROUP_UNION] == "done")
        ], dtype=np.int64)
        natural_nonterminal = np.asarray([
            start for start in all_starts.tolist()
            if not np.any(stage[start : start + GROUP_UNION] == "done")
        ], dtype=np.int64)
        return EpisodePools(
            episode_id=episode_id,
            boundary=np.asarray(centered, dtype=np.int64),
            terminal_boundary=np.asarray(centered_terminal, dtype=np.int64),
            near=np.asarray(near, dtype=np.int64),
            interior=np.asarray(interior, dtype=np.int64),
            interior_segment=np.asarray(interior_segment, dtype=np.int64),
            natural_nonterminal=natural_nonterminal,
            natural_with_terminal=natural_with_terminal,
            natural_touches_done=natural_touches_done,
            episode_start=(
                np.asarray([0], dtype=np.int64)
                if all_starts.size else np.empty(0, dtype=np.int64)),
        )


class EpisodeBundleSampler:
    def __init__(
            self,
            candidates: CandidateStore,
            *,
            seed: int,
            natural_done_policy: NaturalDonePolicy,
            episodes_per_batch: int = 4,
            max_restarts: int = 256,
        ):
        if natural_done_policy not in ("exclude_terminal", "include_terminal"):
            raise ValueError(
                "natural_done_policy must explicitly be exclude_terminal or "
                "include_terminal")
        if episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be positive")
        self.candidates = candidates
        self.seed = seed
        self.natural_done_policy = natural_done_policy
        self.episodes_per_batch = episodes_per_batch
        self.max_restarts = max_restarts

    def _choose_uniform(
            self,
            *,
            pool: np.ndarray,
            count: int,
            selected: list[GroupSampleIndex],
            category: str,
            rng: np.random.Generator,
        ) -> bool:
        for _ in range(count):
            available = _available(pool, (group.group_start for group in selected))
            if available.size == 0:
                return False
            start = int(available[int(rng.integers(available.size))])
            selected.append(GroupSampleIndex(-1, start, category))
        return True

    def _choose_interior(
            self,
            *,
            pools: EpisodePools,
            selected: list[GroupSampleIndex],
            rng: np.random.Generator,
        ) -> bool:
        for _ in range(CATEGORY_QUOTA["interior"]):
            available_mask = np.asarray(
                [
                    all(
                        not intervals_overlap(int(start), group.group_start)
                        for group in selected
                    )
                    for start in pools.interior.tolist()
                ],
                dtype=bool,
            )
            if not np.any(available_mask):
                return False
            starts = pools.interior[available_mask]
            segments = pools.interior_segment[available_mask]
            unique_segments, counts = np.unique(segments, return_counts=True)
            weights = np.sqrt(counts.astype(np.float64))
            weights /= weights.sum()
            segment = int(rng.choice(unique_segments, p=weights))
            segment_starts = starts[segments == segment]
            start = int(segment_starts[int(rng.integers(segment_starts.size))])
            selected.append(
                GroupSampleIndex(
                    -1,
                    start,
                    "interior",
                    segment_number=segment,
                )
            )
        return True

    def sample_episode(self, episode_id: int, *, epoch: int) -> EpisodeBundle:
        pools = self.candidates.episode(episode_id)
        natural = (
            pools.natural_nonterminal
            if self.natural_done_policy == "exclude_terminal"
            else pools.natural_with_terminal
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, episode_id]))

        for _ in range(self.max_restarts):
            selected: list[GroupSampleIndex] = []
            if not self._choose_uniform(
                pool=pools.boundary,
                count=CATEGORY_QUOTA["boundary_centered"],
                selected=selected,
                category="boundary_centered",
                rng=rng,
            ):
                continue
            if not self._choose_uniform(
                pool=pools.near,
                count=CATEGORY_QUOTA["near_negative"],
                selected=selected,
                category="near_negative",
                rng=rng,
            ):
                continue
            if not self._choose_interior(pools=pools, selected=selected, rng=rng):
                continue
            if not self._choose_uniform(
                pool=natural,
                count=CATEGORY_QUOTA["natural"],
                selected=selected,
                category="natural",
                rng=rng,
            ):
                continue
            if not self._choose_uniform(
                pool=pools.episode_start,
                count=CATEGORY_QUOTA["episode_start"],
                selected=selected,
                category="episode_start",
                rng=rng,
            ):
                continue
            groups = tuple(
                GroupSampleIndex(
                    episode_id=episode_id,
                    group_start=group.group_start,
                    category=group.category,
                    segment_number=group.segment_number,
                    terminal_boundary=(
                        group.category == "boundary_centered"
                        and group.group_start in pools.terminal_boundary
                    ),
                    touches_done=(
                        group.category == "natural"
                        and group.group_start in pools.natural_touches_done
                    ),
                )
                for group in selected
            )
            self._assert_bundle(groups)
            return EpisodeBundle(episode_id=episode_id, groups=groups)

        counts = {
            "boundary": int(pools.boundary.size),
            "near": int(pools.near.size),
            "interior": int(pools.interior.size),
            "natural": int(natural.size),
            "episode_start": int(pools.episode_start.size),
        }
        raise RuntimeError(
            f"episode {episode_id}: failed quota/non-overlap after "
            f"{self.max_restarts} restarts; candidate counts={counts}")

    @staticmethod
    def _assert_bundle(groups: tuple[GroupSampleIndex, ...]) -> None:
        observed = {category: 0 for category in CATEGORY_QUOTA}
        for group in groups:
            observed[group.category] += 1
        if observed != CATEGORY_QUOTA:
            raise AssertionError(f"quota mismatch: {observed}")
        for index, left in enumerate(groups):
            for right in groups[index + 1:]:
                if intervals_overlap(left.group_start, right.group_start):
                    raise AssertionError(
                        f"overlap in episode {left.episode_id}: "
                        f"{left.group_start} and {right.group_start}")

    def sample_epoch(self, *, epoch: int) -> EpochSchedule:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, 0xE0C]))
        episode_ids = np.asarray(self.candidates.episode_ids, dtype=np.int64)
        ordered = episode_ids[rng.permutation(episode_ids.size)].tolist()
        bundles = [
            self.sample_episode(int(episode_id), epoch=epoch)
            for episode_id in ordered
        ]
        batches = tuple(
            tuple(bundles[index:index + self.episodes_per_batch])
            for index in range(0, len(bundles), self.episodes_per_batch)
        )
        schedule = EpochSchedule(epoch=epoch, batches=batches)
        flattened_ids = [
            bundle.episode_id for batch in batches for bundle in batch
        ]
        if len(flattened_ids) != len(set(flattened_ids)):
            raise AssertionError("an epoch must not repeat source episodes")
        if set(flattened_ids) != set(self.candidates.episode_ids):
            raise AssertionError("an epoch must cover every source episode exactly once")
        return schedule


class EpisodeBundleBatchSampler(Sampler[list[GroupSampleIndex]]):
    """Flatten each 4-episode physical batch to its 32 logical group instances."""

    def __init__(
            self,
            schedule: EpochSchedule | RankEpochSchedule,
            *,
            start_batch: int = 0,
        ):
        if not 0 <= start_batch <= len(schedule.batches):
            raise ValueError("start_batch is outside the epoch schedule")
        self.schedule = schedule
        self.start_batch = start_batch

    def __iter__(self) -> Iterator[list[GroupSampleIndex]]:
        for batch in self.schedule.batches[self.start_batch:]:
            yield [group for bundle in batch for group in bundle.groups]

    def __len__(self) -> int:
        return len(self.schedule.batches) - self.start_batch


def partition_epoch_for_ddp(
        schedule: EpochSchedule,
        *,
        rank: int,
        world_size: int,
        episodes_per_rank: int = 4,
    ) -> RankEpochSchedule:
    """Partition real episodes once and mask only final-step dummy slots.

    Every rank receives ``episodes_per_rank`` structurally valid bundles on every
    step. Reused dummy bundles carry ``GroupSampleIndex.valid=False`` and must be
    excluded from loss, metrics, and sampler coverage by the training workspace.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must lie in [0, world_size)")
    if episodes_per_rank <= 0:
        raise ValueError("episodes_per_rank must be positive")
    real_bundles = [
        bundle for batch in schedule.batches for bundle in batch
    ]
    if not real_bundles:
        raise ValueError("cannot partition an empty epoch")

    global_width = world_size * episodes_per_rank
    pad_template = real_bundles[0]
    rank_batches: list[tuple[EpisodeBundle, ...]] = []
    rank_masks: list[tuple[bool, ...]] = []
    for start in range(0, len(real_bundles), global_width):
        global_bundles = real_bundles[start:start + global_width]
        global_valid = [True] * len(global_bundles)
        missing = global_width - len(global_bundles)
        global_bundles.extend([pad_template] * missing)
        global_valid.extend([False] * missing)

        local_start = rank * episodes_per_rank
        local_stop = local_start + episodes_per_rank
        local_valid = tuple(global_valid[local_start:local_stop])
        local_bundles = []
        for bundle, valid in zip(
            global_bundles[local_start:local_stop], local_valid
        ):
            local_bundles.append(EpisodeBundle(
                episode_id=bundle.episode_id,
                groups=tuple(
                    replace(group, valid=valid) for group in bundle.groups
                ),
            ))
        rank_batches.append(tuple(local_bundles))
        rank_masks.append(local_valid)

    return RankEpochSchedule(
        epoch=schedule.epoch,
        rank=rank,
        world_size=world_size,
        batches=tuple(rank_batches),
        valid_episode_mask=tuple(rank_masks),
    )
