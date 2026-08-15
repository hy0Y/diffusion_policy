"""Independent single-chunk dataset for DP + Jump training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from diffusion_policy.dataset.p1_mvp_contract import resolve_p1_mvp_split


DEFAULT_CATEGORY_QUOTA = {
    "boundary_centered": 16,
    "near_negative": 16,
    "interior": 16,
    "natural": 8,
    "episode_start": 2,
}


@dataclass(frozen=True)
class ChunkSampleIndex:
    episode_id: int
    base_time: int
    category: str
    terminal_boundary: bool = False
    touches_done: bool = False


@dataclass(frozen=True)
class ChunkCandidatePools:
    boundary_centered: np.ndarray
    boundary_terminal: np.ndarray
    near_negative: np.ndarray
    interior: np.ndarray
    natural_nonterminal: np.ndarray
    natural_with_terminal: np.ndarray
    natural_touches_done: np.ndarray
    episode_start: np.ndarray


@dataclass(frozen=True)
class _CachedEpisode:
    feature: np.ndarray
    action: np.ndarray
    physical_action: np.ndarray
    previous_action: np.ndarray
    boundary: np.ndarray
    subtask: np.ndarray
    stage: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class _CachedAnnotation:
    boundary: np.ndarray
    subtask: np.ndarray
    stage: np.ndarray
    valid: np.ndarray


class JumpMVPCachedEpisodeStore:
    """Read cached features/actions and annotations without mutating them."""

    def __init__(
            self,
            split_artifact: str | Path,
            *,
            split: str,
            horizon: int,
            n_obs_steps: int,
            cache_action_n_obs_steps: int = 2,
            feature_variant: int = 0,
            task: str = "GetToastedBread",
            annotation_root: str | Path | None = None,
            cache_root: str | Path | None = None,
            canonical_sha256: str | None = None,
        ):
        if horizon < 2:
            raise ValueError("horizon must be at least two")
        if n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        if n_obs_steps >= horizon:
            raise ValueError(
                "n_obs_steps must leave at least two executable Jump tokens")
        if not 1 <= cache_action_n_obs_steps <= horizon:
            raise ValueError(
                "cache_action_n_obs_steps must lie within the action horizon")
        if feature_variant < 0:
            raise ValueError("feature_variant must be non-negative")
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
        self.cache_root = contract.cache_root
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.cache_action_n_obs_steps = int(cache_action_n_obs_steps)
        self.feature_variant = int(feature_variant)

    def _paths(self, episode_id: int) -> tuple[Path, Path, Path]:
        if episode_id not in self.episode_ids:
            raise KeyError(f"episode {episode_id} is not in this split")
        stem = f"episode_{episode_id:06d}"
        return (
            self.annotation_root / f"{stem}.parquet",
            self.cache_root / f"{stem}_features.npy",
            self.cache_root / f"{stem}_actions.npy",
        )

    @lru_cache(maxsize=4)
    def load_episode(self, episode_id: int) -> _CachedEpisode:
        annotation_path, feature_path, action_path = self._paths(episode_id)
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
        if features.ndim != 3 or self.feature_variant >= features.shape[1]:
            raise ValueError(f"invalid feature cache shape: {features.shape}")
        frame_count = int(features.shape[0])
        if actions.ndim != 3 or actions.shape[:2] != (
                frame_count, self.horizon):
            raise ValueError(
                f"action cache must have shape [N,{self.horizon},D]")
        annotation = self.load_annotation(episode_id)
        if annotation.valid.shape != (frame_count,):
            raise ValueError("annotation/cache lengths do not match")
        # Cache row r, token cache_n_obs_steps-1 is physical action a[r].
        # Reconstructing the physical sequence makes model n_obs_steps
        # independent of the immutable cache-generation value and also lets
        # episode-start samples reproduce the cache's left zero padding.
        physical_action = np.asarray(
            actions[:, self.cache_action_n_obs_steps - 1],
            dtype=np.float32,
        ).copy()
        for token in range(self.horizon):
            physical_index = (
                np.arange(frame_count, dtype=np.int64)
                - self.cache_action_n_obs_steps
                + 1
                + token
            )
            inside = (physical_index >= 0) & (physical_index < frame_count)
            if not np.array_equal(
                    np.asarray(actions[inside, token]),
                    physical_action[physical_index[inside]]):
                raise ValueError(
                    "cache_action_n_obs_steps disagrees with action windows")
            if bool(np.any(np.asarray(actions[~inside, token]) != 0)):
                raise ValueError(
                    "cache_action_n_obs_steps disagrees with episode padding")
        previous_action = np.zeros(
            (frame_count, actions.shape[-1]), dtype=np.float32)
        if frame_count > 1:
            previous_action[1:] = physical_action[:-1]
        return _CachedEpisode(
            feature=np.asarray(features[:, self.feature_variant]).copy(),
            action=np.asarray(actions).copy(),
            physical_action=physical_action,
            previous_action=previous_action,
            boundary=annotation.boundary,
            subtask=annotation.subtask,
            stage=annotation.stage,
            valid=annotation.valid,
        )

    @lru_cache(maxsize=512)
    def load_annotation(self, episode_id: int) -> _CachedAnnotation:
        annotation_path, _, _ = self._paths(episode_id)
        table = pq.read_table(
            annotation_path,
            columns=[
                "frame_index",
                "subtask_idx",
                "stage_text",
                "boundary_any_hard",
                "valid_mask",
            ],
        ).to_pydict()
        frame = np.asarray(table["frame_index"], dtype=np.int64)
        if not np.array_equal(frame, np.arange(frame.size)):
            raise ValueError("annotation frame index is not contiguous")
        valid = np.asarray(table["valid_mask"], dtype=bool)
        return _CachedAnnotation(
            boundary=np.asarray(table["boundary_any_hard"], dtype=bool),
            subtask=np.asarray(table["subtask_idx"], dtype=np.int64),
            stage=np.asarray(table["stage_text"], dtype=str),
            valid=valid,
        )

    def candidate_pools(
            self,
            episode_id: int,
            *,
            near_max_gap: int | None = None,
        ) -> ChunkCandidatePools:
        annotation = self.load_annotation(episode_id)
        frame_count = annotation.valid.shape[0]
        maximum_start = min(
            frame_count - self.horizon,
            frame_count - self.n_obs_steps,
            frame_count - self.cache_action_n_obs_steps,
        )
        minimum_start = 0
        all_starts = np.arange(
            minimum_start, maximum_start + 1, dtype=np.int64)
        boundary_frames = np.flatnonzero(annotation.boundary).astype(np.int64)
        # Jump begins at the first action that vanilla DP will execute. A
        # boundary already observed inside the condition window is not a
        # future Jump edge and must not drive the sampling category.
        first_edge_offset = self.n_obs_steps
        last_edge_offset = self.horizon - 1
        center = (first_edge_offset + last_edge_offset) // 2
        center_offsets = {
            offset for offset in (center - 1, center, center + 1)
            if first_edge_offset <= offset <= last_edge_offset
        }
        if near_max_gap is None:
            near_max_gap = max(1, self.horizon - 2)

        centered = []
        centered_terminal = []
        near = []
        interior = []
        natural_nonterminal = []
        natural_touches_done = []
        for start in all_starts.tolist():
            stop = start + self.horizon - 1
            if not bool(np.all(annotation.valid[start:stop + 1])):
                continue
            active_start = start + self.n_obs_steps - 1
            edge_boundaries = boundary_frames[
                (boundary_frames >= active_start + 1)
                & (boundary_frames <= stop)]
            if edge_boundaries.size == 1 \
                    and int(edge_boundaries[0] - start) in center_offsets:
                centered.append(start)
                if annotation.stage[int(edge_boundaries[0])] == "done":
                    centered_terminal.append(start)
            if edge_boundaries.size == 0:
                future = boundary_frames[boundary_frames > stop]
                if future.size and int(future[0]) <= stop + near_max_gap \
                        and annotation.stage[int(future[0])] != "done":
                    near.append(start)
            active_slice = slice(active_start, stop + 1)
            if np.all(
                    annotation.subtask[active_slice]
                    == annotation.subtask[active_start]) \
                    and not np.any(annotation.stage[active_slice] == "done"):
                interior.append(start)
            if np.any(annotation.stage[active_slice] == "done"):
                natural_touches_done.append(start)
            else:
                natural_nonterminal.append(start)
        valid_starts = np.asarray([
            start for start in all_starts.tolist()
            if bool(np.all(annotation.valid[start:start + self.horizon]))
        ], dtype=np.int64)
        return ChunkCandidatePools(
            boundary_centered=np.asarray(centered, dtype=np.int64),
            boundary_terminal=np.asarray(centered_terminal, dtype=np.int64),
            near_negative=np.asarray(near, dtype=np.int64),
            interior=np.asarray(interior, dtype=np.int64),
            natural_nonterminal=np.asarray(natural_nonterminal, dtype=np.int64),
            natural_with_terminal=valid_starts,
            natural_touches_done=np.asarray(
                natural_touches_done, dtype=np.int64),
            episode_start=(
                np.asarray([1 - self.n_obs_steps], dtype=np.int64)
                if frame_count >= self.horizon - self.n_obs_steps + 1
                else np.empty(0, dtype=np.int64)),
        )

    def load_chunk(self, index: ChunkSampleIndex) -> dict[str, Any]:
        episode = self.load_episode(index.episode_id)
        start = int(index.base_time)
        stop = start + self.horizon - 1
        episode_start = 1 - self.n_obs_steps
        if start < 0 and not (
                index.category == "episode_start" and start == episode_start):
            raise IndexError("negative base_time is reserved for episode start")
        if stop >= episode.feature.shape[0]:
            raise IndexError("chunk lies outside the episode")
        observation_anchor = start + self.n_obs_steps - 1
        observation_start = observation_anchor - self.n_obs_steps + 1
        if observation_anchor < 0 or observation_anchor >= episode.feature.shape[0]:
            raise IndexError("observation context lies outside the episode")
        observation_indices = np.arange(
            observation_start, observation_anchor + 1)
        observation_indices = np.clip(observation_indices, 0, None)
        physical_indices = start + np.arange(self.horizon, dtype=np.int64)
        in_episode = (
            (physical_indices >= 0)
            & (physical_indices < episode.feature.shape[0])
        )
        action = np.zeros(
            (self.horizon, episode.physical_action.shape[-1]),
            dtype=np.float32,
        )
        action[in_episode] = episode.physical_action[physical_indices[in_episode]]
        action_valid = np.zeros(self.horizon, dtype=bool)
        action_valid[in_episode] = episode.valid[physical_indices[in_episode]]
        full_edge_valid = action_valid[:-1] & action_valid[1:]
        jump_action_start = self.n_obs_steps - 1
        edge_valid = full_edge_valid[jump_action_start:]
        history_length = observation_anchor + 1
        return {
            "episode_id": torch.tensor(index.episode_id, dtype=torch.long),
            "base_time": torch.tensor(start, dtype=torch.long),
            "category": index.category,
            "terminal_boundary": torch.tensor(
                index.terminal_boundary, dtype=torch.bool),
            "touches_done": torch.tensor(index.touches_done, dtype=torch.bool),
            "obs_feature": torch.from_numpy(
                episode.feature[observation_indices].copy()),
            "action": torch.from_numpy(action),
            "action_valid_mask": torch.from_numpy(action_valid.copy()),
            "boundary_target": torch.from_numpy(
                episode.boundary[
                    start + self.n_obs_steps:stop + 1].copy()),
            "edge_valid_mask": torch.from_numpy(edge_valid.copy()),
            "edge_dt_seconds": torch.full(
                (self.horizon - self.n_obs_steps,),
                0.0,
                dtype=torch.float32,
            ),
            "history_feature": torch.from_numpy(
                episode.feature[:history_length].copy()),
            "history_previous_action": torch.from_numpy(
                episode.previous_action[:history_length].copy()),
            "history_length": torch.tensor(history_length, dtype=torch.long),
            "episode_length": torch.tensor(
                episode.feature.shape[0], dtype=torch.long),
        }


def _draw(
        pool: np.ndarray,
        *,
        count: int,
        rng: np.random.Generator,
        episode_id: int,
        category: str,
        terminal_pool: np.ndarray,
        done_pool: np.ndarray,
    ) -> list[ChunkSampleIndex]:
    if count < 0:
        raise ValueError("category quota must be non-negative")
    if count == 0:
        return []
    if pool.size == 0:
        raise RuntimeError(
            f"episode {episode_id} has no candidates for {category}")
    starts = rng.choice(pool, size=count, replace=pool.size < count)
    terminal = set(int(value) for value in terminal_pool.tolist())
    touches_done = set(int(value) for value in done_pool.tolist())
    return [
        ChunkSampleIndex(
            episode_id=episode_id,
            base_time=int(start),
            category=category,
            terminal_boundary=int(start) in terminal,
            touches_done=int(start) in touches_done,
        )
        for start in starts.tolist()
    ]


class JumpMVPChunkDataset(Dataset):
    """Deterministic epoch schedule of mutually uncoupled chunk items."""

    def __init__(
            self,
            split_artifact: str | Path,
            *,
            split: str,
            horizon: int,
            n_obs_steps: int,
            cache_action_n_obs_steps: int = 2,
            edge_dt_seconds: float,
            seed: int,
            feature_variant: int = 0,
            category_quota: Mapping[str, int] | None = None,
            natural_done_policy: str = "include_terminal",
            task: str = "GetToastedBread",
            annotation_root: str | Path | None = None,
            cache_root: str | Path | None = None,
            canonical_sha256: str | None = None,
        ):
        super().__init__()
        if edge_dt_seconds <= 0:
            raise ValueError("edge_dt_seconds must be positive")
        if natural_done_policy not in ("exclude_terminal", "include_terminal"):
            raise ValueError(
                "natural_done_policy must be exclude_terminal or include_terminal")
        self.store = JumpMVPCachedEpisodeStore(
            split_artifact,
            split=split,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cache_action_n_obs_steps=cache_action_n_obs_steps,
            feature_variant=feature_variant,
            task=task,
            annotation_root=annotation_root,
            cache_root=cache_root,
            canonical_sha256=canonical_sha256,
        )
        self.edge_dt_seconds = float(edge_dt_seconds)
        self.seed = int(seed)
        self.natural_done_policy = natural_done_policy
        self.category_quota = dict(
            DEFAULT_CATEGORY_QUOTA if category_quota is None else category_quota)
        if set(self.category_quota) != set(DEFAULT_CATEGORY_QUOTA):
            raise ValueError(
                f"category_quota keys must be {tuple(DEFAULT_CATEGORY_QUOTA)}")
        self.epoch = 0
        self._indices: tuple[ChunkSampleIndex, ...] = ()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        indices = []
        for episode_id in self.store.episode_ids:
            pools = self.store.candidate_pools(episode_id)
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, epoch, episode_id]))
            natural = (
                pools.natural_nonterminal
                if self.natural_done_policy == "exclude_terminal"
                else pools.natural_with_terminal
            )
            category_pools = {
                "boundary_centered": pools.boundary_centered,
                "near_negative": pools.near_negative,
                "interior": pools.interior,
                "natural": natural,
                "episode_start": pools.episode_start,
            }
            for category, count in self.category_quota.items():
                indices.extend(_draw(
                    category_pools[category],
                    count=int(count),
                    rng=rng,
                    episode_id=episode_id,
                    category=category,
                    terminal_pool=pools.boundary_terminal,
                    done_pool=pools.natural_touches_done,
                ))
        shuffle_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, 0x4A554D50]))
        order = shuffle_rng.permutation(len(indices))
        self._indices = tuple(indices[int(index)] for index in order)
        self.epoch = int(epoch)

    @property
    def sample_indices(self) -> tuple[ChunkSampleIndex, ...]:
        return self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int | ChunkSampleIndex) -> dict[str, Any]:
        if isinstance(index, int):
            sample_index = self._indices[index]
        elif isinstance(index, ChunkSampleIndex):
            sample_index = index
        else:
            raise TypeError("index must be int or ChunkSampleIndex")
        result = self.store.load_chunk(sample_index)
        result["edge_dt_seconds"].fill_(self.edge_dt_seconds)
        return result


_HISTORY_KEYS = (
    "history_feature",
    "history_previous_action",
)


def jump_chunk_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad each independent causal prefix without episode-level deduplication."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    for item in batch:
        length = int(item["history_length"].item())
        if any(item[key].shape[0] != length for key in _HISTORY_KEYS):
            raise ValueError("history tensors and history_length disagree")
    result = default_collate([
        {key: value for key, value in item.items()
         if key not in _HISTORY_KEYS}
        for item in batch
    ])
    for key in _HISTORY_KEYS:
        result[key] = pad_sequence(
            [item[key] for item in batch], batch_first=True)
    max_length = result["history_feature"].shape[1]
    result["history_valid_mask"] = (
        torch.arange(max_length)[None]
        < result["history_length"][:, None]
    )
    return result
