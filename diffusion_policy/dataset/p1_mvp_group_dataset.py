"""Runtime data contract for the GetToastedBread P1 MVP.

The production 50-epoch pilots read locked cached observation features. The raw
wrapper remains available for parity checks and explicitly configured ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence

from diffusion_policy.dataset.p1_mvp_contract import resolve_p1_mvp_split


GROUP_WINDOW = 10
GROUP_WINDOWS = 4
GROUP_STRIDE = 5
GROUP_UNION = GROUP_WINDOW + (GROUP_WINDOWS - 1) * GROUP_STRIDE


@dataclass(frozen=True)
class GroupSampleIndex:
    episode_id: int
    group_start: int
    category: str
    segment_number: int | None = None
    terminal_boundary: bool = False
    touches_done: bool = False
    valid: bool = True


@dataclass(frozen=True)
class CachedGroup:
    episode_id: int
    group_start: int
    observation_anchors: np.ndarray
    physical_time: np.ndarray
    observation_features: np.ndarray
    context_vector: np.ndarray
    clean_actions: np.ndarray
    boundary_edge_target: np.ndarray
    subtask_idx: np.ndarray
    stage_local_id: np.ndarray
    semantic_skill_local_id: np.ndarray


@dataclass(frozen=True)
class CachedHistory:
    feature: np.ndarray
    previous_action: np.ndarray
    subtask_idx: np.ndarray
    prefix_length: int
    episode_length: int


class CachedEpisodeStore:
    """Read locked P1 cache and annotation files without mutating them."""

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
        artifact_path = Path(split_artifact)
        contract = resolve_p1_mvp_split(
            artifact_path,
            split=split,
            task=task,
            annotation_root=annotation_root,
            cache_root=cache_root,
            canonical_sha256=canonical_sha256,
        )
        self.artifact_path = artifact_path
        self.episode_ids = contract.episode_ids
        self.annotation_root = contract.annotation_root
        self.cache_root = contract.cache_root

    def _paths(self, episode_id: int) -> tuple[Path, Path, Path]:
        if episode_id not in self.episode_ids:
            raise KeyError(f"episode {episode_id} is not in this split")
        stem = f"episode_{episode_id:06d}"
        return (
            self.annotation_root / f"{stem}.parquet",
            self.cache_root / f"{stem}_features.npy",
            self.cache_root / f"{stem}_actions.npy",
        )

    def load_group(
            self,
            episode_id: int,
            group_start: int,
            *,
            feature_variant: int,
        ) -> CachedGroup:
        if feature_variant not in range(5):
            raise ValueError("feature_variant must be in [0,4]")
        annotation_path, feature_path, action_path = self._paths(episode_id)
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
        frame_count = int(features.shape[0])
        expected_feature_shape = (frame_count, 5, 969)
        expected_action_shape = (frame_count, GROUP_WINDOW, 12)
        if features.shape != expected_feature_shape:
            raise ValueError(
                f"feature shape {features.shape} != {expected_feature_shape}")
        if actions.shape != expected_action_shape:
            raise ValueError(f"action shape {actions.shape} != {expected_action_shape}")

        window_starts = group_start + np.arange(GROUP_WINDOWS) * GROUP_STRIDE
        observation_anchors = window_starts + 1
        if group_start < 0 or int(window_starts[-1] + GROUP_WINDOW) > frame_count:
            raise IndexError(
                f"group [{group_start},{group_start + GROUP_UNION - 1}] "
                f"is outside T={frame_count}")
        physical_time = window_starts[:, None] + np.arange(GROUP_WINDOW)[None, :]
        context_indices = np.stack(
            (observation_anchors - 1, observation_anchors), axis=1)
        annotation = pq.read_table(
            annotation_path,
            columns=[
                "boundary_any_hard",
                "subtask_idx",
                "stage_local_id",
                "semantic_skill_local_id",
            ],
        ).to_pydict()
        boundary = np.asarray(annotation["boundary_any_hard"], dtype=bool)
        subtask = np.asarray(annotation["subtask_idx"], dtype=np.int64)
        stage = np.asarray(annotation["stage_local_id"], dtype=np.int64)
        skill = np.asarray(annotation["semantic_skill_local_id"], dtype=np.int64)
        if not (
            len(boundary) == len(subtask) == len(stage) == len(skill) == frame_count
        ):
            raise ValueError("annotation/cache frame counts do not match")

        edge_endpoint = physical_time[:, :-1] + 1
        observation_features = np.asarray(
            features[context_indices, feature_variant]
        )
        return CachedGroup(
            episode_id=episode_id,
            group_start=group_start,
            observation_anchors=observation_anchors.astype(np.int64),
            physical_time=physical_time.astype(np.int64),
            observation_features=observation_features,
            context_vector=observation_features.reshape(GROUP_WINDOWS, -1),
            clean_actions=np.asarray(actions[observation_anchors]),
            boundary_edge_target=boundary[edge_endpoint],
            subtask_idx=subtask[physical_time],
            stage_local_id=stage[physical_time],
            semantic_skill_local_id=skill[physical_time],
        )

    @lru_cache(maxsize=4)
    def _load_history_episode(
            self,
            episode_id: int,
            feature_variant: int,
        ) -> CachedHistory:
        """Load one writable full-episode history per DataLoader worker."""
        if feature_variant not in range(5):
            raise ValueError("feature_variant must be in [0,4]")
        annotation_path, feature_path, action_path = self._paths(episode_id)
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
        frame_count = int(features.shape[0])
        if actions.shape[0] != frame_count:
            raise ValueError("feature/action frame counts do not match")
        subtask = pq.read_table(
            annotation_path, columns=["subtask_idx"]
        )["subtask_idx"].to_numpy(zero_copy_only=False)
        if len(subtask) != frame_count:
            raise ValueError("annotation/cache frame counts do not match")
        previous_action = np.zeros(
            (frame_count, actions.shape[-1]), dtype=np.float32)
        if frame_count > 1:
            previous_action[1:] = np.asarray(actions[1:, 0])
        return CachedHistory(
            feature=np.asarray(features[:, feature_variant]).copy(),
            previous_action=previous_action,
            subtask_idx=np.asarray(subtask, dtype=np.int64).copy(),
            prefix_length=frame_count,
            episode_length=frame_count,
        )

    def load_sequential_episode(
            self, episode_id: int, *, feature_variant: int
        ) -> dict[str, np.ndarray | int]:
        """Load one episode for a strictly forward, one-edge-at-a-time probe."""
        annotation_path, feature_path, action_path = self._paths(episode_id)
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        actions = np.load(action_path, mmap_mode="r", allow_pickle=False)
        frame_count = int(features.shape[0])
        if features.shape != (frame_count, 5, 969):
            raise ValueError("invalid sequential-probe feature cache shape")
        if actions.shape != (frame_count, GROUP_WINDOW, 12):
            raise ValueError("invalid sequential-probe action cache shape")
        annotation = pq.read_table(
            annotation_path, columns=["boundary_any_hard", "subtask_idx"]
        ).to_pydict()
        boundary = np.asarray(annotation["boundary_any_hard"], dtype=np.float32)
        subtask = np.asarray(annotation["subtask_idx"], dtype=np.int64)
        if len(boundary) != frame_count or len(subtask) != frame_count:
            raise ValueError("annotation/cache frame counts do not match")
        history = self._load_history_episode(episode_id, feature_variant)
        return {
            "episode_id": episode_id,
            "episode_length": frame_count,
            "feature": np.asarray(features[:, feature_variant]).copy(),
            # Row r=t+1 is the cached candidate window beginning at action t.
            "action_candidate": np.asarray(actions[1:]).copy(),
            "previous_action": history.previous_action.copy(),
            "subtask_idx": subtask.copy(),
            "boundary_target": boundary.copy(),
        }

    def load_history(
            self,
            episode_id: int,
            prefix_length: int,
            *,
            feature_variant: int,
        ) -> CachedHistory:
        """Load the causal prefix needed by all window starts in one group.

        At physical time ``t`` the history input uses feature ``f[t]`` and the
        previous expert action ``a[t-1]``. Cached action row ``r=t`` starts at
        physical action ``t-1``, so ``actions[t, 0]`` is exactly that previous
        action. Time zero receives the specified all-zero previous action.
        """
        episode = self._load_history_episode(episode_id, feature_variant)
        frame_count = episode.episode_length
        if not 1 <= prefix_length <= frame_count:
            raise IndexError(
                f"history prefix {prefix_length} is outside episode length {frame_count}"
            )
        return CachedHistory(
            feature=episode.feature[:prefix_length],
            previous_action=episode.previous_action[:prefix_length],
            subtask_idx=episode.subtask_idx[:prefix_length],
            prefix_length=prefix_length,
            episode_length=frame_count,
        )


def shared_physical_time_noise(
        physical_time: np.ndarray,
        *,
        action_dim: int,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    """Create one Gaussian vector per physical time and gather overlap views."""
    physical_time = np.asarray(physical_time, dtype=np.int64)
    if physical_time.ndim != 2:
        raise ValueError("physical_time must have shape [K,W]")
    _, inverse = np.unique(physical_time, return_inverse=True)
    unique_count = int(np.max(inverse)) + 1
    unique_noise = torch.randn(
        (unique_count, action_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    gather_index = torch.as_tensor(inverse, dtype=torch.long, device=device)
    return unique_noise[gather_index].reshape(
        *physical_time.shape, action_dim)


class P1MVPCachedGroupDataset(Dataset):
    """Map a :class:`GroupSampleIndex` to four overlapping cached windows."""

    def __init__(
            self,
            split_artifact: str | Path,
            *,
            split: str,
            feature_variant: int = 0,
            include_history: bool = True,
            task: str = "GetToastedBread",
            annotation_root: str | Path | None = None,
            cache_root: str | Path | None = None,
            canonical_sha256: str | None = None,
        ):
        super().__init__()
        if feature_variant not in range(5):
            raise ValueError("feature_variant must be in [0,4]")
        self.store = CachedEpisodeStore(
            split_artifact,
            split=split,
            task=task,
            annotation_root=annotation_root,
            cache_root=cache_root,
            canonical_sha256=canonical_sha256,
        )
        self.feature_variant = feature_variant
        self.include_history = bool(include_history)

    def __len__(self) -> int:
        return len(self.store.episode_ids)

    def episode_length(self, episode_id: int) -> int:
        return self.store._load_history_episode(
            episode_id, self.feature_variant).episode_length

    def __getitem__(self, index: GroupSampleIndex) -> dict[str, Any]:
        if not isinstance(index, GroupSampleIndex):
            raise TypeError("P1MVPCachedGroupDataset expects GroupSampleIndex values")
        group = self.store.load_group(
            index.episode_id,
            index.group_start,
            feature_variant=self.feature_variant,
        )
        result = {
            "episode_id": torch.tensor(group.episode_id, dtype=torch.long),
            "group_start": torch.tensor(group.group_start, dtype=torch.long),
            "category": index.category,
            "episode_valid": torch.tensor(index.valid, dtype=torch.bool),
            "physical_time": torch.from_numpy(group.physical_time),
            "observation_anchor": torch.from_numpy(group.observation_anchors),
            "obs_feature": torch.from_numpy(group.observation_features.copy()),
            "context_vector": torch.from_numpy(group.context_vector.copy()),
            "action": torch.from_numpy(group.clean_actions.copy()),
            "boundary_edge_target": torch.from_numpy(
                group.boundary_edge_target.copy()),
            "subtask_idx": torch.from_numpy(group.subtask_idx.copy()),
            "stage_local_id": torch.from_numpy(group.stage_local_id.copy()),
            "semantic_skill_local_id": torch.from_numpy(
                group.semantic_skill_local_id.copy()),
            "episode_length": torch.tensor(
                int(group.physical_time.max()) + 1, dtype=torch.long),
        }
        if self.include_history:
            prefix_length = int(group.physical_time[-1, 0]) + 1
            history = self.store.load_history(
                index.episode_id,
                prefix_length,
                feature_variant=self.feature_variant,
            )
            result.update({
                "history_feature": torch.from_numpy(history.feature),
                "history_previous_action": torch.from_numpy(
                    history.previous_action),
                "history_subtask_idx": torch.from_numpy(history.subtask_idx),
                "history_length": torch.tensor(
                    history.prefix_length, dtype=torch.long),
                "episode_length": torch.tensor(
                    history.episode_length, dtype=torch.long),
            })
        return result


class P1MVPRawGroupDataset(Dataset):
    """Primary raw-observation wrapper using the same physical group indices.

    ``base_dataset`` must be one GetToastedBread ``LerobotDataset`` configured
    with ``horizon=10`` and ``n_obs_steps=2``. A local index ``r`` in that dataset
    produces observation context ``[o_{r-1}, o_r]`` and action window
    ``[a_{r-1}, ..., a_{r+8}]``. Therefore the group window whose physical action
    start is ``s`` is fetched at ``r=s+1``.
    """

    def __init__(
            self,
            base_dataset: Dataset,
            split_artifact: str | Path,
            *,
            split: str,
            history_feature_variant: int = 0,
            include_history: bool = True,
            task: str = "GetToastedBread",
            annotation_root: str | Path | None = None,
            cache_root: str | Path | None = None,
            canonical_sha256: str | None = None,
        ):
        super().__init__()
        artifact_path = Path(split_artifact)
        contract = resolve_p1_mvp_split(
            artifact_path,
            split=split,
            task=task,
            annotation_root=annotation_root,
            cache_root=cache_root,
            canonical_sha256=canonical_sha256,
        )
        if getattr(base_dataset, "n_obs_steps", None) != 2:
            raise ValueError("P1 raw group dataset requires n_obs_steps=2")

        self.base_dataset = base_dataset
        self.artifact_path = artifact_path
        self.episode_ids = contract.episode_ids
        self.annotation_root = contract.annotation_root
        self.include_history = bool(include_history)
        self.history_store = (
            CachedEpisodeStore(
                artifact_path,
                split=split,
                task=task,
                annotation_root=annotation_root,
                cache_root=cache_root,
                canonical_sha256=canonical_sha256,
            )
            if self.include_history else None
        )
        if history_feature_variant not in range(5):
            raise ValueError("history_feature_variant must be in [0,4]")
        self.history_feature_variant = history_feature_variant
        available_ids = getattr(base_dataset, "trajectory_ids", None)
        if available_ids is None:
            raise TypeError("base_dataset must expose trajectory_ids")
        missing = sorted(set(self.episode_ids) - set(int(x) for x in available_ids))
        if missing:
            raise ValueError(f"base dataset is missing locked episodes: {missing[:10]}")

    def __len__(self) -> int:
        return len(self.episode_ids)

    def episode_length(self, episode_id: int) -> int:
        return len(self._load_annotation(episode_id)["frame_index"])

    def _global_index(self, episode_id: int, observation_anchor: int) -> int:
        trajectory_index = self.base_dataset.get_trajectory_index(episode_id)
        start_indices = getattr(self.base_dataset, "start_indices", None)
        if start_indices is None:
            raise TypeError("base_dataset must expose start_indices")
        global_index = int(start_indices[trajectory_index] + observation_anchor)
        all_steps = getattr(self.base_dataset, "all_steps", None)
        if all_steps is not None:
            observed_episode, observed_anchor = all_steps[global_index]
            if (int(observed_episode), int(observed_anchor)) != (
                episode_id, observation_anchor
            ):
                raise AssertionError("base dataset episode/local-index mapping drifted")
        return global_index

    @lru_cache(maxsize=8)
    def _load_annotation(self, episode_id: int) -> dict[str, np.ndarray]:
        path = self.annotation_root / f"episode_{episode_id:06d}.parquet"
        columns = [
            "frame_index",
            "subtask_idx",
            "stage_local_id",
            "semantic_skill_local_id",
            "boundary_any_hard",
        ]
        table = pq.read_table(path, columns=columns)
        result = {
            name: table[name].to_numpy(zero_copy_only=False) for name in columns
        }
        if not np.array_equal(result["frame_index"], np.arange(table.num_rows)):
            raise ValueError(f"invalid annotation frame_index: {path}")
        return result

    def __getitem__(self, index: GroupSampleIndex) -> dict[str, Any]:
        if not isinstance(index, GroupSampleIndex):
            raise TypeError("P1MVPRawGroupDataset expects GroupSampleIndex values")
        if index.episode_id not in self.episode_ids:
            raise KeyError(f"episode {index.episode_id} is not in this split")

        window_starts = index.group_start + np.arange(GROUP_WINDOWS) * GROUP_STRIDE
        observation_anchors = window_starts + 1
        samples = [
            self.base_dataset[
                self._global_index(index.episode_id, int(observation_anchor))
            ]
            for observation_anchor in observation_anchors
        ]
        obs_keys = tuple(samples[0]["obs"])
        if any(tuple(sample["obs"]) != obs_keys for sample in samples[1:]):
            raise ValueError("raw group windows have inconsistent observation keys")
        observation = {
            key: torch.stack([sample["obs"][key] for sample in samples])
            for key in obs_keys
        }
        clean_actions = torch.stack([sample["action"] for sample in samples])
        if clean_actions.shape[-2:] != (GROUP_WINDOW, 12):
            raise ValueError(
                f"raw action group must end in {(GROUP_WINDOW, 12)}, "
                f"got {clean_actions.shape}")

        physical_time = window_starts[:, None] + np.arange(GROUP_WINDOW)[None, :]
        annotation = self._load_annotation(index.episode_id)
        frame_count = len(annotation["frame_index"])
        if int(physical_time.max()) >= frame_count:
            raise IndexError(
                f"group [{index.group_start},{index.group_start + GROUP_UNION - 1}] "
                f"is outside T={frame_count}")
        edge_endpoint = physical_time[:, :-1] + 1
        result = {
            "episode_id": torch.tensor(index.episode_id, dtype=torch.long),
            "group_start": torch.tensor(index.group_start, dtype=torch.long),
            "category": index.category,
            "episode_valid": torch.tensor(index.valid, dtype=torch.bool),
            "physical_time": torch.from_numpy(physical_time.astype(np.int64)),
            "observation_anchor": torch.from_numpy(
                observation_anchors.astype(np.int64)),
            "obs": observation,
            "action": clean_actions,
            "boundary_edge_target": torch.from_numpy(
                np.asarray(annotation["boundary_any_hard"])[edge_endpoint].copy()),
            "subtask_idx": torch.from_numpy(
                np.asarray(annotation["subtask_idx"])[physical_time].copy()),
            "stage_local_id": torch.from_numpy(
                np.asarray(annotation["stage_local_id"])[physical_time].copy()),
            "semantic_skill_local_id": torch.from_numpy(
                np.asarray(annotation["semantic_skill_local_id"])[
                    physical_time].copy()),
            "episode_length": torch.tensor(frame_count, dtype=torch.long),
        }
        if self.include_history:
            if self.history_store is None:
                raise AssertionError("history store was not initialized")
            prefix_length = int(window_starts[-1]) + 1
            history = self.history_store.load_history(
                index.episode_id,
                prefix_length,
                feature_variant=self.history_feature_variant,
            )
            result.update({
                "history_feature": torch.from_numpy(history.feature),
                "history_previous_action": torch.from_numpy(
                    history.previous_action),
                "history_subtask_idx": torch.from_numpy(history.subtask_idx),
                "history_length": torch.tensor(
                    history.prefix_length, dtype=torch.long),
            })
        return result

    def get_normalizer(self, **kwargs):
        return self.base_dataset.get_normalizer(**kwargs)


_HISTORY_KEYS = (
    "history_feature",
    "history_previous_action",
    "history_subtask_idx",
)


def p1_group_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate groups while encoding each episode history only once.

    A training batch contains eight groups per episode. Their causal histories
    are nested prefixes of the same episode, so running the history model on all
    eight copies is redundant. Keep the longest prefix for each episode and
    return ``group_to_history`` so every group can query that shared encoding.
    """
    if not batch:
        raise ValueError("cannot collate an empty P1 batch")
    has_history = [all(key in item for key in _HISTORY_KEYS) for item in batch]
    if not any(has_history):
        return default_collate(batch)
    if not all(has_history):
        raise ValueError("a P1 batch cannot mix items with and without history")
    for item in batch:
        missing = [key for key in _HISTORY_KEYS if key not in item]
        if missing:
            raise KeyError(f"P1 item is missing history fields: {missing}")
        lengths = {int(item[key].shape[0]) for key in _HISTORY_KEYS}
        if lengths != {int(item["history_length"].item())}:
            raise ValueError("history tensors and history_length disagree")

    episode_to_history: dict[int, int] = {}
    history_items: list[dict[str, Any]] = []
    group_to_history: list[int] = []
    for item in batch:
        episode_id = int(item["episode_id"].item())
        history_index = episode_to_history.get(episode_id)
        if history_index is None:
            history_index = len(history_items)
            episode_to_history[episode_id] = history_index
            history_items.append(item)
        elif int(item["history_length"].item()) > int(
                history_items[history_index]["history_length"].item()):
            history_items[history_index] = item
        group_to_history.append(history_index)

    fixed = [{
        key: value for key, value in item.items()
        if key not in _HISTORY_KEYS and key != "history_length"
    } for item in batch]
    result = default_collate(fixed)
    for key in _HISTORY_KEYS:
        result[key] = pad_sequence(
            [item[key] for item in history_items], batch_first=True)
    result["history_episode_id"] = torch.stack([
        item["episode_id"] for item in history_items
    ])
    result["history_length"] = torch.stack([
        item["history_length"] for item in history_items
    ])
    result["group_to_history"] = torch.tensor(
        group_to_history, dtype=torch.long)
    max_length = result["history_feature"].shape[1]
    result["history_valid_mask"] = (
        torch.arange(max_length)[None, :] < result["history_length"][:, None]
    )
    return result
