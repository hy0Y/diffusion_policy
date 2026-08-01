"""Runtime data contract for the GetToastedBread P1 MVP.

The primary experiment will read raw observations. This cached-feature dataset is
kept as a contract/parity/profile path and deliberately requires artifact paths to
be supplied by the experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


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
    stage_local_id: np.ndarray
    semantic_skill_local_id: np.ndarray


class CachedEpisodeStore:
    """Read locked P1 cache and annotation files without mutating them."""

    def __init__(self, split_artifact: str | Path, *, split: str):
        artifact_path = Path(split_artifact)
        manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if manifest.get("split_id") != "gettoastedbread_mvp_intersection_v1":
            raise ValueError("unexpected P1 MVP split_id")
        self.artifact_path = artifact_path
        self.split_id = manifest["split_id"]
        self.episode_ids = tuple(
            int(value) for value in manifest["episodes"][split]
        )
        self.annotation_root = Path(manifest["source"]["annotation_root"])
        self.cache_root = Path(manifest["source"]["cache_root"])

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
                "stage_local_id",
                "semantic_skill_local_id",
            ],
        ).to_pydict()
        boundary = np.asarray(annotation["boundary_any_hard"], dtype=bool)
        stage = np.asarray(annotation["stage_local_id"], dtype=np.int64)
        skill = np.asarray(annotation["semantic_skill_local_id"], dtype=np.int64)
        if not (len(boundary) == len(stage) == len(skill) == frame_count):
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
            stage_local_id=stage[physical_time],
            semantic_skill_local_id=skill[physical_time],
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
        ):
        super().__init__()
        if feature_variant not in range(5):
            raise ValueError("feature_variant must be in [0,4]")
        self.store = CachedEpisodeStore(split_artifact, split=split)
        self.feature_variant = feature_variant

    def __len__(self) -> int:
        return len(self.store.episode_ids)

    def __getitem__(self, index: GroupSampleIndex) -> dict[str, Any]:
        if not isinstance(index, GroupSampleIndex):
            raise TypeError("P1MVPCachedGroupDataset expects GroupSampleIndex values")
        group = self.store.load_group(
            index.episode_id,
            index.group_start,
            feature_variant=self.feature_variant,
        )
        return {
            "episode_id": torch.tensor(group.episode_id, dtype=torch.long),
            "group_start": torch.tensor(group.group_start, dtype=torch.long),
            "category": index.category,
            "physical_time": torch.from_numpy(group.physical_time),
            "observation_anchor": torch.from_numpy(group.observation_anchors),
            "obs_feature": torch.from_numpy(group.observation_features.copy()),
            "context_vector": torch.from_numpy(group.context_vector.copy()),
            "action": torch.from_numpy(group.clean_actions.copy()),
            "boundary_edge_target": torch.from_numpy(
                group.boundary_edge_target.copy()),
            "stage_local_id": torch.from_numpy(group.stage_local_id.copy()),
            "semantic_skill_local_id": torch.from_numpy(
                group.semantic_skill_local_id.copy()),
        }
