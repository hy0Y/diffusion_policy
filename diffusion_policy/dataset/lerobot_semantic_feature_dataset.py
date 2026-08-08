"""Semantic RoboCasa dataset backed by precomputed observation features."""

from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from typing import Dict

import numpy as np
import torch

from diffusion_policy.dataset.lerobot_semantic_dataset import (
    LerobotSemanticCotrainingDataset,
    LerobotSemanticDataset,
)


class LerobotSemanticFeatureDataset(LerobotSemanticDataset):
    """Loads cached frozen-encoder features and cached action windows.

    Cache variant 0 is the deterministic center-crop feature. Variants 1..K
    are deterministic random-crop features generated offline. Validation
    always uses variant 0; training rotates through variants 1..K.
    """

    def __init__(
        self,
        *args,
        feature_cache_root: str,
        random_feature_variants: int = 4,
        feature_cache_version: str = "v1",
        **kwargs,
    ):
        self.feature_seed = int(kwargs.get("seed", 42))
        super().__init__(*args, **kwargs)
        self.feature_cache_root = pathlib.Path(feature_cache_root)
        self.random_feature_variants = int(random_feature_variants)
        self.feature_cache_version = str(feature_cache_version)
        self.feature_split_role = "train"
        if self.random_feature_variants < 0:
            raise ValueError("random_feature_variants must be non-negative")

        metadata_path = self.feature_cache_root / "cache_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"feature cache metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format_version") != self.feature_cache_version:
            raise ValueError(
                "feature cache version mismatch: "
                f"expected={self.feature_cache_version}, "
                f"actual={metadata.get('format_version')}"
            )
        actual_variants = int(metadata.get("random_feature_variants", -1))
        if actual_variants != self.random_feature_variants:
            raise ValueError(
                "feature variant mismatch: "
                f"expected={self.random_feature_variants}, actual={actual_variants}"
            )
        self.feature_cache_metadata = metadata
        self._feature_task_dir = (
            self.feature_cache_root / self.semantic_split / self.semantic_task_name
        )
        if not self._feature_task_dir.is_dir():
            raise FileNotFoundError(
                f"feature task directory missing: {self._feature_task_dir}"
            )

    def set_split_role(self, split_role: str) -> None:
        if split_role not in {"train", "val"}:
            raise ValueError(f"invalid feature split role: {split_role}")
        self.feature_split_role = split_role

    @lru_cache(maxsize=16)
    def _load_feature_episode(
        self, episode_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        stem = f"episode_{int(episode_index):06d}"
        feature_path = self._feature_task_dir / f"{stem}_features.npy"
        action_path = self._feature_task_dir / f"{stem}_actions.npy"
        if not feature_path.is_file() or not action_path.is_file():
            raise FileNotFoundError(
                f"feature cache incomplete for {self.semantic_task_name}/{stem}"
            )
        features = np.load(feature_path, mmap_mode="r")
        actions = np.load(action_path, mmap_mode="r")
        expected_variants = 1 + self.random_feature_variants
        if features.ndim != 3 or features.shape[1] != expected_variants:
            raise ValueError(f"invalid cached feature shape: {feature_path} {features.shape}")
        if actions.ndim != 3:
            raise ValueError(f"invalid cached action shape: {action_path} {actions.shape}")
        if features.shape[0] != actions.shape[0]:
            raise ValueError(f"cache length mismatch for {stem}")
        return features, actions

    def _variant_for(self, dataset_index: int, observation_offset: int) -> int:
        if self.feature_split_role == "val" or self.random_feature_variants == 0:
            return 0
        mixed = (
            self.feature_seed * 1_000_003
            + int(self.epoch) * 97_409
            + int(dataset_index) * 65_537
            + int(observation_offset) * 257
        )
        return 1 + mixed % self.random_feature_variants

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory_id, base_index = self.all_steps[idx]
        trajectory_id, base_index = int(trajectory_id), int(base_index)
        features, actions = self._load_feature_episode(trajectory_id)

        observation_indices = np.arange(
            base_index - self.n_obs_steps + 1, base_index + 1, dtype=np.int64
        )
        observation_indices = np.clip(observation_indices, 0, features.shape[0] - 1)
        selected = [
            features[frame_index, self._variant_for(idx, offset)]
            for offset, frame_index in enumerate(observation_indices.tolist())
        ]
        obs_feature = np.stack(selected, axis=0).astype(np.float32, copy=False)
        action = np.asarray(actions[base_index], dtype=np.float32)

        sidecar = self._load_semantic_episode(trajectory_id)
        future_index = base_index + self.future_offset
        valid = bool(sidecar["future_h8_valid"][base_index])
        if valid:
            subtask = int(sidecar["subtask_local_id"][future_index])
            skill = int(sidecar["semantic_skill_local_id"][future_index])
            stage = int(sidecar["stage_local_id"][future_index])
        else:
            subtask = skill = stage = -100

        return {
            "obs_feature": torch.from_numpy(np.array(obs_feature, copy=True)),
            "action": torch.from_numpy(np.array(action, copy=True)),
            "semantic_target": {
                "subtask": torch.tensor(subtask, dtype=torch.long),
                "semantic_skill": torch.tensor(skill, dtype=torch.long),
                "stage": torch.tensor(stage, dtype=torch.long),
                "valid": torch.tensor(valid, dtype=torch.bool),
            },
            "boundary_target": {
                "subtask_future_h8": torch.tensor(
                    bool(sidecar["boundary_subtask_future_h8"][base_index]),
                    dtype=torch.float32,
                ),
                "stage_future_h8": torch.tensor(
                    bool(sidecar["boundary_stage_future_h8"][base_index]),
                    dtype=torch.float32,
                ),
                "any_future_h8": torch.tensor(
                    bool(sidecar["boundary_any_future_h8"][base_index]),
                    dtype=torch.float32,
                ),
                "valid": torch.tensor(valid, dtype=torch.bool),
            },
            "sample_index": {
                "episode_index": torch.tensor(trajectory_id, dtype=torch.long),
                "frame_index": torch.tensor(base_index, dtype=torch.long),
            },
        }


class LerobotSemanticFeatureCotrainingDataset(
    LerobotSemanticCotrainingDataset
):
    """Feature-cache variant of the composite semantic mixture dataset."""

    def __init__(
        self,
        *args,
        feature_cache_root: str,
        random_feature_variants: int = 4,
        feature_cache_version: str = "v1",
        **kwargs,
    ):
        super().__init__(
            *args,
            single_dataset_class=LerobotSemanticFeatureDataset,
            single_dataset_kwargs={
                "feature_cache_root": feature_cache_root,
                "random_feature_variants": random_feature_variants,
                "feature_cache_version": feature_cache_version,
            },
            **kwargs,
        )
