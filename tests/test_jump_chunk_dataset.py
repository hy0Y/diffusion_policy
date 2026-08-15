import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from diffusion_policy.dataset.jump_mvp_chunk_dataset import (
    ChunkSampleIndex,
    JumpMVPChunkDataset,
    JumpMVPCachedEpisodeStore,
    jump_chunk_collate,
)


def make_artifacts(tmp_path):
    annotation = tmp_path / "annotation"
    cache = tmp_path / "cache"
    annotation.mkdir()
    cache.mkdir()
    episode_id = 1
    frames = 64
    horizon = 10
    features = np.zeros((frames, 2, 6), dtype=np.float32)
    for time in range(frames):
        features[time, 0] = time
        features[time, 1] = 1000 + time
    actions = np.zeros((frames, horizon, 3), dtype=np.float32)
    for row in range(frames):
        for token in range(horizon):
            physical_time = row - 1 + token
            if 0 <= physical_time < frames:
                actions[row, token] = (physical_time + 1) * 100
    stem = f"episode_{episode_id:06d}"
    np.save(cache / f"{stem}_features.npy", features)
    np.save(cache / f"{stem}_actions.npy", actions)

    boundary = np.zeros(frames, dtype=bool)
    boundary[[5, 25, 45]] = True
    subtask = np.zeros(frames, dtype=np.int64)
    subtask[5:25] = 1
    subtask[25:45] = 2
    subtask[45:] = 3
    pq.write_table(pa.table({
        "frame_index": np.arange(frames, dtype=np.int64),
        "subtask_idx": subtask,
        "stage_text": ["active"] * frames,
        "boundary_any_hard": boundary,
        "valid_mask": np.ones(frames, dtype=bool),
    }), annotation / f"{stem}.parquet")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({
        "split_id": "gettoastedbread_mvp_intersection_v1",
        "source": {
            "annotation_root": str(annotation),
            "cache_root": str(cache),
        },
        "episodes": {
            "train": [episode_id],
            "validation": [episode_id],
            "test": [episode_id],
        },
    }), encoding="utf-8")
    return manifest, annotation, cache


def test_chunk_alignment_and_causal_history(tmp_path):
    manifest, annotation, cache = make_artifacts(tmp_path)
    store = JumpMVPCachedEpisodeStore(
        manifest,
        split="train",
        horizon=10,
        n_obs_steps=2,
        feature_variant=0,
        annotation_root=annotation,
        cache_root=cache,
    )
    item = store.load_chunk(ChunkSampleIndex(1, 3, "natural"))
    torch.testing.assert_close(item["obs_feature"][:, 0], torch.tensor([3.0, 4.0]))
    assert item["action"][0, 0].item() == 400
    torch.testing.assert_close(
        item["history_feature"][:, 0], torch.arange(5, dtype=torch.float32))
    assert item["history_previous_action"][0].abs().sum().item() == 0
    assert item["history_previous_action"][4, 0].item() == 400
    episode = store.load_episode(1)
    np.testing.assert_array_equal(item["action"].numpy(), episode.action[4])
    np.testing.assert_array_equal(
        item["boundary_target"].numpy(), episode.boundary[5:13])
    assert item["action"].ndim == 2
    assert item["boundary_target"].shape == (8,)
    assert item["edge_valid_mask"].all()

    episode_start = store.load_chunk(
        ChunkSampleIndex(1, -1, "episode_start"))
    torch.testing.assert_close(
        episode_start["obs_feature"][:, 0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(
        episode_start["history_feature"][:, 0], torch.tensor([0.0]))
    assert episode_start["action"][0].abs().sum().item() == 0
    assert episode_start["action"][1, 0].item() == 100
    np.testing.assert_array_equal(
        episode_start["action"].numpy(), episode.action[0])
    assert episode_start["action_valid_mask"].tolist() == (
        [False] + [True] * 9)


def test_model_observation_count_is_independent_of_cache_action_alignment(
        tmp_path):
    manifest, annotation, cache = make_artifacts(tmp_path)
    store = JumpMVPCachedEpisodeStore(
        manifest,
        split="train",
        horizon=10,
        n_obs_steps=3,
        cache_action_n_obs_steps=2,
        feature_variant=0,
        annotation_root=annotation,
        cache_root=cache,
    )
    item = store.load_chunk(ChunkSampleIndex(1, 3, "natural"))

    # Model condition: f[base_time:base_time+n_obs_steps].
    torch.testing.assert_close(
        item["obs_feature"][:, 0], torch.tensor([3.0, 4.0, 5.0]))
    # Physical action a[3] is reconstructed from the immutable cache contract.
    # This does not depend on model n_obs.
    assert item["action"][0, 0].item() == 400
    # initial_z is the current physical-time state and therefore includes the
    # newest condition observation f[base_time+n_obs_steps-1].
    torch.testing.assert_close(
        item["history_feature"][:, 0], torch.arange(6, dtype=torch.float32))
    assert item["history_previous_action"][5, 0].item() == 500
    assert item["boundary_target"].shape == (7,)


def test_wrong_cache_action_observation_contract_fails_closed(tmp_path):
    manifest, annotation, cache = make_artifacts(tmp_path)
    store = JumpMVPCachedEpisodeStore(
        manifest,
        split="train",
        horizon=10,
        n_obs_steps=2,
        cache_action_n_obs_steps=3,
        annotation_root=annotation,
        cache_root=cache,
    )
    with pytest.raises(ValueError, match="cache_action_n_obs_steps"):
        store.load_episode(1)


def test_category_schedule_is_independent_single_chunks(tmp_path):
    manifest, annotation, cache = make_artifacts(tmp_path)
    dataset = JumpMVPChunkDataset(
        manifest,
        split="train",
        horizon=10,
        n_obs_steps=2,
        edge_dt_seconds=0.05,
        seed=42,
        annotation_root=annotation,
        cache_root=cache,
    )
    assert len(dataset) == 58
    categories = [index.category for index in dataset.sample_indices]
    assert categories.count("boundary_centered") == 16
    assert categories.count("near_negative") == 16
    assert categories.count("interior") == 16
    assert categories.count("natural") == 8
    assert categories.count("episode_start") == 2
    assert all(not hasattr(index, "group_start") for index in dataset.sample_indices)
    assert dataset[0]["action"].shape == (10, 3)
    assert torch.all(dataset[0]["edge_dt_seconds"] == 0.05)


def test_collate_pads_each_history_without_episode_dedup(tmp_path):
    manifest, annotation, cache = make_artifacts(tmp_path)
    store = JumpMVPCachedEpisodeStore(
        manifest,
        split="train",
        horizon=10,
        n_obs_steps=2,
        annotation_root=annotation,
        cache_root=cache,
    )
    left = store.load_chunk(ChunkSampleIndex(1, 0, "episode_start"))
    right = store.load_chunk(ChunkSampleIndex(1, 12, "natural"))
    left["edge_dt_seconds"].fill_(0.05)
    right["edge_dt_seconds"].fill_(0.05)
    batch = jump_chunk_collate([left, right])
    assert batch["history_feature"].shape[:2] == (2, 14)
    assert batch["history_valid_mask"][0].sum().item() == 2
    assert batch["history_valid_mask"][1].sum().item() == 14
    assert "group_to_history" not in batch
