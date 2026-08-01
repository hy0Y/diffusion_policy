import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from diffusion_policy.dataset.p1_mvp_group_dataset import (
    CachedEpisodeStore,
    GroupSampleIndex,
    P1MVPCachedGroupDataset,
    P1MVPRawGroupDataset,
    shared_physical_time_noise,
)


def make_fixture(tmp_path):
    annotation_root = tmp_path / "annotations"
    cache_root = tmp_path / "cache"
    annotation_root.mkdir()
    cache_root.mkdir()
    frame_count = 40
    stem = "episode_000007"
    boundary = np.zeros(frame_count, dtype=bool)
    boundary[12] = True
    pq.write_table(
        pa.table({
            "frame_index": np.arange(frame_count, dtype=np.int64),
            "subtask_idx": np.arange(frame_count, dtype=np.int64) // 10,
            "boundary_any_hard": boundary,
            "stage_local_id": np.arange(frame_count, dtype=np.int64),
            "semantic_skill_local_id": np.arange(frame_count, dtype=np.int64) + 100,
        }),
        annotation_root / f"{stem}.parquet",
    )
    features = np.arange(frame_count * 5 * 969, dtype=np.float16).reshape(
        frame_count, 5, 969)
    actions = np.arange(frame_count * 10 * 12, dtype=np.float32).reshape(
        frame_count, 10, 12)
    np.save(cache_root / f"{stem}_features.npy", features)
    np.save(cache_root / f"{stem}_actions.npy", actions)
    artifact = tmp_path / "split.json"
    artifact.write_text(json.dumps({
        "split_id": "gettoastedbread_mvp_intersection_v1",
        "episodes": {"train": [7], "validation": [], "test": []},
        "source": {
            "annotation_root": str(annotation_root),
            "cache_root": str(cache_root),
        },
    }))
    return artifact, features, actions, boundary


class FakeRawDataset(torch.utils.data.Dataset):
    n_obs_steps = 2
    trajectory_ids = np.array([7], dtype=np.int64)
    start_indices = np.array([0], dtype=np.int64)
    all_steps = [(7, index) for index in range(40)]

    def __init__(self):
        self.requested = []

    def __len__(self):
        return 40

    def get_trajectory_index(self, episode_id):
        assert episode_id == 7
        return 0

    def __getitem__(self, index):
        self.requested.append(index)
        return {
            "obs": {
                "state": torch.tensor(
                    [[index - 1.0], [index]], dtype=torch.float32),
            },
            "action": torch.full((10, 12), float(index)),
        }

    def get_normalizer(self, **kwargs):
        return ("normalizer", kwargs)


def test_cached_group_uses_physical_time_contract(tmp_path):
    artifact, features, actions, boundary = make_fixture(tmp_path)
    store = CachedEpisodeStore(artifact, split="train")
    group = store.load_group(7, 0, feature_variant=2)

    assert group.observation_anchors.tolist() == [1, 6, 11, 16]
    assert group.physical_time.shape == (4, 10)
    assert group.observation_features.shape == (4, 2, 969)
    assert group.context_vector.shape == (4, 1938)
    assert group.clean_actions.shape == (4, 10, 12)
    np.testing.assert_array_equal(group.observation_features[0], features[[0, 1], 2])
    np.testing.assert_array_equal(group.clean_actions[0], actions[1])
    np.testing.assert_array_equal(group.boundary_edge_target[0], boundary[1:10])


def test_dataset_accepts_group_index(tmp_path):
    artifact, _, _, _ = make_fixture(tmp_path)
    dataset = P1MVPCachedGroupDataset(
        artifact, split="train", feature_variant=0)
    item = dataset[GroupSampleIndex(7, 0, "episode_start")]

    assert item["obs_feature"].shape == (4, 2, 969)
    assert item["action"].shape == (4, 10, 12)
    assert item["boundary_edge_target"].shape == (4, 9)
    assert item["category"] == "episode_start"
    assert item["episode_valid"].item()


def test_raw_dataset_maps_group_start_to_observation_anchor(tmp_path):
    artifact, _, _, boundary = make_fixture(tmp_path)
    base = FakeRawDataset()
    dataset = P1MVPRawGroupDataset(base, artifact, split="train")
    item = dataset[GroupSampleIndex(7, 0, "episode_start")]

    assert base.requested == [1, 6, 11, 16]
    assert item["obs"]["state"].shape == (4, 2, 1)
    assert item["action"].shape == (4, 10, 12)
    assert item["action"][:, 0, 0].tolist() == [1.0, 6.0, 11.0, 16.0]
    np.testing.assert_array_equal(
        item["boundary_edge_target"][0].numpy(), boundary[1:10])
    assert item["subtask_idx"].shape == (4, 10)
    assert item["episode_valid"].item()
    assert dataset.get_normalizer(probe=True) == ("normalizer", {"probe": True})


def test_overlap_uses_exact_same_noise():
    physical_time = np.stack([
        np.arange(start, start + 10) for start in (20, 25, 30, 35)
    ])
    generator = torch.Generator().manual_seed(123)
    noise = shared_physical_time_noise(
        physical_time, action_dim=12, generator=generator)

    assert noise.shape == (4, 10, 12)
    torch.testing.assert_close(noise[0, 5:], noise[1, :5], rtol=0, atol=0)
    torch.testing.assert_close(noise[1, 5:], noise[2, :5], rtol=0, atol=0)
    assert not torch.equal(noise[0, :5], noise[1, :5])
