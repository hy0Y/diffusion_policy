import numpy as np

from diffusion_policy.dataset.p1_mvp_sampler import (
    CATEGORY_QUOTA,
    EpisodeBundleBatchSampler,
    EpisodeBundleSampler,
    EpisodePools,
    intervals_overlap,
    partition_epoch_for_ddp,
)


def make_pools(episode_id: int) -> EpisodePools:
    boundary = np.array([40, 41, 42, 140, 141, 142, 240, 241, 242])
    near = np.array([0, 1, 2, 90, 91, 92, 190, 191, 192])
    interior = np.arange(0, 400, dtype=np.int64)
    return EpisodePools(
        episode_id=episode_id,
        boundary=boundary,
        terminal_boundary=np.array([240, 241, 242]),
        near=near,
        interior=interior,
        interior_segment=np.where(interior < 200, 0, 1),
        natural_nonterminal=np.arange(0, 400, dtype=np.int64),
        natural_with_terminal=np.arange(0, 425, dtype=np.int64),
        natural_touches_done=np.arange(300, 425, dtype=np.int64),
        episode_start=np.array([400], dtype=np.int64),
    )


class FakeStore:
    def __init__(self, episode_count: int):
        self.episode_ids = tuple(range(episode_count))

    def episode(self, episode_id: int) -> EpisodePools:
        return make_pools(episode_id)


def test_episode_bundle_is_deterministic_disjoint_and_quota_complete():
    sampler = EpisodeBundleSampler(
        FakeStore(1), seed=123, natural_done_policy="include_terminal")
    first = sampler.sample_episode(0, epoch=4)
    second = sampler.sample_episode(0, epoch=4)

    assert first == second
    observed = {category: 0 for category in CATEGORY_QUOTA}
    for group in first.groups:
        observed[group.category] += 1
    assert observed == CATEGORY_QUOTA
    for index, left in enumerate(first.groups):
        for right in first.groups[index + 1:]:
            assert not intervals_overlap(left.group_start, right.group_start)


def test_epoch_is_episode_complete_and_final_batch_is_not_dropped():
    sampler = EpisodeBundleSampler(
        FakeStore(10),
        seed=42,
        natural_done_policy="include_terminal",
        episodes_per_batch=4,
    )
    schedule = sampler.sample_epoch(epoch=3)

    assert schedule.episode_count == 10
    assert schedule.group_count == 80
    assert schedule.window_count == 320
    assert [len(batch) for batch in schedule.batches] == [4, 4, 2]
    batch_sampler = EpisodeBundleBatchSampler(schedule)
    assert [len(batch) for batch in batch_sampler] == [32, 32, 16]


def test_resume_batch_cursor_preserves_remaining_schedule():
    sampler = EpisodeBundleSampler(
        FakeStore(10), seed=42, natural_done_policy="include_terminal")
    schedule = sampler.sample_epoch(epoch=3)
    full = list(EpisodeBundleBatchSampler(schedule))
    resumed = list(EpisodeBundleBatchSampler(schedule, start_batch=1))
    assert resumed == full[1:]


def test_ddp_final_step_uses_masked_dummy_slots_without_repeating_real_data():
    sampler = EpisodeBundleSampler(
        FakeStore(10), seed=42, natural_done_policy="include_terminal")
    schedule = sampler.sample_epoch(epoch=3)
    ranks = [
        partition_epoch_for_ddp(
            schedule,
            rank=rank,
            world_size=3,
            episodes_per_rank=4,
        )
        for rank in range(3)
    ]

    assert all(len(rank_schedule.batches) == 1 for rank_schedule in ranks)
    assert [rank.real_episode_count for rank in ranks] == [4, 4, 2]
    assert [rank.dummy_episode_count for rank in ranks] == [0, 0, 2]
    real_episode_ids = []
    for rank_schedule in ranks:
        batch = rank_schedule.batches[0]
        mask = rank_schedule.valid_episode_mask[0]
        assert len(batch) == len(mask) == 4
        flat_groups = list(EpisodeBundleBatchSampler(rank_schedule))[0]
        assert len(flat_groups) == 32
        for bundle, valid in zip(batch, mask):
            assert all(group.valid is valid for group in bundle.groups)
            if valid:
                real_episode_ids.append(bundle.episode_id)
    assert len(real_episode_ids) == 10
    assert len(set(real_episode_ids)) == 10
