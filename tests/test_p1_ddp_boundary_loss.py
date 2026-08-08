from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from diffusion_policy.model.p1.losses import (
    edge_balanced_bce_from_intensity_views,
)


_OFFSETS = (-0.8, -0.5, -0.2, 0.1, 0.4, 0.7, 1.0, -0.7, -0.3, 0.2, 0.6, 0.9)
_TARGETS = (1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0)
_PARTITIONS = {
    1: ((0, 12),),
    2: ((0, 4), (4, 12)),
    # Rank 1 contains only dummy groups. Rank 0 has only a positive edge.
    4: ((0, 1), (1, 1), (1, 8), (8, 12)),
}


def _edge_inputs(theta, start, end):
    offsets = theta.new_tensor(_OFFSETS[start:end])
    intensity = F.softplus(theta + offsets)
    target = theta.new_tensor(_TARGETS[start:end])
    episode_id = torch.arange(start, end, dtype=torch.int64)
    edge_time = torch.arange(start + 1, end + 1, dtype=torch.int64)
    return intensity, target, episode_id, edge_time


def _ddp_boundary_worker(rank, world_size, init_path, result_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        theta = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
        start, end = _PARTITIONS[world_size][rank]
        intensity, target, episode_id, edge_time = _edge_inputs(
            theta, start, end)
        local_groups = theta.new_tensor(float(end - start))
        result = edge_balanced_bce_from_intensity_views(
            intensity=intensity,
            target=target,
            episode_id=episode_id,
            edge_time=edge_time,
            ddp_local_group_count=local_groups,
        )

        global_groups = local_groups.detach().clone()
        dist.all_reduce(global_groups, op=dist.ReduceOp.SUM)
        outer_scale = local_groups * world_size / global_groups
        scaled_loss = result.loss * outer_scale
        scaled_loss.backward()

        gradient = theta.grad.detach().clone()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient /= world_size
        global_loss = scaled_loss.detach().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        global_loss /= world_size

        if rank == 0:
            torch.save({
                "loss": global_loss,
                "gradient": gradient,
                "positive_count": result.global_positive_count,
                "negative_count": result.global_negative_count,
            }, result_path)
    finally:
        dist.destroy_process_group()


def _single_process_reference():
    theta = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    intensity, target, episode_id, edge_time = _edge_inputs(
        theta, 0, len(_TARGETS))
    result = edge_balanced_bce_from_intensity_views(
        intensity=intensity,
        target=target,
        episode_id=episode_id,
        edge_time=edge_time,
    )
    result.loss.backward()
    return result.loss.detach(), theta.grad.detach()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_global_boundary_bce_loss_and_gradient_match_single_process(
        tmp_path, world_size):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    expected_loss, expected_gradient = _single_process_reference()
    init_path = tmp_path / f"gloo_init_{world_size}"
    result_path = tmp_path / f"result_{world_size}.pt"
    mp.spawn(
        _ddp_boundary_worker,
        args=(world_size, str(init_path), str(result_path)),
        nprocs=world_size,
        join=True,
    )
    observed = torch.load(result_path, weights_only=True)

    torch.testing.assert_close(observed["loss"], expected_loss)
    torch.testing.assert_close(observed["gradient"], expected_gradient)
    assert observed["positive_count"] == sum(_TARGETS)
    assert observed["negative_count"] == len(_TARGETS) - sum(_TARGETS)
