"""Masked single-chunk boundary objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class BoundaryLossOutput:
    loss: Tensor
    probability: Tensor
    target: Tensor
    positive_count: int
    negative_count: int
    global_positive_count: int
    global_negative_count: int


def masked_balanced_boundary_loss(
        *,
        intensity: Tensor,
        target: Tensor,
        edge_dt_seconds: Tensor,
        valid_mask: Tensor,
        epsilon: float = 1e-6,
    ) -> BoundaryLossOutput:
    """Class-balanced BCE over valid edge views, without physical-edge dedup."""
    if intensity.shape != target.shape or intensity.shape != valid_mask.shape:
        raise ValueError("intensity, target, and valid_mask shapes must match")
    valid = valid_mask.to(dtype=torch.bool)
    if edge_dt_seconds.ndim == 0:
        dt = edge_dt_seconds.expand_as(intensity)
    elif edge_dt_seconds.shape == intensity.shape:
        dt = edge_dt_seconds
    elif edge_dt_seconds.shape == (intensity.shape[0], 1):
        dt = edge_dt_seconds.expand_as(intensity)
    else:
        raise ValueError("edge_dt_seconds must be scalar, [B,1], or [B,T-1]")

    selected_intensity = intensity[valid]
    selected_target = target[valid].to(dtype=intensity.dtype)
    selected_dt = dt[valid].to(dtype=intensity.dtype)
    if bool(torch.any(selected_intensity < 0)):
        raise ValueError("intensity must be non-negative")
    if bool(torch.any((selected_target != 0) & (selected_target != 1))):
        raise ValueError("valid boundary targets must be binary")
    probability = -torch.expm1(-selected_intensity * selected_dt)
    probability = probability.clamp(min=epsilon, max=1 - epsilon)
    per_edge = -(
        selected_target * torch.log(probability)
        + (1 - selected_target) * torch.log1p(-probability)
    )
    positive = selected_target == 1
    negative = ~positive
    positive_count = int(positive.sum().item())
    negative_count = int(negative.sum().item())
    positive_sum = per_edge[positive].sum()
    negative_sum = per_edge[negative].sum()

    counts = intensity.new_tensor(
        [positive_count, negative_count], dtype=torch.float64)
    global_counts = counts.clone()
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
    global_positive = int(global_counts[0].item())
    global_negative = int(global_counts[1].item())
    world_size = dist.get_world_size() if distributed else 1
    loss = per_edge.sum() * 0
    if global_positive:
        loss = loss + positive_sum * (0.5 if global_negative else 1.0) \
            * world_size / global_positive
    if global_negative:
        loss = loss + negative_sum * (0.5 if global_positive else 1.0) \
            * world_size / global_negative

    return BoundaryLossOutput(
        loss=loss,
        probability=probability,
        target=selected_target,
        positive_count=positive_count,
        negative_count=negative_count,
        global_positive_count=global_positive,
        global_negative_count=global_negative,
    )
