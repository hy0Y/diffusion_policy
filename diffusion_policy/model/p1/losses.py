"""Group-level P1 losses with real-edge aggregation across overlap views."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from diffusion_policy.model.p1.group import P1GroupOutput


@dataclass(frozen=True)
class EdgeBalancedBCEOutput:
    loss: Tensor
    edge_intensity: Tensor
    edge_probability: Tensor
    edge_target: Tensor
    edge_episode: Tensor
    edge_time: Tensor
    positive_count: int
    negative_count: int
    global_positive_count: int
    global_negative_count: int
    metric_update_valid: bool


@dataclass(frozen=True)
class P1GroupLossOutput:
    total: Tensor
    diffusion: Tensor
    boundary: Tensor
    overlap: Tensor
    initialization: Tensor
    edge: EdgeBalancedBCEOutput


def _edge_balanced_loss_from_sums(
    *,
    per_edge: Tensor,
    positive_sum: Tensor,
    negative_sum: Tensor,
    positive_count: int,
    negative_count: int,
    ddp_local_group_count: Tensor | None,
) -> tuple[Tensor, int, int]:
    """Return the rank contribution for one globally balanced DDP BCE."""
    if ddp_local_group_count is None:
        if positive_count and negative_count:
            loss = (
                0.5 * positive_sum / positive_count
                + 0.5 * negative_sum / negative_count
            )
        elif positive_count:
            loss = positive_sum / positive_count
        elif negative_count:
            loss = negative_sum / negative_count
        else:
            loss = per_edge.sum() * 0
        return loss, positive_count, negative_count

    local_group_count = torch.as_tensor(
        ddp_local_group_count,
        device=per_edge.device,
        dtype=torch.float64,
    )
    if local_group_count.numel() != 1:
        raise ValueError("ddp_local_group_count must be a scalar")
    if not bool(torch.isfinite(local_group_count)) \
            or bool(local_group_count < 0):
        raise ValueError(
            "ddp_local_group_count must be finite and non-negative")
    if per_edge.numel() and not bool(local_group_count > 0):
        raise ValueError("valid real edges require a positive local group count")

    summary = torch.stack((
        per_edge.new_tensor(float(positive_count), dtype=torch.float64),
        per_edge.new_tensor(float(negative_count), dtype=torch.float64),
        local_group_count,
    ))
    global_summary = summary.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_summary, op=dist.ReduceOp.SUM)
    global_positive_count = int(global_summary[0].item())
    global_negative_count = int(global_summary[1].item())
    global_group_count = global_summary[2]

    if global_positive_count and global_negative_count:
        local_class_contribution = (
            0.5 * positive_sum / global_positive_count
            + 0.5 * negative_sum / global_negative_count
        )
    elif global_positive_count:
        local_class_contribution = positive_sum / global_positive_count
    elif global_negative_count:
        local_class_contribution = negative_sum / global_negative_count
    else:
        local_class_contribution = per_edge.sum() * 0

    if bool(local_group_count > 0):
        scale = (global_group_count / local_group_count).to(
            dtype=local_class_contribution.dtype)
        loss = local_class_contribution * scale
    else:
        # Every rank must enter the collective even when it has only dummy
        # groups. Keep a differentiable zero so DDP sees the same graph.
        loss = (positive_sum + negative_sum) * 0
    return loss, global_positive_count, global_negative_count


def edge_balanced_bce_from_intensity_views(
    *,
    intensity: Tensor,
    target: Tensor,
    episode_id: Tensor,
    edge_time: Tensor,
    valid_mask: Tensor | None = None,
    delta_seconds: float = 0.05,
    epsilon: float = 1e-6,
    ddp_local_group_count: Tensor | None = None,
) -> EdgeBalancedBCEOutput:
    """Average duplicate views, then balance real edges by global DDP class.

    ``ddp_local_group_count`` enables the training workspace contract where DDP
    later weights the complete local loss by the rank's number of real groups.
    This function all-reduces class and group counts and returns a local boundary
    contribution that cancels that outer weighting. DDP's gradient mean then
    equals a single process over the union of all rank-local real edges.
    """
    tensors = (intensity, target, episode_id, edge_time)
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("intensity, target, episode_id, and edge_time must be 1-D")
    if len({value.numel() for value in tensors}) != 1:
        raise ValueError("all edge-view inputs must have identical lengths")
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    elif valid_mask.shape != target.shape:
        raise ValueError("valid_mask must have the same shape as target")
    valid_mask = valid_mask.to(dtype=torch.bool)
    if bool(torch.any(intensity < 0)):
        raise ValueError("intensity must be non-negative")
    if bool(torch.any((target[valid_mask] != 0) & (target[valid_mask] != 1))):
        raise ValueError("valid targets must be binary")

    selected_intensity = intensity[valid_mask]
    selected_target = target[valid_mask].to(dtype=intensity.dtype)
    selected_episode = episode_id[valid_mask].to(dtype=torch.int64)
    selected_time = edge_time[valid_mask].to(dtype=torch.int64)
    keys = torch.stack((selected_episode, selected_time), dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    edge_count = unique_keys.shape[0]
    summed_intensity = selected_intensity.new_zeros((edge_count,))
    summed_intensity.scatter_add_(0, inverse, selected_intensity)
    view_count = selected_intensity.new_zeros((edge_count,))
    view_count.scatter_add_(0, inverse, torch.ones_like(selected_intensity))
    mean_intensity = summed_intensity / view_count

    summed_target = selected_intensity.new_zeros((edge_count,))
    summed_target.scatter_add_(0, inverse, selected_target)
    if bool(torch.any((summed_target != 0) & (summed_target != view_count))):
        raise ValueError("overlap views of the same real edge have inconsistent targets")
    edge_target = (summed_target > 0).to(dtype=selected_intensity.dtype)
    probability = -torch.expm1(-mean_intensity * delta_seconds)
    probability = probability.clamp(min=epsilon, max=1 - epsilon)
    per_edge = -(
        edge_target * torch.log(probability)
        + (1 - edge_target) * torch.log1p(-probability)
    )
    positive = edge_target == 1
    negative = ~positive
    positive_count = int(positive.sum().item())
    negative_count = int(negative.sum().item())
    positive_sum = per_edge[positive].sum()
    negative_sum = per_edge[negative].sum()
    loss, global_positive_count, global_negative_count = (
        _edge_balanced_loss_from_sums(
            per_edge=per_edge,
            positive_sum=positive_sum,
            negative_sum=negative_sum,
            positive_count=positive_count,
            negative_count=negative_count,
            ddp_local_group_count=ddp_local_group_count,
        ))

    return EdgeBalancedBCEOutput(
        loss=loss,
        edge_intensity=mean_intensity,
        edge_probability=probability,
        edge_target=edge_target,
        edge_episode=unique_keys[:, 0],
        edge_time=unique_keys[:, 1],
        positive_count=positive_count,
        negative_count=negative_count,
        global_positive_count=global_positive_count,
        global_negative_count=global_negative_count,
        metric_update_valid=edge_count > 0,
    )


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=torch.bool)
    if value.shape[: mask.ndim] != mask.shape:
        raise ValueError("mask must match the leading value dimensions")
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    if not bool(torch.any(expanded)):
        return value.sum() * 0
    return value[expanded].mean()


def compute_p1_group_losses(
    *,
    prediction: Tensor,
    diffusion_target: Tensor,
    output: P1GroupOutput,
    boundary_target: Tensor,
    episode_id: Tensor,
    physical_time: Tensor,
    group_valid: Tensor,
    stride: int = 5,
    beta_boundary: float = 0.10,
    beta_overlap: float = 0.05,
    beta_initialization: float = 0.10,
    delta_seconds: float = 0.05,
) -> P1GroupLossOutput:
    if prediction.shape != diffusion_target.shape:
        raise ValueError("prediction and diffusion_target shapes must match")
    if prediction.ndim != 4:
        raise ValueError("diffusion tensors must have shape [B,K,W,D]")
    batch, windows, window, _ = prediction.shape
    if group_valid.shape != (batch,) or episode_id.shape != (batch,):
        raise ValueError("group_valid and episode_id must have shape [B]")
    if boundary_target.shape != (batch, windows, window - 1):
        raise ValueError("boundary_target shape does not match the group")
    if physical_time.shape != (batch, windows, window):
        raise ValueError("physical_time shape does not match the group")

    diffusion_per_group = (prediction - diffusion_target).square().mean(
        dim=(1, 2, 3))
    diffusion = _masked_mean(diffusion_per_group, group_valid)

    edge_valid = group_valid[:, None, None].expand_as(boundary_target)
    edge_episode = episode_id[:, None, None].expand_as(boundary_target)
    # ``physical_time[..., u]`` is the start of edge u -> u+1.  The probe
    # contract indexes boundary belief by its endpoint t, matching
    # ``boundary_target[..., u] == boundary_any_hard[t]``.
    edge = edge_balanced_bce_from_intensity_views(
        intensity=output.intensity.reshape(-1),
        target=boundary_target.reshape(-1),
        episode_id=edge_episode.reshape(-1),
        edge_time=(physical_time[:, :, :-1] + 1).reshape(-1),
        valid_mask=edge_valid.reshape(-1),
        delta_seconds=delta_seconds,
        ddp_local_group_count=group_valid.sum(dtype=torch.float64),
    )

    if not 0 < stride < window:
        raise ValueError("stride must lie strictly inside the window")
    overlap_error = (
        output.z[:, :-1, stride:] - output.z[:, 1:, : window - stride]
    ).square()
    overlap_mask = group_valid[:, None, None].expand(
        batch, windows - 1, window - stride)
    overlap = _masked_mean(overlap_error, overlap_mask)

    init_error = (
        output.carry_start[:, 1:] - output.history_start[:, 1:]
    ).square()
    init_mask = group_valid[:, None] & output.carry_history_valid[:, 1:]
    initialization = _masked_mean(init_error, init_mask)
    total = (
        diffusion
        + beta_boundary * edge.loss
        + beta_overlap * overlap
        + beta_initialization * initialization
    )
    return P1GroupLossOutput(
        total=total,
        diffusion=diffusion,
        boundary=edge.loss,
        overlap=overlap,
        initialization=initialization,
        edge=edge,
    )
