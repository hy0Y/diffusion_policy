"""Metrics defined on the committed physical-step clock."""

from __future__ import annotations

from collections.abc import Sequence


def ordered_prefix_depth(
    ordered_ids: Sequence[str], completed_ids: set[str] | frozenset[str]
) -> int:
    depth = 0
    for milestone_id in ordered_ids:
        if milestone_id not in completed_ids:
            break
        depth += 1
    return depth


def fixed_horizon_prefix_auc(
    prefix_depths: Sequence[int], *, required_count: int, fixed_horizon: int
) -> float:
    """Area under normalized prefix progress with zero-padded remaining time."""
    if required_count <= 0:
        raise ValueError("required_count must be positive")
    if fixed_horizon <= 0:
        raise ValueError("fixed_horizon must be positive")
    if len(prefix_depths) > fixed_horizon:
        raise ValueError("prefix trace exceeds the fixed physical horizon")
    if any(depth < 0 or depth > required_count for depth in prefix_depths):
        raise ValueError("prefix depth is outside the valid range")
    return float(sum(prefix_depths)) / float(required_count * fixed_horizon)


def executed_horizon_prefix_auc(
    prefix_depths: Sequence[int], *, required_count: int
) -> float:
    if required_count <= 0:
        raise ValueError("required_count must be positive")
    if not prefix_depths:
        return 0.0
    return float(sum(prefix_depths)) / float(required_count * len(prefix_depths))
