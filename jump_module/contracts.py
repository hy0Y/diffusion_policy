"""Public, host-neutral tensor contract for the Jump adapter."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class JumpControls:
    """Independent intervention scales used by attribution experiments."""

    feedback_scale: float | Tensor = 1.0
    jump_scale: float | Tensor = 1.0
    flow_scale: float | Tensor = 1.0


@dataclass(frozen=True)
class JumpAdapterInput:
    action_hidden: Tensor
    initial_z: Tensor
    condition_context: Tensor | None
    noise_level: Tensor
    edge_dt_seconds: Tensor
    edge_valid_mask: Tensor
    boundary_target: Tensor | None
    teacher_ratio: float | Tensor
    controls: JumpControls = JumpControls()


@dataclass(frozen=True)
class JumpAdapterOutput:
    action_hidden: Tensor
    z: Tensor
    z_minus: Tensor
    flow_delta: Tensor
    raw_jump_delta: Tensor
    applied_jump_delta: Tensor
    intensity: Tensor
    probability: Tensor
    gate: Tensor
    feedback_residual: Tensor
