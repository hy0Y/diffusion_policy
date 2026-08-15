"""Right-continuous Flow/Jump latent dynamics with no host dependency."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jump_module.contracts import JumpControls


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class JumpDynamicsConfig:
    latent_dim: int = 32
    flow_condition_dim: int = 64
    jump_condition_dim: int = 64
    hidden_dim: int = 128


@dataclass(frozen=True)
class JumpDynamicsOutput:
    z: Tensor
    z_minus: Tensor
    flow_delta: Tensor
    raw_jump_delta: Tensor
    applied_jump_delta: Tensor
    intensity: Tensor
    probability: Tensor
    gate: Tensor


def _expand_edge_value(
        value: Tensor | float,
        *,
        batch: int,
        edges: int,
        reference: Tensor,
        name: str,
    ) -> Tensor:
    tensor = torch.as_tensor(
        value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor.expand(batch, edges)
    if tensor.shape == (batch,):
        return tensor[:, None].expand(batch, edges)
    if tensor.shape == (batch, 1):
        return tensor.expand(batch, edges)
    if tensor.shape == (batch, edges):
        return tensor
    raise ValueError(
        f"{name} must be scalar, [B], [B,1], or [B,T-1]; got {tensor.shape}")


class JumpDynamicsCore(nn.Module):
    """Scan ``Flow -> pre-Jump intensity -> gated Jump`` over one chunk."""

    def __init__(self, config: JumpDynamicsConfig | None = None):
        super().__init__()
        self.config = config or JumpDynamicsConfig()
        cfg = self.config
        self.flow = _mlp(
            cfg.latent_dim + cfg.flow_condition_dim,
            cfg.hidden_dim,
            cfg.latent_dim,
        )
        self.jump = _mlp(
            cfg.latent_dim + cfg.jump_condition_dim,
            cfg.hidden_dim,
            cfg.latent_dim,
        )
        self.intensity_head = _mlp(
            cfg.latent_dim + cfg.jump_condition_dim,
            cfg.hidden_dim,
            1,
        )

    def forward(
            self,
            *,
            initial_z: Tensor,
            flow_condition: Tensor,
            jump_condition: Tensor,
            edge_dt_seconds: Tensor,
            edge_valid_mask: Tensor,
            boundary_target: Tensor | None = None,
            teacher_ratio: float | Tensor = 0.0,
            controls: JumpControls = JumpControls(),
        ) -> JumpDynamicsOutput:
        cfg = self.config
        if flow_condition.ndim != 3:
            raise ValueError("flow_condition must have shape [B,T-1,C_flow]")
        batch, edges, flow_dim = flow_condition.shape
        if edges < 1:
            raise ValueError("a Jump chunk must contain at least one edge")
        if flow_dim != cfg.flow_condition_dim:
            raise ValueError("flow condition dimension does not match config")
        if jump_condition.shape != (batch, edges, cfg.jump_condition_dim):
            raise ValueError("jump_condition shape does not match config")
        if initial_z.shape != (batch, cfg.latent_dim):
            raise ValueError("initial_z must have shape [B,D_z]")
        if edge_valid_mask.shape != (batch, edges):
            raise ValueError("edge_valid_mask must have shape [B,T-1]")
        valid_mask = edge_valid_mask.to(dtype=torch.bool)

        dt = _expand_edge_value(
            edge_dt_seconds,
            batch=batch,
            edges=edges,
            reference=initial_z,
            name="edge_dt_seconds",
        )
        if bool(torch.any(dt[valid_mask] <= 0)):
            raise ValueError("valid edges require positive edge_dt_seconds")
        ratio = _expand_edge_value(
            teacher_ratio,
            batch=batch,
            edges=edges,
            reference=initial_z,
            name="teacher_ratio",
        )
        if bool(torch.any((ratio < 0) | (ratio > 1))):
            raise ValueError("teacher_ratio must lie in [0,1]")
        if boundary_target is not None and boundary_target.shape != (batch, edges):
            raise ValueError("boundary_target must have shape [B,T-1]")
        if boundary_target is None and bool(torch.any(ratio != 0)):
            raise ValueError(
                "boundary_target is required when teacher_ratio is nonzero")

        flow_scale = _expand_edge_value(
            controls.flow_scale,
            batch=batch,
            edges=edges,
            reference=initial_z,
            name="controls.flow_scale",
        )
        jump_scale = _expand_edge_value(
            controls.jump_scale,
            batch=batch,
            edges=edges,
            reference=initial_z,
            name="controls.jump_scale",
        )

        current = initial_z
        stored = [current]
        z_minus_values = []
        flow_delta_values = []
        raw_jump_values = []
        applied_jump_values = []
        intensity_values = []
        probability_values = []
        gate_values = []

        for edge in range(edges):
            valid = valid_mask[:, edge]
            valid_float = valid.to(dtype=initial_z.dtype)
            flow_input = torch.cat(
                (current, flow_condition[:, edge]), dim=-1)
            raw_flow_delta = (
                dt[:, edge, None] * self.flow(flow_input))
            flow_delta = (
                raw_flow_delta
                * flow_scale[:, edge, None]
                * valid_float[:, None]
            )
            z_minus = current + flow_delta

            jump_input = torch.cat(
                (z_minus, jump_condition[:, edge]), dim=-1)
            intensity = F.softplus(
                self.intensity_head(jump_input).squeeze(-1))
            probability = -torch.expm1(-intensity * dt[:, edge].clamp_min(0))
            edge_ratio = ratio[:, edge]
            if boundary_target is None or not bool(torch.any(edge_ratio != 0)):
                gate = probability
            else:
                # Invalid targets may contain arbitrary sentinels. They are
                # replaced before any arithmetic, so invalid edges are unread.
                target = torch.where(
                    valid,
                    boundary_target[:, edge].to(dtype=probability.dtype),
                    torch.zeros_like(probability),
                )
                gate = edge_ratio * target + (1 - edge_ratio) * probability
            gate = gate * valid_float

            raw_jump_delta = self.jump(jump_input)
            applied_jump_delta = (
                raw_jump_delta
                * gate[:, None]
                * jump_scale[:, edge, None]
            )
            current = z_minus + applied_jump_delta

            stored.append(current)
            z_minus_values.append(z_minus)
            flow_delta_values.append(flow_delta)
            raw_jump_values.append(raw_jump_delta)
            applied_jump_values.append(applied_jump_delta)
            intensity_values.append(intensity)
            probability_values.append(probability)
            gate_values.append(gate)

        return JumpDynamicsOutput(
            z=torch.stack(stored, dim=1),
            z_minus=torch.stack(z_minus_values, dim=1),
            flow_delta=torch.stack(flow_delta_values, dim=1),
            raw_jump_delta=torch.stack(raw_jump_values, dim=1),
            applied_jump_delta=torch.stack(applied_jump_values, dim=1),
            intensity=torch.stack(intensity_values, dim=1),
            probability=torch.stack(probability_values, dim=1),
            gate=torch.stack(gate_values, dim=1),
        )
