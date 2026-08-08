"""One-window latent Flow / Jump dynamics for P1.

The stored state is right-continuous.  Every edge therefore runs

    z[t+] -> Flow -> z[(t+1)-] -> intensity/gate/Jump -> z[(t+1)+].

The oracle symbol is only visible to the two condition projectors.  It is not
concatenated directly into Flow, Jump, the intensity head, or feedback decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class P1DynamicsConfig:
    latent_dim: int = 32
    dit_hidden_dim: int = 512
    context_dim: int = 1938
    oracle_dim: int = 32
    flow_condition_dim: int = 64
    jump_condition_dim: int = 64
    projector_hidden_dim: int = 256
    dynamics_hidden_dim: int = 128
    feedback_hidden_dim: int = 256
    delta_seconds: float = 0.05

    @property
    def projector_input_dim(self) -> int:
        # h, Cvec, normalized diffusion tau, normalized endpoint time, oracle w_s.
        return (
            self.dit_hidden_dim
            + self.context_dim
            + 1
            + 1
            + self.oracle_dim
        )


@dataclass(frozen=True)
class DynamicsOutput:
    # Stored right-continuous states z_t, including the input state at token 0.
    z: Tensor
    # Per-edge pre-Jump states z^-_{t+1}.
    z_minus: Tensor
    flow_delta: Tensor
    jump_delta: Tensor
    intensity: Tensor
    probability: Tensor
    gate: Tensor
    kappa_flow: Tensor
    kappa_jump: Tensor


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class LatentFeedbackDecoder(nn.Module):
    """Zero-initialized latent-to-hidden feedback decoder A_fb: z -> DiT h."""

    def __init__(self, config: P1DynamicsConfig):
        super().__init__()
        self.network = _mlp(
            config.latent_dim,
            config.feedback_hidden_dim,
            config.dit_hidden_dim,
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("feedback decoder must end in nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, latent: Tensor) -> Tensor:
        return self.network(latent)

    def add_to_hidden(self, hidden: Tensor, latent: Tensor) -> Tensor:
        if hidden.shape[:-1] != latent.shape[:-1]:
            raise ValueError("hidden and latent leading dimensions must match")
        return hidden + self(latent)


class LatentDynamics(nn.Module):
    """Euler Flow -> pre-Jump intensity/gate -> gated Jump scan for one window."""

    def __init__(self, config: P1DynamicsConfig | None = None):
        super().__init__()
        self.config = config or P1DynamicsConfig()
        cfg = self.config
        self.flow_projector = _mlp(
            cfg.projector_input_dim,
            cfg.projector_hidden_dim,
            cfg.flow_condition_dim,
        )
        self.jump_projector = _mlp(
            cfg.projector_input_dim,
            cfg.projector_hidden_dim,
            cfg.jump_condition_dim,
        )
        self.flow = _mlp(
            cfg.latent_dim + cfg.flow_condition_dim,
            cfg.dynamics_hidden_dim,
            cfg.latent_dim,
        )
        self.jump = _mlp(
            cfg.latent_dim + cfg.jump_condition_dim,
            cfg.dynamics_hidden_dim,
            cfg.latent_dim,
        )
        self.intensity_head = _mlp(
            cfg.latent_dim + cfg.jump_condition_dim,
            cfg.dynamics_hidden_dim,
            1,
        )

    def _validate_inputs(
        self,
        *,
        initial_z: Tensor,
        hidden: Tensor,
        context: Tensor,
        normalized_tau: Tensor,
        normalized_next_time: Tensor,
        oracle: Tensor | None,
        boundary_target: Tensor | None,
        teacher_ratio: float | Tensor,
    ) -> tuple[int, int, Tensor, Tensor, Tensor]:
        cfg = self.config
        if hidden.ndim != 3 or hidden.shape[-1] != cfg.dit_hidden_dim:
            raise ValueError(
                f"hidden must have shape [B,W,{cfg.dit_hidden_dim}], got {hidden.shape}"
            )
        batch, window, _ = hidden.shape
        if window < 2:
            raise ValueError("a dynamics window must contain at least one edge")
        if initial_z.shape != (batch, cfg.latent_dim):
            raise ValueError(
                f"initial_z must have shape {(batch, cfg.latent_dim)}"
            )
        if context.shape != (batch, cfg.context_dim):
            raise ValueError(f"context must have shape {(batch, cfg.context_dim)}")
        if normalized_tau.shape not in ((batch,), (batch, 1)):
            raise ValueError("normalized_tau must have shape [B] or [B,1]")
        if normalized_next_time.shape != (batch, window - 1):
            raise ValueError("normalized_next_time must have shape [B,W-1]")
        if oracle is None:
            oracle = hidden.new_zeros((batch, cfg.oracle_dim))
        elif oracle.shape != (batch, cfg.oracle_dim):
            raise ValueError(f"oracle must have shape {(batch, cfg.oracle_dim)}")
        if boundary_target is not None and boundary_target.shape != (batch, window - 1):
            raise ValueError("boundary_target must have shape [B,W-1]")
        ratio = torch.as_tensor(
            teacher_ratio, dtype=hidden.dtype, device=hidden.device)
        if ratio.numel() != 1:
            raise ValueError("teacher_ratio is one global training-progress scalar")
        if bool(torch.any((ratio < 0) | (ratio > 1))):
            raise ValueError("teacher_ratio must lie in [0,1]")
        if boundary_target is None and bool(torch.any(ratio != 0)):
            raise ValueError("boundary_target is required when teacher_ratio is nonzero")
        return batch, window, oracle, normalized_tau.reshape(batch, 1), ratio

    def forward(
        self,
        *,
        initial_z: Tensor,
        hidden: Tensor,
        context: Tensor,
        normalized_tau: Tensor,
        normalized_next_time: Tensor,
        oracle: Tensor | None = None,
        boundary_target: Tensor | None = None,
        teacher_ratio: float | Tensor = 0.0,
        max_edges: int | None = None,
    ) -> DynamicsOutput:
        """Scan all edges by default, or a strict prefix for sequential inference."""
        _, window, oracle, tau, ratio = self._validate_inputs(
            initial_z=initial_z,
            hidden=hidden,
            context=context,
            normalized_tau=normalized_tau,
            normalized_next_time=normalized_next_time,
            oracle=oracle,
            boundary_target=boundary_target,
            teacher_ratio=teacher_ratio,
        )
        edge_count = window - 1 if max_edges is None else int(max_edges)
        if not 1 <= edge_count <= window - 1:
            raise ValueError("max_edges must lie in [1,W-1]")
        cfg = self.config
        current = initial_z
        stored = [current]
        z_minus_values: list[Tensor] = []
        flow_delta_values: list[Tensor] = []
        jump_delta_values: list[Tensor] = []
        intensity_values: list[Tensor] = []
        probability_values: list[Tensor] = []
        gate_values: list[Tensor] = []
        flow_condition_values: list[Tensor] = []
        jump_condition_values: list[Tensor] = []

        for edge in range(edge_count):
            projector_input = torch.cat(
                (
                    hidden[:, edge],
                    context,
                    tau,
                    normalized_next_time[:, edge : edge + 1],
                    oracle,
                ),
                dim=-1,
            )
            kappa_flow = self.flow_projector(projector_input)
            flow_velocity = self.flow(torch.cat((current, kappa_flow), dim=-1))
            flow_delta = cfg.delta_seconds * flow_velocity
            z_minus = current + flow_delta

            kappa_jump = self.jump_projector(projector_input)
            jump_input = torch.cat((z_minus, kappa_jump), dim=-1)
            intensity = F.softplus(self.intensity_head(jump_input).squeeze(-1))
            probability = -torch.expm1(-intensity * cfg.delta_seconds)
            if boundary_target is None:
                gate = probability
            else:
                target = boundary_target[:, edge].to(dtype=probability.dtype)
                gate = ratio * target + (1 - ratio) * probability
            jump_delta = self.jump(jump_input)
            current = z_minus + gate.unsqueeze(-1) * jump_delta

            stored.append(current)
            z_minus_values.append(z_minus)
            flow_delta_values.append(flow_delta)
            jump_delta_values.append(jump_delta)
            intensity_values.append(intensity)
            probability_values.append(probability)
            gate_values.append(gate)
            flow_condition_values.append(kappa_flow)
            jump_condition_values.append(kappa_jump)

        return DynamicsOutput(
            z=torch.stack(stored, dim=1),
            z_minus=torch.stack(z_minus_values, dim=1),
            flow_delta=torch.stack(flow_delta_values, dim=1),
            jump_delta=torch.stack(jump_delta_values, dim=1),
            intensity=torch.stack(intensity_values, dim=1),
            probability=torch.stack(probability_values, dim=1),
            gate=torch.stack(gate_values, dim=1),
            kappa_flow=torch.stack(flow_condition_values, dim=1),
            kappa_jump=torch.stack(jump_condition_values, dim=1),
        )
