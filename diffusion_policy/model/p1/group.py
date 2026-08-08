"""Sequential K-window P1 group forward with differentiable carry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn

from diffusion_policy.model.p1.dynamics import (
    DynamicsOutput,
    LatentDynamics,
    LatentFeedbackDecoder,
    P1DynamicsConfig,
)
from diffusion_policy.model.p1.history import (
    P1CausalHistoryEncoder,
    P1HistoryConfig,
)


class P1GateMode(str, Enum):
    ORACLE = "oracle"
    SCHEDULED = "scheduled"
    PREDICTED = "predicted"


@dataclass(frozen=True)
class P1GroupConfig:
    windows: int = 4
    window_length: int = 10
    stride: int = 5
    num_diffusion_steps: int = 100


@dataclass(frozen=True)
class P1GroupOutput:
    corrected_hidden: Tensor
    z: Tensor
    z_minus: Tensor
    flow_delta: Tensor
    jump_delta: Tensor
    intensity: Tensor
    probability: Tensor
    gate: Tensor
    history_start: Tensor
    carry_start: Tensor
    carry_history_valid: Tensor


class P1LatentGroupModel(nn.Module):
    """Run history once, then process overlapping windows in temporal order."""

    def __init__(
        self,
        *,
        dynamics_config: P1DynamicsConfig | None = None,
        history_config: P1HistoryConfig | None = None,
        group_config: P1GroupConfig | None = None,
    ):
        super().__init__()
        self.dynamics_config = dynamics_config or P1DynamicsConfig()
        self.history_config = history_config or P1HistoryConfig()
        self.group_config = group_config or P1GroupConfig()
        if self.dynamics_config.latent_dim != self.history_config.latent_dim:
            raise ValueError("history and dynamics latent dimensions must match")
        if self.dynamics_config.oracle_dim != self.history_config.oracle_dim:
            raise ValueError("history and dynamics oracle dimensions must match")
        self.history = P1CausalHistoryEncoder(self.history_config)
        self.dynamics = LatentDynamics(self.dynamics_config)
        self.feedback = LatentFeedbackDecoder(self.dynamics_config)

    @staticmethod
    def _stack(outputs: list[DynamicsOutput], field: str) -> Tensor:
        return torch.stack([getattr(output, field) for output in outputs], dim=1)

    def forward(
        self,
        *,
        hidden: Tensor,
        context: Tensor,
        diffusion_timestep: Tensor,
        physical_time: Tensor,
        episode_length: Tensor,
        history_feature: Tensor,
        history_previous_action: Tensor,
        history_valid_mask: Tensor,
        group_to_history: Tensor,
        history_symbol: Tensor | None,
        window_symbol: Tensor | None,
        boundary_target: Tensor,
        gate_mode: P1GateMode | str,
        teacher_ratio: float | Tensor,
        carry_ratio: float | Tensor,
    ) -> P1GroupOutput:
        cfg = self.group_config
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B,K,W,H]")
        batch, windows, window_length, _ = hidden.shape
        if (windows, window_length) != (cfg.windows, cfg.window_length):
            raise ValueError(
                f"group geometry must be K={cfg.windows}, W={cfg.window_length}"
            )
        if context.shape[:2] != (batch, windows):
            raise ValueError("context must have shape [B,K,C]")
        if physical_time.shape != (batch, windows, window_length):
            raise ValueError("physical_time must have shape [B,K,W]")
        if boundary_target.shape != (batch, windows, window_length - 1):
            raise ValueError("boundary_target must have shape [B,K,W-1]")
        if episode_length.shape != (batch,):
            raise ValueError("episode_length must have shape [B]")
        if diffusion_timestep.shape != (batch,):
            raise ValueError("diffusion_timestep must have shape [B]")
        expected_starts = physical_time[:, :1, 0] + torch.arange(
            windows, device=physical_time.device) * cfg.stride
        if not torch.equal(physical_time[:, :, 0], expected_starts):
            raise ValueError("window starts do not follow the configured stride")
        if window_symbol is not None and window_symbol.shape != (batch, windows):
            raise ValueError("window_symbol must have shape [B,K]")

        gate_mode = P1GateMode(gate_mode)
        if gate_mode is P1GateMode.ORACLE:
            ratio = hidden.new_tensor(1.0)
        elif gate_mode is P1GateMode.PREDICTED:
            ratio = hidden.new_tensor(0.0)
        else:
            ratio = torch.as_tensor(
                teacher_ratio, device=hidden.device, dtype=hidden.dtype)
        eta = torch.as_tensor(
            carry_ratio, device=hidden.device, dtype=hidden.dtype)
        if eta.numel() != 1 or bool(torch.any((eta < 0) | (eta > 1))):
            raise ValueError("carry_ratio must be one scalar in [0,1]")

        history_latent = self.history(
            feature=history_feature,
            previous_action=history_previous_action,
            valid_mask=history_valid_mask,
            oracle_symbol=history_symbol,
        )
        window_start = physical_time[:, :, 0]
        history_start = self.history.gather_groups_at_time(
            history_latent, group_to_history, window_start)
        if window_symbol is None:
            window_oracle = hidden.new_zeros(
                (batch, windows, self.history_config.oracle_dim))
        else:
            window_oracle = self.history.oracle_embedding(
                window_symbol.to(dtype=torch.long))

        tau_normalized = diffusion_timestep.to(hidden.dtype) / max(
            cfg.num_diffusion_steps - 1, 1)
        time_denominator = (episode_length - 1).clamp_min(1).to(hidden.dtype)
        outputs: list[DynamicsOutput] = []
        carry_values: list[Tensor] = []
        carry_valid: list[Tensor] = []
        previous: DynamicsOutput | None = None
        for window_index in range(windows):
            if previous is None:
                initial_z = history_start[:, window_index]
                carry_values.append(torch.zeros_like(initial_z))
                carry_valid.append(torch.zeros(batch, dtype=torch.bool, device=hidden.device))
            else:
                carry = previous.z[:, cfg.stride]
                initial_z = eta * carry + (1 - eta) * history_start[:, window_index]
                carry_values.append(carry)
                carry_valid.append(torch.ones(batch, dtype=torch.bool, device=hidden.device))
            normalized_next_time = (
                physical_time[:, window_index, 1:].to(hidden.dtype)
                / time_denominator[:, None]
            )
            output = self.dynamics(
                initial_z=initial_z,
                hidden=hidden[:, window_index],
                context=context[:, window_index],
                normalized_tau=tau_normalized,
                normalized_next_time=normalized_next_time,
                oracle=window_oracle[:, window_index],
                boundary_target=boundary_target[:, window_index],
                teacher_ratio=ratio,
            )
            outputs.append(output)
            previous = output

        z = self._stack(outputs, "z")
        corrected_hidden = hidden + self.feedback(z)
        return P1GroupOutput(
            corrected_hidden=corrected_hidden,
            z=z,
            z_minus=self._stack(outputs, "z_minus"),
            flow_delta=self._stack(outputs, "flow_delta"),
            jump_delta=self._stack(outputs, "jump_delta"),
            intensity=self._stack(outputs, "intensity"),
            probability=self._stack(outputs, "probability"),
            gate=self._stack(outputs, "gate"),
            history_start=history_start,
            carry_start=torch.stack(carry_values, dim=1),
            carry_history_valid=torch.stack(carry_valid, dim=1),
        )
