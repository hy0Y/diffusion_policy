"""Torch-only causal history providers for the Jump initial state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalHistoryConfig:
    feature_dim: int
    action_dim: int
    model_dim: int = 128
    latent_dim: int = 32
    num_layers: int = 2
    backend: str = "selective_ssm"
    state_dim: int = 16
    conv_width: int = 4
    expand: int = 2


class _SelectiveStateSpaceLayer(nn.Module):
    """Small causal selective SSM implemented entirely with PyTorch ops."""

    def __init__(
            self,
            model_dim: int,
            *,
            state_dim: int,
            conv_width: int,
            expand: int,
        ):
        super().__init__()
        if min(state_dim, conv_width, expand) <= 0:
            raise ValueError("selective SSM dimensions must be positive")
        inner_dim = model_dim * expand
        dt_rank = max(1, math.ceil(model_dim / 16))
        self.inner_dim = inner_dim
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(model_dim)
        self.in_proj = nn.Linear(model_dim, 2 * inner_dim)
        self.conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_width,
            padding=conv_width - 1,
            groups=inner_dim,
        )
        self.x_proj = nn.Linear(inner_dim, dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner_dim)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
            .repeat(inner_dim, 1))
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, model_dim)

    def forward(self, hidden: Tensor, valid_mask: Tensor) -> Tensor:
        residual = hidden
        projected, gate = self.in_proj(self.norm(hidden)).chunk(2, dim=-1)
        projected = projected * valid_mask.unsqueeze(-1).to(projected.dtype)
        convolved = self.conv(projected.transpose(1, 2))
        convolved = convolved[..., :hidden.shape[1]].transpose(1, 2)
        convolved = F.silu(convolved)
        parameters = self.x_proj(convolved)
        dt_rank = parameters.shape[-1] - 2 * self.state_dim
        dt, B, C = torch.split(
            parameters, (dt_rank, self.state_dim, self.state_dim), dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log).to(dtype=hidden.dtype)

        state = hidden.new_zeros(
            hidden.shape[0], self.inner_dim, self.state_dim)
        values = []
        for index in range(hidden.shape[1]):
            step_dt = dt[:, index]
            transition = torch.exp(step_dt.unsqueeze(-1) * A)
            candidate_state = (
                transition * state
                + step_dt.unsqueeze(-1)
                * B[:, index].unsqueeze(1)
                * convolved[:, index].unsqueeze(-1)
            )
            valid = valid_mask[:, index, None, None]
            state = torch.where(valid, candidate_state, state)
            value = (
                (state * C[:, index].unsqueeze(1)).sum(dim=-1)
                + self.D.to(hidden.dtype) * convolved[:, index]
            )
            value = value * valid_mask[:, index, None].to(value.dtype)
            values.append(value)
        scanned = torch.stack(values, dim=1)
        update = self.out_proj(scanned * F.silu(gate))
        hidden = residual + update
        return hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)


class CausalHistoryEncoder(nn.Module):
    """Map ``(feature[t], executed_action[t-1])`` prefixes to latent states."""

    def __init__(self, config: CausalHistoryConfig):
        super().__init__()
        self.config = config
        cfg = config
        self.input_projection = nn.Linear(
            cfg.feature_dim + cfg.action_dim, cfg.model_dim)
        if cfg.backend == "selective_ssm":
            self.layers = nn.ModuleList([
                _SelectiveStateSpaceLayer(
                    cfg.model_dim,
                    state_dim=cfg.state_dim,
                    conv_width=cfg.conv_width,
                    expand=cfg.expand,
                )
                for _ in range(cfg.num_layers)
            ])
            self.gru = None
        elif cfg.backend == "gru":
            self.layers = nn.ModuleList()
            self.gru = nn.GRU(
                cfg.model_dim,
                cfg.model_dim,
                num_layers=cfg.num_layers,
                batch_first=True,
            )
        else:
            raise ValueError("history backend must be selective_ssm or gru")
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.history_projection = nn.Linear(cfg.model_dim, cfg.latent_dim)

    def forward(
            self,
            *,
            feature: Tensor,
            previous_action: Tensor,
            valid_mask: Tensor,
        ) -> Tensor:
        cfg = self.config
        if feature.ndim != 3 or feature.shape[-1] != cfg.feature_dim:
            raise ValueError("feature must have shape [B,L,F]")
        if previous_action.shape != (
                feature.shape[0], feature.shape[1], cfg.action_dim):
            raise ValueError("previous_action must have shape [B,L,D_action]")
        if valid_mask.shape != feature.shape[:2]:
            raise ValueError("valid_mask must have shape [B,L]")
        valid_mask = valid_mask.to(dtype=torch.bool)
        if bool(torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1])):
            raise ValueError("valid history positions must form a causal prefix")
        model_input = torch.cat((feature, previous_action), dim=-1)
        model_input = model_input * valid_mask.unsqueeze(-1).to(model_input.dtype)
        hidden = self.input_projection(model_input)
        hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)
        if self.gru is not None:
            hidden, _ = self.gru(hidden)
            hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)
        else:
            for layer in self.layers:
                hidden = layer(hidden, valid_mask)
        latent = self.history_projection(self.output_norm(hidden))
        return latent * valid_mask.unsqueeze(-1).to(latent.dtype)

    @staticmethod
    def last_valid(latent: Tensor, valid_mask: Tensor) -> Tensor:
        if latent.ndim != 3 or valid_mask.shape != latent.shape[:2]:
            raise ValueError("latent [B,L,D] and valid_mask [B,L] are required")
        lengths = valid_mask.to(dtype=torch.long).sum(dim=1)
        if bool(torch.any(lengths < 1)):
            raise ValueError("every history must contain at least one valid step")
        index = (lengths - 1)[:, None, None].expand(-1, 1, latent.shape[-1])
        return torch.gather(latent, dim=1, index=index).squeeze(1)
