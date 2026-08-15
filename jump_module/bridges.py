"""Dimension-configurable host condition and feedback bridges."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class HostConditionBridgeConfig:
    host_hidden_dim: int
    context_dim: int
    flow_condition_dim: int = 64
    jump_condition_dim: int = 64
    hidden_dim: int = 256


class HostConditionBridge(nn.Module):
    """Project host hidden/context/noise into Flow and Jump conditions."""

    def __init__(self, config: HostConditionBridgeConfig):
        super().__init__()
        self.config = config
        input_dim = config.host_hidden_dim + config.context_dim + 1
        self.flow_projector = _mlp(
            input_dim, config.hidden_dim, config.flow_condition_dim)
        self.jump_projector = _mlp(
            input_dim, config.hidden_dim, config.jump_condition_dim)

    def forward(
            self,
            action_hidden: Tensor,
            condition_context: Tensor | None,
            noise_level: Tensor,
        ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if action_hidden.ndim != 3 \
                or action_hidden.shape[-1] != cfg.host_hidden_dim:
            raise ValueError("action_hidden shape does not match bridge config")
        batch, tokens, _ = action_hidden.shape
        if tokens < 2:
            raise ValueError("action_hidden requires at least two tokens")
        if cfg.context_dim == 0:
            if condition_context is None:
                condition_context = action_hidden.new_empty((batch, 0))
        if condition_context is None \
                or condition_context.shape != (batch, cfg.context_dim):
            raise ValueError("condition_context shape does not match bridge config")
        if noise_level.shape not in ((batch,), (batch, 1)):
            raise ValueError("noise_level must have shape [B] or [B,1]")
        noise_level = noise_level.reshape(batch, 1).to(action_hidden.dtype)
        if bool(torch.any((noise_level < 0) | (noise_level > 1))):
            raise ValueError("noise_level must lie in [0,1]")

        edges = tokens - 1
        context = condition_context.to(action_hidden.dtype)[:, None].expand(
            batch, edges, cfg.context_dim)
        noise = noise_level[:, None].expand(batch, edges, 1)
        projector_input = torch.cat(
            (action_hidden[:, :-1], context, noise), dim=-1)
        return (
            self.flow_projector(projector_input),
            self.jump_projector(projector_input),
        )


@dataclass(frozen=True)
class HostFeedbackBridgeConfig:
    latent_dim: int = 32
    host_hidden_dim: int = 512
    hidden_dim: int = 256


class HostFeedbackBridge(nn.Module):
    """Zero-initialized same-time latent-to-host residual."""

    def __init__(self, config: HostFeedbackBridgeConfig):
        super().__init__()
        self.config = config
        self.network = _mlp(
            config.latent_dim, config.hidden_dim, config.host_hidden_dim)
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("feedback bridge must end in nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 3 or latent.shape[-1] != self.config.latent_dim:
            raise ValueError("latent must have shape [B,T,D_z]")
        return self.network(latent)
