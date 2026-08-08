"""Causal episode-history encoder for P1.

The default production backend is Transformers' exact sequential Mamba reference
path. The optional ``mamba_ssm`` backend has the same model role but requires a
CUDA extension. A GRU backend is intentionally kept only for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class P1HistoryConfig:
    feature_dim: int = 969
    action_dim: int = 12
    oracle_vocab_size: int = 6
    oracle_dim: int = 32
    model_dim: int = 128
    latent_dim: int = 32
    num_layers: int = 2
    backend: str = "transformers_mamba"
    mamba_state_dim: int = 16
    mamba_conv_width: int = 4
    mamba_expand: int = 2


class P1CausalHistoryEncoder(nn.Module):
    """Map a causal feature/action/symbol prefix to one latent per physical time."""

    def __init__(self, config: P1HistoryConfig | None = None):
        super().__init__()
        self.config = config or P1HistoryConfig()
        cfg = self.config
        self.backend = cfg.backend
        self.oracle_embedding = nn.Embedding(cfg.oracle_vocab_size, cfg.oracle_dim)
        self.input_projection = nn.Linear(
            cfg.feature_dim + cfg.action_dim + cfg.oracle_dim,
            cfg.model_dim,
        )
        if cfg.backend == "mamba_ssm":
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise RuntimeError(
                    "P1 history backend 'mamba_ssm' requires the mamba-ssm package; "
                    "use backend='gru' only for tests/smoke runs"
                ) from exc
            self.sequence_layers = nn.ModuleList([
                Mamba(
                    d_model=cfg.model_dim,
                    d_state=cfg.mamba_state_dim,
                    d_conv=cfg.mamba_conv_width,
                    expand=cfg.mamba_expand,
                )
                for _ in range(cfg.num_layers)
            ])
            self.sequence_norms = nn.ModuleList([
                nn.LayerNorm(cfg.model_dim) for _ in range(cfg.num_layers)
            ])
            self.gru = None
        elif cfg.backend == "transformers_mamba":
            from transformers.models.mamba.configuration_mamba import MambaConfig
            from transformers.models.mamba.modeling_mamba import MambaModel

            mamba_config = MambaConfig(
                hidden_size=cfg.model_dim,
                state_size=cfg.mamba_state_dim,
                num_hidden_layers=cfg.num_layers,
                expand=cfg.mamba_expand,
                conv_kernel=cfg.mamba_conv_width,
                use_cache=False,
                use_mambapy=False,
            )
            # Construct the official bare model once so post_init applies Mamba's
            # dt/A/D initialization, then keep only its initialized blocks. The
            # unused token embedding is deliberately not registered in P1.
            initialized = MambaModel(mamba_config)
            self.sequence_layers = initialized.layers
            self.sequence_norms = nn.ModuleList()
            self.gru = None
        elif cfg.backend == "gru":
            self.sequence_layers = nn.ModuleList()
            self.sequence_norms = nn.ModuleList()
            self.gru = nn.GRU(
                input_size=cfg.model_dim,
                hidden_size=cfg.model_dim,
                num_layers=cfg.num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(
                "history backend must be transformers_mamba, mamba_ssm, or gru")
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.history_projection = nn.Linear(cfg.model_dim, cfg.latent_dim)

    def embed_symbol(self, symbol: Tensor | None, *, reference: Tensor) -> Tensor:
        if symbol is None:
            return reference.new_zeros((*reference.shape[:-1], self.config.oracle_dim))
        if symbol.shape != reference.shape[:-1]:
            raise ValueError(
                f"oracle symbol shape {symbol.shape} must match {reference.shape[:-1]}"
            )
        if bool(torch.any((symbol < 0) | (symbol >= self.config.oracle_vocab_size))):
            raise ValueError("oracle symbol lies outside the configured vocabulary")
        return self.oracle_embedding(symbol.to(dtype=torch.long))

    def forward(
        self,
        *,
        feature: Tensor,
        previous_action: Tensor,
        valid_mask: Tensor,
        oracle_symbol: Tensor | None = None,
    ) -> Tensor:
        if feature.ndim != 3 or feature.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"feature must have shape [B,L,{self.config.feature_dim}]"
            )
        if previous_action.shape != (
            feature.shape[0], feature.shape[1], self.config.action_dim
        ):
            raise ValueError("previous_action must have shape [B,L,action_dim]")
        if valid_mask.shape != feature.shape[:2]:
            raise ValueError("valid_mask must have shape [B,L]")
        valid_mask = valid_mask.to(dtype=torch.bool)
        oracle = self.embed_symbol(oracle_symbol, reference=feature)
        model_input = torch.cat((feature, previous_action, oracle), dim=-1)
        model_input = model_input * valid_mask.unsqueeze(-1).to(model_input.dtype)
        hidden = self.input_projection(model_input)
        if self.gru is not None:
            hidden, _ = self.gru(hidden)
        elif self.backend == "transformers_mamba":
            for layer in self.sequence_layers:
                hidden = layer(hidden)
        else:
            for norm, layer in zip(self.sequence_norms, self.sequence_layers):
                hidden = hidden + layer(norm(hidden))
        latent = self.history_projection(self.output_norm(hidden))
        return latent * valid_mask.unsqueeze(-1).to(latent.dtype)

    @staticmethod
    def gather_at_time(latent: Tensor, physical_time: Tensor) -> Tensor:
        if latent.ndim != 3 or physical_time.ndim != 2:
            raise ValueError("latent [B,L,D] and physical_time [B,K] are required")
        if latent.shape[0] != physical_time.shape[0]:
            raise ValueError("history/query batch dimensions do not match")
        if bool(torch.any((physical_time < 0) | (physical_time >= latent.shape[1]))):
            raise IndexError("history query lies outside the padded prefix")
        index = physical_time.to(dtype=torch.long).unsqueeze(-1).expand(
            -1, -1, latent.shape[-1])
        return torch.gather(latent, dim=1, index=index)

    @staticmethod
    def gather_groups_at_time(
            latent: Tensor,
            group_to_history: Tensor,
            physical_time: Tensor,
        ) -> Tensor:
        """Gather ``[B,K,D]`` queries from ``E`` deduplicated histories."""
        if latent.ndim != 3 or physical_time.ndim != 2:
            raise ValueError("latent [E,L,D] and physical_time [B,K] are required")
        if group_to_history.shape != (physical_time.shape[0],):
            raise ValueError("group_to_history must have shape [B]")
        group_to_history = group_to_history.to(dtype=torch.long)
        if bool(torch.any(
                (group_to_history < 0) | (group_to_history >= latent.shape[0]))):
            raise IndexError("group_to_history lies outside the history batch")
        if bool(torch.any((physical_time < 0) | (physical_time >= latent.shape[1]))):
            raise IndexError("history query lies outside the padded prefix")
        history_index = group_to_history[:, None].expand_as(physical_time)
        return latent[history_index, physical_time.to(dtype=torch.long)]
