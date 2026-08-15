"""Diffusion Transformer host with an explicit Jump insertion boundary.

The vanilla :mod:`transformer_for_diffusion` module remains unchanged.  This
subclass adds only split execution helpers and therefore has exactly the same
parameter and state-dict layout as ``TransformerForDiffusion``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple, Union

import torch

from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)


class DecoderSplitState(NamedTuple):
    hidden: torch.Tensor
    memory: torch.Tensor
    split_layer: int


class JumpTransformerForDiffusion(TransformerForDiffusion):
    """Checkpoint-compatible Transformer with pre/post decoder execution."""

    def _prepare_decoder_inputs(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            cond: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_only or self.decoder is None:
            raise RuntimeError("decoder split-forward requires decoder mode")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_embedding = self.time_emb(timesteps).unsqueeze(1)

        condition_embedding = time_embedding
        if self.obs_as_cond:
            if cond is None:
                raise ValueError("cond is required when obs_as_cond=True")
            condition_embedding = torch.cat(
                (condition_embedding, self.cond_obs_emb(cond)), dim=1)
        condition_tokens = condition_embedding.shape[1]
        memory = self.encoder(self.drop(
            condition_embedding + self.cond_pos_emb[:, :condition_tokens]))

        action_embedding = self.input_emb(sample)
        action_tokens = action_embedding.shape[1]
        hidden = self.drop(action_embedding + self.pos_emb[:, :action_tokens])
        return hidden, memory

    def _run_decoder_layer_range(
            self,
            hidden: torch.Tensor,
            memory: torch.Tensor,
            *,
            start_layer: int,
            end_layer: int,
        ) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("decoder layer range requires decoder mode")
        layer_count = len(self.decoder.layers)
        if not 0 <= start_layer <= end_layer <= layer_count:
            raise ValueError(
                f"decoder layer range [{start_layer}, {end_layer}) is outside "
                f"[0, {layer_count})")
        for layer in self.decoder.layers[start_layer:end_layer]:
            hidden = layer(
                hidden,
                memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
                tgt_is_causal=self.mask is not None,
                memory_is_causal=False,
            )
        return hidden

    def forward_decoder_pre(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            cond: Optional[torch.Tensor] = None,
            *,
            split_layer: int,
        ) -> DecoderSplitState:
        hidden, memory = self._prepare_decoder_inputs(sample, timestep, cond)
        hidden = self._run_decoder_layer_range(
            hidden,
            memory,
            start_layer=0,
            end_layer=split_layer,
        )
        return DecoderSplitState(hidden, memory, split_layer)

    def forward_decoder_post(
            self,
            state: DecoderSplitState,
            *,
            hidden: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("decoder split-forward requires decoder mode")
        if hidden is None:
            hidden = state.hidden
        elif hidden.shape != state.hidden.shape:
            raise ValueError(
                f"replacement hidden shape {hidden.shape} does not match "
                f"pre-block hidden shape {state.hidden.shape}")
        hidden = self._run_decoder_layer_range(
            hidden,
            state.memory,
            start_layer=state.split_layer,
            end_layer=len(self.decoder.layers),
        )
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return self.head(self.ln_f(hidden))
