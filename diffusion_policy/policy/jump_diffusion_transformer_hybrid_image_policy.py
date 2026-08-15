"""Diffusion Policy composed with the portable causal Jump adapter."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.jump_transformer_for_diffusion import (
    JumpTransformerForDiffusion,
)
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from jump_module import (
    CausalHistoryConfig,
    CausalHistoryEncoder,
    HostConditionBridge,
    HostConditionBridgeConfig,
    HostFeedbackBridge,
    HostFeedbackBridgeConfig,
    JumpAdapter,
    JumpAdapterInput,
    JumpControls,
    JumpDynamicsConfig,
    JumpDynamicsCore,
    masked_balanced_boundary_loss,
)


class JumpDiffusionTransformerHybridImagePolicy(
        DiffusionTransformerHybridImagePolicy):
    """A separate DP + Jump architecture with a frozen vanilla DP host."""

    architecture_id = "dp_jump"
    history_input_schema = (
        "full_fused_observation_feature_plus_previous_action_per_timestep")
    history_time_alignment = "includes_current_condition_anchor"
    jump_token_alignment = "starts_at_first_executed_action"

    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            num_inference_steps: int | None = None,
            vision_backbone: str = "resnet18",
            crop_shape: tuple[int, int] | list[int] | None = (76, 76),
            obs_encoder_group_norm: bool = False,
            eval_fixed_crop: bool = False,
            n_layer: int = 8,
            n_cond_layers: int = 0,
            n_head: int = 4,
            n_emb: int = 256,
            p_drop_emb: float = 0.0,
            p_drop_attn: float = 0.3,
            causal_attn: bool = True,
            time_as_cond: bool = True,
            obs_as_cond: bool = True,
            pred_action_steps_only: bool = False,
            jump_split_layer: int = 6,
            jump_history_backend: str = "selective_ssm",
            jump_history_model_dim: int = 128,
            jump_history_layers: int = 2,
            jump_history_state_dim: int = 16,
            jump_history_conv_width: int = 4,
            jump_history_expand: int = 2,
            jump_latent_dim: int = 32,
            jump_flow_condition_dim: int = 64,
            jump_condition_dim: int = 64,
            jump_projector_hidden_dim: int = 256,
            jump_dynamics_hidden_dim: int = 128,
            jump_feedback_hidden_dim: int = 256,
            jump_edge_dt_seconds: float = 0.05,
            jump_beta_boundary: float = 0.10,
            jump_teacher_force_until: float = 0.40,
            jump_teacher_force_end: float = 0.60,
            jump_freeze_host: bool = True,
            jump_store_full_diffusion_trace: bool = False,
            **kwargs,
        ):
        if not obs_as_cond:
            raise ValueError("DP + Jump currently requires obs_as_cond=True")
        if pred_action_steps_only:
            raise ValueError(
                "DP + Jump requires the full vanilla DP action horizon")
        if n_obs_steps >= horizon:
            raise ValueError(
                "n_obs_steps must leave at least two executable Jump tokens")
        if n_action_steps > horizon - n_obs_steps + 1:
            raise ValueError(
                "n_action_steps exceeds the executable full-horizon suffix")
        if not jump_freeze_host:
            raise ValueError("v3 keeps the DP host frozen for DP + Jump")
        super().__init__(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            num_inference_steps=num_inference_steps,
            vision_backbone=vision_backbone,
            crop_shape=crop_shape,
            obs_encoder_group_norm=obs_encoder_group_norm,
            eval_fixed_crop=eval_fixed_crop,
            n_layer=n_layer,
            n_cond_layers=n_cond_layers,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            pred_action_steps_only=pred_action_steps_only,
            **kwargs,
        )
        base_model = self.model
        jump_model = JumpTransformerForDiffusion(
            input_dim=base_model.input_emb.in_features,
            output_dim=base_model.head.out_features,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=self.obs_feature_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )
        jump_model.load_state_dict(base_model.state_dict(), strict=True)
        self.model = jump_model

        if self.model.decoder is None:
            raise ValueError("DP + Jump requires decoder-mode Transformer")
        self.jump_split_layer = int(jump_split_layer)
        if not 0 < self.jump_split_layer < len(self.model.decoder.layers):
            raise ValueError("jump_split_layer must lie inside the decoder")
        if jump_edge_dt_seconds <= 0:
            raise ValueError("jump_edge_dt_seconds must be positive")
        if not 0 <= jump_teacher_force_until <= jump_teacher_force_end <= 1:
            raise ValueError("invalid Jump teacher schedule")
        self.jump_action_start = self.n_obs_steps - 1
        self.jump_token_count = self.horizon - self.jump_action_start
        self.jump_edge_count = self.jump_token_count - 1

        history = CausalHistoryEncoder(CausalHistoryConfig(
            feature_dim=self.obs_feature_dim,
            action_dim=self.action_dim,
            model_dim=jump_history_model_dim,
            latent_dim=jump_latent_dim,
            num_layers=jump_history_layers,
            backend=jump_history_backend,
            state_dim=jump_history_state_dim,
            conv_width=jump_history_conv_width,
            expand=jump_history_expand,
        ))
        condition_bridge = HostConditionBridge(HostConditionBridgeConfig(
            host_hidden_dim=n_emb,
            context_dim=self.obs_feature_dim * self.n_obs_steps,
            flow_condition_dim=jump_flow_condition_dim,
            jump_condition_dim=jump_condition_dim,
            hidden_dim=jump_projector_hidden_dim,
        ))
        dynamics = JumpDynamicsCore(JumpDynamicsConfig(
            latent_dim=jump_latent_dim,
            flow_condition_dim=jump_flow_condition_dim,
            jump_condition_dim=jump_condition_dim,
            hidden_dim=jump_dynamics_hidden_dim,
        ))
        feedback = HostFeedbackBridge(HostFeedbackBridgeConfig(
            latent_dim=jump_latent_dim,
            host_hidden_dim=n_emb,
            hidden_dim=jump_feedback_hidden_dim,
        ))
        self.jump_adapter = JumpAdapter(
            condition_bridge=condition_bridge,
            dynamics=dynamics,
            feedback_bridge=feedback,
            history_encoder=history,
        )

        self.jump_edge_dt_seconds = float(jump_edge_dt_seconds)
        self.jump_beta_boundary = float(jump_beta_boundary)
        self.jump_teacher_force_until = float(jump_teacher_force_until)
        self.jump_teacher_force_end = float(jump_teacher_force_end)
        self.jump_freeze_host = bool(jump_freeze_host)
        self.jump_store_full_diffusion_trace = bool(
            jump_store_full_diffusion_trace)
        self.register_buffer(
            "jump_training_progress", torch.tensor(0.0), persistent=False)
        self.obs_encoder.requires_grad_(False)
        self.model.requires_grad_(False)
        for parameter in self.parameters():
            if parameter.numel() == 0:
                parameter.requires_grad_(False)
        self.last_loss_metrics: dict[str, float] = {}
        self.last_jump_trace: dict[str, Any] = {}
        self.reset()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.jump_freeze_host:
            self.obs_encoder.eval()
            self.model.eval()
        self.jump_adapter.train(mode)
        return self

    def set_training_progress(self, progress: float) -> None:
        if not 0 <= progress <= 1:
            raise ValueError("training progress must lie in [0,1]")
        self.jump_training_progress.fill_(float(progress))

    def teacher_ratio(self) -> float:
        progress = float(self.jump_training_progress.item())
        if progress <= self.jump_teacher_force_until:
            return 1.0
        if progress >= self.jump_teacher_force_end:
            return 0.0
        width = self.jump_teacher_force_end - self.jump_teacher_force_until
        if width == 0:
            return 0.0
        return 1.0 - (progress - self.jump_teacher_force_until) / width

    def trainable_parameter_audit(self) -> dict[str, Any]:
        # LinearNormalizer reconstructs tensor Parameters when a transform is
        # installed. They are data statistics, never optimization variables.
        self.normalizer.requires_grad_(False)
        names = [name for name, parameter in self.named_parameters()
                 if parameter.requires_grad]
        invalid = [name for name in names if not name.startswith("jump_adapter.")]
        if self.jump_freeze_host and invalid:
            raise RuntimeError(f"frozen host has unexpected trainable keys: {invalid}")
        return {
            "architecture_id": self.architecture_id,
            "host_frozen": self.jump_freeze_host,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad),
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
                if not parameter.requires_grad),
            "trainable_names": names,
        }

    def set_normalizer(self, normalizer) -> None:
        super().set_normalizer(normalizer)
        self.normalizer.requires_grad_(False)

    def load_base_model_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Strictly load only the checkpoint-compatible Transformer host."""
        self.model.load_state_dict(state_dict, strict=True)

    def load_jump_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.jump_adapter.load_state_dict(state_dict, strict=True)

    def load_pretrained_backbone(self, state_dict: dict[str, Tensor]) -> None:
        """Load a vanilla policy checkpoint with an explicit mismatch audit."""
        normalizer_state = {
            key: value.detach().clone()
            for key, value in self.normalizer.state_dict().items()
        }
        backbone_state = {
            key: value for key, value in state_dict.items()
            if not key.startswith("normalizer.")
        }
        incompatible = self.load_state_dict(backbone_state, strict=False)
        self.normalizer.load_state_dict(normalizer_state)
        allowed_missing = ("normalizer.", "jump_adapter.")
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "pretrained backbone mismatch: "
                f"missing={invalid_missing}, "
                f"unexpected={incompatible.unexpected_keys}")

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float],
            jump_weight_decay: float = 1.0e-3,
        ) -> torch.optim.Optimizer:
        parameters = [
            parameter for parameter in self.jump_adapter.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("Jump adapter has no trainable parameters")
        return torch.optim.AdamW(
            [{"params": parameters, "weight_decay": jump_weight_decay}],
            lr=learning_rate,
            betas=betas,
        )

    def _encode_chunk_condition(
            self, batch: Dict[str, Any], *, batch_size: int
        ) -> Tensor:
        if "obs_feature" in batch:
            feature = batch["obs_feature"]
            expected = (batch_size, self.n_obs_steps, self.obs_feature_dim)
            if feature.shape != expected:
                raise ValueError(
                    f"cached obs_feature shape {feature.shape} != {expected}")
            return feature.to(device=self.device, dtype=self.dtype)
        if "obs" not in batch:
            raise KeyError("DP + Jump batch requires obs or obs_feature")
        normalized = self.normalizer.normalize(batch["obs"])
        flattened = dict_apply(
            normalized,
            lambda value: value[:, :self.n_obs_steps].reshape(
                -1, *value.shape[2:]),
        )
        with torch.set_grad_enabled(not self.jump_freeze_host):
            encoded = self.obs_encoder(flattened)
        return encoded.reshape(batch_size, self.n_obs_steps, -1)

    def _history_initial_state(self, batch: Dict[str, Any]) -> Tensor:
        required = (
            "history_feature",
            "history_previous_action",
            "history_valid_mask",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"DP + Jump batch is missing history fields: {missing}")
        feature = batch["history_feature"].to(
            device=self.device, dtype=self.dtype)
        valid_mask = batch["history_valid_mask"].to(device=self.device)
        previous_action = self.normalizer["action"].normalize(
            batch["history_previous_action"].to(
                device=self.device, dtype=self.dtype))
        return self.jump_adapter.encode_history(
            feature=feature,
            previous_action=previous_action,
            valid_mask=valid_mask,
        )

    def _jump_model_output(
            self,
            *,
            noisy_action: Tensor,
            timestep: Tensor,
            condition: Tensor,
            initial_z: Tensor,
            edge_valid_mask: Tensor,
            edge_dt_seconds: Tensor,
            boundary_target: Tensor | None,
            teacher_ratio: float,
            controls: JumpControls = JumpControls(),
        ) -> tuple[Tensor, Any]:
        with torch.set_grad_enabled(not self.jump_freeze_host):
            split = self.model.forward_decoder_pre(
                noisy_action,
                timestep,
                condition,
                split_layer=self.jump_split_layer,
            )
        timestep_batch = torch.as_tensor(
            timestep, device=noisy_action.device).expand(noisy_action.shape[0])
        noise_level = timestep_batch.to(noisy_action.dtype) / max(
            self.noise_scheduler.config.num_train_timesteps - 1, 1)
        active_hidden = split.hidden[:, self.jump_action_start:]
        adapter_output = self.jump_adapter(JumpAdapterInput(
            action_hidden=active_hidden,
            initial_z=initial_z,
            condition_context=condition.reshape(condition.shape[0], -1),
            noise_level=noise_level,
            edge_dt_seconds=edge_dt_seconds,
            edge_valid_mask=edge_valid_mask,
            boundary_target=boundary_target,
            teacher_ratio=teacher_ratio,
            controls=controls,
        ))
        # Do not wrap the post blocks in no_grad: frozen parameters still need
        # to propagate the action objective back into the adapter input.
        corrected_hidden = torch.cat((
            split.hidden[:, :self.jump_action_start],
            adapter_output.action_hidden,
        ), dim=1)
        prediction = self.model.forward_decoder_post(
            split, hidden=corrected_hidden)
        return prediction, adapter_output

    def compute_loss(
            self,
            batch: Dict[str, Any],
            *,
            diffusion_timestep_override: int | Tensor | None = None,
        ) -> Tensor:
        action = self.normalizer["action"].normalize(
            batch["action"].to(device=self.device, dtype=self.dtype))
        if action.ndim != 3 or action.shape[1:] != (
                self.horizon, self.action_dim):
            raise ValueError("action must have shape [B,horizon,action_dim]")
        batch_size = action.shape[0]
        condition = self._encode_chunk_condition(batch, batch_size=batch_size)
        initial_z = self._history_initial_state(batch)
        action_valid = batch.get(
            "action_valid_mask",
            torch.ones(action.shape[:2], device=action.device, dtype=torch.bool),
        ).to(device=action.device, dtype=torch.bool)
        if action_valid.shape != action.shape[:2]:
            raise ValueError("action_valid_mask must have shape [B,horizon]")
        edge_valid = batch["edge_valid_mask"].to(
            device=action.device, dtype=torch.bool)
        if edge_valid.shape != (batch_size, self.jump_edge_count):
            raise ValueError(
                "edge_valid_mask must cover executable Jump edges")
        edge_dt = batch["edge_dt_seconds"].to(
            device=action.device, dtype=action.dtype)
        boundary_target = batch["boundary_target"].to(device=action.device)

        noise = torch.randn_like(action)
        if diffusion_timestep_override is None:
            timestep = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=action.device,
            ).long()
        elif isinstance(diffusion_timestep_override, int):
            timestep = torch.full(
                (batch_size,), diffusion_timestep_override,
                device=action.device, dtype=torch.long)
        else:
            timestep = diffusion_timestep_override.to(
                device=action.device, dtype=torch.long)
            if timestep.shape != (batch_size,):
                raise ValueError("diffusion timestep override must have shape [B]")
        if bool(torch.any(timestep < 0)) or bool(torch.any(
                timestep >= self.noise_scheduler.config.num_train_timesteps)):
            raise ValueError("diffusion timestep lies outside the scheduler")
        noisy_action = self.noise_scheduler.add_noise(action, noise, timestep)
        prediction, output = self._jump_model_output(
            noisy_action=noisy_action,
            timestep=timestep,
            condition=condition,
            initial_z=initial_z,
            edge_valid_mask=edge_valid,
            edge_dt_seconds=edge_dt,
            boundary_target=boundary_target,
            teacher_ratio=self.teacher_ratio(),
        )
        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            diffusion_target = noise
        elif prediction_type == "sample":
            diffusion_target = action
        else:
            raise ValueError(f"unsupported prediction type {prediction_type}")
        action_mask = action_valid.clone()
        action_mask[:, :self.jump_action_start] = False
        action_mask = action_mask.unsqueeze(-1).expand_as(prediction)
        if not bool(torch.any(action_mask)):
            diffusion_loss = prediction.sum() * 0
        else:
            diffusion_loss = (
                (prediction - diffusion_target).square()[action_mask].mean())
        boundary = masked_balanced_boundary_loss(
            intensity=output.intensity,
            target=boundary_target,
            edge_dt_seconds=edge_dt,
            valid_mask=edge_valid,
        )
        total = diffusion_loss + self.jump_beta_boundary * boundary.loss
        valid_edge_float = edge_valid.to(output.probability.dtype)
        edge_denominator = valid_edge_float.sum().clamp_min(1)
        self.last_loss_metrics = {
            "loss_total": float(total.detach()),
            "loss_diffusion": float(diffusion_loss.detach()),
            "loss_boundary": float(boundary.loss.detach()),
            "teacher_ratio": self.teacher_ratio(),
            "boundary_probability_mean": float(
                (output.probability * valid_edge_float).sum().detach()
                / edge_denominator),
            "flow_delta_norm": float(
                (output.flow_delta.norm(dim=-1) * valid_edge_float).sum().detach()
                / edge_denominator),
            "applied_jump_delta_norm": float(
                (output.applied_jump_delta.norm(dim=-1) * valid_edge_float)
                .sum().detach() / edge_denominator),
            "feedback_residual_norm": float(
                output.feedback_residual.norm(dim=-1).mean().detach()),
            "boundary_positive_count": float(boundary.positive_count),
            "boundary_negative_count": float(boundary.negative_count),
        }
        return total

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        return self.compute_loss(batch)

    def reset(self) -> None:
        self._jump_history_feature: Tensor | None = None
        self._jump_history_previous_action: Tensor | None = None
        self._jump_history_valid_mask: Tensor | None = None
        self._jump_history_observed_length: Tensor | None = None
        self.last_jump_trace = {}

    def _append_executed_history(
            self,
            latest_feature: Tensor,
            executed_feature: Tensor | None,
            executed_action: Tensor | None,
            executed_valid_mask: Tensor | None,
        ) -> Tensor:
        latest = latest_feature.detach()[:, -1:]
        batch = latest.shape[0]
        if self._jump_history_feature is None:
            if any(value is not None for value in (
                    executed_feature, executed_action, executed_valid_mask)):
                raise RuntimeError(
                    "the first policy call cannot contain an executed segment")
            zero_action = self.normalizer["action"].normalize(
                latest.new_zeros((batch, 1, self.action_dim)))
            self._jump_history_feature = latest
            self._jump_history_previous_action = zero_action
            self._jump_history_valid_mask = torch.ones(
                (batch, 1), device=latest.device, dtype=torch.bool)
            self._jump_history_observed_length = torch.ones(
                batch, device=latest.device, dtype=torch.long)
        else:
            provided = tuple(value is not None for value in (
                executed_feature, executed_action, executed_valid_mask))
            if not all(provided):
                raise RuntimeError(
                    "the next policy call requires the complete executed segment")
            if executed_feature.ndim != 3 \
                    or executed_feature.shape[0] != batch \
                    or executed_feature.shape[2] != self.obs_feature_dim:
                raise ValueError(
                    "executed_feature must have shape [B,M,obs_feature_dim]")
            segment_length = executed_feature.shape[1]
            if executed_action.shape != (
                    batch, segment_length, self.action_dim):
                raise ValueError(
                    "executed_action must have shape [B,M,action_dim]")
            if executed_valid_mask.shape != (batch, segment_length):
                raise ValueError("executed_valid_mask must have shape [B,M]")
            executed_valid_mask = executed_valid_mask.to(dtype=torch.bool)
            if bool(torch.any(
                    executed_valid_mask[:, 1:] & ~executed_valid_mask[:, :-1])):
                raise ValueError(
                    "executed segment valid positions must form a prefix")
            if self._jump_history_feature.shape[0] != batch:
                raise RuntimeError("rollout batch size changed before reset")
            segment_feature = executed_feature.detach()
            for batch_index in range(batch):
                added_count = int(
                    executed_valid_mask[batch_index].sum().item())
                if added_count and not torch.allclose(
                        latest_feature[batch_index, -1],
                        segment_feature[batch_index, added_count - 1],
                        rtol=1e-5,
                        atol=1e-6,
                    ):
                    raise ValueError(
                        "latest observation does not match the executed "
                        "segment endpoint")

            old_length = self._jump_history_observed_length
            added_length = executed_valid_mask.sum(dim=1)
            total_length = old_length + added_length
            width = int(total_length.max().item())
            packed_feature = latest.new_zeros((
                batch, width, self._jump_history_feature.shape[-1]))
            packed_action = latest.new_zeros((batch, width, self.action_dim))
            packed_valid = torch.zeros(
                (batch, width), device=latest.device, dtype=torch.bool)
            for batch_index in range(batch):
                old_count = int(old_length[batch_index].item())
                added_count = int(added_length[batch_index].item())
                packed_feature[batch_index, :old_count] = (
                    self._jump_history_feature[batch_index, :old_count])
                packed_action[batch_index, :old_count] = (
                    self._jump_history_previous_action[batch_index, :old_count])
                if added_count:
                    stop = old_count + added_count
                    packed_feature[batch_index, old_count:stop] = (
                        segment_feature[batch_index, :added_count])
                    packed_action[batch_index, old_count:stop] = (
                        executed_action[batch_index, :added_count].detach())
                # initial_z is the current physical-time latent. Every pair
                # actually observed in the segment contributes immediately.
                valid_count = old_count + added_count
                packed_valid[batch_index, :valid_count] = True
            self._jump_history_feature = packed_feature
            self._jump_history_previous_action = packed_action
            self._jump_history_valid_mask = packed_valid
            self._jump_history_observed_length = total_length
        return self.jump_adapter.encode_history(
            feature=self._jump_history_feature,
            previous_action=self._jump_history_previous_action,
            valid_mask=self._jump_history_valid_mask,
        )

    @torch.no_grad()
    def _jump_conditional_sample(
            self,
            condition_data: Tensor,
            condition_mask: Tensor,
            condition: Tensor,
            initial_z: Tensor,
            generator=None,
            controls: JumpControls = JumpControls(),
        ) -> Tensor:
        trajectory = torch.randn(
            condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        batch, tokens = trajectory.shape[:2]
        active_tokens = tokens - self.jump_action_start
        if active_tokens < 2:
            raise RuntimeError("Jump sampling requires at least two active tokens")
        edge_valid = torch.ones(
            (batch, active_tokens - 1),
            device=trajectory.device,
            dtype=torch.bool,
        )
        edge_dt = trajectory.new_full(
            (batch, active_tokens - 1), self.jump_edge_dt_seconds)
        latest = None
        diffusion_trace = []
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            prediction, latest = self._jump_model_output(
                noisy_action=trajectory,
                timestep=timestep,
                condition=condition,
                initial_z=initial_z,
                edge_valid_mask=edge_valid,
                edge_dt_seconds=edge_dt,
                boundary_target=None,
                teacher_ratio=0.0,
                controls=controls,
            )
            trajectory = self.noise_scheduler.step(
                prediction,
                timestep,
                trajectory,
                generator=generator,
                **self.kwargs,
            ).prev_sample
            row: dict[str, Any] = {
                "timestep": int(torch.as_tensor(timestep).item()),
                "noise_level": float(
                    torch.as_tensor(timestep).item()
                    / max(self.noise_scheduler.config.num_train_timesteps - 1, 1)),
            }
            if self.jump_store_full_diffusion_trace:
                row.update({
                    "z": latest.z.detach().cpu().tolist(),
                    "z_minus": latest.z_minus.detach().cpu().tolist(),
                    "flow_delta": latest.flow_delta.detach().cpu().tolist(),
                    "raw_jump_delta": (
                        latest.raw_jump_delta.detach().cpu().tolist()),
                    "applied_jump_delta": (
                        latest.applied_jump_delta.detach().cpu().tolist()),
                    "intensity": latest.intensity.detach().cpu().tolist(),
                    "probability": latest.probability.detach().cpu().tolist(),
                    "gate": latest.gate.detach().cpu().tolist(),
                })
            diffusion_trace.append(row)
        trajectory[condition_mask] = condition_data[condition_mask]
        if latest is None:
            raise RuntimeError("diffusion scheduler executed zero steps")
        self.last_jump_trace = {
            "jump_action_start": self.jump_action_start,
            "z": latest.z.detach().cpu(),
            "z_minus": latest.z_minus.detach().cpu(),
            "flow_delta": latest.flow_delta.detach().cpu(),
            "raw_jump_delta": latest.raw_jump_delta.detach().cpu(),
            "applied_jump_delta": latest.applied_jump_delta.detach().cpu(),
            "intensity": latest.intensity.detach().cpu(),
            "probability": latest.probability.detach().cpu(),
            "gate": latest.gate.detach().cpu(),
            "diffusion_trace": diffusion_trace,
            "full_diffusion_trace_stored": self.jump_store_full_diffusion_trace,
        }
        return trajectory

    def _prediction_from_feature(
            self,
            feature: Tensor,
            executed_feature: Tensor | None = None,
            executed_action: Tensor | None = None,
            executed_valid_mask: Tensor | None = None,
        ) -> Dict[str, Tensor]:
        initial_z = self._append_executed_history(
            feature,
            executed_feature,
            executed_action,
            executed_valid_mask,
        )
        batch = feature.shape[0]
        token_count = self.n_action_steps \
            if self.pred_action_steps_only else self.horizon
        condition_data = torch.zeros(
            (batch, token_count, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        sample = self._jump_conditional_sample(
            condition_data, condition_mask, feature, initial_z)
        action_pred = self.normalizer["action"].unnormalize(
            sample[..., :self.action_dim])
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1
            action = action_pred[:, start:start + self.n_action_steps]
        return {"action": action, "action_pred": action_pred}

    def predict_action_from_cached_feature(
            self,
            feature: Tensor,
            executed_feature: Tensor | None = None,
            executed_action: Tensor | None = None,
            executed_valid_mask: Tensor | None = None,
        ) -> Dict[str, Tensor]:
        if feature.ndim == 2:
            feature = feature[:, None]
        if feature.shape[1:] != (self.n_obs_steps, self.obs_feature_dim):
            raise ValueError(
                "feature must have shape [B,n_obs_steps,obs_feature_dim]")
        feature = feature.to(device=self.device, dtype=self.dtype)
        if executed_feature is not None:
            executed_feature = executed_feature.to(
                device=self.device, dtype=self.dtype)
        if executed_action is not None:
            executed_action = self.normalizer["action"].normalize(
                executed_action.to(device=self.device, dtype=self.dtype))
        if executed_valid_mask is not None:
            executed_valid_mask = executed_valid_mask.to(device=self.device)
        return self._prediction_from_feature(
            feature,
            executed_feature,
            executed_action,
            executed_valid_mask,
        )

    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raw = dict(obs_dict)
        executed_obs = raw.pop("executed_obs", None)
        executed_action = raw.pop("executed_action", None)
        executed_valid_mask = raw.pop("executed_valid_mask", None)
        if "past_action" in raw:
            raise ValueError(
                "DP + Jump requires executed segments, not past_action")
        normalized = self.normalizer.normalize(raw)
        if not normalized:
            raise ValueError("DP + Jump inference requires observations")
        value = next(iter(normalized.values()))
        batch = value.shape[0]
        flattened = dict_apply(
            normalized,
            lambda tensor: tensor[:, -self.n_obs_steps:].reshape(
                -1, *tensor.shape[2:]),
        )
        feature = self.obs_encoder(flattened).reshape(
            batch, self.n_obs_steps, self.obs_feature_dim)
        provided = tuple(value is not None for value in (
            executed_obs, executed_action, executed_valid_mask))
        if any(provided) and not all(provided):
            raise ValueError("executed segment fields must be provided together")
        executed_feature = None
        normalized_executed_action = None
        if all(provided):
            if executed_action.ndim != 3 \
                    or executed_action.shape[0] != batch \
                    or executed_action.shape[-1] != self.action_dim:
                raise ValueError(
                    "executed_action must have shape [B,M,action_dim]")
            segment_length = executed_action.shape[1]
            if executed_valid_mask.shape != (batch, segment_length):
                raise ValueError("executed_valid_mask must have shape [B,M]")
            normalized_segment = self.normalizer.normalize(executed_obs)
            for value in normalized_segment.values():
                if value.shape[:2] != (batch, segment_length):
                    raise ValueError(
                        "every executed observation must have shape [B,M,...]")
            if segment_length:
                flat_segment = dict_apply(
                    normalized_segment,
                    lambda tensor: tensor.reshape(-1, *tensor.shape[2:]),
                )
                executed_feature = self.obs_encoder(flat_segment).reshape(
                    batch, segment_length, self.obs_feature_dim)
            else:
                executed_feature = feature.new_zeros(
                    (batch, 0, self.obs_feature_dim))
            normalized_executed_action = self.normalizer["action"].normalize(
                executed_action).to(dtype=feature.dtype)
        return self._prediction_from_feature(
            feature,
            executed_feature,
            normalized_executed_action,
            executed_valid_mask,
        )
