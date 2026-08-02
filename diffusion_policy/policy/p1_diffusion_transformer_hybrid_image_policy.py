"""Diffusion Policy integration for the three GetToastedBread P1 experiments."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.p1 import (
    P1DynamicsConfig,
    P1GateMode,
    P1GroupConfig,
    P1GroupLossOutput,
    P1HistoryConfig,
    P1LatentGroupModel,
    compute_p1_group_losses,
)
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)


class P1ExperimentMode(str, Enum):
    BASELINE = "baseline"
    ORACLE_GATE_SYMBOL = "oracle_gate_symbol"
    MAIN = "main"


class P1DiffusionTransformerHybridImagePolicy(
        DiffusionTransformerHybridImagePolicy):
    """One policy class with an audited experiment-mode boundary.

    ``baseline`` fine-tunes the original observation encoder and DiT on the P1
    group sampler. ``oracle_gate_symbol`` and ``main`` freeze those modules and
    train only the P1 latent path. Oracle receives causal past/current symbols
    and uses GT boundaries only as a training-time Jump teacher. Evaluation and
    online inference use predicted soft gates. Main receives no symbol and
    transitions from GT to predicted soft gates during training.
    """

    def __init__(
            self,
            *args,
            p1_mode: str = "baseline",
            p1_history_backend: str = "transformers_mamba",
            p1_history_model_dim: int = 128,
            p1_history_layers: int = 2,
            p1_latent_dim: int = 32,
            p1_oracle_vocab_size: int = 6,
            p1_oracle_dim: int = 32,
            p1_flow_condition_dim: int = 64,
            p1_jump_condition_dim: int = 64,
            p1_projector_hidden_dim: int = 256,
            p1_dynamics_hidden_dim: int = 128,
            p1_feedback_hidden_dim: int = 256,
            p1_split_layer: int = 9,
            p1_group_windows: int = 4,
            p1_group_stride: int = 5,
            p1_beta_boundary: float = 0.10,
            p1_beta_initialization: float = 0.10,
            p1_beta_overlap_max: float = 0.05,
            p1_teacher_force_until: float = 0.40,
            p1_teacher_force_end: float = 0.60,
            p1_online_episode_length: int = 600,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.p1_mode = P1ExperimentMode(p1_mode)
        self.p1_split_layer = int(p1_split_layer)
        self.p1_beta_boundary = float(p1_beta_boundary)
        self.p1_beta_initialization = float(p1_beta_initialization)
        self.p1_beta_overlap_max = float(p1_beta_overlap_max)
        self.p1_teacher_force_until = float(p1_teacher_force_until)
        self.p1_teacher_force_end = float(p1_teacher_force_end)
        self.p1_online_episode_length = int(p1_online_episode_length)
        if self.p1_online_episode_length < 2:
            raise ValueError("p1_online_episode_length must be at least 2")


        if not (
            0 <= self.p1_teacher_force_until
            <= self.p1_teacher_force_end
            <= 1
        ):
            raise ValueError("invalid P1 teacher-forcing schedule")
        if self.model.decoder is None:
            raise ValueError("P1 requires decoder-mode Diffusion Transformer")
        if not 0 < self.p1_split_layer < len(self.model.decoder.layers):
            raise ValueError("p1_split_layer must lie inside the DiT decoder")

        self.p1_model: P1LatentGroupModel | None = None
        if self.p1_mode is not P1ExperimentMode.BASELINE:
            dynamics_config = P1DynamicsConfig(
                latent_dim=p1_latent_dim,
                dit_hidden_dim=self.model.head.in_features,
                context_dim=self.obs_feature_dim * self.n_obs_steps,
                oracle_dim=p1_oracle_dim,
                flow_condition_dim=p1_flow_condition_dim,
                jump_condition_dim=p1_jump_condition_dim,
                projector_hidden_dim=p1_projector_hidden_dim,
                dynamics_hidden_dim=p1_dynamics_hidden_dim,
                feedback_hidden_dim=p1_feedback_hidden_dim,
            )
            history_config = P1HistoryConfig(
                feature_dim=self.obs_feature_dim,
                action_dim=self.action_dim,
                oracle_vocab_size=p1_oracle_vocab_size,
                oracle_dim=p1_oracle_dim,
                model_dim=p1_history_model_dim,
                latent_dim=p1_latent_dim,
                num_layers=p1_history_layers,
                backend=p1_history_backend,
            )
            group_config = P1GroupConfig(
                windows=p1_group_windows,
                window_length=self.horizon,
                stride=p1_group_stride,
                num_diffusion_steps=(
                    self.noise_scheduler.config.num_train_timesteps),
            )
            self.p1_model = P1LatentGroupModel(
                dynamics_config=dynamics_config,
                history_config=history_config,
                group_config=group_config,
            )
            if self.p1_mode is P1ExperimentMode.MAIN:
                # Main receives no privileged symbol. Keeping this table trainable
                # would create a genuinely unused DDP parameter.
                self.p1_model.history.oracle_embedding.requires_grad_(False)
            self._freeze_pretrained_backbone()

        # ModuleAttrMixin and mask generators register zero-element parameters
        # only to expose module device/dtype. They never participate in the
        # forward graph, so leaving them trainable breaks DDP on iteration two
        # when find_unused_parameters=False.
        for parameter in self.parameters():
            if parameter.numel() == 0:
                parameter.requires_grad_(False)

        self.register_buffer(
            "p1_training_progress",
            torch.tensor(0.0),
            persistent=True,
        )
        self.last_loss_metrics: dict[str, float] = {}
        self.last_probe_rows: list[dict[str, float | int]] = []
        self.last_online_probe: dict[str, Any] = {}
        self.reset()

    @property
    def is_p1_latent(self) -> bool:
        return self.p1_model is not None

    def _freeze_pretrained_backbone(self) -> None:
        self.obs_encoder.requires_grad_(False)
        self.model.requires_grad_(False)
        self.obs_encoder.eval()
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.is_p1_latent:
            # Frozen dropout/crop behavior must not drift during P1 training.
            self.obs_encoder.eval()
            self.model.eval()
            if self.p1_model is not None:
                self.p1_model.train(mode)
        return self

    def set_training_progress(self, progress: float) -> None:
        if not 0 <= progress <= 1:
            raise ValueError("training progress must lie in [0,1]")
        self.p1_training_progress.fill_(float(progress))

    def schedule_values(self) -> tuple[float, float, float]:
        progress = float(self.p1_training_progress.item())
        if progress <= 0.20:
            carry_ratio = 1.0
        elif progress < 0.40:
            carry_ratio = 1.0 - 0.5 * ((progress - 0.20) / 0.20)
        else:
            carry_ratio = 0.5

        if progress <= 0.10:
            beta_overlap = 0.0
        elif progress < 0.20:
            beta_overlap = self.p1_beta_overlap_max * (
                (progress - 0.10) / 0.10)
        else:
            beta_overlap = self.p1_beta_overlap_max

        if progress <= self.p1_teacher_force_until:
            teacher_ratio = 1.0
        elif progress >= self.p1_teacher_force_end:
            teacher_ratio = 0.0
        else:
            teacher_ratio = 1.0 - (
                (progress - self.p1_teacher_force_until)
                / (self.p1_teacher_force_end - self.p1_teacher_force_until)
            )
        return carry_ratio, beta_overlap, teacher_ratio

    def trainable_parameter_audit(self) -> dict[str, Any]:
        names = [name for name, parameter in self.named_parameters()
                 if parameter.requires_grad]
        return {
            "mode": self.p1_mode.value,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad),
            "frozen_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
                if not parameter.requires_grad),
            "trainable_names": names,
        }

    def load_pretrained_backbone(self, state_dict: dict[str, Tensor]) -> None:
        """Strictly validate base keys while allowing only new P1 keys to be absent."""
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("p1_model.", "p1_training_progress")
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "pretrained backbone mismatch: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float],
            p1_weight_decay: float = 1.0e-3,
        ) -> torch.optim.Optimizer:
        if self.p1_mode is P1ExperimentMode.BASELINE:
            return super().get_optimizer(
                transformer_weight_decay=transformer_weight_decay,
                obs_encoder_weight_decay=obs_encoder_weight_decay,
                learning_rate=learning_rate,
                betas=betas,
            )
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("P1 latent experiment has no trainable parameters")
        return torch.optim.AdamW(
            [{"params": parameters, "weight_decay": p1_weight_decay}],
            lr=learning_rate,
            betas=betas,
        )

    @staticmethod
    def _shared_group_noise(action: Tensor, physical_time: Tensor) -> Tensor:
        """Sample exactly one Gaussian action vector per group physical time."""
        if action.ndim != 4 or physical_time.shape != action.shape[:3]:
            raise ValueError("action [B,K,W,D] and physical_time [B,K,W] required")
        noise = torch.empty_like(action)
        for group_index in range(action.shape[0]):
            _, inverse = torch.unique(
                physical_time[group_index].reshape(-1),
                sorted=True,
                return_inverse=True,
            )
            unique_noise = torch.randn(
                int(inverse.max().item()) + 1,
                action.shape[-1],
                device=action.device,
                dtype=action.dtype,
            )
            noise[group_index] = unique_noise[inverse].reshape_as(
                action[group_index])
        return noise

    def _encode_group_observation(
            self, batch: Dict[str, Any], *, batch_size: int, windows: int
        ) -> Tensor:
        if "obs_feature" in batch:
            feature = batch["obs_feature"]
            expected = (batch_size, windows, self.n_obs_steps, self.obs_feature_dim)
            if feature.shape != expected:
                raise ValueError(
                    f"cached obs_feature shape {feature.shape} != {expected}")
            return feature.to(dtype=self.dtype)
        if "obs" not in batch:
            raise KeyError("P1 group batch requires obs or obs_feature")
        normalized = self.normalizer.normalize(batch["obs"])
        value = next(iter(normalized.values()))
        if value.shape[:3] != (batch_size, windows, self.n_obs_steps):
            raise ValueError("raw group observations must have shape [B,K,To,...]")
        flattened = dict_apply(
            normalized,
            lambda tensor: tensor.reshape(
                batch_size * windows * self.n_obs_steps, *tensor.shape[3:]),
        )
        grad_enabled = self.p1_mode is P1ExperimentMode.BASELINE
        with torch.set_grad_enabled(grad_enabled):
            feature = self.obs_encoder(flattened)
        return feature.reshape(
            batch_size, windows, self.n_obs_steps, self.obs_feature_dim)

    @staticmethod
    def _masked_group_mean(value: Tensor, valid: Tensor) -> Tensor:
        valid = valid.to(dtype=torch.bool)
        if value.shape != valid.shape:
            raise ValueError("group value/mask shapes must match")
        if not bool(torch.any(valid)):
            return value.sum() * 0
        return value[valid].mean()

    @staticmethod
    def _masked_probe_values(value: Tensor, group_valid: Tensor) -> Tensor:
        """Flatten arbitrary per-group probes while excluding DDP dummy groups."""
        if value.shape[0] != group_valid.shape[0]:
            raise ValueError("probe and group mask batch dimensions must match")
        mask = group_valid.to(dtype=torch.bool).reshape(
            group_valid.shape[0], *([1] * (value.ndim - 1)))
        mask = mask.expand_as(value)
        if not bool(torch.any(mask)):
            return value.new_empty((0,))
        return value[mask]

    @classmethod
    def _masked_probe_mean(cls, value: Tensor, group_valid: Tensor) -> Tensor:
        values = cls._masked_probe_values(value, group_valid)
        if values.numel() == 0:
            return value.sum() * 0
        return values.mean()

    def compute_group_loss(
            self,
            batch: Dict[str, Any],
            *,
            diffusion_timestep_override: int | Tensor | None = None,
        ) -> tuple[Tensor, P1GroupLossOutput | None]:
        if not self.obs_as_cond:
            raise NotImplementedError("P1 MVP requires obs_as_cond=True")
        action = self.normalizer["action"].normalize(batch["action"])
        if action.ndim != 4:
            raise ValueError("P1 group action must have shape [B,K,W,D]")
        batch_size, windows, window, action_dim = action.shape
        if (window, action_dim) != (self.horizon, self.action_dim):
            raise ValueError("P1 action geometry does not match the policy")
        physical_time = batch["physical_time"].to(dtype=torch.long)
        group_valid = batch["episode_valid"].to(dtype=torch.bool)
        condition = self._encode_group_observation(
            batch, batch_size=batch_size, windows=windows)
        context = condition.reshape(batch_size, windows, -1)

        noise = self._shared_group_noise(action, physical_time)
        if diffusion_timestep_override is None:
            diffusion_timestep = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=action.device,
            ).long()
        elif isinstance(diffusion_timestep_override, int):
            diffusion_timestep = torch.full(
                (batch_size,),
                diffusion_timestep_override,
                device=action.device,
                dtype=torch.long,
            )
        else:
            diffusion_timestep = diffusion_timestep_override.to(
                device=action.device, dtype=torch.long)
            if diffusion_timestep.shape != (batch_size,):
                raise ValueError(
                    "diffusion_timestep_override must have shape [B]")
        if bool(torch.any(diffusion_timestep < 0)) or bool(torch.any(
            diffusion_timestep >= self.noise_scheduler.config.num_train_timesteps
        )):
            raise ValueError("diffusion_timestep_override is outside the scheduler")
        flat_timestep = diffusion_timestep[:, None].expand(
            -1, windows).reshape(-1)
        flat_action = action.reshape(batch_size * windows, window, action_dim)
        flat_noise = noise.reshape_as(flat_action)
        noisy_action = self.noise_scheduler.add_noise(
            flat_action, flat_noise, flat_timestep)
        flat_condition = condition.reshape(
            batch_size * windows, self.n_obs_steps, self.obs_feature_dim)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            diffusion_target = noise
        elif prediction_type == "sample":
            diffusion_target = action
        else:
            raise ValueError(f"unsupported prediction type {prediction_type}")

        if self.p1_mode is P1ExperimentMode.BASELINE:
            prediction = self.model(
                noisy_action, flat_timestep, flat_condition).reshape_as(action)
            per_group = (prediction - diffusion_target).square().mean(
                dim=(1, 2, 3))
            loss = self._masked_group_mean(per_group, group_valid)
            self.last_loss_metrics = {
                "loss_total": float(loss.detach()),
                "loss_diff": float(loss.detach()),
            }
            self.last_probe_rows = []
            return loss, None

        if self.p1_model is None:
            raise AssertionError("latent mode requires p1_model")
        with torch.no_grad():
            split_state = self.model.forward_decoder_pre(
                noisy_action,
                flat_timestep,
                flat_condition,
                split_layer=self.p1_split_layer,
            )
        hidden = split_state.hidden.reshape(
            batch_size, windows, window, -1)
        carry_ratio, beta_overlap, teacher_ratio = self.schedule_values()
        if self.p1_mode is P1ExperimentMode.ORACLE_GATE_SYMBOL:
            gate_mode = (
                P1GateMode.ORACLE
                if self.training
                else P1GateMode.PREDICTED
            )
            history_symbol = batch["history_subtask_idx"]
            window_symbol = batch["subtask_idx"][:, :, 0]
        else:
            gate_mode = P1GateMode.SCHEDULED if self.training else P1GateMode.PREDICTED
            history_symbol = None
            window_symbol = None

        output = self.p1_model(
            hidden=hidden,
            context=context,
            diffusion_timestep=diffusion_timestep,
            physical_time=physical_time,
            episode_length=batch["episode_length"].to(dtype=torch.long),
            history_feature=batch["history_feature"].to(dtype=hidden.dtype),
            history_previous_action=self.normalizer["action"].normalize(
                batch["history_previous_action"]).to(dtype=hidden.dtype),
            history_valid_mask=batch["history_valid_mask"],
            group_to_history=batch["group_to_history"],
            history_symbol=history_symbol,
            window_symbol=window_symbol,
            boundary_target=batch["boundary_edge_target"].to(dtype=hidden.dtype),
            gate_mode=gate_mode,
            teacher_ratio=teacher_ratio,
            carry_ratio=carry_ratio,
        )
        prediction = self.model.forward_decoder_post(
            split_state,
            hidden=output.corrected_hidden.reshape_as(split_state.hidden),
        ).reshape_as(action)
        losses = compute_p1_group_losses(
            prediction=prediction,
            diffusion_target=diffusion_target,
            output=output,
            boundary_target=batch["boundary_edge_target"],
            episode_id=batch["episode_id"],
            physical_time=physical_time,
            group_valid=group_valid,
            stride=self.p1_model.group_config.stride,
            beta_boundary=self.p1_beta_boundary,
            beta_overlap=beta_overlap,
            beta_initialization=self.p1_beta_initialization,
            delta_seconds=self.p1_model.dynamics_config.delta_seconds,
        )
        probability = losses.edge.edge_probability.detach()
        intensity = losses.edge.edge_intensity.detach()
        probability_values = probability
        if probability_values.numel():
            p_mean = float(probability_values.mean())
            p_std = float(probability_values.std(unbiased=False))
            p_q10 = float(torch.quantile(probability_values, 0.10))
            p_q50 = float(torch.quantile(probability_values, 0.50))
            p_q90 = float(torch.quantile(probability_values, 0.90))
        else:
            p_mean = p_std = p_q10 = p_q50 = p_q90 = 0.0
        target = losses.edge.edge_target.detach().to(
            device=probability.device, dtype=probability.dtype)
        positive = target.to(dtype=torch.bool)
        negative = ~positive

        def conditional_mean(value: Tensor, mask: Tensor) -> float:
            if not bool(torch.any(mask)):
                return 0.0
            return float(value[mask].mean())

        feedback = output.corrected_hidden.detach() - hidden.detach()
        feedback_ratio = feedback.norm(dim=-1) / hidden.detach().norm(
            dim=-1).clamp_min(1e-6)
        total_delta = output.z.detach()[:, :, 1:] - output.z.detach()[:, :, :-1]
        z_values = self._masked_probe_values(output.z.detach(), group_valid)
        z_variance = float(z_values.var(unbiased=False)) if z_values.numel() else 0.0
        self.last_loss_metrics = {
            "loss_total": float(losses.total.detach()),
            "loss_diff": float(losses.diffusion.detach()),
            "loss_bnd": float(losses.boundary.detach()),
            "loss_ov": float(losses.overlap.detach()),
            "loss_init": float(losses.initialization.detach()),
            "eta": carry_ratio,
            "beta_ov": beta_overlap,
            "rho": (
                1.0
                if gate_mode is P1GateMode.ORACLE
                else 0.0
                if gate_mode is P1GateMode.PREDICTED
                else teacher_ratio
            ),
            "edge_positive_count": losses.edge.positive_count,
            "edge_negative_count": losses.edge.negative_count,
            "lambda_mean": float(intensity.mean()) if intensity.numel() else 0.0,
            "p_mean": p_mean,
            "p_std": p_std,
            "p_q10": p_q10,
            "p_q50": p_q50,
            "p_q90": p_q90,
            "p_positive_mean": conditional_mean(probability, positive),
            "p_negative_mean": conditional_mean(probability, negative),
            "p_brier": float((probability - target).square().mean())
            if probability.numel() else 0.0,
            "gate_mean": float(self._masked_probe_mean(
                output.gate.detach(), group_valid)),
            "z_norm": float(self._masked_probe_mean(
                output.z.detach().norm(dim=-1), group_valid)),
            "z_variance": z_variance,
            "flow_delta_norm": float(
                self._masked_probe_mean(
                    output.flow_delta.detach().norm(dim=-1), group_valid)),
            "jump_delta_norm": float(
                self._masked_probe_mean(
                    output.jump_delta.detach().norm(dim=-1), group_valid)),
            "total_delta_norm": float(self._masked_probe_mean(
                total_delta.norm(dim=-1), group_valid)),
            "feedback_hidden_ratio": float(self._masked_probe_mean(
                feedback_ratio, group_valid)),
        }
        self.last_probe_rows = [
            {
                "episode_id": int(episode),
                "physical_time": int(time),
                "probability": float(p),
                "intensity": float(lam),
                "target": float(y),
            }
            for episode, time, p, lam, y in zip(
                losses.edge.edge_episode.detach().cpu().tolist(),
                losses.edge.edge_time.detach().cpu().tolist(),
                probability.detach().cpu().tolist(),
                intensity.detach().cpu().tolist(),
                target.detach().cpu().tolist(),
            )
        ]
        return losses.total, losses

    def compute_loss(self, batch: Dict[str, Any]):
        action = batch.get("action")
        if action is not None and action.ndim == 4:
            return self.compute_group_loss(batch)[0]
        if self.p1_mode is not P1ExperimentMode.BASELINE:
            raise ValueError("P1 latent experiments require a group batch")
        return super().compute_loss(batch)

    def reset(self) -> None:
        self._p1_online_history_feature: Tensor | None = None
        self._p1_online_history_action: Tensor | None = None
        self.last_online_probe = {}

    def _p1_encode_online_condition(
            self, obs_dict: Dict[str, Tensor]
        ) -> tuple[Tensor, Tensor | None]:
        raw = dict(obs_dict)
        past_action = raw.pop("past_action", None)
        normalized = self.normalizer.normalize(raw)
        if not normalized:
            raise ValueError("P1 online inference requires observations")
        value = next(iter(normalized.values()))
        if value.ndim < 3 or value.shape[1] < self.n_obs_steps:
            raise ValueError("P1 online observations do not contain n_obs_steps")
        batch = value.shape[0]
        current = dict_apply(
            normalized,
            lambda tensor: tensor[:, :self.n_obs_steps].reshape(
                -1, *tensor.shape[2:]),
        )
        feature = self.obs_encoder(current).reshape(
            batch, self.n_obs_steps, self.obs_feature_dim)
        if past_action is not None:
            if past_action.ndim != 3 or past_action.shape[0] != batch:
                raise ValueError("past_action must have shape [B,L,action_dim]")
            if past_action.shape[-1] != self.action_dim:
                raise ValueError("past_action action_dim does not match policy")
            past_action = self.normalizer["action"].normalize(
                past_action[:, -1]).to(dtype=feature.dtype)
        return feature, past_action

    def _p1_update_online_history(
            self,
            feature: Tensor,
            past_action: Tensor | None,
        ) -> Tensor:
        if self.p1_model is None:
            raise AssertionError("P1 online history requires p1_model")
        latest = feature[:, -1:].detach()
        batch = latest.shape[0]
        if self._p1_online_history_feature is None:
            previous = latest.new_zeros((batch, self.action_dim))
            self._p1_online_history_feature = latest
            self._p1_online_history_action = previous[:, None]
        else:
            if past_action is None:
                raise RuntimeError(
                    "persistent P1 inference requires the actually executed "
                    "past_action after the first physical frame")
            if self._p1_online_history_feature.shape[0] != batch:
                raise RuntimeError("online rollout batch size changed within an episode")
            self._p1_online_history_feature = torch.cat(
                (self._p1_online_history_feature, latest), dim=1)
            self._p1_online_history_action = torch.cat(
                (self._p1_online_history_action, past_action[:, None]), dim=1)
        valid = torch.ones(
            self._p1_online_history_feature.shape[:2],
            device=feature.device,
            dtype=torch.bool,
        )
        latent = self.p1_model.history(
            feature=self._p1_online_history_feature,
            previous_action=self._p1_online_history_action,
            valid_mask=valid,
            oracle_symbol=None,
        )
        return latent[:, -1]

    def _p1_online_conditional_sample(
            self,
            condition_data: Tensor,
            condition_mask: Tensor,
            condition: Tensor,
            initial_z: Tensor,
            generator=None,
        ) -> Tensor:
        if self.p1_model is None:
            raise AssertionError("P1 online sampling requires p1_model")
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        batch = trajectory.shape[0]
        context = condition.reshape(batch, -1)
        if self._p1_online_history_feature is None:
            raise AssertionError("P1 online history was not initialized")
        current_time = self._p1_online_history_feature.shape[1] - 1
        denominator = max(self.p1_online_episode_length - 1, 1)
        normalized_next_time = (
            current_time
            + torch.arange(
                1,
                self.horizon,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
        ) / denominator
        normalized_next_time = normalized_next_time.clamp_max(1).expand(
            batch, -1)
        latest_dynamics = None
        for timestep in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            split_state = self.model.forward_decoder_pre(
                trajectory,
                timestep,
                condition,
                split_layer=self.p1_split_layer,
            )
            batch_timestep = torch.as_tensor(
                timestep,
                device=trajectory.device,
            ).expand(batch).long()
            latest_dynamics = self.p1_model.dynamics(
                initial_z=initial_z,
                hidden=split_state.hidden,
                context=context,
                normalized_tau=(
                    batch_timestep.to(dtype=trajectory.dtype)
                    / max(self.p1_model.group_config.num_diffusion_steps - 1, 1)
                ),
                normalized_next_time=normalized_next_time,
                oracle=None,
                boundary_target=None,
                teacher_ratio=0.0,
            )
            corrected = split_state.hidden + self.p1_model.feedback(
                latest_dynamics.z)
            model_output = self.model.forward_decoder_post(
                split_state,
                hidden=corrected,
            )
            trajectory = scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                **self.kwargs,
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        if latest_dynamics is None:
            raise RuntimeError("P1 online sampling executed zero diffusion steps")
        edge_time = list(range(
            current_time,
            current_time + self.horizon - 1,
        ))
        self.last_online_probe = {
            "physical_time": edge_time,
            "probability": latest_dynamics.probability[0].detach().cpu().tolist(),
            "intensity": latest_dynamics.intensity[0].detach().cpu().tolist(),
            "gate": latest_dynamics.gate[0].detach().cpu().tolist(),
        }
        return trajectory

    def predict_action(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.p1_mode is P1ExperimentMode.BASELINE:
            return super().predict_action(obs_dict)
        if self.p1_model is None:
            raise AssertionError("P1 latent inference requires p1_model")

        feature, past_action = self._p1_encode_online_condition(obs_dict)
        initial_z = self._p1_update_online_history(feature, past_action)
        batch = feature.shape[0]
        shape = (batch, self.horizon, self.action_dim)
        if self.pred_action_steps_only:
            shape = (batch, self.n_action_steps, self.action_dim)
        condition_data = torch.zeros(
            size=shape,
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        nsample = self._p1_online_conditional_sample(
            condition_data,
            condition_mask,
            feature,
            initial_z,
        )
        normalized_action = nsample[..., :self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(normalized_action)
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]
        return {
            "action": action,
            "action_pred": action_pred,
        }
