"""Semantic diffusion policy that trains from frozen observation features."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from diffusion_policy.policy.semantic_diffusion_transformer_hybrid_image_policy import (
    SemanticDiffusionTransformerHybridImagePolicy,
)


class SemanticFeatureDiffusionTransformerHybridImagePolicy(
    SemanticDiffusionTransformerHybridImagePolicy
):
    """Uses cached encoder outputs for training while retaining raw-observation inference."""

    def _training_inputs(self, batch):
        action = self.normalizer["action"].normalize(batch["action"])
        batch_size = action.shape[0]
        if not self.obs_as_cond:
            raise NotImplementedError("feature policy requires obs_as_cond=True")
        cond = batch["obs_feature"][:, : self.n_obs_steps].to(
            device=action.device, dtype=action.dtype
        )
        expected = (batch_size, self.n_obs_steps, self.obs_feature_dim)
        if tuple(cond.shape) != expected:
            raise ValueError(f"obs_feature shape {tuple(cond.shape)} != {expected}")

        trajectory = action
        if self.pred_action_steps_only:
            start = self.n_obs_steps - 1
            trajectory = action[:, start : start + self.n_action_steps]
        mask = (
            torch.zeros_like(trajectory, dtype=torch.bool)
            if self.pred_action_steps_only
            else self.mask_generator(trajectory.shape)
        )
        noise = torch.randn_like(trajectory)
        timestep = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=trajectory.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timestep)
        noisy[mask] = trajectory[mask]
        return trajectory, noise, timestep, noisy, mask, cond

    def compute_bifurcation_loss(self, batch, auxiliary_scale=1.0):
        if not self.bifurcation_enabled:
            raise RuntimeError("bifurcation branch disabled")
        batch_size = min(
            self.probe_batch_size_per_device, batch["obs_feature"].shape[0]
        )
        cond = batch["obs_feature"][:batch_size, : self.n_obs_steps].to(
            dtype=self.dtype
        )
        label = {
            key: value[:batch_size] for key, value in batch["boundary_target"].items()
        }
        was_training = self.model.training
        self.model.eval()
        probes = self._reverse_probes(cond, self.m_train)
        repeated = cond.repeat_interleave(self.m_train, dim=0)
        scores = []
        for value in self.probing_timesteps:
            _, hidden = self.model(
                probes[value],
                torch.tensor(value, device=cond.device),
                repeated,
                return_hidden=True,
            )
            logits = self._semantic_logits(hidden)
            subtask = logits["subtask"].reshape(batch_size, self.m_train, -1)
            stage = logits["stage"].reshape(batch_size, self.m_train, -1)
            scores.append(0.5 * (self._js(subtask) + self._js(stage)))
        per_timestep = torch.stack(scores, dim=-1)
        temperature = self.bifurcation_temperature
        score = temperature * (
            torch.logsumexp(per_timestep / temperature, dim=-1)
            - math.log(len(self.probing_timesteps))
        )
        logits = self.bifurcation_calibrator(score[:, None]).squeeze(-1)
        valid = label["valid"].bool()
        if torch.any(valid):
            loss = F.binary_cross_entropy_with_logits(
                logits[valid],
                label["any_future_h8"][valid].to(logits.dtype),
                pos_weight=self.boundary_pos_weight[2],
            )
            accuracy = (
                (logits[valid] >= 0)
                == (label["any_future_h8"][valid] >= 0.5)
            ).float().mean()
        else:
            loss = logits.sum() * 0
            accuracy = logits.detach().sum() * 0
        self.model.train(was_training)
        return {
            "loss_bifurcation": auxiliary_scale * self.lambda_bound * loss,
            "loss_bound_b": loss,
            "bifurcation_score_mean": score.mean(),
            "boundary_b_accuracy": accuracy,
        }

    @torch.no_grad()
    def predict_bifurcation_features(self, obs_feature, sample_count=None):
        """Runs reverse-time probing directly from cached observation features."""
        if not self.bifurcation_enabled:
            raise RuntimeError("bifurcation branch disabled")
        count = self.m_eval if sample_count is None else int(sample_count)
        if count < 2:
            raise ValueError("sample_count must be at least 2")
        cond = obs_feature[:, : self.n_obs_steps].to(dtype=self.dtype)
        batch_size = cond.shape[0]
        was_training = self.model.training
        self.model.eval()
        probes = self._reverse_probes(cond, count)
        repeated = cond.repeat_interleave(count, dim=0)
        scores = []
        subtask_scores = []
        stage_scores = []
        for value in self.probing_timesteps:
            _, hidden = self.model(
                probes[value], torch.tensor(value, device=cond.device), repeated,
                return_hidden=True,
            )
            logits = self._semantic_logits(hidden)
            subtask = logits["subtask"].reshape(batch_size, count, -1)
            stage = logits["stage"].reshape(batch_size, count, -1)
            subtask_score = self._js(subtask)
            stage_score = self._js(stage)
            subtask_scores.append(subtask_score)
            stage_scores.append(stage_score)
            scores.append(0.5 * (subtask_score + stage_score))

        per_timestep = torch.stack(scores, dim=-1)
        subtask_per_timestep = torch.stack(subtask_scores, dim=-1)
        stage_per_timestep = torch.stack(stage_scores, dim=-1)
        temperature = self.bifurcation_temperature
        score = temperature * (
            torch.logsumexp(per_timestep / temperature, dim=-1)
            - math.log(len(self.probing_timesteps))
        )
        logit = self.bifurcation_calibrator(score[:, None]).squeeze(-1)
        critical_index = per_timestep.argmax(dim=-1)
        probing_timesteps = torch.as_tensor(
            self.probing_timesteps, device=score.device, dtype=torch.long
        )
        critical_timestep = probing_timesteps[critical_index]
        self.model.train(was_training)
        return {
            "probability": torch.sigmoid(logit),
            "logit": logit,
            "score": score,
            "per_timestep_score": per_timestep,
            "subtask_per_timestep_score": subtask_per_timestep,
            "stage_per_timestep_score": stage_per_timestep,
            "critical_timestep": critical_timestep,
            "sample_count": count,
        }
