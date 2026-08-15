"""Host-neutral composition of condition bridge, dynamics, and feedback."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from jump_module.bridges import HostConditionBridge, HostFeedbackBridge
from jump_module.contracts import JumpAdapterInput, JumpAdapterOutput
from jump_module.core import JumpDynamicsCore
from jump_module.history import CausalHistoryEncoder


class JumpAdapter(nn.Module):
    def __init__(
            self,
            *,
            condition_bridge: HostConditionBridge,
            dynamics: JumpDynamicsCore,
            feedback_bridge: HostFeedbackBridge,
            history_encoder: CausalHistoryEncoder | None = None,
        ):
        super().__init__()
        self.condition_bridge = condition_bridge
        self.dynamics = dynamics
        self.feedback_bridge = feedback_bridge
        self.history_encoder = history_encoder

    def encode_history(
            self,
            *,
            feature: Tensor,
            previous_action: Tensor,
            valid_mask: Tensor,
        ) -> Tensor:
        if self.history_encoder is None:
            raise RuntimeError("this adapter has no history encoder")
        trace = self.history_encoder(
            feature=feature,
            previous_action=previous_action,
            valid_mask=valid_mask,
        )
        return self.history_encoder.last_valid(trace, valid_mask)

    def forward(self, inputs: JumpAdapterInput) -> JumpAdapterOutput:
        flow_condition, jump_condition = self.condition_bridge(
            inputs.action_hidden,
            inputs.condition_context,
            inputs.noise_level,
        )
        dynamics = self.dynamics(
            initial_z=inputs.initial_z,
            flow_condition=flow_condition,
            jump_condition=jump_condition,
            edge_dt_seconds=inputs.edge_dt_seconds,
            edge_valid_mask=inputs.edge_valid_mask,
            boundary_target=inputs.boundary_target,
            teacher_ratio=inputs.teacher_ratio,
            controls=inputs.controls,
        )
        feedback = self.feedback_bridge(dynamics.z)
        scale = torch.as_tensor(
            inputs.controls.feedback_scale,
            device=inputs.action_hidden.device,
            dtype=inputs.action_hidden.dtype,
        )
        if scale.ndim == 0:
            scaled_feedback = feedback * scale
        elif scale.shape in ((inputs.action_hidden.shape[0],),
                             (inputs.action_hidden.shape[0], 1)):
            scaled_feedback = feedback * scale.reshape(-1, 1, 1)
        else:
            raise ValueError("controls.feedback_scale must be scalar or [B]")
        corrected = inputs.action_hidden + scaled_feedback
        return JumpAdapterOutput(
            action_hidden=corrected,
            z=dynamics.z,
            z_minus=dynamics.z_minus,
            flow_delta=dynamics.flow_delta,
            raw_jump_delta=dynamics.raw_jump_delta,
            applied_jump_delta=dynamics.applied_jump_delta,
            intensity=dynamics.intensity,
            probability=dynamics.probability,
            gate=dynamics.gate,
            feedback_residual=scaled_feedback,
        )
