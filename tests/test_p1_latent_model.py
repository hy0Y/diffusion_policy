import torch

from diffusion_policy.model.p1.dynamics import P1DynamicsConfig
from diffusion_policy.model.p1.group import (
    P1GateMode,
    P1GroupConfig,
    P1GroupOutput,
    P1LatentGroupModel,
)
from diffusion_policy.model.p1.history import (
    P1CausalHistoryEncoder,
    P1HistoryConfig,
)
from diffusion_policy.model.p1.losses import compute_p1_group_losses


def small_configs():
    dynamics = P1DynamicsConfig(
        latent_dim=4,
        dit_hidden_dim=8,
        context_dim=6,
        oracle_dim=3,
        flow_condition_dim=5,
        jump_condition_dim=5,
        projector_hidden_dim=12,
        dynamics_hidden_dim=10,
        feedback_hidden_dim=9,
    )
    history = P1HistoryConfig(
        feature_dim=5,
        action_dim=3,
        oracle_vocab_size=6,
        oracle_dim=3,
        model_dim=7,
        latent_dim=4,
        num_layers=1,
        backend="gru",
    )
    group = P1GroupConfig(
        windows=4,
        window_length=6,
        stride=3,
        num_diffusion_steps=20,
    )
    return dynamics, history, group


def group_inputs(*, batch=2):
    generator = torch.Generator().manual_seed(17)
    starts = torch.arange(4) * 3
    physical_time = starts[None, :, None] + torch.arange(6)[None, None, :]
    physical_time = physical_time.expand(batch, -1, -1).clone()
    boundary = torch.zeros(batch, 4, 5)
    boundary[:, 1, 2] = 1
    return {
        "hidden": torch.randn(batch, 4, 6, 8, generator=generator),
        "context": torch.randn(batch, 4, 6, generator=generator),
        "diffusion_timestep": torch.tensor([3, 11])[:batch],
        "physical_time": physical_time,
        "episode_length": torch.full((batch,), 20, dtype=torch.long),
        "history_feature": torch.randn(batch, 16, 5, generator=generator),
        "history_previous_action": torch.randn(batch, 16, 3, generator=generator),
        "history_valid_mask": torch.ones(batch, 16, dtype=torch.bool),
        "group_to_history": torch.arange(batch, dtype=torch.long),
        "history_symbol": torch.randint(0, 6, (batch, 16), generator=generator),
        "window_symbol": torch.randint(0, 6, (batch, 4), generator=generator),
        "boundary_target": boundary,
    }


def test_history_is_causal_under_future_perturbation():
    _, history_config, _ = small_configs()
    model = P1CausalHistoryEncoder(history_config).eval()
    generator = torch.Generator().manual_seed(3)
    feature = torch.randn(1, 12, 5, generator=generator)
    action = torch.randn(1, 12, 3, generator=generator)
    mask = torch.ones(1, 12, dtype=torch.bool)
    first = model(
        feature=feature,
        previous_action=action,
        valid_mask=mask,
    )
    perturbed = feature.clone()
    perturbed[:, 7:] += 100
    second = model(
        feature=perturbed,
        previous_action=action,
        valid_mask=mask,
    )
    torch.testing.assert_close(first[:, :7], second[:, :7], rtol=0, atol=0)


def test_transformers_mamba_reference_is_causal_and_differentiable():
    config = P1HistoryConfig(
        feature_dim=5,
        action_dim=3,
        oracle_vocab_size=6,
        oracle_dim=3,
        model_dim=8,
        latent_dim=4,
        num_layers=1,
        backend="transformers_mamba",
        mamba_state_dim=4,
        mamba_conv_width=2,
        mamba_expand=1,
    )
    model = P1CausalHistoryEncoder(config)
    generator = torch.Generator().manual_seed(29)
    feature = torch.randn(1, 12, 5, generator=generator, requires_grad=True)
    action = torch.randn(1, 12, 3, generator=generator)
    mask = torch.ones(1, 12, dtype=torch.bool)
    first = model(feature=feature, previous_action=action, valid_mask=mask)
    perturbed = feature.detach().clone()
    perturbed[:, 7:] += 100
    second = model(feature=perturbed, previous_action=action, valid_mask=mask)

    torch.testing.assert_close(first[:, :7], second[:, :7], rtol=0, atol=0)
    first.square().mean().backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()


def test_transformers_mamba_fp32_residual_feeds_fp16_output_head():
    class Float32Residual(torch.nn.Module):
        def forward(self, hidden):
            return hidden.float()

    config = P1HistoryConfig(
        feature_dim=5,
        action_dim=3,
        oracle_vocab_size=6,
        oracle_dim=3,
        model_dim=8,
        latent_dim=4,
        num_layers=1,
        backend="transformers_mamba",
        mamba_state_dim=4,
        mamba_conv_width=2,
        mamba_expand=1,
    )
    model = P1CausalHistoryEncoder(config).to(dtype=torch.float16)
    model.sequence_layers = torch.nn.ModuleList([Float32Residual()])
    feature = torch.randn(1, 12, 5, dtype=torch.float16, requires_grad=True)
    action = torch.randn(1, 12, 3, dtype=torch.float16)
    mask = torch.ones(1, 12, dtype=torch.bool)

    output = model(
        feature=feature,
        previous_action=action,
        valid_mask=mask,
    )

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert model.output_norm.weight.grad is not None
    assert torch.isfinite(model.output_norm.weight.grad).all()
    assert model.history_projection.weight.grad is not None
    assert torch.isfinite(model.history_projection.weight.grad).all()


def test_group_oracle_gate_and_differentiable_carry():
    dynamics, history, group = small_configs()
    model = P1LatentGroupModel(
        dynamics_config=dynamics,
        history_config=history,
        group_config=group,
    )
    values = group_inputs()
    output = model(
        **values,
        gate_mode=P1GateMode.ORACLE,
        teacher_ratio=0.0,
        carry_ratio=0.5,
    )
    assert output.z.shape == (2, 4, 6, 4)
    assert output.corrected_hidden.shape == (2, 4, 6, 8)
    torch.testing.assert_close(output.gate, values["boundary_target"])
    torch.testing.assert_close(
        output.carry_start[:, 1], output.z[:, 0, group.stride])

    output.z[:, -1, -1].square().mean().backward()
    first_flow = model.dynamics.flow[0]
    assert first_flow.weight.grad is not None
    assert torch.count_nonzero(first_flow.weight.grad) > 0


def test_groups_can_share_one_episode_history_encoding():
    dynamics, history, group = small_configs()
    model = P1LatentGroupModel(
        dynamics_config=dynamics,
        history_config=history,
        group_config=group,
    )
    values = group_inputs(batch=2)
    values["history_feature"] = values["history_feature"][:1]
    values["history_previous_action"] = values["history_previous_action"][:1]
    values["history_valid_mask"] = values["history_valid_mask"][:1]
    values["history_symbol"] = values["history_symbol"][:1]
    values["group_to_history"] = torch.zeros(2, dtype=torch.long)
    output = model(
        **values,
        gate_mode=P1GateMode.ORACLE,
        teacher_ratio=1.0,
        carry_ratio=0.5,
    )
    assert output.history_start.shape == (2, 4, 4)


def test_predicted_gate_does_not_read_boundary_target():
    dynamics, history, group = small_configs()
    model = P1LatentGroupModel(
        dynamics_config=dynamics,
        history_config=history,
        group_config=group,
    ).eval()
    values = group_inputs(batch=1)
    first = model(
        **values,
        gate_mode=P1GateMode.PREDICTED,
        teacher_ratio=1.0,
        carry_ratio=1.0,
    )
    values["boundary_target"] = 1 - values["boundary_target"]
    second = model(
        **values,
        gate_mode=P1GateMode.PREDICTED,
        teacher_ratio=1.0,
        carry_ratio=1.0,
    )
    torch.testing.assert_close(first.gate, second.gate)
    torch.testing.assert_close(first.z, second.z)


def test_group_loss_aggregates_duplicate_edges_and_masks_dummy_group():
    batch, windows, window, latent = 2, 4, 6, 4
    prediction = torch.zeros(batch, windows, window, 3, requires_grad=True)
    target = torch.ones_like(prediction)
    starts = torch.arange(windows) * 3
    physical = starts[None, :, None] + torch.arange(window)[None, None, :]
    physical = physical.expand(batch, -1, -1).clone()
    zeros_z = torch.zeros(batch, windows, window, latent)
    output = P1GroupOutput(
        corrected_hidden=torch.zeros(batch, windows, window, 8),
        z=zeros_z,
        z_minus=torch.zeros(batch, windows, window - 1, latent),
        flow_delta=torch.zeros(batch, windows, window - 1, latent),
        jump_delta=torch.zeros(batch, windows, window - 1, latent),
        intensity=torch.ones(batch, windows, window - 1, requires_grad=True),
        probability=torch.zeros(batch, windows, window - 1),
        gate=torch.zeros(batch, windows, window - 1),
        history_start=torch.zeros(batch, windows, latent),
        carry_start=torch.zeros(batch, windows, latent),
        carry_history_valid=torch.tensor([[False, True, True, True]] * batch),
    )
    boundary = torch.zeros(batch, windows, window - 1)
    boundary[0, 0, 3] = 1
    boundary[0, 1, 0] = 1  # Same real edge and same target.
    result = compute_p1_group_losses(
        prediction=prediction,
        diffusion_target=target,
        output=output,
        boundary_target=boundary,
        episode_id=torch.tensor([7, 8]),
        physical_time=physical,
        group_valid=torch.tensor([True, False]),
        stride=3,
    )
    assert result.diffusion.item() == 1.0
    assert result.edge.positive_count == 1
    assert result.edge.edge_time.unique().numel() < windows * (window - 1)
    assert int(result.edge.edge_time.min()) == int(physical.min()) + 1
    result.total.backward()
    assert prediction.grad is not None
    assert output.intensity.grad is not None
