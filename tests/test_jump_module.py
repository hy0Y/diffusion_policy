import ast
from pathlib import Path

import pytest
import torch

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


def make_adapter():
    return JumpAdapter(
        condition_bridge=HostConditionBridge(HostConditionBridgeConfig(
            host_hidden_dim=7,
            context_dim=5,
            flow_condition_dim=4,
            jump_condition_dim=4,
            hidden_dim=9,
        )),
        dynamics=JumpDynamicsCore(JumpDynamicsConfig(
            latent_dim=3,
            flow_condition_dim=4,
            jump_condition_dim=4,
            hidden_dim=8,
        )),
        feedback_bridge=HostFeedbackBridge(HostFeedbackBridgeConfig(
            latent_dim=3,
            host_hidden_dim=7,
            hidden_dim=6,
        )),
        history_encoder=CausalHistoryEncoder(CausalHistoryConfig(
            feature_dim=6,
            action_dim=2,
            model_dim=8,
            latent_dim=3,
            num_layers=1,
            backend="selective_ssm",
            state_dim=4,
            conv_width=3,
            expand=2,
        )),
    )


def make_input(*, boundary=True, teacher_ratio=0.5, controls=JumpControls()):
    generator = torch.Generator().manual_seed(13)
    return JumpAdapterInput(
        action_hidden=torch.randn(2, 5, 7, generator=generator),
        initial_z=torch.randn(2, 3, generator=generator),
        condition_context=torch.randn(2, 5, generator=generator),
        noise_level=torch.tensor([0.2, 0.8]),
        edge_dt_seconds=torch.full((2, 4), 0.05),
        edge_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        boundary_target=(
            torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
            if boundary else None),
        teacher_ratio=teacher_ratio,
        controls=controls,
    )


def test_right_continuous_order_probability_and_zero_init_feedback():
    adapter = make_adapter()
    output = adapter(make_input())
    torch.testing.assert_close(
        output.z[:, 1:], output.z_minus + output.applied_jump_delta)
    assert bool(torch.all(output.intensity > 0))
    expected_probability = -torch.expm1(
        -output.intensity * make_input().edge_dt_seconds)
    torch.testing.assert_close(output.probability, expected_probability)
    torch.testing.assert_close(
        output.action_hidden, make_input().action_hidden, rtol=0, atol=0)
    assert output.feedback_residual.abs().max().item() == 0


def test_probability_increases_with_dt_at_fixed_pre_jump_state():
    core = JumpDynamicsCore(JumpDynamicsConfig(3, 4, 4, 8))
    initial = torch.randn(2, 3)
    flow = torch.randn(2, 1, 4)
    jump = torch.randn(2, 1, 4)
    common = dict(
        initial_z=initial,
        flow_condition=flow,
        jump_condition=jump,
        edge_valid_mask=torch.ones(2, 1, dtype=torch.bool),
        controls=JumpControls(flow_scale=0.0, jump_scale=0.0),
    )
    short = core(edge_dt_seconds=torch.tensor(0.05), **common)
    long = core(edge_dt_seconds=torch.tensor(0.10), **common)
    torch.testing.assert_close(short.intensity, long.intensity)
    assert bool(torch.all(long.probability > short.probability))


def test_interventions_are_independent_and_invalid_edges_hold_state():
    adapter = make_adapter()
    with torch.no_grad():
        adapter.feedback_bridge.network[-1].weight.fill_(0.2)
    baseline = adapter(make_input())
    feedback_zero = adapter(make_input(controls=JumpControls(feedback_scale=0.0)))
    torch.testing.assert_close(feedback_zero.z, baseline.z)
    torch.testing.assert_close(feedback_zero.probability, baseline.probability)
    assert feedback_zero.feedback_residual.abs().max().item() == 0

    jump_zero = adapter(make_input(controls=JumpControls(jump_scale=0.0)))
    assert jump_zero.raw_jump_delta.abs().sum().item() > 0
    assert jump_zero.probability.abs().sum().item() > 0
    assert jump_zero.applied_jump_delta.abs().max().item() == 0

    flow_zero = adapter(make_input(controls=JumpControls(flow_scale=0.0)))
    assert flow_zero.flow_delta.abs().max().item() == 0
    assert flow_zero.raw_jump_delta.abs().sum().item() > 0

    data = make_input()
    valid = data.edge_valid_mask.clone()
    valid[:, 1] = False
    target = data.boundary_target.clone()
    target[:, 1] = torch.nan
    held = adapter(JumpAdapterInput(
        action_hidden=data.action_hidden,
        initial_z=data.initial_z,
        condition_context=data.condition_context,
        noise_level=data.noise_level,
        edge_dt_seconds=data.edge_dt_seconds,
        edge_valid_mask=valid,
        boundary_target=target,
        teacher_ratio=1.0,
    ))
    torch.testing.assert_close(held.z[:, 2], held.z[:, 1])
    assert held.flow_delta[:, 1].abs().max().item() == 0
    assert held.applied_jump_delta[:, 1].abs().max().item() == 0
    assert bool(torch.all(torch.isfinite(held.z)))


def test_predicted_gate_never_requires_boundary_target():
    output = make_adapter()(make_input(boundary=False, teacher_ratio=0.0))
    torch.testing.assert_close(output.gate, output.probability)
    with pytest.raises(ValueError, match="required"):
        make_adapter()(make_input(boundary=False, teacher_ratio=0.1))


@pytest.mark.parametrize("backend", ["gru", "selective_ssm"])
def test_history_is_causal_and_uses_previous_action(backend):
    torch.manual_seed(5)
    encoder = CausalHistoryEncoder(CausalHistoryConfig(
        feature_dim=4,
        action_dim=2,
        model_dim=8,
        latent_dim=3,
        num_layers=1,
        backend=backend,
        state_dim=4,
        conv_width=3,
        expand=2,
    )).eval()
    feature = torch.randn(2, 7, 4)
    action = torch.randn(2, 7, 2)
    valid = torch.ones(2, 7, dtype=torch.bool)
    reference = encoder(
        feature=feature, previous_action=action, valid_mask=valid)
    changed_feature = feature.clone()
    changed_feature[:, 5:] += 100
    changed_action = action.clone()
    changed_action[:, 5:] -= 100
    changed = encoder(
        feature=changed_feature,
        previous_action=changed_action,
        valid_mask=valid)
    torch.testing.assert_close(reference[:, :5], changed[:, :5])
    assert not torch.allclose(reference[:, 5:], changed[:, 5:])


@pytest.mark.parametrize("backend", ["gru", "selective_ssm"])
def test_history_uses_the_full_feature_at_every_timestep(backend):
    torch.manual_seed(17)
    encoder = CausalHistoryEncoder(CausalHistoryConfig(
        feature_dim=7,
        action_dim=2,
        model_dim=8,
        latent_dim=3,
        num_layers=1,
        backend=backend,
        state_dim=4,
        conv_width=3,
        expand=2,
    )).eval()
    feature = torch.randn(2, 5, 7)
    action = torch.randn(2, 5, 2)
    valid = torch.ones(2, 5, dtype=torch.bool)
    reference = encoder(
        feature=feature,
        previous_action=action,
        valid_mask=valid,
    )
    changed_feature = feature.clone()
    changed_feature[:, 3:, 4:] += 1
    changed = encoder(
        feature=changed_feature,
        previous_action=action,
        valid_mask=valid,
    )
    assert reference.shape == (2, 5, 3)
    assert encoder.input_projection.in_features == 7 + 2
    torch.testing.assert_close(reference[:, :3], changed[:, :3])
    assert not torch.allclose(reference[:, 3:], changed[:, 3:])


@pytest.mark.parametrize("backend", ["gru", "selective_ssm"])
def test_padding_invalid_tail_preserves_the_training_prefix_state(backend):
    torch.manual_seed(23)
    encoder = CausalHistoryEncoder(CausalHistoryConfig(
        feature_dim=7,
        action_dim=2,
        model_dim=8,
        latent_dim=3,
        num_layers=1,
        backend=backend,
        state_dim=4,
        conv_width=3,
        expand=2,
    )).eval()
    feature = torch.randn(2, 5, 7)
    action = torch.randn(2, 5, 2)
    training_valid = torch.ones(2, 4, dtype=torch.bool)
    training_trace = encoder(
        feature=feature[:, :4],
        previous_action=action[:, :4],
        valid_mask=training_valid,
    )
    rollout_valid = torch.tensor([
        [True, True, True, True, False],
        [True, True, True, True, False],
    ])
    rollout_trace = encoder(
        feature=feature,
        previous_action=action,
        valid_mask=rollout_valid,
    )
    torch.testing.assert_close(
        encoder.last_valid(training_trace, training_valid),
        encoder.last_valid(rollout_trace, rollout_valid),
        rtol=0,
        atol=0,
    )


def test_boundary_loss_masks_sentinels_and_counts_views_without_dedup():
    intensity = torch.tensor([[1.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    target = torch.tensor([[1.0, float("nan"), 0.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, False, True], [True, True, True]])
    output = masked_balanced_boundary_loss(
        intensity=intensity,
        target=target,
        edge_dt_seconds=torch.full_like(intensity, 0.05),
        valid_mask=valid,
    )
    assert torch.isfinite(output.loss)
    assert output.positive_count == 2
    assert output.negative_count == 3
    assert output.probability.numel() == 5


def test_portable_package_import_graph_has_no_host_dependency():
    root = Path(__file__).parents[1] / "jump_module"
    forbidden = {"diffusion_policy", "diffusers", "gr00t", "robocasa"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden, (path, imported & forbidden)
