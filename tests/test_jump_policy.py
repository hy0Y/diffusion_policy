import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.policy.jump_diffusion_transformer_hybrid_image_policy import (
    JumpDiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from jump_module import JumpControls


def policy_kwargs():
    return dict(
        shape_meta={
            "obs": {"state": {"shape": [5], "type": "low_dim"}},
            "action": {"shape": [3]},
        },
        noise_scheduler=DDPMScheduler(
            num_train_timesteps=20,
            prediction_type="epsilon",
        ),
        horizon=6,
        n_action_steps=4,
        n_obs_steps=2,
        num_inference_steps=2,
        crop_shape=None,
        n_layer=4,
        n_cond_layers=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
    )


def make_policy(beta_boundary=0.1, **overrides):
    kwargs = policy_kwargs()
    kwargs.update(overrides)
    policy = JumpDiffusionTransformerHybridImagePolicy(
        **kwargs,
        jump_split_layer=2,
        jump_history_backend="gru",
        jump_history_model_dim=8,
        jump_history_layers=1,
        jump_latent_dim=4,
        jump_flow_condition_dim=5,
        jump_condition_dim=5,
        jump_projector_hidden_dim=12,
        jump_dynamics_hidden_dim=10,
        jump_feedback_hidden_dim=9,
        jump_beta_boundary=beta_boundary,
    )
    policy.normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.normalizer["state"] = SingleFieldLinearNormalizer.create_identity()
    return policy


def make_batch(batch_size=2):
    generator = torch.Generator().manual_seed(91)
    return {
        "obs_feature": torch.randn(batch_size, 2, 5, generator=generator),
        "action": torch.randn(batch_size, 6, 3, generator=generator),
        "action_valid_mask": torch.ones(batch_size, 6, dtype=torch.bool),
        "boundary_target": torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32),
        "edge_valid_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "edge_dt_seconds": torch.full((batch_size, 4), 0.05),
        "history_feature": torch.randn(
            batch_size, 7, 5, generator=generator),
        "history_previous_action": torch.randn(
            batch_size, 7, 3, generator=generator),
        "history_valid_mask": torch.ones(batch_size, 7, dtype=torch.bool),
    }


def test_policy_freezes_host_and_trains_only_jump_namespace():
    policy = make_policy()
    policy.train()
    audit = policy.trainable_parameter_audit()
    assert not policy.model.training
    assert not policy.obs_encoder.training
    assert all(name.startswith("jump_adapter.") for name in audit["trainable_names"])
    loss = policy.compute_loss(make_batch(), diffusion_timestep_override=7)
    loss.backward()
    assert all(parameter.grad is None for parameter in policy.model.parameters())
    assert any(
        parameter.grad is not None
        for parameter in policy.jump_adapter.parameters())
    assert set(policy.last_loss_metrics) >= {
        "loss_total", "loss_diffusion", "loss_boundary", "teacher_ratio"}


def test_action_loss_reaches_history_flow_and_jump_after_feedback_opens():
    policy = make_policy(beta_boundary=0.0)
    with torch.no_grad():
        policy.jump_adapter.feedback_bridge.network[-1].weight.fill_(0.01)
    loss = policy.compute_loss(make_batch(), diffusion_timestep_override=7)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in policy.jump_adapter.history_encoder.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in policy.jump_adapter.dynamics.flow.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in policy.jump_adapter.dynamics.jump.parameters())


def test_feedback_zero_has_exact_vanilla_noise_prediction_parity():
    torch.manual_seed(22)
    vanilla = DiffusionTransformerHybridImagePolicy(**policy_kwargs()).eval()
    jump = make_policy().eval()
    jump.model.load_state_dict(vanilla.model.state_dict(), strict=True)
    sample = torch.randn(2, 6, 3)
    condition = torch.randn(2, 2, 5)
    timestep = torch.tensor([3, 7])
    initial = torch.randn(2, 4)
    expected = vanilla.model(sample, timestep, condition)
    actual, output = jump._jump_model_output(
        noisy_action=sample,
        timestep=timestep,
        condition=condition,
        initial_z=initial,
        edge_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        edge_dt_seconds=torch.full((2, 4), 0.05),
        boundary_target=None,
        teacher_ratio=0.0,
        controls=JumpControls(feedback_scale=0.0),
    )
    torch.testing.assert_close(output.action_hidden, output.action_hidden - output.feedback_residual)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_online_history_appends_every_valid_executed_segment_step():
    policy = make_policy().eval()
    first_feature = torch.randn(1, 2, 5)
    policy.predict_action_from_cached_feature(first_feature)
    assert policy._jump_history_feature.shape[1] == 1
    with pytest.raises(RuntimeError, match="complete executed segment"):
        policy.predict_action_from_cached_feature(torch.randn(1, 2, 5))

    executed_feature = torch.randn(1, 3, 5)
    executed_action = torch.tensor([[
        [0.3, -0.2, 0.1],
        [0.4, -0.1, 0.2],
        [0.5, 0.0, 0.3],
    ]])
    policy.predict_action_from_cached_feature(
        executed_feature[:, -2:],
        executed_feature=executed_feature,
        executed_action=executed_action,
        executed_valid_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert policy._jump_history_feature.shape[1] == 4
    assert policy._jump_history_valid_mask.tolist() == [[True, True, True, True]]
    torch.testing.assert_close(
        policy._jump_history_feature[:, -3:], executed_feature)
    torch.testing.assert_close(
        policy._jump_history_previous_action[:, -3:], executed_action)

    padded_feature = torch.randn(1, 4, 5)
    padded_action = torch.randn(1, 4, 3)
    policy.predict_action_from_cached_feature(
        padded_feature[:, :2],
        executed_feature=padded_feature,
        executed_action=padded_action,
        executed_valid_mask=torch.tensor([[True, True, False, False]]),
    )
    assert policy._jump_history_feature.shape[1] == 6
    assert policy._jump_history_valid_mask.tolist() == [[
        True, True, True, True, True, True,
    ]]
    torch.testing.assert_close(
        policy._jump_history_feature[:, -2:], padded_feature[:, :2])
    assert "carry_ratio" not in policy.last_jump_trace


def test_single_step_segments_immediately_advance_the_current_state():
    policy = make_policy().eval()
    initial = torch.randn(1, 2, 5)
    policy.predict_action_from_cached_feature(initial)

    first_observation = torch.randn(1, 1, 5)
    first_action = torch.randn(1, 1, 3)
    policy.predict_action_from_cached_feature(
        first_observation.expand(-1, 2, -1),
        executed_feature=first_observation,
        executed_action=first_action,
        executed_valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    assert policy._jump_history_feature.shape[1] == 2
    assert policy._jump_history_valid_mask.tolist() == [[True, True]]

    second_observation = torch.randn(1, 1, 5)
    second_action = torch.randn(1, 1, 3)
    policy.predict_action_from_cached_feature(
        second_observation.expand(-1, 2, -1),
        executed_feature=second_observation,
        executed_action=second_action,
        executed_valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    assert policy._jump_history_feature.shape[1] == 3
    assert policy._jump_history_valid_mask.tolist() == [[True, True, True]]
    torch.testing.assert_close(
        policy._jump_history_feature[:, 1], first_observation[:, 0])
    torch.testing.assert_close(
        policy._jump_history_previous_action[:, 1], first_action[:, 0])


@pytest.mark.parametrize(("n_obs_steps", "expected_mask"), [
    (1, [[True, True, True, True, True]]),
    (3, [[True, True, True, True, True]]),
])
def test_every_observed_pair_is_valid_for_any_configured_observation_count(
        n_obs_steps, expected_mask):
    policy = make_policy(
        n_obs_steps=n_obs_steps,
        n_action_steps=min(4, 6 - n_obs_steps + 1),
    ).eval()
    initial = torch.randn(1, n_obs_steps, 5)
    policy.predict_action_from_cached_feature(initial)

    executed_feature = torch.randn(1, 4, 5)
    executed_action = torch.randn(1, 4, 3)
    current = torch.randn(1, n_obs_steps, 5)
    current[:, -1] = executed_feature[:, -1]
    policy.predict_action_from_cached_feature(
        current,
        executed_feature=executed_feature,
        executed_action=executed_action,
        executed_valid_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert policy._jump_history_feature.shape[1] == 5
    assert policy._jump_history_valid_mask.tolist() == expected_mask


def test_raw_inference_uses_recent_configured_obs_and_full_segment():
    policy = make_policy().eval()
    current = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    policy.predict_action({"state": current})
    torch.testing.assert_close(
        policy._jump_history_feature[:, 0], current[:, -1])

    executed_state = torch.arange(
        100, 115, dtype=torch.float32).reshape(1, 3, 5)
    executed_action = torch.randn(1, 3, 3)
    next_current = torch.arange(
        200, 220, dtype=torch.float32).reshape(1, 4, 5)
    next_current[:, -1] = executed_state[:, -1]
    policy.predict_action({
        "state": next_current,
        "executed_obs": {"state": executed_state},
        "executed_action": executed_action,
        "executed_valid_mask": torch.ones(1, 3, dtype=torch.bool),
    })
    assert policy._jump_history_feature.shape[1] == 4
    assert policy._jump_history_valid_mask.tolist() == [[True, True, True, True]]
    torch.testing.assert_close(
        policy._jump_history_feature[:, -3:], executed_state)


def test_training_and_rollout_use_identical_full_history_tokens(monkeypatch):
    kwargs = policy_kwargs()
    kwargs["shape_meta"] = {
        "obs": {
            "state": {"shape": [5], "type": "low_dim"},
            "lang_emb": {"shape": [3], "type": "low_dim"},
        },
        "action": {"shape": [3]},
    }
    policy = JumpDiffusionTransformerHybridImagePolicy(
        **kwargs,
        jump_split_layer=2,
        jump_history_backend="gru",
        jump_history_model_dim=8,
        jump_history_layers=1,
        jump_latent_dim=4,
        jump_flow_condition_dim=5,
        jump_condition_dim=5,
        jump_projector_hidden_dim=12,
        jump_dynamics_hidden_dim=10,
        jump_feedback_hidden_dim=9,
    )
    action_scale = torch.tensor([2.0, 3.0, 4.0])
    action_offset = torch.tensor([0.5, -0.5, 1.0])
    policy.normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
        action_scale, action_offset, {})
    policy.normalizer["state"] = SingleFieldLinearNormalizer.create_identity()
    policy.normalizer["lang_emb"] = (
        SingleFieldLinearNormalizer.create_identity())
    history_config = policy.jump_adapter.history_encoder.config
    assert history_config.feature_dim == 8
    assert policy.jump_adapter.history_encoder.input_projection.in_features == 11

    dynamic = torch.randn(1, 5, 5)
    language = torch.tensor([[[1.0, 2.0, 3.0]]]).expand(1, 5, 3)
    observed_feature = torch.cat((dynamic, language), dim=-1)
    training_feature = observed_feature
    executed_action = torch.randn(1, 4, 3)
    previous_action = torch.cat((
        torch.zeros(1, 1, 3),
        executed_action,
    ), dim=1)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)

    captured = []
    history_encoder = policy.jump_adapter.history_encoder

    def capture_history_input(module, args, kwargs):
        captured.append({
            key: kwargs[key].detach().clone()
            for key in ("feature", "previous_action", "valid_mask")
        })

    handle = history_encoder.register_forward_pre_hook(
        capture_history_input, with_kwargs=True)
    training_initial_z = policy._history_initial_state({
        "history_feature": training_feature,
        "history_previous_action": previous_action,
        "history_valid_mask": valid_mask,
    })
    training_input = captured[-1]

    def history_only_prediction(
            feature,
            executed_feature=None,
            executed_action=None,
            executed_valid_mask=None,
        ):
        initial_z = policy._append_executed_history(
            feature,
            executed_feature,
            executed_action,
            executed_valid_mask,
        )
        return {"initial_z": initial_z}

    monkeypatch.setattr(policy, "_prediction_from_feature", history_only_prediction)
    policy.reset()
    policy.predict_action_from_cached_feature(
        observed_feature[:, :1].expand(-1, policy.n_obs_steps, -1))
    rollout_result = policy.predict_action_from_cached_feature(
        observed_feature[:, -2:],
        executed_feature=observed_feature[:, 1:],
        executed_action=executed_action,
        executed_valid_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    rollout_input = captured[-1]
    handle.remove()

    assert rollout_input["valid_mask"].tolist() == [[
        True, True, True, True, True,
    ]]
    for key in training_input:
        torch.testing.assert_close(
            training_input[key], rollout_input[key], rtol=0, atol=0)
    torch.testing.assert_close(
        training_initial_z, rollout_result["initial_z"], rtol=0, atol=0)
    torch.testing.assert_close(
        rollout_input["feature"][..., -3:], language, rtol=0, atol=0)


def test_jump_feedback_begins_at_the_first_executed_action_token():
    torch.manual_seed(103)
    vanilla = DiffusionTransformerHybridImagePolicy(**policy_kwargs()).eval()
    jump = make_policy().eval()
    jump.model.load_state_dict(vanilla.model.state_dict(), strict=True)
    with torch.no_grad():
        jump.jump_adapter.feedback_bridge.network[-1].bias[0] = 0.5
    sample = torch.randn(2, 6, 3)
    condition = torch.randn(2, 2, 5)
    timestep = torch.tensor([3, 7])
    initial = torch.randn(2, 4)
    expected = vanilla.model(sample, timestep, condition)
    actual, output = jump._jump_model_output(
        noisy_action=sample,
        timestep=timestep,
        condition=condition,
        initial_z=initial,
        edge_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        edge_dt_seconds=torch.full((2, 4), 0.05),
        boundary_target=None,
        teacher_ratio=0.0,
    )
    assert jump.jump_action_start == jump.n_obs_steps - 1 == 1
    assert output.action_hidden.shape[1] == 5
    assert output.z.shape[1] == 5
    assert output.intensity.shape[1] == 4
    assert output.feedback_residual.abs().sum().item() > 0
    torch.testing.assert_close(
        actual[:, :jump.jump_action_start],
        expected[:, :jump.jump_action_start],
        rtol=1e-6,
        atol=1e-7,
    )
    assert not torch.allclose(
        actual[:, jump.jump_action_start:],
        expected[:, jump.jump_action_start:],
    )


def test_all_denoising_steps_reuse_one_causal_initial_state(monkeypatch):
    policy = make_policy().eval()
    captured = []
    original = policy.jump_adapter.forward

    def record(inputs):
        captured.append(inputs.initial_z.detach().clone())
        return original(inputs)

    monkeypatch.setattr(policy.jump_adapter, "forward", record)
    policy.predict_action_from_cached_feature(torch.randn(1, 2, 5))
    assert len(captured) == policy.num_inference_steps
    for value in captured[1:]:
        torch.testing.assert_close(value, captured[0], rtol=0, atol=0)
