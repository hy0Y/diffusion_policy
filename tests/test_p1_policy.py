import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.policy.p1_diffusion_transformer_hybrid_image_policy import (
    P1DiffusionTransformerHybridImagePolicy,
)


def make_policy(mode: str):
    policy = P1DiffusionTransformerHybridImagePolicy(
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
        crop_shape=None,
        n_layer=12,
        n_cond_layers=1,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        p1_mode=mode,
        p1_history_backend="gru",
        p1_history_model_dim=7,
        p1_history_layers=1,
        p1_latent_dim=4,
        p1_oracle_dim=3,
        p1_flow_condition_dim=5,
        p1_jump_condition_dim=5,
        p1_projector_hidden_dim=12,
        p1_dynamics_hidden_dim=10,
        p1_feedback_hidden_dim=9,
        p1_group_stride=3,
    )
    policy.normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    return policy


def make_batch(batch_size=2):
    generator = torch.Generator().manual_seed(91)
    starts = torch.arange(4) * 3
    physical = starts[None, :, None] + torch.arange(6)[None, None, :]
    physical = physical.expand(batch_size, -1, -1).clone()
    boundary = torch.zeros(batch_size, 4, 5)
    boundary[:, 1, 2] = 1
    return {
        "episode_id": torch.arange(batch_size),
        "episode_valid": torch.ones(batch_size, dtype=torch.bool),
        "physical_time": physical,
        "obs_feature": torch.randn(
            batch_size, 4, 2, 5, generator=generator),
        "action": torch.randn(batch_size, 4, 6, 3, generator=generator),
        "boundary_edge_target": boundary,
        "subtask_idx": torch.randint(
            0, 6, (batch_size, 4, 6), generator=generator),
        "history_feature": torch.randn(
            batch_size, 16, 5, generator=generator),
        "history_previous_action": torch.randn(
            batch_size, 16, 3, generator=generator),
        "history_subtask_idx": torch.randint(
            0, 6, (batch_size, 16), generator=generator),
        "history_valid_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "group_to_history": torch.arange(batch_size, dtype=torch.long),
        "episode_length": torch.full((batch_size,), 20, dtype=torch.long),
    }


def test_shared_noise_is_exact_on_overlap():
    action = torch.zeros(1, 4, 6, 3)
    physical = make_batch(batch_size=1)["physical_time"]
    noise = P1DiffusionTransformerHybridImagePolicy._shared_group_noise(
        action, physical)
    torch.testing.assert_close(noise[:, 0, 3:], noise[:, 1, :3], rtol=0, atol=0)
    torch.testing.assert_close(noise[:, 1, 3:], noise[:, 2, :3], rtol=0, atol=0)


def test_baseline_trains_backbone_and_uses_group_batch():
    policy = make_policy("baseline")
    assert all(parameter.requires_grad for parameter in policy.model.parameters())
    loss, details = policy.compute_group_loss(make_batch())
    assert details is None
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.model.parameters())


def test_oracle_freezes_backbone_and_trains_only_p1_path():
    policy = make_policy("oracle_gate_symbol")
    policy.train()
    assert not policy.model.training
    assert not policy.obs_encoder.training
    assert all(not parameter.requires_grad for parameter in policy.model.parameters())
    assert all(not parameter.requires_grad for parameter in policy.obs_encoder.parameters())
    loss, details = policy.compute_group_loss(make_batch())
    assert details is not None
    loss.backward()
    assert all(parameter.grad is None for parameter in policy.model.parameters())
    assert any(parameter.grad is not None for parameter in policy.p1_model.parameters())
    assert policy.last_loss_metrics["rho"] == 1.0
    for key in (
        "lambda_mean", "p_q10", "p_q50", "p_q90", "p_brier",
        "z_variance", "total_delta_norm", "feedback_hidden_ratio",
    ):
        assert key in policy.last_loss_metrics
        assert torch.isfinite(torch.tensor(policy.last_loss_metrics[key]))


def test_main_schedule_reaches_predicted_gate():
    policy = make_policy("main")
    assert all(
        not parameter.requires_grad
        for parameter in policy.p1_model.history.oracle_embedding.parameters()
    )
    policy.set_training_progress(1.0)
    policy.train()
    _, details = policy.compute_group_loss(make_batch())
    assert details is not None
    assert policy.last_loss_metrics["rho"] == 0.0
    assert policy.last_loss_metrics["eta"] == 0.5
    assert policy.last_loss_metrics["beta_ov"] == policy.p1_beta_overlap_max


def test_all_dummy_rank_has_finite_zero_weight_probes():
    policy = make_policy("oracle_gate_symbol")
    batch = make_batch()
    batch["episode_valid"].zero_()
    loss, _ = policy.compute_group_loss(batch)

    assert loss.item() == 0.0
    for value in policy.last_loss_metrics.values():
        assert torch.isfinite(torch.tensor(value))
