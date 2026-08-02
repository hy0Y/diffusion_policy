from pathlib import Path

from omegaconf import OmegaConf
import torch

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.workspace.train_p1_mvp_workspace import TrainP1MVPWorkspace


def workspace_config():
    return OmegaConf.create({
        "training": {"seed": 17, "use_ema": True},
        "optimizer": {
            "transformer_weight_decay": 1.0e-3,
            "obs_encoder_weight_decay": 1.0e-6,
            "p1_weight_decay": 1.0e-3,
            "learning_rate": 1.0e-4,
            "betas": [0.9, 0.95],
        },
        "policy": {
            "_target_": (
                "diffusion_policy.policy."
                "p1_diffusion_transformer_hybrid_image_policy."
                "P1DiffusionTransformerHybridImagePolicy"
            ),
            "shape_meta": {
                "obs": {"state": {"shape": [5], "type": "low_dim"}},
                "action": {"shape": [3]},
            },
            "noise_scheduler": {
                "_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
                "num_train_timesteps": 20,
                "prediction_type": "epsilon",
            },
            "horizon": 6,
            "n_action_steps": 4,
            "n_obs_steps": 2,
            "crop_shape": None,
            "n_layer": 12,
            "n_cond_layers": 1,
            "n_head": 4,
            "n_emb": 32,
            "p_drop_emb": 0.0,
            "p_drop_attn": 0.0,
            "causal_attn": True,
            "time_as_cond": True,
            "obs_as_cond": True,
            "p1_mode": "main",
            "p1_history_backend": "gru",
            "p1_history_model_dim": 7,
            "p1_history_layers": 1,
            "p1_latent_dim": 4,
            "p1_oracle_dim": 3,
            "p1_flow_condition_dim": 5,
            "p1_jump_condition_dim": 5,
            "p1_projector_hidden_dim": 12,
            "p1_dynamics_hidden_dim": 10,
            "p1_feedback_hidden_dim": 9,
            "p1_group_stride": 3,
        },
    })


def prepare_checkpointable_workspace(tmp_path: Path):
    workspace = TrainP1MVPWorkspace(
        workspace_config(), output_dir=str(tmp_path))
    normalizer = SingleFieldLinearNormalizer.create_identity()
    workspace.model.normalizer["action"] = normalizer
    workspace.ema_model.normalizer["action"] = (
        SingleFieldLinearNormalizer.create_identity())
    workspace.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        workspace.optimizer, lambda _: 1.0)
    return workspace


class UnwrappedAccelerator:
    @staticmethod
    def unwrap_model(model):
        return model


def test_workspace_checkpoint_restores_mid_epoch_and_ema_state(tmp_path):
    source = prepare_checkpointable_workspace(tmp_path)
    source.global_step = 13
    source.epoch = 4
    source.batch_in_epoch = 7
    source.episodes_seen = 123
    source.ema_optimization_step = 11
    with torch.no_grad():
        next(source.model.parameters()).fill_(0.25)
        next(source.ema_model.parameters()).fill_(0.5)
    checkpoint = tmp_path / "checkpoints" / "latest.ckpt"
    source._save_training_checkpoint(
        UnwrappedAccelerator(), path=checkpoint)

    restored = prepare_checkpointable_workspace(tmp_path)
    restored.load_checkpoint(path=checkpoint)

    assert restored.global_step == 13
    assert restored.epoch == 4
    assert restored.batch_in_epoch == 7
    assert restored.episodes_seen == 123
    assert restored.ema_optimization_step == 11
    torch.testing.assert_close(
        next(restored.model.parameters()),
        torch.full_like(next(restored.model.parameters()), 0.25),
    )
    torch.testing.assert_close(
        next(restored.ema_model.parameters()),
        torch.full_like(next(restored.ema_model.parameters()), 0.5),
    )


def test_ddp_loss_scale_recovers_real_group_weighted_mean():
    global_groups = torch.tensor(48.0)
    rank0_scale = TrainP1MVPWorkspace._ddp_loss_scale(
        torch.tensor(32.0), global_groups, 2)
    rank1_scale = TrainP1MVPWorkspace._ddp_loss_scale(
        torch.tensor(16.0), global_groups, 2)
    rank0_mean = torch.tensor(2.0)
    rank1_mean = torch.tensor(5.0)

    ddp_gradient_equivalent = (
        rank0_scale * rank0_mean + rank1_scale * rank1_mean) / 2
    expected = (32 * rank0_mean + 16 * rank1_mean) / 48
    torch.testing.assert_close(ddp_gradient_equivalent, expected)


def test_platform_probe_row_exposes_canonical_value_without_mutation():
    source = {
        "episode_id": 4,
        "physical_time": 19,
        "probability": 0.35,
        "intensity": 1.2,
        "target": 0.0,
    }
    payload = TrainP1MVPWorkspace._platform_probe_row(source)

    assert payload["value"] == 0.35
    assert "value" not in source
