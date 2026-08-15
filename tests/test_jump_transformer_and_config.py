import hashlib
import copy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.model.diffusion.jump_transformer_for_diffusion import (
    JumpTransformerForDiffusion,
)
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy.workspace.train_jump_diffusion_transformer_hybrid_workspace import (
    TrainJumpDiffusionTransformerHybridWorkspace,
)


def git_blob_hash(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode() + data).hexdigest()


def transformer_kwargs():
    return dict(
        input_dim=4,
        output_dim=4,
        horizon=6,
        n_obs_steps=2,
        cond_dim=5,
        n_layer=4,
        n_head=2,
        n_emb=8,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=1,
    )


def test_jump_split_model_has_exact_state_and_output_parity():
    torch.manual_seed(41)
    vanilla = TransformerForDiffusion(**transformer_kwargs()).eval()
    jump = JumpTransformerForDiffusion(**transformer_kwargs()).eval()
    jump.load_state_dict(vanilla.state_dict(), strict=True)
    assert tuple(jump.state_dict()) == tuple(vanilla.state_dict())
    sample = torch.randn(2, 6, 4)
    condition = torch.randn(2, 2, 5)
    timestep = torch.tensor([3, 7])
    expected = vanilla(sample, timestep, condition)
    state = jump.forward_decoder_pre(
        sample, timestep, condition, split_layer=2)
    actual = jump.forward_decoder_post(state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_vanilla_transformer_is_the_pinned_blob_and_has_no_jump_surface():
    root = Path(__file__).parents[1]
    path = root / "diffusion_policy/model/diffusion/transformer_for_diffusion.py"
    assert git_blob_hash(path.read_bytes()) == (
        "2948be7064d8ceb4b2fe88e83c313c68efeed6a8")
    vanilla_paths = [
        path,
        root / "diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py",
        root / "diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py",
        root / "diffusion_policy/config/train_diffusion_transformer_hybrid_workspace.yaml",
    ]
    for vanilla_path in vanilla_paths:
        text = vanilla_path.read_text(encoding="utf-8").lower()
        assert "jump_module" not in text
        assert "jump_enabled" not in text


def test_hydra_architectures_have_distinct_class_identity():
    root = Path(__file__).parents[1]
    config_dir = str(root.joinpath("diffusion_policy/config").resolve())
    vanilla = OmegaConf.load(
        root / "diffusion_policy/config/train_diffusion_transformer_hybrid_workspace.yaml")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        jump = compose(
            config_name="train_jump_diffusion_transformer_hybrid_workspace")
    assert vanilla.policy._target_.endswith(
        ".DiffusionTransformerHybridImagePolicy")
    assert jump.policy._target_.endswith(
        ".JumpDiffusionTransformerHybridImagePolicy")
    assert jump._target_.endswith(
        ".TrainJumpDiffusionTransformerHybridWorkspace")
    assert jump.architecture_id == "dp_jump"
    assert vanilla.policy._target_ != jump.policy._target_
    assert jump.task.input_path == "cached"
    assert jump.task.history_feature_variant == 0
    assert jump.task.cache_action_n_obs_steps == 2
    assert jump.task.train_dataset.n_obs_steps == jump.n_obs_steps
    assert (
        jump.task.train_dataset.cache_action_n_obs_steps
        == jump.task.cache_action_n_obs_steps
    )
    assert jump.policy.eval_fixed_crop is True
    assert jump.policy.pred_action_steps_only is False
    assert jump.task.env_runner._target_.endswith(
        ".JumpRobomimicImageRunner")
    assert jump.task.env_runner.return_executed_observations is True
    assert jump.task.env_runner.past_action is False
    assert jump.task.env_runner.abs_action is False
    resolved = str(jump).lower()
    for removed in (
        "group_windows",
        "group_stride",
        "carry_ratio",
        "beta_overlap",
        "beta_initialization",
        "oracle_vocab",
        "oracle_dim",
        "static_context",
        "jump_static_context",
    ):
        assert removed not in resolved


@pytest.mark.parametrize(("field", "value", "message"), [
    ("task.history_feature_variant", 1, "center-crop"),
    ("policy.eval_fixed_crop", False, "eval_fixed_crop"),
    ("policy.pred_action_steps_only", True, "full vanilla DP horizon"),
    (
        "task.env_runner.return_executed_observations",
        False,
        "every executed observation",
    ),
    ("task.env_runner.past_action", True, "legacy past_action"),
    ("task.env_runner.abs_action", True, "abs_action=False"),
])
def test_workspace_rejects_train_rollout_contract_drift(field, value, message):
    root = Path(__file__).parents[1]
    config_dir = str(root.joinpath("diffusion_policy/config").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="train_jump_diffusion_transformer_hybrid_workspace")
    config = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(
        config, resolve=False)))
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError, match=message):
        TrainJumpDiffusionTransformerHybridWorkspace \
            ._validate_train_rollout_contract(config)
