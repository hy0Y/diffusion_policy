from pathlib import Path

from hydra import compose, initialize_config_dir


def compose_config(name):
    config_dir = str(
        (Path(__file__).resolve().parents[1] / "diffusion_policy" / "config")
        .resolve()
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=name)


def test_vanilla_and_jump_share_physical_evaluator_contract():
    vanilla = compose_config("train_dp_gettoastedbread_milestone_v3")
    jump = compose_config("train_jump_gettoastedbread_milestone_v3")
    vanilla_runner = vanilla.task.env_runner
    jump_runner = jump.task.env_runner

    assert vanilla_runner._target_ == jump_runner._target_
    assert (
        vanilla_runner.milestone_evaluation.spec_target
        == jump_runner.milestone_evaluation.spec_target
    )
    assert (
        vanilla_runner.milestone_evaluation.expected_spec_version
        == jump_runner.milestone_evaluation.expected_spec_version
        == "1.1.0"
    )
    assert vanilla_runner.max_steps == jump_runner.max_steps == 900
    assert vanilla_runner.return_executed_observations is False
    assert jump_runner.return_executed_observations is True
    assert vanilla_runner.past_action is False
    assert jump_runner.past_action is False
    assert vanilla.task.source_task == jump.task.source_task == "GetToastedBread"


def test_policy_targets_remain_distinct():
    vanilla = compose_config("train_dp_gettoastedbread_milestone_v3")
    jump = compose_config("train_jump_gettoastedbread_milestone_v3")
    assert vanilla.architecture_id == "dp_vanilla"
    assert jump.architecture_id == "dp_jump"
    assert "jump_" not in vanilla.policy._target_
    assert "jump_" in jump.policy._target_
