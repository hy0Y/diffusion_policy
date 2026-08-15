import gym
import numpy as np
import pytest

from diffusion_policy.env_runner.milestone_robomimic_image_runner import (
    MilestoneRobomimicImageRunner,
    validate_milestone_evaluation_contract,
)
from diffusion_policy.gym_util.milestone_multistep_wrapper import (
    MilestoneMultiStepWrapper,
)


class FakeMilestoneEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -10, 10, shape=(1,), dtype=np.float32
        )
        self.begin_calls = []
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.array([0.0], dtype=np.float32)

    def begin_policy_call(self, policy_call, proposed_chunk_length):
        self.begin_calls.append((policy_call, proposed_chunk_length))

    def step(self, action):
        self.step_count += 1
        return (
            np.array([self.step_count], dtype=np.float32),
            0.0,
            False,
            {"success": False},
        )

    def finalize_milestone_episode(self, termination_metadata=None):
        return {"steps": self.step_count, "termination": termination_metadata}


def test_multistep_adapter_marks_chunk_without_changing_per_action_loop():
    inner = FakeMilestoneEnv()
    env = MilestoneMultiStepWrapper(
        inner,
        n_obs_steps=2,
        n_action_steps=3,
        max_episode_steps=10,
    )
    env.reset()
    action = np.zeros((3, 1), dtype=np.float32)
    env.step(action)
    assert inner.begin_calls == [(1, 3)]
    assert inner.step_count == 3
    outcome = env.finalize_milestone_episode({"termination_reason": "test"})
    assert outcome == {
        "steps": 3,
        "termination": {"termination_reason": "test"},
    }


def test_policy_call_clock_resets_between_episodes():
    inner = FakeMilestoneEnv()
    env = MilestoneMultiStepWrapper(
        inner, n_obs_steps=1, n_action_steps=2, max_episode_steps=10
    )
    env.reset()
    env.step(np.zeros((2, 1), dtype=np.float32))
    env.reset()
    env.step(np.zeros((2, 1), dtype=np.float32))
    assert inner.begin_calls == [(1, 2), (1, 2)]


def milestone_config(**overrides):
    config = {
        "enabled": True,
        "required": True,
        "spec_target": (
            "robocasa_milestones.specs.get_toasted_bread_v1:"
            "GetToastedBreadV1Spec"
        ),
        "expected_task_name": "GetToastedBread",
        "expected_spec_id": (
            "robocasa365/GetToastedBread/physical_milestones"
        ),
        "expected_spec_version": "1.1.0",
        "fixed_horizon": 900,
        "environment_source_revision": "test-revision",
    }
    config.update(overrides)
    return config


def test_runner_contract_is_fail_closed_before_environment_creation():
    validated = validate_milestone_evaluation_contract(
        milestone_config(), env_name="GetToastedBread", max_steps=900
    )
    assert validated["required"] is True
    assert validated["fixed_horizon"] == 900

    with pytest.raises(ValueError, match="fail-closed"):
        validate_milestone_evaluation_contract(
            milestone_config(required=False),
            env_name="GetToastedBread",
            max_steps=900,
        )
    with pytest.raises(ValueError, match="does not match rollout environment"):
        validate_milestone_evaluation_contract(
            milestone_config(), env_name="PnPCounterToCabinet", max_steps=900
        )
    with pytest.raises(ValueError, match="fixed_horizon"):
        validate_milestone_evaluation_contract(
            milestone_config(), env_name="GetToastedBread", max_steps=500
        )


def test_runner_log_data_exposes_continuous_lever_diagnostics():
    runner = MilestoneRobomimicImageRunner.__new__(
        MilestoneRobomimicImageRunner
    )
    runner.reset_mode = "generated_seed"
    runner.env_seeds = [1111111]
    runner.env_prefixs = ["test/"]
    runner.env_kwargs = {"env_name": "GetToastedBread"}
    runner.episode_root = None
    runner.initial_state_index = 0
    runner.last_milestone_manifest = {
        "path": "/tmp/manifest.json",
        "sha256": "test-sha256",
        "episode_count": 1,
    }
    runner.last_milestone_outcomes = [
        {
            "task_spec_id": (
                "robocasa365/GetToastedBread/physical_milestones"
            ),
            "task_spec_version": "1.1.0",
            "normalized_valid_progress": 0.0,
            "normalized_linear_prefix": 0.0,
            "prefix_auc_fixed_horizon": 0.0,
            "invalid_event_count": 0,
            "official_success_contract_disagreement": False,
            "spec_final_metadata": {
                "max_lever_position_tracked_slot": 0.87,
                "max_lever_position_any_slot": 0.87,
                "lever_threshold_shortfall_tracked_slot": 0.03,
                "lever_threshold_reached_tracked_slot": False,
                "first_lever_threshold_step_tracked_slot": None,
            },
        }
    ]

    log = runner._build_log_data([np.array([0.0])], [None])

    assert log[
        "milestone/episode_000000/max_lever_position_tracked_slot"
    ] == 0.87
    assert np.isclose(
        log["milestone/mean_lever_threshold_shortfall_tracked_slot"], 0.03
    )
    assert log["milestone/lever_threshold_reach_rate"] == 0.0
