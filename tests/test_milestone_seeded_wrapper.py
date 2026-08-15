import numpy as np
import pytest

from diffusion_policy.env.robomimic.milestone_seeded_robomimic_image_wrapper import (
    MilestoneSeededRobomimicImageWrapper,
)
from diffusion_policy.env.robomimic.seeded_robomimic_image_wrapper import (
    SeededRobomimicImageWrapper,
)


class FakeTracker:
    def __init__(self):
        self.calls = []

    def observe_after_committed_step(self, raw_env, **kwargs):
        self.calls.append((raw_env, kwargs))
        return {
            "physical_step": len(self.calls),
            "valid_progress_count": 2,
            "linear_prefix_depth": 2,
            "new_valid_completions": ["p1"],
            "new_invalid_events": [],
        }


class RawHolder:
    def __init__(self, raw_env):
        self.env = raw_env


class GymHolder:
    def __init__(self, raw_env):
        self.unwrapped = RawHolder(raw_env)


def make_wrapper():
    wrapper = object.__new__(MilestoneSeededRobomimicImageWrapper)
    wrapper.env = GymHolder(object())
    wrapper._milestone_enabled = True
    wrapper._milestone_tracker = FakeTracker()
    wrapper._policy_call = 4
    wrapper._offset_within_chunk = 1
    return wrapper


def test_step_observes_only_after_parent_step_commits(monkeypatch):
    wrapper = make_wrapper()
    monkeypatch.setattr(
        SeededRobomimicImageWrapper,
        "step",
        lambda self, action: (np.array([1.0]), 0.0, False, {"success": False}),
    )
    observation, reward, done, info = wrapper.step(np.array([0.0]))
    assert observation.tolist() == [1.0]
    assert reward == 0.0 and done is False
    assert info["milestone_physical_step"] == 1
    assert wrapper._milestone_tracker.calls[0][1]["action_metadata"] == {
        "policy_call": 4,
        "offset_within_executed_chunk": 1,
    }
    assert wrapper._offset_within_chunk == 2


def test_failed_parent_step_does_not_observe(monkeypatch):
    wrapper = make_wrapper()

    def fail(self, action):
        raise RuntimeError("env failed")

    monkeypatch.setattr(SeededRobomimicImageWrapper, "step", fail)
    with pytest.raises(RuntimeError, match="env failed"):
        wrapper.step(np.array([0.0]))
    assert wrapper._milestone_tracker.calls == []
    assert wrapper._offset_within_chunk == 1
