from dataclasses import dataclass

import pytest

from robocasa_milestones.contracts import (
    EpisodeContext,
    MilestoneDefinition,
    SpecFinalState,
    SpecObservation,
)
from robocasa_milestones.engine import MilestoneEpisodeTracker
from robocasa_milestones.metrics import fixed_horizon_prefix_auc


@dataclass
class FakeEnv:
    conditions: dict[str, bool]


class FakeLinearSpec:
    spec_id = "fake/linear"
    spec_version = "1.0.0"
    expected_task_name = "FakeTask"
    thresholds = {"kind": "test"}
    definitions = (
        MilestoneDefinition("a", "first"),
        MilestoneDefinition("b", "second", prerequisites=("a",)),
        MilestoneDefinition("c", "third", prerequisites=("b",)),
    )

    def reset(self, raw_env, context):
        self.reset_count = getattr(self, "reset_count", 0) + 1

    def observe(self, raw_env, physical_step):
        return SpecObservation(
            primitive_values={"step": physical_step},
            milestone_conditions=dict(raw_env.conditions),
            evidence_by_milestone={
                key: {"condition": value}
                for key, value in raw_env.conditions.items()
            },
        )

    def finalize(self, raw_env, physical_step):
        return SpecFinalState(
            primitive_values={"step": physical_step},
            maintained_milestones=dict(raw_env.conditions),
        )


def context(max_steps=6):
    return EpisodeContext(
        task_name="FakeTask",
        task_spec_id="fake/linear",
        task_spec_version="1.0.0",
        environment_source_revision="fake",
        reset_mode="generated_seed",
        environment_seed=7,
        max_physical_steps=max_steps,
    )


def test_linear_tracker_uses_one_based_committed_physical_clock():
    env = FakeEnv({"a": False, "b": False, "c": False})
    tracker = MilestoneEpisodeTracker(FakeLinearSpec(), fixed_horizon=6)
    tracker.reset(env, context())

    prefix = []
    for step, milestone_id in enumerate(("a", "b", "c"), start=1):
        env.conditions = {key: key == milestone_id for key in env.conditions}
        row = tracker.observe_after_committed_step(
            env,
            action_metadata={
                "policy_call": 1 if step < 3 else 2,
                "offset_within_executed_chunk": (step - 1) % 2,
            },
            official_success=step == 3,
        )
        assert row["physical_step"] == step
        prefix.append(row["linear_prefix_depth"])

    outcome = tracker.finalize(env)
    summary = outcome["summary"]
    assert prefix == [1, 2, 3]
    assert summary["completed_required_count"] == 3
    assert summary["prefix_auc_fixed_horizon"] == pytest.approx(1 / 3)
    assert summary["prefix_auc_executed_horizon"] == pytest.approx(2 / 3)
    assert summary["official_success_contract_disagreement"] is False
    assert len(outcome["trace"]) == summary["physical_step_count"] == 3


def test_same_step_does_not_atomically_complete_prerequisite_chain():
    env = FakeEnv({"a": True, "b": True, "c": False})
    tracker = MilestoneEpisodeTracker(FakeLinearSpec(), fixed_horizon=4)
    tracker.reset(env, context(max_steps=4))

    first = tracker.observe_after_committed_step(env)
    assert first["new_valid_completions"] == ["a"]
    assert first["new_invalid_events"] == [
        "milestone_trigger_before_prerequisites:b"
    ]

    second = tracker.observe_after_committed_step(env)
    assert second["new_valid_completions"] == ["b"]
    assert second["new_invalid_events"] == []


def test_reset_removes_episode_latches_and_finalize_is_idempotent():
    env = FakeEnv({"a": True, "b": False, "c": False})
    tracker = MilestoneEpisodeTracker(FakeLinearSpec(), fixed_horizon=3)
    tracker.reset(env, context(max_steps=3))
    tracker.observe_after_committed_step(env)
    assert tracker.finalize(env)["summary"]["completed_required_count"] == 1

    env.conditions = {"a": False, "b": False, "c": False}
    tracker.reset(env, context(max_steps=3))
    first = tracker.finalize(env)
    second = tracker.finalize(env)
    assert first == second
    assert first["summary"]["completed_required_count"] == 0
    assert first["trace"] == []


def test_failed_spec_observation_does_not_commit_clock():
    class FailingSpec(FakeLinearSpec):
        def observe(self, raw_env, physical_step):
            raise RuntimeError("state read failed")

    env = FakeEnv({"a": False, "b": False, "c": False})
    tracker = MilestoneEpisodeTracker(FailingSpec(), fixed_horizon=3)
    tracker.reset(env, context(max_steps=3))
    with pytest.raises(RuntimeError, match="state read failed"):
        tracker.observe_after_committed_step(env)
    assert tracker.physical_step == 0
    assert tracker.snapshot_trace() == []


def test_fixed_horizon_auc_rejects_inconsistent_inputs():
    assert fixed_horizon_prefix_auc(
        [0, 1, 2], required_count=2, fixed_horizon=6
    ) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        fixed_horizon_prefix_auc([0, 1, 2], required_count=2, fixed_horizon=2)
