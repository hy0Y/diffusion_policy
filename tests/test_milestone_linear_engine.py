from robocasa_milestones.contracts import (
    EpisodeContext,
    MilestoneDefinition,
    SpecFinalState,
    SpecObservation,
)
from robocasa_milestones.engine import MilestoneEpisodeTracker


class EventSpec:
    spec_id = "test/event_linear"
    spec_version = "1.0.0"
    expected_task_name = "EventTask"
    thresholds = {}
    definitions = (
        MilestoneDefinition("start", "start event"),
        MilestoneDefinition("finish", "finish event", prerequisites=("start",)),
    )

    def reset(self, raw_env, context):
        return None

    def observe(self, raw_env, physical_step):
        return SpecObservation(
            primitive_values={},
            milestone_conditions=dict(raw_env),
        )

    def finalize(self, raw_env, physical_step):
        return SpecFinalState(
            primitive_values={}, maintained_milestones=dict(raw_env)
        )


def test_past_out_of_order_event_is_not_retroactively_completed():
    state = {"start": False, "finish": True}
    tracker = MilestoneEpisodeTracker(EventSpec(), fixed_horizon=5)
    tracker.reset(
        state,
        EpisodeContext(
            task_name="EventTask",
            task_spec_id="test/event_linear",
            task_spec_version="1.0.0",
            environment_source_revision="test",
            reset_mode="generated_seed",
            max_physical_steps=5,
        ),
    )
    first = tracker.observe_after_committed_step(state)
    assert first["new_valid_completions"] == []

    state.update(start=True, finish=False)
    second = tracker.observe_after_committed_step(state)
    assert second["new_valid_completions"] == ["start"]
    assert second["completed_milestones"] == ["start"]

    state.update(start=False, finish=True)
    third = tracker.observe_after_committed_step(state)
    assert third["new_valid_completions"] == ["finish"]
    assert third["linear_prefix_depth"] == 2
