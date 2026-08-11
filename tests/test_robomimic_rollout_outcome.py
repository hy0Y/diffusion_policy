import numpy as np
import pytest

from diffusion_policy.env_runner.robomimic_image_runner import (
    step_environment_then_commit_trace,
    summarize_rollout_outcome,
)


class RecordingEnvironment:
    def __init__(self, events, *, fail=False):
        self.events = events
        self.fail = fail

    def step(self, action):
        self.events.append(("env.step", np.asarray(action).copy()))
        if self.fail:
            raise RuntimeError("simulated env failure")
        return "obs", "reward", "done", "info"


class RecordingTraceWriter:
    def __init__(self, events):
        self.events = events
        self.rows = []

    def append(self, row):
        self.events.append(("trace.append", row))
        self.rows.append(row)


def test_trace_is_committed_only_after_successful_environment_step() -> None:
    events = []
    environment = RecordingEnvironment(events)
    writer = RecordingTraceWriter(events)
    row = {"action_time": 0, "physical_time": 1}
    transition = step_environment_then_commit_trace(
        environment,
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        trace_writer=writer,
        trace_row=row,
    )
    assert transition == ("obs", "reward", "done", "info")
    assert [event[0] for event in events] == ["env.step", "trace.append"]
    assert writer.rows == [row]


def test_failed_environment_step_does_not_commit_trace() -> None:
    events = []
    environment = RecordingEnvironment(events, fail=True)
    writer = RecordingTraceWriter(events)
    with pytest.raises(RuntimeError, match="simulated env failure"):
        step_environment_then_commit_trace(
            environment,
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            trace_writer=writer,
            trace_row={"action_time": 0, "physical_time": 1},
        )
    assert [event[0] for event in events] == ["env.step"]
    assert writer.rows == []


def test_rollout_outcome_records_first_success_step() -> None:
    outcome = summarize_rollout_outcome([0.0, 0.0, 1.0, 1.0], 10)
    assert outcome == {
        "termination_reason": "task_success",
        "success_step": 3,
        "executed_steps": 4,
    }


def test_rollout_outcome_distinguishes_horizon_and_environment_end() -> None:
    assert summarize_rollout_outcome([0.0] * 4, 4) == {
        "termination_reason": "horizon_exhausted",
        "success_step": None,
        "executed_steps": 4,
    }
    assert summarize_rollout_outcome([0.0] * 2, 4) == {
        "termination_reason": "environment_terminated",
        "success_step": None,
        "executed_steps": 2,
    }
