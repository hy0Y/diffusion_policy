import numpy as np

from robocasa_milestones.contracts import EpisodeContext
from robocasa_milestones.engine import MilestoneEpisodeTracker
from robocasa_milestones.specs.get_toasted_bread_v1 import (
    GetToastedBreadV1Spec,
)


class FakeReader:
    def __init__(self):
        self.turned_on = {3: False, 5: False}
        self.lever = {3: 0.0, 5: 0.0}
        self.contacts = {3: True, 5: False}
        self.grasped = False
        self.in_plate = False
        self.gripper_far = False
        self.base_position = np.array([3.0, 3.0, 0.0])
        self.base_orientation = np.zeros(3)
        self.target_position = np.array([1.0, 2.0, 0.0])
        self.target_orientation = np.zeros(3)

    def validate(self, raw_env):
        return None

    def slot_states(self, raw_env):
        return {
            slot: {"turned_on": value, "lever": self.lever[slot]}
            for slot, value in self.turned_on.items()
        }

    def slot_contacts(self, raw_env, slot_pairs):
        return {slot: self.contacts[slot] for slot in slot_pairs}

    def bread_grasped(self, raw_env):
        return self.grasped

    def bread_in_plate(self, raw_env):
        return self.in_plate

    def gripper_bread_far(self, raw_env):
        return self.gripper_far

    def plate_navigation_target(self, raw_env):
        return self.target_position.copy(), self.target_orientation.copy()

    def robot_base_pose(self, raw_env):
        return self.base_position.copy(), self.base_orientation.copy()


def make_tracker(reader, horizon=20):
    spec = GetToastedBreadV1Spec(state_reader=reader)
    tracker = MilestoneEpisodeTracker(spec, fixed_horizon=horizon)
    tracker.reset(
        object(),
        EpisodeContext(
            task_name="GetToastedBread",
            task_spec_id=spec.spec_id,
            task_spec_version=spec.spec_version,
            environment_source_revision="fake",
            reset_mode="generated_seed",
            environment_seed=123,
            max_physical_steps=horizon,
        ),
    )
    return tracker


def test_dynamic_slot_normal_sequence_records_all_five_milestones():
    reader = FakeReader()
    tracker = make_tracker(reader)

    reader.turned_on[3] = True
    reader.lever[3] = 0.93
    assert tracker.observe_after_committed_step(object())["new_valid_completions"] == [
        "toaster_activated"
    ]

    reader.turned_on[3] = False
    reader.lever[3] = 1.0
    assert tracker.observe_after_committed_step(object())["new_valid_completions"] == [
        "toast_cycle_finished"
    ]

    reader.grasped = True
    reader.contacts[3] = False
    assert tracker.observe_after_committed_step(object())["new_valid_completions"] == [
        "bread_extracted"
    ]

    reader.base_position = reader.target_position.copy()
    assert tracker.observe_after_committed_step(object())["new_valid_completions"] == [
        "reached_plate_area"
    ]

    reader.grasped = False
    reader.in_plate = True
    reader.gripper_far = True
    final_row = tracker.observe_after_committed_step(
        object(), official_success=True
    )
    assert final_row["new_valid_completions"] == ["bread_placed_and_released"]

    summary = tracker.finalize(object())["summary"]
    assert summary["completed_required_count"] == 5
    assert summary["max_linear_prefix_depth"] == 5
    assert summary["spec_final_metadata"]["active_slot_pair"] == 3
    assert summary["spec_final_metadata"][
        "max_lever_position_tracked_slot"
    ] == 1.0
    assert summary["spec_final_metadata"][
        "lever_threshold_reached_tracked_slot"
    ] is True
    assert summary["spec_final_metadata"][
        "first_lever_threshold_step_tracked_slot"
    ] == 1
    assert summary["official_success_contract_disagreement"] is False
    assert summary["milestones"]["bread_placed_and_released"][
        "maintained_at_episode_end"
    ] is True


def test_wrong_slot_and_out_of_order_events_are_visible_but_not_credited():
    reader = FakeReader()
    tracker = make_tracker(reader)

    reader.turned_on[5] = True
    row = tracker.observe_after_committed_step(object())
    assert "toaster_activation_without_bread_contact:5" in row["new_invalid_events"]
    assert row["new_valid_completions"] == []

    reader.grasped = True
    reader.contacts[3] = False
    row = tracker.observe_after_committed_step(object())
    assert "milestone_trigger_before_prerequisites:bread_extracted" in row[
        "new_invalid_events"
    ]
    assert row["valid_progress_count"] == 0


def test_official_success_without_wait_completion_is_disagreement():
    reader = FakeReader()
    tracker = make_tracker(reader)

    reader.turned_on[3] = True
    reader.lever[3] = 0.91
    tracker.observe_after_committed_step(object())
    reader.in_plate = True
    reader.gripper_far = True
    tracker.observe_after_committed_step(object(), official_success=True)

    summary = tracker.finalize(object())["summary"]
    assert summary["official_success"] is True
    assert summary["completed_required_count"] == 1
    assert summary["milestones"]["toast_cycle_finished"]["completed"] is False
    assert summary["official_success_contract_disagreement"] is True


def test_terminal_maintenance_is_separate_from_first_hit_completion():
    reader = FakeReader()
    tracker = make_tracker(reader)
    reader.turned_on[3] = True
    reader.lever[3] = 0.95
    tracker.observe_after_committed_step(object())
    reader.turned_on[3] = False
    tracker.observe_after_committed_step(object())
    reader.grasped = True
    reader.contacts[3] = False
    tracker.observe_after_committed_step(object())
    reader.base_position = reader.target_position.copy()
    tracker.observe_after_committed_step(object())
    reader.grasped = False
    reader.in_plate = True
    reader.gripper_far = True
    tracker.observe_after_committed_step(object())

    reader.in_plate = False
    summary = tracker.finalize(object())["summary"]
    place = summary["milestones"]["bread_placed_and_released"]
    assert place["completed"] is True
    assert place["maintained_at_episode_end"] is False


def test_partial_lever_press_records_depth_and_threshold_shortfall():
    reader = FakeReader()
    tracker = make_tracker(reader)

    reader.lever[3] = 0.42
    first = tracker.observe_after_committed_step(object())
    reader.lever[3] = 0.87
    second = tracker.observe_after_committed_step(object())

    assert first["primitive_values"]["lever_position_tracked_slot"] == 0.42
    assert second["primitive_values"][
        "max_lever_position_tracked_slot_so_far"
    ] == 0.87
    assert second["primitive_values"][
        "lever_threshold_reached_tracked_slot"
    ] is False

    summary = tracker.finalize(object())["summary"]
    metadata = summary["spec_final_metadata"]
    assert metadata["max_lever_position_tracked_slot"] == 0.87
    assert metadata["max_lever_position_any_slot"] == 0.87
    assert metadata["lever_activation_threshold"] == 0.90
    assert np.isclose(metadata["lever_threshold_shortfall_tracked_slot"], 0.03)
    assert metadata["lever_threshold_reached_tracked_slot"] is False
    assert metadata["first_lever_threshold_step_tracked_slot"] is None
