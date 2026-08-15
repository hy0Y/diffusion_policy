"""GetToastedBread physical milestone specification version 1.1.0."""

from __future__ import annotations

from typing import Any

import numpy as np

from robocasa_milestones.contracts import (
    EpisodeContext,
    MilestoneDefinition,
    SpecFinalState,
    SpecObservation,
)
from robocasa_milestones.robocasa_state import GetToastedBreadStateReader


class GetToastedBreadV1Spec:
    spec_id = "robocasa365/GetToastedBread/physical_milestones"
    spec_version = "1.1.0"
    expected_task_name = "GetToastedBread"
    lever_activation_threshold = 0.90
    definitions = (
        MilestoneDefinition(
            id="toaster_activated",
            description="Bread-contact toaster slot changes from off to on",
            trigger_semantics="same-step bread contact and turned_on rising edge",
        ),
        MilestoneDefinition(
            id="toast_cycle_finished",
            description="The activated toaster slot changes from on to off",
            prerequisites=("toaster_activated",),
            trigger_semantics="same active slot turned_on falling edge",
        ),
        MilestoneDefinition(
            id="bread_extracted",
            description="Bread is grasped and no longer contacts the active slot",
            prerequisites=("toast_cycle_finished",),
            trigger_semantics="grasped AND not active-slot contact",
        ),
        MilestoneDefinition(
            id="reached_plate_area",
            description="Robot reaches the plate approach pose while holding bread",
            prerequisites=("bread_extracted",),
            trigger_semantics="held bread AND base pose threshold",
        ),
        MilestoneDefinition(
            id="bread_placed_and_released",
            description="Bread is in the plate and the gripper is far",
            prerequisites=("reached_plate_area",),
            trigger_semantics="bread in plate AND gripper far",
        ),
    )

    def __init__(
        self,
        *,
        state_reader: Any | None = None,
        navigation_xy_threshold: float = 0.20,
        navigation_orientation_cosine_threshold: float = 0.98,
    ):
        self.reader = state_reader or GetToastedBreadStateReader()
        self.navigation_xy_threshold = float(navigation_xy_threshold)
        self.navigation_orientation_cosine_threshold = float(
            navigation_orientation_cosine_threshold
        )
        self.thresholds = {
            "lever_activation_position_min": self.lever_activation_threshold,
            "navigation_xy_distance_max": self.navigation_xy_threshold,
            "navigation_orientation_cosine_min": (
                self.navigation_orientation_cosine_threshold
            ),
            "bread_extracted_requires_grasp_and_slot_exit": True,
            "toaster_slot_pair_is_dynamic": True,
        }
        self._reset_state()

    def _reset_state(self) -> None:
        self._slot_pairs: tuple[int, ...] = ()
        self._previous_turned_on: dict[int, bool] = {}
        self._active_slot_pair: int | None = None
        self._tracked_bread_slot_pair: int | None = None
        self._activation_step: int | None = None
        self._initial_lever_position_tracked_slot: float | None = None
        self._max_lever_position_by_slot: dict[int, float] = {}
        self._first_lever_threshold_step_tracked_slot: int | None = None
        self._target_position: np.ndarray | None = None
        self._target_orientation: np.ndarray | None = None
        self._last_snapshot: dict[str, Any] | None = None

    def reset(self, raw_env: Any, context: EpisodeContext) -> None:
        self._reset_state()
        self.reader.validate(raw_env)
        states = self.reader.slot_states(raw_env)
        self._slot_pairs = tuple(sorted(int(slot) for slot in states))
        self._previous_turned_on = {
            slot: bool(states[slot]["turned_on"]) for slot in self._slot_pairs
        }
        lever_positions = self._lever_positions(states)
        self._max_lever_position_by_slot = dict(lever_positions)
        contacts = self.reader.slot_contacts(raw_env, self._slot_pairs)
        contact_slots = [slot for slot in self._slot_pairs if contacts[slot]]
        if contact_slots:
            self._tracked_bread_slot_pair = contact_slots[0]
            self._initial_lever_position_tracked_slot = lever_positions[
                self._tracked_bread_slot_pair
            ]
        target_position, target_orientation = self.reader.plate_navigation_target(
            raw_env
        )
        self._target_position = np.asarray(target_position, dtype=float)
        self._target_orientation = np.asarray(target_orientation, dtype=float)

    def observe(self, raw_env: Any, physical_step: int) -> SpecObservation:
        snapshot = self._read_snapshot(raw_env)
        self._update_lever_diagnostics(snapshot, physical_step)
        states = snapshot["slot_states"]
        contacts = snapshot["slot_contacts"]
        rising_slots = [
            slot
            for slot in self._slot_pairs
            if not self._previous_turned_on[slot]
            and bool(states[slot]["turned_on"])
        ]
        falling_slots = [
            slot
            for slot in self._slot_pairs
            if self._previous_turned_on[slot]
            and not bool(states[slot]["turned_on"])
        ]
        raw_events: list[str] = []
        invalid_events: list[str] = []

        for slot in rising_slots:
            raw_events.append(f"toaster_turned_on_rising:{slot}")
        for slot in falling_slots:
            raw_events.append(f"toaster_turned_on_falling:{slot}")

        activation_candidates = [slot for slot in rising_slots if contacts[slot]]
        toaster_activated = (
            self._active_slot_pair is None and bool(activation_candidates)
        )
        if toaster_activated:
            self._active_slot_pair = activation_candidates[0]
            if self._tracked_bread_slot_pair is None:
                self._tracked_bread_slot_pair = self._active_slot_pair
            self._activation_step = int(physical_step)
        for slot in rising_slots:
            if not contacts[slot]:
                invalid_events.append(
                    f"toaster_activation_without_bread_contact:{slot}"
                )
            elif self._active_slot_pair is not None and slot != self._active_slot_pair:
                invalid_events.append(f"secondary_toaster_slot_activation:{slot}")

        toast_cycle_finished = (
            self._active_slot_pair is not None
            and self._active_slot_pair in falling_slots
        )
        for slot in falling_slots:
            if self._active_slot_pair is None:
                invalid_events.append(f"toaster_falling_without_activation:{slot}")
            elif slot != self._active_slot_pair:
                invalid_events.append(f"non_active_toaster_slot_falling:{slot}")

        active_contact = (
            contacts[self._active_slot_pair]
            if self._active_slot_pair is not None
            else any(contacts.values())
        )
        bread_extracted = bool(snapshot["bread_grasped"] and not active_contact)
        reached_plate_area = bool(
            snapshot["bread_grasped"]
            and snapshot["base_target_xy_distance"] <= self.navigation_xy_threshold
            and snapshot["base_target_orientation_cosine"]
            >= self.navigation_orientation_cosine_threshold
        )
        bread_placed_and_released = bool(
            snapshot["bread_in_plate"] and snapshot["gripper_bread_far"]
        )

        tracked_lever = self._tracked_lever_value(snapshot["lever_positions"])
        max_tracked_lever = self._max_tracked_lever_value()
        current_max_any_lever = max(snapshot["lever_positions"].values())
        max_any_lever = max(self._max_lever_position_by_slot.values())

        if snapshot["bread_grasped"]:
            raw_events.append("bread_grasped_state")
        if not active_contact:
            raw_events.append("bread_outside_active_slot_state")
        if reached_plate_area:
            raw_events.append("held_bread_at_plate_approach_pose")
        if bread_placed_and_released:
            raw_events.append("bread_in_plate_and_gripper_far")

        primitives = {
            "active_slot_pair": (
                self._active_slot_pair
                if self._active_slot_pair is not None
                else -1
            ),
            "tracked_bread_slot_pair": (
                self._tracked_bread_slot_pair
                if self._tracked_bread_slot_pair is not None
                else -1
            ),
            "lever_position_tracked_slot": tracked_lever,
            "lever_position_max_any_slot": float(current_max_any_lever),
            "max_lever_position_tracked_slot_so_far": max_tracked_lever,
            "max_lever_position_any_slot_so_far": float(max_any_lever),
            "lever_activation_threshold": self.lever_activation_threshold,
            "lever_threshold_shortfall_tracked_slot_so_far": (
                self._lever_threshold_shortfall(max_tracked_lever)
            ),
            "lever_threshold_reached_tracked_slot": bool(
                max_tracked_lever is not None
                and max_tracked_lever >= self.lever_activation_threshold
            ),
            "first_lever_threshold_step_tracked_slot": (
                self._first_lever_threshold_step_tracked_slot
            ),
            "toaster_turned_on": (
                bool(states[self._active_slot_pair]["turned_on"])
                if self._active_slot_pair is not None
                else any(bool(values["turned_on"]) for values in states.values())
            ),
            "bread_contact_active_slot": bool(active_contact),
            "bread_contact_any_slot": bool(any(contacts.values())),
            "bread_grasped": bool(snapshot["bread_grasped"]),
            "bread_in_plate": bool(snapshot["bread_in_plate"]),
            "gripper_bread_far": bool(snapshot["gripper_bread_far"]),
            "base_target_xy_distance": float(snapshot["base_target_xy_distance"]),
            "base_target_orientation_cosine": float(
                snapshot["base_target_orientation_cosine"]
            ),
        }
        conditions = {
            "toaster_activated": toaster_activated,
            "toast_cycle_finished": toast_cycle_finished,
            "bread_extracted": bread_extracted,
            "reached_plate_area": reached_plate_area,
            "bread_placed_and_released": bread_placed_and_released,
        }
        evidence = {
            milestone_id: dict(primitives) for milestone_id in conditions
        }
        evidence["toast_cycle_finished"].update(
            {
                "activation_step": self._activation_step,
                "observed_on_duration_physical_steps": (
                    physical_step - self._activation_step
                    if self._activation_step is not None
                    else None
                ),
            }
        )
        self._previous_turned_on = {
            slot: bool(states[slot]["turned_on"]) for slot in self._slot_pairs
        }
        self._last_snapshot = snapshot
        return SpecObservation(
            primitive_values=primitives,
            milestone_conditions=conditions,
            raw_events=raw_events,
            invalid_events=invalid_events,
            evidence_by_milestone=evidence,
        )

    def finalize(self, raw_env: Any, physical_step: int) -> SpecFinalState:
        snapshot = self._read_snapshot(raw_env)
        self._update_lever_diagnostics(snapshot, physical_step=None)
        active_contact = (
            snapshot["slot_contacts"][self._active_slot_pair]
            if self._active_slot_pair is not None
            else any(snapshot["slot_contacts"].values())
        )
        maintained = {
            "toaster_activated": bool(
                self._active_slot_pair is not None and self._activation_step is not None
            ),
            "toast_cycle_finished": bool(
                self._active_slot_pair is not None
                and not snapshot["slot_states"][self._active_slot_pair]["turned_on"]
            ),
            "bread_extracted": bool(snapshot["bread_grasped"] and not active_contact),
            "reached_plate_area": bool(
                snapshot["bread_grasped"]
                and snapshot["base_target_xy_distance"]
                <= self.navigation_xy_threshold
                and snapshot["base_target_orientation_cosine"]
                >= self.navigation_orientation_cosine_threshold
            ),
            "bread_placed_and_released": bool(
                snapshot["bread_in_plate"] and snapshot["gripper_bread_far"]
            ),
        }
        terminal = {
            "active_slot_pair": (
                self._active_slot_pair
                if self._active_slot_pair is not None
                else -1
            ),
            "bread_grasped": bool(snapshot["bread_grasped"]),
            "bread_in_plate": bool(snapshot["bread_in_plate"]),
            "gripper_bread_far": bool(snapshot["gripper_bread_far"]),
            "base_target_xy_distance": float(snapshot["base_target_xy_distance"]),
            "base_target_orientation_cosine": float(
                snapshot["base_target_orientation_cosine"]
            ),
            "lever_position_tracked_slot": self._tracked_lever_value(
                snapshot["lever_positions"]
            ),
            "lever_position_max_any_slot": float(
                max(snapshot["lever_positions"].values())
            ),
        }
        max_tracked_lever = self._max_tracked_lever_value()
        max_any_lever = max(self._max_lever_position_by_slot.values())
        return SpecFinalState(
            primitive_values=terminal,
            maintained_milestones=maintained,
            metadata={
                "active_slot_pair": (
                    self._active_slot_pair
                    if self._active_slot_pair is not None
                    else -1
                ),
                "activation_step": self._activation_step,
                "slot_pair_count": len(self._slot_pairs),
                "tracked_bread_slot_pair": (
                    self._tracked_bread_slot_pair
                    if self._tracked_bread_slot_pair is not None
                    else -1
                ),
                "initial_lever_position_tracked_slot": (
                    self._initial_lever_position_tracked_slot
                ),
                "max_lever_position_tracked_slot": max_tracked_lever,
                "max_lever_position_any_slot": float(max_any_lever),
                "lever_activation_threshold": self.lever_activation_threshold,
                "lever_threshold_shortfall_tracked_slot": (
                    self._lever_threshold_shortfall(max_tracked_lever)
                ),
                "lever_threshold_reached_tracked_slot": bool(
                    max_tracked_lever is not None
                    and max_tracked_lever >= self.lever_activation_threshold
                ),
                "first_lever_threshold_step_tracked_slot": (
                    self._first_lever_threshold_step_tracked_slot
                ),
            },
        )

    def _update_lever_diagnostics(
        self, snapshot: dict[str, Any], physical_step: int | None
    ) -> None:
        contacts = snapshot["slot_contacts"]
        if self._tracked_bread_slot_pair is None:
            contact_slots = [slot for slot in self._slot_pairs if contacts[slot]]
            if contact_slots:
                self._tracked_bread_slot_pair = contact_slots[0]
                self._initial_lever_position_tracked_slot = snapshot[
                    "lever_positions"
                ][self._tracked_bread_slot_pair]

        for slot, value in snapshot["lever_positions"].items():
            self._max_lever_position_by_slot[slot] = max(
                self._max_lever_position_by_slot.get(slot, value), value
            )

        tracked_value = self._tracked_lever_value(snapshot["lever_positions"])
        if (
            physical_step is not None
            and tracked_value is not None
            and tracked_value >= self.lever_activation_threshold
            and self._first_lever_threshold_step_tracked_slot is None
        ):
            self._first_lever_threshold_step_tracked_slot = int(physical_step)

    def _tracked_lever_value(
        self, lever_positions: dict[int, float]
    ) -> float | None:
        if self._tracked_bread_slot_pair is None:
            return None
        return float(lever_positions[self._tracked_bread_slot_pair])

    def _max_tracked_lever_value(self) -> float | None:
        if self._tracked_bread_slot_pair is None:
            return None
        return float(
            self._max_lever_position_by_slot[self._tracked_bread_slot_pair]
        )

    def _lever_threshold_shortfall(self, value: float | None) -> float | None:
        if value is None:
            return None
        return float(max(0.0, self.lever_activation_threshold - value))

    def _lever_positions(
        self, states: dict[int, dict[str, Any]]
    ) -> dict[int, float]:
        positions: dict[int, float] = {}
        for slot in self._slot_pairs:
            if "lever" not in states[slot]:
                raise RuntimeError(f"toaster slot {slot} state has no lever value")
            value = float(states[slot]["lever"])
            if not np.isfinite(value):
                raise RuntimeError(
                    f"toaster slot {slot} lever value is not finite: {value}"
                )
            positions[slot] = value
        return positions

    def _read_snapshot(self, raw_env: Any) -> dict[str, Any]:
        states = self.reader.slot_states(raw_env)
        actual_slots = tuple(sorted(int(slot) for slot in states))
        if actual_slots != self._slot_pairs:
            raise RuntimeError(
                f"toaster slot pairs changed within episode: {self._slot_pairs} -> {actual_slots}"
            )
        contacts = self.reader.slot_contacts(raw_env, self._slot_pairs)
        base_position, base_orientation = self.reader.robot_base_pose(raw_env)
        if self._target_position is None or self._target_orientation is None:
            raise RuntimeError("navigation target was not initialized")
        xy_distance = float(
            np.linalg.norm(self._target_position[:2] - np.asarray(base_position)[:2])
        )
        orientation_cosine = float(
            np.cos(self._target_orientation[2] - np.asarray(base_orientation)[2])
        )
        return {
            "slot_states": states,
            "slot_contacts": contacts,
            "lever_positions": self._lever_positions(states),
            "bread_grasped": bool(self.reader.bread_grasped(raw_env)),
            "bread_in_plate": bool(self.reader.bread_in_plate(raw_env)),
            "gripper_bread_far": bool(self.reader.gripper_bread_far(raw_env)),
            "base_target_xy_distance": xy_distance,
            "base_target_orientation_cosine": orientation_cosine,
        }
