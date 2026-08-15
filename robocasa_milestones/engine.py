"""Episode tracker and dependency-aware milestone completion engine."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Mapping

from robocasa_milestones.contracts import (
    EpisodeContext,
    MilestoneSpec,
    SpecObservation,
    definition_ids,
)
from robocasa_milestones.metrics import (
    executed_horizon_prefix_auc,
    fixed_horizon_prefix_auc,
    ordered_prefix_depth,
)


EPISODE_SCHEMA_VERSION = "robocasa_milestone_episode_v1"
TRACE_SCHEMA_VERSION = "robocasa_milestone_trace_v1"


class MilestoneEpisodeTracker:
    """Track one episode without reading policy or prediction state."""

    def __init__(self, spec: MilestoneSpec, *, fixed_horizon: int):
        self.spec = spec
        self.fixed_horizon = int(fixed_horizon)
        if self.fixed_horizon <= 0:
            raise ValueError("fixed_horizon must be positive")
        self._ids = definition_ids(spec.definitions)
        self._definition_by_id = {
            definition.id: definition for definition in spec.definitions
        }
        self._required_ids = tuple(
            definition.id for definition in spec.definitions if definition.required
        )
        if not self._required_ids:
            raise ValueError("a milestone spec must define a required milestone")
        self._context: EpisodeContext | None = None
        self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._physical_step = 0
        self._completed: set[str] = set()
        self._previous_conditions = {milestone_id: False for milestone_id in self._ids}
        self._first_raw_hit = {milestone_id: None for milestone_id in self._ids}
        self._first_valid_hit = {milestone_id: None for milestone_id in self._ids}
        self._last_true = {milestone_id: None for milestone_id in self._ids}
        self._valid_evidence: dict[str, dict[str, Any] | None] = {
            milestone_id: None for milestone_id in self._ids
        }
        self._trace: list[dict[str, Any]] = []
        self._invalid_event_count = 0
        self._official_success_seen = False
        self._finalized = False
        self._cached_outcome: dict[str, Any] | None = None

    @property
    def physical_step(self) -> int:
        return self._physical_step

    def reset(self, raw_env: Any, context: EpisodeContext) -> None:
        if context.task_spec_id != self.spec.spec_id:
            raise ValueError(
                f"spec id mismatch: {context.task_spec_id} != {self.spec.spec_id}"
            )
        if context.task_spec_version != self.spec.spec_version:
            raise ValueError(
                "spec version mismatch: "
                f"{context.task_spec_version} != {self.spec.spec_version}"
            )
        if context.task_name != self.spec.expected_task_name:
            raise ValueError(
                f"task mismatch: {context.task_name} != {self.spec.expected_task_name}"
            )
        if int(context.max_physical_steps) != self.fixed_horizon:
            raise ValueError(
                "context max_physical_steps must equal tracker fixed_horizon"
            )
        self._reset_runtime_state()
        self._context = context
        self.spec.reset(raw_env, context)

    def observe_after_committed_step(
        self,
        raw_env: Any,
        *,
        action_metadata: Mapping[str, Any] | None = None,
        official_success: bool = False,
    ) -> dict[str, Any]:
        if self._context is None:
            raise RuntimeError("tracker must be reset before observe")
        if self._finalized:
            raise RuntimeError("cannot observe a finalized episode")
        next_step = self._physical_step + 1
        if next_step > self.fixed_horizon:
            raise RuntimeError("committed physical steps exceed fixed_horizon")

        # Only commit the clock and trace after task-specific observation succeeds.
        observation = self.spec.observe(raw_env, next_step)
        self._validate_observation(observation)

        completed_before = frozenset(self._completed)
        new_valid: list[str] = []
        new_invalid = list(dict.fromkeys(str(x) for x in observation.invalid_events))
        raw_events = list(dict.fromkeys(str(x) for x in observation.raw_events))

        for milestone_id in self._ids:
            condition = bool(observation.milestone_conditions[milestone_id])
            was_true = self._previous_conditions[milestone_id]
            if condition:
                if self._first_raw_hit[milestone_id] is None:
                    self._first_raw_hit[milestone_id] = next_step
                self._last_true[milestone_id] = next_step
            if condition and milestone_id not in completed_before:
                definition = self._definition_by_id[milestone_id]
                prerequisites_met = all(
                    prerequisite in completed_before
                    for prerequisite in definition.prerequisites
                )
                if prerequisites_met:
                    new_valid.append(milestone_id)
                elif not was_true:
                    new_invalid.append(
                        f"milestone_trigger_before_prerequisites:{milestone_id}"
                    )

        # Use the beginning-of-step completion set for every prerequisite check.
        # This prevents several linear milestones from being credited atomically.
        for milestone_id in new_valid:
            self._completed.add(milestone_id)
            self._first_valid_hit[milestone_id] = next_step
            evidence = observation.evidence_by_milestone.get(milestone_id, {})
            self._valid_evidence[milestone_id] = dict(evidence)

        self._physical_step = next_step
        self._official_success_seen = (
            self._official_success_seen or bool(official_success)
        )
        self._previous_conditions = {
            milestone_id: bool(observation.milestone_conditions[milestone_id])
            for milestone_id in self._ids
        }
        new_invalid = list(dict.fromkeys(new_invalid))
        self._invalid_event_count += len(new_invalid)

        completed_ordered = [
            milestone_id for milestone_id in self._ids if milestone_id in self._completed
        ]
        active = [
            milestone_id
            for milestone_id in self._ids
            if milestone_id not in self._completed
            and all(
                prerequisite in self._completed
                for prerequisite in self._definition_by_id[
                    milestone_id
                ].prerequisites
            )
        ]
        required_completed = self._completed.intersection(self._required_ids)
        prefix_depth = ordered_prefix_depth(
            self._required_ids, frozenset(required_completed)
        )
        metadata = dict(action_metadata or {})
        row = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "physical_step": next_step,
            "policy_call": _optional_int(metadata.get("policy_call")),
            "offset_within_executed_chunk": _optional_int(
                metadata.get("offset_within_executed_chunk")
            ),
            "primitive_values": dict(observation.primitive_values),
            "raw_events": raw_events,
            "new_valid_completions": new_valid,
            "new_invalid_events": new_invalid,
            "active_milestones": active,
            "completed_milestones": completed_ordered,
            "valid_progress_count": len(required_completed),
            "linear_prefix_depth": prefix_depth,
            "official_success_from_step_info": bool(official_success),
        }
        self._trace.append(row)
        return copy.deepcopy(row)

    def _validate_observation(self, observation: SpecObservation) -> None:
        actual = set(observation.milestone_conditions)
        expected = set(self._ids)
        if actual != expected:
            raise ValueError(
                "spec observation milestone ids mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    def finalize(
        self,
        raw_env: Any,
        *,
        termination_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._context is None:
            raise RuntimeError("tracker must be reset before finalize")
        if self._cached_outcome is not None:
            return copy.deepcopy(self._cached_outcome)

        final_state = self.spec.finalize(raw_env, self._physical_step)
        required_completed = self._completed.intersection(self._required_ids)
        all_required_complete = len(required_completed) == len(self._required_ids)
        prefix_depths = [int(row["linear_prefix_depth"]) for row in self._trace]
        milestones: dict[str, Any] = {}
        for definition in self.spec.definitions:
            milestone_id = definition.id
            milestones[milestone_id] = {
                **asdict(definition),
                "completed": milestone_id in self._completed,
                "first_raw_hit_step": self._first_raw_hit[milestone_id],
                "first_valid_hit_step": self._first_valid_hit[milestone_id],
                "last_true_step": self._last_true[milestone_id],
                "maintained_at_episode_end": bool(
                    final_state.maintained_milestones.get(milestone_id, False)
                ),
                "first_valid_evidence": self._valid_evidence[milestone_id],
            }

        summary = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "task_name": self._context.task_name,
            "task_spec_id": self.spec.spec_id,
            "task_spec_version": self.spec.spec_version,
            "physical_step_count": self._physical_step,
            "physical_step_semantics": self._context.physical_step_semantics,
            "fixed_physical_horizon": self.fixed_horizon,
            "official_success": self._official_success_seen,
            "required_milestone_count": len(self._required_ids),
            "completed_required_count": len(required_completed),
            "normalized_valid_progress": (
                float(len(required_completed)) / float(len(self._required_ids))
            ),
            "max_linear_prefix_depth": max(prefix_depths, default=0),
            "normalized_linear_prefix": (
                float(max(prefix_depths, default=0)) / float(len(self._required_ids))
            ),
            "prefix_auc_fixed_horizon": fixed_horizon_prefix_auc(
                prefix_depths,
                required_count=len(self._required_ids),
                fixed_horizon=self.fixed_horizon,
            ),
            "prefix_auc_executed_horizon": executed_horizon_prefix_auc(
                prefix_depths, required_count=len(self._required_ids)
            ),
            "invalid_event_count": self._invalid_event_count,
            "all_required_milestones_complete": all_required_complete,
            "official_success_contract_disagreement": (
                self._official_success_seen != all_required_complete
            ),
            "milestones": milestones,
            "thresholds": dict(self.spec.thresholds),
            "context": self._context.to_dict(),
            "terminal_primitive_values": dict(final_state.primitive_values),
            "spec_final_metadata": dict(final_state.metadata),
            "termination_metadata": dict(termination_metadata or {}),
        }
        self._finalized = True
        self._cached_outcome = {"summary": summary, "trace": copy.deepcopy(self._trace)}
        return copy.deepcopy(self._cached_outcome)

    def snapshot_trace(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._trace)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
