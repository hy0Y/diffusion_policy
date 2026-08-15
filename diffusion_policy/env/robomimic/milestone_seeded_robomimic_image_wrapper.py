"""Seeded RoboCasa image wrapper with per-physical-step milestone evaluation.

This is an additive host adapter. The existing seeded and image wrappers remain
unchanged; all task semantics live in the top-level ``robocasa_milestones``
package.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from diffusion_policy.env.robomimic.seeded_robomimic_image_wrapper import (
    SeededRobomimicImageWrapper,
)
from robocasa_milestones.contracts import EpisodeContext
from robocasa_milestones.engine import MilestoneEpisodeTracker
from robocasa_milestones.loader import load_milestone_spec
from robocasa_milestones.robocasa_state import resolve_raw_task_env


class MilestoneSeededRobomimicImageWrapper(SeededRobomimicImageWrapper):
    """Observe raw simulator state after every successful inner env step."""

    def __init__(
        self,
        *args,
        milestone_evaluation: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._milestone_config = _plain_mapping(milestone_evaluation or {})
        self._milestone_enabled = bool(
            self._milestone_config.get("enabled", False)
        )
        self._milestone_required = bool(
            self._milestone_config.get("required", True)
        )
        self._milestone_tracker: MilestoneEpisodeTracker | None = None
        self._milestone_episode_active = False
        self._last_milestone_outcome: dict[str, Any] | None = None
        self._policy_call: int | None = None
        self._offset_within_chunk: int | None = None
        if self._milestone_enabled:
            required_keys = (
                "spec_target",
                "expected_task_name",
                "expected_spec_id",
                "expected_spec_version",
                "fixed_horizon",
                "environment_source_revision",
            )
            missing = [
                key for key in required_keys if self._milestone_config.get(key) is None
            ]
            if missing:
                raise ValueError(f"milestone_evaluation lacks required keys: {missing}")
            spec = load_milestone_spec(
                str(self._milestone_config["spec_target"]),
                expected_task_name=str(
                    self._milestone_config["expected_task_name"]
                ),
                expected_spec_id=str(self._milestone_config["expected_spec_id"]),
                expected_spec_version=str(
                    self._milestone_config["expected_spec_version"]
                ),
                spec_kwargs=_plain_mapping(
                    self._milestone_config.get("spec_kwargs", {})
                ),
            )
            self._milestone_tracker = MilestoneEpisodeTracker(
                spec, fixed_horizon=int(self._milestone_config["fixed_horizon"])
            )

    @property
    def milestone_enabled(self) -> bool:
        return self._milestone_enabled

    def begin_policy_call(self, policy_call: int, proposed_chunk_length: int) -> None:
        if int(policy_call) <= 0:
            raise ValueError("policy_call must be one-based and positive")
        if int(proposed_chunk_length) < 0:
            raise ValueError("proposed_chunk_length cannot be negative")
        self._policy_call = int(policy_call)
        self._offset_within_chunk = 0

    def reset(self):
        if self._milestone_episode_active:
            raise RuntimeError(
                "milestone episode must be finalized before the next reset"
            )
        observation = super().reset()
        self._policy_call = None
        self._offset_within_chunk = None
        self._last_milestone_outcome = None
        if self._milestone_enabled:
            raw_env = resolve_raw_task_env(self.env)
            snapshot = self.last_reset_snapshot or {}
            reset_mode = (
                "demonstration_state"
                if self.last_reset_snapshot is not None
                else "generated_seed"
            )
            context = EpisodeContext(
                task_name=str(self._milestone_config["expected_task_name"]),
                task_spec_id=str(self._milestone_config["expected_spec_id"]),
                task_spec_version=str(
                    self._milestone_config["expected_spec_version"]
                ),
                environment_source_revision=str(
                    self._milestone_config["environment_source_revision"]
                ),
                reset_mode=reset_mode,
                max_physical_steps=int(
                    self._milestone_config["fixed_horizon"]
                ),
                environment_seed=self.last_reset_seed,
                episode_root=snapshot.get("episode_root"),
                initial_state_index=snapshot.get("state_index"),
                extra={
                    "state_backend": "robocasa_raw_simulator_state",
                    "evaluator_required": self._milestone_required,
                },
            )
            assert self._milestone_tracker is not None
            self._milestone_tracker.reset(raw_env, context)
            self._milestone_episode_active = True
        return observation

    def step(self, action):
        transition = super().step(action)
        if not self._milestone_enabled:
            return transition
        observation, reward, done, info = transition
        raw_env = resolve_raw_task_env(self.env)
        assert self._milestone_tracker is not None
        row = self._milestone_tracker.observe_after_committed_step(
            raw_env,
            action_metadata={
                "policy_call": self._policy_call,
                "offset_within_executed_chunk": self._offset_within_chunk,
            },
            official_success=bool(info.get("success", False)),
        )
        if self._offset_within_chunk is not None:
            self._offset_within_chunk += 1
        enriched_info = dict(info)
        enriched_info.update(
            {
                "milestone_physical_step": row["physical_step"],
                "milestone_valid_progress_count": row["valid_progress_count"],
                "milestone_linear_prefix_depth": row["linear_prefix_depth"],
                "milestone_new_completion_count": len(
                    row["new_valid_completions"]
                ),
                "milestone_invalid_event_count_step": len(
                    row["new_invalid_events"]
                ),
            }
        )
        return observation, reward, done, enriched_info

    def finalize_milestone_episode(
        self, termination_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if not self._milestone_enabled:
            return None
        if not self._milestone_episode_active:
            if self._last_milestone_outcome is None:
                raise RuntimeError("there is no active or finalized milestone episode")
            return copy.deepcopy(self._last_milestone_outcome)
        raw_env = resolve_raw_task_env(self.env)
        assert self._milestone_tracker is not None
        outcome = self._milestone_tracker.finalize(
            raw_env, termination_metadata=termination_metadata
        )
        self._last_milestone_outcome = outcome
        self._milestone_episode_active = False
        return copy.deepcopy(outcome)

    def get_milestone_configuration(self) -> dict[str, Any]:
        return copy.deepcopy(self._milestone_config)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    try:
        return {str(key): _plain_value(item) for key, item in value.items()}
    except AttributeError as error:
        raise TypeError("milestone configuration must be mapping-like") from error


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value
