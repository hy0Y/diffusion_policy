"""Additive MultiStep adapter that marks policy-call boundaries."""

from __future__ import annotations

from typing import Any, Mapping

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


class MilestoneMultiStepWrapper(MultiStepWrapper):
    """Preserve existing chunk execution while exposing physical-clock metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._milestone_policy_call = 0

    def reset(self):
        self._milestone_policy_call = 0
        return super().reset()

    def step(self, action):
        self._milestone_policy_call += 1
        begin = _find_inner_method(self.env, "begin_policy_call")
        if begin is not None:
            begin(self._milestone_policy_call, len(action))
        return super().step(action)

    def finalize_milestone_episode(
        self, termination_metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | None:
        finalize = _find_inner_method(self.env, "finalize_milestone_episode")
        if finalize is None:
            return None
        return finalize(termination_metadata)

    def get_milestone_configuration(self) -> dict[str, Any] | None:
        accessor = _find_inner_method(self.env, "get_milestone_configuration")
        return None if accessor is None else accessor()


def _find_inner_method(env: Any, name: str):
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, name, None)
        if callable(method):
            return method
        current = getattr(current, "env", None)
    return None
