"""Public, host-independent milestone data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence


JSONScalar = str | int | float | bool | None


@dataclass(frozen=True)
class EpisodeContext:
    """Identity and clock contract for one simulator episode."""

    task_name: str
    task_spec_id: str
    task_spec_version: str
    environment_source_revision: str
    reset_mode: str
    max_physical_steps: int
    environment_seed: int | None = None
    episode_root: str | None = None
    initial_state_index: int | None = None
    physical_step_semantics: str = (
        "one successfully committed RoboCasa env.step action"
    )
    extra: Mapping[str, JSONScalar] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MilestoneDefinition:
    """Stable milestone identity plus its prerequisite dependency."""

    id: str
    description: str
    prerequisites: tuple[str, ...] = ()
    required: bool = True
    trigger_semantics: str = "state_or_event_condition"
    occurrence: int = 0


@dataclass(frozen=True)
class SpecObservation:
    """Task-specific, read-only evidence sampled after one committed action."""

    primitive_values: Mapping[str, JSONScalar]
    milestone_conditions: Mapping[str, bool]
    raw_events: Sequence[str] = ()
    invalid_events: Sequence[str] = ()
    evidence_by_milestone: Mapping[str, Mapping[str, JSONScalar]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class SpecFinalState:
    """Terminal state used only for maintenance diagnostics."""

    primitive_values: Mapping[str, JSONScalar]
    maintained_milestones: Mapping[str, bool]
    metadata: Mapping[str, JSONScalar] = field(default_factory=dict)


class MilestoneSpec(Protocol):
    """Interface implemented by a versioned task-specific evaluator."""

    spec_id: str
    spec_version: str
    expected_task_name: str
    definitions: tuple[MilestoneDefinition, ...]
    thresholds: Mapping[str, JSONScalar]

    def reset(self, raw_env: Any, context: EpisodeContext) -> None: ...

    def observe(self, raw_env: Any, physical_step: int) -> SpecObservation: ...

    def finalize(self, raw_env: Any, physical_step: int) -> SpecFinalState: ...


def definition_ids(definitions: Sequence[MilestoneDefinition]) -> tuple[str, ...]:
    ids = tuple(definition.id for definition in definitions)
    if not ids:
        raise ValueError("a milestone spec must define at least one milestone")
    if len(ids) != len(set(ids)):
        raise ValueError(f"milestone ids must be unique: {ids}")
    known: set[str] = set()
    for definition in definitions:
        unknown = set(definition.prerequisites) - set(ids)
        if unknown:
            raise ValueError(
                f"{definition.id} has unknown prerequisites: {sorted(unknown)}"
            )
        if definition.id in definition.prerequisites:
            raise ValueError(f"{definition.id} cannot depend on itself")
        known.add(definition.id)
    return ids
