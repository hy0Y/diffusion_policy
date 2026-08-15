"""Exact import-target loader without a central task registry."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from robocasa_milestones.contracts import definition_ids


def load_milestone_spec(
    target: str,
    *,
    expected_task_name: str,
    expected_spec_id: str,
    expected_spec_version: str,
    spec_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    if ":" not in target:
        raise ValueError("spec target must use 'module:ClassName' syntax")
    module_name, attribute_name = target.split(":", maxsplit=1)
    if not module_name or not attribute_name:
        raise ValueError(f"invalid spec target: {target!r}")
    module = importlib.import_module(module_name)
    spec_type = getattr(module, attribute_name)
    spec = spec_type(**dict(spec_kwargs or {}))
    identities = {
        "expected_task_name": (spec.expected_task_name, expected_task_name),
        "spec_id": (spec.spec_id, expected_spec_id),
        "spec_version": (spec.spec_version, expected_spec_version),
    }
    for label, (actual, expected) in identities.items():
        if str(actual) != str(expected):
            raise ValueError(
                f"loaded spec {label} mismatch: actual={actual!r}, expected={expected!r}"
            )
    definition_ids(spec.definitions)
    return spec
