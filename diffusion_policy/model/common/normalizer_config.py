from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

import torch

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


FORMAT_VERSION = "linear_normalizer_transform_v1"


def normalizer_transform_payload(
        normalizer: LinearNormalizer) -> dict[str, dict[str, list[float]]]:
    """Return the inference-relevant affine transform in canonical form."""
    payload: dict[str, dict[str, list[float]]] = {}
    for key in sorted(normalizer.params_dict.keys()):
        params = normalizer.params_dict[key]
        payload[key] = {
            "scale": params["scale"].detach().cpu().to(
                dtype=torch.float32).flatten().tolist(),
            "offset": params["offset"].detach().cpu().to(
                dtype=torch.float32).flatten().tolist(),
        }
    return payload


def normalizer_transform_sha256(normalizer: LinearNormalizer) -> str:
    encoded = json.dumps(
        normalizer_transform_payload(normalizer),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_affine(
        key: str, field: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
    kind = str(field.get("kind", ""))
    if kind == "identity":
        size = int(field.get("size", 0))
        if size <= 0:
            raise ValueError(f"normalizer field {key!r} has invalid identity size")
        return torch.ones(size, dtype=torch.float32), torch.zeros(
            size, dtype=torch.float32)
    if kind == "image_range":
        return torch.tensor([2.0], dtype=torch.float32), torch.tensor(
            [-1.0], dtype=torch.float32)
    if kind == "affine":
        scale = torch.as_tensor(field.get("scale"), dtype=torch.float32).flatten()
        offset = torch.as_tensor(field.get("offset"), dtype=torch.float32).flatten()
        if scale.numel() == 0 or scale.shape != offset.shape:
            raise ValueError(
                f"normalizer field {key!r} scale/offset shapes do not match")
        if not bool(torch.isfinite(scale).all() and torch.isfinite(offset).all()):
            raise ValueError(f"normalizer field {key!r} contains non-finite values")
        if any(math.isclose(float(value), 0.0) for value in scale):
            raise ValueError(f"normalizer field {key!r} contains zero scale")
        return scale, offset
    raise ValueError(f"normalizer field {key!r} has unsupported kind {kind!r}")


def linear_normalizer_from_spec(spec: Mapping[str, Any]) -> LinearNormalizer:
    """Build and SHA-check a compact, inference-complete normalizer spec."""
    if str(spec.get("format_version", "")) != FORMAT_VERSION:
        raise ValueError("unsupported normalizer spec format_version")
    fields = spec.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("normalizer spec requires non-empty fields")

    normalizer = LinearNormalizer()
    for key, field in fields.items():
        if not isinstance(field, Mapping):
            raise TypeError(f"normalizer field {key!r} must be a mapping")
        scale, offset = _field_affine(str(key), field)
        normalizer[str(key)] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict={},
        )

    expected_sha256 = str(spec.get("transform_sha256", ""))
    actual_sha256 = normalizer_transform_sha256(normalizer)
    if not expected_sha256:
        raise ValueError("normalizer spec requires transform_sha256")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "normalizer transform SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return normalizer
