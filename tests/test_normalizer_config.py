import pytest
import torch

from diffusion_policy.model.common.normalizer_config import (
    linear_normalizer_from_spec,
    normalizer_transform_sha256,
)


IDENTITY_SHA256 = "1788614c245557148e030eaab54f5a37bc5706b9fcad0af043a5307e3db08d5a"


def test_linear_normalizer_spec_builds_and_checks_transform_sha256():
    normalizer = linear_normalizer_from_spec({
        "format_version": "linear_normalizer_transform_v1",
        "transform_sha256": IDENTITY_SHA256,
        "fields": {"state": {"kind": "identity", "size": 1}},
    })

    torch.testing.assert_close(
        normalizer.normalize({"state": torch.tensor([[3.0]])})["state"],
        torch.tensor([[3.0]]),
    )
    assert normalizer_transform_sha256(normalizer) == IDENTITY_SHA256


def test_linear_normalizer_spec_rejects_wrong_transform_sha256():
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        linear_normalizer_from_spec({
            "format_version": "linear_normalizer_transform_v1",
            "transform_sha256": "0" * 64,
            "fields": {"state": {"kind": "identity", "size": 1}},
        })
