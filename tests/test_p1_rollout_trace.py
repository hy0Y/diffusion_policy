import json

import numpy as np
import pytest

from diffusion_policy.env_runner.p1_rollout_trace import (
    P1RolloutTraceWriter, committed_first_edge,
)


def test_committed_trace_selects_only_first_edge_and_executed_action(tmp_path):
    probe = {
        "action_time": [0, 1], "physical_time": [1, 2],
        "probability": [0.2, 0.9],
        "intensity": [0.3, 0.8], "gate": [0.5, 0.7],
        "z": [[1, 2], [4, 6], [20, 30]],
        "z_minus": [[2, 3], [10, 10]],
        "flow_delta": [[1, 1], [8, 8]],
        "jump_delta": [[2, 4], [9, 9]],
        "total_delta_norm": [5.0, 99.0], "d_flow": [1.4, 88.0],
        "d_jump": [2.2, 77.0], "d_consistency": 0.125,
    }
    probe["full_diffusion_trace_stored"] = True
    probe["diffusion_trace"] = [
        {
            "timestep": 9, "normalized_tau": 1.0,
            "initial_z": [[0, 0]], "z": [[[1, 2], [3, 5]]],
            "z_minus": [[[2, 3], [0, 0]]],
            "flow_delta": [[[1, 1]]], "jump_delta": [[[2, 4]]],
            "probability": [[0.1]], "intensity": [[0.2]],
            "gate": [[0.3]], "total_delta_norm": [[3.6]],
            "d_flow": [[1.4]], "d_jump": [[1.3]],
            "d_consistency": None,
        },
        {
            "timestep": 0, "normalized_tau": 0.0,
            "initial_z": [[0, 0]], "z": [[[1, 2], [4, 6]]],
            "z_minus": [[[2, 3], [0, 0]]],
            "flow_delta": [[[1, 1]]], "jump_delta": [[[2, 4]]],
            "probability": [[0.2]], "intensity": [[0.3]],
            "gate": [[0.5]], "total_delta_norm": [[5.0]],
            "d_flow": [[1.4]], "d_jump": [[2.2]],
            "d_consistency": [0.125],
        },
    ]
    row = committed_first_edge(
        probe, np.array([1, 2]), np.array([3, 4]), oracle_subtask_idx=6)
    assert row["action_time"] == 0
    assert row["physical_time"] == 1
    assert row["oracle_subtask_idx"] == 6
    assert row["oracle_subtask_time"] == 0
    np.testing.assert_array_equal(row["z_before"], [1, 2])
    np.testing.assert_array_equal(row["z_after"], [4, 6])
    np.testing.assert_array_equal(row["applied_jump_delta"], [1, 2])
    np.testing.assert_array_equal(row["executed_action"], [3, 4])
    assert row["reference_kind"] == "none"
    assert row["reference_action"] is None

    invalid_probe = {**probe, "physical_time": [0, 1]}
    with pytest.raises(
        RuntimeError, match=r"physical_time must equal action_time \+ 1"
    ):
        committed_first_edge(invalid_probe, np.array([1, 2]), np.array([3, 4]))

    writer = P1RolloutTraceWriter(tmp_path, {"p1_mode": "oracle_gate_symbol"})
    writer.append(row)
    artifacts = writer.finalize()
    scalar = json.loads((tmp_path / "rollout_trace.jsonl").read_text())
    assert scalar["action_time"] == 0
    assert scalar["physical_time"] == 1
    assert "z_before" not in scalar
    assert scalar["d_consistency"] == 0.125
    with np.load(tmp_path / "trajectory_signals.npz") as arrays:
        assert arrays["z_before"].shape == (1, 2)
        assert arrays["executed_action"].dtype == np.float32
        assert "reference_action" not in arrays
    diffusion_rows = [
        json.loads(line)
        for line in (tmp_path / "diffusion_scalar_trace.jsonl")
        .read_text().splitlines()
    ]
    np.testing.assert_allclose(diffusion_rows[0]["probability"], [0.1, 0.2])
    assert diffusion_rows[0]["d_consistency"] == [None, 0.125]
    with np.load(tmp_path / "diffusion_signals.npz") as arrays:
        assert arrays["probability"].shape == (1, 2)
        assert arrays["z_after"].shape == (1, 2, 2)
        np.testing.assert_array_equal(arrays["action_time"], [0])
        np.testing.assert_array_equal(arrays["physical_time"], [1])
        np.testing.assert_array_equal(arrays["scheduler_timestep"], [9, 0])
    manifest = json.loads(
        (tmp_path / "diffusion_signal_manifest.json").read_text())
    assert manifest["tau_aliases"]["noise"]["diffusion_index"] == 0
    assert manifest["tau_aliases"]["clean"]["diffusion_index"] == 1
    assert artifacts["p1_trace_steps"] == 1
    assert len(artifacts["p1_trace_npz_sha256"]) == 64
    assert manifest["schema_version"] == "2.1"
    assert manifest["time_semantics"]["p0_defined"] is False
    assert len(artifacts["p1_diffusion_signals_npz_sha256"]) == 64

    metadata = json.loads(
        (tmp_path / "trace_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["reference_kind"] == "none"
    assert metadata["reference_action_present"] is False

    offline_dir = tmp_path / "offline"
    offline_row = committed_first_edge(
        probe,
        np.array([1, 2]),
        np.array([np.nan, np.nan]),
        reference_action=np.array([5, 6]),
        reference_kind="demonstration",
    )
    offline_writer = P1RolloutTraceWriter(
        offline_dir, {"evaluation_mode": "offline_validation"}
    )
    offline_writer.append(offline_row)
    offline_writer.finalize()
    with np.load(offline_dir / "trajectory_signals.npz") as arrays:
        np.testing.assert_array_equal(arrays["reference_action"], [[5, 6]])
    offline_metadata = json.loads(
        (offline_dir / "trace_metadata.json").read_text(encoding="utf-8")
    )
    assert offline_metadata["reference_kind"] == "demonstration"
    assert offline_metadata["reference_action_present"] is True

    mixed_writer = P1RolloutTraceWriter(tmp_path / "mixed", {})
    mixed_writer.append(row)
    with pytest.raises(RuntimeError, match="reference_kind changed"):
        mixed_writer.append({
            **offline_row,
            "action_time": 1,
            "physical_time": 2,
        })
    mixed_writer.abort()

    with pytest.raises(
        ValueError, match="reference_kind=none cannot contain reference_action"
    ):
        committed_first_edge(
            probe,
            np.array([1, 2]),
            np.array([3, 4]),
            reference_action=np.array([3, 4]),
        )
