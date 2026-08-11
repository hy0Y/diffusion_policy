from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_VECTOR_FIELDS = (
    "z_before", "z_minus", "z_after", "total_delta", "flow_delta",
    "raw_jump_delta", "applied_jump_delta", "policy_action", "executed_action",
)

OPTIONAL_VECTOR_FIELDS = ("reference_action",)
VECTOR_FIELDS = REQUIRED_VECTOR_FIELDS + OPTIONAL_VECTOR_FIELDS
REFERENCE_KINDS = {"none", "demonstration"}

DIFFUSION_SCALAR_FIELDS = (
    "probability", "intensity", "gate", "total_delta_norm", "d_flow",
    "d_jump", "d_consistency",
)

DIFFUSION_VECTOR_FIELDS = (
    "z_before", "z_minus", "z_after", "total_delta", "flow_delta",
    "raw_jump_delta", "applied_jump_delta",
)

TIME_SEMANTICS = {
    "action_time": "edge_start_t",
    "physical_time": "edge_endpoint_t_plus_1",
    "probability": "p(physical_time)=p(action_time->physical_time)",
    "p0_defined": False,
}


def committed_diffusion_trace(probe: dict[str, Any]) -> dict[str, np.ndarray]:
    """Materialize every diffusion timestep for the first committed edge."""
    trace = probe.get("diffusion_trace")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("P1 policy did not publish a diffusion trace")
    if not probe.get("full_diffusion_trace_stored"):
        raise RuntimeError("P1 policy did not store full diffusion diagnostics")

    scalars: dict[str, list[float]] = {
        name: [] for name in DIFFUSION_SCALAR_FIELDS
    }
    vectors: dict[str, list[np.ndarray]] = {
        name: [] for name in DIFFUSION_VECTOR_FIELDS
    }
    timesteps: list[int] = []
    normalized_tau: list[float] = []
    for index, step in enumerate(trace):
        required = {
            "timestep", "normalized_tau", "initial_z", "z", "z_minus",
            "flow_delta", "jump_delta", "probability", "intensity", "gate",
            "total_delta_norm", "d_flow", "d_jump", "d_consistency",
        }
        missing = sorted(required - set(step))
        if missing:
            raise RuntimeError(
                f"diffusion trace row {index} is missing fields: {missing}"
            )
        z = np.asarray(step["z"], dtype=np.float32)[0]
        z_minus = np.asarray(step["z_minus"], dtype=np.float32)[0]
        flow_delta = np.asarray(step["flow_delta"], dtype=np.float32)[0]
        raw_jump = np.asarray(step["jump_delta"], dtype=np.float32)[0]
        probability = np.asarray(step["probability"], dtype=np.float32)[0]
        intensity = np.asarray(step["intensity"], dtype=np.float32)[0]
        gate = np.asarray(step["gate"], dtype=np.float32)[0]
        total_delta_norm = np.asarray(
            step["total_delta_norm"], dtype=np.float32)[0]
        d_flow = np.asarray(step["d_flow"], dtype=np.float32)[0]
        d_jump = np.asarray(step["d_jump"], dtype=np.float32)[0]
        if z.shape[0] < 2 or probability.shape[0] < 1:
            raise RuntimeError("diffusion trace has no committed edge")
        consistency = step["d_consistency"]
        consistency_value = (
            np.nan if consistency is None
            else float(np.asarray(consistency, dtype=np.float32)[0])
        )
        timesteps.append(int(step["timestep"]))
        normalized_tau.append(float(step["normalized_tau"]))
        scalars["probability"].append(float(probability[0]))
        scalars["intensity"].append(float(intensity[0]))
        scalars["gate"].append(float(gate[0]))
        scalars["total_delta_norm"].append(float(total_delta_norm[0]))
        scalars["d_flow"].append(float(d_flow[0]))
        scalars["d_jump"].append(float(d_jump[0]))
        scalars["d_consistency"].append(consistency_value)
        vectors["z_before"].append(z[0])
        vectors["z_minus"].append(z_minus[0])
        vectors["z_after"].append(z[1])
        vectors["total_delta"].append(z[1] - z[0])
        vectors["flow_delta"].append(flow_delta[0])
        vectors["raw_jump_delta"].append(raw_jump[0])
        vectors["applied_jump_delta"].append(gate[0] * raw_jump[0])

    return {
        "scheduler_timestep": np.asarray(timesteps, dtype=np.int32),
        "normalized_tau": np.asarray(normalized_tau, dtype=np.float32),
        **{
            name: np.asarray(values, dtype=np.float32)
            for name, values in scalars.items()
        },
        **{
            name: np.stack(values).astype(np.float32, copy=False)
            for name, values in vectors.items()
        },
    }

def committed_first_edge(
        probe: dict[str, Any], policy_action: np.ndarray,
        executed_action: np.ndarray, oracle_subtask_idx: int | None = None,
        reference_action: np.ndarray | None = None,
        reference_kind: str = "none",
    ) -> dict[str, Any]:
    """Select the only latent edge committed by a one-action-step rollout."""
    if not probe:
        raise RuntimeError("P1 policy did not publish last_online_probe")
    action_times = np.asarray(probe.get("action_time", [])).reshape(-1)
    endpoint_times = np.asarray(probe.get("physical_time", [])).reshape(-1)
    if action_times.size < 1 or endpoint_times.size < 1:
        raise RuntimeError("P1 probe did not publish action/endpoint time")
    action_time = int(action_times[0])
    physical_time = int(endpoint_times[0])
    if physical_time < 1 or physical_time != action_time + 1:
        raise RuntimeError(
            "P1 endpoint contract violated: physical_time must equal "
            "action_time + 1 and p(0) is undefined"
        )
    gate = float(probe["gate"][0])
    raw_jump = np.asarray(probe["jump_delta"][0], dtype=np.float32)
    if reference_kind not in REFERENCE_KINDS:
        raise ValueError(f"unsupported reference_kind: {reference_kind}")
    if reference_kind == "none":
        if reference_action is not None:
            raise ValueError(
                "reference_kind=none cannot contain reference_action"
            )
        committed_reference = None
    else:
        if reference_action is None:
            raise ValueError(
                "reference_kind=demonstration requires reference_action"
            )
        committed_reference = np.asarray(reference_action, dtype=np.float32)
    return {
        "action_time": action_time,
        "physical_time": physical_time,
        "oracle_subtask_idx": oracle_subtask_idx,
        "oracle_subtask_time": (
            action_time if oracle_subtask_idx is not None else None
        ),
        "probability": float(probe["probability"][0]),
        "intensity": float(probe["intensity"][0]),
        "gate": gate,
        "total_delta_norm": float(probe["total_delta_norm"][0]),
        "d_flow": float(probe["d_flow"][0]),
        "d_jump": float(probe["d_jump"][0]),
        "d_consistency": probe.get("d_consistency"),
        "flow_delta_norm": float(probe["d_flow"][0]),
        "raw_jump_delta_norm": float(np.linalg.norm(raw_jump)),
        "applied_jump_delta_norm": float(probe["d_jump"][0]),
        "z_before": np.asarray(probe["z"][0], dtype=np.float32),
        "z_minus": np.asarray(probe["z_minus"][0], dtype=np.float32),
        "z_after": np.asarray(probe["z"][1], dtype=np.float32),
        "total_delta": (np.asarray(probe["z"][1], dtype=np.float32)
                        - np.asarray(probe["z"][0], dtype=np.float32)),
        "flow_delta": np.asarray(probe["flow_delta"][0], dtype=np.float32),
        "raw_jump_delta": raw_jump,
        "applied_jump_delta": gate * raw_jump,
        "policy_action": np.asarray(policy_action, dtype=np.float32),
        "executed_action": np.asarray(executed_action, dtype=np.float32),
        "reference_kind": reference_kind,
        "reference_action": committed_reference,
        "_diffusion": committed_diffusion_trace(probe),
    }


class P1RolloutTraceWriter:
    """Stream compact rollout rows plus every diffusion-time diagnostic."""

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "rollout_trace.jsonl"
        self.npz_path = self.output_dir / "trajectory_signals.npz"
        self.metadata_path = self.output_dir / "trace_metadata.json"
        self.diffusion_jsonl_path = (
            self.output_dir / "diffusion_scalar_trace.jsonl")
        self.diffusion_npz_path = self.output_dir / "diffusion_signals.npz"
        self.diffusion_manifest_path = (
            self.output_dir / "diffusion_signal_manifest.json")
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")
        self._diffusion_jsonl = self.diffusion_jsonl_path.open(
            "w", encoding="utf-8")
        self._vectors = {name: [] for name in VECTOR_FIELDS}
        self._diffusion_scalars = {
            name: [] for name in DIFFUSION_SCALAR_FIELDS}
        self._diffusion_vectors = {
            name: [] for name in DIFFUSION_VECTOR_FIELDS}
        self._action_time: list[int] = []
        self._physical_time: list[int] = []
        self._scheduler_timestep: np.ndarray | None = None
        self._normalized_tau: np.ndarray | None = None
        self._metadata = dict(metadata)
        self._reference_kind: str | None = None
        self._steps = 0

    @staticmethod
    def _json_float_array(values: np.ndarray) -> list[float | None]:
        return [
            float(value) if np.isfinite(value) else None
            for value in np.asarray(values).reshape(-1)
        ]

    def append(self, row: dict[str, Any]) -> None:
        diffusion = row.get("_diffusion")
        if not isinstance(diffusion, dict):
            raise RuntimeError("committed row has no full diffusion trace")
        action_time = int(row["action_time"])
        physical_time = int(row["physical_time"])
        if action_time != self._steps or physical_time != self._steps + 1:
            raise RuntimeError(
                "P1 trace must contain contiguous edges t->t+1 beginning at "
                "action_time=0, physical_time=1"
            )
        reference_kind = row.get("reference_kind")
        if reference_kind not in REFERENCE_KINDS:
            raise RuntimeError(
                f"committed row has invalid reference_kind: {reference_kind}"
            )
        reference_action = row.get("reference_action")
        if reference_kind == "none" and reference_action is not None:
            raise RuntimeError(
                "reference_kind=none cannot contain reference_action"
            )
        if reference_kind == "demonstration" and reference_action is None:
            raise RuntimeError(
                "reference_kind=demonstration requires reference_action"
            )
        if (
            self._reference_kind is not None
            and reference_kind != self._reference_kind
        ):
            raise RuntimeError("reference_kind changed within one trace")

        vector_row = {
            name: np.asarray(row[name], dtype=np.float32)
            for name in REQUIRED_VECTOR_FIELDS
        }
        if reference_action is not None:
            vector_row["reference_action"] = np.asarray(
                reference_action, dtype=np.float32
            )
        scalar = {
            key: value for key, value in row.items()
            if key not in VECTOR_FIELDS and key != "_diffusion"
        }
        self._jsonl.write(json.dumps(scalar, sort_keys=True) + "\n")
        self._jsonl.flush()
        for name, values in vector_row.items():
            self._vectors[name].append(values)
        self._reference_kind = reference_kind

        scheduler_timestep = np.asarray(
            diffusion["scheduler_timestep"], dtype=np.int32)
        normalized_tau = np.asarray(
            diffusion["normalized_tau"], dtype=np.float32)
        if self._scheduler_timestep is None:
            self._scheduler_timestep = scheduler_timestep
            self._normalized_tau = normalized_tau
        elif (
            not np.array_equal(self._scheduler_timestep, scheduler_timestep)
            or not np.allclose(
                self._normalized_tau, normalized_tau, rtol=0, atol=1.0e-7)
        ):
            raise RuntimeError("diffusion schedule changed within one rollout")

        diffusion_scalar_row: dict[str, Any] = {
            "action_time": action_time,
            "physical_time": physical_time,
            "scheduler_timestep": scheduler_timestep.tolist(),
            "normalized_tau": self._json_float_array(normalized_tau),
        }
        for name in DIFFUSION_SCALAR_FIELDS:
            values = np.asarray(diffusion[name], dtype=np.float32)
            if values.shape != scheduler_timestep.shape:
                raise RuntimeError(
                    f"diffusion scalar {name} has shape {values.shape}, "
                    f"expected {scheduler_timestep.shape}"
                )
            self._diffusion_scalars[name].append(values)
            diffusion_scalar_row[name] = self._json_float_array(values)
        for name in DIFFUSION_VECTOR_FIELDS:
            values = np.asarray(diffusion[name], dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != scheduler_timestep.size:
                raise RuntimeError(
                    f"diffusion vector {name} has invalid shape {values.shape}"
                )
            self._diffusion_vectors[name].append(values)

        self._diffusion_jsonl.write(
            json.dumps(
                diffusion_scalar_row, allow_nan=False, sort_keys=True
            ) + "\n"
        )
        self._diffusion_jsonl.flush()
        self._action_time.append(action_time)
        self._physical_time.append(physical_time)
        self._steps += 1

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _tau_aliases(self) -> dict[str, dict[str, float | int]]:
        if self._scheduler_timestep is None or self._normalized_tau is None:
            raise RuntimeError("cannot define tau aliases for an empty trace")
        last = self._scheduler_timestep.size - 1
        alias_indices = {
            "noise": 0,
            "mid2": int(round(last / 3)),
            "mid1": int(round(2 * last / 3)),
            "clean": last,
        }
        return {
            alias: {
                "diffusion_index": index,
                "scheduler_timestep": int(self._scheduler_timestep[index]),
                "normalized_tau": float(self._normalized_tau[index]),
            }
            for alias, index in alias_indices.items()
        }

    def finalize(self) -> dict[str, Any]:
        self._jsonl.close()
        self._diffusion_jsonl.close()
        if self._steps == 0:
            raise RuntimeError("cannot finalize an empty P1 rollout trace")
        arrays = {
            name: np.stack(values).astype(np.float32, copy=False)
            for name, values in self._vectors.items() if values
        }
        np.savez_compressed(self.npz_path, **arrays)

        diffusion_arrays: dict[str, np.ndarray] = {
            "action_time": np.asarray(self._action_time, dtype=np.int32),
            "physical_time": np.asarray(self._physical_time, dtype=np.int32),
            "scheduler_timestep": self._scheduler_timestep,
            "normalized_tau": self._normalized_tau,
        }
        diffusion_arrays.update({
            name: np.stack(values).astype(np.float32, copy=False)
            for name, values in self._diffusion_scalars.items()
        })
        diffusion_arrays.update({
            name: np.stack(values).astype(np.float32, copy=False)
            for name, values in self._diffusion_vectors.items()
        })
        np.savez_compressed(self.diffusion_npz_path, **diffusion_arrays)

        tau_aliases = self._tau_aliases()
        signal_manifest = {
            "schema_version": "2.1",
            "selection": "first_committed_edge_at_every_diffusion_timestep",
            "time_semantics": TIME_SEMANTICS,
            "axis_order": {
                "scalar": ["physical_time", "diffusion_time"],
                "vector": ["physical_time", "diffusion_time", "latent"],
            },
            "steps": self._steps,
            "diffusion_steps": int(self._scheduler_timestep.size),
            "tau_order": "noise_to_clean",
            "tau_aliases": tau_aliases,
            "scalar_signals": {
                name: {
                    "dtype": "float32",
                    "shape": list(diffusion_arrays[name].shape),
                }
                for name in DIFFUSION_SCALAR_FIELDS
            },
            "vector_signals": {
                name: {
                    "dtype": "float32",
                    "shape": list(diffusion_arrays[name].shape),
                }
                for name in DIFFUSION_VECTOR_FIELDS
            },
            "web_scalar_jsonl": self.diffusion_jsonl_path.name,
            "full_signal_npz": self.diffusion_npz_path.name,
            "web_scalar_jsonl_sha256": self._sha256(
                self.diffusion_jsonl_path),
            "full_signal_npz_sha256": self._sha256(
                self.diffusion_npz_path),
        }
        self.diffusion_manifest_path.write_text(
            json.dumps(signal_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata = {
            **self._metadata,
            "schema_version": "2.1",
            "selection": "first_committed_edge_only",
            "time_semantics": TIME_SEMANTICS,
            "diffusion_selection": (
                "first_committed_edge_at_every_diffusion_timestep"),
            "steps": self._steps,
            "vector_dtype": "float32",
            "reference_kind": self._reference_kind,
            "reference_action_present": "reference_action" in arrays,
            "trajectory_vector_signals": sorted(arrays),
            "jsonl_path": str(self.jsonl_path),
            "npz_path": str(self.npz_path),
            "diffusion_jsonl_path": str(self.diffusion_jsonl_path),
            "diffusion_npz_path": str(self.diffusion_npz_path),
            "diffusion_manifest_path": str(self.diffusion_manifest_path),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "p1_trace_jsonl": str(self.jsonl_path),
            "p1_trace_npz": str(self.npz_path),
            "p1_trace_metadata": str(self.metadata_path),
            "p1_trace_steps": self._steps,
            "p1_trace_jsonl_sha256": self._sha256(self.jsonl_path),
            "p1_trace_npz_sha256": self._sha256(self.npz_path),
            "p1_diffusion_scalar_jsonl": str(self.diffusion_jsonl_path),
            "p1_diffusion_signals_npz": str(self.diffusion_npz_path),
            "p1_diffusion_signal_manifest": str(
                self.diffusion_manifest_path),
            "p1_diffusion_scalar_jsonl_sha256": self._sha256(
                self.diffusion_jsonl_path),
            "p1_diffusion_signals_npz_sha256": self._sha256(
                self.diffusion_npz_path),
        }

    def abort(self) -> None:
        if not self._jsonl.closed:
            self._jsonl.close()
        if not self._diffusion_jsonl.closed:
            self._diffusion_jsonl.close()
