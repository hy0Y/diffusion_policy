"""Atomic JSON/JSONL milestone artifact serialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_SCHEMA_VERSION = "robocasa_milestone_manifest_v1"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_text(destination, text)
    return destination


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> tuple[Path, int]:
    destination = Path(path)
    row_list = list(rows)
    text = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in row_list
    )
    _atomic_write_text(destination, text)
    return destination, len(row_list)


def write_episode_artifacts(
    output_dir: str | Path,
    *,
    episode_index: int,
    summary: Mapping[str, Any],
    trace: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(output_dir) / "milestones"
    stem = f"episode_{int(episode_index):06d}"
    summary_path = write_json(root / f"{stem}.summary.json", summary)
    trace_path, row_count = write_jsonl(root / f"{stem}.trace.jsonl", trace)
    return {
        "episode_index": int(episode_index),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "trace_row_count": row_count,
        "task_spec_id": summary["task_spec_id"],
        "task_spec_version": summary["task_spec_version"],
        "status": "succeeded",
    }


def write_manifest(
    output_dir: str | Path,
    *,
    episodes: list[Mapping[str, Any]],
    source_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir) / "milestones"
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "succeeded",
        "episode_count": len(episodes),
        "episodes": [dict(episode) for episode in episodes],
        "source_provenance": dict(source_provenance or {}),
    }
    path = write_json(root / "manifest.json", manifest)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "episode_count": len(episodes),
        "schema_version": MANIFEST_SCHEMA_VERSION,
    }
