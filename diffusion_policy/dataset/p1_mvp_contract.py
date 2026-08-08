"""Resolve the locked GetToastedBread MVP split from public data products."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")
MVP_SPLIT_ID = "gettoastedbread_mvp_intersection_v1"
CANONICAL_SPLIT_VERSION = "composite_seen_90_5_5_v1"


@dataclass(frozen=True)
class P1MVPSplitContract:
    episode_ids: tuple[int, ...]
    annotation_root: Path
    cache_root: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_p1_mvp_split(
        split_artifact: str | Path,
        *,
        split: str,
        task: str = "GetToastedBread",
        annotation_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        canonical_sha256: str | None = None,
    ) -> P1MVPSplitContract:
    """Resolve either the old locked intersection or the public split manifest.

    The public 506-episode task split is intersected with the published feature
    cache inventory. This deterministically removes the six train episodes whose
    cache is absent and yields the MVP's locked 450/25/25 split.
    """
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    artifact_path = Path(split_artifact)
    manifest = json.loads(artifact_path.read_text(encoding="utf-8"))

    if manifest.get("split_id") == MVP_SPLIT_ID:
        source = manifest["source"]
        resolved_annotation = Path(annotation_root or source["annotation_root"])
        resolved_cache = Path(cache_root or source["cache_root"])
        episode_ids = tuple(int(value) for value in manifest["episodes"][split])
        return P1MVPSplitContract(
            episode_ids=episode_ids,
            annotation_root=resolved_annotation,
            cache_root=resolved_cache,
        )

    if manifest.get("version") != CANONICAL_SPLIT_VERSION:
        raise ValueError("unexpected P1 split artifact schema/version")
    if canonical_sha256 is not None:
        observed = _sha256(artifact_path)
        if observed != canonical_sha256:
            raise ValueError(
                f"canonical split SHA-256 mismatch: {observed} != "
                f"{canonical_sha256}")
    if task not in manifest.get("tasks", {}):
        raise KeyError(f"task {task!r} is absent from the canonical split")
    if annotation_root is None or cache_root is None:
        raise ValueError(
            "canonical split resolution requires annotation_root and cache_root")
    resolved_annotation = Path(annotation_root)
    resolved_cache = Path(cache_root)
    source_episode_ids = tuple(
        int(value) for value in manifest["tasks"][task][split])
    episode_ids = []
    for episode_id in source_episode_ids:
        stem = f"episode_{episode_id:06d}"
        required = (
            resolved_annotation / f"{stem}.parquet",
            resolved_cache / f"{stem}_features.npy",
            resolved_cache / f"{stem}_actions.npy",
        )
        if all(path.is_file() for path in required):
            episode_ids.append(episode_id)
    if not episode_ids:
        raise ValueError(f"no cached episodes remain in canonical {split} split")
    return P1MVPSplitContract(
        episode_ids=tuple(episode_ids),
        annotation_root=resolved_annotation,
        cache_root=resolved_cache,
    )
