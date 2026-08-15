import json
from pathlib import Path

from robocasa_milestones.loader import load_milestone_spec
from robocasa_milestones.serialization import (
    sha256_file,
    write_episode_artifacts,
    write_manifest,
)


def test_exact_spec_target_and_identity_validation():
    spec = load_milestone_spec(
        "robocasa_milestones.specs.get_toasted_bread_v1:GetToastedBreadV1Spec",
        expected_task_name="GetToastedBread",
        expected_spec_id="robocasa365/GetToastedBread/physical_milestones",
        expected_spec_version="1.1.0",
    )
    assert len(spec.definitions) == 5


def test_episode_artifacts_and_manifest_have_content_hashes(tmp_path: Path):
    summary = {
        "task_spec_id": "fake/spec",
        "task_spec_version": "1.0.0",
        "physical_step_count": 2,
    }
    trace = [{"physical_step": 1}, {"physical_step": 2}]
    episode = write_episode_artifacts(
        tmp_path, episode_index=0, summary=summary, trace=trace
    )
    assert episode["trace_row_count"] == 2
    assert episode["trace_sha256"] == sha256_file(episode["trace_path"])

    manifest = write_manifest(tmp_path, episodes=[episode])
    payload = json.loads(Path(manifest["path"]).read_text())
    assert payload["status"] == "succeeded"
    assert payload["episode_count"] == 1
    assert manifest["sha256"] == sha256_file(manifest["path"])
    assert not list((tmp_path / "milestones").glob("*.tmp"))
