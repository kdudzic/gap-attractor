import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from gap_attractor.harness.foreign_path import ForeignPathMemory


_SPEC = importlib.util.spec_from_file_location(
    "build_foreign_path_control", Path(__file__).parents[1] / "scripts" / "build_foreign_path_control.py"
)
_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILDER)
build = _BUILDER.build


def config(path, *, action_format="text", step_limit=3):
    return OmegaConf.create({"experiment": {
        "foreign_history_file": str(path),
        "action_format": action_format,
        "step_limit": step_limit,
        "system_prompt": "system",
    }})


def history_file(tmp_path, items=None):
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "source_batch": "/tmp/baseline",
        "mapping": {"Spreadsheet": {"items": items or [
            {"reasoning": "r1", "choice": "A", "source_run_id": "run-a"},
            {"reasoning": "r2", "choice": "B", "source_run_id": "run-b"},
        ]}},
    }))
    return path


def test_foreign_path_uses_same_turn_shape_as_path_only_and_records_provenance(tmp_path):
    path = history_file(tmp_path)
    cfg = config(path)
    ForeignPathMemory.validate(cfg)
    memory = ForeignPathMemory(cfg)
    memory.start(SimpleNamespace(doc={"start_page": "Spreadsheet"}))

    messages = memory.messages("rule", [{}, {}], "current page")

    assert [m["role"] for m in messages] == ["system", "assistant", "assistant", "user"]
    assert json.loads(messages[1]["content"]) == {"reasoning": "r1", "chose": "A"}
    assert json.loads(messages[2]["content"]) == {"reasoning": "r2", "chose": "B"}
    assert messages[-1]["content"] == "current page"
    settings = memory.settings()
    assert settings["control"] == "foreign_path"
    assert settings["donor_run_ids"] == ["run-a", "run-b"]
    assert len(settings["history_file_sha256"]) == 64


def test_foreign_path_rejects_wrong_action_format_and_short_history(tmp_path):
    path = history_file(tmp_path, [{"reasoning": "r1", "choice": "A", "source_run_id": "run-a"}])
    with pytest.raises(ValueError, match="action_format=text"):
        ForeignPathMemory.validate(config(path, action_format="tools"))
    memory = ForeignPathMemory(config(path, step_limit=3))
    with pytest.raises(ValueError, match="1 turns; 2 required"):
        memory.start(SimpleNamespace(doc={"start_page": "Spreadsheet"}))


def test_builder_uses_other_runs_and_excludes_stop_and_harness_picks(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    rows = []
    for index, start in enumerate(("A", "B"), 1):
        run_dir = batch / f"run-{index}"
        run_dir.mkdir()
        steps = [
            {"step_index": 1, "parsed_choice": f"{start}-1", "model_reasoning": "ok",
             "invalid_action": {"harness_handling": None}, "resolution": {"resolve_status": "success"}},
            {"step_index": 2, "parsed_choice": "STOP", "model_reasoning": "done",
             "invalid_action": {"harness_handling": None}, "resolution": {"resolve_status": "success"}},
            {"step_index": 3, "parsed_choice": f"{start}-fallback", "model_reasoning": None,
             "invalid_action": {"harness_handling": "skip"}, "resolution": {"resolve_status": "success"}},
        ]
        (run_dir / "steps.jsonl").write_text("\n".join(json.dumps(step) for step in steps) + "\n")
        rows.append({"Run ID": f"run-{index}", "Start Page": start, "Artifact Path": str(run_dir)})
    with (batch / "run_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Run ID", "Start Page", "Artifact Path"])
        writer.writeheader()
        writer.writerows(rows)

    document = build(batch, turns=1)

    assert document["mapping"]["A"]["items"][0]["source_start_page"] == "B"
    assert document["mapping"]["B"]["items"][0]["source_start_page"] == "A"
    assert all(item["choice"] != "STOP" for record in document["mapping"].values() for item in record["items"])
