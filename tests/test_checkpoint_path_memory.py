import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from gap_attractor.harness.checkpoint_path import CheckpointPathMemory


_SPEC = importlib.util.spec_from_file_location(
    "build_checkpoint_path", Path(__file__).parents[1] / "scripts" / "build_checkpoint_path.py"
)
_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILDER)


def config(path, *, action_format="text"):
    return OmegaConf.create({"experiment": {
        "checkpoint_history_file": str(path),
        "action_format": action_format,
        "system_prompt": "system",
    }})


def checkpoint_file(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "source_batch": "batch-prefix",
        "mapping": {"Canton of Glarus": {"items": [
            {"reasoning": "r1", "choice": "A", "source_run_id": "run-prefix", "source_step": 1},
            {"reasoning": "r2", "choice": "B", "source_run_id": "run-prefix", "source_step": 2},
        ]}},
    }))
    return path


def test_checkpoint_path_prepends_prefix_and_then_live_history(tmp_path):
    path = checkpoint_file(tmp_path)
    cfg = config(path)
    CheckpointPathMemory.validate(cfg)
    memory = CheckpointPathMemory(cfg)
    memory.start(SimpleNamespace(doc={"start_page": "Canton of Glarus"}))

    messages = memory.messages("rule", [{"reasoning": "live", "choice": "C"}], "current page")

    assert [message["role"] for message in messages] == ["system", "assistant", "assistant", "assistant", "user"]
    assert json.loads(messages[1]["content"]) == {"reasoning": "r1", "chose": "A"}
    assert json.loads(messages[3]["content"]) == {"reasoning": "live", "chose": "C"}
    settings = memory.settings()
    assert settings["checkpoint_turns"] == 2
    assert settings["source_run_ids"] == ["run-prefix"]
    assert len(settings["history_file_sha256"]) == 64


def test_checkpoint_path_rejects_tools_and_missing_recipient(tmp_path):
    path = checkpoint_file(tmp_path)
    with pytest.raises(ValueError, match="action_format=text"):
        CheckpointPathMemory.validate(config(path, action_format="tools"))
    memory = CheckpointPathMemory(config(path))
    with pytest.raises(ValueError, match="no checkpoint history"):
        memory.start(SimpleNamespace(doc={"start_page": "Spreadsheet"}))


def test_builder_keeps_successful_model_moves_and_excludes_fallbacks(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "batch_id": "batch-prefix", "run_id": "run-prefix", "start_page": "Spreadsheet",
    }))
    steps = [
        {"step_index": 1, "parsed_choice": "A", "model_reasoning": "ok",
         "invalid_action": {"harness_handling": None}, "resolution": {"resolve_status": "success"}},
        {"step_index": 2, "parsed_choice": "B", "model_reasoning": None,
         "invalid_action": {"harness_handling": "skip"}, "resolution": {"resolve_status": "success"}},
    ]
    (run_dir / "steps.jsonl").write_text("\n".join(json.dumps(step) for step in steps) + "\n")

    document = _BUILDER.build(run_dir, ["Canton of Glarus"])

    items = document["mapping"]["Canton of Glarus"]["items"]
    assert [item["choice"] for item in items] == ["A"]
