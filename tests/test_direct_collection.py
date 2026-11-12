import csv
import json

import pytest

from gap_attractor.harness import runner
from gap_attractor.harness.runner import run_experiment
from gap_attractor.providers.base import ProviderResponse
from test_fixture_harness import ROOT, config


USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001}


def call(name, n, **arguments):
    return {"id": f"c{n}", "name": name, "arguments": arguments}


def reply(*calls):
    return ProviderResponse("", USAGE, "tool_calls", 0, None, [], list(calls))


class Scripted:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def chat(self, messages, *, seed, tools=None, **_):
        self.requests.append({"messages": messages, "tools": [t["function"]["name"] for t in tools or []]})
        if not self.replies:
            raise AssertionError("unexpected model call")
        return self.replies.pop(0)


def collection_config(tmp_path, **overrides):
    extra = [
        "experiment.memory=collection_path_only",
        "experiment.action_format=tools",
        "experiment.allow_stop=false",
        "experiment.collection_target=2",
        "experiment.collection_instruction=Build a two-page reference collection.",
        "experiment.step_limit=5",
    ]
    extra += [f"{key}={value}" for key, value in overrides.items()]
    return config(tmp_path, extra=extra)


def test_direct_collection_saves_current_pages_and_ends_at_target(tmp_path, monkeypatch):
    provider = Scripted([
        reply(call("save_page", 1, reason="broad index"), call("follow_link", 2, title="Archive")),
        reply(call("save_page", 3, reason="durable records"), call("follow_link", 4, title="Bridge")),
    ])
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, results = run_experiment(collection_config(tmp_path), ROOT)
    run_dir = next(path for path in dataset.root.iterdir() if (path / "run.json").exists())
    doc = json.loads((run_dir / "run.json").read_text())
    artifact = json.loads((run_dir / "collection.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]

    assert results[0]["termination_reason"] == "collection_complete"
    assert artifact == doc["collection"]
    assert artifact["complete"] is True
    assert [row["title"] for row in artifact["selections"]] == ["Civic Atlas", "Archive"]
    assert [row["step_index"] for row in artifact["selections"]] == [1, 2]
    assert [step["memory"]["collection_saved_this_step"] for step in steps] == [["Civic Atlas"], ["Archive"]]
    assert summary["collection_target"] == 2 and summary["collection_selected"] == 2
    assert summary["collection_complete"] is True and summary["n_steps"] == 2
    assert provider.requests[0]["tools"] == ["follow_link", "save_page"]
    assert "Saved 0/2" in provider.requests[0]["messages"][-1]["content"]
    with (dataset.root / "run_log.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert "Collection saved 2/2" in row["Notes"]


def test_direct_collection_keeps_incomplete_set_as_behavioral_outcome(tmp_path, monkeypatch):
    provider = Scripted([reply(call("follow_link", 1, title="Archive"))])
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    cfg = collection_config(tmp_path, **{"experiment.step_limit": 1})
    dataset, results = run_experiment(cfg, ROOT)
    run_dir = next(path for path in dataset.root.iterdir() if (path / "run.json").exists())
    artifact = json.loads((run_dir / "collection.json").read_text())
    assert results[0]["termination_reason"] == "step_limit"
    assert results[0]["status"] == "Completed"
    assert artifact["complete"] is False and artifact["selections"] == []


def test_collection_completed_on_final_allowed_step_is_not_labelled_step_limit(tmp_path, monkeypatch):
    provider = Scripted([reply(call("save_page", 1, reason="useful"), call("follow_link", 2, title="Archive"))])
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    cfg = collection_config(tmp_path, **{"experiment.collection_target": 1, "experiment.step_limit": 1})
    _, results = run_experiment(cfg, ROOT)
    assert results[0]["termination_reason"] == "collection_complete"
    assert results[0]["collection_complete"] is True


def test_collection_config_rejects_non_tool_or_stoppable_runs(tmp_path):
    with pytest.raises(ValueError, match="action_format=tools"):
        run_experiment(collection_config(tmp_path / "a", **{"experiment.action_format": "text"}), ROOT)
    with pytest.raises(ValueError, match="allow_stop=false"):
        run_experiment(collection_config(tmp_path / "b", **{"experiment.allow_stop": "true"}), ROOT)
