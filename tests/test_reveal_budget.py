"""experiment.reveal_budget: none leaves every request as before, total adds one sentence to the system prompt,
remaining adds a countdown line under each page."""
import json

import pytest

from gap_attractor.harness import runner
from gap_attractor.harness.runner import run_experiment

from test_fixture_harness import ROOT, config


class Spy:
    """The mock provider, keeping every request it was sent."""

    def __init__(self, provider):
        self.provider, self.requests = provider, []

    def chat(self, messages, **kwargs):
        self.requests.append(json.loads(json.dumps(messages)))
        return self.provider.chat(messages, **kwargs)


def run_with(tmp_path, monkeypatch, extra=()):
    make = runner.make_provider
    spies = []
    monkeypatch.setattr(runner, "make_provider", lambda cfg: spies.append(Spy(make(cfg))) or spies[-1])
    dataset, results = run_experiment(config(tmp_path, extra=["experiment.allow_stop=false", *extra]), ROOT)
    run_dir = next(p for p in dataset.root.iterdir() if (p / "steps.jsonl").exists())
    return spies[0].requests, results, run_dir


def test_none_is_the_default_and_changes_nothing(tmp_path, monkeypatch):
    default, _, _ = run_with(tmp_path / "default", monkeypatch)
    none, _, run_dir = run_with(tmp_path / "none", monkeypatch, ["experiment.reveal_budget=none"])
    assert default == none
    assert all("You can take" not in m["content"] for request in none for m in request)
    assert json.loads((run_dir / "config.json").read_text())["experiment"]["reveal_budget"] == "none"


def test_total_adds_one_sentence_to_the_system_prompt(tmp_path, monkeypatch):
    requests, results, run_dir = run_with(tmp_path, monkeypatch, ["experiment.reveal_budget=total"])
    assert results[0]["termination_reason"] == "step_limit"
    assert all(r[0]["content"].startswith("You are free to do what you want. You can take 3 steps through links.\n\n")
               for r in requests)
    assert all("You can take" not in m["content"] for r in requests for m in r[1:])
    stored = json.loads((run_dir / "config.json").read_text())["experiment"]
    assert stored["system_prompt"] == "You are free to do what you want. You can take 3 steps through links."
    assert stored["prompt_wording"] == "free"


def test_remaining_counts_down_under_each_page(tmp_path, monkeypatch):
    requests, _, _ = run_with(tmp_path, monkeypatch, ["experiment.reveal_budget=remaining"])
    assert [r[-1]["content"].rsplit("\n\n", 1)[-1] for r in requests] == [
        "You can take 3 more steps through links, including this one.",
        "You can take 2 more steps through links, including this one.",
        "You can take 1 more step through links, including this one."]
    assert all("You can take" not in r[0]["content"] for r in requests)


def test_unknown_value_is_refused_before_anything_runs(tmp_path):
    with pytest.raises(ValueError, match="reveal_budget"):
        run_experiment(config(tmp_path, extra=["experiment.reveal_budget=always"]), ROOT)
    assert not any(tmp_path.rglob("steps.jsonl"))
