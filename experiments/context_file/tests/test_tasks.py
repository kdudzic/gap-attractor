"""Tasks and openings (21 Oct 2026): the essay (handed in with the stop tool), the opening search (variant
b_search_first) written into the file, and own files that never mark authors."""
import json

import pytest

import run as entry
from gap_attractor.harness import runner

from helpers import Scripted, call, config, links_in, read_steps, reply, text_of


def single(tmp_path, monkeypatch, provider, extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, results = entry.run(config(tmp_path, list(extra)))
    steps, folder = read_steps(dataset)
    return steps, folder, results


def test_an_essay_is_handed_in_with_the_stop_tool(tmp_path, monkeypatch):
    essay = "Archives keep what cities forget. " * 5
    provider = Scripted([reply(call("follow_link", 0, title="Archive")), lambda m: reply(call("follow_link", 1, title=links_in(m)[0])),
                         reply(call("stop", 2, essay=essay))])
    steps, folder, results = single(tmp_path, monkeypatch, provider, ["task=essay", "experiment.step_limit=3"])
    doc = json.loads((folder / "run.json").read_text())
    assert doc["outcome"]["termination_reason"] == "model_stopped" and doc["outcome"]["model_stop_statement"] == essay
    first = provider.requests[0]
    system = first["messages"][0]["content"]
    assert "write an essay on any topic you like" in system and "You will see at most 3 pages." in system
    assert "This is the last page you will see" not in text_of(first["messages"][1:])
    assert "This is the last page you will see: hand in your essay now" in text_of(provider.requests[2]["messages"][1:])
    assert doc["config"]["tools"]["definitions"]["stop"]["parameters"]["required"] == ["essay"]


def test_the_essay_stop_tool_has_no_reason_parameter():
    cfg = entry.compose(["task=essay"])
    spec = next(t for t in runner.tool_specs(cfg, ["stop"]))["function"]
    assert set(spec["parameters"]["properties"]) == {"essay"} and spec["description"].startswith("Hand in your essay")


def test_an_opening_search_is_written_into_the_file(tmp_path, monkeypatch):
    provider = Scripted([reply(call("search", 0, query="Archive")), reply(call("follow_link", 1, title="Archive"))])
    steps, folder, _ = single(tmp_path, monkeypatch, provider, ["variant=b_search_first", "variant.n_runs=1", "experiment.step_limit=2"])
    assert provider.requests[0]["tools"] == ["search", "stop"]                # the opening search offers no editing
    later = provider.requests[2]["messages"]                                  # step 1: the first request from the file
    text = text_of(later[1:])
    assert "[[TURN 1 role=assistant]]\n[searched: Archive]" in text and "[[TURN 2 role=user]]" in text
    assert "Archive" in later[1]["content"] or "Archive" in text                # the result list
    assert any(m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "follow_link" for m in later)
    assert not any(m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "search" for m in later)
    assert "# Archive" in text and [s["step_index"] for s in steps] == [0, 1, 2]   # the search is step 0; the limit counts links


def test_separate_files_never_mark_authors(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    entry.run(config(tmp_path, ["ctx=joint", "batch.start_pages=[Civic Atlas,Archive,Bridge]", "experiment.step_limit=2",
                                "ctx.joint.sharing=separate", "ctx.joint.agent_providers=[{model: m1},{model: m2}]"]))
    assert all("agent=" not in text_of(r["messages"][1:]) for p in providers.values() for r in p.requests)


@pytest.mark.parametrize("allow_stop", [True, False])
def test_nothing_the_context_says_mentions_stopping(tmp_path, monkeypatch, allow_stop):
    """The stop tool describes itself (and experiment.allow_stop=false removes it); neither the prompt, the
    instructions nor the budget notices mention stopping."""
    import re
    from ctxfile.budget import Budget
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    entry.run(config(tmp_path, ["ctx=joint", "ctx.joint.n_agents=2", f"experiment.allow_stop={str(allow_stop).lower()}",
                                "experiment.step_limit=2", "batch.start_pages=[Civic Atlas,Archive]",
                                "ctx.joint.agent_providers=[{model: m1},{model: m2}]"]))
    for p in providers.values():
        assert all(("stop" in r["tools"]) == allow_stop for r in p.requests)
        assert not re.search(r"\bstop", p.requests[0]["messages"][0]["content"], re.I)
        assert "with 1 other agent who browses Wikipedia" in p.requests[0]["messages"][0]["content"]
    budget = Budget(entry.compose([]).ctx, "bash")
    assert "stop" not in budget.over(30000)[1] + budget.held(30000, 3, 3)
    assert "Not stopped" in budget.held(30000, 1, 3, stop=True)     # only the answer to a stop the model tried


@pytest.mark.parametrize("overrides, wording", [([], "browse"), (["prompt=free"], "free"), (["task=essay"], "essay"),
                                                (["prompt=find"], "find")])
def test_the_prompt_states_the_task_only(overrides, wording):
    """No paragraph on how to move, what happens to unusable replies, or how to stop: the tools say how to act."""
    prompt = str(entry.compose(overrides).experiment.system_prompt)
    assert entry.compose(overrides).experiment.prompt_wording == wording and "\n" not in prompt
    assert not any(w in prompt for w in ("stop", "follow_link", "Candidate links", "random"))
