"""21 Oct 2026 additions: how the other agents look (attribution other, awareness), the whole-request count and its
calibration, the per-call log and the request audit, and goal worlds (find a page: assigned, external, pool)."""
import json
import re
from dataclasses import replace

import pytest

import audit_requests
import run as entry
from ctxfile.memory import request_chars
from gap_attractor.harness import runner

from helpers import Scripted, call, config, links_in, reply, text_of

STARTS = "batch.start_pages=[Civic Atlas,Archive,Bridge]"
TARGETS = "ctx.goal.targets=[Garden,Library,Canal]"          # one link away from each start page
THREE = "ctx.joint.agent_providers=[{model: m1},{model: m2},{model: m3}]"


def joint(tmp_path, monkeypatch, providers, extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    cfg = config(tmp_path, ["ctx=joint", STARTS, "experiment.step_limit=2", "ctx.joint.order=fixed",
                            "ctx.joint.agent_providers=[{model: m1},{model: m2}]", *extra])
    dataset, results = entry.run(cfg)
    runs = {json.loads(p.read_text())["world"]["agent_id"]: p.parent for p in dataset.root.rglob("run.json")}
    return dataset, results, runs


def world(dataset):
    return json.loads((dataset.root / "worlds" / "world_001" / "world.json").read_text())


def doc(folder):
    return json.loads((folder / "run.json").read_text())


def steps_of(folder):
    return [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines()]


class Billed(Scripted):
    """Scripted, but each reply bills the request at 3 characters per token (as a provider would)."""
    def chat(self, messages, *, seed, tools=None, **kw):
        out = super().chat(messages, seed=seed, tools=tools, **kw)
        return replace(out, usage={"prompt_tokens": round(request_chars(messages, tools) / 3), "completion_tokens": 2,
                                   "prompt_tokens_details": {"cached_tokens": 0}})


# --- who the others look like --------------------------------------------------------------------------------
def test_other_agents_can_all_look_alike(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    _, _, runs = joint(tmp_path, monkeypatch, providers, ["ctx.attribution=other"])
    last = providers["m1"].requests[-1]["messages"]                          # agent 3, round 2 (headers: edit_context)
    text = text_of(last[1:])
    assert "agent=other]]" in text and "agent=you]]" in text and not re.search(r"agent=\d", text)
    assert "marked as another agent's" in last[0]["content"] and "you are agent" not in last[0]["content"]
    # everyone's moves are assistant tool calls; a move appears with its page, at its maker's next turn: round 1's
    # three moves and nothing yet of round 2's (agents 1 and 2 have moved, their pages come at their next turns)
    assert sum(1 for m in last if m.get("tool_calls")) == 3
    assert doc(runs[1])["memory_settings"]["attribution"] == "other"


def test_without_headers_another_agents_reply_says_so_and_stays_an_assistant_message(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    joint(tmp_path, monkeypatch, providers, ["ctx.attribution=other", "ctx_edit=none", "ctx.budget_tokens=1000000", "ctx.headers=false"])
    last = providers["m1"].requests[-1]["messages"]
    others = [m for m in last if m.get("tool_calls") and (m.get("content") or "").startswith("[another agent]")]
    # agents 1 and 2, round 1 (their round-2 moves are written with their pages, at their next turns)
    assert len(others) == 2 and all(m["role"] == "assistant" for m in others)
    assert "[another agent]\n# Archive" in text_of(last[1:])                       # agent 2's page, its move's result
    assert any(m["role"] == "tool" and m["content"].startswith("[another agent]\n# ") for m in last)
    assert "[[TURN" not in text_of(last[1:])


def test_strangers_are_told_nothing_and_rumour_only_that_it_may_be_shared(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    joint(tmp_path / "s", monkeypatch, providers, ["ctx.attribution=none", "ctx.awareness=strangers"])
    system = providers["m1"].requests[0]["messages"][0]["content"]
    assert "A shared context" not in system and "other agents" not in system and "may be shared" not in system
    assert "are your own replies" in system
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    joint(tmp_path / "r", monkeypatch, providers, ["ctx.attribution=none", "ctx.awareness=rumour"])
    system = providers["m1"].requests[0]["messages"][0]["content"]
    assert "This context may be shared; others may also read and write it." in system and "A shared context" not in system
    with pytest.raises(ValueError, match="needs ctx.attribution=none"):
        joint(tmp_path / "x", monkeypatch, providers, ["ctx.awareness=strangers"])


# --- the size: the whole request, calibrated -----------------------------------------------------------------
def test_the_whole_request_is_counted_in_the_models_own_tokens(tmp_path, monkeypatch):
    provider = Billed()
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, _ = entry.run(config(tmp_path, ["experiment.step_limit=4", "ctx.save_requests=true"]))
    folder = next(p.parent for p in dataset.root.rglob("run.json"))
    calls = [json.loads(l) for l in (folder / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == len(provider.requests) == 4
    assert calls[0]["chars_per_token"] == pytest.approx(3.5, abs=0.01)           # 4 -> halfway to the billed 3
    assert calls[-1]["chars_per_token"] == pytest.approx(3.0, abs=0.1)
    billed = calls[-1]["usage"]["prompt_tokens"]
    assert 0.9 < billed / calls[-1]["estimate"] < 1.15                               # the count tracks the bill
    saved = json.loads((folder / "requests" / "0002.json").read_text())
    assert saved["messages"] == provider.requests[1]["messages"] and [t["function"]["name"] for t in saved["tools"]] == provider.requests[1]["tools"]
    assert calls[1]["turn_ids"] and all(isinstance(i, int) for i in calls[1]["turn_ids"])
    settings = doc(folder)["memory_settings"]
    assert settings["count"] == "whole" and settings["tokenizer"] == "calibrated"
    assert "Its size counts everything you are sent" in provider.requests[0]["messages"][0]["content"]


def test_the_old_count_is_still_there(tmp_path, monkeypatch):
    provider = Scripted()
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, _ = entry.run(config(tmp_path, ["ctx.count=file", "ctx.tokenizer=chars"]))
    folder = next(p.parent for p in dataset.root.rglob("run.json"))
    assert doc(folder)["memory_settings"]["count"] == "file" and doc(folder)["memory_settings"]["tokenizer"] == "chars/4"
    assert "Its size counts" not in provider.requests[0]["messages"][0]["content"]


# --- the audit -----------------------------------------------------------------------------------------------
def test_the_audit_passes_and_a_deleted_turn_is_never_sent_again(tmp_path, monkeypatch, capsys):
    def wipe(messages):          # agent 1, round 2: delete the first turns (its first page and its links list)
        return reply(call("edit_context", 1, delete_turns=[{"first": 1, "last": 3}]))
    providers = {"m1": Billed([reply(call("follow_link", 0, title="Archive")), wipe], model="m1"), "m2": Billed(model="m2")}
    dataset, _, runs = joint(tmp_path, monkeypatch, providers, ["ctx.save_requests=true", "experiment.step_limit=3",
                                                                "ctx.joint.agent_providers=[{model: m1},{model: m2},{model: m2}]"])
    versions = [json.loads(l) for l in (dataset.root / "worlds" / "world_001" / "versions.jsonl").read_text().splitlines()]
    cut = next(v for v in versions if v["op"] == "edit")
    gone = {r["id"] for r in cut["removed"]}
    assert len(gone) == 3
    later = [json.loads(l) for k in (2, 3) for l in (runs[k] / "calls.jsonl").read_text().splitlines()]
    after = [c for c in later if c["round"] and c["round"] >= 2 and c["step"] >= 2]
    assert after and all(not (gone & set(c["turn_ids"])) for c in after)
    with pytest.raises(SystemExit) as ended:
        audit_requests.main([str(dataset.root)])
    assert ended.value.code == 0
    out = capsys.readouterr().out
    assert out.count("problems 0") == 3
    report = json.loads((runs[1] / "audit.json").read_text())
    assert report["requests_saved"] == report["calls"] > 0 and 0.9 < report["billed_over_estimate"]["m1"] < 1.2


def test_the_audit_catches_a_request_built_from_a_removed_turn(tmp_path, monkeypatch):
    def wipe(messages):
        return reply(call("edit_context", 1, delete_turns=[{"first": 1, "last": 3}]))
    providers = {"m1": Scripted([reply(call("follow_link", 0, title="Archive")), wipe], model="m1"), "m2": Scripted(model="m2")}
    dataset, _, runs = joint(tmp_path, monkeypatch, providers, ["experiment.step_limit=3",
                                                                "ctx.joint.agent_providers=[{model: m1},{model: m2},{model: m2}]"])
    path = runs[2] / "calls.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    rows[-1]["turn_ids"] = [1, 2, 3] + rows[-1]["turn_ids"]                       # as if the old turns had been sent
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert audit_requests.audit(runs[2])["n_problems"] == 1


# --- goal worlds ---------------------------------------------------------------------------------------------
def goal_world(tmp_path, monkeypatch, providers, setup, extra=()):
    return joint(tmp_path, monkeypatch, providers, ["prompt=find", "ctx.goal.task=find", f"ctx.goal.setup={setup}",
                                                    TARGETS, THREE, "experiment.step_limit=3", *extra])


def go(title, n=0):
    return reply(call("follow_link", n, title=title))


def test_assigned_goals_are_marked_turns_and_arriving_ends_the_agent(tmp_path, monkeypatch):
    providers = {"m1": Scripted([go("Garden")], model="m1"), "m2": Scripted([go("Library")], model="m2"),
                 "m3": Scripted([go("Canal")], model="m3")}
    dataset, results, runs = goal_world(tmp_path, monkeypatch, providers, "assigned")
    first = text_of(providers["m1"].requests[0]["messages"][1:])
    assert '[[TURN 1 role=user agent=1]]\nYour goal is to reach the Wikipedia page "Garden".' in first
    assert '[[TURN 3 role=user agent=3]]\nYour goal is to reach the Wikipedia page "Canal".' in first
    assert "looking for a Wikipedia page" in providers["m1"].requests[0]["messages"][0]["content"]
    g = world(dataset)["goal"]
    assert [(a["agent"], a["target"], a["round"]) for a in g["arrivals"]] == [(1, "Garden", 1), (2, "Library", 1), (3, "Canal", 1)]
    assert {r["termination_reason"] for r in results} == {"reached_target"} and all(len(steps_of(runs[k])) == 1 for k in runs)
    assert doc(runs[2])["goal"]["reached"] and doc(runs[2])["outcome"]["outcome_category"] == "A-normal"
    assert 'The page "Garden" has been reached.' in (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()


def test_a_goal_still_pursued_cannot_be_edited_away(tmp_path, monkeypatch):
    erase = reply(call("edit_context", 1, delete_turns=[{"first": 1, "last": 1}]))   # agent 1's goal
    providers = {"m1": Scripted([go("Archive"), go("Library", 2)], model="m1"), "m2": Scripted([erase, go("Library")], model="m2"),
                 "m3": Scripted([go("Canal")], model="m3")}
    dataset, _, runs = goal_world(tmp_path, monkeypatch, providers, "assigned")
    edit = steps_of(runs[2])[0]["memory"]["edits"][0]
    assert edit["rejected"] == "goal" and not edit["applied"]
    assert "states a page still to be reached" in text_of(providers["m2"].requests[1]["messages"])


def test_external_goals_live_in_each_agents_own_instructions(tmp_path, monkeypatch):
    providers = {"m1": Scripted([go("Garden")], model="m1"), "m2": Scripted([go("Library")], model="m2"),
                 "m3": Scripted([go("Canal")], model="m3")}
    dataset, results, _ = goal_world(tmp_path, monkeypatch, providers, "external")
    for model, target in (("m1", "Garden"), ("m2", "Library"), ("m3", "Canal")):
        request = providers[model].requests[0]["messages"]
        assert f'reach the Wikipedia page "{target}"' in request[0]["content"]
        assert "Your goal" not in text_of(request[1:])
    assert "Your goal" not in (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()
    assert "has been reached" not in (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()
    assert {r["termination_reason"] for r in results} == {"reached_target"}


def test_a_pool_is_one_list_anyone_can_reach_and_it_shrinks(tmp_path, monkeypatch):
    # agent 1 reaches agent 3's page ("Canal" is one of the pool's targets too)
    providers = {"m1": Scripted([go("Canal"), go("Garden")], model="m1"), "m2": Scripted([go("Library")], model="m2"),
                 "m3": Scripted([go("Market")], model="m3")}
    dataset, results, runs = goal_world(tmp_path, monkeypatch, providers, "pool")
    first = text_of(providers["m1"].requests[0]["messages"][1:])
    assert 'Pages to reach: "Garden", "Library", "Canal". Reaching any one of them counts.' in first
    third = text_of(providers["m3"].requests[0]["messages"][1:])                  # after agents 1 and 2 arrived
    assert 'Pages to reach: "Garden". Reaching any one of them counts.' in third
    g = world(dataset)["goal"]
    assert [(a["agent"], a["target"]) for a in g["arrivals"]] == [(1, "Canal"), (2, "Library"), (1, "Garden")]
    assert {r["termination_reason"] for r in results} == {"world_stopped"}
    assert "Pages to reach: none left." in (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()


def test_the_first_arrival_can_end_the_world(tmp_path, monkeypatch):
    providers = {"m1": Scripted([go("Garden")], model="m1"), "m2": Scripted([go("Bridge")], model="m2"),
                 "m3": Scripted([go("Market")], model="m3")}
    dataset, results, runs = goal_world(tmp_path, monkeypatch, providers, "assigned", ["ctx.goal.stop=first"])
    assert world(dataset)["goal"]["first_arrival_round"] == 1
    reasons = {doc(runs[k])["world"]["agent_id"]: doc(runs[k])["outcome"]["termination_reason"] for k in runs}
    assert reasons == {1: "reached_target", 2: "world_stopped", 3: "world_stopped"}
    assert all(len(steps_of(runs[k])) == 1 for k in runs)                       # no step after the first round


def test_a_goal_needs_the_find_wording(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2"), "m3": Scripted(model="m3")}
    with pytest.raises(ValueError, match="needs prompt=find"):
        joint(tmp_path, monkeypatch, providers, ["ctx.goal.task=find", TARGETS, THREE])


def test_an_arrival_made_over_the_limit_is_still_an_arrival(tmp_path, monkeypatch):
    providers = {"m1": Scripted([go("Garden")], model="m1"), "m2": Scripted([go("Library")], model="m2"),
                 "m3": Scripted([go("Canal")], model="m3")}
    _, results, runs = goal_world(tmp_path, monkeypatch, providers, "assigned",
                                  ["ctx.budget_tokens=1000", "ctx.reserve_tokens=0", "ctx.max_held_moves=0"])
    assert {r["termination_reason"] for r in results} == {"reached_target"}
    assert doc(runs[1])["context_budget_end"]["tokens_kept"] > 1000               # the overrun is still on record


def test_a_common_target_is_one_goal_turn_and_the_first_arrival_ends_the_world(tmp_path, monkeypatch):
    providers = {"m1": Scripted([go("Garden")], model="m1"), "m2": Scripted([go("Library")], model="m2"),
                 "m3": Scripted([go("Canal")], model="m3")}
    dataset, results, runs = joint(tmp_path, monkeypatch, providers, [
        "prompt=find", "ctx.goal.task=find", "ctx.goal.setup=common", "ctx.goal.stop=first",
        "ctx.goal.targets=[Garden,Garden,Garden]", THREE, "experiment.step_limit=3"])
    live = (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()
    assert live.count('Your goal is to reach the Wikipedia page "Garden".') == 1                # one turn, nobody's
    assert '[[TURN 1 role=user]]\nYour goal is to reach the Wikipedia page "Garden".' in live
    reasons = {doc(runs[k])["world"]["agent_id"]: doc(runs[k])["outcome"]["termination_reason"] for k in runs}
    assert reasons == {1: "reached_target", 2: "world_stopped", 3: "world_stopped"}
    with pytest.raises(ValueError, match="one target for all its agents"):
        joint(tmp_path / "x", monkeypatch, providers, ["prompt=find", "ctx.goal.task=find", "ctx.goal.setup=common",
                                                       "ctx.goal.stop=first", TARGETS, THREE])
    with pytest.raises(ValueError, match="common needs ctx.goal.stop=first"):
        joint(tmp_path / "y", monkeypatch, providers, ["prompt=find", "ctx.goal.task=find", "ctx.goal.setup=common",
                                                       "ctx.goal.targets=[Garden,Garden,Garden]", THREE])


def test_all_found_agents_go_on_after_arriving_and_only_owners_count(tmp_path, monkeypatch):
    avoid = lambda messages: go(next(t for t in links_in(messages) if t not in ("Garden", "Library", "Canal")))
    providers = {"m1": Scripted([go("Garden")], model="m1"), "m2": Scripted([go("Library")], model="m2"),
                 "m3": Scripted([avoid] * 5, model="m3")}
    dataset, results, runs = goal_world(tmp_path, monkeypatch, providers, "assigned", ["ctx.goal.stop=all_found"])
    g = world(dataset)["goal"]
    assert [(a["agent"], a["target"]) for a in g["arrivals"]] == [(1, "Garden"), (2, "Library")]   # once each
    assert all(len(steps_of(runs[k])) == 3 for k in runs)                       # arrived agents went on to the limit
    assert {r["termination_reason"] for r in results} == {"step_limit"}
    assert doc(runs[1])["goal"]["reached"] and not doc(runs[3])["goal"]["reached"]


def test_all_found_ends_the_world_once_every_owner_has_arrived(tmp_path, monkeypatch):
    providers = {"m1": Scripted([go("Garden")], model="m1"), "m2": Scripted([go("Library")], model="m2"),
                 "m3": Scripted([go("Canal")], model="m3")}
    dataset, results, runs = goal_world(tmp_path, monkeypatch, providers, "assigned", ["ctx.goal.stop=all_found"])
    assert {r["termination_reason"] for r in results} == {"world_stopped"} and all(len(steps_of(runs[k])) == 1 for k in runs)
    assert all(doc(runs[k])["goal"]["reached"] for k in runs)
    with pytest.raises(ValueError, match="all_found needs"):
        goal_world(tmp_path / "x", monkeypatch, providers, "pool", ["ctx.goal.stop=all_found"])
