"""world.goal=joint (one team score; submit may name another agent's current problem) and world.reasoning_in_context
(the provider's reasoning written into the shared context before the reply)."""
import json

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
from siam.world import World
import run as runner

BASE = ["world.setup=external", "world.order=fixed", "world.rounds=2", "ctx.readout_by_agent=false",
        "world.identity=names", "output.save_requests=true", "world.n_agents=2", "ctx.budget_tokens=60000"]


def world(tmp_path, scripts, *extra):
    cfg = runner.compose(BASE + list(extra))
    OmegaConf.set_struct(cfg, False)
    streams = [[problem(10 * k + i) for i in range(1, 5)] for k in (1, 2)]
    return World(cfg, 1, streams, tmp_path / "w", FakeGrader(), factory(*scripts), label="t")


def test_joint_submit_closes_the_other_agents_problem_and_scores_the_team(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')", for_agent=2), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    w = world(tmp_path, [a1, a2], "world.goal=joint")
    w.run()
    rows = read_jsonl(tmp_path / "w" / "results.jsonl")
    by_other = [r for r in rows if r["how"] == "submit_by_other"]
    assert len(by_other) == 1 and by_other[0]["agent"] == 2 and by_other[0]["submitted_by"] == 1
    assert by_other[0]["problem"] == "p21"                       # agent 2's first problem, closed by agent 1
    system = a1.seen[0][0]["content"]
    assert "Your goal is shared with the other agents" in system
    tools = {t["function"]["name"]: t for t in json.loads((tmp_path / "w" / "requests" / "tools_a1.json").read_text())}
    assert "for_agent" in tools["submit"]["function"]["parameters"]["properties"]
    doc = json.loads((tmp_path / "w" / "world.json").read_text())
    assert doc["goal"] == "joint" and doc["team"]["submitted_for_others"] == 1
    assert doc["team"]["closed"] == len(rows) and doc["team"]["assigned"] == doc["team"]["closed"] + doc["team"]["unsubmitted"]


def test_joint_submit_for_a_missing_agent_is_refused(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')", for_agent=7), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    w = world(tmp_path, [a1, a2], "world.goal=joint")
    w.run()
    assert not any(r["how"] == "submit_by_other" for r in read_jsonl(tmp_path / "w" / "results.jsonl"))
    assert "there is no such agent" in (tmp_path / "w" / "context.txt").read_text()


def test_individual_goal_is_the_default_and_has_no_for_agent(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')")])
    a2 = Scripted([tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2])
    w.run()
    tools = {t["function"]["name"]: t for t in json.loads((tmp_path / "w" / "requests" / "tools_a1.json").read_text())}
    assert "for_agent" not in tools["submit"]["function"]["parameters"]["properties"]
    assert "Your goal is shared" not in a1.seen[0][0]["content"]


def test_joint_needs_external_names_and_no_ring(tmp_path):
    with pytest.raises(ValueError):
        world(tmp_path, [Scripted(), Scripted()], "world.goal=joint", "world.cross_test=ring")


def test_reasoning_in_context_is_a_turn_others_read(tmp_path):
    a1 = Scripted([dict(tool("run_tests", code="print('ONE')"), reasoning="MY PRIVATE THOUGHT"),
                   tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    w = world(tmp_path, [a1, a2], "world.reasoning_in_context=true")
    w.run()
    assert any("MY PRIVATE THOUGHT" in json.dumps(m) for m in a2.seen[0])         # the partner reads it
    assert "is added to the context before your reply" in a2.seen[0][0]["content"]


def test_reasoning_stays_out_by_default(tmp_path):
    a1 = Scripted([dict(tool("run_tests", code="print('ONE')"), reasoning="MY PRIVATE THOUGHT"),
                   tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    w = world(tmp_path, [a1, a2])
    w.run()
    assert not any("MY PRIVATE THOUGHT" in json.dumps(m) for m in a2.seen[0])


def test_the_owner_is_told_privately_when_another_agent_closed_its_problem(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')", for_agent=2), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    w = world(tmp_path, [a1, a2], "world.goal=joint")
    w.run()
    status = a2.seen[0][-1]["content"]
    assert "Your problem 1 was submitted by agent 1: accepted." in status
    assert not any("was submitted by agent" in req[-1]["content"] for req in a1.seen)
    assert len(read_jsonl(tmp_path / "w" / "joint_notices.jsonl")) == 1


def test_for_agent_true_is_not_agent_one(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("submit", code="print('OK')", for_agent=True), tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], "world.goal=joint")
    w.run()
    assert not any(r["how"] == "submit_by_other" for r in read_jsonl(tmp_path / "w" / "results.jsonl"))


def test_an_agents_own_reasoning_does_not_get_its_call_held(tmp_path):
    big = "x " * 6000                                  # ~12k chars of reasoning, past the limit of this small world
    a1 = Scripted([dict(tool("run_tests", code="print('ONE')"), reasoning=big), tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    w = world(tmp_path, [a1, a2], "world.reasoning_in_context=true", "ctx.budget_tokens=4096", "ctx.reserve_tokens=512")
    w.run()
    first = read_jsonl(tmp_path / "w" / "actions.jsonl")[0]
    assert first["agent"] == 1 and first["code"] == "print('ONE')"   # carried out, not held by its own thought
