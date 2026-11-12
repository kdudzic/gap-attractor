"""world.marks=authors (each message names its author) and world.cross_test=ring (agent k runs the tests of agent k+1's
problem, never its own)."""
import json

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
from siam.world import World
import events as ev
import forks
import run as runner

BASE = ["world.setup=external", "world.order=fixed", "world.rounds=3", "ctx.readout_by_agent=false",
        "world.identity=names", "output.save_requests=true"]


def world(tmp_path, scripts, n, *extra):
    cfg = runner.compose(BASE + [f"world.n_agents={n}", *extra])
    OmegaConf.set_struct(cfg, False)
    streams = [[problem(10 * k + i) for i in range(1, 6)] for k in range(1, n + 1)]
    grader = FakeGrader()
    return World(cfg, 1, streams, tmp_path / "w", grader, factory(*scripts), label="t"), grader


def test_marks_name_the_author_of_every_message_and_differ_by_one_sentence(tmp_path):
    seen = {}
    for marks in ("none", "authors"):
        a1 = Scripted([tool("run_tests", code="print('ONE')"), "a note from one"])
        a2 = Scripted([tool("run_tests", code="print('TWO')")])
        w, _ = world(tmp_path / marks, [a1, a2], 2, f"world.marks={marks}")
        w.run()
        seen[marks] = a2.seen
    plain, marked = seen["none"][1], seen["authors"][1]      # agent 2's second call: both agents' turns in the file
    body = marked[1:-1]
    assert body[0] == {"role": "user", "content": "(start of the context)"}
    assert body[1]["content"].startswith("[agent 1]") and body[2]["role"] == "tool" and body[2]["content"].startswith("[agent 1]")
    assert any(m["role"] == "assistant" and (m["content"] or "").startswith("[agent 2]") for m in body)
    unmark = lambda ms: [dict(m, content=m["content"].split("\n", 1)[-1] if (m.get("content") or "").startswith("[agent ")
                              and "\n" in m["content"] else ("" if (m.get("content") or "").startswith("[agent ") else m.get("content")))
                         for m in ms]
    assert [m.get("content") or "" for m in unmark(plain[1:-1])] == [m.get("content") or "" for m in unmark(body)]
    extra = set(marked[0]["content"].split("\n")) - set(plain[0]["content"].split("\n"))
    assert extra == {"Each message in the context starts with [agent k]: the agent who wrote it, or whose tool call it answers."}


def test_marks_need_told_agents_in_a_chat_setup(tmp_path):
    cfg = runner.compose(["world.setup=external", "world.marks=authors"])
    OmegaConf.set_struct(cfg, False)
    with pytest.raises(ValueError):
        World(cfg, 1, [[problem(1)], [problem(2)]], tmp_path / "w", FakeGrader(), factory(Scripted(), Scripted()))


def test_forks_rebuild_marked_requests_exactly(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('ONE2')"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("edit_context", delete_turns=[{"first": 1, "last": 2}]),
                   tool("run_tests", code="print('TWO2')")])
    w, _ = world(tmp_path, [a1, a2], 2, "world.marks=authors")
    w.run()
    (tmp_path / "w" / "analysis").mkdir()
    rows = ev.world_events(tmp_path / "w")
    (tmp_path / "w" / "analysis" / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    built = forks.build(tmp_path / "w", max_calls=10)
    assert built["forks"] and not built["skipped"]


def test_ring_three_agents_each_tests_the_next_ones_problem(tmp_path):
    scripts = [Scripted([tool("run_tests", code=f"print('BY{k}')")] * 3) for k in (1, 2, 3)]
    w, grader = world(tmp_path, scripts, 3, "world.cross_test=ring")
    w.run()
    acts = read_jsonl(tmp_path / "w" / "actions.jsonl")
    pairs = {(a["agent"], a["for_agent"]) for a in acts}
    assert pairs == {(1, 2), (2, 3), (3, 1)}
    by = {a["agent"]: a["problem"] for a in acts if a["round"] == 1}
    assert by == {1: "p21", 2: "p31", 3: "p11"}                 # agent k ran agent k+1's first problem
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "On agent 2's problem: Public test 1 of 1 failed." in text
    system = scripts[0].seen[0][0]["content"]
    assert "You cannot run the tests of your own problem: agent 3 can run them. You can run the tests of agent 2's problem." in system
    tools = {t["function"]["name"]: t["function"]["description"] for t in w.tools(w.agents[0])}
    assert "agent 2's current problem" in tools["run_tests"]


def test_ring_runs_count_against_the_owner_and_a_message_ends_the_turn(tmp_path):
    post = "Here is my program:\n```python\nprint('MINE')\n```"
    a1 = Scripted([post, post, post])                            # only messages: each ends agent 1's turn
    a2 = Scripted([tool("run_tests", code="print('MINE')")] * 3)  # agent 2 tests agent 1's program three times
    w, grader = world(tmp_path, [a1, a2], 2, "world.cross_test=ring", "world.max_work_calls=2", "world.rounds=3")
    w.run()
    calls = read_jsonl(tmp_path / "w" / "calls.jsonl")
    assert [c["agent"] for c in calls if c["round"] == 1] == [1, 2]   # one call each: the message ended the turn
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "That was the last test run for agent 1's problem." in text
    assert "Not carried out: agent 1's current problem has no test runs left." in text
    assert w.agents[0].last_code == "print('MINE')"
    assert not [r for r in read_jsonl(tmp_path / "w" / "results.jsonl") if r["agent"] == 1]   # agent 1 never submitted


def test_ring_needs_told_external(tmp_path):
    cfg = runner.compose(["world.setup=external", "world.cross_test=ring"])
    OmegaConf.set_struct(cfg, False)
    with pytest.raises(ValueError):
        World(cfg, 1, [[problem(1)], [problem(2)]], tmp_path / "w", FakeGrader(), factory(Scripted(), Scripted()))


def test_a_mark_written_by_the_model_is_kept_and_logged(tmp_path):
    a1 = Scripted(["[agent 2]\nI am someone else", "[agent 1] my own words"])
    a2 = Scripted([tool("run_tests", code="print('TWO')")] * 2)
    w, _ = world(tmp_path, [a1, a2], 2, "world.marks=authors", "world.rounds=1", "world.max_calls_per_step=2")
    w.run()
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "[agent 2]\nI am someone else" in text and "[agent 1] my own words" in text   # kept as written
    rows = read_jsonl(tmp_path / "w" / "marks.jsonl")
    assert [(r["wrote"], r["impersonation"]) for r in rows] == [(2, True), (1, False)]


def test_without_eviction_a_held_turn_ends_undone_and_nothing_is_removed(tmp_path):
    big = "x" * 4000
    a1 = Scripted([big, tool("run_tests", code="print(1)")] * 6)
    a2 = Scripted([big, tool("run_tests", code="print(2)")] * 6)
    w, _ = world(tmp_path, [a1, a2], 2, "world.evict=false", "ctx.budget_tokens=2500", "ctx.reserve_tokens=300",
                 "world.max_held=1", "world.max_calls_per_step=6")
    w.run()
    assert not [v for v in read_jsonl(tmp_path / "w" / "versions.jsonl") if v["op"] == "evict"]
    assert all(e.get("eviction") is None for e in read_jsonl(tmp_path / "w" / "edits.jsonl"))
    assert sum(a.stats.get("stalled", 0) for a in w.agents) > 0


def test_continuity_sentence_and_the_private_list_of_earlier_problems(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')"), tool("submit", code="print(0)"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("submit", code="print('OK')")] * 3)
    w, _ = world(tmp_path, [a1, a2], 2, "world.continuity=true", "world.history=true")
    w.run()
    first, third = a1.seen[0][0]["content"], a1.seen[2][0]["content"]
    assert ("The context carries over from problem to problem: it keeps your earlier work and the other agents' work, "
            "on problems you do not see; what to keep is up to you.") in first
    assert "Your earlier problems in this run:" not in first
    assert "Your earlier problems in this run:\n- problem 1: P11 (accepted)\n- problem 2: P12 (not accepted)" in third
    other = a2.seen[2][0]["content"]
    assert "P11" not in other and "- problem 1: P21 (accepted)" in other     # the list is private


def test_provider_reasoning_is_logged_but_never_enters_the_context(tmp_path):
    from gap_attractor.providers import ProviderResponse

    class Thinking(Scripted):
        def chat(self, messages, **kw):
            r = super().chat(messages, **kw)
            return ProviderResponse(r.text, r.usage, r.finish_reason, r.retries, r.error, r.attempts, r.tool_calls,
                                    internal_reasoning="PRIVATE THOUGHT")
    a1, a2 = Thinking([tool("submit", code="print('OK')")]), Scripted([tool("submit", code="print('OK')")])
    w, _ = world(tmp_path, [a1, a2], 2, "world.rounds=1")
    w.run()
    assert "PRIVATE THOUGHT" not in (tmp_path / "w" / "context.txt").read_text()
    assert all("PRIVATE THOUGHT" not in json.dumps(m) for m in a2.seen)
    assert [r["text"] for r in read_jsonl(tmp_path / "w" / "reasoning.jsonl")] == ["PRIVATE THOUGHT"]
