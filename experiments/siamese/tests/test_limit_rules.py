"""world.end_over_limit (that many held calls or replies in a row, with no change that made the context smaller in
between, end an agent's session), held replies when there is no problem to work on, and world.after_stream=stay."""
import json

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
from siam.tasks.open import OpenTask
from siam.world import World
import run as runner

BASE = ["world.setup=external", "world.order=fixed", "ctx.readout_by_agent=false", "world.identity=names",
        "output.save_requests=true", "world.n_agents=2"]
TIGHT = ["ctx.budget_tokens=2500", "ctx.reserve_tokens=300", "world.evict=false"]


def open_world(tmp_path, scripts, *extra):
    cfg = runner.compose(BASE + ["world.rounds=3"] + TIGHT + list(extra))
    OmegaConf.set_struct(cfg, False)
    task = OpenTask("weak")
    return World(cfg, 1, task.streams(2), tmp_path / "w", task, factory(*scripts), label="t")


def lcb_world(tmp_path, scripts, n_problems, *extra):
    cfg = runner.compose(BASE + list(extra))
    OmegaConf.set_struct(cfg, False)
    streams = [[problem(10 * k + i) for i in range(1, n_problems + 1)] for k in (1, 2)]
    return World(cfg, 1, streams, tmp_path / "w", FakeGrader(), factory(*scripts), label="t")


def test_held_replies_in_a_row_end_the_session(tmp_path):
    a1 = Scripted(["x" * 12000, "second", "third"])
    a2 = Scripted(["hello", "again"])
    w = open_world(tmp_path, [a1, a2], "world.end_over_limit=2")
    w.run()
    assert [a.end_reason for a in w.agents] == ["over_limit", "over_limit"]
    ended = read_jsonl(tmp_path / "w" / "ended.jsonl")
    assert [(r["agent"], r["held_in_a_row"]) for r in ended] == [(2, 2), (1, 2)]
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "hello" in text and "again" in text                     # a held message is still added
    assert "Held in a row: 1 of 2." in text and "agent 2's session has ended" in text
    assert "After 2 such held replies in a row" in a2.seen[0][0]["content"]
    assert "a reply without a tool call does not end your turn until the context is at most" in a2.seen[1][-1]["content"]
    assert len(a2.seen) == 2 and len(a1.seen) == 3


def test_a_change_that_makes_the_context_smaller_resets_the_run(tmp_path):
    a1 = Scripted(["x" * 24000, "a", "b", "c", "d"])
    a2 = Scripted(["hello", tool("edit_context", delete_turns=[{"first": 3, "last": 3}]), "again", "more"])
    w = open_world(tmp_path, [a1, a2], "world.end_over_limit=2")
    w.run()
    assert w.agents[1].end_reason == "over_limit" and len(a2.seen) == 4    # without the reset it ends after 2 calls
    assert read_jsonl(tmp_path / "w" / "ended.jsonl")[0]["agent"] == 2


def test_without_the_end_rule_held_replies_stall_the_turn(tmp_path):
    a1 = Scripted(["x" * 12000] + ["more"] * 12)
    a2 = Scripted(["hello"] * 12)
    w = open_world(tmp_path, [a1, a2], "world.max_held=2")
    w.run()
    assert all(not a.end_reason or a.end_reason == "rounds" for a in w.agents)
    assert w.agents[1].stats["stalled"] == 3 and "The turn ends without it." in (tmp_path / "w" / "context.txt").read_text()


def test_held_work_calls_in_a_row_end_the_session(tmp_path):
    a1 = Scripted(["x" * 12000, tool("run_tests", code="print(1)"), tool("run_tests", code="print(1)")])
    a2 = Scripted([tool("submit", code="print('OK')")] * 3)
    w = lcb_world(tmp_path, [a1, a2], 3, "world.rounds=2", "world.end_over_limit=2", *TIGHT)
    w.run()
    assert w.agents[0].end_reason == "over_limit"
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "Held in a row: 1 of 2." in text and "That is 2 held calls in a row: your session has ended." in text
    assert "After 2 such held calls in a row" in a1.seen[0][0]["content"]


def test_after_its_stream_an_agent_stays_without_a_problem(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')"), "I am done", tool("run_tests", code="print(1)"), "still here"])
    a2 = Scripted([tool("submit", code="print(0)"), tool("submit", code="print('OK')"), tool("submit", code="print('OK')"),
                   tool("submit", code="print('OK')")])
    w = lcb_world(tmp_path, [a1, a2], 1, "world.rounds=3", "world.after_stream=stay", "ctx.budget_tokens=60000")
    w.run()
    one = w.agents[0]
    assert one.stayed and one.end_reason == "rounds" and len(a1.seen) == 4
    assert [t["function"]["name"] for t in w.tools(one)] == ["edit_context"]
    system = a1.seen[1][0]["content"]
    assert "You have finished your series of 1 problem. You stay in the session" in system
    assert "run_tests" not in system and "Your current problem" not in system
    assert "Your series is finished (1 problem); you have no current problem." in a1.seen[1][-1]["content"]
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "I am done" in text and "Not carried out: you have no current problem" in text
    assert len(read_jsonl(tmp_path / "w" / "results.jsonl")) == 2      # one problem each, nothing more


@pytest.mark.parametrize("extra", [["world.after_stream=stay"], ["world.rounds=2", "world.end_over_limit=2"]])
def test_settings_that_need_others_are_refused(tmp_path, extra):
    with pytest.raises(ValueError):
        lcb_world(tmp_path, [Scripted(), Scripted()], 1, *extra)


def test_a_turn_that_uses_up_its_calls_over_the_limit_counts_as_held(tmp_path):
    a1 = Scripted(["x" * 12000] + [tool("edit_context", delete_turns=[{"first": 99, "last": 99}])] * 40)
    a2 = Scripted([tool("edit_context", delete_turns=[{"first": 99, "last": 99}])] * 40)
    w = open_world(tmp_path, [a1, a2], "world.end_over_limit=2", "world.max_calls_per_step=6")
    w.run()
    assert w.agents[1].end_reason == "over_limit" and len(a2.seen) == 12          # two whole turns, then the end
    assert "with no more calls left in it. Held in a row: 1 of 2." in (tmp_path / "w" / "context.txt").read_text()


def test_a_stayed_agent_can_submit_for_another_in_a_joint_world(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')"), tool("submit", code="print('OK')", for_agent=2)])
    a2 = Scripted([tool("run_tests", code="print(0)")] + [tool("submit", code="print(0)")] * 4)
    w = lcb_world(tmp_path, [a1, a2], 2, "world.rounds=2", "world.after_stream=stay", "world.goal=joint",
                  "ctx.budget_tokens=60000")
    w.agents[0].stream = w.agents[0].stream[:1]                    # agent 1 has one problem, so it stays from round 2
    w.run()
    rows = read_jsonl(tmp_path / "w" / "results.jsonl")
    assert [(r["agent"], r.get("submitted_by")) for r in rows if r.get("submitted_by")] == [(2, 1)]
    assert [t["function"]["name"] for t in w.tools(w.agents[0])] == ["submit", "edit_context"]
    assert "- submit(code, for_agent): submit a complete program for another agent's current problem" in a1.seen[1][0]["content"]


def test_an_agent_ended_on_a_problem_leaves_it_unsubmitted(tmp_path):
    a1 = Scripted(["x" * 12000] + [tool("run_tests", code="print(1)")] * 3)
    a2 = Scripted([tool("run_tests", code="print(1)")] * 6)
    w = lcb_world(tmp_path, [a1, a2], 3, "world.rounds=2", "world.end_over_limit=2", "world.goal=joint", *TIGHT)
    w.run()
    assert [a.end_reason for a in w.agents] == ["over_limit", "over_limit"]
    team = json.loads((tmp_path / "w" / "world.json").read_text())["team"]
    assert team["unsubmitted"] == 2 and team["assigned"] == 2


def test_stay_is_refused_with_the_ring(tmp_path):
    with pytest.raises(ValueError):
        lcb_world(tmp_path, [Scripted(), Scripted()], 1, "world.rounds=2", "world.after_stream=stay", "world.cross_test=ring")


def test_gated_items_wait_for_their_first_item_to_close(tmp_path):
    a1 = Scripted([tool("run_tests", code="print(1)"), tool("run_tests", code="print(1)")] + [tool("submit", code="print('OK')")] * 4)
    a2 = Scripted([tool("submit", code="print('OK')")] * 6)
    cfg = runner.compose(BASE + ["world.rounds=4", "ctx.budget_tokens=60000"])
    OmegaConf.set_struct(cfg, False)
    s1 = [problem(11), problem(12)]
    s2 = [dict(problem(21), after="p11"), problem(22)]            # agent 2's first item waits for agent 1's first
    w = World(cfg, 1, [s1, s2], tmp_path / "w", FakeGrader(), factory(a1, a2), label="t")
    w.run()
    order = [(r["agent"], r["problem"], r["round"]) for r in read_jsonl(tmp_path / "w" / "results.jsonl")]
    assert (2, "p22", 1) in order                                  # deferred: agent 2 took its next item first
    assert [r[2] for r in order if r[1] == "p11"] == [3] and [r[2] for r in order if r[1] == "p21"] == [3]
    events = [e["event"] for e in read_jsonl(tmp_path / "w" / "gate.jsonl")]
    assert events[0] == "defer" and "wait" in events and "release" in events and "serve" in events


def test_gated_items_are_refused_in_free_for_all(tmp_path):
    cfg = runner.compose(BASE + ["world.schedule=free-for-all", "ctx.budget_tokens=60000"])
    OmegaConf.set_struct(cfg, False)
    with pytest.raises(ValueError):
        World(cfg, 1, [[problem(11)], [dict(problem(21), after="p11")]], tmp_path / "w", FakeGrader(),
              factory(Scripted(), Scripted()), label="t")


def test_counting_only_the_turns_leaves_the_instructions_out(tmp_path):
    a1, a2 = Scripted([tool("submit", code="print('OK')")] * 3), Scripted([tool("submit", code="print('OK')")] * 3)
    w = lcb_world(tmp_path, [a1, a2], 3, "world.rounds=1", "ctx.count=turns", "ctx.budget_tokens=2400",
                  "ctx.reserve_tokens=300")
    one = w.agents[0]
    assert w.tokens(one) == 0 and w.chars(one) == 0               # nothing in the file yet: the instructions don't count
    w.run()
    system = a1.seen[0][0]["content"]
    assert "Your working context (the messages after these instructions, except the status at the end) may hold at " \
           "most 2100 tokens; these instructions are not counted." in system
    assert "~0/2100" in a1.seen[0][-1]["content"]
    whole = lcb_world(tmp_path / "b", [Scripted(), Scripted()], 3, "world.rounds=1", "ctx.budget_tokens=2400",
                      "ctx.reserve_tokens=300")
    assert whole.tokens(whole.agents[0]) > 300                    # whole: the instructions count


def test_turns_counting_needs_a_chat_setup(tmp_path):
    with pytest.raises(ValueError):
        lcb_world(tmp_path, [Scripted(), Scripted()], 1, "world.setup=assigned", "ctx.count=turns")


def test_curation_advice_and_dashboard_are_off_by_default_and_opt_in(tmp_path):
    plain = lcb_world(tmp_path / "p", [Scripted(), Scripted()], 2, "world.rounds=1")
    plain.run()
    assert "Turns in the context" not in plain.agents[0].provider.seen[0][-1]["content"]
    a1 = Scripted([tool("run_tests", code="print(1)"),
                   tool("edit_context", replace_turns=[{"first": 1, "last": 1, "summary": "summary of my first call"}]),
                   tool("run_tests", code="print(2)"), tool("submit", code="print('OK')")])
    w = lcb_world(tmp_path / "d", [a1, Scripted()], 2, "world.rounds=3", "world.dashboard=true",
                  "world.curation_advice='Keep preferences, commitments, and facts you cannot fetch again.'")
    w.run()
    first, later = a1.seen[0], a1.seen[-1]
    assert "Keep preferences, commitments, and facts you cannot fetch again." in first[0]["content"]
    assert first[-1]["content"].rstrip().endswith("age (rounds)") or "Turns in the context:\nturn | agent" in first[-1]["content"]
    table = later[-1]["content"].split("Turns in the context:\n")[1].split("\n")
    rows = [r.split(" | ") for r in table[1:] if r and r[0].isdigit()]
    assert rows and rows[0][0] == "1" and rows[0][1] == "1" and rows[0][2] == "summary" and int(rows[0][3]) > 0
    assert int(rows[1][4]) >= 1 and int(rows[1][4]) >= int(rows[-1][4])      # the oldest turn aged at least as much
    assert 1 <= int(rows[0][4]) < int(rows[1][4])                             # a rewritten turn ages from its rewrite
    assert "'" not in json.loads((tmp_path / "d" / "w" / "world.json").read_text())["curation_advice"]
    doc = json.loads((tmp_path / "d" / "w" / "world.json").read_text())
    assert doc["dashboard"] is True and doc["curation_advice"].startswith("Keep preferences, commitments,")
