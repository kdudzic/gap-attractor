import json

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool, turn_numbers
from siam.world import World, code_of
import run as runner


def config(*overrides):
    cfg = runner.compose(list(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def world(tmp_path, scripts, streams, *overrides):
    cfg = config(*overrides)
    return World(cfg, 1, streams, tmp_path / "w", FakeGrader(), factory(*scripts), label="t")


def test_single_agent_runs_to_the_end(tmp_path):
    w = world(tmp_path, [Scripted([tool("run_tests", code="print(0)"), tool("submit", code="print('OK')")])],
              [[problem(1), problem(2)]], "world.n_agents=1")
    out = w.run()
    assert out["status"] == "complete"
    res = read_jsonl(tmp_path / "w" / "results.jsonl")
    assert [r["passed"] for r in res] == [True, True]
    text = (tmp_path / "w" / "context_agent_1.txt").read_text()
    assert "Public test 1 of 1 failed" in text and "Accepted" in text and "[[TURN 1 role=user]]" in text


def test_full_sharing_lets_an_agent_delete_the_partners_turns(tmp_path):
    def delete_partner_code(messages):
        nums = turn_numbers(messages, r"(?s)agent=1\]\]\n.*\[run_tests\]")
        return tool("edit_context", delete_turns=[{"first": nums[0], "last": nums[0]}])
    a1 = Scripted([tool("run_tests", code="print(0)"), tool("submit", code="print('OK')")])
    a2 = Scripted([delete_partner_code, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.order=fixed", "world.sharing=full")
    w.run()
    edits = read_jsonl(tmp_path / "w" / "edits.jsonl")
    assert edits[0]["agent"] == 2 and edits[0]["applied"] and edits[0]["removed_others"] == 1
    assert "print(0)" not in (tmp_path / "w" / "context.txt").read_text()


def test_readonly_others_refuses_the_same_edit(tmp_path):
    def delete_partner_code(messages):
        nums = turn_numbers(messages, r"(?s)agent=1\]\]\n.*\[run_tests\]")
        return tool("edit_context", delete_turns=[{"first": nums[0], "last": nums[0]}])
    a1 = Scripted([tool("run_tests", code="print(0)"), tool("submit", code="print('OK')")])
    a2 = Scripted([delete_partner_code, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.order=fixed", "world.sharing=readonly_others")
    w.run()
    edits = read_jsonl(tmp_path / "w" / "edits.jsonl")
    assert edits[0]["rejected"] == "others_turns" and not edits[0]["applied"]
    assert "print(0)" in (tmp_path / "w" / "context.txt").read_text()


def test_a_current_problem_cannot_be_changed_by_anyone(tmp_path):
    def delete_partner_problem(messages):
        nums = turn_numbers(messages, r"agent=1\]\]\nProblem 1")
        return tool("edit_context", delete_turns=[{"first": nums[0], "last": nums[0]}])
    def delete_own_problem(messages):
        nums = turn_numbers(messages, r"agent=2\]\]\nProblem 1")
        return tool("edit_context", delete_turns=[{"first": nums[0], "last": nums[0]}])
    a1 = Scripted([tool("run_tests", code="print(0)"), tool("submit", code="print('OK')")])
    a2 = Scripted([delete_partner_problem, delete_own_problem, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.order=fixed")
    w.run()
    rejected = [e["rejected"] for e in read_jsonl(tmp_path / "w" / "edits.jsonl")]
    assert rejected == ["protected", "protected"]


def test_a_finished_problem_can_be_removed(tmp_path):
    def delete_partner_old_problem(messages):
        nums = turn_numbers(messages, r"agent=1\]\]\nProblem 1")
        return tool("edit_context", delete_turns=[{"first": nums[0], "last": nums[0]}])
    a1 = Scripted([tool("submit", code="print('OK')"), tool("run_tests", code="x"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="y"), delete_partner_old_problem, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1), problem(3)], [problem(2)]], "world.order=fixed")
    w.run()
    edits = read_jsonl(tmp_path / "w" / "edits.jsonl")
    assert edits[0]["applied"] and edits[0]["removed_kinds"] == ["problem"]


def test_separate_files_never_leak(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('SECRET_ONE')"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print('SECRET_TWO')"), tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.sharing=separate")
    w.run()
    f1 = (tmp_path / "w" / "context_agent_1.txt").read_text()
    f2 = (tmp_path / "w" / "context_agent_2.txt").read_text()
    assert "SECRET_ONE" in f1 and "SECRET_TWO" not in f1 and "SECRET_TWO" in f2 and "SECRET_ONE" not in f2
    assert not any("SECRET_TWO" in json.dumps(m) for m in a1.seen) and not any("SECRET_ONE" in json.dumps(m) for m in a2.seen)
    assert "agent" not in a1.seen[0][0]["content"].split("Tools")[0].lower().replace("agent=", "")


def test_work_limit_submits_the_last_program(tmp_path):
    a1 = Scripted([tool("run_tests", code="v1"), tool("run_tests", code="v2 OK")])
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1", "world.max_work_calls=2")
    w.run()
    res = read_jsonl(tmp_path / "w" / "results.jsonl")
    assert res == [res[0]] and res[0]["how"] == "work_limit" and res[0]["passed"] and res[0]["work_calls"] == 2


def test_over_the_limit_work_is_held_then_evicted(tmp_path):
    big = "x" * 4000
    a1 = Scripted([tool("run_tests", code=big)] + [tool("run_tests", code="print('OK')")] * 4)
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1", "ctx.budget_tokens=1400", "ctx.reserve_tokens=200", "ctx.count=file",
              "world.max_held=2", "world.max_work_calls=3")
    w.run()
    calls = [c["tools"] for c in read_jsonl(tmp_path / "w" / "calls.jsonl")]
    assert len(calls) >= 4
    stats = json.loads((tmp_path / "w" / "world.json").read_text())["agents"][0]
    assert stats["held"] == 3 and stats["evictions"] == 1
    assert any(e.get("eviction") for e in read_jsonl(tmp_path / "w" / "edits.jsonl"))
    snaps = "".join(p.read_text() for p in sorted((tmp_path / "w" / "snapshots").glob("*.txt")))
    assert "Not carried out" in snaps


def test_per_agent_models_are_cycled(tmp_path):
    w = world(tmp_path, [Scripted(), Scripted(), Scripted()], [[problem(1)], [problem(2)], [problem(3)]],
              "world.n_agents=3", "world.agent_providers=[{model: m/a}, {model: m/b}]")
    w.run()
    assert [a["model"] for a in json.loads((tmp_path / "w" / "world.json").read_text())["agents"]] == ["m/a", "m/b", "m/a"]


def test_the_system_message_says_any_turn_can_be_changed(tmp_path):
    a1, a2 = Scripted(), Scripted()
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]])
    w.run()
    system = a1.seen[0][0]["content"]
    assert "any turn, whoever wrote it" in system and "agent 2" in system
    tools = {t["function"]["name"]: t for t in w.tools(w.agents[0])}
    assert "whoever wrote it" in tools["edit_context"]["function"]["description"]


def test_text_only_replies_are_kept_and_reminded(tmp_path):
    a1 = Scripted(["Let me think about the algorithm first.", tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1")
    w.run()
    assert "Let me think about the algorithm first." in (tmp_path / "w" / "context_agent_1.txt").read_text()
    assert "Call one tool" in a1.seen[1][-1]["content"]


@pytest.mark.parametrize("given,expected", [("print(1)", "print(1)"), ("```python\nprint(1)\n```", "print(1)"),
                                            ("text\n```python\nprint(2)\n```\n", "print(2)")])
def test_code_is_unwrapped(given, expected):
    assert code_of({"code": given}) == expected


def test_run_entry_point_with_fakes(tmp_path):
    cfg = config(f"output.directory={tmp_path}", "problems.file=" + str(tmp_path / "p.jsonl"), "problems.per_agent=1",
                 "world.seeds=[1,2]", "problems.difficulty=[]")
    rows = []
    for i in range(1, 5):
        p = problem(i)
        rows.append(json.dumps({"question_id": p["id"], "question_title": p["title"], "question_content": p["question_content"],
                                "platform": "atcoder", "difficulty": "medium", "contest_date": p["contest_date"],
                                "starter_code": "", "public_test_cases": json.dumps(p["public"]),
                                "private_test_cases": json.dumps([]), "metadata": "{}"}))
    (tmp_path / "p.jsonl").write_text("\n".join(rows))
    root, results = runner.run(cfg, make_provider=lambda pc: Scripted(model=str(pc.model)), grader_factory=FakeGrader)
    assert [r["status"] for r in results] == ["complete", "complete"]
    root2, again = runner.run(cfg, make_provider=lambda pc: Scripted(model=str(pc.model)), grader_factory=FakeGrader)
    assert all("skipped" in r["status"] for r in again)


def test_cut_off_replies_are_marked_and_logged(tmp_path):
    a1 = Scripted([{"text": "a very long chain of reasoning", "finish": "length"}, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1")
    w.run()
    text = (tmp_path / "w" / "context_agent_1.txt").read_text()
    assert "a very long chain of reasoning" in text and "[this reply was cut off at the limit of 8000 tokens]" in text
    rows = read_jsonl(tmp_path / "w" / "texts.jsonl")
    assert rows[0]["truncated"] and rows[0]["turn_id"]
    assert json.loads((tmp_path / "w" / "world.json").read_text())["agents"][0]["truncated"] == 1
    assert "at most 8000 tokens long" in a1.seen[0][0]["content"]


def test_an_empty_length_reply_with_no_tokens_writes_no_cut_off_notice(tmp_path):
    """A provider artefact (Gemini in ring worlds, grid2/grid3): finish_reason length, no text, 0 tokens. Nothing was cut."""
    a1 = Scripted([{"text": "", "finish": "length", "tokens": 0}, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1")
    w.run()
    assert "cut off" not in (tmp_path / "w" / "context_agent_1.txt").read_text()
    agent = json.loads((tmp_path / "w" / "world.json").read_text())["agents"][0]
    assert agent["truncated"] == 0 and agent["empty_length"] == 1


def test_status_shows_each_agents_share_of_a_shared_file(tmp_path):
    a1 = Scripted([tool("run_tests", code="print(0)"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.order=fixed")
    w.run()
    status = a2.seen[0][-1]["content"]
    assert "In the file: agent 1 ~" in status and "agent 2 ~" in status
    solo = Scripted()
    w1 = World(config("world.n_agents=1"), 1, [[problem(1)]], tmp_path / "w1", FakeGrader(), factory(solo))
    w1.run()
    assert "In the file" not in solo.seen[0][-1]["content"]


def test_an_agent_override_may_add_provider_keys(tmp_path):
    w = world(tmp_path, [Scripted(), Scripted()], [[problem(1)], [problem(2)]],
              "world.agent_providers=[{model: m/a, extra_body: {reasoning: {enabled: false}}}, "
              "{model: m/b, extra_body: {reasoning: {effort: low}}}]")
    assert w.agents[0].pcfg.extra_body.reasoning.enabled is False
    assert w.agents[1].pcfg.extra_body.reasoning.effort == "low" and "enabled" not in w.agents[1].pcfg.extra_body.reasoning


# --- other designs (opt-in) ------------------------------------------------------------------------------------

from siam.world import remap_edit


def test_round_robin_keeps_one_order_from_a_seeded_first_mover(tmp_path):
    a1, a2 = Scripted([tool("run_tests", code="x")] * 3), Scripted([tool("run_tests", code="y")] * 3)
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.order=round_robin", "world.max_work_calls=4")
    w.run()
    calls = [c["agent"] for c in read_jsonl(tmp_path / "w" / "calls.jsonl")]
    first = w.first_mover
    assert first in (1, 2) and calls[0] == first
    assert all(calls[i] != calls[i + 1] for i in range(len(calls) - 1) if i < 6)      # alternating while both are busy
    assert json.loads((tmp_path / "w" / "world.json").read_text())["first_mover"] == first


def test_unaware_agents_are_not_told_about_each_other(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('MINE_ONE')"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print('MINE_TWO')"), tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.awareness=unaware", "world.order=fixed")
    w.run()
    sent = json.dumps(a2.seen[0])
    system = a2.seen[0][0]["content"]
    assert "agent" not in system.lower().replace("agent=", "") and "share" not in system
    assert "agent=" not in sent and "for agent" not in sent and "In the file" not in sent
    assert "MINE_ONE" in sent                                   # the other agent's work is there, unexplained
    assert "agent=" not in (tmp_path / "w" / "context.txt").read_text()


def test_unaware_others_as_own_shows_the_partners_replies_as_ones_own(tmp_path):
    a1 = Scripted(["the partner is thinking out loud"])
    a2 = Scripted()
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.awareness=unaware", "world.others_as=own",
              "world.order=fixed")
    w.run()
    partner = [m for m in a2.seen[0] if "the partner is thinking out loud" in (m.get("content") or "")]
    assert partner and partner[0]["role"] == "assistant"


def test_remap_edit_follows_turns_that_moved_and_refuses_ones_that_went():
    seen = [10, 11, 12, 13]
    assert remap_edit({"delete_turns": [{"first": 2, "last": 3}]}, seen, seen + [14]) == \
        ({"delete_turns": [{"first": 2, "last": 3}]}, None)                          # only appended since: unchanged
    moved, why = remap_edit({"delete_turns": [{"first": 3, "last": 4}], "replace_text": [{"turn": 4, "old": "a", "new": "b"}]},
                            seen, [11, 12, 13, 15])                                  # turn 10 went, others shifted up
    assert why is None and moved["delete_turns"] == [{"first": 2, "last": 3}] and moved["replace_text"][0]["turn"] == 3
    assert remap_edit({"delete_turns": [{"first": 1, "last": 2}]}, seen, [11, 12, 13])[1]          # 10 is gone
    assert remap_edit({"delete_turns": [{"first": 2, "last": 3}]}, seen, [10, 11, 99, 12, 13])[1]  # no longer one block


def test_async_world_runs_both_agents_to_the_end(tmp_path):
    import time as _t

    class Slow(Scripted):
        def chat(self, messages, **kw):
            _t.sleep(0.02)
            return super().chat(messages, **kw)
    a1 = Slow([tool("run_tests", code="x"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="y"), tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1), problem(3)], [problem(2), problem(4)]], "world.schedule=async")
    out = w.run()
    assert out["status"] == "complete"
    res = read_jsonl(tmp_path / "w" / "results.jsonl")
    assert sorted((r["agent"], r["problem"]) for r in res) == [(1, "p1"), (1, "p3"), (2, "p2"), (2, "p4")]
    doc = json.loads((tmp_path / "w" / "world.json").read_text())
    assert doc["schedule"] == "free-for-all" and "any of you may act at any time" in a1.seen[0][0]["content"]


def test_default_design_is_unchanged(tmp_path):
    a1, a2 = Scripted(), Scripted()
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]])
    w.run()
    doc = json.loads((tmp_path / "w" / "world.json").read_text())
    assert (doc["order"], doc["schedule"], doc["awareness"], doc["others_as"], doc["setup"], doc["edit"]) == \
        ("shuffled", "rounds", "introduced", "user", "assigned", "tool")
    assert "you take turns." in a1.seen[0][0]["content"] and "agent=" in json.dumps(a1.seen[0])


def test_unaware_worlds_keep_a_labelled_copy_for_the_logs(tmp_path):
    w = world(tmp_path, [Scripted(["someone else's code is in here"]), Scripted()], [[problem(1)], [problem(2)]],
              "world.awareness=unaware", "world.order=fixed")
    w.run()
    snaps = sorted((tmp_path / "w" / "snapshots").glob("*_labelled_r*.txt"))
    assert snaps and "agent=1" in snaps[0].read_text()
    from inspect_world import story
    text = story(tmp_path / "w")
    assert "FIRST a1 notices someone else" in text


def test_hint_adds_one_sentence_and_nothing_else(tmp_path):
    a1, a2 = Scripted(), Scripted()
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.awareness=hint")
    w.run()
    system = a1.seen[0][0]["content"]
    assert "This context may be shared: others may also read and write it." in system
    assert "agent=" not in json.dumps(a1.seen[0]) and "You are agent" not in system and "In the file" not in json.dumps(a1.seen[0])


def test_a_malformed_tool_call_is_said_once_in_the_context(tmp_path):
    """Gemini's MALFORMED_FUNCTION_CALL: finish_reason error, no text, no tool call, no cost. Before 28 Oct nothing was
    written, the next request was identical, and the agent looped."""
    a1 = Scripted([{"text": "", "finish": "error", "tokens": 0}, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1")
    w.run()
    assert "[your last reply could not be read as a tool call; nothing was carried out]" in \
        (tmp_path / "w" / "context_agent_1.txt").read_text()
    assert a1.seen[1] != a1.seen[0]
    assert json.loads((tmp_path / "w" / "world.json").read_text())["agents"][0]["malformed"] == 1
