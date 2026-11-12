import json

from analyze import collaboration, curve, paired, working_sets, world_rows


def call(agent, problem, ctx, out, rnd=1, tools=("run_tests",)):
    return {"agent": agent, "problem": problem, "context_tokens": ctx, "usage": {"completion_tokens": out},
            "round": rnd, "tools": list(tools), "finish_reason": "tool_calls"}


def test_working_set_is_what_each_problem_adds():
    calls = [call(1, "a", 1000, 100), call(1, "a", 1500, 200),          # problem a: ends at 1500 + 200 = 1700
             call(1, "b", 2600, 50), call(1, "b", 4000, 300)]           # problem b: ends at 4300, began after 1700
    probs = {"a": {"question_content": "x" * 1600, "starter_code": ""}}  # statement ~ (1600 + template) / 4 tokens
    ws = working_sets(calls, 1, probs)
    assert ws["b"] == 4300 - 1700
    assert 1700 - 1000 < ws["a"] < 1700                                # first problem: first request minus its statement


def act(agent, problem, code, tool="submit"):
    return {"agent": agent, "problem": problem, "code": code, "tool": tool}


CODE = "\n".join(f"x{i} = compute_{i}(alpha, beta, gamma)" for i in range(10))


def test_reuse_is_split_into_help_and_mixup():
    probs = {"p1": {"fn_name": "solveOne"}, "p2": {"fn_name": "solveTwo"}}
    helped = "class Solution:\n    def solveOne(self):\n" + CODE        # agent 2 wrote code for agent 1's problem
    actions = [act(2, "p2", helped, "run_tests"), act(1, "p1", helped),  # agent 1 submits it for p1: help
               act(2, "p2", CODE + "\nprint(1)"), act(1, "p3", CODE + "\nprint(1)")]   # p2's code reused for p3: mix-up
    results = [{"agent": 1, "problem": "p1", "passed": True}, {"agent": 1, "problem": "p3", "passed": False}]
    c1 = collaboration(actions, results, probs, 1)
    assert c1["reuse_help"] == [1] and c1["reuse_mixup"] == [0]
    c2 = collaboration(actions, results, probs, 2)
    assert c2["for_partner_programs"] == 1                              # it defined solveOne while on p2


def world(tmp_path, name, sharing, versions, texts, agents=(1, 2)):
    w = tmp_path / name
    w.mkdir()
    doc = {"seed": 1, "status": "complete", "sharing": sharing, "relation": "different", "capped": True, "limit": 7168,
           "agents": [{"agent": k, "model": "m", "edits": 0, "edits_applied": 0, "held": 0, "evictions": 0, "cost": 0,
                       "calls": 0, "no_tool": 0, "api_errors": 0} for k in agents], "config": {}}
    (w / "world.json").write_text(json.dumps(doc))
    (w / "results.jsonl").write_text("\n".join(json.dumps({"agent": k, "problem": f"p{k}", "passed": True, "work_calls": 1})
                                               for k in agents))
    (w / "texts.jsonl").write_text("\n".join(json.dumps(t) for t in texts))
    for fname, rows in versions.items():
        (w / fname).write_text("\n".join(json.dumps(v) for v in rows))
    return w


def test_reasoning_turn_fate_uses_each_agents_own_file(tmp_path):
    # separate files: agent 2's file also has a turn with id 5, removed by agent 2 itself; agent 1's turn 5 survives
    w = world(tmp_path, "sep", "separate",
              {"versions_agent_1.jsonl": [], "versions_agent_2.jsonl": [
                  {"n": 1, "agent": 2, "op": "edit", "step": 3, "removed": [{"id": 5, "agent": 2, "kind": "text"}]}]},
              [{"agent": 1, "turn_id": 5}, {"agent": 2, "turn_id": 5}])
    rows = {r["agent"]: r for r in world_rows(w)}
    assert rows[1]["text_removed"] == {} and rows[2]["text_removed"] == {"self": 1}
    # a shared file: agent 2 removing agent 1's reasoning counts as "other"
    j = world(tmp_path, "joint", "full",
              {"versions.jsonl": [{"n": 1, "agent": 2, "op": "edit", "step": 3, "removed": [{"id": 7, "agent": 1, "kind": "text"}]}]},
              [{"agent": 1, "turn_id": 7}])
    rows = {r["agent"]: r for r in world_rows(j)}
    assert rows[1]["text_removed"] == {"other": 1}


def row(model, sharing, limit, seed, agent, by_problem, evicted=0):
    return {"model": model, "sharing": sharing, "limit": limit, "seed": seed, "agent": agent, "by_problem": by_problem,
            "evicted_problems": evicted}


def test_paired_and_curve():
    uncapped = [row("m", "single", None, s, 1, {"a": 1, "b": 1, "c": 0}) for s in (1, 2)]
    small = [row("m", "single", 7168, s, 1, {"a": 0, "b": 0, "c": 0}) for s in (1, 2)]
    big = [row("m", "single", 28672, s, 1, {"a": 1, "b": 1, "c": 0}) for s in (1, 2)]
    cells = {"A/unc": uncapped, "B/8k": small, "B/32k": big}
    pr = paired(cells)
    assert pr["m | A/unc - B/8k"]["diff"] == round(4 / 6, 3) and pr["m | A/unc - B/8k"]["problems"] == 6
    cur = curve(cells)["m"]
    assert cur["B_star_limit"] == 28672                 # 8k loses 2 of 3 in every seed: its interval excludes 0
    assert cur["by_limit"][0]["ci"] == [-0.667, -0.667]
    evicting = {"A/unc": uncapped, "B/32k": [row("m", "single", 28672, s, 1, {"a": 1, "b": 1, "c": 0}, evicted=1) for s in (1, 2)]}
    assert curve(evicting)["m"]["B_star_limit"] is None   # as good as uncapped, but it evicts in 1/3 of problems
