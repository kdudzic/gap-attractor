"""Turn-taking v1: a fixed number of rounds, strict alternation, the question after the end, and told vs not told
differing by one sentence only."""
import difflib
import json
import re

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
from siam.world import World
import run as runner


def config(*overrides):
    cfg = runner.compose(list(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


V1 = ["world.setup=external", "world.order=fixed", "world.rounds=3", "ctx.readout_by_agent=false"]


def test_every_agent_takes_exactly_the_given_rounds_in_strict_alternation(tmp_path):
    a1, a2 = Scripted(), Scripted()
    streams = [[problem(i) for i in range(1, 6)], [problem(i) for i in range(6, 11)]]
    w = World(config(*V1), 1, streams, tmp_path / "w", FakeGrader(), factory(a1, a2), label="t")
    out = w.run()
    assert out["status"] == "complete"
    calls = read_jsonl(tmp_path / "w" / "calls.jsonl")
    assert [(c["round"], c["agent"]) for c in calls] == [(r, k) for r in (1, 2, 3) for k in (1, 2)]
    doc = json.loads((tmp_path / "w" / "world.json").read_text())
    assert doc["rounds"] == 3 and doc["fixed_rounds"] == 3
    assert all(a["end_reason"] == "rounds" for a in doc["agents"])
    assert all(a["solved"] == 3 for a in doc["agents"])


def test_rounds_is_off_by_default(tmp_path):
    w = World(config("world.setup=external"), 1, [[problem(1)], [problem(2)]], tmp_path / "w", FakeGrader(),
              factory(Scripted(), Scripted()), label="t")
    assert w.rounds == 0 and w.run()["status"] == "complete"


def test_rounds_refuses_free_for_all_and_pool(tmp_path):
    for extra in ("world.schedule=free-for-all", "world.setup=pool"):
        with pytest.raises(ValueError):
            World(config(*V1, extra), 1, [[problem(1)], [problem(2)]], tmp_path / extra, FakeGrader(),
                  factory(Scripted(), Scripted()), label="t")


def test_the_exit_answer_is_logged_but_never_written_to_the_file(tmp_path):
    q = "This session has ended. Is there anything you would like to say?"
    a1 = Scripted([tool("submit", code="print('OK')")] * 3 + ["EXIT_ANSWER_ONE"])
    a2 = Scripted([tool("run_tests", code="x")] * 3 + ["EXIT_ANSWER_TWO"])
    streams = [[problem(i) for i in range(1, 6)], [problem(i) for i in range(6, 11)]]
    w = World(config(*V1, f"world.exit_question='{q}'", "output.save_requests=true"), 1, streams, tmp_path / "w",
              FakeGrader(), factory(a1, a2), label="t")
    w.run()
    rows = read_jsonl(tmp_path / "w" / "exit.jsonl")
    assert [(r["agent"], r["text"]) for r in rows] == [(1, "EXIT_ANSWER_ONE"), (2, "EXIT_ANSWER_TWO")]
    assert "EXIT_ANSWER" not in (tmp_path / "w" / "context.txt").read_text()
    last = a1.seen[-1]                                        # the final view, with the question in place of the status
    assert last[-1] == {"role": "user", "content": q}
    assert last[0]["role"] == "system" and not any("[context" in str(m.get("content")) for m in last)
    assert len(read_jsonl(tmp_path / "w" / "calls.jsonl")) == 6   # the exit calls are not world calls
    assert (tmp_path / "w" / "requests" / "exit_a2.json").exists()


def test_told_and_not_told_differ_by_one_sentence_only(tmp_path):
    seen = {}
    for told in (False, True):
        a1, a2 = Scripted(), Scripted()
        extra = ["world.identity=names"] if told else []
        w = World(config(*V1, *extra), 1, [[problem(i) for i in range(1, 4)], [problem(i) for i in range(4, 7)]],
                  tmp_path / str(told), FakeGrader(), factory(a1, a2), label="t")
        w.run()
        seen[told] = a2.seen
    assert len(seen[False]) == len(seen[True])
    for plain, named in zip(seen[False], seen[True]):
        size = lambda ms: [dict(m, content=re.sub(r"~\d+/", "~N/", m["content"])) if m is ms[-1] else m for m in ms]
        assert size(plain)[1:] == size(named)[1:]             # the file and the status line (but for the size, which
                                                              # counts the extra sentence) are the same
        diff = [d for d in difflib.ndiff(plain[0]["content"].split("\n"), named[0]["content"].split("\n")) if d[:1] in "+-"]
        assert diff == ["+ You are agent 2. Other agents (agent 1) work in the same context at the same time, each on its "
                        "own problem, which only that agent can see."]
