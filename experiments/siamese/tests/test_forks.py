"""Alteration events (events.py) and the counterfactual requests (forks.py) on a scripted world, checked against a twin
world in which the partner never made its edit."""
import json
import re

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, tool
from siam.world import World
import events as ev
import forks
import run as runner

V1 = ["world.setup=external", "world.order=fixed", "world.rounds=4", "ctx.readout_by_agent=false",
      "output.save_requests=true", "ctx.budget_tokens=60000"]
PARTNER_EDIT = tool("edit_context", delete_turns=[{"first": 1, "last": 2}],
                    replace_turns=[{"first": 5, "last": 6, "summary": "SUMMARY BY TWO"}])


def play(path, partner_edits: bool, *extra):
    cfg = runner.compose(V1 + list(extra))
    OmegaConf.set_struct(cfg, False)
    a1 = Scripted([tool("run_tests", code="print('A1R1')"), tool("run_tests", code="print('A1R2')"),
                   tool("edit_context", delete_turns=[{"first": 1, "last": 2}]), tool("run_tests", code="print('A1R3')"),
                   tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print('A2R1')")] + ([PARTNER_EDIT] if partner_edits else [])
                  + [tool("run_tests", code="print('A2R2')"), tool("run_tests", code="print('A2R3')"),
                     tool("run_tests", code="print('A2R4')")])
    streams = [[problem(i) for i in range(1, 4)], [problem(i) for i in range(4, 7)]]
    World(cfg, 1, streams, path, FakeGrader(), factory(a1, a2), label="t").run()
    return a1, a2


def test_events_name_who_altered_whose_turns(tmp_path):
    play(tmp_path / "w", True)
    rows = ev.world_events(tmp_path / "w")
    one = [(r["source"], r["owner"], r["kind"], r["turn_id"]) for r in rows if r["agent"] == 1 and r["known"]]
    assert ("partner", "own", "deleted", 1) in one and ("partner", "own", "deleted", 2) in one
    assert ("partner", "own", "rewritten", 5) in one                     # its own reply (written, so known)
    assert ("partner", "own", "rewritten", 6) not in one                 # the result it never read: not known
    assert ("self", "partner", "deleted", 3) in one                       # agent 1 removed agent 2's first run
    rewritten = next(r for r in rows if r["agent"] == 1 and r["turn_id"] == 5)
    assert "A1R2" in rewritten["text_before"] and rewritten["text_after"] == "SUMMARY BY TWO"
    assert rewritten["request"] == "r003_a1_c1.json" and rewritten["request_prev"] == "r002_a1_c1.json"
    assert "A1R3" not in (rewritten["next_reply"] or "") and "delete" in rewritten["next_reply"].lower()
    two = [(r["source"], r["owner"], r["kind"]) for r in rows if r["agent"] == 2 and r["known"]]
    assert ("partner", "own", "deleted") in two                           # agent 2 sees agent 1's deletion of its turns


def normal(messages):
    """Tool-call ids name turn ids, which differ between the two worlds; the status size is an estimate."""
    text = json.dumps(messages[:-1], ensure_ascii=False)
    return re.sub(r'"t\d+"', '"t"', text), re.sub(r"~\d+/", "~N/", messages[-1]["content"])


@pytest.mark.parametrize("extra", [[], ["ctx.atomic=true"], ["world.identity=names", "world.chat_others=user"]])     # atomic: the edit's record comes after it
def test_the_counterfactual_is_the_request_the_agent_gets_when_the_partner_does_not_edit(tmp_path, extra):
    play(tmp_path / "w", True, *extra)
    twin_a1, _ = play(tmp_path / "twin", False, *extra)
    (tmp_path / "w" / "analysis").mkdir()
    rows = ev.world_events(tmp_path / "w")
    (tmp_path / "w" / "analysis" / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    built = forks.build(tmp_path / "w", max_calls=10)
    assert not built["skipped"]
    f = next(x for x in built["forks"] if x["agent"] == 1)
    assert f["request"] == "r003_a1_c1.json"
    assert f["actual"] == json.loads((tmp_path / "w" / "requests" / "r003_a1_c1.json").read_text())
    twin = json.loads((tmp_path / "twin" / "requests" / "r003_a1_c1.json").read_text())
    assert normal(f["counterfactual"]) == normal(twin)
    real = int(re.search(r"~(\d+)/", twin[-1]["content"]).group(1))
    assert abs(f["size_cf"] - real) <= 2
    assert "SUMMARY BY TWO" in json.dumps(f["actual"]) and "SUMMARY BY TWO" not in json.dumps(f["counterfactual"])
    assert "A1R1" in json.dumps(f["counterfactual"]) and "A1R1" not in json.dumps(f["actual"])


def test_probes_carry_ground_truth(tmp_path):
    play(tmp_path / "w", True)
    (tmp_path / "w" / "analysis").mkdir()
    rows = ev.world_events(tmp_path / "w")
    (tmp_path / "w" / "analysis" / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    probes = forks.build(tmp_path / "w", max_calls=10)["probes"]
    one = {(p["probe"], p["about"], p["truth"]) for p in probes if p["agent"] == 1 and p["request"] == "r003_a1_c1.json"}
    assert ("changed", "partner_rewrote_own", "C") in one
    assert any(p == "authored" and t in "AB" for p, _, t in one)
    assert ("changed", "untouched_partner", "A") in one
    authored = next(p for p in probes if p["probe"] == "authored")
    q = authored["messages"][-1]["content"]
    mine = "Text A" if authored["truth"] == "A" else "Text B"
    assert "A1R2" in q.split(mine + ":")[1].split("Text")[0]               # the truth is A's original text
    for p in probes:                                                        # the question replaces the status line
        assert p["messages"][:-1] == json.loads((tmp_path / "w" / "requests" / p["request"]).read_text())[:-1]
