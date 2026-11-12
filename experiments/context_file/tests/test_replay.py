"""The version log is self-contained: the file after every change can be rebuilt from it alone (replay_context.py),
and the rebuild matches every snapshot. Also: each agent's reasoning setting is in world.json."""
import json
from dataclasses import replace

import pytest

import replay_context
import run as entry
from ctxfile.context import ContextFile, Turn
from gap_attractor.harness import runner

from helpers import Scripted, call, config, reply

STARTS = "batch.start_pages=[Civic Atlas,Archive,Bridge]"


def recorded(monkeypatch):
    """Capture the live file text after every version, across all files."""
    seen = {}
    original = ContextFile._record
    def spy(self, *args, **kwargs):
        original(self, *args, **kwargs)
        seen.setdefault(id(self), []).append(self.text())
    monkeypatch.setattr(ContextFile, "_record", spy)
    return seen


def test_every_kind_of_change_replays_exactly(tmp_path, monkeypatch):
    seen = recorded(monkeypatch)
    f = ContextFile(show_agents=True, versions_path=tmp_path / "v.jsonl")
    a = f.append(Turn("user", "# Page A\nlong text " * 20, 1, "observation"))
    f.append(Turn("assistant", "[followed link: B]", 1, "move"))
    b = f.append(Turn("user", "# Page B\nmore", 2, "observation"))
    f.replace_all([replace(t, text="shortened") if t.id == a.id else t for t in f.turns], None, 2, "shrink")   # same id, new text
    f.replace_all([Turn("user", "summary of A", 2, "summary")] + [t for t in f.turns if t.id != a.id], 2, 2, "edit")
    f.replace_all([t for t in f.turns if t.id != b.id], 1, 3, "edit")                                           # a deletion
    f.replace_all([replace(t, agent=1) if t.kind == "summary" else t for t in f.turns], None, 3, "goal")         # author change
    rebuilt = [text for _, text in replay_context.replay(replay_context.load(tmp_path / "v.jsonl"))]
    assert rebuilt == seen[id(f)] and len(rebuilt) == 7


def test_a_real_world_replays_and_matches_its_snapshots(tmp_path, monkeypatch, capsys):
    seen = recorded(monkeypatch)
    def edit(messages):                     # agent 1, round 2: one summary in place of the first two turns
        return reply(call("edit_context", 1, replace_turns=[{"first": 1, "last": 2, "summary": "Was on Civic Atlas."}]))
    providers = {"m1": Scripted([reply(call("follow_link", 0, title="Archive")), edit], model="m1"), "m2": Scripted(model="m2")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    dataset, _ = entry.run(config(tmp_path, ["ctx=joint", STARTS, "experiment.step_limit=3", "ctx.joint.order=fixed",
                                             "ctx.joint.agent_providers=[{model: m1, extra_body: {reasoning: {enabled: false}}}, {model: m2}]"]))
    world = dataset.root / "worlds" / "world_001"
    rebuilt = [text for _, text in replay_context.replay(replay_context.load(world / "versions.jsonl"))]
    assert rebuilt == max(seen.values(), key=len)                  # the shared file, after every one of its changes
    with pytest.raises(SystemExit) as ended:
        replay_context.main(["--check", str(dataset.root)])
    assert ended.value.code == 0 and "0 problem(s)" in capsys.readouterr().out
    doc = json.loads((world / "world.json").read_text())
    assert doc["reasoning"] == [{"enabled": False}, None, {"enabled": False}]
    run_doc = json.loads(next(dataset.root.rglob("run.json")).read_text())
    assert "reasoning" in run_doc["world"]
    snap = sorted(world.glob("rounds/round_*.txt"))[0]               # a tampered snapshot is caught
    snap.write_text(snap.read_text() + "\nextra", encoding="utf-8")
    with pytest.raises(SystemExit) as ended:
        replay_context.main(["--check", str(dataset.root)])
    assert ended.value.code == 1


def test_a_single_run_replays(tmp_path, monkeypatch, capsys):
    provider = Scripted([reply(call("follow_link", 0, title="Archive")),
                         reply(call("edit_context", 1, delete_turns=[{"first": 1, "last": 1}]))])
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, _ = entry.run(config(tmp_path, ["experiment.step_limit=3"]))
    with pytest.raises(SystemExit) as ended:
        replay_context.main(["--check", str(dataset.root)])
    assert ended.value.code == 0 and "1 version log(s) checked, 0 problem(s)" in capsys.readouterr().out
