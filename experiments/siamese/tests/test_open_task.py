"""The open task (siam/tasks/open.py): no problems; a message ends the turn; weak (an invitation) or none (no goal)."""
import json

from omegaconf import OmegaConf

from helpers import Scripted, factory, read_jsonl, tool
from siam.tasks.open import OpenTask
from siam.world import World
import run as runner

BASE = ["world.setup=external", "world.order=fixed", "world.rounds=3", "ctx.readout_by_agent=false",
        "world.identity=names", "output.save_requests=true", "world.n_agents=2", "ctx.budget_tokens=60000",
        "world.exit_question='How did it go?'"]


def world(tmp_path, scripts, level="weak", *extra):
    cfg = runner.compose(BASE + list(extra))
    OmegaConf.set_struct(cfg, False)
    task = OpenTask(level)
    return World(cfg, 1, task.streams(2), tmp_path / "w", task, factory(*scripts), label="t")


def test_messages_end_turns_and_nothing_is_graded(tmp_path):
    a1 = Scripted(["hello from one", "still here", "bye"])
    a2 = Scripted(["hello from two", tool("edit_context", delete_turns=[{"first": 1, "last": 1}]), "after editing", "end"])
    w = world(tmp_path, [a1, a2])
    w.run()
    assert not (tmp_path / "w" / "results.jsonl").exists()
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "still here" in text and "after editing" in text
    system = a1.seen[0][0]["content"]
    assert "what would you like to explore?" in system and "Your current problem" not in system
    assert "A reply without a tool call is a message to the context and ends your turn." in system
    assert "Your current problem" not in a1.seen[0][-1]["content"]
    assert len(read_jsonl(tmp_path / "w" / "exit.jsonl")) == 2
    assert json.loads((tmp_path / "w" / "world.json").read_text())["task"] == "open"


def test_none_level_has_no_goal(tmp_path):
    a1 = Scripted(["one", "two", "three"])
    a2 = Scripted(["one", "two", "three"])
    world(tmp_path, [a1, a2], "none").run()
    system = a1.seen[0][0]["content"]
    assert "explore" not in system and system.startswith("The messages after these instructions are your working context.")


def test_run_py_builds_an_open_world_without_problems_or_grader(tmp_path):
    cfg = runner.compose(BASE + ["task.name=open", "task.level=none", f"output.directory={tmp_path}",
                                 "world.seeds=[1]", "problems.file=/nonexistent.jsonl"])
    OmegaConf.set_struct(cfg, False)
    root, results = runner.run(cfg, make_provider=lambda pc: Scripted(["hi", "hi", "hi"], model=str(pc.model)),
                               grader_factory=lambda: (_ for _ in ()).throw(AssertionError("no grader for open")))
    assert results[0]["status"] == "complete"


def test_open_agents_are_not_told_of_private_problems_and_silence_ends_a_turn(tmp_path):
    a1 = Scripted(["", "", ""])                       # silence, every turn
    a2 = Scripted(["two", "two", "two"])
    w = world(tmp_path, [a1, a2])
    w.run()
    assert "each on its own problem" not in a1.seen[0][0]["content"]
    assert len(a1.seen) == 3 + 1                      # one call per round (silence ends the turn) + the exit question
    assert json.loads((tmp_path / "w" / "world.json").read_text())["agents"][0]["silent"] == 3
