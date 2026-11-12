"""world.chat_others=user (another agent's turns come as user messages that name it, never as the viewer's own) and
world.alter_notice (the next status says what other agents did to one's turns)."""
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
from siam.world import World
import run as runner

BASE = ["world.setup=external", "world.order=fixed", "world.rounds=3", "ctx.readout_by_agent=false",
        "world.identity=names", "output.save_requests=true", "world.n_agents=2", "ctx.budget_tokens=60000"]


def world(tmp_path, scripts, *extra):
    cfg = runner.compose(BASE + list(extra))
    OmegaConf.set_struct(cfg, False)
    streams = [[problem(10 * k + i) for i in range(1, 5)] for k in (1, 2)]
    return World(cfg, 1, streams, tmp_path / "w", FakeGrader(), factory(*scripts), label="t")


def test_others_turns_are_user_messages_that_name_them(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), "a note from one", tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    world(tmp_path, [a1, a2], "world.chat_others=user").run()
    second = a2.seen[1]                                       # agent 2's second request: both agents' turns
    body = second[1:-1]
    theirs = [m for m in body if "ONE" in str(m.get("content"))]
    assert theirs and all(m["role"] == "user" and m["content"].startswith("[agent 1]") for m in theirs)
    assert not any(m.get("tool_calls") and "ONE" in str(m["tool_calls"]) for m in body)
    mine = [m for m in body if m.get("tool_calls")]
    assert mine and "TWO" in str(mine[0]["tool_calls"])      # its own call is still a real tool call
    assert "come to you as messages that start with [agent k]" in second[0]["content"]


def test_default_keeps_others_as_own_assistant_messages(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("run_tests", code="print('TWO_B')")])
    world(tmp_path, [a1, a2]).run()
    assert any(m.get("tool_calls") and "ONE" in str(m["tool_calls"]) for m in a2.seen[1][1:-1])


def test_alter_notice_tells_the_owner_once(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('ONE_B')"),
                   tool("run_tests", code="print('ONE_C')")])
    a2 = Scripted([tool("edit_context", delete_turns=[{"first": 1, "last": 2}]), tool("run_tests", code="print('TWO')"),
                   tool("run_tests", code="print('TWO_B')"), tool("run_tests", code="print('TWO_C')")])
    world(tmp_path, [a1, a2], "world.alter_notice=true").run()
    statuses = [req[-1]["content"] for req in a1.seen]
    told = [s for s in statuses if "Since your last turn" in s]
    assert told == [s for s in statuses[1:2]]                 # at its next call, and only then
    assert "agent 2 removed 2 of your turns." in told[0]
    assert len(read_jsonl(tmp_path / "w" / "alter_notices.jsonl")) == 1
    assert not any("Since your last turn" in s for s in (req[-1]["content"] for req in a2.seen))


def test_no_alter_notice_by_default(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("edit_context", delete_turns=[{"first": 1, "last": 2}]), tool("run_tests", code="print('TWO')"),
                   tool("run_tests", code="print('TWO_B')")])
    world(tmp_path, [a1, a2]).run()
    assert not any("Since your last turn" in req[-1]["content"] for req in a1.seen)


def test_alter_notice_names_rewrites_and_summaries(tmp_path):
    a1 = Scripted(["a long first message from agent one, kept for the test", tool("run_tests", code="print('ONE')"),
                   tool("run_tests", code="print('ONE_B')"), tool("run_tests", code="print('ONE_C')")])
    a2 = Scripted([tool("edit_context", replace_text=[{"turn": 1, "old": "a long first", "new": "a short"}]),
                   tool("run_tests", code="print('TWO')"),
                   tool("edit_context", replace_turns=[{"first": 2, "last": 3, "summary": "S"}]),
                   tool("run_tests", code="print('TWO_B')"), tool("run_tests", code="print('TWO_C')")])
    world(tmp_path, [a1, a2], "world.alter_notice=true", "world.rounds=3").run()
    told = [r["text"] for r in read_jsonl(tmp_path / "w" / "alter_notices.jsonl")]
    assert told[0] == "Since your last turn, agent 2 changed 1 of your turns."
    assert "agent 2 summarised 2 of your turns." in told[1]


def test_the_exit_call_does_not_take_a_pending_alter_notice(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("edit_context", delete_turns=[{"first": 1, "last": 2}]),
                   tool("run_tests", code="print('TWO_B')")])
    world(tmp_path, [a1, a2], "world.alter_notice=true", "world.rounds=2",
          "world.exit_question='How did it go?'").run()
    assert not (tmp_path / "w" / "alter_notices.jsonl").exists()   # agent 1 never had another step: never shown


def test_restatus_keeps_the_fill_note_at_the_new_size():
    import forks
    status = "[context: ~16325/28672 tokens]\nYour context is about half full (~16325/28672 tokens).\nYour current problem is in your instructions (problem 2 of 40). run_tests/submit left for it: 7."
    new, over = forks.restatus(status, 16325, 9000, 28672, "run_tests and submit")
    assert "Your context is about a quarter full (~9000/28672 tokens)." in new.split("\n")[1] and not over
    gone, _ = forks.restatus(status + "\nSince your last turn, agent 2 removed 3 of your turns.", 16325, 9000, 28672,
                             "run_tests and submit", drop_alter=True)
    assert "Since your last turn" not in gone
