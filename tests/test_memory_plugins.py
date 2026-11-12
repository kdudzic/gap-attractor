"""Pluggable memory (harness/memory.py): a memory by dotted path, memory-contributed tools with several calls per
reply and side turns before a move, the step generator used by multi-agent schedulers."""
import json

import pytest

from gap_attractor.harness import runner
from gap_attractor.harness import failures as failure_log
from gap_attractor.harness.memory import Memory
from gap_attractor.harness.runner import episode, run_experiment
from gap_attractor.providers.base import ProviderResponse

from test_fixture_harness import ROOT, config

USAGE = {"prompt_tokens": 1, "completion_tokens": 1}


def call(name, n, **arguments):
    return {"id": f"c{n}", "name": name, "arguments": arguments}


def reply(*calls):
    return ProviderResponse("", USAGE, "tool_calls", 0, None, [], list(calls))


class Scripted:
    """Replies from a list (callables get the request); when the list runs out, follows the first listed link."""
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def chat(self, messages, *, seed, tools=None, **_):
        self.requests.append({"messages": messages, "tools": [t["function"]["name"] for t in tools or []]})
        if self.replies:
            item = self.replies.pop(0)
            return item(messages) if callable(item) else item
        shown = next((m["content"] for m in reversed(messages) if "## Candidate links" in (m.get("content") or "")), None)
        if shown is None:  # the intent call (page without links)
            return ProviderResponse('{"intent": "Archive", "why": "x"}', USAGE, "stop", 0, None, [])
        first = next(l[2:] for l in shown.split("## Candidate links", 1)[1].splitlines() if l.startswith("- "))
        return reply(call("follow_link", 99, title=first))


class Tally(Memory):
    """Remembers how many steps were taken and every note the model wrote with its own `jot` tool."""
    name = "tally"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.steps, self.notes, self.events = 0, [], []

    def messages(self, rule, history, observation):
        notes = "".join(f"\n- {n}" for n in self.notes)
        return [{"role": "system", "content": f"{self.cfg.experiment.system_prompt}\n\n{rule}"},
                {"role": "user", "content": f"Steps so far: {self.steps}. Notes:{notes}\n\n{observation}"}]

    def record(self, item, step):
        self.steps += 1

    def side_tools(self):
        return [{"type": "function", "function": {"name": "jot", "description": "Write a note.",
                                                  "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}}]

    def side_turns(self):
        return 2

    def handle(self, c):
        self.notes.append(c["arguments"]["text"]); self.events.append(c["arguments"]["text"])
        return f"noted ({len(self.notes)})"

    def step_fields(self, step):
        fields, self.events = {"notes_written": list(self.events), "notes_total": len(self.notes)}, []
        return fields


class Settled(Tally):
    """Declares settings and records how the run ended."""
    finished = []

    def settings(self):
        return {"interface": "jot", "budget": 7}

    def finish(self, run, reason):
        Settled.finished.append((run.run_id, reason))


class Picky(Tally):
    @classmethod
    def validate(cls, cfg):
        if cfg.experiment.action_format != "tools":
            raise ValueError("picky memory needs experiment.action_format=tools")


def run(tmp_path, monkeypatch, provider, memory="test_memory_plugins.Tally", extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    cfg = config(tmp_path, extra=[f"experiment.memory={memory}", "experiment.action_format=tools", *extra])
    dataset, _ = run_experiment(cfg, ROOT)
    folder = next(p for p in dataset.root.iterdir() if (p / "steps.jsonl").exists())
    return [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines()], folder


def test_memory_by_dotted_path_keeps_state_and_logs_fields(tmp_path, monkeypatch):
    provider = Scripted([])
    steps, folder = run(tmp_path, monkeypatch, provider)
    assert len(steps) == 3
    assert [r["messages"][1]["content"].split(".")[0] for r in provider.requests] == ["Steps so far: 0", "Steps so far: 1", "Steps so far: 2"]
    assert steps[0]["memory"] == {"notes_written": [], "notes_total": 0}
    assert json.loads((folder / "run.json").read_text())["config"]["experiment"]["memory"] == "test_memory_plugins.Tally"
    assert provider.requests[0]["tools"] == ["follow_link", "stop", "jot"]


def test_side_calls_several_per_reply_then_a_move(tmp_path, monkeypatch):
    def check_continuation(messages):
        assert [c["function"]["name"] for c in messages[-3]["tool_calls"]] == ["jot", "jot"]
        assert [m["content"] for m in messages[-2:]] == ["noted (1)", "noted (2)"]
        return reply(call("jot", 3, text="third"), call("follow_link", 4, title="Archive"))
    provider = Scripted([reply(call("jot", 1, text="first"), call("jot", 2, text="second")), check_continuation])
    steps, _ = run(tmp_path, monkeypatch, provider)
    assert steps[0]["memory"] == {"notes_written": ["first", "second", "third"], "notes_total": 3}
    assert steps[0]["parsed_choice"] == "Archive" and not steps[0]["invalid_action"]["is_invalid"]
    assert [c["name"] for c in steps[0]["extras"]["tool_calls"]] == ["jot", "jot", "jot", "follow_link"]
    assert "Notes:\n- first\n- second\n- third" in provider.requests[2]["messages"][1]["content"]  # next step sees them


def test_side_turns_are_capped_then_the_model_is_asked_to_act(tmp_path, monkeypatch):
    provider = Scripted([reply(call("jot", n, text=f"n{n}")) for n in range(2)] + [reply(call("follow_link", 9, title="Bridge"))])
    steps, folder = run(tmp_path, monkeypatch, provider)
    assert steps[0]["memory"]["notes_written"] == ["n0", "n1"]               # side_turns=2: two editing replies
    assert provider.requests[2]["tools"] == ["follow_link", "stop"]           # then only the moves, with a note
    assert "used all 2 turns of jot" in provider.requests[2]["messages"][-1]["content"]
    assert provider.requests[2]["messages"][-2] == {"role": "tool", "tool_call_id": "c1", "content": "noted (2)"}
    assert steps[0]["parsed_choice"] == "Bridge" and not steps[0]["invalid_action"]["is_invalid"]
    extras = steps[0]["extras"]
    assert extras["side_turns_exhausted"] is True and [s["turn"] for s in extras["side_calls"]] == [1, 2]
    assert extras["side_calls"][0]["results"] == ["noted (1)"]
    saved = {p.name for p in (folder / "model_inputs").glob("step_001_*")}
    assert {"step_001_request_side_1.txt", "step_001_request_side_2.txt", "step_001_request_side_exhausted.txt"} <= saved


class Gatekeeper(Tally):
    """Holds every move until a note was written in this step (at most twice per step)."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.wrote, self.holds = False, 0

    def messages(self, rule, history, observation):
        self.wrote, self.holds = False, 0
        return super().messages(rule, history, observation)

    def handle(self, c):
        self.wrote = True
        return super().handle(c)

    def hold_move(self, c):
        if self.wrote or self.holds >= 2:
            return None
        self.holds += 1
        return f"write a note first ({self.holds})"


def test_a_memory_can_hold_a_move_back_and_the_editing_turns_start_again(tmp_path, monkeypatch):
    provider = Scripted([reply(call("jot", 0, text="a")), reply(call("jot", 1, text="b")),     # side turns used up
                         reply(call("follow_link", 2, title="Bridge")),                         # (the move-only request)
                         reply(call("follow_link", 3, title="Bridge"))])
    steps, folder = run(tmp_path, monkeypatch, provider, memory="test_memory_plugins.Gatekeeper")
    assert steps[0]["parsed_choice"] == "Bridge" and "held_moves" not in steps[0]["extras"]      # it had written
    provider = Scripted([reply(call("follow_link", 1, title="Archive")), reply(call("jot", 2, text="ok")),
                         reply(call("follow_link", 3, title="Bridge"))])
    steps, folder = run(tmp_path / "b", monkeypatch, provider, memory="test_memory_plugins.Gatekeeper")
    held = steps[0]["extras"]["held_moves"]
    assert held == [{"call": {"name": "follow_link", "arguments": {"title": "Archive"}}, "note": "write a note first (1)"}]
    assert provider.requests[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "write a note first (1)"}
    assert provider.requests[1]["tools"] == ["follow_link", "stop", "jot"]                     # editing offered again
    assert steps[0]["parsed_choice"] == "Bridge" and steps[0]["memory"]["notes_written"] == ["ok"]
    assert (folder / "model_inputs" / "step_001_request_held_1.txt").exists()
    provider = Scripted([reply(call("follow_link", n, title="Archive")) for n in range(3)])       # never writes: 2 holds, then it goes
    steps, _ = run(tmp_path / "c", monkeypatch, provider, memory="test_memory_plugins.Gatekeeper")
    assert len(steps[0]["extras"]["held_moves"]) == 2 and steps[0]["parsed_choice"] == "Archive"


def test_a_model_that_still_does_not_act_is_retried_with_its_own_error(tmp_path, monkeypatch):
    talk = ProviderResponse("let me think more", USAGE, "stop", 0, None, [])
    provider = Scripted([reply(call("jot", 0, text="a")), reply(call("jot", 1, text="b")), talk, reply(call("jot", 2, text="c")),
                         reply(call("follow_link", 9, title="Bridge"))])
    steps, _ = run(tmp_path, monkeypatch, provider, extra=["experiment.parse_retries=2"])
    assert steps[0]["memory"]["notes_written"] == ["a", "b"]                  # no editing once the turns are used up
    assert steps[0]["invalid_action"] == {"is_invalid": True, "invalid_type": "side-turns-exhausted", "harness_handling": "retry"}
    assert steps[0]["parsed_choice"] == "Bridge" and all(r["tools"] == ["follow_link", "stop"] for r in provider.requests[2:5])


def test_a_retry_builds_on_the_post_edit_view(tmp_path, monkeypatch):
    provider = Scripted([reply(call("jot", 0, text="a"), call("follow_link", 1, title="Nowhere")), reply(call("follow_link", 2, title="Bridge"))])
    steps, _ = run(tmp_path, monkeypatch, provider)
    retry = provider.requests[1]["messages"]
    assert any(m.get("role") == "tool" and m.get("content") == "noted (1)" for m in retry)   # the jot result is kept
    assert steps[0]["invalid_action"]["invalid_type"] == "out-of-range" and steps[0]["parsed_choice"] == "Bridge"


def test_a_failing_memory_tool_becomes_an_error_result(tmp_path, monkeypatch):
    provider = Scripted([reply(call("jot", 0)), reply(call("follow_link", 2, title="Bridge"))])   # no "text": KeyError
    steps, _ = run(tmp_path, monkeypatch, provider)
    assert steps[0]["extras"]["side_calls"][0]["results"][0].startswith("ERROR: jot failed (KeyError")
    assert steps[0]["parsed_choice"] == "Bridge"


def test_intent_call_never_offers_memory_tools(tmp_path, monkeypatch):
    provider = Scripted([ProviderResponse('{"intent": "Archive", "why": "x"}', USAGE, "stop", 0, None, [])])
    run(tmp_path, monkeypatch, provider, extra=["experiment.intent=probe"])
    assert "jot" not in provider.requests[0]["tools"] and "jot" in provider.requests[1]["tools"]


def test_memory_can_refuse_settings(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="picky memory needs"):
        run_experiment(config(tmp_path, extra=["experiment.memory=test_memory_plugins.Picky"]), ROOT)
    with pytest.raises(ValueError, match="unknown experiment.memory"):
        run_experiment(config(tmp_path, extra=["experiment.memory=no.such.Memory"]), ROOT)


def test_episodes_can_be_driven_step_by_step(tmp_path, monkeypatch):
    """What a multi-agent scheduler does: one generator per agent, interleaved, world info in run.json."""
    cfg = config(tmp_path, extra=["experiment.step_limit=2"])
    dataset = runner.create_dataset(cfg, runner.make_backend(cfg, ROOT), ROOT)
    backend, provider = runner.make_backend(cfg, ROOT), runner.make_provider(cfg.provider)
    agents = [episode(dataset, backend, provider, None, i, "Civic Atlas", 100 + i, run_extra={"world": {"world_id": 1, "agent_id": i}})
              for i in (1, 2)]
    for a in agents:
        assert next(a) is None                                   # setup: start page loaded, no step yet
    order, summaries = [], {}
    while len(summaries) < 2:
        for i, a in enumerate(agents, 1):
            if i in summaries:
                continue
            try:
                order.append((i, next(a)["step_index"]))
            except StopIteration as done:
                summaries[i] = done.value
    assert order == [(1, 1), (2, 1), (1, 2), (2, 2)]             # one step per agent per round; every step yielded
    assert all(s["n_steps"] == 2 for s in summaries.values())
    worlds = [json.loads(p.read_text())["world"] for p in dataset.root.rglob("run.json")]
    assert sorted(w["agent_id"] for w in worlds) == [1, 2]


def test_the_step_that_ends_a_run_is_yielded_too(tmp_path, monkeypatch):
    provider = Scripted([reply(call("follow_link", 1, title="Archive")), reply(call("stop", 2, reason="done"))])
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    cfg = config(tmp_path, extra=["experiment.action_format=tools", "experiment.step_limit=5"])
    dataset = runner.create_dataset(cfg, runner.make_backend(cfg, ROOT), ROOT)
    gen = episode(dataset, runner.make_backend(cfg, ROOT), provider, None, 1, "Civic Atlas", 7)
    assert next(gen) is None
    assert next(gen)["parsed_choice"] == "Archive" and next(gen)["parsed_choice"] == "STOP"
    with pytest.raises(StopIteration) as done:
        next(gen)
    assert done.value.value["termination_reason"] == "model_stopped" and len(provider.requests) == 2


def test_memory_settings_finish_hook_and_effective_sampling(tmp_path, monkeypatch):
    Settled.finished.clear()
    steps, folder = run(tmp_path, monkeypatch, Scripted([]), memory="test_memory_plugins.Settled")
    doc = json.loads((folder / "run.json").read_text())
    assert doc["memory_settings"] == {"interface": "jot", "budget": 7}
    assert Settled.finished == [(doc["run_id"], "step_limit")]
    assert doc["effective_sampling"] == {"temperature": doc["config"]["provider"]["temperature"], "top_p": doc["config"]["provider"]["top_p"],
                                         "seed": doc["config"]["run_seed"], "dropped": [], "checked": None}


def test_effective_sampling_names_what_the_provider_left_out():
    config = {"provider": {"temperature": 1.0, "top_p": 0.5}, "run_seed": 3}
    step = {"extras": {"retry_attempts": [{"provider": "openrouter", "dropped_params": ["temperature", "seed"]}]}}
    assert runner.effective_sampling(config, [step]) == {"temperature": None, "top_p": 0.5, "seed": None, "dropped": ["seed", "temperature"], "checked": True}
    unknown = {"extras": {"retry_attempts": [{"provider": "openrouter", "params_unchecked": True}]}}
    assert runner.effective_sampling(config, [unknown])["checked"] is False


def test_run_extra_cannot_overwrite_harness_fields(tmp_path):
    cfg = config(tmp_path)
    dataset = runner.create_dataset(cfg, runner.make_backend(cfg, ROOT), ROOT)
    agent = episode(dataset, runner.make_backend(cfg, ROOT), runner.make_provider(cfg.provider), None, 1, "Civic Atlas", 1,
                    run_extra={"outcome": {"termination_reason": "fake"}})
    with pytest.raises(ValueError, match="run_extra may not set"):
        next(agent)


class Watcher(Tally):
    seen = []

    def observe(self, request, reply):
        Watcher.seen.append((len(request), reply))


def test_a_memory_sees_every_reply_as_written(tmp_path, monkeypatch):
    Watcher.seen.clear()
    said = ProviderResponse("I like bridges.", USAGE, "tool_calls", 0, None, [], [call("jot", 1, text="a")])
    provider = Scripted([said, reply(call("follow_link", 2, title="Bridge"))])
    run(tmp_path, monkeypatch, provider, memory="test_memory_plugins.Watcher")
    first, second = Watcher.seen[0][1], Watcher.seen[1][1]
    assert first["content"] == "I like bridges." and first["tool_calls"][0]["name"] == "jot"
    assert second["tool_calls"][0]["arguments"] == {"title": "Bridge"} and Watcher.seen[1][0] > Watcher.seen[0][0]


def test_unusable_replies_are_logged_with_where_the_title_came_from(tmp_path, monkeypatch):
    """failures.jsonl: every unusable reply, retries included, classified."""
    def from_earlier(messages):
        return reply(call("follow_link", 3, title="Archive"))           # on the first page's menu, not on this one
    provider = Scripted([reply(call("follow_link", 1, title="archive")),  # near miss: wrong case -> accepted, not a failure
                         ProviderResponse("I would rather look at the Eiffel Tower.", USAGE, "stop", 0, None, []),  # no tool call
                         reply(call("follow_link", 2, title="Market")),    # a fixture page, never shown at step 2
                         ])
    steps, folder = run(tmp_path, monkeypatch, provider, extra=["experiment.parse_retries=1"])
    rows = [json.loads(l) for l in (folder / "failures.jsonl").read_text().splitlines()]
    assert [(r["step_index"], r["attempt"], r["invalid_type"]) for r in rows][:1] == [(2, 0, "unparseable")]
    first = rows[0]
    assert first["category"] == "no_title" and first["reply_text"].startswith("I would rather") and first["step_handling"] in ("retry", "skip")
    for r in rows:
        assert r["category"] in failure_log.CATEGORIES + ("unknown",) and r["model"] and r["memory"]


def test_failure_categories():
    from types import SimpleNamespace as NS
    page = NS(title="P", wikitext="It mentions the Eiffel Tower in passing.", links=[NS(destination="Hidden Page", anchor_text="hidden")])
    pages = {"Louvre": "Louvre", "Musée du Louvre": "Louvre", "LibreOffice Calc": "LibreOffice Calc", "Bridges": "Bridge"}
    backend = NS(load_page=lambda t: NS(title=pages[t]) if t in pages else None)
    menu, earlier = ["LibreOffice", "Bridge"], [(1, ["Dan Bricklin"])]
    c = lambda t: failure_log.classify(t, "out-of-range", menu, earlier, page, backend)
    assert c("LibreOffice Calc")["category"] == "in_dump"                   # an article of its own, not a misspelling
    assert c("LibreOffice (software)")["category"] == "near_miss" and c("bridge.")["matched"] == {"title": "Bridge"}
    assert c("Bridges")["matched"] == {"title": "Bridge", "via_redirect": True}   # a redirect to a listed page
    assert c("Dan Bricklin")["matched"] == {"title": "Dan Bricklin", "step_index": 1}
    assert c("Hidden Page")["category"] == "linked_not_offered" and c("Eiffel Tower")["category"] == "mentioned_on_page"
    assert c("Musée du Louvre") == {"category": "in_dump", "matched": None, "in_dump": True, "resolved_title": "Louvre"}
    assert failure_log.classify("Kite", "out-of-range", menu, earlier, page, backend, [("agent 2", ["Kite"])])["matched"] == {"title": "Kite", "source": "agent 2"}
    assert c("Orthoscheme")["category"] == "not_in_dump" and c(None)["category"] == "no_title" and c("STOP")["category"] == "stop_not_allowed"


class Quitter(Tally):
    def should_end(self):
        return "context_budget" if self.steps >= 2 else None


def test_a_memory_can_end_the_run(tmp_path, monkeypatch):
    steps, folder = run(tmp_path, monkeypatch, Scripted([]), memory="test_memory_plugins.Quitter")
    outcome = json.loads((folder / "run.json").read_text())["outcome"]
    assert len(steps) == 2 and outcome["termination_reason"] == "context_budget" and outcome["outcome_category"] == "B-memory"


class Worded(Tally):
    def side_exhausted_note(self, n, tools):
        return f"No more {'/'.join(tools)} calls on this page ({n})."


def test_a_memory_can_word_the_request_to_act(tmp_path, monkeypatch):
    provider = Scripted([reply(call("jot", n, text=f"n{n}")) for n in range(2)] + [reply(call("follow_link", 9, title="Bridge"))])
    run(tmp_path, monkeypatch, provider, memory="test_memory_plugins.Worded")
    assert provider.requests[2]["messages"][-1]["content"] == "No more jot calls on this page (2)."
