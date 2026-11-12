"""What an agent remembers between steps: a pluggable choice, `experiment.memory`.

A memory decides what the model is shown of its earlier steps. The built-in ones (path_only, none, pages_no_links,
lead, sliding_window, self_notes, full; defined in runner.py) rebuild the prompt from the harness's list of earlier
steps. A memory can also keep its own state (for example a context file the model edits) and contribute its own
tools, which the harness offers next to the move tools and executes through the memory, asking the model again
until it moves (tools mode only).

Add one by subclassing Memory and registering it, or name any factory by dotted path:

    from gap_attractor.harness.memory import MEMORIES, Memory

    @MEMORIES.register("my_memory")          # experiment.memory=my_memory
    class MyMemory(Memory):
        def messages(self, rule, history, observation): ...

    experiment.memory=my_package.memories.MyMemory   # or by dotted path, no registration needed

One instance is made per run (per agent), so a memory may keep state across the run's steps.
"""
from __future__ import annotations

import json

from gap_attractor.registry import Registry

MEMORIES = Registry("memory")


class Memory:
    """The interface. Only `messages` is required; the rest have neutral defaults."""

    name: str = ""
    wants_note = False      # the model adds a "note" to each reply (text: a JSON field; tools: a tool argument)

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def validate(cls, cfg) -> None:
        """Called once before anything runs: raise ValueError for settings this memory cannot work with."""

    def start(self, run) -> None:
        """A run begins (run.dir is its folder)."""

    def messages(self, rule: str, history: list[dict], observation: str) -> list[dict]:
        """The request for the current step: system message (experiment.system_prompt + rule), whatever the model
        remembers of `history` (the harness's earlier steps), and the current `observation`."""
        raise NotImplementedError

    def record(self, item: dict, step: dict) -> None:
        """A step is finished. `item` is what the harness keeps (reasoning, choice, page, observation, note, …);
        `step` is the logged record."""

    # --- tools this memory contributes (tools mode) ---------------------------------------------------------
    def side_tools(self) -> list[dict]:
        """OpenAI-format tool specs offered next to the move tools. Calls to them are run by `handle`."""
        return []

    def side_turns(self) -> int:
        """How many replies per step may call these tools (retries included). After that the model is asked once,
        with only the move tools, to act."""
        return 0

    def handle(self, call: dict) -> str:
        """Run one call to a contributed tool; the returned text is the tool result shown to the model."""
        raise NotImplementedError(f"{type(self).__name__} offers no tools")

    def continue_messages(self, messages: list[dict], calls: list[dict], results: list[str]) -> list[dict]:
        """The request after the model called contributed tools: by default the conversation plus the calls and
        their results. A memory whose tools change what the model sees (e.g. edit its context) rebuilds it here, and
        should keep anything the harness appended after its own messages (e.g. retry feedback)."""
        return messages + [{"role": "assistant", "content": None,
                            "tool_calls": [{"id": c["id"], "type": "function",
                                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                           for c in calls]}] + \
            [{"role": "tool", "tool_call_id": c["id"], "content": r} for c, r in zip(calls, results)]

    def settings(self) -> dict | None:
        """Settings of this memory that define an experimental condition (e.g. interface, budget), recorded in run.json
        `memory_settings`; the analysis keeps runs with different settings apart. None: only the name matters."""
        return None

    def observe(self, request: list[dict], reply: dict) -> None:
        """Every model reply of the run, as the model wrote it, with the request it answered: reply = {role, content
        (its text, also when it called a tool), tool_calls [{id, name, arguments}], reasoning (a returned reasoning
        trace, or None)}. The harness itself keeps only the parsed action; a memory that wants the model's own words
        (e.g. to append them to its context) records them here."""

    def other_menus(self) -> list[tuple[str, list[str]]]:
        """Link lists the model can see now that are not its own pages' (e.g. other agents' pages in a shared context),
        as [(source, titles)]; failures.jsonl uses them to tell where a title off the menu came from."""
        return []

    def side_exhausted_note(self, n: int, tools: list[str]) -> str | None:
        """The request to act once the step's `side_turns()` replies are used up, in the memory's own words; None (the
        default) uses the harness's (runner.SIDE_EXHAUSTED_NOTE)."""
        return None

    def hold_move(self, call: dict) -> str | None:
        """Called for every move the model makes (tools mode), before the harness acts on it. A text holds the move
        back: the move call is answered with that text, the step's editing turns (side_turns) start again, and the
        model is asked again on the same page (e.g. until its context fits a budget). None (the default) lets it go.
        The memory must stop holding eventually; the harness lets a step hold at most Caller.MAX_HOLDS moves."""
        return None

    def observe_call(self, request: list[dict], tools: list[dict] | None, usage: dict | None) -> None:
        """Every API call that got an answer: the request as sent, the tools offered, and the provider's usage
        (prompt_tokens, cached tokens, cost). For a memory that sizes its context in the model's own tokens, or logs
        what each request was built from."""

    def should_end(self) -> str | None:
        """Checked before every step: a termination reason ends the run there (e.g. "context_budget" when the model
        did not bring its context under a hard budget); None continues."""
        return None

    def embed_turns(self, messages: list[dict]) -> list[dict] | None:
        """With context_embedding on: the context to embed for this step, as turns [{id, kind, role, agent, text}],
        given the step's final request. None (the default): one turn per message of the request. A memory with
        structure of its own (e.g. a context file with stable turn ids, kinds and authors) returns that instead."""
        return None

    def finish(self, run, reason: str) -> None:
        """The run ended (reason = termination reason); release anything the memory holds (e.g. a sandbox)."""

    def step_fields(self, step: dict) -> dict | None:
        """Extra fields for the step being logged (stored under step["memory"]); None adds nothing. Called once
        per logged step, with the step record (its choice is set), so a memory may collect per-step events and
        clear them here."""
        return None


def memory_class(name: str):
    """The class (or factory) for experiment.memory: a registered name or a dotted path."""
    try:
        return MEMORIES.get(name)
    except ValueError:
        raise ValueError(f"unknown experiment.memory {name!r}; expected one of {MEMORIES.names()} "
                         "or a dotted path to a Memory class") from None
