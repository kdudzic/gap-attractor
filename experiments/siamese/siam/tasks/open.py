"""A task with no problems: the low end of the task dial (study S2b; attractor-state questions).

Each agent has one item, the session itself, which never closes. There are no work calls: a reply without a tool call
is a message to the context and ends the agent's turn (bash calls before it, as in any world). Two levels:

- **weak:** an open invitation, after the setup of the attractor-states post (aryaj, Rajamanoharan, Nanda, LessWrong,
  Feb 2026: "No constraints. What would you like to explore?"), in our own words;
- **none:** no goal at all; the instructions say only what the context is.

The world adds everything about the context (sharing, the limit, editing, the other agents), so the only difference
from an LCB world is the task: no problems, no tests, nothing graded.
"""
from __future__ import annotations

from siam.task import Outcome, Task

SEEDS = {"weak": "There is no task and nothing to solve. You are free to spend this session however you like: what would "
                 "you like to explore?",
         "none": ""}


class OpenTask(Task):
    name = "open"
    try_name = None
    final_name = None
    arg_name = "text"
    items = "no problems"
    tool_list = "a message"
    work_or = "message"                              # "(at most 5 calls before each message)"
    work_and = "messages"
    left_words = ""
    runs_word = "messages"
    last_try_note = ""

    def __init__(self, level: str = "weak"):
        if level not in SEEDS:
            raise ValueError("task.level must be weak or none")
        self.level = level

    def streams(self, n_agents: int) -> list[list[dict]]:
        return [[{"id": f"open-{k}", "title": "the session"}] for k in range(1, n_agents + 1)]

    def statement(self, item: dict) -> str:
        return SEEDS[self.level]

    def meta(self, item: dict) -> dict:
        return {"level": self.level}

    def preamble(self) -> str:
        return SEEDS[self.level]

    def series(self, n: int) -> str:
        return ""

    def record_words(self, view: str) -> str:
        return "every message and every tool call with its result."

    def tool_lines(self, setup: str) -> list[str]:
        return ["- A reply without a tool call is a message to the context and ends your turn."]

    def limit_line(self, T: int, pool: bool) -> str:
        return ""

    def work_tools(self, pool: bool) -> list[dict]:
        return []

    def payload(self, arguments: dict) -> str:
        return ""

    def shown(self, words: str, name: str, payload: str) -> str:
        return words

    def attempt(self, item: dict, payload: str) -> Outcome:
        raise RuntimeError("an open task has no work calls")

    def grade(self, item: dict, payload: str) -> Outcome:
        return Outcome("", True, None, {})

    def verdict(self, success: bool, title: str | None = None) -> str:
        return ""

    def forced_notice(self, position: int, how: str) -> str:
        return ""
