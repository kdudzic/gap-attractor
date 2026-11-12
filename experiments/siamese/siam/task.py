"""What a world needs from a task (benchmark), so that the shared-context world (scheduling, the file, editing, the
budget, logging) does not depend on LiveCodeBench. LCB is one adapter (siam/tasks/lcb.py); another benchmark is
another subclass.

The task family the world runs: each agent has a stream of items ("problems"); on its current item an agent makes
work calls. A *try* call (LCB: run_tests) gets feedback and the item stays open; the *final* call (LCB: submit) is
graded and closes the item. After `T` work calls, or `max_steps_per_problem` steps, the item is closed with the
latest attempt (`forced`). Everything the model reads about the task comes from here; the world adds only what is
about the context (sharing, the limit, editing, turns).

For tasks with one action per step and an environment observation after each (text games), `attempt` is the move
and its observation, `terminal` ends the item, and `final_name` may be None (no separate final call).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Outcome:
    """The answer to one work call: the text added to the file, whether the item is now closed, whether it was
    solved (None while open), and fields for the logs."""
    text: str
    terminal: bool = False
    success: bool | None = None
    log: dict = field(default_factory=dict)


class Task:
    name = "task"
    try_name: str | None = None          # a work call with feedback (item stays open)
    final_name: str | None = None        # the graded call that closes the item

    @property
    def work_names(self) -> tuple:
        return tuple(n for n in (self.try_name, self.final_name) if n)

    # --- items --------------------------------------------------------------------------------------------------
    def statement(self, item: dict) -> str:
        raise NotImplementedError

    def meta(self, item: dict) -> dict:
        """Item fields for results.jsonl."""
        return {}

    # --- instructions -------------------------------------------------------------------------------------------
    def preamble(self) -> str:
        raise NotImplementedError

    def series(self, n: int) -> str:
        raise NotImplementedError

    def work_tools(self, pool: bool) -> list[dict]:
        raise NotImplementedError

    # --- work calls ---------------------------------------------------------------------------------------------
    def payload(self, arguments: dict) -> str:
        """The attempt in a work call (LCB: the program)."""
        raise NotImplementedError

    def shown(self, words: str, name: str, payload: str) -> str:
        """How a work call is written into the file."""
        raise NotImplementedError

    def attempt(self, item: dict, payload: str) -> Outcome:
        """A try call."""
        raise NotImplementedError

    def grade(self, item: dict, payload: str) -> Outcome:
        """The final grading of an attempt (a final call, or a forced close)."""
        raise NotImplementedError

    def files(self, item: dict) -> dict:
        """Files the item needs in its agent's sandbox ({relative path: text}; bash editing only). The world puts
        them in /work/env/<slug>/ while the item is open and removes them when it closes. Default: none."""
        return {}

    def close(self) -> None:
        pass
