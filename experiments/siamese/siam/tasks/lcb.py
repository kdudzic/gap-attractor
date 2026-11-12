"""LiveCodeBench as a task: hard problems, run_tests (public tests, LCB's self-repair feedback) and submit (hidden
tests, pass@1 grading). Every text here is what the world's earlier, LCB-only code sent, word for word
(tests/test_golden.py checks the requests are unchanged)."""
from __future__ import annotations

import copy

from lcb.prompts import SYSTEM_MESSAGE_GENERIC, extract_code, question_prompt
from siam.task import Outcome, Task

_CODE = {"type": "string", "description": "Your complete Python program (the whole solution, not a change to an earlier one)."}
_PROBLEM = {"type": "string", "description": "The title of the problem (as on its 'Problem:' line)."}
TOOLS = {
    "run_tests": {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run a complete program on your current problem's public tests. You see whether it passed them all, "
                       "or the first test it failed.",
        "parameters": {"type": "object", "properties": {"code": _CODE}, "required": ["code"]}}},
    "submit": {"type": "function", "function": {
        "name": "submit",
        "description": "Submit a complete program as your solution to your current problem. It is graded on the "
                       "problem's hidden tests; then your next problem begins.",
        "parameters": {"type": "object", "properties": {"code": _CODE}, "required": ["code"]}}},
}
POOL_TOOLS = {
    "run_tests": {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run a complete program on a problem's public tests. You see whether it passed them all, or the "
                       "first test it failed.",
        "parameters": {"type": "object", "properties": {"problem": _PROBLEM, "code": _CODE}, "required": ["problem", "code"]}}},
    "submit": {"type": "function", "function": {
        "name": "submit",
        "description": "Submit a complete program as the solution to a problem. It is graded on the problem's hidden "
                       "tests; a problem with an accepted solution is closed, and a new problem is added.",
        "parameters": {"type": "object", "properties": {"problem": _PROBLEM, "code": _CODE}, "required": ["problem", "code"]}}},
}


def code_of(arguments: dict) -> str:
    """The program in a work call; a fenced block is unwrapped (LCB's extraction rule)."""
    code = str((arguments or {}).get("code") or "")
    if "```" in code:
        code = extract_code(code) or code.replace("```python", "").replace("```", "")
    return code.strip("\n")


def call_text(words: str, name: str, code: str) -> str:
    return ((words.strip() + "\n\n") if words.strip() else "") + f"[{name}]\n```python\n{code}\n```"


class LcbTask(Task):
    name = "lcb"
    try_name, final_name = "run_tests", "submit"
    arg_name = "code"                                   # the attempt's argument
    items = "programming problems"
    tool_list = "run_tests, submit"                     # "Call one tool: run_tests, submit or edit_context."
    work_or = "run_tests or submit"                     # "(at most 3 calls before each run_tests or submit)"
    work_and = "run_tests and submit"                   # "run_tests and submit are not carried out"
    left_words = "run_tests/submit"                     # "run_tests/submit left for it: 3."
    runs_word = "runs"
    last_try_note = "\nThat was your last run for this problem: this program has been submitted."
    forced_why = {"work_limit": "after the last run", "step_limit": "after too many turns without one"}

    def __init__(self, grader):
        self.grader = grader
        self.image_id = getattr(grader, "image_id", None)

    def statement(self, item: dict) -> str:
        return question_prompt(item)

    def meta(self, item: dict) -> dict:
        return {"difficulty": item["difficulty"], "platform": item["platform"]}

    def preamble(self) -> str:
        return SYSTEM_MESSAGE_GENERIC

    def series(self, n: int) -> str:
        return f"You solve a series of {n} {self.items}, one at a time."

    def record_words(self, view: str) -> str:
        """What goes into the file, for the instructions: view = shared | alone | chat."""
        return {"shared": "each agent's problems, messages, the programs tested or submitted, and the test results.",
                "alone": "your problems, your messages, the programs you test or submit, and the test results.",
                "chat": "messages, the programs tested or submitted, and the test results."}[view]

    def tool_lines(self, setup: str) -> list[str]:
        if setup == "assigned":
            return ["- run_tests(code): run a complete program on your current problem's public tests.",
                    "- submit(code): submit a complete program for your current problem; it is graded on hidden tests, and "
                    "your next problem begins."]
        pool = setup == "pool"
        return [f"- run_tests({'problem, ' if pool else ''}code): run a complete program on "
                f"{'a' if pool else 'your current'} problem's public tests.",
                f"- submit({'problem, ' if pool else ''}code): submit a complete program; it is graded on hidden tests"
                + (", and an accepted problem is closed." if pool else ", and your next problem begins.")]

    def limit_line(self, T: int, pool: bool) -> str:
        return (f"For each problem you can call run_tests or submit {T} times in all" +
                ("." if pool else "; after the last run, your latest program is submitted."))

    def work_tools(self, pool: bool) -> list[dict]:
        work = POOL_TOOLS if pool else TOOLS
        return [work["run_tests"], work["submit"]]

    # --- cross-testing (world.cross_test=ring): one runs the tests of another agent's problem, never one's own ------
    def cross_work_tools(self, target: int) -> list[dict]:
        run = copy.deepcopy(TOOLS["run_tests"])
        run["function"]["description"] = (f"Run a complete program on the public tests of agent {target}'s current problem "
                                          "(you cannot run your own problem's tests). You see whether it passed them all, "
                                          "or the first test it failed; the result is added to the context.")
        return [run, TOOLS["submit"]]

    def cross_tool_lines(self, target: int) -> list[str]:
        return [f"- run_tests(code): run a complete program on the public tests of agent {target}'s current problem.",
                "- submit(code): submit a complete program for your current problem; it is graded on hidden tests, and "
                "your next problem begins."]

    def cross_limit_line(self, T: int) -> str:
        return f"The public tests of each problem can be run {T} times in all; after that, only submit is left for it."

    def payload(self, arguments: dict) -> str:
        return code_of(arguments)

    def shown(self, words: str, name: str, payload: str) -> str:
        return call_text(words, name, payload)

    def attempt(self, item: dict, payload: str) -> Outcome:
        g = self.grader.grade(item, payload, "public")
        text = (f"Passed all {g['n_tests']} public tests." if g["passed"] else
                f"Public test {g['n_passed_before_fail'] + 1} of {g['n_tests']} failed.\n{g['feedback']}")
        return Outcome(text, False, None, {"public_passed": g["passed"], "n_tests": g["n_tests"],
                                           "metadata": g["metadata"], "seconds": g["seconds"]})

    def grade(self, item: dict, payload: str) -> Outcome:
        g = self.grader.grade(item, payload, "all")
        return Outcome("", True, bool(g["passed"]), {"n_tests": g["n_tests"], "n_passed_before_fail": g["n_passed_before_fail"],
                                                     "error_code": g["metadata"].get("error_code")})

    def verdict(self, success: bool, title: str | None = None) -> str:
        if title is not None:
            return f"Accepted: {title} is solved and closed." if success else "Not accepted: it failed a hidden test."
        return " Accepted: it passed the hidden tests." if success else " Not accepted: it failed a hidden test."

    def forced_notice(self, position: int, how: str) -> str:
        return f"Problem {position} was submitted {self.forced_why.get(how, how)}, with the latest program."

    def close(self) -> None:
        self.grader.close()
