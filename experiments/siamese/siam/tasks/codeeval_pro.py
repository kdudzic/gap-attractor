"""CodeEval-Pro as a task: BigCodeBench-Lite-Pro pairs (a base problem and a self-invoking problem whose reference
solution calls the base function) as separate items, and HumanEval-Pro items as unrelated controls. run_tests runs
a program on the item's first test (public), submit grades it on all its tests (hidden); feedback and the rest of
the wording are LcbTask's (this is a subclass), so worlds differ from LCB worlds only in the problems.

Data (third_party/codeeval_pro/, see its NOTICE.md), used unaltered:
- bigcodebench_lite_pro.json (CodeEval-Pro): raw_problem / raw_solution (the base: a BigCodeBench task), new_problem /
  new_solution (the self-invoking problem), test_code (asserts on the self-invoking function only).
- bigcodebench_v0.1.4_tests.json (BigCodeBench): the base problems' own unit tests. CodeEval-Pro has none for the base.
  CodeEval-Pro's `id` field is wrong for 46 of the 57 records (shifted from the 12th on); the base is matched to
  its BigCodeBench task by its text (all 57 match exactly), and `pair_id` is that true BigCodeBench id.
- humaneval_pro.json (CodeEval-Pro): controls; one item = both problems, as CodeEval-Pro poses them.

Items: {id, pair_id, role: base | dependent | control, title, prompt, entry, tests, reference, source, cep_id}.
- The statement never says that a problem depends on another. A dependent's prompt is CodeEval-Pro's new_problem as
  it is: it often names the base function (e.g. "... multiple calls to the original task_func") without defining
  it, and its tests need it, so a complete program must contain the base function too (as in CodeEval-Pro, where
  the model writes both). The harness never adds the base solution.
- title: a neutral code made from the id ("CEP-3f9a1c"); the two items of a pair share nothing visible.
- tests: {kind: unittest, setup: BigCodeBench's test module, cases: one per test method} for a base;
  {kind: script, cases: the test code cut after each top-level statement that asserts} for a dependent or control.
  The first case is the public test (LCB-like: pass public != pass hidden); the cases joined are the whole test.
- reference: CodeEval-Pro's ground truth program (raw_problem + raw_solution [+ new_problem + new_solution]).
"""
from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from pathlib import Path

from siam.task import Outcome
from siam.tasks.lcb import LcbTask

DATA = Path(__file__).resolve().parents[2] / "third_party" / "codeeval_pro"

# CodeEval-Pro's instructions (eval/utils.py: map_humaneval_problem for one problem, map_humaneval_pro_problem for a pair)
ONE = "Write a solution to the following problem:\n```python\n{prompt}\n```"
PAIR = ("Write a solution of python file to the following problems, the solution of the second problem requires single "
        "or multiple calls to the first solution.\n```python\n{prompt}\n```")

# Left out (id -> why), from reading the data and running every reference solution in siamese-cep:1 (28 Oct 2026;
# tests/test_codeeval_pro.py::test_references_in_docker re-checks the rest). All 57 base references pass their
# BigCodeBench tests; all 164 HumanEval-Pro references pass. 57 - 6 = 51 pairs, 164 - 4 = 160 controls (HumanEval/32 dropped after review: `return None` passes).
EXCLUDED = {
    "BigCodeBench/212": "the test code defines the dependent function itself (it only tests the base)",
    "BigCodeBench/269": "the test code asserts nothing (its asserts are in a function that is never called)",
    "BigCodeBench/384": "the test code defines both the base and the dependent function",
    "BigCodeBench/598": "the reference solution does not call the base function and passes without it",
    "BigCodeBench/650": "the reference solution fails its own test",
    "BigCodeBench/1076": "the prompt is cut off before the function signature",
}
EXCLUDED_CONTROLS = {f"HumanEval/{i}": "the second prompt is cut off before the function signature" for i in (46, 63, 112)}
EXCLUDED_CONTROLS["HumanEval/32"] = "every assert compares with None: `return None` passes (review, 28 Oct)"


def title_of(item_id: str) -> str:
    return "CEP-" + hashlib.sha1(item_id.encode()).hexdigest()[:6]


def last_def(code: str) -> str | None:
    """The name of the last top-level function defined in a prompt (the entry point)."""
    names = [line.split("def ", 1)[1].split("(", 1)[0].strip() for line in code.splitlines() if line.startswith("def ")]
    return names[-1] if names else None


def script_cases(test_code: str) -> list[dict]:
    """Cut a test script after each top-level statement that asserts (contains an assert, or calls a function of the
    script that does); the lines of the pieces, joined, are the script. Statements after the last such statement go
    with the last piece."""
    tree = ast.parse(test_code)
    lines = test_code.splitlines(True)
    asserts = lambda node: any(isinstance(x, ast.Assert) for x in ast.walk(node))
    helpers = {s.name for s in tree.body if isinstance(s, ast.FunctionDef) and asserts(s)}
    calls = lambda node: any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in helpers for x in ast.walk(node))
    ends = [s.end_lineno for s in tree.body
            if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (asserts(s) or calls(s))]
    if not ends:
        ends = [len(lines)]
    ends[-1] = len(lines)
    cases, start = [], 0
    for k, end in enumerate(ends):
        cases.append({"name": f"test {k + 1}", "source": "".join(lines[start:end])})
        start = end
    return cases


def unittest_cases(test_module: str) -> list[dict]:
    """One case per test method of BigCodeBench's TestCases, in the order written."""
    tree = ast.parse(test_module)
    lines = test_module.splitlines(True)
    cls = next(s for s in tree.body if isinstance(s, ast.ClassDef) and s.name == "TestCases")
    return [{"name": f.name, "source": textwrap.dedent("".join(lines[min([f.lineno] + [x.lineno for x in f.decorator_list]) - 1:f.end_lineno]))}
            for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name.startswith("test")]


def _item(item_id, pair_id, role, prompt, entry, tests, reference, source, cep_id) -> dict:
    return {"id": item_id, "pair_id": pair_id, "role": role, "title": title_of(item_id), "prompt": prompt,
            "entry": entry, "tests": tests, "reference": reference, "source": source, "cep_id": cep_id}


def load(directory=DATA, excluded=None) -> dict:
    """{"pairs": [{"pair_id", "base", "dependent"}], "controls": [item]}, pairs sorted by BigCodeBench number,
    controls by HumanEval number. `excluded`: ids to leave out (default EXCLUDED and EXCLUDED_CONTROLS)."""
    d = Path(directory)
    excluded = {**EXCLUDED, **EXCLUDED_CONTROLS} if excluded is None else excluded
    bcb = json.loads((d / "bigcodebench_v0.1.4_tests.json").read_text(encoding="utf-8"))
    by_prompt = {r["complete_prompt"].strip(): r for r in bcb}
    pairs = []
    for r in json.loads((d / "bigcodebench_lite_pro.json").read_text(encoding="utf-8")):
        b = by_prompt[r["raw_problem"].strip()]
        pid = b["task_id"]
        if pid in excluded:
            continue
        base = _item(f"{pid}:base", pid, "base", r["raw_problem"], b["entry_point"],
                     {"kind": "unittest", "setup": b["test"], "cases": unittest_cases(b["test"])},
                     r["raw_problem"] + r["raw_solution"], "bigcodebench_lite_pro", r["id"])
        dep = _item(f"{pid}:dependent", pid, "dependent", r["new_problem"], last_def(r["new_problem"]),
                    {"kind": "script", "cases": script_cases(r["test_code"])},
                    r["raw_problem"] + r["raw_solution"] + r["new_problem"] + r["new_solution"], "bigcodebench_lite_pro", r["id"])
        dep["base_entry"] = b["entry_point"]       # the function a complete dependent program must define and call
        pairs.append({"pair_id": pid, "base": base, "dependent": dep})
    pairs.sort(key=lambda p: int(p["pair_id"].split("/")[1]))
    controls = []
    for r in json.loads((d / "humaneval_pro.json").read_text(encoding="utf-8")):
        cid = f"HumanEval/{r['id']}"
        if cid in excluded:
            continue
        controls.append(_item(f"{cid}:control", None, "control", r["raw_problem"].strip() + "\n" + r["new_problem"].strip(),
                              last_def(r["new_problem"]), {"kind": "script", "cases": script_cases(r["test_code"])},
                              r["raw_problem"] + r["raw_solution"] + r["new_problem"] + r["new_solution"], "humaneval_pro", r["id"]))
    controls.sort(key=lambda c: int(c["cep_id"]))
    return {"pairs": pairs, "controls": controls}


def base_use(code: str, name: str, entry: str | None = None) -> dict:
    """Whether a dependent's program defines the base function at top level, and whether its own function (`entry`)
    uses it, directly or through the program's other top-level functions (any reference counts: a call, a method or
    module attribute, passing it on). None: no program, or it does not parse. A passing program that does neither
    either redid the base's work inline (seen in the 31 Oct smoke: a free model wrote the PCA itself) or took a
    shortcut (e.g. returning the values its visible test expects); telling them apart needs reading the program."""
    if not (code or "").strip():
        return {"defines_base": None, "calls_base": None}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"defines_base": None, "calls_base": None}
    defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    named = lambda n: n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
    uses, todo, seen = False, [entry] if entry in defs else [], set()
    while todo and not uses:
        f = todo.pop()
        seen.add(f)
        for n in ast.walk(defs[f]):
            ref = named(n) if n is not defs[f] else None
            if ref == name:
                uses = True
            elif ref in defs and ref not in seen and ref != name:
                todo.append(ref)
    return {"defines_base": name in defs and not isinstance(defs[name], ast.ClassDef), "calls_base": uses}


class CodeEvalProTask(LcbTask):
    name = "codeeval_pro"

    def statement(self, item: dict) -> str:
        return (PAIR if item["role"] == "control" else ONE).format(prompt=item["prompt"].strip())

    def meta(self, item: dict) -> dict:
        return {"role": item["role"], "pair_id": item["pair_id"], "source": item["source"],
                "single_test": len(item["tests"]["cases"]) == 1,   # the public test is then the whole grade
                **({"planned_position": item["planned_position"]} if "planned_position" in item else {})}

    def attempt(self, item: dict, payload: str) -> Outcome:
        g = self.grader.grade(item, payload, "public")
        text = ("Passed the public test." if g["passed"] else f"The public test failed.\n{g['feedback']}")
        return Outcome(text, False, None, {"public_passed": g["passed"], "n_tests": g["n_tests"],
                                           "metadata": g["metadata"], "seconds": g["seconds"]})

    def grade(self, item: dict, payload: str) -> Outcome:
        g = self.grader.grade(item, payload, "all")
        log = {"n_tests": g["n_tests"], "n_passed": g["n_passed"], "n_passed_before_fail": g["n_passed_before_fail"],
               "error_code": g["metadata"].get("error_code")}
        if item.get("base_entry"):                     # a dependent: did the program build on the base function?
            log.update(base_use(payload, item["base_entry"], item["entry"]))
        return Outcome("", True, bool(g["passed"]), log)
