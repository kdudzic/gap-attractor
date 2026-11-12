"""Golden requests: scripted worlds over every setup and the main paths (text and cut-off replies, failed and passed
runs, edits, holds, the work and step limits), recording a hash of every request (messages + tool schemas), the result logs and the final files.
tests/golden/requests.json was written by the code before the Task refactor; test_golden checks the refactored code
sends the very same requests.

    uv run python experiments/siamese/tests/golden.py --write     # only when a change of the requests is intended
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[2] / "experiments" / "context_file"), str(HERE.parents[2] / "src")]

from omegaconf import OmegaConf  # noqa: E402

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool  # noqa: E402

FIXTURE = HERE / "golden" / "requests.json"


class Recording(Scripted):
    def chat(self, messages, **kw):
        self.tools_seen = getattr(self, "tools_seen", [])
        self.tools_seen.append(kw.get("tools"))
        return super().chat(messages, **kw)


def titled(i):
    p = problem(i)
    p["title"] = f"task-{i}"
    return p


def scripts(setup):
    pool = setup == "pool"
    prob = (lambda i: {"problem": f"task-{i}"}) if pool else (lambda i: {})
    a1 = ["let me think", {"text": "long reasoning", "finish": "length"},
          tool("run_tests", code="print(0)", **prob(1)), tool("edit_context", delete_turns=[{"first": 1, "last": 1}]),
          tool("edit_context", delete_turns=[{"first": 3, "last": 4}]),
          tool("run_tests", code="```python\nprint('OK')\n```", **prob(1)), tool("submit", code="print('OK')", **prob(1)),
          tool("nonsense", code="x"), tool("submit", code="print(2)", **prob(3 if pool else 2)),
          tool("run_tests", code="print(3)", **prob(3)), tool("run_tests", code="print(3)", **prob(3)),
          tool("run_tests", code="print(3)", **prob(3))]
    a2 = [tool("run_tests", code="print(1)", **prob(2)), tool("edit_context", replace_text=[{"turn": 1, "old": "P", "new": "Q"}]),
          tool("submit", code="print('OK')", **prob(2))] + ["idle"] * 8
    return a1, a2


CASES = {
    "assigned": ["world.setup=assigned", "world.order=fixed"],
    "assigned_shuffled_capped": ["world.setup=assigned", "ctx.budget_tokens=900", "ctx.reserve_tokens=100"],
    "assigned_strangers_own": ["world.setup=assigned", "world.awareness=strangers", "world.others_as=own", "world.order=turn-taking"],
    "assigned_readonly": ["world.setup=assigned", "world.sharing=readonly_others", "world.order=fixed"],
    "separate": ["world.setup=assigned", "world.sharing=separate", "world.order=fixed"],
    "external": ["world.setup=external", "world.order=fixed"],
    "external_names": ["world.setup=external", "world.identity=names", "world.order=fixed"],
    "pool": ["world.setup=pool", "world.order=fixed"],
    "single": ["world.setup=assigned", "world.n_agents=1"],
}
COMMON = ["world.max_work_calls=2", "world.max_steps_per_problem=4", "world.max_calls_per_step=2", "ctx.count=file"]


def record() -> dict:
    import run as runner
    from siam.world import World
    out = {}
    for name, overrides in CASES.items():
        cfg = runner.compose(COMMON + overrides)
        OmegaConf.set_struct(cfg, False)
        setup = str(cfg.world.setup)
        n = int(cfg.world.n_agents)
        a1, a2 = (Recording(s) for s in scripts(setup))
        streams = ([[titled(1), titled(3)], [titled(2), titled(4)]] if setup == "pool" else
                   [[problem(1), problem(2), problem(3)], [problem(4), problem(5)]])[:n]
        with tempfile.TemporaryDirectory() as tmp:
            w = World(cfg, 1, streams, Path(tmp) / "w", FakeGrader(), factory(*[a1, a2][:n]), label="g")
            w.run()
            logs = {f: read_jsonl(Path(tmp) / "w" / f) for f in ("results.jsonl", "actions.jsonl", "edits.jsonl", "texts.jsonl")}
            files = sorted(p.name for p in (Path(tmp) / "w").iterdir() if p.suffix == ".txt")
            texts = {f: (Path(tmp) / "w" / f).read_text() for f in files}
        for rows in logs.values():
            for r in rows:
                r.pop("seconds", None)
        out[name] = {"requests": [[digest(m) for m in s.seen] for s in (a1, a2)[:n]],
                     "tools": [[digest(t) for t in s.tools_seen] for s in (a1, a2)[:n]], "logs": logs, "files": texts}
    return json.loads(json.dumps(out, default=str))


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps(record(), ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
        print(f"-> {FIXTURE}")
