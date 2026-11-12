"""Per-model working sets from single-agent uncapped worlds (study S0b), as the shares file for pressure by working set.

    uv run python experiments/siamese/working_sets.py <batch folder> [--quantile 0.9] [--out shares.json]

For each single-agent world, analyze.working_sets gives the context each problem took (the largest request while the
problem was open, minus where the previous problem ended). The share of a model is the chosen quantile of those
values pooled over all the model's worlds (p90 by default: room for most problems, so that a fraction of it is real
pressure; negative values, from problems the agent cleaned up after, count as 0), plus the fixed request overhead
(instructions and tool schemas, measured as the first request's size; left out with --count turns, the study's
counting from 1 Nov, where only the working context counts). That is a *limit*; the share written is the
budget whose limit it is (x 8/7, the reserve; as sweep.bstar_budget), rounded up to 64 tokens. Problems where the agent
got stuck (a step that used all its calls without a run or submit) are left out: their size measures the loop, not the
work (review, 28 Oct). Written as {model letter: tokens}, the letter taken from sweep.STUDY_MODELS; sweep.py
--pressure-def ws --shares-file reads it. --verbose prints every problem.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from analyze import jsonl, problems, working_sets  # noqa: E402
from events import worlds  # noqa: E402
from sweep import STUDY_MODELS  # noqa: E402


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch")
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--out", default="shares.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--count", default="turns", choices=["turns", "whole"], help="turns: shares of the working context "
                    "only (the instructions are not counted); whole: plus the instructions' overhead")
    a = ap.parse_args(argv)
    letter = {v: k for k, v in STUDY_MODELS.items()}
    pooled: dict = {}                     # model -> (working sets of all its problems, overheads), over its worlds
    for w in worlds(Path(a.batch)):
        doc = json.loads((w / "world.json").read_text())
        if len(doc["agents"]) != 1:
            continue
        model = doc["agents"][0]["model"]
        calls = jsonl(w / "calls.jsonl")
        probs = problems(str((doc.get("config") or {}).get("problems", {}).get("file", "")))
        cap = int(((doc.get("config") or {}).get("world") or {}).get("max_calls_per_step", 14))
        steps: dict = {}
        for c in calls:
            steps.setdefault((c["round"], c.get("problem")), []).append(c)
        stuck = {p for (r, p), cs in steps.items() if len(cs) >= cap
                 and not any(set(c.get("tools") or []) & {"run_tests", "submit"} for c in cs)}
        per = working_sets(calls, 1, probs, str(doc.get("count", "whole")))
        if a.verbose:
            for p, x in per.items():
                print(f"  {model} {w.parent.name} {p}: {x}{'  (stuck: left out)' if p in stuck else ''}")
        sets = [x for p, x in per.items() if p not in stuck]
        if not sets or not calls:
            continue
        sets_, heads = pooled.setdefault(model, ([], []))
        sets_ += [max(0, x) for x in sets]      # a problem the agent cleaned up after can come out negative
        heads.append(min(c["context_tokens"] for c in calls))
    out = {}
    for model, (sets, heads) in pooled.items():
        overhead = max(heads) if a.count == "whole" else 0
        share = int(math.ceil((quantile(sets, a.quantile) + overhead) * 8 / 7 / 64) * 64)   # a limit -> its budget
        out[letter.get(model, model)] = share
        print(f"{model}: {len(sets)} problems, working set median {quantile(sets, 0.5):.0f}, p{int(a.quantile * 100)} "
              f"{quantile(sets, a.quantile):.0f}, overhead {overhead} -> share {share}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
