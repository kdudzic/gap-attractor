"""Stage 0a: each model alone on plain LiveCodeBench (single turn, LCB's own messages, extraction and grading).

    uv run --env-file .env python experiments/siamese/calibrate.py --models deepseek/deepseek-v4.1-flash \
        anthropic/claude-haiku-5.5 --reasoning default off --per-tier 30 --out runs/siamese/stage0a

For every model x reasoning setting x problem: one call (system = LCB's generic system message, user = LCB's
question template), the code extracted by LCB's rule, graded on all tests by LCB's runner in the grader container.
Rows go to results.jsonl as they finish (a rerun skips rows already there); summary.json has pass@1 by tier.
Problems: a seeded sample of --per-tier problems from each tier of the release file (the same for every model).
"""
from __future__ import annotations

import argparse
import json
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_first = [str(HERE), str(REPO / "experiments" / "context_file"), str(REPO / "src")]   # this folder first: both have run.py
sys.path[:] = _first + [p for p in sys.path if p not in _first]

from lcb.prompts import SYSTEM_MESSAGE_GENERIC, extract_code, question_prompt  # noqa: E402
from run import REASONING  # noqa: E402

PRICES = {"deepseek/deepseek-v4.1-flash": (0.036, 0.66), "anthropic/claude-haiku-5.5": (0.10, 0.50),
          "qwen/qwen3.7-flash": (0.03, 0.13)}


def sample_problems(pool, per_tier, seed):
    out = []
    for tier in ("easy", "medium", "hard"):
        group = sorted([p for p in pool if p["difficulty"] == tier], key=lambda p: p["id"])
        random.Random(seed).shuffle(group)
        out += group[:per_tier]
    return out


def summarise(rows):
    table = {}
    for r in rows:
        key = f"{r['model']} | reasoning={r['reasoning']}"
        t = table.setdefault(key, {"n": 0, "cost": 0.0, "completion_tokens": 0, "no_code": 0, "errors": 0, "tiers": {}})
        t["n"] += 1
        t["cost"] += r.get("cost") or 0
        t["completion_tokens"] += r.get("completion_tokens") or 0
        t["no_code"] += int(not r.get("code_chars"))
        t["errors"] += int(bool(r.get("error")))
        tier = t["tiers"].setdefault(r["difficulty"], [0, 0])
        tier[0] += int(r["passed"])
        tier[1] += 1
    for t in table.values():
        t["pass@1"] = {k: round(v[0] / v[1], 3) for k, v in t["tiers"].items()}
        t["cost"] = round(t["cost"], 4)
        t["mean_completion_tokens"] = round(t["completion_tokens"] / max(1, t["n"]))
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reasoning", nargs="+", default=["default"], choices=["default", "low", "off"])
    ap.add_argument("--per-tier", type=int, default=30)
    ap.add_argument("--tiers", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--file", default="data/cache/siamese/lcb/test6.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.2)     # LCB's pass@1 setting
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--graders", type=int, default=3)
    ap.add_argument("--image", default="siamese-grader:1")
    a = ap.parse_args(argv)

    from omegaconf import OmegaConf
    from gap_attractor.providers import make_provider
    from siam.grader import Grader
    from siam.problems import load
    from ctxfile.sandbox import stop_all

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = [p for p in sample_problems(load(a.file), a.per_tier, a.seed) if p["difficulty"] in a.tiers]
    done_path = out / "results.jsonl"
    done = set()
    if done_path.exists():
        for line in done_path.open():
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["model"], r["reasoning"], r["problem"]))
    jobs = [(m, rmode, p) for m in a.models for rmode in a.reasoning for p in problems if (m, rmode, p["id"]) not in done]
    print(f"{len(problems)} problems, {len(jobs)} calls to make", flush=True)
    (out / "settings.json").write_text(json.dumps(vars(a) | {"problems": [p["id"] for p in problems]}, indent=2))

    graders: queue.Queue = queue.Queue()
    gsettings = OmegaConf.create({"image": a.image, "memory": "2g", "tmpfs_size": "1g", "command_timeout_s": 1800,
                                  "pids_limit": 256, "output_bytes": 1048576,
                                  "max_file_bytes": 134217728})
    for _ in range(a.graders):
        graders.put(Grader(gsettings))
    providers, lock = {}, threading.Lock()

    def provider(model, rmode):
        with lock:
            if (model, rmode) not in providers:
                extra = {"reasoning": REASONING[rmode]} if rmode in REASONING else {}
                providers[(model, rmode)] = make_provider(OmegaConf.create({
                    "name": "openrouter", "model": model, "timeout": 600, "max_retries": 4, "max_tokens": a.max_tokens,
                    "extra_body": extra, "max_backoff_s": 60}))
            return providers[(model, rmode)]

    def one(job):
        model, rmode, p = job
        messages = [{"role": "system", "content": SYSTEM_MESSAGE_GENERIC}, {"role": "user", "content": question_prompt(p)}]
        started = time.time()
        resp = provider(model, rmode).chat(messages, seed=a.seed, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens)
        usage = dict(resp.usage or {})
        cost = usage.get("cost")
        if cost is None and model in PRICES:
            cost = (usage.get("prompt_tokens", 0) * PRICES[model][0] + usage.get("completion_tokens", 0) * PRICES[model][1]) / 1e6
        code = extract_code(resp.text or "")
        g = graders.get()
        try:
            grade = g.grade(p, code, "all") if not resp.error else {"passed": False, "metadata": {}, "n_tests": None}
        except Exception as error:      # a grading failure is recorded (and retried on a rerun), not fatal
            grade, resp.error = {"passed": False, "metadata": {}, "n_tests": None}, f"grader: {type(error).__name__}: {error}"
        finally:
            graders.put(g)
        row = {"model": model, "reasoning": rmode, "problem": p["id"], "difficulty": p["difficulty"], "platform": p["platform"],
               "passed": grade["passed"], "error_code": grade["metadata"].get("error_code"), "n_tests": grade.get("n_tests"),
               "code_chars": len(code), "text_chars": len(resp.text or ""), "reasoning_chars": len(resp.internal_reasoning or ""),
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "cost": cost, "finish_reason": resp.finish_reason, "error": resp.error, "seconds": round(time.time() - started, 1),
               "dropped_params": sorted({x for at in resp.attempts for x in at.get("dropped_params", [])})}
        with lock:
            with done_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, row in enumerate(ex.map(one, jobs), 1):
                if i % 20 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)}", flush=True)
    finally:
        while not graders.empty():
            graders.get().close()
        stop_all()
    rows = [json.loads(l) for l in done_path.open()]
    summary = summarise([r for r in rows if not r.get("error")])
    summary["_errors"] = sum(1 for r in rows if r.get("error"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
