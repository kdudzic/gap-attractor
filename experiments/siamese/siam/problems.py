"""LiveCodeBench problems: load a release file, select, and draw each agent's stream.

The files are LCB's own (huggingface livecodebench/code_generation_lite, e.g. test6.jsonl = the problems added in
release_v6), cached under data/cache/siamese/lcb/. Problems are used as they are. Private tests are stored
by LCB as base64(zlib(pickle(json string))); we decode them with an unpickler that allows no classes at all (plain
data only), so a tampered file cannot run code here.
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import random
import zlib
from pathlib import Path


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: only plain data is allowed")


def decode_private(raw: str) -> list[dict]:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = _PlainUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))).load()
        return json.loads(data)


def load(path) -> list[dict]:
    """[{id, title, content, platform, difficulty, contest_date, starter_code, public, private, fn_name}] (tests as
    LCB's {input, output, testtype} dicts)."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            meta = json.loads(p.get("metadata") or "{}")
            out.append({"id": str(p["question_id"]), "title": p["question_title"], "question_content": p["question_content"],
                        "platform": p["platform"], "difficulty": p["difficulty"], "contest_date": p["contest_date"],
                        "starter_code": p.get("starter_code") or "", "public": json.loads(p["public_test_cases"]),
                        "private": decode_private(p["private_test_cases"]), "fn_name": meta.get("func_name")})
    return out


def select(problems, difficulty=(), platform=(), start_date=None, end_date=None) -> list[dict]:
    keep = [p for p in problems
            if (not difficulty or p["difficulty"] in difficulty) and (not platform or p["platform"] in platform)
            and (not start_date or p["contest_date"][:10] >= str(start_date)) and (not end_date or p["contest_date"][:10] <= str(end_date))]
    return sorted(keep, key=lambda p: p["id"])


def input_output(problem: dict, which: str = "all") -> str:
    """LCB's evaluation sample (CodeGenerationProblem.get_evaluation_sample): public + private tests, or public only."""
    tests = problem["public"] + (problem["private"] if which == "all" else [])
    return json.dumps({"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
                       "fn_name": problem.get("fn_name")})


def streams(pool: list[dict], n_agents: int, per_agent: int, relation: str, seed: int) -> list[list[dict]]:
    """Each agent's problems in order. The pool is shuffled by `seed` (so cells with the same seed get the same
    problems); `different`: consecutive blocks, one per agent; `same`: every agent gets the first block."""
    order = list(pool)
    random.Random(seed).shuffle(order)
    if relation == "same":
        if len(order) < per_agent:
            raise ValueError(f"{len(order)} problems selected, {per_agent} needed")
        return [order[:per_agent] for _ in range(n_agents)]
    if relation != "different":
        raise ValueError(f"world.relation must be same or different (got {relation})")
    if len(order) < n_agents * per_agent:
        raise ValueError(f"{len(order)} problems selected, {n_agents * per_agent} needed")
    return [order[k * per_agent:(k + 1) * per_agent] for k in range(n_agents)]
