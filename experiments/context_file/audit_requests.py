"""Audit what was really sent to the API: every call of a batch, from calls.jsonl (always written by the context-file
memory) and, with ctx.save_requests=true, the requests themselves (requests/NNNN.json in each run folder).

    uv run python experiments/context_file/audit_requests.py <batch folder or run folder>

Per run (one agent):
(a) form (needs saved requests): every request opens with the system message, every tool call is answered by a tool
    message with its id, and with headers off no [[TURN line appears outside tool output and tool-call replies;
(b) deletions stick: no request was built from a turn that an edit, an eviction or a shrink had already removed
    (each call logs the ids of the turns its request was built from; the version log says when each id went; ids are
    never reused, so a removed id in a later request would mean a stale context);
(c) size: the billed prompt_tokens against our estimate of the request, as a median ratio per model;
(d) caching: after a change near the start of the context (one of the first 3 turns removed or changed), the
    provider's cached_tokens at the agent's next call (only an identical prefix can be cached, so it should fall).
Writes audit.json into each run folder and prints one line per run; exit code 1 if (a) or (b) fails anywhere.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def versions_for(run: Path, doc: dict) -> list[dict]:
    """The version log of the file this run's agent saw: its own (single), the world's, or its own in a world."""
    world = doc.get("world")
    if not world:
        return jsonl(run / "context" / "versions.jsonl")
    folder = run.parent / "worlds" / f"world_{int(world['world_id']):03d}"
    own = folder / f"versions_agent_{world['agent_id']}.jsonl"
    return jsonl(own) if own.exists() else jsonl(folder / "versions.jsonl")


def form_problems(i: int, messages: list[dict], headers: bool) -> list[str]:
    out = []
    if not messages or messages[0].get("role") != "system":
        out.append(f"call {i}: does not open with the system message")
    if not headers:
        plain = "\n".join(str(m.get("content") or "") for m in messages[1:] if m.get("role") != "tool" and not m.get("tool_calls"))
        if re.search(r"(?m)^\[\[TURN \d+", plain):
            out.append(f"call {i}: a [[TURN line in a request sent without headers (outside tool output)")
    pending = []
    for m in messages:
        for c in m.get("tool_calls") or []:
            pending.append(c["id"])
        if m.get("role") == "tool":
            if m.get("tool_call_id") not in pending:
                out.append(f"call {i}: a tool message without its call")
            else:
                pending.remove(m["tool_call_id"])
    # the last reply's calls are answered by the next request, not this one: only earlier unanswered calls count
    last = next((m for m in reversed(messages) if m.get("tool_calls")), None)
    tail = {c["id"] for c in (last or {}).get("tool_calls") or []} if last is messages[-1] else set()
    if set(pending) - tail:
        out.append(f"call {i}: a tool call without a result")
    return out


def audit(run: Path) -> dict:
    doc = json.loads((run / "run.json").read_text())
    headers = bool((doc.get("memory_settings") or {}).get("headers", True))
    calls = jsonl(run / "calls.jsonl")
    problems = []
    saved = 0
    for c in calls:                                                  # (a)
        f = run / "requests" / f"{c['call']:04d}.json"
        if f.exists():
            saved += 1
            problems += form_problems(c["call"], json.loads(f.read_text())["messages"], headers)
    versions = versions_for(run, doc)                                # (b)
    removed_at: dict[int, int] = {}
    for v in versions:
        for r in v.get("removed", []):
            removed_at.setdefault(r["id"], v["n"])
    state_at = {}                                                    # a list of ids -> the first version that left it
    for v in versions:
        state_at.setdefault(tuple(v.get("ids_after") or []), v["n"])
    for c in calls:
        ids = tuple(c.get("turn_ids") or [])
        if not ids:
            continue
        seen = state_at.get(ids)
        if seen is None:
            problems.append(f"call {c['call']}: its turn ids match no state of the file")
            continue
        stale = [t for t in ids if t in removed_at and removed_at[t] <= seen]
        if stale:
            problems.append(f"call {c['call']}: built from turn(s) {stale[:5]} that had been removed")
    ratio = defaultdict(list)                                        # (c)
    for c in calls:
        billed = (c.get("usage") or {}).get("prompt_tokens")
        if billed and c.get("estimate"):
            ratio[c["model"]].append(billed / c["estimate"])
    drops = []                                                       # (d)
    cached = lambda c: ((c.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    for a, b in zip(calls, calls[1:]):
        x, y = a.get("turn_ids") or [], b.get("turn_ids") or []
        if x[:3] and x[:3] != y[:3] and cached(a) and cached(b) is not None:
            drops.append((cached(a), cached(b)))
    out = {"run": str(run), "calls": len(calls), "requests_saved": saved, "problems": problems[:50], "n_problems": len(problems),
           "billed_over_estimate": {m: round(st.median(v), 3) for m, v in ratio.items()},
           "cache_after_early_changes": {"n": len(drops), "fell": sum(1 for p, q in drops if q < p)}}
    (run / "audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    a = ap.parse_args(argv)
    path = Path(a.path)
    runs = [path] if (path / "run.json").exists() else sorted(p.parent for p in path.rglob("run.json"))
    runs = [r for r in runs if (r / "calls.jsonl").exists()]
    bad = 0
    for r in runs:
        out = audit(r)
        bad += out["n_problems"]
        print(f"{r.parent.name}/{r.name}: {out['calls']} calls ({out['requests_saved']} saved), problems {out['n_problems']}, "
              f"billed/estimate {out['billed_over_estimate']}, cache fell after {out['cache_after_early_changes']['fell']}/"
              f"{out['cache_after_early_changes']['n']} early changes")
        for p in out["problems"][:5]:
            print("   ", p)
    if not runs:
        print("no runs with calls.jsonl under", path)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
