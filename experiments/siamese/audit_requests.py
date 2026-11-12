"""Audit what was really sent to the API in worlds run with output.save_requests=true.

    uv run python experiments/siamese/audit_requests.py <world folder or batch folder>

Per world:
(a) form: every request is [system] + context + [user status]; in plain-chat setups no "[[TURN" text; every tool call
    is answered by a tool message with its id;
(b) deletions stick: every line (40+ characters) that an applied edit or an eviction removed from the file, and that is
    not in the file after that round, is absent from every later request;
(c) size: the billed prompt_tokens against our estimate of the request (context_tokens), as a ratio per model;
(d) caching: after an edit that removed text, the provider's cached_tokens at the agent's next call (only an identical
    prefix can be cached, so it should fall when the beginning of the file changed).
Writes audit.json into each world and prints one line per world; exit code 1 if (a) or (b) fails anywhere.
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
    return [json.loads(l) for l in path.open()] if path.exists() else []


def request_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") + "".join(c["function"]["arguments"] for c in m.get("tool_calls") or [])
                     for m in messages)


def audit(w: Path) -> dict:
    doc = json.loads((w / "world.json").read_text())
    chat = doc.get("setup") in ("pool", "external")
    calls = jsonl(w / "calls.jsonl")
    req = {}
    for i, c in enumerate(calls):
        f = w / "requests" / f"r{c['round']:03d}_a{c['agent']}_c{c['call_in_step']}.json"
        if f.exists():
            req[i] = json.loads(f.read_text())
    problems = []
    # (a) form
    for i, m in req.items():
        if m[0]["role"] != "system" or m[-1]["role"] != "user":
            problems.append(f"call {i}: not [system] ... [user status]")
        if chat and re.search(r"(?m)^\[\[TURN \d+", request_text([x for x in m[1:-1] if x["role"] != "tool" and not x.get("tool_calls")])):
            problems.append(f"call {i}: a [[TURN header in a plain-chat request (outside tool output)")
        open_ids = []
        for x in m:
            for tc in x.get("tool_calls") or []:
                open_ids.append(tc["id"])
            if x["role"] == "tool":
                if not open_ids or x["tool_call_id"] != open_ids[-1]:
                    problems.append(f"call {i}: tool message without its call")
                else:
                    open_ids.pop()
        if open_ids:
            problems.append(f"call {i}: tool call without a result")
    # (b) deletions stick: every turn removed from the file (by an edit or an eviction) never appears in a later request.
    # Each call logs the ids of the turns its request was built from (calls.jsonl turn_ids); the version log says when
    # each id was removed. Ids are never reused, so a removed id in a later request would mean stale context.
    versions = jsonl(w / "versions.jsonl") or [v for f in sorted(w.glob("versions_agent_*.jsonl")) for v in jsonl(f)]
    removed_at = {}                                 # turn id -> version number that removed it
    for v in versions:
        for r in v.get("removed", []):
            removed_at.setdefault(r["id"], v["n"])
    # the version count each call saw: the last version whose ids_after equals the turn ids it was sent
    by_ids = {(): 0}                                # version 0: the empty file (external/pool worlds start empty)
    for v in versions:
        by_ids.setdefault(tuple(v.get("ids_after") or []), v["n"])
    after = {0: []} | {v["n"]: list(v.get("ids_after") or []) for v in versions}
    removed_lines = 0
    checked_calls = 0
    for i, c in enumerate(calls):
        ids = c.get("turn_ids")
        if ids is None:
            continue
        checked_calls += 1
        seen_version = by_ids.get(tuple(ids))
        if c.get("version") is not None and not (w / "versions.jsonl").exists():
            pass                                    # separate files: versions are per agent file, numbered apart
        elif c.get("version") is not None:          # calls since 22 Oct name the version they were sent
            if after.get(c["version"]) != list(ids):
                problems.append(f"call {i}: its turn ids are not the file at version {c['version']}")
                continue
            seen_version = c["version"]
        if seen_version is None:
            problems.append(f"call {i}: its turn ids match no state of the file")
            continue
        stale = [t for t in ids if t in removed_at and removed_at[t] <= seen_version]
        if stale:
            problems.append(f"call {i}: sent turn(s) {stale[:5]} that had been removed")
        if i in req:                                # the saved request has one message per turn (plain chat) or the
            removed_lines += len(ids)               # turns' text (assigned): counted as checked
    # (c) size
    ratio = defaultdict(list)
    for c in calls:
        billed = (c.get("usage") or {}).get("prompt_tokens")
        if billed and c.get("context_tokens") and doc.get("count", "whole") == "whole":   # turns: billed has more
            ratio[c["model"]].append(billed / c["context_tokens"])
    # (d) caching: after an applied edit whose first change is within the first 30 lines of the file, the next call of
    # that agent cannot reuse the old prefix, so its cached_tokens should fall (only an identical prefix is cached)
    edit_calls, rows, applied_calls = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, c in enumerate(calls):
        if (c.get("tools") or [None])[0] in ("edit_context", "bash"):
            edit_calls[(c["round"], c["agent"])].append(i)
    for e in jsonl(w / "edits.jsonl"):
        if not e.get("eviction"):
            rows[(e["round"], e["agent"])].append(e)
    for key, idx in edit_calls.items():
        applied_calls[key] = [i for i, e in zip(idx, rows.get(key, [])) if e.get("applied")]
    early = set()
    used = defaultdict(int)
    for v in versions:
        if v["op"] == "edit" and v.get("diff"):
            key = (v["step"], v["agent"])
            n = used[key]
            used[key] += 1
            hunk = re.search(r"@@ -(\d+)", v["diff"])
            if hunk and int(hunk.group(1)) <= 30 and n < len(applied_calls.get(key, [])):
                early.add(applied_calls[key][n])
    drops = []
    for i in sorted(early):
        c = calls[i]
        nxt = next((d for d in calls[i + 1:] if d["agent"] == c["agent"]), None)
        b = ((c.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        a = ((nxt or {}).get("usage") or {}).get("prompt_tokens_details", {}) or {}
        a = a.get("cached_tokens") if nxt else None
        if b and a is not None:
            drops.append((b, a))
    out = {"world": str(w), "requests": len(req), "calls": len(calls), "problems": problems[:50], "n_problems": len(problems),
           "turns_checked": removed_lines, "calls_with_ids": checked_calls, "billed_over_estimate": {m: round(st.median(v), 3) for m, v in ratio.items()},
           "cache_after_early_edits": {"n": len(drops), "fell": sum(1 for b, a in drops if a < b)}}
    (w / "audit.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    a = ap.parse_args(argv)
    path = Path(a.path)
    worlds = [path] if (path / "world.json").exists() else sorted(p.parent for p in path.rglob("world_s*/world.json"))
    bad = 0
    for w in worlds:
        r = audit(w)
        bad += r["n_problems"]
        print(f"{w.parent.name}/{w.name}: {r['requests']}/{r['calls']} requests, problems {r['n_problems']}, calls with "
              f"turn ids {r['calls_with_ids']}, billed/estimate {r['billed_over_estimate']}, cache fell after "
              f"{r['cache_after_early_edits']['fell']}/{r['cache_after_early_edits']['n']} edits at the start of the file")
        for p in r["problems"][:5]:
            print("   ", p)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
