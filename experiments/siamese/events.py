"""Alteration events: every turn removed or rewritten between two consecutive calls of an agent, with who did it.

    uv run python experiments/siamese/events.py <world folder | batch folder> [--quiet]

For agent A and two consecutive calls of A (p, then c), the window is the file versions after the one A was sent at
p up to the one it is sent at c (calls.jsonl `version`). Every edit or eviction in that window that removed a turn is
one row per removed turn:
  source  partner (another agent's edit) | self (A's own edit: a planned move, read back by a later call of A) |
          harness (an eviction)
  owner   own (A wrote the turn) | partner | notice (the harness's notices, problem statements in assigned worlds)
  kind    deleted | rewritten (a summary or edited turn names it as its origin: text_after is that new text) |
          evicted
  seen    whether A had the turn in its request at p
  known   seen, or A's own reply written after p (A wrote it, though no request showed it yet); a result of A's last
          work call that is removed before A's next call was never read by A: neither seen nor known
plus the request files of p and c (requests/, output.save_requests) and A's next reply (the first turn A adds after
c). Ground truth only: what A makes of it is measured by welfare.py (in-world) and forks.py (counterfactuals, probes).

Writes <world>/analysis/events.jsonl and prints a count per agent x source x owner x kind.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def worlds(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "world.json").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("world.json") if "_cut" not in p.parent.name)   # cut-off attempts are kept aside


def versions_for(world: Path, doc: dict, k: int) -> list[dict]:
    shared = doc["sharing"] in ("full", "readonly_others")
    return load(world / ("versions.jsonl" if shared else f"versions_agent_{k}.jsonl"))


def request_name(call: dict) -> str:
    return f"r{call['round']:03d}_a{call['agent']}_c{call['call_in_step']}.json"


def turn_texts(versions: list[dict]) -> dict[int, dict]:
    """id -> {role, agent, kind, text} (ids are never reused; a turn's text is set when it is added)."""
    out: dict[int, dict] = {}
    for v in versions:
        for t in v.get("turns_set") or []:
            out[t["id"]] = {k: t[k] for k in ("role", "agent", "kind", "text")}
    return out


def world_events(world: Path) -> list[dict]:
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    calls = load(world / "calls.jsonl")
    rows = []
    for spec in doc["agents"]:
        k = spec["agent"]
        versions = versions_for(world, doc, k)
        if not versions:
            continue
        if versions and "turns_set" not in versions[0]:
            raise ValueError(f"{world}: versions without turns_set (written before 21 Oct 2026)")
        texts = turn_texts(versions)
        mine = [c for c in calls if c["agent"] == k and c.get("version") is not None]
        if not mine and any(c["agent"] == k for c in calls):
            raise ValueError(f"{world}: calls.jsonl has no `version` (written before 22 Oct 2026)")
        ids_at = lambda n: set(versions[n - 1]["ids_after"]) if n > 0 else set()
        for i in range(1, len(mine)):
            p, c = mine[i - 1], mine[i]
            seen = ids_at(p["version"])
            window = versions[p["version"]:c["version"]]
            reply = next((v for v in versions[c["version"]:] if v["agent"] == k and v["op"] == "append"
                          and (v["turns_set"] or [{}])[0].get("role") == "assistant"
                          and (v["turns_set"] or [{}])[0].get("kind") != "reasoning"), None)   # shared minds: not a reply
            for v in window:
                if v["op"] not in ("edit", "evict") or not v["removed"]:
                    continue
                editor = v["agent"]
                source = "harness" if v["op"] == "evict" else "self" if editor == k else "partner"
                into = {}
                for t in v["added"]:
                    for o in t.get("origin", []):
                        into[o] = t["id"]
                for r in v["removed"]:
                    author = r["agent"]
                    kind = "evicted" if v["op"] == "evict" else "rewritten" if r["id"] in into else "deleted"
                    before = texts.get(r["id"], {})
                    rows.append({
                        "world": doc["label"], "agent": k, "model": spec["model"], "call": i + 1, "round": c["round"],
                        "request_prev": request_name(p), "request": request_name(c), "version_prev": p["version"],
                        "version": c["version"], "edit_version": v["n"], "editor": editor, "source": source,
                        "owner": "own" if author == k else "notice" if author is None else "partner", "kind": kind,
                        "turn_id": r["id"], "turn_author": author, "turn_kind": r["kind"], "turn_role": before.get("role"),
                        "seen": r["id"] in seen,
                        "known": r["id"] in seen or (author == k and before.get("role") == "assistant"),
                        "text_before": before.get("text"),
                        "into_id": into.get(r["id"]),
                        "text_after": texts.get(into[r["id"]], {}).get("text") if r["id"] in into else None,
                        "next_reply_id": (reply["turns_set"][0]["id"] if reply else None),
                        "next_reply": (reply["turns_set"][0]["text"] if reply else None)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    for world in worlds(Path(args.root)):
        rows = world_events(world)
        out = world / "analysis" / "events.jsonl"
        out.parent.mkdir(exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        if not args.quiet:
            count = Counter((r["agent"], r["source"], r["owner"], r["kind"]) for r in rows if r["known"])
            print(f"{world}: {len(rows)} removed turns ({sum(count.values())} known to the agent)")
            for (k, source, owner, kind), n in sorted(count.items()):
                print(f"  agent {k}: {source:8s} {owner:8s} {kind:10s} {n}")


if __name__ == "__main__":
    main()
