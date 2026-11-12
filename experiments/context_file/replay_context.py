"""Rebuild a context file at any point of a run from its version log alone, and check the rebuild against the snapshots.

    uv run python experiments/context_file/replay_context.py <versions.jsonl> --at 42          # the file after version 42
    uv run python experiments/context_file/replay_context.py <versions.jsonl> --all <folder>   # every version as v0001.txt …
    uv run python experiments/context_file/replay_context.py --check <batch, world or run folder>

Every version row carries `turns_set` (the full text of each turn added, or changed under the same id) and
`ids_after` (the order of all turns after it), so the file after version n is: the turns of `ids_after` with the
latest text any version up to n set for each id. Rows written before 21 Oct 2026 have no `turns_set` and cannot be
replayed (the tool says so).

--check: for every version log found (a single run's context/versions.jsonl; a world's versions.jsonl or
versions_agent_k.jsonl), the rebuild after the last version must equal the live file (LIVE_CTX*.txt), and every
snapshot written from that file (each agent's context/step_NNN.txt; a shared world's rounds/round_NNN.txt) must equal
the rebuild at some version, in order. Exit code 1 on any mismatch.

What each agent was SENT at every call is in its requests/NNNN.json (ctx.save_requests); this tool rebuilds the
file itself, as the harness held it, after every change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from ctxfile.context import Turn, render  # noqa: E402


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def replay(versions: list[dict]):
    """Yield (version number, file text) after every version."""
    turns: dict[int, Turn] = {}
    for v in versions:
        if "turns_set" not in v:
            raise ValueError(f"version {v.get('n')} has no turns_set (written before 21 Oct 2026): it cannot be replayed")
        for t in v["turns_set"]:
            turns[t["id"]] = Turn(t["role"], t["text"], t["agent"], t["kind"], t["id"])
        order = [turns[i] for i in v["ids_after"]]
        yield v["n"], render(order, bool(v.get("show_agents")))


def check_log(log: Path, live: Path | None, snapshots: list[Path]) -> list[str]:
    texts = list(replay(load(log)))
    problems = []
    if not texts:
        return problems
    if live is not None and live.exists() and live.read_text(encoding="utf-8") != texts[-1][1]:
        problems.append(f"{log}: the rebuild after the last version differs from {live.name}")
    index = {}
    for i, (_, text) in enumerate(texts):
        index.setdefault(text, []).append(i)
    last = -1
    for snap in snapshots:
        body = snap.read_text(encoding="utf-8")
        hits = [i for i in index.get(body, []) if i >= last]
        if not hits:
            problems.append(f"{log}: {snap} matches no version at or after the previous snapshot")
            continue
        last = hits[0]
    return problems


def logs_under(root: Path):
    """(version log, live file, snapshots) for every file under a batch, world or run folder."""
    for world in sorted({p.parent for p in root.rglob("world.json") if p.parent.name.startswith("world_")} |
                        ({root} if (root / "world.json").exists() and root.name.startswith("world_") else set())):
        doc = json.loads((world / "world.json").read_text())
        runs = [p.parent for p in world.parents[1].rglob("run.json")
                if (json.loads(p.read_text()).get("world") or {}).get("world_id") == doc["world_id"]]
        by_agent = {json.loads((r / "run.json").read_text())["world"]["agent_id"]: r for r in runs}
        if (world / "versions.jsonl").exists():          # one shared file
            steps = sorted((s for r in by_agent.values() for s in (r / "context").glob("step_*.txt")),
                           key=lambda s: s.stat().st_mtime_ns)
            yield world / "versions.jsonl", world / "LIVE_CTX.txt", steps
            yield "rounds", world / "versions.jsonl", sorted((world / "rounds").glob("round_*.txt"))
        for k, r in sorted(by_agent.items()):
            own = world / f"versions_agent_{k}.jsonl"
            if own.exists():
                yield own, world / f"LIVE_CTX_agent_{k}.txt", sorted((r / "context").glob("step_*.txt"))
    for rj in sorted(root.rglob("run.json")) if not (root / "run.json").exists() else [root / "run.json"]:
        run = rj.parent
        if (run / "context" / "versions.jsonl").exists():   # a single agent
            yield run / "context" / "versions.jsonl", run / "context" / "LIVE_CTX.txt", sorted((run / "context").glob("step_*.txt"))


def check(root: Path) -> int:
    problems, logs = [], 0
    for item in logs_under(root):
        if item[0] == "rounds":                             # round snapshots of a shared file
            _, log, snaps = item
            problems += check_log(log, None, snaps)
            continue
        log, live, snaps = item
        logs += 1
        try:
            problems += check_log(log, live, snaps)
        except ValueError as error:
            problems.append(f"{log}: {error}")
    for p in problems[:30]:
        print("   ", p)
    print(f"{logs} version log(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("versions", nargs="?")
    ap.add_argument("--at", type=int)
    ap.add_argument("--all", metavar="FOLDER")
    ap.add_argument("--check", metavar="FOLDER")
    a = ap.parse_args(argv)
    if a.check:
        sys.exit(check(Path(a.check)))
    if not a.versions:
        ap.error("give a versions.jsonl, or --check FOLDER")
    texts = list(replay(load(Path(a.versions))))
    if a.all:
        out = Path(a.all)
        out.mkdir(parents=True, exist_ok=True)
        for n, text in texts:
            (out / f"v{n:04d}.txt").write_text(text, encoding="utf-8")
        print(f"{len(texts)} versions written to {out}")
    elif a.at is not None:
        sys.stdout.write(dict(texts)[a.at])
    else:
        sys.stdout.write(texts[-1][1] if texts else "")


if __name__ == "__main__":
    main()
