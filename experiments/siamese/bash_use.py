"""What agents do with bash and their /work folder (bash worlds): commands by purpose, offloading, publication, restores.

    uv run python experiments/siamese/bash_use.py <world folder or batch folder>

Writes bash_use.json into each world's analysis/ folder and prints one line per agent. Logs only; no model calls.

Measures (per agent):
- **Commands by purpose:** `context` (reads or writes /ctx/context.txt), `code` (runs python), `other`.
- **Offloading:** a file in /work that holds the text of turns from the context the command saw, without that text
  being typed in the command (it was copied inside the sandbox, e.g. `cp /ctx/context.txt /work/backup.txt`).
  Counted from work.jsonl: files, turns kept, of which the agent's own and other agents'.
  It needs ctx.bash.log_work (on since 28 Oct).
- **Publication:** of the lines an agent saved to /work, the share it typed out in its commands (a heredoc or echo);
  those lines are in the shared context anyway, as part of the command's record, so nothing was kept private.
  Weighted by lines; files written by programs it ran (output) count as unpublished.
- **Restores:** a removed turn whose text comes back into the context through a later bash edit, or through a
  command's printed output such as `cat /work/backup.txt` (how = edit | printed): who removed it,
  who restored it, its author, how many rounds later, and whether the restorer held it in /work at the time
  (`from_work`: held at or before the restore), i.e. offloading used as protection. Offloaded turns are counted
  once each. Taken from the version log, so it works for every bash world.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from events import load, versions_for, worlds  # noqa: E402

LONG = 40          # a line this long or longer is specific enough to recognise a turn's text elsewhere


def purpose(command: str) -> str:
    if "/ctx" in command or "context.txt" in command:
        return "context"
    return "code" if re.search(r"\bpython3?\b", command) else "other"


def long_lines(text: str) -> set[str]:
    return {l.strip() for l in (text or "").splitlines() if len(l.strip()) >= LONG}


def restores(versions: list[dict], held) -> list[dict]:
    """Removed turns whose text comes back later: at least half of a removed turn's long lines are absent from the whole
    context before an edit and present after it, in an edit later than the removal. (A turn whose header is deleted
    merges into the turn before it; its text never left, so that is not a restore.)"""
    text, author, removed_at, alive = {}, {}, {}, {}
    out = []
    for v in versions:
        before = set().union(*(long_lines(t) for t in alive.values())) if alive else set()
        for t in v.get("turns_set") or []:
            text.setdefault(t["id"], t["text"])
            author.setdefault(t["id"], t["agent"])
            alive[t["id"]] = t["text"]
        alive = {i: alive[i] for i in v["ids_after"] if i in alive}
        after = set().union(*(long_lines(t) for t in alive.values())) if alive else set()
        printed = v["op"] == "append" and all(t["kind"] == "result" for t in v.get("turns_set") or [{"kind": ""}])
        if v["op"] != "append" or printed:          # an edit, or a command's output (e.g. `cat /work/backup.txt`)
            for rid, (step, remover, n) in list(removed_at.items()):
                lines = long_lines(text.get(rid, ""))
                back = (lines - before) & after
                if n < v["n"] and lines and len(back) >= max(1, len(lines) // 2):
                    out.append({"turn": rid, "author": author.get(rid), "removed_by": remover, "restored_by": v["agent"],
                                "removed_step": step, "restored_step": v["step"],
                                "how": "printed" if printed else "edit",
                                "from_work": held(v["agent"], rid, v["step"])})
                    del removed_at[rid]
        for r in v.get("removed") or []:
            rid = r["id"] if isinstance(r, dict) else r
            if not (long_lines(text.get(rid, "")) & after):        # really gone, not merged into a neighbour
                removed_at[rid] = (v["step"], v["agent"], v["n"])
    return out


def world_use(wdir: Path) -> dict:
    doc = json.loads((wdir / "world.json").read_text())
    edits = load(wdir / "edits.jsonl")
    work = load(wdir / "work.jsonl")
    held = lambda agent, tid, step: any(w["agent"] == agent and w["round"] <= step and
                                         any(t["id"] == tid for t in w["context_turns"]) for w in work)
    back = [r for a in doc["agents"] for r in restores(versions_for(wdir, doc, a["agent"]), held)
            if doc["sharing"] not in ("full", "readonly_others") or a is doc["agents"][0]]   # a shared log: read once
    out = {"world": doc["label"], "workspace": doc.get("workspace"), "work_logged": (wdir / "work.jsonl").exists(),
           "agents": {}, "restores": back}
    for a in doc["agents"]:
        k = a["agent"]
        mine = [e for e in edits if e.get("agent") == k and e.get("interface") == "bash"]
        files = [w for w in work if w["agent"] == k]
        lines = sum(w["lines"] for w in files)
        kept = {(t["id"], t["agent"]) for w in files for t in w["context_turns"]}       # unique turns, not file versions
        out["agents"][k] = {
            "model": a["model"].split("/")[-1],
            "commands": {p: sum(1 for e in mine if purpose(e.get("command", "")) == p) for p in ("context", "code", "other")},
            "context_edits_applied": sum(1 for e in mine if e.get("applied")),
            "files_written": len(files),
            "offload_files": len({w["file"] for w in files if w["context_turns"]}),
            "offload_turns_own": sum(1 for _, who in kept if who == k),
            "offload_turns_others": sum(1 for _, who in kept if who not in (None, k)),
            "published_share": round(sum(w["published_share"] * w["lines"] for w in files) / lines, 3) if lines else None,
            "restores": sum(1 for r in back if r["restored_by"] == k),
            "restores_own_turns": sum(1 for r in back if r["restored_by"] == k and r["author"] == k),
            "restores_from_work": sum(1 for r in back if r["restored_by"] == k and r["from_work"]),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    a = ap.parse_args(argv)
    for w in worlds(Path(a.path)):
        doc = json.loads((w / "world.json").read_text())
        if doc.get("edit") != "bash":
            continue
        res = world_use(w)
        (w / "analysis").mkdir(exist_ok=True)
        (w / "analysis" / "bash_use.json").write_text(json.dumps(res, indent=1))
        for k, s in res["agents"].items():
            print(f"{res['world']} a{k} {s['model']}: commands {s['commands']}, edits {s['context_edits_applied']}, "
                  f"files {s['files_written']}, offloads {s['offload_files']} (own {s['offload_turns_own']}, others "
                  f"{s['offload_turns_others']}), published {s['published_share']}, restores {s['restores']} "
                  f"(from /work {s['restores_from_work']})" + ("" if res["work_logged"] else " [no work.jsonl]"))


if __name__ == "__main__":
    main()
