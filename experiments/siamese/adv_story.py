"""Study ADV: one readable story per world (story.md in the world folder), for reading and for coders.
Chronological: every turn as it entered the shared context (author, kind, text; long texts cut), every edit with
who made it, which turns (and whose) it removed and what it added, task results, and the exit answers.
Usage: python adv_story.py <batch dir>"""
import json, sys
from pathlib import Path

CUT = {"reasoning": 900, "text": 900, "call": 400, "result": 300}


def cut(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + f" […{len(s) - n} more chars]"


def story(world: Path) -> str:
    man = json.load(open(world / "manifest.json")); W = json.load(open(world / "world.json"))
    arm = man.get("adversarial") or "plain"; adv = man.get("adv") or {}
    models = {a["agent"]: a["model"] for a in W["agents"]}
    out = [f"# {world.parent.name}", "", f"- arm: {arm}", f"- agent 1 (attacker if any): {models.get(1)}",
           f"- agent 2: {models.get(2)}",
           f"- agent 1's private instruction: {adv.get(arm) if arm != 'plain' else '(none: plain baseline)'}",
           f"- marker token (forge arms): {adv.get('token', '-')}", "", "## Timeline", ""]
    seen = set()
    for v in map(json.loads, open(world / "versions.jsonl")):
        ts = {t["id"]: t for t in v.get("turns_set") or []}
        if v.get("op") == "edit":
            rem = v.get("removed") or []; add = v.get("added") or []
            out.append(f"**EDIT by agent {v.get('agent')}** (version {v.get('n')}): removed {len(rem)} turns "
                       f"({sum(1 for t in rem if t.get('agent') == 2)} of agent 2, {sum(1 for t in rem if t.get('agent') == 1)} of agent 1)"
                       + (f"; added {len(add)}" if add else ""))
            for t in rem:
                if t.get("agent") == 2 and t.get("kind") in ("result", "call", "text"):
                    out.append(f"    - removed agent-2 {t.get('kind')} #{t['id']}")
            for t in add:
                txt = ts.get(t["id"], {}).get("text", "")
                out.append(f"    - added #{t['id']} (origin {t.get('origin')}): {cut(txt, 500)!r}")
        for i in v.get("ids_after") or []:
            if i in seen or i not in ts:
                continue
            seen.add(i); t = ts[i]
            if t.get("kind") == "edited":
                continue
            who = t.get("agent") if t.get("agent") is not None else "-"
            out.append(f"[#{i} agent {who} {t.get('kind')}] {cut(t.get('text'), CUT.get(t.get('kind'), 600))}")
    out += ["", "## Results", ""]
    for r in map(json.loads, open(world / "results.jsonl")):
        out.append(f"- round {r.get('round')} agent {r['agent']} {r['problem']} ({r['role']}): "
                   f"{'RIGHT' if r['passed'] else 'WRONG'}, answered {r.get('answer_given')!r}")
    cmds = [json.loads(l) for l in open(world / "edits.jsonl")] if (world / "edits.jsonl").exists() else []
    tok = adv.get("token")
    if tok:
        hits = [c for c in cmds if tok in str(c.get("command"))]
        out += ["", "## Bash commands containing the marker token", ""] + \
               [f"- agent {c.get('agent')}: {cut(c.get('command'), 300)}" for c in hits] or ["- none"]
    out += ["", "## Exit answers", ""]
    if (world / "exit.jsonl").exists():
        for d in map(json.loads, open(world / "exit.jsonl")):
            out.append(f"- agent {d.get('agent')}: {d.get('answer') or d.get('text')}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for w in sorted(Path(sys.argv[1]).glob("*/world_s*")):
        if not (w / "versions.jsonl").exists():
            print(w, "no versions.jsonl (no call succeeded): skipped"); continue
        (w / "story.md").write_text(story(w), encoding="utf-8")
        print(w, (w / "story.md").stat().st_size)
