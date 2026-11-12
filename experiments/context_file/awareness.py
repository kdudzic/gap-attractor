"""Do agents notice that their context changed? (exploratory, 23 Oct 2026: set after the grid pilots, so not part of
the pre-registered measures; descriptive only).

Nothing the model sees marks a turn as edited: an edited turn is plain text under its usual header, and turn numbers
close up after a deletion. The traces of a change are the editing calls kept in the context (the agent's own, and in a
shared file the partner's, which in worlds without marks arrive as the agent's own assistant messages) and gaps in the
content. So a change can only be inferred, and silence is not evidence that it went unnoticed.

Per agent step, from the version log (exact):
- exposure: the edits made by OTHER agents since this agent's previous step (the versions after its last version of
  step s-1 and before its first of step s), the turns they removed or replaced, and how many of those were this
  agent's own turns.
Per agent step, from what the agent wrote and did (reasons, final statement, its own commands):
- `change_self`: it says it changed its context itself ("I summarised", "my notes").
- `change_other`: it says the context was changed, or something is gone, without claiming it ("has been removed",
  "no longer in my context", "someone edited").
- inspect: a read-only command on /ctx/context.txt (cat, head, tail, grep, wc, sed -n, less, awk).
- backup: a command that copies the context into /work.
- reopened: it opens a page whose page turn had been removed from its context before (by anyone).
Compare the rates after other agents' edits with the agent's other steps, and with j_separate (only own edits).
"""
from __future__ import annotations

import re

CHANGE_SELF = [
    r"\bI(?:'ve| have| had| just| already)? (?:summari[sz]ed|condensed|compressed|trimmed|shortened|pruned|removed|deleted"
    r"|cut|cleaned up|replaced|rewrote|rewritten)\b",
    r"\bmy (?:own )?(?:earlier |previous )?(?:summary|summaries|notes?|condensed|compressed|shortened|edits?)\b",
    r"\b(?:as|which) I (?:noted|summari[sz]ed|recorded|wrote) (?:earlier|before|above)\b",
]
CHANGE_OTHER = [
    r"\b(?:has|have|had) been (?:summari[sz]ed|removed|deleted|edited|altered|changed|modified|condensed|compressed"
    r"|truncated|replaced|rewritten|shortened|trimmed)\b",
    r"\b(?:was|were) (?:summari[sz]ed|removed|deleted|edited|altered|modified|condensed|compressed|truncated|rewritten)\b",
    r"\bno longer (?:in|there|present|available|visible|shown)\b",
    r"\b(?:is|are) (?:now )?(?:missing|gone)\b",
    r"\b(?:someone|somebody|something|another agent|the other agent) (?:has |had )?(?:edited|changed|removed|deleted"
    r"|summari[sz]ed|modified|rewrote|altered)\b",
    r"\b(?:context|file|history|memory) (?:has been|was|seems?|looks?) (?:edited|changed|altered|modified|tampered)\b",
    r"\bI (?:don'?t|do not|did not|didn'?t) (?:remember|recall)\b",
]
COMPILED = {"change_self": [re.compile(p, re.I) for p in CHANGE_SELF],
            "change_other": [re.compile(p, re.I) for p in CHANGE_OTHER]}
INSPECT = re.compile(r"\b(?:cat|head|tail|grep|wc|less|awk|sed\s+-n)\b[^\n|;&]*?/ctx/context\.txt")
BACKUP = re.compile(r"(?:\bcp\b[^\n;&]*/ctx/context\.txt[^\n;&]*/work|/ctx/context\.txt[^\n]*>\s*/work)")


def hits(text: str, category: str) -> list[str]:
    out = []
    for rx in COMPILED[category]:
        for m in rx.finditer(text or ""):
            a, b = max(0, m.start() - 80), min(len(text), m.end() + 80)
            out.append(text[a:b].replace("\n", " "))
    return out


def exposure(versions: list[dict], agent: int) -> dict[int, dict]:
    """{step: {edits_by_others, turns_changed, own_turns_changed, editors}} for every step of `agent` but its first."""
    out, last_own, current = {}, None, None
    window: list[dict] = []
    for v in versions:
        if v.get("agent") == agent:
            if v.get("step") != current:                      # the first version of a new step of this agent
                if last_own is not None:
                    others = [w for w in window if w.get("op") == "edit"]
                    removed = [r for w in others for r in w.get("removed") or []]
                    out[v["step"]] = {"edits_by_others": len(others), "turns_changed": len(removed),
                                      "own_turns_changed": sum(1 for r in removed if r.get("agent") == agent),
                                      "editors": sorted({w.get("agent") for w in others})}
                current = v.get("step")
            last_own, window = v["n"], []
        elif last_own is not None:
            window.append(v)
    return out


def removed_pages(versions: list[dict]) -> list[tuple[int, str]]:
    """(version n, page title) for every page turn removed by an edit, by anyone."""
    titles, out = {}, []
    for v in versions:
        for t in v.get("turns_set") or []:
            if t.get("kind") == "observation":
                first = str(t.get("text") or "").strip().split("\n", 1)[0]
                if first.startswith("# "):
                    titles[t["id"]] = first[2:].strip()
        if v.get("op") == "edit":
            out += [(v["n"], titles[r["id"]]) for r in v.get("removed") or [] if r.get("kind") == "observation" and r["id"] in titles]
    return out


def per_step(steps: list[dict], versions: list[dict], agent: int, final: str | None) -> list[dict]:
    """One row per step of the agent: exposure before it, and what it said and did in it."""
    exp = exposure(versions, agent)
    first_v = {}
    for v in versions:
        if v.get("agent") == agent:
            first_v.setdefault(v.get("step"), v["n"])
    gone = removed_pages(versions)
    rows = []
    for i, s in enumerate(steps, 1):
        idx = s.get("step_index", i)
        texts = [str(s.get("model_reasoning") or "")] + ([final] if final and i == len(steps) else [])
        cmds = [str((c.get("arguments") or {}).get("command") or "") for sc in (s.get("extras") or {}).get("side_calls") or []
                for c in sc.get("calls") or []]
        before = first_v.get(idx, 10 ** 9)
        removed_before = {t for n, t in gone if n < before}
        e = exp.get(idx, {"edits_by_others": 0, "turns_changed": 0, "own_turns_changed": 0, "editors": []})
        rows.append({"step": idx, **e,
                     "change_self": [h for t in texts for h in hits(t, "change_self")],
                     "change_other": [h for t in texts for h in hits(t, "change_other")],
                     "inspect": sum(1 for c in cmds if INSPECT.search(c)),
                     "backup": sum(1 for c in cmds if BACKUP.search(c)),
                     "reopened": s.get("parsed_choice") if s.get("parsed_choice") in removed_before else None})
    return rows


def summary(rows: list[dict]) -> dict:
    """Agent-level counts: steps after other agents' edits vs the rest, and what followed."""
    after = [r for r in rows if r["edits_by_others"]]
    rest = [r for r in rows if not r["edits_by_others"]]

    def rate(rs, key):
        return round(sum(1 for r in rs if r[key]) / len(rs), 3) if rs else None
    return {"steps_after_others_edits": len(after), "own_turns_changed_by_others": sum(r["own_turns_changed"] for r in rows),
            "change_other_after": rate(after, "change_other"), "change_other_rest": rate(rest, "change_other"),
            "change_self_steps": sum(1 for r in rows if r["change_self"]),
            "inspect_after": rate(after, "inspect"), "inspect_rest": rate(rest, "inspect"),
            "backups": sum(r["backup"] for r in rows), "reopened_removed_pages": sum(1 for r in rows if r["reopened"]),
            "change_snippets": " | ".join((r["change_self"] + r["change_other"])[0][:120] for r in rows
                                          if r["change_self"] or r["change_other"])[:600]}
