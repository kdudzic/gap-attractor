"""Which turns an agent cuts from its context, and how hard: where the cut turns rank on fixed scores (no fitted model).

The context-file counterpart of the click hypotheses (src/gap_attractor/analysis/hypotheses.py; plan §7 in
team analysis notes, not in this repository). Agents edit freely (bash rewrites the file any way it likes);
the harness re-reads the file into turns, and any turn whose text changed, by one character or entirely, comes back
under a new id as kind "edited" by the editor (kind "summary" for the edit tool's replace_turns). So at every
applied edit (version op "edit") the menu is the file just before it, and each menu turn has a fate:
  untouched   its id is still there;
  rewritten   an edited turn names it as its only origin: shortened, reworded or expanded in place;
  merged      a new turn names it among several origins: summarised together with other turns;
  deleted     its id is gone and no new turn names it.
Origins are logged from 22 Oct 2026 (added turns carry `origin`). Older logs are aligned by position: where a run of
old ids gave way to as many new turns, they pair one to one; to a single new turn, they merged; any other
replacement leaves its turns unlinked, and the edit counts for `touched` only. A bash rewrite that dropped a turn's
[[TURN]] line cannot be linked either: its old turn reads as deleted and its text as new.

cut: the share of a turn's characters the edit took away (1 deleted, 0 untouched; rewritten: 1 - new/old, at least
0; merged: 1 - new / all its origins), from the turn texts (turns_set, logged from 21 Oct 2026).
Turns the agent could not remove are left out of the menu: its current link list (protected until it moves), goal
turns, and the editing calls and results of the current step.

Per edit, score s (declared below, higher = "more of it") and outcome w (touched: 1 for every changed turn; deleted:
1 for deleted turns; cut: the cut share), r = sum w_i p_i / sum w_i, with p_i the turn's mid-rank percentile on the
menu by s, (#lower + (#tied - 1) / 2) / (n - 1). If the outcome is blind to s (the same amounts handed to the turns
at random), the expected r is exactly 0.5 whatever the menu. Edits whose menu does not vary in s say nothing about
it and are left out (counted), e.g. `others` in a file with no other agent's turns.
"""
from __future__ import annotations

import difflib
import hashlib

TRAIL = ("call", "result")
OUTCOMES = ("touched", "deleted", "cut")


def _placebo(turn_id: int, n: int) -> float:
    return int(hashlib.sha1(f"{turn_id}:{n}".encode()).hexdigest()[:8], 16) / 16 ** 8


# name -> (role, function(meta, editor, version n) -> score or None). meta: agent, kind, first (version number the id
# first appeared), length (characters, None if unknown).
SCORES = {
    "others": ("primary", lambda m, ed, n: float(m["agent"] is not None and m["agent"] != ed)),
    "older": ("exploratory", lambda m, ed, n: -float(m["first"])),
    "longer": ("exploratory", lambda m, ed, n: None if m["length"] is None else float(m["length"])),
    "page": ("exploratory", lambda m, ed, n: float(m["kind"] == "observation")),
    "edit_trail": ("exploratory", lambda m, ed, n: float(m["kind"] in TRAIL)),
    "notice": ("exploratory", lambda m, ed, n: float(m["kind"] == "notice")),
    "summary": ("exploratory", lambda m, ed, n: float(m["kind"] in ("edited", "summary"))),
    "placebo": ("placebo", lambda m, ed, n: _placebo(m["id"], n)),
}


def mid_rank(values: list[float], i: int) -> float:
    x, n = values[i], len(values)
    lower = sum(v < x for v in values)
    tied = sum(v == x for v in values)
    return (lower + (tied - 1) / 2) / (n - 1)


def fates(before: list[int], after: list[int], added: list[dict]) -> dict[int, tuple[str, list[int]]]:
    """Removed id -> (fate, new ids it went into). Logged origins when the version has them, else aligned."""
    removed = [i for i in before if i not in set(after)]
    if any("origin" in t for t in added) or not added:
        into: dict[int, list[int]] = {}
        sizes = {t["id"]: len(t.get("origin") or []) for t in added}
        for t in added:
            for o in t.get("origin") or []:
                into.setdefault(o, []).append(t["id"])
        return {i: (("merged" if any(sizes[j] > 1 for j in into[i]) else "rewritten"), into[i]) if i in into
                else ("deleted", []) for i in removed}
    new = {t["id"] for t in added}
    out = {}
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        olds, news = before[i1:i2], [j for j in after[j1:j2] if j in new]
        if op == "delete" or (op == "replace" and not news):
            out.update({i: ("deleted", []) for i in olds})
        elif op == "replace" and len(olds) == len(news):
            out.update({i: ("rewritten", [j]) for i, j in zip(olds, news)})
        elif op == "replace" and len(news) == 1:
            out.update({i: ("merged", news) for i in olds})
        elif op == "replace":
            out.update({i: ("unlinked", []) for i in olds})
    return {i: out.get(i, ("unlinked", [])) for i in removed}


def events(versions: list[dict]) -> list[dict]:
    """One row per applied edit that changed turns: editor, step, menu size, what happened to the turns, and r per
    score and outcome (`<score>` = touched, `<score>_deleted`, `<score>_cut`; None where it cannot be computed)."""
    meta: dict[int, dict] = {}
    ids: list[int] = []
    out = []
    for v in versions:
        before = list(ids)
        for t in v.get("added") or []:
            meta.setdefault(t["id"], {"id": t["id"], "agent": t["agent"], "kind": t["kind"], "first": v["n"], "length": None})
        for t in v.get("turns_set") or []:
            if t["id"] in meta:
                meta[t["id"]]["length"] = len(t["text"])
        ids = list(v.get("ids_after") or [])
        if v.get("op") != "edit":
            continue
        editor, step = v.get("agent"), v.get("step")
        fate = fates(before, ids, v.get("added") or [])
        if not fate:
            continue
        mine = [i for i in before if meta[i]["agent"] == editor]
        links = max((i for i in mine if meta[i]["kind"] == "links"), default=None)
        in_progress = {i for i in mine if meta[i]["kind"] in TRAIL and step_of(meta[i], versions) == step}
        menu = [i for i in before if i != links and meta[i]["kind"] != "goal" and i not in in_progress]
        if len(menu) < 2 or not any(i in fate for i in menu):
            continue
        weights = {"touched": [float(i in fate) for i in menu],
                   "deleted": [float(fate.get(i, ("",))[0] == "deleted") for i in menu],
                   "cut": [cut_share(i, fate, meta) for i in menu]}
        if any(f == "unlinked" for f, _ in fate.values()):
            weights["deleted"] = weights["cut"] = None
        elif any(w is None for w in weights["cut"]):
            weights["cut"] = None
        others = lambda f: sum(1 for i in menu if fate.get(i, ("",))[0] == f and meta[i]["agent"] not in (None, editor))
        row = {"version": v["n"], "editor": editor, "step": step, "menu": len(menu), "touched": sum(i in fate for i in menu),
               **{f: sum(1 for i in menu if fate.get(i, ("",))[0] == f) for f in ("deleted", "rewritten", "merged", "unlinked")},
               "touched_others": sum(1 for i in menu if i in fate and meta[i]["agent"] not in (None, editor)),
               "deleted_others": others("deleted"),
               "taken_over_others": others("rewritten") + others("merged"),   # their text now stands as the editor's
               "chars_cut": round(sum((meta[i]["length"] or 0) * (w or 0) for i, w in zip(menu, weights["cut"] or []))),
               "chars_cut_others": round(sum((meta[i]["length"] or 0) * (w or 0) for i, w in zip(menu, weights["cut"] or [])
                                             if meta[i]["agent"] not in (None, editor)))}
        for name, (_, fn) in SCORES.items():
            values = [fn(meta[i], editor, v["n"]) for i in menu]
            varies = not any(x is None for x in values) and len(set(values)) > 1
            for outcome in OUTCOMES:
                w = weights[outcome]
                key = name if outcome == "touched" else f"{name}_{outcome}"
                row[key] = (sum(wi * mid_rank(values, k) for k, wi in enumerate(w)) / sum(w)
                            if varies and w is not None and sum(w) > 0 else None)
        out.append(row)
    return out


def cut_share(i: int, fate: dict, meta: dict) -> float | None:
    """The share of turn i's characters this edit took away (None when a length is unknown)."""
    if i not in fate:
        return 0.0
    kind, into = fate[i]
    if kind == "deleted":
        return 1.0
    if kind == "unlinked" or meta[i]["length"] is None or any(meta[j]["length"] is None for j in into):
        return None
    new = sum(meta[j]["length"] for j in into)
    if kind == "merged":
        sources = [o for o, (f, js) in fate.items() if f == "merged" and set(js) & set(into)]
        if any(meta[o]["length"] is None for o in sources):
            return None
        old = sum(meta[o]["length"] for o in sources)
    else:
        old = meta[i]["length"]
    return max(0.0, 1 - new / old) if old else 0.0


def step_of(m: dict, versions: list[dict]) -> int | None:
    """The step in which a turn was added (its first version's step)."""
    return versions[m["first"] - 1].get("step") if 0 < m["first"] <= len(versions) else None
