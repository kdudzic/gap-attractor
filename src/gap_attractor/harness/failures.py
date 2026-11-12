"""Unusable replies, kept for their own sake (run folder: failures.jsonl, one line per unusable reply).

A reply is unusable when the harness cannot act on it: no readable choice (unparseable, refusal), a title that is not
on the menu (out-of-range), STOP when stopping is off, or editing until the memory's turns ran out. The step itself
records only its first error and how it was handled; this log keeps every attempt, what the model wrote, and where
the title it named could have come from:

  near_miss           a title on the menu, written differently (case, punctuation, a missing or extra qualifier); a
                      title that is an article of its own is never a near miss, even if it contains a listed one
  earlier_menu        a title offered on an earlier page of this run, not on the current one
  others_menu         a title on a link list the memory showed from elsewhere (another agent's page in a shared
                      context: Memory.other_menus)
  linked_not_offered  linked from the current page, but not offered (the link target is not an article in the dump,
                      or was filtered out)
  mentioned_on_page   named in the current page's text without a link
  in_dump             an article that exists in the dump (directly or by redirect) but was never shown: a page the
                      model knows of and might have searched for
  not_in_dump         no such article
  no_title            no title at all (unparseable, refusal, editing turns used up)
  stop_not_allowed    STOP while stopping is off
  unknown             a named title not traced to any menu or page, when no backend was given to look it up (e.g.
                      scripts/failures.py --reconstruct without --wikipedia)
`in_dump` and `resolved_title` are filled for every named title, whatever its category.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

CATEGORIES = ("near_miss", "earlier_menu", "others_menu", "linked_not_offered", "mentioned_on_page", "in_dump", "not_in_dump",
              "no_title", "stop_not_allowed")


def norm(title: str) -> str:
    """Comparable form of a title: case, accents, punctuation, spacing and underscores ignored."""
    text = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode().lower().replace("_", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def near(attempted: str, titles, loose: bool = True) -> str | None:
    """The listed title the attempt most plausibly meant: the same normalised form or, with loose, one containing the
    other as whole words or the same title without its "(qualifier)". classify() keeps a loose match only when the
    named title is not an article of its own (e.g. "LibreOffice Calc" is one, so it is not a near miss of
    "LibreOffice")."""
    a = norm(attempted)
    if not a:
        return None
    exact = [t for t in titles if norm(t) == a]
    if exact or not loose:
        return exact[0] if exact else None
    bare = norm(re.sub(r"\s*\([^)]*\)\s*$", "", attempted))
    for t in titles:
        n = norm(t)
        if n and (f" {n} " in f" {a} " or f" {a} " in f" {n} " or (bare and norm(re.sub(r"\s*\([^)]*\)\s*$", "", t)) == bare)):
            return t
    return None


def classify(attempted, invalid_type, menu, earlier, page, backend, others=()) -> dict:
    """Category and evidence for one unusable reply. `earlier`: [(step_index, [titles])] of the run's earlier menus."""
    out = {"category": None, "matched": None, "in_dump": None, "resolved_title": None}
    if not attempted:
        out["category"] = "no_title"
        return out
    if str(attempted).strip().lower() == "stop":
        out["category"] = "stop_not_allowed"
        return out
    resolved = None
    try:
        resolved = backend.load_page(attempted) if backend is not None else None
    except Exception:
        resolved = None
    if backend is not None:
        out["in_dump"], out["resolved_title"] = resolved is not None, (resolved.title if resolved is not None else None)
    def same(hit):          # a loose match counts unless the named title is a different article of its own
        return near(attempted, [hit], loose=False) is not None or resolved is None or norm(resolved.title) == norm(hit)

    via = near(resolved.title, menu, loose=False) if resolved is not None else None    # a redirect onto a listed page
    hit = via or near(attempted, menu)
    if hit is not None and (via or same(hit)):
        out.update(category="near_miss", matched={"title": hit, **({"via_redirect": True} if via and norm(via) != norm(attempted) else {})})
        return out
    for step_index, titles in reversed(earlier):
        via = near(resolved.title, titles, loose=False) if resolved is not None else None
        hit = via or (attempted if attempted in titles else near(attempted, titles))
        if hit is not None and (via or same(hit)):
            out.update(category="earlier_menu", matched={"title": hit, "step_index": step_index})
            return out
    for source, titles in others:
        via = near(resolved.title, titles, loose=False) if resolved is not None else None
        hit = via or (attempted if attempted in titles else near(attempted, titles))
        if hit is not None and (via or same(hit)):
            out.update(category="others_menu", matched={"title": hit, "source": source})
            return out
    if page is not None:
        targets = [x for l in (page.links or []) for x in (getattr(l, "destination", None), getattr(l, "anchor_text", None)) if x]
        hit = near(attempted, targets, loose=False)
        if hit is not None:
            out.update(category="linked_not_offered", matched={"link": hit})
            return out
        if norm(attempted) and f" {norm(attempted)} " in f" {norm(page.wikitext or '')} ":
            out["category"] = "mentioned_on_page"
            return out
    out["category"] = "in_dump" if out["in_dump"] else ("not_in_dump" if out["in_dump"] is False else "unknown")
    return out


def write(run, backend, page, step, notes) -> None:
    """Append this step's unusable replies to <run>/failures.jsonl, with how the step ended."""
    earlier = [(s["step_index"], [c["title"] for c in s.get("choice_set") or []] + [r["title"] for r in (s.get("search") or {}).get("results") or []])
               for s in run.steps]
    cfg = run.config
    try:
        others = run.memory.other_menus() if run.memory is not None else []
    except Exception:
        others = []
    with (Path(run.dir) / "failures.jsonl").open("a", encoding="utf-8") as f:
        for n in notes:
            row = {"run_id": run.run_id, "step_index": step["step_index"], "attempt": n["attempt"], "invalid_type": n["invalid_type"],
                   "attempted": n["attempted"], **classify(n["attempted"], n["invalid_type"], n["menu"], earlier, page, backend, others),
                   "current_page": page.title if page is not None else None, "menu_size": len(n["menu"]),
                   "reply_text": (n["reply"].get("content") or "")[:2000], "tool_calls": n["reply"].get("tool_calls") or [], "cut_off": n.get("cut_off", False),
                   "step_handling": step["invalid_action"].get("harness_handling"), "step_choice": step.get("parsed_choice"),
                   "model": cfg.get("provider", {}).get("model"), "memory": cfg.get("experiment", {}).get("memory")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
