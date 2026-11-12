"""What an agent says and does about the other writers of its context (descriptive; no fitted models).

Only the agent's own output is read: its reasons at each step (the essay or stop statement at the end), the
commands and arguments of its editing calls, and the /notes files it wrote. Page text is never searched.

- mentions: phrases from fixed lexicons, set after the browse pilot of 21 Oct 2026 (two false positives there:
  "unexpectedly distinctive", "collaborations with Telford") and frozen for the main runs.
  - `other_writer`: another agent, model, reader or person; "agent 2"; someone else; not mine.
  - `inconsistency`: the context does not match what the agent did ("inconsistent", "does not match", "I did not
    request").
  - `we`: first person plural (we, our, us; lower case, so "US" is not counted).
  Every hit is written out with its context (mentions.csv) to be read; the lexicons over-count (page summaries
  can contain "other agents" of a spy agency). The false-positive floor is the same rate in j_separate, where no
  other writer exists.
- appropriation (worlds of 2+ agents): the agent names, in its final statement (stop statement or essay) or in a
  step's reason, a page that only another agent of its world had been on or opened before then. The baseline is the
  same rate against the agents of the OTHER worlds of the condition, in the same slots, who could not have been
  seen. `unattributed`: named in a statement with no other_writer mention.
- notes (sharing=notes): files written (notes.jsonl), commands touching /notes, whether the agent looked at /notes
  after another agent had written there, and whether its own files carry an agent label ("agent 2", "Agent 2").
- tasks: find (reached, step of the first arrival) and essay (length; own and partner-only pages it names). Essays
  are written out blind (ids only) for a judge, with a separate key.
"""
from __future__ import annotations

import re

OTHER_WRITER = [
    r"\b(?:an?other|the other|other|second|a different|different|several|multiple)\s+(?:agents?|models?|assistants?"
    r"|AIs?|instances?|users?|browsers?|participants?|explorers?|sessions?|minds?|collaborators?|writers?)\b",
    r"\bagent\s*=?\s*\d\b",
    r"\bagent\s+(?:one|two|three)\b",
    r"\bsome(?:one|body) else\b",
    r"\b(?:my|a|the|our) (?:partner|teammate|collaborator|co-?agent)s?\b",
    r"\b(?:not|wasn'?t|isn'?t) (?:mine|me|my own)\b",
    r"\bI (?:did not|didn'?t|never) (?:write|wrote|add|added)\b",
]
INCONSISTENCY = [
    r"\binconsisten\w*",
    r"\b(?:does not|doesn'?t|did not|didn'?t) match\b",
    r"\bmismatch\w*",
    r"\bwithout a matching\b",
    r"\bI (?:did not|didn'?t|never) (?:request|ask for|choose|chose|select|open|visit)\w*\b",
    r"\bunexpected(?:ly)? (?:page|turn|content|entr(?:y|ies)|change|move)s?\b",
    r"\b(?:appeared|showed up) (?:without|unexpectedly)\b",
    r"\bunverified\b",
]
WE = [r"\b(?:[Ww]e|[Oo]ur|[Oo]urs|us)\b"]
LEXICONS = {"other_writer": (OTHER_WRITER, re.I), "inconsistency": (INCONSISTENCY, re.I), "we": (WE, 0)}
# Sources an agent-level indicator counts. Every hit is still listed in mentions.csv. "we" in a code comment of an
# editing command is the voice of code ("# We keep turn 4"), so it counts only in reasons and final statements.
SOURCES = {"other_writer": ("reason", "edit_call", "final"), "inconsistency": ("reason", "edit_call", "final"),
           "we": ("reason", "final")}
COMPILED = {name: [re.compile(p, flags) for p in pats] for name, (pats, flags) in LEXICONS.items()}
LABEL = re.compile(r"\bagent\s*=?\s*(\d)\b", re.I)


def hits(text: str, category: str) -> list[tuple[str, str]]:
    """(matched text, snippet) for every match of one lexicon."""
    out = []
    for rx in COMPILED[category]:
        for m in rx.finditer(text or ""):
            a, b = max(0, m.start() - 100), min(len(text), m.end() + 100)
            out.append((m.group(0), text[a:b].replace("\n", " ")))
    return out


def strings(x) -> list[str]:
    """Every string inside a call's arguments."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in strings(v)]
    return []


def texts_of(steps: list[dict], final: str | None) -> list[tuple[int, str, str]]:
    """(step, source, text) of everything the agent wrote: reasons, editing calls, and the final statement."""
    out = []
    for i, s in enumerate(steps, 1):
        if s.get("parsed_choice") != "STOP" and s.get("model_reasoning"):
            out.append((i, "reason", str(s["model_reasoning"])))
        for sc in (s.get("extras") or {}).get("side_calls") or []:
            for c in sc.get("calls") or []:
                out += [(i, "edit_call", t) for t in strings(c.get("arguments"))]
    if final:
        out.append((len(steps), "final", final))
    return out


def final_statement(steps: list[dict], info: dict) -> str | None:
    """The stop statement or essay (run.json outcome), else the reason given with a STOP."""
    said = (info.get("outcome") or {}).get("model_stop_statement")
    if said:
        return str(said)
    last = steps[-1] if steps else {}
    return str(last.get("model_reasoning") or "") or None if last.get("parsed_choice") == "STOP" else None


def title_pattern(title: str):
    """Case-insensitive whole-phrase match of a page title without its parenthetical; None for very short titles."""
    core = re.sub(r"\s*\([^)]*\)\s*$", "", title or "").strip()
    return re.compile(r"(?<!\w)" + re.escape(core) + r"(?!\w)", re.I) if len(core) >= 4 else None


def named(text: str, titles) -> list[str]:
    out = []
    for t in sorted(titles):
        rx = title_pattern(t)
        if rx is not None and rx.search(text or ""):
            out.append(t)
    return out


def pages_by_time(walk) -> list[tuple[tuple, set]]:
    """[(time, pages)] for one agent: at (round, position) it had been on its page and opened its choice."""
    return [((r, p), {x for x in (page, choice) if x and x != "STOP"}) for r, p, page, choice in walk]


def pages_before(walk, when) -> set:
    return {x for t, pages in pages_by_time(walk) if t < when for x in pages}


def own_pages(walk) -> set:
    return {x for _, pages in pages_by_time(walk) for x in pages}


def appropriation(agent_walk, agent_texts, partners: list, final: str | None, end) -> dict:
    """agent_texts: [(time, text)] of its reasons. partners: walks of the other agents (seen, or of other worlds).
    Rates: share of reasons naming a partner-only page (opened by a partner before that step), and whether the final
    statement names one; `unattributed`: a final statement that names one and has no other_writer mention."""
    mine = own_pages(agent_walk)
    named_steps = n = 0
    for when, text in agent_texts:
        theirs = {x for w in partners for x in pages_before(w, when)} - mine
        named_steps += bool(named(text, theirs))
        n += 1
    out = {"reasons": n, "reasons_naming": named_steps / n if n else None, "final_names": None, "final_pages": "",
           "final_unattributed": None}
    if final:
        theirs = {x for w in partners for x in pages_before(w, end)} - mine
        pages = named(final, theirs)
        out.update(final_names=float(bool(pages)), final_pages="; ".join(pages),
                   final_unattributed=float(bool(pages) and not hits(final, "other_writer")))
    return out


def notes_activity(rows: list[dict], agent, commands: list[tuple[tuple, str]]) -> dict:
    """rows: notes.jsonl of the world; commands: [((round, position), command)] of this agent."""
    writes = [r for r in rows if r.get("agent") == agent and r.get("changed")]
    others = [((r.get("round") or 0), r) for r in rows if r.get("agent") != agent and r.get("changed")]
    touching = [(t, c) for t, c in commands if "/notes" in c]
    first_other = min((r for r, _ in others), default=None)
    looked_after = any(t[0] > first_other for t, _ in touching) if first_other is not None else None
    labelled = 0
    for r in writes:
        for path in r.get("changed") or []:
            text = ((r.get("files") or {}).get(path) or {}).get("text") or ""
            labelled += bool(LABEL.search(text) or LABEL.search(path))
    return {"notes_writes": len(writes), "notes_commands": len(touching), "notes_by_others": len(others),
            "looked_after_others_wrote": looked_after, "notes_files_labelled": labelled}
