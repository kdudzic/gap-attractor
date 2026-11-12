"""The `tool` edit interface: one tool, `edit_context`, with three lists of edits that apply together or not at all.

  replace_turns  [{first, last, summary}]  turns first..last -> one turn holding `summary` (written by the editor)
  delete_turns   [{first, last}]           turns first..last removed
  replace_text   [{turn, old, new}]        inside one turn, an exact piece of its text that occurs once -> `new`
Turn numbers are those of the context before the call. Ranges must not overlap, and a replace_text turn must not be
inside one of them. Every field is required by the schema (models left ranges out when they were optional), and an
invalid call changes nothing and says what a valid one looks like. replace_text works inside a turn's text: it never
sees turn headers or the echoes of earlier edit calls. The gate decides whether an edit that makes the context
bigger is kept: `fit` only if it stays under the target, `shrink` never.
"""
from __future__ import annotations

from ctxfile.context import Turn, neutralize

MAX_EDITS = 50
OPS = ("replace_turns", "delete_turns", "replace_text")
_RANGE = {"first": {"type": "integer", "description": "First turn number."},
          "last": {"type": "integer", "description": "Last turn number (inclusive; equal to first for one turn)."}}
EXAMPLE = {
    "replace_turns": '{"replace_turns": [{"first": 1, "last": 6, "summary": "..."}]}',
    "delete_turns": '{"delete_turns": [{"first": 2, "last": 3}]}',
    "replace_text": '{"replace_text": [{"turn": 4, "old": "exact text in turn 4", "new": "shorter text"}]}',
}

TOOL = {"type": "function", "function": {
    "name": "edit_context",
    "description": ("Change your context: the numbered [[TURN i ...]] turns you see. Give one or more of the lists "
                    "replace_turns, delete_turns and replace_text. Turn numbers are the ones you see now, ranges must not "
                    "overlap, and all edits apply together or not at all (an invalid call changes nothing and says why). "
                    "The newest list of links cannot be changed until you have moved. You then see the result and the "
                    "new size, and can edit again or move."),
    "parameters": {"type": "object", "properties": {
        "replace_turns": {"type": "array", "description": "Put one text of your own in place of turns first..last.",
                          "items": {"type": "object", "properties": {**_RANGE, "summary": {"type": "string", "description": "The text that replaces those turns."}},
                                    "required": ["first", "last", "summary"]}},
        "delete_turns": {"type": "array", "description": "Delete turns first..last.",
                         "items": {"type": "object", "properties": dict(_RANGE), "required": ["first", "last"]}},
        "replace_text": {"type": "array", "description": "Inside one turn, replace an exact piece of its text.",
                         "items": {"type": "object", "properties": {
                             "turn": {"type": "integer", "description": "The turn number."},
                             "old": {"type": "string", "description": "Exact text from that turn's body (not its [[TURN]] line), occurring once."},
                             "new": {"type": "string", "description": "Replacement text."}},
                             "required": ["turn", "old", "new"]}}}}}}


class EditError(ValueError):
    pass


def _number(value, name, where, op):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise EditError(f"{where}: `{name}` must be a turn number (got {value!r}). A valid call: {EXAMPLE[op]}")


def normalise(arguments) -> list[dict]:
    """The call's edits as [{'op', ...}] in list order. Also accepts the older form {'edits': [{'op': ...}]}."""
    if not isinstance(arguments, dict):
        raise EditError("edit_context takes an object with replace_turns, delete_turns and/or replace_text lists")
    items = []
    if isinstance(arguments.get("edits"), list):
        for e in arguments["edits"]:
            if not isinstance(e, dict) or e.get("op") not in OPS:
                raise EditError(f"each edit needs one of {', '.join(OPS)}. A valid call: {EXAMPLE['replace_turns']}")
            items.append(dict(e))
    for op in OPS:
        given = arguments.get(op)
        if given is None:
            continue
        if not isinstance(given, list):
            given = [given]
        for e in given:
            if not isinstance(e, dict):
                raise EditError(f"each {op} item is an object. A valid call: {EXAMPLE[op]}")
            items.append({**e, "op": op})
    if not items:
        raise EditError("no edits given. A valid call: " + EXAMPLE["replace_turns"])
    if len(items) > MAX_EDITS:
        raise EditError(f"at most {MAX_EDITS} edits per call")
    return items


def check(arguments, n_turns: int) -> list[dict]:
    """Validate the whole call; returns normalised edits or raises EditError naming the first problem."""
    out, spans, texts = [], [], []
    for i, e in enumerate(normalise(arguments), 1):
        op = e["op"]
        where = f"{op} item {sum(1 for x in out if x['op'] == op) + 1}"
        if op == "replace_text":
            turn = _number(e.get("turn"), "turn", where, op)
            old, new = e.get("old"), e.get("new", "")
            if not isinstance(old, str) or not old:
                raise EditError(f"{where}: needs `old`, exact text from turn {turn}. A valid call: {EXAMPLE[op]}")
            if not isinstance(new, str):
                raise EditError(f"{where}: `new` must be text")
            if not 1 <= turn <= n_turns:
                raise EditError(f"{where}: turn {turn} does not exist (the context has turns 1..{n_turns})")
            out.append({"op": op, "turn": turn, "old": old, "new": new})
            texts.append((turn, where))
            continue
        for key in ("first", "last"):
            if e.get(key) is None:
                raise EditError(f"{where}: needs `first` and `last` (turn numbers). A valid call: {EXAMPLE[op]}")
        first, last = _number(e["first"], "first", where, op), _number(e["last"], "last", where, op)
        if not 1 <= first <= last <= n_turns:
            raise EditError(f"{where}: turns {first}..{last} do not exist (the context has turns 1..{n_turns})")
        for (a, b, w) in spans:
            if first <= b and a <= last:
                raise EditError(f"{where}: turns {first}..{last} overlap {w} (turns {a}..{b})")
        spans.append((first, last, where))
        item = {"op": op, "first": first, "last": last}
        if op == "replace_turns":
            summary = e.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise EditError(f"{where}: needs a non-empty `summary`. A valid call: {EXAMPLE[op]}")
            item["summary"] = summary
        out.append(item)
    for turn, where in texts:
        for (a, b, w) in spans:
            if a <= turn <= b:
                raise EditError(f"{where}: turn {turn} is also changed by {w}")
    return out


def apply(turns: list[Turn], arguments, editor: int | None, show_agents: bool = False) -> tuple[list[Turn], list[str]]:
    """New list of turns (the input is not changed) and one note per edit. Raises EditError if anything is invalid."""
    edits = check(arguments, len(turns))
    starts = {e["first"]: e for e in edits if e["op"] != "replace_text"}
    covered = {k for e in starts.values() for k in range(e["first"], e["last"] + 1)}
    texts: dict[int, list[dict]] = {}
    for e in edits:
        if e["op"] == "replace_text":
            texts.setdefault(e["turn"], []).append(e)
    work = []
    for k, t in enumerate(turns, 1):
        e = starts.get(k)
        if e is not None:
            gone = turns[e["first"] - 1:e["last"]]
            if e["op"] == "replace_turns":
                work.append(Turn(gone[0].role, neutralize(e["summary"].strip()), editor, "summary",
                                 origin=[{"id": g.id, "agent": g.agent, "kind": g.kind} for g in gone]))
            continue
        if k in covered:
            continue
        if k not in texts:
            work.append(Turn(t.role, t.text, t.agent, t.kind, t.id, t.call, t.reply_to, list(t.origin)))
            continue
        text = t.text
        for e in texts[k]:
            count = text.count(e["old"])
            if count != 1:
                raise EditError(f"replace_text in turn {k}: `old` occurs {'nowhere' if count == 0 else f'{count} times'} in that "
                                "turn's text" + (" (give a longer, unique piece)" if count else " (copy it exactly, without the [[TURN]] line)"))
            text = text.replace(e["old"], neutralize(e["new"]), 1)
        work.append(Turn(t.role, text, editor, "edited", origin=[{"id": t.id, "agent": t.agent, "kind": t.kind}]))
    notes = [f"{e['op']} {e['first']}..{e['last']}" if e["op"] != "replace_text" else f"replace_text in turn {e['turn']}" for e in edits]
    return work, notes


def describe(arguments) -> str:
    """A short description of a call, for the file (the full arguments travel in the tool call)."""
    try:
        edits = normalise(arguments)
    except EditError:
        return "edit_context: (invalid call)"
    show = lambda v: "?" if v is None else v               # an incomplete call: never a Python None
    return "edit_context: " + ", ".join(f"{e['op']} {show(e.get('first'))}..{show(e.get('last'))}" if e["op"] != "replace_text"
                                         else f"replace_text in turn {show(e.get('turn'))}" for e in edits)


def gate(mode: str, before: int, after: int, target: int) -> bool:
    """Keep an edit? fit: unless it makes the context bigger and over the target; shrink: only if smaller."""
    if mode == "shrink":
        return after < before
    return after <= before or after <= target
