"""The context file: an agent's (or a whole world's) live context as editable text.

File format, after Context Language Models (Shao et al., 2026, arXiv 2609.37725; repo
facebookresearch/context-language-models): every turn is a block

    [[TURN 3 role=user agent=2]]
    <text>

(CLM writes [[CTX_TURN ...]]; ours says only TURN, so the format does not point a model to the paper. The parser
also reads the CLM spelling, for files written before 16 Oct 2026.)

numbered from 1 in file order. `agent=` appears only in a joint (multi-agent) file with attribution on: the author's
number (attribution=labelled), or `agent=you` / `agent=other` (attribution=other: other agents all look the same).
The code here is our own (MIT); see NOTICE for what the format and protocol owe to CLM.

Parsing text back (after a bash edit or a text replacement):
- a header is a whole line `[[TURN <n> role=<user|assistant>[ agent=<k|you|other>]]]`; anything else stays text;
- text before the first header is a user turn; empty turns are dropped; turns are never merged.
Turn bodies never contain a header line: text coming from a model is passed through `neutralize`, so a summary
cannot smuggle in a turn of another agent.

Turns carry hidden bookkeeping (never in the file): a stable id, the kind of turn, the tool call that made it, and
for summaries or rewritten turns the ids they came from. After a parse, `reconcile` gives unchanged turns back
their id and metadata (matched by role and text), so the logs can say which turn survived, and whose. Who wrote a
turn always comes from that record, never from a header's agent=: a turn whose text was changed belongs to the one
who changed it, whatever its header says.
"""
from __future__ import annotations

import difflib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROLES = ("user", "assistant")
ATTRIBUTIONS = ("labelled", "other", "none")   # agent=k | agent=you / agent=other | no agent
_OPEN = "[[TURN "
_OPENS = (_OPEN, "[[CTX_TURN ")      # ours; the CLM spelling (older files)


@dataclass
class Turn:
    role: str                    # user | assistant
    text: str                    # the body as it is in the file
    agent: int | None = None     # author (joint files; in a single agent's file None)
    kind: str = ""               # observation | links | notice | move | attempt | feedback | note | call | result | summary | edited
                                 # | goal | search | results (the opening search of variant b_search_first)
    id: int = 0                  # stable within a file; assigned by ContextFile
    call: dict | None = None     # a turn made by a tool call: {name, arguments, content, result}; sent as a tool call
    reply_to: int | None = None  # a tool result turn: the id of its call turn
    origin: list = field(default_factory=list)   # summary/edited: [{id, agent, kind}] of the turns it replaced


def neutralize(text: str) -> str:
    """Make header-looking lines inert (a model's summary must not create turns)."""
    lines = str(text).split("\n")
    return "\n".join(("[ " + l.lstrip()[1:]) if l.lstrip().startswith(_OPENS) else l for l in lines)


def label(turn: Turn, attribution: str, viewer: int | None = None) -> str | None:
    """How a turn's author is shown to `viewer`: its number (labelled), you/other (other), or nothing (none, or a
    turn of the session itself)."""
    if turn.agent is None or attribution == "none":
        return None
    if attribution == "other":
        return "you" if turn.agent == viewer else "other"
    return str(turn.agent)


def header(n: int, turn: Turn, show_agents, viewer: int | None = None) -> str:
    """`show_agents`: an attribution (labelled | other | none), or a bool (True = labelled; older callers)."""
    attribution = show_agents if isinstance(show_agents, str) else ("labelled" if show_agents else "none")
    who = label(turn, attribution, viewer)
    return f"{_OPEN}{n} role={turn.role}{f' agent={who}' if who else ''}]]"


def render(turns: list[Turn], show_agents=False, viewer: int | None = None) -> str:
    return "\n\n".join(f"{header(i, t, show_agents, viewer)}\n{t.text.strip()}" for i, t in enumerate(turns, 1)) + ("\n" if turns else "")


def read_header(line: str) -> dict | None:
    """{'n', 'role', 'agent'} for a header line, or None for any other line."""
    s = line.strip()
    opened = next((o for o in _OPENS if s.startswith(o)), None)
    if opened is None or not s.endswith("]]"):
        return None
    words = s[len(opened):-2].split()
    if not words or not words[0].isdigit():
        return None
    out = {"n": int(words[0]), "role": None, "agent": None}
    for word in words[1:]:
        key, eq, value = word.partition("=")
        if not eq or key not in ("role", "agent") or out[key] is not None:
            return None
        out[key] = value
    if out["role"] not in ROLES or (out["agent"] is not None and not (out["agent"].isdigit() or out["agent"] in ("you", "other"))):
        return None
    if out["agent"] is not None and out["agent"].isdigit():
        out["agent"] = int(out["agent"])
    return out


def parse(text: str) -> list[dict]:
    """File text -> [{'n', 'role', 'agent', 'text'}] in file order (n None for text before the first header)."""
    blocks, current, body = [], {"n": None, "role": "user", "agent": None}, []
    for line in (text or "").split("\n"):
        h = read_header(line)
        if h is None:
            body.append(line)
            continue
        blocks.append(dict(current, text="\n".join(body).strip()))
        current, body = h, []
    blocks.append(dict(current, text="\n".join(body).strip()))
    return [b for b in blocks if b["text"]]


def reconcile(old: list[Turn], blocks: list[dict], editor: int | None, labelled: bool) -> tuple[list[Turn], list[str]]:
    """Turns for parsed blocks. Blocks are aligned with the old turns by (role, text) in order (difflib), so a block
    that IS an old turn keeps its id and metadata even when duplicates exist (a revisited page). Any other block is
    new text by `editor`.
    labelled=False (what the memory uses since 21 Oct 2026): headers' agent= is ignored; authors come from the record
    (an unchanged turn keeps its author, changed or new text is the editor's), so no header can forge anything and an
    agent may shorten another agent's turn in place, keeping its header line, as it can with edit_context.
    labelled=True (strict, the earlier rule): a block may not claim another agent; problems are returned and the
    caller rejects the edit. It refused that natural edit (shortening a partner's turn, header kept) as forgery."""
    keys_old = [(t.role, t.text.strip()) for t in old]
    keys_new = [(b["role"], b["text"]) for b in blocks]
    match = {}
    for a, b, size in difflib.SequenceMatcher(None, keys_old, keys_new, autojunk=False).get_matching_blocks():
        for k in range(size):
            match[b + k] = a + k
    out, problems = [], []
    for j, b in enumerate(blocks):
        if j in match:
            t = old[match[j]]
            if labelled and b["agent"] is not None and b["agent"] != t.agent:
                problems.append(f"turn {b['n']}: its author is agent {t.agent}; turns cannot be relabelled")
            out.append(Turn(t.role, t.text, t.agent, t.kind, t.id, t.call, t.reply_to, list(t.origin)))
            continue
        if labelled and b["agent"] is not None and b["agent"] != editor:
            problems.append(f"turn {b['n']}: you can only write turns as agent {editor}")
        at = old[b["n"] - 1] if b["n"] and 0 < b["n"] <= len(old) else None
        origin = [{"id": at.id, "agent": at.agent, "kind": at.kind}] if at is not None else []
        out.append(Turn(b["role"], neutralize(b["text"]), editor, "edited", 0, None, None, origin))
    return out, problems


@dataclass
class Version:
    """One change of the file: who, when, what, sizes, and which turns (by id) went and came. Self-contained: with
    `turns_set` (the full text of every turn added, or changed under the same id) and `ids_after` (the order), the
    file after any version can be rebuilt from the log alone (replay_context.py)."""
    n: int
    agent: int | None
    step: int | None
    op: str                      # append | edit | evict | shrink (the harness shortens a finished step's echoes)
    chars_before: int
    chars_after: int
    diff: str = ""
    removed: list = field(default_factory=list)  # [{id, agent, kind}]
    added: list = field(default_factory=list)    # [{id, agent, kind}]
    ids_after: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    turns_set: list = field(default_factory=list)   # [{id, role, agent, kind, text}]: added, or text/role/author changed
    show_agents: bool = False                       # how the file text is rendered (agent= in headers)


class ContextFile:
    """Turns + version history; one per agent (single) or one per world (joint). The version log is appended to
    `versions_path` as it happens; `mirror` (a host copy for people to read, never mounted into a sandbox) is
    replaced atomically after every change."""

    def __init__(self, show_agents: bool = False, mirror: Path | None = None, versions_path: Path | None = None):
        self.turns: list[Turn] = []
        self.show_agents = show_agents
        self.mirror_path = Path(mirror) if mirror else None
        self.versions_path = Path(versions_path) if versions_path else None
        self.versions: list[Version] = []
        self.next_id = 1

    # --- reading -------------------------------------------------------------------------------------------
    def text(self) -> str:
        return render(self.turns, self.show_agents)

    def index_of(self, turn_id: int) -> int | None:
        return next((i for i, t in enumerate(self.turns, 1) if t.id == turn_id), None)

    def view(self, viewer: int | None, attribution: str) -> str:
        """The file as one agent sees it (bash: /ctx/context.txt): agent= as the attribution shows it to this viewer."""
        return render(self.turns, attribution, viewer)

    def messages(self, viewer: int | None = None, labelled=True, others_as: str = "user", headers: bool = True,
                 turns: list[Turn] | None = None) -> list[dict]:
        """The turns (default: the file's) as chat messages for one agent.
        - labelled: an attribution (labelled | other | none), or a bool (True = labelled).
        - headers: each message starts with its [[TURN i …]] line (as in the file), so turns can be named by number.
        - A turn made by a tool call (a move, an editing call) is sent as a real tool call with its result, while it is
          unedited and the viewer may see it in its own role; otherwise as text.
        - A move's result is the page it opened (its text and link list, when they follow it unedited): moves are
          atomic, so nothing comes between a move and its page.
        - Joint files: another agent's turn is sent in its own role (others_as=own: a reply is an assistant message,
          with its tool calls) or as a user message (user); without headers it starts with "[agent k]" (labelled) or
          "[another agent]" (other).
        Consecutive plain messages of one role are merged."""
        attribution = labelled if isinstance(labelled, str) else ("labelled" if labelled else "none")
        out: list[dict] = []
        turns, skip = (self.turns if turns is None else turns), set()

        def plain(role, content):
            if out and out[-1]["role"] == role and "tool_calls" not in out[-1] and role != "tool":
                out[-1]["content"] += "\n\n" + content
            else:
                out.append({"role": role, "content": content})

        for i, t in enumerate(turns):
            if i in skip:
                continue
            own = viewer is None or t.agent is None or t.agent == viewer
            head = header(i + 1, t, attribution, viewer) + "\n" if headers else ""
            who = label(t, attribution, viewer)
            tag = "" if headers or own or who is None else ("[another agent]\n" if who == "other" else f"[agent {who}]\n")
            role = t.role if (own or others_as == "own") else "user"
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            paired = nxt is not None and nxt.kind == "result" and nxt.reply_to == t.id
            if t.call and role == "assistant" and (t.kind == "move" or (t.kind in ("call", "attempt") and paired)):
                content = (head + tag + str(t.call.get("content") or "")).strip() or None
                out.append({"role": "assistant", "content": content, "tool_calls": [{"id": f"t{t.id}", "type": "function",
                            "function": {"name": t.call["name"], "arguments": json.dumps(t.call.get("arguments") or {}, ensure_ascii=False)}}]})
                if t.kind == "move":
                    # the page the move opened is its result (atomic moves): its text and link list, if they follow
                    # unedited, are sent as the tool result rather than as separate messages
                    page = []
                    for j in (i + 1, i + 2):
                        nt = turns[j] if j < len(turns) else None
                        want = "observation" if j == i + 1 else "links"
                        if nt is None or nt.kind != want or nt.agent != t.agent or (j == i + 2 and not page):
                            break
                        page.append(j)
                    if page:
                        content = "\n\n".join((header(j + 1, turns[j], attribution, viewer) + "\n" if headers else "") + tag
                                              + turns[j].text.strip() for j in page)
                        skip.update(page)
                    else:
                        content = str(t.call.get("result") or "ok")
                    out.append({"role": "tool", "tool_call_id": f"t{t.id}", "content": content})
                else:
                    nhead = header(i + 2, nxt, attribution, viewer) + "\n" if headers else ""
                    out.append({"role": "tool", "tool_call_id": f"t{t.id}", "content": nhead + nxt.text.strip()})
                    skip.add(i + 1)
                continue
            plain(role, head + tag + t.text.strip())
        if out and out[0]["role"] != "user":   # some providers refuse a conversation that opens with the assistant
            out.insert(0, {"role": "user", "content": "[context]"})
        return out

    # --- changing ------------------------------------------------------------------------------------------
    def assign(self, turn: Turn) -> Turn:
        if not turn.id:
            turn.id, self.next_id = self.next_id, self.next_id + 1
        return turn

    def append(self, turn: Turn, step: int | None = None, op: str = "append") -> Turn:
        before = len(self.text())
        self.turns.append(self.assign(turn))
        self._record(turn.agent, step, op, before, "", [], [turn], {"kind": turn.kind, "role": turn.role}, [turn])
        return turn

    def replace_all(self, turns: list[Turn], agent: int | None, step: int | None, op: str = "edit", detail=None) -> dict:
        """Swap in a new list of turns (ids kept where present, new ones assigned). Returns {removed, added}."""
        old_text, old = self.text(), {t.id: t for t in self.turns}
        self.turns = [self.assign(t) for t in turns]
        new = {t.id for t in self.turns}
        removed = [t for i, t in old.items() if i not in new]
        added = [t for t in self.turns if t.id not in old]
        changed = [t for t in self.turns if t.id in old and (old[t.id].text, old[t.id].role, old[t.id].agent) != (t.text, t.role, t.agent)]
        diff = "".join(difflib.unified_diff(old_text.splitlines(True), self.text().splitlines(True), "before", "after", n=1))
        self._record(agent, step, op, len(old_text), diff, removed, added, detail or {}, added + changed)
        return {"removed": removed, "added": added}

    def _record(self, agent, step, op, before, diff, removed, added, detail, turns_set=()) -> None:
        # an edited or summary turn also names the turns it replaced (origin ids, from 22 Oct 2026), so the log tells
        # a turn shortened in place or summarised from one deleted outright
        brief = lambda ts: [{"id": t.id, "agent": t.agent, "kind": t.kind, **({"origin": [o["id"] for o in t.origin]} if t.origin else {})}
                            for t in ts]
        full = [{"id": t.id, "role": t.role, "agent": t.agent, "kind": t.kind, "text": t.text} for t in turns_set]
        v = Version(len(self.versions) + 1, agent, step, op, before, len(self.text()), diff, brief(removed), brief(added),
                    [t.id for t in self.turns], detail, full, bool(self.show_agents))
        self.versions.append(v)
        if self.versions_path is not None:
            self.versions_path.parent.mkdir(parents=True, exist_ok=True)
            with self.versions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(v), ensure_ascii=False) + "\n")
        self.mirror()

    def mirror(self) -> None:
        if self.mirror_path is None:
            return
        self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.mirror_path.with_name(self.mirror_path.name + ".tmp")
        tmp.write_text(self.text(), encoding="utf-8")
        os.replace(tmp, self.mirror_path)      # replaces a link instead of writing through it
