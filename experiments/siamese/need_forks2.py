"""Stated-need replays, second batch: what in a stated need changes the edit, through which channel, and for how long.

    uv run python experiments/siamese/need_forks2.py <stage dir> [...] --out <dir> --dry-run
    uv run python experiments/siamese/need_forks2.py <stage dir> [...] --out <dir> --fake            (no API: mechanics)
    uv run --env-file .env python experiments/siamese/need_forks2.py <stage dir> [...] --out <dir> \
        --arms actual,s_N,s_DR --samples 2 --calls 12 --max-cost 3

Events are need_forks.py's: the call at which an agent's edit destroyed a value V that a not-yet-issued task needed
(state_chain worlds, saved requests; each rebuilds exactly). Every arm is the saved request with ONE addition, and the
same additions are used for every event (whose need it is only changes the owner word), so channel and content are
crossed within events:

  channel   s_   one more line in the editing agent's private status (the harness's voice), kept in the status for
                 the whole replay, as the world keeps its standing lines
            pa_  one more turn from the other agent, at the end, as an assistant-role message (how these worlds show
                 the partner)
            pu_  the same turn as a user-role message
            e_   (DR only) a notice in the context right after the turn that first held V: the need stated when the
                 value appeared, many turns before the edit
            l_   (DR only) the same notice at the end of the context (so e_ and l_ differ in position only)
  content   N    nothing about the context ("Each of your answers is checked when you submit it." / "a note from me...")
            PT   pointer only: names turn i, says nothing else
            D    directive only: "Keep turn i."
            R    reason only: "A later task of <owner> will ask for what turn i holds."
            DR   directive and reason
            G    the need without the turn: "A later task of <owner> will ask for something that is in the context now."
            DC   decoy: DR about a turn d that holds nothing any later task needs
            CA   caution: "Be careful with the context: do not remove anything that is still needed."
            I    instruction: "Keep turn i. Make room by removing other turns." (can they do it when told how)
            J    joint score: "Your score is the number of tasks that you and agent j get right, together."
            CP   (pu_ only) the partner says it has saved what turn i holds, so it need not be kept
  questions (text answer, one call, no edit): q_rule gives the rule and asks which earlier answers are in the context;
            q_open asks whether anything may be asked for later, uncapped. With --points N they are asked at calls
            where a needed value was in the context and no loss followed (N per model), not only at loss events.
No addition names V. `actual` is the saved request unchanged. Compare a content with N in the SAME channel (the N
sentences differ by channel), and split by whose need it is (in pa_/pu_ the other agent speaks about the editor's own
need for events of kind self, which the setup makes implausible).

The replay continues as the world would, for at most --calls calls (1 = the first reply only):
- the agent's current task's files are in /work/env/<slug>/ (rebuilt from the task generator; whatever else the agent
  had in /work is not restored);
- a bash call runs on the context file; the file read back is reconciled with the turns; an edit that would grow the
  context past the limit is not applied; the call and its result (the output in full, as these worlds showed it) are
  added; the agent has as many bash calls left as it had in the world at that call, and then is told to submit;
- a submit over the limit is held with the world's words and gives the bash calls back; three holds in a row (without
  the context getting smaller in between) end the session; a submit under the limit is carried out and ends the replay
  (the task is not graded);
- a reply without a tool call is added as a turn and the agent is asked again ("Call one tool: submit or bash.").
- three replies in a row that do nothing (no tool, a refused bash call, an unknown tool) end the replay.
Not taken from the world (the same in every arm): holds made earlier in the same step do not count towards the three
per step (9 of 88 requests had one); the 'held in a row' count is read from the editing agent's latest result only
(wrong for 1 of 88); the step's cap of 14 calls; the fill line after the first request; the world's running
re-calibration of characters per token; the notice after a reply cut at the length limit.
Outcomes are taken after the first call and at the end. They come from which of the ORIGINAL turns are still there
unchanged (named turn, any turn that held V, the decoy) and from whether V is anywhere in the context or was copied
to /work, /tmp, /dev/shm or beside the context file. The first request is exact; later ones follow the world's rules
with the size from the saved call's characters per token.
--interface tool: every message starts with its turn number and a tool remove_turns(turns) is offered next to bash
(is wiping a consequence of editing a flat file by script?). --system: need_forks.system_variant, or env (one line
saying that the other agent can change the context at any time and that /work keeps its files).
--as-model (with --reasoning): a cross-model replay (the context was written by another model).
Chains are written to <out>/chains.jsonl as they finish and loaded again on a restart; events that share a request
share the chain (same first request) and are scored each for its own value: they are not independent (cluster by
request). Nothing is written into a world. Output: rows.jsonl, chains.jsonl, config.json (one configuration a folder).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import shlex
import signal
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_first = [str(HERE), str(REPO / "experiments" / "context_file"), str(REPO / "src")]
sys.path[:] = _first + [p for p in sys.path if p not in _first]

from ctxfile.context import Turn, neutralize, parse, reconcile, render  # noqa: E402
from events import load, request_name, worlds  # noqa: E402
from forks import History, body_of, provider_for, restatus, sample  # noqa: E402
from need_forks import ASK_RULE, SYSTEMS, WORK_AND, events_of, system_variant  # noqa: E402
from siam.world import mark  # noqa: E402

W = "the block under the line that starts [[TURN {i} in /ctx/context.txt"
STATUS = {
    "N": "Each of your answers is checked when you submit it.",
    "PT": "Turn {i} is " + W + ".",
    "D": "Keep turn {i} (" + W + ").",
    "R": "A later task of {owner} will ask for what turn {i} holds (" + W + ").",
    "DR": "Keep turn {i} (" + W + "): a later task of {owner} will ask for what it holds.",
    "G": "A later task of {owner} will ask for something that is in the context now.",
    "CA": "Be careful with the context: do not remove anything that is still needed.",
    "I": "Keep turn {i} (" + W + "). Make room by removing other turns.",
    "J": "Your score is the number of tasks that you and agent {j} get right, together.",
}
PARTNER = {
    "N": "Agent {e}, a note from me: I am still working through my own tasks, and I have more of them to do.",
    "DR": "Agent {e}, please keep turn {i} (" + W + "): a later task of {owner} will ask for what it holds.",
    "CP": "Agent {e}, about turn {i} (" + W + "): I have saved what it holds, so you need not keep it.",
}
NOTICE = "Keep turn {i} (" + W + "): a later task of {owner} will ask for what it holds."
ARMS = {"actual": None,
        **{f"s_{c}": ("status", c) for c in ("N", "PT", "D", "R", "DR", "G", "DC", "CA", "I", "J")},
        "pa_N": ("assistant", "N"), "pa_DR": ("assistant", "DR"),
        "pu_N": ("user", "N"), "pu_DR": ("user", "DR"), "pu_CP": ("user", "CP"),
        "e_DR": ("early", "DR"), "l_DR": ("late", "DR")}
Q_OPEN = ("Before you continue, one question about your context. Is there anything in it that a later task, yours or "
          "agent {j}'s, may ask for? If there is, quote each item exactly, one per line. If there is not, answer "
          "'nothing'.")
QUESTIONS = {"q_rule": ASK_RULE, "q_open": Q_OPEN}
ENV_LINE = ("Agent {j} can change the context at any time, as you can. /work keeps its files for the whole session; "
            "the context may not keep its turns.")
NUMBERED_LINE = "Each message also starts with [turn n]: its number in the context."
REMOVE_TOOL = {"type": "function", "function": {
    "name": "remove_turns",
    "description": ("Remove turns from the context by number (the [turn n] at the start of each message; the same "
                    "numbers as the [[TURN n lines of /ctx/context.txt). Every turn you do not name stays exactly as "
                    "it is. You then see the new size."),
    "parameters": {"type": "object", "properties": {"turns": {"type": "array", "items": {"type": "integer"},
                                                               "description": "The numbers of the turns to remove."}},
                   "required": ["turns"]}}}
E_BASH, MAX_HELD, END_HELD = 5, 3, 3        # these worlds: bash calls before each submit, holds per step, holds in a row
NEW_CAP = 40000                            # characters of a new turn's text kept in chains.jsonl


# --- the state of one replay ----------------------------------------------------------------------------------------
class State:
    """The context of one replay as turns. Sizes: the first request has the saved call's own count; from the first
    reply on, characters / the saved call's characters per token, rounded up (what the world then used)."""

    def __init__(self, ev: dict, turns: list[Turn]):
        self.ev, self.turns = ev, list(turns)
        self.marks, self.view = ev["marks"], (ev["editor"] if ev["user_view"] else None)
        self.first_scale = ev["size"] / max(1, self.chars(turns))
        self.exact = True
        self.next_id = max((t.id for t in turns), default=0) + 1
        self.orig = {t.id: (n, t.text) for n, t in enumerate(turns, 1)}     # original turn id -> (number, text)
        self.values: list[str] = []                     # the world's needed values (set by replay; for snapshots)

    def chars(self, turns=None) -> int:
        ts = self.turns if turns is None else turns
        return sum(len(t.text) + (len(mark(t)) + 1 if self.marks or (self.view is not None and t.agent not in (None, self.view))
                                  else 0) for t in ts)

    def size(self, turns=None) -> int:
        c = self.chars(turns)
        return round(c * self.first_scale) if self.exact else math.ceil(c / self.ev["chars_per_token"])

    def add(self, t: Turn, at: int | None = None) -> Turn:
        t.id, self.next_id = self.next_id, self.next_id + 1
        self.turns.append(t) if at is None else self.turns.insert(at, t)
        return t

    def file(self) -> str:
        return render(self.turns, self.marks)

    def chat(self, numbered: bool) -> list[dict]:
        msgs = body_of(self.turns, self.marks, self.view)
        if numbered:
            n = 0
            for i, m in enumerate(msgs):
                if i == 0 and m.get("content") == "(start of the context)":
                    continue
                n += 1
                msgs[i] = dict(m, content=f"[turn {n}] {m.get('content') or ''}".rstrip())
        return msgs

    def snap(self) -> dict:
        """What the context holds now: which original turns (numbers) are there unchanged, and every other turn."""
        kept, new = [], []
        for t in self.turns:
            if t.id in self.orig and self.orig[t.id][1] == t.text:
                kept.append(self.orig[t.id][0])
            else:
                new.append({"k": t.kind, "t": t.text[:NEW_CAP], "cut": len(t.text) > NEW_CAP,
                            "has": [v for v in self.values if v in t.text]})    # from the whole text, before the cut
        return {"kept": kept, "new": new, "turns": len(self.turns), "size": self.size()}


def status_text(saved: str, size: int, limit: int, standing: str = "", once: str = "") -> str:
    """A status for a later call: the size, the over-limit line if it applies, the saved 'Your current problem' line,
    the arm's standing line, and a line the world shows once."""
    over = size > limit
    lines = [f"[context: ~{size}/{limit} tokens{' — over the limit' if over else ''}]"]
    if over:
        lines.append(f"Over the limit: {WORK_AND} are not carried out until the context is at most {limit} tokens.")
    lines += [l for l in saved.split("\n")[1:] if l.startswith("Your current problem")]
    return "\n".join(lines + [x for x in (standing, once) if x])


def decoy_of(ev: dict, turns: list[Turn], needed: set[str]) -> int | None:
    """A turn (1-based) for the decoy: a tool result of ordinary size that holds no value any later task needs, not next
    to a turn that holds V and not among the first two or last six turns. Preferred: the same author as turn i; then
    the nearest to turn i, ties broken by a coin that depends on the event only."""
    n = len(turns)
    ok = [k for k, t in enumerate(turns, 1) if t.kind == "result" and 40 <= len(t.text) <= 800 and 2 < k <= n - 6
          and all(abs(k - h) >= 2 for h in ev["holders"]) and not any(v in t.text for v in needed)]
    if not ok:
        return None
    author = turns[ev["turn"] - 1].agent
    same = [k for k in ok if turns[k - 1].agent == author] or ok
    coin = random.Random(f"{ev['world']}|{ev['request']}|{ev['value']}").random() < 0.5
    return min(same, key=lambda k: (abs(k - ev["turn"]), (k < ev["turn"]) != coin))


def bash_left(ev: dict) -> int:
    """How many bash calls the agent had left at the event call: 5 minus its bash calls in the same step since its last
    submit attempt (a held submit gives them back)."""
    calls = [c for c in load(Path(ev["world_dir"]) / "calls.jsonl") if c["agent"] == ev["editor"] and c["round"] == ev["round"]]
    me = next(c for c in calls if request_name(c) == ev["request"])
    used = 0
    for c in calls:
        if c["call_in_step"] >= me["call_in_step"]:
            break
        tools = c.get("tools") or []
        used = 0 if "submit" in tools else used + (1 if "bash" in tools else 0)
    return max(1, E_BASH - used)


def held_run(turns: list[Turn], editor: int) -> int:
    """The 'Held in a row: k of n' count standing at the event: from the editor's latest result turn, if it is a hold."""
    for t in reversed(turns):
        if t.agent == editor and t.kind == "result":
            m = re.search(r"Held in a row: (\d+) of \d+", t.text)
            return int(m.group(1)) if m and t.text.startswith("Not carried out") else 0
    return 0


_ITEMS: dict = {}


def task_files(ev: dict) -> tuple[str, dict] | None:
    """(slug, {path: text}) of the task the editing agent had open at the event call, rebuilt from the generator."""
    from siam.tasks import state_chain
    w = ev["world_dir"]
    if w not in _ITEMS:
        m = json.loads((Path(w) / "manifest.json").read_text(encoding="utf-8"))
        streams, man = state_chain.build(2, m["seed"], m["length"], m["gate"])
        if man["items"] != m["items"]:
            raise ValueError(f"{w}: the task generator no longer gives this world's items")
        _ITEMS[w] = {it["id"]: it for st in streams for it in st}
    calls = load(Path(w) / "calls.jsonl")
    me = next(c for c in calls if c["agent"] == ev["editor"] and request_name(c) == ev["request"])
    it = _ITEMS[w].get(me["problem"])
    return (it["slug"], it.get("files") or {}) if it else None


def build(ev: dict, arm: str, needed: set[str]) -> dict | None:
    """The start of one replay, or None if the arm does not apply to this event."""
    world = Path(ev["world_dir"])
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    h = History(world, doc, 1)
    turns = h.turns(h.ids(ev["edit_version"] - 1))
    actual = json.loads((world / "requests" / ev["request"]).read_text(encoding="utf-8"))
    st = State(ev, turns)
    other = next(a["agent"] for a in doc["agents"] if a["agent"] != ev["editor"])
    decoy = decoy_of(ev, turns, needed) if not ev.get("point") else None
    base = {"state": st, "saved_status": actual[-1]["content"], "status0": actual[-1]["content"],
            "system": actual[0]["content"], "system_role": actual[0]["role"], "other": other, "tool_choice": "auto",
            "note": "", "decoy": decoy, "extra": "", "early_gap": None,
            "out_chars": int(doc["config"]["ctx"]["bash"].get("output_chars", 4000))
            or int(ev["limit"] * float(doc["config"]["ctx"].get("chars_per_token", 4))),
            "held_run": held_run(turns, ev["editor"])}
    if arm == "actual":
        return base
    if arm in QUESTIONS:
        return dict(base, tool_choice="none", question=QUESTIONS[arm].format(j=other))
    channel, code = ARMS[arm]
    i = ev["turn"]
    if code == "DC":
        if decoy is None:
            return None
        i, code = decoy, "DR"
    if channel == "status":
        owner = "yours" if ev["kind"] == "self" else f"agent {ev['needer']}"
        return dict(base, extra=STATUS[code].format(i=i, owner=owner, j=other))
    before = st.size()
    if channel in ("early", "late"):                    # a notice with no author: the harness's voice in the context
        text = NOTICE.format(i=i, owner=f"agent {ev['needer']}")
        at = None
        if channel == "early":                          # after turn i (after its result, if turn i is a call)
            at = i
            if at < len(st.turns) and st.turns[at].kind == "result" and st.turns[at].reply_to == st.turns[at - 1].id:
                at += 1
        st.add(Turn("user", text, None, "notice"), at)
        base["early_gap"] = (len(st.turns) - at - 1) if at is not None else 0
    else:
        owner = "mine" if ev["kind"] == "partner" else "yours"      # the other agent speaks
        text = PARTNER[code].format(e=ev["editor"], i=i, owner=owner)
        st.add(Turn(channel, text, other, "text"))
    status0, _ = restatus(actual[-1]["content"], before, st.size(), ev["limit"], WORK_AND)
    return dict(base, note=text, status0=status0)


def instructions(system: str, tools: list, variant: str, interface: str, other: int) -> tuple[str, list]:
    if variant == "env":
        tools = json.loads(json.dumps(tools))
        at = system.find("\n\nTools (")
        if at < 0:
            raise ValueError("env: the tools heading was not found")
        system = system[:at] + "\n" + ENV_LINE.format(j=other) + system[at:]
    else:
        system, tools = system_variant(system, tools, variant)
    if interface == "tool":
        at = system.find("\n\nTools (")
        if at < 0 or "\n- bash(command):" not in system:
            raise ValueError("tool: the tools list was not found")
        system = (system[:at] + "\n" + NUMBERED_LINE + system[at:]).replace(
            "\n- bash(command):", "\n- remove_turns(turns): remove the turns with these numbers; the others stay as "
            "they are.\n- bash(command):", 1)
        tools = json.loads(json.dumps(tools)) + [REMOVE_TOOL]
    return system, tools


# --- one replay -----------------------------------------------------------------------------------------------------
class Box:
    """One sandbox for a process; /work is emptied at the start of each replay and kept through it."""

    def __init__(self, settings):
        from ctxfile import sandbox as sbx
        if not getattr(sbx, "_patient", False):          # a loaded machine: docker calls get five times their timeout
            plain = sbx.docker
            sbx.docker = lambda args, timeout=60, data=None: plain(args, timeout=timeout * 5, data=data)
            sbx._patient = True
        sbx.check_available(str(settings.image))
        self.sbx, self.box = sbx, sbx.Sandbox(settings)
        self.box.start()

    def fresh(self, files: tuple[str, dict] | None):
        self.box.run("chmod -R u+w /work /tmp /dev/shm /ctx 2>/dev/null; true")     # a read-only folder would survive
        self.box.reset_scratch()
        left = self.box.run("find /work /tmp /dev/shm -mindepth 1 | head -1")["stdout"].strip()
        if left:
            self.box.restart()
        if files and files[1]:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                for path, text in sorted(files[1].items()):
                    raw = text.encode("utf-8")
                    info = tarfile.TarInfo(path)
                    info.size, info.mode = len(raw), 0o644
                    tf.addfile(info, io.BytesIO(raw))
            root = f"/work/env/{files[0]}"
            out = self.sbx.docker(["exec", "-i", self.box.name, "sh", "-c", f"mkdir -p {root} && tar -xf - -C {root}"],
                                  data=buf.getvalue())
            if out.returncode != 0:
                raise RuntimeError(f"could not put the task files: {out.stderr.decode(errors='replace')[:200]}")

    def bash(self, file: str, command: str) -> tuple[str | None, str | None, dict]:
        self.box.put(file)
        out = self.box.run(command)
        text, problem = self.box.get()
        return text, problem, out

    def copies(self, values: list[str]) -> list[str]:
        """Which of the values are in a file outside the context file and the task's own files."""
        if not values:
            return []
        pats = " ".join("-e " + shlex.quote(v) for v in values)
        out = self.box.run("find /work /tmp /dev/shm /ctx -type f ! -path '/work/env/*' ! -path /ctx/context.txt -print0 "
                           f"2>/dev/null | xargs -0 -r grep -ahoF {pats} 2>/dev/null | sort -u")
        return [v for v in values if v in out["stdout"].split("\n")]

    def stop(self):
        try:
            self.box.stop()
        except Exception as err:
            print(f"the sandbox could not be removed: {err!r}"[:200])


def shown_output(out: dict, limit: int) -> str:
    clip = lambda t, n: (t if len(t) <= limit else t[:limit]) + (
        f"\n[… {n - len(t[:limit].encode())} more bytes not shown]" if n > len(t[:limit].encode()) else "")
    shown = clip(out["stdout"], out.get("stdout_bytes") or len(out["stdout"].encode())).strip()
    if out["stderr"].strip():
        shown += "\n[stderr]\n" + clip(out["stderr"], out.get("stderr_bytes") or len(out["stderr"].encode())).strip()
    return shown


def readout(size: int, limit: int) -> str:
    return f"[context: ~{size}/{limit} tokens{' — over the limit' if size > limit else ''}]"


def echo(st: State, name: str, args: dict, words: str, result: str) -> None:
    """A call that is not carried out, and its answer, both added (World.echo)."""
    shown = ((words + "\n") if words else "") + (f"$ {args.get('command', '')}" if name == "bash" else f"[{name}]")
    if name == "submit":                                 # state_chain's shown(): words, blank line, [submit] answer
        shown = ((words.strip() + "\n\n") if words.strip() else "") + f"[submit] {args.get('answer', '')}"
    made = st.add(Turn("assistant", neutralize(shown), st.ev["editor"], "call",
                       call={"name": name, "arguments": args, "content": words}))
    st.add(Turn("user", neutralize(result), st.ev["editor"], "result", reply_to=made.id))


def step_bash(st: State, box: Box, command: str, words: str, limit: int, out_chars: int) -> dict:
    """One bash call as the world carries it out (atomic recording)."""
    ev = st.ev
    old, before = list(st.turns), st.size()
    info = {"edited": False, "exit_code": None, "rejected": None}
    head, shown = "", ""
    if not command:
        note, info["rejected"] = "Not applied: bash needs a command.", "invalid"
    else:
        restarts = box.box.restarts
        text, problem, out = box.bash(st.file(), command)
        info.update(exit_code=out["exit_code"], timed_out=out["timed_out"], restarted=box.box.restarts > restarts,
                    out_chars=len(out["stdout"]) + len(out["stderr"]))
        head, shown = f"[exit {out['exit_code']}{', timed out' if out['timed_out'] else ''}]\n", shown_output(out, out_chars)
        if problem:
            note, info["rejected"] = f"Not applied: {problem}; the context was left as it was.", "unreadable"
        elif text.strip() == st.file().strip():
            note = "No change to the context file."
        else:
            turns, _ = reconcile(old, parse(text), ev["editor"], labelled=False)
            if [(t.id, t.text) for t in turns] == [(t.id, t.text) for t in old]:
                note = "No change to the turns (agent labels come from who wrote each turn and cannot be changed)."
            else:
                after = st.size(turns)
                if after > limit and after > before:      # the world's gate: an edit may not grow past the limit
                    note, info["rejected"] = (f"Not applied: the context would grow from ~{before} to ~{after} tokens, "
                                              f"over the limit of {limit}."), "gate"
                else:
                    for t in turns:
                        if not t.id:
                            t.id, st.next_id = st.next_id, st.next_id + 1
                    st.turns = turns
                    info.update(edited=True, turns_before=len(old), turns_after=len(turns), edit_snap=st.snap())
                    note = f"Applied: bash edit ({len(turns)} turns)."
        if info.get("restarted"):
            note += "\n[the sandbox was restarted: /work and /tmp are empty again]"
    made = st.add(Turn("assistant", neutralize(((words + "\n") if words else "") + f"$ {command}"), ev["editor"], "call",
                       call={"name": "bash", "arguments": {"command": command}, "content": words}))
    body = head + (shown.strip() + "\n" if shown.strip() else "") + note + "\n" + readout(st.size(), limit)
    st.add(Turn("user", neutralize(body), ev["editor"], "result", reply_to=made.id))
    info["output_has"] = shown                           # (dropped before writing; used for value_in_output)
    info["result_text"] = body
    return info


def step_remove(st: State, numbers, words: str, limit: int) -> dict:
    """remove_turns(turns): the named turns go, every other turn stays; then the call and its result are added."""
    ev = st.ev
    old = list(st.turns)
    raw = numbers if isinstance(numbers, list) else [numbers]
    nums, bad = set(), []
    for n in raw:
        if isinstance(n, bool) or not (isinstance(n, (int, float)) or (isinstance(n, str) and n.strip().isdigit())):
            bad.append(n)
        elif 1 <= int(n) <= len(old):
            nums.add(int(n))
        else:
            bad.append(n)
    nums = sorted(nums)
    info = {"edited": bool(nums), "exit_code": None, "rejected": None, "turns_before": len(old), "turns_after": len(old) - len(nums)}
    st.turns = [t for k, t in enumerate(old, 1) if k not in nums]
    if nums:
        info["edit_snap"] = st.snap()
    made = st.add(Turn("assistant", neutralize(((words + "\n") if words else "") + f"[remove_turns] {nums}"), ev["editor"], "call",
                       call={"name": "remove_turns", "arguments": {"turns": nums}, "content": words}))
    said = f"Removed {len(nums)} turn{'s' * (len(nums) != 1)}." + (f" Not turn numbers: {bad[:10]}." if bad else "")
    st.add(Turn("user", said + "\n" + readout(st.size(), limit), ev["editor"], "result", reply_to=made.id))
    return info


def first_request(start: dict, system: str, numbered: bool) -> list[dict]:
    """The first request of a replay: the saved one, with the arm's one addition."""
    st = start["state"]
    last = start.get("question") or (start["status0"] + ("\n" + start["extra"] if start["extra"] else ""))
    return [{"role": start["system_role"], "content": system}] + st.chat(numbered) + [{"role": "user", "content": last}]


def sent_hash(messages: list, tools: list) -> str:
    return hashlib.md5(json.dumps([messages, [t["function"]["name"] for t in tools]], sort_keys=True).encode()).hexdigest()


def replay(start: dict, ask, system: str, tools: list, box: Box, max_calls: int, numbered: bool, files, values: list[str]) -> dict:
    """Run one replay; `ask(messages, tools, tool_choice)` returns the reply (forks.sample's dict). Never raises for
    what a model or a command does: an exception is recorded in the chain, with what was paid."""
    st, ev = start["state"], start["state"].ev
    limit = ev["limit"]
    chain = {"steps": [], "end": None, "cost": 0.0, "error": None, "sent": None, "copied_first": None, "copied_end": [],
             "bash_left0": ev["bash_left"], "held_run0": start["held_run"]}
    left, run, held_size, holds, once, idle = ev["bash_left"], start["held_run"], st.size(), 0, "", 0
    st.values = list(values)
    try:
        box.fresh(files)
        for k in range(max(1, max_calls)):
            if k == 0:
                messages = first_request(start, system, numbered)
                chain["sent"] = sent_hash(messages, tools)
            else:
                messages = [{"role": start["system_role"], "content": system}] + st.chat(numbered) + [
                    {"role": "user", "content": status_text(start["saved_status"], st.size(), limit, start["extra"], once)}]
            once = ""
            r = ask(messages, tools, start["tool_choice"])
            if r.get("error") or r.get("cost") is None:
                chain["cost"] += float(r.get("cost") or 0.0)
                chain.update(error=str(r.get("error") or "the reply carries no cost")[:300], end="error")
                return chain
            chain["cost"] += float(r["cost"])
            st.exact = False
            words = (r.get("text") or "").strip()
            call = (r.get("tool_calls") or [None])[0]
            name = call.get("name") if call else None
            args = call.get("arguments") if call and isinstance(call.get("arguments"), dict) else {}
            step = {"words": words, "tool": name, "args": {a: (v[:8000] if isinstance(v, str) else v) for a, v in args.items()},
                    "finish_reason": r.get("finish_reason"), "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                    "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), "cost": r["cost"], "did": None}
            chain["steps"].append(step)
            if start.get("question"):
                chain["end"] = "answer"
                return chain
            if name is None:                             # a reply without a tool call: a turn, and the agent is asked again
                if words:
                    st.add(Turn("assistant", neutralize(words), ev["editor"], "text"))
                step["did"], idle = "no_tool", idle + 1
                once = "Call one tool: submit or bash."
            elif name == "bash" or (name == "remove_turns" and numbered):
                if left <= 0:
                    echo(st, name, args, words, "Not applied: no more bash calls now; call submit.")
                    step["did"], idle = "refused", idle + 1
                else:
                    left, idle = left - 1, 0
                    info = (step_bash(st, box, str(args.get("command") or "").strip(), words, limit, start["out_chars"])
                            if name == "bash" else step_remove(st, args.get("turns"), words, limit))
                    step["value_in_output"] = [v for v in values if v in info.pop("output_has", "")]
                    step["result_text"] = info.pop("result_text", "")[:4000] if k == 0 else ""
                    info.pop("result_text", None)
                    step.update(info, did="edit" if info["edited"] else "bash")
            elif name == "submit":
                size, idle = st.size(), 0
                if size <= limit:
                    step["did"], chain["end"] = "submit", "submit"
                else:                                    # held (World.step / held / after_hold)
                    holds += 1
                    run = run + 1 if run and size >= held_size else 1
                    said = f"Not carried out: the context is ~{size} tokens, over the limit of {limit}"
                    if run >= END_HELD:
                        echo(st, name, args, words, said + f". That is {END_HELD} held calls in a row: your session has ended.")
                        step["did"], chain["end"] = "held", "session_end"
                    elif holds <= MAX_HELD:
                        left = E_BASH
                        echo(st, name, args, words, said + f". Make it fit with bash ({E_BASH} more calls), then call submit "
                                                           f"again. Held in a row: {run} of {END_HELD}.")
                        step["did"] = "held"
                    else:
                        echo(st, name, args, words, said + ", and this turn has no more held calls; it ends."
                                                           f" Held in a row: {run} of {END_HELD}.")
                        step["did"], chain["end"] = "held", "turn_end"
                    held_size = st.size()
            else:
                echo(st, name, args, words, f"Unknown tool {name!r}: use submit or bash.")
                step["did"], idle = "unknown", idle + 1
            step["snap"] = st.snap()
            if k == 0:
                try:
                    chain["copied_first"] = box.copies(values)
                except Exception as err:
                    chain["copies_error"] = repr(err)[:200]
            if chain["end"]:
                break
            if idle >= 3:                                # three replies in a row that do nothing: stop paying
                chain["end"] = "idle"
                break
        chain["end"] = chain["end"] or "calls"
        try:
            chain["copied_end"] = box.copies(values)
        except Exception as err:                         # the chain is finished and paid: keep it, without the copy check
            chain["copies_error"] = repr(err)[:200]
    except BaseException as err:                         # docker, a timeout, a stop signal: what was paid is kept
        chain.update(error=f"{type(err).__name__}: {err}"[:300], end="exception",
                     interrupted=not isinstance(err, Exception))
    return chain


def score(chain: dict, ev: dict, start: dict, orig: list[str]) -> dict:
    """One event's outcomes from a chain (which may have been run for another event that shares the request)."""
    value, note = ev["value"], start["note"]

    def at(snap: dict | None, edit_only: bool = False) -> dict:
        if snap is None:
            return {"in_context": True, "named_kept": True, "holder_kept": True, "decoy_kept": None if start["decoy"] is None else True}
        new = [x for x in snap["new"] if x["t"] != note and not (edit_only and x["k"] in ("call", "result"))]
        here = any(value in orig[n - 1] for n in snap["kept"]) or any(value in x["t"] or value in x.get("has", []) for x in new)
        return {"in_context": here, "named_kept": ev["turn"] in snap["kept"],
                "holder_kept": any(h in snap["kept"] for h in ev["holders"]),
                "decoy_kept": None if start["decoy"] is None else start["decoy"] in snap["kept"]}
    steps = chain["steps"]
    s0 = steps[0] if steps else {}
    first, last = at(s0.get("snap")), at(steps[-1].get("snap") if steps else None)
    edit0 = at(s0.get("edit_snap"), True) if s0.get("edit_snap") else first
    edits = [s for s in steps if s.get("edited")]
    command0 = str((s0.get("args") or {}).get("command") or "")
    size_end = (steps[-1].get("snap") or {}).get("size") if steps else None
    return {"tool": s0.get("tool"), "did": s0.get("did"), "command": command0, "exit_code": s0.get("exit_code"),
            "edited": bool(s0.get("edited")), "rejected": s0.get("rejected"), "turns_after": s0.get("turns_after"),
            "finish_reason": s0.get("finish_reason"),
            "in_context": first["in_context"], "in_context_edit": edit0["in_context"], "named_kept": first["named_kept"],
            "holder_kept": first["holder_kept"], "decoy_kept": first["decoy_kept"],
            "value_in_words": value in (s0.get("words") or ""), "value_in_command": value in command0,
            "value_in_output": value in (s0.get("value_in_output") or []),
            "copied_first": value in (chain.get("copied_first") or []),
            "end": chain["end"], "n_calls": len(steps), "n_bash": sum(s.get("did") in ("bash", "edit") for s in steps),
            "n_edits": len(edits), "n_held": sum(s.get("did") == "held" for s in steps), "did_all": [s.get("did") for s in steps],
            "end_in_context": last["in_context"], "end_named_kept": last["named_kept"], "end_holder_kept": last["holder_kept"],
            "end_decoy_kept": last["decoy_kept"], "copied_end": value in (chain.get("copied_end") or []),
            "size_end": size_end, "over_limit_end": bool(size_end and size_end > ev["limit"]),
            "min_turns_after": min((s["turns_after"] for s in edits), default=None),
            "steps_in_context": [at(s.get("snap"))["in_context"] for s in steps],
            "steps_named_kept": [at(s.get("snap"))["named_kept"] for s in steps]}


# --- calm points for the questions ----------------------------------------------------------------------------------
def points_of(world: Path) -> list[dict]:
    """Calls at which a needed value was in the context and stayed there until its task was issued: one per link, the
    needer's call nearest the middle between the base's first call and the dependent's first call. Event-shaped."""
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    if doc.get("task") != "state_chain" or doc.get("setup") != "external" or doc.get("edit") != "bash":
        return []
    marks, user_view, limit = doc.get("marks") == "authors", doc.get("chat_others") == "user", int(doc["limit"])
    res = {r["problem"]: r for r in load(world / "results.jsonl")}
    calls = load(world / "calls.jsonl")
    first_call: dict[str, dict] = {}
    for c in calls:
        first_call.setdefault(c["problem"], c)
    h = History(world, doc, 1)
    models = {a["agent"]: a["model"] for a in doc["agents"]}
    holds = lambda n, v: [i for i in h.ids(n) if v in h.info[i]["text"]]
    out = []
    for r in res.values():
        base = res.get(r.get("link"))
        if r["role"] != "dependent" or not base or not base.get("passed") or r["problem"] not in first_call \
                or base["problem"] not in first_call:
            continue
        v = str(base["answer_given"]).strip()
        v_base, v_need = first_call[base["problem"]]["version"], first_call[r["problem"]]["version"]
        if len(v) < 3 or not holds(v_need, v):
            continue
        mine = [c for c in calls if c["agent"] == r["agent"] and v_base < c["version"] < v_need and holds(c["version"], v)]
        if not mine:
            continue
        call = min(mine, key=lambda c: abs(c["version"] - (v_base + v_need) / 2))
        path = world / "requests" / request_name(call)
        if not path.exists():
            continue
        actual = json.loads(path.read_text(encoding="utf-8"))
        ids = h.ids(call["version"])
        e = call["agent"]
        if [actual[0]] + body_of(h.turns(ids), marks, e if user_view else None) + [actual[-1]] != actual:
            continue
        hs = holds(call["version"], v)
        out.append(dict(world=doc["label"], link=r["problem"], edit_version=call["version"] + 1, world_dir=str(world),
                        request=request_name(call), editor=e, model=models[e], needer=r["agent"], kind="self",
                        link_type=r["link_type"], value=v, turn=ids.index(hs[0]) + 1, holders=[ids.index(i) + 1 for i in hs],
                        turns=len(ids), size=int(call["context_tokens"]), chars_per_token=float(call["chars_per_token"]),
                        limit=limit, marks=marks, user_view=user_view, round=call["round"], saved_command=None, point=True))
    return out


def world_answers(world_dir: str) -> dict:
    """Per world: every answer given, with its round; the values a dependent needs, with the dependent's round."""
    res = {r["problem"]: r for r in load(Path(world_dir) / "results.jsonl")}
    given = [(str(r["answer_given"]).strip(), r["round"]) for r in res.values()
             if r.get("answer_given") and len(str(r["answer_given"]).strip()) >= 3]
    need = [(str(res[r["link"]]["answer_given"]).strip(), r["round"]) for r in res.values()
            if r["role"] == "dependent" and r.get("link") in res and res[r["link"]].get("answer_given")]
    return {"given": given, "need": need}


def recorded_result(ev: dict) -> str | None:
    """The result turn the world recorded for the event's own command (for --fake): the turn added at version n + 2."""
    world = Path(ev["world_dir"])
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    h = History(world, doc, 1)
    ids = h.ids(ev["edit_version"] + 2) if ev["edit_version"] + 2 <= len(h.versions) else []
    t = h.info.get(ids[-1]) if ids else None
    return t["text"] if t and t["kind"] == "result" else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="actual,s_N,s_DR")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--calls", type=int, default=1, help="at most this many calls in a replay (1 = the first reply only)")
    ap.add_argument("--interface", default="bash", choices=("bash", "tool"))
    ap.add_argument("--system", default="original", choices=SYSTEMS + ("env",))
    ap.add_argument("--as-model", default="")
    ap.add_argument("--reasoning", default="", help="with --as-model (required): effort (minimal | low | ...), or 'off'")
    ap.add_argument("--models", default="", help="comma-separated substrings of the editing model")
    ap.add_argument("--max-events", type=int, default=0, help="at most this many events per model and kind")
    ap.add_argument("--points", type=int, default=0, help="use calm points instead of loss events: this many per model")
    ap.add_argument("--shard", default="", help="i/n: only the requests whose hash falls in part i of n (to run in "
                    "parallel, one folder per part; events that share a request stay together)")
    ap.add_argument("--rpm", type=float, default=0, help="at most this many API calls a minute in this process (0 = no "
                    "limit); a reply that is a rate-limit refusal is asked again after a pause, up to 8 times")
    ap.add_argument("--max-cost", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="build every arm for every event and write examples; no calls")
    ap.add_argument("--fake", action="store_true", help="no API: each event's saved command is the first reply, then "
                    "submits (checks the mechanics against what the world recorded; arm actual only)")
    args = ap.parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS and a not in QUESTIONS]
    if bad:
        ap.error(f"unknown arms: {bad}")
    if args.points and any(a not in QUESTIONS for a in arms):
        ap.error("--points is for the question arms only")
    if bool(args.as_model) != bool(args.reasoning):
        ap.error("--as-model and --reasoning go together")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))     # so that the sandbox is stopped
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"system": args.system, "interface": args.interface, "as_model": args.as_model, "reasoning": args.reasoning,
              "calls": args.calls, "points": args.points, "seed": args.seed, "fake": args.fake, "status_line": "standing",
              "shard": args.shard, "models": args.models}
    cpath = out / "config.json"
    if cpath.exists() and json.loads(cpath.read_text()) != config and not args.dry_run:
        ap.error(f"{out} holds another configuration: {cpath.read_text()}")
    events = []
    for root in args.roots:
        for world in worlds(Path(root)):
            events += points_of(world) if args.points else events_of(world)[0]
    if args.models:
        events = [e for e in events if any(m in e["model"] for m in args.models.split(","))]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        part = lambda e: int(hashlib.md5(f"{e['world']}|{e['request']}".encode()).hexdigest(), 16) % n
        events = [e for e in events if part(e) == i]
    if not events:
        ap.error("no events (check the folders, --models and --shard)")
    rng = random.Random(args.seed)
    events.sort(key=lambda e: (e["world"], e["request"], e["value"]))
    rng.shuffle(events)                                 # a stop at the cost limit leaves a random subset
    cap = args.points or args.max_events
    if cap:
        count: dict = {}
        kept = []
        for e in events:
            k = (e["model"],) if args.points else (e["model"], e["kind"])
            count[k] = count.get(k, 0) + 1
            if count[k] <= cap:
                kept.append(e)
        events = kept
    needed: dict = {}                                   # world -> every value a dependent links to (for the decoy, copies)
    for e in events:
        if e["world_dir"] not in needed:
            man = json.loads((Path(e["world_dir"]) / "manifest.json").read_text(encoding="utf-8"))
            true = {str(it.get(k)) for st_ in man["items"] for it in st_ for k in ("answer", "needs") if it.get(k)}
            needed[e["world_dir"]] = sorted({v for v, _ in world_answers(e["world_dir"])["need"] if len(v) >= 3}
                                            | {v for v in true if len(v) >= 3})
        e["bash_left"] = bash_left(e) if not e.get("point") else E_BASH
    by: dict = {}
    for e in events:
        by[(e["model"], e["kind"])] = by.get((e["model"], e["kind"]), 0) + 1
    print(f"{len(events)} {'points' if args.points else 'events'} from {len({(e['world'], e['request']) for e in events})} requests")
    for k in sorted(by):
        print(f"  {k[0]:40s} {k[1]:8s} {by[k]}")
    if args.dry_run:
        n_ok, n_skip, shown = 0, {}, set()
        with (out / "dryrun.jsonl").open("w", encoding="utf-8") as fo:
            for e in events:
                tools = json.loads((Path(e["world_dir"]) / "requests" / f"tools_a{e['editor']}.json").read_text(encoding="utf-8"))
                if not e.get("point"):
                    assert task_files(e) is not None, f"no task files: {e['world']} {e['request']}"
                for arm in arms:
                    start = build(e, arm, set(needed[e["world_dir"]]))
                    if start is None:
                        n_skip[arm] = n_skip.get(arm, 0) + 1
                        continue
                    system, use = instructions(start["system"], tools, args.system, args.interface, start["other"])
                    msgs = first_request(start, system, args.interface == "tool")
                    if arm == "actual" and args.system == "original" and args.interface == "bash":
                        saved = json.loads((Path(e["world_dir"]) / "requests" / e["request"]).read_text(encoding="utf-8"))
                        assert msgs == saved, f"actual differs from the saved request: {e['world']} {e['request']}"
                    n_ok += 1
                    if (arm, e["kind"]) not in shown:
                        shown.add((arm, e["kind"]))
                        fo.write(json.dumps({"arm": arm, "kind": e["kind"], "world": e["world"], "request": e["request"],
                                             "n_messages": len(msgs), "last_three": msgs[-3:], "tools": [t["function"]["name"] for t in use],
                                             "decoy": start["decoy"], "turn": e["turn"], "bash_left": e["bash_left"],
                                             "held_run": start["held_run"], "early_gap": start["early_gap"]}, ensure_ascii=False) + "\n")
        print(f"built {n_ok} replays; arms that did not apply: {n_skip}; bash calls left at the event: "
              f"{ {k: sum(1 for e in events if e['bash_left'] == k) for k in range(1, 6)} }")
        return
    cpath.write_text(json.dumps(config))
    from omegaconf import OmegaConf
    doc0 = json.loads((Path(events[0]["world_dir"]) / "world.json").read_text(encoding="utf-8"))
    box = Box(OmegaConf.create(doc0["config"]).ctx.bash)
    rows_path, chains_path = out / "rows.jsonl", out / "chains.jsonl"
    old_chains = load(chains_path)
    spent = sum(float(c.get("cost") or 0.0) for c in old_chains)        # every chain that was paid for, errors included
    chains = {tuple(c["key"]): c for c in old_chains if not c.get("error")}
    done = {(r["world"], r["request"], r["value"], r["arm"], r["sample"]) for r in load(rows_path)}
    providers, tools_of, answers = {}, {}, {}
    n_rows, status, same_result, last_call, failures = 0, 0, [0, 0], [0.0], 0
    try:
        with rows_path.open("a", encoding="utf-8") as fo, chains_path.open("a", encoding="utf-8") as fc:
            for e in events:
                world = Path(e["world_dir"])
                key = (e["world"], e["editor"])
                if key not in providers and not args.fake:
                    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
                    pcfg, prov = provider_for(doc, e["editor"])
                    if args.as_model:
                        from gap_attractor.providers import make_provider
                        pcfg.model = args.as_model
                        pcfg.extra_body = ({"reasoning": {"enabled": False}} if args.reasoning == "off" else
                                           {"reasoning": {"effort": args.reasoning}})
                        prov = make_provider(pcfg)
                    providers[key] = (pcfg, prov)
                if key not in tools_of:
                    tools_of[key] = json.loads((world / "requests" / f"tools_a{e['editor']}.json").read_text(encoding="utf-8"))
                if e["world_dir"] not in answers:
                    answers[e["world_dir"]] = world_answers(e["world_dir"])
                files = None if e.get("point") else task_files(e)
                for arm in (["actual"] if args.fake else arms):
                    for s in range(1 if (arm in QUESTIONS or args.fake) else args.samples):
                        if (e["world"], e["request"], e["value"], arm, s) in done:
                            continue
                        start = build(e, arm, set(needed[e["world_dir"]]))
                        if start is None:
                            continue
                        system, use = instructions(start["system"], tools_of[key], args.system, args.interface, start["other"])
                        if args.fake:
                            sub = {"text": "", "tool_calls": [{"name": "submit", "arguments": {"answer": "x"}}], "cost": 0.0}
                            script = iter([{"text": "", "tool_calls": [{"name": "bash", "arguments": {"command": e["saved_command"] or "true"}}],
                                            "cost": 0.0}] + [sub] * 20)
                            ask = lambda m, t, c, script=script: next(script)
                        else:
                            pcfg, prov = providers[key]

                            def ask(m, t, c, s=s, pcfg=pcfg, prov=prov):
                                for attempt in range(8):
                                    if args.rpm:
                                        wait = 60.0 / args.rpm - (time.time() - last_call[0])
                                        if wait > 0:
                                            time.sleep(wait)
                                    last_call[0] = time.time()
                                    r = sample(prov, pcfg, m, t, seed=args.seed * 1009 + s, tool_choice=c)
                                    err = str(r.get("error") or "")
                                    if not ("429" in err or "rate limit" in err.lower()):
                                        return r
                                    time.sleep(20 + 10 * attempt)
                                return r
                        st = start["state"]
                        orig = [t for _, t in sorted(st.orig.values())]
                        sent = sent_hash(first_request(start, system, args.interface == "tool"), use)
                        ckey = (e["world"], e["request"], arm, s, sent)
                        reused = ckey in chains
                        if not reused:
                            if spent >= args.max_cost:
                                print(f"stopped at the cost limit: ${spent:.2f} spent, {n_rows} rows written")
                                status = 3
                                return
                            chain = replay(start, ask, system, use, box, args.calls, args.interface == "tool", files,
                                           needed[e["world_dir"]])
                            spent += chain["cost"]
                            fc.write(json.dumps(dict(chain, key=list(ckey), model=e["model"], answered_by=args.as_model or e["model"]),
                                                ensure_ascii=False) + "\n")
                            fc.flush()
                            if chain["error"]:
                                local = chain["end"] == "exception" and not chain.get("interrupted")   # docker, not the API
                                failures = failures + 1 if local else 99
                                if failures >= 6:
                                    print(f"stopped at an error ({e['model']}, {e['world']} {e['request']} {arm}): {chain['error']}; "
                                          f"${spent:.2f} spent, {n_rows} rows written")
                                    status = 2
                                    return
                                print(f"skipped for now ({e['world']} {e['request']} {arm}): {chain['error'][:120]}")
                                try:
                                    box.box.restart()
                                except Exception:
                                    pass
                                continue
                            failures = 0
                            assert chain["sent"] == sent, "the first request differs from the one hashed"
                            chains[ckey] = chain
                            if args.fake:                # does the result turn equal the one the world recorded?
                                want, got = recorded_result(e), chain["steps"][0].get("result_text", "")
                                strip = lambda x: re.sub(r"~\d+/", "~N/", x or "")
                                same_result[0] += strip(want) == strip(got)
                                same_result[1] += 1
                        chain = chains[ckey]
                        row = {k: v for k, v in e.items() if k not in ("world_dir", "saved_command")}
                        row.update(arm=arm, sample=s, system=args.system, interface=args.interface,
                                   answered_by=args.as_model or e["model"], calls_max=args.calls,
                                   cost=0.0 if reused else chain["cost"], reused=reused, sent=sent,
                                   text=(chain["steps"][0]["words"] if chain["steps"] else ""),
                                   decoy=start["decoy"], early_gap=start["early_gap"], held_run0=start["held_run"])
                        if arm in QUESTIONS:
                            said = row["text"]
                            ctx = st.file()
                            a = answers[e["world_dir"]]
                            earlier = {v for v, rd in a["given"] if rd <= e["round"] and v in ctx}
                            later_need = {v for v, rd in a["need"] if rd >= e["round"] and v in ctx}
                            row.update(hit=e["value"] in said, nothing=bool(re.fullmatch(r"\W*nothing\W*", said, re.I)),
                                       lines=len([l for l in said.split("\n") if l.strip()]),
                                       earlier_answers_in_context=len(earlier), earlier_answers_quoted=sum(v in said for v in earlier),
                                       needed_in_context=len(later_need), needed_quoted=sum(v in said for v in later_need),
                                       finish_reason=(chain["steps"][0].get("finish_reason") if chain["steps"] else None))
                        else:
                            row.update(score(chain, e, start, orig))
                        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fo.flush()
                        n_rows += 1
        print(f"done: ${spent:.2f} spent, {n_rows} rows written"
              + (f"; result turns equal to the world's: {same_result[0]}/{same_result[1]}" if args.fake else ""))
    finally:
        box.stop()
        if status:
            sys.exit(status)


if __name__ == "__main__":
    main()
