"""A world: N agents (any models) solving a task's problems (LiveCodeBench by default; siam/task.py) in ONE shared,
capped context file.

Every agent has its own stream of problems and acts in turns (each round, every active agent takes one step, in a
seeded random order). Each step the agent is sent its own instructions, the whole context file and a short status
line, and makes tool calls until it has made one work call:
  run_tests(code)  its program on the current problem's public tests (LCB's grader; feedback in LCB's self-repair words)
  submit(code)     graded on all tests (LCB pass@1 grading); its next problem starts at its next step
  edit_context     change the file (replace_turns / delete_turns / replace_text, ctxfile.edits), up to max_edit_calls
Everything goes into the file: problem statements, the agents' messages, every call with its program, every result.

Sharing (world.sharing): full = anyone may change any turn, the other agents' included (the study's core
assumption; only a problem statement still being worked on is protected) | readonly_others = only one's own turns
(control) | separate = one file per agent (control; budget per file as given). With one agent it is the single case.

The cap (ctx.cap): the model is shown the limit (budget - reserve) and its size; nudges at 25/50/75 %. A work call
made while the context is over the limit is held (not carried out) and the agent gets its edit calls back; after
max_held holds in one step the oldest unprotected turns are removed until it fits (logged as an eviction) and the
call goes ahead. Size (ctx.count=whole): system message + messages + tool schemas + status line, at the model's
calibrated characters per token; ctx.count=turns (the study, from 1 Nov): the messages only, so the instructions
always fit.

Designs (all opt-in; the defaults are the original design; old names in brackets are still accepted):
- world.order: shuffled [seeded_shuffle] (default: a new seeded order every round) | fixed (by agent number) |
  turn-taking [round_robin] (the first mover is drawn once from the seed, then the order stays fixed);
- world.rounds: 0 (default: until every stream is done) | R: exactly R rounds (every agent takes R steps);
- world.exit_question: asked once after the end (exit.jsonl; never in the file);
- world.schedule: rounds (default) | free-for-all [async]: every agent runs its own loop at its own pace (its model's speed decides how
  many steps it takes); every change of the file is atomic (one lock), but the file may change between an agent reading
  it and acting on it: an edit names turns by the numbers the agent saw, they are mapped to the turns themselves, and an
  edit whose turns changed or went in between is refused ("the context changed since you read it");
- world.awareness (setup=assigned): introduced [told] (default: the instructions name the other agents, turns carry agent=k, the status line shows
  each agent's share) | strangers [unaware] (none of that: the instructions read as for one agent, problem statements carry no
  owner, and the agent must notice for itself that someone else writes in its context) | rumour [hint] (as strangers, plus one
  sentence, world.hint_text, saying the context may be shared); world.others_as: user (another
  agent's turns arrive as user messages) | own (in their own role, i.e. another agent's replies look like one's own).

Per problem at most max_work_calls work calls; after the last run_tests its program is submitted. Logs (one folder
per world): world.json, calls.jsonl, actions.jsonl, edits.jsonl, results.jsonl, rounds.jsonl (tokens by author),
the live file(s) and version logs (ctxfile.context.ContextFile), snapshots/ after every round.
"""
from __future__ import annotations

import copy
import io
import json
import math
import random
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

from ctxfile import edits as edit_ops
from ctxfile.budget import Budget
from ctxfile.context import ContextFile, Turn, neutralize, parse, reconcile, render
from siam.task import Task
from siam.tasks.lcb import LcbTask, code_of  # noqa: F401  (code_of: kept importable from here)


# world.need_line: the status sentences of need_forks2.py (STATUS["R"], STATUS["DR"]; W is its W), so that the live run and
# the replays use the same words.
NEED_W = "the block under the line that starts [[TURN {i} in /ctx/context.txt"
NEED_LINES = {"reason": "A later task of {owner} will ask for what turn {i} holds (" + NEED_W + ").",
              "directive": "Keep turn {i} (" + NEED_W + "): a later task of {owner} will ask for what it holds."}


AWARENESS = {"introduced": "introduced", "told": "introduced", "strangers": "strangers", "unaware": "strangers",
             "rumour": "rumour", "hint": "rumour"}
ORDER = {"shuffled": "shuffled", "seeded_shuffle": "shuffled", "fixed": "fixed", "turn-taking": "turn-taking",
         "round_robin": "turn-taking"}
SCHEDULE = {"rounds": "rounds", "free-for-all": "free-for-all", "async": "free-for-all"}
SETUPS = ("assigned", "pool", "external")
CHAT_NUMBERING = ("Turns are numbered 1, 2, 3, ... in the order of the messages after these instructions, not counting a "
                  "first message '(start of the context)' or the status message at the end: every other message is one "
                  "turn (a tool call and its result are two turns).")   # 26 Oct: the placeholder and status are no turns


def scope_words(scope: str) -> str:
    return {"full": "any turn, whoever wrote it (yours or another agent's)",
            "readonly_others": "your own turns (other agents' turns cannot be changed)", "single": "any turn"}[scope]


WORKSPACE = {"kept": "/work is a folder of your own: it keeps its files for the whole session, and the other agents cannot "
                      "see it.",
             "kept_alone": "/work is a folder of your own: it keeps its files for the whole session.",
             "emptied": "/work is a scratch folder; it is emptied after each command."}
BASH_EXAMPLE = ("A short script is the reliable way to change many lines; this one puts one text in place of turns 3 to 5:\n"
                "python3 - <<'EOF'\n"
                "p = \"/ctx/context.txt\"; lines = open(p).read().split(\"\\n\")\n"
                "starts = [i for i, l in enumerate(lines) if l.startswith(\"[[TURN \")]\n"
                "end = starts[5] if len(starts) > 5 else len(lines)\n"
                "lines[starts[2] + 1:end] = [\"your text\", \"\"]\n"
                "open(p, \"w\").write(\"\\n\".join(lines))\n"
                "EOF")


def bash_tool(scope: str, chat: bool, workspace: str = "kept", marks: bool = False, shown: int = 4000) -> dict:
    if chat:            # pool / external: the file holds the messages the agent sees, numbered as they are counted
        header = "[[TURN i role=... agent=k]] (agent=k: who wrote it, from the record)" if marks else "[[TURN i role=...]]"
        what = ("Run one bash command in your own sandbox (no network). Your context is also the file /ctx/context.txt: "
                f"one block per turn, each opened by a line {header} with the turn's text below it (a tool call is written "
                "as its words, then [run_tests] or [submit] and the program, or $ and the command). " + CHAT_NUMBERING +
                " When a command changes the file, your context changes the same way from your next call on; the file "
                "holds the same turns you see, so there is no need to print it. You can change "
                f"{scope_words(scope)}. A block whose [[TURN ...]] line is gone merges into the block before it. What the "
                "command prints is added to your context" + (f" (at most {shown} characters of each stream are shown). "
                                                             if shown > 0 else ", all of it unless it alone is larger than "
                                                             "the context limit. ") +
                f"{WORKSPACE[workspace + ('_alone' if scope == 'single' and workspace == 'kept' else '')]}\n{BASH_EXAMPLE}")
    else:
        what = ("Run one bash command in your own sandbox (no network). Your context is the file "
                "/ctx/context.txt, one block per turn, each opened by a line [[TURN i role=...]]: what you change "
                f"in it is your context from your next call on. You can change {scope_words(scope)}, except a "
                "problem statement that is protected. What the command prints is shown to you and added to your "
                "context. /work is a scratch folder that stays between commands.")
    return {"type": "function", "function": {
        "name": "bash",
        "description": what,
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "One bash command."}},
                       "required": ["command"]}}}


def edit_tool(scope: str, chat: bool = False) -> dict:
    """edit_context with ctxfile.edits' parameters; the description says what may be changed."""
    tool = copy.deepcopy(edit_ops.TOOL)
    who = {"full": "any turn, whoever wrote it (yours or another agent's)",
           "readonly_others": "your own turns (other agents' turns cannot be changed)",
           "single": "any turn"}[scope]
    where = (CHAT_NUMBERING + " " if chat else "the numbered [[TURN i ...]] turns you see. ")
    tool["function"]["description"] = (
        f"Change your context: {where}You can change {who}, except a problem "
        "statement that is still being worked on. Give one or more of the lists replace_turns, delete_turns and "
        "replace_text. Turn numbers are the ones you see now, ranges must not overlap, and all edits apply together or "
        "not at all (an invalid call changes nothing and says why). You then see the result and the new size.")
    return tool


def remap_edit(arguments: dict, seen: list, now: list) -> tuple[dict, str | None]:
    """An edit names turns by the numbers the agent saw (ids `seen`, in order); map them to the numbers the same turns
    have now (ids `now`). Returns (arguments, None) or (arguments, why) when a named turn is gone or a range is no longer
    one contiguous block of the same turns. Numbers outside what was seen are left for the edit check to reject."""
    if seen == now[:len(seen)]:
        return arguments, None
    pos = {i: n for n, i in enumerate(now, 1)}
    out = dict(arguments)
    for op in ("replace_turns", "delete_turns"):
        items = arguments.get(op)
        if not isinstance(items, list):
            continue
        new = []
        for e in items:
            e = dict(e) if isinstance(e, dict) else e
            try:
                first, last = int(e["first"]), int(e["last"])
            except (TypeError, KeyError, ValueError):
                new.append(e)
                continue
            if not 1 <= first <= last <= len(seen):
                new.append(e)
                continue
            ids = seen[first - 1:last]
            if any(i not in pos for i in ids):
                return arguments, f"turns {first}..{last} are no longer there"
            start = pos[ids[0]]
            if now[start - 1:start - 1 + len(ids)] != ids:
                return arguments, f"turns {first}..{last} are no longer one block"
            e["first"], e["last"] = start, start + len(ids) - 1
            new.append(e)
        out[op] = new
    items = arguments.get("replace_text")
    if isinstance(items, list):
        new = []
        for e in items:
            e = dict(e) if isinstance(e, dict) else e
            try:
                turn = int(e["turn"])
            except (TypeError, KeyError, ValueError):
                new.append(e)
                continue
            if 1 <= turn <= len(seen):
                if seen[turn - 1] not in pos:
                    return arguments, f"turn {turn} is no longer there"
                e["turn"] = pos[seen[turn - 1]]
            new.append(e)
        out["replace_text"] = new
    return out, None


def mark(t) -> str:
    """world.marks=authors: what a turn's message starts with (its author; a result: the agent whose call it answers)."""
    return f"[agent {t.agent}]" if t.agent is not None else ""


def chat_messages(turns: list, marks: bool = False, viewer: int | None = None) -> list[dict]:
    """setup=pool / external: the file as plain chat, one message per turn, no headers. Every reply (whoever wrote it)
    is an assistant message; an unedited call followed by its result is a real tool call + tool message; problem
    statements, notices and any other user turn are user messages. If the file starts with a reply, a neutral first user
    message '(start of the context)' is put in front (not a turn). marks: every message starts with mark(turn).
    viewer (world.chat_others=user): another agent's turns, its calls and their results included, are user messages that
    start with its mark, never the viewer's own assistant messages or tool calls."""
    out, skip = [], set()
    tag = (lambda t, text: f"{mark(t)}\n{text}".strip() if mark(t) else text) if marks else (lambda t, text: text)
    for i, t in enumerate(turns):
        if i in skip:
            continue
        if viewer is not None and t.agent not in (None, viewer):
            out.append({"role": "user", "content": f"{mark(t)}\n{t.text.strip()}".strip()})
            continue
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        paired = nxt is not None and nxt.kind == "result" and nxt.reply_to == t.id
        if t.role == "assistant" and t.call and t.kind == "call" and paired:
            out.append({"role": "assistant", "content": tag(t, str(t.call.get("content") or "")) or None,
                        "tool_calls": [{"id": f"t{t.id}", "type": "function", "function": {
                            "name": t.call["name"], "arguments": json.dumps(t.call.get("arguments") or {}, ensure_ascii=False)}}]})
            out.append({"role": "tool", "tool_call_id": f"t{t.id}", "content": tag(nxt, nxt.text.strip())})
            skip.add(i + 1)
        else:
            out.append({"role": "assistant" if t.role == "assistant" else "user", "content": tag(t, t.text.strip())})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(start of the context)"})
    return out


_LOG_LOCK = threading.Lock()


def write_jsonl(path: Path, row: dict) -> None:
    with _LOG_LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class Agent:
    def __init__(self, k, pcfg, provider, stream, file, budget):
        self.k, self.pcfg, self.provider, self.stream = k, pcfg, provider, stream
        self.model = str(pcfg.model)
        self.file, self.budget = file, budget
        self.pos, self.problem, self.problem_turn = -1, None, None
        self.history: list = []                # closed problems: (position, title, accepted)
        self.work, self.steps_on_problem, self.last_code = 0, 0, ""
        self.done, self.end_reason = False, None
        self.stayed = False                    # world.after_stream=stay: the stream is finished, the agent stays on
        self.held_run, self.held_size = 0, 0   # held calls/replies in a row; the context's size after the last one
        self.waiting_since = None              # task.gate: the round since which only waiting items are left
        self.cpt = 4.0                         # characters per token, calibrated from the billed prompt (ctx.count=whole)
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "api_errors": 0, "no_tool": 0,
                      "edits": 0, "edits_applied": 0, "held": 0, "truncated": 0, "evictions": 0, "idle_steps": 0, "solved": 0, "submitted": 0}


class World:
    def __init__(self, cfg, seed: int, streams: list[list[dict]], out: Path, task, make_provider, label: str = ""):
        """`task`: a siam.task.Task, or a grader (then the task is LiveCodeBench graded by it)."""
        self.cfg, self.seed, self.out, self.label = cfg, int(seed), Path(out), label
        self.task = task if isinstance(task, Task) else LcbTask(task)
        w, c = cfg.world, cfg.ctx
        self.n = len(streams)
        self.sharing = str(w.sharing) if self.n > 1 else "single"
        if self.sharing not in ("full", "readonly_others", "separate", "single"):
            raise ValueError(f"world.sharing must be full, readonly_others or separate (got {self.sharing})")
        self.shared = self.sharing in ("full", "readonly_others")
        self.setup = str(w.get("setup", "assigned"))
        if self.setup not in SETUPS:
            raise ValueError(f"world.setup must be one of {SETUPS}")
        self.chat = self.setup in ("pool", "external")             # plain chat: no headers, every reply an assistant
        self.identity = str(w.get("identity", "none"))             # pool / external: none | names
        self.marks = str(w.get("marks", "none")) == "authors"      # pool / external: each message names its author
        self.cross = str(w.get("cross_test", "none"))              # external: none | ring (agent k tests agent k+1's)
        self.awareness = AWARENESS.get(str(w.get("awareness", "introduced")))
        if self.awareness is None:
            raise ValueError("world.awareness must be introduced, strangers or rumour")
        if self.chat:
            self.awareness = "introduced" if self.identity == "names" else "strangers"
        self.aware = self.awareness == "introduced"
        self.labelled = self.shared and self.aware and not self.chat and str(w.get("attribution", "labelled")) == "labelled"
        self.alone_view = not self.shared or not self.aware        # the instructions read as for a single agent
        self.others_as = "own" if self.chat else str(w.get("others_as", "user"))
        self.chat_others = str(w.get("chat_others", "own"))        # pool / external: others' turns as own | user
        self.open_task = not self.task.work_names                # no problems (siam/tasks/open.py): messages only
        self.goal = str(w.get("goal", "individual"))               # individual | joint (one team score for the pool)
        self.reasoning_in_context = bool(w.get("reasoning_in_context", False))   # the provider's reasoning as a turn
        self.alter_notice = bool(w.get("alter_notice", False))     # the status says when others changed one's turns
        self.need_line = str(w.get("need_line", "") or "")         # "" | reason | directive (state_chain: pending links)
        if self.need_line not in ("", *NEED_LINES):
            raise ValueError("world.need_line must be empty, reason or directive")
        self.need_seq: dict = {}                                   # per agent: need lines shown so far (need_lines.jsonl)
        self.altered: dict = {}                                    # owner -> editor -> {"removed": n, "changed": n}
        self.closed_ids: set = set()                               # items closed (graded), for items with `after`
        self.closed_by_other: dict = {}                            # joint: owner -> lines for its next status
        if self.chat_others not in ("own", "user") or (self.chat_others == "user" and not (self.chat and self.shared
                                                                                            and self.identity == "names")):
            raise ValueError("world.chat_others=user needs setup pool or external, a shared file and world.identity=names")
        self.schedule = SCHEDULE.get(str(w.get("schedule", "rounds")))
        if self.schedule is None:
            raise ValueError("world.schedule must be rounds or free-for-all")
        self.order = ORDER.get(str(w.get("order", "shuffled")))
        if self.order is None:
            raise ValueError("world.order must be shuffled, fixed or turn-taking")
        self.edit_mode = str(c.get("edit", "tool"))
        if self.edit_mode not in ("tool", "bash"):
            raise ValueError("ctx.edit must be tool or bash")
        self.edit_name = "bash" if self.edit_mode == "bash" else "edit_context"
        self.atomic = bool(c.get("atomic", False))                 # an editing call is written after it ran (main d545615)
        self.workspace = str((c.get("bash") or {}).get("workspace", "kept"))   # bash /work: kept for the run | emptied
        if self.workspace not in ("kept", "emptied"):
            raise ValueError("ctx.bash.workspace must be kept or emptied")
        self.count = str(c.get("count", "whole"))
        if self.count not in ("whole", "file", "turns") or (self.count == "turns" and not self.chat):
            raise ValueError("ctx.count must be whole, file or turns; turns needs setup pool or external")
        self.boxes: dict = {}
        self.log_work = bool((c.get("bash") or {}).get("log_work", True))   # bash: what /work holds after each command
        self.work_seen: dict = {}                                  # agent -> {file: sha1} at its previous command
        self.typed: dict = {}                                      # agent -> lines it typed in its bash commands
        self.lock = threading.RLock()                              # every change of the file (async: between agents)
        self.capped = bool(c.get("cap", True))
        self.T, self.E = int(w.max_work_calls), int(w.max_edit_calls)
        self.max_held, self.max_calls = int(w.max_held), int(w.max_calls_per_step)
        self.max_steps = int(w.get("max_steps_per_problem", 3 * self.T))
        self.rounds = int(w.get("rounds") or 0)                   # > 0: the world lasts exactly this many rounds
        if self.rounds and (self.schedule != "rounds" or self.setup == "pool"):
            raise ValueError("world.rounds needs schedule=rounds and setup assigned or external")
        if self.marks and not (self.chat and self.shared and self.identity == "names"):
            raise ValueError("world.marks=authors needs setup pool or external, a shared file and world.identity=names")
        if self.goal not in ("individual", "joint") or (self.goal == "joint" and (self.open_task or not (
                self.setup == "external" and self.shared and self.identity == "names" and self.cross == "none"))):
            raise ValueError("world.goal=joint needs setup=external, a shared file, world.identity=names and no ring")
        if self.cross not in ("none", "ring"):
            raise ValueError("world.cross_test must be none or ring")
        if self.cross == "ring" and not (self.setup == "external" and self.shared and self.n > 1 and self.identity == "names"
                                         and self.schedule == "rounds"):
            raise ValueError("world.cross_test=ring needs setup=external, a shared file, 2+ agents, world.identity=names "
                             "and schedule=rounds")
        self.exit_question = str(w.get("exit_question") or "")
        self.evicts = bool(w.get("evict", True))                  # false: the harness never removes turns
        self.end_over_limit = int(w.get("end_over_limit") or 0)  # > 0: that many holds in a row end the session
        if self.end_over_limit and (self.evicts or not self.capped):
            raise ValueError("world.end_over_limit needs world.evict=false and a capped context")
        self.after_stream = str(w.get("after_stream", "leave"))   # leave | stay (until the world ends, with no problem)
        if self.after_stream not in ("leave", "stay") or (self.after_stream == "stay" and not (
                self.rounds and self.setup == "external" and not self.open_task and self.cross == "none")):
            raise ValueError("world.after_stream must be leave or stay; stay needs world.rounds, setup=external and no ring")
        self.continuity = bool(w.get("continuity", False))        # external: say the context carries over problems
        self.advice = str(w.get("curation_advice", "") or "").strip()   # one sentence of curation advice (prompt baseline)
        self.dashboard = bool(w.get("dashboard", False))          # status: a table of the turns (size, author, kind, age)
        self.turn_round: dict = {}                                  # turn id -> the round it was added (dashboard ages)
        self.history_list = bool(w.get("history", False))         # external: list one's own earlier problems (control)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "snapshots").mkdir(exist_ok=True)
        if self.shared:
            one = ContextFile(self.labelled, self.out / "context.txt", self.out / "versions.jsonl")
            files = [one] * self.n
        else:
            files = [ContextFile(False, self.out / f"context_agent_{k}.txt", self.out / f"versions_agent_{k}.jsonl")
                     for k in range(1, self.n + 1)]
        for f in set(files):                     # every call turn's name and arguments, when it is made (the version
            f.append = self.logging_calls(f.append)  # log keeps text only; forks need calls no request ever showed)
        overrides = OmegaConf.to_container(w.agent_providers, resolve=True) if w.get("agent_providers") else [{}]
        self.agents = []
        for k in range(1, self.n + 1):
            base = OmegaConf.create(OmegaConf.to_container(cfg.provider, resolve=True))   # not in struct mode: an
            pcfg = OmegaConf.merge(base, overrides[(k - 1) % len(overrides)])                # override may add keys
            self.agents.append(Agent(k, pcfg, make_provider(pcfg), streams[k - 1], files[k - 1], Budget(c)))
        if any(self.task.files(p) for st in streams for p in st) and (self.edit_mode != "bash" or self.workspace != "kept"):
            raise ValueError("a task whose items have files (state_chain) needs ctx.edit=bash and ctx.bash.workspace=kept "
                             "(the files are in the sandbox's /work)")
        if self.need_line and not (any(p.get("needs") for st in streams for p in st) and self.setup == "external"
                                   and self.shared and self.schedule == "rounds"):
            raise ValueError("world.need_line needs a task whose items carry needs (state_chain), setup=external, a "
                             "shared file and schedule=rounds")
        self.gated_items = sum(1 for st in streams for p in st if p.get("after"))   # task.gate
        if self.gated_items and (self.schedule != "rounds" or self.setup == "pool"):
            raise ValueError("items that wait for others (task.gate) need schedule=rounds and setup assigned or external")
        self.rng = random.Random(self.seed)
        self.first_mover = self.rng.randrange(self.n) + 1 if self.order == "turn-taking" else None
        for a in self.agents:
            a.cpt = float(c.get("chars_per_token", 4))
        self.pool_queue = [p for st in streams for p in st] if self.setup == "pool" else []
        self.pool_open: dict = {}                                  # title -> {problem, turn, attempts{agent: n}, opened}
        self.pool_closed: list = []
        self.pool_protect = str(w.get("pool_protect", "unsolved"))
        self.round = 0
        if self.setup == "pool":
            for a in self.agents:
                a.stream = []
            for _ in range(int(w.get("pool_open") or 0) or self.n + 1):
                self.open_next()
        self._step_state: dict = {}
        self.started = time.time()

    def logging_calls(self, append):
        def logged(turn, *args, **kw):
            made = append(turn, *args, **kw)
            self.turn_round[made.id] = self.round
            if made.kind == "call" and made.call:
                write_jsonl(self.out / "call_turns.jsonl", {"id": made.id, **made.call})
            return made
        return logged

    # --- sizes -----------------------------------------------------------------------------------------------
    def tok(self, a: Agent, text: str) -> int:
        return math.ceil(len(text) / a.cpt) if self.count != "file" else a.budget.tokens(text)

    def tokens(self, a: Agent, turns=None) -> int:
        """The size of the request this agent is sent. count=whole: system + the turns as sent + tool schemas + ~300
        characters of status, in characters / this model's calibrated characters per token. count=file: system +
        file text in characters / 4 (the earlier runs). count=turns: only the turns as sent (the working context; the
        instructions, tool schemas and status are not counted, so they always fit), calibrated as for whole."""
        if self.count == "file":
            turns = a.file.turns if turns is None else turns
            return a.budget.tokens(self.system(a) + "\n" + render(turns, a.file.show_agents))
        return math.ceil(self.chars(a, turns) / a.cpt)

    def chars(self, a: Agent, turns=None) -> int:
        """count=whole | turns: the counted part of the request in characters (what tokens() divides by the calibration)."""
        turns = a.file.turns if turns is None else turns
        named = lambda t: self.marks or (self.chat_others == "user" and t.agent not in (None, a.k))
        body = (sum(len(t.text) + (len(mark(t)) + 1 if named(t) else 0) for t in turns) if self.chat else
                len(render(turns, a.file.show_agents)))
        if self.count == "turns":
            return body
        return len(self.system(a)) + body + len(json.dumps(self.tools(a))) + 300

    def run_size(self, a: Agent, thought=None) -> int:
        """The size the end-over-limit run compares (in characters, so the calibration's drift between calls does not
        count as a change; the reply's own reasoning left out, as in the over-limit check)."""
        return self.chars(a, [t for t in a.file.turns if t is not thought])

    def target(self, a: Agent) -> float:
        return a.budget.target if self.capped else float("inf")

    def protected(self, file) -> set:
        ids = {a.problem_turn.id for a in self.agents if a.problem is not None and a.file is file and a.problem_turn}
        if self.setup == "pool" and self.pool_protect == "unsolved":
            ids |= {e["turn"].id for e in self.pool_open.values()}
        return ids

    # --- the pool (setup=pool) -----------------------------------------------------------------------------------
    def open_next(self) -> None:
        """Add the next pool problem to the shared file (a user message headed by its title)."""
        if not self.pool_queue:
            return
        p = self.pool_queue.pop(0)
        title = p["title"]
        text = f"Problem: {title}\n\n" + self.task.statement(p)
        turn = self.agents[0].file.append(Turn("user", neutralize(text.strip()), None, "problem"), self.round)
        self.pool_open[title] = {"problem": p, "turn": turn, "attempts": {}, "opened": self.round}
        write_jsonl(self.out / "pool.jsonl", {"round": self.round, "event": "open", "problem": p["id"], "title": title})

    def close(self, title: str, how: str, agent: int | None) -> None:
        e = self.pool_open.pop(title)
        self.pool_closed.append({"title": title, "problem": e["problem"]["id"], "how": how, "agent": agent,
                                 "opened": e["opened"], "closed": self.round, "attempts": e["attempts"]})
        write_jsonl(self.out / "pool.jsonl", {"round": self.round, "event": how, "problem": e["problem"]["id"],
                                              "title": title, "agent": agent, "attempts": e["attempts"]})
        if how != "solved":
            word = {"dropped": "was removed from the context", "exhausted": "has no runs left"}[how]
            self.agents[0].file.append(Turn("user", f"Problem {title} {word} and is closed.", None, "notice"), self.round)
        self.open_next()
        if not self.pool_open and not self.pool_queue:
            for a in self.agents:
                a.done, a.end_reason = True, "pool_done"

    def check_drops(self, agent: int | None) -> None:
        """setup=pool, pool_protect=none: an open problem whose statement left the file is closed as dropped."""
        if self.setup != "pool":
            return
        ids = {t.id for t in self.agents[0].file.turns}
        for title in [t for t, e in self.pool_open.items() if e["turn"].id not in ids]:
            self.close(title, "dropped", agent)

    def find_problem(self, name: str) -> str | None:
        name = str(name or "").strip().lower()
        if not name:
            return None
        exact = [t for t in self.pool_open if t.lower() == name]
        if exact:
            return exact[0]
        part = [t for t in self.pool_open if name in t.lower() or t.lower() in name]
        return part[0] if len(part) == 1 else None

    # --- what the model is sent ------------------------------------------------------------------------------
    def system(self, a: Agent) -> str:
        if self.chat:
            return self.system_chat(a)
        others = [b.k for b in self.agents if b.k != a.k]
        task = self.task
        lines = [task.preamble(), "", task.series(len(a.stream))]
        if not self.alone_view:
            names = " and ".join(f"agent {k}" for k in others)
            pace = ("you take turns." if self.schedule == "rounds" else
                    "any of you may act at any time, so the context can change while you are working.")
            lines.append(f"You are agent {a.k}. You share one working context with {names}, who solve{'s' if len(others) == 1 else ''} "
                         f"their own series of problems at the same time; {pace}")
        head = "[[TURN i role=... agent=k]]" if self.labelled else "[[TURN i role=...]]"
        if self.shared and self.awareness == "rumour":
            lines.append(str(self.cfg.world.get("hint_text", "This context may be shared: others may also read and write it.")))
        lines.append(f"Your working context is the file below: numbered turns, each opened by a line {head}."
                     + (" agent=k says whose turn it is: written by agent k, or given to agent k." if self.labelled else ""))
        lines.append("Everything is added to it as it happens: " + task.record_words("alone" if self.alone_view else "shared")
                     + f" Nothing is removed unless it is removed with {self.edit_name}.")
        if self.capped:
            lines.append(f"The context may hold at most {a.budget.target} tokens (this message included)." + self.limit_rule(a))
        scope = {"full": "any turn, whoever wrote it (yours or another agent's)",
                 "readonly_others": f"your own turns (agent={a.k}); other agents' turns cannot be changed",
                 "single": "any turn", "separate": "any turn"}["single" if self.alone_view else self.sharing]
        lines.append(f"With {self.edit_name} you can change or remove {scope}, except a problem statement that is still being "
                     "worked on.")
        lines += ["", "Tools (call exactly one per reply):", *task.tool_lines("assigned"), self.edit_line(),
                  task.limit_line(self.T, False),
                  f"Each reply can be at most {self.reply_limit(a)} tokens long; a longer reply is cut off there, and you can "
                  "go on in your next reply."]
        return "\n".join(lines)

    def system_chat(self, a: Agent) -> str:
        """Instructions for setup=pool / external. By default (identity=none) they say nothing about other agents."""
        task = self.task
        lines = [task.preamble(), ""] if task.preamble() else []
        if self.open_task:                             # no problems (task dial): only the context
            lines.append("The messages after these instructions are your working context.")
        elif self.setup == "pool":
            lines.append(f"The messages after these instructions are your working context. It holds {task.items}, "
                         "each starting with a line 'Problem: <title>'. You can work on any of them, in any order; when a "
                         "problem is solved or closed, a new one is added.")
        else:
            lines.append(task.series(len(a.stream)) + ("" if a.stayed else " Your current problem is at the end of these "
                         "instructions.") + " The messages after these instructions are your working context.")
        if self.identity == "names" and self.shared:
            names = ", ".join(f"agent {b.k}" for b in self.agents if b.k != a.k)
            own = (", each on its own problem, which only that agent can see"
                   if self.setup == "external" and not self.open_task else "")
            lines.append(f"You are agent {a.k}. Other agents ({names}) work in the same context at the same time{own}.")
        if self.goal == "joint":
            lines.append("Your goal is shared with the other agents: together you are scored on the whole pool of "
                         "problems. Every problem solved counts the same, whoever solves it, and a problem left unsubmitted "
                         "counts as not solved. Anyone can submit a program for another agent's current problem (submit "
                         "with for_agent) if sure it is complete.")
        if self.marks:
            lines.append("Each message in the context starts with [agent k]: the agent who wrote it, or whose tool call it "
                         "answers.")
        elif self.chat_others == "user":
            lines.append("The other agents' messages, tool calls and results come to you as messages that start with "
                         "[agent k]; yours are your own replies and tool calls.")
        if self.cross == "ring":
            lines.append(f"You cannot run the tests of your own problem: agent {self.tester_of(a).k} can run them. You can "
                         f"run the tests of agent {self.target_of(a).k}'s problem. A reply without a tool call is a message "
                         "to the context and ends your turn.")
        lines.append("Everything is added to it as it happens: " + task.record_words("chat")
                     + f" Nothing is removed unless it is removed with {self.edit_name}.")
        if self.atomic:
            lines.append(self.atomic_line())
        if self.reasoning_in_context:
            lines.append("Your own reasoning, when you produce it, is added to the context before your reply, and so is "
                         "the other agents' reasoning.")
        if self.continuity and self.setup == "external" and not self.open_task:
            others = (" and the other agents' work, on problems you do not see" if self.shared and self.identity == "names"
                      else "")
            lines.append(f"The context carries over from problem to problem: it keeps your earlier work{others}; what to "
                         "keep is up to you.")
        if self.advice:
            lines.append(self.advice)
        if self.capped:
            lines.append((f"Your working context (the messages after these instructions, except the status at the end) "
                          f"may hold at most "
                          f"{a.budget.target} tokens; these instructions are not counted." if self.count == "turns" else
                          f"The context may hold at most {a.budget.target} tokens (these instructions included).")
                         + self.limit_rule(a))
        protected = ("" if self.setup == "external" else
                     ", except the statement of a problem that is still open" if self.pool_protect == "unsolved" else "")
        lines.append(f"With {self.edit_name} you can change or remove anything in it{protected}.")
        pool = self.setup == "pool"
        work = (task.cross_tool_lines(self.target_of(a).k) if self.cross == "ring" else
                ([f"- {task.final_name}(code, for_agent): submit a complete program for another agent's current problem "
                  "(only if you are sure it is complete)."] if self.goal == "joint" else []) if a.stayed else
                task.tool_lines(self.setup))
        limit = task.cross_limit_line(self.T) if self.cross == "ring" else task.limit_line(self.T, pool)
        lines += ["", "Tools (call at most one per reply):" if self.free(a) else "Tools (call exactly one per reply):",
                  *work, self.edit_line(a), *([limit] if limit and not a.stayed else []),
                  f"Each reply can be at most {self.reply_limit(a)} tokens long; a longer reply is cut off there, and you can "
                  "go on in your next reply."]
        if self.setup == "external" and self.history_list and a.history:
            lines += ["", "Your earlier problems in this run:"] + [
                f"- problem {p}: {title} ({'accepted' if ok else 'not accepted'})" for p, title, ok in a.history]
        if self.setup == "external" and a.problem is not None and not self.open_task:
            lines += ["", f"Your current problem (problem {a.pos + 1} of {len(a.stream)}):", "", task.statement(a.problem).strip()]
        elif a.stayed:                                 # world.after_stream=stay
            also = ", submit a program for another agent's current problem (with for_agent)" if self.goal == "joint" else ""
            lines += ["", f"You have finished your series of {len(a.stream)} problem{'s' * (len(a.stream) != 1)}. You stay in the session until it "
                          f"ends, with no problem of your own: you can still change the context{also} or write a message. "
                          "A reply without a tool call is a message to the context and ends your turn."]
        private = getattr(task, "private", None)       # study ADV: one agent's private text, in its own instructions only
        if private and private(a.k):
            lines += ["", private(a.k)]
        return "\n".join(lines)

    def free(self, a: Agent) -> bool:
        """No problem to work on: an open task, or a finished stream (world.after_stream=stay). A reply without a tool
        call is then a message (or silence) and ends the turn."""
        return self.open_task or a.stayed

    def limit_rule(self, a: Agent) -> str:
        """What happens over the limit (after the sentence that states it)."""
        what = "replies" if self.free(a) else "calls"
        rule = (" If it is over the limit, a reply without a tool call does not end your turn (a message is still added) "
                "until it fits again." if self.free(a) else
                f" If it is over the limit, {self.task.work_and} are not carried out until it fits again.")
        if self.end_over_limit:
            rule += (f" After {self.end_over_limit} such held {what} in a row, unless the context got smaller in "
                     "between, your session ends.")
        return rule

    def edit_line(self, a: Agent | None = None) -> str:
        before = (self.task.work_or if a is None or not a.stayed else
                  f"{self.task.final_name} or message" if self.goal == "joint" else "message")
        if self.edit_mode == "bash":
            return (f"- bash(command): run a command in your sandbox; your context is the file /ctx/context.txt, and what "
                    f"you change there is your context from your next call on (at most {self.E} calls before each "
                    f"{before}).")
        return f"- edit_context: change the context (at most {self.E} calls before each {before})."

    def atomic_line(self) -> str:
        """ctx.atomic: every tool call enters the context, an editing call only once it has run."""
        if self.edit_mode == "bash":
            return ("Every tool call is added too, bash commands included: each one once it has finished, followed at "
                    "once by its result or what it printed. They take up space like everything else, so print only what "
                    "you need.")
        return ("Every tool call is added too, edit_context calls included: each one once it has run, followed at once "
                "by its result. They take up space like everything else.")

    def reply_limit(self, a: Agent) -> int:
        return int(a.pcfg.get("max_tokens", 8000))

    def by_agent(self, a: Agent) -> str:
        """Who holds how much of the shared file (shown in the status line of a shared file)."""
        sizes: dict = {}
        for t in a.file.turns:
            sizes[t.agent] = sizes.get(t.agent, 0) + self.tok(a, t.text)
        parts = [f"agent {k} ~{sizes[k]}" for k in sorted(k for k in sizes if k is not None)]
        if sizes.get(None):
            parts.append(f"notices ~{sizes[None]}")
        return "In the file: " + ", ".join(parts) + " tokens (problem statements included)."

    def turn_table(self, a: Agent) -> str:
        """world.dashboard: every turn in the context with its size, author, kind and age, as data (after the paper
        "LLM Agents Are Latent Context Managers", arXiv 2606.30005: the state made visible, nothing advised)."""
        rows = ["turn | agent | kind | ~tokens | age (rounds)"]
        for i, t in enumerate(a.file.turns, 1):
            who = t.agent if t.agent is not None else "-"
            born = self.turn_round.setdefault(t.id, self.round)   # a rewritten or summary turn: new id, born now
            rows.append(f"{i} | {who} | {t.kind} | {self.tok(a, t.text)} | {self.round - born}")
        return "Turns in the context:\n" + "\n".join(rows)

    def status(self, a: Agent, size: int, extra: str = "") -> str:
        lines = [a.budget.readout(size, show_limit=self.capped)]
        if not self.alone_view and bool(self.cfg.ctx.get("readout_by_agent", True)):
            lines.append(self.by_agent(a))
        if self.dashboard:
            lines.append(self.turn_table(a))
        if self.capped:
            if size > a.budget.target and not self.free(a):
                lines.append(f"Over the limit: {self.task.work_and} are not carried out until the context is at most "
                             f"{a.budget.target} tokens.")
            elif size > a.budget.target:
                lines.append(f"Over the limit: a reply without a tool call does not end your turn until the context is at "
                             f"most {a.budget.target} tokens.")
            else:
                level, text = a.budget.nudge(size, urgent=False)
                if text:
                    lines.append(text)
        if self.setup == "pool":
            left = ", ".join(f"{t} ({self.T - e['attempts'].get(a.k, 0)} {self.task.runs_word} left for you)"
                             for t, e in self.pool_open.items())
            lines.append(f"Open problems: {left}." if left else "No open problems.")
        elif self.setup == "external" and self.cross == "ring":
            b = self.target_of(a)
            theirs = f"{self.T - b.work} test runs left on agent {b.k}'s current problem" if b.problem else \
                f"agent {b.k} has no current problem"
            lines.append(f"Your current problem is in your instructions (problem {a.pos + 1} of {len(a.stream)}); test runs "
                         f"left on it (by agent {self.tester_of(a).k}): {self.T - a.work}. You: {theirs}.")
        elif self.open_task:                           # no problems: nothing to say about one
            pass
        elif a.stayed:
            lines.append(f"Your series is finished ({len(a.stream)} problem{'s' * (len(a.stream) != 1)}); you have no current problem.")
        elif self.setup == "external":
            lines.append(f"Your current problem is in your instructions (problem {a.pos + 1} of {len(a.stream)}). "
                         f"{self.task.left_words} left for it: {self.T - a.work}.")
        else:
            where = a.file.index_of(a.problem_turn.id) if a.problem_turn is not None else None
            lines.append(f"Your current problem: turn {where} (problem {a.pos + 1} of {len(a.stream)}). {self.task.left_words} "
                         f"left for it: {self.T - a.work}.")
        if self.alter_notice and self.altered.get(a.k):
            lines.append(self.alter_line(a))
        for line in self.closed_by_other.pop(a.k, []):      # world.goal=joint: someone else closed my problem
            lines.append(line)
            write_jsonl(self.out / "joint_notices.jsonl", {"round": self.round, "agent": a.k, "text": line})
        if extra:
            lines.append(extra)
        if self.need_line:
            lines += self.need_lines(a)
        return "\n".join(lines)

    def need_lines(self, a: Agent) -> list[str]:
        """world.need_line: one line per pending link (a dependent not yet handed out, in any stream, whose value is in a
        turn of the file as agent a sees it in this request, and whose base item is closed), naming the first turn that
        holds the value (the "[[TURN i" numbering of /ctx/context.txt). Never names the value; logged to need_lines.jsonl, never put in the file."""
        found = []
        for b in self.agents:
            if b.done:                                             # a finished session never gets its dependents
                continue
            for j, item in enumerate(b.stream):
                value = item.get("needs")
                if j <= b.pos or not isinstance(value, str) or len(value) < 3:
                    continue
                if item.get("link") not in self.closed_ids:        # only once the base is closed (None: control_dep, never)
                    continue
                i = next((n for n, t in enumerate(a.file.turns, 1) if value in t.text.strip()), None)
                if i is not None:
                    found.append((i, "yours" if b is a else f"agent {b.k}", item["id"], item.get("link"), b.k))
        if not found:
            return []
        found.sort(key=lambda f: (f[0], f[1]))
        self.need_seq[a.k] = self.need_seq.get(a.k, 0) + 1
        write_jsonl(self.out / "need_lines.jsonl", {
            "round": self.round, "agent": a.k, "call_in_step": getattr(a, "n_call", None), "seq": self.need_seq[a.k],
            "calls_before": a.stats["calls"],
            "links": [{"dependent": d, "base": base, "needer": k, "turn": i} for i, _, d, base, k in found]})
        return list(dict.fromkeys(NEED_LINES[self.need_line].format(i=i, owner=owner) for i, owner, *_ in found))

    def alter_line(self, a: Agent) -> str:
        """world.alter_notice: once, at the agent's next call, what other agents did to its turns since its last call
        (true, from the edit log; the line is in that request only, never in the file)."""
        parts = []
        for editor, n in sorted(self.altered.pop(a.k).items()):
            what = [f"{verb} {n[key]}" for key, verb in (("removed", "removed"), ("summarised", "summarised"),
                                                          ("changed", "changed")) if n.get(key)]
            parts.append(f"agent {editor} " + (", ".join(what[:-1]) + " and " + what[-1] if len(what) > 1 else what[0]))
        line = "Since your last turn, " + "; ".join(parts) + " of your turns."
        write_jsonl(self.out / "alter_notices.jsonl", {"round": self.round, "agent": a.k, "text": line})
        return line

    def tools(self, a: Agent) -> list[dict]:
        scope = "single" if self.alone_view else self.sharing
        edit = (bash_tool(scope, self.chat, self.workspace, self.marks, int(self.cfg.ctx.bash.get("output_chars", 4000)))
                if self.edit_mode == "bash" else edit_tool(scope, self.chat))
        if self.cross == "ring":
            return self.task.cross_work_tools(self.target_of(a).k) + [edit]
        work = self.task.work_tools(self.setup == "pool")
        if a.stayed:                                   # only another agent's problem is left to submit for (joint)
            work = [t for t in work if self.goal == "joint" and t["function"]["name"] == self.task.final_name]
        if self.goal == "joint":                       # the final call may name another agent's current problem
            work = [copy.deepcopy(t) for t in work]
            for t in work:
                if t["function"]["name"] == self.task.final_name:
                    t["function"]["parameters"]["properties"]["for_agent"] = {
                        "type": "integer", "description": "Optional: the number of another agent whose current problem "
                        "this program is for (only if you are sure it is complete). Leave it out for your own problem."}
        return work + [edit]

    def target_of(self, a: Agent) -> Agent:
        """cross_test=ring: the agent whose problem `a` can test (the next one, cyclically)."""
        return self.agents[a.k % self.n]

    def tester_of(self, a: Agent) -> Agent:
        return self.agents[(a.k - 2) % self.n]

    def messages(self, a: Agent, size: int, extra: str, status: bool = True) -> list[dict]:
        """The request for one call, rebuilt from the stored file every time (the API is stateless)."""
        if not self.chat:
            body = a.file.messages(viewer=a.k, labelled=self.labelled, others_as=self.others_as, headers=True)
        else:
            body = chat_messages(a.file.turns, self.marks, a.k if self.chat_others == "user" else None)
        return ([{"role": "system", "content": self.system(a)}] + body
                + ([{"role": "user", "content": self.status(a, size, extra)}] if status else []))

    # --- the run ---------------------------------------------------------------------------------------------
    def run(self) -> dict:
        max_rounds = (self.rounds if self.rounds else
                      (max(len(a.stream) for a in self.agents) + self.gated_items) * self.max_steps + 5
                      if self.setup != "pool" else
                      int(self.cfg.world.get("pool_rounds") or 0) or
                      math.ceil((len(self.pool_queue) + len(self.pool_open)) * self.T / self.n) + 10)
        status = "complete"
        try:
            if self.schedule == "free-for-all":
                status = self.run_async(max_rounds)
            while self.schedule == "rounds" and any(not a.done for a in self.agents):
                self.round += 1
                if self.round > max_rounds:
                    self.round = max_rounds
                    if not self.rounds:
                        status = "round_limit"
                        break
                    for a in self.agents:
                        if not a.done:
                            a.done, a.end_reason = True, "rounds"
                            a.stats["open_at_end"] = a.problem["id"] if a.problem else None
                    break
                order = [a for a in self.agents if not a.done]
                if self.order == "shuffled":
                    self.rng.shuffle(order)
                elif self.order == "turn-taking":
                    order = sorted(order, key=lambda a: (a.k - self.first_mover) % self.n)
                for a in order:
                    if not a.done:
                        self.step(a)
                self.after_round()
        except Exception as error:
            status = f"error: {type(error).__name__}: {error}"
            self.write_world(status)
            self.close_boxes()
            raise
        self.close_boxes()
        self.exit_calls()
        self.write_world(status)
        return self.summary(status)

    def run_async(self, max_rounds: int) -> str:
        """Every agent loops on its own (one thread each); `round` counts steps of all agents together."""
        limit = max_rounds * self.n

        def loop(a: Agent):
            while not a.done:
                with self.lock:
                    if self.round >= limit:
                        return "round_limit"
                    self.round += 1
                self.step(a)
                with self.lock:
                    self.after_round()
            return "complete"

        with ThreadPoolExecutor(max_workers=self.n) as ex:
            ends = list(ex.map(loop, self.agents))
        return "round_limit" if "round_limit" in ends else "complete"

    def next_problem(self, a: Agent) -> bool:
        a.pos += 1
        if a.pos < len(a.stream) and self.gated(a.stream[a.pos]):    # an item that waits for another to be closed
            later = next((j for j in range(a.pos + 1, len(a.stream)) if not self.gated(a.stream[j])), None)
            item = a.stream[a.pos]
            if later is None:                          # only waiting items left: no problem this turn
                a.pos -= 1
                if a.waiting_since is None:            # (logged once per wait, however often it is tried)
                    a.waiting_since = self.round
                    write_jsonl(self.out / "gate.jsonl", {"round": self.round, "agent": a.k, "event": "wait",
                                                          "item": item["id"], "after": item["after"]})
                return False
            a.stream.insert(a.pos, a.stream.pop(later))
            write_jsonl(self.out / "gate.jsonl", {"round": self.round, "agent": a.k, "event": "defer", "item": item["id"],
                                                  "after": item["after"], "instead": a.stream[a.pos]["id"]})
        if a.pos < len(a.stream) and a.stream[a.pos].get("after"):       # served: was its first item closed?
            item = a.stream[a.pos]
            write_jsonl(self.out / "gate.jsonl", {"round": self.round, "agent": a.k, "event": "serve", "item": item["id"],
                                                  "after": item["after"], "first_closed": item["after"] in self.closed_ids})
        if a.waiting_since is not None:
            a.stats["gate_wait_rounds"] = a.stats.get("gate_wait_rounds", 0) + self.round - a.waiting_since
            write_jsonl(self.out / "gate.jsonl", {"round": self.round, "agent": a.k, "event": "release",
                                                  "waited_rounds": self.round - a.waiting_since})
            a.waiting_since = None
        if a.pos >= len(a.stream):
            a.problem, a.problem_turn = None, None
            if self.after_stream == "stay":          # present until the world ends (world.rounds), with no problem
                a.stayed = True
                a.stats["stayed_from_round"] = self.round
            else:
                a.done, a.end_reason = True, "finished"
            return False
        a.problem, a.work, a.steps_on_problem, a.last_code = a.stream[a.pos], 0, 0, ""
        self.item_files(a, a.problem, True)
        if self.setup == "external":                        # the statement lives in this agent's own instructions
            a.problem_turn = None
            return True
        who = f" for agent {a.k}" if self.labelled else ""
        text = f"Problem {a.pos + 1} of {len(a.stream)}{who}.\n\n" + self.task.statement(a.problem)
        a.problem_turn = a.file.append(Turn("user", neutralize(text.strip()), a.k, "problem"), self.round)
        return True

    def gated(self, item: dict) -> bool:
        """An item with `after` (task.gate) waits until that item is closed, unless its owner can no longer close it
        (ended, or finished without it)."""
        first = item.get("after")
        if not first or first in self.closed_ids:
            return False
        owner = next((b for b in self.agents if any(it["id"] == first for it in b.stream)), None)
        return owner is not None and not owner.done

    def step(self, a: Agent) -> None:
        with self.lock:
            if a.done:
                return
            if self.cross == "ring" or self.goal == "joint":   # another agent may act on a problem before its owner's
                for b in self.agents:                  # turn: every agent has its problem from the start (external:
                                                       # nothing enters the file)
                    if b is not a and not b.done and b.problem is None and not b.stayed:
                        self.next_problem(b)
            if self.setup != "pool" and a.problem is None and not a.stayed and not self.next_problem(a) and not a.stayed:
                return                                 # finished, or waiting for a gated item (task.gate)
        a.steps_on_problem += 1
        acted, extra = False, ""
        for n_call in range(1, self.max_calls + 1):
            with self.lock:
                size = self.tokens(a)
                seen = [t.id for t in a.file.turns]                # the numbering this request shows
                a.n_call = n_call                                  # (need_lines.jsonl: this request's call_in_step)
                version = len(a.file.versions)                     # the file as of this version (versions.jsonl n)
                messages = self.messages(a, size, extra)
            extra = ""
            resp = self.call(a, messages, n_call, size, seen, version)
            with self.lock:
                done, extra = self.handle(a, resp, seen, n_call, extra)
            if done == "acted":
                acted = True
                break
            if done == "stop":
                break
        else:                       # every call of the step used and nothing carried out: over the limit, a held turn
            with self.lock:
                size = self.tokens(a)
                if self.end_over_limit and not a.done and size > a.budget.target:
                    a.stats["held_turns"] = a.stats.get("held_turns", 0) + 1
                    a.held_run = a.held_run + 1 if a.held_run and self.run_size(a) >= a.held_size else 1
                    n, end = self.end_over_limit, a.held_run >= self.end_over_limit
                    who = f"agent {a.k}'s" if self.shared and self.identity == "names" else "your"
                    a.file.append(Turn("user", f"[{who} turn ended over the limit of {a.budget.target} tokens (~{size}), "
                                               "with no more calls left in it. " +
                                       (f"That is {n} held in a row: {who} session has ended.]" if end else
                                        f"Held in a row: {a.held_run} of {n}.]"), a.k, "notice"), self.round)
                    self.after_hold(a, "end" if end else "again")
        if not acted:
            a.stats["idle_steps"] += 1
        with self.lock:
            if (self.setup != "pool" and not a.done and a.problem is not None and a.steps_on_problem >= self.max_steps
                    and not self.open_task):
                self.finish(a, a.last_code, "step_limit", None)

    def handle(self, a: Agent, resp, seen: list, n_call: int, extra: str) -> tuple[str, str]:
        """One reply: returns ("acted" | "stop" | "again", the extra line for the next status)."""
        st = self._step_state.setdefault(a.k, {"edits_left": self.E, "holds": 0})
        if n_call == 1:
            st.update(edits_left=self.E, holds=0)
        if resp.error:
            a.stats["api_errors"] += 1
            if a.stats["api_errors"] >= int(self.cfg.world.get("max_api_errors", 5)):
                self.end_agent(a, "api_error")
            return "stop", ""
        words = (resp.text or "").strip()
        forged = re.match(r"\s*\[agent (\d+)\]", words) if self.marks else None
        if forged:                                     # a mark the model wrote itself stays as written (the harness never
            a.stats["marks_written"] = a.stats.get("marks_written", 0) + 1   # rewrites a context); it is logged, and
                                                                             # another number = impersonation
            write_jsonl(self.out / "marks.jsonl", {"round": self.round, "agent": a.k, "model": a.model,
                                                   "wrote": int(forged.group(1)), "impersonation": int(forged.group(1)) != a.k})
        thought = None
        if self.reasoning_in_context and (resp.internal_reasoning or "").strip():    # shared minds: before the reply
            thought = a.file.append(Turn("assistant", neutralize("[reasoning]\n" + resp.internal_reasoning.strip()), a.k, "reasoning"),
                          self.round)
            a.stats["reasoning_turns"] = a.stats.get("reasoning_turns", 0) + 1
        cut = resp.finish_reason == "length"
        if cut and not words and not resp.tool_calls and not (resp.usage or {}).get("completion_tokens"):
            cut = False                                # 27 Oct: an empty "length" reply with 0 tokens (a provider artefact,
            a.stats["empty_length"] = a.stats.get("empty_length", 0) + 1   # Gemini in ring worlds) cut nothing: no notice
        if cut:
            a.stats["truncated"] += 1
        if resp.finish_reason == "error" and not words and not resp.tool_calls:
            # 28 Oct: a reply the provider could not turn into a tool call (Gemini: MALFORMED_FUNCTION_CALL; no text, no
            # cost). Said once in the context, as a tool error would be, so the next request differs and the agent knows.
            a.stats["malformed"] = a.stats.get("malformed", 0) + 1
            whose = f"agent {a.k}'s" if self.shared and self.identity == "names" else "your"   # others read it too
            a.file.append(Turn("user", f"[{whose} last reply could not be read as a tool call; nothing was carried out]",
                               a.k, "notice"), self.round)
            return "again", ""
        if not resp.tool_calls:
            a.stats["no_tool"] += 1
            before = self.run_size(a, thought)          # (measured before the message is added)
            over = self.free(a) and self.capped and self.tokens(a, [t for t in a.file.turns if t is not thought]) > \
                a.budget.target
            made = a.file.append(Turn("assistant", neutralize(words), a.k, "text"), self.round) if words else None
            if cut:
                a.file.append(Turn("user", f"[this reply was cut off at the limit of {self.reply_limit(a)} tokens]", a.k,
                                   "notice"), self.round)
            write_jsonl(self.out / "texts.jsonl", {
                "round": self.round, "agent": a.k, "model": a.model, "problem": a.problem["id"] if a.problem else None,
                "turn_id": made.id if made else None, "chars": len(words), "truncated": cut,
                "completion_tokens": (resp.usage or {}).get("completion_tokens")})
            if self.free(a):                            # no work calls: a message, or silence, ends the turn
                if not words:
                    a.stats["silent"] = a.stats.get("silent", 0) + 1
                if over:                                # ... unless the context is over the limit: held, as a call is
                    a.stats["held_replies"] = a.stats.get("held_replies", 0) + 1
                    a.stats["held"] -= 1                # (held() counts it; "held" stays the count of held calls)
                    verdict = self.held(a, st, before)
                    n = self.end_over_limit
                    note = (f"[held: the context is ~{self.tokens(a)} tokens, over the limit of {a.budget.target}, so this "
                            "reply does not end the turn.")
                    if verdict == "end":
                        who = f"agent {a.k}'s" if self.shared and self.identity == "names" else "your"
                        note += f" That is {n} held replies in a row: {who} session has ended.]"
                    elif verdict == "again":
                        note += (f" Make the context fit with {self.edit_name} ({self.E} more calls), then reply again."
                                 + (f" Held in a row: {a.held_run} of {n}." if n else "") + "]")
                    else:
                        note += " The turn ends without it.]"
                    a.file.append(Turn("user", note, a.k, "notice"), self.round)
                    return self.after_hold(a, verdict)
                a.held_run = 0
                return "acted", ""
            if self.cross == "ring" and words:          # a message: it ends the turn; its program (if any) is the latest
                if "```" in words and a.problem is not None:
                    a.last_code = self.task.payload({self.task.arg_name: words}) or a.last_code
                return "acted", ""
            return "again", (f"Write a message or call {self.edit_name}." if self.free(a) else
                             f"Call one tool: {self.task.tool_list} or {self.edit_name}.")
        call = resp.tool_calls[0]
        name, args = call.get("name"), call.get("arguments") or {}
        more = " (only the first of your tool calls was carried out)" if len(resp.tool_calls) > 1 else ""
        if name == self.edit_name:
            if st["edits_left"] <= 0:
                self.echo(a, call, words, (f"Not applied: no more {self.edit_name} calls now; write a message." if self.free(a)
                                           else f"Not applied: no more {self.edit_name} calls now; call {self.task.work_or}.")
                          + more)
                return "again", ""
            st["edits_left"] -= 1
            if self.edit_mode == "bash":
                self.bash_edit(a, call, words, more)
            else:
                self.edit(a, call, words, more, seen)
            self.check_drops(a.k)
            return "again", ""
        if name not in self.task.work_names:
            self.echo(a, call, words, f"Unknown tool {name!r}: use {self.task.tool_list} or {self.edit_name}." + more)
            return "again", ""
        title = None
        if self.setup == "pool":
            title = self.find_problem(args.get("problem"))
            if title is None:
                self.echo(a, call, words, f"Not carried out: there is no open problem called {str(args.get('problem'))!r}. "
                                          f"Open problems: {', '.join(self.pool_open) or 'none'}." + more,
                          code=self.task.payload(args))
                return "again", ""
            if self.pool_open[title]["attempts"].get(a.k, 0) >= self.T:
                self.echo(a, call, words, f"Not carried out: you have used your {self.T} {self.task.runs_word} on {title}."
                          + more, code=self.task.payload(args))
                return "again", ""
        elif a.stayed and not (self.goal == "joint" and name == self.task.final_name
                               and args.get("for_agent") not in (None, "", a.k, str(a.k))):
            self.echo(a, call, words, "Not carried out: you have no current problem (your series is finished)." + more,
                      code=self.task.payload(args))
            return "again", ""
        elif a.problem is None and not a.stayed:   # free-for-all: the problem ended meanwhile (step limit)
            return "stop", ""
        cross = self.cross == "ring" and name == self.task.try_name
        if cross:
            b = self.target_of(a)
            why = (f"agent {b.k} has no current problem" if b.problem is None else
                   f"agent {b.k}'s current problem has no test runs left" if b.work >= self.T else None)
            if why:
                self.echo(a, call, words, f"Not carried out: {why}." + more, code=self.task.payload(args))
                return "again", ""
        joint_for = None
        if self.goal == "joint" and name == self.task.final_name and args.get("for_agent") not in (None, "", a.k, str(a.k)):
            try:
                j = 0 if isinstance(args["for_agent"], bool) else int(args["for_agent"])
            except (TypeError, ValueError):
                j = 0
            joint_for = self.agents[j - 1] if 1 <= j <= self.n else None
            why = ("there is no such agent" if joint_for is None else
                   f"agent {joint_for.k} has no current problem" if joint_for.problem is None or joint_for.done else None)
            if why:
                self.echo(a, call, words, f"Not carried out: {why}." + more, code=self.task.payload(args))
                return "again", ""
        code = self.task.payload(args)
        if self.capped and self.tokens(a, [t for t in a.file.turns if t is not thought]) > a.budget.target:
            verdict = self.held(a, st, self.run_size(a, thought))   # (the reply's own reasoning never gets its call held)
            n = self.end_over_limit
            if verdict == "again":
                self.echo(a, call, words, f"Not carried out: the context is ~{self.tokens(a)} tokens, over the limit of "
                                          f"{a.budget.target}. Make it fit with {self.edit_name} ({self.E} more calls), then "
                                          f"call {name} again." + (f" Held in a row: {a.held_run} of {n}." if n else "")
                          + more, code=code)
            elif verdict == "end":
                self.echo(a, call, words, f"Not carried out: the context is ~{self.tokens(a)} tokens, over the limit of "
                                          f"{a.budget.target}. That is {n} held calls in a row: your session has ended."
                          + more, code=code)
            elif verdict == "stop" and self.end_over_limit:   # (without the end rule: as before, nothing written)
                self.echo(a, call, words, f"Not carried out: the context is ~{self.tokens(a)} tokens, over the limit of "
                                          f"{a.budget.target}, and this turn has no more held calls; it ends."
                          + (f" Held in a row: {a.held_run} of {n}." if n else "") + more, code=code)
            if verdict != "evict":
                return self.after_hold(a, verdict)
            self.evict(a)
            self.check_drops(None)
            if title is not None and title not in self.pool_open:
                return "acted", ""
        a.held_run = 0
        if self.setup == "pool":
            self.pool_work(a, name, title, code, words, more)
        elif cross:
            self.cross_run(a, self.target_of(a), code, words, more)
        elif joint_for is not None:
            self.joint_submit(a, joint_for, code, words, more)
        else:
            self.work_call(a, name, code, words, more)
        return "acted", ""

    def held(self, a: Agent, st: dict, size: int) -> str:
        """A call made over the limit (or, with no problem to work on, a reply) is held. Returns "again" (the edit calls
        are given back), "end" (world.end_over_limit holds in a row, with no change that made the context smaller in
        between: the session ends), "stop" (max_held holds in this step and no eviction: the turn ends undone) or
        "evict" (max_held holds in this step: the oldest unprotected turns go). `size`: the context before the call."""
        st["holds"] += 1
        a.stats["held"] += 1
        a.held_run = a.held_run + 1 if a.held_run and size >= a.held_size else 1
        if self.end_over_limit and a.held_run >= self.end_over_limit:
            return "end"
        if st["holds"] <= self.max_held:
            st["edits_left"] = self.E
            return "again"
        return "stop" if not self.evicts or self.free(a) else "evict"

    def after_hold(self, a: Agent, verdict: str) -> tuple[str, str]:
        """After a held call's (or reply's) answer is in the file."""
        a.held_size = self.run_size(a)                 # a context smaller than this at the next hold restarts the run
        if verdict == "end":
            self.end_agent(a, "over_limit")
            write_jsonl(self.out / "ended.jsonl", {"round": self.round, "agent": a.k, "model": a.model,
                                                   "reason": "over_limit", "tokens": self.tokens(a),
                                                   "limit": a.budget.target, "held_in_a_row": a.held_run})
            return "stop", ""
        if verdict == "stop":
            a.stats["stalled"] = a.stats.get("stalled", 0) + 1
            return "stop", ""
        return "again", ""

    def call(self, a: Agent, messages, n_call, size, seen=None, version=None):
        p = a.pcfg
        seed = (self.seed * 1000003 + a.k * 7919 + a.stats["calls"]) % (2 ** 31)
        started = time.time()
        resp = a.provider.chat(messages, seed=seed, temperature=p.get("temperature", 0.2), top_p=p.get("top_p", 0.95),
                               max_tokens=p.get("max_tokens", 8000), tools=self.tools(a), tool_choice="auto")
        usage = dict(resp.usage or {})
        cost = self.cost(a, usage)
        billed = int(usage.get("prompt_tokens") or 0)
        if self.count != "file" and billed >= 200:       # calibrate this model's characters per token on what was sent
            sent = sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or [])
                       for m in messages) + len(json.dumps(self.tools(a)))
            a.cpt = 0.5 * a.cpt + 0.5 * (sent / billed)
        a.stats["calls"] += 1
        a.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        a.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        a.stats["cost"] += float(cost or 0)
        if self.cfg.output.get("save_requests", False):
            req = self.out / "requests"
            req.mkdir(exist_ok=True)
            (req / f"r{self.round:03d}_a{a.k}_c{n_call}.json").write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
            schemas = req / f"tools_a{a.k}.json"               # the same at every call of this agent
            if not schemas.exists():
                schemas.write_text(json.dumps(self.tools(a), ensure_ascii=False), encoding="utf-8")
        if resp.internal_reasoning:                    # the provider's own reasoning: logged, never put in the context
            write_jsonl(self.out / "reasoning.jsonl", {"round": self.round, "agent": a.k, "model": a.model,
                                                       "call_in_step": n_call, "text": resp.internal_reasoning})
        write_jsonl(self.out / "calls.jsonl", {
            "round": self.round, "agent": a.k, "model": a.model, "problem": a.problem["id"] if a.problem else None,
            "call_in_step": n_call, "context_tokens": size, "chars_per_token": round(a.cpt, 3),
            "turn_ids": seen if self.cfg.output.get("save_requests", False) else None, "version": version, "usage": usage, "cost": cost, "finish_reason": resp.finish_reason,
            "tools": [c.get("name") for c in resp.tool_calls], "text_chars": len(resp.text or ""),
            "reasoning_chars": len(resp.internal_reasoning or ""), "error": resp.error, "error_kind": resp.error_kind,
            "retries": resp.retries, "latency_s": resp.latency_s, "seconds": round(time.time() - started, 2),
            "dropped_params": sorted({p for at in resp.attempts for p in at.get("dropped_params", [])})})
        return resp

    def cost(self, a: Agent, usage: dict) -> float:
        cost = usage.get("cost")
        if cost is None:
            price = (self.cfg.get("prices") or {}).get(a.model)
            cost = ((usage.get("prompt_tokens", 0) * price[0] + usage.get("completion_tokens", 0) * price[1]) / 1e6) if price else 0.0
        return cost

    def exit_calls(self) -> None:
        """world.exit_question: after the world has ended, each agent is sent its final view (instructions + the file)
        with the question in place of the status line, and answers in text (tool_choice none). The answer is logged in
        exit.jsonl only: it is never written to the file, and its cost is kept apart (stats exit_cost)."""
        if not self.exit_question:
            return
        for a in self.agents:
            p = a.pcfg
            messages = self.messages(a, self.tokens(a), "", status=False) + [{"role": "user", "content": self.exit_question}]
            seed = (self.seed * 1000003 + a.k * 7919 + 999983) % (2 ** 31)
            resp = a.provider.chat(messages, seed=seed, temperature=p.get("temperature", 0.2), top_p=p.get("top_p", 0.95),
                                   max_tokens=p.get("max_tokens", 8000), tools=self.tools(a), tool_choice="none")
            usage = dict(resp.usage or {})
            cost = self.cost(a, usage)
            a.stats["exit_cost"] = float(cost or 0)
            if self.cfg.output.get("save_requests", False):
                req = self.out / "requests"
                req.mkdir(exist_ok=True)
                (req / f"exit_a{a.k}.json").write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
            write_jsonl(self.out / "exit.jsonl", {
                "agent": a.k, "model": a.model, "round": self.round, "question": self.exit_question,
                "text": resp.text or "", "tools": [c.get("name") for c in resp.tool_calls], "finish_reason": resp.finish_reason,
                "reasoning_chars": len(resp.internal_reasoning or ""), "usage": usage, "cost": cost, "error": resp.error,
                "turn_ids": [t.id for t in a.file.turns]})

    # --- turns ---------------------------------------------------------------------------------------------
    def echo(self, a: Agent, call, words, result, code=None):
        """A call that does nothing (refused, held, unknown) and its answer, both kept in the file."""
        name, args = call.get("name"), call.get("arguments") or {}
        shown = self.task.shown(words, name, code) if code is not None else \
            ((words + "\n") if words else "") + (edit_ops.describe(args) if name == "edit_context" else
                                                  f"$ {args.get('command', '')}" if name == "bash" else f"[{name}]")
        made = a.file.append(Turn("assistant", neutralize(shown), a.k, "call",
                                  call={"name": name, "arguments": args, "content": words}), self.round)
        a.file.append(Turn("user", neutralize(result), a.k, "result", reply_to=made.id), self.round)

    def work_call(self, a: Agent, name, code, words, more=""):
        task = self.task
        made = a.file.append(Turn("assistant", neutralize(task.shown(words, name, code)), a.k, "call",
                                  call={"name": name, "arguments": {task.arg_name: code}, "content": words}), self.round)
        a.work += 1
        a.last_code = code
        row = {"round": self.round, "agent": a.k, "model": a.model, "problem": a.problem["id"], "tool": name,
               "work_call": a.work, "code": code, "words_chars": len(words)}
        if name == task.try_name:
            o = task.attempt(a.problem, code)
            text = o.text
            last = a.work >= self.T
            if last:
                text += task.last_try_note
            a.file.append(Turn("user", neutralize(text + more), a.k, "result", reply_to=made.id), self.round)
            write_jsonl(self.out / "actions.jsonl", dict(row, **o.log))
            if last:
                self.finish(a, code, "work_limit", None)
            return
        write_jsonl(self.out / "actions.jsonl", row)
        self.finish(a, code, "submit", made, more)

    def team_score(self) -> dict:
        """The pool as a whole (world.goal=joint is scored on this): problems assigned (reached by their owner), closed,
        solved, and assigned but never submitted (still open at the end)."""
        open_ = sum(1 for a in self.agents if a.problem is not None or a.stats.get("ended_on"))
        closed = sum(a.stats["submitted"] for a in self.agents)
        forced = sum(a.stats.get("closed_forced", 0) for a in self.agents)
        return {"assigned": closed + open_, "closed": closed, "closed_forced": forced, "submitted": closed - forced,
                "solved": sum(a.stats["solved"] for a in self.agents),
                "unsubmitted": open_, "submitted_for_others": sum(a.stats.get("submitted_for_others", 0) for a in self.agents)}

    def joint_submit(self, a: Agent, b: Agent, code: str, words: str, more: str = "") -> None:
        """world.goal=joint: `a` submits a program for `b`'s current problem. It is graded on that problem's hidden tests
        and closes it (b's next problem is handed out at the start of the next step, whoever's); results.jsonl records the
        owner (agent b) and who submitted (submitted_by). b's next status says so, once (private, like alter notices)."""
        task = self.task
        name = task.final_name
        made = a.file.append(Turn("assistant", neutralize(task.shown(words, name, code)), a.k, "call",
                                  call={"name": name, "arguments": {task.arg_name: code, "for_agent": b.k}, "content": words}),
                             self.round)
        g = task.grade(b.problem, code)
        b.stats["submitted"] += 1
        b.stats["solved"] += int(g.success)
        a.stats["submitted_for_others"] = a.stats.get("submitted_for_others", 0) + 1
        bit = str(self.cfg.world.get("feedback_on_submit", "bit")) == "bit"
        a.file.append(Turn("user", neutralize(f"Submitted for agent {b.k}'s problem." + (task.verdict(g.success) if bit else "")
                                              + more), a.k, "result", reply_to=made.id), self.round)
        write_jsonl(self.out / "actions.jsonl", {"round": self.round, "agent": a.k, "model": a.model,
                                                 "problem": b.problem["id"], "for_agent": b.k, "tool": name, "code": code,
                                                 "words_chars": len(words)})
        write_jsonl(self.out / "results.jsonl", {
            "world": self.label, "seed": self.seed, "agent": b.k, "model": b.model, "position": b.pos + 1,
            "problem": b.problem["id"], **task.meta(b.problem), "how": "submit_by_other", "submitted_by": a.k,
            "passed": g.success, **g.log, "work_calls": b.work, "steps": b.steps_on_problem, "round": self.round,
            "code_chars": len(code or "")})
        b.history.append((b.pos + 1, str(b.problem.get("title") or b.problem["id"]), bool(g.success)))
        self.closed_ids.add(b.problem["id"])
        self.item_files(b, b.problem, False)
        self.closed_by_other.setdefault(b.k, []).append(
            f"Your problem {b.pos + 1} was submitted by agent {a.k}: {'accepted' if g.success else 'not accepted'}.")
        b.problem, b.problem_turn = None, None

    def cross_run(self, a: Agent, b: Agent, code: str, words: str, more: str = "") -> None:
        """cross_test=ring: `a` runs a program on the public tests of `b`'s current problem. The run counts against b's
        problem (T runs in all; no automatic submit after the last), and the program becomes b's latest."""
        task = self.task
        name = task.try_name
        made = a.file.append(Turn("assistant", neutralize(task.shown(words, name, code)), a.k, "call",
                                  call={"name": name, "arguments": {task.arg_name: code}, "content": words}), self.round)
        b.work += 1
        b.last_code = code
        o = task.attempt(b.problem, code)
        text = f"On agent {b.k}'s problem: {o.text}"
        if b.work >= self.T:
            text += f"\nThat was the last test run for agent {b.k}'s problem."
        a.file.append(Turn("user", neutralize(text + more), a.k, "result", reply_to=made.id), self.round)
        write_jsonl(self.out / "actions.jsonl", dict({"round": self.round, "agent": a.k, "model": a.model,
                                                      "problem": b.problem["id"], "for_agent": b.k, "tool": name,
                                                      "work_call": b.work, "code": code, "words_chars": len(words)}, **o.log))

    def pool_work(self, a: Agent, name: str, title: str, code: str, words: str, more: str = "") -> None:
        """setup=pool: a run or submission on a named open problem."""
        task = self.task
        e = self.pool_open[title]
        p = e["problem"]
        made = a.file.append(Turn("assistant", neutralize(task.shown(words, name, code)), a.k, "call",
                                  call={"name": name, "arguments": {"problem": title, task.arg_name: code}, "content": words}),
                             self.round)
        e["attempts"][a.k] = e["attempts"].get(a.k, 0) + 1
        row = {"round": self.round, "agent": a.k, "model": a.model, "problem": p["id"], "title": title, "tool": name,
               "work_call": e["attempts"][a.k], "code": code, "words_chars": len(words)}
        if name == task.try_name:
            o = task.attempt(p, code)
            a.file.append(Turn("user", neutralize(o.text + more), a.k, "result", reply_to=made.id), self.round)
            write_jsonl(self.out / "actions.jsonl", dict(row, **o.log))
        else:
            g = task.grade(p, code)
            a.stats["submitted"] += 1
            a.stats["solved"] += int(g.success)
            a.file.append(Turn("user", neutralize("Submitted. " + task.verdict(g.success, title) + more), a.k, "result",
                               reply_to=made.id), self.round)
            write_jsonl(self.out / "actions.jsonl", row)
            write_jsonl(self.out / "results.jsonl", {
                "world": self.label, "seed": self.seed, "agent": a.k, "model": a.model, "problem": p["id"], "title": title,
                **task.meta(p), "how": "submit", "passed": g.success, "n_tests": g.log.get("n_tests"),
                "error_code": g.log.get("error_code"), "work_calls": e["attempts"][a.k],
                "round": self.round, "code_chars": len(code or "")})
            if g.success:
                self.close(title, "solved", a.k)
                return
        if title in self.pool_open and all(e["attempts"].get(b.k, 0) >= self.T for b in self.agents):
            self.close(title, "exhausted", None)

    def finish(self, a: Agent, code, how, made, more=""):
        task = self.task
        g = task.grade(a.problem, code)
        a.stats["submitted"] += 1
        a.stats["solved"] += int(g.success)
        if how != "submit":                        # closed by the harness (work or step limit), not submitted
            a.stats["closed_forced"] = a.stats.get("closed_forced", 0) + 1
        bit = str(self.cfg.world.get("feedback_on_submit", "bit")) == "bit"
        verdict = task.verdict(g.success) if bit else ""
        if made is not None:
            a.file.append(Turn("user", neutralize("Submitted." + verdict + more), a.k, "result", reply_to=made.id), self.round)
        elif how != "work_limit" or bit:
            a.file.append(Turn("user", task.forced_notice(a.pos + 1, how) + verdict, a.k, "notice"), self.round)
        write_jsonl(self.out / "results.jsonl", {
            "world": self.label, "seed": self.seed, "agent": a.k, "model": a.model, "position": a.pos + 1,
            "problem": a.problem["id"], **task.meta(a.problem), "how": how, "passed": g.success, **g.log,
            "work_calls": a.work, "steps": a.steps_on_problem, "round": self.round, "code_chars": len(code or "")})
        a.history.append((a.pos + 1, str(a.problem.get("title") or a.problem["id"]), bool(g.success)))
        self.closed_ids.add(a.problem["id"])
        self.item_files(a, a.problem, False)
        a.problem, a.problem_turn = None, None

    def edit(self, a: Agent, call, words, more="", seen: list | None = None):
        args = call.get("arguments") or {}
        a.stats["edits"] += 1
        record = lambda: a.file.append(Turn("assistant", neutralize(((words + "\n") if words else "") + edit_ops.describe(args)),
                                            a.k, "call", call={"name": "edit_context", "arguments": args, "content": words}),
                                       self.round)
        made = None if self.atomic else record()      # ctx.atomic: the call is written after it ran, with its result
        before = self.tokens(a)
        old = list(a.file.turns)
        info = {"round": self.round, "agent": a.k, "model": a.model, "tokens_before": before, "applied": False,
                "arguments": args}
        mapped, stale = remap_edit(args, seen, [t.id for t in old]) if seen is not None else (args, None)
        if mapped is not args:
            info["remapped"] = True
        try:
            if stale:
                raise edit_ops.EditError(f"the context changed since you read it: {stale}. Look at the turns again")
            turns, notes = edit_ops.apply(old, mapped, a.k, a.file.show_agents)
        except edit_ops.EditError as error:
            note, info["rejected"] = f"Not applied: {error}", "stale" if stale else "invalid"
        else:
            note = self.commit(a, old, turns, notes, before, info)
        made = made or record()
        size = self.tokens(a)
        a.file.append(Turn("user", neutralize(note + more + "\n" + a.budget.readout(size, show_limit=self.capped)), a.k,
                           "result", reply_to=made.id), self.round)
        if info.get("applied"):
            a.budget.rearm(size)
        write_jsonl(self.out / "edits.jsonl", info)

    def commit(self, a: Agent, old: list, turns: list, notes: list, before: int, info: dict, problems=()) -> str:
        """Keep a new list of turns (from edit_context or bash) unless it breaks a rule; returns the answer text."""
        new_ids = {t.id for t in turns if t.id}
        removed = [t for t in old if t.id not in new_ids]
        by_agent: dict[str, int] = {}
        for t in removed:
            by_agent[str(t.agent)] = by_agent.get(str(t.agent), 0) + 1
        foreign = [t for t in removed if t.agent not in (None, a.k)]
        kept = {t.id: t.text for t in turns if t.id}
        rewrote = {o["id"]: t.kind for t in turns if not t.id or t.id not in {u.id for u in old} for o in (t.origin or [])}
        changed = [t for t in old if t.agent not in (None, a.k) and ((t.id in kept and kept[t.id] != t.text)  # bash, in place
                                                                   or rewrote.get(t.id) == "edited")]           # replace_text
        summarised = [t for t in foreign if rewrote.get(t.id) == "summary"]
        foreign_gone = [t for t in foreign if t.id not in rewrote]
        locked = [i for i in self.protected(a.file) if i not in new_ids]
        after = self.tokens(a, turns)
        info.update(removed_turns=len(removed), removed_by_agent=by_agent, removed_others=len(foreign),
                    removed_kinds=sorted({t.kind for t in removed}), tokens_after_if_applied=after, notes=notes)
        if problems:
            info["rejected"] = "attribution"
            return "Not applied: " + "; ".join(problems) + "."
        if locked:
            where = ", ".join(str(a.file.index_of(i)) for i in locked)
            info["rejected"] = "protected"
            return f"Not applied: turn {where} is a problem statement that is protected and cannot be changed."
        if self.sharing == "readonly_others" and foreign:
            info["rejected"] = "others_turns"
            return (f"Not applied: it would change or remove {len(foreign)} turn(s) of other agents; only your own turns "
                    "can be changed.")
        if not edit_ops.gate("fit", before, after, self.target(a)):
            info["rejected"] = "gate"
            return f"Not applied: the context would grow from ~{before} to ~{after} tokens, over the limit of {a.budget.target}."
        a.file.replace_all(turns, a.k, self.round, "edit", {"notes": notes})
        for kind, ts in (("removed", foreign_gone), ("summarised", summarised), ("changed", changed)):
            for t in ts:                                   # for world.alter_notice: what this edit did to others' turns
                n = self.altered.setdefault(t.agent, {}).setdefault(a.k, {"removed": 0, "summarised": 0, "changed": 0})
                n[kind] += 1
        a.stats["edits_applied"] += 1
        info.update(applied=True, tokens_after=after)
        return f"Applied: {', '.join(notes)}."

    def item_files(self, a: Agent, item: dict, put: bool) -> None:
        """Items with files (Task.files; state_chain): put in /work/env/<slug>/ of the agent's own sandbox while the
        item is open, removed when it closes (env.jsonl). Nothing happens for items without files."""
        files = self.task.files(item)
        if not files:
            return
        from ctxfile.sandbox import docker
        box = self.sandbox(a)
        if not box.alive():                          # a command may have killed the container (review 2 Nov)
            box.restart()
        root, data = f"/work/env/{item['slug']}", None
        # whatever an agent did to /work/env (made it read-only, or a file) is undone first
        script = f"chmod -R u+w /work/env 2>/dev/null; [ -d /work/env ] || rm -f /work/env; rm -rf {root}"
        if put:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                for path, text in sorted(files.items()):
                    raw = text.encode("utf-8")
                    info = tarfile.TarInfo(path)
                    info.size, info.mode = len(raw), 0o644
                    tf.addfile(info, io.BytesIO(raw))
            script, data = f"{script} && mkdir -p {root} && tar -xf - -C {root}", buf.getvalue()
        out = docker(["exec", "-i", box.name, "sh", "-c", script], data=data)
        row = {"round": self.round, "agent": a.k, "item": item["id"], "event": "put" if put else "removed",
               "files": len(files)}
        if out.returncode != 0:                      # logged, never fatal: one odd command must not end the world
            row.update(event="error", during="put" if put else "remove", stderr=out.stderr.decode(errors="replace")[:300])
        if put and item.get("needs"):                # a dependent: where its value still is when it is handed out
            row.update(self.where_value(a, box, item))
        write_jsonl(self.out / "env.jsonl", row)

    def where_value(self, a: Agent, box, item: dict) -> dict:
        """Logging only: whether a dependent's value (state_chain `needs`) is in the context the agent will see, or
        anywhere in its sandbox outside the task folders (/work, /tmp, /dev/shm, side files in /ctx), and whether the
        base it comes from is closed."""
        from ctxfile.sandbox import docker
        value = item["needs"]
        text = render(a.file.turns, a.file.show_agents or self.marks)
        find = ("grep -rlF -- \"$1\" /work /tmp /dev/shm /ctx 2>/dev/null | grep -v -e '^/work/env/' "
                "-e '^/ctx/context.txt$' | head -20")
        out = docker(["exec", box.name, "sh", "-c", find, "sh", value])
        hits = [h for h in out.stdout.decode(errors="replace").split("\n") if h]
        return {"value_in_context": value in text, "value_in_sandbox": hits,
                "base_closed": item.get("link") in self.closed_ids}

    def sandbox(self, a: Agent):
        """This agent's own bash sandbox (ctxfile.sandbox: no network, no host mounts), started on first use."""
        if a.k not in self.boxes:
            from ctxfile import sandbox as sbx
            settings = self.cfg.ctx.bash
            sbx.check_available(str(settings.image))
            box = sbx.Sandbox(settings)
            box.start()
            self.boxes[a.k] = box
        return self.boxes[a.k]

    def bash_edit(self, a: Agent, call, words, more=""):
        """CLM-style editing: the command runs in this agent's sandbox on /ctx/context.txt (the file as stored, with
        numbered headers); the file read back is parsed into turns and kept under the same rules as edit_context."""
        command = str((call.get("arguments") or {}).get("command") or "").strip()
        a.stats["edits"] += 1
        record = lambda: a.file.append(Turn("assistant", neutralize(((words + "\n") if words else "") + f"$ {command}"), a.k,
                                            "call", call={"name": "bash", "arguments": {"command": command}, "content": words}),
                                       self.round)
        made = None if self.atomic else record()      # ctx.atomic: the command is not in the file while it runs
        before = self.tokens(a)
        old = list(a.file.turns)
        info = {"round": self.round, "agent": a.k, "model": a.model, "tokens_before": before, "applied": False,
                "interface": "bash", "command": command}
        head, shown, tail = "", "", ""
        view = lambda: render(a.file.turns, a.file.show_agents or self.marks)   # marks: the file names authors too
        if not command:
            note, info["rejected"] = "Not applied: bash needs a command.", "invalid"
        else:
            box = self.sandbox(a)
            restarts = box.restarts
            box.put(view())
            out = box.run(command)
            text, problem = box.get()
            limit = int(self.cfg.ctx.bash.get("output_chars", 4000))
            if limit <= 0:                             # no fixed cap: only what could never fit (the whole limit) is cut
                limit = int(a.budget.target * a.budget.chars_per_token)
            if self.atomic:                            # main d545615: the start of the output, and how much more there was
                clip = lambda t, n: (t if len(t) <= limit else t[:limit]) + (
                    f"\n[… {n - len(t[:limit].encode())} more bytes not shown]" if n > len(t[:limit].encode()) else "")
                shown = clip(out["stdout"], out["stdout_bytes"]).strip()
                if out["stderr"].strip():
                    shown += "\n[stderr]\n" + clip(out["stderr"], out["stderr_bytes"]).strip()
            else:
                clip = lambda t: t if len(t) <= limit else t[:limit // 2] + f"\n[... {len(t) - limit} characters not shown ...]\n" + t[-limit // 2:]
                shown = clip(out["stdout"].strip())
                if out["stderr"].strip():
                    shown += "\n[stderr]\n" + clip(out["stderr"].strip())
            head = f"[exit {out['exit_code']}{', timed out' if out['timed_out'] else ''}]\n"
            if self.log_work:
                self.log_workspace(a, box, command, old)
            if self.workspace == "emptied":
                box.reset_scratch()
            if box.restarts > restarts:
                tail = "\n[the sandbox was restarted: /work and /tmp are empty again]"
                if a.problem is not None and self.task.files(a.problem):
                    self.item_files(a, a.problem, True)
                    tail = "\n[the sandbox was restarted: /work and /tmp are empty again, except your current task's files]"
            info.update(exit_code=out["exit_code"], timed_out=out["timed_out"], seconds=out["seconds"],
                        stdout_chars=len(out["stdout"]), runs_python=" python" in f" {command}",
                        stdout_bytes=out.get("stdout_bytes"), stderr_bytes=out.get("stderr_bytes"),
                        killed_processes=out.get("killed_processes"), container_died=out.get("container_died"),
                        sandbox_restarts=box.restarts, workspace=self.workspace)
            if problem:
                note, info["rejected"] = f"Not applied: {problem}; the context was left as it was.", "unreadable"
            elif text.strip() == view().strip():
                note = "No change to the context file."
            else:
                # unchanged turns keep their author; any changed or new text is the editor's, whatever agent= label
                # the command left on it (labels are re-derived from the stored authors, so they cannot be forged)
                blocks = parse(text)
                turns, _ = reconcile(old, blocks, a.k, labelled=False)
                info["relabelled"] = sum(1 for b, t in zip(blocks, turns) if b["agent"] is not None and b["agent"] != t.agent)
                if [(t.id, t.text) for t in turns] == [(t.id, t.text) for t in old]:
                    note = "No change to the turns (agent labels come from who wrote each turn and cannot be changed)."
                else:
                    note = self.commit(a, old, turns, [f"bash edit ({len(turns)} turns)"], before, info)
        made = made or record()
        size = self.tokens(a)
        body = (head + (shown.strip() + "\n" if shown.strip() else "") + note + tail + more + "\n"
                + a.budget.readout(size, show_limit=self.capped))
        a.file.append(Turn("user", neutralize(body), a.k, "result", reply_to=made.id), self.round)
        if info.get("applied"):
            a.budget.rearm(size)
        write_jsonl(self.out / "edits.jsonl", info)

    def log_workspace(self, a: Agent, box, command: str, seen: list):
        """Logging only (never shown to any agent): after a bash command, every file in this agent's /work that is new
        or changed since its previous command, with two measures (work.jsonl):
        - publication: the share of the file's lines (8+ characters) that the agent typed out in this or an earlier
          command, i.e. already in the shared context as part of a command's record (a heredoc or echo publishes what it
          saves; editing such a file later with sed does not make its lines private);
        - offloading: the context turns (of the context the command saw) whose text is in the file without being in the
          command, i.e. copied from /ctx/context.txt inside the sandbox; each with its author (own or another agent's)."""
        files = box.read_tree("/work", cap=200000)
        before = self.work_seen.setdefault(a.k, {})
        typed = self.typed.setdefault(a.k, set())     # lines this agent typed in any command so far (already shown)
        typed.update(l.strip() for l in command.splitlines() if len(l.strip()) >= 8)
        for name, f in files.items():
            if before.get(name) == f["sha1"] or name.startswith("env/"):   # env/: task files the world put there
                continue
            lines = [l.strip() for l in f["text"].splitlines() if len(l.strip()) >= 8]
            published = sum(1 for l in lines if l in typed) / len(lines) if lines else 0.0
            held = set(lines) - typed
            turns = [t for t in seen if any(len(l.strip()) >= 40 and l.strip() in held for l in t.text.splitlines())]
            write_jsonl(self.out / "work.jsonl", {
                "round": self.round, "agent": a.k, "model": a.model, "file": name, "bytes": f["bytes"], "sha1": f["sha1"],
                "new": name not in before, "published_share": round(published, 3), "lines": len(lines),
                "context_turns": [{"id": t.id, "agent": t.agent} for t in turns],
                "own_turns": sum(1 for t in turns if t.agent == a.k),
                "others_turns": sum(1 for t in turns if t.agent not in (None, a.k)),
                "workspace": self.workspace})
        self.work_seen[a.k] = {name: f["sha1"] for name, f in files.items()}

    def close_boxes(self):
        for box in self.boxes.values():
            try:
                box.stop()
            except Exception:
                pass
        self.boxes = {}

    def evict(self, a: Agent):
        """Remove the oldest unprotected turns until the context fits (after max_held holds in one step)."""
        keep = list(a.file.turns)
        locked = self.protected(a.file)
        removed = []
        while self.tokens(a, keep) > a.budget.target:
            i = next((j for j, t in enumerate(keep) if t.id not in locked), None)
            if i is None:
                break
            removed.append(keep.pop(i))
        if not removed:
            return
        a.file.replace_all(keep, None, self.round, "evict", {"by_harness_for": a.k})
        a.file.append(Turn("user", f"[the context was over the limit: its {len(removed)} oldest turns were removed]", None,
                           "notice"), self.round)
        a.stats["evictions"] += 1
        write_jsonl(self.out / "edits.jsonl", {"round": self.round, "agent": a.k, "eviction": True, "removed_turns": len(removed),
                                               "removed_by_agent": {str(k): sum(1 for t in removed if t.agent == k)
                                                                    for k in {t.agent for t in removed}}})

    def end_agent(self, a: Agent, reason: str):
        if a.problem is not None:                      # left open: unsubmitted (team_score)
            a.stats["ended_on"] = a.problem["id"]
        a.done, a.end_reason = True, reason
        a.problem, a.problem_turn = None, None

    # --- logs ----------------------------------------------------------------------------------------------
    def files(self):
        seen, out = set(), []
        for a in self.agents:
            if id(a.file) not in seen:
                seen.add(id(a.file))
                out.append(a.file)
        return out

    def after_round(self):
        for i, f in enumerate(self.files(), 1):
            name = "context" if len(self.files()) == 1 else f"context_agent_{i}"
            (self.out / "snapshots" / f"{name}_r{self.round:03d}.txt").write_text(f.text(), encoding="utf-8")
            if self.shared and not f.show_agents:     # unlabelled shared file: a labelled copy for the logs only
                (self.out / "snapshots" / f"{name}_labelled_r{self.round:03d}.txt").write_text(render(f.turns, True), encoding="utf-8")
            budget = self.agents[0].budget
            by_agent: dict[str, int] = {}
            for t in f.turns:
                by_agent[str(t.agent)] = by_agent.get(str(t.agent), 0) + self.tok(self.agents[0], t.text)
            write_jsonl(self.out / "rounds.jsonl", {"round": self.round, "file": name, "turns": len(f.turns),
                                                    "tokens_by_agent": by_agent, "tokens": self.tok(self.agents[0], f.text())})
        self.write_world("running")

    def write_world(self, status):
        doc = {"label": self.label, "seed": self.seed, "status": status, "rounds": self.round, "sharing": self.sharing,
               "setup": self.setup, "identity": self.identity, "edit": self.edit_mode, "atomic": self.atomic,
               "workspace": self.workspace if self.edit_mode == "bash" else None, "count": self.count,
               "pool": ({"open": list(self.pool_open), "closed": self.pool_closed, "left": len(self.pool_queue),
                         "solved": sum(1 for c in self.pool_closed if c["how"] == "solved")} if self.setup == "pool" else None),
               "order": self.order, "first_mover": self.first_mover, "schedule": self.schedule, "fixed_rounds": self.rounds,
               "marks": "authors" if self.marks else "none", "cross_test": self.cross,
               "continuity": self.continuity, "curation_advice": self.advice, "dashboard": self.dashboard, "history": self.history_list,
               "awareness": self.awareness, "others_as": self.others_as, "chat_others": self.chat_others,
               "alter_notice": self.alter_notice, "need_line": self.need_line, "goal": self.goal, "reasoning_in_context": self.reasoning_in_context,
               "team": self.team_score(),
               "relation": str(self.cfg.world.relation), "capped": self.capped,
               "budget_tokens": int(self.cfg.ctx.budget_tokens), "limit": self.agents[0].budget.target,
               "tokenizer": self.agents[0].budget.tokenizer, "grader_image": getattr(self.task, "image_id", None), "task": self.task.name,
               "seconds": round(time.time() - self.started, 1),
               "agents": [{"agent": a.k, "model": a.model, "problems": [p["id"] for p in a.stream], "done": a.done,
                           "end_reason": a.end_reason, "chars_per_token": round(a.cpt, 3), **a.stats} for a in self.agents],
               "config": OmegaConf.to_container(self.cfg, resolve=True)}
        (self.out / "world.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def summary(self, status):
        return {"label": self.label, "seed": self.seed, "status": status,
                "agents": [{"agent": a.k, "model": a.model, "solved": a.stats["solved"], "submitted": a.stats["submitted"],
                            "cost": round(a.stats["cost"], 4)} for a in self.agents]}
