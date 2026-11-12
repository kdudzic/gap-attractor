"""ContextFileMemory: the agent's memory is a context file it edits itself (a harness memory plug-in).

Each step: the page the agent is on is appended to the file as a user turn (its article shortened if it does not
fit under the soft budget, never its links), then a notice turn with the size readout and any budget nudge. The
model is sent the system message followed by the file (see ContextFile.messages). It may edit the file
(`edit_context`, or `bash` in its own sandbox), up to `max_edit_turns` editing replies per step, none of which count
as a step; every editing call is appended to the file before it runs, and its result after (as in CLM). Its move is
appended as an assistant turn made by a follow_link call. Under the hard budget (overflow=wall) a move is held while
the context is over the limit, and the model edits again (hold_move; budget.py). Nothing is removed unless a model
removes it, except past hard_limit under the soft budget.

A joint world (joint.py) gives every agent of the world a ContextFileMemory over the same ContextFile.
Used as `experiment.memory=context_file` (registered when ctxfile is imported; run.py does this) with the
settings under `ctx` (configs/ctx/*.yaml, configs/ctx_edit/*.yaml, configs/ctx_prompt/*.yaml).
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from gap_attractor.harness.memory import MEMORIES, Memory
from gap_attractor.harness.runner import harness_pick_note, intent_mode, tool_specs, variant_of

from ctxfile import edits as edit_ops
from ctxfile.budget import LINKS_HEADING, Budget
from ctxfile.context import ATTRIBUTIONS, ContextFile, Turn, label, neutralize, parse, reconcile, render

BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": ("Run one bash command in your own sandbox (no network). Your context is the file /ctx/context.txt: "
                    "what you change in it is what you see from your next call on. What the command prints is shown to "
                    "you and added to your context. /work is a scratch folder."),
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "One bash command."}},
                   "required": ["command"]}}}
INTERFACES = ("tool", "bash", "none")
SHARING = ("full", "readonly_others", "separate", "notes")   # notes: own files + one /notes folder in every sandbox
SHARED_CONTEXT = ("full", "readonly_others")
AWARENESS = ("introduced", "strangers", "rumour")   # joint: told about the others | told nothing | told it may be shared
COUNTS = ("whole", "file")                          # what the size counts (see _tokens)


def option(ctx, key, default):
    value = ctx.get(key, default)
    return default if value is None else value


def headers_on(ctx) -> bool:
    value = str(option(ctx, "headers", "auto")).lower()
    return str(option(ctx, "interface", "tool")) == "tool" if value == "auto" else value == "true"


def elide(value, limit: int = 120):
    """Long strings in tool arguments shortened (the replayed edit call; its effect is in the file already)."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"… [{len(value) - limit} characters elided]"
    if isinstance(value, list):
        return [elide(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: elide(v, limit) for k, v in value.items()}
    return value


def call_text(call) -> str:
    """A tool call as text in the file (verbatim mode): its name and its arguments as given."""
    return neutralize(f"{call.get('name')} {json.dumps(call.get('arguments') or {}, ensure_ascii=False)}")


def named(title: str, text: str) -> bool:
    """The title occurs in the text as a whole title (not inside a longer word or title), ignoring case."""
    return re.search(r"(?<![\w])" + re.escape(title) + r"(?![\w])", text, re.I) is not None


def request_chars(messages, tools=None) -> int:
    """The characters of a request as sent: every message's text and tool calls, and the tool definitions."""
    n = 0
    for m in messages:
        n += len(str(m.get("content") or "")) + 8
        for c in m.get("tool_calls") or []:
            f = c.get("function") or {}
            n += len(str(f.get("name") or "")) + len(str(f.get("arguments") or "")) + 8
    return n + (len(json.dumps(tools, ensure_ascii=False)) if tools else 0)


def page_title(observation: str) -> str | None:
    first = observation.split("\n", 1)[0]
    return first[2:].strip() if first.startswith("# ") else None


@MEMORIES.register("context_file")
class ContextFileMemory(Memory):
    name = "context_file"

    def __init__(self, cfg, shared: ContextFile | None = None, agent: int | None = None, n_agents: int = 1, sandbox=None):
        super().__init__(cfg)
        self.ctx = cfg.ctx
        self.interface = str(option(self.ctx, "interface", "tool"))
        self.tool_name = {"tool": "edit_context", "bash": "bash"}.get(self.interface)
        self.budget = Budget(self.ctx, self.tool_name if option(self.ctx, "nudges", True) else None)
        self.agent, self.n_agents = agent, n_agents
        self.sharing = str((self.ctx.get("joint") or {}).get("sharing", "full")) if agent is not None else "full"
        self.shared_context = agent is not None and self.sharing in SHARED_CONTEXT    # the file itself is shared
        # an agent's own file (single, separate, notes) never marks authors: every turn is its own
        self.attribution = str(option(self.ctx, "attribution", "labelled")) if self.shared_context else "none"
        self.labelled = self.attribution != "none"
        self.awareness = str(option(self.ctx, "awareness", "introduced"))
        others = str(option(self.ctx, "others_as", "auto"))
        self.others_as = "own" if others == "auto" else others      # auto: a reply is an assistant message, whoever wrote it
        self.count = str(option(self.ctx, "count", "whole"))
        self.cpt = float(option(self.ctx, "chars_per_token", 4))   # count=whole: calibrated on the billed prompt tokens
        if self.count == "whole":
            self.budget._encoder, self.budget.chars_per_token, self.budget.tokenizer = None, self.cpt, "calibrated"
        self._tool_list = None
        self._view_ids, self._n_calls = [], 0
        # goal worlds (joint.py sets these): {setup, target, targets (pool), key_of}; protected turn ids; arrivals
        self.goal, self.protect = None, (lambda: set())
        self.arrivals, self.world_stop = [], None
        self._pending, self._after_move = None, []   # the latest move, until its page is written (atomic moves)
        self.headers = headers_on(self.ctx)
        self.file = shared if shared is not None else ContextFile(show_agents=False)
        self.gate = str(option(self.ctx, "edit_gate", "fit"))
        self.sandbox, self.own_sandbox = sandbox, False
        self.steps_done, self.run_dir, self._current, self._system = 0, None, None, {"role": "system", "content": ""}
        self._view, self._extra, self.visited, self.round = [], [], [], None
        self._page = None                           # the newest page's article turn (none before the first page)
        self.verbatim = str(option(self.ctx, "append", "structured")) == "verbatim"
        self.wall = str(option(self.ctx, "overflow", "wall")) == "wall"
        self.shrink = False                         # hard rule: never (validated above); kept so old code paths stay inert
        self._shrink: dict[int, tuple] = {}         # this step's echoes: turn id -> (full text, short text, short arguments)
        self._over = None                           # hard budget: the model moved while still over the limit (the run ends)
        self.max_holds = int(option(self.ctx, "max_held_moves", 3))
        self._holds = 0                             # moves held back in this step (hold_move)
        self._replies, self._used = [], set()      # this step's replies as written (observe); those already in the file
        self._events = self._fresh()

    # --- harness hooks ---------------------------------------------------------------------------------------
    @classmethod
    def validate(cls, cfg):
        if "ctx" not in cfg:
            raise ValueError("experiment.memory=context_file needs the ctx settings: run it through experiments/context_file/run.py")
        if str(cfg.experiment.get("action_format")) != "tools":
            raise ValueError("experiment.memory=context_file needs experiment.action_format=tools (the model edits with a tool)")
        if intent_mode(cfg) != "none":
            raise ValueError("experiment.memory=context_file does not support the intent probe yet")
        if variant_of(cfg) not in ("a_hub", "b_search_first"):
            raise ValueError("experiment.memory=context_file supports variant=a_hub and b_search_first")
        interface = str(option(cfg.ctx, "interface", "tool"))
        if interface not in INTERFACES:
            raise ValueError(f"ctx.interface must be one of {INTERFACES}, not {interface!r}")
        if str(option(cfg.ctx, "others_as", "auto")) not in ("auto", "user", "own"):
            raise ValueError("ctx.others_as must be auto, user or own")
        if str(option(cfg.ctx, "headers", "auto")).lower() not in ("auto", "true", "false"):
            raise ValueError("ctx.headers must be auto, true or false")
        if interface == "tool" and not headers_on(cfg.ctx):
            raise ValueError("ctx.interface=tool needs ctx.headers (edit_context refers to turns by number)")
        if str(option(cfg.ctx, "overflow", "wall")) not in ("wall", "soft"):
            raise ValueError("ctx.overflow must be wall or soft")
        # Hard rule (23 Oct 2026): the harness never shortens, elides or rewrites anything in an agent's context;
        # only the agents' own edits change it. Give a larger budget instead (README, "The study's settings").
        if bool(option(cfg.ctx, "shrink_edits", False)):
            raise ValueError("ctx.shrink_edits is not allowed: the harness never shortens the context (hard rule)")
        if str(option(cfg.ctx, "overflow", "wall")) == "soft":
            raise ValueError("ctx.overflow=soft is not allowed: it cuts pages and drops turns (hard rule: the harness "
                             "never shortens the context); use wall, with a budget too large to reach if none should bind")
        if str(option(cfg.ctx, "overflow", "wall")) == "wall" and interface == "none" and int(cfg.ctx.get("budget_tokens", 0)) < 10 ** 6:
            raise ValueError("ctx.overflow=wall needs an editing interface (ctx_edit=tool or bash), or a budget too large to reach")
        if str((cfg.ctx.get("joint") or {}).get("sharing", "full")) not in SHARING:
            raise ValueError(f"ctx.joint.sharing must be one of {SHARING}")
        attribution, awareness = str(option(cfg.ctx, "attribution", "labelled")), str(option(cfg.ctx, "awareness", "introduced"))
        if attribution not in ATTRIBUTIONS:
            raise ValueError(f"ctx.attribution must be one of {ATTRIBUTIONS}")
        if awareness not in AWARENESS:
            raise ValueError(f"ctx.awareness must be one of {AWARENESS}")
        if awareness != "introduced" and attribution != "none":
            raise ValueError("ctx.awareness=strangers or rumour needs ctx.attribution=none (labels would tell)")
        if str((cfg.ctx.get("joint") or {}).get("sharing", "full")) == "notes" and str(cfg.ctx.get("mode")) == "joint":
            if interface != "bash":
                raise ValueError("ctx.joint.sharing=notes needs ctx_edit=bash (/notes is a folder in the sandbox)")
            if str(cfg.ctx.bash.get("joint_workspace")) == "shared":
                raise ValueError("ctx.joint.sharing=notes needs ctx.bash.joint_workspace=private or per_step (/notes is the shared part)")
        if (str((cfg.ctx.get("joint") or {}).get("sharing", "full")) == "separate" and str(cfg.ctx.get("mode")) == "joint"
                and interface == "bash" and str(cfg.ctx.bash.get("joint_workspace")) == "shared"):
            raise ValueError("ctx.joint.sharing=separate needs ctx.bash.joint_workspace=private or per_step (nothing is shared)")
        if str(option(cfg.ctx, "count", "whole")) not in COUNTS:
            raise ValueError(f"ctx.count must be one of {COUNTS}")
        if interface == "bash":
            from ctxfile.sandbox import check_available
            check_available(str(cfg.ctx.bash.image))

    def settings(self):
        """What defines the condition (run.json memory_settings; the analysis keeps different settings apart)."""
        out = {"interface": self.interface, "budget": self.budget.budget, "nudges": self.budget.tool is not None,
               "headers": self.headers, "prompt": str(option(self.ctx, "prompt_style", "clm")),
               "max_edit_turns": self.side_turns(), "gate": self.gate, "readout": bool(option(self.ctx, "readout", True)),
               "edits_in_context": bool(option(self.ctx, "edits_in_context", True)), "hard_limit": self.budget.hard_limit,
               "min_page_tokens": int(option(self.ctx, "min_observation_tokens", 2000)), "nudge_tiers": list(self.budget.tiers),
               "tokenizer": self.budget.tokenizer, "append": "verbatim" if self.verbatim else "structured",
               "append_reasoning": bool(option(self.ctx, "append_reasoning", False)),
               "overflow": "wall" if self.wall else "soft", "reserve": self.budget.reserve, "limit": self.budget.target,
               "shrink_edits": self.shrink, "count": self.count}
        if self.wall:
            out["max_held_moves"] = self.max_holds
        else:
            out["urgent_room"] = self.budget.urgent_room
        if self.agent is not None:
            out.update(sharing=self.sharing, attribution=self.attribution, others_as=self.others_as, awareness=self.awareness)
        if self.agent is not None:
            joint = self.ctx.get("joint") or {}
            out.update(order=str(joint.get("order", "")), start_assignment=str(joint.get("start_assignment", "")))
        if self.goal is not None:
            out.update(goal=self.goal["setup"])
        if self.interface == "bash":
            out.update(workspace=str(self.ctx.bash.get("joint_workspace" if self.agent is not None else "workspace", "")),
                       command_timeout_s=self.ctx.bash.get("command_timeout_s"))
        return out

    def start(self, run):
        self.run_dir = Path(run.dir)
        if self.agent is None:                       # single agent: the file lives in the run folder
            self.file.mirror_path = self.run_dir / "context" / "LIVE_CTX.txt"
            self.file.versions_path = self.run_dir / "context" / "versions.jsonl"
            self.file.mirror()
            if self.interface == "bash" and self.sandbox is None:
                from ctxfile.sandbox import Sandbox, image_id
                self.sandbox, self.own_sandbox = Sandbox(self.ctx.bash), True
                self.sandbox.start()
                run.doc["sandbox_image"] = image_id(str(self.ctx.bash.image)); run.write()

    def finish(self, run, reason):
        self._flush_move()                           # a last move whose page was never opened
        self._flush_after_move()
        over = self.wall and self.budget.tool is not None and self._kept_tokens() > self.budget.target
        if reason == "model_stopped" and (over or self._over):
            # a STOP that went through while over the limit (after max_held_moves holds): a budget-related ending
            # competing with stopping, not a free choice to end
            run.doc["stopped_over_limit"] = {"step": self.steps_done, "tokens_kept": self._kept_tokens(),
                                             "limit": self.budget.target, "held_moves": self._holds}
            run.write()
        elif self._over:
            run.doc["context_budget_end"] = self._over
            run.write()
        if self.goal is not None:
            run.doc["goal"] = {"setup": self.goal["setup"], "target": self.goal.get("target"), "arrivals": self.arrivals,
                               "reached": bool(self.arrivals), "world_stop": self.world_stop}
            run.write()
        if self.own_sandbox and self.sandbox is not None:
            self.sandbox.stop()

    def _flush_move(self):
        """The move made at the end of the previous step, written just before the page it opened (or at the end of the
        session, with no page), so a move and its page are always adjacent: no other agent's turn comes between."""
        if self._pending is not None:
            turn, step = self._pending
            self._pending = None
            self.file.append(turn, step)

    def after_move(self, turn):
        """A notice about this agent's latest move (e.g. a target reached): written after the page the move opened, or
        after the move itself if the session ends there."""
        self._after_move.append(turn)

    def _flush_after_move(self):
        for turn in self._after_move:
            self.file.append(turn, self.steps_done)
        self._after_move = []

    def messages(self, rule, history, observation):
        self._flush_move()
        step = self.steps_done + 1
        self._events, self._extra, self._replies, self._used = self._fresh(), [], [], set()
        self._holds, self._shrink = 0, {}
        if self.sandbox is not None and self._reset_each_step():
            self.sandbox.reset_scratch()
        self._system = {"role": "system", "content": self._system_text(rule)}
        self._events["evicted_turns"] = 0 if self.wall else self._evict_if_over(step)
        now = self._tokens(self.file.turns)
        self.budget.rearm(now)
        page_tokens = self.budget.tokens(observation)
        if self.wall:                      # a page is atomic: it always arrives whole
            text, cap = neutralize(observation), {"capped": False, "article_chars": len(observation), "article_chars_kept": len(observation)}
        else:
            room = max(self.budget.target - now, int(option(self.ctx, "min_observation_tokens", 2000)))
            text, cap = self.budget.cap_page(neutralize(observation), room)
        self.budget.recent_pages.append(page_tokens)
        # the page is two turns: its article (editable at once) and its link list (protected until the agent moves)
        cut = text.rfind(LINKS_HEADING)
        article, links = (text[:cut], text[cut + 1:]) if cut >= 0 else (text, "")
        self._page = self.file.append(Turn("user", article, self.agent, "observation"), step)
        self._current = self.file.append(Turn("user", links, self.agent, "links"), step) if links.strip() else None
        after, kept = self._tokens(self.file.turns), self._kept_tokens()
        # nudge only when there is something to compact besides this page (never on a first, oversized page)
        editable = after - self._tokens([t for t in (self._page, self._current) if t is not None])
        per_round = self.n_agents if self.shared_context else 1
        floor = self.budget.urgent_room * self.budget.target
        readout = option(self.ctx, "readout", True)
        size = kept if self.wall else after              # the size the readout shows
        level, nudge = None, None
        if self.wall:                                       # pages are not charged, so no "next page may not fit" nudge
            if editable >= floor:
                level, nudge = self.budget.nudge(kept, per_round, urgent=False)
            # the notice turn counts too: if it is what takes the context over, the model gets the over warning
            shown = "\n".join(([self.budget.readout(kept)] if readout else []) + ([nudge] if nudge else []))
            total = kept + (self.budget.tokens(shown) + 2 if shown else 0)      # the notice turn counts too
            if self.budget.tool is not None and total > self.budget.target:
                size = total                                # shown in the readout and the warning alike
                level, nudge = self.budget.over(total, self.side_turns())   # only the strongest warning
                self._events["over_warning"] = True
        elif editable >= floor:
            level, nudge = self.budget.nudge(after, per_round)
        notice = [self.budget.readout(size)] if readout else []
        if nudge:
            notice.append(nudge)
        last = self.cfg.experiment.get("last_page_note")
        if last and step >= int(self.cfg.experiment.step_limit):
            notice.append(str(last))
        if notice:
            self.file.append(Turn("user", "\n".join(notice), self.agent, "notice"), step)
        self._flush_after_move()
        shown = text.rsplit(LINKS_HEADING, 1)[-1] if LINKS_HEADING in text else ""
        self._events.update(tokens_before=now, page_tokens=page_tokens, tokens_with_page=after, tokens_kept_with_page=kept,
                            pressure=round((kept if self.wall else after) / self.budget.target, 3), nudge=level, links_visible=sum(
                                1 for l in shown.splitlines() if l.startswith("- ")), **cap)
        self._view = [self._system] + self._file_messages()
        self._view_ids = [t.id for t in self.file.turns]
        return self._view

    def side_tools(self):
        if self._page is None:          # before the first page (variant b's opening search): no editing tools
            return []
        return {"tool": [edit_ops.TOOL], "bash": [BASH_TOOL]}.get(self.interface, [])

    def side_turns(self):
        return int(option(self.ctx, "max_edit_turns", 3)) if self.interface != "none" else 0

    def handle(self, call):
        try:
            if call.get("name") == "bash" and self.interface == "bash":
                return self._handle_bash(call)
            if call.get("name") == "edit_context" and self.interface == "tool":
                return self._handle_tool(call)
            return f"There is no tool named {call.get('name')}."
        except Exception as error:                    # a failing edit must never end the run
            self._events["edits"].append({"applied": False, "error": f"{type(error).__name__}: {error}"[:300]})
            return "The edit failed. Nothing was changed."

    def continue_messages(self, messages, calls, results):
        """The post-edit view: system + the file (which now holds the editing calls and results), then whatever the
        harness had appended after the last view (e.g. a retry exchange)."""
        if messages[:len(self._view)] == self._view:
            self._extra = self._extra + messages[len(self._view):]
        rebuilt = [self._system] + self._file_messages()
        self._view_ids = [t.id for t in self.file.turns]
        if not option(self.ctx, "edits_in_context", True):   # not in the file: show only the latest calls and results
            rebuilt += [{"role": "assistant", "content": None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for c in calls]}] + [{"role": "tool", "tool_call_id": c["id"], "content": r} for c, r in zip(calls, results)]
        self._view = rebuilt + list(self._extra)
        return self._view

    def side_exhausted_note(self, n, tools):
        """The request to move once this page's editing calls are used up (the harness's says "turns" and "step",
        which models copied)."""
        held = (f" If your context is still over the limit, the move is held and you get {n} more calls."
                if self.wall and self.budget.tool is not None else "")
        return f"You have used your {n} {self.tool_name} calls before this move. Now call follow_link.{held}"

    def hold_move(self, call):
        """Hard budget: a move (follow_link, or stop) is held while the context (all but the newest list of links) is
        over the limit, and the model edits again; the attempt and the answer stay in the file. After max_held_moves
        holds in a step the move goes ahead: a link ends the run after it (record), a STOP is flagged (finish)."""
        if not self.wall or self.budget.tool is None or call.get("name") not in ("follow_link", "stop"):
            return None
        stop = call.get("name") == "stop"
        tokens = self._kept_tokens()
        if tokens <= self.budget.target or self._holds >= self.max_holds:
            return None
        self._holds += 1
        note = self.budget.held(tokens, self._holds, self.max_holds, stop=stop, calls=self.side_turns())
        args = call.get("arguments") or {}
        if self.verbatim:
            words = self._words(self._reply_with(call.get("id")))
            text, content = (words + "\n" if words else "") + call_text(call), words
        else:
            words = neutralize(str(args.get("reasoning") or "").strip())
            tried = "[tried to stop]" if stop else f"[tried to follow link: {args.get('title')}]"
            text, content = (words + "\n" if words else "") + tried, ""
        made = self.file.append(Turn("assistant", text, self.agent, "attempt",
                                     call={"name": call.get("name"), "arguments": args, "content": content}), self.steps_done + 1)
        self.file.append(Turn("user", neutralize(note), self.agent, "result", reply_to=made.id), self.steps_done + 1)
        self._events["held_moves"].append({"tokens_kept": tokens, "title": "STOP" if stop else args.get("title")})
        return note

    def other_menus(self):
        """Other agents' link lists still in the shared file (for failures.jsonl: a click on one is `others_menu`)."""
        out = []
        for t in self.file.turns:
            if t.kind == "links" and t.agent not in (None, self.agent):
                out.append((f"agent {t.agent}, turn {self.file.index_of(t.id)}", [l[2:] for l in t.text.splitlines() if l.startswith("- ")]))
        return out

    def observe(self, request, reply):
        self._replies.append((request, reply))

    def observe_call(self, request, tools, usage):
        """Every API call with what it billed: calibrates characters per token for this agent's model (count=whole),
        and logs the call (calls.jsonl: the ids of the turns the request was built from, our size estimate, the
        usage with cached tokens) and, with ctx.save_requests, the request itself (requests/NNNN.json)."""
        usage = dict(usage or {})
        sent = request_chars(request, tools)
        estimate = round(sent / self.cpt)
        billed = usage.get("prompt_tokens")
        if self.count == "whole" and isinstance(billed, (int, float)) and billed >= 200:
            self.cpt = 0.5 * self.cpt + 0.5 * sent / billed
            self.budget.chars_per_token = self.cpt
        if self.run_dir is None:
            return
        self._n_calls += 1
        if option(self.ctx, "save_requests", False):
            folder = self.run_dir / "requests"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{self._n_calls:04d}.json").write_text(json.dumps({"messages": request, "tools": tools}, ensure_ascii=False),
                                                              encoding="utf-8")
        row = {"call": self._n_calls, "step": self.steps_done + 1, "agent": self.agent,
               "round": (self.round or {}).get("round"), "model": str(self.cfg.provider.model), "turn_ids": list(self._view_ids),
               "estimate": estimate, "chars_per_token": round(self.cpt, 4), "usage": usage}
        with (self.run_dir / "calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _words(self, i) -> str:
        """The text of reply i as written (with its reasoning trace if ctx.append_reasoning), once per reply."""
        if i is None or i in self._used:
            return ""
        self._used.add(i)
        reply = self._replies[i][1]
        parts = []
        if option(self.ctx, "append_reasoning", False) and (reply.get("reasoning") or "").strip():
            parts.append("[thinking]\n" + reply["reasoning"].strip() + "\n[/thinking]")
        if (reply.get("content") or "").strip():
            parts.append(reply["content"].strip())
        return neutralize("\n".join(parts))

    def _reply_with(self, call_id):
        return next((i for i, (_, r) in enumerate(self._replies) if any(c.get("id") == call_id for c in r["tool_calls"])), None)

    def _feedback(self, call_id=None, text=None) -> str:
        """What the harness answered to a failed attempt (found in this step's later requests)."""
        for request, _ in self._replies:
            for k, m in enumerate(request):
                if call_id and m.get("role") == "tool" and m.get("tool_call_id") == call_id:
                    return str(m.get("content") or "")
                if text and m.get("role") == "assistant" and m.get("content") == text and k + 1 < len(request):
                    return str(request[k + 1].get("content") or "")
        return ""

    def _append_attempts(self, final):
        """Verbatim: replies of this step that were neither editing calls nor the move, with the harness's answer."""
        sides = {t["function"]["name"] for t in self.side_tools()}
        for i, (_, reply) in enumerate(self._replies):
            if i == final or i in self._used:
                continue
            calls = reply["tool_calls"]
            if calls and all(c["name"] in sides for c in calls):
                continue                                  # editing replies are in the file already (_echo)
            words, move = self._words(i), next((c for c in calls if c["name"] not in sides), None)
            if move is not None:
                made = self.file.append(Turn("assistant", (words + "\n" if words else "") + call_text(move), self.agent, "attempt",
                                             call={"name": move["name"], "arguments": move["arguments"], "content": words}), self.steps_done)
                self.file.append(Turn("user", neutralize(self._feedback(call_id=move.get("id")) or "(no answer)"), self.agent, "result",
                                      reply_to=made.id), self.steps_done)
            elif words:
                self.file.append(Turn("assistant", words, self.agent, "attempt"), self.steps_done)
                feedback = self._feedback(text=reply.get("content"))
                if feedback:
                    self.file.append(Turn("user", neutralize(feedback), self.agent, "feedback"), self.steps_done)

    def record(self, item, step):
        if item.get("page") is None and item.get("search_query") is not None and self._page is None:
            self._record_search(item)
            return
        self._check_arrival(item)
        self._check_limit()
        self._apply_shrink()
        self.steps_done += 1
        if self.verbatim:
            return self._record_verbatim(item)
        title = page_title(self._page.text) if self._page is not None else None
        if title:
            self.visited.append((title, self._page.id))
        if item.get("harness_pick"):
            turn = Turn("user", harness_pick_note(self.cfg, item), self.agent, "note")
        else:
            reasoning = neutralize((item.get("reasoning") or "").strip())
            args = {"title": item["choice"], **({"reasoning": reasoning} if reasoning else {})}
            turn = Turn("assistant", (reasoning + "\n" if reasoning else "") + f"[followed link: {item['choice']}]", self.agent, "move",
                        call={"name": "follow_link", "arguments": args, "content": "", "result": f'Opened: {item["choice"]}'})
        self._pending = (turn, self.steps_done)       # written with its page, at this agent's next turn (atomic move)
        self._snapshot()

    def _record_search(self, item):
        """Variant b_search_first: the opening search (asked outside the file, before any page) is written into it as
        the agent's search call with the result list it got, then the result it opened."""
        reasoning = neutralize((item.get("reasoning") or "").strip())
        query = str(item["search_query"])
        # as text, not a replayed tool call: later requests do not offer the search tool
        self.file.append(Turn("assistant", (reasoning + "\n" if reasoning and not item.get("choice") else "") + f"[searched: {query}]",
                              self.agent, "search"), 0)
        self.file.append(Turn("user", neutralize(item.get("results_text") or ""), self.agent, "results"), 0)
        if item.get("choice") and item["choice"] != "STOP":
            args = {"title": item["choice"], **({"reasoning": reasoning} if reasoning else {})}
            self.file.append(Turn("assistant", (reasoning + "\n" if reasoning else "") + f"[opened: {item['choice']}]", self.agent,
                                  "move", call={"name": "follow_link", "arguments": args, "content": "",
                                                "result": f'Opened: {item["choice"]}'}), 0)
        self._check_arrival(item)

    def _record_verbatim(self, item):
        title = page_title(self._page.text) if self._page is not None else None
        if title:
            self.visited.append((title, self._page.id))
        final = None
        if not item.get("harness_pick"):
            final = next((i for i in range(len(self._replies) - 1, -1, -1) if any(
                c["name"] == "follow_link" and c["arguments"].get("title") for c in self._replies[i][1]["tool_calls"])), None)
        self._append_attempts(final)
        if item.get("harness_pick") or final is None:
            turn = Turn("user", harness_pick_note(self.cfg, item), self.agent, "note") if item.get("harness_pick") else \
                Turn("assistant", f"[followed link: {item['choice']}]", self.agent, "move",
                     call={"name": "follow_link", "arguments": {"title": item["choice"]}, "content": "", "result": f'Opened: {item["choice"]}'})
        else:
            move = next(c for c in self._replies[final][1]["tool_calls"] if c["name"] == "follow_link")
            words = self._words(final)
            turn = Turn("assistant", (words + "\n" if words else "") + call_text(move), self.agent, "move",
                        call={"name": "follow_link", "arguments": move["arguments"], "content": words, "result": f'Opened: {item["choice"]}'})
        self._pending = (turn, self.steps_done)       # written with its page, at this agent's next turn (atomic move)
        self._snapshot()

    def _check_limit(self):
        """Hard budget, at the move (before it is appended): if the context is still over the limit (its moves were
        held max_held_moves times, or the harness picked a link after unusable replies), the run ends after this step."""
        if not self.wall or self.budget.tool is None:
            return
        tokens = self._kept_tokens()
        if tokens > self.budget.target:
            self._over = {"step": self.steps_done + 1, "tokens_kept": tokens, "limit": self.budget.target,
                          "budget": self.budget.budget, "held_moves": self._holds}

    def embed_turns(self, messages):
        """context_embedding=steps (main): embed the file's turns, with their stable ids, kinds and authors, rather than
        the request's messages. Called at the end of the step, before the move is appended: the file as the model
        last saw it in this step, after its edits."""
        return [{"id": t.id, "kind": t.kind, "role": t.role, "agent": t.agent, "text": t.text.strip()} for t in self.file.turns]

    def should_end(self):
        # an arrival wins over the budget: the move that arrived may also have been made over the limit (finish
        # still records context_budget_end in run.json)
        if self.goal is not None and self.goal["setup"] != "pool" and self.arrivals and not self.goal.get("stay"):
            return "reached_target"
        if self._over:
            return "context_budget"
        return self.world_stop

    def _check_arrival(self, item):
        """Goal worlds: the page this move opens is a target (own target; pool: any target still open)."""
        if self.goal is None or not item.get("choice") or item["choice"] == "STOP":
            return
        if self.goal.get("stay") and self.arrivals:   # all_found: its own target is reached once; going on is not arriving
            return
        key = self.goal["key_of"](item["choice"])
        open_keys = self.goal["open_keys"]() if self.goal["setup"] == "pool" else {self.goal["key"]: self.goal["target"]}
        if key is not None and key in open_keys:
            self.arrivals.append({"target": open_keys[key], "step": self.steps_done + 1, "round": (self.round or {}).get("round"),
                                  "harness_pick": bool(item.get("harness_pick"))})

    def _snapshot(self):
        if self.run_dir is not None and option(self.ctx, "snapshots", True):
            snap = self.run_dir / "context" / f"step_{self.steps_done:03d}.txt"
            snap.parent.mkdir(parents=True, exist_ok=True)
            snap.write_text(self.file.text(), encoding="utf-8")

    def step_fields(self, step):
        """Logged per step (before the move is appended): sizes, cap, nudge, edits, and what the agent could see.
        pages_kept: share of the pages it visited before whose article turn is still in the file unchanged;
        path_recall: share whose title the file still names outside link lists (in a summary, or another page's text).
        named_by_*: whether the chosen title is named, as a whole title and outside link lists, in its own earlier turns
        or in other agents' turns; menu_named_by_*: the share of the menu that is. Descriptive only: the
        no-exposure baseline for joint worlds is computed by analyze.py from agents that could not see each other."""
        text = self.file.text()
        current = {t.id for t in (self._page, self._current) if t is not None}
        alive = {t.id for t in self.file.turns}
        prose = lambda ts: "\n".join(t.text.rsplit(LINKS_HEADING, 1)[0] for t in ts if t.kind != "links")
        n = len(self.visited)
        said = prose(self.file.turns)
        kept = self._kept_tokens()
        fields = dict(self._events, tokens_after_edits=self._tokens(self.file.turns), tokens_kept_at_move=kept,
                      over_limit_at_move=kept > self.budget.target if self.wall else None, turns=len(self.file.turns),
                      file_chars=len(text), agent=self.agent, round=self.round, visited_before=n,
                      pages_kept=round(sum(i in alive for _, i in self.visited) / n, 3) if n else None,
                      path_recall=round(sum(named(v, said) for v, _ in self.visited) / n, 3) if n else None)
        menu = [c["title"] for c in step.get("choice_set") or []]
        choice = step.get("parsed_choice")
        pools = {"own": prose([t for t in self.file.turns if t.id not in current and t.agent in (None, self.agent)])}
        if self.agent is not None:
            pools["others"] = prose([t for t in self.file.turns if t.agent not in (None, self.agent)])
            owners: dict[str, int] = {}                     # who holds the kept context (characters, by author)
            for t in self.file.turns:
                if self._current is None or t.id != self._current.id:
                    owners[str(t.agent)] = owners.get(str(t.agent), 0) + len(t.text)
            fields["kept_chars_by_agent"] = owners
        for who, pool in pools.items():
            if menu:
                fields[f"menu_named_by_{who}"] = round(sum(named(m, pool) for m in menu) / len(menu), 3)
            if choice and choice != "STOP":
                fields[f"named_by_{who}"] = named(choice, pool)
        self._events = self._fresh()
        return fields

    # --- editing ---------------------------------------------------------------------------------------------
    def _handle_tool(self, call):
        edits = call.get("arguments") or {}
        before = self._measure(self.file.turns)
        try:
            turns, notes = edit_ops.apply(self.file.turns, edits, self.agent, self.file.show_agents)
        except edit_ops.EditError as error:
            result = f"Not applied: {error}"
            self._events["edits"].append({"applied": False, "rejected": "invalid", "note": str(error), "tokens_before": before})
            return self._echo_result(self._echo_call(call), result)
        kept, note, info = self._commit(turns, before, {"edits": edits})
        result = f"Applied ({'; '.join(notes)}): {len(turns)} turns." if kept else note
        self._events["edits"].append(dict(info, ops=notes))
        return self._echo_result(self._echo_call(call), result)     # atomic: the call and its result, after it ran

    def _handle_bash(self, call):
        command = str((call.get("arguments") or {}).get("command") or "").strip()
        if not command:
            return "Not applied: bash needs a `command`."
        before = self._measure(self.file.turns)       # atomic: the command is not in the file while it runs
        restarts = self.sandbox.restarts
        view = self._file_view()
        self.sandbox.put(view)
        out = self.sandbox.run(command)
        text, problem = self.sandbox.get()
        info = {"applied": False, "tokens_before": before, "tokens_after": before}
        if problem:
            note = f"Not applied: {problem}; it was put back as it was."
            info["rejected"] = "unreadable"
        elif text.strip() == view.strip():
            note = "No change to the context file."
            info["rejected"] = None
        else:
            # who wrote a turn comes from the record, never from a header: a turn shortened in place (its header kept)
            # becomes the editor's, as with edit_context; an agent= changed by hand changes nothing
            blocks = parse(text)
            turns, _ = reconcile(self.file.turns, blocks, self.agent, labelled=False)
            relabelled = sum(1 for b, t in zip(blocks, turns) if t.id and b["agent"] is not None      # unchanged turns only
                             and str(b["agent"]) != str(label(t, self.attribution, self.agent)))
            if [(t.id, t.role, t.text) for t in turns] == [(t.id, t.role, t.text) for t in self.file.turns]:
                note = "No change to the turns (who added a turn cannot be changed: agent= comes from who wrote it)."
                info.update(rejected=None, relabelled=relabelled)
            else:
                kept, note, info = self._commit(turns, before, {"command": command})
                info["relabelled"] = relabelled
        cut = lambda s, n: f"\n[… {n - len(s.encode())} more bytes not shown]" if n > len(s.encode()) else ""
        shown = self._clip(out["stdout"]) + cut(out["stdout"], out["stdout_bytes"])
        if out["stderr"].strip():
            shown += "\n[stderr]\n" + self._clip(out["stderr"]) + cut(out["stderr"], out["stderr_bytes"])
        head = f"[exit {out['exit_code']}{', timed out' if out['timed_out'] else ''}]\n"
        emptied = "/tmp is" if self.agent is not None and str(self.ctx.bash.get("joint_workspace", "")) == "shared" else "/work and /tmp are"
        tail = note + (f"\n[the sandbox was restarted: {emptied} empty again]" if self.sandbox.restarts > restarts else "")
        result = head + (shown.strip() + "\n" if shown.strip() else "") + tail
        made = self._echo_call(call)                  # atomic: the command and its output, appended after it ran
        self._events["edits"].append(dict(info, command=command, exit_code=out["exit_code"], timed_out=out["timed_out"],
                                          seconds=out["seconds"], stdout_bytes=out["stdout_bytes"], stderr_bytes=out["stderr_bytes"],
                                          killed_processes=out["killed_processes"], leftover_processes=out["leftover_processes"],
                                          container_died=out["container_died"], sandbox_restarts=self.sandbox.restarts))
        return self._echo_result(made, result)

    def _commit(self, turns, before, detail, problems=()):
        """Keep a new list of turns unless it is invalid, touches the current link list, edits other agents' turns
        where that is not allowed, or fails the gate. Returns (kept, note, info)."""
        after = self._measure(turns)
        new_ids = {t.id for t in turns if t.id}
        removed = [t for t in self.file.turns if t.id not in new_ids]
        by_agent: dict[str, int] = {}
        for t in removed:
            by_agent[str(t.agent)] = by_agent.get(str(t.agent), 0) + 1
        foreign = [t for t in removed if t.agent not in (None, self.agent)]
        cur = self._current
        cur_ok = cur is None or any(t.id == cur.id and t.text == cur.text for t in turns)
        info = {"applied": False, "tokens_before": before, "tokens_after": before, "removed_turns": len(removed),
                "removed_by_agent": by_agent, "removed_others": len(foreign), "rejected": None}
        if problems:
            info["rejected"] = "attribution"
            return False, "Not applied: " + "; ".join(problems) + ".", info
        if not cur_ok:
            where = self.file.index_of(cur.id)
            info["rejected"] = "current_page"
            page = self.file.index_of(self._page.id) if self._page is not None else None
            hint = f" To shorten the page you are on, edit only turn {page}, its text." if page else ""
            return False, (f"Not applied: turn {where} is the list of links you can follow now, and it must stay as it is "
                           f"until you have moved.{hint}"), info
        kept_now = {t.id: t.text for t in turns if t.id}
        goals = [t for t in self.file.turns if t.id in self.protect() and kept_now.get(t.id) != t.text]
        if goals:
            info["rejected"] = "goal"
            where = ", ".join(str(self.file.index_of(t.id)) for t in goals)
            return False, (f"Not applied: turn {where} states a page still to be reached, and it must stay as it is."), info
        if self.sharing == "readonly_others" and foreign:
            info["rejected"] = "others_turns"
            return False, (f"Not applied: it would remove or change {len(foreign)} turn(s) of other agents; only your own "
                           "turns can be edited."), info
        if not edit_ops.gate(self.gate, before, after, self.budget.target):
            info["rejected"] = "gate"
            return False, (f"Not applied: the context would grow from ~{before} to ~{after} tokens, over the limit of "
                           f"{self.budget.target}. An edit may make the context larger only while it stays within the limit."), info
        change = self.file.replace_all(turns, self.agent, self.steps_done + 1, "edit", detail)
        info.update(applied=True, tokens_after=after, added_turns=len(change["added"]))
        return True, f"Applied: {len(turns)} turns.", info

    def _echo_call(self, call):
        """The editing call, recorded in full (never shortened: hard rule) and appended after it has run, just before
        its result: a call and its result are one unit, so a command never sees or changes its own record (CLM instead
        appends the call before it runs; changed 23 Oct 2026). Sent as a tool call with its result while unedited.
        Returns the turn, or None (edits_in_context off)."""
        if not option(self.ctx, "edits_in_context", True):
            return None
        args = call.get("arguments") or {}
        words = self._words(self._reply_with(call.get("id"))) if self.verbatim else ""
        shown = f"$ {args.get('command', '')}" if call.get("name") == "bash" else call_text(call)
        text = (words + "\n" if words else "") + neutralize(shown)
        return self.file.append(Turn("assistant", text, self.agent, "call",
                                     call={"name": call.get("name"), "arguments": args, "content": words}), self.steps_done + 1)

    def _echo_result(self, made, result) -> str:
        """The result with the size readout (the context including this result), appended right after its call."""
        size = (self._kept_tokens() if self.wall else self._tokens(self.file.turns)) + self.budget.tokens(result) + 12
        result = result + "\n" + self.budget.readout(size, show_limit=bool(option(self.ctx, "readout", True)))
        if made is not None:
            self.file.append(Turn("user", neutralize(result), self.agent, "result", reply_to=made.id), self.steps_done + 1)
        return result

    def _shrunk(self, turns):
        """The turns as they will be kept after the step: this step's long editing calls and command output shortened
        (unless the model has changed them)."""
        out = []
        for t in turns:
            entry = self._shrink.get(t.id)
            if entry is not None and t.text == entry[0]:
                call = dict(t.call, arguments=entry[2]) if (t.call and entry[2] is not None) else t.call
                t = replace(t, text=entry[1], call=call)
            out.append(t)
        return out

    def _apply_shrink(self):
        if self._shrink:
            turns = self._shrunk(self.file.turns)
            if any(a is not b for a, b in zip(turns, self.file.turns)):
                self.file.replace_all(turns, None, self.steps_done + 1, "shrink", {"turns": sorted(self._shrink)})
        self._shrink = {}

    # --- helpers ---------------------------------------------------------------------------------------------
    @staticmethod
    def _fresh():
        return {"tokens_before": None, "page_tokens": None, "capped": False, "nudge": None, "evicted_turns": 0, "edits": [],
                "held_moves": []}

    def _file_messages(self, turns=None):
        return self.file.messages(self.agent, labelled=self.attribution, others_as=self.others_as, headers=self.headers, turns=turns)

    def _file_view(self) -> str:
        """The file as this agent sees it (bash: /ctx/context.txt)."""
        return self.file.view(self.agent, self.attribution)

    def _tools(self):
        if self._tool_list is None:
            self._tool_list = tool_specs(self.cfg, ["follow_link", "stop"]) + {"tool": [edit_ops.TOOL], "bash": [BASH_TOOL]}.get(self.interface, [])
        return self._tool_list

    def _kept_tokens(self) -> int:
        """What counts against the limit: the whole request except the newest list of links (the agent cannot edit it
        before it moves), with this step's long echoes as they will be kept."""
        turns = [t for t in self.file.turns if self._current is None or t.id != self._current.id]
        return self._tokens(self._shrunk(turns) if self.shrink else turns)

    def _measure(self, turns) -> int:
        """A list of turns sized as the limit sizes the file (wall), or whole (soft): for the gate and edit notes."""
        if not self.wall:
            return self._tokens(turns)
        kept = [t for t in turns if self._current is None or t.id != self._current.id]
        return self._tokens(self._shrunk(kept) if self.shrink else kept)

    def _tokens(self, turns) -> int:
        """The size of a context made of these turns.
        count=whole (default from 21 Oct 2026): the whole request as sent, the system message, the turns as messages
        and the tool definitions, at this model's characters per token (calibrated on its billed prompt tokens).
        count=file (earlier runs): the turns as file text, at chars/4 or tiktoken; the system message and tools are
        not counted (for Haiku and Gemini that was 1.3-1.4x below the billed size)."""
        if self.count == "whole":
            return round(request_chars([self._system] + self._file_messages(turns), self._tools()) / self.cpt)
        return self.budget.tokens(render(turns, self.attribution, self.agent))

    def _clip(self, text):
        limit = int((self.ctx.get("bash") or {}).get("output_chars", 10000))
        return text if len(text) <= limit else text[:limit // 2] + f"\n[… {len(text) - limit} characters omitted …]\n" + text[-limit // 2:]

    def _reset_each_step(self) -> bool:
        key = "joint_workspace" if self.agent is not None else "workspace"
        return str((self.ctx.get("bash") or {}).get(key, "")) == "per_step"

    def _others(self):
        """Placeholders for the joint instructions: "1 other agent who browses", "2 other agents who browse"."""
        n = self.n_agents - 1
        return {"n_agents": self.n_agents, "n_others": n, "agent": self.agent,
                "others": "1 other agent" if n == 1 else f"{n} other agents", "s": "s" if n == 1 else ""}

    def _system_text(self, rule):
        prompts = self.ctx.instructions
        text = str(prompts.get(self.interface) or "")
        parts = [str(self.cfg.experiment.system_prompt)]
        if text:
            readout, limit = bool(option(self.ctx, "readout", True)), self.budget.target
            calls = self.side_turns()
            if self.wall and self.budget.tool is not None:
                budget = (f"Your context may hold up to {limit} tokens; the newest list of links does not count. A page is "
                          "always added in full, even if that takes your context over the limit. While it is over, a move "
                          "is held instead of made: shorten your context first. "
                          "If you run out of "
                          f"{self.tool_name} calls on a page while you are still over, call follow_link anyway: the move is "
                          f"held and you get {calls} more calls. A move is held at most {self.max_holds} times on a page; "
                          "the next one goes ahead, and the session ends after it.")
                if self.shrink and self.interface == "bash":
                    budget += (" Everything a command prints, and long commands, are shown in full while you are on a "
                               "page; when you move, the output is dropped and long commands are shortened. The limit "
                               "counts them as they will be kept.")
                elif self.shrink and self.verbatim:
                    budget += (" Long edit calls are shown in full while you are on a page and shortened when you move; "
                               "the limit counts them shortened.")
            else:
                budget = f"Your context may hold up to {limit} tokens." if (readout or self.budget.tool is not None) else ""
            if budget and self.count == "whole":
                budget += " Its size counts everything you are sent, these instructions included."
            notice = (f", followed by a notice with the size of your context, such as [context: ~N/{limit} tokens]"
                      if readout else "")
            joint = self.shared_context and self.awareness == "introduced"
            roles = ("Turns with role=assistant are replies (moves and edits) of the agent that added them; turns with "
                     "role=user come from the session (pages, notices, and answers to calls)." if joint else
                     "Turns with role=assistant are your own replies; turns with role=user come from the session (pages, "
                     "notices, and answers to your calls).")
            turn_format = (" It is a numbered list of turns, each starting with a line such as [[TURN 3 role=user]]. "
                           + roles if self.headers else "")
            calls_kept = ""                          # calls are kept in full (hard rule: nothing is shortened)
            more = " (more if a move is held)" if self.wall and self.budget.tool is not None else ""
            bash = self.ctx.get("bash") or {}
            space = str(bash.get("joint_workspace" if self.agent is not None else "workspace", ""))
            told = self.awareness == "introduced" and (self.shared_context or self.sharing == "notes")
            if space in ("shared", "private") and not told:
                space = "persistent"                 # not told about others (or none to tell of): described as kept
            workspace = {"per_step": "/work is a scratch folder; it is emptied each time you open a page.",
                         "persistent": "/work is a scratch folder that keeps its files for the whole session.",
                         "shared": "/work is a scratch folder shared by all agents for the whole session.",
                         "private": "/work is your own scratch folder for the whole session."}.get(space, "/work is a scratch folder.")
            kept_files = "" if space == "per_step" else ", except files you leave in the sandbox"
            filled = text.format(budget=budget, notice=notice, max_edit_turns=calls, tool=self.tool_name,
                                 turn_format=turn_format, roles=roles, calls_kept=calls_kept, more=more,
                                 workspace=workspace, kept_files=kept_files, timeout=bash.get("command_timeout_s", 120))
            filled = re.sub(r"(?<=\S) {2,}", " ", filled).replace(" \n", "\n")     # spaces left by an empty placeholder
            parts.append(re.sub(r"\n{3,}", "\n\n", filled).strip())
        if self.shared_context and self.awareness == "introduced":
            joint = self.ctx.joint_instructions[{"labelled": "labelled", "other": "other", "none": "anonymous"}[self.attribution]]
            extra = str(self.ctx.joint_instructions.get(self.sharing) or "") if self.interface != "none" else ""
            parts.append(str(joint).format(**self._others()) + (" " + extra if extra else ""))
        elif self.shared_context and self.awareness == "rumour":
            parts.append(str(self.ctx.joint_instructions.get("rumour") or ""))
        elif self.agent is not None and self.sharing == "notes":
            key = {"introduced": "notes", "rumour": "notes_rumour", "strangers": "notes_strangers"}[self.awareness]
            parts.append(str(self.ctx.joint_instructions[key]).format(**self._others()))
        jcfg = self.ctx.get("joint") or {}           # the joint texts describe the study's settings (a new order every round,
        if self.agent is not None:                    # different start pages); keep them true under the other settings
            if str(jcfg.get("order", "seeded_shuffle")) == "fixed":
                parts = [re.sub(r"in\s+an order that changes every round", "always in the same order", p) for p in parts]
            if str(jcfg.get("start_assignment", "")) == "same":
                parts = [re.sub(r",\s+each from its own\s+page(?=;|\.)", "", p) for p in parts]
                parts = [re.sub(r",\s+each from its own page and with its own context", ", each with its own context", p)
                         for p in parts]
        page_note = self.cfg.experiment.get("pages_note")
        if page_note:                                # a task that ends with something written (essay): the page budget
            parts.append(str(page_note).format(pages=int(self.cfg.experiment.step_limit)))
        if self.goal is not None and self.goal["setup"] == "external":
            parts.append(str(self.ctx.goal.text).format(target=self.goal["target"]))
        return "\n\n".join(parts) + "\n\n" + rule

    def _evict_if_over(self, step) -> int:
        """Past hard_limit x budget, the harness drops the oldest turns until the file is under the target, and says
        so in a notice turn (logged as an `evict` version)."""
        if self._tokens(self.file.turns) <= self.budget.hard_limit * self.budget.budget:
            return 0
        turns, n = list(self.file.turns), 0
        while turns and self._tokens(turns) > self.budget.target:
            turns.pop(0); n += 1
        if n == 0:                   # nothing to remove (count=whole: the instructions alone are over a tiny budget)
            return 0
        self.file.replace_all(turns, None, step, "evict", {"evicted_turns": n})
        self.file.append(Turn("user", f"[the context had grown too large, so its {n} oldest turns were removed]", None, "notice"), step)
        return n
