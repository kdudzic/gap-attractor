"""Stated-need forks: does an agent spare a turn when the need for it is stated, at the very call where it wiped it?

    uv run python experiments/siamese/need_forks.py <stage dir> [...] --out <dir> --dry-run
    uv run python experiments/siamese/need_forks.py <stage dir> [...] --out <dir> --validate          (no API calls)
    uv run --env-file .env python experiments/siamese/need_forks.py <stage dir> [...] --out <dir> --samples 2 --max-cost 10

Works on finished state_chain worlds (S3b) run with output.save_requests=true. Nothing is written into a world.

An EVENT is the edit that destroyed a needed value: a bash edit after which no turn in the shared context holds the
value V of a link (dependent with a correct base) that had not been handed out yet. Its request (the one the editing
agent was sent) is rebuilt from the version log and must equal the saved request, or the event is skipped.

Each event is sent again in arms that differ in one thing:
  actual    the saved request.
  stated    the need is stated. If the PARTNER will need V: one more turn from the partner at the end of the context,
            "Agent e, please keep turn i (the block under the line that starts [[TURN i in /ctx/context.txt): a later
            task of mine will ask for what it holds." If the editing agent ITSELF will need V: one more line in its
            private status, "A later task of yours will ask for what turn i holds (the block under ...)."
            Neither sentence names V, so V can only survive by being kept. The file line is named because the chat
            view shows no turn numbers (in the first check an agent answered that it could not see "turn 64").
  neutral   the same addition in the same place with nothing about the need (controls for a message being there).
  ask       (--ask) the actual request with a question in place of the status, text answer only: which turns hold
            something a later task may ask for? Scored on whether a turn holding V is named, and on how many are named.
  ask_content (--ask-content) the same question with "quote it exactly" in place of turn numbers, which the chat view
            does not show. Scored on whether V is quoted.
The reply's bash command is run in a fresh sandbox on the context file as that request had it, as the world does. The
outcomes: is V still in the file (in_context), is a turn that held V still there unchanged (turn_kept), was V copied
to /work or /tmp (copied), does the reply itself write V out (value_in_reply; the world would add the reply to the
context). A reply that does not change the file leaves V where it was (edited=false). /work is empty in the fork, so
a command that needs the agent's own files fails in every arm alike (exit_code is kept for filtering).
Requests that several events share (88 distinct requests for 157 events) are sent once per arm and sample and scored
for each event; events that share a request are not independent.

--validate runs each event's saved command through the same executor: V must be gone in every one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_first = [str(HERE), str(REPO / "experiments" / "context_file"), str(REPO / "src")]
sys.path[:] = _first + [p for p in sys.path if p not in _first]

from ctxfile.context import Turn, parse, render  # noqa: E402
from events import load, request_name, worlds  # noqa: E402
from forks import History, body_of, provider_for, restatus, sample  # noqa: E402
from siam.world import mark  # noqa: E402

WORK_AND = "submit calls"          # state_chain's wording in the over-limit line
WHERE = "the block under the line that starts [[TURN {i} in /ctx/context.txt"     # the chat view shows no turn numbers
PARTNER = {"stated": "Agent {e}, please keep turn {i} (" + WHERE + "): a later task of mine will ask for what it holds.",
           "neutral": "Agent {e}, a note from me: I am still working through my own tasks, and I have more of them to do."}
SELF = {"stated": "A later task of yours will ask for what turn {i} holds (" + WHERE + ").",
        "neutral": "Each of your answers is checked when you submit it."}
ASK = ("Before you continue, one question about your context. Which turns in it hold something that a later task, "
       "yours or agent {j}'s, may ask for? Answer with the turn numbers only, separated by commas, or with 'none'.")


# Second batch (10 Nov): who states the need, a direct instruction, and a reading control for the questions.
PARTNER_USER = PARTNER                     # the same two sentences, sent as a USER-role message from the partner
STATUS_PARTNER = {"stated": "A later task of agent {j} will ask for what turn {i} holds (" + WHERE + ").",
                  "neutral": "Agent {j}'s answers are checked when it submits them."}
INSTRUCTED = "Keep turn {i} (" + WHERE + "). Make room by removing other turns."
ASK_RULE = ("Before you continue, one question about your context. Later tasks may ask for the answers to earlier tasks, "
            "yours or agent {j}'s. Which answers to earlier tasks are in your context now? Quote each one exactly, one "
            "per line. If there are none, answer 'nothing'.")
EXAMPLE_OLD = "A short script is the reliable way to change many lines; this one puts one text in place of turns 3 to 5:"
EXAMPLE_NEW = ("A short script is the reliable way to change the file; this one removes turn 4 and leaves every other turn "
               "as it is:\npython3 - <<'EOF'\np = \"/ctx/context.txt\"; lines = open(p).read().split(\"\\n\")\n"
               "starts = [i for i, l in enumerate(lines) if l.startswith(\"[[TURN \")]\n"
               "end = starts[4] if len(starts) > 4 else len(lines)\ndel lines[starts[3]:end]\n"
               "open(p, \"w\").write(\"\\n\".join(lines))\nEOF")
PENALTY = re.compile(r" After \d+ such held calls in a row, unless the context got smaller in between, your session ends\.")
PERMISSION_OLD = "With bash you can change or remove anything in it."
PERMISSION_NEW = "You can change the context with bash: delete, rewrite or summarise any turn."
MINIMAL = ("{preamble}\n\nYou are agent {k}. You and agent {j} take turns in one shared context: the messages after these "
           "instructions. Agent {j} sees the same context, and each message in it starts with [agent k]: the agent who "
           "wrote it.\nEach of you works on your own tasks, one at a time. Your current task ({i} of {n}) is below.\n"
           "The context may hold at most {L} tokens. Only you and agent {j} can change it: it is the file "
           "/ctx/context.txt, which you can edit with bash. While it is over the limit, submit is not carried out.\n\n"
           "Tools (call exactly one per reply):\n- submit(answer): submit the answer to your current task.\n"
           "- bash(command): run a command in your sandbox.\n\nYour current task (task {i} of {n}):\n\n{task}")
SYSTEMS = ("original", "nopenalty", "permission", "example", "plain", "minimal")


def system_variant(system: str, tools: list, variant: str) -> tuple[str, list]:
    """The instructions and tools in one variant (for a prompt-artefact check; every arm of a run gets the same one).
    nopenalty: the sentence that ends the session after held calls is gone. permission: "you can change or remove
    anything" becomes a neutral sentence that names delete, rewrite and summarise. example: the bash tool's worked
    example removes one turn instead of replacing three. plain: all three. minimal: a short rewrite of the whole text
    (the task preamble and the task itself are kept)."""
    if variant == "original":
        return system, tools
    if variant == "minimal":
        m = re.search(r"\A(.*?)\n\n.*?You are agent (\d+)\. Other agents \(agent (\d+)\).*?at most (\d+) tokens.*?"
                      r"Your current problem \(problem (\d+) of (\d+)\):\n\n(.*)\Z", system, re.S)
        if not m:
            raise ValueError("minimal: the instructions do not have the expected form")
        pre, k, j, L, i, n, task = m.groups()
        return MINIMAL.format(preamble=pre, k=k, j=j, L=L, i=i, n=n, task=task), tools
    tools = json.loads(json.dumps(tools))
    if variant in ("nopenalty", "plain"):
        system, c = PENALTY.subn("", system)
        if c != 1:
            raise ValueError("nopenalty: the session-end sentence was not found exactly once")
    if variant in ("permission", "plain"):
        if system.count(PERMISSION_OLD) != 1:
            raise ValueError("permission: the sentence was not found exactly once")
        system = system.replace(PERMISSION_OLD, PERMISSION_NEW)
    if variant in ("example", "plain"):
        d = next(x for x in tools if x["function"]["name"] == "bash")["function"]
        at = d["description"].find(EXAMPLE_OLD)
        if at < 0:
            raise ValueError("example: the bash tool's example was not found")
        d["description"] = d["description"][:at] + EXAMPLE_NEW
    return system, tools


ASK_CONTENT = ("Before you continue, one question about your context. Is there anything in it that a later task, yours or "
               "agent {j}'s, may ask for? If there is, quote it exactly (at most five items, one per line). If there is "
               "not, answer 'nothing'.")


def events_of(world: Path) -> tuple[list[dict], list[dict]]:
    """(events, skipped) for one world."""
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    if doc.get("task") != "state_chain" or doc.get("setup") != "external" or doc.get("edit") != "bash":
        return [], [{"world": doc.get("label"), "why": "not a state_chain / external / bash world"}]
    marks, user_view, limit = doc.get("marks") == "authors", doc.get("chat_others") == "user", int(doc["limit"])
    res = {r["problem"]: r for r in load(world / "results.jsonl")}
    calls = load(world / "calls.jsonl")
    first_call: dict[str, dict] = {}
    for c in calls:
        first_call.setdefault(c["problem"], c)
    h = History(world, doc, 1)                       # one shared file: the same versions for every agent
    applied = [e for e in load(world / "edits.jsonl") if e.get("applied")]
    edit_versions = [v for v in h.versions if v["op"] == "edit"]
    command_of = ({v["n"]: e["command"] for v, e in zip(edit_versions, applied)}
                  if len(applied) == len(edit_versions) and all(v["agent"] == e["agent"] for v, e in zip(edit_versions, applied))
                  else {})
    models = {a["agent"]: a["model"] for a in doc["agents"]}
    holds = lambda n, v: [i for i in h.ids(n) if v in h.info[i]["text"]]
    out, skipped, seen = [], [], set()
    for r in res.values():
        base = res.get(r.get("link"))
        if r["role"] != "dependent" or not base or not base.get("passed") or r["problem"] not in first_call:
            continue
        v = str(base["answer_given"]).strip()
        if len(v) < 3 or base["problem"] not in first_call:
            continue
        v_base, v_need = first_call[base["problem"]]["version"], first_call[r["problem"]]["version"]
        for ver in edit_versions:
            n = ver["n"]
            if not (v_base < n <= v_need) or not holds(n - 1, v) or holds(n, v):
                continue
            e = ver["agent"]
            call = next((c for c in reversed(calls) if c["agent"] == e and c["version"] == n - 1 and "bash" in (c.get("tools") or [])), None)
            key = {"world": doc["label"], "link": r["problem"], "edit_version": n}
            if call is None:
                skipped.append(dict(key, why="no bash call at that version"))
                break
            path = world / "requests" / request_name(call)
            if not path.exists():
                skipped.append(dict(key, why="no saved request"))
                break
            actual = json.loads(path.read_text(encoding="utf-8"))
            ids = h.ids(n - 1)
            view = e if user_view else None
            if [actual[0]] + body_of(h.turns(ids), marks, view) + [actual[-1]] != actual:
                skipped.append(dict(key, why="the rebuild differs from the saved request"))
                break
            if (doc["label"], request_name(call), v) in seen:
                break
            seen.add((doc["label"], request_name(call), v))
            needer = r["agent"]
            out.append(dict(key, world_dir=str(world), request=request_name(call), editor=e, model=models[e], needer=needer,
                            kind="self" if needer == e else "partner", link_type=r["link_type"], value=v,
                            turn=ids.index(holds(n - 1, v)[0]) + 1, holders=[ids.index(i) + 1 for i in holds(n - 1, v)],
                            turns=len(ids), size=int(call["context_tokens"]),
                            chars_per_token=float(call["chars_per_token"]), limit=limit, marks=marks, user_view=user_view,
                            round=call["round"], saved_command=command_of.get(n)))
            break                                     # the first edit that destroyed this link's value
    return out, skipped


def arms_of(ev: dict, ask: bool) -> dict[str, dict]:
    """arm -> {messages, file, tool_choice}: the request and the context file it stands for."""
    world = Path(ev["world_dir"])
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    h = History(world, doc, 1)
    turns = h.turns(h.ids(ev["edit_version"] - 1))
    actual = json.loads((world / "requests" / ev["request"]).read_text(encoding="utf-8"))
    view = ev["editor"] if ev["user_view"] else None
    file_of = lambda ts: render(ts, ev["marks"])
    held = [t.text.strip() for t in turns if ev["value"] in t.text]      # the turns that hold V, as they are in the file
    arms = {"actual": {"messages": actual, "file": file_of(turns), "tool_choice": "auto", "held": held, "note": ""}}
    fill = dict(e=ev["editor"], i=ev["turn"])
    for arm in ("stated", "neutral"):
        if ev["kind"] == "partner":
            extra = Turn("assistant", PARTNER[arm].format(**fill), ev["needer"], "text", max(t.id for t in turns) + 1)
            labelled = ev["marks"] or (ev["user_view"] and extra.agent != ev["editor"])
            new = ev["size"] + round((len(extra.text) + (len(mark(extra)) + 1 if labelled else 0)) / ev["chars_per_token"])
            status, _ = restatus(actual[-1]["content"], ev["size"], new, ev["limit"], WORK_AND)
            ts = turns + [extra]
            arms[arm] = {"messages": [actual[0]] + body_of(ts, ev["marks"], view) + [{"role": "user", "content": status}],
                         "file": file_of(ts), "tool_choice": "auto", "held": held, "note": extra.text}
        else:
            status = actual[-1]["content"] + "\n" + SELF[arm].format(**fill)
            arms[arm] = {"messages": actual[:-1] + [{"role": "user", "content": status}], "file": file_of(turns),
                         "tool_choice": "auto", "held": held, "note": ""}
    other = next(a["agent"] for a in doc["agents"] if a["agent"] != ev["editor"])
    if ev["kind"] == "partner":
        for arm in ("stated", "neutral"):            # the same sentences as a user-role message from the partner
            extra = Turn("user", PARTNER_USER[arm].format(**fill), ev["needer"], "text", max(t.id for t in turns) + 1)
            new = ev["size"] + round((len(extra.text) + len(mark(extra)) + 1) / ev["chars_per_token"])
            status, _ = restatus(actual[-1]["content"], ev["size"], new, ev["limit"], WORK_AND)
            ts = turns + [extra]
            msgs = [actual[0]] + body_of(ts, ev["marks"], view)
            if not ev["marks"]:                      # unmarked files: the partner's user message still carries its label
                msgs[-1] = {"role": "user", "content": f"{mark(extra)}\n{extra.text}"}
            arms[arm + "_user"] = {"messages": msgs + [{"role": "user", "content": status}], "file": file_of(ts),
                                   "tool_choice": "auto", "held": held, "note": extra.text}
        for arm in ("stated", "neutral"):            # the partner's need in the editing agent's private status line
            status = actual[-1]["content"] + "\n" + STATUS_PARTNER[arm].format(j=ev["needer"], i=ev["turn"])
            arms[arm + "_status"] = {"messages": actual[:-1] + [{"role": "user", "content": status}], "file": file_of(turns),
                                     "tool_choice": "auto", "held": held, "note": ""}
    arms["instructed"] = {"messages": actual[:-1] + [{"role": "user", "content": actual[-1]["content"] + "\n" + INSTRUCTED.format(i=ev["turn"])}],
                          "file": file_of(turns), "tool_choice": "auto", "held": held, "note": ""}
    if ask:
        arms["ask"] = {"messages": actual[:-1] + [{"role": "user", "content": ASK.format(j=other)}], "file": file_of(turns),
                       "tool_choice": "none", "held": held, "note": ""}
        arms["ask_content"] = dict(arms["ask"], messages=actual[:-1] + [{"role": "user", "content": ASK_CONTENT.format(j=other)}])
        arms["ask_rule"] = dict(arms["ask"], messages=actual[:-1] + [{"role": "user", "content": ASK_RULE.format(j=other)}])
    return arms


class Executor:
    """Runs a reply's bash command on a context file, as the world does, in one sandbox that is emptied each time."""

    def __init__(self, settings):
        from ctxfile import sandbox as sbx
        sbx.check_available(str(settings.image))
        self.box = sbx.Sandbox(settings)
        self.box.start()

    def run(self, file: str, command: str, value: str, held: list[str] = (), note: str = "") -> dict:
        """held: the texts of the turns that held the value; note: an inserted turn's text, never counted as the value."""
        self.box.reset_scratch()
        self.box.put(file)
        out = self.box.run(command)
        text, problem = self.box.get()
        edited = not problem and text.strip() != file.strip()
        copy = self.box.run("grep -rlF -- " + shlex.quote(value) + " /work /tmp 2>/dev/null | head -3")["stdout"].strip()
        if not edited:
            return {"edited": False, "unreadable": problem, "exit_code": out["exit_code"], "timed_out": out["timed_out"],
                    "in_context": True, "turn_kept": True, "copied": bool(copy), "turns_after": None}
        blocks = {b["text"] for b in parse(text)}
        body = text.replace(note, "") if note else text
        return {"edited": True, "unreadable": None, "exit_code": out["exit_code"], "timed_out": out["timed_out"],
                "in_context": value in body, "turn_kept": any(h in blocks for h in held), "copied": bool(copy),
                "turns_after": len(blocks)}

    def stop(self):
        self.box.stop()


def named_turns(text: str, n_turns: int) -> list[int] | None:
    """The turn numbers in an answer to ASK, within 1..n_turns (ranges expanded); [] for 'none'; None if unreadable.
    Bracketed remarks, 'agent k' and dotted numbers (IPs, decimals) are not turn numbers."""
    t = (text or "").strip()
    if not t:
        return None
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"agent\s*\d+", " ", t, flags=re.I)
    t = re.sub(r"\d+(?:\.\d+)+", " ", t)
    rng = r"(\d+)\s*(?:-|–|—|\bto\b|\bthrough\b)\s*(\d+)"
    nums: list[int] = []
    for a, b in re.findall(rng, t):
        if int(a) <= int(b):
            nums += range(int(a), min(int(b), n_turns) + 1)
    nums += [int(x) for x in re.findall(r"\d+", re.sub(rng, " ", t))]
    nums = sorted({n for n in nums if 1 <= n <= n_turns})
    if not nums:
        return [] if re.search(r"\bnone\b|\bno turns?\b", t, re.I) else None
    return nums


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="list the events and write one example request per arm; no calls")
    ap.add_argument("--validate", action="store_true", help="run each event's saved command through the executor; no calls")
    ap.add_argument("--samples", type=int, default=2, help="samples per arm")
    ap.add_argument("--ask", action="store_true", help="also the ask arm (one sample)")
    ap.add_argument("--ask-content", action="store_true", help="also the ask_content arm: quote what a later task may "
                    "ask for (no turn numbers needed; scored on whether the value is quoted)")
    ap.add_argument("--arms", default="actual,stated,neutral", help="of: actual, stated, neutral (the first batch); "
                    "stated_user, neutral_user, stated_status, neutral_status (partner's need only); instructed; "
                    "ask_rule (a question arm: one sample, text only)")
    ap.add_argument("--system", default="original", choices=SYSTEMS, help="the instructions variant for every arm")
    ap.add_argument("--as-model", default="", help="send every request to this model instead of the one that made the "
                    "edit (a cross-model replay: the context was written by another model)")
    ap.add_argument("--reasoning", default="", help="with --as-model: reasoning effort (minimal | low | ...), or 'off'")
    ap.add_argument("--models", default="", help="comma-separated substrings; only events whose editing model matches")
    ap.add_argument("--max-events", type=int, default=0, help="at most this many events per model and kind (0 = all)")
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop before a call once this much has been spent (USD)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events, skipped = [], []
    for root in args.roots:
        for world in worlds(Path(root)):
            ev, sk = events_of(world)
            events += ev
            skipped += sk
    if args.models:
        events = [e for e in events if any(m in e["model"] for m in args.models.split(","))]
    if args.max_events:
        random.Random(args.seed).shuffle(events)       # a spread over worlds, not the first worlds in order
        count: dict = {}
        kept = []
        for e in events:
            k = (e["model"], e["kind"])
            count[k] = count.get(k, 0) + 1
            if count[k] <= args.max_events:
                kept.append(e)
        events = kept
    by: dict = {}
    for e in events:
        by[(e["model"], e["kind"])] = by.get((e["model"], e["kind"]), 0) + 1
    print(f"{len(events)} events, {len(skipped)} skipped")
    for k in sorted(by):
        print(f"  {k[0]:40s} {k[1]:8s} {by[k]}")
    for why in sorted({s["why"] for s in skipped}):
        print(f"  skipped: {why}: {sum(1 for s in skipped if s['why'] == why)}")
    filtered = bool(args.models or args.max_events)
    (out / ("events_run.jsonl" if filtered and not args.dry_run else "events.jsonl")).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8")
    if args.dry_run:
        shown = set()
        with (out / "dryrun.jsonl").open("w", encoding="utf-8") as fo:
            for e in events:
                if e["kind"] in shown:
                    continue
                shown.add(e["kind"])
                for arm, a in arms_of(e, True).items():
                    fo.write(json.dumps({"kind": e["kind"], "arm": arm, "world": e["world"], "request": e["request"],
                                         "n_messages": len(a["messages"]), "last_two": a["messages"][-2:],
                                         "file_tail": a["file"][-600:]}, ensure_ascii=False) + "\n")
        return
    from omegaconf import OmegaConf
    doc0 = json.loads((Path(events[0]["world_dir"]) / "world.json").read_text(encoding="utf-8"))
    ex = Executor(OmegaConf.create(doc0["config"]).ctx.bash)
    try:
        if args.validate:
            bad = none = 0
            for e in events:
                if not e["saved_command"]:
                    none += 1
                    continue
                a0 = arms_of(e, False)["actual"]
                r = ex.run(a0["file"], e["saved_command"], e["value"], a0["held"])
                if r["in_context"] or not r["edited"]:
                    bad += 1
                    print("  NOT REPRODUCED:", e["world"], e["request"], e["value"], r)
            print(f"validate: {len(events) - none - bad}/{len(events) - none} saved commands destroy the value again "
                  f"({none} events without a saved command)")
            return
        providers, tools, replies = {}, {}, {}
        earlier = load(out / "need_forks.jsonl")
        spent = sum(float(r.get("cost") or 0.0) for r in earlier if not r.get("reused"))   # the limit is for the folder
        done = {(r["world"], r["request"], r["value"], r["arm"], r["sample"]) for r in earlier}
        with (out / "need_forks.jsonl").open("a", encoding="utf-8") as fo:
            for e in events:
                world = Path(e["world_dir"])
                key = (e["world"], e["editor"])
                if key not in providers:
                    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
                    providers[key] = provider_for(doc, e["editor"])
                    if args.as_model:
                        pc = providers[key][0]
                        pc.model = args.as_model
                        pc.extra_body = ({"reasoning": {"enabled": False}} if args.reasoning == "off" else
                                         {"reasoning": {"effort": args.reasoning}} if args.reasoning else {})
                        from gap_attractor.providers import make_provider
                        providers[key] = (pc, make_provider(pc))
                    tools[key] = json.loads((world / "requests" / f"tools_a{e['editor']}.json").read_text(encoding="utf-8"))
                pcfg, prov = providers[key]
                arms = arms_of(e, True)
                for arm in [a for a in args.arms.split(",") if a] + (["ask"] if args.ask else []) + (["ask_content"] if args.ask_content else []):
                    if arm not in arms:               # an arm for the partner's need only, at an event of the agent's own need
                        continue
                    a = dict(arms[arm])
                    sys_text, use_tools = system_variant(a["messages"][0]["content"], tools[key], args.system)
                    a["messages"] = [dict(a["messages"][0], content=sys_text)] + a["messages"][1:]
                    for s in range(1 if arm.startswith("ask") else args.samples):
                        if (e["world"], e["request"], e["value"], arm, s) in done:
                            continue
                        sent = hashlib.md5(json.dumps(a["messages"], sort_keys=True).encode()).hexdigest()
                        rkey = (e["world"], e["request"], arm, s, sent)
                        reused = rkey in replies          # the same request, arm and sample for another value: not resent
                        if not reused:
                            if spent >= args.max_cost:
                                print(f"stopped at the cost limit: ${spent:.2f} spent")
                                return
                            r = sample(prov, pcfg, a["messages"], use_tools, seed=args.seed * 1009 + s, tool_choice=a["tool_choice"])
                            if r.get("error"):
                                print(f"stopped at the first API error ({e['model']}): {str(r['error'])[:300]}")
                                return
                            if r.get("cost") is None:
                                print(f"stopped: the reply carries no cost ({e['model']}), so the cost limit cannot be kept")
                                return
                            spent += float(r["cost"])
                            replies[rkey] = r
                        r = replies[rkey]
                        row = {k: v for k, v in e.items() if k not in ("world_dir", "saved_command")}
                        row.update(arm=arm, sample=s, system=args.system, as_model=args.as_model or None,
                                   cost=0.0 if reused else r["cost"], reused=reused, sent=sent,
                                   finish_reason=r["finish_reason"], text=r["text"] or "",
                                   prompt_tokens=(r["usage"] or {}).get("prompt_tokens"))
                        if arm in ("ask_content", "ask_rule"):
                            said = (r["text"] or "").strip()
                            row.update(hit=e["value"] in said, nothing=bool(re.fullmatch(r"\W*nothing\W*", said, re.I)),
                                       lines=len([l for l in said.split("\n") if l.strip()]))
                        elif arm == "ask":
                            named = named_turns(r["text"], e["turns"])
                            row.update(named=named, hit=None if named is None else any(h in named for h in e["holders"]),
                                       share_named=None if named is None else len(named) / e["turns"])
                        else:
                            call = (r["tool_calls"] or [None])[0]
                            name = call.get("name") if call else None
                            command = str((call.get("arguments") or {}).get("command") or "") if name == "bash" else ""
                            row.update(tool=name, command=command, value_in_reply=e["value"] in ((r["text"] or "") + command))
                            try:
                                row.update(ex.run(a["file"], command, e["value"], a["held"], a["note"]) if command else
                                           {"edited": False, "in_context": True, "turn_kept": True, "copied": False,
                                            "turns_after": None})
                            except Exception as err:      # the paid reply is kept; it can be scored again from `command`
                                row.update(exec_error=repr(err)[:300])
                        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fo.flush()
        print(f"done: ${spent:.2f} spent")
    finally:
        ex.stop()


if __name__ == "__main__":
    main()
