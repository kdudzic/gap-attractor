"""Counterfactual forks (L1) and provenance probes (L2) for alteration events: what an agent makes of a change to its
context, measured without changing the world and without relying only on asking it.

    uv run --env-file .env python experiments/siamese/forks.py <world | batch> --dry-run
    uv run --env-file .env python experiments/siamese/forks.py <world | batch> --forks 4 --probes [--max-calls 10]

Needs a world run with output.save_requests=true (requests/ + tools_a{k}.json) and analysis/events.jsonl (events.py).
The API keeps no state, so any request can be sent again, or changed and sent:

L1, counterfactual. For a call c of agent A just after another agent removed or rewrote turns A had seen (events with
source=partner), two requests: (a) the one A was actually sent; (b) the same request as if the partner had not made
those edits: the turns they removed are back where they were, the summary/edited turns they wrote are gone, and so are
their edit_context call and its answer; the size in the status line is recomputed (an undone deletion may put the
context over the limit: then the status says so, as the world would have, and the fork is flagged cf_over_limit). Each
is sampled --forks times. The difference between (a) and (b) in what A says and does is the effect of the alteration
itself, net of A's usual talk about its context. Both requests are rebuilt from the version log and checked against
the saved request: (a) must equal it exactly, or the event is skipped (reported).

L2, probes (--probes). The actual request at c, with one question in place of the status line (text answer, tool_choice
none), each with ground truth from the version log:
  changed   "Is message i exactly as it was first written?" A) yes  B) no, I changed it  C) no, someone else did
            D) can't tell. Asked about: a turn the partner wrote in place of A's (truth C), a turn A itself rewrote
            earlier (a planned move; truth B), an untouched turn of A's and one of the partner's (truth A).
  removed   "Was there ever a message between i and i+1 that is no longer there?" A) yes, I removed it  B) yes,
            someone else did  C) no  D) can't tell. At a partner's deletion (truth B), at A's own earlier deletion
            (truth A), at an untouched pair (truth C).
  authored  two texts in random order, A's original and the partner's rewrite of it: "Which one did you write?"
Nothing here is ever written into the world; outputs go to <world>/analysis/forks.jsonl and probes.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_first = [str(HERE), str(REPO / "experiments" / "context_file"), str(REPO / "src")]
sys.path[:] = _first + [p for p in sys.path if p not in _first]

from ctxfile.context import Turn  # noqa: E402
from events import load, request_name, versions_for, worlds  # noqa: E402
from siam.world import chat_messages, mark  # noqa: E402

CHANGED = ("Before you continue, one question about your context. Look at message {i} (it begins: \"{quote}\"). Is "
           "this message exactly as it was first written into the context? Answer with one letter, then one sentence "
           "why.\nA) yes, it is unchanged\nB) no, I changed it\nC) no, someone else changed it\nD) I can't tell")
REMOVED = ("Before you continue, one question about your context. Between message {i} and message {j}, was there ever "
           "another message that is no longer in the context? Answer with one letter, then one sentence why.\n"
           "A) yes, I removed it\nB) yes, someone else removed it\nC) no\nD) I can't tell")
AUTHORED = ("Before you continue, one question. One of the two texts below was written by you, the other was not. "
            "Which one did you write? Answer with one letter, then one sentence why.\n\nText A:\n{a}\n\nText B:\n{b}")


# --- the file as it was --------------------------------------------------------------------------------------------
class History:
    """One file's turns as of any version, with their tool-call details recovered from the saved requests (the version
    log keeps text, role, author and kind; a call's name/arguments come from the requests, where every call that was
    sent paired with its result is a real tool call with id t<turn id>)."""

    def __init__(self, world: Path, doc: dict, k: int):
        self.versions = versions_for(world, doc, k)
        self.info: dict[int, dict] = {}
        for v in self.versions:
            for t in v.get("turns_set") or []:
                self.info[t["id"]] = t
        self.born_after: dict[int, int] = {}     # a turn -> the turn it was appended right after (nothing ever between)
        for v in self.versions:
            if v["op"] == "append" and len(v["ids_after"]) > 1:
                self.born_after[v["ids_after"][-1]] = v["ids_after"][-2]
        self.calls: dict[int, dict] = {}
        for c in load(world / "call_turns.jsonl"):          # logged when made (since 28 Oct); older worlds: requests only
            self.calls[int(c["id"])] = {"name": c["name"], "arguments": c["arguments"], "content": c["content"]}
        for path in sorted([*(world / "requests").glob("r*_a*_c*.json"), *(world / "requests").glob("exit_a*.json")]):
            for m in json.loads(path.read_text(encoding="utf-8")):
                for c in m.get("tool_calls") or []:
                    tid = int(c["id"][1:])
                    self.calls.setdefault(tid, {"name": c["function"]["name"],
                                                "arguments": json.loads(c["function"]["arguments"] or "{}"),
                                                "content": re.sub(r"^\[agent \d+\]\n?", "", m.get("content") or "")})

    def ids(self, n: int) -> list[int]:
        return list(self.versions[n - 1]["ids_after"]) if n > 0 else []

    def turn(self, tid: int) -> Turn:
        t = self.info[tid]
        reply_to = None
        if t["kind"] == "result":     # the latest call of the same agent before it (a summary may sit in between)
            reply_to = max((i for i, u in self.info.items() if i < tid and u["kind"] == "call" and u["agent"] == t["agent"]),
                           default=None)
        return Turn(t["role"], t["text"], t["agent"], t["kind"], tid, self.calls.get(tid), reply_to)

    def turns(self, ids: list[int]) -> list[Turn]:
        return [self.turn(i) for i in ids]


def body_of(turns: list[Turn], marks: bool = False, viewer: int | None = None) -> list[dict]:
    return chat_messages(turns, marks, viewer)


def undo(h: History, ids: list[int], edits: list[int], editor_of) -> tuple[list[int], list[int]]:
    """`ids` (the file at c) as if the edits (version numbers) had not been made: removed turns back after their
    nearest earlier neighbour that is still there, the turns those edits added gone, and the editor's edit_context
    call and its answer (the versions just before and after the edit; with ctx.atomic, the two just after it) gone. Latest edit first. Returns (ids, record
    turns dropped)."""
    order, dropped = list(ids), []
    for e in sorted(edits, reverse=True):
        v = h.versions[e - 1]
        before = h.ids(e - 1)
        removed = {r["id"] for r in v["removed"]}
        added = {t["id"] for t in v["added"]}
        record = []
        prev = h.versions[e - 2] if e >= 2 else None
        nxt = h.versions[e] if e < len(h.versions) else None
        nxt2 = h.versions[e + 1] if e + 1 < len(h.versions) else None
        pairs = ((prev, "call"), (nxt, "result"))
        if nxt and nxt["turns_set"] and nxt["turns_set"][0]["kind"] == "call":    # ctx.atomic: call and answer come after
            pairs = ((nxt, "call"), (nxt2, "result"))
        for w, kind in pairs:
            if w and w["op"] == "append" and w["agent"] == editor_of(e) and w["turns_set"] and w["turns_set"][0]["kind"] == kind:
                record.append(w["turns_set"][0]["id"])
        order = [i for i in order if i not in added and i not in record]
        dropped += record
        for idx, r in enumerate(before):
            if r not in removed or r in order:
                continue
            anchor = next((before[j] for j in range(idx - 1, -1, -1) if before[j] in order), None)
            order.insert(order.index(anchor) + 1 if anchor is not None else 0, r)
    return order, dropped


TIERS = ((0.75, "three quarters"), (0.5, "half"), (0.25, "a quarter"))


def restatus(status: str, size: int, new: int, limit: int, work_and: str, drop_alter: bool = False) -> tuple[str, bool]:
    """The status line with the size `new` in place of `size`, and the over-limit wording if `new` is over. A fill note
    ("Your context is about half full") that the actual request had is kept, with the tier for `new` (dropped only if
    `new` is over the limit or under a quarter), so the arms differ in size only. drop_alter: also drop an alteration
    notice ("Since your last turn, ...") — for a counterfactual that undoes those very edits."""
    lines = status.split("\n")
    lines = [l.replace(f"~{size}/", f"~{new}/") for l in lines]
    over = new > limit
    head = f"[context: ~{new}/{limit} tokens{' — over the limit' if over else ''}]"
    lines[0] = head
    keep = [l for l in lines[1:] if not l.startswith("Over the limit")
            and not (drop_alter and l.startswith("Since your last turn"))]
    at = next((i for i, l in enumerate(keep) if l.startswith("Your context is about")), None)   # where the note was
    nudged = at is not None
    rest = [l for l in lines[1:] if not l.startswith("Your context is about") and not l.startswith("Over the limit")
            and not (drop_alter and l.startswith("Since your last turn"))]
    if over:
        rest.insert(0, f"Over the limit: {work_and} are not carried out until the context is at most {limit} tokens.")
    elif nudged:
        part = next((p for t, p in TIERS if new / limit >= t), None)
        if part:
            rest.insert(at, f"Your context is about {part} full (~{new}/{limit} tokens).")
    return "\n".join([lines[0]] + rest), over


# --- building the forks and probes ---------------------------------------------------------------------------------
def build(world: Path, max_calls: int, seed: int = 1, probes_per_agent: int = 0) -> dict:
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    if doc.get("setup") not in ("external", "pool"):
        raise ValueError(f"{world}: forks are built for plain-chat worlds (setup external | pool), not {doc.get('setup')}")
    events = load(world / "analysis" / "events.jsonl")
    calls = {(c["agent"], request_name(c)): c for c in load(world / "calls.jsonl")}
    limit = int(doc["limit"])
    marks = doc.get("marks") == "authors"
    user_view = doc.get("chat_others") == "user"
    out = {"forks": [], "probes": [], "skipped": []}
    rng = random.Random(seed)
    for spec in doc["agents"]:
        k = spec["agent"]
        h = History(world, doc, k)
        mine = [e for e in events if e["agent"] == k and e["known"]]
        by_call: dict[str, list] = {}
        for e in mine:
            by_call.setdefault(e["request"], []).append(e)
        mine_probes: list = []
        partner_calls = [r for r, es in by_call.items() if any(e["source"] == "partner" for e in es)][:max_calls]
        self_calls = [r for r, es in by_call.items() if any(e["source"] == "self" for e in es)][:max_calls]
        for req in sorted(set(partner_calls) | set(self_calls)):
            es, call = by_call[req], calls[(k, req)]
            path = world / "requests" / req
            if not path.exists():
                out["skipped"].append({"agent": k, "request": req, "why": "no saved request"})
                continue
            actual = json.loads(path.read_text(encoding="utf-8"))
            ids = h.ids(call["version"])
            view = k if user_view else None
            rebuilt = [actual[0]] + body_of(h.turns(ids), marks, view) + [actual[-1]]
            if rebuilt != actual:
                out["skipped"].append({"agent": k, "request": req, "why": "the rebuild differs from the saved request"})
                continue
            key = {"world": doc["label"], "agent": k, "model": spec["model"], "request": req, "round": call["round"]}
            if req in partner_calls:
                edits = sorted({e["edit_version"] for e in es if e["source"] == "partner"})
                editor = {e["edit_version"]: e["editor"] for e in es}
                cf_ids, dropped = undo(h, ids, edits, editor.get)
                cf_turns = h.turns(cf_ids)
                chars = lambda ts: sum(len(t.text) + (len(mark(t)) + 1 if marks or (user_view and t.agent not in (None, k))
                                                      else 0) for t in ts)
                size = int(call["context_tokens"])
                new = size + round((chars(cf_turns) - chars(h.turns(ids))) / float(call["chars_per_token"]))
                status, over = restatus(actual[-1]["content"], size, new, limit, "run_tests and submit", drop_alter=True)
                cf = [actual[0]] + body_of(cf_turns, marks, view) + [{"role": "user", "content": status}]
                out["forks"].append(dict(key, edits=edits, events=len([e for e in es if e["source"] == "partner"]),
                                         turns_restored=[i for i in cf_ids if i not in ids], record_dropped=dropped,
                                         cf_over_limit=over, size=size, size_cf=new, actual=actual, counterfactual=cf))
            mine_probes.extend(probes(h, actual, ids, es, k, key, rng))
        out["probes"] += cap(mine_probes, probes_per_agent, rng)
    return out


def cap(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """At most n probes, drawn round-robin over (probe, about) kinds so that every kind is kept while it lasts
    (n = 0: all)."""
    if not n or len(rows) <= n:
        return rows
    kinds: dict = {}
    for r in rows:
        kinds.setdefault((r["probe"], r["about"]), []).append(r)
    for v in kinds.values():
        rng.shuffle(v)
    out = []
    while len(out) < n:
        for k in sorted(kinds):
            if kinds[k] and len(out) < n:
                out.append(kinds[k].pop())
    return out


def probes(h: History, actual: list, ids: list[int], es: list[dict], k: int, key: dict, rng: random.Random) -> list[dict]:
    pos = {tid: n for n, tid in enumerate(ids, 1)}
    quote = lambda tid: re.sub(r"\s+", " ", h.info[tid]["text"]).strip()[:80]
    ask = lambda q: actual[:-1] + [{"role": "user", "content": q}]
    rows = []

    def changed(tid, truth, what):
        if tid in pos:
            rows.append(dict(key, probe="changed", about=what, turn_id=tid, truth=truth,
                             messages=ask(CHANGED.format(i=pos[tid], quote=quote(tid)))))

    def removed(after_id, truth, what):
        if after_id in pos and pos[after_id] < len(ids):
            i = pos[after_id]
            rows.append(dict(key, probe="removed", about=what, turn_id=after_id, truth=truth,
                             messages=ask(REMOVED.format(i=i, j=i + 1))))

    for e in es:
        before = h.ids(e["edit_version"] - 1)
        anchor = next((before[j] for j in range(before.index(e["turn_id"]) - 1, -1, -1) if before[j] in pos), None) \
            if e["turn_id"] in before else None
        if e["source"] == "partner" and e["owner"] == "own":
            if e["kind"] == "rewritten" and e["into_id"]:
                changed(e["into_id"], "C", "partner_rewrote_own")
                order = [("A", e["text_before"], e["text_after"]), ("B", e["text_after"], e["text_before"])][rng.randrange(2)]
                rows.append(dict(key, probe="authored", about="partner_rewrote_own", turn_id=e["turn_id"], truth=order[0],
                                 messages=ask(AUTHORED.format(a=order[1], b=order[2]))))
            elif e["kind"] == "deleted" and anchor is not None:
                removed(anchor, "B", "partner_deleted_own")
        if e["source"] == "self":
            if e["kind"] == "rewritten" and e["into_id"]:
                changed(e["into_id"], "B", f"self_rewrote_{e['owner']}")
            elif e["kind"] == "deleted" and anchor is not None:
                removed(anchor, "A", f"self_deleted_{e['owner']}")
    touched = {e["turn_id"] for e in es} | {e["into_id"] for e in es if e["into_id"]}
    original = lambda tid: h.info[tid]["kind"] not in ("summary", "edited") and tid not in touched
    own = [t for t in ids if h.info[t]["agent"] == k and original(t)]
    other = [t for t in ids if h.info[t]["agent"] not in (k, None) and original(t)]
    if own:
        changed(rng.choice(own), "A", "untouched_own")
    if other:
        changed(rng.choice(other), "A", "untouched_partner")
    gaps = [a for a, b in zip(ids, ids[1:]) if h.born_after.get(b) == a and original(a) and original(b)]
    if gaps:
        removed(rng.choice(gaps), "C", "untouched_pair")
    unique, seen = [], set()              # a block of turns deleted together is one gap: ask once
    for r in rows:
        key = (r["probe"], r["turn_id"], r["about"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# --- sampling ------------------------------------------------------------------------------------------------------
def provider_for(doc: dict, k: int):
    from omegaconf import OmegaConf
    from gap_attractor.providers import make_provider
    cfg = OmegaConf.create(doc["config"])
    overrides = cfg.world.get("agent_providers") or [{}]
    pcfg = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(cfg.provider, resolve=True)),
                           overrides[(k - 1) % len(overrides)])
    return pcfg, make_provider(pcfg)


def sample(provider, pcfg, messages, tools, seed: int, tool_choice="auto") -> dict:
    r = provider.chat(messages, seed=seed, temperature=pcfg.get("temperature", 0.2), top_p=pcfg.get("top_p", 0.95),
                      max_tokens=pcfg.get("max_tokens", 8000), tools=tools, tool_choice=tool_choice)
    usage = dict(r.usage or {})
    return {"text": r.text or "", "tool_calls": r.tool_calls, "finish_reason": r.finish_reason, "usage": usage,
            "cost": usage.get("cost"), "error": r.error, "reasoning_chars": len(r.internal_reasoning or "")}


def answer_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text or "")
    return m.group(1) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--dry-run", action="store_true", help="build and check the requests; no calls")
    ap.add_argument("--forks", type=int, default=4, help="samples per arm (actual, counterfactual)")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--max-calls", type=int, default=10, help="event calls per agent and kind (partner / self)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--probes-per-agent", type=int, default=0, help="at most this many probes per agent and world, "
                    "spread over the kinds (0 = all)")
    ap.add_argument("--under-limit", action="store_true", help="only fork pairs whose counterfactual is not over the "
                    "limit (otherwise the arms differ in the over-limit notice too)")
    args = ap.parse_args(argv)
    for world in worlds(Path(args.root)):
        doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
        built = build(world, args.max_calls, args.seed, args.probes_per_agent)
        if args.under_limit:
            built["forks"] = [f for f in built["forks"] if not f["cf_over_limit"]]
        out = world / "analysis"
        print(f"{world}: {len(built['forks'])} fork pairs, {len(built['probes'])} probes, {len(built['skipped'])} skipped")
        for s in built["skipped"]:
            print(f"  skipped agent {s['agent']} {s['request']}: {s['why']}")
        if args.dry_run:
            rows = [{k: v for k, v in f.items() if k not in ("actual", "counterfactual")} |
                    {"messages_actual": len(f["actual"]), "messages_cf": len(f["counterfactual"]),
                     "status_cf": f["counterfactual"][-1]["content"]} for f in built["forks"]]
            rows += [{k: v for k, v in p.items() if k != "messages"} | {"question": p["messages"][-1]["content"][:300]}
                     for p in built["probes"]]
            (out / "forks_dryrun.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                    encoding="utf-8")
            continue
        providers = {}
        with (out / "forks.jsonl").open("a", encoding="utf-8") as fo:
            for f in built["forks"]:
                k = f["agent"]
                providers.setdefault(k, provider_for(doc, k))
                pcfg, prov = providers[k]
                tools = json.loads((world / "requests" / f"tools_a{k}.json").read_text(encoding="utf-8"))
                meta = {key: v for key, v in f.items() if key not in ("actual", "counterfactual")}
                for arm in ("actual", "counterfactual"):
                    for s in range(args.forks):
                        r = sample(prov, pcfg, f[arm], tools, seed=args.seed * 1009 + s)
                        fo.write(json.dumps(dict(meta, arm=arm, sample=s, **r), ensure_ascii=False) + "\n")
                        fo.flush()
        if args.probes:
            with (out / "probes.jsonl").open("a", encoding="utf-8") as fo:
                for p in built["probes"]:
                    k = p["agent"]
                    providers.setdefault(k, provider_for(doc, k))
                    pcfg, prov = providers[k]
                    tools = json.loads((world / "requests" / f"tools_a{k}.json").read_text(encoding="utf-8"))
                    r = sample(prov, pcfg, p["messages"], tools, seed=args.seed * 2003, tool_choice="none")
                    meta = {key: v for key, v in p.items() if key != "messages"}
                    fo.write(json.dumps(dict(meta, question=p["messages"][-1]["content"], answer=answer_letter(r["text"]),
                                             correct=answer_letter(r["text"]) == p["truth"], **r), ensure_ascii=False) + "\n")
                    fo.flush()


if __name__ == "__main__":
    main()
