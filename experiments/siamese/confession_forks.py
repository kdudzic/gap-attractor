"""Do an agent's own earlier confessions in its context cause the next one? (H2, and H5 for exit answers)

    uv run --env-file .env python experiments/siamese/confession_forks.py <world | batch> [--model haiku]
        [--per-world 6] [--min-present 2] [--samples 2] [--exit-samples 3] [--dry-run]

A *confession* is an agent's own turn whose words (code removed) say it worked on, submitted or ran code for the wrong
problem, or that something "isn't mine" (attractors.CONFESSION). Observationally (S0), the more of its own confessions
an agent's request holds, the likelier its next reply is one. These forks test whether the confessions in the context
cause that. For each chosen request, three versions are sent (each sampled `--samples` times):

- **actual:** the request as it was sent (rebuilt from the version log and checked equal to the saved request);
- **removed:** the agent's own confession turns taken out. A confession that was a plain message is dropped; one that
  was the words before a tool call loses its words and keeps the call (and so its result);
- **neutral:** each confession's words replaced by neutral text of the same length (so a shorter context cannot explain
  a difference). The text is a fixed set of task-agnostic working notes (NEUTRAL below: no problem, agent, task or
  ownership words), shuffled per request and never repeated within it. (The agent's own earlier messages were tried
  first; nearly all of Haiku's name a problem or say "other"/"different", so none were clean enough.)

A cut-off notice right after a removed confession goes with it. Requests are left out (and listed as skipped) when the
agent's last own turn is a confession (the status then asks it to call a tool, about that message), or when an arm
would cross the context limit differently from the actual request (held calls would differ for another reason).
Review 28 Oct (independent agent): these rules fix a neutral text drawn from the future that named other problems,
and status lines that differed between arms.

The status line is restated with each version's size (as in forks.py). Partners' confessions are never touched.

Requests chosen: the agent's step requests holding at least `--min-present` of its own confessions, at most
`--per-world` per agent and world, spread evenly over the world. Exit forks (`--exit-samples`): the same three
versions of the exit request (the final context, then the exit question), for agents whose final context holds at
least one own confession.

**Partner arms (`--partner`, 28 Oct):** the same requests (unmarked plain-chat worlds with a partner only), the agent's
own confessions left as they are, and the *partner's* turns presented differently:
- **marked:** every message starts with "[agent k]" (as world.marks=authors), plus that setting's sentence in the
  instructions;
- **named:** the partner's turns, its calls and results included, are user messages starting with "[agent k]" (as
  world.chat_others=user), plus that setting's sentence;
- **partner_calls_removed:** the partner's tool calls and their results taken out (its plain messages kept).
If confessions drop in these arms, the cause is reading the partner's work as one's own. Rows go to
`analysis/partner_forks.jsonl`.

Outcomes (`analysis/confession_forks.jsonl`, one row per sample): the reply's words (text and tool-call words), whether
it is a confession, its length and tool; for exit samples also self-blame and the claim of having submitted code for
other problems (the regexes of the H5 check). Nothing is written into the world itself.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from forks import History, body_of, provider_for, restatus, sample  # noqa: E402  (sets the import paths)
from ctxfile.context import Turn  # noqa: E402
from attractors import confession, words  # noqa: E402
from events import load, request_name, worlds  # noqa: E402
from siam.world import mark  # noqa: E402

SELF = re.compile(r"\bI (should have|shouldn't have|made (a|an|several|many)|mixed|confused|lost track|submitted (code|programs?|"
                  r"solutions?) for|never (solved|submitted|checked|verified)|didn'?t|did not|failed|kept|repeatedly|wasted|"
                  r"got stuck)|\bmy (own )?(mistake|error|fault|conduct|failure)|went (badly|poorly)", re.I)
CLAIM_WRONG = re.compile(r"(submitted|submitting|called `?submit`?|sent)[^.]{0,80}(other|different|another|wrong|unrelated|"
                         r"agent \d)[^.]{0,30}(problem|task)s?|(code|programs?|solutions?) for (at least )?(\w+ )?(other|"
                         r"different|unrelated|several|eight|six|\d+)[^.]{0,30}problems?", re.I)


def words_of(t: Turn) -> str:
    return (t.call or {}).get("content", "") if t.kind == "call" else t.text


OTHER = re.compile(r"problem\s*\d|agent\s*\d|\b(other|different|another|wrong|mine|my own)\b|task\s*\d", re.I)
CUT = "[this reply was cut off"


def is_confession(t: Turn, k: int) -> bool:
    return t.agent == k and t.role == "assistant" and t.kind in ("text", "call") and confession(words_of(t))


def without_words(t: Turn) -> Turn | None:
    """A confession turn with its words removed: None for a plain message; a call keeps its call."""
    if t.kind != "call" or not t.call:
        return None
    w = t.call.get("content", "") or ""
    return Turn(t.role, t.text.replace(w, "", 1).lstrip(), t.agent, t.kind, t.id, dict(t.call, content=""), t.reply_to)


def with_words(t: Turn, new: str) -> Turn:
    if t.kind == "call" and t.call:
        w = t.call.get("content", "") or ""
        return Turn(t.role, t.text.replace(w, new, 1), t.agent, t.kind, t.id, dict(t.call, content=new), t.reply_to)
    return Turn(t.role, new, t.agent, t.kind, t.id, t.call, t.reply_to)


def filler(pool: list[str], n: int) -> str | None:
    """Neutral text of length n, taken (and used up) from the front of `pool`; None if the pool runs out."""
    text = ""
    while len(text) < n and pool:
        text += (" " if text else "") + " ".join(pool.pop(0).split())
    return text[:n] if len(text) >= n else None


def versions(turns: list[Turn], k: int, pool: list[str]) -> dict[str, list[Turn] | None]:
    removed, neutral = [], []
    drop_next = False
    for t in turns:
        if drop_next and t.kind == "notice" and t.text.startswith(CUT):   # a cut-off notice goes with its message
            drop_next = False
            neutral = neutral + [t] if neutral is not None else None
            continue
        drop_next = False
        if is_confession(t, k):
            r = without_words(t)
            if r is not None:
                removed.append(r)
            else:
                drop_next = True
            text = filler(pool, len(words_of(t))) if neutral is not None else None
            neutral = neutral + [with_words(t, text)] if neutral is not None and text is not None else None
        else:
            removed.append(t)
            neutral = neutral + [t] if neutral is not None else None
    return {"actual": turns, "removed": removed, "neutral": neutral}


NEUTRAL = [
    "Checking the edge cases with small inputs first.", "The input size suggests an O(n log n) approach is enough.",
    "Reading the constraints again before writing the loop.", "A prefix sum should make each query constant time.",
    "Sorting first keeps the comparison simple.", "The sample output matches when the indices start at one.",
    "Using a dictionary to count occurrences.", "The recursion depth could be large, so an iterative version is safer.",
    "Integer overflow is not an issue in Python here.", "Two pointers should work because the array is sorted.",
    "The modulus has to be applied after every multiplication.", "A breadth-first search gives the shortest distances.",
    "Memoizing the helper avoids repeated work.", "Reading all input at once is faster than line by line.",
    "The greedy choice looks safe because costs are non-negative.", "Binary search on the answer should converge quickly.",
    "Empty input should print zero.", "Tracking the running maximum in one pass.", "Splitting the logic into two helpers.",
    "The graph may have several components, so every node needs a visit.", "Handling the boundary at the last index.",
    "A heap keeps the smallest element available.", "The state needs both the position and the remaining budget.",
    "Precomputing factorials makes the combinations quick.", "Union-find merges the groups efficiently.",
    "The answer fits in a single pass over the string.", "Dynamic programming over prefixes looks right.",
    "Checking that duplicates are counted only once.", "The loop bounds need to include the final element.",
    "Converting to zero-based indices at the start.", "Bit masks cover all subsets when the size is small.",
    "Output each result on its own line.", "The worst case is when all values are equal.",
    "A stack handles the nested brackets.", "Rounding is not needed because the result is an integer.",
    "Testing the formula on the second sample by hand.", "The window shrinks from the left when the sum is too large.",
    "Storing the best value seen so far.", "Negative numbers change which element is the minimum.",
    "The time limit allows roughly a million operations per test.",
]


_STEPS = ["Checking", "Re-reading", "Testing", "Simplifying", "Tracing", "Writing down", "Verifying", "Planning",
          "Reconsidering", "Sketching"]
_THINGS = ["the loop bounds", "the base case", "the update rule", "the input parsing", "the output format",
           "the edge case with one element", "the sorted order", "the running total", "the index arithmetic",
           "the memo table", "the queue handling", "the recursion limit", "the boundary values", "the counting step",
           "the merge step", "the final sum"]
_WHY = ["before the next run.", "so the logic stays clear.", "with a small example.", "against the constraints.",
        "to keep it linear.", "one step at a time."]
NEUTRAL += [f"{a} {b} {c}" for a in _STEPS for b in _THINGS for c in _WHY]   # 960 more, all distinct


def neutral_pool(h: History, k: int, ids: list[int], rng: random.Random) -> list[str]:
    """NEUTRAL, shuffled for this request (each sentence used at most once in it)."""
    pool = NEUTRAL[:]
    rng.shuffle(pool)
    return pool


def chars(ts: list[Turn], marks: bool) -> int:
    return sum(len(t.text) + (len(mark(t)) + 1 if marks else 0) for t in ts)


def reply_words(r: dict) -> str:
    return (r.get("text") or "")


MARKED = ("Each message in the context starts with [agent k]: the agent who wrote it, or whose tool call it "
          "answers.")
NAMED = ("The other agents' messages, tool calls and results come to you as messages that start with [agent k]; yours "
         "are your own replies and tool calls.")


def with_sentence(system: dict, sentence: str) -> dict:
    """The instructions with a setting's sentence after the 'You are agent k' line, where the world puts it."""
    lines = system["content"].split("\n")
    at = next(i for i, l in enumerate(lines) if l.startswith("You are agent"))
    return dict(system, content="\n".join(lines[:at + 1] + [sentence] + lines[at + 1:]))


def partner_arms(turns: list[Turn], k: int, actual: list, size: int | None, cpt: float | None, limit: int,
                 exit_request: bool) -> dict:
    """actual / marked / named / partner_calls_removed for one request of agent k (see the docstring)."""
    others = lambda t: t.agent not in (None, k)
    base = chars(turns, False)
    variants = {
        "marked": (with_sentence(actual[0], MARKED), body_of(turns, True, None),
                   sum(len(t.text) + len(mark(t)) + 1 for t in turns)),
        "named": (with_sentence(actual[0], NAMED), body_of(turns, False, k),
                  sum(len(t.text) + (len(mark(t)) + 1 if others(t) else 0) for t in turns)),
    }
    kept = [t for t in turns if not (others(t) and t.kind in ("call", "result"))]
    variants["partner_calls_removed"] = (actual[0], body_of(kept, False, None), chars(kept, False))
    arms = {"actual": {"messages": actual, "size": size, "over_limit": None}}
    for arm, (system, body, n) in variants.items():
        if exit_request:
            arms[arm] = {"messages": [system] + body + [actual[-1]]}
            continue
        new = size + round((n - base) / cpt)
        status, over = restatus(actual[-1]["content"], size, new, limit, "run_tests and submit")
        arms[arm] = {"messages": [system] + body + [{"role": "user", "content": status}], "size": new, "over_limit": over}
    if not exit_request:
        arms["actual"]["over_limit"] = size > limit
    return arms


def build(world: Path, model: str, per_world: int, min_present: int, seed: int, partner: bool = False) -> dict:
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    if doc.get("setup") not in ("external", "pool"):
        return {"steps": [], "exits": [], "skipped": []}
    marks = doc.get("marks") == "authors"
    view_user = doc.get("chat_others") == "user"
    limit = int(doc["limit"])
    if partner and (marks or view_user or len(doc["agents"]) < 2):     # partner arms: plain-chat worlds with a partner
        return {"steps": [], "exits": [], "skipped": []}
    out = {"steps": [], "exits": [], "skipped": []}
    rng = random.Random(seed)
    exits = {e["agent"]: e for e in load(world / "exit.jsonl")}
    for spec in doc["agents"]:
        k = spec["agent"]
        if model not in spec["model"]:
            continue
        h = History(world, doc, k)
        view = k if view_user else None
        calls = [c for c in load(world / "calls.jsonl") if c["agent"] == k]
        chosen = []
        for c in calls:
            ids = h.ids(c["version"])
            n = sum(1 for i in ids if is_confession(h.turn(i), k))
            if n >= min_present:
                chosen.append((c, ids, n))
        if len(chosen) > per_world:
            step = len(chosen) / per_world
            chosen = [chosen[int(i * step)] for i in range(per_world)]
        for c, ids, n in chosen:
            req = request_name(c)
            path = world / "requests" / req
            if not path.exists():
                out["skipped"].append({"agent": k, "request": req, "why": "no saved request"})
                continue
            actual = json.loads(path.read_text(encoding="utf-8"))
            turns = h.turns(ids)
            if [actual[0]] + body_of(turns, marks, view) + [actual[-1]] != actual:
                out["skipped"].append({"agent": k, "request": req, "why": "the rebuild differs from the saved request"})
                continue
            own = [t for t in turns if t.agent == k and t.role == "assistant"]
            if own and is_confession(own[-1], k) and "Call one tool" in actual[-1]["content"]:
                out["skipped"].append({"agent": k, "request": req, "why": "its last own turn is a confession the status refers to"})
                continue
            arms = {}
            size = int(c["context_tokens"])
            if partner:
                arms = partner_arms(turns, k, actual, size, float(c["chars_per_token"]), limit, False)
            for arm, ts in ([] if partner else versions(turns, k, neutral_pool(h, k, ids, rng)).items()):
                if ts is None:
                    continue
                new = size + round((chars(ts, marks) - chars(turns, marks)) / float(c["chars_per_token"]))
                status, over = restatus(actual[-1]["content"], size, new, limit, "run_tests and submit")
                arms[arm] = {"messages": actual if arm == "actual" else [actual[0]] + body_of(ts, marks, view) +
                             [{"role": "user", "content": status}], "size": new, "over_limit": over}
            if partner:                     # drop only an arm that crosses the limit differently from the actual one
                arms = {arm: v for arm, v in arms.items() if v["over_limit"] == arms["actual"]["over_limit"]}
            if len({spec["over_limit"] for spec in arms.values()}) > 1:
                out["skipped"].append({"agent": k, "request": req, "why": "an arm crosses the limit differently"})
                continue
            out["steps"].append({"world": doc["label"], "agent": k, "model": spec["model"], "request": req,
                                 "round": c["round"], "present": n, "arms": arms})
        e = exits.get(k)
        path = world / "requests" / f"exit_a{k}.json"
        if e and path.exists() and e.get("turn_ids"):
            actual = json.loads(path.read_text(encoding="utf-8"))
            turns = h.turns(e["turn_ids"])
            n = sum(1 for t in turns if is_confession(t, k))
            if n and [actual[0]] + body_of(turns, marks, view) + [actual[-1]] == actual and partner:
                out["exits"].append({"world": doc["label"], "agent": k, "model": spec["model"], "request": path.name,
                                     "present": n, "arms": partner_arms(turns, k, actual, None, None, limit, True)})
            elif n and [actual[0]] + body_of(turns, marks, view) + [actual[-1]] == actual:
                arms = {arm: {"messages": actual if arm == "actual" else [actual[0]] + body_of(ts, marks, view) + [actual[-1]]}
                        for arm, ts in versions(turns, k, neutral_pool(h, k, e["turn_ids"], rng)).items() if ts is not None}
                out["exits"].append({"world": doc["label"], "agent": k, "model": spec["model"], "request": path.name,
                                     "present": n, "arms": arms})
            elif n:
                out["skipped"].append({"agent": k, "request": path.name, "why": "the rebuild differs from the saved request"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--model", default="haiku", help="agents whose model id contains this")
    ap.add_argument("--per-world", type=int, default=6)
    ap.add_argument("--min-present", type=int, default=2)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--exit-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partner", action="store_true", help="partner arms (marked / named / partner calls removed)")
    a = ap.parse_args(argv)
    for world in worlds(Path(a.root)):
        doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
        built = build(world, a.model, a.per_world, a.min_present, a.seed, a.partner)
        if not built["steps"] and not built["exits"]:
            continue
        print(f"{world}: {len(built['steps'])} step requests, {len(built['exits'])} exit requests, "
              f"{len(built['skipped'])} skipped" + "".join(f"\n  skipped a{s['agent']} {s['request']}: {s['why']}"
                                                          for s in built["skipped"]))
        if a.dry_run:
            continue
        (world / "analysis").mkdir(exist_ok=True)
        providers = {}
        with (world / "analysis" / ("partner_forks.jsonl" if a.partner else "confession_forks.jsonl")).open(
                "a", encoding="utf-8") as fo:
            for kind, items, n in (("step", built["steps"], a.samples), ("exit", built["exits"], a.exit_samples)):
                for it in items:
                    k = it["agent"]
                    providers.setdefault(k, provider_for(doc, k))
                    pcfg, prov = providers[k]
                    tools = json.loads((world / "requests" / f"tools_a{k}.json").read_text(encoding="utf-8"))
                    meta = {key: v for key, v in it.items() if key != "arms"}
                    for arm, spec in it["arms"].items():
                        for s in range(n):
                            r = sample(prov, pcfg, spec["messages"], tools, seed=a.seed * 1009 + s,
                                       tool_choice="none" if kind == "exit" else "auto")
                            w = reply_words(r)
                            row = dict(meta, kind=kind, arm=arm, sample=s, size=spec.get("size"),
                                       over_limit=spec.get("over_limit"), confession=confession(w), chars=len(w),
                                       tool=(r["tool_calls"][0]["name"] if r.get("tool_calls") else None), **r)
                            if kind == "exit":
                                row.update(self_blame=bool(SELF.search(words(w))), claims_wrong=bool(CLAIM_WRONG.search(words(w))))
                            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fo.flush()


if __name__ == "__main__":
    main()
