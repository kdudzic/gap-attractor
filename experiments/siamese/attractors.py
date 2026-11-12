"""Self-reinforcement (attractor) measures on existing worlds: repetition, loops, confession self-reinforcement, style
contagion. Logs only; no model calls.

    uv run python experiments/siamese/attractors.py <world folder or batch folder> [--window 5]

Writes analysis/attractors.json into every world and prints one line per agent. Background: IDEAS_attractors.md.

Units: an agent's **own turns** are the assistant turns it added (replies, tool calls with their arguments as recorded
in the file, i.e. its words plus the program or command), in the order they were added (version log), with the round
they were added in. Leading "[agent k]" marks are stripped before any comparison.

1. **Repetition** (per agent, per window of `--window` rounds):
   - `ratio`: lzma-compressed size / raw size of the window's own turns (lower = more repetitive inside the window);
   - `novelty`: (compressed size of all own turns so far including this window - compressed size before it) / raw size
     of the window, i.e. how much new information the window adds beyond the agent's earlier output (lower = it
     repeats itself across time). `novelty_slope`: least-squares slope of novelty over windows (negative = drifting
     toward self-repetition).
   - `context_ratio`: lzma ratio of the whole context after each round (from snapshots/), first vs last window.
2. **Loops** (per agent): consecutive own turns that are near-identical (difflib ratio >= 0.9 on the first 2,000
   characters after whitespace normalisation; identical counts as 1.0). A loop is a run of >= 3 such turns.
   Reported: number of loops, the longest run, and the top runs with their text.
3. **Confession self-reinforcement** (per agent call): a *confession* is an own turn whose words (code blocks removed)
   say the agent worked on or submitted for the wrong / another problem, or that something is not its own (regex
   CONFESSION below). For every call of the agent (calls.jsonl), the request's turns (`turn_ids`) give the number of
   confessions present (the agent's own, and other agents'); the reply is the agent's first own turn added after the
   request (none if the call produced no turn). Output: rows (round, fill = context_tokens / limit, problem, own
   confessions written so far, own present, others present, reply is a confession). Read as P(confession | k present),
   stratified by fill, and, as a natural experiment, with the number written so far held fixed: agents delete some of
   their own confessions, so at equal `written`, a different `own` present isolates the effect of presence from that
   of being stuck.
4. **Style contagion** (per agent, per window of 10 rounds):
   - `untested_rate`: of its submissions (results how = submit), the share with no earlier run_tests on that problem
     (work_calls == 1);
   - `overlap_partner` / `overlap_self`: for each program it ran or submitted (actions.jsonl, >= 20 six-token grams),
     the largest share of its grams found in one earlier program by another agent (any problem) vs in one of its own
     earlier programs for a *different* problem; mean per window.
"""
from __future__ import annotations

import argparse
import difflib
import json
import lzma
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from analyze import grams  # noqa: E402
from events import load, versions_for, worlds  # noqa: E402

CONFESSION = re.compile(
    r"wrong problem|wrong task|(for|on|to|into) (a |an |the )?(different|other|another|wrong) (problem|task)|"
    r"not (my|mine)\b(?! own)|\bisn'?t mine|\bwasn'?t mine|not my (problem|code|submission|program|work)|"
    r"belongs? to (another|a different|other|agent)|mislabel|\bmix(ed)?[- ]?up\b|instead of (mine|my (own )?problem)|"
    r"(code|program|submission)s? for (a |an )?(different|other|another) problem", re.I)
MARK = re.compile(r"^\s*\[agent \d+\]\s*")
CODE = re.compile(r"```.*?(```|$)", re.S)


def lz(b: bytes) -> int:
    return len(lzma.compress(b, preset=6))


def words(text: str) -> str:
    return CODE.sub(" ", MARK.sub("", text or ""))


def confession(text: str) -> bool:
    return bool(CONFESSION.search(words(text)))


def norm(text: str) -> str:
    return " ".join(MARK.sub("", text or "").split())[:2000]


def similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sm.ratio() if sm.real_quick_ratio() >= 0.9 and sm.quick_ratio() >= 0.9 else 0.0


def slope(ys: list[float]) -> float | None:
    n = len(ys)
    if n < 3:
        return None
    mx, my = (n - 1) / 2, sum(ys) / n
    den = sum((i - mx) ** 2 for i in range(n))
    return round(sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den, 5)


def own_turns(versions: list[dict], k: int) -> list[dict]:
    """The agent's assistant turns as first written, in order: {id, round, text}."""
    seen, out = set(), []
    for v in versions:
        for t in v.get("turns_set") or []:
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            if t["agent"] == k and t["role"] == "assistant" and t.get("kind") != "reasoning":   # not replies
                out.append({"id": t["id"], "round": v["step"], "text": t["text"]})
    return out


def all_texts(versions: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for v in versions:
        for t in v.get("turns_set") or []:
            out.setdefault(t["id"], t)
    return out


def repetition(turns: list[dict], window: int) -> dict:
    if not turns:
        return {}
    last = max(t["round"] for t in turns)
    ratios, novelty, prev = [], [], b""
    for w0 in range(1, last + 1, window):
        chunk = "\n".join(t["text"] for t in turns if w0 <= t["round"] < w0 + window).encode()
        if len(chunk) < 200:
            continue
        ratios.append(round(lz(chunk) / len(chunk), 4))
        novelty.append(round((lz(prev + chunk) - lz(prev)) / len(chunk) if prev else lz(chunk) / len(chunk), 4))
        prev += chunk
    return {"ratio": ratios, "novelty": novelty, "novelty_slope": slope(novelty[1:]) if len(novelty) > 3 else None}


def loops(turns: list[dict], top: int = 3) -> dict:
    runs, cur = [], [turns[0]] if turns else []
    for a, b in zip(turns, turns[1:]):
        if similar(norm(a["text"]), norm(b["text"])) >= 0.9:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    if cur:
        runs.append(cur)
    long_runs = sorted((r for r in runs if len(r) >= 3), key=len, reverse=True)
    return {"loops": len(long_runs), "longest": max((len(r) for r in runs), default=0),
            "turns_in_loops": sum(len(r) for r in long_runs),
            "top": [{"length": len(r), "rounds": [r[0]["round"], r[-1]["round"]], "text": norm(r[0]["text"])[:240]}
                    for r in long_runs[:top]]}


def confession_rows(calls: list[dict], k: int, texts: dict, mine: list[dict], limit: int) -> list[dict]:
    own_ids = [t["id"] for t in mine]
    rows = []
    mycalls = [c for c in calls if c["agent"] == k]
    for i, c in enumerate(mycalls):
        ids = c.get("turn_ids") or []
        top = max(ids, default=0)
        nxt_top = max(mycalls[i + 1].get("turn_ids") or [0], default=0) if i + 1 < len(mycalls) else None
        reply = next((t for t in own_ids if t > top), None)
        if reply is None or (nxt_top is not None and reply > nxt_top):
            continue                                  # this call added no turn (refused, empty, malformed)
        present = [texts[j] for j in ids if j in texts and texts[j]["role"] == "assistant"
                   and texts[j].get("kind") != "reasoning" and confession(texts[j]["text"])]
        rows.append({"round": c["round"], "fill": round(int(c["context_tokens"]) / limit, 3) if limit else None,
                     "problem": c.get("problem"),
                     "written": sum(1 for t in mine if t["id"] <= top and confession(t["text"])),
                     "own": sum(1 for t in present if t["agent"] == k),
                     "others": sum(1 for t in present if t["agent"] not in (None, k)),
                     "reply": confession(texts[reply]["text"]) if reply in texts else False})
    return rows


def contagion(k: int, actions: list[dict], results: list[dict], window: int = 10) -> dict:
    out = []
    last = max([a["round"] for a in actions] + [r["round"] for r in results] + [1])
    progs = [(a, grams(a.get("code") or "")) for a in actions]
    for w0 in range(1, last + 1, window):
        subs = [r for r in results if r["agent"] == k and r["how"] == "submit" and w0 <= r["round"] < w0 + window]
        ovp, ovs = [], []
        for i, (a, g) in enumerate(progs):
            if a["agent"] != k or not (w0 <= a["round"] < w0 + window) or len(g) < 20:
                continue
            earlier = progs[:i]
            p = max((len(g & h) / len(g) for b, h in earlier if b["agent"] != k), default=0.0)
            s = max((len(g & h) / len(g) for b, h in earlier if b["agent"] == k and b["problem"] != a["problem"]),
                    default=0.0)
            ovp.append(p)
            ovs.append(s)
        out.append({"rounds": [w0, w0 + window - 1], "submits": len(subs),
                    "untested_rate": round(sum(1 for r in subs if r["work_calls"] == 1) / len(subs), 3) if subs else None,
                    "overlap_partner": round(sum(ovp) / len(ovp), 3) if ovp else None,
                    "overlap_self": round(sum(ovs) / len(ovs), 3) if ovs else None})
    return {"windows": out}


def snapshots(wdir: Path) -> list[Path]:
    labelled = sorted((wdir / "snapshots").glob("*_labelled_r*.txt"))
    return labelled or sorted((wdir / "snapshots").glob("*.txt"))


def world_measures(wdir: Path, window: int = 5) -> dict:
    doc = json.loads((wdir / "world.json").read_text())
    calls, actions, results = load(wdir / "calls.jsonl"), load(wdir / "actions.jsonl"), load(wdir / "results.jsonl")
    out = {"world": doc["label"], "dir": str(wdir), "marks": doc.get("marks"), "cross": doc.get("cross_test"),
           "edit": doc.get("edit"), "models": [a["model"].split("/")[-1] for a in doc["agents"]], "agents": {}}
    ctx = []
    for snap in snapshots(wdir):
        b = snap.read_bytes()
        if len(b) > 200:
            ctx.append(round(lz(b) / len(b), 4))
    out["context_ratio"] = {"first": ctx[0] if ctx else None, "last": ctx[-1] if ctx else None, "slope": slope(ctx)}
    for a in doc["agents"]:
        k = a["agent"]
        versions = versions_for(wdir, doc, k)
        texts = all_texts(versions)
        mine = own_turns(versions, k)
        rows = confession_rows(calls, k, texts, mine, int(doc.get("limit") or 0))
        out["agents"][k] = {"model": a["model"].split("/")[-1], "own_turns": len(mine),
                            "confession_turns": sum(1 for t in mine if confession(t["text"])),
                            "repetition": repetition(mine, window), "loops": loops(mine),
                            "confession_rows": rows, "contagion": contagion(k, actions, results)}
    return out


def body_kind(m: dict) -> str:
    models = m["models"]
    return "solo" if len(models) == 1 else "same" if len(set(models)) == 1 else "mixed"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args(argv)
    for w in worlds(Path(a.path)):
        m = world_measures(w, a.window)
        (w / "analysis").mkdir(exist_ok=True)
        (w / "analysis" / "attractors.json").write_text(json.dumps(m, indent=1))
        for k, s in m["agents"].items():
            r, l = s["repetition"], s["loops"]
            print(f"{m['world']} ({body_kind(m)}) a{k} {s['model']}: novelty {r.get('novelty')} slope {r.get('novelty_slope')}"
                  f" | loops {l['loops']} longest {l['longest']} | confessions {s['confession_turns']}/{s['own_turns']}")


if __name__ == "__main__":
    main()
