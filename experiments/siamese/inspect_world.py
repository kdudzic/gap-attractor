"""Read a world as a story: what happened in the shared file, round by round, in plain text.

    uv run python experiments/siamese/inspect_world.py <world folder or batch folder> [--excerpt 300]

Writes story.md into every world folder (and stories.md next to a batch's cells, all worlds one after another):
- the file over time: tokens held by each agent every few rounds (from rounds.jsonl), as a small bar per agent;
- problems: when each agent started and finished each problem, how (submit / work limit), accepted or not;
- edits: who edited, what (delete / summary / text replacement), whose turns went, sizes before/after; summaries that
  replaced another agent's turns are quoted (taken from the version diff); refused edits with the reason;
- talk about the partner: the agent's own turns that name another agent (first lines quoted);
- code moving between agents: a submission that reuses another agent's code (help = written for MY problem; mix-up =
  written for another problem), and programs written for another agent's problem.
Nothing here is a statistic; it is for building intuition. analyze.py has the numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_first = [str(HERE), str(HERE.parents[1] / "experiments" / "context_file"), str(HERE.parents[1] / "src")]
sys.path[:] = _first + [p for p in sys.path if p not in _first]

from analyze import grams, jsonl, problems  # noqa: E402


def short(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


def bar(tokens: int, scale: int, width: int = 30) -> str:
    return "█" * max(0, round(width * tokens / scale)) if scale else ""


def summary_texts(diff: str) -> str:
    """The lines a version added (its new turns' text), without the diff headers."""
    added = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++") and not l[1:].startswith("[[TURN")]
    return "\n".join(added)


DISCOVERY = (r"\b(another|other|second|different|separate) (agent|user|model|assistant|person|programmer|solver|participant|"
             r"session|instance|process|conversation)s?\b|someone else|somebody else|not (my|mine)\b|isn'?t mine|"
             r"(didn'?t|did not|never) (write|wrote|ask)|wasn'?t (me|mine)|not my problem|two (different )?(problems|tasks|series)|"
             r"shared (context|file|workspace)|multiple (agents|users)")


def snapshots(wdir: Path) -> list[Path]:
    """The file after each round; for an unlabelled shared file, the labelled copies kept for the logs."""
    labelled = sorted((wdir / "snapshots").glob("*_labelled_r*.txt"))
    return labelled or sorted((wdir / "snapshots").glob("*.txt"))


def story(wdir: Path, excerpt: int = 300) -> str:
    doc = json.loads((wdir / "world.json").read_text())
    agents = {a["agent"]: a["model"].split("/")[-1] for a in doc["agents"]}
    shared = doc["sharing"] in ("full", "readonly_others")
    probs = problems(str((doc.get("config") or {}).get("problems", {}).get("file", "")))
    results, edits, rounds, actions, texts = (jsonl(wdir / f) for f in (
        "results.jsonl", "edits.jsonl", "rounds.jsonl", "actions.jsonl", "texts.jsonl"))
    calls = jsonl(wdir / "calls.jsonl")
    versions = jsonl(wdir / "versions.jsonl") if shared else []
    lines = [f"# {doc.get('label', wdir.name)}", "",
             f"sharing {doc['sharing']}, relation {doc['relation']}, limit {doc['limit']} tokens, {doc['rounds']} rounds, "
             f"status {doc['status']}", ""]
    for a in doc["agents"]:
        lines.append(f"- agent {a['agent']} = {agents[a['agent']]}: solved {a['solved']}/{a['submitted']}, calls {a['calls']}, "
                     f"edits {a['edits_applied']}/{a['edits']}, held {a['held']}, evictions {a['evictions']}, "
                     f"cut-off replies {a.get('truncated', 0)}, cost ${a['cost']:.3f}")
    lines.append("")

    # the file over time
    lines += ["## The file over time (tokens held per agent)", ""]
    limit = doc["limit"] or max((r["tokens"] for r in rounds), default=1)
    step = max(1, len(rounds) // 12)
    for r in rounds[::step] + ([rounds[-1]] if rounds and (len(rounds) - 1) % step else []):
        parts = "  ".join(f"a{k} {r['tokens_by_agent'].get(str(k), 0) / 1000:5.1f}k {bar(r['tokens_by_agent'].get(str(k), 0), limit, 20):<20}"
                          for k in agents)
        lines.append(f"    r{r['round']:03d} {r['file'][:16]:<16} total {r['tokens'] / 1000:5.1f}k | {parts}")
    lines.append("")

    # events, in round order
    events = []
    started = {}
    for c in calls:
        key = (c["agent"], c.get("problem"))
        if c.get("problem") and key not in started:
            started[key] = c["round"]
            events.append((c["round"], 0, f"a{c['agent']} starts problem {c['problem']}"))
    for r in results:
        events.append((r["round"], 1, f"a{r['agent']} {'ACCEPTED' if r['passed'] else 'failed'} {r['problem']} "
                                      f"({r['how']}, {r['work_calls']} work calls, {r['steps']} steps)"))
    by_round_versions = {}
    for v in versions:
        by_round_versions.setdefault((v["step"], v["agent"]), []).append(v)
    for e in edits:
        k = e.get("agent")
        if e.get("eviction"):
            events.append((e["round"], 2, f"EVICTION for a{k}: {e['removed_turns']} turns removed {e.get('removed_by_agent')}"))
            continue
        if not e.get("applied"):
            events.append((e["round"], 2, f"a{k} edit refused ({e.get('rejected')})"))
            continue
        ops = ", ".join(sorted((e.get("arguments") or {}).keys()))
        whose = {kk: v for kk, v in (e.get("removed_by_agent") or {}).items()}
        text = (f"a{k} edits ({ops}): removed {e.get('removed_turns', 0)} turns {whose}, "
                f"{e.get('tokens_before')}->{e.get('tokens_after')} tokens")
        if e.get("removed_others") and "replace_turns" in (e.get("arguments") or {}):
            for item in (e["arguments"].get("replace_turns") or []):
                text += f"\n        summary written: \"{short(item.get('summary', ''), excerpt)}\""
        events.append((e["round"], 2, text))
    # talk about the partner (own turns naming another agent), first time each text appears
    seen = set()
    for snap in snapshots(wdir) if shared else []:
        rnd = int(re.search(r"_r(\d+)", snap.name).group(1))
        for block in re.split(r"(?=\[\[TURN \d+ role=)", snap.read_text()):
            m = re.match(r"\[\[TURN \d+ role=assistant agent=(\d+)\]\]\n", block)
            if not m:
                continue
            k, body = int(m.group(1)), block[m.end():]
            others = [o for o in agents if o != k]
            if not any(re.search(rf"\bagent {o}\b", body, re.I) for o in others):
                continue
            key = body[:200]
            if key in seen:
                continue
            seen.add(key)
            events.append((rnd, 3, f"a{k} mentions its partner: \"{short(body, excerpt)}\""))
    # noticing someone else (meant for unaware worlds, works for all): own turns whose words say the context holds
    # someone else's work; the first one per agent is marked FIRST
    found, first = set(), set()
    pattern = re.compile(DISCOVERY, re.I)
    for snap in snapshots(wdir) if shared else []:
        rnd = int(re.search(r"_r(\d+)", snap.name).group(1))
        for block in re.split(r"(?=\[\[TURN \d+ role=)", snap.read_text()):
            m = re.match(r"\[\[TURN \d+ role=assistant(?: agent=(\d+))?\]\]\n", block)
            if not m:
                continue
            body = block[m.end():]
            hit = pattern.search(body)
            if not hit or body[:200] in found:
                continue
            found.add(body[:200])
            who = m.group(1) or "?"
            tag = "FIRST " if who not in first else ""
            first.add(who)
            events.append((rnd, 3, f"{tag}a{who} notices someone else (\"{hit.group(0)}\"): \"{short(body, excerpt)}\""))
    # code moving between agents
    fn = lambda pid: (probs.get(pid) or {}).get("fn_name")
    for i, a in enumerate(actions if shared else []):
        others_problems = {b["problem"] for b in actions if b["agent"] != a["agent"]}
        names = {fn(p) for p in others_problems} - {None, fn(a["problem"])}
        hit = [n for n in names if re.search(rf"def {re.escape(n)}\(", a["code"] or "")]
        if hit:
            events.append((a["round"], 4, f"a{a['agent']} writes a program for another agent's problem ({hit[0]}) "
                                          f"while on {a['problem']}"))
        g = grams(a["code"])
        prior = [b for b in actions[:i] if b["agent"] != a["agent"]]
        if a["tool"] != "submit" or len(g) < 20 or not prior:
            continue
        src = max(prior, key=lambda b: len(g & grams(b["code"])))
        overlap = len(g & grams(src["code"])) / len(g)
        if overlap > 0.5:
            mine = fn(a["problem"])
            helped = src["problem"] == a["problem"] or (mine and re.search(rf"def {re.escape(mine)}\(", src["code"] or ""))
            events.append((a["round"], 4, f"a{a['agent']} submits for {a['problem']} code {overlap:.0%} from a{src['agent']}'s "
                                          f"program (written while a{src['agent']} was on {src['problem']}): "
                                          f"{'HELP' if helped else 'MIX-UP'}"))
    # alteration events (events.py): an agent's next call after another agent altered turns it knew
    altered: dict = {}
    for e in jsonl(wdir / "analysis" / "events.jsonl"):
        if e["known"] and e["source"] == "partner" and e["owner"] == "own":
            altered.setdefault((e["round"], e["agent"], e["editor"]), []).append(e)
    for (rnd, k, editor), es in altered.items():
        kinds = ", ".join(f"{sum(1 for e in es if e['kind'] == kd)} {kd}" for kd in ("deleted", "rewritten")
                          if any(e["kind"] == kd for e in es))
        reply = es[0].get("next_reply") or ""
        events.append((rnd, 5, f"ALTERED a{k}'s own turns by a{editor} ({kinds}); a{k} next says: \"{short(reply, excerpt)}\""))
    lines += ["## Events", ""]
    for rnd, _, text in sorted(events, key=lambda x: (x[0], x[1])):
        lines.append(f"- r{rnd:03d} {text}")
    exits = jsonl(wdir / "exit.jsonl")
    if exits:
        lines += ["", "## After the end (exit question)", "", f"> {exits[0]['question']}", ""]
        for x in exits:
            lines.append(f"- a{x['agent']} ({agents[x['agent']]}): \"{short(x['text'], 4 * excerpt)}\"")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--excerpt", type=int, default=300)
    a = ap.parse_args(argv)
    path = Path(a.path)
    worlds = [path] if (path / "world.json").exists() else sorted(p.parent for p in path.rglob("world_s*/world.json") if "_cut" not in p.parent.name)
    out = []
    for w in worlds:
        text = story(w, a.excerpt)
        (w / "story.md").write_text(text)
        out.append(text)
    if len(worlds) > 1:
        (path / "stories.md").write_text("\n\n---\n\n".join(out))
        print(f"{len(worlds)} worlds -> {path / 'stories.md'}")
    else:
        print(out[0])


if __name__ == "__main__":
    main()
