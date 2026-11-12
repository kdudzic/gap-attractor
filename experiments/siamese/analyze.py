"""Summaries of siamese worlds: one row per cell x model, the context each model needs, and how agents interact.

    uv run python experiments/siamese/analyze.py <batch> [<batch> ...] [--out <dir>] [--curve]

Per cell (a folder holding world_s*/) and model:
- solved / submitted, pass rate with a percentile bootstrap over worlds (the cluster), work calls per problem;
- working set per problem (single-agent files only): the context a problem took, i.e. the largest request while the
  agent worked on it plus its last reply, minus the context before its statement arrived (the previous problem's end;
  for the first problem, the first request minus the statement);
- tokens per solved problem (all completion tokens / problems solved): the token-efficiency measure;
- allocation: each agent's share of the file's tokens, and the share of the turns it removed that others wrote;
- budget: held work calls, evictions (and the share of problems with one), rejected edits by reason; cost;
- reasoning chains: runs of consecutive replies with no tool call (count, tokens, cut-off replies), and who removed
  the agent's reasoning turns later (itself, another agent, or an eviction). Turn ids are per file, so each agent's
  turns are looked up in its own file (separate) or the one shared file;
- collaboration (shared files): programs written for another agent's problem, submissions that reuse another
  agent's code split into help (code written for MY problem) and mix-up (code written for another problem), with
  acceptance; summaries that replaced other agents' turns.
Paired contrasts: for every model and every two cells holding the same (seed, problem stream), the difference in pass
rate on the same problems, with a bootstrap over seeds (paired.json / paired.md).
--curve: pass rate by context limit for single-agent worlds (uncapped = None), and B* per model: the smallest limit
whose paired difference from uncapped (same problems, bootstrap over seeds) has a 95 % interval that includes 0, with
an eviction in at most 10 % of problems (curve.json / curve.md).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
_first = [str(HERE), str(HERE.parents[1] / "experiments" / "context_file"), str(HERE.parents[1] / "src")]
sys.path[:] = _first + [p for p in sys.path if p not in _first]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def boot(values_by_cluster: list[list[float]], n=2000, seed=0):
    """Percentile 95 % interval of the pooled mean, resampling clusters."""
    flat = [v for c in values_by_cluster for v in c]
    if not flat:
        return None, (None, None)
    rng, means = random.Random(seed), []
    for _ in range(n):
        pick = [v for _ in values_by_cluster for v in rng.choice(values_by_cluster)]
        if pick:
            means.append(sum(pick) / len(pick))
    means.sort()
    return sum(flat) / len(flat), (means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1])


@lru_cache(maxsize=4)
def problems(path: str) -> dict:
    """{id: problem} of the release file a world used (statement size, LeetCode function name)."""
    if not path or not Path(path).exists():
        return {}
    from siam.problems import load
    return {p["id"]: p for p in load(path)}


def statement_tokens(problem: dict | None, chars_per_token: float = 4.0) -> int:
    if not problem:
        return 0
    from lcb.prompts import question_prompt
    return int(len(question_prompt(problem)) / chars_per_token)


def working_sets(calls: list[dict], agent: int, probs: dict, count: str = "whole") -> dict:
    """{problem: tokens the problem took} for one agent of a single-agent file (see the module docstring). count:
    the world's ctx.count; with turns the first request does not hold the statement, so nothing is taken off it."""
    order, by = [], defaultdict(list)
    for c in calls:
        if c["agent"] == agent and c.get("problem") and not c.get("error"):
            if c["problem"] not in by:
                order.append(c["problem"])
            by[c["problem"]].append((c["context_tokens"], int((c.get("usage") or {}).get("completion_tokens") or 0)))
    out, prev_end = {}, None
    for p in order:
        v = by[p]
        end = max(x for x, _ in v) + v[-1][1]
        base = (prev_end if prev_end is not None else v[0][0] if count == "turns" else
                v[0][0] - statement_tokens(probs.get(p)))
        out[p] = end - base
        prev_end = end
    return out


def grams(code: str, n: int = 6) -> set:
    t = re.findall(r"\w+|[^\s\w]", code or "")
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def collaboration(actions: list[dict], results: list[dict], probs: dict, agent: int) -> dict:
    """Programs this agent wrote for another agent's problem, and its submissions that reuse other agents' code."""
    accepted = {(r["agent"], r["problem"]): r["passed"] for r in results}
    fn = lambda pid: (probs.get(pid) or {}).get("fn_name")
    others_problems = {a["problem"] for a in actions if a["agent"] != agent}
    for_partner = 0
    help_rows, mixup_rows = [], []
    for i, a in enumerate(actions):
        if a["agent"] != agent:
            continue
        names = {fn(p) for p in others_problems} - {None, fn(a["problem"])}
        if any(re.search(rf"def {re.escape(n)}\(", a["code"] or "") for n in names):
            for_partner += 1
        if a["tool"] != "submit":
            continue
        g = grams(a["code"])
        if len(g) < 20:
            continue
        prior = [b for b in actions[:i] if b["agent"] != agent]
        if not prior:
            continue
        src = max(prior, key=lambda b: len(g & grams(b["code"])))
        if len(g & grams(src["code"])) / len(g) <= 0.5:
            continue
        mine = fn(a["problem"])
        written_for_mine = src["problem"] == a["problem"] or (mine and re.search(rf"def {re.escape(mine)}\(", src["code"] or ""))
        (help_rows if written_for_mine else mixup_rows).append(int(accepted.get((agent, a["problem"]), 0)))
    return {"for_partner_programs": for_partner, "reuse_help": help_rows, "reuse_mixup": mixup_rows}


def world_rows(wdir: Path) -> list[dict]:
    doc = json.loads((wdir / "world.json").read_text())
    results, calls, edits, rounds, actions, texts = (jsonl(wdir / f) for f in (
        "results.jsonl", "calls.jsonl", "edits.jsonl", "rounds.jsonl", "actions.jsonl", "texts.jsonl"))
    shared = doc["sharing"] in ("full", "readonly_others")
    probs = problems(str((doc.get("config") or {}).get("problems", {}).get("file", "")))
    out = []
    for ag in doc["agents"]:
        k = ag["agent"]
        mine = [r for r in results if r["agent"] == k]
        ws = working_sets(calls, k, probs, str(doc.get("count", "whole"))) if not shared else {}
        shares = []
        for r in rounds:
            total = sum(r["tokens_by_agent"].values())
            if total and (shared or r["file"].endswith(f"_{k}") or doc["sharing"] == "single"):
                shares.append(r["tokens_by_agent"].get(str(k), 0) / total)
        applied = [e for e in edits if e.get("agent") == k and e.get("applied")]
        removed = sum(e.get("removed_turns", 0) for e in applied)
        removed_others = sum(e.get("removed_others", 0) for e in applied)
        summaries_of_others = sum(1 for e in applied if e.get("removed_others") and "replace_turns" in (e.get("arguments") or {}))
        rejected = defaultdict(int)
        for e in edits:
            if e.get("agent") == k and e.get("rejected"):
                rejected[e["rejected"]] += 1
        # problems during which an eviction happened (the round of the eviction -> the agent's problem then)
        problem_at = {(c["agent"], c["round"]): c["problem"] for c in calls if c.get("problem")}
        evicted_problems = {problem_at.get((k, e["round"])) for e in edits if e.get("eviction") and e.get("agent") == k} - {None}
        chains, cur = [], None
        for c in calls:
            if c["agent"] != k or c.get("error"):
                continue
            if not c["tools"]:
                cur = cur or {"replies": 0, "tokens": 0, "cut": 0}
                cur["replies"] += 1
                cur["tokens"] += int((c.get("usage") or {}).get("completion_tokens") or 0)
                cur["cut"] += int(c.get("finish_reason") == "length")
            elif cur:
                chains.append(cur)
                cur = None
        if cur:
            chains.append(cur)
        own_file = wdir / ("versions.jsonl" if shared else f"versions_agent_{k}.jsonl")
        if not own_file.exists() and doc["sharing"] == "single":
            own_file = wdir / "versions_agent_1.jsonl"
        mine_text = {t["turn_id"] for t in texts if t["agent"] == k and t.get("turn_id")}
        fate = defaultdict(int)
        for v in jsonl(own_file):
            for r in v.get("removed", []):
                if r["id"] in mine_text:
                    fate["eviction" if v["op"] == "evict" else "self" if v["agent"] == k else "other"] += 1
        completion = sum(int((c.get("usage") or {}).get("completion_tokens") or 0) for c in calls if c["agent"] == k)
        collab = collaboration(actions, results, probs, k) if shared else {"for_partner_programs": 0, "reuse_help": [], "reuse_mixup": []}
        out.append({"world": wdir.name, "seed": doc["seed"], "status": doc["status"], "agent": k, "model": ag["model"],
                    "sharing": doc["sharing"], "relation": doc["relation"], "capped": doc["capped"],
                    "limit": doc["limit"] if doc["capped"] else None,
                    "by_problem": {r["problem"]: int(r["passed"]) for r in mine},
                    "passed": [int(r["passed"]) for r in mine], "work_calls": [r["work_calls"] for r in mine],
                    "working_set": list(ws.values()), "completion_tokens": completion,
                    "share": st.mean(shares) if shares else None,
                    "edits": ag["edits"], "edits_applied": ag["edits_applied"], "removed": removed,
                    "removed_others": removed_others, "summaries_of_others": summaries_of_others,
                    "held": ag["held"], "evictions": ag["evictions"], "evicted_problems": len(evicted_problems),
                    "rejected": dict(rejected), "cost": ag["cost"], "calls": ag["calls"], "no_tool": ag["no_tool"],
                    "api_errors": ag["api_errors"], "truncated": ag.get("truncated", 0),
                    "chains": len(chains), "chain_replies": [c["replies"] for c in chains],
                    "chain_tokens": [c["tokens"] for c in chains], "text_turns": len(mine_text), "text_removed": dict(fate),
                    **collab})
    return out


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def summarise(rows: list[dict]) -> dict:
    mean = lambda xs: round(st.mean(xs), 3) if xs else None
    med = lambda xs: round(st.median(xs)) if xs else None
    rate, (lo, hi) = boot([r["passed"] for r in rows])
    allv = lambda key: [v for r in rows for v in r[key]]
    rejected = defaultdict(int)
    for r in rows:
        for k, v in r["rejected"].items():
            rejected[k] += v
    removed = sum(r["removed"] for r in rows)
    solved = sum(allv("passed"))
    help_, mix = allv("reuse_help"), allv("reuse_mixup")
    return {"worlds": len({r["world"] for r in rows}), "agents": len(rows), "problems": len(allv("passed")), "solved": solved,
            "pass_rate": round(rate, 3) if rate is not None else None, "pass_ci": [round(lo, 3), round(hi, 3)] if lo is not None else None,
            "work_calls_mean": mean(allv("work_calls")),
            "working_set_median": med(allv("working_set")), "working_set_p75": q(allv("working_set"), .75),
            "working_set_p90": q(allv("working_set"), .9),
            "tokens_per_solved": round(sum(r["completion_tokens"] for r in rows) / solved) if solved else None,
            "share_mean": mean([r["share"] for r in rows if r["share"] is not None]),
            "edits": sum(r["edits"] for r in rows), "edits_applied": sum(r["edits_applied"] for r in rows),
            "removed_turns": removed, "removed_others_frac": round(sum(r["removed_others"] for r in rows) / removed, 3) if removed else None,
            "summaries_of_others": sum(r["summaries_of_others"] for r in rows),
            "held": sum(r["held"] for r in rows), "evictions": sum(r["evictions"] for r in rows),
            "evicted_problem_frac": round(sum(r["evicted_problems"] for r in rows) / max(1, len(allv("passed"))), 3),
            "rejected": dict(rejected), "no_tool_replies": sum(r["no_tool"] for r in rows), "api_errors": sum(r["api_errors"] for r in rows),
            "cost": round(sum(r["cost"] for r in rows), 4), "cost_per_problem": round(sum(r["cost"] for r in rows) / max(1, len(allv("passed"))), 4),
            "statuses": sorted({r["status"] for r in rows}),
            "truncated": sum(r["truncated"] for r in rows), "chains": sum(r["chains"] for r in rows),
            "chain_tokens_median": med(allv("chain_tokens")), "chain_replies_max": max(allv("chain_replies")) if allv("chain_replies") else None,
            "text_turns": sum(r["text_turns"] for r in rows),
            "text_removed": {w: sum(r["text_removed"].get(w, 0) for r in rows) for w in ("self", "other", "eviction")},
            "for_partner_programs": sum(r["for_partner_programs"] for r in rows),
            "reuse_help": f"{sum(help_)}/{len(help_)}", "reuse_mixup": f"{sum(mix)}/{len(mix)}"}


def paired(cells: dict) -> dict:
    """For each model, every two cells: mean pass difference on the same (seed, agent slot, problem), bootstrap by seed."""
    by = defaultdict(lambda: defaultdict(dict))        # model -> cell -> {(seed, slot, problem): passed}
    for cell, rows in cells.items():
        for r in rows:
            for p, v in r["by_problem"].items():
                by[r["model"]][cell][(r["seed"], r["agent"], p)] = v
    out = {}
    for model, cs in by.items():
        names = sorted(cs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = sorted(set(cs[a]) & set(cs[b]))
                if not common:
                    continue
                per_seed = defaultdict(list)
                for key in common:
                    per_seed[key[0]].append(cs[a][key] - cs[b][key])
                d, (lo, hi) = boot(list(per_seed.values()))
                out[f"{model} | {a} - {b}"] = {"problems": len(common), "diff": round(d, 3),
                                               "ci": [round(lo, 3), round(hi, 3)], "a_solved": sum(cs[a][k] for k in common),
                                               "b_solved": sum(cs[b][k] for k in common)}
    return out


def curve(cells: dict) -> dict:
    """Pass rate by context limit for single-agent rows (single or separate files), and B* per model."""
    by = defaultdict(lambda: defaultdict(dict))         # model -> limit -> {(seed, slot, problem): passed}
    ev = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for rows in cells.values():
        for r in rows:
            if r["sharing"] in ("full", "readonly_others"):
                continue
            for p, v in r["by_problem"].items():
                by[r["model"]][r["limit"]][(r["seed"], r["agent"], p)] = v
            ev[r["model"]][r["limit"]][0] += r["evicted_problems"]
            ev[r["model"]][r["limit"]][1] += len(r["by_problem"])
    out = {}
    for model, limits in by.items():
        ref = limits.get(None)
        rows, best = [], None
        for lim in sorted(limits, key=lambda x: (x is None, x or 0)):
            got = limits[lim]
            common = sorted(set(got) & set(ref)) if ref else sorted(got)
            solved = sum(got[k] for k in common)
            ref_solved = sum(ref[k] for k in common) if ref else None
            frac = ev[model][lim][0] / max(1, ev[model][lim][1])
            diff, ci = None, (None, None)
            if ref and lim is not None:
                per_seed = defaultdict(list)
                for key in common:
                    per_seed[key[0]].append(got[key] - ref[key])
                diff, ci = boot(list(per_seed.values()))
            rows.append({"limit": lim, "problems": len(common), "solved": solved, "uncapped_solved": ref_solved,
                         "diff_vs_uncapped": None if diff is None else round(diff, 3),
                         "ci": None if ci[0] is None else [round(ci[0], 3), round(ci[1], 3)],
                         "evicted_problem_frac": round(frac, 3)})
            if lim is not None and best is None and ci[1] is not None and ci[1] >= 0 and frac <= 0.10:
                best = lim
        out[model] = {"by_limit": rows, "B_star_limit": best}
    return out


def table(rows: dict, cols: list[str]) -> str:
    lines = ["| key | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {k} | " + " | ".join(str(v.get(c)) for c in cols) + " |" for k, v in rows.items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--curve", action="store_true", help="pass rate by context limit and B* (single-agent worlds)")
    a = ap.parse_args(argv)
    cells = {}
    for batch in map(Path, a.batches):
        for cell in sorted({p.parent.parent for p in batch.rglob("world_s*/world.json")}):
            name = f"{batch.name}/{cell.relative_to(batch)}"
            cells[name] = [r for w in sorted(cell.glob("world_s*")) if (w / "world.json").exists() and "_cut" not in w.name
                           for r in world_rows(w)]
    summary = {}
    for cell, rows in cells.items():
        for model in sorted({r["model"] for r in rows}):
            summary[f"{cell} | {model}"] = summarise([r for r in rows if r["model"] == model])
    out = Path(a.out) if a.out else Path(a.batches[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cols = ["problems", "solved", "pass_rate", "pass_ci", "working_set_median", "working_set_p90", "tokens_per_solved",
            "share_mean", "removed_others_frac", "summaries_of_others", "for_partner_programs", "reuse_help", "reuse_mixup",
            "held", "evictions", "truncated", "text_removed", "cost", "cost_per_problem"]
    (out / "summary.md").write_text(table(summary, cols))
    pair = paired(cells)
    (out / "paired.json").write_text(json.dumps(pair, indent=2))
    (out / "paired.md").write_text(table(pair, ["problems", "a_solved", "b_solved", "diff", "ci"]))
    print(table(summary, cols))
    if a.curve:
        cur = curve(cells)
        (out / "curve.json").write_text(json.dumps(cur, indent=2))
        lines = []
        for model, c in cur.items():
            lines.append(f"## {model}  (B* limit: {c['B_star_limit']})\n")
            lines.append(table({str(r["limit"]): r for r in c["by_limit"]}, ["problems", "solved", "uncapped_solved", "diff_vs_uncapped", "ci", "evicted_problem_frac"]))
        (out / "curve.md").write_text("\n".join(lines))
        print("\n".join(lines))


if __name__ == "__main__":
    main()
