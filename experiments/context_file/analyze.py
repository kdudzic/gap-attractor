"""What the context-file conditions did with their context, per condition (descriptive; no fitted models).

    uv run python experiments/context_file/analyze.py runs/ctx_pilot --out <folder>

Reads run folders (run.json + steps.jsonl; `_superseded` skipped) and the version logs. Conditions are
experiment.condition + model + the analysis labels of main (memory_setup, world_setup, effective temperature) + the
step limit + the task (browse | find | essay, "+search" for the opening search) + the prompt wording
(experiment.prompt_wording, e.g. browse_explicit vs browse_explicit_v2) + the reasoning setting of each
agent (off, an effort, or default), so nothing with different settings is pooled (a joint world counts as one
model: its composition). Intervals: percentile bootstrap over clusters (a world, or for single agents the start
page: runs from the same start are near-twins at temperature 0, so they count once).
Writes runs.csv, conditions.csv, survival.csv, exposure.csv, convergence.csv, removals.csv (+ removal_events.csv),
identity.csv (+ identity_agents.csv, mentions.csv), contrasts.csv, tasks.csv, essays_blind.jsonl + essays_key.csv
and summary.md. Navigation itself (outcomes, clicks, walks against random walks and clickstream) is the main
pipeline: gap-attractor-analyze.

- compaction: editing calls per step, how many applied, why the others were not (invalid call, current link list,
  gate, others' turns), how much an applied edit shrank the context, pressure (context / limit), the share of steps
  whose page took the context over the limit, held moves (hard budget), capped pages, evictions, nudges, steps where
  the editing turns ran out, and how runs ended (context_budget is a competing ending, not a failure to stop; a STOP
  made while over the limit is its own ending, model_stopped_over_limit);
- memory: pages_kept and path_recall at fixed steps (5, 10, 20; runs that reached them) and at the last step (which
  depends on run length: compare only within one ending);
- survival of page (article) turns: Kaplan-Meier per condition AND per run ending (runs ended by the budget remove
  pages under duress, so they are not pooled with runs that reached the step limit); who removed them;
- exposure (joint): how often an agent chose a page another agent of its world had already visited, against the
  same rate computed with the agents of the OTHER worlds of the condition, who could not have been seen (the
  no-exposure baseline); the difference is the exposure effect;
- convergence (joint): pages visited by two or more agents of a world, against worlds re-formed by permuting each
  agent slot across the worlds of the condition (world size kept, averaged over all worlds);
- removals (removals.py): which turns an edit touches, deletes outright, and how much of each it cuts (a turn
  rewritten in place or merged into a summary is told apart from one deleted), as the mean rank of those turns on
  fixed scores (another agent's turn, older, longer, kind; a placebo), against 0.5 (blind to the score); sign-flip
  test over clusters. Per agent also: other agents' turns deleted, and taken over (rewritten, so their text now
  stands as the editor's);
- identity (identity.py): mentions of another writer and of an inconsistent context in the agent's own words,
  naming pages only another agent of the world had opened (against agents of other worlds, who could not be seen),
  /notes activity, find and essay outcomes;
- awareness (awareness.py; exploratory, 23 Oct 2026): whether an agent says or shows that its context changed: other
  agents' edits since its previous step (exact, from the version log) and its own turns among them, mentions of a
  change (claimed as its own, or not), reading the context file, backups into /work, reopening removed pages;
  columns aware_* in identity_agents.csv and identity.csv;
- contrasts: the same measures between grid conditions with everything else equal (difference of means, bootstrap
  interval over clusters, permutation p over clusters). PRIMARY lists a proposed primary family (Holm-adjusted);
  it is to be fixed before the main runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import awareness  # noqa: E402
import identity  # noqa: E402
import numpy as np  # noqa: E402
import removals  # noqa: E402
from gap_attractor.analysis.hypotheses import holm, sign_flip_p  # noqa: E402
from gap_attractor.analysis.logs import effective_temperature, memory_setup, world_setup  # noqa: E402

KEYS = ("condition", "model", "memory", "world", "temperature", "step_limit", "task", "prompt", "reasoning")
MATCH = ("model", "composition", "memory_base", "temperature", "step_limit", "task", "prompt", "reasoning")   # equal in a contrast
GRID_SETTINGS = ("sharing", "attribution", "awareness", "others_as", "workspace")    # what the grid varies
PER_AGENT = ("budget", "limit", "reserve")      # a shared file holds N x these: compared per agent
AT = (5, 10, 20)
END = (10 ** 9, 0)                                         # after every step of a world
REFERENCE = "j_separate"
CONTRASTS = [("j_rumour", "j_full"), ("j_strangers", "j_full"), ("j_strangers", "j_rumour"),
             ("j_notes_rumour", "j_notes"), ("j_notes_strangers", "j_notes"), ("j_notes_strangers", "j_notes_rumour"),
             ("j_full", "j_notes"), ("j_rumour", "j_notes_rumour"), ("j_strangers", "j_notes_strangers")]
CONTRASTS += [(c, REFERENCE) for c in ("j_full", "j_rumour", "j_strangers", "j_notes", "j_notes_rumour", "j_notes_strangers")]
# Proposed primary family (measure, condition, against); Holm within it. To be fixed with the team before main runs.
EVENT_TOTALS = ("touched", "deleted", "rewritten", "merged", "touched_others", "deleted_others", "taken_over_others",
                "chars_cut", "chars_cut_others")      # per agent, summed over its edits (removals.events)
PRIMARY = [("mentions_other_writer", "j_strangers", REFERENCE), ("removal_rank_others_cut", "j_strangers", "j_full"),
           ("appropriation_final_excess", "j_strangers", "j_full")]


def task_of(info: dict) -> str:
    config = info.get("config") or {}
    wording = str((config.get("experiment") or {}).get("prompt_wording") or "")
    goal = ((config.get("ctx") or {}).get("goal") or {}).get("task")
    task = "essay" if wording.startswith("essay") else "find" if goal == "find" else "browse"
    variant = info.get("variant")
    variant = variant.get("name") if isinstance(variant, dict) else variant
    return task + ("+search" if variant == "b_search_first" else "")


def reasoning_label(r) -> str:
    if not r:
        return "default"
    if r.get("enabled") is False:
        return "off"
    if r.get("effort"):
        return str(r["effort"])
    return json.dumps(r, sort_keys=True)


def reasoning_of(info: dict) -> tuple[str, str]:
    """(this agent's setting, the world's settings in agent order)."""
    world = info.get("world") or {}
    if world.get("reasoning") is not None:
        labels = [reasoning_label(r) for r in world["reasoning"]]
        k = world.get("agent_id")
        return (labels[k - 1] if k and k <= len(labels) else "default"), "+".join(labels)
    own = reasoning_label((((info.get("config") or {}).get("provider") or {}).get("extra_body") or {}).get("reasoning"))
    return own, own


def memory_base(info: dict) -> str:
    """The memory settings without those the grid varies, with a shared file's budget per agent, so a contrast
    between grid conditions holds everything else equal."""
    settings = dict(info.get("memory_settings") or {})
    n = (info.get("world") or {}).get("n_agents") or 1
    if settings.get("sharing") in ("full", "readonly_others"):
        settings.update({k: settings[k] // n for k in PER_AGENT if isinstance(settings.get(k), int)})
    return ",".join(f"{k}={settings[k]}" for k in sorted(settings) if k not in GRID_SETTINGS)


def find_runs(paths):
    for p in map(Path, paths):
        for run in sorted(p.rglob("run.json")):
            if (run.parent / "steps.jsonl").exists() and "_superseded" not in run.parts:
                yield run.parent


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def boot(values_by_cluster: dict, n=2000, seed=0):
    """Percentile 95 % interval of the mean over clusters (each cluster: list of values)."""
    clusters = [v for v in values_by_cluster.values() if v]
    if len(clusters) < 2:
        return None, None
    rng, stats = random.Random(seed), []
    for _ in range(n):
        draw = [x for c in (rng.choice(clusters) for _ in clusters) for x in c]
        stats.append(sum(draw) / len(draw))
    stats.sort()
    return round(stats[int(0.025 * n)], 4), round(stats[int(0.975 * n) - 1], 4)


def versions_for(folder: Path, info: dict):
    world = info.get("world") or {}
    if world.get("world_id") is not None:
        root = folder.parent / "worlds" / f"world_{int(world['world_id']):03d}"
        name = "versions.jsonl" if world.get("sharing", "full") != "separate" else f"versions_agent_{world['agent_id']}.jsonl"
        path = root / name
    else:
        path = folder / "context" / "versions.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def run_row(folder: Path):
    info = json.loads((folder / "run.json").read_text())
    steps = [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines() if l.strip()]
    world = info.get("world") or {}
    mem = [s.get("memory") or {} for s in steps]
    edits = [e for m in mem for e in m.get("edits") or []]
    applied = [e for e in edits if e.get("applied")]
    rejected = defaultdict(int)
    for e in edits:
        if not e.get("applied"):
            rejected[e.get("rejected") or "no_change"] += 1
    nudges = defaultdict(int)
    for m in mem:
        if m.get("nudge"):
            nudges[m["nudge"]] += 1
    shrink = [1 - e["tokens_after"] / e["tokens_before"] for e in applied if e.get("tokens_before")]
    last = next((m for m in reversed(mem) if m.get("pages_kept") is not None), {})
    batch = info.get("batch_id")
    pressures = [m.get("pressure") for m in mem if m.get("pressure") is not None]
    held = [len((s.get("extras") or {}).get("held_moves") or []) for s in steps]
    config = info.get("config") or {}
    models = sorted(set(world.get("models") or []))
    model = (config.get("provider") or {}).get("model")
    own_reasoning, reasoning = reasoning_of(info)
    row = {
        "run": f"{batch}/{info['run_id']}", "folder": str(folder),
        "cluster": f"{batch}/world{world['world_id']}" if world.get("world_id") is not None else f"start/{info.get('start_page') or info['run_id']}",
        "condition": info.get("condition"), "model": ("world:" + "+".join(models)) if len(models) > 1 else model, "agent_model": model,
        "step_limit": (config.get("experiment") or {}).get("step_limit"),
        "limit": (info.get("memory_settings") or {}).get("limit"),
        "memory": memory_setup(info), "world": world_setup(info), "temperature": effective_temperature(info, folder),
        "composition": world_setup(info).split("|", 1)[-1] if world else world_setup(info), "memory_base": memory_base(info),
        "task": task_of(info), "prompt": (config.get("experiment") or {}).get("prompt_wording"),
        "reasoning": reasoning, "agent_reasoning": own_reasoning,
        "sharing": world.get("sharing"), "attribution": world.get("attribution"), "awareness": world.get("awareness"),
        "agent": world.get("agent_id"), "steps": len(steps),
        "end": "model_stopped_over_limit" if info.get("stopped_over_limit") else (info.get("outcome") or {}).get("termination_reason"),
        "invalid_steps": sum(bool((s.get("invalid_action") or {}).get("is_invalid")) for s in steps),
        "harness_picks": sum((s.get("invalid_action") or {}).get("harness_handling") == "skip" for s in steps),
        "edit_turns_ran_out": sum(bool((s.get("extras") or {}).get("side_turns_exhausted")) for s in steps),
        "edit_calls": len(edits), "edits_applied": len(applied), "edits_per_step": round(len(edits) / max(1, len(steps)), 3),
        "edit_validity": round(1 - rejected.get("invalid", 0) / len(edits), 3) if edits else None,
        "shrink_per_applied_edit": mean(shrink),
        "pressure_mean": mean(pressures), "pressure_max": max(pressures or [0]),
        "steps_over_target": round(sum(p > 1 for p in pressures) / len(pressures), 3) if pressures else None,
        "capped_pages": sum(bool(m.get("capped")) for m in mem), "evicted_turns": sum(m.get("evicted_turns") or 0 for m in mem),
        "removed_others_turns": sum(e.get("removed_others") or 0 for e in applied),
        "pages_kept_last": last.get("pages_kept"), "path_recall_last": last.get("path_recall"),
        **{f"pages_kept_at_{k}": mem[k - 1].get("pages_kept") for k in AT if len(mem) >= k},
        **{f"path_recall_at_{k}": mem[k - 1].get("path_recall") for k in AT if len(mem) >= k},
        "held_moves": sum(held), "steps_with_held_moves": round(sum(h > 0 for h in held) / len(held), 3) if held else None,
        "over_limit_at_move": sum(bool(m.get("over_limit_at_move")) for m in mem),
        "named_by_others": mean([float(m["named_by_others"]) for m in mem if "named_by_others" in m]),
        "menu_named_by_others": mean([m.get("menu_named_by_others") for m in mem]),
        "sandbox_restarts": max([e.get("sandbox_restarts") or 0 for e in edits] or [0]),
        **{f"rejected_{k}": v for k, v in rejected.items()}, **{f"nudge_{k}": v for k, v in nudges.items()},
    }
    return row, info, steps


def survival_rows(folder, info):
    """One row per article turn of this agent: lifetime (in the editor's steps) and whether it was removed, by whom."""
    versions, me = versions_for(folder, info), (info.get("world") or {}).get("agent_id")
    added, out = {}, []
    for v in versions:
        for t in v.get("added") or []:
            if t["kind"] == "observation" and t["agent"] == me:
                added[t["id"]] = v.get("step") or 0
        for t in v.get("removed") or []:
            if t["id"] in added:
                who = "harness" if v["op"] == "evict" else ("self" if v.get("agent") == me else "other")
                out.append({"lifetime": (v.get("step") or 0) - added.pop(t["id"]), "removed": True, "by": who})
    end = max([v.get("step") or 0 for v in versions] or [0])
    out += [{"lifetime": end - s, "removed": False, "by": None} for s in added.values()]
    return out


def kaplan_meier(rows):
    """{t: S(t)} at the event times, from (lifetime, removed) with censoring."""
    times = sorted({r["lifetime"] for r in rows if r["removed"]})
    s, curve = 1.0, {}
    for t in times:
        at_risk = sum(r["lifetime"] >= t for r in rows)
        events = sum(r["lifetime"] == t and r["removed"] for r in rows)
        if at_risk:
            s *= 1 - events / at_risk
        curve[t] = s
    return curve


def surviving(curve, t):
    return round(min([v for k, v in curve.items() if k <= t] or [1.0]), 3)


def walk_of(steps):
    """[(round, position, page, choice)] for one agent of a joint world."""
    out = []
    for s in steps:
        r = (s.get("memory") or {}).get("round") or {}
        out.append((r.get("round", s.get("step_index")), r.get("position", 0), s.get("current_page_title"), s.get("parsed_choice")))
    return out


def seen_rate(agent, others):
    """Share of the agent's choices that are pages one of `others` had been on or opened before that choice (by
    round, then position in the round)."""
    hits = n = 0
    for r, p, _, choice in agent:
        if not choice or choice == "STOP":
            continue
        visited = {x for o in others for (r2, p2, page, chosen) in o if (r2, p2) < (r, p) for x in (page, chosen) if x and x != "STOP"}
        hits += choice in visited
        n += 1
    return (hits, n)


def exposure(worlds):
    """worlds: world -> {slot: walk}. Observed: choices of pages other agents of the same world had visited.
    Baseline: the same against the other agents of each OTHER world (unseen), averaged over those worlds."""
    rows = []
    for w, agents in worlds.items():
        for k, walk in agents.items():
            hits, n = seen_rate(walk, [a for j, a in agents.items() if j != k])
            base = [seen_rate(walk, [a for j, a in other.items() if j != k]) for w2, other in worlds.items() if w2 != w]
            base = [h / m for h, m in base if m]
            rows.append({"world": w, "slot": k, "choices": n, "observed": hits / n if n else None, "baseline": mean(base)})
    return rows


def convergence(worlds: dict, rng: random.Random, draws=2000):
    """worlds: world -> {slot: set of pages}. Statistic: mean over worlds of the pages visited by >= 2 of its agents
    (start pages shared by several agents left out). Null: each slot's page sets permuted across worlds."""
    slots = sorted({k for a in worlds.values() for k in a})
    full = {w: a for w, a in worlds.items() if sorted(a) == slots}
    if len(full) < 2:
        return {"worlds": len(full), "observed": None, "null": None, "p": None}

    def shared(sets):
        return len({p for p in set().union(*sets) if sum(p in s for s in sets) >= 2})
    ws = list(full)
    observed = sum(shared([full[w][k] for k in slots]) for w in ws) / len(ws)
    null = []
    for _ in range(draws):
        columns = {k: rng.sample([full[w][k] for w in ws], len(ws)) for k in slots}
        null.append(sum(shared([columns[k][i] for k in slots]) for i in range(len(ws))) / len(ws))
    return {"worlds": len(ws), "observed": round(observed, 3), "null": round(sum(null) / draws, 3),
            "p": round((1 + sum(x >= observed for x in null)) / (1 + draws), 4)}


def run_identity(folder: Path, info: dict, steps: list[dict], row: dict) -> dict:
    """What the agent wrote and did, for the identity measures: its walk, its reasons by time, its final statement,
    lexicon hits, the commands it ran, the world's notes log and its task outcome."""
    walk = walk_of(steps)
    times = [(r, p) for r, p, _, _ in walk]
    final = identity.final_statement(steps, info)
    texts = identity.texts_of(steps, final)
    mention_rows, per = [], {}
    for category in identity.COMPILED:
        found = [(i, src, m, snip) for i, src, text in texts for m, snip in identity.hits(text, category)]
        mention_rows += [{"run": row["run"], "condition": row["condition"], "model": row["agent_model"], "agent": row["agent"],
                          "step": i, "source": src, "category": category, "match": m, "snippet": snip} for i, src, m, snip in found]
        counted = [x for x in found if x[1] in identity.SOURCES[category]]
        per[f"mentions_{category}"] = float(bool(counted))
        per[f"first_step_{category}"] = min((i for i, *_ in counted), default=None)
        per[f"steps_with_{category}"] = len({i for i, *_ in counted})
    commands = [(times[i - 1], text) for i, src, text in texts if src == "edit_call"]
    reasons = [(times[i - 1], text) for i, src, text in texts if src == "reason"]
    world = info.get("world") or {}
    notes = []
    if world.get("world_id") is not None:
        nf = folder.parent / "worlds" / f"world_{int(world['world_id']):03d}" / "notes.jsonl"
        notes = [json.loads(l) for l in nf.read_text().splitlines() if l.strip()] if nf.exists() else []
    goal = info.get("goal") or {}
    task = {"reached": float(bool(goal.get("reached"))) if goal else None,
            "first_arrival_step": min((a.get("step") for a in goal.get("arrivals") or []), default=None),
            "essay_chars": len(final) if final and row["task"].startswith("essay") else None}
    aware = awareness.summary(awareness.per_step(steps, versions_for(folder, info), row["agent"], final))
    return {"walk": walk, "reasons": reasons, "final": final, "per": per, "mentions": mention_rows,
            "commands": commands, "notes": notes, "task": task, "awareness": aware}


def contrast(a: dict, b: dict, rng: random.Random, n=2000) -> dict:
    """Clusters -> values for two conditions: difference of means, bootstrap interval (clusters resampled within
    each condition) and a two-sided permutation p (clusters reassigned between the conditions, sizes kept)."""
    a = {k: v for k, v in a.items() if v}
    b = {k: v for k, v in b.items() if v}
    if not a or not b:
        return {"clusters_a": len(a), "clusters_b": len(b), "diff": None, "ci": None, "p": None}

    def m(cs):
        xs = [x for c in cs for x in c]
        return sum(xs) / len(xs)
    ca, cb = list(a.values()), list(b.values())
    diff = m(ca) - m(cb)
    boots = sorted(m([rng.choice(ca) for _ in ca]) - m([rng.choice(cb) for _ in cb]) for _ in range(n))
    pooled, extreme = ca + cb, 0
    for _ in range(n):
        rng.shuffle(pooled)
        extreme += abs(m(pooled[:len(ca)]) - m(pooled[len(ca):])) >= abs(diff) - 1e-12
    return {"clusters_a": len(ca), "clusters_b": len(cb), "diff": round(diff, 4),
            "ci": f"{boots[int(0.025 * n)]:.4f}..{boots[int(0.975 * n) - 1]:.4f}", "p": round((1 + extreme) / (1 + n), 4)}


def removal_summary(events: list[dict], seed: int) -> list[dict]:
    """Per score and outcome (touched, deleted, cut): edits informative about it, mean r, interval (bootstrap over
    clusters), sign-flip p."""
    out = []
    for name, (role, _) in removals.SCORES.items():
        for outcome in removals.OUTCOMES:
            field = name if outcome == "touched" else f"{name}_{outcome}"
            clusters = defaultdict(list)
            for e in events:
                if e.get(field) is not None:
                    clusters[e["cluster"]].append(e[field] - 0.5)
            n = sum(len(v) for v in clusters.values())
            lo, hi = boot(clusters, seed=seed)
            sums = np.array([sum(v) for v in clusters.values()])
            out.append({"score": name, "outcome": outcome, "role": role, "edits": n, "skipped": len(events) - n,
                        "clusters": len(clusters), "r": round(0.5 + sums.sum() / n, 4) if n else None,
                        "r_ci": f"{0.5 + lo:.4f}..{0.5 + hi:.4f}" if lo is not None else None,
                        "p_signflip": round(sign_flip_p(sums, n, 5000, seed), 4) if len(sums) >= 2 else None})
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, surv, events, idn = [], [], [], {}
    walks = defaultdict(lambda: defaultdict(dict))
    loaded = [run_row(folder) for folder in find_runs(a.paths)]
    temps = defaultdict(set)              # a world's agents can differ (one model takes no temperature): label the world
    for row, _, _ in loaded:
        if row["agent"] is not None:
            temps[row["cluster"]].add(str(row["temperature"]))
    for row, info, steps in loaded:
        row["agent_temperature"] = row["temperature"]
        if row["agent"] is not None:
            row["temperature"] = "+".join(sorted(temps[row["cluster"]]))
        folder = Path(row["folder"])
        rows.append(row)
        key = tuple(row[k] for k in KEYS)
        surv += [dict(s, **{k: row[k] for k in KEYS}, end=row["end"]) for s in survival_rows(folder, info)]
        if row["agent"] is not None:
            walks[key][row["cluster"]][row["agent"]] = walk_of(steps)
        idn[row["run"]] = run_identity(folder, info, steps, row)
        events += [dict(e, **{k: row[k] for k in KEYS}, cluster=row["cluster"], run=row["run"])
                   for e in removals.events(versions_for(folder, info)) if e["editor"] == row["agent"]]
    if not rows:
        raise SystemExit("no runs found")
    write_csv(out / "runs.csv", rows)

    by = defaultdict(list)
    for r in rows:
        by[tuple(r[k] for k in KEYS)].append(r)
    cond_rows = []
    for key, rs in sorted(by.items(), key=lambda kv: [str(x) for x in kv[0]]):
        c = dict(zip(KEYS, key), limit=rs[0].get("limit"), runs=len(rs), clusters=len({r["cluster"] for r in rs}))
        for field in ("steps", "edits_per_step", "edit_validity", "shrink_per_applied_edit", "pressure_mean", "pressure_max",
                      "steps_over_target", "capped_pages", "evicted_turns", "invalid_steps", "harness_picks", "edit_turns_ran_out",
                      "pages_kept_last", "path_recall_last", "named_by_others", "menu_named_by_others", "removed_others_turns",
                      "held_moves", "steps_with_held_moves", "over_limit_at_move",
                      *[f"{m}_at_{k}" for m in ("pages_kept", "path_recall") for k in AT]):
            c[field] = mean([r.get(field) for r in rs])
        for field in [k for k in dict.fromkeys(k for r in rs for k in r) if k.startswith(("rejected_", "nudge_"))]:
            c[field] = sum(r.get(field) or 0 for r in rs)
        clusters = defaultdict(list)
        for r in rs:
            if r.get("pages_kept_last") is not None:
                clusters[r["cluster"]].append(r["pages_kept_last"])
        c["pages_kept_last_ci"] = "%s..%s" % boot(clusters, seed=a.seed)
        ends = defaultdict(int)
        for r in rs:
            ends[r["end"]] += 1
        c["ends"] = ", ".join(f"{k}:{v}" for k, v in sorted(ends.items(), key=lambda kv: str(kv[0])))
        c["context_budget_share"] = round(ends.get("context_budget", 0) / len(rs), 3)
        c["stopped_over_limit_share"] = round(ends.get("model_stopped_over_limit", 0) / len(rs), 3)
        cond_rows.append(c)
    write_csv(out / "conditions.csv", cond_rows)

    groups = defaultdict(list)
    for s in surv:                       # per condition, and per ending (all, then each ending on its own)
        groups[tuple(s[k] for k in KEYS) + ("all",)].append(s)
        groups[tuple(s[k] for k in KEYS) + (s["end"],)].append(s)
    surv_rows = []
    for key, ss in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        curve = kaplan_meier(ss)
        median = next((t for t, v in sorted(curve.items()) if v <= 0.5), None)
        removed = [s for s in ss if s["removed"]]
        surv_rows.append({**dict(zip(KEYS + ("ending",), key)), "page_turns": len(ss), "removed": len(removed),
                          **{f"surviving_{k}_steps": surviving(curve, k) for k in (1, 3, 5, 10)},
                          "median_survival_steps": median,
                          **{f"removed_by_{w}": sum(s["by"] == w for s in removed) for w in ("self", "other", "harness")}})
    write_csv(out / "survival.csv", surv_rows)

    rng, exp_rows, conv_rows = random.Random(a.seed), [], []
    for key, ws in sorted(walks.items(), key=lambda kv: [str(x) for x in kv[0]]):
        per = exposure(ws)
        clusters = defaultdict(list)
        for r in per:
            if r["observed"] is not None and r["baseline"] is not None:
                clusters[r["world"]].append(r["observed"] - r["baseline"])
        lo, hi = boot(clusters, seed=a.seed)
        exp_rows.append({**dict(zip(KEYS, key)), "worlds": len(ws), "observed": mean([r["observed"] for r in per]),
                         "unseen_baseline": mean([r["baseline"] for r in per]),
                         "excess": mean([x for v in clusters.values() for x in v]), "excess_ci": f"{lo}..{hi}"})
        sets = {}
        for w, agents in ws.items():
            firsts = [walk[0][2] for walk in agents.values() if walk]
            shared_starts = {x for x in firsts if firsts.count(x) > 1}      # start_assignment=same: not convergence
            sets[w] = {k: {page for (_, _, page, _) in walk if page} - shared_starts for k, walk in agents.items()}
        conv_rows.append({**dict(zip(KEYS, key)), **convergence(sets, rng)})
    if exp_rows:
        write_csv(out / "exposure.csv", exp_rows)
        write_csv(out / "convergence.csv", conv_rows)

    # removals: which turns edits take out
    rem_rows = []
    if events:
        write_csv(out / "removal_events.csv", events)
        by_key = defaultdict(list)
        for e in events:
            by_key[tuple(e[k] for k in KEYS)].append(e)
        for key, es in sorted(by_key.items(), key=lambda kv: [str(x) for x in kv[0]]):
            rem_rows += [{**dict(zip(KEYS, key)), **r} for r in removal_summary(es, a.seed)]
        write_csv(out / "removals.csv", rem_rows)

    # identity: per agent, then per condition
    agents, row_of = [], {r["run"]: r for r in rows}
    events_of = defaultdict(list)
    for e in events:
        events_of[e["run"]].append(e)
    runs_by = defaultdict(lambda: defaultdict(dict))          # key -> cluster -> slot -> run id
    for r in rows:
        runs_by[tuple(r[k] for k in KEYS)][r["cluster"]][r["agent"]] = r["run"]
    for key, worlds in runs_by.items():
        for w, slots in worlds.items():
            for k, run in slots.items():
                r, d = row_of[run], idn[run]
                ag = {**{f: r[f] for f in ("run", *KEYS, "cluster", "agent", "agent_model", "agent_reasoning", "sharing",
                                           "attribution", "awareness", "steps", "end", "removed_others_turns")}, **d["per"],
                      **{f: sum(e[f] for e in events_of[run]) for f in EVENT_TOTALS}}
                if len(slots) > 1:
                    seen = identity.appropriation(d["walk"], d["reasons"], [idn[o]["walk"] for j, o in slots.items() if j != k],
                                                  d["final"], END)
                    unseen = [identity.appropriation(d["walk"], d["reasons"], [idn[o]["walk"] for j, o in other.items() if j != k],
                                                     d["final"], END) for w2, other in worlds.items() if w2 != w]
                    for f in ("reasons_naming", "final_names"):
                        base = mean([u[f] for u in unseen])
                        ag[f"appropriation_{f}"] = seen[f]
                        ag[f"appropriation_{f}_unseen"] = base
                        ag[f"appropriation_{f}_excess"] = (round(seen[f] - base, 4) if seen[f] is not None and base is not None
                                                           else None)
                    ag["appropriation_final_unattributed"] = seen["final_unattributed"]
                    ag["appropriation_final_pages"] = seen["final_pages"]
                if r["sharing"] == "notes":
                    ag.update(identity.notes_activity(d["notes"], r["agent"], d["commands"]))
                ag.update(d["task"])
                ag.update({f"aware_{f}": v for f, v in d["awareness"].items()})      # exploratory (awareness.py)
                agents.append(ag)
    write_csv(out / "identity_agents.csv", agents)
    mention_rows = [m for d in idn.values() for m in d["mentions"]]
    if mention_rows:
        write_csv(out / "mentions.csv", mention_rows)

    def clusters_of(rs, field):
        cs = defaultdict(list)
        for x in rs:
            if x.get(field) is not None:
                cs[x["cluster"]].append(float(x[field]))
        return cs

    agents_by = defaultdict(list)
    for ag in agents:
        agents_by[tuple(ag[k] for k in KEYS)].append(ag)
    id_rows = []
    for key, ags in sorted(agents_by.items(), key=lambda kv: [str(x) for x in kv[0]]):
        c = {**dict(zip(KEYS, key)), "agents": len(ags), "clusters": len({x["cluster"] for x in ags})}
        for field in ("mentions_other_writer", "mentions_inconsistency", "mentions_we", "steps_with_other_writer",
                      "appropriation_reasons_naming", "appropriation_reasons_naming_unseen", "appropriation_reasons_naming_excess",
                      "appropriation_final_names", "appropriation_final_names_unseen", "appropriation_final_names_excess",
                      "appropriation_final_unattributed", "notes_writes", "notes_commands", "looked_after_others_wrote",
                      "notes_files_labelled", "reached", "first_arrival_step", "essay_chars",
                      "aware_steps_after_others_edits", "aware_own_turns_changed_by_others", "aware_change_other_after",
                      "aware_change_other_rest", "aware_change_self_steps", "aware_inspect_after", "aware_inspect_rest",
                      "aware_backups", "aware_reopened_removed_pages"):
            vals = [x.get(field) for x in ags if x.get(field) is not None]
            if not vals:
                continue
            c[field] = mean([float(v) for v in vals])
            if field.startswith(("mentions_", "appropriation_")) and field.endswith(("writer", "inconsistency", "excess")):
                lo, hi = boot(clusters_of(ags, field), seed=a.seed)
                c[field + "_ci"] = f"{lo}..{hi}" if lo is not None else None
        id_rows.append(c)
    write_csv(out / "identity.csv", id_rows)

    # contrasts between grid conditions with everything else equal
    measures = defaultdict(dict)                          # measure -> key -> {cluster: values}
    for key, ags in agents_by.items():
        for field, name in (("mentions_other_writer", None), ("mentions_inconsistency", None), ("mentions_we", None),
                            ("appropriation_final_names_excess", "appropriation_final_excess"),
                            ("appropriation_reasons_naming_excess", "appropriation_reasons_excess"),
                            ("steps", None), ("touched_others", None), ("deleted_others", None),
                            ("taken_over_others", None), ("chars_cut_others", None)):
            measures[name or field][key] = clusters_of(ags, field)
    for e in events:
        for field, name in (("others", "removal_rank_others"), ("others_deleted", "removal_rank_others_deleted"),
                            ("others_cut", "removal_rank_others_cut")):
            if e.get(field) is not None:
                measures[name].setdefault(tuple(e[k] for k in KEYS), defaultdict(list))[e["cluster"]].append(e[field])
    match_of = {tuple(r[k] for k in KEYS): tuple(r[k] for k in MATCH) for r in rows}
    con_rows, rng_c = [], random.Random(a.seed)
    for measure, by in sorted(measures.items()):
        for ca, cb in CONTRASTS:
            for ka in [k for k in by if k[0] == ca]:
                for kb in [k for k in by if k[0] == cb and match_of[k] == match_of[ka]]:
                    con_rows.append({"measure": measure, "a": ca, "b": cb, **dict(zip(MATCH, match_of[ka])),
                                     "primary": (measure, ca, cb) in PRIMARY, **contrast(by[ka], by[kb], rng_c)})
    groups_p = defaultdict(list)
    for i, c in enumerate(con_rows):
        if c["primary"]:
            groups_p[tuple(c[k] for k in MATCH)].append(i)
    for idx in groups_p.values():
        for i, p in zip(idx, holm([con_rows[i]["p"] if con_rows[i]["p"] is not None else float("nan") for i in idx])):
            con_rows[i]["p_holm_primary"] = None if p != p else round(p, 4)
    if con_rows:
        write_csv(out / "contrasts.csv", con_rows)

    # tasks and essays
    task_rows = []
    for key, ags in sorted(agents_by.items(), key=lambda kv: [str(x) for x in kv[0]]):
        if key[KEYS.index("task")].startswith(("find", "essay")):
            essays = [x for x in ags if x.get("essay_chars")]
            task_rows.append({**dict(zip(KEYS, key)), "agents": len(ags), "reached": mean([x.get("reached") for x in ags]),
                              "first_arrival_step": mean([x.get("first_arrival_step") for x in ags]),
                              "essays": len(essays), "essay_chars": mean([x["essay_chars"] for x in essays]),
                              "essays_naming_partner_pages": mean([x.get("appropriation_final_names") for x in essays])})
    if task_rows:
        write_csv(out / "tasks.csv", task_rows)
    key_rows = []
    with open(out / "essays_blind.jsonl", "w", encoding="utf-8") as f:
        for ag in agents:
            final = idn[ag["run"]]["final"]
            if ag["task"].startswith("essay") and final:
                eid = hashlib.sha1(ag["run"].encode()).hexdigest()[:12]
                f.write(json.dumps({"id": eid, "essay": final}, ensure_ascii=False) + "\n")
                key_rows.append({"id": eid, **{k: ag[k] for k in ("run", *KEYS, "cluster", "agent", "agent_model", "agent_reasoning")}})
    if key_rows:
        write_csv(out / "essays_key.csv", key_rows)
    else:
        (out / "essays_blind.jsonl").unlink()

    lines = ["# Context-file conditions", "", f"{len(rows)} runs from {', '.join(a.paths)}.", "",
             "| condition | model | limit | steps | runs | edits/step | valid edits | shrink | pressure (max) | over limit | held moves | "
             "pages kept at 10 | ends |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cond_rows:
        lines.append(f"| {c['condition']} | {c['model']} | {c['limit']} | {c['step_limit']} | {c['runs']} | {c['edits_per_step']} | {c['edit_validity']} | "
                     f"{c['shrink_per_applied_edit']} | {c['pressure_mean']} ({c['pressure_max']}) | {c['steps_over_target']} | "
                     f"{c['held_moves']} | {c['pages_kept_at_10']} | {c['ends']} |")
    lines += ["", "Page turns surviving (Kaplan-Meier), all runs and by ending:", "",
              "| condition | ending | page turns | 1 step | 3 steps | 5 steps | median |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['condition']} | {r['ending']} | {r['page_turns']} | {r['surviving_1_steps']} | {r['surviving_3_steps']} | "
              f"{r['surviving_5_steps']} | {r['median_survival_steps']} |" for r in surv_rows]
    if exp_rows:
        lines += ["", "Joint worlds: choices of pages another agent had visited (observed) vs agents of other worlds (unseen):", "",
                  "| condition | worlds | observed | unseen baseline | excess (95 % CI) | convergence | permuted | p |", "|---|---|---|---|---|---|---|---|"]
        for e, c in zip(exp_rows, conv_rows):
            lines.append(f"| {e['condition']} | {e['worlds']} | {e['observed']} | {e['unseen_baseline']} | {e['excess']} ({e['excess_ci']}) | "
                         f"{c['observed']} | {c['null']} | {c['p']} |")
    if rem_rows:
        lines += ["", "What edits cut: mean rank of the cut turns by each score, weighted by the share of each turn cut "
                  "(0.5 = blind to the score; > 0.5 = cuts more of it). Touched at all / deleted outright are in removals.csv. "
                  "Edits informative about the score in brackets:", "",
                  "| condition | model | others | older | longer | page | edit trail | placebo |", "|---|---|---|---|---|---|---|---|"]
        rem_by = defaultdict(dict)
        for r in rem_rows:
            if r["outcome"] == "cut":
                rem_by[(r["condition"], r["model"])][r["score"]] = r
        for (cond, model), sc in rem_by.items():
            cell = lambda s: f"{sc[s]['r']} ({sc[s]['edits']}, p={sc[s]['p_signflip']})" if sc.get(s, {}).get("r") is not None else "–"
            lines.append(f"| {cond} | {model} | {cell('others')} | {cell('older')} | {cell('longer')} | {cell('page')} | "
                         f"{cell('edit_trail')} | {cell('placebo')} |")
    if id_rows:
        lines += ["", "Other writers: share of agents whose own words mention another writer / an inconsistent context / "
                  "'we'; naming pages only a partner had opened (final statement; against agents of other worlds):", "",
                  "| condition | model | reasoning | agents | other writer | inconsistency | we | partner pages named (unseen) | "
                  "unattributed | notes written |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in id_rows:
            lines.append(f"| {c['condition']} | {c['model']} | {c['reasoning']} | {c['agents']} | {c.get('mentions_other_writer')} | "
                         f"{c.get('mentions_inconsistency')} | {c.get('mentions_we')} | {c.get('appropriation_final_names', '–')} "
                         f"({c.get('appropriation_final_names_unseen', '–')}) | {c.get('appropriation_final_unattributed', '–')} | "
                         f"{c.get('notes_writes', '–')} |")
    prim = [c for c in con_rows if c["primary"]]
    if prim:
        lines += ["", "Proposed primary contrasts (Holm within each model/task group; the family is to be fixed before main runs):", "",
                  "| measure | a | b | model | diff (95 % CI) | p | p Holm |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {c['measure']} | {c['a']} | {c['b']} | {c['model']} | {c['diff']} ({c['ci']}) | {c['p']} | "
                  f"{c.get('p_holm_primary')} |" for c in prim]
    lines += ["", "Small samples: read these as descriptions. Convergence and exposure need about 20 worlds per condition."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
