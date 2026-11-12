"""Do runs from different start pages end up in the same place, faster or tighter than random walks? Measured hop
by hop on the observed trails, without a transition model (docs/analysis.md "convergence").

For every agent run, `walks_per_run` walks of each baseline (analysis/walks.py: uniform, no_revisit, clickstream,
indegree, ...) start on its first page and make as many moves (capped at max_hop). At hop t, over the runs that
reached t (each hop reports how many), for pairs of runs with DIFFERENT start pages:

  similarity   mean vec(p_t) . vec(p'_t): topic overlap (categories) or cosine (embeddings)
  same_page    share of pairs on the same page

The null distribution comes from the walks: in replicate k, walk k of every run stands in for the run. One-sided
p = (1 + #{replicates >= agent}) / (1 + replicates); the band is the replicates' 2.5-97.5% range.

Absorption, split in half to avoid defining the attractor on the runs it is tested on: runs are split at random in
groups that are never separated (convergence.split_by: start_page, the default, keeps runs of a condition from the
same start page together, since at low temperature they repeat each other; worlds stay whole; or run); the attractor set is the top_k pages visited by the most runs of one half (hops >= 1, a run's
start page left out); in the other half we measure the share of moves that land in the set (`absorption`) and, per
hop, the share of runs that have entered it by then (`entered_by_hop`). Both halves take a turn, averaged over
n_splits random splits; the walks of the held-out runs give the null. Pages a walk can reach that are hubs for every
walker (the set may be all hubs) are why the indegree baseline matters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gap_attractor.analysis.walks import Walker


def pair_similarity(V: np.ndarray, starts: np.ndarray) -> float:
    """Mean dot product over ordered pairs (i != j) with different start pages."""
    if len(V) < 2 or V.shape[1] == 0:
        return np.nan
    total = V.sum(0)
    num, den = float(total @ total), float(len(V)) ** 2
    for g in np.unique(starts):
        Vg = V[starts == g].sum(0)
        num -= float(Vg @ Vg); den -= float((starts == g).sum()) ** 2
    return num / den if den > 0 else np.nan


def pair_same_page(pages: np.ndarray, starts: np.ndarray) -> float:
    """Share of ordered pairs (i != j) with different start pages that are on the same page."""
    n = len(pages)
    den = n * n - sum(int((starts == g).sum()) ** 2 for g in np.unique(starts))
    if den <= 0:
        return np.nan
    num = 0
    df = pd.DataFrame({"p": pages, "s": starts})
    for _, g in df.groupby("p"):
        if len(g) > 1:
            num += len(g) ** 2 - (g.s.value_counts() ** 2).sum()
    return num / den


def _one_sided(agent: float, null: np.ndarray) -> tuple[float, float, float, float]:
    null = null[np.isfinite(null)]
    if not np.isfinite(agent) or not len(null):
        return np.nan, np.nan, np.nan, np.nan
    p = (1 + (null >= agent - 1e-12).sum()) / (1 + len(null))
    return float(null.mean()), float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975)), float(p)


def attractor_set(paths: list[np.ndarray], top_k: int) -> set:
    """The top_k pages visited (hops >= 1, start page excluded) by the most runs; ties by page id."""
    count = {}
    for p in paths:
        for page in set(p[1:].tolist()) - {int(p[0])}:
            count[page] = count.get(page, 0) + 1
    ranked = sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))
    return {page for page, _ in ranked[:top_k]}


def absorption_of(paths: list[np.ndarray], A: set, max_hop: int) -> tuple[float, np.ndarray]:
    """(share of moves landing in A, per hop t = 1..max_hop the share of runs that reached t and entered A by t)."""
    moves = np.concatenate([np.isin(p[1:], list(A)) for p in paths if len(p) > 1]) if paths else np.zeros(0)
    entered = np.full(max_hop, np.nan)
    for t in range(1, max_hop + 1):
        alive = [p for p in paths if len(p) > t]
        if alive:
            entered[t - 1] = np.mean([np.isin(p[1:t + 1], list(A)).any() for p in alive])
    return (float(moves.mean()) if len(moves) else np.nan), entered


def condition(runs: pd.DataFrame, walks: dict, vec, cfg, seed: int) -> tuple[list, list, dict]:
    """runs: one condition's runs (path as page ids, truncated); walks[baseline]: (n_runs, W, moves+1) page ids,
    padded with -1. vec(page_ids) -> vectors."""
    max_hop, W = int(cfg.max_hop), int(cfg.walks_per_run)
    paths = [np.asarray(p, np.int64) for p in runs.path]
    starts = np.array([int(p[0]) for p in paths])
    lens = np.array([len(p) - 1 for p in paths])
    curves = []
    for t in range(1, max_hop + 1):
        alive = np.flatnonzero(lens >= t)
        if len(np.unique(starts[alive])) < 2:
            break
        pages = np.array([paths[i][t] for i in alive])
        a_sim, a_same = pair_similarity(vec(pages), starts[alive]), pair_same_page(pages, starts[alive])
        for b, arr in walks.items():
            wp = arr[alive, :, t]                                          # (n_alive, W)
            null_sim = np.array([pair_similarity(vec(wp[:, k]), starts[alive]) for k in range(W)])
            null_same = np.array([pair_same_page(wp[:, k], starts[alive]) for k in range(W)])
            for stat, a, null in (("similarity", a_sim, null_sim), ("same_page", a_same, null_same)):
                m, lo, hi, p = _one_sided(a, null)
                curves.append({"hop": t, "runs": int(len(alive)), "starts": int(len(np.unique(starts[alive]))),
                               "baseline": b, "statistic": stat, "agent": a, "walks": m, "walks_lo": lo,
                               "walks_hi": hi, "p_walks": p})
    # split-half absorption
    rng = np.random.default_rng(seed)
    if str(cfg.get("split_by", "start_page")) == "start_page":
        world = runs.cluster.astype(str).str.contains("/world")
        clusters = np.where(world, runs.cluster.astype(str), "start:" + runs.start_page.astype(str))
    else:
        clusters = runs.cluster.astype(str).to_numpy()
    uniq = np.unique(clusters)
    absorb, entered_rows = [], []
    if len(uniq) >= 4:
        agent_abs, agent_ent = [], []
        null_abs = {b: [] for b in walks}
        null_ent = {b: [] for b in walks}
        for _ in range(int(cfg.n_splits)):
            half = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
            inA = np.array([c in half for c in clusters])
            for define, test in ((inA, ~inA), (~inA, inA)):
                A = attractor_set([paths[i] for i in np.flatnonzero(define)], int(cfg.top_k))
                held = np.flatnonzero(test)
                ab, en = absorption_of([paths[i] for i in held], A, max_hop)
                agent_abs.append(ab); agent_ent.append(en)
                for b, arr in walks.items():
                    reps = [absorption_of([arr[i, k][arr[i, k] >= 0] for i in held], A, max_hop) for k in range(W)]
                    null_abs[b].append([r[0] for r in reps]); null_ent[b].append(np.stack([r[1] for r in reps]))
        a_abs, a_ent = float(np.nanmean(agent_abs)), np.nanmean(np.stack(agent_ent), 0)
        for b in walks:
            na = np.nanmean(np.array(null_abs[b]), 0)                        # (W,) averaged over splits
            m, lo, hi, p = _one_sided(a_abs, na)
            absorb.append({"baseline": b, "top_k": int(cfg.top_k), "n_splits": int(cfg.n_splits), "groups": int(len(uniq)),
                           "split_by": str(cfg.get("split_by", "start_page")), "agent": a_abs,
                           "walks": m, "walks_lo": lo, "walks_hi": hi, "p_walks": p})
            ne = np.nanmean(np.stack(null_ent[b]), 0)                        # (W, max_hop)
            for t in range(max_hop):
                m, lo, hi, p = _one_sided(float(a_ent[t]), ne[:, t])
                if np.isfinite(a_ent[t]):
                    entered_rows.append({"hop": t + 1, "baseline": b, "statistic": "entered_by_hop", "agent": float(a_ent[t]),
                                         "walks": m, "walks_lo": lo, "walks_hi": hi, "p_walks": p})
    return curves + entered_rows, absorb, {"runs": len(runs), "split_groups": int(len(uniq)), "max_moves": int(lens.max()) if len(lens) else 0}


def run(runs: pd.DataFrame, keys: list[str], graph, sim, clickstream, cfg, unlisted: float = 5.0, seed: int = 0):
    """All conditions: (curves table, absorption table, audit). cfg = the `convergence` node."""
    walker = Walker(graph, clickstream, unlisted, seed)
    runs = runs.copy()
    runs["start_local"] = [int(graph.locals_of_page_ids([p[0]])[0]) if p else -1 for p in runs.path]
    use = runs[(runs.moves > 0) & (runs.start_local >= 0)].copy()
    max_hop, W = int(cfg.max_hop), int(cfg.walks_per_run)
    use["path"] = [list(p[: max_hop + 1]) for p in use.path]
    baselines = [b for b in cfg.baselines if b != "clickstream" or clickstream is not None]
    audit = {"runs": len(runs), "runs_without_moves": int((runs.moves == 0).sum()),
             "runs_start_outside_core": int(((runs.moves > 0) & (runs.start_local < 0)).sum()), "conditions": {}}
    per_cond, pids = [], set()
    for cond, g in use.groupby(keys):
        cond = cond if isinstance(cond, tuple) else (cond,)
        walks = {}
        for b in baselines:
            arr = np.full((len(g), W, max_hop + 1), -1, np.int64)
            for i, (_, r) in enumerate(g.iterrows()):
                m = len(r.path) - 1
                arr[i, :, : m + 1] = walker.pid[walker.walk(b, int(r.start_local), m, W)]
            walks[b] = arr
            pids.update(np.unique(arr[arr >= 0]).tolist())
        pids.update(int(x) for p in g.path for x in p)
        per_cond.append((cond, g, walks))
    pid_list = np.array(sorted(pids), np.int64)
    sim.prepare(pid_list)
    M = sim.vectors(pid_list)
    vec = lambda ids: M[np.searchsorted(pid_list, np.asarray(ids, np.int64))]
    curves, absorb = [], []
    for cond, g, walks in per_cond:
        c, a, info = condition(g, walks, vec, cfg, seed)
        label = dict(zip(keys, cond))
        curves += [{**label, "similarity_option": sim.name, **r} for r in c]
        absorb += [{**label, **r} for r in a]
        top = attractor_set([np.asarray(p, np.int64) for p in g.path], int(cfg.top_k))
        info["top_pages_all_runs"] = sorted(int(x) for x in top)
        audit["conditions"][" | ".join(map(str, cond))] = info
    return pd.DataFrame(curves), pd.DataFrame(absorb), audit


def write(out: Path, curves: pd.DataFrame, absorb: pd.DataFrame, audit: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out / "convergence.csv", index=False)
    absorb.to_csv(out / "absorption.csv", index=False)
    (out / "convergence_audit.json").write_text(json.dumps(audit, indent=1, default=str), encoding="utf-8")
