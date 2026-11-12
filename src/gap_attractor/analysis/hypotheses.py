"""Hypothesis tests on clicks: where the chosen link ranks on fixed scores, against reference choosers.

The non-parametric counterpart of HypTrails (Singer et al. 2015), without its concentration parameter and without
fitting a choice model (docs/hypothesis_testing_plan.md in the results docs; docs/analysis.md "hypotheses").

A hypothesis is a fixed score s(i, j, history) for every link j on the menu the agent was shown at page i, declared
before the data are seen (registry SCORES; a new one is a function registered here or named by dotted path).
Per link step with at least two scored links that differ:

  r      the chosen link's mid-rank percentile on the menu by s: (#lower + (#tied - 1) / 2) / (n - 1), in [0, 1].
         If the agent picked uniformly, its expectation is exactly 0.5 whatever the menu.
  E_B r  the expected rank under a reference chooser B on the same menu (uniform: 0.5; clickstream: human clicks;
         indegree: links in proportion to how linked their page is). B carries a control: "s beyond B".
  d      r - E_B r.

Per condition and (hypothesis, reference): R = mean r, lift = mean d, a 95% cluster-bootstrap interval, and two
p-values (two-sided):
  p_cluster  sign-flip test over clusters: each cluster's summed deviation keeps or flips its sign at random. It
             allows any dependence within a cluster, so it is the one to read (Holm within the primary family uses it).
  p_mc       B itself picks on the same menus many times, every click independently. Too small whenever the agent's
             clicks repeat (temperature 0, loops, runs from the same start page); kept as a reference only.
Clusters (hypotheses.cluster_by): start_page (default: runs of one condition from the same start page are one
cluster, since at low temperature they repeat each other; worlds stay whole) or run (runs, or worlds).
Steps whose links all score the same tell nothing about s and are left out (counted). Roles: primary, exploratory,
placebo (a random score: must come out null) and check (display position under a seeded shuffle: a primacy bias is
real behaviour, not a failure; the shuffle keeps it from confounding the other scores).

Bits per step: each hypothesis as a fixed chooser P(j) = (1 - lam) b_j / sum b + lam / n, with b = s - min over the
menu (lam not fitted; reported at a few declared values), against uniform: mean log2 P(chosen) + log2 n.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gap_attractor.analysis.inference import cluster_bootstrap
from gap_attractor.analysis.logs import FOLLOW, Steps

SCORES: dict[str, Callable] = {}


def score(name):
    def deco(fn):
        SCORES[name] = fn
        return fn
    return deco


def get_score(name: str) -> Callable:
    if name in SCORES:
        return SCORES[name]
    module, _, attr = name.replace(":", ".").rpartition(".")
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError, ValueError) as e:
        raise KeyError(f"unknown hypothesis {name!r}; registered: {sorted(SCORES)}") from e


def _seg(steps: Steps) -> np.ndarray:
    """Step index of every menu entry."""
    return np.repeat(np.arange(len(steps)), steps.menu_sizes())


def visited_before(steps: Steps) -> list[set]:
    """Per step, the page rows the run had been on before choosing (its earlier current pages and choices, and the
    current page). Steps the loader left out (harness picks) are missing, so this slightly undercounts."""
    out, seen_by_run = [None] * len(steps), {}
    for s in np.lexsort((steps.t, steps.run)):
        seen = seen_by_run.setdefault(int(steps.run[s]), set())
        seen.add(int(steps.x[s]))
        out[s] = set(seen)
        seen.add(int(steps.y[s]))
    return out


# ------------------------------------------------------------------ hypotheses (scores per menu entry)

@score("similarity")
def similarity(steps: Steps, ctx: dict) -> np.ndarray:
    """How alike the link's page is to the current page: topic overlap (similarity=categories) or cosine (embeddings)."""
    X = ctx["X"]
    if X.shape[1] == 0:
        raise ValueError("the similarity hypothesis needs page vectors: analysis/similarity=categories or embeddings")
    seg = _seg(steps)
    return (X[steps.x[seg]] * X[steps.menu]).sum(1)


@score("hub")
def hub(steps: Steps, ctx: dict) -> np.ndarray:
    """log(1 + in-degree) of the link's page on the link graph (configs/analysis/graph/); NaN off the graph."""
    indeg, loc = ctx["indegree"](), ctx["graph"]().locals_of_page_ids(steps.page_ids[steps.menu].astype(np.int64))
    return np.where(loc >= 0, np.log1p(indeg[np.maximum(loc, 0)]), np.nan)


@score("human")
def human(steps: Steps, ctx: dict) -> np.ndarray:
    """The share of human readers' clicks from the current page that went to the link (clickstream; unlisted links
    count as clickstream_unlisted clicks)."""
    return _clickstream_probs(steps, ctx)


@score("novelty")
def novelty(steps: Steps, ctx: dict) -> np.ndarray:
    """1 if the run has not been on the link's page yet."""
    seen, seg = ctx["visited"](), _seg(steps)
    return np.array([float(int(steps.menu[k]) not in seen[s]) for k, s in enumerate(seg)])


@score("backtrack")
def backtrack(steps: Steps, ctx: dict) -> np.ndarray:
    """1 if the link leads back to the page the run was on just before the current one."""
    prev = np.full(len(steps), -1)
    order = np.lexsort((steps.t, steps.run))
    for a, b in zip(order[:-1], order[1:]):
        if steps.run[a] == steps.run[b]:
            prev[b] = steps.x[a]
    return (steps.menu == prev[_seg(steps)]).astype(float)


@score("self_interest")
def self_interest(steps: Steps, ctx: dict) -> np.ndarray:
    """Cosine of the link's page to a fixed prototype text (hypotheses.self_text, about AI and minds). Needs
    similarity=embeddings."""
    sim = ctx["sim"]
    if getattr(sim, "name", "") != "embeddings":
        raise ValueError("self_interest needs analysis/similarity=embeddings")
    v = np.asarray(sim.embedder.embed([str(ctx["self_text"])])[0], np.float32)
    v = v / (np.linalg.norm(v) or 1.0)
    return ctx["X"][steps.menu] @ v


@score("position")
def position(steps: Steps, ctx: dict) -> np.ndarray:
    """Placebo under a seeded shuffle: minus the link's position in the list as shown (first = highest). With
    display_order_policy=source it measures a bias toward links listed early instead."""
    seg = _seg(steps)
    return -(np.arange(len(steps.menu)) - steps.menu_ptr[seg]).astype(float)


@score("random")
def random_score(steps: Steps, ctx: dict) -> np.ndarray:
    """Placebo: a fixed pseudo-random number per (current page, link page)."""
    seg, ids = _seg(steps), steps.page_ids
    key = lambda a, b: int(hashlib.blake2b(f"{a}>{b}".encode(), digest_size=8).hexdigest(), 16) / 2 ** 64
    return np.array([key(ids[steps.x[s]], ids[steps.menu[k]]) for k, s in enumerate(seg)])


# ------------------------------------------------------------------ reference choosers (probability per menu entry)

def _clickstream_probs(steps: Steps, ctx: dict) -> np.ndarray:
    cs = ctx.get("clickstream")
    if cs is None:
        raise ValueError("needs clickstream.npz (scripts/build_wiki_resources.py clickstream)")
    out = np.zeros(len(steps.menu))
    for s in range(len(steps)):
        a, b = steps.menu_ptr[s], steps.menu_ptr[s + 1]
        if b > a:
            out[a:b], _ = cs.probs(int(steps.page_ids[steps.x[s]]), steps.page_ids[steps.menu[a:b]], float(ctx.get("unlisted", 5)))
    return out


def reference_probs(name: str, steps: Steps, ctx: dict) -> np.ndarray:
    """Unnormalised weight of every menu entry under a reference chooser (normalised per step, over scored links)."""
    if name == "uniform":
        return np.ones(len(steps.menu))
    if name == "clickstream":
        return _clickstream_probs(steps, ctx)
    if name == "indegree":
        indeg, loc = ctx["indegree"](), ctx["graph"]().locals_of_page_ids(steps.page_ids[steps.menu].astype(np.int64))
        return np.where(loc >= 0, indeg[np.maximum(loc, 0)].astype(float), 0.0) + 1.0
    raise ValueError(f"unknown reference chooser {name!r}; expected uniform, clickstream or indegree")


# ------------------------------------------------------------------ ranks

def midranks(s: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of every entry within its own vector: (#lower + (#tied - 1) / 2) / (n - 1)."""
    n = len(s)
    order = np.argsort(s, kind="stable")
    sorted_s = s[order]
    lo = np.searchsorted(sorted_s, s, "left")
    hi = np.searchsorted(sorted_s, s, "right")
    return (lo + (hi - lo - 1) / 2) / (n - 1)


def step_table(steps: Steps, scores: np.ndarray, weights: dict[str, np.ndarray]) -> dict:
    """Per informative link step: the chosen rank, and per reference the expected rank and the per-entry
    (rank - expected, probability) needed for the Monte Carlo test."""
    out = {"step": [], "r": [], "n_links": [], **{f"E_{k}": [] for k in weights}}
    entries = {k: [] for k in weights}
    skipped = {"jump": 0, "menu_lt2": 0, "chosen_unscored": 0, "no_variation": 0}
    for s in range(len(steps)):
        if steps.kind[s] != FOLLOW:
            skipped["jump"] += 1; continue
        a, b = steps.menu_ptr[s], steps.menu_ptr[s + 1]
        c = steps.chosen[s] - a
        sc = scores[a:b]
        ok = np.isfinite(sc)
        if b - a < 2 or ok.sum() < 2:
            skipped["menu_lt2"] += 1; continue
        if not ok[c]:
            skipped["chosen_unscored"] += 1; continue
        if np.ptp(sc[ok]) == 0:
            skipped["no_variation"] += 1; continue
        idx = np.flatnonzero(ok)
        r = midranks(sc[idx])
        rc = r[np.searchsorted(idx, c)]
        out["step"].append(s); out["r"].append(rc); out["n_links"].append(len(idx))
        for k, w in weights.items():
            p = np.clip(np.nan_to_num(w[a:b][idx]), 0, None)
            p = p / p.sum() if p.sum() > 0 else np.full(len(idx), 1 / len(idx))
            e = float(p @ r)
            out[f"E_{k}"].append(e)
            entries[k].append((r - e, p))
    return {k: np.array(v) for k, v in out.items()}, entries, skipped


def monte_carlo_lift(entries: list, observed: float, n: int, seed: int) -> float:
    """Two-sided p of the mean lift when the reference chooser itself picks on every menu (n replicates)."""
    if not entries:
        return np.nan
    rng = np.random.default_rng(seed)
    null = np.zeros(n)
    for dev, p in entries:
        null += dev[np.minimum(np.searchsorted(np.cumsum(p), rng.random(n) * p.sum()), len(p) - 1)]
    null /= len(entries)
    return float((1 + (np.abs(null) >= abs(observed) - 1e-12).sum()) / (n + 1))


def sign_flip_p(d_sums: np.ndarray, n_total: float, n: int, seed: int) -> float:
    """Two-sided p of sum(d) / N against random sign flips of whole clusters' summed deviations."""
    if len(d_sums) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    observed = abs(d_sums.sum()) / n_total
    null = np.abs((rng.choice([-1.0, 1.0], size=(n, len(d_sums))) * d_sums).sum(1)) / n_total
    return float((1 + (null >= observed - 1e-12).sum()) / (n + 1))


def bits_per_step(steps: Steps, scores: np.ndarray, rows: np.ndarray, lam: float) -> np.ndarray:
    """Per given step: log2 P(chosen) + log2 n under P = (1 - lam) b / sum b + lam / n, b = s - min over the menu."""
    out = np.empty(len(rows))
    for i, s in enumerate(rows):
        a, b = steps.menu_ptr[s], steps.menu_ptr[s + 1]
        sc = scores[a:b]
        ok = np.isfinite(sc)
        idx = np.flatnonzero(ok)
        v = sc[idx] - sc[idx].min()
        n = len(idx)
        p = (1 - lam) * (v / v.sum() if v.sum() > 0 else np.full(n, 1 / n)) + lam / n
        out[i] = np.log2(p[np.searchsorted(idx, steps.chosen[s] - a)]) + np.log2(n)
    return out


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN kept)."""
    p = np.asarray(p, float)
    ok = np.flatnonzero(np.isfinite(p))
    out = np.full(len(p), np.nan)
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        out[k] = running
    return out.tolist()


# ------------------------------------------------------------------ one condition

def condition_tests(steps: Steps, ctx: dict, hypotheses, references, lams, n_boot: int, seed: int,
                    cluster: np.ndarray | None = None) -> tuple[list, list, dict]:
    rows, bits_rows, audit = [], [], {}
    weights = {k: reference_probs(k, steps, ctx) for k in references}
    cluster = steps.cluster if cluster is None else cluster
    for h in hypotheses:
        try:
            sc = np.asarray(get_score(h)(steps, ctx), float)
        except (ValueError, FileNotFoundError) as e:
            audit[h] = {"skipped": str(e)}
            continue
        tab, entries, skipped = step_table(steps, sc, weights)
        audit[h] = {"informative_steps": int(len(tab["r"])), **skipped}
        if not len(tab["r"]):
            continue
        cl = cluster[tab["step"]]
        uniq, inv = np.unique(cl, return_inverse=True)
        for k in references:
            d = tab["r"] - tab[f"E_{k}"]
            sums = {"d": np.bincount(inv, d, len(uniq)), "r": np.bincount(inv, tab["r"], len(uniq)),
                    "n": np.bincount(inv, None, len(uniq)).astype(float)}
            ci = cluster_bootstrap(sums, lambda t: {"lift": t["d"] / t["n"], "R": t["r"] / t["n"]}, n_boot, seed)
            rows.append({"hypothesis": h, "reference": k, "steps": len(d), "clusters": len(uniq),
                         "R": float(tab["r"].mean()), "R_lo": float(ci["R"][0]), "R_hi": float(ci["R"][1]),
                         "expected": float(tab[f"E_{k}"].mean()), "lift": float(d.mean()),
                         "lift_lo": float(ci["lift"][0]), "lift_hi": float(ci["lift"][1]),
                         "p_cluster": sign_flip_p(sums["d"], float(len(d)), n_boot, seed),
                         "p_mc": monte_carlo_lift(entries[k], float(d.mean()), n_boot, seed),
                         "mean_links": float(tab["n_links"].mean())})
        for lam in lams:
            bpd = bits_per_step(steps, sc, tab["step"], lam)
            sums = {"b": np.bincount(inv, bpd, len(uniq)), "n": np.bincount(inv, None, len(uniq)).astype(float)}
            ci = cluster_bootstrap(sums, lambda t: {"bits": t["b"] / t["n"]}, n_boot, seed)["bits"]
            bits_rows.append({"hypothesis": h, "lambda": lam, "steps": len(bpd), "bits_vs_uniform": float(bpd.mean()),
                              "bits_lo": float(ci[0]), "bits_hi": float(ci[1])})
    return rows, bits_rows, audit


def cluster_labels(steps: Steps, by: str) -> np.ndarray:
    """Per step, its cluster: the run's cluster (run, or world) or, with start_page, its start page (worlds whole)."""
    if by == "run":
        return steps.cluster
    runs = steps.runs
    world = runs["cluster"].astype(str).str.contains("/world")
    label = np.where(world, runs["cluster"].astype(str), "start:" + runs["start_page"].astype(str))
    return pd.factorize(label)[0][steps.run]


def run(steps: Steps, keys: list[str], ctx: dict, cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All conditions: (hypotheses table, bits table, audit). cfg = the `hypotheses` node."""
    hyps, refs = list(cfg.tests), list(cfg.references)
    primary, placebos, checks = set(cfg.get("primary") or []), set(cfg.get("placebos") or []), set(cfg.get("checks") or [])
    by = str(cfg.get("cluster_by", "start_page"))
    if by not in ("start_page", "run"):
        raise ValueError("hypotheses.cluster_by must be start_page or run")
    run_key = steps.runs[keys].astype(str).agg("\x1f".join, axis=1).to_numpy()
    table, bits, audit = [], [], {}
    for key in sorted(set(run_key)):
        cond = dict(zip(keys, key.split("\x1f")))
        sub = steps.subset(np.isin(steps.run, np.flatnonzero(run_key == key)))
        cache = {}
        sub_ctx = {**ctx, "visited": lambda sub=sub, cache=cache: cache["v"] if "v" in cache else cache.setdefault("v", visited_before(sub))}
        rows, brows, a = condition_tests(sub, sub_ctx, hyps, refs, list(cfg.lambdas), int(cfg.n_boot), int(cfg.seed),
                                         cluster_labels(sub, by))
        prim = [r for r in rows if r["hypothesis"] in primary and r["reference"] == "uniform"]
        for r, adj in zip(prim, holm([r["p_cluster"] for r in prim])):
            r["p_holm_primary"] = adj
        for r in rows:
            r["role"] = ("placebo" if r["hypothesis"] in placebos else "check" if r["hypothesis"] in checks
                         else "primary" if r["hypothesis"] in primary else "exploratory")
            r["cluster_by"] = by
        table += [{**cond, **r} for r in rows]
        bits += [{**cond, **r} for r in brows]
        audit[key.replace("\x1f", " | ")] = a
    return pd.DataFrame(table), pd.DataFrame(bits), audit


def write(out: Path, table: pd.DataFrame, bits: pd.DataFrame, audit: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "hypotheses.csv", index=False)
    bits.to_csv(out / "hypotheses_bits.csv", index=False)
    (out / "hypotheses_audit.json").write_text(json.dumps(audit, indent=1, default=str), encoding="utf-8")
