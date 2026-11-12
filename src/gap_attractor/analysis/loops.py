"""Two-page loops (A -> B -> A) and escapes from them, per condition (docs/analysis.md "loops").

A move t is a bounce when it returns to the page of two moves before (p_t == p_{t-2}, p_t != p_{t-1}). Consecutive
bounces form a loop between two pages. After c consecutive bounces the agent either bounces again or escapes (its
next page is neither of the two). Statistics from the observed paths, no fitted model:

  bounce_share       share of moves (from the second on) that are bounces
  first_bounce       move of the first bounce (runs with a bounce)
  hazard(c)          P(escape at the next move | c consecutive bounces), c = 1 .. max_bounces (the last bin pools
                     c >= max_bounces). Opportunities end with the run, so only observed next moves count.
  decline            hazard(1) - hazard(>= 3): positive if escaping gets harder the longer a loop lasts. NOT specific
                     to in-context learning: page pairs differ in how sticky they are, so the loops that last are the
                     sticky ones and the pooled hazard falls even when every pair's hazard is constant (the 15 Oct
                     memoryless runs show it). Compare against a memoryless condition on the same start pages instead
                     (paired below; and hazard_1, the escape chance after the first bounce, by memory)

Intervals by cluster bootstrap; clusters are start pages within a condition (loops.cluster_by: start_page | run).
Paired (loops.paired_field / paired_baseline, e.g. memory vs path_only): per start page, the difference in
bounce_share between two values of one condition field with all others fixed; mean, bootstrap interval, exact sign
test over start pages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gap_attractor.analysis.inference import cluster_bootstrap, sign_test


def path_events(path) -> dict:
    """Bounce flags and loop-escape opportunities of one path (page ids, start page first)."""
    p = list(path)
    m = len(p) - 1
    bounce = [False] * (m + 1)
    for t in range(2, m + 1):
        bounce[t] = p[t] == p[t - 2] and p[t] != p[t - 1]
    opp = []                                   # (c consecutive bounces so far, escaped at the next move?)
    run_len = 0
    for t in range(2, m + 1):
        run_len = run_len + 1 if bounce[t] else 0
        if run_len and t + 1 <= m:
            nxt = p[t + 1]
            opp.append((run_len, nxt != p[t - 1] and nxt != p[t]))   # neither of the loop's two pages
    firsts = [t for t in range(2, m + 1) if bounce[t]]
    return {"moves": m, "bounces": int(sum(bounce)), "bounce_share": (sum(bounce) / (m - 1)) if m >= 2 else np.nan,
            "first_bounce": firsts[0] if firsts else np.nan, "opportunities": opp}


def clusters_of(runs: pd.DataFrame, by: str) -> np.ndarray:
    if by == "run":
        return runs.cluster.astype(str).to_numpy()
    world = runs.cluster.astype(str).str.contains("/world")
    return np.where(world, runs.cluster.astype(str), "start:" + runs.start_page.astype(str))


def summarise(runs: pd.DataFrame, keys: list[str], cfg, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per condition, hazard table per condition and bounce count)."""
    C = int(cfg.max_bounces)
    rows, hz = [], []
    for cond, g in runs[runs.moves >= 2].groupby(keys, sort=True):
        cond = dict(zip(keys, cond if isinstance(cond, tuple) else (cond,)))
        cl = clusters_of(g, str(cfg.get("cluster_by", "start_page")))
        uniq, inv = np.unique(cl, return_inverse=True)
        k = len(uniq)
        ev = [path_events(p) for p in g.path]
        b = np.bincount(inv, [e["bounces"] for e in ev], k)
        n = np.bincount(inv, [e["moves"] - 1 for e in ev], k).astype(float)
        opp = np.zeros((k, C)); esc = np.zeros((k, C))
        for i, e in zip(inv, ev):
            for c, escaped in e["opportunities"]:
                j = min(c, C) - 1
                opp[i, j] += 1; esc[i, j] += escaped
        def stat(t):
            h = t["esc"] / np.where(t["opp"] > 0, t["opp"], np.nan)
            late = t["esc"][..., 2:].sum(-1) / np.where(t["opp"][..., 2:].sum(-1) > 0, t["opp"][..., 2:].sum(-1), np.nan)
            return {"bounce_share": t["b"] / t["n"], "hazard": h, "decline": h[..., 0] - late, "late": late}
        sums = {"b": b, "n": n, "opp": opp, "esc": esc}
        point = stat({k_: v.sum(0) for k_, v in sums.items()})
        ci = cluster_bootstrap(sums, stat, n_boot, seed)
        firsts = np.array([e["first_bounce"] for e in ev], float)
        rows.append({**cond, "runs": len(g), "clusters": k, "moves": int(sum(e["moves"] for e in ev)),
                     "bounce_share": float(point["bounce_share"]), "bounce_lo": float(ci["bounce_share"][0]),
                     "bounce_hi": float(ci["bounce_share"][1]), "runs_with_bounce": int(np.isfinite(firsts).sum()),
                     "median_first_bounce": float(np.nanmedian(firsts)) if np.isfinite(firsts).any() else np.nan,
                     "loop_opportunities": int(opp.sum()), "escapes": int(esc.sum()),
                     "hazard_1": float(point["hazard"][0]), "hazard_ge3": float(point["late"]),
                     "decline": float(point["decline"]), "decline_lo": float(ci["decline"][0]), "decline_hi": float(ci["decline"][1])})
        for j in range(C):
            hz.append({**cond, "bounces": f"{j + 1}" if j < C - 1 else f">={C}", "opportunities": int(opp[:, j].sum()),
                       "escapes": int(esc[:, j].sum()), "hazard": float(point["hazard"][j]),
                       "hazard_lo": float(ci["hazard"][0][j]), "hazard_hi": float(ci["hazard"][1][j])})
    return pd.DataFrame(rows), pd.DataFrame(hz)


def paired(runs: pd.DataFrame, keys: list[str], field: str, baseline: str, n_boot: int, seed: int) -> pd.DataFrame:
    """bounce_share under each other value of `field` minus under `baseline`, per start page, others fixed."""
    rest = [k for k in keys if k != field]
    runs = runs[runs.moves >= 2].assign(bs=[path_events(p)["bounce_share"] for p in runs[runs.moves >= 2].path])
    out = []
    for cond, g in runs.groupby(rest, sort=True) if rest else [((), runs)]:
        per = {v: h.groupby("start_page").bs.mean() for v, h in g.groupby(field)}
        if baseline not in per:
            continue
        for value in sorted(set(per) - {baseline}):
            common = per[baseline].index.intersection(per[value].index)
            d = (per[value][common] - per[baseline][common]).to_numpy(float)
            if not len(d):
                continue
            ci = cluster_bootstrap({"d": d, "n": np.ones(len(d))}, lambda t: {"m": t["d"] / t["n"]}, n_boot, seed)["m"]
            st = sign_test(d)
            out.append({**dict(zip(rest, cond if isinstance(cond, tuple) else (cond,))), "field": field, "baseline": baseline,
                        "other": value, "start_pages": len(d), "diff_bounce_share": float(d.mean()),
                        "diff_lo": float(ci[0]), "diff_hi": float(ci[1]), "n_pos": st["n_pos"], "n_neg": st["n_neg"],
                        "p_sign": st["p"]})
    return pd.DataFrame(out)


def write(out: Path, table: pd.DataFrame, hazard: pd.DataFrame, pairs: pd.DataFrame | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "loops.csv", index=False)
    hazard.to_csv(out / "loops_hazard.csv", index=False)
    if pairs is not None:
        pairs.to_csv(out / "loops_paired.csv", index=False)
