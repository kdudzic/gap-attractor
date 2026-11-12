"""Each click of the agent versus a reference chooser on the same menu, with pages as vectors (analysis/similarity.py).

Per step s: u = vec(x) (current page), v = vec(y) (chosen page), w = the chooser's expected vector of the next page.
  sim_agent = mean u.v, sim_model = mean u.w, sim_excess = their difference: how much more alike consecutive pages
  are for the agent than for the chooser. With similarity=categories (vec = topic mix chi) u.v is the topic
  overlap and sim_* are the trace shares of the topic tables; with embeddings u.v is the cosine.
Topic tables (similarity=categories only): F_agent = sum u v^T, F_model = sum u w^T. Both have row sums m = sum u
(chi rows sum to 1), so they compare cell by cell. Row e of A = F / m is the next page's topic distribution given
that a randomly drawn topic of the current page is e. Retention (the diagonal) is the chance of staying in topic e.
bits_per_step_vs_uniform needs no vectors and exists for every option.
"""
from __future__ import annotations

import numpy as np

from gap_attractor.analysis.inference import cluster_bootstrap, sign_test
from gap_attractor.analysis.logs import FOLLOW, Steps
from gap_attractor.analysis.models import Prediction
from gap_attractor.analysis.topics import TopicMap

LN2 = np.log(2)


def chi_matrix(steps: Steps, tm: TopicMap, weighting: str) -> np.ndarray:
    """(n_pages, K) membership matrix for every page seen in the logs."""
    return np.stack([tm.chi(c or [], weighting) for c in steps.page_cats]) if steps.page_cats else np.zeros((0, tm.k))


def page_matrix(steps: Steps, sim) -> np.ndarray:
    """(n_pages, dim) vectors of every page seen in the logs, from the chosen similarity option."""
    ids = [int(p) if p is not None else -1 for p in steps.page_ids]
    sim.prepare([p for p in ids if p >= 0])
    return sim.vectors(ids, categories=steps.page_cats if sim.has_topics else None)


def _stats(t: dict) -> dict:
    """Statistics from (possibly resampled) totals; works with any number of leading axes."""
    n, out = t["n"], {}
    if "s_agent" in t:
        out.update(sim_agent=t["s_agent"] / n, sim_model=t["s_model"] / n, sim_excess=(t["s_agent"] - t["s_model"]) / n)
    if "F_agent" in t:
        Fa, Fm, m = t["F_agent"], t["F_model"], t["m"]
        out.update({"A_agent": Fa / m[..., :, None], "A_model": Fm / m[..., :, None],
                    "D": (Fa - Fm) / n[..., None, None], "L": np.log((Fa + 0.5) / (Fm + 0.5))})
        out["retention_agent"] = np.diagonal(out["A_agent"], axis1=-2, axis2=-1)
        out["retention_model"] = np.diagonal(out["A_model"], axis1=-2, axis2=-1)
    if "ll_diff" in t:
        out["bits_per_step_vs_uniform"] = t["ll_diff"] / t["n_ll"] / LN2
        out["cross_entropy_bits"] = -t["ll"] / t["n_ll"] / LN2
    return out


def cluster_sums(steps: Steps, chi: np.ndarray, pred: Prediction, uniform_logp: np.ndarray, topics: bool = True) -> dict[str, np.ndarray]:
    """Additive per-cluster totals over the steps the model covers. `chi` is the page-vector matrix (any width);
    topic tables only with topics=True, similarity sums only when vectors have a width."""
    cov = pred.coverage
    cl = steps.cluster
    n_cl = int(steps.runs["cluster_index"].max()) + 1 if len(steps.runs) else 0
    U, V, W = chi[steps.x], chi[steps.y], pred.w
    K = chi.shape[1]
    s = {"n": np.zeros(n_cl)}
    np.add.at(s["n"], cl[cov], 1.0)
    if K:
        s["s_agent"], s["s_model"] = np.zeros(n_cl), np.zeros(n_cl)
        np.add.at(s["s_agent"], cl[cov], (U[cov] * V[cov]).sum(1))
        np.add.at(s["s_model"], cl[cov], (U[cov] * W[cov]).sum(1))
    if topics and K:
        s.update({"F_agent": np.zeros((n_cl, K, K)), "F_model": np.zeros((n_cl, K, K)), "m": np.zeros((n_cl, K))})
        np.add.at(s["F_agent"], cl[cov], U[cov, :, None] * V[cov, None, :])
        np.add.at(s["F_model"], cl[cov], U[cov, :, None] * W[cov, None, :])
        np.add.at(s["m"], cl[cov], U[cov])
    ll = cov & (steps.kind == FOLLOW) & np.isfinite(pred.logp) & np.isfinite(uniform_logp)
    s["ll"], s["ll_diff"], s["n_ll"] = np.zeros(n_cl), np.zeros(n_cl), np.zeros(n_cl)
    np.add.at(s["ll"], cl[ll], pred.logp[ll])
    np.add.at(s["ll_diff"], cl[ll], pred.logp[ll] - uniform_logp[ll])
    np.add.at(s["n_ll"], cl[ll], 1.0)
    return s


def compare(steps: Steps, chi: np.ndarray, pred: Prediction, uniform_logp: np.ndarray, *, min_row_mass: float,
            n_boot: int, seed: int, topics: bool = True) -> tuple[dict, dict]:
    """Point estimates, bootstrap intervals and sign tests for one model. Returns (arrays, summary); arrays (the
    topic tables) only with topics=True."""
    sums = cluster_sums(steps, chi, pred, uniform_logp, topics)
    present = sums["n"] > 0
    sums = {k: v[present] for k, v in sums.items()}
    if not present.any():
        return {}, {"n_steps": 0}
    tot = {k: v.sum(0) for k, v in sums.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        est = _stats(tot)
        ci = cluster_bootstrap(sums, _stats, n_boot, seed)
        per_cluster = _stats({k: v for k, v in sums.items()})
    scalar = lambda k: {"estimate": float(est[k]), "ci95": [float(x) for x in ci[k]]}
    summary = {"n_steps": int(tot["n"]), "n_clusters": int(present.sum())}
    arrays = {}
    if "F_agent" in tot:
        flagged = tot["m"] < min_row_mass
        arrays = {"F_agent": tot["F_agent"], "F_model": tot["F_model"], "m": tot["m"], "flagged_rows": flagged,
                  **{k: est[k] for k in ("A_agent", "A_model", "D", "L", "retention_agent", "retention_model")},
                  **{f"{k}_ci": ci[k] for k in ("A_agent", "A_model", "D", "L", "retention_agent", "retention_model")}}
        summary["flagged_rows"] = int(flagged.sum())
    if "s_agent" in tot:
        summary.update(sim_agent=scalar("sim_agent"), sim_model=scalar("sim_model"),
                       sim_excess={**scalar("sim_excess"), "sign_test": sign_test(per_cluster["sim_excess"])})
    if tot["n_ll"] > 0:
        ok = sums["n_ll"] > 0
        summary["n_link_steps"] = int(tot["n_ll"])
        summary["cross_entropy_bits"] = scalar("cross_entropy_bits")
        summary["bits_per_step_vs_uniform"] = {**scalar("bits_per_step_vs_uniform"),
                                               "sign_test": sign_test(sums["ll_diff"][ok])}
    return arrays, summary


def by_time(steps: Steps, chi: np.ndarray, pred: Prediction, uniform_logp: np.ndarray, *, n_boot: int, seed: int,
            bucket: int = 5) -> dict:
    """Similarity of consecutive pages (sim_*) in early/late halves of each run and in buckets of `bucket` steps."""
    if not chi.shape[1]:
        return {}
    t_max = np.zeros(len(steps.runs), int)
    np.maximum.at(t_max, steps.run, steps.t)
    groups = {"early_half": steps.t <= t_max[steps.run] / 2, "late_half": steps.t > t_max[steps.run] / 2}
    for b in range(0, int(steps.t.max()) + 1 if len(steps) else 0, bucket):
        groups[f"t{b}-{b + bucket - 1}"] = (steps.t >= b) & (steps.t < b + bucket)
    out = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        sub = steps.subset(mask)
        p = Prediction(pred.w[mask], pred.logp[mask], None if pred.covered is None else pred.covered[mask])
        _, s = compare(sub, chi, p, uniform_logp[mask], min_row_mass=0, n_boot=n_boot, seed=seed, topics=False)
        if s.get("n_steps"):
            out[name] = {k: s[k] for k in ("n_steps", "sim_agent", "sim_model", "sim_excess")}
    return out
