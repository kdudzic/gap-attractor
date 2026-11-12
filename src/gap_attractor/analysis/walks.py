"""Stage walks: whole runs of the agent versus random walks from the same start page, with the same number of moves.

For every agent run r with m >= 1 moves, `per_run` walks of each baseline start on r's first page and make m moves
on the link graph (configs/analysis/graph/, by default the enwiki strongly connected core from scripts/graph/).
Each path gives a few statistics; the comparison is
paired: d_r = statistic(agent run r) - mean over r's walks, and diff = the mean of d_r. Its 95% interval comes from
resampling clusters (runs, or worlds), as in the topics stage. Its p-value is a Monte Carlo test against the walks
themselves: in each replicate one walk of every run stands in for the agent, which gives the distribution of diff
when the agent is just another walk (no assumption about the shape; right for skewed counts such as revisits).
Nothing is fitted:

  uniform      every out-link equally likely (random walk)
  jump         uniform, but each move is a jump to a random core page with the agent's own jump rate in that
               condition (search results and intent jumps / moves); left out where the agent never jumped
  no_revisit   uniform over out-links not yet visited in this walk (all of them if every one was visited)
  clickstream  out-links in proportion to human clicks (analysis/clickstream.py; unlisted links count as `unlisted`)
  indegree     out-links in proportion to 1 + the in-degree of their page (a hub-seeking walker; a control for
               attraction to hubs)

Statistics per path p_0 .. p_m (pages by page id). The first three need page vectors (analysis/similarity.py) and
are left out with similarity=none; the rest need only links:
  step_similarity   mean over moves of vec(p_t-1) . vec(p_t): topic overlap (categories) or cosine (embeddings);
                    with categories it is the per-step trace of the clicks stage
  breadth           how widely the visited pages spread (PageVectors.breadth: entropy of the mean topic mix in
                    bits for categories, 1 - |mean unit vector| for embeddings)
  drift_from_start  1 - vec(p_0) . vec(p_m): how far in meaning the run ends from its start page
  revisit_rate      share of moves landing on a page already on the path
  hub_share         share of moves landing on a hub (graph.hubs: Claire's hub pool, or the most linked-to pages)
  end_distance      directed link distance from p_0 to p_m (only with walks.distance=true; one BFS per start page)

The agent's path is every page it opened, harness picks included: those are part of what it saw next.
The graph is the 2026-10-08 build, so a few agent links are not graph edges and some pages are outside the core;
agent pages outside the core count as not-a-hub and have no distance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gap_attractor.analysis.inference import cluster_bootstrap
from gap_attractor.analysis.logs import find_runs, run_row

BASELINES = ("uniform", "jump", "no_revisit", "clickstream", "indegree")
VECTOR_STATS = ("step_similarity", "breadth", "drift_from_start")
LINK_STATS = ("revisit_rate", "hub_share", "end_distance")
STATS = VECTOR_STATS + LINK_STATS


def load_paths(paths, group_by: dict[str, str]) -> pd.DataFrame:
    """One row per run: condition fields, cluster, the page-id path (start page first) and the number of jumps."""
    rows = []
    for folder in find_runs(paths):
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        steps = [json.loads(l) for l in (folder / "steps.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        path, jumps = [], 0
        for st in sorted(steps, key=lambda s: s.get("step_index", 0)):
            action = st.get("action") or "follow"
            y = (st.get("resolution") or {}).get("resolved_page_id")
            if action == "search_start":
                if y is not None:
                    path = [int(y)]
                continue
            if not path and st.get("current_page_id") is not None:
                path = [int(st["current_page_id"])]
            if st.get("parsed_choice") in (None, "STOP") or y is None:
                continue
            path.append(int(y))
            jumps += action == "intent_jump" or (action == "search" and bool((st.get("search") or {}).get("results")))
        rows.append({**run_row(folder, info, group_by), "path": path, "moves": max(len(path) - 1, 0), "jumps": jumps})
    return pd.DataFrame(rows)


def in_degree(graph) -> np.ndarray:
    """In-degree of every local node (from the reverse CSR when loaded, else counted from the forward edges)."""
    rev = getattr(graph, "rev", None)
    if rev is not None:
        return np.diff(rev[0]).astype(np.int64)
    return np.bincount(graph.fwd[1], minlength=graph.n).astype(np.int64)


class Walker:
    """Random walks on the core graph, all from one start page."""

    def __init__(self, graph, clickstream=None, unlisted: float = 5.0, seed: int = 0):
        self.g, self.cs, self.unlisted = graph, clickstream, unlisted
        self.indptr, self.indices = graph.fwd
        self.deg = np.diff(self.indptr)
        self.pid = graph.attrs["page_id"]
        self.rng = np.random.default_rng(seed)
        self._cs_cache: dict[int, np.ndarray] = {}
        self._indeg = None

    def in_degree(self) -> np.ndarray:
        if self._indeg is None:
            self._indeg = in_degree(self.g)
        return self._indeg

    def walk(self, kind: str, start: int, moves: int, n: int, jump_rate: float = 0.0) -> np.ndarray:
        """(n, moves + 1) local ids."""
        out = np.empty((n, moves + 1), np.int64)
        out[:, 0] = start
        if kind in ("uniform", "jump"):
            cur = np.full(n, start, np.int64)
            for t in range(1, moves + 1):
                dead = self.deg[cur] == 0                # stays put on a page with no out-link
                pos = self.indptr[cur] + (self.rng.random(n) * self.deg[cur]).astype(np.int64)
                nxt = np.where(dead, cur, self.indices[np.minimum(pos, len(self.indices) - 1)])
                if kind == "jump" and jump_rate > 0:
                    j = self.rng.random(n) < jump_rate
                    nxt[j] = self.rng.integers(0, self.g.n, int(j.sum()))
                out[:, t] = cur = nxt
            return out
        for w in range(n):
            cur, seen = start, {start}
            for t in range(1, moves + 1):
                nb = self.g.out(cur)
                if not len(nb):                  # a page with no out-link (only in graphs that are not strongly connected)
                    out[w, t] = cur
                    continue
                if kind == "no_revisit":
                    fresh = nb[~np.isin(nb, list(seen))]
                    cur = int(self.rng.choice(fresh if len(fresh) else nb))
                elif kind == "indegree":
                    wt = self.in_degree()[nb] + 1.0
                    cur = int(nb[min(np.searchsorted(np.cumsum(wt), self.rng.random() * wt.sum()), len(nb) - 1)])
                elif kind == "clickstream":
                    cum = self._cs_cache.get(cur)
                    if cum is None:
                        p, _ = self.cs.probs(int(self.pid[cur]), self.pid[nb], self.unlisted)
                        cum = self._cs_cache[cur] = np.cumsum(p)
                    cur = int(nb[min(np.searchsorted(cum, self.rng.random() * cum[-1]), len(nb) - 1)])
                else:
                    raise ValueError(f"unknown walk baseline {kind!r}; expected one of {BASELINES}")
                seen.add(cur)
                out[w, t] = cur
        return out


def path_stats(V: np.ndarray, hub: np.ndarray, dist: np.ndarray | None, path, sim) -> dict:
    """Statistics of one path. V: (m+1, dim) page vectors; hub: (m+1,) bool; dist: (m+1,) or None; path: page ids."""
    m = len(path) - 1
    out = {"revisit_rate": _revisit_rate(list(path)), "hub_share": float(hub[1:].mean()) if m else np.nan,
           "end_distance": float(dist[-1]) if dist is not None and dist[-1] >= 0 else np.nan}
    if V.shape[1]:
        out["step_similarity"] = float((V[:-1] * V[1:]).sum(1).mean()) if m else np.nan
        out["breadth"] = sim.breadth(V)
        out["drift_from_start"] = float(1 - V[0] @ V[-1]) if m else np.nan
    return out


def _revisit_rate(path) -> float:
    seen, r = {path[0]}, 0
    for p in path[1:]:
        r += p in seen
        seen.add(p)
    return r / (len(path) - 1) if len(path) > 1 else np.nan


def simulate(runs: pd.DataFrame, keys: list[str], graph, sim, clickstream=None, *, baselines=BASELINES,
             per_run: int = 100, unlisted: float = 5.0, distance: bool = False, max_distance: int = 8,
             seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Per run, baseline and statistic: the agent's value, the mean over its walks and (column walk_values, not
    written to CSV) every walk's value. `sim` is the PageVectors option. Two passes: all walks first, then the
    vectors of every page they and the agents visited are prepared at once (embeddings fetch them in batches).
    Returns (table, audit)."""
    walker = Walker(graph, clickstream, unlisted, seed)
    hub_flag = getattr(graph, "hub", None)
    if hub_flag is None:
        hub_flag = graph.attrs["pool"].astype(bool) if "pool" in graph.attrs else np.zeros(graph.n, bool)
    runs = runs.copy()
    runs["start_local"] = [int(graph.locals_of_page_ids([p[0]])[0]) if p else -1 for p in runs.path]
    audit = {"runs": len(runs), "runs_without_moves": int((runs.moves == 0).sum()),
             "runs_start_outside_core": int(((runs.moves > 0) & (runs.start_local < 0)).sum())}
    use = runs[(runs.moves > 0) & (runs.start_local >= 0)]
    rate = {k: (g.jumps.sum() / g.moves.sum()) for k, g in use.groupby(keys)} if len(use) else {}
    plan = []                                           # pass 1: every walk
    for _, r in use.iterrows():
        cond = tuple(r[k] for k in keys)
        jr = float(rate.get(cond if len(keys) > 1 else cond[0], 0.0))
        for b in baselines:
            if (b == "jump" and jr == 0) or (b == "clickstream" and clickstream is None):
                continue
            plan.append((r, b, walker.walk(b, int(r.start_local), int(r.moves), per_run, jr)))
    pids = set(np.concatenate([np.asarray(p, np.int64) for p in use.path]).tolist()) if len(use) else set()
    for _, _, locs in plan:
        pids.update(walker.pid[np.unique(locs)].tolist())
    pid_list = np.array(sorted(pids), np.int64)
    sim.prepare(pid_list)                               # pass 2: vectors for every visited page
    M = sim.vectors(pid_list)
    row_of = lambda ids: np.searchsorted(pid_list, np.asarray(ids, np.int64))
    stats = [s for s in STATS if (s in LINK_STATS or M.shape[1]) and (s != "end_distance" or distance)]
    agent, dists, rows = {}, {}, []
    for r, b, locs in plan:
        start = int(r.start_local)
        if distance and start not in dists:
            dists.clear()   # one start at a time keeps memory at one distance array
            dists[start] = graph.bfs(start, dmax=max_distance)
        d = dists.get(start)
        if r.folder not in agent:                     # keyed by folder: run ids repeat across batches
            apath = np.asarray(r.path, np.int64)
            aloc = graph.locals_of_page_ids(apath)
            agent[r.folder] = path_stats(M[row_of(apath)], np.where(aloc >= 0, hub_flag[np.maximum(aloc, 0)], False),
                                         np.where(aloc >= 0, d[np.maximum(aloc, 0)], -1) if d is not None else None, apath, sim)
        a = agent[r.folder]
        wpids = walker.pid[locs]
        ws = [path_stats(M[row_of(wpids[i])], hub_flag[locs[i]], d[locs[i]] if d is not None else None, locs[i], sim)
              for i in range(len(locs))]
        for s in stats:
            vals = np.array([w[s] for w in ws], float)
            vals = vals[np.isfinite(vals)]
            rows.append({**{k: r[k] for k in keys}, "run_id": r.run_id, "cluster": r.cluster, "baseline": b,
                         "statistic": s, "similarity": sim.name, "moves": int(r.moves), "agent": a[s],
                         "walks": float(vals.mean()) if len(vals) else np.nan, "walk_values": vals})
    return pd.DataFrame(rows), audit


def monte_carlo_p(observed: float, walk_values: list[np.ndarray], n: int, seed: int) -> float:
    """Two-sided p-value of the mean paired difference when the agent is replaced by one of its own walks."""
    vals = [v for v in walk_values if len(v) >= 2]
    if not vals:
        return np.nan
    rng = np.random.default_rng(seed)
    null = np.zeros(n)
    for v in vals:
        j = rng.integers(0, len(v), n)
        null += v[j] - (v.sum() - v[j]) / (len(v) - 1)
    null /= len(vals)
    return float((1 + (np.abs(null) >= abs(observed) - 1e-12).sum()) / (n + 1))


def compare(table: pd.DataFrame, keys: list[str], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per condition, baseline and statistic: agent mean, walk mean, the mean paired difference with a
    cluster-bootstrap 95% interval, and a Monte Carlo p-value against the walks (monte_carlo_p)."""
    out = []
    if table.empty:
        return pd.DataFrame(out)
    for cond, g in table.dropna(subset=["agent", "walks"]).groupby(keys + ["baseline", "statistic"], sort=True):
        g = g.assign(d=g.agent - g.walks)
        cl = g.groupby("cluster").agg(d=("d", "sum"), n=("d", "size"))
        sums = {k: cl[k].to_numpy(float) for k in ("d", "n")}
        ci = cluster_bootstrap(sums, lambda t: {"diff": t["d"] / t["n"]}, n_boot, seed)["diff"]
        out.append({**dict(zip(keys + ["baseline", "statistic"], cond)), "similarity": g.similarity.iloc[0] if "similarity" in g else None,
                    "runs": len(g), "clusters": len(cl),
                    "agent": g.agent.mean(), "walks": g.walks.mean(), "diff": g.d.mean(),
                    "diff_lo": float(ci[0]), "diff_hi": float(ci[1]),
                    "p_walks": monte_carlo_p(g.d.mean(), list(g.walk_values), n_boot, seed)})
    return pd.DataFrame(out)


def write(out: Path, table: pd.DataFrame, summary: pd.DataFrame, audit: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "walks.csv", index=False)
    table.drop(columns=["walk_values"], errors="ignore").to_csv(out / "walks_runs.csv", index=False)
    (out / "walks_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
