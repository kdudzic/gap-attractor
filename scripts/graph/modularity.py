"""Louvain modularity and sampled clustering of the core graphs, measured the same way as the other surveyed graphs:
undirected (edges collapsed) version of the largest strongly connected component, igraph community_multilevel.

usage: python3 scripts/graph/modularity.py full|pool [--seed 0] [--runs 2]   (needs python-igraph and numpy)
Writes <graph dir>/modularity_<which>.json.
"""
import argparse
import json
import random
import os
import time
from pathlib import Path

import igraph as ig
import numpy as np

GRAPH_DIR = Path(os.environ.get("GAP_ATTRACTOR_GRAPH", "data/graph"))


def edges(which):
    if which == "full":  # cached CSR from core_graph.py, local ids
        z = np.load(GRAPH_DIR / "full_core_csr.npz")
        indptr, indices = z["fwd_indptr"], z["fwd_indices"]
        src = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
        return len(indptr) - 1, src, indices.astype(np.int64)
    z = np.load(GRAPH_DIR / "pool_core.npz")
    nodes = z["node_ids"]
    local = {int(g): i for i, g in enumerate(nodes)}
    src = np.array([local[int(x)] for x in z["src"]], np.int64)
    dst = np.array([local[int(x)] for x in z["dst"]], np.int64)
    return len(nodes), src, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["full", "pool"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    n, src, dst = edges(args.which)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    undirected = np.unique(lo * n + hi)
    del src, dst, lo, hi
    print(f"{args.which}: {n:,} nodes, {len(undirected):,} undirected edges ({time.time() - t0:.0f}s)", flush=True)
    g = ig.Graph(n=n, edges=np.column_stack((undirected // n, undirected % n)), directed=False)
    del undirected
    out = {"graph": args.which, "nodes": n, "undirected_edges": g.ecount(), "undir_edges_per_node": round(g.ecount() / n, 2), "runs": []}
    for run in range(args.runs):
        random.seed(args.seed + run); ig.set_random_number_generator(random)
        t = time.time()
        cl = g.community_multilevel()
        sizes = sorted(cl.sizes(), reverse=True)
        out["runs"].append({"seed": args.seed + run, "modularity": round(cl.modularity, 4), "communities": len(cl),
                            "largest_communities": sizes[:10], "seconds": round(time.time() - t)})
        print(json.dumps(out["runs"][-1]), flush=True)
    random.seed(args.seed)
    sample = random.sample(range(n), min(5000, n))
    out["avg_clustering_undir_sampled"] = round(float(np.mean(g.transitivity_local_undirected(vertices=sample, mode="zero"))), 3)
    out["louvain_modularity"] = max(r["modularity"] for r in out["runs"])
    (GRAPH_DIR / f"modularity_{args.which}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}), f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
