"""Restrict Claire's pool subgraph (and the full graph) to the largest SCC and save dumps.

usage: make_core.py <graph dir> <start_pages dir> <out dir>
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

GD, SP, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
z = np.load(f"{GD}/graph.npz")
keys = np.array(open(f"{GD}/keys.txt", encoding="utf-8").read().split("\n")[:-1], dtype=object)
src, dst, pool = z["src"], z["dst"], z["pool"]
N = len(keys)
starts = {}
for lab, fn in (("A", "start_set_A_final.tsv"), ("B", "start_set_B_final.tsv")):
    starts[lab] = {l.split("\t")[2] for l in open(f"{SP}/{fn}", encoding="utf-8").read().split("\n")[1:] if l}


def core_of(nodes, s, d, label):
    """nodes: global ids; s,d: global edge endpoints (both in nodes). Returns core ids, summary."""
    n = len(nodes)
    remap = np.full(N, -1, np.int64); remap[nodes] = np.arange(n)
    ls, ld = remap[s], remap[d]
    A = sparse.csr_matrix((np.ones(len(ls), np.int8), (ls, ld)), shape=(n, n))
    n_comp, lab = connected_components(A, directed=True, connection="strong")
    core = lab == np.bincount(lab).argmax()
    # classify removed nodes relative to the core
    _, wl = connected_components(A, directed=True, connection="weak")
    outd, ind = np.diff(A.indptr), np.bincount(A.indices, minlength=n)
    has_in_core = np.zeros(n, bool); has_out_core = np.zeros(n, bool)
    has_out_core[ls[core[ld]]] = True       # links into the core
    has_in_core[ld[core[ls]]] = True        # linked from the core
    ek = core[ls] & core[ld]
    summ = {
        "graph": label, "nodes_before": n, "edges_before": int(len(ls)), "scc_count": int(n_comp),
        "nodes_core": int(core.sum()), "edges_core": int(ek.sum()),
        "nodes_removed": int(n - core.sum()), "nodes_removed_frac": round(float(1 - core.mean()), 6),
        "edges_removed": int(len(ls) - ek.sum()), "edges_removed_frac": round(float(1 - ek.mean()), 6),
        "removed_out_degree0": int((~core & (outd == 0)).sum()),
        "removed_in_degree0": int((~core & (ind == 0)).sum()),
        "removed_isolated": int((~core & (outd == 0) & (ind == 0)).sum()),
        "start_pages_removed": {k: sorted(set(keys[nodes[~core]]) & v) for k, v in starts.items()},
    }
    rem = np.flatnonzero(~core)
    reason = np.where((outd[rem] == 0) & (ind[rem] == 0), "isolated",
             np.where(has_out_core[rem] & ~has_in_core[rem], "IN (links to core, never linked back)",
             np.where(has_in_core[rem] & ~has_out_core[rem], "OUT (linked from core, no way back)",
                      "other (tendril/small SCC)")))
    return nodes[core], ek, summ, list(zip(keys[nodes[rem]], outd[rem], ind[rem], reason))


def save(label, gnodes, s, d, summ, removed, with_tsv):
    np.savez_compressed(f"{OUT}/{label}_core.npz", src=s, dst=d, node_ids=gnodes)
    with open(f"{OUT}/{label}_core_nodes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(keys[gnodes]) + "\n")
    if with_tsv:
        with open(f"{OUT}/{label}_core_edges.tsv", "w", encoding="utf-8") as f:
            f.write("src\tdst\n")
            for a, b in zip(keys[s], keys[d]):
                f.write(f"{a}\t{b}\n")
    with open(f"{OUT}/{label}_removed_nodes.tsv", "w", encoding="utf-8") as f:
        f.write("key\tout_deg\tin_deg\treason\n")
        for r in removed:
            f.write("\t".join(map(str, r)) + "\n")
    from collections import Counter
    summ["removed_by_reason"] = dict(Counter(r[3] for r in removed))
    json.dump(summ, open(f"{OUT}/{label}_core_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


# pool (Claire's pruned graph)
pidx = np.flatnonzero(pool)
pk = pool[src] & pool[dst]
core_nodes, ek, summ, removed = core_of(pidx, src[pk], dst[pk], "pool")
save("pool", core_nodes, src[pk][ek], dst[pk][ek], summ, removed, with_tsv=True)

# full graph, for comparison
core_nodes, ek, summ, removed = core_of(np.arange(N), src, dst, "full")
summ["pool_nodes_outside_full_core"] = int(pool.sum() - pool[core_nodes].sum())
save("full", core_nodes, src[ek], dst[ek], summ, removed, with_tsv=False)

# full constructed graph dump (nodes = all articles, global ids = line numbers of keys.txt)
shutil.copy(f"{GD}/keys.txt", f"{OUT}/full_nodes.txt")
np.savez_compressed(f"{OUT}/full_graph.npz", **{k: z[k] for k in z.files})
shutil.copy(f"{GD}/build_stats.json", f"{OUT}/build_stats.json")
print("DONE", flush=True)
