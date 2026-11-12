"""Graph statistics for the full article graph and Claire's pool subgraph.

usage: stats.py <graph dir> <start_pages dir> <which: pool|full> <max_bfs>
"""
import json
import sys
import time

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

GD, SP, WHICH, MAX_BFS = (sys.argv + [""]*4)[1:5]
rng = np.random.default_rng(0)
T0 = time.time()


def log(*a):
    print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)


# --- BFS ------------------------------------------------------------------------
def bfs(indptr, indices, s, chunk=20_000_000):
    n = len(indptr) - 1
    dist = np.full(n, -1, np.int32)
    mark = np.zeros(n, bool)
    dist[s] = 0
    frontier = np.array([s], np.int64)
    d = 0
    while frontier.size:
        starts, lens = indptr[frontier], indptr[frontier + 1] - indptr[frontier]
        cl = np.cumsum(lens)
        lo = 0
        while lo < frontier.size:            # split frontier so the gather stays < chunk edges
            base = cl[lo - 1] if lo else 0
            hi = int(np.searchsorted(cl, base + chunk, side="right"))
            hi = max(hi, lo + 1)
            st, ln = starts[lo:hi], lens[lo:hi]
            tot = int(ln.sum())
            if tot:
                offs = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(tot)
                nb = indices[offs]
                mark[nb[dist[nb] < 0]] = True
            lo = hi
        frontier = np.flatnonzero(mark)
        mark[frontier] = False
        d += 1
        dist[frontier] = d
    return dist


def describe(x):
    x = np.asarray(x)
    q = np.percentile(x, [50, 90, 99]) if x.size else [0, 0, 0]
    return {"mean": round(float(x.mean()), 2), "median": float(q[0]), "p90": float(q[1]),
            "p99": float(q[2]), "max": int(x.max())}


def bounding(fwd, bwd, max_bfs, deg, directed=True):
    """Exact out-eccentricity diameter & radius of a strongly connected graph
    (Takes & Kosters BoundingDiameters, directed variant). Returns bounds if capped."""
    n = len(fwd[0]) - 1
    lo = np.zeros(n, np.int32)
    hi = np.full(n, np.iinfo(np.int32).max // 2, np.int32)
    cand = np.ones(n, bool)
    D_lo, R_hi, n_bfs, pick_hi = 0, np.iinfo(np.int32).max, 0, True
    center = None
    w = int(np.argmax(deg))
    while n_bfs < max_bfs:
        df = bfs(*fwd, w); n_bfs += 1
        if directed:
            db = bfs(*bwd, w); n_bfs += 1
        else:
            db = df
        e = int(df.max())
        np.maximum(lo, np.maximum(db, e - df), out=lo)
        np.minimum(hi, db + e, out=hi)
        lo[w] = hi[w] = e
        D_lo = max(D_lo, int(lo.max()))
        if int(hi.min()) < R_hi:
            R_hi = int(hi.min()); center = int(np.argmin(hi))
        # a node is decided once it can no longer change the diameter or the radius
        cand &= ~((lo == hi) | ((hi <= D_lo) & (lo >= R_hi)))
        D_hi = int(hi[cand].max()) if cand.any() else D_lo
        R_lo = int(lo[cand].min()) if cand.any() else R_hi
        D_hi, R_lo = max(D_hi, D_lo), min(R_lo, R_hi)
        if n_bfs % 10 < 2:
            log(f"   bounding bfs={n_bfs} diam in [{D_lo},{D_hi}] radius in [{R_lo},{R_hi}] cand={int(cand.sum()):,}")
        if D_lo == D_hi and R_lo == R_hi:
            break
        idx = np.flatnonzero(cand)
        if pick_hi:
            m = hi[idx]; sel = idx[m == m.max()]
        else:
            m = lo[idx]; sel = idx[m == m.min()]
        w = int(sel[np.argmax(deg[sel])])
        pick_hi = not pick_hi
    return {"diameter": [D_lo, D_hi], "radius": [R_lo, R_hi], "exact": D_lo == D_hi and R_lo == R_hi,
            "n_bfs": n_bfs, "center_idx": center}


def dist_sample(fwd, sources):
    hist, reach, ecc = {}, [], []
    for s in sources:
        d = bfs(*fwd, int(s))
        f = d[d > 0]
        reach.append(int(f.size)); ecc.append(int(f.max()) if f.size else 0)
        for k, v in zip(*np.unique(f, return_counts=True)):
            hist[int(k)] = hist.get(int(k), 0) + int(v)
    tot = sum(hist.values())
    ks = sorted(hist)
    cum, eff = 0, None
    for k in ks:
        cum += hist[k]
        if eff is None and cum >= 0.9 * tot:
            eff = k
    mean = sum(k * v for k, v in hist.items()) / tot if tot else 0
    return {"n_sources": len(sources), "mean_distance": round(mean, 3), "effective_diameter_p90": eff,
            "hist_frac": {k: round(hist[k] / tot, 5) for k in ks},
            "reach": describe(reach), "ecc_from_source": describe(ecc)}


def analyse(name, src, dst, n, keys, start_idx, max_bfs, undirected_only=False):
    R = {"name": name}
    log(f"== {name}: building CSR")
    A = sparse.csr_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n))
    A.sum_duplicates()
    m = A.nnz
    outd = np.diff(A.indptr); ind = np.bincount(A.indices, minlength=n)
    R["nodes"], R["edges"] = n, int(m)
    R["density"] = m / (n * (n - 1))
    R["out_degree"], R["in_degree"] = describe(outd), describe(ind)
    R["sinks_out0"] = int((outd == 0).sum()); R["sources_in0"] = int((ind == 0).sum())
    R["isolated"] = int(((outd == 0) & (ind == 0)).sum())
    code = A.indptr  # noqa
    rows = np.repeat(np.arange(n, dtype=np.int64), outd)
    fw = rows * n + A.indices; bw = A.indices.astype(np.int64) * n + rows
    fw.sort()
    pos = np.searchsorted(fw, bw); pos[pos >= len(fw)] = 0
    R["reciprocity"] = round(float((fw[pos] == bw).mean()), 4)
    del fw, bw, pos, rows
    log(f"   n={n:,} m={m:,} reciprocity={R['reciprocity']}")

    # weak components
    nw, wl = csgraph.connected_components(A, directed=True, connection="weak")
    ws = np.bincount(wl)
    R["wcc"] = {"count": int(nw), "largest": int(ws.max()), "largest_frac": round(ws.max() / n, 5),
                "singletons": int((ws == 1).sum())}
    # strong components + bow-tie
    ns_, sl = csgraph.connected_components(A, directed=True, connection="strong")
    ss = np.bincount(sl); g = int(np.argmax(ss)); inS = sl == g
    rep = int(np.flatnonzero(inS)[0])
    AT = A.T.tocsr()
    fwd_reach = bfs(A.indptr, A.indices, rep) >= 0
    bwd_reach = bfs(AT.indptr, AT.indices, rep) >= 0
    out_set, in_set = fwd_reach & ~inS, bwd_reach & ~inS
    R["scc"] = {"count": int(ns_), "largest": int(ss.max()), "largest_frac": round(ss.max() / n, 5),
                "second_largest": int(np.sort(ss)[-2]) if ns_ > 1 else 0, "singletons": int((ss == 1).sum())}
    R["bowtie"] = {"core_scc": int(inS.sum()), "IN": int(in_set.sum()), "OUT": int(out_set.sum()),
                   "other(tendrils/tubes/disconnected)": int(n - inS.sum() - in_set.sum() - out_set.sum())}
    R["reachability"] = {
        "frac_nodes_that_can_reach_core": round(float(bwd_reach.mean()), 5),
        "frac_nodes_reachable_from_core": round(float(fwd_reach.mean()), 5),
        "reach_set_of_any_core_node": int(fwd_reach.sum()),
    }
    if start_idx is not None:
        for lab, idx in start_idx.items():
            R["reachability"][f"start_{lab}_in_core"] = f"{int(inS[idx].sum())}/{len(idx)}"
    log(f"   wcc={R['wcc']} scc={R['scc']} bowtie={R['bowtie']}")

    # giant SCC: exact directed diameter / radius
    if not undirected_only:
        core = np.flatnonzero(inS)
        C = A[core][:, core].tocsr(); CT = C.T.tocsr()
        log(f"   core subgraph {len(core):,} nodes {C.nnz:,} edges; bounding diameters")
        b = bounding((C.indptr, C.indices), (CT.indptr, CT.indices), max_bfs,
                     np.diff(C.indptr) + np.diff(CT.indptr), directed=True)
        b["center"] = keys[core[b.pop("center_idx")]]
        R["core_directed"] = b
        log(f"   core directed {b}")
        srcs = rng.choice(len(core), size=min(100, len(core)), replace=False)
        R["core_distance_sample_random"] = dist_sample((C.indptr, C.indices), srcs)
        log(f"   distance sample: mean {R['core_distance_sample_random']['mean_distance']}")
        del C, CT
        if start_idx is not None:
            for lab, idx in start_idx.items():
                sm = rng.choice(idx, size=min(100, len(idx)), replace=False)
                R[f"start_{lab}_distance_sample"] = dist_sample((A.indptr, A.indices), sm)
                log(f"   start {lab}: {R[f'start_{lab}_distance_sample']['ecc_from_source']}")

    # undirected view (largest WCC)
    U = (A + AT).tocsr(); U.data[:] = 1
    big = np.flatnonzero(wl == int(np.argmax(ws)))
    UC = U[big][:, big].tocsr()
    log(f"   undirected largest WCC {len(big):,} nodes {UC.nnz//2:,} edges; bounding")
    b = bounding((UC.indptr, UC.indices), None, max_bfs, np.diff(UC.indptr), directed=False)
    b["center"] = keys[big[b.pop("center_idx")]]
    R["undirected_largest_wcc"] = b
    R["undirected_edges"] = int(U.nnz // 2)
    srcs = rng.choice(len(big), size=min(100, len(big)), replace=False)
    R["undirected_distance_sample"] = dist_sample((UC.indptr, UC.indices), srcs)
    log(f"   undirected {b}")
    return R


def main():
    z = np.load(f"{GD}/graph.npz")
    keys = open(f"{GD}/keys.txt", encoding="utf-8").read().split("\n")[:-1]
    kid = {k: i for i, k in enumerate(keys)}
    starts = {}
    for lab, fn in (("A", "start_set_A_final.tsv"), ("B", "start_set_B_final.tsv")):
        ks = [l.split("\t")[2] for l in open(f"{SP}/{fn}", encoding="utf-8").read().split("\n")[1:] if l]
        starts[lab] = np.array([kid[k] for k in ks if k in kid])
        log(f"start set {lab}: {len(starts[lab])}/{len(ks)} found in graph")
    src, dst = z["src"], z["dst"]
    if WHICH == "full":
        R = analyse("full article graph (redirect-merged)", src, dst, len(keys), keys, starts, int(MAX_BFS))
    elif WHICH == "full_core":     # restrict the full graph to its largest SCC
        n = len(keys)
        A = sparse.csr_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(n, n))
        _, lab = csgraph.connected_components(A, directed=True, connection="strong")
        del A
        cidx = np.flatnonzero(lab == np.bincount(lab).argmax())
        remap = np.full(n, -1, np.int64); remap[cidx] = np.arange(len(cidx))
        keep = (remap[src] >= 0) & (remap[dst] >= 0)
        ckeys = [keys[i] for i in cidx]
        st = {lab_: remap[idx][remap[idx] >= 0] for lab_, idx in starts.items()}
        for lab_ in st:
            log(f"start set {lab_}: {len(st[lab_])}/{len(starts[lab_])} in core")
        R = analyse("full article graph, largest SCC", remap[src[keep]], remap[dst[keep]], len(cidx), ckeys, st, int(MAX_BFS))
    else:
        pool = z["pool"].copy()
        if WHICH == "pool_core":   # restrict the pool to its largest SCC
            pidx = np.flatnonzero(pool); r = np.full(len(keys), -1); r[pidx] = np.arange(len(pidx))
            k = pool[src] & pool[dst]
            P = sparse.csr_matrix((np.ones(k.sum(), np.int8), (r[src[k]], r[dst[k]])), shape=(len(pidx),) * 2)
            _, lab = csgraph.connected_components(P, directed=True, connection="strong")
            pool[pidx[lab != np.bincount(lab).argmax()]] = False
        pidx = np.flatnonzero(pool)
        remap = np.full(len(keys), -1, np.int64); remap[pidx] = np.arange(len(pidx))
        keep = pool[src] & pool[dst]
        pkeys = [keys[i] for i in pidx]
        st = {lab: remap[idx][remap[idx] >= 0] for lab, idx in starts.items()}
        for lab in st:
            log(f"start set {lab}: {len(st[lab])}/{len(starts[lab])} in pool")
        R = analyse(f"Claire pool subgraph ({WHICH})", remap[src[keep]], remap[dst[keep]], len(pidx), pkeys, st, int(MAX_BFS))
    json.dump(R, open(f"{GD}/stats_{WHICH}.json", "w"), indent=1, default=int)
    log("DONE")


if __name__ == "__main__":
    main()
