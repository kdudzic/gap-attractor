"""The redirect-merged article graph's strongly connected core as CSR arrays, with BFS in both directions.

Built by scripts/graph/extract.py -> build_graph.py -> make_core.py (scripts/graph/README.md). Global node ids are line numbers of
full_nodes.txt; the core lists its global ids in full_core.npz["node_ids"]. Everything here works in core-local
ids 0..n-1 (positions in node_ids). Used by the walks stage of gap-attractor-analyze and by scripts/graph/.

    g = CoreGraph.load("data/graph")
    dist_to_t = g.bfs(g.local_of_key("Iliad"), reverse=True, dmax=8)   # d(v, Iliad) for every core node v
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

GRAPH_DIR = os.environ.get("GAP_ATTRACTOR_GRAPH", "data/graph")   # built 2026-10-08


def log(*parts, t0=time.time()):
    print(f"[{(time.time() - t0) / 60:5.1f}m]", *parts, flush=True)


def csr(src, dst, n):
    """(indptr, indices) of the edges src -> dst, sorted by src (counting sort, no scipy needed)."""
    order = np.argsort(src, kind="stable")
    indices = dst[order].astype(np.int32)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, indices


def bfs(indptr, indices, sources, dmax=None, chunk=20_000_000):
    """Distances from `sources` (int or array) to every node; -1 = not reached (or farther than dmax).
    Level-synchronous over the CSR, the frontier gathered in chunks so memory stays bounded."""
    n = len(indptr) - 1
    dist = np.full(n, -1, np.int16)
    frontier = np.unique(np.atleast_1d(np.asarray(sources, np.int64)))
    dist[frontier] = 0
    mark, d = np.zeros(n, bool), 0
    while frontier.size and (dmax is None or d < dmax):
        starts, lens = indptr[frontier], indptr[frontier + 1] - indptr[frontier]
        cl, lo = np.cumsum(lens), 0
        while lo < frontier.size:
            base = cl[lo - 1] if lo else 0
            hi = max(int(np.searchsorted(cl, base + chunk, side="right")), lo + 1)
            st, ln = starts[lo:hi], lens[lo:hi]
            total = int(ln.sum())
            if total:
                offsets = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(total)
                neighbours = indices[offsets]
                mark[neighbours[dist[neighbours] < 0]] = True
            lo = hi
        frontier = np.flatnonzero(mark)
        mark[frontier] = False
        d += 1
        dist[frontier] = d
    return dist


class CoreGraph:
    """Core SCC in local ids: forward and reverse CSR, keys, page ids and the per-node flags of full_graph.npz."""

    def __init__(self, graph_dir, fwd, rev, node_ids, keys, attrs):
        self.graph_dir, self.fwd, self.rev, self.node_ids, self.keys, self.attrs = Path(graph_dir), fwd, rev, node_ids, keys, attrs
        self.n = len(node_ids)
        self._local_by_key = None
        self._local_by_page_id = None

    @classmethod
    def load(cls, graph_dir=GRAPH_DIR, cache=None, reverse=True, keys=True):
        """Load full_core.npz as CSR. The CSR arrays are cached in `cache` (default <graph_dir>/full_core_csr.npz).
        reverse=False skips the reverse CSR and keys=False the page titles (the walks stage needs neither)."""
        graph_dir = Path(graph_dir)
        cache = Path(cache) if cache else graph_dir / "full_core_csr.npz"
        if cache.exists():
            log(f"loading {cache}")
            z = np.load(cache)
            node_ids, fwd = z["node_ids"], (z["fwd_indptr"], z["fwd_indices"])
            rev = (z["rev_indptr"], z["rev_indices"]) if reverse else None
        else:
            log(f"building CSR from {graph_dir / 'full_core.npz'} (once; cached at {cache})")
            z = np.load(graph_dir / "full_core.npz")
            node_ids = z["node_ids"]
            local = np.full(int(node_ids.max()) + 1, -1, np.int32)
            local[node_ids] = np.arange(len(node_ids), dtype=np.int32)
            src, dst = local[z["src"]], local[z["dst"]]
            if (src < 0).any() or (dst < 0).any():
                raise ValueError("full_core.npz has edges leaving the core")
            fwd, rev = csr(src, dst, len(node_ids)), csr(dst, src, len(node_ids))
            del src, dst
            np.savez(cache, node_ids=node_ids, fwd_indptr=fwd[0], fwd_indices=fwd[1], rev_indptr=rev[0], rev_indices=rev[1])
        log("loading node attributes" + (" and keys" if keys else ""))
        if keys:
            all_keys = (graph_dir / "full_nodes.txt").read_text(encoding="utf-8").split("\n")
            keys = np.array([all_keys[i] for i in node_ids], dtype=object)
        else:
            keys = None
        full = np.load(graph_dir / "full_graph.npz")
        attrs = {name: full[name][node_ids] for name in ("page_id", "out_m", "in_cont", "cat_lists", "title_junk", "pool")
                 if name in full.files}      # only page_id is required (save_graph writes page_id and pool)
        log(f"core: {len(node_ids):,} nodes, {len(fwd[1]):,} edges")
        return cls(graph_dir, fwd, rev, node_ids, keys, attrs)

    def bfs(self, sources, reverse=False, dmax=None):
        """reverse=False: d(source, v); reverse=True: d(v, source) for every v."""
        if reverse and self.rev is None:
            raise ValueError("loaded with reverse=False")
        indptr, indices = self.rev if reverse else self.fwd
        return bfs(indptr, indices, sources, dmax)

    def local_of_key(self, key):
        if self._local_by_key is None:
            self._local_by_key = {k: i for i, k in enumerate(self.keys)}
        return self._local_by_key.get(key)

    def local_of_page_id(self, page_id):
        if self._local_by_page_id is None:
            self._local_by_page_id = {int(p): i for i, p in enumerate(self.attrs["page_id"])}
        return self._local_by_page_id.get(int(page_id))

    def locals_of_page_ids(self, page_ids) -> np.ndarray:
        """Vectorised local_of_page_id: -1 where a page is not in the core."""
        pid = self.attrs["page_id"]
        if not hasattr(self, "_pid_order"):
            self._pid_order = np.argsort(pid, kind="stable")
        q = np.asarray(page_ids, np.int64)
        pos = np.clip(np.searchsorted(pid[self._pid_order], q), 0, len(pid) - 1)
        loc = self._pid_order[pos]
        return np.where(pid[loc] == q, loc, -1)

    def out(self, local) -> np.ndarray:
        """Out-neighbours (local ids) of one node."""
        indptr, indices = self.fwd
        return indices[indptr[local]:indptr[local + 1]]

    def title(self, local):
        return str(self.keys[local]).replace("_", " ")

    def plain(self, min_out=5):
        """Nodes that make reasonable start pages / targets: not a list, not a junk title, enough out-links."""
        return (self.attrs["cat_lists"] == 0) & (self.attrs["title_junk"] == 0) & (self.attrs["out_m"] >= min_out)


GRAPHS = None


def _graphs():
    global GRAPHS
    if GRAPHS is None:
        from gap_attractor.registry import Registry
        GRAPHS = Registry("graph")
        GRAPHS.register("core_npz")(load_core_npz)
    return GRAPHS


def load_core_npz(cfg) -> CoreGraph:
    """A graph folder in the format of scripts/graph/ (full_core_csr.npz + full_graph.npz), e.g. the enwiki core,
    or one written by save_graph. Hubs: cfg.hubs = pool (the folder's `pool` flag) or top_in_degree (the
    cfg.hub_top_n most linked-to pages)."""
    g = CoreGraph.load(str(cfg.dir), reverse=False, keys=False)
    hubs = str(cfg.get("hubs", "pool"))
    if hubs == "pool" and "pool" in g.attrs:
        g.hub = g.attrs["pool"].astype(bool)
    elif hubs in ("pool", "top_in_degree"):
        indeg = np.bincount(g.fwd[1], minlength=g.n)
        g.hub = np.zeros(g.n, bool)
        g.hub[np.argsort(-indeg, kind="stable")[:int(cfg.get("hub_top_n", 1000))]] = True
    else:
        raise ValueError(f"graph.hubs must be pool or top_in_degree, not {hubs!r}")
    return g


def load_graph(cfg) -> CoreGraph:
    """The graph named by configs/analysis/graph/<name>.yaml: `kind` is a registered loader (core_npz) or a dotted
    path to factory(cfg) -> an object with fwd (indptr, indices), n, attrs["page_id"], hub (bool per node), out(),
    bfs() and locals_of_page_ids(), like CoreGraph."""
    return _graphs().get(str(cfg.kind))(cfg)


def save_graph(folder, src_page_ids, dst_page_ids, hub_page_ids=()) -> Path:
    """Write any directed graph, given as page-id edges, in the core_npz format, so the walks stage can use it
    (configs/analysis/graph/: kind core_npz, dir <folder>). Walks follow out-links, so give a strongly connected
    graph (or its largest strongly connected component): a walk on a page with no out-link stays there."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    src, dst = np.asarray(src_page_ids, np.int64), np.asarray(dst_page_ids, np.int64)
    page_id = np.unique(np.concatenate([src, dst]))
    s, d = np.searchsorted(page_id, src), np.searchsorted(page_id, dst)
    fwd, rev = csr(s, d, len(page_id)), csr(d, s, len(page_id))
    np.savez(folder / "full_core_csr.npz", node_ids=np.arange(len(page_id), dtype=np.int64), fwd_indptr=fwd[0],
             fwd_indices=fwd[1], rev_indptr=rev[0], rev_indices=rev[1])
    np.savez(folder / "full_graph.npz", page_id=page_id, pool=np.isin(page_id, np.asarray(hub_page_ids, np.int64)))
    return folder
