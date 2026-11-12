# Graph tools

The hyperlink graph of the English Wikipedia dump (enwiki-20260901) as arrays: build it once from the dump, then
measure it or walk it (BFS distances, reachability, communities). The analysis pipeline's `walks` stage and the
random-walk baselines read the built graph.

- The built files live in `data/graph` by default; set `GAP_ATTRACTOR_GRAPH` to use another folder.
- The scripts need the `graph` extra: `uv run --extra graph python scripts/graph/<script>.py ...`.
  `modularity.py` also needs `python-igraph`, which the extra does not install.
- Dump download and general setup: [docs/reproduce.md](../../docs/reproduce.md).

The graph is **directed**: a link A -> B is not a link B -> A (only about 20% of links are reciprocated). Distances
and reachability follow link direction; `modularity.py` ignores it on purpose.
[docs/analysis.md](../../docs/analysis.md#the-link-graph-is-directed) explains why this matters for the analysis.

## Scripts

| Script | What it does |
|---|---|
| `extract.py` | Parallel pass over the multistream dump: per article its links and list flag, per redirect its target (`art_*.tsv`, `red_*.tsv`) |
| `build_graph.py` | Redirect-merged article graph (`graph.npz`, `keys.txt`, `build_stats.json`) and the hub pool (a boolean per node in `graph.npz`) |
| `make_core.py` | Largest strongly connected component of the pool and of the full graph (`pool_core.npz`, `full_core.npz`, `*_core_nodes.txt`, `*_removed_nodes.tsv` with the reason each node was left out, `*_core_summary.json`), plus `full_nodes.txt` and `full_graph.npz` |
| `stats.py` | Degree, reachability, exact diameter and sampled distance statistics (`stats_<which>.json`) |
| `modularity.py` | Louvain modularity and sampled clustering of a core, on its undirected version (`modularity_<which>.json`) |
| `core_graph.py` | Re-exports the graph library, which lives in the package as `gap_attractor.analysis.graph`: the full core as forward and reverse CSR (cached as `full_core_csr.npz`), BFS in both directions, key and page-id lookups |

Usage, in build order:

```bash
uv run --extra graph python scripts/graph/extract.py <hub_analysis dir> <dump.xml.bz2> <index.txt.bz2> <raw dir> <workers>
uv run --extra graph python scripts/graph/build_graph.py <hub_analysis dir> <index.txt.bz2> <raw dir> <graph dir>
uv run --extra graph python scripts/graph/make_core.py <graph dir> <start_pages dir> <out dir>
uv run --extra graph python scripts/graph/stats.py <graph dir> <start_pages dir> <full|full_core|pool|pool_core> <max_bfs>
uv run --extra graph python scripts/graph/modularity.py full|pool [--seed 0] [--runs 2]
```

- `<dump.xml.bz2>` and `<index.txt.bz2>` are the multistream dump and its index under
  `data/wikipedia/enwiki-20260901`.
- `<start_pages dir>` is `data/start_pages`; `make_core.py` and `stats.py` read `start_set_A_final.tsv` and
  `start_set_B_final.tsv` from it to report how many start pages fall inside each core.
- `<max_bfs>` caps the number of BFS runs spent on the exact diameter.
- `modularity.py` reads and writes the folder named by `GAP_ATTRACTOR_GRAPH`.

Using the built graph from Python:

```python
from gap_attractor.analysis.graph import CoreGraph

g = CoreGraph.load("data/graph")
dist_to_target = g.bfs(g.local_of_key("Iliad"), reverse=True, dmax=8)   # d(v, Iliad) for every core node v
```

## Known limitation: the first two steps need code that is not in this repository

`extract.py` and `build_graph.py` reuse Claire Xu's dump parsing and hub pool (the `hub_analysis` folder: `wikitext`,
`dump_reader`, `build_lookup`, `collect_redirects`, `analyze`, and its `config.yaml`). They take a checkout of that
folder as their first argument and put it on `sys.path`. That folder is not part of this repository, so these two
scripts document how the graph was built but cannot be run from this repository alone. Everything downstream
(`make_core.py`, `stats.py`, `modularity.py`, `gap_attractor.analysis.graph`) needs only the built files. To use
another graph, write it in the same format with `gap_attractor.analysis.graph.save_graph`
([docs/analysis.md](../../docs/analysis.md#another-graph)).

## Measured

Values from `stats_*.json` and `modularity_*.json` of the graph built on 2026-10-08.

| | full core | hub-pool core |
|---|---|---|
| nodes / edges | 6,611,107 / 224,620,274 | 47,447 / 4,014,552 |
| reciprocity (share of links linked back) | 0.20 | 0.24 |
| diameter following links / ignoring direction | 51 / 9 | 7 / 5 |
| Louvain modularity (undirected) | 0.67 | 0.50 |

- Sampled directed distance on the full core: mean 4.8, 90th percentile 6; about 3.5% of pairs are 7 or more apart.
  The diameter (51) comes from a thin tail of far-apart pairs, not from typical distances.
- Both cores are strongly connected by construction: every page can reach every other, and nothing is a dead end.
  Pages outside them are listed with the reason in `*_removed_nodes.tsv`.
