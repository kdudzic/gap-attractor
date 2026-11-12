"""Synthetic pages, topics and agent runs written in the harness log format, for the analysis tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gap_attractor.analysis.topics import TopicMap, UNLABELED, save_topic_map

N_TOPICS = 6


def make_world(n_pages=300, out_degree=20, homophily=0.6, seed=0, multi=True):
    """Pages with 1-2 categories 'Cat i' (topic i), a few unlabeled ones, and links mostly within topic."""
    rng = np.random.default_rng(seed)
    main = rng.integers(0, N_TOPICS, n_pages)
    cats = []
    for p in range(n_pages):
        if p % 50 == 7:
            cats.append(["Living people"])                       # no topic -> unlabeled
            continue
        c = [f"Cat {main[p]}"]
        if multi and rng.random() < 0.4:
            c.append(f"Cat {rng.integers(0, N_TOPICS)}")
        cats.append(c)
    links = []
    for p in range(n_pages):
        same = np.flatnonzero(main == main[p])
        n_same = int(out_degree * homophily)
        nb = set(rng.choice(same, n_same).tolist()) | set(rng.choice(n_pages, out_degree - n_same).tolist())
        nb.discard(p)
        links.append(sorted(nb))
    names = tuple(f"T{i}" for i in range(N_TOPICS)) + (UNLABELED,)
    tm = TopicMap(names, {f"Cat {i}": (i,) for i in range(N_TOPICS)}, {}, frozenset(), np.full(N_TOPICS + 1, 1 / (N_TOPICS + 1)))
    return {"cats": cats, "links": links, "main": main, "tm": tm}


def topic_vec(world, p):
    return world["tm"].chi(world["cats"][p])


def policy_probs(world, policy, page, menu, visited):
    if policy == "uniform":
        return np.full(len(menu), 1 / len(menu))
    C = np.stack([topic_vec(world, j) for j in menu])
    if policy == "novelty":
        n = C.sum(0)
        score = np.where(n > 0, C / np.where(n > 0, n, 1), 0).sum(1)
        return score / score.sum()
    if policy == "clickstream":                               # follows the synthetic human clicks
        c = np.array([world["clicks"].get((page, j), 0) for j in menu], float)
        c[c == 0] = 5
        return c / c.sum()
    if policy == "hub":                                       # strongly favours well-linked pages
        if "indeg" not in world:
            world["indeg"] = np.bincount(np.concatenate([np.asarray(l) for l in world["links"]]), minlength=len(world["links"]))
        w = world["indeg"][np.asarray(menu)].astype(float) ** 3
        return w / w.sum()
    if policy == "attractor":                                 # 30x weight on links into a fixed set of pages
        w = 1 + 30 * np.isin(np.asarray(menu), list(world.get("attractor", range(5)))).astype(float)
        return w / w.sum()
    if policy in ("pingpong", "icl"):                         # goes back to the previous page; icl: more so each bounce
        back = visited[-2] if len(visited) >= 2 else None
        c = 0                                                 # consecutive bounces so far
        while len(visited) >= c + 3 and visited[-1 - c] == visited[-3 - c] and visited[-1 - c] != visited[-2 - c]:
            c += 1
        weight = 12.0 if policy == "pingpong" else 3.0 * 4 ** min(c, 4)
        w = np.array([weight if j == back else 1.0 for j in menu])
        return w / w.sum()
    if policy == "sticky":                                    # 4x weight on links sharing a topic with the page
        share = (C @ topic_vec(world, page) > 0).astype(float)
        w = 1 + 3 * share
        return w / w.sum()
    raise ValueError(policy)


def write_runs(root: Path, world, policy="uniform", n_runs=40, n_steps=15, seed=1, schema="3.1", world_size=None,
               batch="batch_01_synth", extra_steps=False):
    """Simulate runs and write run.json + steps.jsonl. world_size groups consecutive runs into shared worlds."""
    rng = np.random.default_rng(seed)
    n_pages = len(world["cats"])
    folder = Path(root) / batch
    for r in range(n_runs):
        run_id = f"run{r:03d}"
        d = folder / run_id
        d.mkdir(parents=True, exist_ok=True)
        info = {"schema_version": schema, "batch_id": batch, "run_id": run_id, "start_page": "p0", "variant": "a_hub",
                "config": {"provider": {"model": f"synthetic-{policy}"}, "experiment": {"memory": "path_only", "intent": "none"}}}
        if world_size:
            info["world"] = {"world_id": r // world_size + 1, "agent_id": r % world_size + 1, "n_agents": world_size}
        page, visited, recs = int(rng.integers(0, n_pages)), [], []
        if extra_steps:
            recs.append({"step_index": 0, "action": "search_start", "current_page_id": None, "choice_set": [],
                         "search": {"results": [_row(world, page)]}, "parsed_choice": f"p{page}",
                         "resolution": {"resolved_page_id": page}, "invalid_action": {"harness_handling": None}})
        for t in range(1, n_steps + 1):
            visited.append(page)
            menu = world["links"][page]
            rows = [_row(world, j) for j in menu]
            if schema.startswith("2") and len(rows) > 1:
                rows.append(dict(rows[0]))                          # redirect duplicate, as in schema 2.x
            p = policy_probs(world, policy, page, menu, visited)
            nxt = int(menu[rng.choice(len(menu), p=p)])
            handling = "skip" if extra_steps and t == 3 else None
            recs.append({"step_index": t, "action": "follow", "current_page_id": page, "current_page_title": f"p{page}",
                         "semantic_cluster_label": world["cats"][page], "choice_set": rows, "search": None,
                         "parsed_choice": f"p{nxt}", "resolution": {"resolve_status": "success", "resolved_page_id": nxt},
                         "invalid_action": {"is_invalid": handling is not None, "harness_handling": handling}})
            page = nxt
        if extra_steps:
            recs.append({"step_index": n_steps + 1, "action": "follow", "current_page_id": page,
                         "semantic_cluster_label": world["cats"][page], "choice_set": [_row(world, j) for j in world["links"][page]],
                         "parsed_choice": "STOP", "resolution": {"resolved_page_id": None}, "invalid_action": {}})
        (d / "run.json").write_text(json.dumps(info))
        (d / "steps.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    return folder


def _row(world, j):
    return {"page_id": int(j), "title": f"p{j}", "semantic_cluster_label": world["cats"][j]}


def write_resources(root: Path, world) -> Path:
    tm = world["tm"]
    save_topic_map(tm, {"roots_config": [f"T{i}" for i in range(N_TOPICS)]}, Path(root))
    return Path(root)


def add_clicks(world, seed=3):
    """Synthetic human clicks: readers favour low page numbers; about half the links are unlisted (<= 10 clicks)."""
    rng = np.random.default_rng(seed)
    world["clicks"] = {(p, j): int(1000 // (1 + j % 40)) for p, links in enumerate(world["links"]) for j in links
                       if rng.random() < 0.5 and 1000 // (1 + j % 40) > 10}
    return world


def write_clickstream(root: Path, world) -> None:
    from gap_attractor.analysis.clickstream import Clickstream  # noqa: F401  (format check happens on load)
    pairs = sorted(world["clicks"].items())
    src = np.array([a for (a, _), _ in pairs], np.int64)
    dst = np.array([b for (_, b), _ in pairs], np.int64)
    n = np.array([c for _, c in pairs], np.int64)
    sources, starts = np.unique(src, return_index=True)
    np.savez(Path(root) / "clickstream.npz", source=sources, indptr=np.append(starts, len(src)), target=dst, clicks=n)


def write_page_topics(root: Path, world) -> None:
    tm, rows = world["tm"], [topic_vec(world, p) for p in range(len(world["cats"]))]
    topic = [np.flatnonzero(v) for v in rows]
    np.savez(Path(root) / "page_topics.npz", page_id=np.arange(len(rows), dtype=np.int64),
             indptr=np.concatenate([[0], np.cumsum([len(t) for t in topic])]).astype(np.int64),
             topic=np.concatenate(topic).astype(np.int8), weight=np.concatenate([v[t] for v, t in zip(rows, topic)]).astype(np.float32),
             names=np.array(tm.names), weighting=np.array("multiplicity"), topic_hash=np.array(tm.hash()))


def write_graph(root: Path, world) -> Path:
    """The world's links in the scripts/graph/ format (local id = global id = page id); hubs = the 30 most linked."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    n = len(world["links"])
    src = np.repeat(np.arange(n), [len(l) for l in world["links"]])
    dst = np.concatenate([np.asarray(l) for l in world["links"]])
    indeg = np.bincount(dst, minlength=n)
    from gap_attractor.analysis.graph import csr
    fwd, rev = csr(src, dst, n), csr(dst, src, n)
    np.savez(root / "full_core_csr.npz", node_ids=np.arange(n, dtype=np.int64), fwd_indptr=fwd[0], fwd_indices=fwd[1],
             rev_indptr=rev[0], rev_indices=rev[1])
    pool = np.zeros(n, bool); pool[np.argsort(-indeg)[:30]] = True
    np.savez(root / "full_graph.npz", page_id=np.arange(n, dtype=np.int64), out_m=np.diff(fwd[0]).astype(np.int32),
             in_cont=indeg.astype(np.int64), cat_lists=np.zeros(n, np.int8), title_junk=np.zeros(n, np.int8), pool=pool)
    return root


def analysis_cfg(res: Path, overrides=()):
    """The analysis config with the synthetic resources (configs/analysis.yaml + overrides)."""
    from hydra import compose, initialize_config_dir
    from gap_attractor.analysis.cli import CONFIG_DIR
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="analysis", overrides=[f"resources={res}", *overrides])


def category_vectors(res: Path):
    from gap_attractor.analysis.similarity import make_similarity
    return make_similarity(analysis_cfg(res))


def write_leads(root: Path, world) -> Path:
    """leads.sqlite whose text names each page's topics, so a bag-of-words embedder sees topic similarity."""
    import sqlite3
    rng = np.random.default_rng(5)
    con = sqlite3.connect(Path(root) / "leads.sqlite")
    con.execute("CREATE TABLE pages (page_id INTEGER PRIMARY KEY, title TEXT NOT NULL, lead TEXT NOT NULL)")
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    rows = []
    for p, cats in enumerate(world["cats"]):
        words = " ".join(f"subject{c.split()[-1]} " * 4 for c in cats if c.startswith("Cat "))
        filler = " ".join(f"w{int(x)}" for x in rng.integers(0, 500, 6))
        rows.append((p, f"p{p}", f"{words} {filler}".strip()))
    con.executemany("INSERT INTO pages VALUES (?, ?, ?)", rows)
    con.executemany("INSERT INTO meta VALUES (?, ?)", [("built_at", "synthetic"), ("dump", "synthetic")])
    con.commit(); con.close()
    return Path(root) / "leads.sqlite"
