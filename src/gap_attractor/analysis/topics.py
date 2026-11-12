"""Category -> topic rollup and per-page topic membership vectors (chi).

The rollup is Claire Xu's nearest-root walk (origin/claire/hub-analysis, hub_analysis/rollup_categories.py):
walk the category graph down from a fixed list of root categories; a category takes the roots that reach
it at the smallest depth (ties kept). Changes here: category redirects are followed, hidden/maintenance
categories are dropped, and a page votes with all of its categories (her enrichment kept only the first 8).
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNLABELED = "unlabeled"


def norm_category(name: str) -> str:
    """'category_name' / 'Category:category name' -> 'Category name' (MediaWiki capitalises the first letter)."""
    name = " ".join((name or "").replace("_", " ").split())
    if name.lower().startswith("category:"):
        name = name.split(":", 1)[1].strip()
    return name[:1].upper() + name[1:]


def resolve_category(name: str, redirects: dict[str, str], max_hops: int = 5) -> str:
    for _ in range(max_hops):
        if name not in redirects:
            break
        name = redirects[name]
    return name


def rollup(parents: dict[str, list[str]], roots: list[str], max_depth: int = 7) -> dict[str, tuple[int, frozenset]]:
    """Breadth-first walk from all roots at once over parent -> child edges.

    `parents` maps a category to its parent categories (already normalised and redirect-resolved).
    Returns category -> (depth, roots reaching it at that depth). Roots map to themselves at depth 0."""
    children: dict[str, list[str]] = {}
    for child, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(child)
    best: dict[str, int] = {}
    topics: dict[str, set] = {}
    q = deque()
    for r in roots:
        best[r] = 0; topics[r] = {r}; q.append((r, 0, r))
    while q:
        cat, d, root = q.popleft()
        if d >= max_depth:
            continue
        for ch in children.get(cat, ()):
            nd = d + 1
            if ch not in best or nd < best[ch]:
                best[ch] = nd; topics[ch] = {root}; q.append((ch, nd, root))
            elif nd == best[ch] and root not in topics[ch]:
                topics[ch].add(root); q.append((ch, nd, root))
    return {c: (best[c], frozenset(t)) for c, t in topics.items()}


@dataclass(frozen=True)
class TopicMap:
    """Topic names (fixed order, `unlabeled` last) and category -> topics."""
    names: tuple[str, ...]
    cat_topics: dict[str, tuple[int, ...]]
    redirects: dict[str, str]
    hidden: frozenset
    prevalence: np.ndarray | None = None      # pi: mean chi over all articles, if the resources provide it

    @property
    def k(self) -> int:
        return len(self.names)

    def hash(self) -> str:
        return hashlib.sha256("\n".join(self.names).encode()).hexdigest()[:12]

    def chi(self, categories: list[str], weighting: str = "multiplicity") -> np.ndarray:
        """Membership vector over `names`, summing to 1. Pages without a topic go to the unlabeled bucket.

        multiplicity: each category splits one vote over its topics; chi = votes / total votes.
        uniform: 1/|S| on each distinct topic S the page reaches."""
        v = np.zeros(self.k)
        for c in categories:
            c = resolve_category(norm_category(c), self.redirects)
            if c in self.hidden:
                continue
            ts = self.cat_topics.get(c)
            if ts:
                v[list(ts)] += 1.0 / len(ts)
        if v.sum() == 0:
            v[-1] = 1.0
        elif weighting == "uniform":
            v = (v > 0) / (v > 0).sum()
        elif weighting == "multiplicity":
            v /= v.sum()
        else:
            raise ValueError(f"unknown chi weighting {weighting!r}")
        return v


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def read_category_pages(path: Path) -> tuple[dict[str, list[str]], dict[str, str], set[str]]:
    """category_pages.tsv(.gz): category, redirect_to, hidden, parents ('|'-joined). Returns parents, redirects, hidden."""
    parents, redirects, hidden = {}, {}, set()
    with _open(path) as f:
        next(f)
        for line in f:
            cat, red, hid, ps = line.rstrip("\n").split("\t")
            if red:
                redirects[cat] = red
            if hid == "1":
                hidden.add(cat)
            parents[cat] = [p for p in ps.split("|") if p]
    return parents, redirects, hidden


def build_topic_map(parents, redirects, hidden, roots: list[str], max_depth: int = 7) -> tuple[TopicMap, dict]:
    """Resolve redirects and drop hidden categories, then roll up. Root labels are the redirect targets."""
    clean = {}
    for cat, ps in parents.items():
        if cat in redirects or cat in hidden:
            continue
        clean[cat] = [q for q in {resolve_category(p, redirects) for p in ps} if q not in hidden and q != cat]
    labels = []
    for r in roots:
        lab = resolve_category(norm_category(r), redirects)
        if lab not in labels:
            labels.append(lab)
    depth_topics = rollup(clean, labels, max_depth)
    idx = {lab: i for i, lab in enumerate(labels)}
    cat_topics = {c: tuple(sorted(idx[t] for t in ts)) for c, (_, ts) in depth_topics.items()}
    info = {"roots_config": list(roots), "roots_resolved": labels, "categories_total": len(parents),
            "categories_redirect": len(redirects), "categories_hidden": len(hidden),
            "categories_reached": len(cat_topics), "max_depth": max_depth}
    return TopicMap(tuple(labels) + (UNLABELED,), cat_topics, dict(redirects), frozenset(hidden)), info


def save_topic_map(tm: TopicMap, info: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "category_topics.tsv", "w", encoding="utf-8") as f:
        f.write("category\ttopics\n")
        for c, ts in sorted(tm.cat_topics.items()):
            f.write(f"{c}\t{'|'.join(tm.names[i] for i in ts)}\n")
    with open(out_dir / "category_redirects.tsv", "w", encoding="utf-8") as f:
        f.write("category\tredirect_to\n")
        for c, r in sorted(tm.redirects.items()):
            f.write(f"{c}\t{r}\n")
    (out_dir / "hidden_categories.txt").write_text("\n".join(sorted(tm.hidden)) + "\n", encoding="utf-8")
    meta = {**info, "topics": list(tm.names), "topic_hash": tm.hash()}
    if tm.prevalence is not None:
        meta["prevalence"] = dict(zip(tm.names, map(float, tm.prevalence)))
    (out_dir / "topics.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load_topic_map(resources: str | Path) -> TopicMap:
    """Read what scripts/build_wiki_resources.py wrote into the resources folder."""
    d = Path(resources)
    meta = json.loads((d / "topics.json").read_text(encoding="utf-8"))
    names = tuple(meta["topics"])
    idx = {n: i for i, n in enumerate(names)}
    cat_topics = {}
    with open(d / "category_topics.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            c, ts = line.rstrip("\n").split("\t")
            cat_topics[c] = tuple(idx[t] for t in ts.split("|"))
    redirects = {}
    with open(d / "category_redirects.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            c, r = line.rstrip("\n").split("\t")
            redirects[c] = r
    hidden = frozenset(x for x in (d / "hidden_categories.txt").read_text(encoding="utf-8").split("\n") if x)
    prev = meta.get("prevalence")
    pi = np.array([prev[n] for n in names]) if prev else None
    return TopicMap(names, cat_topics, redirects, hidden, pi)


class PageTopics:
    """chi of every article (page_topics.npz from scripts/build_wiki_resources.py pages), looked up by page id."""

    def __init__(self, page_id: np.ndarray, indptr: np.ndarray, topic: np.ndarray, weight: np.ndarray, k: int):
        self.page_id, self.indptr, self.topic, self.weight, self.k = page_id, indptr, topic, weight, k

    @classmethod
    def load(cls, resources: str | Path, tm: TopicMap) -> "PageTopics":
        z = np.load(Path(resources) / "page_topics.npz")
        if str(z["topic_hash"]) != tm.hash():
            raise ValueError("page_topics.npz was built with another topic map; rerun build_wiki_resources.py pages")
        return cls(z["page_id"], z["indptr"], z["topic"], z["weight"], tm.k)

    def chi(self, page_ids) -> np.ndarray:
        """(n, K) membership rows; pages not in the table count as unlabeled."""
        q = np.atleast_1d(np.asarray(page_ids, np.int64))
        pos = np.clip(np.searchsorted(self.page_id, q), 0, len(self.page_id) - 1)
        found = self.page_id[pos] == q
        out = np.zeros((len(q), self.k))
        out[~found, -1] = 1.0
        for i in np.flatnonzero(found):
            a, b = self.indptr[pos[i]], self.indptr[pos[i] + 1]
            out[i, self.topic[a:b]] = self.weight[a:b]
        return out
