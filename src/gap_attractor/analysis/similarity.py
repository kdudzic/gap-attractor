"""How alike two pages are: the choice configs/analysis/similarity/ makes (categories | embeddings | none).

Every option gives each page a vector, and the similarity of two pages is the dot product of their vectors:

  categories  the page's topic mix chi (categories rolled up to topic roots: the hypergraph). The dot product is
              the topic overlap, so the mean over clicks is the "trace share" of the topic tables. Topic tables
              (F, A, D, L) exist only for this option (has_topics).
  embeddings  a unit-length text embedding of the page (analysis/embedding.py). The dot product is the cosine.
  none        no vector (dim 0): only measures that need links alone (revisits, hubs, distances, bits per click).

The analysis only uses this interface, so a new option is a class registered here (or a factory named by dotted
path in `kind`): factory(cfg) -> PageVectors, where cfg is the whole analysis config.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gap_attractor.registry import Registry

SIMILARITIES = Registry("similarity")


class PageVectors:
    """The interface. `prepare` before `vectors` for pages that may need computing (embeddings fetch them)."""
    name = "none"
    has_topics = False
    topic_names: tuple = ()
    dim = 0

    def prepare(self, page_ids) -> None:
        pass

    def vectors(self, page_ids, categories=None) -> np.ndarray:
        """(n, dim) rows. `categories` (one list per page, as logged) may replace a lookup where an option uses them."""
        return np.zeros((len(page_ids), 0))

    def prevalence(self) -> np.ndarray:
        """Mean vector of a random article: the expected next page after a jump to a random page."""
        return np.zeros(self.dim)

    def breadth(self, V: np.ndarray) -> float:
        """How widely a set of pages spreads (higher = more spread)."""
        return float("nan")

    def describe(self) -> dict:
        return {"similarity": self.name}


@SIMILARITIES.register("none")
class NoVectors(PageVectors):
    def __init__(self, cfg=None):
        pass


@SIMILARITIES.register("categories")
class CategoryVectors(PageVectors):
    """chi from categories. Logged categories are used when given (clicks stage); otherwise page_topics.npz
    (scripts/build_wiki_resources.py pages) by page id (walks stage). breadth = entropy (bits) of the mean chi."""
    name, has_topics = "categories", True

    def __init__(self, cfg):
        from gap_attractor.analysis.topics import load_topic_map
        self.resources = Path(str(cfg.resources))
        if not (self.resources / "topics.json").exists():
            raise FileNotFoundError(f"no topic resources in {self.resources}: run scripts/build_wiki_resources.py "
                                    "(docs/analysis.md), or use analysis/similarity=embeddings or none")
        self.weighting = str(cfg.analysis.similarity.get("weighting", "multiplicity"))
        self.tm = load_topic_map(self.resources)
        self.topic_names, self.dim = tuple(self.tm.names), self.tm.k
        self._pages = None

    def vectors(self, page_ids, categories=None) -> np.ndarray:
        if categories is not None:
            return np.stack([self.tm.chi(c or [], self.weighting) for c in categories]) if len(categories) else np.zeros((0, self.dim))
        if self._pages is None:
            from gap_attractor.analysis.topics import PageTopics
            self._pages = PageTopics.load(self.resources, self.tm)
        return self._pages.chi(np.asarray(page_ids, np.int64))

    def prevalence(self) -> np.ndarray:
        return self.tm.prevalence if self.tm.prevalence is not None else np.full(self.dim, 1 / self.dim)

    def breadth(self, V: np.ndarray) -> float:
        m = V.mean(0)
        m = m[m > 0]
        return float(-(m * np.log2(m)).sum())

    def describe(self) -> dict:
        return {"similarity": self.name, "weighting": self.weighting, "topic_hash": self.tm.hash(), "topics": list(self.topic_names)}


@SIMILARITIES.register("embeddings")
class EmbeddingVectors(PageVectors):
    """Unit-length embeddings; a page without text gets a zero vector (similarity 0) and is counted in `missing`.
    breadth = 1 - |mean of the unit vectors| (0 when all pages are the same, towards 1 when they point everywhere)."""
    name = "embeddings"

    def __init__(self, cfg):
        from gap_attractor.analysis.embedding import EMBEDDERS, PAGE_TEXTS, EmbeddingCache, make
        s = cfg.analysis.similarity
        self.embedder, self.text = make(EMBEDDERS, cfg.analysis.embedder), make(PAGE_TEXTS, cfg.analysis.page_text)
        self.cache = EmbeddingCache(s.cache_dir, self.embedder.describe(), self.text.describe())
        self.max_new_pages = int(s.get("max_new_pages", 1_000_000))
        self.prevalence_sample, self.seed = int(s.get("prevalence_sample", 2000)), int(cfg.get("seed", 0))
        self.mem: dict[int, np.ndarray] = {}
        self.stats = {"embedded_now": 0, "from_cache": 0, "no_text": 0}
        self.dim = 0
        self._prev = None

    def prepare(self, page_ids) -> None:
        want = sorted({int(p) for p in page_ids} - self.mem.keys())
        if not want:
            return
        got = self.cache.get(want)
        self.stats["from_cache"] += len(got)
        for pid, v in got.items():
            self.mem[pid] = self._unit(v)
        missing = [p for p in want if p not in got]
        if not missing:
            return
        texts = self.text.texts(missing)
        todo = [(p, t) for p, t in zip(missing, texts) if t]
        no_text = [p for p, t in zip(missing, texts) if not t]
        self.stats["no_text"] += len(no_text)
        for p in no_text:
            self.mem[p] = None
        if len(todo) > self.max_new_pages:
            raise RuntimeError(f"{len(todo):,} pages would be embedded, more than analysis.similarity.max_new_pages "
                               f"({self.max_new_pages:,}); raise it if that is intended")
        for i in range(0, len(todo), 2048):           # write to the cache as we go, so an interrupted run keeps its work
            ids, txt = zip(*todo[i:i + 2048])
            vecs = self.embedder.embed(list(txt))
            self.cache.put(ids, vecs)
            for p, v in zip(ids, vecs):
                self.mem[p] = self._unit(v)
            self.stats["embedded_now"] += len(ids)

    def _unit(self, v) -> np.ndarray:
        v = np.asarray(v, np.float32)
        if not self.dim:
            self.dim = len(v)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def vectors(self, page_ids, categories=None) -> np.ndarray:
        ids = [int(p) for p in page_ids]
        self.prepare(ids)
        out = np.zeros((len(ids), self.dim), np.float32)
        for i, p in enumerate(ids):
            v = self.mem.get(p)
            if v is not None:
                out[i] = v
        return out

    def prevalence(self) -> np.ndarray:
        if self._prev is None:
            self._prev = self.vectors(self.text.sample(self.prevalence_sample, self.seed)).mean(0)
        return self._prev

    def breadth(self, V: np.ndarray) -> float:
        return float(1 - np.linalg.norm(V.mean(0))) if len(V) else float("nan")

    def describe(self) -> dict:
        return {"similarity": self.name, "embedder": self.embedder.describe(), "page_text": self.text.describe(),
                "cache": str(self.cache.path), "dim": self.dim, **self.stats, "usage": dict(self.embedder.usage)}


def make_similarity(cfg) -> PageVectors:
    """The option named by cfg.analysis.similarity.kind (a registered name or a dotted path to a factory)."""
    return SIMILARITIES.get(str(cfg.analysis.similarity.kind))(cfg)
