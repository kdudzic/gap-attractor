"""Page embeddings for similarity=embeddings, in three exchangeable parts (each a Hydra config group):

  embedder   (configs/analysis/embedder/)   text -> vectors. `openai_compatible` calls any /embeddings endpoint
             (OpenRouter by default); `mock` is a deterministic bag-of-words embedder for tests.
  page_text  (configs/analysis/page_text/)  page id -> the text to embed. `lead_db` reads leads.sqlite
             (scripts/build_wiki_resources.py leads): the title, the lead section, or both.
  cache      EmbeddingCache: vectors already computed, on disk and shared, one file per (embedder, page text), so
             a page is embedded once and every later analysis reuses it.

A new embedder or text source is a class registered here, or any factory named by dotted path in `kind`:
factory(cfg_node) -> object with the methods of the classes below.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from gap_attractor.registry import Registry

EMBEDDERS = Registry("embedder")
PAGE_TEXTS = Registry("page text")


def make(registry: Registry, cfg):
    return registry.get(str(cfg.kind))(cfg)


# ---------------------------------------------------------------- embedders

@EMBEDDERS.register("openai_compatible")
class OpenAICompatibleEmbedder:
    """POST {base_url}/embeddings with {"model", "input": [texts]} in batches, retrying rate limits and server errors.
    Works with OpenRouter, OpenAI and most self-hosted servers. The key is read from the environment variable named
    by `api_key_env`. Token use and cost (where the server reports it) are added up in `usage`."""

    def __init__(self, cfg):
        self.model, self.base_url = str(cfg.model), str(cfg.base_url).rstrip("/")
        self.api_key_env = str(cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        self.batch_size, self.timeout, self.retries = int(cfg.get("batch_size", 64)), float(cfg.get("timeout", 60)), int(cfg.get("retries", 5))
        self.dimensions = cfg.get("dimensions")
        self.max_input_chars = int(cfg.get("max_input_chars", 2000))
        self.concurrency = max(1, int(cfg.get("concurrency", 1)))
        import threading
        self._lock = threading.Lock()
        self.usage = {"requests": 0, "texts": 0, "tokens": 0, "cost_usd": 0.0}

    def describe(self) -> dict:   # concurrency is left out: it does not change the vectors
        return {"kind": "openai_compatible", "model": self.model, "base_url": self.base_url,
                "dimensions": self.dimensions, "max_input_chars": self.max_input_chars}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Batches of batch_size, `concurrency` requests at a time; vectors come back in the order of `texts`."""
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"set {self.api_key_env} (e.g. uv run --env-file .env ...) to call {self.base_url}")
        batches = [[t[:self.max_input_chars] or " " for t in texts[i:i + self.batch_size]] for i in range(0, len(texts), self.batch_size)]
        if self.concurrency > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(self.concurrency) as pool:
                out = list(pool.map(lambda b: self._batch(b, key), batches))
        else:
            out = [self._batch(b, key) for b in batches]
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def _batch(self, batch: list[str], key: str) -> np.ndarray:
        """One request, retried with backoff on rate limits, server errors and timeouts."""
        import requests
        body = {"model": self.model, "input": batch}
        if self.dimensions:
            body["dimensions"] = int(self.dimensions)
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.base_url}/embeddings", json=body, timeout=self.timeout,
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                if r.status_code == 200 and "data" in (d := r.json()):
                    break
                err = f"HTTP {r.status_code}: {r.text[:300]}"
            except (requests.RequestException, ValueError) as e:
                err = repr(e)
            if attempt == self.retries:
                raise RuntimeError(f"embedding request failed after {self.retries + 1} attempts: {err}")
            time.sleep(min(60, 2 ** attempt))
        vecs = [e["embedding"] for e in sorted(d["data"], key=lambda e: e.get("index", 0))]
        if len(vecs) != len(batch):
            raise RuntimeError(f"embedding server returned {len(vecs)} vectors for {len(batch)} texts")
        u = d.get("usage") or {}
        with self._lock:
            self.usage["requests"] += 1; self.usage["texts"] += len(batch)
            self.usage["tokens"] += int(u.get("total_tokens") or u.get("prompt_tokens") or 0)
            self.usage["cost_usd"] += float(u.get("cost") or 0.0)
        return np.asarray(vecs, np.float32)


@EMBEDDERS.register("mock")
class MockEmbedder:
    """Deterministic and offline: hashed bag of words, so texts sharing words get similar vectors. For tests."""

    def __init__(self, cfg):
        self.dims = int(cfg.get("dims", 64))
        self.usage = {"requests": 0, "texts": 0, "tokens": 0, "cost_usd": 0.0}

    def describe(self) -> dict:
        return {"kind": "mock", "dims": self.dims}

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dims] += 1.0
        self.usage["texts"] += len(texts)
        return out


# ---------------------------------------------------------------- page text

@PAGE_TEXTS.register("lead_db")
class LeadDB:
    """leads.sqlite (table pages: page_id, title, lead). field: title_lead ("Title. Lead"), title or lead."""

    FIELDS = ("title_lead", "title", "lead")

    def __init__(self, cfg):
        self.path, self.field, self.max_chars = Path(str(cfg.db)), str(cfg.get("field", "title_lead")), int(cfg.get("max_chars", 2000))
        if self.field not in self.FIELDS:
            raise ValueError(f"page_text.field must be one of {self.FIELDS}, not {self.field!r}")
        if not self.path.exists():
            raise FileNotFoundError(f"no {self.path}: run scripts/build_wiki_resources.py leads")
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self._ids = None

    def describe(self) -> dict:
        meta = dict(self.con.execute("SELECT key, value FROM meta").fetchall())
        return {"kind": "lead_db", "db": str(self.path), "field": self.field, "max_chars": self.max_chars,
                "built_at": meta.get("built_at"), "dump": meta.get("dump")}

    def texts(self, page_ids) -> list[str | None]:
        ids = [int(p) for p in page_ids]
        got = {}
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            q = f"SELECT page_id, title, lead FROM pages WHERE page_id IN ({','.join('?' * len(chunk))})"
            for pid, title, lead in self.con.execute(q, chunk):
                text = {"title": title, "lead": lead, "title_lead": f"{title}. {lead}" if lead else title}[self.field]
                got[pid] = text[:self.max_chars]
        return [got.get(p) for p in ids]

    def sample(self, n: int, seed: int) -> np.ndarray:
        """n random article ids (for the expected similarity to a random page)."""
        if self._ids is None:
            self._ids = np.fromiter((r[0] for r in self.con.execute("SELECT page_id FROM pages")), np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(self._ids, size=min(n, len(self._ids)), replace=False))


# ---------------------------------------------------------------- cache

class EmbeddingCache:
    """page_id -> vector (float16) in <dir>/<key>.sqlite, where key names the embedder and the page text, so
    changing either never mixes vectors. Shared between analyses and people."""

    def __init__(self, cache_dir, embedder_desc: dict, text_desc: dict):
        ident = {"embedder": embedder_desc, "text": {k: text_desc.get(k) for k in ("kind", "field", "max_chars", "built_at")}}
        h = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{embedder_desc.get('model') or embedder_desc.get('kind')}__{text_desc.get('field')}")
        self.path = Path(str(cache_dir)) / f"{name}__{h}.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS vec (page_id INTEGER PRIMARY KEY, v BLOB NOT NULL)")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.con.execute("INSERT OR REPLACE INTO meta VALUES ('identity', ?)", (json.dumps(ident, sort_keys=True),))
        self.con.commit()
        for f in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):   # shared: teammates add to it too
            try:
                f.chmod(0o664)
            except OSError:
                pass

    def get(self, page_ids) -> dict[int, np.ndarray]:
        ids, out = [int(p) for p in page_ids], {}
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            for pid, blob in self.con.execute(f"SELECT page_id, v FROM vec WHERE page_id IN ({','.join('?' * len(chunk))})", chunk):
                out[pid] = np.frombuffer(blob, np.float16).astype(np.float32)
        return out

    def put(self, page_ids, vectors: np.ndarray) -> None:
        self.con.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)",
                             ((int(p), np.asarray(v, np.float16).tobytes()) for p, v in zip(page_ids, vectors)))
        self.con.commit()

    def count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM vec").fetchone()[0]
