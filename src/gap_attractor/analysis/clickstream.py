"""Human link clicks from the Wikimedia clickstream, as a reference chooser with no fitted parameters.

clickstream.npz (scripts/build_wiki_resources.py clickstream) holds, per source article, how many readers clicked
each linked article in one month. Wikimedia drops every pair with 10 or fewer clicks, so a link that is not listed
had between 0 and 10 clicks. It is counted as `unlisted` clicks (default 5, the middle of that range); rerun with 1
and 10 to see how much a result depends on it. A page with no listed link at all therefore gets a uniform choice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Clickstream:
    def __init__(self, source, indptr, target, clicks, info=None):
        self.source, self.indptr, self.target, self.clicks, self.info = source, indptr, target, clicks, info or {}

    @classmethod
    def load(cls, resources: str | Path) -> "Clickstream":
        d = Path(resources)
        if not (d / "clickstream.npz").exists():
            raise FileNotFoundError(f"no clickstream.npz in {d}: run scripts/build_wiki_resources.py clickstream")
        z = np.load(d / "clickstream.npz")
        info = json.loads((d / "clickstream.json").read_text(encoding="utf-8")) if (d / "clickstream.json").exists() else {}
        return cls(z["source"], z["indptr"], z["target"], z["clicks"], info)

    def counts(self, source_pid: int, target_pids) -> np.ndarray:
        """Listed clicks from source to each target page id (0 where the pair is not listed)."""
        targets = np.asarray(target_pids, np.int64)
        out = np.zeros(len(targets), np.int64)
        i = np.searchsorted(self.source, source_pid)
        if i >= len(self.source) or self.source[i] != source_pid:
            return out
        a, b = self.indptr[i], self.indptr[i + 1]
        row_t, row_n = self.target[a:b], self.clicks[a:b]
        pos = np.clip(np.searchsorted(row_t, targets), 0, max(b - a - 1, 0))
        hit = (b > a) & (row_t[pos] == targets) if b > a else np.zeros(len(targets), bool)
        out[hit] = row_n[pos[hit]]
        return out

    def probs(self, source_pid: int, target_pids, unlisted: float = 5.0) -> tuple[np.ndarray, bool]:
        """Choice probabilities over the given links, and whether any of them was listed."""
        c = self.counts(source_pid, target_pids).astype(float)
        listed = bool((c > 0).any())
        c[c == 0] = unlisted
        return c / c.sum(), listed
