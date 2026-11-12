"""Embeddings of the context the model was given, once per step, to follow how it develops (opt-in).

Off by default (configs/context_embedding/none.yaml); `context_embedding=steps` turns it on for any memory. It adds
one embedding request per step and changes nothing the model sees. At the end of every step (close_step) the step's
final request (Caller.last_messages, the request the model's last reply answered) is embedded at two levels:

  turns   one text per turn. By default a turn is a message of the request (system, user, assistant, tool; a tool
          call as its name and arguments). A memory can give richer turns with Memory.embed_turns (the context-file
          memory gives its file turns with ids, kinds and authors). Vectors are keyed by a hash of the text, so a
          turn that comes back unchanged keeps its vector; changed text gets a new one. Texts longer than
          max_input_chars are cut at paragraph breaks and their pieces length-averaged (never truncated).
  file    the whole context: `mean` = length-weighted mean of the turn vectors (no extra request), `direct` = all
          turns as one text, or `both`.

Written to <run>/context_embedding/:
  steps.jsonl     one line per step: step, agent, file keys, and per turn: id, kind, role, author, key, characters.
                  A step whose embedding failed gets "error" and the run goes on (counted in run.json).
  vectors.sqlite  key -> unit vector (float16), with the embedder's description in table meta.
run.json gets `context_embedding` (embedder, settings, counts, cost). It is not part of memory_settings: it changes
nothing the model sees, so runs with and without it are the same condition.

load_trace(run_dir) returns the arrays; trace_summary(run_dir) per-step drift and kept turns.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

from gap_attractor.analysis.embedding import EMBEDDERS, make

FOLDER = "context_embedding"

FILE_MODES = ("mean", "direct", "both")


def enabled(cfg) -> bool:
    """cfg: the whole run config (its context_embedding node)."""
    return bool((cfg.get("context_embedding") or {}).get("enabled", False))


def key_of(prefix: str, text: str) -> str:
    return prefix + ":" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def pieces(text: str, limit: int) -> list[str]:
    """The text cut into pieces of at most `limit` characters, at paragraph breaks where possible."""
    text = text or " "
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for para in re.split(r"(\n\s*\n)", text):
        while len(para) > limit:                      # a paragraph longer than the limit: hard cuts
            if cur:
                out.append(cur); cur = ""
            out.append(para[:limit]); para = para[limit:]
        if len(cur) + len(para) > limit:
            out.append(cur); cur = ""
        cur += para
    if cur.strip():
        out.append(cur)
    return [p for p in out if p.strip()] or [text[:limit]]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


class VectorStore:
    """key -> unit vector in one sqlite file (float16, like the page-embedding cache)."""

    def __init__(self, path: Path, embedder_desc: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, timeout=60, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        old = self.con.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        desc = json.dumps(embedder_desc, sort_keys=True)
        if old and old[0] != desc:
            raise ValueError(f"{path} holds vectors of another embedder ({old[0]}); use a new folder")
        self.con.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (desc,))
        self.con.commit()

    def has(self, keys) -> set:
        keys, got = list(keys), set()
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            got |= {r[0] for r in self.con.execute(f"SELECT key FROM vec WHERE key IN ({','.join('?' * len(chunk))})", chunk)}
        return got

    def put(self, items: dict) -> None:
        self.con.executemany("INSERT OR REPLACE INTO vec VALUES (?, ?)",
                             ((k, np.asarray(v, np.float16).tobytes()) for k, v in items.items()))
        self.con.commit()

    def get(self, keys) -> dict:
        keys, out = list(keys), {}
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            for k, blob in self.con.execute(f"SELECT key, v FROM vec WHERE key IN ({','.join('?' * len(chunk))})", chunk):
                out[k] = np.frombuffer(blob, np.float16).astype(np.float32)
        return out


class ContextEmbedTrace:
    """Embeds the context once per step; one instance per run (a multi-agent world: one per agent run)."""

    def __init__(self, cfg, out_dir: Path, agent: int | None = None, embedder=None):
        self.cfg, self.agent, self.out_dir = cfg, agent, Path(out_dir)
        self.embedder = embedder if embedder is not None else make(EMBEDDERS, cfg.embedder)
        self.turns_on = bool(cfg.get("turns", True))
        self.file_mode = str(cfg.get("file", "both"))
        if self.file_mode not in FILE_MODES:
            raise ValueError(f"context_embedding.file must be one of {FILE_MODES}, not {self.file_mode!r}")
        self.limit = int(cfg.get("max_input_chars") or getattr(self.embedder, "max_input_chars", 2000))
        if hasattr(self.embedder, "max_input_chars"):  # we cut texts ourselves; the embedder must not cut them again
            self.embedder.max_input_chars = max(int(self.embedder.max_input_chars), self.limit)
        self.store = VectorStore(self.out_dir / "vectors.sqlite", self.describe()["embedder"])
        self.log_path = self.out_dir / "steps.jsonl"
        self.stats = {"steps": 0, "errors": 0, "texts_embedded": 0, "texts_reused": 0, "pieces": 0}

    def describe(self) -> dict:
        return {"embedder": self.embedder.describe(), "turns": self.turns_on, "file": self.file_mode,
                "max_input_chars": self.limit}

    def _embed(self, texts: dict) -> dict:
        """key -> text for keys not stored yet: embedded (pieces averaged by length), stored, returned."""
        stored = self.store.has(texts)
        todo = {k: t for k, t in texts.items() if k not in stored}
        self.stats["texts_reused"] += len(texts) - len(todo)
        if not todo:
            return {}
        parts, owner = [], []
        for k, t in todo.items():
            for p in pieces(t, self.limit):
                parts.append(p); owner.append(k)
        vecs = self.embedder.embed(parts)
        out = {}
        for k in todo:
            idx = [i for i, o in enumerate(owner) if o == k]
            w = np.array([len(parts[i]) for i in idx], np.float32)
            out[k] = unit((vecs[idx] * w[:, None]).sum(axis=0))
        self.store.put(out)
        self.stats["texts_embedded"] += len(out); self.stats["pieces"] += len(parts)
        return out

    def record(self, step: int, turns: list[dict], file_text: str) -> dict:
        """turns: [{id, kind, role, agent, text}] in file order. Logs one line; never raises."""
        line = {"step": step, "agent": self.agent, "n_turns": len(turns), "file_chars": len(file_text)}
        try:
            rows = [{"id": t.get("id"), "kind": t.get("kind"), "role": t.get("role"), "author": t.get("agent"),
                     "chars": len(t.get("text") or ""), "key": key_of("t", t.get("text") or "")} for t in turns]
            want = {}
            if self.turns_on or self.file_mode in ("mean", "both"):
                want.update({r["key"]: t.get("text") or "" for r, t in zip(rows, turns)})
            if self.file_mode in ("direct", "both"):
                line["file_direct"] = key_of("f", file_text)
                want[line["file_direct"]] = file_text
            self._embed(want)
            if self.file_mode in ("mean", "both") and rows:
                vec = self.store.get([r["key"] for r in rows])
                w = np.array([max(1, r["chars"]) for r in rows], np.float32)
                mean = unit((np.stack([vec[r["key"]] for r in rows]) * w[:, None]).sum(axis=0))
                line["file_mean"] = f"m:{self.agent if self.agent is not None else 0}:{step}"
                self.store.put({line["file_mean"]: mean})
            line["turns"] = rows
        except Exception as e:                       # embedding must never end a run
            line["error"] = repr(e)[:500]
            self.stats["errors"] += 1
        self.stats["steps"] += 1
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return line

    def record_request(self, step: int, messages, memory=None) -> dict | None:
        """The step's final request -> turns (the memory's, else one per message) -> record(). None if no request."""
        if not messages:
            return None
        turns = None
        if memory is not None:
            try:
                turns = memory.embed_turns(messages)
            except Exception as e:                   # a broken hook must not end the run: fall back to messages
                turns, self.stats["hook_errors"] = None, self.stats.get("hook_errors", 0) + 1
                print(f"[context_embedding] embed_turns failed: {type(e).__name__}: {e}", file=sys.stderr)
        turns = turns if turns is not None else turns_from_messages(messages)
        return self.record(step, turns, render_turns(turns))

    def summary(self) -> dict:
        usage = dict(getattr(self.embedder, "usage", {}) or {})
        return {**self.describe(), **self.stats, "usage": usage, "folder": FOLDER}


def message_text(m: dict) -> str:
    """A request message as text: its content, and any tool calls as `name {arguments}`."""
    parts = [m["content"]] if isinstance(m.get("content"), str) and m["content"] else []
    if isinstance(m.get("content"), list):            # content parts (text only)
        parts += [p.get("text", "") for p in m["content"] if isinstance(p, dict) and p.get("text")]
    for c in m.get("tool_calls") or []:
        f = c.get("function") or c
        args = f.get("arguments")
        parts.append(f"{f.get('name')} {args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}")
    return "\n".join(parts).strip()


def turns_from_messages(messages) -> list[dict]:
    """The default turns: one per message of the request, in order (id = position from 1, kind = role)."""
    return [{"id": i, "kind": m.get("role"), "role": m.get("role"), "agent": None, "text": message_text(m)}
            for i, m in enumerate(messages, 1)]


def render_turns(turns) -> str:
    """All turns as one text (the `direct` file vector)."""
    return "\n\n".join(f"[{t.get('role')}]\n{t.get('text') or ''}" for t in turns)


def make_trace(cfg, run_dir, agent=None):
    """The trace for one run when context_embedding is enabled, else None."""
    return ContextEmbedTrace(cfg.context_embedding, Path(run_dir) / FOLDER, agent) if enabled(cfg) else None


def load_trace(run_dir) -> dict:
    """The embeddings of one run folder as arrays:
      steps         [S] step numbers (failed steps left out)
      file_mean     [S, d] or None; file_direct [S, d] or None
      turn_keys     [T] distinct turn keys; turn_vectors [T, d]
      membership    list of S lists: per step, the indices into turn_keys of its turns in file order
      turn_info     list of S lists of the per-turn rows (id, kind, role, author, chars)
    """
    folder = Path(run_dir) / FOLDER
    lines = [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines() if l.strip()]
    lines = [l for l in lines if "error" not in l]
    con = sqlite3.connect(folder / "vectors.sqlite")
    vec = {k: np.frombuffer(b, np.float16).astype(np.float32) for k, b in con.execute("SELECT key, v FROM vec")}
    keys = sorted({r["key"] for l in lines for r in l.get("turns", [])})
    index = {k: i for i, k in enumerate(keys)}

    def stack(field):
        ks = [l.get(field) for l in lines]
        return np.stack([vec[k] for k in ks]) if ks and all(k in vec for k in ks) else None

    return {"steps": np.array([l["step"] for l in lines]), "file_mean": stack("file_mean"), "file_direct": stack("file_direct"),
            "turn_keys": keys, "turn_vectors": np.stack([vec[k] for k in keys]) if keys else np.zeros((0, 0), np.float32),
            "membership": [[index[r["key"]] for r in l.get("turns", [])] for l in lines],
            "turn_info": [[{k: r[k] for k in ("id", "kind", "role", "author", "chars")} for r in l.get("turns", [])] for l in lines]}


def trace_summary(run_dir) -> list[dict]:
    """Per step, descriptive: how the context moves in embedding space and how much of it is still there.
      drift_prev / drift_start   1 - cosine of the file vector to the previous step's / the first step's
      kept_turns                 share of the previous step's turn texts still present unchanged
      new_turns                  turns whose text was not in the previous step
    The file vector is file_direct when present, else file_mean."""
    t = load_trace(run_dir)
    files = t["file_direct"] if t["file_direct"] is not None else t["file_mean"]
    out, prev = [], None
    for i, step in enumerate(t["steps"]):
        cur = set(t["membership"][i])
        row = {"step": int(step), "n_turns": len(t["membership"][i])}
        if files is not None:
            row["drift_start"] = round(max(0.0, 1 - float(files[i] @ files[0])), 4)
            row["drift_prev"] = round(max(0.0, 1 - float(files[i] @ files[i - 1])), 4) if i else 0.0
        if prev is not None:
            row["kept_turns"] = round(len(cur & prev) / max(1, len(prev)), 3)
            row["new_turns"] = len(cur - prev)
        out.append(row)
        prev = cur
    return out
