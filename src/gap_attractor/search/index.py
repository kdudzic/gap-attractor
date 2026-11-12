"""Local Wikipedia search for Variants B, C and D. Runs entirely on the local dump: no API, no network.

Build once per dump (reads only the multistream index file, a few minutes):
    uv run python -m gap_attractor.search.index build \
        --index-file <dump folder>/enwiki-...-pages-articles-multistream-index.txt.bz2 \
        --out <folder>/enwiki-...-titles.sqlite

Try a query:
    uv run python -m gap_attractor.search.index query --db <sqlite> "black holes"

Smoke-test design: titles only (redirects included, e.g. "USA" finds United States), SQLite FTS5 BM25 with
Porter stemming, exact title match first, "(disambiguation)" last. All terms must match; if none do, any term
may match (logged as fts_mode). Snippets come from the page itself, loaded from the same dump.
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sqlite3
import time
from pathlib import Path

NAMESPACES = {"Talk", "User", "User talk", "Wikipedia", "Wikipedia talk", "WP", "Project", "File", "File talk",
              "Image", "Media", "MediaWiki", "MediaWiki talk", "Template", "Template talk", "Help", "Help talk",
              "Category", "Category talk", "Portal", "Portal talk", "Draft", "Draft talk", "TimedText",
              "TimedText talk", "Module", "Module talk", "Book", "Book talk", "Special", "Gadget", "Gadget talk",
              "Gadget definition", "Gadget definition talk", "Education Program", "Topic"}
STOPWORDS = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "is", "at", "by", "with"}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SCHEMA = """
    CREATE TABLE pages(rowid INTEGER PRIMARY KEY, title TEXT NOT NULL, title_norm TEXT NOT NULL, page_id INTEGER);
    CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""
FTS = """
    CREATE INDEX idx_title_norm ON pages(title_norm);
    CREATE VIRTUAL TABLE pages_fts USING fts5(title, content='pages', content_rowid='rowid',
                                              tokenize='porter unicode61 remove_diacritics 2');
    INSERT INTO pages_fts(pages_fts) VALUES('rebuild');
"""


def is_article_title(title):
    return ":" not in title or title.split(":", 1)[0] not in NAMESPACES


def norm(title):
    return " ".join((title or "").replace("_", " ").casefold().split())


def _fill(con, rows, meta):
    con.executescript(SCHEMA)
    batch = []
    for title, page_id in rows:
        if is_article_title(title):
            batch.append((title, norm(title), page_id))
        if len(batch) >= 200_000:
            con.executemany("INSERT INTO pages(title, title_norm, page_id) VALUES (?,?,?)", batch); batch = []
    if batch:
        con.executemany("INSERT INTO pages(title, title_norm, page_id) VALUES (?,?,?)", batch)
    con.executescript(FTS)
    meta["titles_indexed"] = con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    con.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
    con.commit()


def build_index(index_file, out_path):
    """Build a title index from a *-multistream-index.txt.bz2 file (lines: offset:page_id:title)."""
    index_file, out_path = Path(index_file), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    t0 = time.time()

    def rows():
        with bz2.open(index_file, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) == 3 and parts[0].isdigit():
                    yield parts[2], int(parts[1])

    con = sqlite3.connect(tmp)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    _fill(con, rows(), {"source_index_file": index_file.name, "indexed_fields": "title",
                        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    con.close()
    tmp.replace(out_path)
    print(f"done -> {out_path} ({time.time() - t0:.0f}s)")


class SearchIndex:
    def __init__(self, con, source):
        self.con, self.source = con, source
        self.meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}

    @classmethod
    def open(cls, db_path):
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"search index not found: {db_path}. Build it with "
                                    "`python -m gap_attractor.search.index build --index-file ... --out ...`")
        return cls(sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False), str(db_path))

    @classmethod
    def from_titles(cls, titles_and_ids, source="in-memory"):
        """In-memory index, used for the fixture backend and tests."""
        con = sqlite3.connect(":memory:", check_same_thread=False)
        _fill(con, titles_and_ids, {"source_index_file": source, "indexed_fields": "title"})
        return cls(con, source)

    def provenance(self):
        return {"search_index": self.source, **{f"search_index_{k}": v for k, v in self.meta.items()}}

    def search(self, query, limit=50):
        """Return (hits, fts_mode). hit = {matched_title, exact, bm25}."""
        tokens = [t.casefold() for t in TOKEN_RE.findall(query or "")]
        if not tokens:
            return [], "empty"
        exact = [{"matched_title": t, "exact": True, "bm25": None} for (t,) in
                 self.con.execute("SELECT title FROM pages WHERE title_norm = ? LIMIT 5", (norm(query),))]
        hits, mode = self._match(" ".join(f'"{t}"' for t in tokens), limit), "and"
        if not hits:
            content = [t for t in tokens if t not in STOPWORDS] or tokens
            hits, mode = self._match(" OR ".join(f'"{t}"' for t in content), limit), "or"
        seen, merged = set(), []
        for hit in exact + hits:
            if hit["matched_title"] not in seen:
                seen.add(hit["matched_title"]); merged.append(hit)
        merged.sort(key=lambda h: (not h["exact"], "(disambiguation)" in h["matched_title"]))
        return merged[:limit], mode

    def _match(self, fts_query, limit):
        sql = ("SELECT p.title, bm25(pages_fts) AS score FROM pages_fts JOIN pages p ON p.rowid = pages_fts.rowid "
               "WHERE pages_fts MATCH ? ORDER BY score LIMIT ?")
        return [{"matched_title": t, "exact": False, "bm25": round(s, 4)} for t, s in self.con.execute(sql, (fts_query, int(limit)))]


SHORTDESC_RE = re.compile(r"\{\{\s*short[ _]description\s*\|\s*([^{}|]*?)\s*(?:\|[^{}]*)?\}\}", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def make_snippet(wikitext, clean_article_text, mode="short_description_then_lead", n_sentences=2, max_chars=300):
    """Return (snippet, source). Short description = the one-liner under Wikipedia's search box."""
    match = SHORTDESC_RE.search(wikitext or "")
    description = match.group(1).strip() if match else ""
    if mode == "short_description_then_lead" and description and description.lower() != "none":
        return description, "short_description"
    lead, _ = clean_article_text((wikitext or "").split("\n==", 1)[0], None)
    return " ".join(SENTENCE_RE.split(lead)[:n_sentences])[:max_chars].strip(), "lead_sentences"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--index-file", required=True); b.add_argument("--out", required=True)
    q = sub.add_parser("query"); q.add_argument("--db", required=True); q.add_argument("text"); q.add_argument("-n", type=int, default=10)
    args = parser.parse_args()
    if args.cmd == "build":
        build_index(args.index_file, args.out)
    else:
        hits, mode = SearchIndex.open(args.db).search(args.text, args.n)
        print(f"mode={mode}, {len(hits)} hits")
        for i, hit in enumerate(hits, 1):
            print(f"{i:>2}. {hit['matched_title']}  (bm25={hit['bm25']}, exact={hit['exact']})")


if __name__ == "__main__":
    main()
