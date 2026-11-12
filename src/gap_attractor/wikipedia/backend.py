"""Wikipedia page retrieval used by both synthetic and multistream-dump runs."""
from __future__ import annotations

import bz2
import json
import os
import pickle
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol
from xml.sax.saxutils import escape as _xml_escape


def escape(text: str) -> str:
    """How MediaWiki XML dumps write a title inside <title>: &, <, > and " escaped."""
    return _xml_escape(text, {'"': "&quot;"})

LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
SEC_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
SKIP_NS = ("File:", "Image:", "Media:", "Wikipedia:", "WP:", "Help:", "Template:",
           "User:", "Draft:", "Module:", "Portal:", "Talk:", "Special:", "TimedText:",
           "MediaWiki:", "Book:", "Category:")


def raw_key(title: str | None) -> str:
    """Spaces to underscores, leading ':' removed; case kept. How dump index titles are stored."""
    return (title or "").replace(" ", "_").lstrip(":")


def wiki_key(title: str | None) -> str:
    """Title -> page key the way MediaWiki normalises it: the first letter is capitalised, so [[oxygen]] is Oxygen.
    (Schema 1.x keys kept the case, which silently dropped every lowercase-first link.) If upper() would widen
    the letter (ß -> SS), it is left as is, like MediaWiki."""
    key = raw_key(title)
    first = key[:1].upper()
    return first + key[1:] if len(first) == 1 else key


@dataclass(frozen=True)
class Link:
    """A source hyperlink, retained separately from its article-text rendering."""
    candidate_key: str
    destination: str
    anchor_text: str
    raw_position: int


def parse_links_categories(wikitext: str) -> tuple[list[Link], list[str]]:
    links, categories, seen = [], [], set()
    for raw_position, raw in enumerate(LINK_RE.findall(wikitext or "")):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip().lstrip(":")
        if not target:
            continue
        if target.lower().startswith("category:"):
            category = target.split(":", 1)[1].replace("_", " ").strip()
            if category and category not in categories:
                categories.append(category)
            continue
        if target.startswith(SKIP_NS):
            continue
        key = wiki_key(target)
        if key not in seen:
            seen.add(key)
            anchor = raw.split("|", 1)[1].strip() if "|" in raw else target.replace("_", " ")
            links.append(Link(key, target.replace("_", " "), anchor, raw_position))
    return links, categories


def _strip_balanced(text: str, opener: str, closer: str) -> str:
    """Remove balanced MediaWiki constructs without leaving nested closing fragments."""
    output, index, depth = [], 0, 0
    while index < len(text):
        if text.startswith(opener, index):
            depth += 1
            index += len(opener)
        elif depth and text.startswith(closer, index):
            depth -= 1
            index += len(closer)
        elif depth:
            index += 1
        else:
            output.append(text[index])
            index += 1
    return "".join(output)


def _render_wikilinks(text: str) -> str:
    """Keep ordinary link anchor prose and remove full file/image/category constructs."""
    output, index = [], 0
    while index < len(text):
        if not text.startswith("[[", index):
            output.append(text[index])
            index += 1
            continue
        start, cursor, depth = index, index + 2, 1
        while cursor < len(text) and depth:
            if text.startswith("[[", cursor):
                depth += 1; cursor += 2
            elif text.startswith("]]", cursor):
                depth -= 1; cursor += 2
            else:
                cursor += 1
        if depth:  # malformed opening marker: omit it, preserve following prose.
            index += 2
            continue
        content = text[start + 2:cursor - 2]
        target, separator, anchor = content.partition("|")
        target = target.split("#", 1)[0].strip().lstrip(":")
        if not target.lower().startswith(("file:", "image:", "category:")):
            visible = anchor if separator else target
            output.append(_render_wikilinks(visible))
        index = cursor
    return "".join(output)


def clean_article_text(wikitext: str, max_chars: int | None = None) -> tuple[str, bool]:
    """Render all available article prose while removing navigation destinations/markup.

    Link destinations are extracted before this function is called.  The visible anchor text stays
    in the prose, but wiki markup and external URLs do not provide uncontrolled navigation paths.
    """
    text = wikitext or ""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = _strip_balanced(text, "{{", "}}")
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<ref\b[^/]*/>", " ", text, flags=re.I)
    text = _render_wikilinks(text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?m)^={2,6}\s*(.+?)\s*={2,6}$", r"\1", text)
    # Remaining delimiters are malformed, unmatched syntax after structured rendering.
    text = text.replace("[[", "").replace("]]", "").replace("{{", "").replace("}}", "")
    text = re.sub(r"\s+", " ", re.sub(r"'{2,}", "", text)).strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars], True
    return text, False


@dataclass
class Page:
    key: str
    title: str
    page_id: int
    wikitext: str
    links: list[Link]
    categories: list[str]
    section_names: list[str]
    created: str | None = None
    modified: str | None = None
    storage_cluster_id: int = 0
    ns: int = 0
    rev_id: int = 0

    @property
    def wikitext_chars(self) -> int:
        return len(self.wikitext)


class WikipediaBackend(Protocol):
    """What the runner needs from a page graph. Any object with these members can be plugged in with
    `wikipedia.backend=<dotted.path.to.factory>` (factory(wikipedia_cfg, project_root) -> backend).

    Optional: `iter_titles() -> iterable of (title, page_id)` lets search (Variants B/C, intent) run on an
    in-memory title index instead of a prebuilt SQLite index; the fixture backend uses this.
    """
    backend_name: str
    snapshot_date: str | None

    def load_page(self, title: str) -> Page | None: ...
    def contains(self, key: str) -> bool: ...
    def categories_for_keys(self, keys: list[str]) -> dict[str, list[str]]: ...
    def provenance(self) -> dict: ...


class FixtureBackend:
    backend_name = "fixture"

    def __init__(self, fixture_path: str | Path):
        self.fixture_path = Path(fixture_path)
        document = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self.fixture_metadata = {k: v for k, v in document.items() if k != "pages"}
        self.raw = {wiki_key(p["title"]): p for p in document.get("pages", [])}
        if not self.raw:
            raise ValueError("fixture contains no pages")
        self.snapshot_date = None
        self._cache: dict[str, Page] = {}

    def contains(self, key: str) -> bool:
        return wiki_key(key) in self.raw

    def iter_titles(self):
        """(title, page_id) of every page, in fixture order: the fixture's in-memory search index."""
        return ((page["title"], int(page["page_id"])) for page in self.raw.values())

    def load_page(self, title: str, _depth: int = 0) -> Page | None:
        key = wiki_key(title)
        if key in self._cache:
            return self._cache[key]
        item = self.raw.get(key)
        if not item or _depth > 4:
            return None
        if item.get("redirect"):  # {"title": ..., "page_id": ..., "redirect": "Target"}, like a dump redirect page
            return self.load_page(item["redirect"], _depth + 1)
        links, categories = parse_links_categories(item["wikitext"])
        page = Page(key, item["title"], int(item["page_id"]), item["wikitext"], links, categories,
                    [name for level, name in SEC_RE.findall(item["wikitext"]) if len(level) == 2],
                    item.get("created"), item.get("modified"))
        self._cache[key] = page
        return page

    def categories_for_keys(self, keys: list[str]) -> dict[str, list[str]]:
        return {wiki_key(key): (self.load_page(key).categories if self.load_page(key) else []) for key in keys}

    def provenance(self) -> dict:
        return {"backend": self.backend_name, "fixture_path": self.fixture_path.as_posix(),
                "fixture_purpose": self.fixture_metadata.get("purpose"), "snapshot_date": None}


class PageCache:
    """Least-recently-used Page cache with a size bound (the unbounded dict grew for the whole process)."""
    def __init__(self, max_pages: int = 4096):
        self.max_pages, self._pages = int(max_pages), OrderedDict()

    def get(self, key):
        page = self._pages.get(key)
        if page is not None:
            self._pages.move_to_end(key)
        return page

    def put(self, key, page):
        self._pages[key] = page
        self._pages.move_to_end(key)
        while len(self._pages) > self.max_pages:
            self._pages.popitem(last=False)


def _index_rows(index_path):
    """(raw key, block offset, page id) for each line of a *-multistream-index.txt.bz2, in file order."""
    with bz2.open(index_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":", 2)
            if len(parts) == 3:
                yield raw_key(parts[2]), int(parts[0]), int(parts[1])


class PickleLookup:
    """Schema 1.x lookup: a dict {raw key: (offset, page_id, storage_cluster_id)} held in RAM (~7 GB for enwiki)."""
    format = "pickle"

    def __init__(self, cache_dir, index_path):
        self.path = Path(cache_dir) / "title_lookup.pkl"
        if self.path.exists():
            with self.path.open("rb") as handle:
                self.table = pickle.load(handle)
            return
        first: dict[str, tuple[int, int]] = {}
        for key, offset, page_id in _index_rows(index_path):
            first.setdefault(key, (offset, page_id))
        offsets = {offset: index for index, offset in enumerate(sorted({v[0] for v in first.values()}))}
        self.table = {key: (offset, page_id, offsets[offset]) for key, (offset, page_id) in first.items()}
        with self.path.open("wb") as handle:
            pickle.dump(self.table, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get(self, key):
        return self.table.get(key)

    def any_key(self):
        return next(iter(self.table))


class SqliteLookup:
    """Same table as PickleLookup (first index line per key wins; storage_cluster_id = rank of the block offset),
    stored in SQLite and read through the OS page cache, so concurrent processes share one copy instead of each
    holding ~7 GB. Built once per dump next to the pickle; a lock file makes concurrent first runs wait."""
    format = "sqlite"
    BUILD_VERSION = 1

    def __init__(self, cache_dir, index_path, wait_seconds=3600):
        self.path = Path(cache_dir) / "title_lookup.sqlite"
        if not self.path.exists():
            self._build_once(Path(index_path), wait_seconds)
        self.con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self._get = lru_cache(maxsize=262_144)(self._query)

    def _build_once(self, index_path, wait_seconds):
        lock = self.path.with_name(self.path.name + ".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        except FileExistsError:  # someone else is building it
            print(f"waiting for another process to finish building {self.path} (lock: {lock})", file=sys.stderr)
            deadline = time.time() + wait_seconds
            while not self.path.exists():
                if time.time() > deadline:
                    raise TimeoutError(f"{self.path} was not built within {wait_seconds}s; if no build is running, delete {lock}")
                time.sleep(5)
            return
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        try:
            os.write(fd, f"{os.getpid()}\n".encode()); os.close(fd)
            print(f"building {self.path} from {index_path.name} (once per dump, a few minutes)", file=sys.stderr)
            con = sqlite3.connect(tmp)
            con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-1000000;"
                              "CREATE TABLE lookup(key TEXT PRIMARY KEY, offset INTEGER NOT NULL, page_id INTEGER NOT NULL) WITHOUT ROWID;")
            con.executemany("INSERT OR IGNORE INTO lookup VALUES (?,?,?)", _index_rows(index_path))
            con.executescript("CREATE TABLE clusters(offset INTEGER PRIMARY KEY, cluster INTEGER NOT NULL);"
                              "INSERT INTO clusters SELECT offset, ROW_NUMBER() OVER (ORDER BY offset) - 1 FROM (SELECT DISTINCT offset FROM lookup);"
                              "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);")
            con.executemany("INSERT INTO meta VALUES (?,?)", [("source_index_file", index_path.name),
                            ("build_version", str(self.BUILD_VERSION)), ("built_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))])
            con.commit(); con.close()
            os.chmod(tmp, 0o664)
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)

    def _query(self, key):
        row = self.con.execute("SELECT l.offset, l.page_id, c.cluster FROM lookup l JOIN clusters c ON c.offset = l.offset "
                               "WHERE l.key = ?", (key,)).fetchone()
        return tuple(row) if row else None

    def get(self, key):
        return self._get(key)

    def any_key(self):
        return self.con.execute("SELECT key FROM lookup LIMIT 1").fetchone()[0]


LOOKUP_FORMATS = {"sqlite": SqliteLookup, "pickle": PickleLookup}


class DumpBackend:
    """Read the existing multistream XML/index layout without network access.

    The title lookup (key -> block offset, page id) is built once per dump in the cache directory, as SQLite
    (default, shared between processes) or as the schema 1.x pickle. Dump mode never calls MediaWiki;
    `created` is therefore unavailable from current-revision articles alone.
    """
    backend_name = "dump"

    def __init__(self, dump_path: str | Path, cache_dir: str | Path,
                 xml_filename="enwiki-latest-pages-articles-multistream.xml.bz2",
                 index_filename="enwiki-latest-pages-articles-multistream-index.txt.bz2",
                 max_compressed_block_bytes=32_000_000, lookup_format="sqlite", page_cache_size=4096):
        self.dump_path, self.cache_dir = Path(dump_path), Path(cache_dir)
        self.xml_path = self.dump_path / xml_filename
        self.index_path = self.dump_path / index_filename
        self.max_compressed_block_bytes = int(max_compressed_block_bytes)
        if self.max_compressed_block_bytes <= 0:
            raise ValueError("max_compressed_block_bytes must be positive")
        if lookup_format not in LOOKUP_FORMATS:
            raise ValueError(f"unknown wikipedia.lookup_format: {lookup_format!r}; expected one of {tuple(LOOKUP_FORMATS)}")
        if not self.xml_path.exists() or not self.index_path.exists():
            raise FileNotFoundError("dump backend needs configured files "
                                    f"{self.xml_path} and {self.index_path}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lookup = LOOKUP_FORMATS[lookup_format](self.cache_dir, self.index_path)
        self.cache_path = self.lookup.path
        self.snapshot_date = None
        self._cache = PageCache(page_cache_size)

    def _find(self, title):
        """(stored key, lookup row) for a title: the MediaWiki-normalised key first, then the key exactly as written.
        The dump index stores titles XML-escaped (AT&amp;T, &quot;Weird Al&quot; Yankovic: ~150k titles), so the
        escaped spelling is tried too; without it every link to such a page was dropped."""
        normalised, as_written = wiki_key(title), raw_key(title)
        for key in dict.fromkeys((normalised, as_written, escape(normalised), escape(as_written))):
            row = self.lookup.get(key)
            if row is not None:
                return key, row
        return None, None

    def contains(self, key: str) -> bool:
        return self._find(key)[0] is not None

    @lru_cache(maxsize=128)
    def _stream(self, offset: int) -> str:
        decompressor, chunks, read = bz2.BZ2Decompressor(), [], 0
        with self.xml_path.open("rb") as handle:
            handle.seek(offset)
            while read < self.max_compressed_block_bytes:
                block = handle.read(min(65536, self.max_compressed_block_bytes - read))
                if not block:
                    break
                read += len(block)
                try:
                    chunks.append(decompressor.decompress(block))
                except EOFError:
                    break
                if decompressor.eof:
                    break
        if not decompressor.eof:
            raise RuntimeError("compressed multistream block exceeded "
                               f"max_compressed_block_bytes={self.max_compressed_block_bytes}; "
                               "increase the configured retrieval guard instead of presenting partial content")
        return b"".join(chunks).decode("utf-8", errors="replace")

    @staticmethod
    def _page_xml(blob: str, title: str) -> str | None:
        for candidate in (title, title.replace("_", " "), title.replace(" ", "_")):
            index = blob.find(f"<title>{escape(candidate)}</title>") if candidate != escape(candidate) else -1
            if index < 0:
                index = blob.find(f"<title>{candidate}</title>")
            if index >= 0:
                start, end = blob.rfind("<page>", 0, index), blob.find("</page>", index)
                if start >= 0 and end >= 0:
                    return blob[start:end + 7]
        return None

    def load_page(self, title: str, _depth: int = 0) -> Page | None:
        key = wiki_key(title)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        stored_key, row = self._find(title)
        if row is None or _depth > 4:
            return None
        offset, fallback_id, cluster = row
        raw = self._page_xml(self._stream(offset), stored_key)
        if not raw:
            return None
        root = ET.fromstring(raw)
        values, timestamps, seen_id = {"title": "", "ns": 0, "page_id": 0, "rev_id": 0, "wikitext": "", "redirect": None}, [], False
        for element in root.iter():
            tag = element.tag.split("}")[-1]
            if tag == "title" and not values["title"]: values["title"] = element.text or ""
            elif tag == "ns": values["ns"] = int(element.text or 0)
            elif tag == "id":
                if not seen_id: values["page_id"], seen_id = int(element.text or 0), True
                elif not values["rev_id"]: values["rev_id"] = int(element.text or 0)
            elif tag == "timestamp" and element.text: timestamps.append(element.text)
            elif tag == "redirect": values["redirect"] = element.attrib.get("title")
            elif tag == "text": values["wikitext"] = element.text or ""
        if values["redirect"]:
            return self.load_page(values["redirect"], _depth + 1)
        links, categories = parse_links_categories(values["wikitext"])
        page = Page(wiki_key(values["title"]) or key, values["title"], values["page_id"] or fallback_id, values["wikitext"], links, categories,
                    [name for level, name in SEC_RE.findall(values["wikitext"]) if len(level) == 2], None,
                    timestamps[-1] if timestamps else None, cluster, values["ns"], values["rev_id"])
        self._cache.put(key, page); self._cache.put(page.key, page)
        return page

    def categories_for_keys(self, keys: list[str]) -> dict[str, list[str]]:
        return {wiki_key(key): (self.load_page(key).categories if self.load_page(key) else []) for key in keys}

    def provenance(self) -> dict:
        return {"backend": self.backend_name, "dump_path": self.dump_path.as_posix(),
                "xml_path": self.xml_path.name, "index_path": self.index_path.name,
                "max_compressed_block_bytes": self.max_compressed_block_bytes,
                "lookup_format": self.lookup.format, "lookup_path": self.lookup.path.as_posix(),
                "snapshot_date": self.snapshot_date, "creation_date_source": "not available in current-revision dump"}
