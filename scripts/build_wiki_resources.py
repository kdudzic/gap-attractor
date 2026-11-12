"""Build the topic resources the analysis needs from the multistream dump. Calls no model.

    uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py extract
    uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py rollup

extract  one parallel pass over the dump (~15 min with 12 workers). Writes to $GAP_ATTRACTOR_RESOURCES:
           category_pages.tsv.gz      category, redirect_to, hidden, parents   (every Category: page)
           article_categories.tsv.gz  page_id, title, categories               (every ns-0 non-redirect article)
rollup   reads those two files and the topic roots in configs/analysis/topics/<name>.yaml. Writes
           category_topics.tsv, category_redirects.tsv, hidden_categories.txt, topics.json (incl. prevalence pi)
pages    the topic mix chi of every article, for the walks stage. Writes page_topics.npz (~2 min)
leads    title and lead section (prose before the first heading, cleaned like the harness's article text, at most
           --lead-chars characters) of every article, for embeddings. One parallel pass over the dump (~15 min).
           Writes leads.sqlite (table pages: page_id, title, lead)
clickstream --clickstream <clickstream-enwiki-YYYY-MM.tsv.gz>: human link clicks between articles (Wikimedia
           clickstream, type "link" only), matched to page ids. Writes clickstream.npz + clickstream.json (~5 min)

    uv run --extra analysis python scripts/build_wiki_resources.py pages
    uv run --extra analysis python scripts/build_wiki_resources.py clickstream \
        --clickstream data/cache/clickstream/clickstream-enwiki-2026-09.tsv.gz

Categories are parsed with the harness's own parse_links_categories, so they match `semantic_cluster_label`.
Dump paths come from the same environment variables as the harness (GAP_ATTRACTOR_DUMP, ..._DUMP_XML, ..._DUMP_INDEX).
"""
import argparse
import bz2
import gzip
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gap_attractor.analysis.topics import (build_topic_map, load_topic_map, norm_category,  # noqa: E402
                                           read_category_pages, save_topic_map, TopicMap)
from gap_attractor.wikipedia.backend import clean_article_text, parse_links_categories  # noqa: E402

REDIRECT_TPL = re.compile(r"\{\{\s*(?:category[ _]?redirect|cat[ _]?redirect|categoryredirect)\s*\|\s*(?:1\s*=\s*)?"
                          r"(?::?\s*category\s*:)?\s*([^|}]+)", re.I)
HIDDEN_RE = re.compile(r"__HIDDENCAT__|\{\{\s*(?:hidden[ _]?category|hiddencat|tracking[ _]?category|"
                       r"maintenance[ _]?category|wikipedia[ _]category\s*\|[^}]*hidden\s*=\s*yes)", re.I)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _pages(raw):
    a, b = raw.find(b"<page>"), raw.rfind(b"</page>")
    if a < 0:
        return
    for elem in ET.fromstring(b"<r>" + raw[a:b + 7] + b"</r>"):
        rec = {"title": "", "ns": None, "id": None, "redirect": None, "text": ""}
        for ch in elem:
            t = _local(ch.tag)
            if t == "title": rec["title"] = ch.text or ""
            elif t == "ns": rec["ns"] = int(ch.text)
            elif t == "id" and rec["id"] is None: rec["id"] = int(ch.text)
            elif t == "redirect": rec["redirect"] = ch.attrib.get("title")
            elif t == "revision":
                for sub in ch:
                    if _local(sub.tag) == "text": rec["text"] = sub.text or ""
        yield rec


def _read_stream(f, offset):
    f.seek(offset)
    d, parts = bz2.BZ2Decompressor(), []
    while not d.eof:
        buf = f.read(1 << 20)
        if not buf:
            break
        parts.append(d.decompress(buf))
    return b"".join(parts)


def _work(job):
    dump, out, chunk, offsets = job
    n = {"articles": 0, "categories": 0}
    with open(dump, "rb") as f, open(out / f"art_{chunk:05d}.tsv", "w", encoding="utf-8") as art, \
            open(out / f"cat_{chunk:05d}.tsv", "w", encoding="utf-8") as cat:
        for off in offsets:
            for p in _pages(_read_stream(f, off)):
                if p["ns"] == 0 and not p["redirect"] and not p["text"].lstrip()[:9].upper() == "#REDIRECT":
                    _, cats = parse_links_categories(p["text"])
                    art.write(f"{p['id']}\t{p['title']}\t{'|'.join(norm_category(c) for c in cats)}\n")
                    n["articles"] += 1
                elif p["ns"] == 14:
                    name = norm_category(p["title"])
                    red = p["redirect"] or (m.group(1) if (m := REDIRECT_TPL.search(p["text"])) else "")
                    red = norm_category(red) if red else ""
                    _, parents = parse_links_categories(p["text"])
                    hidden = int(bool(HIDDEN_RE.search(p["text"])))
                    cat.write(f"{name}\t{red}\t{hidden}\t{'|'.join(norm_category(c) for c in parents)}\n")
                    n["categories"] += 1
    return n


FIRST_HEADING = re.compile(r"^==[^=].*?==\s*$", re.M)


def lead_text(wikitext: str, max_chars: int) -> str:
    """The prose before the first section heading, cleaned like the harness's article text, on one line."""
    m = FIRST_HEADING.search(wikitext or "")
    text, _ = clean_article_text((wikitext or "")[:m.start()] if m else (wikitext or ""))
    return " ".join(text.split())[:max_chars]


def _lead_work(job):
    dump, out, chunk, offsets, max_chars = job
    n = 0
    with open(dump, "rb") as f, open(out / f"lead_{chunk:05d}.tsv", "w", encoding="utf-8") as fo:
        for off in offsets:
            for p in _pages(_read_stream(f, off)):
                if p["ns"] == 0 and not p["redirect"] and not p["text"].lstrip()[:9].upper() == "#REDIRECT":
                    fo.write(f"{p['id']}\t{' '.join(p['title'].split())}\t{lead_text(p['text'], max_chars)}\n")
                    n += 1
    return n


def dump_paths():
    dump_dir = Path(os.environ.get("GAP_ATTRACTOR_DUMP", "data/wikipedia/enwiki-20260901/"))   # same default as configs/wikipedia/dump.yaml
    dump = dump_dir / os.environ.get("GAP_ATTRACTOR_DUMP_XML", "enwiki-latest-pages-articles-multistream.xml.bz2")
    index = dump_dir / os.environ.get("GAP_ATTRACTOR_DUMP_INDEX", "enwiki-latest-pages-articles-multistream-index.txt.bz2")
    return dump, index


def stream_offsets(index: Path) -> list[int]:
    offs, last = [], None
    with bz2.open(index, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            o = int(line.split(":", 1)[0])
            if o != last:
                offs.append(o); last = o
    return offs


def leads_cmd(res: Path, workers: int, max_chars: int):
    import sqlite3
    dump, index = dump_paths()
    parts = res / "_lead_parts"
    parts.mkdir(parents=True, exist_ok=True)
    offs, step = stream_offsets(index), 400
    jobs = [(str(dump), parts, i // step, offs[i:i + step], max_chars) for i in range(0, len(offs), step)]
    print(f"{dump}: {len(offs):,} streams, {len(jobs)} chunks, {workers} workers", flush=True)
    total, t0 = 0, time.time()
    with Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_lead_work, jobs), 1):
            total += n
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} chunks, {total:,} articles, {(time.time() - t0) / 60:.1f} min", flush=True)
    db = res / "leads.sqlite"
    tmp = res / "leads.sqlite.tmp"
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE pages (page_id INTEGER PRIMARY KEY, title TEXT NOT NULL, lead TEXT NOT NULL)")
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    for p in sorted(parts.glob("lead_*.tsv")):
        rows = (line.rstrip("\n").split("\t", 2) for line in p.read_text(encoding="utf-8").splitlines() if line)
        con.executemany("INSERT OR REPLACE INTO pages VALUES (?, ?, ?)", ((int(a), b, c) for a, b, c in rows))
    con.executemany("INSERT INTO meta VALUES (?, ?)", [("dump", str(dump)), ("lead_chars", str(max_chars)),
                                                      ("articles", str(total)), ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S"))])
    con.commit(); con.close()
    tmp.replace(db)
    for p in parts.glob("lead_*.tsv"):
        p.unlink()
    parts.rmdir()
    print(f"leads.sqlite: {total:,} articles", flush=True)


def extract(res: Path, workers: int):
    dump_dir = Path(os.environ.get("GAP_ATTRACTOR_DUMP", "data/wikipedia/enwiki-20260901/"))   # same default as configs/wikipedia/dump.yaml
    dump = dump_dir / os.environ.get("GAP_ATTRACTOR_DUMP_XML", "enwiki-latest-pages-articles-multistream.xml.bz2")
    index = dump_dir / os.environ.get("GAP_ATTRACTOR_DUMP_INDEX", "enwiki-latest-pages-articles-multistream-index.txt.bz2")
    parts = res / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    offs, last = [], None
    with bz2.open(index, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            o = int(line.split(":", 1)[0])
            if o != last:
                offs.append(o); last = o
    step = 400
    jobs = [(str(dump), parts, i // step, offs[i:i + step]) for i in range(0, len(offs), step)]
    print(f"{dump}: {len(offs):,} streams, {len(jobs)} chunks, {workers} workers", flush=True)
    tot, t0 = {"articles": 0, "categories": 0}, time.time()
    with Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_work, jobs), 1):
            for k in tot: tot[k] += n[k]
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} chunks {tot} {(time.time() - t0) / 60:.1f} min", flush=True)
    for prefix, name, header in (("art", "article_categories.tsv.gz", "page_id\ttitle\tcategories\n"),
                                 ("cat", "category_pages.tsv.gz", "category\tredirect_to\thidden\tparents\n")):
        with gzip.open(res / name, "wt", encoding="utf-8") as out:
            out.write(header)
            for p in sorted(parts.glob(f"{prefix}_*.tsv")):
                out.write(p.read_text(encoding="utf-8"))
                p.unlink()
    parts.rmdir()
    manifest = {"dump": str(dump), "dump_size": dump.stat().st_size, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **tot}
    (res / "manifest_extract.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("extract done", manifest, flush=True)


def rollup_cmd(res: Path, topics_cfg: Path, weighting: str):
    cfg = yaml.safe_load(topics_cfg.read_text(encoding="utf-8"))
    parents, redirects, hidden = read_category_pages(res / "category_pages.tsv.gz")
    tm, info = build_topic_map(parents, redirects, hidden, cfg["roots"], cfg.get("max_depth", 7))
    print(json.dumps(info, indent=1), flush=True)
    total, n, unl = np.zeros(tm.k), 0, 0
    with gzip.open(res / "article_categories.tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            cats = line.rstrip("\n").split("\t", 2)[2]
            v = tm.chi(cats.split("|") if cats else [], weighting)
            total += v; n += 1; unl += v[-1] == 1.0
    pi = total / n
    tm = TopicMap(tm.names, tm.cat_topics, tm.redirects, tm.hidden, pi)
    info.update({"articles": n, "articles_unlabeled": int(unl), "prevalence_weighting": weighting,
                 "topics_config": str(topics_cfg.relative_to(ROOT)) if topics_cfg.is_relative_to(ROOT) else str(topics_cfg)})
    save_topic_map(tm, info, res)
    print(f"{n:,} articles, unlabeled {unl / n:.1%}")
    for name, p in sorted(zip(tm.names, pi), key=lambda x: -x[1]):
        print(f"  {name:20s} {p:.4f}")


def articles(res: Path):
    """(page_id, title, categories) for every article in article_categories.tsv.gz."""
    with gzip.open(res / "article_categories.tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, title, cats = line.rstrip("\n").split("\t", 2)
            yield int(pid), title, (cats.split("|") if cats else [])


def pages_cmd(res: Path, weighting: str):
    """page_topics.npz: chi of every article as a sparse matrix (rows sorted by page_id)."""
    tm = load_topic_map(res)
    pids, indptr, topic, weight = [], [0], [], []
    for pid, _, cats in articles(res):
        v = tm.chi(cats, weighting)
        nz = np.flatnonzero(v)
        pids.append(pid); topic.extend(nz.tolist()); weight.extend(v[nz].tolist()); indptr.append(len(topic))
    pids, indptr = np.array(pids, np.int64), np.array(indptr, np.int64)
    order = np.argsort(pids, kind="stable")
    if (order != np.arange(len(order))).any():   # re-sort rows by page_id
        starts, ends = indptr[:-1][order], indptr[1:][order]
        take = np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)])
        topic, weight = np.asarray(topic)[take], np.asarray(weight)[take]
        indptr = np.concatenate([[0], np.cumsum(ends - starts)])
        pids = pids[order]
    np.savez(res / "page_topics.npz", page_id=pids, indptr=indptr, topic=np.asarray(topic, np.int8),
             weight=np.asarray(weight, np.float32), names=np.array(tm.names), weighting=np.array(weighting),
             topic_hash=np.array(tm.hash()))
    print(f"page_topics.npz: {len(pids):,} articles, {len(topic):,} entries, topic_hash {tm.hash()}")


def clickstream_cmd(res: Path, path: Path):
    """clickstream.npz: per source article, human clicks to each linked article (type "link"), keyed by page id.
    Titles are matched to articles exactly (underscores -> spaces). Pairs with 10 or fewer clicks in the month
    are missing from the source data (Wikimedia's privacy threshold)."""
    import pandas as pd
    pid_of = {title: pid for pid, title, _ in articles(res)}
    src, dst, n = [], [], []
    rows = matched = clicks = clicks_matched = 0
    for chunk in pd.read_csv(path, sep="\t", names=["prev", "curr", "type", "n"], quoting=3, dtype={"n": np.int64},
                             chunksize=2_000_000, keep_default_na=False):
        chunk = chunk[chunk["type"] == "link"]
        rows += len(chunk); clicks += int(chunk["n"].sum())
        a = chunk["prev"].str.replace("_", " ").map(pid_of)
        b = chunk["curr"].str.replace("_", " ").map(pid_of)
        ok = a.notna() & b.notna()
        matched += int(ok.sum()); clicks_matched += int(chunk["n"][ok].sum())
        src.append(a[ok].to_numpy(np.int64)); dst.append(b[ok].to_numpy(np.int64)); n.append(chunk["n"][ok].to_numpy(np.int64))
    src, dst, n = np.concatenate(src), np.concatenate(dst), np.concatenate(n)
    order = np.lexsort((dst, src))
    src, dst, n = src[order], dst[order], n[order]
    sources, starts = np.unique(src, return_index=True)
    indptr = np.append(starts, len(src)).astype(np.int64)
    np.savez(res / "clickstream.npz", source=sources, indptr=indptr, target=dst, clicks=n)
    info = {"source_file": str(path), "link_rows": rows, "link_rows_matched": matched, "link_clicks": clicks,
            "link_clicks_matched": clicks_matched, "sources": int(len(sources)), "min_clicks_listed": 11,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (res / "clickstream.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["extract", "rollup", "pages", "clickstream", "leads"])
    ap.add_argument("--lead-chars", type=int, default=2000, help="step leads: characters kept per lead")
    ap.add_argument("--clickstream", help="clickstream-enwiki-YYYY-MM.tsv.gz (step clickstream)")
    ap.add_argument("--resources", default=os.environ.get("GAP_ATTRACTOR_RESOURCES", str(ROOT / "outputs" / "resources")))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--topics", default=str(ROOT / "configs" / "analysis" / "topics" / "roots.yaml"))
    ap.add_argument("--weighting", default="multiplicity", choices=["multiplicity", "uniform"])
    a = ap.parse_args()
    res = Path(a.resources)
    res.mkdir(parents=True, exist_ok=True)
    if a.step == "extract":
        extract(res, a.workers)
    elif a.step == "rollup":
        rollup_cmd(res, Path(a.topics), a.weighting)
    elif a.step == "pages":
        pages_cmd(res, a.weighting)
    elif a.step == "leads":
        leads_cmd(res, a.workers, a.lead_chars)
    else:
        if not a.clickstream:
            ap.error("clickstream needs --clickstream <file>")
        clickstream_cmd(res, Path(a.clickstream))


if __name__ == "__main__":
    main()
