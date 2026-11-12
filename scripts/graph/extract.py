"""Parallel pass over the multistream dump, reproducing Claire's hub_analysis parsing.

Per article (ns0, non-redirect): page_id, key, cat_lists, raw parsed link keys
(_parse_links_cats order, before lookup/redirect filtering).
Per ns0 redirect: from_key, to_key (collect_redirects.py rules).

usage: extract.py <hub_analysis dir> <dump.xml.bz2> <index.txt.bz2> <out_dir> <workers>
"""
import bz2
import os
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool

sys.path.insert(0, sys.argv[1])
from build_lookup import index_key            # noqa: E402
from collect_redirects import redirect_target  # noqa: E402
from dump_reader import _parse_page            # noqa: E402
from wikitext import _parse_links_cats, wiki_key  # noqa: E402

DUMP, INDEX, OUT, WORKERS = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])


def read_stream(f, offset):
    f.seek(offset)
    d = bz2.BZ2Decompressor()
    parts = []
    while not d.eof:
        buf = f.read(1 << 20)
        if not buf:
            break
        parts.append(d.decompress(buf))
    return b"".join(parts)


def work(job):
    chunk_id, offsets = job
    art = open(os.path.join(OUT, f"art_{chunk_id:05d}.tsv"), "w", encoding="utf-8")
    red = open(os.path.join(OUT, f"red_{chunk_id:05d}.tsv"), "w", encoding="utf-8")
    n = {"pages": 0, "articles": 0, "redirects": 0}
    with open(DUMP, "rb") as f:
        for off in offsets:
            raw = read_stream(f, off)
            a, b = raw.find(b"<page>"), raw.rfind(b"</page>")
            if a < 0:
                continue
            root = ET.fromstring(b"<r>" + raw[a:b + 7] + b"</r>")
            for elem in root:
                p = _parse_page(elem)
                n["pages"] += 1
                if p["ns"] != 0:
                    continue
                tgt = redirect_target(p)
                if tgt is not None:
                    to_key = wiki_key(tgt)
                    if to_key:
                        red.write(f"{index_key(p['title'])}\t{to_key}\n")
                        n["redirects"] += 1
                    continue
                links, cats = _parse_links_cats(p["text"])
                cl = int(any("lists" in c.lower() for c in cats))
                links = [k.replace("\t", " ").replace("\n", " ").replace("\x1f", " ") for k in links]
                art.write(f"{p['page_id']}\t{index_key(p['title'])}\t{cl}\t" + "\x1f".join(links) + "\n")
                n["articles"] += 1
    art.close(); red.close()
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    offs, last = [], None
    with bz2.open(INDEX, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            o = int(line.split(":", 1)[0])
            if o != last:
                offs.append(o); last = o
    print(f"{len(offs):,} streams", flush=True)
    step = 400
    jobs = [(i // step, offs[i:i + step]) for i in range(0, len(offs), step)]
    tot = {"pages": 0, "articles": 0, "redirects": 0}
    with Pool(WORKERS) as pool:
        for i, n in enumerate(pool.imap_unordered(work, jobs), 1):
            for k in tot:
                tot[k] += n[k]
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} chunks  {tot}", flush=True)
    print("DONE", tot, flush=True)


if __name__ == "__main__":
    main()
