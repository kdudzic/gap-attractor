"""Build the redirect-merged article graph + Claire's hub pool from extract.py output.

usage: build_graph.py <hub_analysis dir> <index.txt.bz2> <raw dir> <out dir>
"""
import bz2
import glob
import json
import os
import re
import sys
from array import array

import numpy as np
import yaml

HA, INDEX, RAW, OUT = sys.argv[1:5]
sys.path.insert(0, HA)
from analyze import junk_flags, percentile  # noqa: E402
from build_lookup import index_key          # noqa: E402
from wikitext import SKIP_NS, resolve       # noqa: E402

os.makedirs(OUT, exist_ok=True)
cfg = yaml.safe_load(open(os.path.join(HA, "config.yaml")))
stats = {}

print("lookup ...", flush=True)
lookup = set()
with bz2.open(INDEX, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.rstrip("\n").split(":", 2)
        if len(parts) == 3:
            lookup.add(index_key(parts[2]))
stats["lookup_keys"] = len(lookup)

print("redirects ...", flush=True)
redirects = {}
for p in sorted(glob.glob(os.path.join(RAW, "red_*.tsv"))):
    with open(p, encoding="utf-8") as f:
        for line in f:
            s, d = line.rstrip("\n").split("\t", 1)
            redirects[s] = d
stats["redirects"] = len(redirects)

print("articles ...", flush=True)
art_files = sorted(glob.glob(os.path.join(RAW, "art_*.tsv")))
keys, pids, catl = [], array("q"), array("b")
kid = {}
for p in art_files:
    with open(p, encoding="utf-8") as f:
        for line in f:
            pid, key, cl, _ = line.split("\t", 3)
            if key in kid:          # duplicate key: first wins
                continue
            kid[key] = len(keys)
            keys.append(key); pids.append(int(pid)); catl.append(int(cl))
N = len(keys)
stats["articles"] = N
print(f"  {N:,} articles", flush=True)

print("edges ...", flush=True)
src, dst = array("i"), array("i")
out_h = np.zeros(N, np.int32)       # harness menu size (offered, unmerged)
out_m = np.zeros(N, np.int32)       # merged out-degree (Claire's out_degree_merged)
in_all = np.zeros(N, np.int64)
in_cont = np.zeros(N, np.int64)
cnt = {"merged_links": 0, "merged_to_non_article": 0, "self_links": 0}
non_article_in = {}
for p in art_files:
    with open(p, encoding="utf-8") as f:
        for line in f:
            pid, key, cl, links = line.rstrip("\n").split("\t", 3)
            u = kid[key]
            if pids[u] != int(pid):
                continue
            offered = [k for k in links.split("\x1f") if k and k in lookup and not k.startswith(SKIP_NS)] if links else []
            out_h[u] = len(offered)
            seen = set()
            for k in offered:
                r = resolve(k, redirects)
                if r not in lookup or r.startswith(SKIP_NS) or r in seen:
                    continue
                seen.add(r)
                v = kid.get(r)
                if v is None:
                    cnt["merged_to_non_article"] += 1
                    non_article_in[r] = non_article_in.get(r, 0) + 1
                    continue
                in_all[v] += 1
                if cl == "0":
                    in_cont[v] += 1
                if v == u:
                    cnt["self_links"] += 1
                    continue
                src.append(u); dst.append(v)
            out_m[u] = len(seen)
            cnt["merged_links"] += len(seen)
stats.update(cnt)
stats["edges_no_self"] = len(src)
print(f"  {len(src):,} edges", stats, flush=True)

# --- Claire's pool (select_pool.py) -------------------------------------------
regexes = [re.compile(x, re.I) for x in cfg["junk_title_regex"]]
title_junk = np.array([int(any(junk_flags(k.replace("_", " "), cfg["junk_title_prefixes"],
                                          cfg["junk_title_suffixes"], regexes))) for k in keys], np.int8)
catl = np.frombuffer(catl, np.int8)
thr = percentile(sorted(in_cont.tolist()), 1 - cfg["pool_in_pct"] / 100)
pool = (catl == 0) & (title_junk == 0) & (out_m >= cfg["pool_min_out"]) & (in_cont >= thr)
stats["pool_in_threshold"] = int(thr)
stats["pool_size"] = int(pool.sum())
stats["not_list"] = int(((catl == 0) & (title_junk == 0)).sum())
print("pool", stats["pool_size"], "threshold", thr, flush=True)

np.savez(os.path.join(OUT, "graph.npz"), src=np.frombuffer(src, np.int32), dst=np.frombuffer(dst, np.int32),
         page_id=np.frombuffer(pids, np.int64), out_h=out_h, out_m=out_m, in_all=in_all, in_cont=in_cont,
         cat_lists=catl, title_junk=title_junk, pool=pool)
with open(os.path.join(OUT, "keys.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(keys) + "\n")
json.dump(stats, open(os.path.join(OUT, "build_stats.json"), "w"), indent=1)
print("DONE", flush=True)
