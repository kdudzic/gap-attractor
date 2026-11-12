"""Collect unusable replies (failures.jsonl, schema 3.6+) across run folders: one table of all of them, counts per
condition and category, and the titles models named that were never on their menu.

    uv run python scripts/failures.py runs/<runs> --out <folder>
    uv run python scripts/failures.py <older runs> --out <folder> --reconstruct --wikipedia dump

--reconstruct rebuilds the attempts of runs without failures.jsonl (before schema 3.6) from steps.jsonl: every reply
of a step is in model_output_raw. --wikipedia dump|fixture also checks whether named titles exist (loads the dump
index, ~8.5 GB); without it, titles not traced to a menu or page are 'unknown'. Categories: harness/failures.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap_attractor.analysis.logs import memory_setup, world_setup  # noqa: E402
from gap_attractor.harness import failures as F  # noqa: E402
from gap_attractor.harness.runner import attempted_choice, parse_reply  # noqa: E402

SEP = "\n---next call---\n"


def runs(paths):
    for p in map(Path, paths):
        for r in sorted(p.rglob("run.json")):
            if (r.parent / "steps.jsonl").exists() and "_superseded" not in r.parts:
                yield r.parent


def reconstruct(folder, info, backend):
    """Failures of an older run, from its steps: each reply of an invalid step that did not give a usable choice."""
    steps = [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines() if l.strip()]
    stop_ok = str(((info.get("config") or {}).get("experiment") or {}).get("allow_stop", True)).lower() != "false"
    out, earlier = [], []
    for s in steps:
        menu = [c["title"] for c in s.get("choice_set") or []] + [r["title"] for r in (s.get("search") or {}).get("results") or []]
        ia = s.get("invalid_action") or {}
        if ia.get("is_invalid") and ia.get("harness_handling") in ("retry", "skip"):
            page = backend.load_page(s["current_page_title"]) if backend is not None and s.get("current_page_title") else None
            replies = (s.get("model_output_raw") or "").split(SEP)      # an empty reply is a reply (no text, no call)
            for k, text in enumerate(replies):
                if text.lstrip().startswith('{"memory_tool_calls"'):
                    continue
                title, _, error = parse_reply(text, {t: t for t in menu}, stop_ok)
                if title is not None and k == len(replies) - 1 and ia["harness_handling"] == "retry":
                    continue                                   # the retry that worked
                attempted = attempted_choice(text) or None
                out.append({"run_id": info["run_id"], "step_index": s["step_index"], "attempt": k,
                            "invalid_type": ia["invalid_type"] if k == 0 else (error or "out-of-range"), "attempted": attempted,
                            **F.classify(attempted, error, menu, earlier, page, backend), "current_page": s.get("current_page_title"),
                            "menu_size": len(menu), "reply_text": text[:2000], "tool_calls": [], "step_handling": ia["harness_handling"],
                            "step_choice": s.get("parsed_choice"), "reconstructed": True})
        earlier.append((s["step_index"], menu))
    return out


def make_backend(name):
    if not name:
        return None
    from hydra import compose, initialize_config_dir
    from gap_attractor.harness.runner import make_backend as build
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"wikipedia={name}"])
    return build(cfg, ROOT)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--reconstruct", action="store_true", help="rebuild failures of runs without failures.jsonl")
    p.add_argument("--wikipedia", default=None, help="dump | fixture: check whether named titles exist")
    a = p.parse_args(argv)
    backend = make_backend(a.wikipedia)
    rows, steps_per_group = [], Counter()
    for folder in runs(a.paths):
        info = json.loads((folder / "run.json").read_text())
        group = {"condition": info.get("condition"), "model": ((info.get("config") or {}).get("provider") or {}).get("model"),
                 "memory": memory_setup(info), "world": world_setup(info)}
        n_steps = sum(1 for l in (folder / "steps.jsonl").read_text().splitlines() if l.strip())
        steps_per_group[tuple(group.values())] += n_steps
        log = folder / "failures.jsonl"
        if log.exists():
            found = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        elif a.reconstruct:
            found = reconstruct(folder, info, backend)
        else:
            continue
        rows += [{**group, "folder": str(folder), **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                                                      for k, v in r.items()}} for r in found]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("no failures found" + ("" if a.reconstruct else " (older runs need --reconstruct)"))
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "failures.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    counts = defaultdict(Counter)
    for r in rows:
        counts[(r["condition"], r["model"], r["memory"], r["world"])][r["category"]] += 1
    cats = list(F.CATEGORIES) + ["unknown"]
    lines = ["Full condition labels (memory settings, world) are in failures.csv.", "",
             "| condition | model | memory | steps | failures / 100 steps | " + " | ".join(cats) + " |",
             "|---" * (5 + len(cats)) + "|"]
    for key, c in sorted(counts.items(), key=lambda kv: [str(x) for x in kv[0]]):
        n = steps_per_group[key]
        lines.append(f"| {key[0]} | {key[1]} | {str(key[2]).split('|')[0]} | {n} | {round(100 * sum(c.values()) / max(1, n), 1)} | "
                     + " | ".join(str(c.get(k, 0)) for k in cats) + " |")
    unseen = Counter((r["attempted"], r["category"], r.get("resolved_title")) for r in rows if r["category"] in ("in_dump", "not_in_dump", "unknown"))
    lines += ["", "Titles named but never shown (most frequent first):", "", "| title | category | resolves to | times |", "|---|---|---|---|"]
    lines += [f"| {t} | {c} | {res or ''} | {n} |" for (t, c, res), n in unseen.most_common(50)]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
