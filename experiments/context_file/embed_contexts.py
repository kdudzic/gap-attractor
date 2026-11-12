"""Embed the context files of finished runs from their step snapshots (context/step_NNN.txt), in the format of a
run with context_embedding=steps (main: harness/context_embed.py), or print how the context developed in runs that
have embeddings. A snapshot is the file after the step's move; a live run embeds it just before the move.

    uv run --env-file .env python experiments/context_file/embed_contexts.py RESULTS_DIR [--dry-run]
    uv run python experiments/context_file/embed_contexts.py RESULTS_DIR --summary

Snapshots carry no turn ids or kinds: a turn is its position in the file and its role, and turns are matched across
steps by their text (as in a live run). Runs that already have context_embedding/ are skipped unless --force.
Embedder and settings: the harness's configs/context_embedding/steps.yaml, any key overridable as key=value.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def runs_with_snapshots(root: Path):
    for ctx in sorted(root.rglob("context")):
        if ctx.is_dir() and any(ctx.glob("step_*.txt")) and (ctx.parent / "run.json").exists():
            yield ctx.parent


def settings(overrides):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(REPO / "configs" / "context_embedding" / "steps.yaml")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def embed_run(run_dir: Path, cfg, embedder=None) -> dict:
    from ctxfile.context import parse
    from gap_attractor.harness.context_embed import FOLDER, ContextEmbedTrace
    agent = (json.loads((run_dir / "run.json").read_text()).get("world") or {}).get("agent_id")
    trace = ContextEmbedTrace(cfg, run_dir / FOLDER, agent, embedder)
    for snap in sorted((run_dir / "context").glob("step_*.txt")):
        text = snap.read_text(encoding="utf-8")
        turns = [{"id": b["n"], "kind": None, "role": b["role"], "agent": b["agent"], "text": b["text"]} for b in parse(text)]
        trace.record(int(snap.stem.split("_")[1]), turns, text)
    summary = {**trace.summary(), "source": "backfill from context/step_NNN.txt"}
    (run_dir / FOLDER / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("overrides", nargs="*", help="settings as key=value (configs/context_embedding/steps.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="list the runs and the characters that would be embedded")
    ap.add_argument("--force", action="store_true", help="replace existing context_embedding/")
    ap.add_argument("--summary", action="store_true", help="print drift and kept turns per step for runs with embeddings")
    a = ap.parse_args(argv)
    if a.summary:
        from gap_attractor.harness.context_embed import trace_summary
        for emb in sorted(a.root.rglob("context_embedding/steps.jsonl")):
            run_dir = emb.parents[1]
            print(f"== {run_dir.relative_to(a.root)}")
            for row in trace_summary(run_dir):
                print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))
        return
    cfg = settings(a.overrides)
    runs = list(runs_with_snapshots(a.root))
    for run_dir in runs:
        out = run_dir / "context_embedding"
        if out.exists() and not a.force:
            print(f"skip (has embeddings): {run_dir.relative_to(a.root)}")
            continue
        if a.dry_run:
            chars = sum(len(p.read_text(encoding="utf-8")) for p in (run_dir / "context").glob("step_*.txt"))
            print(f"would embed: {run_dir.relative_to(a.root)}  ({chars:,} characters of snapshots, fewer after reuse)")
            continue
        if out.exists():
            shutil.rmtree(out)
        s = embed_run(run_dir, cfg)
        print(f"embedded: {run_dir.relative_to(a.root)}  steps={s['steps']} errors={s['errors']} "
              f"new={s['texts_embedded']} reused={s['texts_reused']} cost=${s['usage'].get('cost_usd', 0):.5f}")
    print(f"{len(runs)} run(s) with snapshots under {a.root}")


if __name__ == "__main__":
    main()
