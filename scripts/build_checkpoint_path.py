#!/usr/bin/env python3
"""Build a checkpoint-path memory file from one completed prefix run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def usable_items(run_dir: Path) -> tuple[dict, list[dict]]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    items = []
    for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines():
        step = json.loads(line)
        choice = step.get("parsed_choice")
        if not choice or choice == "STOP":
            continue
        if (step.get("invalid_action") or {}).get("harness_handling") == "skip":
            continue
        if (step.get("resolution") or {}).get("resolve_status") != "success":
            continue
        items.append({
            "reasoning": step.get("model_reasoning"),
            "choice": choice,
            "source_start_page": run["start_page"],
            "source_run_id": run["run_id"],
            "source_step": step.get("step_index"),
        })
    if not items:
        raise ValueError(f"prefix run has no usable successful moves: {run_dir}")
    return run, items


def build(run_dir: Path, recipients: list[str]) -> dict:
    run, items = usable_items(run_dir)
    return {
        "schema_version": 1,
        "source_batch": run.get("batch_id"),
        "source_run_id": run["run_id"],
        "source_start_page": run["start_page"],
        "construction": "successful non-STOP prefix replies, excluding harness-selected fallback actions",
        "mapping": {
            recipient: {"source_run_id": run["run_id"], "items": items}
            for recipient in recipients
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recipient", action="append", required=True)
    args = parser.parse_args()
    document = build(args.prefix_run.resolve(), args.recipient)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(document['mapping'])} recipient mapping(s) to {args.output}")


if __name__ == "__main__":
    main()
