#!/usr/bin/env python3
"""Build deterministic foreign-history controls from a completed baseline batch.

For each recipient start page, replies are taken only from the other starts in
the same baseline batch. Donor runs are visited in a deterministic rotation
and concatenated until the requested number of history turns is available.
STOP replies and harness-selected fallback actions are excluded because the
built-in path_only memory would not show either as an earlier assistant reply.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def usable_items(run_dir: Path, run_id: str, start_page: str) -> list[dict]:
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
            "source_start_page": start_page,
            "source_run_id": run_id,
            "source_step": step.get("step_index"),
        })
    return items


def build(batch: Path, turns: int) -> dict:
    rows = list(csv.DictReader((batch / "run_log.csv").open(encoding="utf-8")))
    if len(rows) < 2:
        raise ValueError("at least two completed donor runs are required")
    donors = []
    for row in rows:
        run_dir = Path(row["Artifact Path"])
        items = usable_items(run_dir, row["Run ID"], row["Start Page"])
        donors.append({"start_page": row["Start Page"], "run_id": row["Run ID"], "items": items})

    mapping = {}
    for recipient_index, recipient in enumerate(donors):
        ordered = [donors[(recipient_index + offset) % len(donors)] for offset in range(1, len(donors))]
        selected = [item for donor in ordered for item in donor["items"]]
        if len(selected) < turns:
            raise ValueError(
                f"only {len(selected)} usable foreign turns are available for {recipient['start_page']!r}; "
                f"{turns} required"
            )
        mapping[recipient["start_page"]] = {
            "recipient_baseline_run_id": recipient["run_id"],
            "items": selected[:turns],
        }
    return {
        "schema_version": 1,
        "source_batch": str(batch.resolve()),
        "required_history_turns": turns,
        "construction": "deterministic rotated concatenation of successful non-STOP replies from other starts",
        "mapping": mapping,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_batch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--turns", type=int, required=True)
    args = parser.parse_args()
    document = build(args.baseline_batch.resolve(), args.turns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(document['mapping'])} mappings with {args.turns} turns each to {args.output}")


if __name__ == "__main__":
    main()
