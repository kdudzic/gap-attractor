"""Verify that every source claimed in a structured stop statement was actually opened.

Ask the model to end with a stop reason whose full contents are JSON, for example:
    {"brief_complete": true, "sources": ["Montreal Protocol", "Ozone depletion"]}

Usage:
    uv run python scripts/check_claimed_sources.py <run folder>
"""
import argparse
import sys
from pathlib import Path

from gap_attractor.analysis.source_claims import validate_claimed_sources
from gap_attractor.wikipedia import wiki_key


def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_doc = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    steps = [json.loads(line) for line in (args.run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    try:
        claims, opened, missing = validate_claimed_sources(run_doc, steps)
    except ValueError as error:
        print(f"SOURCE CLAIM CHECK FAILED: {error}")
        return 1
    print(f"claimed sources ({len(claims)}): {', '.join(claims)}")
    print(f"opened pages ({len({wiki_key(title) for title in opened})})")
    if missing:
        print(f"SOURCE CLAIM CHECK FAILED: never opened: {', '.join(missing)}")
        return 1
    print("ALL CLAIMED SOURCES WERE OPENED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
