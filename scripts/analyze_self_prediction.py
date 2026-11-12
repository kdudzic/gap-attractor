#!/usr/bin/env python3
"""Score stateless self/generic forecasts against the later navigation trajectory."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FRAMINGS = ("self", "generic")
CONTAMINATION_PHRASES = (
    "preregistered forecast",
    "your own likely navigation",
    "typical language-model agent",
)


def normalized(value: str) -> str:
    return " ".join(value.replace("_", " ").split()).casefold()


def unique(values):
    seen, result = set(), []
    for value in values:
        key = normalized(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def score_prediction(record, selected):
    predicted = unique(record.get("pages") or [])
    selected_unique = unique(selected)
    predicted_keys = {normalized(page) for page in predicted}
    hits = [page for page in selected_unique if normalized(page) in predicted_keys]
    time_to_first = next(
        (index for index, page in enumerate(selected, 1) if normalized(page) in predicted_keys),
        None,
    )
    first_click = selected[0] if selected else None
    first_click_rank = next(
        (index for index, page in enumerate(predicted, 1) if normalized(page) == normalized(first_click or "")),
        None,
    )
    return {
        "predicted_pages": predicted,
        "predicted_topics": record.get("topics") or [],
        "confidence": record.get("confidence"),
        "exact_hits": hits,
        "exact_precision_at_5": len(hits) / len(predicted) if predicted else None,
        "first_click_hit": first_click_rank is not None,
        "first_click_rank_in_prediction": first_click_rank,
        "time_to_first_exact_hit": time_to_first,
        "parse_ok": bool(record.get("parse_ok")),
    }


def navigation_requests_contain_prediction_text(run_dir: Path):
    matches = []
    for path in sorted((run_dir / "model_inputs").glob("step_*.txt")):
        if path.name.startswith("step_000"):
            continue
        text = path.read_text(encoding="utf-8").casefold()
        for phrase in CONTAMINATION_PHRASES:
            if phrase in text:
                matches.append({"file": str(path), "phrase": phrase})
    return matches


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def common_prefix(paths):
    result = []
    for values in zip(*paths):
        if len({normalized(value) for value in values}) != 1:
            break
        result.append(values[0])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    run_dirs = sorted(path for path in batch.iterdir() if path.is_dir() and (path / "run.json").exists())
    if not run_dirs:
        raise SystemExit(f"no run directories in {batch}")

    runs = []
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        predictions = json.loads((run_dir / "pre_run_prediction.json").read_text(encoding="utf-8"))
        steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        selected = [
            step.get("parsed_choice")
            for step in steps
            if step.get("parsed_choice") not in (None, "STOP")
        ]
        records = {record["framing"]: record for record in predictions["records"]}
        scores = {framing: score_prediction(records[framing], selected) for framing in FRAMINGS}
        self_pages = {normalized(page) for page in scores["self"]["predicted_pages"]}
        generic_pages = {normalized(page) for page in scores["generic"]["predicted_pages"]}
        union = self_pages | generic_pages
        runs.append({
            "run_id": summary["run_id"],
            "run_index": summary["run_index"],
            "run_seed": json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["run_seed"],
            "status": summary["status"],
            "termination_reason": summary["termination_reason"],
            "n_steps": summary["n_steps"],
            "invalid_actions": summary["invalid_actions"],
            "prediction_order": [record["framing"] for record in predictions["records"]],
            "selected_first": selected[0] if selected else None,
            "selected_unique_pages": len(unique(selected)),
            "selected_prefix": selected[:5],
            "scores": scores,
            "self_generic_page_jaccard": len(self_pages & generic_pages) / len(union) if union else None,
            "navigation_contamination_matches": navigation_requests_contain_prediction_text(run_dir),
            "input_tokens": summary["input_tokens"],
            "output_tokens": summary["output_tokens"],
            "reasoning_tokens": summary["reasoning_tokens"],
            "cost_usd": summary["cost_usd"],
            "artifact_path": summary["artifact_path"],
        })

    by_framing = {
        framing: {
            "mean_exact_precision_at_5": mean([run["scores"][framing]["exact_precision_at_5"] for run in runs]),
            "first_click_hits": sum(run["scores"][framing]["first_click_hit"] for run in runs),
            "runs_with_any_exact_hit": sum(bool(run["scores"][framing]["exact_hits"]) for run in runs),
            "mean_first_click_rank_when_hit": mean([
                run["scores"][framing]["first_click_rank_in_prediction"] for run in runs
                if run["scores"][framing]["first_click_hit"]
            ]),
            "mean_time_to_first_exact_hit": mean([
                run["scores"][framing]["time_to_first_exact_hit"] for run in runs
            ]),
        }
        for framing in FRAMINGS
    }
    deltas = [
        run["scores"]["self"]["exact_precision_at_5"] - run["scores"]["generic"]["exact_precision_at_5"]
        for run in runs
    ]
    selected_sets = []
    selected_paths = []
    for run_dir in run_dirs:
        steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
        path = [step.get("parsed_choice") for step in steps if step.get("parsed_choice") not in (None, "STOP")]
        selected_paths.append(path)
        selected_sets.append({normalized(page) for page in path})
    pairwise_selected_jaccards = [
        len(selected_sets[left] & selected_sets[right]) / len(selected_sets[left] | selected_sets[right])
        for left in range(len(selected_sets))
        for right in range(left + 1, len(selected_sets))
    ]
    output = {
        "analysis": "self_prediction_exact_page_pilot",
        "batch": str(batch),
        "n_runs": len(runs),
        "smoke_excluded": True,
        "primary_metric": "per-run exact-page precision@5 over selected navigation pages; paired self minus generic",
        "runs": runs,
        "aggregate": {
            "technical_gate_pass": (
                len(runs) == 3
                and all(run["status"] == "Completed" and run["n_steps"] == 30 for run in runs)
                and all(run["scores"][framing]["parse_ok"] for run in runs for framing in FRAMINGS)
                and all(not run["navigation_contamination_matches"] for run in runs)
            ),
            "all_runs_completed_30_steps": all(run["status"] == "Completed" and run["n_steps"] == 30 for run in runs),
            "all_six_predictions_parsed": all(
                run["scores"][framing]["parse_ok"] for run in runs for framing in FRAMINGS
            ),
            "navigation_contamination_audit_pass": all(
                not run["navigation_contamination_matches"] for run in runs
            ),
            "by_framing": by_framing,
            "paired_self_minus_generic_precision_deltas": deltas,
            "mean_paired_precision_delta": mean(deltas),
            "mean_self_generic_page_jaccard": mean([run["self_generic_page_jaccard"] for run in runs]),
            "first_clicks": [run["selected_first"] for run in runs],
            "common_selected_prefix": common_prefix(selected_paths),
            "common_selected_pages": sorted(set.intersection(*selected_sets)),
            "pairwise_selected_page_jaccards": pairwise_selected_jaccards,
            "mean_pairwise_selected_page_jaccard": mean(pairwise_selected_jaccards),
            "total_cost_usd": sum(run["cost_usd"] for run in runs),
            "total_input_tokens": sum(run["input_tokens"] for run in runs),
            "total_output_tokens": sum(run["output_tokens"] for run in runs),
            "total_reasoning_tokens": sum(run["reasoning_tokens"] for run in runs),
        },
        "interpretation_guardrail": (
            "Descriptive n=3 pilot only. Exact page-set overlap cannot establish self-knowledge, an attractor, "
            "curiosity, or a latent mental state; local graph salience and training familiarity remain alternatives."
        ),
    }
    destination = args.output or batch / "self_prediction_pilot_summary.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
