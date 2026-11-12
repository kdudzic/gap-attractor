"""Check that step_XXX.json files are filled as intended, and rebuild what the model saw.

Usage (from the repo root):
    uv run --env-file .env python scripts/check_steps.py                 # latest run (GAP_ATTRACTOR_OUTPUT or outputs/)
    uv run --env-file .env python scripts/check_steps.py <run folder>    # a specific run
    uv run --env-file .env python scripts/check_steps.py --show 2        # also print the full prompt of step 2

For every step it saves the rebuilt model input to <run folder>/observations/step_XXX.txt.
"""
import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from gap_attractor.harness.runner import link_mask_spec, make_backend, page_view

CANDIDATE_FIELDS = {"original_position", "shuffled_position", "title", "hyperlink", "page_id", "semantic_cluster_label"}
LINK_FIELDS = CANDIDATE_FIELDS | {"link_title", "via_redirect"}  # schema 3.0: page links keep the link as written
RESULT_FIELDS = CANDIDATE_FIELDS | {"snippet", "snippet_source", "matched_title", "via_redirect"}
INTENT_FIELDS = {"text", "why", "raw", "parse_ok", "mapped_title", "mapped_page_id", "match_method", "top_results", "in_links",
                 "intended_shuffled_position", "intended_original_position", "is_current_page", "already_visited",
                 "choice_matches_intent", "tokens", "latency_s", "provider_usage", "internal_reasoning"}
TOP_LEVEL = {"run_id", "step_index", "timestamp", "dwell_time", "action", "search", "intent", "current_page_id", "current_page_title", "choice_set",
             "model_reasoning", "model_internal_reasoning", "model_note", "model_output_raw", "parsed_choice", "invalid_action", "token_counts", "resolution",
             "extras", "semantic_cluster_label"}
OPTIONAL_TOP_LEVEL = {"memory"}   # schema 3.4+: fields a plug-in memory logs per step
REMOVED = {"available_nodes", "candidate_table", "displayed_candidates", "eligible_candidates_source_order",
           "choice_set_size", "chosen_page_id"}
REMOVED_EXTRAS = {"rendered_request", "raw_links_source_order", "article_source_wikitext", "chosen_candidate_key",
                  "chosen_position_norm"}
MUST_NOT_REACH_MODEL = ("[[", "]]", "{{", "}}", "http://", "https://", "<ref", "original_position",
                        "shuffled_position", "displayed_position", "page_id", "semantic_cluster_label")


def latest_run(root=None):
    import os
    root = Path(root or os.environ.get("GAP_ATTRACTOR_OUTPUT") or "outputs")
    runs = [p.parent for p in root.rglob("steps.jsonl")]
    if not runs:
        sys.exit(f"no runs found under {root.resolve()}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def check_step(step, problems, expected_mask=None):
    name = f"step {step['step_index']}"
    missing, extra = TOP_LEVEL - set(step), set(step) - TOP_LEVEL - OPTIONAL_TOP_LEVEL
    if missing: problems.append(f"{name}: missing keys {sorted(missing)}")
    if extra: problems.append(f"{name}: unexpected keys {sorted(extra)}")
    for key in REMOVED & set(step): problems.append(f"{name}: duplicate '{key}' is back")
    for key in REMOVED_EXTRAS & set(step.get("extras", {})): problems.append(f"{name}: duplicate 'extras.{key}' is back")

    rows = step["choice_set"]
    n = len(rows)
    if n != step["extras"]["n_links_offered"]: problems.append(f"{name}: {n} candidates but n_links_offered={step['extras']['n_links_offered']}")
    for row in rows:
        if set(row) != LINK_FIELDS:
            problems.append(f"{name}: candidate fields {sorted(set(row))}"); break
    if [r["shuffled_position"] for r in rows] != list(range(n)): problems.append(f"{name}: shuffled_position is not 0..{n - 1} in list order")
    if sorted(r["original_position"] for r in rows) != list(range(n)): problems.append(f"{name}: original_position is not a permutation of 0..{n - 1}")
    if len({r["title"] for r in rows}) != n: problems.append(f"{name}: duplicate titles in choice_set")
    if step["extras"]["display_order_policy"] == "source" and any(r["original_position"] != r["shuffled_position"] for r in rows):
        problems.append(f"{name}: policy is 'source' but order differs from the page")
    if expected_mask is not None and step["current_page_id"] is not None:
        mask = (step["extras"].get("candidate_filter_provenance") or {}).get("link_mask")
        if mask is None:
            problems.append(f"{name}: missing candidate_filter_provenance.link_mask")
        else:
            for key, expected in (("mode", expected_mask["mode"]), ("mask_id", expected_mask["mask_id"]),
                                  ("mask_spec_sha256", expected_mask["sha256"])):
                if mask.get(key) != expected: problems.append(f"{name}: link mask {key}={mask.get(key)!r}, expected {expected!r}")
            masked = mask.get("masked_candidates") or []
            if mask.get("masked_count") != len(masked): problems.append(f"{name}: masked_count does not match masked_candidates")
            if step["extras"].get("n_links_masked") != len(masked): problems.append(f"{name}: n_links_masked does not match masked_candidates")
            if step["extras"].get("n_links_eligible_before_mask") != n + len(masked):
                problems.append(f"{name}: eligible-before-mask count does not equal offered + masked")
            masked_titles = {row.get("title") for row in masked}
            if masked_titles & {row["title"] for row in rows}: problems.append(f"{name}: a masked title remains in choice_set")
            if mask.get("mode") in ("visited", "visited_all") and any(not row.get("was_visited") for row in masked):
                problems.append(f"{name}: visited mask removed an unvisited candidate")
            if mask.get("mode") in ("sham_visited_count", "sham_visited_all_count"):
                if any(row.get("was_visited") for row in masked): problems.append(f"{name}: sham mask removed a visited candidate")
                if len(masked) + int(mask.get("sham_shortfall") or 0) != int(mask.get("visited_candidate_count") or 0):
                    problems.append(f"{name}: sham mask did not match the visited-candidate count")
            if mask.get("mode") == "semantic_visited" and any(not row.get("is_semantic_target") for row in masked):
                problems.append(f"{name}: semantic mask removed a non-target candidate")
            if mask.get("mode") == "sham_semantic_count":
                if any(row.get("is_semantic_target") for row in masked):
                    problems.append(f"{name}: semantic sham removed a target candidate")
                if len(masked) + int(mask.get("sham_shortfall") or 0) != int(mask.get("semantic_target_candidate_count") or 0):
                    problems.append(f"{name}: semantic sham did not match the semantic-target count")

    options = rows
    if step["action"] in ("search", "search_start"):
        search = step.get("search") or {}
        search_options = search.get("results") or []
        options = search_options or rows
        if not search.get("query"): problems.append(f"{name}: search step without a query")
        for row in search_options:
            if set(row) != RESULT_FIELDS:
                problems.append(f"{name}: search result fields {sorted(set(row))}"); break
        if [r["shuffled_position"] for r in search_options] != list(range(len(search_options))): problems.append(f"{name}: search results out of shown order")
        if expected_mask is not None and expected_mask["mode"] in ("visited_all", "sham_visited_all_count") and step["action"] == "search":
            search_mask = (step["extras"].get("candidate_filter_provenance") or {}).get("search_mask")
            if search_mask is None:
                problems.append(f"{name}: missing candidate_filter_provenance.search_mask")
            else:
                for key, expected in (("mode", expected_mask["mode"]), ("mask_id", expected_mask["mask_id"]),
                                      ("mask_spec_sha256", expected_mask["sha256"])):
                    if search_mask.get(key) != expected:
                        problems.append(f"{name}: search mask {key}={search_mask.get(key)!r}, expected {expected!r}")
                masked = search_mask.get("masked_candidates") or []
                if step["extras"].get("n_search_results_masked") != len(masked):
                    problems.append(f"{name}: n_search_results_masked does not match search mask")
                if step["extras"].get("n_search_results_eligible_before_mask") != len(search.get("results") or []) + len(masked):
                    problems.append(f"{name}: search eligible-before-mask count does not equal shown + masked")
                if expected_mask["mode"] == "visited_all" and any(not row.get("was_visited") for row in masked):
                    problems.append(f"{name}: visited_all search mask removed an unvisited result")
                if expected_mask["mode"] == "sham_visited_all_count":
                    if any(row.get("was_visited") for row in masked):
                        problems.append(f"{name}: search sham removed a visited result")
                    if len(masked) + int(search_mask.get("sham_shortfall") or 0) != int(search_mask.get("visited_candidate_count") or 0):
                        problems.append(f"{name}: search sham did not match the visited-result count")
    elif step.get("search") is not None:
        problems.append(f"{name}: follow step has a search record")

    intent = step.get("intent")
    if intent is not None:
        if set(intent) != INTENT_FIELDS: problems.append(f"{name}: intent fields {sorted(set(intent))}")
        elif intent["in_links"] and rows and rows[intent["intended_shuffled_position"]]["title"] != intent["mapped_title"]:
            problems.append(f"{name}: intended_shuffled_position does not point at the intended link")
    if step["action"] == "intent_jump":
        if intent is None or step["parsed_choice"] != intent["mapped_title"]: problems.append(f"{name}: intent_jump does not go to the mapped intent")
        return n

    choice, position = step["parsed_choice"], step["extras"]["chosen_position"]
    if choice not in (None, "STOP"):
        if position is None or not 0 <= position < len(options): problems.append(f"{name}: chosen_position={position} out of range")
        else:
            chosen = options[position]
            if chosen["title"] != choice: problems.append(f"{name}: option[{position}] is '{chosen['title']}', parsed_choice is '{choice}'")
            if chosen["original_position"] != step["extras"]["chosen_original_position"]: problems.append(f"{name}: chosen_original_position mismatch")
            if step["resolution"]["resolve_status"] == "success" and chosen["page_id"] != step["resolution"]["resolved_page_id"]:
                problems.append(f"{name}: resolved_page_id {step['resolution']['resolved_page_id']} != chosen page_id {chosen['page_id']}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--show", type=int, help="print the full rebuilt prompt of this step number")
    args = parser.parse_args()
    run_dir = args.run_dir or latest_run()
    print(f"run: {run_dir}\n")

    step_files = sorted((run_dir / "steps").glob("step_*.json"))
    steps = [json.loads(f.read_text(encoding="utf-8")) for f in step_files]
    jsonl = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    problems = []
    if steps != jsonl: problems.append("steps/step_XXX.json and steps.jsonl disagree")
    for name in ("config.json", "run.json", "summary.json", "metrics.json", "trajectory.jsonl", "pages.csv", "transcript.md"):
        if not (run_dir / name).exists(): problems.append(f"missing file {name}")

    cfg = OmegaConf.create(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    run_doc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    specs = list(cfg.experiment.get("readouts") or [])
    if specs:
        path = run_dir / "readouts.json"
        if not path.exists():
            problems.append("configured post-run readouts but readouts.json is missing")
        else:
            readouts = json.loads(path.read_text(encoding="utf-8")).get("readouts") or []
            if readouts != (run_doc.get("readouts") or []): problems.append("readouts.json and run.json readouts disagree")
            if len(readouts) != len(specs): problems.append(f"expected {len(specs)} readouts, got {len(readouts)}")
            if readouts:
                candidates, seed = readouts[0].get("candidate_titles"), readouts[0].get("paired_sampling_seed")
                for readout in readouts:
                    if readout.get("candidate_titles") != candidates: problems.append("paired readouts have different candidate lists")
                    if readout.get("paired_sampling_seed") != seed: problems.append("paired readouts have different sampling seeds")
                    selected = [row.get("title") for row in readout.get("selections") or []]
                    if readout.get("valid") and (len(selected) != int(cfg.experiment.readout_selection_count) or
                                                  len(set(selected)) != len(selected) or not set(selected) <= set(candidates or [])):
                        problems.append(f"readout {readout.get('id')}: valid flag conflicts with selections")
    if str(cfg.experiment.get("memory")) == "collection_path_only":
        path = run_dir / "collection.json"
        if not path.exists():
            problems.append("collection_path_only run is missing collection.json")
        else:
            collection = json.loads(path.read_text(encoding="utf-8"))
            if collection != run_doc.get("collection"): problems.append("collection.json and run.json collection disagree")
            target = int(cfg.experiment.collection_target)
            selected = collection.get("selections") or []
            titles = [row.get("title") for row in selected]
            if collection.get("target") != target: problems.append("collection target differs from config")
            if len(selected) > target or len(set(titles)) != len(titles): problems.append("collection selections are not unique or exceed target")
            if [row.get("selection_index") for row in selected] != list(range(1, len(selected) + 1)):
                problems.append("collection selection indexes are not 1..N")
            by_step = {step["step_index"]: step for step in steps}
            for row in selected:
                step = by_step.get(row.get("step_index"))
                if step is None or row.get("title") != step.get("current_page_title") or row.get("page_id") != step.get("current_page_id"):
                    problems.append(f"collection selection {row.get('selection_index')} does not match its recorded step")
            if bool(collection.get("complete")) != (len(selected) == target):
                problems.append("collection complete flag conflicts with selection count")
    expected_mask = link_mask_spec(cfg) if cfg.experiment.get("link_mask_mode") is not None else None
    backend = make_backend(cfg, Path.cwd())
    out_dir = run_dir / "observations"; out_dir.mkdir(exist_ok=True)
    max_chars = cfg.experiment.article_max_chars

    print(f"{'step':>4}  {'action':<12} {'page':<28} {'links':>5}  {'chose':<28} {'shown@':>6} {'page@':>6}  prompt size")
    for step in steps:
        n = check_step(step, problems, expected_mask)
        if step["current_page_id"] is None:  # Variant B search step: no page shown
            print(f"{step['step_index']:>4}  {step['action']:<12} {'[search] ' + str(step['search']['query']):<28} {'-':>5}  {str(step['parsed_choice'])[:28]:<28} "
                  f"{str(step['extras']['chosen_position']):>6} {str(step['extras']['chosen_original_position']):>6}  {len(step['search']['results'])} results")
            continue
        page = backend.load_page(step["current_page_title"])
        if page is None:
            problems.append(f"step {step['step_index']}: page '{step['current_page_title']}' not found in the dump"); continue
        observation, _ = page_view(page, step["choice_set"], None if max_chars is None else int(max_chars))
        (out_dir / f"step_{step['step_index']:03d}.txt").write_text(observation, encoding="utf-8")
        for token in MUST_NOT_REACH_MODEL:
            if token in observation: problems.append(f"step {step['step_index']}: '{token}' appears in what the model saw")
        shown = [line[2:] for line in observation.split("## Candidate links", 1)[1].splitlines() if line.startswith("- ")]
        if step["action"] != "intent_jump" and shown != [r["title"] for r in step["choice_set"]]: problems.append(f"step {step['step_index']}: shown titles differ from choice_set order")
        article = observation.split("## Article content", 1)[1].split("## Candidate links", 1)[0].strip()
        ex = step["extras"]
        print(f"{step['step_index']:>4}  {step['action']:<12} {step['current_page_title'][:28]:<28} {n:>5}  {str(step['parsed_choice'])[:28]:<28} "
              f"{str(ex['chosen_position']):>6} {str(ex['chosen_original_position']):>6}  "
              f"article {len(article):,} chars + {len(shown)} titles")

    if args.show is not None and (out_dir / f"step_{args.show:03d}.txt").exists():
        text = (out_dir / f"step_{args.show:03d}.txt").read_text(encoding="utf-8")
        print(f"\n===== what the model saw at step {args.show} (user message) =====\n{text[:3000]}"
              + (f"\n... [{len(text) - 3000:,} more chars, full text in observations/step_{args.show:03d}.txt]" if len(text) > 3000 else ""))
        print(f"\n===== system message =====\n{cfg.experiment.system_prompt}\n\n{cfg.experiment.response_rule}")

    print("\nshown@ = position in the list the model saw; page@ = position on the wiki page (for searches: search rank)")
    print(f"\n{'ALL CHECKS PASSED' if not problems else f'{len(problems)} PROBLEM(S):'}")
    for problem in problems: print("  -", problem)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
