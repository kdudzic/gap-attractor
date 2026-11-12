"""Outcomes stage: condition labels for prompt wording and stop option, per-condition stop rates, paired test."""
import json

import pytest

from gap_attractor.analysis.logs import allow_stop, effective_temperature, memory_setup, prompt_wording, world_setup
from gap_attractor.analysis.outcomes import load_outcomes, mcnemar_exact, paired, summarise, wilson

GROUP = {"llm": "config.provider.model", "prompt": "derived.prompt_wording", "stop": "derived.allow_stop"}
KEYS = list(GROUP) + ["schema_major"]


def write_run(root, run_id, start, stopped_after, experiment, model="m", n_steps=3, ended=None):
    """A run that stops after `stopped_after` moves (None: goes to the step limit, or ends as `ended` after n_steps)."""
    d = root / run_id
    d.mkdir(parents=True)
    steps, n = [], n_steps if stopped_after is None else stopped_after
    for t in range(1, n + 1):
        steps.append({"step_index": t, "action": "follow", "parsed_choice": f"p{t}", "current_page_id": t,
                      "resolution": {"resolved_page_id": t + 1}, "invalid_action": {"is_invalid": False},
                      "token_counts": {"input": 100, "output": 10}})
    if stopped_after is not None:
        steps.append({"step_index": n + 1, "action": "follow", "parsed_choice": "STOP", "model_output_raw": '{"choice": "STOP"}',
                      "resolution": {"resolved_page_id": None}, "invalid_action": {"is_invalid": False},
                      "token_counts": {"input": 100, "output": 10}})
    info = {"schema_version": "3.3", "run_id": run_id, "start_page": start, "variant": "a_hub",
            "config": {"provider": {"model": model}, "experiment": experiment},
            "outcome": {"termination_reason": (ended or "step_limit") if stopped_after is None else "model_stopped",
                        "model_stop_statement": None if stopped_after is None else "done"}}
    (d / "run.json").write_text(json.dumps(info))
    (d / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps) + "\n")


def test_prompt_wording_and_stop_labels_cover_old_runs():
    old = {"config": {"experiment": {"system_prompt": "You are free to do what you want."}}}
    assert prompt_wording(old) == "free" and allow_stop(old) == "true"
    browse = {"config": {"experiment": {"system_prompt": "You have time to browse Wikipedia however you like.  Follow whatever interests you."}}}
    assert prompt_wording(browse) == "browse"                     # whitespace differences do not matter
    assert prompt_wording({"config": {"experiment": {"system_prompt": "Hi."}}}).startswith("custom-")
    new = {"config": {"experiment": {"prompt_wording": "notask", "allow_stop": False}}}
    assert prompt_wording(new) == "notask" and allow_stop(new) == "false"


def test_memory_world_and_temperature_conditions(tmp_path):
    ctx = {"config": {"experiment": {"memory": "context_file"}}, "memory_settings": {"interface": "tool", "budget": 32768}}
    assert memory_setup(ctx) == "context_file|budget=32768,interface=tool"
    assert memory_setup({"config": {"experiment": {"memory": "path_only"}}}) == "path_only"
    assert world_setup({"config": {}}) == "none" and world_setup({"config": {"world": {"name": "solo"}}}) == "solo"
    joint = {"config": {"world": {"name": "joint_labelled"}}, "world": {"world_id": 1, "agent_id": 2, "n_agents": 3, "models": ["b", "a", "b"]}}
    assert world_setup(joint) == "joint_labelled|n=3|a+bx2"
    assert effective_temperature({"config": {"provider": {"temperature": 1}}, "effective_sampling": {"temperature": None}}) == "provider-default"
    assert effective_temperature({"config": {"provider": {"temperature": 1}}, "effective_sampling": {"temperature": 1}}) == "1"
    assert effective_temperature({"effective_sampling": {"temperature": 1, "checked": False}}) == "1-unchecked"
    old = tmp_path / "old"; old.mkdir()                          # 14 Oct runs: only the step logs name dropped params
    (old / "steps.jsonl").write_text(json.dumps({"extras": {"retry_attempts": [{"dropped_params": ["temperature", "seed"]}]}}) + "\n")
    assert effective_temperature({"config": {"provider": {"temperature": 0}}}, old) == "provider-default"
    assert effective_temperature({"config": {"provider": {"temperature": 0}}}) == "0"


def test_summary_and_paired_test(tmp_path):
    free = {"prompt_wording": "free", "allow_stop": True}
    browse = {"prompt_wording": "browse", "allow_stop": True}
    for i in range(6):                                            # free: 5 of 6 stop on the first page
        write_run(tmp_path, f"f{i}", f"S{i}", 0 if i < 5 else None, free)
        write_run(tmp_path, f"b{i}", f"S{i}", 2 if i == 0 else None, browse)   # browse: 1 stops, after 2 moves
    runs = load_outcomes([tmp_path], GROUP)
    table = summarise(runs, KEYS).set_index("prompt")
    assert table.loc["free", "stopped"] == 5 and table.loc["free", "stopped_first_page"] == 5
    assert table.loc["browse", "stopped"] == 1 and table.loc["browse", "median_moves_when_stopped"] == 2
    assert table.loc["browse", "step_limit"] == 5 and table.loc["free", "stop_replies"] == 5
    lo, hi = wilson(5, 6)
    assert table.loc["free", "stop_lo"] == pytest.approx(lo) and lo < 5 / 6 < hi
    p = paired(runs, KEYS, "prompt", "free").iloc[0]
    assert (p.pairs, p.only_baseline, p.only_other) == (6, 4, 0)
    assert p.p_mcnemar == pytest.approx(mcnemar_exact(4, 0)) == pytest.approx(0.125)


def test_competing_endings_and_repeated_runs(tmp_path):
    free = {"prompt_wording": "free", "allow_stop": True}
    browse = {"prompt_wording": "browse", "allow_stop": True}
    for i in range(4):
        write_run(tmp_path, f"f{i}", f"S{i}", None, free, n_steps=1, ended="context_budget")   # free: ended by the budget at step 1
        write_run(tmp_path, f"b{i}", f"S{i}", 3, browse)                                       # browse: stops at step 4
        write_run(tmp_path, f"r{i}", f"S{i}", None, browse)                                    # and a repeat that runs on
    runs = load_outcomes([tmp_path], GROUP)
    table = summarise(runs, KEYS).set_index("prompt")
    assert table.loc["free", "context_budget"] == 4 and table.loc["free", "other_endings"] == 0
    assert table.loc["browse", "stop_hazard"] == pytest.approx(4 / (4 * 4 + 4 * 3))           # 4 stops in 28 steps
    p = paired(runs, KEYS, "prompt", "free").iloc[0]
    assert p.pairs == 4 and p.censored_pages == 4 and p.repeated_runs                        # both runs of each page kept
    assert (p.only_baseline, p.only_other) == (0, 0)          # browse's stops came after step 1, where free was cut off
