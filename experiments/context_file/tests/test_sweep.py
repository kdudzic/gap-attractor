"""sweep.py builds the same runs for every condition; only the condition's own overrides differ."""
import json

import sweep


def test_dry_run_lists_every_condition(capsys):
    sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "Paris, Texas", "Silk Road", "Bridge", "--dry-run", "--budget-pages", "5",
                "--page-tokens", "6400", "--steps", "8", "--window", "131072", "--agents", "3", "--embed", "--orders", "random",
                "--exploratory"])
    plan = json.loads(capsys.readouterr().out)
    jobs = plan["conditions"]
    assert set(jobs) == set(sweep.conditions(1, 3)) and plan["budget_tokens"] == 32000
    shared = [o for o in jobs["a_full"] if not o.startswith(("output.directory", "experiment.condition", "experiment.memory"))]
    assert all(set(shared) <= set(j) for j in jobs.values())                     # same model, starts, steps, seed
    assert 'batch.start_pages=["Paris, Texas", "Silk Road", "Bridge"]' in shared
    assert {"ctx.budget_tokens=96000", "ctx.reserve_tokens=12288"} <= set(jobs["j_full"])          # N x the single limit
    assert {"ctx.budget_tokens=32000", "ctx.reserve_tokens=4096", "ctx.bash.joint_workspace=private"} <= set(jobs["j_separate"])
    assert "ctx_edit=bash" in jobs["j_full"] and "ctx_edit=tool" in jobs["j_full_tool"] and "ctx_edit=bash" not in jobs["j_full_tool"]
    assert {"ctx.joint.sharing=notes", "ctx.bash.joint_workspace=private", "ctx.budget_tokens=32000"} <= set(jobs["j_notes"])
    table = sweep.conditions(32000, 3)
    assert set(table["j_strangers"]) - set(table["j_full"]) == {"ctx.attribution=none", "ctx.awareness=strangers"}
    assert "context_embedding=steps" in shared and plan["limit_tokens"] == 32000 - 4096 and plan["start_rule"] == "listed"


def test_starts_are_drawn_by_a_recorded_seeded_rule(tmp_path, capsys):
    pool = tmp_path / "starts.tsv"
    pool.write_text("page_id\ttitle\n" + "".join(f"{i}\tPage {i}\n" for i in range(50)))
    args = ["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--n-starts", "6", "--starts-file", str(pool), "--dry-run", "--conditions", "d_budget"]
    sweep.main(args)
    first = json.loads(capsys.readouterr().out)
    sweep.main(args)
    assert json.loads(capsys.readouterr().out)["starts"] == first["starts"] and len(set(first["starts"])) == 6
    assert first["start_rule"].startswith("6 drawn with seed 9001")
    sweep.main(args + ["--seed", "5"])
    assert json.loads(capsys.readouterr().out)["starts"] != first["starts"]


def test_resume_skips_finished_conditions_and_supersedes_unfinished_ones(tmp_path, monkeypatch):
    import subprocess
    ran = []

    def fake_run(cmd, **kw):                    # run.py is not started; git calls answer empty
        conditions = [c for c in cmd if str(c).startswith("experiment.condition=")]
        ran.extend(conditions)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    args = ["--out", str(tmp_path), "--model", "m", "--reasoning", "default", "--starts", "A", "B", "--conditions", "d_budget,j_full",
            "--orders", "random", "--parallel", "1"]
    sweep.main(args)
    assert sorted(ran) == ["experiment.condition=d_budget", "experiment.condition=j_full"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["results"]["j_full"]["exit_code"] = 1                               # j_full failed half-way
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "j_full").mkdir()
    import pytest
    with pytest.raises(SystemExit, match="already holds a sweep"):
        sweep.main(args)
    ran.clear()
    sweep.main(args + ["--resume"])
    assert ran == ["experiment.condition=j_full"] and not (tmp_path / "j_full").exists()
    assert len(list((tmp_path / "_superseded").iterdir())) == 1
    assert set(json.loads((tmp_path / "manifest.json").read_text())["results"]) == {"d_budget", "j_full"}


def test_rungs_change_one_thing_and_long_uncapped_runs_are_refused(capsys):
    import pytest
    table = sweep.conditions(32000, 3)
    assert set(table["c_tool"]) - set(table["b_format"]) == {"ctx_edit=tool", "ctx.nudges=false"}
    assert set(table["e_bash"]) - set(table["d_budget"]) == {"ctx_edit=bash", "ctx.headers=true", "ctx.bash.workspace=per_step"}
    assert set(table["d_clm"]) ^ set(table["d_budget"]) == {"ctx_prompt=clm", "ctx_prompt=neutral"}
    with pytest.raises(SystemExit, match="uncapped rungs would need"):
        sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "A", "B", "--steps", "30", "--window", "131072", "--dry-run"])
    sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "A", "B", "C", "--steps", "30", "--dry-run",
                "--conditions", "d_budget,e_bash"])                       # no uncapped rung: no window needed
    assert json.loads(capsys.readouterr().out)["git_dirty"] in (True, False)


def test_goal_worlds_take_their_starts_and_targets_from_a_worlds_file(tmp_path, capsys):
    import pytest
    worlds = tmp_path / "worlds.tsv"
    worlds.write_text("world_id\tagent_id\tstart\ttarget\n1\t1\tA\tX\n1\t2\tB\tY\n")
    sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--agents", "2", "--conditions", "j_full,j_strangers", "--goal-setup", "pool",
                "--worlds-file", str(worlds), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    job = plan["conditions"]["j_full"]
    assert {"ctx.goal.task=find", "ctx.goal.setup=pool", "prompt=find", f"ctx.goal.worlds_file={worlds}"} <= set(job)
    assert not any(o.startswith("batch.start_pages") for o in job) and plan["start_rule"].startswith("from ")
    with pytest.raises(SystemExit, match="joint conditions only"):
        sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--conditions", "d_budget", "--goal-setup", "pool", "--worlds-file", str(worlds), "--dry-run"])


def test_the_joint_grid_and_the_essay_and_search_options(capsys):
    import pytest
    sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "A", "B", "--conditions", "joint", "--task", "essay", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert list(plan["conditions"]) == [name for n in sweep.GRID for name in (n, n + "_rr")]       # turn order: a factor
    assert all("task=essay" in job for job in plan["conditions"].values())
    sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--opening", "search", "--n-runs", "6", "--conditions", "j_notes", "--dry-run"])
    job = json.loads(capsys.readouterr().out)["conditions"]["j_notes"]
    assert {"variant=b_search_first", "variant.n_runs=6"} <= set(job) and not any(o.startswith("batch.start_pages") for o in job)
    with pytest.raises(SystemExit, match="needs --goal-setup"):
        sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "A", "B", "C", "--task", "find", "--dry-run"])


def test_starts_can_be_drawn_from_a_plain_list(tmp_path, capsys):
    nodes = tmp_path / "nodes.txt"
    nodes.write_text("".join(f"Page {i}\n" for i in range(30)))
    sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--n-starts", "6", "--starts-file", str(nodes), "--conditions", "d_budget", "--dry-run"])
    assert len(set(json.loads(capsys.readouterr().out)["starts"])) == 6


def test_models_get_their_lowest_reasoning_and_the_same_sampling(capsys):
    h, g = "anthropic/claude-haiku-5.5", "google/gemini-3.8-flash"
    sweep.main(["--out", "/tmp/never", "--model", h, g, "--agents", "2", "--starts", "A", "B", "--conditions", "j_full,j_separate",
                "--task", "free", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    for job in plan["conditions"].values():
        assert {"provider.temperature=1", "provider.top_p=0.5", "prompt=free", f"provider.model={h}",
                "+provider.extra_body.reasoning={enabled: false}",
                f"ctx.joint.agent_providers=[{{model: {h}, extra_body: {{reasoning: {{enabled: false}}}}}}, "
                f"{{model: {g}, extra_body: {{reasoning: {{effort: low}}}}}}]"} <= set(job)
    assert plan["reasoning"] == {h: {"enabled": False}, g: {"effort": "low"}}


def test_an_unknown_model_needs_an_explicit_reasoning_choice():
    import pytest
    with pytest.raises(SystemExit, match="no known lowest reasoning"):
        sweep.main(["--out", "/tmp/never", "--model", "x/unknown", "--starts", "A", "--conditions", "a_path", "--dry-run"])
    with pytest.raises(SystemExit, match="joint world"):
        sweep.main(["--out", "/tmp/never", "--model", "a", "b", "--reasoning", "default", "--starts", "A", "--conditions", "a_path", "--dry-run"])


def test_turn_order_is_a_factor_and_identifiability_a_cell(capsys):
    sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "A", "B", "--conditions",
                "j_full,j_full_anon,d_budget", "--dry-run"])
    jobs = json.loads(capsys.readouterr().out)["conditions"]
    assert list(jobs) == ["j_full", "j_full_rr", "j_full_anon", "j_full_anon_rr", "d_budget"]      # single agents: one order
    assert "ctx.joint.order=seeded_shuffle" in jobs["j_full"] and "ctx.joint.order=fixed" in jobs["j_full_rr"]
    assert set(jobs["j_full_anon"]) - set(jobs["j_full"]) == {"ctx.attribution=none", "experiment.condition=j_full_anon",
                                                              f"output.directory={sweep.Path('/tmp/never/j_full_anon')}"}
    assert "experiment.condition=j_full_rr" in jobs["j_full_rr"] and not any(o.startswith("context_embedding") for o in jobs["d_budget"])


def test_settings_outside_the_study_need_exploratory_and_are_labelled(capsys):
    import pytest
    base = ["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "A", "B", "--conditions", "j_full",
            "--orders", "random", "--dry-run"]
    for extra in (["--steps", "10"], ["--agents", "1"], ["--budget-tokens", "8192"], ["--extra", "ctx.max_edit_turns=3"]):
        with pytest.raises(SystemExit, match="outside the study"):
            sweep.main(base + extra)
    sweep.main(base + ["--extra", "experiment.allow_stop=false", "--extra", "experiment.tracker_owner=X"])   # admissible
    assert "experiment.condition=j_full" in json.loads(capsys.readouterr().out)["conditions"]["j_full"]
    sweep.main(base + ["--steps", "10", "--exploratory"])
    plan = json.loads(capsys.readouterr().out)
    assert "experiment.condition=j_full__exploratory" in plan["conditions"]["j_full"] and plan["exploratory"] == ["--steps 10"]


def test_the_two_find_worlds(tmp_path, capsys):
    worlds = tmp_path / "worlds.tsv"
    worlds.write_text("world_id\tagent_id\tstart\ttarget\n1\t1\tA\tX\n1\t2\tB\tX\n")
    for setup, stop in (("common", "first"), ("assigned", "all_found")):
        sweep.main(["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--conditions", "j_full", "--orders", "random",
                    "--goal-setup", setup, "--worlds-file", str(worlds), "--dry-run"])
        job = json.loads(capsys.readouterr().out)["conditions"]["j_full"]
        assert {f"ctx.goal.setup={setup}", f"ctx.goal.stop={stop}", "prompt=find"} <= set(job)


def test_same_start_and_top_p_are_exploratory_options(capsys):
    import pytest
    base = ["--out", "/tmp/never", "--model", "m", "--reasoning", "default", "--starts", "Cadmium", "--conditions", "j_full",
            "--orders", "round_robin", "--dry-run"]
    with pytest.raises(SystemExit, match="outside the study"):
        sweep.main(base + ["--same-start"])
    sweep.main(base + ["--same-start", "--top-p", "1", "--exploratory"])
    plan = json.loads(capsys.readouterr().out)
    job = plan["conditions"]["j_full_rr"]
    assert "ctx.joint.start_assignment=same" in job and "provider.top_p=1.0" in job and "ctx.joint.order=fixed" in job
    assert plan["exploratory"] == ["--top-p 1.0", "--same-start"]
