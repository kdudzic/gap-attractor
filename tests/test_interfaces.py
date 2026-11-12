"""Pluggable components, start-page sources, batch.index=auto, early validation and the Hydra CLI.

None of these may change what a default run logs; test_fixture_harness.py covers the log contents."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gap_attractor.harness.runner import make_backend, read_start_titles, run_experiment, start_pages
from gap_attractor.providers import MockProvider, make_provider
from gap_attractor.registry import PROVIDERS, Registry
from gap_attractor.wikipedia import FixtureBackend

from test_fixture_harness import ROOT, config


def run_titles(dataset):
    return [json.loads((d / "run.json").read_text(encoding="utf-8"))["start_page"]
            for d in sorted(dataset.root.iterdir()) if (d / "run.json").exists()]


def test_registry_resolves_names_and_dotted_paths():
    registry = Registry("thing")
    registry.register("one")(lambda: 1)
    assert registry.get("one")() == 1
    assert registry.get("os.path.join") is os.path.join
    with pytest.raises(ValueError, match="unknown thing: 'two'"):
        registry.get("two")
    with pytest.raises(ValueError, match="cannot import thing"):
        registry.get("no_such_module.factory")


def test_builtin_providers_are_registered(tmp_path):
    assert {"mock", "openai", "groq", "gemini", "openrouter"} <= set(PROVIDERS.names())
    assert isinstance(make_provider(config(tmp_path).provider), MockProvider)


def test_provider_and_backend_plugins_by_dotted_path(tmp_path, monkeypatch):
    calls = []
    plugin = types.ModuleType("fake_plugin")
    plugin.provider = lambda cfg: calls.append("provider") or MockProvider(cfg.model)
    plugin.backend = lambda wcfg, root: calls.append("backend") or FixtureBackend(Path(root) / wcfg.fixture_path)
    monkeypatch.setitem(sys.modules, "fake_plugin", plugin)
    cfg = config(tmp_path, extra=["provider.name=fake_plugin.provider", "wikipedia.backend=fake_plugin.backend"])
    _, results = run_experiment(cfg, ROOT)
    assert calls == ["backend", "provider"] and results[0]["n_steps"] == 3


def test_start_pages_tsv_column_and_runs_per_start(tmp_path):
    tsv = tmp_path / "starts.tsv"
    tsv.write_text("page_id\ttitle\ttopic\n1\tCivic Atlas\tA\n2\tArchive\tB\n3\tBridge\tC\n", encoding="utf-8")
    assert read_start_titles(tsv) == ["Civic Atlas", "Archive", "Bridge"]
    cfg = config(tmp_path, extra=[f"variant.start_pages_file={tsv.as_posix()}", "variant.n_runs=2", "variant.runs_per_start=3"])
    starts = start_pages(cfg)
    assert len(starts) == 6 and starts[0] == starts[1] == starts[2] and starts[3] == starts[4] == starts[5]
    dataset, _ = run_experiment(cfg, ROOT)
    assert run_titles(dataset) == starts
    seeds = [json.loads((d / "config.json").read_text(encoding="utf-8"))["run_seed"] for d in sorted(dataset.root.iterdir()) if d.is_dir()]
    assert len(set(seeds)) == 6  # repeats still get their own run seed


def test_tsv_without_title_column_fails(tmp_path):
    tsv = tmp_path / "starts.tsv"
    tsv.write_text("page_id\tname\n1\tCivic Atlas\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no column 'title'"):
        read_start_titles(tsv)


def test_txt_start_file_is_sampled_exactly_as_before(tmp_path):
    txt = tmp_path / "hubs.txt"
    txt.write_text("# comment\nCivic Atlas\nArchive\n\nBridge\nCanal\n", encoding="utf-8")
    import random
    expected = random.Random(9001).sample(["Civic Atlas", "Archive", "Bridge", "Canal"], 3)
    assert start_pages(config(tmp_path, extra=[f"variant.start_pages_file={txt.as_posix()}", "variant.n_runs=3"])) == expected


def test_custom_start_page_source(tmp_path, monkeypatch):
    plugin = types.ModuleType("fake_starts")
    plugin.pick = lambda cfg: ["Archive", "Bridge"]
    monkeypatch.setitem(sys.modules, "fake_starts", plugin)
    assert start_pages(config(tmp_path, extra=["variant.start_pages_source=fake_starts.pick"])) == ["Archive", "Bridge"]


def test_batch_index_auto_takes_next_free_number(tmp_path):
    first, _ = run_experiment(config(tmp_path, extra=["batch.index=auto", "experiment.step_limit=1"]), ROOT)
    second, _ = run_experiment(config(tmp_path, extra=["batch.index=auto", "experiment.step_limit=1"]), ROOT)
    assert first.batch_id.startswith("batch_01_") and second.batch_id.startswith("batch_02_")
    logged = json.loads((second.root / "batch.json").read_text(encoding="utf-8"))
    assert logged["batch_index"] == 2 and logged["config"]["batch"]["index"] == 2


@pytest.mark.parametrize("override, message", [
    ("experiment.display_order_policy=shufle", "unknown display_order_policy"),
    ("provider.name=nope", "unknown provider"),
    ("wikipedia.backend=nope", "unknown wikipedia backend"),
    ("batch.index=first", "batch.index must be"),
    ("variant.runs_per_start=0", "runs_per_start must be >= 1"),
])
def test_bad_settings_fail_before_anything_is_written(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        run_experiment(config(tmp_path, extra=[override]), ROOT)
    assert not tmp_path.exists() or not any(tmp_path.rglob("*.json"))


def test_bad_search_settings_fail_early(tmp_path):
    with pytest.raises(ValueError, match="variant.search.display_order"):
        run_experiment(config(tmp_path, variant="b_search_first", extra=["variant.search.display_order=ranked"]), ROOT)


def _cli(tmp_path, *args):
    env = {**os.environ, "GAP_ATTRACTOR_OUTPUT": str(tmp_path / "out"), "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run([sys.executable, "-m", "gap_attractor.cli", *args], cwd=tmp_path, env=env, capture_output=True, text=True)


def test_cli_multirun_sweeps_and_writes_no_hydra_files(tmp_path):
    fixture = (ROOT / "tests/fixtures/wiki/fixture_pages.json").as_posix()  # relative paths resolve against the cwd
    proc = _cli(tmp_path, "-m", "variant=a_hub,c_search_anytime", "experiment.step_limit=1", "batch.index=auto",
                f"wikipedia.fixture_path={fixture}")
    assert proc.returncode == 0, proc.stderr
    batches = sorted(p.name for p in (tmp_path / "out" / "smoketest_agent_traversal").iterdir())
    assert len(batches) == 2 and batches[0].startswith("batch_01_") and batches[1].startswith("batch_02_")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]  # no .hydra/, multirun/ or *.log in the cwd
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["_sweeps", "smoketest_agent_traversal"]
    assert len(list((tmp_path / "out" / "_sweeps").glob("*/multirun.yaml"))) == 1


def test_cli_prints_composed_config(tmp_path):
    proc = _cli(tmp_path, "--cfg", "job", "variant=b_search_first")
    assert proc.returncode == 0 and "b_search_first" in proc.stdout
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("provider", ["mock", "openai", "groq", "openrouter"])
@pytest.mark.parametrize("wikipedia", ["fixture", "dump"])
@pytest.mark.parametrize("variant", ["a_hub", "b_search_first", "c_search_anytime", "d_search_first_anytime"])
def test_every_config_group_combination_composes(provider, wikipedia, variant, tmp_path):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"provider={provider}", f"wikipedia={wikipedia}", f"variant={variant}"])
    from omegaconf import OmegaConf
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["provider"]["name"] == provider and resolved["variant"]["name"] == variant
    if wikipedia == "dump":
        assert resolved["wikipedia"]["lookup_format"] in ("sqlite", "pickle") and resolved["wikipedia"]["variant"] == "full"
    if provider != "mock":
        assert resolved["provider"]["max_tokens"] == (4000 if provider == "openrouter" else 700) and resolved["provider"]["extra_body"] == {}
