"""Hypothesis tests on clicks (analysis/hypotheses.py) and convergence hop by hop (analysis/convergence.py), on
synthetic walkers whose rule is known: each must be recovered, and the placebos must stay null."""
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from analysis_synth import (add_clicks, category_vectors, make_world, write_clickstream, write_graph, write_page_topics,
                            write_resources, write_runs)
from gap_attractor.analysis import convergence, hypotheses, kernel, walks
from gap_attractor.analysis.clickstream import Clickstream
from gap_attractor.analysis.graph import CoreGraph
from gap_attractor.analysis.logs import load_steps

GROUP = {"llm": "config.provider.model"}
KEYS = ["llm", "schema_major"]
HCFG = OmegaConf.create({"tests": ["similarity", "hub", "human", "novelty", "backtrack", "position", "random"],
                         "references": ["uniform", "clickstream", "indegree"], "primary": ["similarity", "hub", "human"],
                         "placebos": ["random"], "checks": ["position"], "cluster_by": "run",
                         "lambdas": [0.1, 0.5], "n_boot": 400, "seed": 0})
CCFG = OmegaConf.create({"baselines": ["uniform", "indegree"], "walks_per_run": 40, "max_hop": 8, "top_k": 5, "n_splits": 6, "split_by": "run"})


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    w = add_clicks(make_world())
    write_resources(d, w); write_clickstream(d, w); write_page_topics(d, w)
    graph = CoreGraph.load(write_graph(d / "graph", w), keys=False)
    return w, Clickstream.load(d), category_vectors(d), graph


def table_for(tmp_path, world, policy, n_runs=40):
    w, cs, sim, graph = world
    write_runs(tmp_path / policy, w, policy=policy, n_runs=n_runs, n_steps=10)
    steps = load_steps([tmp_path / policy], GROUP)
    ctx = {"X": kernel.page_matrix(steps, sim), "sim": sim, "clickstream": cs, "unlisted": 5,
           "graph": lambda: graph, "indegree": lambda: walks.in_degree(graph)}
    table, bits, audit = hypotheses.run(steps, KEYS, ctx, HCFG)
    return table.set_index(["hypothesis", "reference"]), bits, audit


def test_midranks_average_one_half_with_ties():
    r = hypotheses.midranks(np.array([3.0, 1.0, 3.0, 2.0, 5.0]))
    assert r.tolist() == [0.625, 0.0, 0.625, 0.25, 1.0] and r.mean() == pytest.approx(0.5)
    assert hypotheses.holm([0.01, 0.04, np.nan, 0.03]) == pytest.approx([0.03, 0.06, np.nan, 0.06], nan_ok=True)


def test_a_random_walker_shows_nothing(tmp_path, world):
    t, _, audit = table_for(tmp_path, world, "uniform")
    for h in HCFG.tests:
        row = t.loc[(h, "uniform")]
        assert row.p_cluster > 0.01 and row.p_mc > 0.01 and abs(row.R - 0.5) < 0.06, h
    assert audit[next(iter(audit))]["similarity"]["informative_steps"] > 300


def test_a_topic_sticky_walker_is_found_and_the_placebos_stay_null(tmp_path, world):
    t, bits, _ = table_for(tmp_path, world, "sticky")
    s = t.loc[("similarity", "uniform")]
    assert s.lift_lo > 0 and s.p_cluster < 0.01 and s.role == "primary" and s.p_holm_primary < 0.05
    assert t.loc[("random", "uniform")].p_cluster > 0.01 and t.loc[("random", "uniform")].role == "placebo"
    assert t.loc[("position", "uniform")].p_cluster > 0.01 and t.loc[("position", "uniform")].role == "check"
    b = bits.set_index(["hypothesis", "lambda"])
    assert b.loc[("similarity", 0.5)].bits_vs_uniform > 0


def test_a_human_like_walker_is_explained_by_the_clickstream_reference(tmp_path, world):
    t, _, _ = table_for(tmp_path, world, "clickstream")
    assert t.loc[("human", "uniform")].lift_lo > 0                    # ranks above chance on human clicks...
    assert t.loc[("human", "clickstream")].p_cluster > 0.01           # ...but not beyond what humans would do


def test_a_hub_seeking_walker_is_explained_by_the_indegree_reference(tmp_path, world):
    t, _, _ = table_for(tmp_path, world, "hub")
    assert t.loc[("hub", "uniform")].lift_lo > 0.1 and t.loc[("hub", "uniform")].p_cluster < 0.01
    assert t.loc[("hub", "indegree")].lift < t.loc[("hub", "uniform")].lift


def converge(tmp_path, world, policy):
    w, cs, sim, graph = world
    write_runs(tmp_path / policy, w, policy=policy, n_runs=40, n_steps=8)
    runs = walks.load_paths([tmp_path / policy], GROUP)
    curves, absorb, audit = convergence.run(runs, KEYS, graph, sim, cs, CCFG)
    return curves, absorb.set_index("baseline"), audit


def test_pair_statistics_by_hand():
    V = np.array([[1.0, 0], [1.0, 0], [0, 1.0]])
    assert convergence.pair_similarity(V, np.array([1, 2, 3])) == pytest.approx(2 / 6)   # 2 of 6 ordered pairs alike
    assert convergence.pair_similarity(V, np.array([1, 1, 3])) == pytest.approx(0.0)     # the alike pair shares a start
    assert convergence.pair_same_page(np.array([7, 7, 9]), np.array([1, 2, 3])) == pytest.approx(2 / 6)


def test_random_walkers_do_not_converge_more_than_walks(tmp_path, world):
    curves, absorb, audit = converge(tmp_path, world, "uniform")
    assert absorb.loc["uniform"].p_walks > 0.01
    same = curves[(curves.statistic == "same_page") & (curves.baseline == "uniform")]
    assert (same.p_walks > 0.01).mean() >= 0.75 and same.runs.iloc[0] == 40


def test_an_attractor_is_found_on_held_out_runs(tmp_path, world):
    curves, absorb, audit = converge(tmp_path, world, "attractor")
    assert absorb.loc["uniform"].agent > absorb.loc["uniform"].walks_hi and absorb.loc["uniform"].p_walks < 0.05
    assert absorb.loc["indegree"].p_walks < 0.05                       # not just hubs
    late = curves[(curves.statistic == "same_page") & (curves.baseline == "uniform") & (curves.hop >= 4)]
    assert late.agent.mean() > late.walks_hi.mean() and (late.agent > late.walks).mean() >= 0.6   # single hops are noisy
    top = set(audit["conditions"][next(iter(audit["conditions"]))]["top_pages_all_runs"])
    assert len(top & set(range(5))) >= 4


def test_cli_runs_both_stages(tmp_path):
    from hydra import compose, initialize_config_dir
    from gap_attractor.analysis.cli import CONFIG_DIR, run_analysis
    w = add_clicks(make_world())
    write_runs(tmp_path / "runs", w, policy="sticky", n_runs=20, n_steps=6)
    res = write_resources(tmp_path / "res", w)
    write_clickstream(res, w); write_page_topics(res, w)
    graph = write_graph(tmp_path / "graph", w)
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="analysis", overrides=[
            f"resources={res}", f"data.paths=[{tmp_path / 'runs'}]", f"output={tmp_path / 'out'}", "n_boot=100",
            "group_by={llm: config.provider.model}", f"analysis.graph.dir={graph}", "stages=[hypotheses,convergence]",
            "convergence.walks_per_run=10", "convergence.n_splits=2", "convergence.max_hop=5"])
    out = run_analysis(cfg)
    table = (out / "hypotheses.csv").read_text()
    assert "similarity" in table and "placebo" in table and "check" in table and "p_cluster" in table and (out / "hypotheses_bits.csv").exists()
    assert (out / "convergence.csv").exists() and (out / "absorption.csv").exists()
    assert json.loads((out / "convergence_audit.json").read_text())["runs"] == 20


def test_runs_from_one_start_page_are_one_cluster(tmp_path, world):
    w, _, _, _ = world
    write_runs(tmp_path / "u", w, policy="uniform", n_runs=6, n_steps=3)
    steps = load_steps([tmp_path / "u"], GROUP)
    steps.runs["start_page"] = ["A", "A", "B", "B", "C", "C"]            # replicate runs per start page
    by_start = hypotheses.cluster_labels(steps, "start_page")
    assert len(np.unique(by_start)) == 3 and len(np.unique(hypotheses.cluster_labels(steps, "run"))) == 6
    assert hypotheses.sign_flip_p(np.array([1.0, -1.0]), 2.0, 200, 0) == 1.0
