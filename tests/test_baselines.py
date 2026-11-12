"""Baselines with no fitted parameters: the human clickstream chooser and whole-run random walks."""
import numpy as np
import pytest

from analysis_synth import (add_clicks, category_vectors, make_world, write_clickstream, write_graph, write_page_topics,
                            write_resources, write_runs)
from gap_attractor.analysis import walks
from gap_attractor.analysis.clickstream import Clickstream
from gap_attractor.analysis.graph import CoreGraph
from gap_attractor.analysis.logs import load_steps
from gap_attractor.analysis.models import call_model, clickstream_coverage

GROUP = {"llm": "config.provider.model"}
KEYS = ["llm", "schema_major"]


def test_clickstream_counts_and_unlisted_links():
    cs = Clickstream(np.array([7]), np.array([0, 2]), np.array([3, 9]), np.array([30, 12]))
    assert list(cs.counts(7, [9, 3, 4])) == [12, 30, 0]
    p, listed = cs.probs(7, [3, 4], unlisted=5)
    assert listed and p == pytest.approx([30 / 35, 5 / 35])
    p, listed = cs.probs(8, [3, 4])            # a page with no listed clicks: uniform
    assert not listed and p == pytest.approx([0.5, 0.5])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("w")
    w = add_clicks(make_world())
    write_resources(d, w); write_clickstream(d, w); write_page_topics(d, w)
    return w, d, Clickstream.load(d), category_vectors(d), CoreGraph.load(write_graph(d / "graph", w), keys=False)


def ll(tmp_path, world, policy):
    w, _, cs, _, _ = world
    write_runs(tmp_path / policy, w, policy=policy, n_runs=40, n_steps=10)
    steps = load_steps([tmp_path / policy], GROUP)
    chi = np.stack([w["tm"].chi(c or []) for c in steps.page_cats])
    ctx = {"clickstream": cs}
    return {m: np.nanmean(call_model(m, steps, chi, w["tm"].prevalence, ctx).logp) for m in ("uniform", "clickstream")}, steps


def test_a_human_like_agent_is_predicted_by_the_clickstream(tmp_path, world):
    human, steps = ll(tmp_path, world, "clickstream")
    assert human["clickstream"] > human["uniform"] + 0.1
    cov = clickstream_coverage(steps, world[2])
    assert cov["link_steps"] == len(steps) and 0 < cov["chosen_link_listed"] <= 1
    rand, _ = ll(tmp_path, world, "uniform")
    assert rand["uniform"] > rand["clickstream"]


def sim(tmp_path, world, policy, **kw):
    w, _, cs, pages, graph = world
    write_runs(tmp_path / policy, w, policy=policy, n_runs=30, n_steps=10)
    runs = walks.load_paths([tmp_path / policy], GROUP)
    table, audit = walks.simulate(runs, KEYS, graph, pages, cs, per_run=40, **kw)
    return walks.compare(table, KEYS, n_boot=300).set_index(["baseline", "statistic"]), audit


def test_walks_find_nothing_for_a_random_walker(tmp_path, world):
    s, audit = sim(tmp_path, world, "uniform", distance=True)
    assert audit["runs_without_moves"] == 0 and audit["runs"] == 30
    for stat in ("step_similarity", "revisit_rate", "breadth", "drift_from_start", "hub_share", "end_distance"):
        assert s.loc[("uniform", stat)].p_walks > 0.01, stat
    assert ("jump", "step_similarity") not in s.index            # the agent never jumped
    assert s.loc[("no_revisit", "revisit_rate")].walks == 0


def test_walks_detect_a_sticky_walker(tmp_path, world):
    s, _ = sim(tmp_path, world, "sticky")
    row = s.loc[("uniform", "step_similarity")]
    assert row.diff_lo > 0 and row.p_walks < 0.01
