"""Analysis package: log parsing, topic vectors, reference models, flows, bootstrap and the CLI on synthetic runs."""
import json

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from analysis_synth import (add_clicks, make_world, write_clickstream, write_graph, write_page_topics,
                            write_resources, write_runs)
from gap_attractor.analysis import kernel
from gap_attractor.analysis.cli import CONFIG_DIR, run_analysis
from gap_attractor.analysis.inference import sign_test
from gap_attractor.analysis.logs import FOLLOW, Steps, load_steps
from gap_attractor.analysis.models import get_model, novelty, uniform
from gap_attractor.analysis.topics import build_topic_map, norm_category

GROUP = {"llm": "config.provider.model"}


def setup(tmp_path, policy="uniform", multi=True, **kw):
    world = make_world(multi=multi)
    write_runs(tmp_path / "runs", world, policy=policy, **kw)
    steps = load_steps([tmp_path / "runs"], GROUP)
    chi = kernel.chi_matrix(steps, world["tm"], "multiplicity")
    return world, steps, chi


def summary(steps, chi, model, n_boot=500):
    u = uniform(steps, chi).logp
    return kernel.compare(steps, chi, get_model(model)(steps, chi, np.full(chi.shape[1], 1 / chi.shape[1])), u,
                          min_row_mass=20, n_boot=n_boot, seed=0)


def test_norm_category_and_rollup_follow_redirects_and_drop_hidden():
    assert norm_category("category:foo_bar") == "Foo bar"
    parents = {"Arts": [], "The arts": [], "Painting": ["Arts"], "Oil": ["Painting", "Tracking"], "Tracking": ["The arts"],
               "Nature": [], "Natural environment": [], "Forests": ["Nature"]}
    tm, info = build_topic_map(parents, {"Arts": "The arts", "Nature": "Natural environment"}, {"Tracking"},
                               ["Arts", "Nature"], max_depth=7)
    assert tm.names == ("The arts", "Natural environment", "unlabeled")
    assert tm.cat_topics["Oil"] == (0,) and tm.cat_topics["Forests"] == (1,) and "Tracking" not in tm.cat_topics
    assert np.allclose(tm.chi(["oil", "Forests", "Forests"]), [1 / 3, 2 / 3, 0])
    assert np.allclose(tm.chi(["Tracking"]), [0, 0, 1])            # hidden only -> unlabeled
    assert np.allclose(tm.chi(["Oil", "Forests", "Forests"], "uniform"), [0.5, 0.5, 0])


def test_parsing_skips_non_decisions_dedupes_menus_and_clusters_worlds(tmp_path):
    world = make_world()
    write_runs(tmp_path / "a", world, n_runs=4, n_steps=5, schema="2.3", extra_steps=True)
    write_runs(tmp_path / "b", world, n_runs=6, n_steps=5, schema="3.1", world_size=3, batch="batch_02_world")
    s = load_steps([tmp_path], GROUP)
    assert s.audit["search_start"] == 4 and s.audit["excluded_skip"] == 4 and s.audit["stop_or_unresolved"] == 4
    assert len(s) == 4 * 4 + 6 * 5 and (s.kind == FOLLOW).all()
    for i in range(len(s)):                                         # 2.3 duplicate rows removed, choice on the menu
        m = s.menu[s.menu_ptr[i]:s.menu_ptr[i + 1]]
        assert len(set(m.tolist())) == len(m) and s.menu[s.chosen[i]] == s.y[i]
    assert sorted(s.runs.schema_major.unique()) == ["2", "3"]
    assert s.runs.cluster.nunique() == 4 + 2                        # runs, plus two worlds of three agents


def test_choice_outside_menu_raises(tmp_path):
    world = make_world()
    folder = write_runs(tmp_path, world, n_runs=1, n_steps=3)
    p = folder / "run000" / "steps.jsonl"
    recs = [json.loads(line) for line in p.read_text().splitlines()]
    recs[1]["resolution"]["resolved_page_id"] = 999999
    p.write_text("\n".join(map(json.dumps, recs)))
    with pytest.raises(ValueError, match="not in their displayed menu"):
        load_steps([tmp_path], GROUP)
    assert load_steps([tmp_path], GROUP, on_choice_outside_menu="drop").audit["dropped_choice_outside_menu"] == 1


def test_single_label_flow_is_the_transition_count_matrix(tmp_path):
    world, steps, chi = setup(tmp_path, multi=False, n_runs=10)
    arr, _ = summary(steps, chi, "uniform", n_boot=50)
    lab = chi.argmax(1)
    counts = np.zeros((chi.shape[1],) * 2)
    np.add.at(counts, (lab[steps.x], lab[steps.y]), 1)
    assert np.allclose(arr["F_agent"], counts)
    rows = counts.sum(1) > 0
    assert np.allclose(arr["A_agent"][rows], counts[rows] / counts[rows].sum(1, keepdims=True))


@pytest.mark.parametrize("model", ["uniform", "novelty", "jump"])
def test_row_sums_match_mass(tmp_path, model):
    world, steps, chi = setup(tmp_path, n_runs=10)
    arr, _ = summary(steps, chi, model, n_boot=50)
    assert np.allclose(arr["F_agent"].sum(1), arr["m"], atol=1e-9) and np.allclose(arr["F_model"].sum(1), arr["m"], atol=1e-9)


def test_hand_checked_menu():
    chi = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])
    s = Steps(None, np.zeros(1, int), np.ones(1, int), np.zeros(1, int), np.array([0]), np.array([3]), np.array([0, 4]),
              np.array([1, 2, 3, 4]), np.array([2]), np.arange(5), [[]] * 5)
    u = uniform(s, chi)
    assert np.allclose(u.w[0], [0.625, 0.375, 0]) and np.isclose(u.logp[0], np.log(0.25))
    # topic 0 mass 2.5, topic 1 mass 1.5 -> p = (1/2.5)/2 for pure-0 links, (1/1.5)/2 for the pure-1 link
    p = np.array([0.2, 0.2, 1 / 3, 0.5 * (0.5 / 2.5 + 0.5 / 1.5)])
    assert np.isclose(p.sum(), 1)
    n = novelty(s, chi)
    assert np.allclose(n.w[0], p @ chi[[1, 2, 3, 4]]) and np.isclose(n.logp[0], np.log(p[2]))


def test_novelty_probabilities_sum_to_one(tmp_path):
    world, steps, chi = setup(tmp_path, n_runs=5)
    pred = novelty(steps, chi)
    assert np.allclose(pred.w.sum(1), 1)


def test_uniform_agent_matches_uniform_and_beats_novelty(tmp_path):
    world, steps, chi = setup(tmp_path, policy="uniform", n_runs=60)
    _, s = summary(steps, chi, "uniform")
    lo, hi = s["sim_excess"]["ci95"]
    assert lo < 0 < hi
    _, nov = summary(steps, chi, "novelty")
    assert nov["bits_per_step_vs_uniform"]["ci95"][1] < 0.02


def test_novelty_agent_is_recognised(tmp_path):
    world, steps, chi = setup(tmp_path, policy="novelty", n_runs=60)
    _, nov = summary(steps, chi, "novelty")
    assert nov["bits_per_step_vs_uniform"]["ci95"][0] > 0


def test_sticky_agent_has_positive_retention_excess(tmp_path):
    world, steps, chi = setup(tmp_path, policy="sticky", n_runs=60)
    _, s = summary(steps, chi, "uniform")
    assert s["sim_excess"]["ci95"][0] > 0 and s["sim_excess"]["sign_test"]["p"] < 0.01


def test_sign_test():
    assert sign_test(np.array([1.0] * 10))["p"] == pytest.approx(2 / 1024)
    assert sign_test(np.array([1.0, -1.0, 0.0]))["p"] == 1.0


def _cli(tmp_path, out):
    world = add_clicks(make_world())
    write_runs(tmp_path / "runs", world, n_runs=12, n_steps=8)
    write_runs(tmp_path / "runs", world, policy="sticky", n_runs=12, n_steps=8, batch="batch_02_sticky")
    res = write_resources(tmp_path / "res", world)
    write_clickstream(res, world); write_page_topics(res, world)
    graph = write_graph(tmp_path / "graph", world)
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="analysis", overrides=[f"resources={res}", f"data.paths=[{tmp_path / 'runs'}]",
                                                         f"output={out}", "n_boot=200", "group_by={llm: config.provider.model}",
                                                         f"analysis.graph.dir={graph}", "walks.per_run=20", "walks.distance=true"])
    return run_analysis(cfg)


def test_cli_end_to_end_is_deterministic(tmp_path):
    a, b = _cli(tmp_path, tmp_path / "out1"), _cli(tmp_path, tmp_path / "out2")
    ra = sorted(a.glob("*/results.json"))
    assert len(ra) == 2 and (a / "summary.csv").exists()
    for fa in ra:
        ja, jb = json.loads(fa.read_text()), json.loads((b / fa.parent.name / "results.json").read_text())
        ja.pop("created"); jb.pop("created")
        assert ja == jb
        assert set(ja["models"]) == {"uniform", "jump", "clickstream"}
        assert ja["counts"]["clickstream"]["link_steps"] == ja["counts"]["link_steps"]
        assert np.load(fa.parent / "flows_uniform.npz")["D_ci"].shape[0] == 2
    wa, wb = (a / "walks.csv").read_text(), (b / "walks.csv").read_text()
    assert wa == wb and "no_revisit" in wa and "end_distance" in wa
