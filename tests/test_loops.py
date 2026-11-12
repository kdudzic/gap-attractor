"""Two-page loops and escapes (analysis/loops.py) on synthetic walkers: a constant escape chance shows no decline,
a walker that bounces more the longer it has bounced (as in-context learning predicts) shows one."""
import pytest
from omegaconf import OmegaConf

from analysis_synth import make_world, write_runs
from gap_attractor.analysis import loops, walks

GROUP = {"llm": "config.provider.model"}
KEYS = ["llm", "schema_major"]
CFG = OmegaConf.create({"max_bounces": 4, "cluster_by": "run"})


@pytest.fixture(scope="module")
def world():
    w = make_world(out_degree=12)
    links = [set(l) for l in w["links"]]
    for p, l in enumerate(w["links"]):                      # reciprocal links, so a walker can go back
        for j in l:
            links[j].add(p)
    w["links"] = [sorted(l) for l in links]
    return w


def run_table(tmp_path, world, policy, n_runs=40, n_steps=15):
    write_runs(tmp_path / policy, world, policy=policy, n_runs=n_runs, n_steps=n_steps)
    runs = walks.load_paths([tmp_path / policy], GROUP)
    table, hazard = loops.summarise(runs, KEYS, CFG, n_boot=400, seed=0)
    return table.iloc[0], hazard, runs


def test_events_by_hand():
    e = loops.path_events([1, 2, 1, 2, 1, 3])
    assert e["bounces"] == 3 and e["bounce_share"] == pytest.approx(3 / 4) and e["first_bounce"] == 2
    assert e["opportunities"] == [(1, False), (2, False), (3, True)]
    assert loops.path_events([1, 1, 2])["bounces"] == 0          # staying put is not a bounce


def test_a_random_walker_rarely_bounces(tmp_path, world):
    row, _, _ = run_table(tmp_path, world, "uniform")
    assert row.bounce_share < 0.15


def test_a_ping_pong_walker_loops_with_a_flat_escape_chance(tmp_path, world):
    row, hazard, _ = run_table(tmp_path, world, "pingpong")
    assert row.bounce_share > 0.3 and row.loop_opportunities > 100
    assert row.decline_lo < 0 < row.decline_hi                  # no trend: loops come from the pages, not the past


def test_a_walker_that_learns_its_own_loop_escapes_less_and_less(tmp_path, world):
    row, hazard, _ = run_table(tmp_path, world, "icl")
    assert row.decline_lo > 0 and row.hazard_1 > row.hazard_ge3
    h = hazard.set_index("bounces").hazard
    assert h["1"] > h[">=4"]


def test_paired_comparison_on_the_same_start_pages(tmp_path, world):
    _, _, a = run_table(tmp_path, world, "uniform", n_runs=20)
    _, _, b = run_table(tmp_path, world, "pingpong", n_runs=20)
    runs = __import__("pandas").concat([a, b], ignore_index=True)
    runs["start_page"] = [f"s{i % 20}" for i in range(len(runs))]
    out = loops.paired(runs, KEYS, "llm", "synthetic-uniform", n_boot=300, seed=0).iloc[0]
    assert out.other == "synthetic-pingpong" and out.start_pages == 20
    assert out.diff_lo > 0 and out.p_sign < 0.01
