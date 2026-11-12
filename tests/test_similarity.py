"""Similarity options (categories | embeddings | none), embedders, page text, the embedding cache and the graph plug-in."""
import json

import numpy as np
import pandas as pd
import pytest

from analysis_synth import (add_clicks, analysis_cfg, make_world, write_clickstream, write_graph, write_leads,
                            write_page_topics, write_resources, write_runs)
from gap_attractor.analysis import walks
from gap_attractor.analysis.cli import run_analysis
from gap_attractor.analysis.embedding import EMBEDDERS, EmbeddingCache
from gap_attractor.analysis.graph import load_graph, save_graph
from gap_attractor.analysis.similarity import NoVectors, PageVectors, make_similarity

GROUP = {"llm": "config.provider.model"}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    w = add_clicks(make_world())
    write_resources(d, w); write_clickstream(d, w); write_page_topics(d, w); write_leads(d, w)
    write_runs(d / "runs", w, policy="uniform", n_runs=12, n_steps=8)
    write_runs(d / "runs", w, policy="sticky", n_runs=12, n_steps=8, batch="batch_02_sticky")
    return w, d, write_graph(d / "graph", w)


def run(setup, tmp_path, *overrides):
    w, d, graph = setup
    cfg = analysis_cfg(d, [f"data.paths=[{d / 'runs'}]", f"output={tmp_path}", "n_boot=200", "group_by={llm: config.provider.model}",
                           f"analysis.graph.dir={graph}", "walks.per_run=20", f"++analysis.similarity.cache_dir={d / 'cache'}",
                           *overrides])
    return run_analysis(cfg)


def test_none_keeps_link_measures_only(setup, tmp_path):
    out = run(setup, tmp_path, "analysis/similarity=none")
    s = pd.read_csv(out / "summary.csv")
    assert s.sim_excess.isna().all() and s.bits_vs_uniform.notna().any()
    assert set(pd.read_csv(out / "walks.csv").statistic) == {"revisit_rate", "hub_share"}
    assert not list(out.glob("*/flows_*.npz"))                       # no topic tables without topics
    assert json.loads((out / "similarity.json").read_text())["similarity"] == "none"


def test_categories_similarity_is_the_trace_of_the_topic_tables(setup, tmp_path):
    out = run(setup, tmp_path, "stages=[clicks]")
    for f in out.glob("*/results.json"):
        r = json.loads(f.read_text())
        arr = np.load(f.parent / "flows_uniform.npz")
        n = r["models"]["uniform"]["n_steps"]
        assert r["models"]["uniform"]["sim_agent"]["estimate"] == pytest.approx(np.trace(arr["F_agent"]) / n)
        assert r["models"]["uniform"]["sim_model"]["estimate"] == pytest.approx(np.trace(arr["F_model"]) / n)


def test_embeddings_find_the_sticky_walker_and_reuse_the_cache(setup, tmp_path):
    over = ("analysis/similarity=embeddings", "analysis/embedder=mock")
    out = run(setup, tmp_path / "a", *over)
    first = json.loads((out / "similarity.json").read_text())
    assert first["embedded_now"] > 0 and first["no_text"] == 0
    s = pd.read_csv(out / "summary.csv").set_index(["llm", "reference"])
    assert s.loc[("synthetic-sticky", "uniform"), "sim_excess_lo"] > 0          # sticky clicks stay on similar pages
    w = pd.read_csv(out / "walks.csv").set_index(["llm", "baseline", "statistic"])
    assert w.loc[("synthetic-sticky", "uniform", "step_similarity"), "diff_lo"] > 0
    assert {"breadth", "drift_from_start"} <= set(w.index.get_level_values("statistic"))
    again = json.loads((run(setup, tmp_path / "b", *over) / "similarity.json").read_text())
    assert again["embedded_now"] == 0 and again["from_cache"] >= first["embedded_now"]


def test_page_text_and_cache_keys(setup, tmp_path):
    w, d, _ = setup
    title = make_similarity(analysis_cfg(d, ["analysis/similarity=embeddings", "analysis/embedder=mock",
                                              "analysis/page_text=title", f"analysis.similarity.cache_dir={tmp_path}"]))
    lead = make_similarity(analysis_cfg(d, ["analysis/similarity=embeddings", "analysis/embedder=mock",
                                             f"analysis.similarity.cache_dir={tmp_path}"]))
    assert title.text.texts([3]) == ["p3"] and lead.text.texts([3])[0].startswith("p3. subject")
    assert title.cache.path != lead.cache.path                         # different text never shares vectors
    v = lead.vectors([3, 3, 999999])
    assert v[0] @ v[1] == pytest.approx(1) and not v[2].any()          # unit length; a page without text is zero


def test_cost_guard(setup, tmp_path):
    w, d, _ = setup
    sim = make_similarity(analysis_cfg(d, ["analysis/similarity=embeddings", "analysis/embedder=mock",
                                            f"analysis.similarity.cache_dir={tmp_path}", "analysis.similarity.max_new_pages=5"]))
    with pytest.raises(RuntimeError, match="max_new_pages"):
        sim.prepare(range(10))


def test_openai_compatible_embedder_batches_and_counts_usage(monkeypatch):
    calls = []

    class Resp:
        status_code = 200

        def __init__(self, n):
            self.n = n

        def json(self):
            return {"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(self.n)][::-1],
                    "usage": {"total_tokens": 3 * self.n, "cost": 0.001}}

    def post(url, json, timeout, headers):
        calls.append(json)
        return Resp(len(json["input"]))

    import requests
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setenv("TEST_KEY", "k")
    from omegaconf import OmegaConf
    e = EMBEDDERS.get("openai_compatible")(OmegaConf.create({"model": "m", "base_url": "http://x/v1", "api_key_env": "TEST_KEY", "batch_size": 2}))
    v = e.embed(["a", "b", "c"])
    assert v.shape == (3, 2) and list(v[:, 0]) == [0, 1, 0]           # vectors come back in input order
    assert len(calls) == 2 and e.usage["tokens"] == 9 and e.usage["cost_usd"] == pytest.approx(0.002)
    par = EMBEDDERS.get("openai_compatible")(OmegaConf.create({"model": "m", "base_url": "http://x/v1", "api_key_env": "TEST_KEY",
                                                               "batch_size": 1, "concurrency": 4}))
    assert par.embed(["a", "b", "c", "d"]).shape == (4, 2) and par.usage["requests"] == 4


def test_cache_round_trip(tmp_path):
    c = EmbeddingCache(tmp_path, {"kind": "mock", "dims": 4}, {"kind": "lead_db", "field": "title"})
    c.put([1, 2], np.array([[1, 0, 0, 0], [0, 0.5, 0, 0]], np.float32))
    got = c.get([1, 2, 3])
    assert set(got) == {1, 2} and got[2][1] == pytest.approx(0.5) and c.count() == 2


def test_any_graph_can_be_saved_and_walked(tmp_path):
    from omegaconf import OmegaConf
    folder = save_graph(tmp_path / "g", [10, 20, 30, 30], [20, 30, 10, 20], hub_page_ids=[30])
    g = load_graph(OmegaConf.create({"kind": "core_npz", "dir": str(folder), "hubs": "pool"}))
    assert list(g.attrs["page_id"]) == [10, 20, 30] and list(g.hub) == [False, False, True]
    assert list(g.attrs["page_id"][g.out(g.locals_of_page_ids([30])[0])]) == [10, 20]
    top = load_graph(OmegaConf.create({"kind": "core_npz", "dir": str(folder), "hubs": "top_in_degree", "hub_top_n": 1}))
    assert g.attrs["page_id"][top.hub][0] == 20                       # the most linked-to page
    dead = save_graph(tmp_path / "d", [1], [2])                       # page 2 has no out-link: walks stay there
    w = walks.Walker(load_graph(OmegaConf.create({"kind": "core_npz", "dir": str(dead)})))
    assert w.walk("uniform", 0, 3, 2).tolist() == [[0, 1, 1, 1]] * 2


class OnlyFirstLetter(PageVectors):
    """A plug-in by dotted path: pages are alike when their titles start with the same letter."""
    name, dim = "first_letter", 2

    def __init__(self, cfg):
        pass

    def vectors(self, page_ids, categories=None):
        return np.array([[1.0, 0.0] if int(p) % 2 else [0.0, 1.0] for p in page_ids])


def test_similarity_plugin_by_dotted_path(setup, tmp_path):
    out = run(setup, tmp_path, "analysis.similarity.kind=test_similarity.OnlyFirstLetter", "stages=[clicks]")
    assert pd.read_csv(out / "summary.csv").similarity.eq("first_letter").all()
    assert isinstance(make_similarity(analysis_cfg(setup[1], ["analysis/similarity=none"])), NoVectors)
