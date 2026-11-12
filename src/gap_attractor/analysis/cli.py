"""gap-attractor-analyze: the analysis pipeline, per condition (docs/analysis.md).

Stages (config `stages`): outcomes (how runs end; run folders only), clicks (each click versus reference choosers
on the same menu; `topics` is accepted as its old name), walks (whole runs versus random walks from the same start
page on the graph of configs/analysis/graph/), hypotheses (where the chosen link ranks on fixed scores, against
reference choosers; analysis/hypotheses.py) and convergence (do runs from different starts end up in the same place,
hop by hop, against walks; analysis/convergence.py) and loops (A -> B -> A bounces and escapes; analysis/loops.py). How alike two pages are comes from configs/analysis/similarity/:
categories (topic mixes), embeddings or none.

    uv run --env-file .env --extra analysis gap-attractor-analyze data.paths=[runs/for_analysis]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gap_attractor.analysis import convergence, hypotheses, kernel, loops, outcomes, report, walks
from gap_attractor.analysis.clickstream import Clickstream
from gap_attractor.analysis.logs import JUMP, load_steps
from gap_attractor.analysis.models import call_model, clickstream_coverage
from gap_attractor.analysis.graph import load_graph
from gap_attractor.analysis.similarity import make_similarity

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def slug(values) -> str:
    return "__".join(str(v).replace("/", "-").replace(" ", "-") for v in values)


STAGE_ALIASES = {"topics": "clicks"}   # the clicks stage was called topics before similarity options existed
STAGES = ("outcomes", "clicks", "walks", "hypotheses", "convergence", "loops")


class GraphHolder:
    """The link graph, loaded once on first use (walks, convergence and the hub hypothesis share it)."""
    def __init__(self, cfg):
        self.cfg, self._g, self._indeg = cfg, None, None

    def __call__(self):
        if self._g is None:
            self._g = load_graph(self.cfg.analysis.graph)
        return self._g

    def in_degree(self):
        if self._indeg is None:
            self._indeg = walks.in_degree(self())
        return self._indeg


def run_analysis(cfg) -> Path:
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    group_by = OmegaConf.to_container(cfg.group_by)
    stages = [STAGE_ALIASES.get(s, s) for s in cfg.get("stages", ["clicks"])]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; expected {', '.join(STAGES)}")
    if "outcomes" in stages:
        runs = outcomes.load_outcomes(list(cfg.data.paths), group_by)
        if len(runs):
            oc = cfg.get("outcomes") or {}
            table = outcomes.write(out, runs, list(group_by) + ["schema_major"], oc.get("paired_field"), oc.get("paired_baseline"))
            print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")
    sim = make_page_vectors(cfg) if {"clicks", "walks", "hypotheses", "convergence"} & set(stages) else None
    graph = GraphHolder(cfg)
    if "clicks" in stages:
        run_clicks(cfg, out, group_by, sim)
    if "walks" in stages:
        run_walks(cfg, out, group_by, sim, graph)
    if "hypotheses" in stages:
        run_hypotheses(cfg, out, group_by, sim, graph)
    if "convergence" in stages:
        run_convergence(cfg, out, group_by, sim, graph)
    if "loops" in stages:
        run_loops(cfg, out, group_by)
    if sim is not None:
        (out / "similarity.json").write_text(json.dumps(sim.describe(), indent=1, default=str), encoding="utf-8")
        print(f"similarity: {json.dumps(sim.describe(), default=str)[:400]}")
    print(f"results: {out}")
    return out


def make_page_vectors(cfg):
    try:
        return make_similarity(cfg)
    except FileNotFoundError as e:
        raise SystemExit(str(e))


def load_clickstream(cfg, wanted: bool):
    if not wanted:
        return None
    try:
        return Clickstream.load(cfg.resources)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}. Or drop clickstream from `models` / `walks.baselines`.")


def run_walks(cfg, out: Path, group_by: dict, sim, graph_holder=None) -> None:
    """Whole runs versus random walks from the same start page with the same number of moves."""
    cs = load_clickstream(cfg, "clickstream" in cfg.walks.baselines)
    graph = graph_holder() if graph_holder is not None else load_graph(cfg.analysis.graph)
    keys = list(group_by) + ["schema_major"]
    runs = walks.load_paths(list(cfg.data.paths), group_by)
    table, audit = walks.simulate(runs, keys, graph, sim, cs, baselines=list(cfg.walks.baselines),
                                  per_run=cfg.walks.per_run, unlisted=cfg.clickstream_unlisted,
                                  distance=cfg.walks.distance, max_distance=cfg.walks.max_distance, seed=cfg.seed)
    summary = walks.compare(table, keys, cfg.n_boot, cfg.seed)
    walks.write(out, table, summary, audit)
    if len(summary):
        cols = keys + ["baseline", "statistic", "runs", "agent", "walks", "diff", "diff_lo", "diff_hi", "p_walks"]
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"walks audit: {audit}")


def run_hypotheses(cfg, out: Path, group_by: dict, sim, graph) -> None:
    """Where the chosen link ranks on each declared hypothesis, against the reference choosers (hypotheses.csv)."""
    h = cfg.hypotheses
    cs = load_clickstream(cfg, "human" in h.tests or "clickstream" in h.references)
    steps = load_steps(list(cfg.data.paths), group_by, cfg.data.menu_dedupe, tuple(cfg.data.exclude_handling),
                       cfg.data.on_choice_outside_menu)
    ctx = {"X": kernel.page_matrix(steps, sim), "sim": sim, "clickstream": cs, "unlisted": cfg.clickstream_unlisted,
           "graph": graph, "indegree": graph.in_degree, "self_text": h.get("self_text", "")}
    keys = list(group_by) + ["schema_major"]
    table, bits, audit = hypotheses.run(steps, keys, ctx, h)
    hypotheses.write(out, table, bits, audit)
    if len(table):
        cols = keys + ["hypothesis", "reference", "role", "steps", "R", "lift", "lift_lo", "lift_hi", "p_mc"]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"hypotheses audit: {json.dumps(audit, default=str)[:600]}")


def run_convergence(cfg, out: Path, group_by: dict, sim, graph) -> None:
    """Runs from different start pages, hop by hop, against walks from the same starts (convergence.csv)."""
    c = cfg.convergence
    cs = load_clickstream(cfg, "clickstream" in c.baselines)
    keys = list(group_by) + ["schema_major"]
    runs = walks.load_paths(list(cfg.data.paths), group_by)
    curves, absorb, audit = convergence.run(runs, keys, graph(), sim, cs, c, cfg.clickstream_unlisted, cfg.seed)
    convergence.write(out, curves, absorb, audit)
    if len(absorb):
        print(absorb.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"convergence audit: {json.dumps(audit, default=str)[:600]}")


def run_loops(cfg, out: Path, group_by: dict) -> None:
    """Two-page loops and escapes per condition, and optionally paired across one condition field (loops.csv)."""
    c = cfg.loops
    keys = list(group_by) + ["schema_major"]
    runs = walks.load_paths(list(cfg.data.paths), group_by)
    table, hazard = loops.summarise(runs, keys, c, cfg.n_boot, cfg.seed)
    pairs = loops.paired(runs, keys, c.paired_field, c.paired_baseline, cfg.n_boot, cfg.seed) if c.get("paired_field") else None
    loops.write(out, table, hazard, pairs)
    if len(table):
        cols = keys + ["runs", "bounce_share", "bounce_lo", "bounce_hi", "hazard_1", "hazard_ge3", "decline", "decline_lo", "decline_hi"]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if pairs is not None and len(pairs):
        print(pairs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def run_clicks(cfg, out: Path, group_by: dict, sim) -> None:
    """Each click versus the reference choosers on the same menu: one folder per condition, plus summary.csv."""
    cs = load_clickstream(cfg, "clickstream" in cfg.models)
    ctx = {"clickstream": cs, "clickstream_unlisted": cfg.clickstream_unlisted, "similarity": sim}
    if sim.has_topics:
        meta = json.loads((Path(cfg.resources) / "topics.json").read_text(encoding="utf-8"))
        if list(meta.get("roots_config", [])) != list(cfg.analysis.topics.roots):
            print(f"WARNING: resources were built with other topic roots than configs/analysis/topics/{cfg.analysis.topics.name}.yaml")
    steps = load_steps(list(cfg.data.paths), group_by, cfg.data.menu_dedupe, tuple(cfg.data.exclude_handling),
                       cfg.data.on_choice_outside_menu)
    X = kernel.page_matrix(steps, sim)
    pi = sim.prevalence()
    unlabeled = X[:, -1] == 1.0 if sim.has_topics else None
    hashed = OmegaConf.to_yaml({k: v for k, v in OmegaConf.to_container(cfg).items() if k != "output"})
    base = {"config_hash": hashlib.sha256(hashed.encode()).hexdigest()[:12], "similarity": sim.name,
            "created": datetime.now().isoformat(timespec="seconds")}
    if sim.has_topics:
        base.update(topic_hash=sim.describe().get("topic_hash"), topics=list(sim.topic_names))
    keys = list(group_by) + ["schema_major"]     # never pool across schema 2.x and 3.x
    run_key = steps.runs[keys].astype(str).agg("\x1f".join, axis=1).to_numpy()
    rows = []
    for key in sorted(set(run_key)):
        cond = tuple(key.split("\x1f"))
        mask = np.isin(steps.run, np.flatnonzero(run_key == key))
        sub = steps.subset(mask)
        condition = dict(zip(keys, cond))
        u_logp = call_model("uniform", sub, X, pi).logp
        counts = {"runs": int((run_key == key).sum()), "runs_with_steps": int(len(np.unique(sub.run))),
                  "clusters": int(len(np.unique(sub.cluster))), "steps": len(sub),
                  "link_steps": int((sub.kind != JUMP).sum()), "jump_steps": int((sub.kind == JUMP).sum()),
                  "jump_rate": float((sub.kind == JUMP).mean()) if len(sub) else None}
        if unlabeled is not None and len(sub):
            counts.update(unlabeled_share_x=float(unlabeled[sub.x].mean()), unlabeled_share_y=float(unlabeled[sub.y].mean()))
        results = {**base, "condition": condition, "audit": steps.audit, "counts": counts, "models": {}}
        if cs is not None:
            results["counts"]["clickstream"] = clickstream_coverage(sub, cs)
        arrays = {}
        for name in cfg.models:
            pred = call_model(name, sub, X, pi, ctx)
            arr, summ = kernel.compare(sub, X, pred, u_logp, min_row_mass=cfg.min_row_mass, n_boot=cfg.n_boot,
                                       seed=cfg.seed, topics=sim.has_topics)
            summ["by_time"] = kernel.by_time(sub, X, pred, u_logp, n_boot=cfg.n_boot, seed=cfg.seed, bucket=cfg.time_bucket)
            results["models"][name] = summ
            arrays[name] = arr
            rows.append({"condition": condition, "model": name, "similarity": sim.name, "summary": summ})
        report.write_condition(out / slug(cond), results, arrays, sim.topic_names, cfg.plots and sim.has_topics)
    table = report.summary_table(rows)
    table.to_csv(out / "summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\naudit: {steps.audit}")


def main():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="analysis", overrides=sys.argv[1:])
    run_analysis(cfg)


if __name__ == "__main__":
    main()
