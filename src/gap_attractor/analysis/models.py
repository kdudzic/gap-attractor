"""Reference choosers. Each one says, for every step, what the next page's topic would be on average (w)
and how likely the agent's actual choice was (logp). No parameters are fitted.

Defaults (configs/analysis.yaml `models`): uniform (random walk), jump (random walk + random-article jumps) and
clickstream (human readers' link clicks). novelty is kept for exploration but is not a default.

A model is a function (steps, chi, pi, ctx=None) -> Prediction; ctx holds optional resources (ctx["clickstream"]). Rows of `w` and entries of `logp` are NaN on steps the
model does not cover (uniform and novelty only describe link choices, so they skip jumps).
Register a new model with @register("name"), or name it in the config by dotted path "package.module.function".
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np

from gap_attractor.analysis.logs import FOLLOW, JUMP, Steps


@dataclass(frozen=True)
class Prediction:
    w: np.ndarray       # (S, dim) expected vector of the next page under the model (dim 0 with similarity=none)
    logp: np.ndarray    # (S,) log-probability of the agent's actual choice (NaN where not defined)
    covered: np.ndarray | None = None   # (S,) steps the model describes; default: rows of w without NaN

    @property
    def coverage(self) -> np.ndarray:
        if self.covered is not None:
            return self.covered
        return ~np.isnan(self.w).any(1) if self.w.shape[1] else np.ones(len(self.w), bool)


MODELS: dict[str, Callable] = {}


def register(name):
    def deco(fn):
        MODELS[name] = fn
        return fn
    return deco


def get_model(name: str) -> Callable:
    if name in MODELS:
        return MODELS[name]
    module, _, attr = name.replace(":", ".").rpartition(".")
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError, ValueError) as e:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}") from e


def call_model(name: str, steps: Steps, chi: np.ndarray, pi, ctx: dict | None = None) -> Prediction:
    """Run a model; ctx is passed only to models that take it (older custom models have three arguments)."""
    fn = get_model(name)
    return fn(steps, chi, pi, ctx=ctx) if "ctx" in inspect.signature(fn).parameters else fn(steps, chi, pi)


def _segments(steps: Steps):
    """Step index of every menu entry, and menu sizes."""
    sizes = steps.menu_sizes()
    return np.repeat(np.arange(len(steps)), sizes), sizes


def _segment_sum(values: np.ndarray, seg: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n,) + values.shape[1:])
    np.add.at(out, seg, values)
    return out


def menu_choice(steps: Steps, chi: np.ndarray, p_entry: np.ndarray) -> Prediction:
    """Prediction for a chooser that picks menu entry j with probability p_entry[j] (follow steps only)."""
    seg, _ = _segments(steps)
    S = len(steps)
    w = _segment_sum(p_entry[:, None] * chi[steps.menu], seg, S)
    logp = np.full(S, np.nan)
    f = steps.kind == FOLLOW
    with np.errstate(divide="ignore"):
        logp[f] = np.log(p_entry[steps.chosen[f]])
    w[~f] = np.nan
    return Prediction(w, logp, f)


@register("uniform")
def uniform(steps: Steps, chi: np.ndarray, pi: np.ndarray | None = None, ctx=None) -> Prediction:
    """Random walk on pages: every displayed link equally likely."""
    seg, sizes = _segments(steps)
    return menu_choice(steps, chi, 1.0 / sizes[seg])


@register("novelty")
def novelty(steps: Steps, chi: np.ndarray, pi: np.ndarray | None = None, ctx=None) -> Prediction:
    """Inverse topic frequency on the menu: pick a topic present on the menu uniformly, then a link within it
    in proportion to its membership. p(j) = sum_e chi(j,e) / n_menu(e) / #topics on the menu,
    n_menu(e) = sum over the menu of chi(k,e). Links in rare topics of the menu get more weight.
    The unlabeled bucket counts as a topic. Needs similarity=categories (topic mixes)."""
    sim = (ctx or {}).get("similarity")
    if sim is not None and not sim.has_topics:
        raise ValueError("the novelty chooser needs topic mixes: use analysis/similarity=categories")
    seg, _ = _segments(steps)
    C = chi[steps.menu]
    n_menu = _segment_sum(C, seg, len(steps))
    n_topics = (n_menu > 0).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(n_menu[seg] > 0, C / n_menu[seg], 0.0).sum(1)
    return menu_choice(steps, chi, score / n_topics[seg])


@register("jump")
def jump(steps: Steps, chi: np.ndarray, pi: np.ndarray | None = None, ctx=None) -> Prediction:
    """Random jumps for search regimes: a jump (search or intent jump) lands on a random article, so its expected
    topic vector is the global prevalence pi. Link steps follow `uniform`. The jump rate itself is reported, not modelled."""
    if pi is None:
        raise ValueError("the jump model needs topic prevalence pi (topics.json from scripts/build_wiki_resources.py)")
    pred = uniform(steps, chi)
    w = pred.w.copy()
    w[steps.kind == JUMP] = pi
    return Prediction(w, pred.logp, pred.covered | (steps.kind == JUMP))


@register("clickstream")
def clickstream(steps: Steps, chi: np.ndarray, pi: np.ndarray | None = None, ctx=None) -> Prediction:
    """Human readers: each displayed link is chosen in proportion to how often readers clicked it from this page
    in the clickstream month (analysis/clickstream.py; unlisted links count as ctx["clickstream_unlisted"] clicks).
    Link steps only, like uniform."""
    cs = (ctx or {}).get("clickstream")
    if cs is None:
        raise ValueError("the clickstream model needs clickstream.npz (scripts/build_wiki_resources.py clickstream)")
    unlisted = float((ctx or {}).get("clickstream_unlisted", 5.0))
    p_entry = np.zeros(len(steps.menu))
    for s in range(len(steps)):
        a, b = steps.menu_ptr[s], steps.menu_ptr[s + 1]
        if b > a:
            p_entry[a:b], _ = cs.probs(int(steps.page_ids[steps.x[s]]), steps.page_ids[steps.menu[a:b]], unlisted)
    return menu_choice(steps, chi, p_entry)


def clickstream_coverage(steps: Steps, cs) -> dict:
    """How much of the agent's link steps the clickstream actually informs."""
    f = np.flatnonzero((steps.kind == FOLLOW) & (np.diff(steps.menu_ptr) > 0))
    page_listed = chosen_listed = 0
    for s in f:
        a, b = steps.menu_ptr[s], steps.menu_ptr[s + 1]
        c = cs.counts(int(steps.page_ids[steps.x[s]]), steps.page_ids[steps.menu[a:b]])
        page_listed += bool((c > 0).any())
        chosen_listed += bool(c[steps.chosen[s] - a] > 0)
    n = len(f)
    return {"link_steps": n, "page_has_listed_link": page_listed / n if n else None,
            "chosen_link_listed": chosen_listed / n if n else None}
