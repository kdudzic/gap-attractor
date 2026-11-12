"""Read run folders into one flat table of analysable steps.

A step is analysable when the agent made the choice itself and it led to a page:
- `search_start` (search-first variants B/D, step 0) has no current page; it only supplies the start page.
- steps the harness handled as `skip` (random pick after failed parses) or `terminate` are dropped.
- STOP and unresolved choices end the run and are dropped.
Kinds: FOLLOW = a link chosen from the displayed menu; JUMP = a search result or an intent jump
(the destination is not on the page's menu). A `search` step whose results were empty falls back to the
page links (the harness does the same), so it is a FOLLOW.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FOLLOW, JUMP = 0, 1


@dataclass
class Steps:
    runs: pd.DataFrame              # one row per run: run_id, folder, schema, cluster, group fields
    run: np.ndarray                 # per step: row in `runs`
    t: np.ndarray                   # step_index as logged
    kind: np.ndarray                # FOLLOW | JUMP
    x: np.ndarray                   # current page (row in page tables)
    y: np.ndarray                   # chosen page
    menu_ptr: np.ndarray            # menu of step s is menu[menu_ptr[s]:menu_ptr[s+1]] (empty for JUMP)
    menu: np.ndarray
    chosen: np.ndarray              # index into `menu` of the chosen link (-1 for JUMP)
    page_ids: np.ndarray            # page row -> page_id
    page_cats: list                 # page row -> list of categories as logged
    audit: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def cluster(self) -> np.ndarray:
        return self.runs["cluster_index"].to_numpy()[self.run]

    def menu_sizes(self) -> np.ndarray:
        return np.diff(self.menu_ptr)

    def subset(self, mask: np.ndarray) -> "Steps":
        """Steps where mask is true; menus are re-packed, runs and pages are kept."""
        idx = np.flatnonzero(mask)
        sizes = self.menu_sizes()[idx]
        ptr = np.concatenate([[0], np.cumsum(sizes)])
        take = np.concatenate([np.arange(self.menu_ptr[i], self.menu_ptr[i + 1]) for i in idx]) if len(idx) else np.zeros(0, int)
        chosen = np.where(self.chosen[idx] >= 0, self.chosen[idx] - self.menu_ptr[idx] + ptr[:-1], -1)
        return Steps(self.runs, self.run[idx], self.t[idx], self.kind[idx], self.x[idx], self.y[idx], ptr,
                     self.menu[take].astype(np.int64), chosen, self.page_ids, self.page_cats, dict(self.audit))


def find_runs(paths) -> list[Path]:
    """Every folder below `paths` holding run.json and steps.jsonl (superseded runs are skipped)."""
    out = []
    for p in map(Path, paths):
        cands = [p] if (p / "run.json").exists() else sorted(q.parent for q in p.rglob("run.json"))
        out += [c for c in cands if (c / "steps.jsonl").exists() and "_superseded" not in c.parts]
    return out


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


PROMPT_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompt"


def _known_wordings() -> dict[str, str]:
    """system prompt text -> wording name, from configs/prompt/*.yaml."""
    out = {}
    for f in sorted(PROMPT_DIR.glob("*.yaml")):
        d = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if d.get("system_prompt"):
            out[" ".join(str(d["system_prompt"]).split())] = d.get("prompt_wording", f.stem)
    return out


def prompt_wording(info: dict) -> str:
    """The run's prompt wording (prompt=free|browse|notask). Runs before schema 3.3 did not record it, so their
    system prompt is matched against configs/prompt/; an unknown text becomes custom-<hash of the text>."""
    exp = _get(info, "config.experiment") or {}
    if exp.get("prompt_wording"):
        return str(exp["prompt_wording"])
    text = " ".join(str(exp.get("system_prompt") or "").split())
    if not text:
        return "none"
    return _known_wordings().get(text) or "custom-" + hashlib.sha1(text.encode()).hexdigest()[:8]


def allow_stop(info: dict) -> str:
    """'true' unless the run had experiment.allow_stop=false (runs before the option always offered a stop)."""
    v = _get(info, "config.experiment.allow_stop")
    return "true" if v is None or str(v).lower() == "true" else "false"


def memory_setup(info: dict) -> str:
    """The memory plus the settings its plug-in declared (run.json memory_settings, e.g. a context file's interface
    and budget), so runs of one memory with different settings are not pooled."""
    name = str(_get(info, "config.experiment.memory") or "none")
    settings = info.get("memory_settings")
    if not settings:
        return name
    return name + "|" + ",".join(f"{k}={settings[k]}" for k in sorted(settings))


def world_setup(info: dict) -> str:
    """config.world.name for single runs (as before); for a run in a multi-agent world also how many agents and which
    models it held, with how many agents ran each (run.json world.n_agents, world.models), so worlds of different
    composition are not pooled."""
    name = _get(info, "config.world.name")
    world = info.get("world") or {}
    if not world:
        return str(name) if name is not None else "none"
    parts = [str(name or "world"), f"n={world.get('n_agents', '?')}"]
    if world.get("models"):   # with multiplicity: [a, a, b] and [a, b, b] are different worlds
        counts = {}
        for m in map(str, world["models"]):
            counts[m] = counts.get(m, 0) + 1
        parts.append("+".join(f"{m}x{c}" if c > 1 else m for m, c in sorted(counts.items())))
    return "|".join(parts)


def effective_temperature(info: dict, folder: Path | None = None) -> str:
    """The temperature the model actually got. A provider that left it out (the model does not accept it: run.json
    effective_sampling, or dropped_params in the step logs of runs from 14 Oct 2026 on) gives 'provider-default'; a
    run whose provider could not check the model's parameters gets '<t>-unchecked'.
    Runs before that recorded nothing, so a model that ignored the setting cannot be told apart there."""
    es = info.get("effective_sampling")
    if es is not None:
        if es.get("temperature") is None:
            return "provider-default"
        return str(es["temperature"]) + ("-unchecked" if es.get("checked") is False else "")
    if folder is not None and (folder / "steps.jsonl").exists():
        with (folder / "steps.jsonl").open(encoding="utf-8") as handle:
            first = handle.readline()
        attempts = ((json.loads(first).get("extras") or {}).get("retry_attempts") or []) if first.strip() else []
        if any("temperature" in (a.get("dropped_params") or []) for a in attempts):
            return "provider-default"
    v = _get(info, "config.provider.temperature")
    return str(v) if v is not None else "none"


# group_by values starting with "derived." are computed from run.json (and, for runs that predate a field, the
# first logged step) by these functions, not read from it
DERIVED = {"prompt_wording": prompt_wording, "allow_stop": allow_stop, "memory_setup": memory_setup,
           "world_setup": world_setup, "effective_temperature": effective_temperature}


def condition_value(info: dict, path: str, folder: Path | None = None) -> str:
    if path.startswith("derived."):
        fn = DERIVED[path.split(".", 1)[1]]
        return fn(info, folder) if fn is effective_temperature else fn(info)
    v = _get(info, path)
    return str(v) if v is not None else "none"


def run_row(folder: Path, info: dict, group_by: dict[str, str]) -> dict:
    """One row of the run table: identity, schema, cluster for the uncertainty estimates, and condition fields."""
    schema = str(info.get("schema_version", "0"))
    world = info.get("world") or {}
    # run ids are unique only within a batch, so clusters (and run keys elsewhere) carry the batch too
    cluster = f"{info.get('batch_id')}/world{world['world_id']}" if world.get("world_id") is not None else f"{info.get('batch_id')}/{info['run_id']}"
    return {"run_id": info["run_id"], "folder": str(folder), "schema": schema, "schema_major": schema.split(".")[0],
            "cluster": cluster, "start_page": info.get("start_page"),
            **{k: condition_value(info, path, folder) for k, path in group_by.items()}}


def load_steps(paths, group_by: dict[str, str] | None = None, menu_dedupe: str = "page_id",
               exclude_handling=("skip", "terminate"), on_choice_outside_menu: str = "raise") -> Steps:
    """Parse all runs below `paths`. `group_by` maps a label to a dotted path in run.json (e.g. config.provider.model)
    or to a derived field (derived.<name>, see DERIVED)."""
    group_by = dict(group_by or {})
    audit, bad = Counter(), []
    pid_row, page_ids, page_cats = {}, [], []

    def page(pid, cats):
        i = pid_row.get(pid)
        if i is None:
            i = pid_row[pid] = len(page_ids)
            page_ids.append(pid); page_cats.append(None)
        if page_cats[i] is None and cats is not None:
            page_cats[i] = list(cats)
        return i

    runs, R, T, K, X, Y, ptr, menu, chosen = [], [], [], [], [], [], [0], [], []
    for folder in find_runs(paths):
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        r = len(runs); runs.append(run_row(folder, info, group_by))
        recs = [json.loads(line) for line in (folder / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for st in sorted(recs, key=lambda s: s.get("step_index", 0)):
            audit["steps_logged"] += 1
            action = st.get("action") or "follow"
            for row_ in st.get("choice_set") or []:
                page(row_.get("page_id"), row_.get("semantic_cluster_label"))
            for row_ in (st.get("search") or {}).get("results") or []:
                page(row_.get("page_id"), row_.get("semantic_cluster_label"))
            if action == "search_start":
                audit["search_start"] += 1; continue
            handling = (st.get("invalid_action") or {}).get("harness_handling")
            if handling in exclude_handling:
                audit[f"excluded_{handling}"] += 1; continue
            y_id = (st.get("resolution") or {}).get("resolved_page_id")
            if st.get("parsed_choice") == "STOP" or y_id is None or st.get("current_page_id") is None:
                audit["stop_or_unresolved"] += 1; continue
            x = page(st["current_page_id"], st.get("semantic_cluster_label"))
            y = page(y_id, None)
            results = (st.get("search") or {}).get("results")
            is_jump = action == "intent_jump" or (action == "search" and bool(results))
            if is_jump:
                audit[f"jump_{action}"] += 1
                R.append(r); T.append(st.get("step_index", 0)); K.append(JUMP); X.append(x); Y.append(y)
                ptr.append(ptr[-1]); chosen.append(-1); continue
            ids = [row_.get("page_id") for row_ in st.get("choice_set") or [] if row_.get("page_id") is not None]
            if menu_dedupe == "page_id":
                ids = list(dict.fromkeys(ids))
            if y_id not in ids:
                bad.append((info["run_id"], st.get("step_index"), y_id, st.get("parsed_choice"))); continue
            m = [pid_row[i] for i in ids]
            audit["follow"] += 1
            R.append(r); T.append(st.get("step_index", 0)); K.append(FOLLOW); X.append(x); Y.append(y)
            chosen.append(ptr[-1] + ids.index(y_id)); menu += m; ptr.append(ptr[-1] + len(m))
    if bad:
        if on_choice_outside_menu == "raise":
            raise ValueError(f"{len(bad)} followed links are not in their displayed menu "
                             f"(run_id, step_index, page_id, parsed_choice): {bad[:20]}")
        audit["dropped_choice_outside_menu"] = len(bad)
    runs_df = pd.DataFrame(runs)
    if len(runs_df):
        runs_df["cluster_index"] = pd.factorize(runs_df["cluster"])[0]
    steps = Steps(runs_df, np.array(R, int), np.array(T, int), np.array(K, int), np.array(X, int), np.array(Y, int),
                  np.array(ptr, int), np.array(menu, int), np.array(chosen, int), np.array(page_ids, dtype=object),
                  page_cats, {})
    # a jump destination may never appear with its categories (e.g. a run's last intent jump): drop those steps
    unknown = np.array([c is None for c in page_cats], bool)
    keep = ~(unknown[steps.x] | unknown[steps.y]) if len(steps) else np.zeros(0, bool)
    audit["dropped_page_without_categories"] = int((~keep).sum())
    steps = steps.subset(keep)
    steps.audit = {**dict(audit), "runs": len(runs_df), "steps_used": len(steps)}
    return steps
