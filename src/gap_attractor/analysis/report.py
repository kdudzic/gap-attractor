"""Write per-condition results: flows_<model>.npz, results.json, optional heatmaps; and a summary table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_condition(folder: Path, results: dict, arrays: dict[str, dict], topic_names, make_plots: bool) -> None:
    """results.json, and the topic tables flows_<model>.npz where there are any (similarity=categories)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(results, indent=1, default=_json), encoding="utf-8")
    for model, arr in arrays.items():
        if arr:
            np.savez_compressed(folder / f"flows_{model}.npz", topics=np.array(topic_names), **arr)
            if make_plots:
                plot_flows(folder / f"flows_{model}.png", arr, topic_names, model)


def summary_table(rows: list[dict]) -> pd.DataFrame:
    """One row per condition x model with the headline numbers. sim_* (similarity of consecutive pages, see
    kernel.py) are empty with similarity=none."""
    out = []
    for r in rows:
        s = r["summary"]
        if not s.get("n_steps"):
            continue                                      # model does not cover any step of this condition
        out.append({**r["condition"], "reference": r["model"], "similarity": r.get("similarity"),
                    "n_clusters": s.get("n_clusters", 0), "n_steps": s.get("n_steps", 0),
                    "sim_agent": _est(s, "sim_agent"), "sim_model": _est(s, "sim_model"),
                    "sim_excess": _est(s, "sim_excess"), "sim_excess_lo": _ci(s, "sim_excess", 0),
                    "sim_excess_hi": _ci(s, "sim_excess", 1), "sim_excess_sign_p": s.get("sim_excess", {}).get("sign_test", {}).get("p"),
                    "bits_vs_uniform": _est(s, "bits_per_step_vs_uniform"), "bits_lo": _ci(s, "bits_per_step_vs_uniform", 0),
                    "bits_hi": _ci(s, "bits_per_step_vs_uniform", 1),
                    "bits_sign_p": s.get("bits_per_step_vs_uniform", {}).get("sign_test", {}).get("p")})
    return pd.DataFrame(out)


def plot_flows(path: Path, arr: dict, names, model: str) -> None:
    """Heatmaps of L (log-ratio agent/model) and D (difference per step); flagged rows greyed out, cells whose
    bootstrap interval of D excludes 0 marked with a dot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(2 * (4 + 0.28 * k), 4 + 0.28 * k), constrained_layout=True)
    sig = (arr["D_ci"][0] > 0) | (arr["D_ci"][1] < 0)
    for ax, key, title in ((axes[0], "L", "log(F_agent / F_model)"), (axes[1], "D", "D = (F_agent - F_model) / steps")):
        M = np.ma.masked_array(arr[key], mask=np.repeat(arr["flagged_rows"][:, None], k, 1))
        lim = np.nanmax(np.abs(M)) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ys, xs = np.nonzero(sig & ~arr["flagged_rows"][:, None])
        ax.scatter(xs, ys, s=6, c="k")
        ax.set_xticks(range(k), names, rotation=90, fontsize=6); ax.set_yticks(range(k), names, fontsize=6)
        ax.set_xlabel("next page topic"); ax.set_ylabel("current page topic"); ax.set_title(f"{model}: {title}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _est(s, k):
    return s.get(k, {}).get("estimate")


def _ci(s, k, i):
    ci = s.get(k, {}).get("ci95")
    return ci[i] if ci else None


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
