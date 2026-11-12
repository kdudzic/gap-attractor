"""Summary charts for docs/results.md.

    uv run --extra plots python docs/figures/make_figures.py

The numbers are the reported estimates from docs/results.md, typed in below. The charts are not recomputed from run
logs: the full run set is not in this repository (see data/README.md).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
MODEL, HUMAN = "#2a78d6", "#eb6834"          # models / human readers and other comparison values
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200,
})


def finish(fig, ax, title, note, name, left=0.0):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11.5, weight="bold")
    fig.text(0.01, 0.015, note, fontsize=7.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(left, 0.05 + 0.045 * note.count("\n"), 1, 0.93))
    fig.savefig(OUT / name)
    plt.close(fig)


def bars(ax, rows, fmt="{:.2f}"):
    """rows: (label, value, (lo, hi) or None, colour), drawn top to bottom."""
    for i, (label, value, ci, colour) in enumerate(rows):
        ax.barh(i, value, height=0.56, color=colour)
        if ci:
            ax.plot(ci, (i, i), color=INK, lw=1.1)
        ax.text(max(value, ci[1] if ci else value) + 0.012, i, fmt.format(value), va="center", color=INK, fontsize=9.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()


def agreement_ladder():
    # docs/results.md, finding 1 (10 models, 20 start pages, 11 Nov 2026 repeat-rate runs; readers: clickstream)
    rows = [("Two human readers", 0.10, (0.08, 0.14), HUMAN),
            ("Two different models", 0.42, (0.33, 0.53), MODEL),
            ("The same model, run twice", 0.56, (0.47, 0.66), MODEL)]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    bars(ax, rows)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("Chance that two first clicks from the same page pick the same link")
    finish(fig, ax, "Different models pick the same link four times as often as two readers",
           "First step from 20 hub pages, 10 models, shuffled link order. Readers: Wikimedia clickstream, Sep 2026.\nLines: 95% CI over start pages.",
           "agreement_ladder.png")


def personas():
    # docs/results.md, finding 2 (agreement between different models, by persona arm)
    rows = [("No persona", 0.44, None, MODEL), ("Assistant", 0.43, None, MODEL), ("Bored", 0.31, None, MODEL),
            ("Researcher", 0.28, None, MODEL), ("Explorer", 0.25, None, MODEL)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bars(ax, rows)
    ax.axvline(0.10, color=HUMAN, lw=1.6, ls="--")
    ax.text(0.108, -0.52, "two human readers 0.10", color=INK2, fontsize=8.5, va="center")
    ax.set_ylim(4.5, -0.75)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Chance that two different models pick the same first link")
    finish(fig, ax, "A persona loosens the herd, but not to the level of readers",
           "Five persona arms on the same 20 start pages and seeds, 4-5 open models, pre-registered.", "personas.png")


def loops():
    # docs/results.md, finding 5 (share of moves that go straight back; runs without a STOP option)
    rows = [("All four models", 0.81, 0.25), ("DeepSeek V4.1 Flash", 0.90, 0.15), ("Grok 4.7", 0.69, 0.06),
            ("MiMo 2.6 Pro", 0.84, 0.38), ("Qwen 3.8 Flash", 0.83, 0.42)]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for i, (label, before, after) in enumerate(rows):
        ax.plot((after, before), (i, i), color=GRID, lw=3, zorder=1)
        ax.scatter(before, i, s=70, color=HUMAN, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.scatter(after, i, s=70, color=MODEL, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.text(before + 0.025, i, f"{before:.2f}", va="center", fontsize=9, color=INK)
        ax.text(after + 0.025, i - 0.3, f"{after:.2f}", va="center", fontsize=9, color=INK)
    ax.scatter([], [], s=70, color=HUMAN, label="no sentence about steps")
    ax.scatter([], [], s=70, color=MODEL, label='with "You can take 15 steps through links."')
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of moves that go straight back to the previous page")
    finish(fig, ax, "One sentence about the step budget ends most of the pacing",
           "Runs with no STOP option, 20 start pages, the four models with a matched arm. Lower with the sentence on 20 of 20 start pages.",
           "loops_budget_sentence.png")


def attractor():
    # docs/results.md, finding 4 (split-half absorption: agents against random walks from the same starts)
    rows = [("Following links: agents", 0.0011, None, MODEL), ("Following links: random walks", 0.0004, None, HUMAN),
            ("Free search: agents", 0.178, None, MODEL), ("Free search: random walks", 0.003, None, HUMAN)]
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    bars(ax, rows, fmt="{:.4g}")
    ax.set_xlim(0, 0.23)
    ax.set_xlabel("Split-half absorption: how strongly independent runs end in the same places")
    finish(fig, ax, "The one real attractor appears only when agents can search freely",
           "Link browsing: mean over grid cells. Free search: blank page, model searches first; above walks in 9 of 9 cells.",
           "attractor_search_vs_links.png")


if __name__ == "__main__":
    agreement_ladder()
    personas()
    loops()
    attractor()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
