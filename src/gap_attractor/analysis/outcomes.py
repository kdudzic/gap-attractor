"""How runs end, per condition: stop rates, where the stop happened, invalid replies and token use.

One row per run (`load_outcomes`), then one row per condition (`summarise`). Needs only the run folders, not the
topic resources. Optional paired test (`paired`): the same start pages under two values of one condition field
(e.g. prompt=free versus prompt=browse), compared with an exact McNemar test on "did the run stop".

Endings other than stopping and the step limit (a dead end, an API error, a memory's budget: `context_budget`) are
competing endings: a run that ended that way could not stop later. So stop_share alone can differ between
conditions with the same per-step tendency to stop; `stop_hazard` (stops per step taken) does not, and the paired
test counts a stop only within the steps both arms of a start page observed.

Generalised from the 11 Oct 2026 stopping sweep (team analysis notes, not in this repository).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from gap_attractor.analysis.logs import find_runs, run_row

_STOP_REPLY = re.compile(r'"choice"\s*:\s*"\s*stop\s*"', re.I)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_outcomes(paths, group_by: dict[str, str]) -> pd.DataFrame:
    """One row per run: condition fields, how it ended, and per-run counts recomputed from steps.jsonl."""
    rows = []
    for folder in find_runs(paths):
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        steps = [json.loads(l) for l in (folder / "steps.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        outcome = info.get("outcome") or {}
        moves = sum(1 for s in steps if s.get("action") != "search_start" and s.get("parsed_choice") not in (None, "STOP")
                    and (s.get("resolution") or {}).get("resolved_page_id") is not None)
        handling = [(s.get("invalid_action") or {}).get("harness_handling") for s in steps]
        tokens = [s.get("token_counts") or {} for s in steps]
        rows.append({**run_row(folder, info, group_by),
                     "termination": outcome.get("termination_reason") or "unknown",
                     "moves": moves,                     # pages opened by the agent's own or the harness's choice
                     "steps": len(steps),                # decisions taken (each one could have been STOP)
                     "invalid_replies": sum(1 for s in steps if (s.get("invalid_action") or {}).get("is_invalid")),
                     "harness_picks": handling.count("skip"),
                     # STOP replies, including those refused when experiment.allow_stop=false
                     "stop_replies": sum(1 for s in steps if _STOP_REPLY.search(str(s.get("model_output_raw") or ""))),
                     "tokens_in": sum(t.get("input") or 0 for t in tokens),
                     "tokens_out": sum(t.get("output") or 0 for t in tokens),
                     "stop_statement": outcome.get("model_stop_statement")})
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per condition. stop_share = model_stopped / runs; stopped_first_page = stopped before opening any page."""
    out = []
    for cond, g in runs.groupby(keys, sort=True):
        stopped = g.termination == "model_stopped"
        k, n = int(stopped.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({**dict(zip(keys, cond if isinstance(cond, tuple) else (cond,))),
                    "runs": n, "stopped": k, "stop_share": k / n, "stop_lo": lo, "stop_hi": hi,
                    "stopped_first_page": int((stopped & (g.moves == 0)).sum()),
                    "step_limit": int((g.termination == "step_limit").sum()),
                    "other_endings": int((~stopped & ~g.termination.isin(["step_limit", "context_budget", "no_valid_move"])).sum()),
                    "context_budget": int((g.termination == "context_budget").sum()),
                    "no_valid_move": int((g.termination == "no_valid_move").sum()),
                    "stop_hazard": k / int(g.steps.sum()) if int(g.steps.sum()) else float("nan"),
                    "median_moves": float(g.moves.median()),
                    "median_moves_when_stopped": float(g.moves[stopped].median()) if k else float("nan"),
                    "invalid_replies": int(g.invalid_replies.sum()), "harness_picks": int(g.harness_picks.sum()),
                    "stop_replies": int(g.stop_replies.sum()),
                    "tokens_in": int(g.tokens_in.sum()), "tokens_out": int(g.tokens_out.sum())})
    return pd.DataFrame(out)


COMPLETE = ("model_stopped", "step_limit")


def paired(runs: pd.DataFrame, keys: list[str], field: str, baseline: str) -> pd.DataFrame:
    """For every other value of `field`, compare stopping with `baseline` on the same start pages, holding all other
    condition fields fixed. Per start page and value, the share of its runs that stopped (0 or 1 with one run each;
    repeated runs are averaged, never overwritten). A run that ended any other way (competing ending) at step T means
    the start page is only observed up to T: a stop in either arm counts only if it came at step T or earlier
    (`censored_pages` start pages were cut so). only_baseline / only_other = start pages where the share is higher
    under that value; p_mcnemar = exact McNemar on those counts (a sign test when runs are repeated)."""
    rest = [k for k in keys if k != field]
    out = []
    for cond, g in runs.groupby(rest, sort=True) if rest else [((), runs)]:
        if baseline not in set(g[field]):
            continue
        for value in sorted(set(g[field]) - {baseline}):
            pair = g[g[field].isin([baseline, value])]
            b = c = censored = n = 0
            repeats = False
            for start, h in pair.groupby("start_page"):
                arms = {v: a for v, a in h.groupby(field)}
                if len(arms) < 2:
                    continue
                cut = h[~h.termination.isin(COMPLETE)].steps
                horizon = int(cut.min()) if len(cut) else None
                censored += horizon is not None
                repeats |= any(len(a) > 1 for a in arms.values())
                share = {v: ((a.termination == "model_stopped") & (a.steps <= horizon if horizon is not None else True)).mean()
                         for v, a in arms.items()}
                b += share[baseline] > share[value]
                c += share[value] > share[baseline]
                n += 1
            out.append({**dict(zip(rest, cond if isinstance(cond, tuple) else (cond,))), "field": field,
                        "baseline": baseline, "other": value, "pairs": n, "only_baseline": int(b), "only_other": int(c),
                        "p_mcnemar": mcnemar_exact(int(b), int(c)), "censored_pages": censored, "repeated_runs": repeats})
    return pd.DataFrame(out)


def write(out: Path, runs: pd.DataFrame, keys: list[str], paired_field: str | None = None,
          paired_baseline: str | None = None) -> pd.DataFrame:
    """outcomes.csv (per condition), outcomes_runs.csv (per run), outcomes_paired.csv, stop_statements.md."""
    out.mkdir(parents=True, exist_ok=True)
    table = summarise(runs, keys)
    table.to_csv(out / "outcomes.csv", index=False)
    runs.drop(columns=["stop_statement"]).to_csv(out / "outcomes_runs.csv", index=False)
    if paired_field:
        paired(runs, keys, paired_field, paired_baseline).to_csv(out / "outcomes_paired.csv", index=False)
    lines = ["# Stop statements, by condition\n"]
    for cond, g in runs[runs.termination == "model_stopped"].groupby(keys, sort=True):
        label = " / ".join(map(str, cond if isinstance(cond, tuple) else (cond,)))
        lines.append(f"\n## {label} ({len(g)} stopped)\n")
        lines += [f"- after {r.moves} moves, start {r.start_page}: {(r.stop_statement or '').strip()[:300]}"
                  for r in g.itertuples()]
    (out / "stop_statements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table
