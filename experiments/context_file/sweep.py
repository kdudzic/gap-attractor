"""Run the context-file conditions side by side: the single-agent ablation ladder and the joint-world controls,
all on the same start pages, model and sampling. Each condition is one run.py process (its own batch folder under
<out>/<condition>/); a manifest records the exact overrides, commit and exit codes.

    uv run --env-file .env python experiments/context_file/sweep.py --out runs/ctx_pilot \
        --model deepseek/deepseek-v4.1-flash --n-starts 6 --steps 30 --window 131072 --dry-run

Start pages: --starts (listed), or --n-starts N drawn with --seed from --starts-file (default: the team's start set
A, topic-stratified); the drawn order is random, so consecutive starts grouped into joint worlds are a random
grouping. The list is recorded in manifest.json. Context embeddings (context_embedding=steps) are on only with
--embed (off by default since 22 Oct 2026: backfill with embed_contexts.py). --resume skips conditions that finished (exit 0) and moves an unfinished one to _superseded/ before
rerunning it, so no partial batch is ever pooled with its rerun.

Every condition has the same prompt (the task only: --task; how to act is in the tools), tool calls, tool
descriptions, sampling (temperature 1, top_p 0.5) and reasoning (--reasoning, by default the lowest each model allows:
LOWEST_REASONING). --model takes one model, or one per agent of a joint world (cycled by agent number).

Single-agent ladder (each rung adds one thing; all use the mechanics-only instructions, ctx_prompt=neutral):
  a_path       the harness's default memory, `path_only` (its own earlier moves and reasons, no pages): the plain agent
  a_full       the harness's `full` memory (everything it saw, no file, no editing): the reference
  b_format     the context file with turn headers, no editing, no readout, a budget too large to matter
  c_tool       + the edit_context tool (still no budget shown, no nudges): self-curation without pressure
  d_budget     + the budget (--budget-tokens, 32768 by default; hard): readout, nudges, held moves ("pressure")
  e_bash       d_budget with bash in a sandbox instead of the tool (headers kept, /work emptied every step)
Side rungs:
  d_clm        d_budget with the CLM-style advice on what to keep (vs d_budget: the instructions alone)
  d_verbatim   d_budget with the model's replies appended as written (text, calls, failed attempts), not just the move
  e_clm        closest to the paper: bash, CLM advice, plain messages, /work kept for the run, replies appended
               verbatim (echoes and command output are kept in full everywhere: nothing is ever shortened)
Joint worlds (--agents N per world, worlds from consecutive start pages; built on d_budget, with bash in a sandbox
per agent). A reply is an assistant message whoever wrote it (others_as=own). The grid (--conditions joint) crosses
where the agents meet with what they are told; numbered always goes with introduced, no marks with rumour/strangers:
  one shared context (N x the single budget; the whole context is the file):
    j_full             introduced, numbered (agent=k / "[agent k]")
    j_full_anon        introduced, no marks (identifiability: numbered vs none)
    j_rumour           only told the context may be shared; no marks
    j_strangers        told nothing about other agents; no marks
  own contexts (the single budget each) + one shared folder /notes in every sandbox:
    j_notes            introduced (told its number, that /notes is shared, that it cannot see their contexts)
    j_notes_rumour     told /notes keeps its files and may be shared
    j_notes_strangers  told /notes keeps its files
  j_separate           own contexts, nothing shared, same scheduler (the reference without a channel)
Extras, not in the grid: j_readonly (j_full, but only one's own turns are editable), j_full_tool and
j_separate_tool (edit_context instead of bash).

Tasks (--task): browse (default) | free (the harness's default wording, "You are free to do what you want.") | find (target pages; --goal-setup assigned|external|pool with --worlds-file or
--targets) | essay (an essay on any topic, handed in with the stop tool; the page budget is stated). Opening
(--opening): page (a start page) | search (the harness's variant b_search_first: one opening search, then links;
--n-runs episodes instead of start pages). --starts-file takes a .tsv/.csv with a title column or a plain list,
e.g. the core graph's nodes for uniformly random starts.

Goal worlds (--goal-setup assigned|external|pool, with --worlds-file or --targets): every joint condition looks for
target pages (prompt=find; see ctxfile/joint.py) instead of browsing.

The budget is 32768 tokens by default (--budget-tokens) with a 4096 reserve (--reserve-tokens): the model is shown
and held to the limit, budget - reserve = 28672. Or --budget-pages x --page-tokens (the median visited page, 4 100
tokens, characters / 4, measured on 1 258 pages of earlier runs). It is hard (ctx.overflow=wall): pages arrive
whole; a move is held while the context (all but the newest links) is over the limit and the model edits again; after
ctx.max_held_moves holds a move goes ahead and the run ends after it (context_budget). Joint worlds sharing one file
get N x the budget and N x the reserve (N x the single limit).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUGE = 1_000_000
START_SET = Path("data/start_pages/start_set_A_final.tsv")


def draw_starts(path: Path, n: int, seed: int) -> list[str]:
    """n titles drawn without replacement, in random order, with `seed` (the rule recorded in the manifest)."""
    if path.suffix in (".tsv", ".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            titles = [row["title"] for row in csv.DictReader(f, delimiter="\t" if str(path).endswith(".tsv") else ",")]
    else:                                           # a plain list, one title per line (e.g. a graph's node list)
        titles = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    if n > len(titles):
        sys.exit(f"--n-starts {n} is more than the {len(titles)} titles in {path}")
    return random.Random(seed).sample(titles, n)


def conditions(budget: int, n_agents: int, reserve: int = 4096, joint_prompt: str = "neutral") -> dict[str, list[str]]:
    neutral, huge = ["ctx_prompt=neutral"], f"ctx.budget_tokens={HUGE}"
    joint = ["ctx=joint", f"ctx.joint.n_agents={n_agents}", f"ctx_prompt={joint_prompt}", "ctx_edit=bash"]
    shared = [f"ctx.budget_tokens={n_agents * budget}", f"ctx.reserve_tokens={n_agents * reserve}"]   # N x the single limit
    separate = [f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve}", "ctx.joint.sharing=separate",
                "ctx.bash.joint_workspace=private"]                       # nothing shared, /work included
    notes = [f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve}", "ctx.joint.sharing=notes",
             "ctx.bash.joint_workspace=private"]
    return {
        "a_path": ["experiment.memory=path_only"],
        "a_full": ["experiment.memory=full"],
        "b_format": ["ctx_edit=none", "ctx.headers=true", "ctx.readout=false", huge, *neutral],      # wall, never reached
        "c_tool": ["ctx_edit=tool", "ctx.readout=false", "ctx.nudges=false", huge, *neutral],
        "d_budget": ["ctx_edit=tool", f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve}", *neutral],
        "e_bash": ["ctx_edit=bash", "ctx.headers=true", "ctx.bash.workspace=per_step", f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve}", *neutral],
        "d_clm": ["ctx_edit=tool", f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve}", "ctx_prompt=clm"],
        "d_verbatim": ["ctx_edit=tool", f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve}", "ctx.append=verbatim", *neutral],
        "e_clm": ["ctx_edit=bash", f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve}", "ctx_prompt=clm", "ctx.append=verbatim",
                  "ctx.shrink_edits=false"],
        "j_full": joint + shared,
        "j_full_anon": joint + shared + ["ctx.attribution=none"],
        "j_rumour": joint + shared + ["ctx.attribution=none", "ctx.awareness=rumour"],
        "j_strangers": joint + shared + ["ctx.attribution=none", "ctx.awareness=strangers"],
        "j_notes": joint + notes,
        "j_notes_rumour": joint + notes + ["ctx.attribution=none", "ctx.awareness=rumour"],
        "j_notes_strangers": joint + notes + ["ctx.attribution=none", "ctx.awareness=strangers"],
        "j_separate": joint + separate,
        "j_readonly": joint + shared + ["ctx.joint.sharing=readonly_others"],
        "j_full_tool": [*joint[:-1], "ctx_edit=tool"] + shared,
        "j_separate_tool": [*joint[:-1], "ctx_edit=tool"] + separate,
    }


UNCAPPED = ("a_full", "b_format", "c_tool")
# The lowest reasoning each model allows (OpenRouter's `reasoning` request field), checked in the 21 Oct pilots.
LOWEST_REASONING = {
    "anthropic/claude-haiku-5.5": {"enabled": False},
    "openai/gpt-6-luna": {"effort": "minimal"},
    "google/gemini-3.8-flash": {"effort": "low"},     # cannot be turned off
    # 23 Oct 2026 probe (one tool-call request per setting, reasoning tokens billed): off gives 0 reasoning tokens
    "qwen/qwen3.8-flash": {"enabled": False},
    "deepseek/deepseek-v4.1-flash-1017": {"enabled": False},
    "xiaomi/mimo-v2.6-pro": {"enabled": False},
    "z-ai/glm-5.3-flash": {"effort": "minimal"},      # 10 Nov 2026 probe: off is refused (mandatory), minimal gives 3 reasoning tokens
    "x-ai/grok-4.7": {"effort": "minimal"},           # "Reasoning is mandatory for this endpoint and cannot be disabled"
}
# The study's fixed settings (decided 22 Oct 2026). A sweep that departs from them, or passes an --extra override
# outside ADMISSIBLE_EXTRA, needs --exploratory: its conditions are then labelled <name>__exploratory everywhere
# (run.json, run logs, the tracker), so they are never pooled with the study's runs by accident.
STUDY = {"steps": 30, "agents": 2, "budget_tokens": 32768, "reserve_tokens": 4096, "budget_pages": None}
ADMISSIBLE_EXTRA = ("experiment.name=", "experiment.tracker_owner=", "experiment.tracker_config_id=",
                    "experiment.contributors=", "experiment.allow_stop=", "ctx.save_requests=")
ORDERS = {"random": ("", "seeded_shuffle"), "round_robin": ("_rr", "fixed")}     # turn order: a design factor


def reasoning_for(model: str, choice: str) -> dict | None:
    """--reasoning: lowest (LOWEST_REASONING), default (the provider's own; nothing sent) or a JSON object for every
    model, e.g. '{"effort": "high"}'."""
    if choice == "default":
        return None
    if choice == "lowest":
        if model not in LOWEST_REASONING:
            sys.exit(f"no known lowest reasoning setting for {model}: add it to LOWEST_REASONING, or pass --reasoning "
                     "default (the provider's own) or a JSON object")
        return LOWEST_REASONING[model]
    return json.loads(choice)


def flow(value) -> str:
    """A Hydra override value (flow style: no quotes needed around simple strings)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {flow(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(flow(v) for v in value) + "]"
    return json.dumps(value) if isinstance(value, bool) else str(value)


GRID = ("j_full", "j_full_anon", "j_rumour", "j_strangers", "j_notes", "j_notes_rumour", "j_notes_strangers", "j_separate")


def git_state() -> dict:
    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return {"git_commit": head or None, "git_dirty": bool(dirty)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="results folder; one subfolder per condition")
    p.add_argument("--model", required=True, nargs="+", help="one model, or one per agent of a joint world (cycled)")
    p.add_argument("--reasoning", default="lowest", help="lowest (default) | default (the provider's own) | a JSON object")
    p.add_argument("--provider", default="openrouter")
    p.add_argument("--starts", nargs="+", help="start pages, one argument each (a multiple of --agents for joint worlds)")
    p.add_argument("--n-starts", type=int, default=None, help="instead of --starts: draw this many from --starts-file with --seed")
    p.add_argument("--starts-file", default=str(START_SET), help="a .tsv/.csv with a 'title' column")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--top-p", type=float, default=0.5, help="sampling top_p (the study's is 0.5; another value needs --exploratory)")
    p.add_argument("--same-start", action="store_true", help="joint worlds: every agent of a world starts on one page "
                   "(ctx.joint.start_assignment=same); --starts then lists one page per world; needs --exploratory")
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--orders", default="random,round_robin", help="joint worlds: turn orders to run, each a condition of its "
                   "own (round_robin adds _rr to the name): random (a seeded shuffle every round) | round_robin (by agent number)")
    p.add_argument("--exploratory", action="store_true", help="allow settings outside the study's (STUDY, ADMISSIBLE_EXTRA); "
                   "the conditions are labelled <name>__exploratory")
    p.add_argument("--budget-tokens", type=int, default=32768, help="the single-agent budget (joint: N x this)")
    p.add_argument("--reserve-tokens", type=int, default=4096, help="the single-agent reserve: the limit is budget - reserve")
    p.add_argument("--page-tokens", type=int, default=4100, help="the median visited page, for --budget-pages")
    p.add_argument("--mean-page-tokens", type=int, default=6000, help="the mean visited page, for the --window check")
    p.add_argument("--budget-pages", type=float, default=None, help="budget as pages x --page-tokens, instead of --budget-tokens")
    p.add_argument("--conditions", default="all", help="comma-separated names, or all, single, joint")
    p.add_argument("--parallel", type=int, default=3, help="processes at once (each holds the dump index, ~8.5 GB)")
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--extra", action="append", default=[], help="more overrides for every condition (repeatable)")
    p.add_argument("--joint-prompt", choices=["neutral", "recipe"], default="neutral",
                   help="instructions of the joint (j_*) conditions: neutral (mechanics only, the study's) | recipe (adds a "
                        "routine for keeping the context small; exploratory)")
    p.add_argument("--window", type=int, default=None, help="the model's context window in tokens (required with the uncapped "
                   "rungs a, b, c): refuse a plan in which they would outgrow it, so no rung ends early for size alone")
    p.add_argument("--embed", action="store_true", help="context embeddings after every step (context_embedding=steps)")
    p.add_argument("--no-embed", action="store_true", help="(the default now; kept so older scripts run)")
    p.add_argument("--resume", action="store_true", help="skip finished conditions; move unfinished ones to _superseded/ and rerun")
    p.add_argument("--goal-setup", choices=["assigned", "external", "pool", "common"], default=None,
                   help="goal worlds: every joint condition looks for target pages (needs --worlds-file or --targets)")
    p.add_argument("--worlds-file", default=None, help="goal worlds: a TSV with world_id, agent_id, start, target")
    p.add_argument("--targets", nargs="+", default=None, help="goal worlds: target titles, consecutive per world like --starts")
    p.add_argument("--goal-stop", choices=["all", "first", "all_found"], default=None,
                   help="default: first with --goal-setup common, all_found with assigned or external, all with pool")
    p.add_argument("--task", choices=["browse", "free", "find", "essay"], default="browse")
    p.add_argument("--opening", choices=["page", "search"], default="page")
    p.add_argument("--n-runs", type=int, default=None, help="--opening search: episodes (a multiple of --agents for joint worlds)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    budget = int(a.page_tokens * a.budget_pages) if a.budget_pages else int(a.budget_tokens)
    if a.reserve_tokens >= budget:
        sys.exit(f"--reserve-tokens {a.reserve_tokens} leaves no room in a budget of {budget}")
    table = conditions(budget, a.agents, a.reserve_tokens, a.joint_prompt)
    names = {"all": list(table), "single": [k for k in table if not k.startswith("j_")],
             "joint": list(GRID)}.get(a.conditions, a.conditions.split(","))
    unknown = [n for n in names if n not in table]
    if unknown:
        sys.exit(f"unknown condition(s) {unknown}; known: {list(table)}")
    orders = [o.strip() for o in a.orders.split(",") if o.strip()]
    if not orders or any(o not in ORDERS for o in orders):
        sys.exit(f"--orders takes {', '.join(ORDERS)} (comma-separated)")
    chosen = {"steps": a.steps, "agents": a.agents, "budget_tokens": a.budget_tokens, "reserve_tokens": a.reserve_tokens,
              "budget_pages": a.budget_pages}
    departures = [f"--{k.replace('_', '-')} {v}" for k, v in chosen.items() if v != STUDY[k]]
    departures += ([f"--top-p {a.top_p}"] if a.top_p != 0.5 else []) + (["--same-start"] if a.same_start else [])
    departures += [f"--joint-prompt {a.joint_prompt}"] if a.joint_prompt != "neutral" else []
    departures += [f"--extra {x}" for x in a.extra if not x.startswith(ADMISSIBLE_EXTRA)]
    if departures and not a.exploratory:
        sys.exit("outside the study's settings: " + "; ".join(departures) + ". Pass --exploratory to run anyway "
                 "(the conditions are labelled <name>__exploratory)")
    label = "__exploratory" if a.exploratory else ""
    base = {}                                                  # condition name -> (table entry, order overrides)
    for n in names:
        for o in (orders if n.startswith("j_") else ["random"]):
            suffix, rule = ORDERS[o]
            base[n + suffix] = (n, [f"ctx.joint.order={rule}"] if n.startswith("j_") else [])
    names = list(base)
    goal = []
    if a.goal_setup and a.task == "browse":
        a.task = "find"
    if a.task == "find" and not a.goal_setup:
        sys.exit("--task find needs --goal-setup assigned|external|pool|common")
    if a.goal_setup and a.goal_stop is None:
        a.goal_stop = {"common": "first", "pool": "all"}.get(a.goal_setup, "all_found")
    if a.task == "essay" and a.goal_setup:
        sys.exit("--goal-setup goes with --task find, not essay")
    if a.task == "essay":
        goal = ["task=essay"]
    if a.task == "free":
        goal = ["prompt=free"]
    if len(a.model) > 1 and any(not n.startswith("j_") for n in names):
        sys.exit("several --model values make a joint world's agents: run single-agent conditions with one model")
    reasoning = {m: reasoning_for(m, a.reasoning) for m in a.model}
    agents_providers = [{"model": m, **({"extra_body": {"reasoning": reasoning[m]}} if reasoning[m] else {})} for m in a.model]
    models = [f"provider.model={a.model[0]}", "provider.temperature=1", f"provider.top_p={a.top_p}",
              *([f"+provider.extra_body.reasoning={flow(reasoning[a.model[0]])}"] if reasoning[a.model[0]] else [])]
    if a.opening == "search":
        if a.task == "find" or a.starts or a.n_starts is not None or not a.n_runs:
            sys.exit("--opening search: give --n-runs (no start pages), and not with --task find")
        goal += ["variant=b_search_first", f"variant.n_runs={a.n_runs}"]
    if a.goal_setup:
        if any(not n.startswith("j_") for n in names):
            sys.exit("--goal-setup runs joint conditions only (--conditions joint, or j_* names)")
        if bool(a.worlds_file) == bool(a.targets):
            sys.exit("--goal-setup needs either --worlds-file or --targets")
        goal = ["ctx.goal.task=find", f"ctx.goal.setup={a.goal_setup}", f"ctx.goal.stop={a.goal_stop}", "prompt=find",
                *([f"ctx.goal.worlds_file={Path(a.worlds_file).resolve()}"] if a.worlds_file else
                  ["ctx.goal.targets=" + json.dumps(a.targets, ensure_ascii=False)])]
    if a.worlds_file and a.goal_setup:
        starts, start_rule = [], f"from {a.worlds_file}"
    elif a.opening == "search":
        starts, start_rule = [], f"{a.n_runs} opening searches"
    elif bool(a.starts) == (a.n_starts is not None):
        sys.exit("give either --starts or --n-starts")
    else:
        starts = [s.strip() for s in a.starts if s.strip()] if a.starts else draw_starts(Path(a.starts_file), a.n_starts, a.seed)
        start_rule = "listed" if a.starts else f"{a.n_starts} drawn with seed {a.seed} from {a.starts_file}"
    same = ["ctx.joint.start_assignment=same"] if a.same_start else []
    if starts and not a.same_start and any(n.startswith("j_") for n in names) and len(starts) % a.agents:
        sys.exit(f"--starts must be a multiple of --agents ({a.agents}) for joint conditions")
    need = int(a.steps * a.mean_page_tokens * 1.05 + 4000)   # pages + moves + system, for the uncapped rungs
    if a.window is None and any(n in UNCAPPED for n in names):
        sys.exit(f"--window is required with the uncapped rungs ({', '.join(UNCAPPED)}): they need ~{need} tokens at {a.steps} steps")
    if a.window and need > a.window and any(n in UNCAPPED for n in names):
        sys.exit(f"the uncapped rungs would need ~{need} tokens at {a.steps} steps, over --window {a.window}: use fewer "
                 f"--steps (about {int((a.window - 4000) / (a.mean_page_tokens * 1.05))}) or leave out {', '.join(UNCAPPED)}")
    out = Path(a.out).resolve()
    common = [f"provider={a.provider}", *models, "wikipedia=dump", f"experiment.step_limit={a.steps}",
              f"experiment.seed={a.seed}", *(["batch.start_pages=" + json.dumps(starts, ensure_ascii=False)] if starts else []),
              *(["context_embedding=steps"] if a.embed else []), *same, *goal, *a.extra]
    joint_models = [f"ctx.joint.agent_providers={flow(agents_providers)}"]
    jobs = {n: [*common, f"output.directory={out / n}", f"experiment.condition={n}{label}", *table[base[n][0]], *base[n][1],
                *(joint_models if n.startswith("j_") else [])] for n in names}
    manifest = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **git_state(), "window": a.window,
                "uncapped_tokens_estimate": need,
                "model": a.model if len(a.model) > 1 else a.model[0], "reasoning": reasoning, "starts": starts, "start_rule": start_rule, "steps": a.steps, "agents": a.agents,
                "budget_tokens": budget, "reserve_tokens": a.reserve_tokens, "limit_tokens": budget - a.reserve_tokens,
                "page_tokens": a.page_tokens, "budget_pages": a.budget_pages, "orders": orders,
                "exploratory": departures if a.exploratory else False, "conditions": jobs, "results": {}}
    if a.dry_run:
        print(json.dumps(manifest, indent=1))
        return
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else None
    if previous and not a.resume:
        sys.exit(f"{out} already holds a sweep: use --resume to finish it, or another --out")
    if previous:
        if previous.get("conditions") and any(previous["conditions"].get(n) != jobs[n] for n in names if n in previous["conditions"]):
            sys.exit("--resume: the settings differ from the earlier sweep in this folder (compare manifest.json)")
        manifest["results"] = {k: v for k, v in previous.get("results", {}).items() if v.get("exit_code") == 0}
        manifest["resumed"] = previous.get("resumed", []) + [manifest["created"]]
        for n in [n for n in names if n not in manifest["results"] and (out / n).exists()]:
            target = out / "_superseded" / f"{n}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out / n), str(target))
            print(f"{n}: unfinished, moved to {target}", flush=True)
        names = [n for n in names if n not in manifest["results"]]

    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    def launch(name):
        log = out / f"{name}.log"
        t0 = time.time()
        with log.open("w") as handle:
            code = subprocess.run([sys.executable, str(HERE / "run.py"), *jobs[name]], stdout=handle, stderr=subprocess.STDOUT).returncode
        manifest["results"][name] = {"exit_code": code, "minutes": round((time.time() - t0) / 60, 1), "log": log.name}
        save()
        print(f"{name}: exit {code} after {manifest['results'][name]['minutes']} min", flush=True)

    save()
    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as pool:
        list(pool.map(launch, names))


if __name__ == "__main__":
    main()
