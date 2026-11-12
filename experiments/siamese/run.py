"""Run siamese worlds: N agents sharing one capped context file, each solving its own LiveCodeBench problems.

    uv run --env-file .env python experiments/siamese/run.py provider=openrouter \
        'world.agent_providers=[{model: deepseek/deepseek-v4.1-flash}, {model: anthropic/claude-haiku-5.5}]' \
        world.seeds=[1,2,3] output.batch=pilot

One folder per world: <output.directory>/<output.name>/<output.batch>/<cell>/world_s<seed>/ (see siam/world.py).
A world whose world.json says complete is skipped, so a cut-off batch can be run again.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_first = [str(HERE), str(REPO / "experiments" / "context_file"), str(REPO / "src")]   # this folder first: both have run.py
sys.path[:] = _first + [p for p in sys.path if p not in _first]

REASONING = {"low": {"effort": "low"}, "off": {"enabled": False}}


def compose(overrides=()):
    from hydra import compose as hydra_compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(HERE / "configs")):
        return hydra_compose("siamese", overrides=[f"hydra.searchpath=[file://{REPO / 'configs'}]", *overrides])


def with_reasoning(cfg):
    """reasoning=low|off -> provider.extra_body.reasoning for every agent (an agent override with extra_body wins)."""
    from omegaconf import OmegaConf
    mode = str(cfg.get("reasoning", "default"))
    if mode == "default":
        return cfg
    if mode not in REASONING:
        raise ValueError(f"reasoning must be default, low or off (got {mode})")
    OmegaConf.set_struct(cfg, False)
    cfg.provider.extra_body = OmegaConf.merge(cfg.provider.get("extra_body") or {}, {"reasoning": REASONING[mode]})
    return cfg


def models(cfg) -> list[str]:
    over = list(cfg.world.get("agent_providers") or [])
    return [str((over[(k - 1) % len(over)] if over else {}).get("model", cfg.provider.model)) for k in range(1, int(cfg.world.n_agents) + 1)]


def cell_name(cfg) -> str:
    if cfg.output.get("cell"):
        return str(cfg.output.cell)
    short = lambda m: m.split("/")[-1].replace("deepseek-", "ds-").replace("claude-", "").replace(".", "")
    n = int(cfg.world.n_agents)
    sharing = "single" if n == 1 else str(cfg.world.sharing)
    cap = f"b{int(cfg.ctx.budget_tokens)}" if cfg.ctx.get("cap", True) else "uncapped"
    rel = "" if n == 1 else f"_{cfg.world.relation}"
    tc = cfg.get("task") or {}
    cep = f"_cep-{tc.arrangement}-g{int(tc.gap)}" if str(tc.get("name", "lcb")) == "codeeval_pro" else ""
    return f"{sharing}{rel}{cep}_{'+'.join(short(m) for m in models(cfg))}_{cap}_r{cfg.get('reasoning', 'default')}"


def run(cfg, make_provider=None, grader_factory=None):
    from siam.problems import load, select, streams
    from siam.world import World
    from ctxfile.sandbox import stop_all
    if make_provider is None:
        from gap_attractor.providers import make_provider
    tc = cfg.get("task") or {}
    cep = str(tc.get("name", "lcb")) == "codeeval_pro"
    if grader_factory is None:
        if cep:
            from siam.cep_grader import CepGrader
            grader_factory = lambda: CepGrader(cfg.cep_grader)
        else:
            from siam.grader import Grader
            grader_factory = lambda: Grader(cfg.grader)
    cfg = with_reasoning(cfg)
    pc = cfg.problems
    open_level = str(tc.get("level", "weak")) if str(tc.get("name", "lcb")) == "open" else None   # task dial: no problems
    chain = str(tc.get("name", "lcb")) == "state_chain"     # binding by state: generated shell tasks, no grader
    pool = [] if open_level or cep or chain else select(load(pc.file), list(pc.difficulty or []), list(pc.platform or []),
                                               pc.start_date, pc.end_date)
    root = Path(cfg.output.directory) / str(cfg.output.name) / str(cfg.output.batch) / cell_name(cfg)
    root.mkdir(parents=True, exist_ok=True)

    def one(seed):
        out = root / f"world_s{seed}"
        done = out / "world.json"
        if done.exists() and json.loads(done.read_text()).get("status") == "complete":
            return {"seed": seed, "status": "skipped (complete)"}
        if out.exists():                      # a cut-off world: keep it aside, start again
            out.rename(out.with_name(out.name + f"_cut{len(list(root.glob(out.name + '_cut*')))}"))
        if open_level:
            from siam.tasks.open import OpenTask
            task = OpenTask(open_level)
            return World(cfg, int(seed), task.streams(int(cfg.world.n_agents)), out, task, make_provider,
                         label=f"{cell_name(cfg)}/s{seed}").run()
        if chain:                             # manifest = ground truth (every answer and link)
            from siam.tasks import state_chain
            st, manifest = state_chain.build(int(cfg.world.n_agents), int(seed), int(tc.length),
                                             gate=bool(tc.get("gate", False)))
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            adv = str(tc.get("adversarial", "") or "")
            if adv and str(cfg.world.setup) != "external":
                raise SystemExit("task.adversarial needs world.setup=external (elsewhere a statement can reach the shared file)")
            if adv:
                manifest["adversarial"], manifest["adv"] = adv, state_chain.adversarial_texts(int(seed))
                (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            return World(cfg, int(seed), st, out, state_chain.StateChainTask(bool(tc.get("hint", False)), adv, int(seed)), make_provider,
                         label=f"{cell_name(cfg)}/s{seed}").run()
        grader = grader_factory()
        try:
            if cep:                           # CodeEval-Pro: streams with known dependencies; manifest = ground truth
                from siam.tasks import cep_streams, codeeval_pro
                st, manifest = cep_streams.build(codeeval_pro.load(), int(cfg.world.n_agents), int(seed),
                                                 str(tc.arrangement), int(tc.length), int(tc.gap),
                                                 rotate=int(tc.get("rotate") or 0), gate=bool(tc.get("gate", False)))
                out.mkdir(parents=True, exist_ok=True)
                (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
                return World(cfg, int(seed), st, out, codeeval_pro.CodeEvalProTask(grader), make_provider,
                             label=f"{cell_name(cfg)}/s{seed}").run()
            st = streams(pool, int(cfg.world.n_agents), int(pc.per_agent), str(cfg.world.relation), int(seed))
            return World(cfg, int(seed), st, out, grader, make_provider, label=f"{cell_name(cfg)}/s{seed}").run()
        finally:
            grader.close()

    try:
        seeds = list(cfg.world.seeds)
        with ThreadPoolExecutor(max_workers=max(1, int(cfg.get("parallel", 1)))) as ex:
            results = list(ex.map(one, seeds))
    finally:
        stop_all()
    (root / "cell.json").write_text(json.dumps({"cell": cell_name(cfg), "pool": len(pool), "results": results}, indent=2), encoding="utf-8")
    return root, results


def main(argv=None):
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # a stopped run (kill, timeout) still removes its containers
    root, results = run(compose(list(sys.argv[1:] if argv is None else argv)))
    for r in results:
        print(json.dumps(r))
    print(f"-> {root}")


if __name__ == "__main__":
    main()
