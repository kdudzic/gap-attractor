"""Run the context-file experiment: single agents (ctx=single) or joint worlds (ctx=joint).

    uv run --env-file .env python experiments/context_file/run.py provider=openrouter \
        provider.model=deepseek/deepseek-v4.1-flash wikipedia=dump batch.start_pages=[Spreadsheet]
    uv run --env-file .env python experiments/context_file/run.py ctx=joint wikipedia=dump ...

The harness's configs/ are composed first (provider, wikipedia, experiment, prompt, tools, variant), then this
folder's configs/ (ctx, …); any key of either can be overridden on the command line.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def compose(overrides=()):
    from hydra import compose as hydra_compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(HERE / "configs")):
        return hydra_compose("context_file", overrides=[f"hydra.searchpath=[file://{REPO / 'configs'}]", *overrides])


def run(cfg):
    import ctxfile.memory  # noqa: F401  (registers experiment.memory=context_file)
    from ctxfile.sandbox import stop_all
    try:
        if str(cfg.ctx.mode) == "joint":
            from ctxfile.joint import run_worlds
            return run_worlds(cfg, REPO)
        from gap_attractor.harness.runner import run_experiment
        return run_experiment(cfg, REPO)
    finally:
        stop_all()                    # bash interface: no container outlives the run


def main(argv=None):
    dataset, results = run(compose(list(sys.argv[1:] if argv is None else argv)))
    print(f"{len(results)} run(s) in {dataset.root}")


if __name__ == "__main__":
    main()
