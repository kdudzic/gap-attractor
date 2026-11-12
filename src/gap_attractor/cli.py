from pathlib import Path

import hydra

from gap_attractor.harness.runner import run_experiment

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    """Run one batch. Overrides as before (`variant=b_search_first experiment.step_limit=5`), plus Hydra's own flags:
    `-m` sweeps (`-m variant=a_hub,c_search_anytime experiment.seed=1,2`), `--cfg job` prints the composed config,
    `--help` lists the config groups. Hydra writes no files of its own (see the hydra: block in configs/config.yaml)."""
    dataset, results = run_experiment(cfg, Path.cwd())
    print(f"batch: {dataset.root}")
    for result in results: print(f"{result['run_id']}: {result['n_steps']} steps, {result['termination_reason']}")


if __name__ == "__main__": main()
