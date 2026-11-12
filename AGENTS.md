# Guide for coding agents

This repository is the public version of the Gap Attractor project. Read [README.md](README.md) and
[docs/README.md](docs/README.md) first.

- Install and test: `uv sync --extra dev --extra analysis`, then `uv run pytest`. The tests run offline.
- Use `provider=mock wikipedia=fixture` for any change to logging, parsing or control flow. It costs nothing.
- Real-provider runs cost money. Do not start them unless asked.
- Never edit the run folders under `data/sample_runs/`.
- Keep the three logging layers (dataset, run, step) and keep raw model output next to parsed output. If a logged
  field is added, removed or changes meaning, bump `SCHEMA_VERSION` in `src/gap_attractor/harness/runner.py` and
  note it in `docs/technical-changelog.md`.
- Do not pool runs across the condition fields listed under `group_by` in `configs/analysis.yaml`.
- Secrets live only in `.env`. Never print a key or write one into a log.

Most code and documents here are generated from the team's working repository (the list is in
`.public-sync.json`). The hand-written files are the top-level `README.md`, `NOTICE.md`, `CITATION.cff`,
`.env.example`, this file, `docs/README.md`, `docs/results.md`, `docs/reproduce.md`, `docs/figures/`, `data/`,
and the READMEs under `experiments/` and `scripts/graph/`.
