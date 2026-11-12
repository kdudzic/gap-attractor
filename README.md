# The Gap Attractor: Investigating Autonomous Exploration Behavior in LLM Agents

This repository is one of the deliverables of the Gap Attractor project conducted as part of the Sentient Futures Project Incubator (September to November
2026). It holds the experiment harness, the analysis pipeline, a sample of real runs, and a summary of the
results.

What happens when LLM agents are placed in an open environment without an explicit goal or reward? Using Wikipedia as the exploration environment, we investigate how agents navigate it under varying experimental conditions, looking for consistent patterns, biases, and behavioral attractors. We share our framework, observations, and implications.

**Explore the runs:** the interactive [Field Map](https://observational.github.io/navigators/) shows where the
agents went, next to where human readers click.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer. Clone with git, because the harness records
its own commit in every run.

**1. Run the harness offline (no key, no download, about a minute).** A mock model walks a small synthetic wiki:

```bash
uv sync --extra dev
uv run gap-attractor provider=mock wikipedia=fixture experiment=smoke
uv run pytest
```

The run is written to `outputs/`. Open its `transcript.md` to see what a run looks like.

**2. Read and analyse real runs (no key, no download).** [data/sample_runs/](data/sample_runs/) holds 14 real runs
from the start page "Albert Schweitzer": five models with and without a STOP option, and four with the
step-budget sentence. All five models make the same first click.

```bash
uv sync --extra dev --extra analysis
uv run gap-attractor-analyze 'data.paths=[data/sample_runs]' 'stages=[outcomes]'
```

**3. Run a real model on the synthetic wiki (API key, a few cents).** Copy `.env.example` to `.env`, add an
[OpenRouter](https://openrouter.ai/) key, then:

```bash
uv run --env-file .env gap-attractor provider=openrouter provider.model=deepseek/deepseek-v4.1-flash \
  wikipedia=fixture experiment.step_limit=5
```

**4. Reproduce the real setting.** This needs the English Wikipedia dump (about 27 GB) and one-time index builds.
[docs/reproduce.md](docs/reproduce.md) has the steps and the commands behind the main experiments.

## What is in this repository

| Path | Content |
|---|---|
| `src/gap_attractor/harness/` | The run loop, memory plug-ins, logging |
| `src/gap_attractor/wikipedia/`, `search/` | Offline dump reader, link extraction, title search |
| `src/gap_attractor/providers/` | Model providers: mock, OpenRouter, OpenAI, Groq |
| `src/gap_attractor/analysis/` | `gap-attractor-analyze`: outcomes, clicks, walks, hypotheses, convergence, loops |
| `configs/` | Hydra configs: variants, prompts, providers, experiments, analysis |
| `scripts/` | Resource builders (topics, clickstream, link graph) and log checks |
| `experiments/context_file/` | Agents whose memory is a context file they edit, alone or shared by two agents |
| `experiments/siamese/` | Side study: several agents share one capped context while solving coding tasks |
| `data/start_pages/` | The start-page pools |
| `data/sample_runs/` | 14 real runs with full step logs |
| `tests/` | Test suite; runs offline on a synthetic wiki |
| `docs/` | Results, methods, design, configuration, reproduction guide. Start at [docs/README.md](docs/README.md) |

## Team

Claire Xu, Gagana M D, Kacper Dudzic, Pooja Manvikar, Sachin Bagoriya, Sebastian Abundis, Sergei Kudriashov,
Sydney Bosibori and Yuliya Antanovich (in alphabetical order by first name). Mentor: Julia Bossmann.

The project was carried out in the Sentient Futures Project Incubator, autumn 2026.

## Licence and credits

Code and documentation: [CC BY-NC 4.0](LICENSE), free to use and adapt with credit for noncommercial purposes. Wikipedia-derived
data (CC BY-SA 4.0) and third-party material: see [NOTICE.md](NOTICE.md).
To cite the project, use [CITATION.cff](CITATION.cff).
