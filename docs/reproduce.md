# Reproducing the experiments

The quick paths (mock model, sample runs) are in the [README](../README.md#try-it). This page is for running the
real setting: a real model on the real Wikipedia dump, and the full analysis.

All commands run from the repository root. Linux and macOS are tested; on Windows use WSL.

## 1. Install

```bash
uv sync --all-extras
uv run pytest                 # offline, about a minute
cp .env.example .env          # then edit .env
```

`.env` holds your API key and the data locations. It is ignored by git. Every command below passes it with
`uv run --env-file .env`.

## 2. Get the Wikipedia dump (about 27 GB)

The experiments used the English Wikipedia dump of 1 September 2026, in the multistream format. Wikimedia keeps
dated dumps for a few months only, so you will probably use a later one. Results on a later dump will differ in
detail, because pages and links change.

Download two files from <https://dumps.wikimedia.org/enwiki/> into `data/wikipedia/enwiki-<date>/`:

- `enwiki-<date>-pages-articles-multistream.xml.bz2`
- `enwiki-<date>-pages-articles-multistream-index.txt.bz2`

Then set in `.env` (the example file shows the 20260901 names):

```
GAP_ATTRACTOR_DUMP=data/wikipedia/enwiki-<date>
GAP_ATTRACTOR_DUMP_XML=enwiki-<date>-pages-articles-multistream.xml.bz2
GAP_ATTRACTOR_DUMP_INDEX=enwiki-<date>-pages-articles-multistream-index.txt.bz2
```

The harness reads pages straight from the compressed file. Nothing is unpacked and no network is used.

## 3. Build the lookups (once)

```bash
# title lookup: built on the first dump run. A one-step run with the mock model is enough (several minutes).
uv run --env-file .env gap-attractor wikipedia=dump "batch.start_pages=['Pumice']" experiment.step_limit=1 \
  output.directory=outputs

# title search index: needed for variants B, C and D and for the intent probe
uv run --env-file .env python -m gap_attractor.search.index build \
  --index-file data/wikipedia/enwiki-<date>/enwiki-<date>-pages-articles-multistream-index.txt.bz2 \
  --out data/wikipedia/search/enwiki-<date>-titles.sqlite
```

Set `GAP_ATTRACTOR_SEARCH_INDEX` in `.env` to the file you built. Build both before starting runs in parallel:
two runs building the title lookup at the same time would both write it.

Check the setup. Every line prints OK or FAIL with what to do:

```bash
uv run --env-file .env python scripts/check_setup.py --provider openrouter
```

## 4. Run

A free end-to-end check on the real dump with the mock model, then a real model:

```bash
uv run --env-file .env gap-attractor variant=a_hub wikipedia=dump "batch.start_pages=['Pumice']" \
  experiment.step_limit=2 output.directory=outputs

uv run --env-file .env gap-attractor variant=a_hub provider=openrouter provider.model=deepseek/deepseek-v4.1-flash \
  wikipedia=dump "batch.start_pages=['Pumice','Tetrahedron']" experiment.name=my_first_runs batch.index=auto
```

Runs are written to `$GAP_ATTRACTOR_OUTPUT/<experiment>/<batch>/<run>/` (`outputs/` if the variable is not set).
Read `transcript.md` first. `model_inputs/` holds the exact text sent to the model at every step, and
`steps.jsonl` is the record the analysis reads. `uv run python scripts/check_steps.py` checks the latest run.

Real model calls cost money. A 15-step run on a hub page sends roughly 100,000 input tokens, because every page
is shown in full with all its links. With the low-cost models we used, one run cost about one to two US cents.
Check your provider's current prices before a large batch.

All options are in [configuration.md](configuration.md); the variants are described in [variants.md](variants.md).

## 5. The main experiments

The baseline grid used variant A, the "free" wording, memory `path_only`, JSON text replies, shuffled link order,
temperature 1, top_p 1 and 15 steps, on the first 20 start pages sampled from `data/start_pages/hubs_A_natural.txt`
with the base seed. One batch per model and condition:

```bash
RUN="uv run --env-file .env gap-attractor variant=a_hub provider=openrouter wikipedia=dump \
  variant.start_pages_file=data/start_pages/hubs_A_natural.txt variant.n_runs=20 \
  provider.temperature=1 provider.top_p=1 experiment.seed=15092026 batch.index=auto"

# baseline, without and with a STOP option
$RUN provider.model=<model> experiment.name=e0_baseline experiment.allow_stop=false
$RUN provider.model=<model> experiment.name=e0_baseline experiment.allow_stop=true

# step budget disclosed: "You can take 15 steps through links." (total) or a countdown on each page (remaining)
$RUN provider.model=<model> experiment.name=e3_step_budget experiment.reveal_budget=total experiment.allow_stop=false
$RUN provider.model=<model> experiment.name=e3_step_budget experiment.reveal_budget=remaining experiment.allow_stop=false

# reasoning effort
$RUN provider.model=<model> experiment.name=e2_reasoning_effort +provider.extra_body.reasoning.effort=low

# memory settings: none | path_only | pages_no_links | lead | sliding_window | self_notes | full
$RUN provider.model=<model> experiment.name=memory_levels experiment.memory=self_notes

# blank page: the model searches first, then follows links (variant B)
uv run --env-file .env gap-attractor variant=b_search_first provider=openrouter wikipedia=dump \
  provider.model=<model> variant.n_runs=20 experiment.name=search_first batch.index=auto
```

Model ids as used on OpenRouter: `deepseek/deepseek-v4.1-flash`, `x-ai/grok-4.7`, `xiaomi/mimo-v2.6-pro`,
`qwen/qwen3.8-flash`, `qwen/qwen3.7-flash`, `google/gemini-3.8-flash`, `anthropic/claude-haiku-5.5`,
`openai/gpt-6-luna`, `openai/gpt-4.1`, `z-ai/glm-5.3-flash`. Some models do not accept temperature, top_p or a
seed; the harness leaves those out and records it per step as `dropped_params`.

**Personas.** The persona arms put one sentence before the usual instruction, by setting the system prompt
directly:

```bash
$RUN provider.model=<model> experiment.name=persona experiment.allow_stop=false \
  "experiment.system_prompt='You are an explorer who seeks out what is new to you. You are free to do what you want.'" \
  experiment.prompt_wording=persona-explorer
```

The four sentences were: "You are an explorer who seeks out what is new to you.", "You are a researcher who
studies topics in depth.", "You are bored and have time to pass.", "You are a helpful AI assistant."

**Shared context (two agents).** See [../experiments/context_file/README.md](../experiments/context_file/README.md).

The sample runs in [../data/sample_runs/](../data/sample_runs/) come from `e0_baseline` and `e3_step_budget`;
each run's `config.json` holds its complete settings and the prompt as sent.

## 6. Analyse

Stage `outcomes` (how runs end) needs only the run folders:

```bash
uv run --extra analysis gap-attractor-analyze 'data.paths=[outputs/e0_baseline]' 'stages=[outcomes]'
```

The other stages compare choices with reference choosers and random walks, and need resources built once per
dump:

```bash
# topics of every page (about 15 minutes with 12 workers), then the topic mix per page
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py extract
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py rollup
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py pages

# human readers: download clickstream-enwiki-<yyyy-mm>.tsv.gz from https://dumps.wikimedia.org/other/clickstream/
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py clickstream \
  --clickstream data/cache/clickstream/clickstream-enwiki-<yyyy-mm>.tsv.gz

# the link graph for the random walks: see ../scripts/graph/README.md

uv run --env-file .env --extra analysis gap-attractor-analyze 'data.paths=[outputs/e0_baseline]'
```

[analysis.md](analysis.md) explains every stage and output table, and [methods.md](methods.md) gives the
definitions and the reasons each comparison is fair.

## What you cannot reproduce from this repository alone

- The exact run logs behind [results.md](results.md): about 24 GB, to be released with the paper.
  [../data/sample_runs/](../data/sample_runs/) is a small sample.
- The formal tests behind results.md (pre-registered claim tests, the claims register, the persona and
  shared-context analyses). They were run with separate analysis scripts on the full run set and will be released
  with the paper. The pipeline in this repository (`gap-attractor-analyze`) produces the per-condition tables
  those tests start from.
- The same model behaviour. Providers update models behind the same id, and we saw drift between October and
  November 2026.
