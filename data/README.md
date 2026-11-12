# Data in this repository

Only small data is kept here. The Wikipedia dump, the lookups built from it and the full run logs are not; see
[../docs/reproduce.md](../docs/reproduce.md) for how to get or build them. Folders created by that guide
(`data/wikipedia/`, `data/cache/`, `data/graph/`) are ignored by git.

## `start_pages/`: where runs begin

| File | Content |
|---|---|
| `hubs_A_natural.txt` | 492 titles of well-connected "hub" articles, one per line. The main experiments sample their 20 start pages from this file |
| `start_set_A_final.tsv` | The same pages with `page_id`, `key`, merged out-degree, content in-degree and topic |
| `hubs_B_stratified.txt`, `start_set_B_final.tsv` | A second pool of 465 pages, stratified by topic |

Titles refer to the English Wikipedia dump of 1 September 2026.

## `sample_runs/`: 14 real runs

All 14 runs start on the page "Albert Schweitzer" with the same run seed, so every model saw the same links in
the same order. They are a sample for reading and for trying the analysis, not a dataset to draw conclusions
from.

| Experiment folder | Runs | Condition |
|---|---|---|
| `e0_baseline/` `*_stop-false` | 5 models | No STOP option, step limit not disclosed |
| `e0_baseline/` `*_stop-true` | 5 models | STOP option offered |
| `e3_step_budget/` `*_budget-total_stop-false` | 4 models | No STOP option; the prompt ends with "You can take 15 steps through links." |

Models: DeepSeek V4.1 Flash, Gemini 3.8 Flash, Grok 4.7, MiMo 2.6 Pro, Qwen 3.8 Flash (no Gemini in the budget
arm). Run on 21 to 23 October 2026 through OpenRouter, log schema 3.11.

What the sample shows, in line with [../docs/results.md](../docs/results.md):

- Without a STOP option, all five models make the same first click, "Reverence for Life".
- Three of the five then pace between that page and the start page for the rest of the run.
- With the step-budget sentence, DeepSeek and Qwen leave the loop; MiMo does not.
- With a STOP option, DeepSeek and Gemini stop on the first page.

Files in each run folder:

| File | Content |
|---|---|
| `transcript.md` | The run in readable form: each page, the links shown, the choice and the stated reason. Start here |
| `steps.jsonl` | One record per step: the full choice set, raw model output, parsed choice, reasoning, token counts. The analysis reads this |
| `run.json`, `config.json` | Settings of the run, including the prompt as sent |
| `summary.json` | How the run ended |

A complete run folder also holds the exact request text of every step, the article text shown, and per-step
JSON files. They are left out here to keep the sample small.

Try the analysis on the sample:

```bash
uv run --extra analysis gap-attractor-analyze 'data.paths=[data/sample_runs]' 'stages=[outcomes]'
```

**Changes made for publication.** File paths inside the logs pointed at the team's server; they were replaced
with repo-relative paths (`data/...`, `runs/...`). Nothing else was edited.

**Licence.** The step logs contain Wikipedia page titles, category names and link structure, and outputs of the
models named above. See [../NOTICE.md](../NOTICE.md).
