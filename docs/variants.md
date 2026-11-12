# Running the three variants

Pick a variant on the command line. Everything else (provider, dump, steps) works as before.

| Variant | What the model can do | Command option |
|---|---|---|
| **A** (default) | Start page from a list, then follow one link per step | `variant=a_hub` |
| **B** | Search once at the start (logged as step 0), open a result, then follow links | `variant=b_search_first` |
| **C** | Every step: follow a link **or** search; search results replace the links for that step | `variant=c_search_anytime` |

In all variants the model sees only the cleaned article and titles (plus snippets for search results).
Positions, links, page ids and categories are logged, never shown. The condition labels
(`variant-a_hub-start` etc.) only appear in folder names.

## One-time setup for B and C: build the search index

```bash
uv run python -m gap_attractor.search.index build \
  --index-file data/wikipedia/enwiki-20260901/enwiki-20260901-pages-articles-multistream-index.txt.bz2 \
  --out data/wikipedia/search/enwiki-20260901-titles.sqlite
```

Then add to `.env`:
```
GAP_ATTRACTOR_SEARCH_INDEX=data/wikipedia/search/enwiki-20260901-titles.sqlite
```
Check it: `uv run python -m gap_attractor.search.index query --db <that path> "black holes"`

The fixture (`wikipedia=fixture`) needs no index: it searches its own titles in memory.

## Run

```bash
# A: 10 start pages from batch.start_pages (or variant.start_pages_file, see below)
uv run --env-file .env python -m gap_attractor.cli variant=a_hub provider=groq wikipedia=dump "batch.start_pages=['Pumice','Tetrahedron']" batch.index=1

# B: 10 episodes, each starting with a search
uv run --env-file .env python -m gap_attractor.cli variant=b_search_first provider=groq wikipedia=dump variant.n_runs=10 batch.index=2

# C: same start pages as A
uv run --env-file .env python -m gap_attractor.cli variant=c_search_anytime provider=groq wikipedia=dump "batch.start_pages=['Pumice','Tetrahedron']" batch.index=3

# free offline check of all three (mock model + fixture)
uv run python -m gap_attractor.cli variant=a_hub
uv run python -m gap_attractor.cli variant=b_search_first variant.n_runs=2
uv run python -m gap_attractor.cli variant=c_search_anytime
```

Use a new `batch.index` for each run on the same day. Check any run with
`uv run --env-file .env python scripts/check_steps.py`.

## Hub start pages (A and C)

Put Claire's list in a text file, one title per line, and set
`variant.start_pages_file=<path>`. The run then samples `variant.n_runs` pages from it using the experiment seed,
so the same seed gives the same start pages.

## Dump file names

The config defaults to file names starting `enwiki-latest-`. For a dated dump, set in `.env`:
```
GAP_ATTRACTOR_DUMP_XML=enwiki-20260901-pages-articles-multistream.xml.bz2
GAP_ATTRACTOR_DUMP_INDEX=enwiki-20260901-pages-articles-multistream-index.txt.bz2
```

## What changed in the step record

- `action`: `follow`, `search` (C) or `search_start` (B, step 0).
- `search`: `null` for link steps; for search steps `{query, fts_mode, n_hits, results: [...]}`.
  Each result has the same fields as a link (`original_position` = search rank, `shuffled_position` = shown position,
  `title`, `hyperlink`, `page_id`, `semantic_cluster_label`) plus `snippet`, `snippet_source`, `matched_title`, `via_redirect`.
- `extras.chosen_position` / `chosen_original_position` refer to the search results on search steps, to the links otherwise.
- `extras.tool_order` (C only): which action was described first; it alternates every step (cyclic shift).

## Open design points

- Variant C uses a single call (page + links + both options), then a second call only if it searched.
  The alternative "two-step" design is still to be agreed with Sergei.
- B's step 0 is not counted in `experiment.step_limit` (link steps only), so B runs have one extra step.

## Memory: how earlier steps are shown (`experiment.memory`)

Works the same in all three variants. The default keeps the current behaviour. The setting is saved in every
run's `config.json`, so results from different memory setups never get mixed up.

| Setting | Earlier steps appear as | Cost |
|---|---|---|
| `path_only` (default) | its previous reasoning + choice only | low |
| `pages_no_links` | earlier articles in full, without their link lists | medium |
| `lead` | each earlier page's lead paragraph | low |
| `sliding_window` | last `experiment.memory_window` pages in full (default 2), older ones as a path line | medium |
| `self_notes` | no pages: the model adds a "note" to each reply and sees its path + notes (no extra call) | low |
| `full` | everything it saw at every step | very high |

```bash
uv run --env-file .env python -m gap_attractor.cli variant=a_hub experiment.memory=lead ...
uv run --env-file .env python -m gap_attractor.cli variant=a_hub experiment.memory=sliding_window experiment.memory_window=3 ...
```
All settings except `path_only` send earlier steps as alternating user/assistant messages. `path_only` is left
exactly as before (replies only) so earlier runs stay comparable. `model_inputs/step_XXX_request.txt` shows exactly
what the memory looked like at each step; with `self_notes`, each step's note is logged as `model_note`.

## Intent (`experiment.intent`), off by default

| Setting | What happens each step |
|---|---|
| `none` (default) | Exactly as before: no intent call |
| `probe` | First a separate call shows the page **without links** and asks which article it would open next. Then the normal choice call runs, unchanged. The intent is logged, never shown back to the model |
| `walk` | Variant C only: the model goes wherever its intent points (resolved by local search). No link list is shown |

```bash
uv run --env-file .env python -m gap_attractor.cli variant=a_hub experiment.intent=probe ...
uv run --env-file .env python -m gap_attractor.cli variant=c_search_anytime experiment.intent=walk ...
```

`probe` works in all variants and with every memory setting (the intent call uses the same memory as the choice
call). Variant B's search step (step 0) has no page, so it gets no intent. On the dump, the intent is matched to an
article with the search index, so `GAP_ATTRACTOR_SEARCH_INDEX` must be set in `.env` (also for Variant A).

Each step's `intent` record (null when off):
- `text`: what the model named; `mapped_title` / `mapped_page_id`: the article it maps to; `match_method`:
  `link_title` (exactly a link on the page) or `search`; `top_results`: top 3 search matches, to check the mapping.
- `in_links`: was the intended article on the page's list? `intended_shuffled_position` / `intended_original_position`:
  where it sat in the list shown / on the wiki page.
- `choice_matches_intent`: did the actual click go there? `already_visited`, `is_current_page`.
- `raw`, `parse_ok`, `tokens`, `latency_s` of the intent call.

`summary.json` adds `intent_steps` and `intent_matches`; `trajectory.jsonl` / `pages.csv` add `intent`,
`intent_mapped_title`, `intent_in_links`, `choice_matches_intent`. The exact intent request is saved as
`model_inputs/step_XXX_request_intent.txt`. Note that `probe` makes one extra call per step (without the link list,
so smaller than the choice call).