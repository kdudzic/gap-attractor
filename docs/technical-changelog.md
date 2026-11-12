# Technical changelog

## 2026-11-12 — schema 3.19: the side branches merged into main

Main now holds kacper/work, codex/memory-history-control, codex/perturb-return-pilot, codex/attractor-mask,
xenomirant/reasoning-table, xenomirant/ctx-inject and xenomirant/siamese. The branches had numbered their schema
changes independently, so 3.11-3.13 each mean two things. A run's `code.git_commit` (dataset.json, summary.json)
says which; this table maps the numbers.

| schema | branch, commit | change |
|---|---|---|
| 3.11 | main | Memory.observe_call, goal endings |
| 3.11 | kacper/work 6cdcc09 | `experiment.reveal_budget` (none / total / remaining) |
| 3.12 | main ff3f6e6 | termination `no_valid_move` |
| 3.12 | codex/memory-history-control 164ab9a | paired post-run readouts |
| 3.13 | codex/memory-history-control 2f640be | direct page-collection experiment |
| 3.13 | codex/attractor-mask db1946a | auditable link masks |
| 3.14-3.18 | codex/attractor-mask 3abfa6e, d2b99c4, 9782460, 2a42e0a, 0066dd5 | semantic masks, pre-run self-prediction, forecaster control, forecasts; later commits on 3.18 add search-aware revisit controls |
| 3.19 | main | all of the above together |

Code behind the Nov 2026 runs: persona, scale and repeat runs 6cdcc09; time capsule 2f640be; visited mask 9cec9a9;
search access 41b8d4f; shared-context pilot and confirmation ac02f41 plus the curation instruction (81903c2), which
predates d545615 (those runs had `ctx.shrink_edits=true`).

## 2026-10-23 — one team folder for runs

- New runs go to `runs/<experiment>/...` (`GAP_ATTRACTOR_OUTPUT=runs`), not
  `data/results/<name>/`; who ran a run is in its metadata and folder owner, not its path. Old runs stay where
  they are. Rules: `runs/README.md`. Docs, `.env.example` and the config comment point there.
- `scripts/index_runs.py` writes `runs/INDEX.csv`: one row per run in both places, the Run Log columns
  plus Account (folder owner), Reasoning and Superseded. Code and schema unchanged.

## 2026-10-23 — opt-in Variant D: search-first and search-anytime

- `variant=d_search_first_anytime` combines the existing Variant B opening search with Variant C's later
  follow-or-search action loop. The opening search is step 0 and does not consume `experiment.step_limit`.
- Together A/B/C/D form a 2×2 design: fixed versus model-searched start, crossed with links-only versus search
  available later. Variant D defaults to seeded search-result order; matched comparisons should explicitly use
  that order in every search-enabled arm.
- Existing A/B/C configs and requests are unchanged. No schema field was added or changed.

## 2026-10-22 — schema 3.12: termination `no_valid_move`

- Three steps in a row where no listed page was chosen (each reply and its retry unusable, so a random link was
  opened) used to end the run as `parse_error` (`C-technical`, run log status "Technical failure"). In the
  context-file pilots almost all of these replies were readable: the model wrote a closing summary instead of a
  move, declined to continue, or named a page not on the list. That is the model's behaviour, not a harness fault.
- Such a run now ends as `no_valid_move` (`A-normal`, status "Completed"). It stays `parse_error` (`C-technical`)
  when any failed reply in the streak was empty (no text, no tool call) or cut off at the token limit
  (`finish_reason` "length"); `runner.unreadable` decides. `failures.jsonl` rows gain `cut_off`.
- `outcomes.csv` gains a `no_valid_move` column (no longer counted in `other_endings`); like other competing
  endings it censors the paired stop comparison.
- Nothing the model sees changes. Runs written before 3.12 keep `parse_error`.

## 2026-10-21 — tool parameters can be removed; an essay stop

- `tool_specs` drops a tool parameter that a config sets to null, so a config can replace a tool's parameters
  (e.g. the context-file experiment's `task=essay` gives `stop` one parameter, `essay`, instead of `reason`).
- `tool_call_as_text` reads a stop's statement from `reasoning`, `reason` or `essay`. Built-in configs are unchanged
  (no null parameters; no `essay`), so their requests and logs are byte-identical.

## 2026-10-21 — schema 3.11: Memory.observe_call, goal endings

- New optional hook `Memory.observe_call(request, tools, usage)`: every API call that got an answer, with the tools
  offered and the provider's usage (prompt tokens, cached tokens, cost). Built-in memories ignore it, so their
  requests and logs are unchanged. The context-file memory uses it to size its context in each model's own tokens
  (calibrated on the billed prompt tokens) and to log every call (`calls.jsonl`; `ctx.save_requests` also saves the
  requests).
- `OUTCOME_CATEGORY` gains `reached_target` and `world_stopped` (A-normal). They are the endings of the
  context-file goal worlds (`ctx.goal.task=find`): an agent arrived at its target page, or another agent's arrival
  ended the world. Nothing else ends that way.

## 2026-10-16 — schema 3.9: nothing the model sees names "the harness"

- The note shown after a random pick (`experiment.harness_pick_note`, default in `runner.HARNESS_PICK_NOTE` and
  `configs/experiment/smoke.yaml`) was 'Your reply could not be read, so the harness opened "X" at random.' It named
  our apparatus, and models in the context-file pilots took the word up in their own reasoning ("if the harness
  blocks it", "harness error"). It is now 'No listed page was chosen, so "X" was opened at random.' (also more
  accurate: the reply was often readable but named no listed page).
- The path line of the `self_notes` and `sliding_window` memories said "(opened by the harness at random: your reply
  could not be read)"; it is now "(opened at random: no listed page was chosen)".
- Nothing else changes: requests and steps of every built-in memory are byte-identical to 3.8 except in a step after
  a random pick. Runs before 3.9 that had random picks showed the old wording from then on (search
  `harness_handling: skip` in their steps); compare such runs with care.

## 2026-10-16 — Memory.side_exhausted_note (optional)

- A memory may word the request to act once its tool replies for the step are used up
  (`side_exhausted_note(n, tools)`); None, the default, keeps `SIDE_EXHAUSTED_NOTE`. Built-in memories are unchanged
  (byte-identical requests). The context-file experiment uses it: the harness's wording ("turns", "step") was copied
  by models in its pilots.
- (The default random-pick note named "the harness"; changed for everyone in schema 3.9, above.)

## 2026-10-15 — outcomes: competing endings and repeated runs

- `outcomes.csv` gains `context_budget` (runs a memory ended over its budget; no longer in `other_endings`) and
  `stop_hazard` (stops per step taken). An ending other than STOP and the step limit competes with stopping, so
  `stop_share` can differ between conditions with the same per-step tendency to stop; `stop_hazard` does not.
- `paired()`: repeated runs of a start page were silently overwritten (one run per page kept); they are now averaged
  (the test becomes a sign test). A stop counts only within the steps both arms of a start page observed when a run
  of that page ended another way (`censored_pages`). With one run per page and no such endings, results are as
  before.

## 2026-10-15 — schema 3.8: a memory may hold a move back (`Memory.hold_move`)

- New optional hook `Memory.hold_move(call)`, asked for every move (tools mode) before the harness acts on it. A text
  holds the move: the move call is answered with that text, the step's editing turns (`side_turns`) start again, and
  the model is asked again on the same page. None (the default) lets the move go. At most `Caller.MAX_HOLDS` (20)
  holds per step, so a faulty memory cannot loop; an exception in the hook lets the move go.
- Logged per step as `extras.held_moves` ([{call, note}]); each request after a hold is saved as `request_held_N`.
- Built-in memories never hold: requests and step logs of every built-in memory × text/tools × variants A/B/C, with
  and without the intent probe, are byte-identical to 3.7 apart from the schema number.
- Used by the context-file experiment: a move is held until the context fits its budget.

## 2026-10-15 — analysis stage `loops`

- `loops` (analysis/loops.py): per condition, the share of moves that bounce back (A -> B -> A), when the first
  bounce happens, and the escape hazard after c consecutive bounces with `decline` = hazard(1) - hazard(>=3)
  (cluster bootstrap over start pages). Optional paired comparison of bounce_share across one condition field on
  the same start pages (e.g. memory path_only vs none), with an exact sign test.

## 2026-10-15 — analysis stages `hypotheses` and `convergence`; walk baseline `indegree`

- `hypotheses` (analysis/hypotheses.py): per declared hypothesis (a fixed score per shown link: similarity, hub,
  human, novelty, backtrack, self_interest; placebos position and random), the chosen link's mid-rank percentile
  (0.5 under uniform choice) and its lift over reference choosers (uniform, clickstream, indegree), with cluster
  bootstrap intervals, a Monte Carlo test against the reference chooser, Holm within the primary family, and bits
  per step at declared mixing weights. No fitted parameters.
- `convergence` (analysis/convergence.py): pairwise similarity and same-page share of runs from different start
  pages at each hop, and split-half absorption into the most-visited pages, against walks from the same starts.
- Walk baseline `indegree` (out-links in proportion to 1 + in-degree), for the walks and convergence stages.
- Default `stages` unchanged (outcomes, clicks, walks); the new stages run with `stages=[hypotheses,convergence]`.

## 2026-10-15 — schema 3.7: optional context embeddings after every step (`context_embedding=steps`)

- New config group `context_embedding` (default `none`: nothing changes; requests and step logs of every built-in
  memory × text/tools × variants A/B/C, with and without the intent probe, are byte-identical to 3.6).
- With `context_embedding=steps`, at the end of every step (`close_step`) the step's final request
  (`Caller.last_messages`) is embedded (`harness/context_embed.py`):
  - **per turn**, one text per message by default; a memory can give its own turns with the new optional hook
    `Memory.embed_turns(messages)`, as the context-file memory does with its file turns. Vectors are keyed by the
    text, so an unchanged turn keeps its vector across steps;
  - **as a whole**: `direct` (all turns as one text), `mean` (length-weighted mean of the turns), or `both`;
  - texts longer than `max_input_chars` are cut at paragraph breaks and their pieces averaged, never truncated.
- Output: `<run>/context_embedding/steps.jsonl` (per step: turns with id, kind, role, author, characters, vector
  keys) and `vectors.sqlite`; run.json `context_embedding` (embedder, counts, cost). Not part of
  `memory_settings`: it changes nothing the model sees, so it does not split conditions.
- A failed embedding request is logged on its step and the run goes on. Steps without a request (a dead end) are
  not embedded. Needs the analysis extra (numpy); imported only when enabled.
- `load_trace(run_dir)` and `trace_summary(run_dir)` (per-step drift and kept turns) for analysis.

## 2026-10-15 — Memory.should_end: a memory may end a run (termination `context_budget`)

- Before every step the harness asks the memory `should_end()`; a reason ends the run there. Built-in memories never
  do. New termination reason `context_budget` (outcome category `B-memory`): the context-file experiment's hard
  budget (the model did not get its context under the budget in time). Logs of built-in memories unchanged.

## 2026-10-15 — schema 3.6: failures.jsonl, every unusable reply with where its title came from

- Each run folder gets `failures.jsonl` (written only when there are any): one line per unusable reply, retries
  included. Each line has:
  - the attempt number, the invalid type, and the title named;
  - the reply as written (text and tool calls);
  - the current page and the menu size;
  - how the step ended (`step_handling`, `step_choice`);
  - a category (`harness/failures.py`):
    - `near_miss`: a menu title written differently, or a redirect onto one. A title that is an article of its own
      is never a near miss.
    - `earlier_menu`: offered on an earlier page of the run.
    - `others_menu`: on a link list the memory showed from elsewhere, e.g. another agent's page in a shared context
      (`Memory.other_menus`).
    - `linked_not_offered`
    - `mentioned_on_page`
    - `in_dump`: an existing article never shown.
    - `not_in_dump`
    - `no_title`
    - `stop_not_allowed`
  - `in_dump` and `resolved_title` (looked up in the dump) for every named title.
- From 3.6 on, no failures.jsonl means no unusable replies. The steps themselves are unchanged (56 configurations
  checked).
- `scripts/failures.py` collects them across runs: a CSV plus counts per condition and category, and a list of the
  never-shown titles. `--reconstruct` rebuilds the failures of older runs from steps.jsonl (every reply is in
  `model_output_raw`), and `--wikipedia dump` adds the dump lookups.
- First use: the 15 Oct context-file pilot (DeepSeek V4.1 Flash, 12 conditions). Every out-of-range reply named a
  link from one of the run's earlier pages, sometimes by a redirect; none named a page never shown. The `full`
  memory had the most (57 per 100 steps, retries included).

## 2026-10-15 — Memory.observe: every reply as the model wrote it

- In tools mode the harness keeps only the parsed action: text a model writes next to a tool call never reaches the
  logs or the memory. A memory may now implement `observe(request, reply)`: every reply of the run (its text, tool
  calls and any returned reasoning trace) with the request it answered. Built-in memories ignore it; their logs are
  unchanged (checked on all 56 configurations). No schema change.

## 2026-10-15 — schema 3.5: memory-tool turns, effective sampling, finer analysis conditions

From the independent review of the context-file experiment (team analysis notes, not in this repository).
- Memory tools (`Caller`): `side_turns()` now counts replies with memory-tool calls per STEP, retries included
  (before it was per call, so a retry reset it). When they are used up, the model is asked once, with a note and only
  the move tools, to act. Before, the harness answered with an invented assistant reply and the step usually ended
  with a seeded random link. A model that still does not move is retried as usual, with invalid type
  `side-turns-exhausted`.
- A retry or follow-up after memory tools ran builds on the memory's post-edit view (`continue_messages`), not on
  the request before the edits. An exception in `memory.handle` becomes an `ERROR: …` tool result instead of ending
  the batch.
- Logged for steps whose memory offered tools: `extras.side_calls` (turn, calls, results, whether a move came in the
  same reply) and `extras.side_turns_exhausted`; every extra request is saved to `model_inputs/`
  (`request_side_N`, `request_side_exhausted`). A reply with only memory-tool calls is kept in `model_output_raw` as
  `{"memory_tool_calls": [...], "text": ...}`.
- `Memory.settings()` -> run.json `memory_settings`; `Memory.finish(run, reason)` at the end of every run;
  `Memory.step_fields(step)` now gets the step record being logged (its choice is set).
- run.json `effective_sampling`: temperature, top_p and seed as sent (None when the provider left one out),
  `dropped`, and `checked` (False when OpenRouter's model list did not cover the model: attempts then carry
  `params_unchecked`).
- `Run(extra=...)` refuses keys the harness writes itself (`outcome`, `config`, ...).
- Analysis `group_by`: `memory` = `derived.memory_setup` (name + memory_settings), `world` = `derived.world_setup`
  (name + n agents + models), `temperature` = `derived.effective_temperature`. Built-in memories and single runs keep
  their labels. `scripts/check_steps.py` accepts the optional `memory` key.
- Built-in memories: requests and step logs identical to 3.4 apart from the schema number (56 configurations:
  every memory x action format x variant, with and without the intent probe).

## 2026-10-14 — OpenRouter: unsupported sampling parameters are left out, and logged

- `provider.drop_unsupported_params` (default true for openrouter): temperature, top_p and seed are dropped from a
  request when the model does not accept them, read once per process from OpenRouter's `/models` list. Each attempt
  record names them (`dropped_params`). Fixes tools mode for models such as Claude Haiku 5.5 (no endpoint passed
  `require_parameters`), and makes visible that such models never used the configured temperature (OpenRouter
  ignored it silently before, e.g. Haiku in the 11 Oct stopping sweep). No log schema change.

## 2026-10-14 — schema 3.4: pluggable memory, memory tools, step-by-step episodes

- `experiment.memory` is a plug-in (`src/gap_attractor/harness/memory.py`, registry `MEMORIES`): a registered name
  or a dotted path to a `Memory` class. The seven built-in memories are registered classes; their prompts and logs
  are byte-identical to 3.3 (checked on every memory × action format × variant, with and without the intent probe).
- A memory may offer its own tools in tools mode. `Caller` runs calls to them through the memory, answers them and
  asks again (at most `side_turns()` times) until the model moves; several calls per reply are all run. Follow-ups
  (retries, search results) build on the request the latest reply answered (`Caller.base`).
- A plug-in memory may log per-step fields under `step["memory"]` (new optional key: schema 3.4).
- `runner.episode(...)` is a run as a generator (one `next()` per step, every logged step yielded once);
  `run_episode` drives it to the end. `Run(..., extra=...)` / `run_extra` add fields to `run.json` (e.g. a world).
- Tests: `tests/test_memory_plugins.py`.

## 2026-10-14 — default sampling temperature 1

- `provider.temperature` is 1 in every provider file (was 0 for groq and openrouter, 0.5 for openai and mock);
  `provider.top_p` is unchanged (0.5). Runs record it in `config.provider.temperature`, so no schema change.
- The analysis groups by temperature (`group_by.temperature`), so runs at 0 and 1 are never pooled.

## 2026-10-13 — page similarity as an option: categories, embeddings or none; graphs as a plug-in

- `configs/analysis/` gains config groups `similarity` (categories | embeddings | none), `embedder` (bge_m3 |
  qwen3_embedding_8b | mock), `page_text` (lead | title) and `graph` (wikipedia_core). Pages are vectors (topic mix
  or unit embedding) and similarity is their dot product, so every measure works under every option; topic tables
  only with categories. New: `analysis/similarity.py`, `analysis/embedding.py` (OpenAI-compatible embedder via
  OpenRouter, shared SQLite cache), `build_wiki_resources.py leads` (leads.sqlite), `graph.save_graph`/`load_graph`.
- Renamed: stage `topics` -> `clicks` (old name accepted); summary columns `trace_*` -> `sim_*` (same numbers with
  categories); walks statistics `topic_retention` -> `step_similarity`, `topic_breadth` -> `breadth`, new
  `drift_from_start`; config `chi_weighting` -> `analysis.similarity.weighting`, `walks.graph` -> `analysis.graph.dir`.
- Fixed: walks keyed agent runs and bootstrap clusters by run id, which repeats across batches; now batch + run id.
  The 11 Oct sweep has no repeated run ids, so its published numbers are unaffected.

## 2026-10-13 — baselines without fitted parameters: human clickstream and whole-run walks

- `gap-attractor-analyze` gains stage `walks` (whole runs versus uniform, jump, no-revisit and clickstream walks
  from the same start pages, paired, with a Monte Carlo test) and the `clickstream` chooser (Wikimedia
  clickstream 2026-09, human link clicks). Default choosers: uniform, jump, clickstream; novelty stays available.
- `scripts/build_wiki_resources.py pages|clickstream` build `page_topics.npz` and `clickstream.npz`.
- The graph library moved into the package (`gap_attractor.analysis.graph`); `scripts/graph/core_graph.py`
  re-exports it. No harness or log change.

## 2026-10-13 — schema 3.3 on main: prompt wording and stop option as variables; analysis pipeline

- `prompt=free|browse|notask` (`configs/prompt/`) sets the opening wording and records `experiment.prompt_wording`;
  `experiment.allow_stop` turns the stop option off; `tools=detailed` gives fuller tool descriptions. Only "STOP"
  stops (no more none/exit/quit/finish). 3.1 and 3.2 are skipped: unmerged branches logged runs with them.
- `gap-attractor-analyze` runs two stages, `outcomes` (how runs end) and `topics` (topic flows versus reference
  choosers), and splits conditions by prompt wording and stop option. `scripts/graph/` holds the link-graph tools.
  See docs/analysis.md.

## 2026-10-04 — schema 3.0: merge of sachin/variants-memory-intent (schema 2.3) and xenomirant/interface

Both lines had moved on from schema 1.9 in parallel (2.1-2.3 and 2.0), so the merged harness is 3.0.
- From Sachin's line: native tool calling (`experiment.action_format: text | tools`, `configs/tools/`), memory `none`,
  reasoning capture, richer transcripts, true start pages in `batch.json` (`batch.start_page_source`), docs/reproduce.md.
- From the interface line: everything in the 2026-10-02 entry below.
- Shared names and defaults: reasoning is `model_internal_reasoning` (step) and `internal_reasoning` (intent record,
  ProviderResponse); `provider.max_tokens` is 4000 for OpenRouter and 700 elsewhere, set in each provider file.
- Honest retries also in tools mode: an invalid tool call gets its feedback as the tool result
  (`tools.retry_feedback` overrides the wording), and a harness pick is never replayed as a tool call.
- Start-page provenance covers the new sources (`runs_per_start`, `start_pages_source`).
- Text-mode and tools-mode prompts are byte-identical to both parents on the fixture scenarios.

## 2026-10-02 — link resolution, honest retries, provider errors, shared lookup, sweeps (released in 3.0)

Interface only (logs and model inputs unchanged with default settings):
- Hydra entry point (`@hydra.main`): `-m` sweeps, `--cfg`, `--help`. Hydra writes nothing for single runs;
  a sweep records `<output.directory>/_sweeps/<time>/multirun.yaml`.
- `batch.index=auto`; settings validated before anything is written; providers, Wikipedia backends and
  start-page sources are pluggable by name or dotted import path (`registry.py`); `.tsv`/`.csv` start-page
  files, `variant.runs_per_start`, `variant.start_pages_source`.

Behaviour (see AGENTS.md before pooling statistics across the 3.0 boundary):
- Link targets are normalised like MediaWiki (first letter capitalised) and XML-escaped titles in the dump
  index are matched. Links are resolved to their destination page: one row per destination (first link kept),
  links to the current page dropped, unloadable targets dropped; the model sees the destination's title.
  New per-row fields `link_title`, `via_redirect`; new extras `n_links_self`, `n_links_duplicate`,
  `n_links_unresolved`.
- Unusable replies get specific feedback, with the model's own reply shown back, and are asked again
  (`experiment.parse_retries`, `retry_feedback`). A STOP on retry is respected; an API error on retry ends the
  run as `api_error`. A harness pick is shown later as `experiment.harness_pick_note`, never as the model's reply.
- Provider: `max_tokens`, `max_backoff_s`, `extra_body` configurable (defaults reproduce 1.x requests);
  reasoning traces logged as `model_internal_reasoning` (and `intent.internal_reasoning`); Retry-After honoured;
  413 / context-length errors end the run as `context_limit` (B-environment) without retries; other 4xx are not retried.
- `extras.latency_s` = time in model HTTP requests only; `dwell_time` = whole step.
- `batch.resume`: finish a batch, re-running only cut-off and api_error runs (moved to `_superseded/`).
- Title lookup in SQLite (`wikipedia.lookup_format: sqlite`, default; built once, shared through the OS page
  cache, ~1.5 GB per process instead of ~7.3 GB); identical to the pickle on all 25,859,722 keys. Page cache bounded.
- Git provenance from the harness's own repository, readable in shared worktrees; adds `git_branch`,
  `git_dirty`, `git_error` (and `harness_git_dirty` per run); a warning is printed if git cannot be read.

## 2026-09-28 — module-based harness refactor

- The notebook-era smoke-test logic is now runnable as `gap-attractor`; the notebook remains a
  historical/exploratory artifact.
- The display-order / position-bias control from `d54cbef` is retained: source and deterministic
  per-run/per-step seeded-shuffle policies log source order, display order, shuffle seed, selected
  key, position, and normalized position.
- The synthetic fixture remains an engineering-only validation backend and now exercises normal
  Python modules rather than notebook cell execution.
- Hydra externalizes provider, model, prompt condition/text, turns, seed, sampling, display policy,
  backend/path, output, and batch settings. `pyproject.toml` provides uv-compatible dependency
  management.
- Runtime paths are configuration-relative or environment-driven, with a documented Linux dump
  configuration and no developer-specific source paths.
- Dump mode is offline. The notebook's optional creation-date HTTP lookup was intentionally not
  retained because it conflicts with the harness's offline dump requirement; its absence is logged
  as provenance rather than being silently omitted.
