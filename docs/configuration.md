# Configuration reference

Every setting you can change for an experiment.

Settings go on the command line as `name=value`; you only type the ones you want to change. There are two kinds:

- **Choosing a config group** (a whole file): `variant=…`, `provider=…`, `wikipedia=…`
- **Changing one value**: `group.key=value`, e.g. `experiment.step_limit=5`

The base command is:
```bash
uv run --env-file .env python -m gap_attractor.cli <settings>
```

Where the defaults live: `configs/config.yaml`, `configs/experiment/smoke.yaml`, `configs/variant/*.yaml`,
`configs/provider/*.yaml` and `configs/wikipedia/*.yaml`.

---

## 1. Variant: what the model can do

`variant=a_hub | b_search_first | c_search_anytime`. The default is `a_hub`.

| Variant | Start | Each step |
|---|---|---|
| `a_hub` | A start page (from `batch.start_pages` or a file) | Follow one link |
| `b_search_first` | Searches once (logged as step 0), opens a result | Follow one link |
| `c_search_anytime` | A start page | Follow a link **or** search (search results replace the links for that step) |

### Variant settings

| Setting | Variants | Default | Meaning |
|---|---|---|---|
| `variant.n_runs` | B; A/C with a file | `10` | Number of episodes |
| `variant.start_pages_file` | A, C | `null` | Text file with one title per line. `n_runs` pages are sampled from it using the seed (e.g. Claire's hub list) |
| `variant.search.top_n` | B, C | `10` | How many search results are shown |
| `variant.search.candidate_pool` | B, C | `50` | Raw hits fetched before redirect de-duplication |
| `variant.search.max_queries` | B | `3` | Re-search attempts if nothing matches |
| `variant.search.display_order` | B, C | `rank` | `rank` or `seeded_shuffle` (shuffle the results) |
| `variant.search.snippet_mode` | B, C | `short_description_then_lead` | Or `lead` (always the first sentences) |
| `variant.search.snippet_sentences` | B, C | `2` | Sentences in a lead snippet |
| `variant.search.snippet_max_chars` | B, C | `300` | Snippet length cap |
| `variant.search.index_path` | B, C | `GAP_ATTRACTOR_SEARCH_INDEX` | Search index file |
| `variant.prompts.*` | B, C | in the yaml | Search, choose and action instructions. Edit them in `configs/variant/*.yaml` |

Two parts of Variant C are fixed in code, not settings:
- the order the two actions are described in alternates every step (a cyclic shift over the two options), logged
  as `extras.tool_order`;
- it uses one call, plus a second only when the model searches.

---

## 2. Memory: what the model remembers from earlier steps

`experiment.memory=`, default `path_only`.

| Value | Earlier steps shown as | Cost |
|---|---|---|
| `path_only` | Its past reasoning and choices only | Low |
| `none` | **Nothing**: memoryless; each step sees only the current page | Lowest |
| `pages_no_links` | Past articles in full, no link lists | Medium |
| `lead` | Each past page's first paragraph | Low |
| `sliding_window` | The last `memory_window` pages in full, older ones as one-line path entries | Medium |
| `self_notes` | No pages; the model adds a `"note"` to each reply and sees its path plus notes (no extra call) | Low |
| `full` | Everything it saw, every step | Very high |

`experiment.memory_window` (default `2`) is used only by `sliding_window`.

Memory is a plug-in (`src/gap_attractor/harness/memory.py`): `experiment.memory` takes one of the names above
or a dotted path to a `Memory` class of your own, with no change to the harness. A memory builds each step's
request (`messages`), may keep its own state across the run (`record` after each step), may refuse settings it
cannot work with (`validate`), and may log its own per-step fields (stored under `step["memory"]`). In tools mode
it may also offer tools of its own (`side_tools`): the harness runs those calls through the memory (`handle`; an
exception becomes an `ERROR: …` result), answers them, and asks the model again with the memory's view of the
conversation (`continue_messages`). At most `side_turns()` replies per step may call them, retries included; after
that the model is asked once, with a note and only the move tools, to act. If it still does not move, the step's
invalid type is `side-turns-exhausted` and the usual retry follows. Several calls per reply are all run; a call made
in the same reply as a move runs before the move. A retry builds on the post-edit view. Steps of such a memory log
`extras.side_calls` (turn, calls, results) and `extras.side_turns_exhausted`; each extra request is saved in
`model_inputs/` (`request_side_N`, `request_side_exhausted`). A memory may also see every reply as the model wrote it (`observe(request, reply)`: text, tool calls, reasoning trace), declare `settings()` (recorded in
run.json `memory_settings`, so the analysis keeps runs with different settings apart) and release resources in
`finish(run, reason)`. It may hold a move back (`hold_move(call)` returns a note): the move call is answered with the note, the step's
editing turns start again and the model is asked again on the same page (at most 20 times per step; logged as
`extras.held_moves`, requests saved as `request_held_N`), and it may end the run before a step (`should_end()`,
termination `context_budget`, outcome category `B-memory`).

Use the same memory setting across A, B and C when comparing variants. Every run records it in its `config.json`.

---

## 2b. Action format: how the model acts

`experiment.action_format=`, default `text`.

| Value | How the model acts |
|---|---|
| `text` | The reply format is written into the prompt, and the model answers with JSON (`{"reasoning", "choice"}`) |
| `tools` | The actions are defined as tools and sent in the API's standard `tools` field. The model calls `follow_link(title)`, `search(query)` or `stop(reason)` |

The tools are defined in **`configs/tools/default.yaml`**: their names, descriptions and parameters, the short
instruction (`tools.rule`) and the retry message. You can edit them there without touching code.
`tools=detailed` (`configs/tools/detailed.yaml`) replaces the one-line descriptions with fuller ones that say what
each tool does and what the model sees next (e.g. `stop`: "End the session. No further pages are shown after this,
and the session cannot be resumed."); everything else comes from `default`. Which tools are
offered at each call depends on the variant:

| Variant | Tools offered |
|---|---|
| A, and B after its search | `follow_link`, `stop` |
| B's first step | `search`, `stop` |
| C | `follow_link` and `search` (their order alternates every step), `stop` |

- In `tools` mode, a search result is handed back as a proper tool result.
- Every step logs `extras.action_format` and the raw `extras.tool_calls`.
- With `experiment.memory=self_notes`, the note becomes a `note` argument of each tool.
- The intent probe always uses plain JSON, since it has no actions.
- On OpenRouter, requests are routed only to providers that support tool calling.

---

## 3. Intent

`experiment.intent=`, default `none`.

| Value | Works with | What happens |
|---|---|---|
| `none` | All | Off: runs as normal |
| `probe` | All | Before each choice, a separate call shows the page **without links** and asks which article it would open next. The answer is logged in the step's `intent` record and never fed back. The choice call is unchanged |
| `walk` | **C only** | The model goes wherever its intent points (resolved by local search). No link list is shown |

| Setting | Default | Meaning |
|---|---|---|
| `experiment.intent_rule` | in `smoke.yaml` | The wording of the intent question |
| `experiment.intent_index_path` | `GAP_ATTRACTOR_SEARCH_INDEX` | Search index used to map intents to articles (needed for A too) |

The probe adds one call per step. That call has no link list, so it's smaller than the choice call.

---

## 4. Experiment settings

| Setting | Default | Meaning |
|---|---|---|
| `experiment.step_limit` | `15` | Moves per episode (B's search step is extra) |
| `experiment.seed` | `15092026` | Base seed (episode *n* uses seed + *n*). Controls the shuffle and the sampled start pages |
| `experiment.display_order_policy` | `seeded_shuffle` (set by every variant) | `source` = Wikipedia's order, `seeded_shuffle` = randomised (position-bias control) |
| `experiment.article_max_chars` | `null` (full article) | Cap the article length. Needs team agreement, since it changes what the model sees |
| `prompt=` | `free` | Opening wording of the system prompt, from `configs/prompt/`: `free` ("You are free to do what you want."), `browse` ("You have time to browse Wikipedia however you like. Follow whatever interests you."), `notask` ("There is no task and no right answer. Open links as long as you want; you may stop at any point."). Recorded as `experiment.prompt_wording` |
| `experiment.system_prompt` | set by `prompt=` | The opening instruction. Setting it directly overrides the preset (then set `experiment.prompt_wording` too, so the run is labelled) |
| `experiment.allow_stop` | `true` | `false`: no stop tool, the "To stop ..." sentences are removed from the reply rules and retry feedback, a STOP reply is asked again, and runs go to `step_limit`. Refused with a system prompt that mentions stopping (`prompt=notask`) |
| `experiment.response_rule` | in `smoke.yaml` | The reply format for link choices |
| `experiment.name` | `smoketest_agent_traversal` | The top-level results folder name |
| `experiment.condition` | set by the variant | A label in folder names; never shown to the model |
| `experiment.contributors` | `[sachin]` | Recorded in the metadata |
| `experiment.parse_retries` | `1` | After an unusable reply, how many times the model is shown its reply plus feedback and asked again |
| `experiment.retry_feedback.*` | in `smoke.yaml` | Feedback text per problem: `unparseable`, `out-of-range` (`{choice}` = what it wrote), `refusal` |
| `experiment.harness_pick_note` | in `smoke.yaml` | If every retry fails, the harness opens a random link; this note (`{title}`) is what the model is told afterwards. Default: 'No listed page was chosen, so "{title}" was opened at random.' Never name the harness or any of our tooling in it |

### Context embeddings (`context_embedding=`)

Default `none`. `context_embedding=steps` embeds the context the model was given after every step, for any memory
(`harness/context_embed.py`; needs `uv sync --extra analysis` and the embedder's API key). Writes
`<run>/context_embedding/` and run.json `context_embedding`; changes nothing the model sees.

| Setting | Default (`steps`) | Meaning |
|---|---|---|
| `context_embedding.turns` | `true` | One vector per turn (a message, or the memory's own turns via `Memory.embed_turns`) |
| `context_embedding.file` | `both` | Whole-context vector: `direct` (one text), `mean` (length-weighted mean of the turns), `both` |
| `context_embedding.max_input_chars` | `16000` | Longer texts are cut at paragraph breaks and their pieces averaged |
| `context_embedding.embedder.*` | bge-m3 via OpenRouter | Same keys as `configs/analysis/embedder/` (e.g. `context_embedding.embedder.model=qwen/qwen3-embedding-8b`) |

---

## 5. Provider: which model service

`provider=mock | groq | openrouter | openai`. The default is `mock` (a free fake model).

| Group | Model | Key in `.env` |
|---|---|---|
| `mock` | random choices (free, for testing) | none |
| `groq` | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| `openrouter` | `openai/gpt-oss-120b` | `OPENROUTER_API_KEY` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |

### Per-provider settings

| Setting | Default | Meaning |
|---|---|---|
| `provider.model` | per file | Any model id the service offers |
| `provider.temperature` | `1` (every provider, since 2026-10-14; before: 0 for groq and openrouter) | Randomness; 0 = most deterministic. Runs at different temperatures are separate conditions in the analysis |
| `provider.top_p` | `0.5` | Sampling cutoff (no effect at temperature 0) |
| `provider.drop_unsupported_params` | `true` (openrouter) | Leave temperature/top_p/seed out of a request when the model does not accept them (OpenRouter's model list); logged per attempt as `dropped_params` |
| `provider.timeout` | `120` | Seconds per request |
| `provider.max_retries` | `4` | Retries on network errors, timeouts, 408/425/429/5xx. Never on 413/context-length (run ends as `context_limit`) or other 4xx |
| `provider.max_tokens` | `4000` (openrouter), `700` (others) | Reply length cap per call. Reasoning models count their reasoning against it: raise it for them |
| `provider.max_backoff_s` | `60` | Longest wait between retries; the server's `Retry-After` is honoured up to this |
| `provider.extra_body` | `{}` | Extra request fields, e.g. `+provider.extra_body.reasoning.effort=low` on OpenRouter (note the `+`: the default is empty) |
| `provider.base_url`, `provider.api_key_env` | per file | For other OpenAI-compatible services |

- `gemini` is also supported in the code, but there's no `gemini.yaml` file yet.
- Reasoning traces are logged when the API returns them (`model_internal_reasoning`, `intent.internal_reasoning`);
  nothing is requested specially, so whether they appear depends on the provider and model.

---

## 6. Wikipedia source

`wikipedia=fixture | dump`. The default is `fixture`, a tiny fake Wikipedia used by the tests. **Use `dump` for real
runs.**

| Setting | Default (dump) | Meaning |
|---|---|---|
| `wikipedia.dump_path` | `GAP_ATTRACTOR_DUMP` | Dump folder |
| `wikipedia.dump_xml_filename`, `wikipedia.dump_index_filename` | `GAP_ATTRACTOR_DUMP_XML` / `_INDEX` | The exact file names |
| `wikipedia.cache_dir` | `GAP_ATTRACTOR_CACHE` | Shared title lookup |
| `wikipedia.max_compressed_block_bytes` | `32000000` | Safety limit when reading the dump |
| `wikipedia.lookup_format` | `sqlite` | Title lookup: `sqlite` (built once in the cache folder, ~2 min; shared between processes) or `pickle` (~7 GB RAM per process) |
| `wikipedia.page_cache_size` | `4096` | Parsed pages kept in memory per process |

---

## 7. Start pages, batches and output

| Setting | Default | Meaning |
|---|---|---|
| `batch.start_pages` | `[Civic Atlas]` | Start titles for A/C: `"batch.start_pages=['Pumice','Keziz']"`. Capitalise the first letter |
| `batch.index` | `1` | **Must be new for each run on the same day**, or `auto` (section 11) |
| `batch.resume` | `false` | `true` (today's batch with this `batch.index`) or a batch folder: finish it. Finished runs stay; cut-off and `api_error` runs move to `_superseded/` and run again with the same seed. Refuses if the config differs |
| `output.directory` | `GAP_ATTRACTOR_OUTPUT` | Results folder |
| `output.save_model_inputs` | `true` | Saves the `model_inputs/` copies of what the model saw; `false` saves disk space |

For a title with an apostrophe, use escaped double quotes: `"batch.start_pages=[\"Schrödinger's cat\"]"`, or put it
in a `start_pages_file`.

---

## 8. Environment variables in `.env`

| Variable | Used for |
|---|---|
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY` | Provider keys (private: never in `data`) |
| `GAP_ATTRACTOR_DUMP`, `GAP_ATTRACTOR_DUMP_XML`, `GAP_ATTRACTOR_DUMP_INDEX` | Dump location and file names |
| `GAP_ATTRACTOR_CACHE` | Title lookup cache |
| `GAP_ATTRACTOR_SEARCH_INDEX` | Search index (B, C, intent) |
| `GAP_ATTRACTOR_OUTPUT` | Results folder |

---

## 9. Example combinations

```bash
P="uv run --env-file .env python -m gap_attractor.cli provider=groq wikipedia=dump"

# A, two start pages, 5 steps
$P variant=a_hub "batch.start_pages=['Pumice','Keziz']" experiment.step_limit=5 batch.index=31

# A from a hub list, 10 sampled pages
$P variant=a_hub variant.start_pages_file=data/start_pages/hubs.txt variant.n_runs=10 batch.index=32

# B, 3 episodes, shuffled search results
$P variant=b_search_first variant.n_runs=3 variant.search.display_order=seeded_shuffle batch.index=33

# C with the intent probe and lead-paragraph memory
$P variant=c_search_anytime "batch.start_pages=['Keziz']" experiment.intent=probe experiment.memory=lead batch.index=34

# tool calling instead of JSON-in-prompt, memoryless
$P variant=a_hub "batch.start_pages=['Keziz']" experiment.action_format=tools experiment.memory=none batch.index=37

# intent-only walk (C)
$P variant=c_search_anytime "batch.start_pages=['Keziz']" experiment.intent=walk batch.index=35

# original Wikipedia link order (no shuffle), for position-bias comparison
$P variant=a_hub "batch.start_pages=['Keziz']" experiment.display_order_policy=source batch.index=36

# the 11 Oct 2026 stopping sweep's four cells (team analysis notes, not in this repository)
$P variant=a_hub prompt=free                           # c0: current wording + stop
$P variant=a_hub prompt=free experiment.allow_stop=false # c1: no stop option
$P variant=a_hub prompt=browse                         # c2: "browse however you like" + stop
$P variant=a_hub prompt=notask                         # c3: "no task ... you may stop"

# the same with tool calling and the fuller tool descriptions
$P variant=a_hub prompt=browse experiment.action_format=tools tools=detailed

# free test of any of the above: drop provider=groq (the fake "mock" model is used)
```

---

## 10. Checking what a run used

Every run saves its complete, resolved settings in its `config.json`:
```bash
cat runs/<experiment>/batch_31_*/*/config.json | head -60
```
`model_inputs/step_XXX_request.txt` in the same run folder shows exactly what the model was sent at each step.

---

## 11. Sweeps, start-page sources and plug-ins

None of these change what the model is sent. Their settings are recorded in every run's `config.json`.

### Sweeps (`-m`) and automatic batch numbers

```bash
# 2 variants x 3 seeds = 6 batches, one after another, numbered automatically
$P -m variant=a_hub,c_search_anytime experiment.seed=1,2,3 batch.index=auto
# print the composed config without running anything
$P --cfg job variant=b_search_first
```
- `batch.index=auto` takes 1 + the highest `batch_NN_` already in `<output.directory>/<experiment.name>/`. If two
  people start at the same moment, the second one moves on to the next number instead of failing. Logs record the
  actual number. Use it for every sweep: a fixed index makes the second job of a sweep fail with `FileExistsError`.
- A sweep records its overrides once in `<output.directory>/_sweeps/<time>/multirun.yaml`. Single runs write
  nothing extra: no `.hydra/`, no log file.
- Jobs in a sweep run one after another. Parallel jobs need a Hydra launcher plug-in (not installed yet).

### Start pages

| Setting | Default | Meaning |
|---|---|---|
| `variant.start_pages_file` | `null` | `.txt`: one title per line (as before). `.tsv` / `.csv`: a header row, titles in `start_pages_column`, e.g. `data/start_pages/start_set_A_final.tsv` |
| `variant.start_pages_column` | `title` | Column to read from a `.tsv` / `.csv` file |
| `variant.runs_per_start` | `1` | Repeat each start page this many times in a row. Every episode still gets its own run seed |
| `variant.start_pages_source` | automatic | `batch`, `file`, `search` (Variant B), or a dotted path to your own `factory(cfg) -> list of titles` |

These keys are in `configs/variant/a_hub.yaml` and `c_search_anytime.yaml`.

```bash
# 5 sampled start pages from Claire's set A, 6 episodes each
$P variant=a_hub variant.start_pages_file=data/start_pages/start_set_A_final.tsv variant.n_runs=5 variant.runs_per_start=6 batch.index=auto
```

### Plug-ins: providers, page graphs and start-page sources

Each of these takes a registered name or a dotted import path to a factory on your `PYTHONPATH`, so new
components need no change to the harness (see `src/gap_attractor/registry.py`):

| Setting | Built-in names | Factory signature |
|---|---|---|
| `provider.name` | `mock`, `openai`, `groq`, `gemini`, `openrouter` | `factory(provider_cfg)` returning an object with `chat(messages, *, seed, temperature, top_p) -> ProviderResponse` |
| `wikipedia.backend` | `fixture`, `dump` | `factory(wikipedia_cfg, project_root)` returning a `WikipediaBackend` (see `wikipedia/backend.py`) |
| `variant.start_pages_source` | `batch`, `file`, `search` | `factory(cfg)` returning a list of start titles |
| `experiment.memory` | `path_only`, `none`, `pages_no_links`, `lead`, `sliding_window`, `self_notes`, `full` | a `Memory` subclass (`harness/memory.py`); one instance per run |

```bash
PYTHONPATH=~/my_plugins $P provider.name=my_providers.make_vllm wikipedia.backend=my_graphs.SyntheticGraph
```

**Driving runs step by step.** `runner.episode(...)` is one run as a generator: the first `next()` sets up, each
further `next()` takes one step and yields its record, and the summary is the `StopIteration` value. A
multi-agent scheduler interleaves one generator per agent; `run_extra={"world": {...}}` adds fields to `run.json`
(the analysis clusters runs by `world.world_id`), and `memory=` passes a memory object, which several agents may
share through a common object of your own.

### Checked before anything is written

Unknown values for the variant, memory, intent, `action_format`, `display_order_policy`, `variant.search.display_order`,
`variant.search.snippet_mode`, provider, backend and start-page source now stop the run before any folder is
created. Before, a typo was only caught mid-run, and a mistyped `variant.search.display_order` or `snippet_mode`
silently fell back to `rank` / `lead`.

---

See also: `docs/variants.md` (behaviour and logging details), docs/reproduce.md (setup on the shared server) and
docs/reproduce.md (ready-made command lines).
