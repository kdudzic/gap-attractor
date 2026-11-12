# Analysis pipeline

How run logs become results. Read this before analysing runs or changing `src/gap_attractor/analysis/`. The
mathematics (definitions, why each comparison is fair, what to expect) is in [methods.md](methods.md).

```
 ONCE PER DUMP                              PER EXPERIMENT
 ─────────────                              ──────────────
 enwiki dump + Wikimedia clickstream        gap-attractor (harness)
   │                                          │
   ├─ scripts/build_wiki_resources.py         ▼
   │    topics (χ, π), every page's χ,      run folders: run.json + steps.jsonl per run
   │    page leads, human clicks ───────┐     │
   │                                    │     ▼
   └─ scripts/graph/*                   │   gap-attractor-analyze   (similarity: categories | embeddings | none)
        link graph, strongly connected  │     ├─ outcomes: how runs end                  (run folders only)
        core, statistics ───────────────┼───► ├─ clicks:   each click vs choosers on the same menu
                                        └───► └─ walks:    whole runs vs random walks from the same start
                                                │
                                                ▼
                                              <output>/ outcomes.csv, summary.csv, walks.csv, ...
```

| Part | Code | Answers |
|---|---|---|
| Resources | `scripts/build_wiki_resources.py`, `analysis/topics.py`, `analysis/clickstream.py` | Which topics each page belongs to; each page's lead text; how often human readers click each link |
| Similarity | `analysis/similarity.py`, `analysis/embedding.py` | How alike two pages are: topic overlap, embedding cosine, or nothing |
| Graph | `analysis/graph.py`, `scripts/graph/` ([README](../scripts/graph/README.md)) | The link graph itself: distances, reachability, hubs, communities |
| Stage `outcomes` | `analysis/outcomes.py` | Per condition: how often runs stop, where, and how clean the replies were |
| Stage `clicks` | `analysis/logs.py`, `models.py`, `kernel.py`, `inference.py`, `report.py` | Per click: does the agent choose differently from a random walk or a human reader on the same menu? |
| Stage `walks` | `analysis/walks.py` | Per run: does the agent's whole path differ from random walks from the same start page (similarity, loops, hubs, distance)? |

The click stage was called `topics` before the similarity options existed; `stages=[topics]` still works.

## Baselines, not fitted models

Every comparison is against a baseline with **no fitted parameters**: a random walk, a random walk with jumps, a
walk that never revisits, or human readers (the Wikimedia clickstream). We do not fit choice models. A condition
has about 20 runs × 15 steps; that cannot pin down the hundreds of cells of a topic-by-topic matrix, let alone a
model of the whole graph. Baselines need no fitting, and the walk side can be simulated as often as we like, so
the only scarce sample is the agent's runs. The full topic matrices are reported for looking at, not for testing;
tests are on a few scalars per condition.

## Quick start

```bash
# once per dump
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py extract    # ~15 min, 12 workers
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py rollup
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py pages      # ~2 min
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py clickstream \
    --clickstream data/cache/clickstream/clickstream-enwiki-2026-09.tsv.gz  # ~5 min
uv run --env-file .env --extra analysis python scripts/build_wiki_resources.py leads      # ~10 min, for embeddings
# the link graph: scripts/graph/README.md (built 2026-10-08 in data/graph)

# analyse runs: any mix of run, batch or experiment folders
uv run --env-file .env --extra analysis gap-attractor-analyze data.paths=[runs/for_analysis]

# all three stages take ~5 min and ~3.5 GB for 240 runs (the walks stage loads the graph)
# the same with embeddings instead of category topics (needs OPENROUTER_API_KEY; first run embeds, later runs reuse)
uv run --env-file .env --extra analysis gap-attractor-analyze data.paths=[...] analysis/similarity=embeddings
uv run --env-file .env --extra analysis gap-attractor-analyze data.paths=[...] analysis/similarity=embeddings \
    analysis/embedder=qwen3_embedding_8b          # the cross-check embedder
# links only (e.g. a graph with neither categories nor page text)
uv run --extra analysis gap-attractor-analyze data.paths=[...] analysis/similarity=none

# only how runs end (no resources needed), with a paired test of prompt wordings on the same start pages
uv run --extra analysis gap-attractor-analyze data.paths=[...] stages=[outcomes] \
    outcomes.paired_field=prompt outcomes.paired_baseline=free
```

Settings are in `configs/analysis.yaml` (Hydra; override any key on the command line). Results go to
`$GAP_ATTRACTOR_OUTPUT/analysis/<timestamp>/`, or `output=<folder>`:

| File | Stage | Contents |
|---|---|---|
| `config.yaml` | all | The full analysis config, for reproducing the run |
| `outcomes.csv` | outcomes | One row per condition (columns below) |
| `outcomes_runs.csv` | outcomes | One row per run, to recompute anything |
| `outcomes_paired.csv` | outcomes | Paired stop comparison, if `outcomes.paired_field` is set |
| `stop_statements.md` | outcomes | Every stop reason, by condition |
| `similarity.json` | clicks, walks | Which similarity option was used; for embeddings the embedder, page text, cache file, pages embedded now or reused, and the cost |
| `summary.csv` | clicks | One row per condition × reference chooser: similarity of consecutive pages (`sim_*`), bits per step, with intervals |
| `<condition>/results.json` | clicks | Everything for one condition: counts, audit, per-model summaries, early/late |
| `<condition>/flows_<model>.npz` | clicks | similarity=categories only: the topic-by-topic matrices (F, A, D, L) and their intervals; heatmaps with `plots=true` |
| `walks.csv` | walks | One row per condition × baseline × statistic: agent, walks, paired difference, interval, p-value |
| `walks_runs.csv` | walks | One row per run × baseline × statistic: the agent's value and the mean of its walks |
| `walks_audit.json` | walks | Runs used, runs with no move (stopped at once), starts outside the graph core |

## Conditions: what is never pooled

Runs are grouped into conditions by `group_by` in `configs/analysis.yaml`. Each entry maps a label to a dotted
path in `run.json`:

| Label | Source | Notes |
|---|---|---|
| `variant` | `variant` | a_hub, b_search_first, c_search_anytime, d_search_first_anytime |
| `llm` | `config.provider.model` | |
| `memory` | derived | `config.experiment.memory`, plus the settings a plug-in memory declared (run.json `memory_settings`), e.g. `context_file\|budget=32768,interface=tool` |
| `intent` | `config.experiment.intent` | |
| `world` | derived | `config.world.name` (`none` for single agents); for a run in a multi-agent world also the number of agents and the set of models, e.g. `joint_labelled\|n=3\|a+bx2` |
| `prompt` | derived | `free` / `browse` / `notask` (`prompt=` in the harness). Runs before schema 3.3 did not record it, so their system prompt text is matched against `configs/prompt/`; any other text becomes `custom-<hash>` |
| `stop` | derived | `true` unless the run had `experiment.allow_stop=false` |
| `temperature` | derived | The temperature the model actually got: run.json `effective_sampling` (schema 3.5), or `dropped_params` in the first step of 14 Oct 2026 runs. `provider-default` when the provider left it out (e.g. Claude Haiku 5.5 on OpenRouter). Older runs show the configured value even where the model ignored it. The default became 1 on 2026-10-14; earlier OpenRouter and Groq runs used 0 |
| `schema_major` | always added | 2.x and 3.x menus differ (see AGENTS.md), so they are never pooled |

Prompt wording and the stop option are separate conditions because they change behaviour a lot. In the
11 Oct 2026 sweep, the wording alone moved stop rates from about 0.1 to about 0.9 for the same model and start
pages. To add a field, add a line to `group_by`; to compute one, add a function to `DERIVED` in `logs.py`.

**Clusters.** Intervals resample clusters, not steps, because steps within a run are not independent. A cluster
is a run, or a whole world when several agents shared one, since they read each other's board and marks.

## Stage `outcomes`: how runs end

Recomputed from `run.json` (`outcome.termination_reason`) and `steps.jsonl`. Columns of `outcomes.csv`:

| Column | Meaning |
|---|---|
| `runs`, `stopped`, `stop_share` | Runs, runs the model ended itself (`model_stopped`), and their share |
| `stop_lo`, `stop_hi` | 95% Wilson interval for `stop_share` |
| `stopped_first_page` | Stopped before opening any page |
| `step_limit`, `context_budget`, `no_valid_move`, `other_endings` | Reached `step_limit`; ended by a memory's budget (`B-memory`); three steps in a row with no listed page chosen, every reply readable (prose, refusals, unlisted pages); ended any other way (dead end, API or parse error) |
| `stop_hazard` | Stops per step taken. Endings other than STOP and the step limit compete with stopping (a run ended at step 3 could not stop at step 10), so compare this, not `stop_share`, when such endings differ between conditions |
| `median_moves`, `median_moves_when_stopped` | Pages opened per run |
| `invalid_replies`, `harness_picks` | Unusable replies; of those, steps where every retry failed and the harness picked a random link |
| `stop_replies` | Replies asking to STOP, including those refused when the stop option is off |
| `tokens_in`, `tokens_out` | Summed token counts, where the provider reported them |

`outcomes.paired_field=prompt outcomes.paired_baseline=free` compares every other wording with `free` on the same
start pages, with all other condition fields held fixed. `only_baseline` and `only_other` count start pages that
stopped under one wording only (with repeated runs per start page: where the share of runs that stopped is higher;
they are averaged, never overwritten); `p_mcnemar` is the exact two-sided McNemar test on those counts (a sign test
with repeats). When a run of a start page ended another way (dead end, error, `context_budget`, `no_valid_move`) at step T, a stop
counts for that page only if it came by step T, in both arms (`censored_pages`).

## Page similarity: categories, embeddings or none

The clicks and walks stages need to know how alike two pages are. That is a choice, `analysis/similarity=`, made
of exchangeable parts, each a Hydra config group in `configs/analysis/`:

| Option | A page is | Similarity of two pages | Needs | Gives |
|---|---|---|---|---|
| `categories` (default) | its topic mix χ (the hypergraph below) | topic overlap Σ_e χ(x,e) χ(y,e) | topic resources | everything, including topic tables |
| `embeddings` | a unit-length text embedding | cosine | page text + an embedder | everything except topic tables |
| `none` | nothing | — | nothing | link measures only: bits per click, revisits, hubs, distance |

Every option gives each page a vector, and similarity is always the dot product. So every measure built on it
means the same thing under every option, and switching is one setting.

**Embeddings** are made of three parts:

- `analysis/embedder=`: the model. `bge_m3` (default; BAAI, 1024 dims), `qwen3_embedding_8b` (4096 dims; the
  cross-check), `mock` (offline, for tests). Both real ones run through OpenRouter at about $0.01 per million
  tokens; any OpenAI-compatible `/embeddings` endpoint works (change `base_url`, `api_key_env`). The default is not
  from an agent family we test, since an embedder from the same family as an agent might share its sense of what
  is similar. Results should not hinge on one embedder: rerun with the other.
- `analysis/page_text=`: what is embedded. `lead` (title + lead section, the default) or `title`, both from
  `leads.sqlite` (`build_wiki_resources.py leads`).
- The cache: `<cache_dir>/<model>__<text>__<hash>.sqlite`, shared by everyone. A page is embedded once per embedder
  and text; changing either starts a new cache file, so vectors never mix. Only pages the analysis meets are
  embedded (agent paths, menus, simulated walks): a few hundred thousand for a sweep of 240 runs, well under $1.
  `analysis.similarity.max_new_pages` refuses runs that would embed more. `similarity.json` reports what was
  embedded, reused and paid.

**Adding an option** is a new class or config file, not a change to the stages:

- a new similarity: a `PageVectors` subclass in `analysis/similarity.py` with `@SIMILARITIES.register("name")`,
  or any factory named by dotted path in `analysis.similarity.kind`;
- a new embedder or text source: a class in `analysis/embedding.py` registered in `EMBEDDERS` or `PAGE_TEXTS`
  (or a dotted path in `kind`), plus a YAML file in `configs/analysis/embedder/` or `page_text/`;
- a new embedding model on an existing endpoint: only a YAML file (see `qwen3_embedding_8b.yaml`, which inherits
  everything from `bge_m3.yaml` but the model).

## Another graph

The harness takes any graph whose pages can be loaded (`wikipedia.backend=<dotted.path>`, see
`src/gap_attractor/registry.py`); categories are optional. In the analysis:

- `outcomes` and `clicks` read only the run folders. Without categories, use `analysis/similarity=embeddings` (if
  pages have text) or `none`.
- `walks` needs the graph itself: `analysis/graph=` (default `wikipedia_core`). Write any directed graph with
  `gap_attractor.analysis.graph.save_graph(folder, src_page_ids, dst_page_ids, hub_page_ids)` and set
  `analysis.graph.dir=<folder>`; with no hub list, use `analysis.graph.hubs=top_in_degree`. Give a strongly
  connected graph: a walk that reaches a page without out-links stays there. A loader of your own goes in
  `analysis.graph.kind` as a dotted path.
- `clickstream` exists only for Wikipedia: drop it from `models` and `walks.baselines` elsewhere.

## Stage `clicks`: each click versus reference choosers

The question, per condition: **when the agent picks a link, does it move differently from a simple reference
chooser looking at the same page and the same menu?** With any similarity option it compares how alike consecutive
pages are; with `categories` it also gives topic-by-topic tables.

### Topics: Wikipedia as a hypergraph (similarity=categories)

Pages are nodes and topics are hyperedges: a page belongs to several topics at once.

- Topics are root categories (`configs/analysis/topics/roots.yaml`, Claire Xu's list of 29). Each category takes
  the roots that reach it first when walking down the category graph (up to 7 levels; ties kept). Category
  redirects are followed (`Arts` becomes `The arts`); hidden and maintenance categories are dropped.
- Each page gets a membership vector χ(page) over the topics plus an `unlabeled` bucket, summing to 1. With
  `analysis.similarity.weighting=multiplicity` (default), each of the page's categories splits one vote over its topics. With
  `uniform`, every topic the page reaches gets equal weight. Unlabeled pages are kept, and their share is reported.
- π is the share of each topic over all articles (the mean of χ).
- Changing the roots changes every number: rebuild the resources and compare `topic_hash` in `results.json`.

### Which steps count

Only choices the agent made itself:

- dropped: harness random picks (`harness_handling: skip`), infrastructure failures (`terminate`), STOP and
  unresolved choices, and search-first variants B/D's opening search (there is no current page yet);
- kept: steps answered on a retry (`harness_handling: retry`), because the model chose the page itself;
- **link steps** choose from the displayed menu. Menus are de-duplicated by page_id (schema 2.x lists a page once
  per redirect). A chosen page outside its menu is an error (`data.on_choice_outside_menu`);
- **jump steps** are search results and intent jumps: their destination is not on the page's menu.

`results.json` → `audit` counts every exclusion. Check it before reading any number.

### Reference choosers

Each sees exactly the page and menu the agent saw, and none has a fitted parameter. Defaults: `uniform`, `jump`,
`clickstream`.

| Chooser | Link step | Jump step |
|---|---|---|
| `uniform` | Every displayed link equally likely (a random walk) | not covered |
| `jump` | As `uniform` | Lands on a random article: expected topic = π. Identical to `uniform` for variant A, which has no jumps |
| `clickstream` | Each link in proportion to how often human readers clicked it from this page (see below) | not covered |
| `novelty` (not a default) | Pick a topic present on the menu uniformly, then a link in it. Links in topics rare *on this menu* get more weight | not covered |

`novelty` needs topic mixes and refuses other similarity options. With embeddings, π (where a jump lands on
average) is the mean embedding of `prevalence_sample` random articles. The jump rate is reported
(`counts.jump_rate`), not modelled. To add a chooser, write a function `(steps, chi, pi, ctx=None) -> Prediction`
in `analysis/models.py` with `@register("name")`, or name any function by dotted path in `models:`; `chi` holds the
page vectors of whichever similarity option is in use.

### Human readers: the clickstream

The Wikimedia clickstream counts, for one month, how many readers went from article A to article B by clicking a
link. We use the 2026-09 file, the month of our dump: 21.2 million link pairs, 99.5% of them matched to our page
ids, covering 2.1 million source pages. Two limits come from the source data:

- Pairs with **10 or fewer clicks** in the month are removed. A link that is not listed therefore had 0-10 clicks;
  it counts as `clickstream_unlisted` clicks (default 5, the middle). Rerun with 1 and 10 to check that a result
  does not depend on it. A page with no listed link at all gets a uniform choice.
- It records pairs, not reading sessions, and readers browse for every reason. It is the average reader's next
  click from each page, not a free-browsing person. Readers also see links in page order, while variant A shows
  the agent a shuffled list; for the closest human comparison, also run the agent with
  `experiment.display_order_policy=source`.

`results.json` → `counts.clickstream` reports how many of the agent's link steps were on a page with listed clicks
(`page_has_listed_link`) and how many chosen links were listed (`chosen_link_listed`).

### Similarity of consecutive pages (every option)

For a step x → y, with u, v the vectors of x and y and w = Σ_j p(j) vec(j) the chooser's expected next vector:

| Quantity | Meaning |
|---|---|
| `sim_agent`, `sim_model` | Mean u·v and mean u·w: how alike consecutive pages are for the agent and for the chooser |
| `sim_excess` | `sim_agent − sim_model`. Positive: the agent moves to pages more like the current one than the chooser would |
| `bits_per_step_vs_uniform` | Link steps: mean log2 p_model(chosen link) − log2 p_uniform(chosen link). Positive: the chooser predicts the agent's choices better than a random walk. Needs no vectors |
| `by_time` | `sim_*` in the early and late half of each run, and in buckets of `time_bucket` steps |

With categories, `sim_*` are exactly the trace shares of the topic tables below (the share of flow that stays in
its topic); with embeddings they are mean cosines. Uncertainty: 95% intervals from a bootstrap over clusters
(`n_boot`), and an exact sign test over clusters for `sim_excess` and `bits_per_step_vs_uniform`.

### Topic tables (similarity=categories)

A step x → y connects two pages, but topics are sets. Each step is spread over topic pairs by an outer product:

- u = χ(x), v = χ(y), and the agent's step adds `u vᵀ` to F_agent. Example: a page half History, half Military
  moving to one half Military, half Politics adds 0.25 to each of History→Military, History→Politics,
  Military→Military and Military→Politics.
- For a model, w = Σ_j p(j) χ(j) over the menu is the expected χ of the next page, and the step adds `u wᵀ` to
  F_model. The same u is used, so agent and model are compared step by step.

Both matrices have row sums m = Σ u, so they compare cell by cell:

| Quantity | Meaning |
|---|---|
| `A = F / m` | Row e: the next page's topic mix, given that a randomly drawn topic of the current page is e |
| `D = (F_agent - F_model) / steps` | Where the agent sends more (positive) or less flow than the model |
| `L = log((F_agent + 0.5) / (F_model + 0.5))` | The same, as a log-ratio |
| retention | Diagonal of A: the chance of staying in the same topic |
| trace share | trace(F) / steps = `sim_agent` / `sim_model` above |

Read only the agent-minus-model contrasts. Pages are topic mixtures, so retention never reaches 1: a step between
two pages with identical χ adds only Σ_e u_e v_e to the diagonal. Absolute retention mostly measures how much the
topics overlap.

Topic rows with mass below `min_row_mass` are flagged: shown, not interpreted.

## Stage `walks`: whole runs versus random walks

The clicks stage looks at one click at a time on the pages the agent visited. The walks stage asks where whole
runs go. For every agent run with at least one move, `walks.per_run` walks of each baseline start on the same page
and make the same number of moves on the link graph (`analysis/graph`, by default the enwiki strongly connected
core from `scripts/graph/`).

| Baseline | Each move |
|---|---|
| `uniform` | A random out-link |
| `jump` | Like uniform, but jumps to a random page at the agent's own jump rate in that condition (search results and intent jumps per move). Left out when the agent never jumped |
| `no_revisit` | A random out-link not yet visited in this walk (any out-link if all were visited) |
| `clickstream` | An out-link in proportion to human clicks, as in the clicks stage |

| Statistic | Per path p_0 … p_m | Needs |
|---|---|---|
| `step_similarity` | Mean over moves of vec(p_t−1)·vec(p_t): topic overlap or cosine (the per-step version of `sim_agent`) | vectors |
| `breadth` | How widely the visited pages spread: entropy (bits) of the mean topic mix for categories, 1 − ‖mean unit vector‖ for embeddings | vectors |
| `drift_from_start` | 1 − vec(p_0)·vec(p_m): how far in meaning the run ends from its start page | vectors |
| `revisit_rate` | Share of moves that land on a page already on the path | links |
| `hub_share` | Share of moves that land on a hub (`analysis.graph.hubs`: Claire's hub pool, pages with 340+ in-links, or the most linked-to pages) | links |
| `end_distance` | Link distance from the start page to the last page, following link direction (`walks.distance=true`) | links |

(Before the similarity options, `step_similarity` and `breadth` were called `topic_retention` and `topic_breadth`.)

**The comparison is paired**: for run r, d_r = statistic(agent) − mean over r's walks, and `diff` is the mean of
d_r. `diff_lo` and `diff_hi` are a 95% bootstrap interval over clusters. `p_walks` is a Monte Carlo test against
the walks themselves: in each replicate one walk of every run stands in for the agent, which gives the
distribution of `diff` when the agent is just another walk. It makes no assumption about the shape of the
statistic, which matters for skewed counts such as revisits (a sign test there is wrong: it tests the median).

The agent's path is every page it opened, harness picks included. Runs that stopped before opening a page have no
path and are counted in `walks_audit.json`. The graph comes from the 2026-10-08 build: a few agent links are not
graph edges, and agent pages outside the core count as not-a-hub and have no distance.

## Stage `hypotheses`: where the chosen link ranks on fixed scores

`analysis/hypotheses.py`. This is the non-parametric counterpart of HypTrails (Singer et al. 2015): no
concentration parameter, and no fitted choice model.

**Hypotheses.** A hypothesis is a fixed score for every link on the menu the agent was shown, declared in advance
(`hypotheses.tests`):

| Name | Score |
|---|---|
| `similarity` | topic overlap (categories) or cosine (embeddings) of the link's page with the current page |
| `hub` | log(1 + in-degree) on the link graph |
| `human` | share of human clicks from the current page (clickstream) |
| `novelty` | 1 if the run has not been on the page yet |
| `backtrack` | 1 if the link leads back to the previous page |
| `self_interest` | cosine to `hypotheses.self_text` (embeddings only) |
| `position` | placebo: minus the shown position (links are shown in a seeded shuffle) |
| `random` | placebo: a fixed pseudo-random number |

**Statistic.** Per link step:
- r = the chosen link's mid-rank percentile on the menu. Its expectation is exactly 0.5 under a uniform choice.
- E_B r = its expectation under a reference chooser B (`hypotheses.references`: `uniform`, `clickstream`,
  `indegree`). B carries a control: "s beyond B".
- Steps whose links all score the same are left out and counted in `hypotheses_audit.json`.

**Per condition, hypothesis and reference:**
- R, the mean of r, and lift = mean(r − E_B r);
- a 95% cluster-bootstrap interval (runs, or worlds);
- `p_cluster`: a sign-flip test over clusters. This is the p-value to read: it allows any dependence within a
  cluster. With `hypotheses.cluster_by: start_page` (the default), runs of one condition from the same start page
  form one cluster.
- `p_mc`: shown for reference only. It treats every click as independent, so it is too small when clicks repeat (in
  the 15 Oct check its random-score placebo came out p < 0.05 in 14% of rows);
- `p_holm_primary`: Holm-adjusted within `hypotheses.primary`, uniform reference, from `p_cluster`;
- `role`: primary, exploratory, placebo (`random`) or check (`position`: a primacy bias is real behaviour).

**Bits per step.** `hypotheses_bits.csv`: each hypothesis as a fixed chooser P = (1 − λ) b / Σb + λ/n, with
b = score − menu minimum, at the declared λs, against uniform.

**Reading it.**
- A placebo that comes out non-null means the inference is off; look for duplicated runs, or a harness change.
- If a hypothesis lifts against uniform but not against `clickstream`, the agent ranks as human readers do.
- If it lifts against uniform but not against `indegree`, its effect is popularity.

## Stage `convergence`: do runs from different starts end up in the same place?

`analysis/convergence.py`. The attractor question measured on the observed trails, hop by hop, without a
transition model.

**Walks.** For every run, `convergence.walks_per_run` walks of each baseline (`uniform`, `no_revisit`, `indegree`,
`clickstream`, `jump`) start on its first page and make as many moves, up to `convergence.max_hop`.

**`convergence.csv`.** At hop t, over runs from **different** start pages that reached t (`runs`, `starts`):
- `similarity`: mean topic overlap or cosine of their pages;
- `same_page`: the share of pairs on the same page.

The null replaces every run by one of its walks, once per replicate. We report a one-sided p and the walks'
2.5–97.5% band.

**`absorption.csv`: split-half absorption.**
- Clusters are split at random. The attractor set is the `top_k` pages visited by the most runs of one half (start
  pages left out). In the other half we measure the share of moves in the set, and per hop the share of runs that
  have entered it (`entered_by_hop` rows in convergence.csv).
- Both halves take a turn, over `n_splits` splits, so the set is never tested on the runs that defined it.
- `split_by: start_page` (the default) keeps runs from the same start page in the same half. At low temperature
  they repeat each other, and splitting them would let one half predict its twin.
- The `indegree` walk is the control for "it is just hubs".
- The p-values' floor is 1 / (1 + `walks_per_run`).

## Stage `loops`: two-page loops and escapes

`analysis/loops.py`. Measured from the observed paths, with no fitted model.

**Bounces.** A move is a bounce when it returns to the page of two moves before (A → B → A); consecutive bounces
form a loop. We report:
- `bounce_share` (bounces among moves from the second on), with a cluster-bootstrap interval;
- `median_first_bounce`.

**Escapes.** `loops_hazard.csv` gives the escape chance after c consecutive bounces. Escaping means the next page is
neither of the two; only observed next moves count. `decline` = hazard(1) − hazard(≥3), with an interval.

**Reading it.**
- `decline` > 0 alone does **not** show in-context learning. Page pairs differ in stickiness, so the loops that
  last are the sticky ones, and the pooled hazard falls even if each pair's hazard is constant. Memoryless agents
  show it too (15 Oct runs).
- In-context learning shows as more bouncing, and a lower `hazard_1`, when the agent sees its path than when it does
  not, on the same start pages.
- Clean comparison: the same start pages with `experiment.memory=path_only` (sees its path) vs `none` (memoryless),
  via `loops.paired_field=memory loops.paired_baseline=path_only`. That gives `loops_paired.csv`: the per-start-page
  difference in bounce_share, a bootstrap interval, and an exact sign test.
- At temperature 0 a decline is confounded with determinism (the same state gives the same choice), so run it at
  temperature 1.

## The link graph is directed

A link A → B does not imply B → A. On the full core only 20% of links are reciprocated, and the diameter following
links is 51 against 9 when direction is ignored (`scripts/graph/README.md`). Consequences:

1. **Menus are out-links, so models are scored on the agent's own menus.** Out-link topic mixes are lopsided:
   for example, a hundred thousand pages may link to one country page, which can link back to only a few of them. A comparison with topic co-membership,
   or with an undirected neighbourhood, would count that structure as agent preference.
2. **F, D and A are not symmetric. Read them by row.** D[History→Military] and D[Military→History] are separate
   claims, and nothing makes a topic's inflow equal its outflow.
3. **A is not a Markov chain over topics, so do not take powers of it.** A_model is computed on the pages the
   agent visited, so it inherits the agent's path, and the out-links depend on the page within a topic.
   Whole-run questions go to the walks stage, which simulates on the page graph.
4. **Revisits and loops need a directed null model.** With few reciprocated links, returning to a page usually
   takes a different route: on the sweep's start pages, random walks revisit on about 1% of moves and
   human-click walks on about 5%. The walks stage compares revisits with
   walks on the directed graph.
5. **Some measures ignore direction on purpose:** modularity (computed on the undirected graph), and jumps, which
   leave the link structure entirely.

## What this does not say

- The clicks stage compares choices **on the pages the agent visited**. It says nothing about where runs end up
  over many steps.
- With ~10-20 runs per condition, intervals are wide. Read effect sizes and intervals, not single p-values.
- A stop rate measures the prompt wording and the model family as much as the model's interest (11 Oct 2026
  sweep). Compare stop rates only within one `prompt` and `stop` setting.
- A difference from a baseline says the agent is not that baseline. It does not say why. Read transcripts
  (`transcript.md`) before naming a cause.
- An embedder is a measuring instrument with its own idea of "similar". Check a result with a second embedder
  before relying on it.
- Possible extensions, all without fitted parameters: more walk adjustments (popularity-weighted, no immediate
  backtrack), human paths from other sources, and the same walks on the pool core.

## Code map

```
src/gap_attractor/analysis/
  cli.py        gap-attractor-analyze: runs the stages in `stages`
  logs.py       run folders -> run table and step table (find_runs, run_row, load_steps); condition labels
  outcomes.py   stage outcomes
  similarity.py page vectors: categories | embeddings | none (PageVectors, SIMILARITIES registry)
  embedding.py  embedders (EMBEDDERS), page text (PAGE_TEXTS), the shared embedding cache
  topics.py     category -> topic rollup, χ, π (TopicMap); every page's χ (PageTopics)
  clickstream.py human clicks per link (Clickstream)
  models.py     reference choosers (uniform, jump, clickstream, novelty) and the registry
  kernel.py     page matrix, similarity of consecutive pages, topic tables F, A, D, L, bits per step, by_time
  walks.py      stage walks: paths, random-walk baselines (incl. indegree), statistics, Monte Carlo test
  hypotheses.py stage hypotheses: scores (SCORES registry), mid-rank lift vs reference choosers, bits, Holm
  convergence.py stage convergence: pairwise convergence by hop and split-half absorption against walks
  loops.py      stage loops: A -> B -> A bounces, escape hazard by bounce count, paired comparison
  graph.py      graphs for the walks: CoreGraph (CSR, BFS), load_graph, save_graph
  inference.py  cluster bootstrap, sign test
  report.py     results.json, flows_*.npz, summary.csv, heatmaps
scripts/build_wiki_resources.py   topic resources, page_topics.npz, clickstream.npz, leads.sqlite
scripts/graph/                    enwiki link graph build, strongly connected core, statistics
configs/analysis.yaml             all analysis settings and the default choice of each group below
configs/analysis/                 similarity/ embedder/ page_text/ graph/ topics/: one YAML file per option
tests/test_analysis.py, test_outcomes.py, test_baselines.py, test_similarity.py, test_hypotheses.py, test_loops.py; tests/analysis_synth.py
```
