# Results

Status: **preliminary**, as of 11 November 2026. A paper is in preparation. Every number below comes from the
team's analysis notes of 9 to 11 November; the full run logs behind them are not in this repository
(see [../data/README.md](../data/README.md)). The charts in the README are drawn from these numbers by
[figures/make_figures.py](figures/make_figures.py).

## How to read this page

- **Data.** About 5,000 single-agent runs collected in October 2026, 1,640 more on 10 and 11 November (personas,
  model size, top-ups, repeat rate), and 260 two-agent worlds. Models, through OpenRouter or the provider's API:
  DeepSeek V4.1 Flash, Grok 4.7, MiMo 2.6 Pro, Qwen 3.8 Flash, Qwen 3.7 Flash, Gemini 3.8 Flash, Claude Haiku 5.5,
  GPT-6 Luna, GPT-4.1 and GLM 5.3 Flash. Not every model is in every experiment.
- **Unit.** The unit of analysis is the start page. Tests are exact sign-flip or permutation tests, and intervals
  are cluster bootstraps over start pages (95% unless stated).
- **Multiple tests.** Sixteen headline tests were declared as the primary family before computing and are
  Holm-corrected together. Everything else is exploratory and carries its raw p-value.
- **Never pooled.** Runs with different decoding settings, prompt wordings, stop settings or memory settings are
  kept as separate conditions.
- **Human baseline.** "Readers" are the Wikimedia clickstream for September 2026: how often people click each
  link from the same page.
- **Status words.** *Supported*: the test passed as stated. *Supported with caveats* or *scope changed*: it passed
  after the claim was narrowed. *Weakened*: the direction holds but a number or a sub-claim failed. *Not
  supported*: it failed.
- **Checks on ourselves.** The 16 primary tests were recomputed independently and all verdicts reproduced. A
  placebo score that should show nothing was run alongside the real ones.

Definitions used below: *agreement* is the chance that two first clicks from the same page pick the same link.
*Backtracking* is a move straight back to the previous page. *Absorption* measures how strongly independent runs
end in the same places, computed on split halves of the runs so that it is not inflated by chance.

## Agents alone

Each run is separate. No agent ever saw another.

### 1. Monoculture of decisions (supported)

Different models shown the same page pick the same first link far more often than two human readers do. The
models stay distinguishable, so they are highly correlated but not interchangeable.

| Pair | Agreement |
|---|---|
| Two human readers | 0.10 [0.08, 0.14] |
| Two different models | 0.42 [0.33, 0.53] |
| The same model, run twice | 0.56 [0.47, 0.66] |

- October grid: 0.383 for different models against 0.103 for two readers, a difference of +0.280 [0.21, 0.35],
  higher on 20 of 20 start pages, Holm-corrected p = 1.4e-5. The table gives the November estimate on 10 models
  (400 one-step runs, two replicates on the same day).
- A different model agrees with a model about three quarters as often as the model agrees with itself.
- Ten models act like about 3.6 independent readers [2.6, 4.8]. They choose among about 4 effective options where
  readers choose among 14.
- Display position is not the cause: with shuffled link order the position effect is −0.004 [−0.010, 0.003]
  (equivalence within ±0.05, p = 0.007).
- Caveat: agreement without a persona was 0.10 lower in November than in October (90% CI −0.19 to −0.01).

### 2. Context steers the herd but rarely decorrelates it (weakened)

- Of 16 context levers tested, only the two step-budget wordings lower agreement between models after correction:
  budget total, no STOP: −0.21 [−0.33, −0.09], Holm 0.047; budget remaining, with STOP: −0.14, Holm 0.023.
- Reasoning effort has no detectable effect (all p > 0.45; effects above about 0.08 are excluded).
- Every lever level still agrees above the reader level of 0.10, except the "browse" wording.
- Personas do lower agreement between different models: 0.44 with none, 0.43 assistant, 0.31 bored, 0.28
  researcher, 0.25 explorer (exploratory against none: explorer −0.21, p = 0.028; researcher −0.18, p = 0.009).
  Under every persona, different models still agree more than two readers (+0.15 to +0.33, all Holm p < 0.02).

### 3. What the herd picks (supported with caveats)

- At the first step, models choose links to people at 0.38 [0.27, 0.56] of the reader rate (Holm p = 0.008). Over
  all steps the ratio is about 0.8, so this is a first-step effect.
- They choose the article's broader "parent" topic more often: 22.4% against 12.0% for readers (Holm p = 0.025).
- About 70% of first picks come from the article's first 20 links, which are 11% of the menu.
- The most linked-to page on the menu was chosen 0 times in 1,083 decisions. Models do not chase popular pages.
- A held-out model's choice is predicted best by the other models' consensus, then by the reader clickstream,
  then by position among the first 20 links. This order is the same for all 8 models tested.

### 4. No deep attractors in link browsing; a real one in free search (scope changed)

- **Link browsing.** Beyond the first hop there is no attractor. Two runs meet on a page new to both at a rate
  within 0.001 of random walks from step 5 on. Split-half absorption is 0.0011 against 0.0004 for walks. The
  page agreement that remains late in a run (0.121 at step 15) is the two-page cycle between the start page and
  the shared first pick.
- **Free search.** Starting from a blank page with a search box, absorption is 0.178 against 0.003 for random
  walks, with p at its floor in 9 of 9 cells.
- Topic-level convergence (the original H2) is weak or mixed. Settling over a run (H6) is not supported: 0 of 39
  cells pass.

### 5. Loops come from framing and memory (supported)

- With no STOP option, models bounce between two pages: 0.81 of moves go straight back. The sentence "You can
  take 15 steps through links." cuts that to 0.25, lower on 20 of 20 start pages (Holm 1.4e-5).
- Per model: DeepSeek 0.90 → 0.15, Grok 0.69 → 0.06, MiMo 0.84 → 0.38, Qwen 3.8 0.83 → 0.42. Only these four
  models have a matched arm. Gemini does not backtrack in either case.
- Once a loop starts it holds: without STOP, the chance of another backtrack after one is 0.89 to 0.99 for five
  models.
- Memory matters. With STOP allowed, memoryless agents bounce more (0.63 against 0.20, Holm 0.008). Self-written
  notes give broader exploration than no memory in all 7 models tested.
- Reading: loops are produced by the interface (no way to stop, plus a visible path), not by a model preference.

### 6. Stopping follows the interface and the persona (supported with caveats)

- Models that stop mostly say no task was given: 0.91 to 0.97 of stops for DeepSeek, Gemini, Qwen 3.8, Luna and
  MiMo. Haiku instead says the page already answers the question (0.69).
- The "browse" wording nearly removes stopping (free minus browse stop rate 0.51 [0.42, 0.59], Holm 1.9e-5).
- With actions as tool calls, 60 of 60 runs stopped, against 20 of 60 with text replies (one design, six models).
- Search access lowers it: DeepSeek stops at the first decision 0.20 of the time with search available, against
  0.67 with links only (30 new start pages, p = 0.0005).
- Personas change it. At the first step with STOP on: 0.225 assistant, 0.125 no persona, 0.013 explorer, 0.013
  researcher, 0.000 bored.
- Stop rate separates groups of models under one prompt (Kendall W 0.7 to 0.8 across experiments), but among
  similar open models the ranking is not stable (0.30), and across persona arms it is 0.045. A persona moves one
  model's stop rate by up to 0.55.
- With no stop option, agents "consider stopping" on 30 of 26,023 coded steps.

### 7. Curiosity canon on a blank page (weakened)

- On a blank page, 36 of 80 openings land on pages from Wikipedia's "Unusual articles" list.
- Three of four models carry it (MiMo 16/20, Qwen 12/20, DeepSeek 8/20, Grok 0/20), and with four models the
  cluster test cannot go below p = 0.0625 (observed 0.31).
- None of the 52 opened pages is in the readers' top 10,000 (p = 2.4e-6). They are mid-popularity pages, not
  obscure ones.

### 8. Planted history: wording, not routes (weakened)

- A planted browsing history does not detectably steer where models go: pull +0.015 [−0.004, 0.034]. Equivalence
  within ±0.03 is not shown (p = 0.11).
- Models do adopt its phrasing: +0.35 [0.27, 0.43], Holm 0.007.
- 0 of 1,710 choices went to an injected page, and 0 of 1,653 steps disowned the history.

## Agents together

Two agents, one shared working memory, the same start page. This uses the context-file memory in
[../experiments/context_file/](../experiments/context_file/).

### 9. Shared context: fewer revisits, no herding (supported in part)

- 120 two-agent worlds from 20 start pages, pre-registered, Holm over 4 tests. Arms: nothing shared against one
  shared, jointly edited context. Pairs: MiMo + DeepSeek, MiMo + MiMo, DeepSeek + DeepSeek. 15 steps.
- Supported: sharing lowers revisiting (−0.053 [−0.077, −0.026], p = 0.003) and raises the pages a pair covers
  (+1.78 [0.57, 3.00], p = 0.015).
- Not supported: staying together longer (+0.75 steps, p = 0.13); a change in the second agent's first pick
  (−0.24 [−0.44, −0.02], p = 0.13; the direction is avoidance, not copying).
- Exploratory: a DeepSeek agent that sees another DeepSeek's move takes a different link (first-step agreement
  0.55 → 0.10, p = 0.002). The two agents' wording grows more alike over a run.
- With nothing shared, these agents reproduce the agreement of finding 1 (same model 0.55, different models 0.40).
- Reading: sharing a workspace does not herd agents. It pushes their actions apart and aligns their language. The
  sameness of the population is prior correlation, not coupling.
- Limits: two open models, 15 steps, the same start page for both agents. These runs had context shortening
  switched on in the harness, which later versions disable.

## The registered hypotheses

The design sketch ([design.md](design.md)) registered ten hypotheses before data collection.

| | Hypothesis | Verdict |
|---|---|---|
| H1 | Attractors exist | Link browsing: not beyond the first hop. Free search: supported |
| H2 | Attractors are topics, not pages | Weak or mixed |
| H3 | Exploration styles exist | Supported: the persona arm is recovered from run features on unseen start pages (accuracy 0.38, chance 0.20) |
| H4 | Models prefer less familiar knowledge | Not supported (agents minus readers, popularity percentile of the first pick: +0.03 [−0.02, 0.07]) |
| H5 | Attractors are robust to ablations | Partly: the knockout result did not replicate |
| H6 | Entropy decays over a run; scale matters | Decay not supported (0 of 39 cells). No slope with model size within four families |
| H7 | Stated and revealed preferences differ | Untested: the coding scheme cannot separate them |
| H8 | No position bias | Supported under shuffled display. GPT-6 Luna shows primacy in a further check |
| H9 | Biases persist under personas | Weakened: agreement stays above readers under every persona; avoidance of person links is not shown |
| H10 | Pages created after the training cutoff attract | Dropped: the dump has no page creation dates |

## What did not survive

We kept a register of 83 claims taken from the design, the team's notes and early write-ups. As of 11 November:
44 supported, 14 mixed, 11 not supported, 8 inconclusive, 3 untestable, 1 dropped, and 2 that are not
statistical. These earlier statements are withdrawn:

- A time-capsule task framing lowering agreement from 0.38 to 0.20: that comparison mixed decoding settings.
  Matched, the change is 0.31 → 0.20 with p = 0.38, and a later top-up found no framing effect.
- "The whole herd shifts under every lever": not significant for any lever after correction.
- "Fewer person links on 17 of 20 start pages": it was 13 of 19 below the reader rate.
- Models picking from the first 20 links more than readers do: 68.7% against 58.5%, Holm 0.15.
- Models piling onto the readers' top link: post hoc, and not significant over all 20 start pages.
- Attractors in link browsing beyond the first hop, entropy decay, and "runs forget their start".
- A planted-history pull of +0.004: the estimator was unbalanced; the paired estimate is +0.015.
- "5 of 5 seeds agree" as evidence of robustness: those runs were at temperature 0 and deterministic.
- "Stop rate is a stable model trait": it holds between groups of models under one prompt only.

## Qualitative coding

An LLM judge coded the reasoning text of 4,085 runs (66,890 steps). Human labels exist for one run (105 labels;
the judge matched 103). A second judge on 300 steps agrees with the first at kappa 0.87 for user framing, 0.88
for stop intent, 0.64 for information need and 0.67 for evaluation awareness. We report three codes as descriptive
shares per model (user framing, stop intent, information need), treat evaluation awareness as an upper bound, and
make no claims about motives.

## Limits

- One environment. Link choice on Wikipedia is not task performance, and agreement here is not error correlation
  on a task.
- 20 hub start pages for the main grid. The p-values generalise to new start pages for these models, not to
  language models in general.
- Sampling temperature 1 throughout the main results; other settings are separate conditions.
- Google, Anthropic and OpenAI models are missing from the November experiments (provider access was blocked at
  the time), so personas, model size and shared context rest on four to five other models.
- Some model ids are not pinned by the provider, and behaviour drifted between October and November.
- The free-search attractor rests on one design and four models.
- The placebo score was non-null in 7.3% of 288 cells in one draw (5% expected); over 200 placebo draws the test
  is calibrated (5.15%). An isolated p near 0.05 is not evidence.

## Side study: several agents, one capped context, coding tasks

[../experiments/siamese/](../experiments/siamese/) holds a separate study in which agents share one size-limited
context file while each solves its own coding problems. Its analysis is not finished and no results are reported
here.
