# Documentation

Read in this order if you are new to the project.

| Document | What it answers |
|---|---|
| [../README.md](../README.md) | What the project is, the main findings, how to try it |
| [Navigators](https://observational.github.io/navigators/) | Interactive site: the runs on a map of Wikipedia, next to human readers |
| [results.md](results.md) | Every finding with its test, the registered hypotheses, what did not survive, limits |
| [design.md](design.md) | The original experiment sketch written before data collection: definitions, questions, hypotheses H1 to H10. Kept as written, so parts of it describe plans that changed |
| [architecture.md](architecture.md) | How the harness is put together, in one page |
| [variants.md](variants.md) | The navigation variants (links only, search first, search anytime), memory settings, the intent probe |
| [configuration.md](configuration.md) | Reference for every setting, with example commands |
| [reproduce.md](reproduce.md) | Running a real model on the real Wikipedia dump, and the commands behind the main experiments |
| [analysis.md](analysis.md) | The analysis pipeline `gap-attractor-analyze`: stages, inputs, output tables |
| [methods.md](methods.md) | The definitions and mathematics behind the analysis, and why each comparison is fair |
| [technical-changelog.md](technical-changelog.md) | Log schema versions and what changed in each. Needed when reading run logs from different dates |
| [../data/README.md](../data/README.md) | The sample runs and the start-page pools |
| [../experiments/context_file/README.md](../experiments/context_file/README.md) | Agents with a self-edited context file, alone or sharing one |
| [../experiments/siamese/README.md](../experiments/siamese/README.md) | Side study: agents sharing one capped context on coding tasks |

## Reading the code

The package is `src/gap_attractor/`. A run flows through it in this order:

1. `cli.py` reads the Hydra config (`configs/`) and calls `harness/runner.py: run_experiment`.
2. `wikipedia/backend.py` loads a page from the dump (or the synthetic fixture) and extracts its links.
3. `harness/runner.py` builds the choice set, shows the page to the model, parses the reply, and logs the step.
4. `providers/base.py` makes the model call, with retries.
5. `harness/memory.py` decides what the model sees of earlier steps. New memories plug in here.

`harness/runner.py` is one large module (about 2,400 lines). Its parts, in file order:

| Part | Main names |
|---|---|
| Link menu and link masks | `build_choice_set`, `link_mask_spec`, `apply_search_result_mask` |
| What the model sees: page text, reply rules, tools | `page_view`, `action_rule`, `tool_specs`, `build_messages` |
| Built-in memories | `BuiltinMemory`, `CollectionPathMemory` |
| Intent probe and step-budget sentence | `run_intent_call`, `reveal_budget`, `steps_sentence` |
| Parsing replies | `parse_reply`, `parse_query`, `parse_action` |
| Pre-run predictions and post-run readouts | `run_pre_run_predictions`, `run_post_run_readouts` |
| Output files | `Dataset` (a batch), `Run` (one run: `steps.jsonl`, `transcript.md`, ...) |
| Setup and validation | `make_backend`, `start_pages`, `validate_config`, `run_experiment` |
| The model call with retries and fallbacks | `Caller`, `choose`, `retry_or_pick` |
| The run loop | `episode` (a generator, one step per `next`), `run_episode` |

The analysis (`analysis/`) is separate from collection: `logs.py` reads run folders, `models.py` and `walks.py`
hold the reference choosers and random walks, and each stage has its own module (`outcomes.py`, `topics.py`,
`walks.py`, `hypotheses.py`, `convergence.py`, `loops.py`). `analysis/cli.py` is the entry point.

The two folders under `experiments/` are self-contained studies built on the harness. Each has one `run.py`, one
`sweep.py`, a library package (`ctxfile/`, `siam/`), and analysis scripts for single questions at the top level.

## Vocabulary

- **Run**: one start page plus up to N link steps by one agent, in a fresh context. It ends on the step limit, a
  STOP choice, a dead end, or a technical failure.
- **Step**: one page view plus one choice. The unit of logging.
- **Batch**: all runs launched together with one configuration. Only the start page and the per-run seed differ.
- **Condition**: the combination of settings a batch was run with (model, prompt wording, stop option, memory,
  and so on). Conditions are never pooled in the analysis.
- **Attractor**: a page or topic that independent runs reach more often than random walks from the same start
  pages do. A loop between two pages is not an attractor by itself.

## A note on the internal references

These documents were written during the project. Some still mention team members by first name (for example,
whose start-page pool or topic list a setting uses) and analysis notes that are not part of this repository.
