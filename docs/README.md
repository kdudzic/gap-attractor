# Documentation

Read in this order if you are new to the project.

| Document | What it answers |
|---|---|
| [../README.md](../README.md) | What the project is, the main findings, how to try it |
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
