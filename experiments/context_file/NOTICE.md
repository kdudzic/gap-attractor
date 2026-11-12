# What this folder owes to Context Language Models

This experiment follows **Context Language Models** (Shao et al., 2026, arXiv 2609.37725; code:
`facebookresearch/context-language-models`, licensed CC BY-NC 4.0). This folder is part of a repository under the PolyForm Noncommercial License 1.0.0.

**Taken from the paper and repository (ideas and conventions):**
- the file format: one block per turn, opened by a header line (CLM: `[[CTX_TURN i role=…]]`; ours: `[[TURN i role=…]]`);
- append-by-default: new turns are added and nothing is removed unless the model removes it;
- a token budget with a reserve, showing the model the enforced limit (budget − reserve), nudges at 25 / 50 / 75 %
  plus an urgent nudge, and the edit gate (an edit may not grow the context past the limit);
- the size readout in the context (`[context: ~N/limit tokens]`) and the gist of the nudge texts, adapted from CLM
  (CC BY-NC 4.0) in our own words;
- demanding a compaction before the agent goes on when the context is over the limit (CLM rolls back and retries;
  we hold the move);
- editing the file with bash in a container, with a time limit per command (the repo's default is 120 s);
- echoing every editing call and its result into the context.

The repository's own numbers:
- budget: `context_budget_tokens: 28672` in the bcp configuration (a 32 768-token window minus max_tokens) and
  32 000 in edgebench and the CLI default, with a 2 048-token reserve;
- over the limit: roll back and retry, `max_num_retry_on_limit` 50 by default (6 in bcp); with 0, one final turn,
  then the end;
- command timeout: 120 s;
- sampling: temperature 0.7, top_p 0.95.

Ours differ (README, "Differences from the paper"): a budget of 16 384 tokens with the same reserve, held moves
instead of rollback, and the project's sampling.

**History of the wording (15 Oct 2026).**
- **14 Oct review.** Until commit `cd1c16c`, this folder said that no CLM code was copied. The 14 Oct review found
  that three parts were near-verbatim from the CLM repository:
  - the header regular expression (`context_utils/context_string.py`);
  - the parse-back rules (the same file);
  - the regex example in the bash instructions (`clm_agent/prompts.yaml`).
- **The fix.** All three were replaced. The parser (`ctxfile/context.py`) reads headers line by line with its own
  rules, and the instructions are our own.
- **15 Oct review.** It then found that a few instruction sentences and the gate's rejection message were still
  close paraphrases of CLM's `prompts.yaml` and `edit_gate.py`. They have been reworded (`configs/ctx_prompt/`,
  `ctxfile/memory.py`). The overlap left is the format convention and the protocol listed above, credited here.

No file in this folder copies CLM source code. If you find wording that still follows CLM closely, either credit it
here as "adapted from CLM (CC BY-NC 4.0)" or reword it.
