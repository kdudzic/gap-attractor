# What this folder owes to others

**LiveCodeBench** (Jain et al., 2024; github.com/LiveCodeBench/LiveCodeBench, MIT, commit 28fef95; licence in
`lcb/LICENSE`).
- `lcb/testing_util.py` is copied unchanged.
- `lcb/driver.py` follows `compute_code_generation_metrics.check_correctness`.
- `lcb/prompts.py` copies the generic system message, the question template, the self-repair feedback text and the
  code extraction rule. The function signatures are adapted, and there is one documented change for programs that
  do not compile.
- The problems are LCB's `code_generation_lite` dataset, used as published.

**Context Language Models** (Shao et al., 2026, arXiv 2609.37725). The context-file format, append-by-default, the
budget with a reserve, the nudges and the edit gate come through experiments/context_file; see its NOTICE.md. That
code is ours (MIT); nothing is copied from the CLM repository.
