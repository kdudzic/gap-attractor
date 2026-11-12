# Notices and credits

## Code

The code and documentation in this repository are released under the Creative Commons Attribution-NonCommercial
4.0 International licence, CC BY-NC 4.0 ([LICENSE](LICENSE)): free to use, change and share with credit, for
noncommercial purposes. For other uses, ask the authors. Exceptions, where a folder states its own terms:

- `experiments/siamese/lcb/` contains code from LiveCodeBench (MIT). See `experiments/siamese/NOTICE.md` and
  `experiments/siamese/lcb/LICENSE`.
- `experiments/context_file/` follows the protocol of *Context Language Models* (Shao et al., 2026); no code from
  that project is included. See `experiments/context_file/NOTICE.md`.

## Wikipedia

Page titles, category names, the link structure between pages, and passages of article text that appear in run
logs and model outputs are taken from English Wikipedia, dump of 1 September 2026 (`enwiki-20260901`). Wikipedia
text is available under the Creative Commons Attribution-ShareAlike 4.0 licence (CC BY-SA 4.0,
<https://creativecommons.org/licenses/by-sa/4.0/>). Credit: Wikipedia contributors; each article's history lists
its authors, at <https://en.wikipedia.org/>.

This applies to `data/start_pages/` and `data/sample_runs/`. If you reuse those files, keep this attribution and
share adaptations of the Wikipedia content under the same licence.

The synthetic wiki in `tests/fixtures/` was written for this project and contains no Wikipedia text.

## Wikimedia clickstream

The human-reader baseline uses the Wikimedia clickstream (September 2026), released under CC0. Credit: Wikimedia
Foundation. The clickstream data itself is not in this repository.

## Model outputs

`data/sample_runs/` contains outputs of DeepSeek V4.1 Flash, Gemini 3.8 Flash (Google), Grok 4.7 (xAI), MiMo 2.6
Pro (Xiaomi) and Qwen 3.8 Flash (Alibaba), obtained through OpenRouter in October 2026.
