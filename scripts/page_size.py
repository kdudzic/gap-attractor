"""Measure how large the model input would be for candidate start pages, without calling any model.

Usage (from the repo root):
    uv run --env-file .env python scripts/page_size.py                       # the suggested list below
    uv run --env-file .env python scripts/page_size.py "Pumice" "Tetrahedron"   # your own titles
    uv run --env-file .env python scripts/page_size.py --budget 8000           # token limit per request

Token counts are rough (characters / 4). "fits" means the first request stays under the budget,
leaving room for the reply. Later steps can still be large if the walk reaches a big page.
"""
import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from gap_attractor.harness.runner import make_backend, page_view
from gap_attractor.wikipedia.backend import SKIP_NS

SUGGESTED = ["Pumice", "Tumbleweed", "Sourdough", "Mancala", "Tetrahedron", "Quokka", "Origami",
             "Lighthouse", "Haiku", "Abacus", "Chess clock", "Paper clip", "Axolotl", "Bonsai"]
REPLY_RESERVE = 700  # max_tokens the provider asks for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("titles", nargs="*")
    parser.add_argument("--wikipedia", default="dump", help="config group to read pages from (dump or fixture)")
    parser.add_argument("--budget", type=int, default=8000, help="tokens allowed per request (Groq free gpt-oss-120b: 8000/min)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="config", overrides=[f"wikipedia={args.wikipedia}", "experiment=smoke"])
    backend = make_backend(cfg, Path.cwd())
    system = len(cfg.experiment.system_prompt) + len(cfg.experiment.response_rule)

    print(f"{'page':<28} {'links':>6} {'article chars':>14} {'~tokens':>8}  fits {args.budget:,}?")
    rows = []
    for title in args.titles or SUGGESTED:
        page = backend.load_page(title)
        if page is None:
            print(f"{title:<28} {'not in dump':>6}"); continue
        eligible = [l for l in page.links if backend.contains(l.candidate_key) and not l.candidate_key.startswith(SKIP_NS)]
        table = [{"title": l.destination} for l in eligible]
        observation, _ = page_view(page, table)
        article = observation.split("## Candidate links", 1)[0]
        tokens = (len(observation) + system) // 4
        fits = tokens + REPLY_RESERVE <= args.budget
        rows.append((tokens, page.title, len(eligible), len(article), fits))
    for tokens, title, n, chars, fits in sorted(rows):
        print(f"{title[:28]:<28} {n:>6} {chars:>14,} {tokens:>8,}  {'yes' if fits else 'NO'}")
    if not any(r[4] for r in rows):
        sys.exit("\nnone of these fit; try smaller pages, a paid plan, or experiment.article_max_chars")


if __name__ == "__main__":
    main()
