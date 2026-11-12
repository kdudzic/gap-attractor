"""Check that this machine is set up to run the experiments. Changes nothing, calls no model.

    uv run --env-file .env python scripts/check_setup.py
    uv run --env-file .env python scripts/check_setup.py --provider openrouter

Every line prints OK or FAIL with what to do. Exit code 0 only if everything needed passed.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name, ok, fix=""):
    results.append(ok)
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + ("" if ok or not fix else f"\n        -> {fix}"))
    return ok


def writable(folder):
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder): pass
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openrouter", help="provider config to check the API key for")
    args = parser.parse_args()

    print("1. Code and packages")
    try:
        from gap_attractor.harness.runner import make_backend, variant_of  # noqa: F401
        from gap_attractor.search.index import SearchIndex
        import hydra, omegaconf  # noqa: F401
        check("gap_attractor, hydra, omegaconf import", True)
    except Exception as error:
        check("gap_attractor, hydra, omegaconf import", False, f"run `uv sync --all-extras` in the repo ({error})")
        sys.exit(1)
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=["wikipedia=dump", f"provider={args.provider}"])
    check("configs load (variant a_hub, b_search_first, c_search_anytime present)",
          all((ROOT / "configs" / "variant" / f"{v}.yaml").exists() for v in ("a_hub", "b_search_first", "c_search_anytime")))

    print("2. Wikipedia dump")
    dump = Path(cfg.wikipedia.dump_path)
    xml, index = dump / cfg.wikipedia.dump_xml_filename, dump / cfg.wikipedia.dump_index_filename
    check(f"dump xml    {xml}", xml.exists(), "set GAP_ATTRACTOR_DUMP / GAP_ATTRACTOR_DUMP_XML in your env file")
    check(f"dump index  {index}", index.exists(), "set GAP_ATTRACTOR_DUMP / GAP_ATTRACTOR_DUMP_INDEX in your env file")
    cache = Path(cfg.wikipedia.cache_dir)
    lookup_format = cfg.wikipedia.get("lookup_format", "sqlite")
    lookup = cache / ("title_lookup.sqlite" if lookup_format == "sqlite" else "title_lookup.pkl")
    check(f"cache folder writable  {cache}", writable(cache), "create it and make it group-writable (see docs/reproduce.md)")
    if lookup.exists():
        check(f"title lookup ({lookup_format}) already built ({lookup.stat().st_size // 2**20} MB)", True)
    else:
        print(f"  NOTE  title lookup ({lookup_format}) not built yet: the first run builds it (sqlite: ~2 minutes; "
              "other runs started meanwhile wait for it).")
    if xml.exists() and index.exists() and lookup.exists():
        backend = make_backend(cfg, ROOT)
        title = "Pumice" if backend.contains("Pumice") else backend.lookup.any_key()
        page = backend.load_page(title)
        check(f"read a page from the dump ({title}: {len(page.links) if page else 0} links)", page is not None)

    print("3. Search index (Variants B/C, intent)")
    index_path = Path(os.environ.get("GAP_ATTRACTOR_SEARCH_INDEX", ""))
    if check(f"search index  {index_path}", index_path.is_file(), "build it once: see docs/reproduce.md, Part 2"):
        hits, _ = SearchIndex.open(index_path).search("black hole", 5)
        check(f"search works ('black hole' -> {hits[0]['matched_title'] if hits else 'nothing'})", bool(hits))

    print("4. Output folder")
    output = Path(cfg.output.directory)
    check(f"results folder writable  {output.resolve()}", writable(output), "set GAP_ATTRACTOR_OUTPUT to a folder you can write to")

    print("5. API key")
    key_env = cfg.provider.get("api_key_env")
    if key_env:
        check(f"{key_env} is set (provider={args.provider}, model={cfg.provider.model})", bool(os.environ.get(key_env)),
              f"add {key_env}=... to your PRIVATE env file and run with --env-file")
    env_file_inside_shared = [p for p in ("data/.env", str(ROOT / ".env")) if Path(p).exists() and str(ROOT).startswith("data")]
    check("no .env with keys inside the shared folder", not env_file_inside_shared, f"move {env_file_inside_shared} to .env")

    print(f"\n{'READY' if all(results) else 'NOT READY'}: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()