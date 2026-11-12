"""Collection runner. It logs observations; it does not compute scientific metrics.

Variants (select with `variant=...`, see configs/variant/):
  a_hub             start from a page list; each step the model follows one link.
  b_search_first    the model searches once, opens a result, then follows links.
  c_search_anytime  each step the model either follows a link or searches; search results replace the links.
  d_search_first_anytime  the model searches for its start, then may follow links or search at every step.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import replace

from gap_attractor.harness import failures as failure_log
from datetime import datetime, timezone
from pathlib import Path

from gap_attractor.analysis.topics import load_topic_map
from gap_attractor.harness.memory import MEMORIES, Memory, memory_class
from gap_attractor.providers import make_provider
from gap_attractor.registry import BACKENDS, PROVIDERS, START_PAGE_SOURCES
from gap_attractor.wikipedia import DumpBackend, FixtureBackend, wiki_key
from gap_attractor.wikipedia.backend import SKIP_NS, clean_article_text

SCHEMA_VERSION = "3.19"
OUTCOME_CATEGORY = {"step_limit": "A-normal", "model_stopped": "A-normal", "dead_end": "B-environment", "context_limit": "B-environment", "api_error": "C-technical", "parse_error": "C-technical",
                    "context_budget": "B-memory",   # context_budget: a memory ended the run (Memory.should_end), e.g. over its budget
                    "collection_complete": "A-normal",   # direct collection task reached its preregistered target
                    "reached_target": "A-normal", "world_stopped": "A-normal",   # goal worlds: arrived; another agent's arrival ended the world
                    "no_valid_move": "A-normal"}   # 3 steps in a row with no listed page chosen, every reply readable (see unreadable)
VARIANTS = ("a_hub", "b_search_first", "c_search_anytime", "d_search_first_anytime")
SEARCH_FIRST_VARIANTS = ("b_search_first", "d_search_first_anytime")
SEARCH_ANYTIME_VARIANTS = ("c_search_anytime", "d_search_first_anytime")
RETRY_TEXT = "Your reply could not be parsed. Reply with JSON only, choosing one title exactly as written in the list."
# Feedback after an unusable reply, by what went wrong (experiment.retry_feedback overrides these). {choice} = what it wrote.
RETRY_FEEDBACK = {
    "unparseable": "Your reply could not be read: it was not the JSON asked for. Reply with JSON only, choosing one title exactly as written in the list.",
    "out-of-range": '"{choice}" is not one of the listed titles. Reply with JSON only, choosing one title exactly as written in the list.',
    "refusal": "Your reply did not choose a title. Reply with JSON only, choosing one title exactly as written in the list, or STOP to stop.",
}
HARNESS_PICK_NOTE = 'No listed page was chosen, so "{title}" was opened at random.'   # never names our apparatus (schema 3.9)
RUN_LOG_COLUMNS = [
    "Config ID", "Batch ID", "Run ID", "Owner", "Experiment", "Condition", "Run Index", "Start Page",
    "Model", "Provider", "Started At (UTC)", "Finished At (UTC)", "Status", "Termination Reason", "Steps",
    "Input Tokens", "Output Tokens", "Reasoning Tokens", "Cost (USD)", "Artifact Path", "Git Commit",
    "Synced At (UTC)", "Notes",
]
READOUT_LOG_COLUMNS = [
    "Config ID", "Batch ID", "Run ID", "Owner", "Experiment", "Readout ID", "Condition", "Run Index",
    "Start Page", "Model", "Provider", "Order", "Candidate Count", "Selection Count", "Valid", "Error",
    "Input Tokens", "Output Tokens", "Reasoning Tokens", "Cost (USD)", "Candidate Shuffle Seed",
    "Paired Sampling Seed", "Selections", "Selection Principle", "Artifact Path", "Git Commit",
]



def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def slug(value, n=40): return re.sub(r"[^\w\-]+", "-", str(value or "none")).strip("-")[:n] or "none"
HARNESS_REPO = Path(__file__).resolve().parents[3]


@functools.lru_cache(maxsize=None)
def git_provenance(repo=HARNESS_REPO):
    """Commit, branch and dirty flag of the repository the harness code was loaded from (not the cwd).
    safe.directory=* lets a shared worktree owned by another user be read; this only reads. If git cannot
    answer, the reason is logged and printed once instead of silently recording null."""
    def git(*args):
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(repo), *args], text=True, capture_output=True, timeout=10)
    try:
        head = git("rev-parse", "HEAD")
        if head.returncode:
            raise RuntimeError(head.stderr.strip() or "git rev-parse failed")
        status = git("status", "--porcelain", "--untracked-files=no")
        branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() or None
        return {"git_commit": head.stdout.strip(), "git_branch": branch,
                "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None, "git_error": None}
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        print(f"warning: harness git provenance unavailable ({message}); runs will log git_commit=null", file=sys.stderr)
        return {"git_commit": None, "git_branch": None, "git_dirty": None, "git_error": message}


def git_commit(root=HARNESS_REPO):
    return git_provenance(Path(root).resolve() if root != HARNESS_REPO else HARNESS_REPO)["git_commit"]


def variant_of(cfg):
    name = cfg.variant.name if cfg.get("variant") else "a_hub"
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")
    return name


def display_keys_for_step(eligible_keys_source_order, policy, run_seed, step_index):
    """Order only: membership/cardinality are invariant under every supported policy."""
    displayed = list(eligible_keys_source_order)
    if policy == "source": return displayed, None
    if policy == "seeded_shuffle":
        shuffle_seed = (int(run_seed) * 1_000_003 + int(step_index)) % (2 ** 63 - 1)
        random.Random(shuffle_seed).shuffle(displayed)
        return displayed, shuffle_seed
    raise ValueError(f"unknown display_order_policy: {policy!r}")


STATIC_LINK_MASK_MODES = ("static_titles", "visited_static_titles", "visited_sham_static_count")
SEMANTIC_LINK_MASK_MODES = ("semantic_visited", "sham_semantic_count")
SEARCH_AWARE_LINK_MASK_MODES = ("visited_all", "sham_visited_all_count")
LINK_MASK_MODES = ("none", "visited", "sham_visited_count", *STATIC_LINK_MASK_MODES,
                   *SEMANTIC_LINK_MASK_MODES, *SEARCH_AWARE_LINK_MASK_MODES)
DEFAULT_TOPIC_RESOURCES = "data/cache/resources"


def link_mask_spec(cfg):
    """Validated, serialisable link-mask configuration with a stable audit hash."""
    mode = str(cfg.experiment.get("link_mask_mode", "none"))
    if mode not in LINK_MASK_MODES:
        raise ValueError(f"unknown link_mask_mode: {mode!r}; expected one of {LINK_MASK_MODES}")
    raw_titles = list(cfg.experiment.get("link_mask_titles") or [])
    titles = [str(title).strip() for title in raw_titles]
    if any(not title for title in titles):
        raise ValueError("experiment.link_mask_titles must contain non-empty page titles")
    keys = [wiki_key(title) for title in titles]
    if len(set(keys)) != len(keys):
        raise ValueError("experiment.link_mask_titles contains duplicate canonical page titles")
    mask_id = cfg.experiment.get("link_mask_id")
    mask_id = None if mask_id in (None, "", "null") else str(mask_id)
    semantic_threshold, semantic_resources, semantic_topic_hash = None, None, None
    if mode not in STATIC_LINK_MASK_MODES and titles:
        raise ValueError(f"experiment.link_mask_titles is only valid for {STATIC_LINK_MASK_MODES}")
    if mode in STATIC_LINK_MASK_MODES and not titles:
        raise ValueError(f"link_mask_mode={mode} requires experiment.link_mask_titles")
    if mode != "none" and not mask_id:
        raise ValueError(f"link_mask_mode={mode} requires experiment.link_mask_id")
    if mode in SEMANTIC_LINK_MASK_MODES:
        semantic_threshold = float(cfg.experiment.get("link_mask_semantic_threshold", 0.5))
        if not 0 < semantic_threshold <= 1:
            raise ValueError("experiment.link_mask_semantic_threshold must be in (0, 1]")
        semantic_resources = Path(str(cfg.experiment.get("link_mask_semantic_resources") or
                                           os.environ.get("GAP_ATTRACTOR_RESOURCES") or
                                           DEFAULT_TOPIC_RESOURCES)).expanduser().resolve()
        topics_json = semantic_resources / "topics.json"
        category_topics = semantic_resources / "category_topics.tsv"
        if not topics_json.is_file() or not category_topics.is_file():
            raise ValueError(f"semantic link mask resources are incomplete under {semantic_resources}")
        semantic_topic_hash = json.loads(topics_json.read_text(encoding="utf-8")).get("topic_hash")
        if not semantic_topic_hash:
            raise ValueError(f"semantic link mask topics.json has no topic_hash: {topics_json}")
        semantic_resources = str(semantic_resources)
    # Preserve the historical hash for pre-semantic modes so old run artifacts remain verifiable.
    canonical = {"mode": mode, "mask_id": mask_id, "configured_titles": titles, "configured_keys": keys}
    if mode in SEMANTIC_LINK_MASK_MODES:
        canonical.update(semantic_threshold=semantic_threshold, semantic_resources=semantic_resources,
                         semantic_topic_hash=semantic_topic_hash)
    canonical["sha256"] = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return canonical


@functools.lru_cache(maxsize=4)
def _semantic_topic_map(resources):
    return load_topic_map(resources)


def _semantic_match(rows, visited_categories, topic_map, threshold):
    """Max coarse-topic dot product to any visited page, excluding the unlabelled bucket."""
    histories = [topic_map.chi(list(categories))[:-1] for categories in (visited_categories or ())]
    output = []
    for row in rows:
        candidate = topic_map.chi(list(row.get("semantic_cluster_label") or []))[:-1]
        similarities = [float(candidate @ previous) for previous in histories]
        maximum = max(similarities, default=0.0)
        output.append((maximum, maximum >= threshold))
    return output


def _apply_link_mask(rows, spec, visited_keys, run_seed, step_index, visited_categories=(), topic_map=None,
                     renumber_original=True):
    """Filter resolved candidates and return an auditable per-step mask record.

    `visited` removes links back to earlier pages. `sham_visited_count` keeps those links but removes the same
    number of unvisited candidates with a deterministic seed, controlling for menu-size reduction.
    `visited_static_titles` removes revisits plus a preregistered set of candidate destinations.
    `visited_sham_static_count` removes revisits in both arms, preserves the candidate destinations, and removes
    the same number of other destinations as the target arm would remove incrementally. The semantic pair does
    the analogous operation for candidates whose coarse-topic overlap with any visited page reaches the
    preregistered threshold. The unlabelled topic never creates a semantic match.
    """
    mode, visited = spec["mode"], set(visited_keys or ())
    effective_mode = {"visited_all": "visited", "sham_visited_all_count": "sham_visited_count"}.get(mode, mode)
    visited_indexes = [i for i, row in enumerate(rows) if row["_destination_key"] in visited]
    configured = set(spec["configured_keys"])
    static_indexes = [i for i, row in enumerate(rows) if row["_destination_key"] in configured]
    static_incremental_indexes = [i for i in static_indexes if i not in visited_indexes]
    semantic_scores = [(0.0, False)] * len(rows)
    if effective_mode in SEMANTIC_LINK_MASK_MODES:
        if topic_map is None:
            raise ValueError(f"link_mask_mode={mode} requires a loaded semantic topic map")
        if topic_map.hash() != spec["semantic_topic_hash"]:
            raise ValueError("semantic link mask topic map hash differs from the recorded specification")
        semantic_scores = _semantic_match(rows, visited_categories, topic_map, spec["semantic_threshold"])
    semantic_indexes = [i for i, (_, matched) in enumerate(semantic_scores) if matched]
    semantic_target_indexes = sorted(set(visited_indexes) | set(semantic_indexes))
    mask_seed, sham_shortfall = None, 0
    if effective_mode == "none":
        masked_indexes = set()
    elif effective_mode == "static_titles":
        masked_indexes = set(static_indexes)
    elif effective_mode == "visited":
        masked_indexes = set(visited_indexes)
    elif effective_mode == "visited_static_titles":
        masked_indexes = set(visited_indexes) | set(static_indexes)
    elif effective_mode == "visited_sham_static_count":
        protected = set(visited_indexes) | set(static_indexes)
        candidates = [i for i in range(len(rows)) if i not in protected]
        target_count = len(static_incremental_indexes)
        mask_seed = (int(run_seed) * 1_000_033 + int(step_index) * 97 + 67) % (2 ** 63 - 1)
        sample_count = min(target_count, len(candidates))
        masked_indexes = set(visited_indexes) | set(random.Random(mask_seed).sample(candidates, sample_count))
        sham_shortfall = target_count - sample_count
    elif effective_mode == "semantic_visited":
        masked_indexes = set(semantic_target_indexes)
    else:
        target_indexes = visited_indexes if effective_mode == "sham_visited_count" else semantic_target_indexes
        protected = set(target_indexes)
        seed_offset = 41 if effective_mode == "sham_visited_count" else 53
        candidates = [i for i, row in enumerate(rows) if row["_destination_key"] not in visited]
        candidates = [i for i in candidates if i not in protected]
        target_count = len(target_indexes)
        mask_seed = (int(run_seed) * 1_000_033 + int(step_index) * 97 + seed_offset) % (2 ** 63 - 1)
        sample_count = min(target_count, len(candidates))
        masked_indexes = set(random.Random(mask_seed).sample(candidates, sample_count))
        sham_shortfall = target_count - sample_count

    masked, kept = [], []
    for eligible_position, row in enumerate(rows):
        clean = {key: value for key, value in row.items() if key != "_destination_key"}
        if eligible_position in masked_indexes:
            audit = {
                "eligible_position": eligible_position,
                "title": clean["title"],
                "link_title": clean.get("link_title", clean.get("matched_title")),
                "via_redirect": clean["via_redirect"],
                "page_id": clean["page_id"],
                "was_visited": row["_destination_key"] in visited,
            }
            if effective_mode in SEMANTIC_LINK_MASK_MODES:
                audit.update(
                    semantic_similarity_max=round(semantic_scores[eligible_position][0], 12),
                    semantic_match=semantic_scores[eligible_position][1],
                    is_semantic_target=eligible_position in semantic_target_indexes,
                )
            if effective_mode in STATIC_LINK_MASK_MODES:
                audit.update(
                    is_static_target=eligible_position in static_indexes,
                    is_incremental_static_target=eligible_position in static_incremental_indexes,
                )
            masked.append(audit)
        else:
            if renumber_original:
                clean["original_position"] = len(kept)
            kept.append(clean)
    record = {
        "mode": mode,
        "mask_id": spec["mask_id"],
        "mask_spec_sha256": spec["sha256"],
        "match_rule": "resolved destination key",
        "visited_candidate_count": len(visited_indexes),
        "static_target_candidate_count": len(static_indexes) if effective_mode in STATIC_LINK_MASK_MODES else None,
        "static_incremental_target_count": (len(static_incremental_indexes)
                                            if effective_mode in STATIC_LINK_MASK_MODES else None),
        "semantic_match_rule": ("exact revisit OR max coarse-topic dot product to any visited page; "
                                "unlabelled excluded") if effective_mode in SEMANTIC_LINK_MASK_MODES else None,
        "semantic_threshold": spec.get("semantic_threshold"),
        "semantic_topic_hash": spec.get("semantic_topic_hash"),
        "semantic_target_candidate_count": len(semantic_target_indexes) if effective_mode in SEMANTIC_LINK_MASK_MODES else None,
        "mask_seed": mask_seed,
        "masked_count": len(masked),
        "sham_shortfall": sham_shortfall,
        "masked_candidates": masked,
    }
    return kept, record


def apply_search_result_mask(results, spec, visited_keys, run_seed, step_index):
    """Apply explicit search-aware modes to an already ranked search result list.

    Original positions remain the pre-mask search ranks; shown positions are compacted after filtering.
    Other modes intentionally retain their historical direct-link-only semantics.
    """
    if spec["mode"] not in SEARCH_AWARE_LINK_MASK_MODES:
        return results, None
    rows = [{**row, "_destination_key": wiki_key(row["title"])} for row in results]
    kept, record = _apply_link_mask(rows, spec, visited_keys, run_seed, 10_000 + step_index,
                                    renumber_original=False)
    for shown_position, row in enumerate(kept):
        row["shuffled_position"] = shown_position
    return kept, record


def build_choice_set(page, backend, policy, run_seed, step_index, mask_spec=None, visited_keys=(),
                     visited_categories=(), topic_map=None):
    """The links offered on `page`: one row per distinct destination article, in page order, then ordered for display.

    Each link is resolved to the page it lands on (redirects followed). Kept: the first link to each destination.
    Dropped and counted: links back to the current page (also via a redirect), further links to a destination
    already offered, and links whose target cannot be loaded. The model sees the destination's own title;
    `link_title` keeps the link as written on the page.
    When `mask_spec` is supplied, masking is applied after redirect resolution/deduplication and before display
    ordering. The model sees only the retained rows; removed rows and the deterministic mask seed are logged.
    Returns (choice_set, shuffle_seed, counts for step extras)."""
    rows, seen = [], set()
    counts = {"n_links_self": 0, "n_links_duplicate": 0, "n_links_unresolved": 0}
    for link in page.links:
        if link.candidate_key.startswith(SKIP_NS):
            continue
        destination = backend.load_page(link.candidate_key) if backend.contains(link.candidate_key) else None
        if destination is None:
            counts["n_links_unresolved"] += 1
        elif destination.key == page.key:
            counts["n_links_self"] += 1
        elif destination.key in seen:
            counts["n_links_duplicate"] += 1
        else:
            seen.add(destination.key)
            rows.append({"original_position": len(rows), "shuffled_position": None, "title": destination.title,
                         "link_title": link.destination, "via_redirect": destination.key != link.candidate_key,
                         "hyperlink": f"https://en.wikipedia.org/wiki/{destination.key}", "page_id": destination.page_id,
                         "semantic_cluster_label": list(destination.categories), "_destination_key": destination.key})
    if mask_spec is None:
        rows = [{key: value for key, value in row.items() if key != "_destination_key"} for row in rows]
    else:
        eligible_before_mask = len(rows)
        rows, mask_record = _apply_link_mask(rows, mask_spec, visited_keys, run_seed, step_index,
                                             visited_categories, topic_map)
        counts.update(n_links_eligible_before_mask=eligible_before_mask,
                      n_links_masked=mask_record["masked_count"], link_mask=mask_record)
    order, shuffle_seed = display_keys_for_step(list(range(len(rows))), policy, run_seed, step_index)
    choice_set = [rows[i] for i in order]
    for shuffled_position, row in enumerate(choice_set):
        row["shuffled_position"] = shuffled_position
    return choice_set, shuffle_seed, counts


# ----------------------------------------------------------------------------- what the model sees
def page_view(page, candidate_table, article_max_chars=None):
    """The observation: full cleaned prose, then the candidate titles only (no positions, no URLs)."""
    article_text, article_was_truncated = clean_article_text(page.wikitext, article_max_chars)
    titles = [row["title"] for row in candidate_table]
    observation = (f"# {page.title}\n\n## Article content\n{article_text}\n\n"
                   f"## Candidate links ({len(titles)})\n"
                   + "\n".join(f"- {title}" for title in titles))
    return observation, article_was_truncated


def results_view(query, results):
    lines = [f'Search results for "{query}" ({len(results)}):']
    lines += [f"- {r['title']} — {r['snippet']}" if r["snippet"] else f"- {r['title']}" for r in results]
    return "\n".join(lines)


_STOP_SENTENCE = re.compile(r"\s*To stop\b[^.]*\.")
_OR_STOP = re.compile(r",? or STOP to stop")  # "... one title exactly as written in the list, or STOP to stop."


def stop_allowed(cfg):
    """experiment.allow_stop (default true): whether the model is offered a way to end the run itself."""
    return True if cfg is None else bool(cfg.experiment.get("allow_stop", True))


def stop_text(cfg, text):
    """A prompt text with its "To stop ..." sentence removed when experiment.allow_stop is false."""
    return text if stop_allowed(cfg) else _STOP_SENTENCE.sub("", str(text)).rstrip()


def action_rule(cfg, order):
    """Search-anytime reply rule, with the two actions described in cyclic order."""
    parts = {"follow": cfg.variant.prompts.follow_option, "search": cfg.variant.prompts.search_option}
    rule = cfg.variant.prompts.action_intro + "\n" + "\nor\n".join(parts[a] for a in order)
    return rule + ("\n" + cfg.variant.prompts.stop_option if stop_allowed(cfg) else "")


MEMORY_MODES = ("path_only", "none", "pages_no_links", "lead", "sliding_window", "self_notes", "full")  # built in
NOTE_RULE = 'Also add a "note" field to your JSON: 1-2 sentences you want to remember from this page.'


def memory_mode(cfg):
    """experiment.memory: a registered memory (harness/memory.py; the built-in ones are below) or a dotted path."""
    mode = str(cfg.experiment.get("memory", "path_only"))
    memory_class(mode)
    return mode


def make_memory(cfg):
    """One memory object for one run (one agent)."""
    return memory_class(memory_mode(cfg))(cfg)


def memory_wants_note(cfg):
    return bool(getattr(memory_class(memory_mode(cfg)), "wants_note", False))


ACTION_FORMATS = ("text", "tools")
PRE_RUN_PREDICTION_MODES = ("none", "self", "generic", "self_and_generic", "self_and_external")
PRE_RUN_PREDICTION_TASKS = ("pages_topics", "memory_effect", "coarse_single_run")
COARSE_TOPIC_ROOTS = (
    "Arts", "Sports", "Entertainment", "Mass media", "Geography", "History", "Military", "Politics",
    "Government", "Law", "Economy", "Business", "Society", "Culture", "Humans", "Education", "Language",
    "Religion", "Philosophy", "Science", "Mathematics", "Technology", "Engineering", "Health", "Life",
    "Nature", "Universe", "Food and drink", "Time", "unlabeled",
)
TOOL_FOR_ACTION = {"follow": "follow_link", "search": "search"}


def action_format(cfg):
    fmt = str(cfg.experiment.get("action_format", "text"))
    if fmt not in ACTION_FORMATS:
        raise ValueError(f"unknown experiment.action_format {fmt!r}; expected one of {ACTION_FORMATS}")
    return fmt


def pre_run_prediction_mode(cfg):
    spec = cfg.experiment.get("pre_run_prediction") or {}
    mode = str(spec.get("mode", "none"))
    if mode not in PRE_RUN_PREDICTION_MODES:
        raise ValueError(f"unknown experiment.pre_run_prediction.mode {mode!r}; expected one of {PRE_RUN_PREDICTION_MODES}")
    if mode == "self_and_external":
        external_model = str(spec.get("external_model") or "").strip()
        if not external_model:
            raise ValueError("experiment.pre_run_prediction.external_model is required for self_and_external")
        if external_model == str(cfg.provider.model):
            raise ValueError("experiment.pre_run_prediction.external_model must differ from the navigation model")
    return mode


def pre_run_prediction_task(cfg):
    spec = cfg.experiment.get("pre_run_prediction") or {}
    task = str(spec.get("task", "pages_topics"))
    if task not in PRE_RUN_PREDICTION_TASKS:
        raise ValueError(
            f"unknown experiment.pre_run_prediction.task {task!r}; expected one of {PRE_RUN_PREDICTION_TASKS}"
        )
    return task


def tool_specs(cfg, names):
    """OpenAI-format tool list for the named tools, in this order, from configs/tools/*.yaml."""
    from omegaconf import OmegaConf
    defined = OmegaConf.to_container(cfg.tools.definitions, resolve=True)
    specs = []
    for name in (n for n in names if n != "stop" or stop_allowed(cfg)):  # experiment.allow_stop=false: no stop tool
        spec = defined[name]
        parameters = json.loads(json.dumps(spec["parameters"]))
        props = parameters.get("properties") or {}
        for key in [k for k, v in props.items() if v is None]:   # a config may remove a parameter by setting it to null
            del props[key]
        if memory_wants_note(cfg):  # self_notes in tools mode: the note travels as a tool argument
            parameters.setdefault("properties", {})["note"] = {"type": "string", "description": cfg.tools.note_description}
        specs.append({"type": "function", "function": {"name": name, "description": spec["description"], "parameters": parameters}})
    return specs


def tool_call_as_text(call):
    """Translate a tool call into the reply format the parsers already understand, so parsing stays in one place."""
    args, name = call.get("arguments") or {}, call.get("name")
    payload = {"reasoning": args.get("reasoning") or args.get("reason") or args.get("essay") or ""}   # essay: a stop that hands one in
    if name == "follow_link":
        payload.update(action="follow", choice=args.get("title"))
    elif name == "search":
        payload.update(action="search", query=args.get("query"))
    elif name == "stop":
        payload.update(action="follow", choice="STOP")
    else:
        payload.update(action="unknown", tool=name)
    if args.get("note"): payload["note"] = args["note"]
    return json.dumps(payload, ensure_ascii=False)


def followup(cfg, caller, response, result_text, text_suffix=""):
    """Messages that hand a result back to the model: a tool result in tools mode, a user message in text mode."""
    content = result_text + (("\n\n" + text_suffix) if text_suffix else "")
    call = caller.last_call
    if action_format(cfg) == "tools" and call is not None:
        return [{"role": "assistant", "content": None,
                 "tool_calls": [{"id": call["id"], "type": "function",
                                 "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": call["id"], "content": content}]
    return [{"role": "assistant", "content": response.text or ""},
            {"role": "user", "content": content}]


def search_result_instruction(cfg):
    """Tell the model how to act on the current search results, using the active action protocol."""
    if action_format(cfg) != "tools":
        return stop_text(cfg, cfg.variant.prompts.choose_result)
    instruction = ("Choose exactly one current search result now. Call follow_link exactly once with a title "
                   "exactly as listed above. Do not call search again and do not choose a title from an earlier "
                   "page or search-result list.")
    if stop_allowed(cfg):
        instruction += " To stop instead, call stop."
    return instruction




def harness_pick_note(cfg, item):
    """What the model is told about a step the harness decided (its reply could not be read): never shown as its own reply."""
    return str(cfg.experiment.get("harness_pick_note") or HARNESS_PICK_NOTE).format(title=item["choice"])


def history_messages(history, cfg=None):
    """path_only (default): each past step as the model's own reply only; a harness pick as a harness note."""
    return [{"role": "user", "content": harness_pick_note(cfg, item)} if item.get("harness_pick") else
            {"role": "assistant", "content": json.dumps({"reasoning": item["reasoning"], "chose": item["choice"]})} for item in history]


def _past_exchange(cfg, item, page_text, n):
    """An earlier page and the model's own action on it (past_actions), or, for a harness pick, the page with the
    harness note: a step the harness decided is never shown as the model's reply or tool call."""
    if item.get("harness_pick"):
        return [{"role": "user", "content": page_text + "\n\n" + harness_pick_note(cfg, item)}]
    return [{"role": "user", "content": page_text}] + past_actions(cfg, item, n)


def _reply(item):
    payload = {"reasoning": item["reasoning"], "chose": item["choice"]}
    if item.get("note"): payload["note"] = item["note"]
    return {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)}


def _tool_pair(call_id, name, arguments, result):
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": call_id, "content": result}]


def past_actions(cfg, item, n):
    """How the model's own earlier action appears in the conversation.

    text mode (unchanged): its JSON reply.
    tools mode: the tool call it made, followed by the tool's result, exactly like a live tool conversation, so
    the model never sees a JSON reply it could copy instead of calling a tool. Only tools that are actually offered
    in this variant appear: a search-first opening is described in the result text, not as a past call.
    """
    if item.get("harness_pick"):  # the harness decided this step: a note, never a reply or tool call of the model's
        return [{"role": "user", "content": harness_pick_note(cfg, item)}]
    if action_format(cfg) != "tools":
        return [_reply(item)]
    pairs, searched_here = [], f' (from the search results for "{item["search_query"]}")' if item.get("search_query") else ""
    if item.get("search_query") and variant_of(cfg) in SEARCH_ANYTIME_VARIANTS:
        args = {"query": item["search_query"]}
        pairs += _tool_pair(f"past_{n}_search", "search", args, f'Search results for "{item["search_query"]}" were shown.')
        searched_here = ""
    args = {"title": item["choice"]}
    if item.get("reasoning"): args["reasoning"] = item["reasoning"]
    if item.get("note"): args["note"] = item["note"]
    pairs += _tool_pair(f"past_{n}_follow", "follow_link", args, f"Opened: {item['choice']}{searched_here}")
    return pairs


def _past_page(item, mode, article_max_chars):
    """How one earlier step is shown, depending on the memory mode."""
    parts, page = [], item.get("page")
    if page is not None:
        if mode == "full":
            parts.append(item["observation"])
        elif mode == "pages_no_links":
            article, _ = clean_article_text(page.wikitext, article_max_chars)
            parts.append(f"# {page.title}\n\n## Article content\n{article}")
        elif mode == "lead":
            lead, _ = clean_article_text((page.wikitext or "").split("\n==", 1)[0], article_max_chars)
            parts.append(f"# {page.title}\n\n{lead}")
    if item.get("search_query"):
        parts.append(item["results_text"] if mode == "full" else f'(You searched Wikipedia for "{item["search_query"]}".)')
    return "\n\n".join(parts) or "(earlier step)"


def _path_line(n, item):
    where = item["page"].title if item.get("page") is not None else f'search "{item.get("search_query")}"'
    if item.get("harness_pick"):
        return f"{n}. {where} -> {item['choice']} (opened at random: no listed page was chosen)."
    note = f' Note: {item["note"]}' if item.get("note") else ""
    return f"{n}. {where} -> {item['choice']}. {item['reasoning'] or ''}{note}".strip()


def build_messages(cfg, rule, history, observation):
    """System + memory of earlier steps + the current page, with a fresh memory object (stateless built-ins)."""
    return make_memory(cfg).messages(rule, history, observation)


def builtin_messages(cfg, mode, rule, history, observation):
    """The built-in memories: system + earlier steps shown according to `mode` + the current page."""
    max_chars = cfg.experiment.article_max_chars
    max_chars = None if max_chars is None else int(max_chars)
    note_rule = "\n" + NOTE_RULE if mode == "self_notes" and action_format(cfg) == "text" else ""  # tools: "note" is a tool argument
    system = {"role": "system", "content": cfg.experiment.system_prompt + "\n\n" + rule + note_rule}
    if mode == "none":  # memoryless: only the current page, nothing from earlier steps
        return [system, {"role": "user", "content": observation}]
    if mode == "path_only":
        if action_format(cfg) != "tools":
            return [system] + history_messages(history, cfg) + [{"role": "user", "content": observation}]
        earlier = [m for n, item in enumerate(history, 1) for m in past_actions(cfg, item, n)]
        return [system] + earlier + [{"role": "user", "content": observation}]
    if mode == "self_notes":
        if not history:
            return [system, {"role": "user", "content": observation}]
        notes = "\n".join(_path_line(n, item) for n, item in enumerate(history, 1))
        return [system, {"role": "user", "content": f"Your earlier steps and notes:\n{notes}\n\n{observation}"}]
    if mode == "sliding_window":
        k = int(cfg.experiment.get("memory_window", 2))
        older, recent = (history[:-k], history[-k:]) if k > 0 else (history, [])
        messages, prefix = [system], ""
        if older:
            prefix = "Your earlier steps:\n" + "\n".join(_path_line(n, item) for n, item in enumerate(older, 1)) + "\n\n"
        for n, item in enumerate(recent, len(older) + 1):
            messages += _past_exchange(cfg, item, prefix + _past_page(item, "full", max_chars), n); prefix = ""
        return messages + [{"role": "user", "content": prefix + observation}]
    messages = [system]  # pages_no_links, lead, full: alternating user (earlier page) / assistant (its reply or tool call)
    for n, item in enumerate(history, 1):
        messages += _past_exchange(cfg, item, _past_page(item, mode, max_chars), n)
    return messages + [{"role": "user", "content": observation}]


class BuiltinMemory(Memory):
    """A built-in memory: the prompt is rebuilt every step from the harness's list of earlier steps."""
    def messages(self, rule, history, observation):
        return builtin_messages(self.cfg, self.name, rule, history, observation)


for _name, _doc in (("path_only", "Only the model's own earlier replies (default)."),
                    ("none", "Memoryless: only the current page."),
                    ("pages_no_links", "Earlier articles in full, without their link lists."),
                    ("lead", "Earlier pages as their lead paragraph."),
                    ("sliding_window", "The last experiment.memory_window pages in full, older ones as path lines."),
                    ("self_notes", "No pages; the model adds a note to each reply and sees its path and notes."),
                    ("full", "Everything it saw at every step.")):
    MEMORIES.register(_name)(type(f"Memory_{_name}", (BuiltinMemory,),
                                  {"name": _name, "__doc__": _doc, "wants_note": _name == "self_notes"}))


@MEMORIES.register("collection_path_only")
class CollectionPathMemory(Memory):
    """Path-only memory plus a direct, machine-resolvable tool for saving the current page.

    Saving is deliberately a side action: the model may save and follow a link in one reply, or save first and
    then choose a link after the tool result. The tool never accepts a title, so a saved item is always the page the
    model was actually shown. Duplicate saves are prevented by withdrawing the tool on an already-saved page.
    """
    name = "collection_path_only"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.target = int(cfg.experiment.get("collection_target", 10))
        self.instruction = str(cfg.experiment.get("collection_instruction") or "").strip()
        self.run, self.current_title = None, None
        self.selections, self.pending = [], []

    @classmethod
    def validate(cls, cfg):
        if action_format(cfg) != "tools":
            raise ValueError("collection_path_only requires experiment.action_format=tools")
        if int(cfg.experiment.get("collection_target", 0)) < 1:
            raise ValueError("collection_path_only requires experiment.collection_target >= 1")
        if not str(cfg.experiment.get("collection_instruction") or "").strip():
            raise ValueError("collection_path_only requires experiment.collection_instruction")
        if stop_allowed(cfg):
            raise ValueError("collection_path_only requires experiment.allow_stop=false")

    def start(self, run):
        self.run = run
        run.doc["collection"] = {"target": self.target, "instruction": self.instruction,
                                 "selections": [], "complete": False}
        run.write()

    def settings(self):
        return {"base_memory": "path_only", "interface": "save_current_page",
                "target": self.target, "instruction": self.instruction}

    def messages(self, rule, history, observation):
        match = re.match(r"# ([^\n]+)", observation)
        self.current_title = match.group(1).strip() if match else None
        saved = [row["title"] for row in self.selections]
        task = (f"Collection task: {self.instruction}\n"
                f"Save exactly {self.target} unique pages. On a page you want in the final collection, call "
                "save_page. You may also call follow_link in the same reply. Saving keeps you on the current page; "
                "navigation continues until the collection is complete or the step limit is reached.")
        progress = (f"## Collection progress\nSaved {len(saved)}/{self.target}.\n"
                    + ("Saved pages: " + "; ".join(saved) if saved else "Saved pages: none"))
        return builtin_messages(self.cfg, "path_only", task + "\n\n" + rule, history,
                                progress + "\n\n" + observation)

    def side_tools(self):
        if not self.current_title or any(row["title"] == self.current_title for row in self.selections):
            return []
        return [{"type": "function", "function": {
            "name": "save_page",
            "description": "Save the current Wikipedia page as one of the final collection items.",
            "parameters": {"type": "object", "properties": {
                "reason": {"type": "string", "description": "Why this page belongs in the final collection, in 1-3 sentences."}
            }, "required": ["reason"]}}}]

    def side_turns(self):
        return 1

    def handle(self, call):
        if call.get("name") != "save_page":
            raise ValueError(f"unknown collection tool {call.get('name')!r}")
        if not self.current_title:
            raise ValueError("current page is unavailable")
        if any(row["title"] == self.current_title for row in self.selections):
            return f"Already saved: {self.current_title}. Saved {len(self.selections)}/{self.target}."
        if len(self.selections) >= self.target:
            return f"Collection already complete ({self.target}/{self.target})."
        record = {"selection_index": len(self.selections) + 1, "title": self.current_title,
                  "reason": str((call.get("arguments") or {}).get("reason") or "")[:1000],
                  "step_index": None, "page_id": None, "semantic_cluster_label": []}
        self.selections.append(record); self.pending.append(record)
        self._sync()
        return f"Saved: {self.current_title}. Saved {len(self.selections)}/{self.target}."

    def _sync(self):
        if self.run is None:
            return
        self.run.doc["collection"] = {"target": self.target, "instruction": self.instruction,
                                      "selections": list(self.selections),
                                      "complete": len(self.selections) == self.target}
        self.run.write()

    def step_fields(self, step):
        saved_now = []
        for record in self.pending:
            record.update(step_index=step["step_index"], page_id=step["current_page_id"],
                          semantic_cluster_label=list(step["semantic_cluster_label"]))
            saved_now.append(record["title"])
        self.pending = []
        self._sync()
        return {"collection_saved_this_step": saved_now, "collection_saved_total": len(self.selections),
                "collection_target": self.target}

    def should_end(self):
        return "collection_complete" if len(self.selections) >= self.target else None

    def finish(self, run, reason):
        self._sync()
        (run.dir / "collection.json").write_text(json.dumps(run.doc["collection"], ensure_ascii=False, indent=2),
                                                  encoding="utf-8")


INTENT_MODES = ("none", "probe", "walk")


def intent_mode(cfg):
    mode = str(cfg.experiment.get("intent", "none"))
    if mode not in INTENT_MODES:
        raise ValueError(f"unknown experiment.intent {mode!r}; expected one of {INTENT_MODES}")
    if mode == "walk" and variant_of(cfg) not in SEARCH_ANYTIME_VARIANTS:
        raise ValueError("experiment.intent=walk (go wherever the intent points) needs "
                         "variant=c_search_anytime or d_search_first_anytime")
    return mode


REVEAL_BUDGETS = ("none", "total", "remaining")


def reveal_budget(cfg):
    """experiment.reveal_budget (default none): whether the model is told its step limit.
    total: one sentence in the system prompt; remaining: a line under each page's link list."""
    mode = str(cfg.experiment.get("reveal_budget", "none"))
    if mode not in REVEAL_BUDGETS:
        raise ValueError(f"unknown experiment.reveal_budget {mode!r}; expected one of {REVEAL_BUDGETS}")
    return mode


def steps_sentence(n, more=False):
    """'You can take 15 steps through links.' / 'You can take 3 more steps through links, including this one.'"""
    word = "step" if n == 1 else "steps"
    return f"You can take {n} more {word} through links, including this one." if more else f"You can take {n} {word} through links."


def page_without_links(page, article_max_chars=None):
    """What the intent call shows: the same cleaned article, and no link list at all."""
    article_text, _ = clean_article_text(page.wikitext, article_max_chars)
    return f"# {page.title}\n\n## Article content\n{article_text}"


def parse_intent(text):
    """Return (intent, why) or (None, raw)."""
    raw = (text or "").strip()
    data = _json_object(raw)
    if data and str(data.get("intent") or "").strip():
        return str(data["intent"]).strip()[:300], str(data.get("why") or "")[:500] or None
    match = re.search(r'"intent"\s*:\s*"([^"]+)"', raw)
    return (match.group(1).strip(), None) if match else (None, raw[:400])


def map_intent(search, backend, text, choice_set, top_k=3):
    """Turn free-text intent into an article: an exact link title first, otherwise the top search results."""
    by_title = {row["title"].casefold(): row for row in choice_set}
    exact = by_title.get((text or "").strip().casefold())
    top = []
    if search is not None and text:
        hits, _ = search.search(text, 20)
        seen = set()
        for hit in hits:
            key = wiki_key(hit["matched_title"])
            if not backend.contains(key) or key.startswith(SKIP_NS):
                continue
            target = backend.load_page(key)
            if target is None or target.key in seen:
                continue
            seen.add(target.key); top.append({"title": target.title, "page_id": target.page_id})
            if len(top) >= top_k:
                break
    if exact:
        return {"title": exact["title"], "page_id": exact["page_id"]}, "link_title", top
    return (top[0], "search", top) if top else (None, None, top)


def intent_record(text, why, raw, parse_ok, mapped, method, top, choice_set, visited, page, tokens, latency,
                  provider_usage, internal_reasoning=None):
    in_list = next((row for row in choice_set if mapped and (row["title"] == mapped["title"] or row["page_id"] == mapped["page_id"])), None)
    return {"text": text, "why": why, "raw": raw, "parse_ok": parse_ok,
            "mapped_title": mapped["title"] if mapped else None, "mapped_page_id": mapped["page_id"] if mapped else None,
            "match_method": method, "top_results": top,
            "in_links": in_list is not None,
            "intended_shuffled_position": in_list["shuffled_position"] if in_list else None,
            "intended_original_position": in_list["original_position"] if in_list else None,
            "is_current_page": bool(mapped) and wiki_key(mapped["title"]) == page.key,
            "already_visited": bool(mapped) and wiki_key(mapped["title"]) in visited,
            "choice_matches_intent": None, "tokens": tokens, "latency_s": latency,
            "provider_usage": provider_usage, "internal_reasoning": internal_reasoning}


def run_intent_call(cfg, provider, run_seed, history, page, max_chars, save, search, backend, choice_set, visited, memory=None):
    """The intent call: same system prompt and memory as the choice, the page WITHOUT links, nothing fed back.
    Returns (record, error text, termination reason for an error)."""
    caller = Caller(provider, cfg, run_seed)
    memory = memory if memory is not None else make_memory(cfg)
    messages = memory.messages(cfg.experiment.intent_rule, history, page_without_links(page, max_chars))
    save("request_intent", messages)
    if action_format(cfg) == "tools":
        names = ["follow_link", "search", "stop"] if variant_of(cfg) in SEARCH_ANYTIME_VARIANTS else ["follow_link", "stop"]
        response = caller(messages, tool_specs(cfg, names), tool_choice="none")
    else:
        response = caller(messages)
    if response.error:
        return None, response.error, caller.termination()
    text, why = parse_intent(response.text)
    internal = getattr(response, "internal_reasoning", None)
    mapped, method, top = map_intent(search, backend, text, choice_set) if text else (None, None, [])
    return intent_record(text, why, response.text, text is not None, mapped, method, top, choice_set, visited, page,
                         dict(caller.tokens), round(caller.api_seconds, 3), caller.provider_usage(),
                         caller.internal_reasoning()), None, None


def reply_note(caller):
    """self_notes: the note field of the reply that made the choice."""
    data = _json_object((caller.raw[-1] if caller.raw else "") or "") or {}
    note = str(data.get("note") or "").strip()
    return note[:500] or None


def save_model_inputs(run_dir, step_index, name, messages, page=None, titles=None):
    """Human-readable copy of what the model was given (debugging aid, not analysis data).

    model_inputs/step_XXX_raw_wikitext.txt   the page as stored in the dump (NOT sent to the model)
    model_inputs/step_XXX_links.txt          the titles offered, in the order shown
    model_inputs/step_XXX_<name>.txt         every message actually sent in that call
    """
    folder = Path(run_dir) / "model_inputs"; folder.mkdir(exist_ok=True)
    stem = f"step_{step_index:03d}"
    if page is not None:
        (folder / f"{stem}_raw_wikitext.txt").write_text(page.wikitext or "", encoding="utf-8")
    if titles is not None:
        (folder / f"{stem}_links.txt").write_text(f"# {len(titles)} links shown to the model, in this order\n" + "\n".join(titles), encoding="utf-8")
    if messages is not None:
        def shown(m):
            text = m.get("content") or ""
            if m.get("tool_calls"):
                text += "\n".join(f"[tool call] {c['function']['name']}({c['function']['arguments']})" for c in m["tool_calls"])
            return text
        total = sum(len(shown(m)) for m in messages)
        body = "\n\n".join(f"===== [{m['role']}] ({len(shown(m)):,} chars) =====\n{shown(m)}" for m in messages)
        (folder / f"{stem}_{name}.txt").write_text(f"# {len(messages)} messages, {total:,} chars (~{total // 4:,} tokens, rough)\n\n" + body, encoding="utf-8")


# ----------------------------------------------------------------------------- parsing replies
def _json_object(raw):
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else None
            except Exception: return None
    return None


def _prediction_messages(cfg, page, framing):
    spec = cfg.experiment.get("pre_run_prediction") or {}
    task = pre_run_prediction_task(cfg)
    max_pages, max_topics = int(spec.get("max_pages", 5)), int(spec.get("max_topics", 5))
    if framing == "self":
        subject = "your own likely navigation in a fresh, independent run"
        role = "You are forecasting a later, stateless navigation run by another instance of yourself."
    elif framing == "external":
        subject = f"the likely navigation of target model {cfg.provider.model} in a fresh, independent run"
        role = "You are an independent external forecaster, not the later navigation agent."
    else:
        subject = "the likely navigation of a typical language-model agent in a fresh, independent run"
        role = "You are forecasting a later, stateless navigation run."
    if task == "memory_effect":
        system = (
            role + " Make a preregistered forecast before two matched Wikipedia navigation runs. "
            "Do not navigate and do not choose an action now. The forecast will be stored separately and will not be "
            "shown to either navigation agent. Return the JSON object immediately, with no prose before or after it: "
            '{"higher_unique_pages": "none|path_only|tie", '
            '"unique_pages_none": "1-5|6-15|16-30", '
            '"unique_pages_path_only": "1-5|6-15|16-30", '
            '"higher_revisit_rate": "none|path_only|tie", '
            '"more_likely_to_reach_step_limit": "none|path_only|tie", '
            '"p_none_reaches_limit": <number from 0 to 1>, '
            '"p_path_only_reaches_limit": <number from 0 to 1>, '
            '"confidence": <number from 0 to 1>}'
        )
    elif task == "coarse_single_run":
        roots = " | ".join(COARSE_TOPIC_ROOTS)
        system = (
            role + " Make a preregistered coarse forecast before one Wikipedia navigation run. "
            "Do not navigate and do not choose an action now. The forecast will be stored separately and will not be "
            "shown to the later navigation agent. A loop means that the run visits a page it visited earlier. "
            "Return the JSON object immediately, with no prose before or after it: "
            f'{{"first_topic_root": "{roots}", '
            '"unique_pages": "1-5|6-15|16+", '
            '"any_page_revisit": <true or false>, '
            '"reaches_step_limit": <true or false>, '
            '"confidence": <number from 0 to 1>}'
        )
    else:
        system = (
            role + " Make a preregistered forecast before a Wikipedia navigation run. "
            "Do not navigate and do not choose an action now. The forecast will be stored separately and will not be "
            "shown to the later navigation agent. Return JSON only, no other text: "
            f'{{"pages": ["up to {max_pages} exact Wikipedia page titles"], '
            f'"topics": ["up to {max_topics} concise topic labels"], "confidence": <number from 0 to 1>}}'
        )
    setup = (
        f"Target navigation model: {cfg.provider.model}\n"
        f"Target temperature: {cfg.provider.temperature}\n"
        f"Target memory: {cfg.experiment.memory}\n"
        f"Target step limit: {cfg.experiment.step_limit}\n"
        f"Target may stop early: {bool(cfg.experiment.get('allow_stop', True))}\n"
        f"Link presentation policy: {cfg.experiment.display_order_policy}; the exact seeded order is selected after this forecast "
        "and is not available to you.\n"
    )
    if task == "memory_effect":
        comparison_steps = int(spec.get("comparison_step_limit", 30))
        comparison_allow_stop = bool(spec.get("comparison_allow_stop", True))
        comparison_mask = str(spec.get("comparison_link_mask_mode", "visited"))
        user = (
            f"Target navigation model: {cfg.provider.model}\n"
            f"Target temperature: {cfg.provider.temperature}\n"
            f"Matched conditions: memory=none versus memory=path_only\n"
            f"Steps per run: {comparison_steps}\n"
            f"May stop early: {comparison_allow_stop}\n"
            f"Exact-revisit masking: {comparison_mask}\n"
            f"Link presentation policy: {cfg.experiment.display_order_policy}; exact seeded orders are selected after this "
            "forecast and are unavailable to you. The two later runs use the same start page and matched run seed.\n"
            f"Starting page: {page.title}\n\n"
            f"Predict the direction and coarse magnitude of the memory-condition effect for {subject}. "
            "Use only the fixed response categories in the JSON schema. Base the forecast only on the starting article "
            "below; no candidate-link list is shown.\n\n"
            + page_without_links(page, cfg.experiment.article_max_chars)
        )
    elif task == "coarse_single_run":
        user = (
            setup + f"Starting page: {page.title}\n\n"
            f"Predict the coarse outcomes of {subject}. For first_topic_root, predict the broad topic root of the first "
            "page selected after the starting page. Use only the fixed response categories. Base the forecast only on "
            "the starting article below; no candidate-link list is shown.\n\n"
            + page_without_links(page, cfg.experiment.article_max_chars)
        )
    else:
        user = (
            setup + f"Starting page: {page.title}\n\n"
            f"Predict {subject}. Base the forecast only on the starting article below; no candidate-link list is shown.\n\n"
            + page_without_links(page, cfg.experiment.article_max_chars)
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _prediction_record(response, framing, seed, order_index, max_pages, max_topics, forecaster_provider, forecaster_model,
                       task="pages_topics", request_controls=None):
    data = _json_object(response.text or "") or {}
    pages = [str(v).strip() for v in data.get("pages", []) if str(v).strip()] if isinstance(data.get("pages"), list) else []
    topics = [str(v).strip() for v in data.get("topics", []) if str(v).strip()] if isinstance(data.get("topics"), list) else []
    confidence = data.get("confidence")
    confidence = float(confidence) if isinstance(confidence, (int, float)) and 0 <= float(confidence) <= 1 else None
    effect, coarse = None, None
    if task == "memory_effect":
        direction_values = {"none", "path_only", "tie"}
        bin_values = {"1-5", "6-15", "16-30"}
        probability_keys = ("p_none_reaches_limit", "p_path_only_reaches_limit")
        effect = {
            "higher_unique_pages": data.get("higher_unique_pages"),
            "unique_pages_none": data.get("unique_pages_none"),
            "unique_pages_path_only": data.get("unique_pages_path_only"),
            "higher_revisit_rate": data.get("higher_revisit_rate"),
            "more_likely_to_reach_step_limit": data.get("more_likely_to_reach_step_limit"),
            **{key: float(data[key]) if isinstance(data.get(key), (int, float)) else None for key in probability_keys},
        }
        effect_ok = (
            effect["higher_unique_pages"] in direction_values
            and effect["unique_pages_none"] in bin_values
            and effect["unique_pages_path_only"] in bin_values
            and effect["higher_revisit_rate"] in direction_values
            and effect["more_likely_to_reach_step_limit"] in direction_values
            and all(effect[key] is not None and 0 <= effect[key] <= 1 for key in probability_keys)
            and confidence is not None
        )
    elif task == "coarse_single_run":
        coarse = {
            "first_topic_root": data.get("first_topic_root"),
            "unique_pages": data.get("unique_pages"),
            "any_page_revisit": data.get("any_page_revisit"),
            "reaches_step_limit": data.get("reaches_step_limit"),
        }
        effect_ok = (
            coarse["first_topic_root"] in set(COARSE_TOPIC_ROOTS)
            and coarse["unique_pages"] in {"1-5", "6-15", "16+"}
            and isinstance(coarse["any_page_revisit"], bool)
            and isinstance(coarse["reaches_step_limit"], bool)
            and confidence is not None
        )
    else:
        effect_ok = bool(pages or topics)
    usage = response.usage or {}
    provider_usage = {
        "cost_usd": float(usage["cost"]) if isinstance(usage.get("cost"), (int, float)) else None,
        "generation_ids": [response.response_id] if response.response_id else [],
        "response_models": [response.response_model] if response.response_model else [],
    }
    return {
        "framing": framing, "order_index": order_index, "seed": seed,
        "forecaster_provider": forecaster_provider, "forecaster_model": forecaster_model,
        "task": task, "parse_ok": effect_ok and not response.error,
        "pages": pages[:max_pages], "topics": topics[:max_topics], "confidence": confidence,
        "memory_effect": effect,
        "coarse_single_run": coarse,
        "request_controls": dict(request_controls or {}),
        "raw": response.text or "", "internal_reasoning": response.internal_reasoning,
        "finish_reason": response.finish_reason, "error": response.error, "error_kind": response.error_kind,
        "tokens": {
            "input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens"),
            "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        },
        "latency_s": response.latency_s, "retry_count": response.retries,
        "retry_attempts": response.attempts, "provider_usage": provider_usage,
    }


def _external_prediction_provider(cfg):
    """Build the external forecaster from the navigation provider config with only its model replaced."""
    from omegaconf import OmegaConf, open_dict
    provider_cfg = OmegaConf.create(OmegaConf.to_container(cfg.provider, resolve=True))
    with open_dict(provider_cfg):
        provider_cfg.model = str(cfg.experiment.pre_run_prediction.external_model)
    return make_provider(provider_cfg)


def _prediction_json_schema(task, max_pages, max_topics):
    probability = {"type": "number", "minimum": 0, "maximum": 1}
    if task == "memory_effect":
        direction = {"type": "string", "enum": ["none", "path_only", "tie"]}
        page_bin = {"type": "string", "enum": ["1-5", "6-15", "16-30"]}
        properties = {
            "higher_unique_pages": direction, "unique_pages_none": page_bin,
            "unique_pages_path_only": page_bin, "higher_revisit_rate": direction,
            "more_likely_to_reach_step_limit": direction, "p_none_reaches_limit": probability,
            "p_path_only_reaches_limit": probability, "confidence": probability,
        }
    elif task == "coarse_single_run":
        properties = {
            "first_topic_root": {"type": "string", "enum": list(COARSE_TOPIC_ROOTS)},
            "unique_pages": {"type": "string", "enum": ["1-5", "6-15", "16+"]},
            "any_page_revisit": {"type": "boolean"}, "reaches_step_limit": {"type": "boolean"},
            "confidence": probability,
        }
    else:
        properties = {
            "pages": {"type": "array", "items": {"type": "string"}, "maxItems": max_pages},
            "topics": {"type": "array", "items": {"type": "string"}, "maxItems": max_topics},
            "confidence": probability,
        }
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _prediction_request_controls(spec, framing, task, max_pages, max_topics):
    controls = {}
    effort = spec.get(f"{framing}_reasoning_effort") or spec.get("reasoning_effort")
    if effort:
        controls["reasoning"] = {"effort": str(effort)}
    if bool(spec.get("structured_output", False)):
        controls["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": f"pre_run_{task}", "strict": True,
                "schema": _prediction_json_schema(task, max_pages, max_topics),
            },
        }
    return controls


def run_pre_run_predictions(cfg, provider, run, page, run_seed, save_inputs):
    """Run stateless forecasts before navigation; never add them to the navigation memory or request."""
    mode = pre_run_prediction_mode(cfg)
    if mode == "none" or page is None:
        return None
    if mode == "self_and_generic":
        framings = ["self", "generic"]
    elif mode == "self_and_external":
        framings = ["self", "external"]
    else:
        framings = [mode]
    if len(framings) == 2 and run_seed % 2:
        framings.reverse()  # counterbalance harmless order/provider-load effects
    spec = cfg.experiment.pre_run_prediction
    task = pre_run_prediction_task(cfg)
    max_pages, max_topics = int(spec.max_pages), int(spec.max_topics)
    external_provider = _external_prediction_provider(cfg) if "external" in framings else None
    records = []
    for order_index, framing in enumerate(framings, 1):
        seed = run_seed + 100000 + order_index
        messages = _prediction_messages(cfg, page, framing)
        if save_inputs:
            save_model_inputs(run.dir, 0, f"pre_run_prediction_{framing}", messages)
        forecaster = external_provider if framing == "external" else provider
        forecaster_model = str(spec.external_model) if framing == "external" else str(cfg.provider.model)
        temperature = float(spec.get("external_temperature", 0)) if framing == "external" else float(cfg.provider.temperature)
        top_p = float(spec.get("external_top_p", 1)) if framing == "external" else float(cfg.provider.top_p)
        request_controls = _prediction_request_controls(spec, framing, task, max_pages, max_topics)
        response = forecaster.chat(
            messages, seed=seed, temperature=temperature, top_p=top_p,
            max_tokens=int(spec.max_tokens),
            request_overrides=request_controls,
        )
        records.append(_prediction_record(
            response, framing, seed, order_index, max_pages, max_topics,
            str(cfg.provider.name), forecaster_model, task, request_controls,
        ))
    result = {
        "mode": mode, "task": task, "navigation_context_contaminated": False,
        "article_content_only": True, "candidate_links_shown": False,
        "target_model": str(cfg.provider.model),
        "external_forecaster_model": str(spec.external_model) if mode == "self_and_external" else None,
        "records": records,
    }
    run.doc["pre_run_prediction"] = result
    (run.dir / "pre_run_prediction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    run.write()
    return result


def parse_reply(text, available, stop_ok=True):
    """(title | 'STOP' | None, reasoning, error). A listed title always wins, so a page called "None" or "Finish" is
    opened, not read as stopping; only "STOP" stops. With stop_ok=False (experiment.allow_stop=false) a STOP is
    out-of-range, so the model is asked again."""
    raw, reasoning, choice = (text or "").strip(), "", None
    data = _json_object(raw)
    if data: reasoning, choice = data.get("reasoning", ""), data.get("choice") if data.get("choice") is not None else data.get("chose")
    if choice is None:
        match = re.search(r'"(?:choice|chose)"\s*:\s*"([^"]+)"', raw)
        choice = match.group(1) if match else None
    if choice is None:
        return None, raw[:400], "refusal" if re.search(r"\b(cannot|can't|won't|refuse|unable)\b", raw, re.I) else "unparseable"
    choice = str(choice).strip().strip('"').split(" — ")[0].strip()  # a copied "Title — snippet" still counts
    if choice in available: return choice, reasoning, None
    if choice.lower() == "stop":
        return ("STOP", reasoning or raw[:400], None) if stop_ok else (None, reasoning or raw[:400], "out-of-range")
    lowered = {item.lower(): item for item in available}
    if choice.lower() in lowered: return lowered[choice.lower()], reasoning, None
    matches = [item for item in available if choice.lower() and choice.lower() in item.lower()]
    return (matches[0], reasoning, None) if len(matches) == 1 else (None, reasoning or raw[:400], "out-of-range")


def parse_query(text):
    """Return (query, reasoning) or (None, raw)."""
    raw = (text or "").strip()
    data = _json_object(raw)
    if data and str(data.get("query") or "").strip():
        return str(data["query"]).strip(), str(data.get("reasoning") or "")
    match = re.search(r'"query"\s*:\s*"([^"]+)"', raw)
    return (match.group(1).strip(), "") if match else (None, raw[:400])


def parse_action(text):
    """Search-anytime: 'search' if the reply asks to search (with a query), otherwise 'follow'."""
    data = _json_object((text or "").strip()) or {}
    action = str(data.get("action") or "").strip().lower()
    if action == "search" or (not action and data.get("query") and not (data.get("choice") or data.get("chose"))):
        return "search"
    return "follow"


# ----------------------------------------------------------------------------- paired post-run readouts
def readout_specs(cfg):
    """Resolved readout definitions, empty for ordinary navigation experiments."""
    return list(cfg.experiment.get("readouts") or [])


def readout_candidates(steps):
    """Unique visited titles in trajectory order, including the last successfully opened destination."""
    titles = []
    for step in steps:
        if step.get("current_page_title") and step["current_page_title"] != "[search]":
            titles.append(step["current_page_title"])
        if (step.get("resolution") or {}).get("resolve_status") == "success" and step.get("parsed_choice") not in (None, "STOP"):
            titles.append(step["parsed_choice"])
    return list(dict.fromkeys(titles))


def parse_readout(text, candidates, count):
    """Strictly parse exactly `count` distinct titles from the supplied candidate list."""
    data = _json_object((text or "").strip())
    if not data or not isinstance(data.get("selections"), list):
        return None, None, "unparseable"
    rows, seen = [], set()
    for item in data["selections"]:
        if not isinstance(item, dict):
            return None, None, "selection is not an object"
        title = str(item.get("title") or "").strip()
        if title not in candidates:
            return None, None, f"title not in candidate list: {title!r}"
        if title in seen:
            return None, None, f"duplicate title: {title!r}"
        seen.add(title)
        rows.append({"title": title, "reason": str(item.get("reason") or "").strip()})
    if len(rows) != count:
        return None, None, f"expected {count} selections, got {len(rows)}"
    return rows, str(data.get("selection_principle") or "").strip(), None


def run_post_run_readouts(run, cfg, provider, run_seed, save_inputs=True):
    """Run independent, matched readouts over one completed trajectory.

    Every condition receives the identical deterministically shuffled candidate list and identical sampling seed.
    Calls are stateless and their execution order is counterbalanced by run index.
    """
    specs = readout_specs(cfg)
    if not specs:
        return []
    count = int(cfg.experiment.get("readout_selection_count", 10))
    retries = max(0, int(cfg.experiment.get("readout_parse_retries", 1)))
    candidates = readout_candidates(run.steps)
    candidate_seed = (int(run_seed) * 1_000_003 + 91_991) % (2 ** 63 - 1)
    paired_seed = (int(run_seed) * 1_000_003 + 77_777) % (2 ** 63 - 1)
    shown = list(candidates); random.Random(candidate_seed).shuffle(shown)
    ordered = list(specs)
    if run.doc["run_index"] % 2 == 0:
        ordered.reverse()
    records = []
    for order, spec in enumerate(ordered, 1):
        readout_id = str(spec["id"])
        base = {
            "id": readout_id, "condition": str(spec["condition"]),
            "tracker_config_id": str(spec.get("tracker_config_id") or ""), "order": order,
            "candidate_titles": shown, "candidate_shuffle_seed": candidate_seed,
            "paired_sampling_seed": paired_seed, "requested_selection_count": count,
        }
        if len(shown) < count:
            records.append({**base, "valid": False, "error": f"insufficient_candidates: {len(shown)} < {count}",
                            "selections": [], "selection_principle": None, "raw_responses": [], "attempts": [],
                            "token_counts": {"input": 0, "output": 0, "reasoning": 0},
                            "provider_usage": {"cost_usd": None, "generation_ids": [], "response_models": []},
                            "latency_s": 0.0, "internal_reasoning": None})
            continue
        rule = ("Return JSON only: {\"selections\":[{\"title\":\"exact title from the list\","
                "\"reason\":\"one sentence\"}],\"selection_principle\":\"one sentence\"}. "
                f"Select exactly {count} distinct titles. Copy every title exactly; do not add titles.")
        candidate_text = "\n".join(f"- {title}" for title in shown)
        prompt = f"{str(spec['instruction']).strip()}\n\nPages available for selection ({len(shown)}):\n{candidate_text}"
        messages = [{"role": "system", "content": rule}, {"role": "user", "content": prompt}]
        caller = Caller(provider, cfg, paired_seed)
        raw, selections, principle, error = [], None, None, None
        for attempt in range(retries + 1):
            name = f"readout_{slug(readout_id)}_request_{attempt + 1}"
            if save_inputs:
                save_model_inputs(run.dir, int(cfg.experiment.step_limit) + order, name, messages)
            response = caller(messages)
            raw.append(response.text or "")
            if response.error:
                error = response.error
                break
            selections, principle, error = parse_readout(response.text, shown, count)
            if error is None:
                break
            if attempt < retries:
                messages = messages + [{"role": "assistant", "content": response.text or ""},
                                       {"role": "user", "content": f"Invalid response ({error}). Return the requested JSON with exactly {count} distinct titles copied from the list."}]
        records.append({**base, "valid": error is None, "error": error, "selections": selections or [],
                        "selection_principle": principle, "raw_responses": raw, "attempts": list(caller.attempts),
                        "token_counts": dict(caller.tokens), "provider_usage": caller.provider_usage(),
                        "latency_s": round(caller.api_seconds, 3), "internal_reasoning": caller.internal_reasoning()})
    # Store in canonical spec order even though call order is counterbalanced.
    by_id = {record["id"]: record for record in records}
    records = [by_id[str(spec["id"])] for spec in specs]
    run.doc["readouts"] = records
    run.write()
    (run.dir / "readouts.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "readouts": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


# ----------------------------------------------------------------------------- output files
class Dataset:
    def __init__(self, cfg, backend, root):
        self.cfg, self.backend, self.project_root = cfg, backend, Path(root)
        day = datetime.now().strftime("%y%m%d")
        self.batch_id = f"batch_{int(cfg.batch.index):02d}_{day}_{slug(cfg.provider.model, 28)}_{slug(cfg.experiment.condition, 24)}"
        self.experiment_root = Path(cfg.output.directory) / cfg.experiment.name
        self.root = self.experiment_root / self.batch_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.data = {"schema_version": SCHEMA_VERSION, "experiment_name": cfg.experiment.name, "variant": variant_of(cfg), "created_date": utc_now(), "contributors": list(cfg.experiment.contributors), "wikipedia_dump_info": {"source_url": "synthetic development fixture" if backend.backend_name == "fixture" else "https://dumps.wikimedia.org/enwiki/latest/", "size": None, "variants_included": [cfg.wikipedia.variant], **backend.provenance()}, "code": {"repo_url": cfg.experiment.repo_url, **git_provenance()}, "run_summary": {"total_runs": 0, "conditions_included": [], "models_included": []}, "batch_id": self.batch_id, "batch_index": cfg.batch.index}
        self.write()
        shared = resolved_config(cfg)
        (self.root / "batch.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "batch_id": self.batch_id, "batch_index": cfg.batch.index, "started_at": utc_now(), "start_pages": list(cfg.batch.start_pages), "n_starts": len(cfg.batch.start_pages), "config": shared}, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.root / "config.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "batch_id": self.batch_id, "batch_index": cfg.batch.index, **shared}, ensure_ascii=False, indent=2), encoding="utf-8")
    def write(self): (self.root / "dataset_metadata.json").write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def reopen(cls, cfg, backend, root, batch_dir):
        """An existing batch, for batch.resume. Refuses unless it was made by the same config (batch.resume aside)."""
        self = cls.__new__(cls)
        self.cfg, self.backend, self.project_root, self.root = cfg, backend, Path(root), Path(batch_dir)
        self.experiment_root, self.batch_id = self.root.parent, self.root.name
        stored_path = self.root / "config.json"
        if not stored_path.exists():
            raise FileNotFoundError(f"batch.resume: no batch at {self.root} (looked for config.json)")
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        if stored.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"batch.resume: {self.root} was written with schema {stored.get('schema_version')}, this harness writes {SCHEMA_VERSION}")
        def comparable(config):
            config = {k: v for k, v in config.items() if k not in ("schema_version", "batch_id", "batch_index")}
            config = json.loads(json.dumps(config))
            config.get("batch", {}).pop("resume", None)
            return config
        current = resolved_config(cfg)
        current["batch"]["index"] = stored.get("batch_index")
        differences = sorted(_config_differences(comparable(stored), comparable(current)))
        if differences:
            raise ValueError(f"batch.resume: the current config differs from the one {self.batch_id} was started with: {differences}")
        _set_config(cfg, "batch.index", stored.get("batch_index"))
        self.data = json.loads((self.root / "dataset_metadata.json").read_text(encoding="utf-8"))
        return self

    def completed_runs(self, starts):
        """Finished runs stay. Runs without summary.json (cut off) or that ended in api_error move to _superseded/ and
        are run again with the same run index and seed. Returns the run indexes to skip."""
        done, superseded = set(), []
        for run_dir in sorted(p for p in self.root.iterdir() if p.is_dir() and (p / "run.json").exists()):
            doc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            index, reason = doc.get("run_index"), (doc.get("outcome") or {}).get("termination_reason")
            if index is None or not 1 <= index <= len(starts) or doc.get("start_page") != starts[index - 1]:
                raise ValueError(f"batch.resume: {run_dir.name} (run {index}, start {doc.get('start_page')!r}) does not match this config's start pages")
            if (run_dir / "summary.json").exists() and reason != "api_error" and index not in done:
                done.add(index)
                continue
            target = self.root / "_superseded" / run_dir.name
            target.parent.mkdir(exist_ok=True)
            run_dir.rename(target); superseded.append(run_dir.name)
        summary = self.data["run_summary"]
        summary["total_runs"] = len(done)
        self.write()
        batch_path = self.root / "batch.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        batch.setdefault("resumes", []).append({"at": utc_now(), "kept_run_indexes": sorted(done), "superseded": superseded,
                                                "rerun_indexes": [i for i in range(1, len(starts) + 1) if i not in done]})
        batch_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        return done

    def register(self, condition, model):
        summary = self.data["run_summary"]; summary["total_runs"] += 1
        for key, value in (("conditions_included", condition), ("models_included", model)):
            if value not in summary[key]: summary[key].append(value)
        self.write()

    def append_run_log(self, summary):
        """Append one portable, Google-Sheets-shaped row per completed run."""
        exp = self.cfg.experiment
        contributors = list(exp.get("contributors") or [])
        readout_note = (f"Paired post-run readouts valid {summary.get('readouts_valid', 0)}/{summary.get('readouts_total', 0)}"
                        if summary.get("readouts_total") else "")
        collection_note = (f"Collection saved {summary.get('collection_selected', 0)}/{summary.get('collection_target')}"
                           if summary.get("collection_target") else "")
        cost_note = "Provider-reported cost unavailable" if summary["cost_usd"] is None else ""
        row = {
            "Config ID": exp.get("tracker_config_id") or "",
            "Batch ID": summary["batch_id"], "Run ID": summary["run_id"],
            "Owner": exp.get("tracker_owner") or (contributors[0] if contributors else ""),
            "Experiment": exp.name, "Condition": summary["condition"], "Run Index": summary["run_index"],
            "Start Page": summary["start_page"], "Model": summary["model"], "Provider": summary["provider"],
            "Started At (UTC)": summary["started_at"], "Finished At (UTC)": summary["finished_at"],
            "Status": summary["status"], "Termination Reason": summary["termination_reason"],
            "Steps": summary["n_steps"], "Input Tokens": summary["input_tokens"],
            "Output Tokens": summary["output_tokens"], "Reasoning Tokens": summary["reasoning_tokens"],
            "Cost (USD)": "" if summary["cost_usd"] is None else summary["cost_usd"],
            "Artifact Path": summary["artifact_path"], "Git Commit": summary["git_commit"] or "",
            "Synced At (UTC)": "",
            "Notes": "; ".join(note for note in (collection_note, readout_note, cost_note) if note),
        }
        path = self.root / "run_log.csv"
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUN_LOG_COLUMNS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def append_readout_log(self, run, readout):
        """Append one row per post-run readout; navigation and readout logs remain separately analysable."""
        exp = self.cfg.experiment
        contributors = list(exp.get("contributors") or [])
        usage = readout.get("provider_usage") or {}
        row = {
            "Config ID": readout.get("tracker_config_id") or "", "Batch ID": self.batch_id,
            "Run ID": run.run_id, "Owner": exp.get("tracker_owner") or (contributors[0] if contributors else ""),
            "Experiment": exp.name, "Readout ID": readout["id"], "Condition": readout["condition"],
            "Run Index": run.doc["run_index"], "Start Page": run.doc["start_page"],
            "Model": run.config["provider"]["model"], "Provider": run.config["provider"]["name"],
            "Order": readout["order"], "Candidate Count": len(readout["candidate_titles"]),
            "Selection Count": len(readout.get("selections") or []), "Valid": readout["valid"],
            "Error": readout.get("error") or "", "Input Tokens": (readout.get("token_counts") or {}).get("input", 0),
            "Output Tokens": (readout.get("token_counts") or {}).get("output", 0),
            "Reasoning Tokens": (readout.get("token_counts") or {}).get("reasoning", 0),
            "Cost (USD)": "" if usage.get("cost_usd") is None else usage["cost_usd"],
            "Candidate Shuffle Seed": readout["candidate_shuffle_seed"],
            "Paired Sampling Seed": readout["paired_sampling_seed"],
            "Selections": json.dumps(readout.get("selections") or [], ensure_ascii=False),
            "Selection Principle": readout.get("selection_principle") or "",
            "Artifact Path": str(run.dir.resolve()), "Git Commit": self.data["code"]["git_commit"] or "",
        }
        path = self.root / "readout_log.csv"
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=READOUT_LOG_COLUMNS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)


def resolved_config(cfg):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(cfg, resolve=True)


def effective_sampling(config, steps):
    """The sampling settings the model actually got: a setting the provider left out (dropped_params in any attempt)
    is None. `checked`: False when the provider could not tell which settings the model accepts (params_unchecked),
    None for providers that do not check."""
    attempts = [a for s in steps for a in (s.get('extras') or {}).get('retry_attempts') or []]
    dropped = sorted({p for a in attempts for p in a.get('dropped_params') or []})
    prov = config.get('provider', {})
    out = {"temperature": prov.get('temperature'), "top_p": prov.get('top_p'), "seed": config.get('run_seed')}
    out = {k: (None if k in dropped else v) for k, v in out.items()}
    checked = None if not attempts or not any(a.get('provider') == "openrouter" for a in attempts) else not any(a.get('params_unchecked') for a in attempts)
    return {**out, "dropped": dropped, "checked": checked}


RESERVED_RUN_KEYS = ("schema_version", "batch_id", "batch_index", "run_id", "run_index", "start_page", "variant", "condition",
                     "config", "outcome", "effective_sampling", "memory_settings", "context_embedding")


class Run:
    def __init__(self, dataset, run_index, start_page, config, extra=None):
        self.dataset, self.config, self.t0 = dataset, config, time.time()
        self.started_at = utc_now()
        base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__llm__{slug(config['provider']['model'], 28)}__seed{config['run_seed']}__{slug(start_page, 24)}"
        self.run_id, suffix = base, 2
        while (dataset.root / self.run_id).exists(): self.run_id = f"{base}__{suffix}"; suffix += 1
        self.dir = dataset.root / self.run_id; (self.dir / "steps").mkdir(parents=True)
        self.doc = {"schema_version": SCHEMA_VERSION, "batch_id": dataset.batch_id, "batch_index": dataset.cfg.batch.index, "run_id": self.run_id, "run_index": run_index, "start_page": start_page, "variant": variant_of(dataset.cfg), "condition": config['experiment']['condition'], "config": config, "outcome": {"outcome_category": None, "termination_reason": None, "model_stop_statement": None}}
        clash = sorted(set(extra or {}) & set(RESERVED_RUN_KEYS))
        if clash: raise ValueError(f"run_extra may not set run.json fields the harness writes: {clash}")
        self.doc.update(extra or {})   # e.g. world {world_id, agent_id, n_agents} from a multi-agent scheduler
        self.steps, self.rows, self.memory, self.yielded = [], [], None, None   # yielded: last step handed to a scheduler
        self.context_trace = None   # context_embedding=steps: embeds each step's final request (harness/context_embed.py)
        (self.dir / "config.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, **config, "run_id": self.run_id, "run_index": run_index, "start_page": start_page, "started_at": self.started_at}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.write()
    def write(self): (self.dir / "run.json").write_text(json.dumps(self.doc, ensure_ascii=False, indent=2), encoding="utf-8")
    def log(self, step):
        fields = self.memory.step_fields(step) if self.memory is not None else None
        if fields is not None:
            step["memory"] = fields
        self.steps.append(step)
        with (self.dir / "steps.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(step, ensure_ascii=False) + "\n")
        (self.dir / "steps" / f"step_{step['step_index']:03d}.json").write_text(json.dumps(step, ensure_ascii=False, indent=2), encoding="utf-8")
        row = trajectory_row(step); self.rows.append(row)
        with (self.dir / "trajectory.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    def finish(self, reason, statement=None):
        self.doc['outcome'] = {"outcome_category": OUTCOME_CATEGORY.get(reason, "C-technical"), "termination_reason": reason, "model_stop_statement": statement}
        self.doc['effective_sampling'] = effective_sampling(self.config, self.steps); self.write()
        if self.context_trace is not None:
            self.doc['context_embedding'] = self.context_trace.summary(); self.write()
        if self.memory is not None:
            try: self.memory.finish(self, reason)
            except Exception as error: print(f"[memory] finish failed: {type(error).__name__}: {error}", file=sys.stderr)
        categories = [category for step in self.steps for category in step['semantic_cluster_label']]
        token_sources = [s.get("token_counts") or {} for s in self.steps]
        token_sources += [(s.get("intent") or {}).get("tokens") or {} for s in self.steps if s.get("intent")]
        predictions = (self.doc.get("pre_run_prediction") or {}).get("records") or []
        token_sources += [record.get("tokens") or {} for record in predictions]
        provider_usage = [s.get("extras", {}).get("provider_usage") or {} for s in self.steps]
        provider_usage += [(s.get("intent") or {}).get("provider_usage") or {} for s in self.steps if s.get("intent")]
        readouts = self.doc.get("readouts") or []
        token_sources += [r.get("token_counts") or {} for r in readouts]
        provider_usage += [r.get("provider_usage") or {} for r in readouts]
        provider_usage += [record.get("provider_usage") or {} for record in predictions]
        costs = [usage["cost_usd"] for usage in provider_usage if usage.get("cost_usd") is not None]
        generation_ids = [request_id for usage in provider_usage for request_id in usage.get("generation_ids") or []]
        finished_at = utc_now()
        readouts_valid = sum(bool(r.get("valid")) for r in readouts)
        readout_failed = bool(readouts) and readouts_valid != len(readouts)
        collection = self.doc.get("collection") or {}
        mask = self.doc.get("link_mask") or {}
        summary = {"schema_version": SCHEMA_VERSION, "batch_id": self.dataset.batch_id, "batch_index": self.dataset.cfg.batch.index, "run_id": self.run_id, "run_index": self.doc['run_index'], "variant": self.doc['variant'], "condition": self.doc['condition'], "start_page": self.doc['start_page'], "model": self.config['provider']['model'], "provider": self.config['provider']['name'], "started_at": self.started_at, "finished_at": finished_at, "status": "Technical failure" if self.doc['outcome']['outcome_category'] == "C-technical" or readout_failed else "Completed", "termination_reason": reason, "outcome_category": self.doc['outcome']['outcome_category'], "n_steps": len(self.steps), "n_searches": sum(1 for s in self.steps if s['action'] != "follow"), "unique_pages": len({s['current_page_title'] for s in self.steps}), "distinct_categories": len(set(categories)), "invalid_actions": sum(s['invalid_action']['is_invalid'] for s in self.steps), "intent_steps": sum(1 for s in self.steps if s.get('intent')), "intent_matches": sum(1 for s in self.steps if (s.get('intent') or {}).get('choice_matches_intent')), "pre_run_prediction_mode": (self.doc.get("pre_run_prediction") or {}).get("mode", "none"), "valid_pre_run_predictions": sum(bool(record.get("parse_ok")) for record in predictions), "mean_choice_set_size": round(sum(s['extras']['n_links_offered'] for s in self.steps) / max(1, len(self.steps)), 1), "collection_target": collection.get("target"), "collection_selected": len(collection.get("selections") or []), "collection_complete": bool(collection.get("complete")) if collection else None, "readouts_total": len(readouts), "readouts_valid": readouts_valid, "link_mask_mode": mask.get("mode"), "link_mask_id": mask.get("mask_id"), "link_mask_sha256": mask.get("sha256"), "masked_candidates_total": sum(s.get("extras", {}).get("n_links_masked") or 0 for s in self.steps), "input_tokens": sum(t.get("input") or 0 for t in token_sources), "output_tokens": sum(t.get("output") or 0 for t in token_sources), "reasoning_tokens": sum(t.get("reasoning") or 0 for t in token_sources), "cost_usd": round(sum(costs), 12) if costs else None, "provider_generation_ids": generation_ids, "wall_seconds": round(time.time() - self.t0, 1), "folder": self.dir.name, "artifact_path": str(self.dir.resolve()), "git_commit": self.dataset.data["code"]["git_commit"]}
        for name in ('summary.json', 'metrics.json'): (self.dir / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.rows:
            with (self.dir / "pages.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader(); writer.writerows(self.rows)
        self.transcript(); self.dataset.append_run_log(summary)
        for readout in readouts:
            self.dataset.append_readout_log(self, readout)
        self.dataset.register(self.doc['condition'], self.config['provider']['model']); return summary
    def transcript(self):
        """Human-readable run report: settings, then per step what was shown, what the model said and did."""
        cfg, steps = self.config, self.steps
        exp, prov = cfg.get("experiment", {}), cfg.get("provider", {})
        lines = [f"# {self.run_id}", "",
                 "| Setting | Value |", "|---|---|",
                 f"| Variant | {self.doc['variant']} |",
                 f"| Start page | {self.doc['start_page']} |",
                 f"| Start page came from | {cfg.get('batch', {}).get('start_page_source', 'batch.start_pages')} |",
                 f"| Model | {prov.get('name')} / {prov.get('model')} (temperature {prov.get('temperature')}) |",
                 f"| Action format | {exp.get('action_format', 'text')} |",
                 f"| Memory | {exp.get('memory', 'path_only')} |",
                 f"| Intent | {exp.get('intent', 'none')} |",
                 f"| Link order | {exp.get('display_order_policy')} |",
                 f"| Link mask | {exp.get('link_mask_mode', 'none')} ({exp.get('link_mask_id') or 'no id'}) |",
                 f"| Seed | {cfg.get('run_seed')} |",
                 f"| Outcome | {self.doc['outcome']['termination_reason']} after {len(steps)} steps |", "",
                 "Exact prompts sent to the model: `model_inputs/step_XXX_request*.txt`. Full records: `steps/step_XXX.json`.", "",
                 "## Steps", ""]
        predictions = (self.doc.get("pre_run_prediction") or {}).get("records") or []
        if predictions:
            block = ["## Pre-run predictions", "",
                     "These stateless forecasts were stored separately and were not shown to the navigation agent.", ""]
            for record in predictions:
                block += [f"### {record['framing']}", "",
                          f"- **Parse OK:** {record['parse_ok']}",
                          f"- **Pages:** {record['pages']}",
                          f"- **Topics:** {record['topics']}",
                          f"- **Coarse single-run forecast:** {record.get('coarse_single_run')}",
                          f"- **Memory-effect forecast:** {record.get('memory_effect')}",
                          f"- **Confidence:** {record['confidence']}",
                          f"- **Tokens:** {record['tokens']}", "",
                          "<details><summary>Raw prediction</summary>", "", "```",
                          record.get("raw") or "", "```", "", "</details>", ""]
            lines[-2:-2] = block
        for step in steps:
            ex, intent, search = step["extras"], step.get("intent"), step.get("search")
            options = search["results"] if search and search.get("results") else step["choice_set"]
            shown, original = ex.get("chosen_position"), ex.get("chosen_original_position")
            where = f" (shown at #{shown + 1} of {len(options)}, #{original + 1} on the {'search ranking' if search else 'page'})" if shown is not None else ""
            lines += [f"### Step {step['step_index']}: {step['current_page_title']}", "",
                      f"- **Action:** {step['action']}",
                      f"- **Links offered:** {ex.get('n_links_offered', 0)}",
                      f"- **Links masked:** {ex.get('n_links_masked', 0)}"]
            if search:
                lines.append(f"- **Search query:** \"{search.get('query')}\" → {len(search.get('results') or [])} results")
            lines.append(f"- **Chose:** {step['parsed_choice']}{where}")
            lines.append(f"- **Reasoning:** {step.get('model_reasoning') or '—'}")
            if step.get("model_note"):
                lines.append(f"- **Note:** {step['model_note']}")
            for call in ex.get("tool_calls") or []:
                lines.append(f"- **Tool called:** `{call.get('name')}` with `{json.dumps(call.get('arguments'), ensure_ascii=False)}`")
            if intent:
                position = f"#{intent['intended_shuffled_position'] + 1} in the list" if intent.get("in_links") else "not on this page"
                lines.append(f"- **Intent (asked before the links were shown):** \"{intent.get('text')}\" → maps to *{intent.get('mapped_title') or 'no article'}* "
                             f"({position}; choice matches intent: {intent.get('choice_matches_intent')})")
            if step["invalid_action"].get("is_invalid"):
                lines.append(f"- **Invalid reply:** {step['invalid_action'].get('invalid_type')} → {step['invalid_action'].get('harness_handling')}")
            resolution = step.get("resolution") or {}
            if resolution.get("resolve_status") not in (None, "success"):
                lines.append(f"- **Problem:** {resolution.get('resolve_status')}: {resolution.get('resolve_error')}")
            lines.append(f"- **Tokens:** {step.get('token_counts')}")
            if search and search.get("results"):
                lines += ["", "<details><summary>Search results shown (in order)</summary>", ""]
                lines += [f"{r['shuffled_position'] + 1}. {r['title']} — {r.get('snippet') or ''}" for r in search["results"]]
                lines += ["", "</details>"]
            if step["choice_set"]:
                lines += ["", f"<details><summary>Links shown (in order, {len(step['choice_set'])})</summary>", ""]
                lines += [f"{c['shuffled_position'] + 1}. {c['title']}" for c in step["choice_set"]]
                lines += ["", "</details>"]
            if step.get("model_internal_reasoning"):
                lines += ["", "<details><summary>Model's internal reasoning (its own thinking; never shown back to it)</summary>", "",
                          "```", step["model_internal_reasoning"], "```", "", "</details>"]
            lines += ["", "<details><summary>Raw model output</summary>", "", "```", step.get("model_output_raw") or "", "```", "", "</details>", ""]
        if self.doc.get("collection"):
            lines += ["## Collection", "", "```json", json.dumps(self.doc["collection"], ensure_ascii=False, indent=2), "```", ""]
        lines += ["## Outcome", "", "```json", json.dumps(self.doc["outcome"], ensure_ascii=False, indent=2), "```", "",
                  "<details><summary>Full settings (config)</summary>", "", "```json", json.dumps(cfg, ensure_ascii=False, indent=2), "```", "", "</details>", ""]
        (self.dir / "transcript.md").write_text("\n".join(lines), encoding="utf-8")


def chosen_list(step):
    """The actual choice list: non-empty search results, otherwise page links after a search fallback."""
    search = step.get('search') or {}
    return search.get('results') or step['choice_set']


def trajectory_row(step):
    ex, options = step['extras'], chosen_list(step)
    position = ex['chosen_position']
    searched = step.get('search') or {}
    transition = 'intent' if step['action'] == 'intent_jump' else 'search' if searched.get('results') else 'link'
    return {"schema_version": SCHEMA_VERSION, "run_id": step['run_id'], "step": step['step_index'], "action": step['action'], "search_query": searched.get('query'), "page_id": step['current_page_id'], "title": step['current_page_title'], "created": ex['created'], "modified": ex['modified'], "storage_cluster_id": ex['storage_cluster_id'], "semantic_cluster_label": step['semantic_cluster_label'], "wikitext_chars": ex['wikitext_chars'], "n_actions": len(options), "chosen_key": wiki_key(step['parsed_choice']) if step['parsed_choice'] not in (None, 'STOP') else None, "chosen_title": step['parsed_choice'], "chosen_position": position, "chosen_original_position": ex['chosen_original_position'], "chosen_position_norm": round(position / len(options), 4) if position is not None and options else None, "transition": transition if step['parsed_choice'] not in (None, 'STOP') else ('stop' if step['parsed_choice'] == 'STOP' else None), "thought": step['model_reasoning'] or '', "intent": (step.get('intent') or {}).get('text'), "intent_mapped_title": (step.get('intent') or {}).get('mapped_title'), "intent_in_links": (step.get('intent') or {}).get('in_links'), "choice_matches_intent": (step.get('intent') or {}).get('choice_matches_intent'), "latency_s": ex['latency_s'], "parse_ok": not step['invalid_action']['is_invalid'], "resolve_status": step['resolution']['resolve_status'], "error": step['resolution']['resolve_error'] or '', "menu": options}


# ----------------------------------------------------------------------------- environment
def _portable(project_root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


@BACKENDS.register("fixture")
def _fixture_backend(wcfg, project_root):
    return FixtureBackend(_portable(project_root, wcfg.fixture_path))


@BACKENDS.register("dump")
def _dump_backend(wcfg, project_root):
    return DumpBackend(_portable(project_root, wcfg.dump_path), _portable(project_root, wcfg.cache_dir),
                       wcfg.dump_xml_filename, wcfg.dump_index_filename, wcfg.max_compressed_block_bytes,
                       wcfg.get("lookup_format", "sqlite"), wcfg.get("page_cache_size", 4096))


def make_backend(cfg, project_root):
    """`wikipedia.backend`: a registered name (fixture, dump) or a dotted import path to
    factory(wikipedia_cfg, project_root) returning a WikipediaBackend."""
    return BACKENDS.get(cfg.wikipedia.backend)(cfg.wikipedia, project_root)


def make_search(cfg, backend, project_root):
    """Search-enabled variants B/C/D, or any variant with the intent on (to map intents to articles).
    A backend with iter_titles() (the fixture) gets an in-memory index of its own titles; the dump uses the built index."""
    if variant_of(cfg) == "a_hub" and intent_mode(cfg) == "none":
        return None
    from gap_attractor.search.index import SearchIndex
    if hasattr(backend, "iter_titles"):
        return SearchIndex.from_titles(backend.iter_titles(), f"{backend.backend_name} titles")
    path = Path(cfg.variant.search.index_path if variant_of(cfg) != "a_hub" else cfg.experiment.intent_index_path)
    return SearchIndex.open(path if path.is_absolute() else Path(project_root) / path)


def resolve_results(search, backend, query, scfg, run_seed, step_index):
    """Search, load each hit (redirects resolve to their target), dedupe, cut to top_n, order for display."""
    from gap_attractor.search.index import make_snippet
    hits, mode = search.search(query, int(scfg.candidate_pool))
    results, seen = [], set()
    for hit in hits:
        key = wiki_key(hit["matched_title"])
        if not backend.contains(key) or key.startswith(SKIP_NS):
            continue
        page = backend.load_page(key)
        if page is None or page.key in seen:
            continue
        seen.add(page.key)
        snippet, source = make_snippet(page.wikitext, clean_article_text, scfg.snippet_mode, int(scfg.snippet_sentences), int(scfg.snippet_max_chars))
        results.append({"original_position": len(results), "shuffled_position": None, "title": page.title,
                        "hyperlink": f"https://en.wikipedia.org/wiki/{page.key}", "page_id": page.page_id,
                        "semantic_cluster_label": list(page.categories), "snippet": snippet, "snippet_source": source,
                        "matched_title": hit["matched_title"], "via_redirect": wiki_key(hit["matched_title"]) != page.key})
        if len(results) >= int(scfg.top_n):
            break
    order = list(range(len(results)))
    if scfg.display_order == "seeded_shuffle":
        order, _ = display_keys_for_step(order, "seeded_shuffle", run_seed, 10_000 + step_index)
    shown = [results[i] for i in order]
    for shown_position, result in enumerate(shown):
        result["shuffled_position"] = shown_position
    return shown, mode, len(hits)


# ----------------------------------------------------------------------------- experiment
def read_start_titles(path, column="title"):
    """Titles from a start-page file: .txt has one title per line (# comments skipped); .tsv/.csv have a header
    row and the titles in `column` (e.g. Claire's start_set_*.tsv)."""
    path = Path(path)
    if path.suffix.lower() in (".tsv", ".csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            if column not in (reader.fieldnames or []):
                raise ValueError(f"{path} has no column {column!r}; columns: {reader.fieldnames}")
            return [row[column].strip() for row in reader if (row[column] or "").strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


@START_PAGE_SOURCES.register("search")
def _search_starts(cfg):
    """Search-first variants: no fixed start page; one placeholder label per episode."""
    return [f"search-{i + 1}" for i in range(int(cfg.variant.n_runs))]


@START_PAGE_SOURCES.register("file")
def _file_starts(cfg):
    """A seeded sample of variant.n_runs titles from variant.start_pages_file."""
    titles = read_start_titles(cfg.variant.start_pages_file, cfg.variant.get("start_pages_column") or "title")
    return random.Random(int(cfg.experiment.seed)).sample(titles, min(int(cfg.variant.n_runs), len(titles)))


@START_PAGE_SOURCES.register("batch")
def _batch_starts(cfg):
    return list(cfg.batch.start_pages)


def start_pages_source(cfg):
    """variant.start_pages_source when set (a registered name or a dotted path to factory(cfg) -> titles);
    otherwise: search for a search-first variant, the file when variant.start_pages_file is set, else batch.start_pages."""
    configured = cfg.variant.get("start_pages_source") if cfg.get("variant") else None
    if configured:
        return str(configured)
    if variant_of(cfg) in SEARCH_FIRST_VARIANTS:
        return "search"
    return "file" if cfg.get("variant") and cfg.variant.get("start_pages_file") else "batch"


def start_pages(cfg):
    """One entry per episode. variant.runs_per_start (default 1) repeats each start page that many times in a row;
    every episode still gets its own run seed (experiment.seed + run index)."""
    starts = list(START_PAGE_SOURCES.get(start_pages_source(cfg))(cfg))
    repeats = int(cfg.variant.get("runs_per_start", 1)) if cfg.get("variant") else 1
    if repeats < 1:
        raise ValueError(f"variant.runs_per_start must be >= 1, got {repeats}")
    return [start for start in starts for _ in range(repeats)]


DISPLAY_ORDER_POLICIES = ("source", "seeded_shuffle")
SEARCH_DISPLAY_ORDERS = ("rank", "seeded_shuffle")
SNIPPET_MODES = ("short_description_then_lead", "lead")


def validate_config(cfg):
    """Check every enumerated setting before anything is loaded or written, so a typo never leaves a half-made batch."""
    variant_of(cfg); intent_mode(cfg); action_format(cfg); reveal_budget(cfg); pre_run_prediction_mode(cfg); pre_run_prediction_task(cfg); link_mask_spec(cfg)
    validate_memory = getattr(memory_class(memory_mode(cfg)), "validate", None)
    if validate_memory is not None:
        validate_memory(cfg)
    policy = cfg.experiment.display_order_policy
    if policy not in DISPLAY_ORDER_POLICIES:
        raise ValueError(f"unknown display_order_policy: {policy!r}; expected one of {DISPLAY_ORDER_POLICIES}")
    if not stop_allowed(cfg) and re.search(r"\bstop\b", str(cfg.experiment.system_prompt), re.I):
        raise ValueError("experiment.allow_stop=false, but the system prompt mentions stopping "
                         f"({cfg.experiment.system_prompt!r}); use prompt=free or prompt=browse, or reword it")
    search_cfg = cfg.variant.get("search") if cfg.get("variant") else None
    if search_cfg is not None:
        if search_cfg.get("display_order", "rank") not in SEARCH_DISPLAY_ORDERS:
            raise ValueError(f"unknown variant.search.display_order: {search_cfg.display_order!r}; expected one of {SEARCH_DISPLAY_ORDERS}")
        if search_cfg.get("snippet_mode", SNIPPET_MODES[0]) not in SNIPPET_MODES:
            raise ValueError(f"unknown variant.search.snippet_mode: {search_cfg.snippet_mode!r}; expected one of {SNIPPET_MODES}")
    for registry, name in ((PROVIDERS, cfg.provider.name), (BACKENDS, cfg.wikipedia.backend), (START_PAGE_SOURCES, start_pages_source(cfg))):
        if not registry.is_known(name):
            registry.get(name)  # raises with the list of known names
    index = cfg.batch.index
    if str(index) != "auto" and not str(index).isdigit():
        raise ValueError(f"batch.index must be a non-negative integer or 'auto', got {index!r}")
    specs = readout_specs(cfg)
    if specs:
        count = int(cfg.experiment.get("readout_selection_count", 10))
        if count < 1:
            raise ValueError(f"experiment.readout_selection_count must be >= 1, got {count}")
        ids = [str(spec.get("id") or "") for spec in specs]
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("experiment.readouts must have non-empty unique ids")
        for spec in specs:
            missing = [key for key in ("condition", "tracker_config_id", "instruction") if not str(spec.get(key) or "").strip()]
            if missing:
                raise ValueError(f"readout {spec.get('id')!r} is missing {missing}")


def next_batch_index(experiment_root):
    """1 + the highest batch_NN_ index already under the experiment folder (any day, model or condition)."""
    root = Path(experiment_root)
    found = [int(m.group(1)) for p in (root.iterdir() if root.is_dir() else []) if (m := re.match(r"batch_(\d+)_", p.name))]
    return max(found, default=0) + 1


def _config_differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in set(a) | set(b):
            yield from _config_differences(a.get(key), b.get(key), f"{path}.{key}" if path else key)
    elif a != b:
        yield path


def _set_config(cfg, key, value):
    """Write a resolved value back into cfg (e.g. the batch index chosen by auto/resume), so every log records it."""
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, key, value)


def resume_target(cfg):
    """batch.resume: false (default) | true (today's batch with this batch.index) | a batch folder path or name."""
    value = cfg.batch.get("resume", False)
    if value in (None, False, "false", "False"):
        return None
    experiment_root = Path(cfg.output.directory) / cfg.experiment.name
    if value in (True, "true", "True"):
        if str(cfg.batch.index) == "auto":
            raise ValueError("batch.resume=true needs the batch's own batch.index (or pass the batch folder: batch.resume=<path>)")
        day = datetime.now().strftime("%y%m%d")
        return experiment_root / f"batch_{int(cfg.batch.index):02d}_{day}_{slug(cfg.provider.model, 28)}_{slug(cfg.experiment.condition, 24)}"
    path = Path(str(value))
    return path if path.is_absolute() else experiment_root / path


def create_dataset(cfg, backend, project_root):
    """batch.index=auto picks the next free index; if another process claims it first, the next one is tried.
    The chosen number is written back into cfg, so every log records the actual index.
    With batch.resume, the existing batch is reopened instead (see Dataset.reopen)."""
    target = resume_target(cfg)
    if target is not None:
        return Dataset.reopen(cfg, backend, project_root, target)
    if str(cfg.batch.index) != "auto":
        return Dataset(cfg, backend, project_root)
    index = next_batch_index(Path(cfg.output.directory) / cfg.experiment.name)
    while True:
        _set_config(cfg, "batch.index", index)
        try:
            return Dataset(cfg, backend, project_root)
        except FileExistsError:
            index += 1


def run_experiment(cfg, project_root="."):
    validate_config(cfg)  # fail fast, before anything is loaded or written
    starts = start_pages(cfg)
    source = start_pages_source(cfg)
    from omegaconf import open_dict
    with open_dict(cfg):  # so config.json / batch.json show the start pages actually used, not the default
        if source == "search":
            label = "Variant B" if variant_of(cfg) == "b_search_first" else "Variant D"
            cfg.batch.start_pages, cfg.batch.start_page_source = [], f"search ({label} starts with a search)"
        elif source == "file":
            cfg.batch.start_pages = starts
            cfg.batch.start_page_source = f"variant.start_pages_file: {cfg.variant.start_pages_file} (sampled with seed {cfg.experiment.seed})"
        elif source == "batch":
            cfg.batch.start_page_source = "batch.start_pages"
        else:
            cfg.batch.start_pages, cfg.batch.start_page_source = starts, f"variant.start_pages_source: {source}"
        budget = steps_sentence(int(cfg.experiment.step_limit))
        if reveal_budget(cfg) == "total" and not str(cfg.experiment.system_prompt).endswith(budget):  # a resumed batch has it already
            cfg.experiment.system_prompt = f"{cfg.experiment.system_prompt} {budget}"  # stored verbatim in config.json
    backend, provider = make_backend(cfg, project_root), make_provider(cfg.provider)
    search = make_search(cfg, backend, project_root)
    dataset, results = create_dataset(cfg, backend, project_root), []
    skip = dataset.completed_runs(starts) if resume_target(cfg) is not None else set()
    for run_index, start in enumerate(starts, 1):
        if run_index in skip:
            continue
        run_seed = int(cfg.experiment.seed) + run_index
        results.append(run_episode(dataset, backend, provider, search, run_index, start, run_seed))
    return dataset, results


SIDE_EXHAUSTED_NOTE = ("You have used all {n} turns of {tools} for this step. Now act: call one of the tools you are "
                       "offered.")


class Caller:
    """Calls the model and accumulates tokens/attempts/raw text/reasoning traces/API time for one step.

    With a memory that contributes tools (`memory.side_tools()`, tools mode), those tools are offered next to the
    move tools. Calls to them are run by the memory (an exception becomes an error result) and answered, and the
    model is asked again with the memory's view of the conversation (`memory.continue_messages`). At most
    `memory.side_turns()` replies per STEP may call them (retries included); after that the model is asked once,
    with an explicit note and the move tools only, to act. Side calls made in the same reply as a move are run
    before it. `last_messages` is the request a follow-up (retry, search results) should build on: after memory
    tools ran, their post-edit view. Logged per step: extras.side_calls and extras.side_turns_exhausted.
    A memory may also hold a move back (`memory.hold_move`, e.g. until its context fits a budget): the move call is
    answered with the memory's note, the step's editing turns start again, and the model is asked again on the same
    page (at most MAX_HOLDS times per step). Logged per step: extras.held_moves."""
    MAX_HOLDS = 20
    def __init__(self, provider, cfg, run_seed, memory=None):
        self.provider, self.seed, self.memory = provider, run_seed, memory
        self.temperature, self.top_p = float(cfg.provider.temperature), float(cfg.provider.top_p)
        self.tokens, self.attempts, self.raw, self.retries, self.finish_reason = {"input": 0, "output": 0, "reasoning": 0}, [], [], 0, None
        self.cost_usd, self.cost_seen = 0.0, False
        self.generation_ids, self.response_models = [], []
        self.tool_calls, self.last_call, self.internal = [], None, []
        self.api_seconds, self.error_kind = 0.0, None
        self.last_messages, self.side_calls, self.side_used, self.side_offered, self.side_exhausted = None, [], 0, False, False
        self.last_reply, self.failures = None, []        # the latest reply as written; this step's unusable replies
        self.held = []                                    # moves the memory held back in this step (hold_move)
        self.sink = None                                  # the step sets it: writes the failures (failures.jsonl)
        self.embed = None                                 # context_embedding on: the step sets it (embeds last_messages)
        self.save = lambda name, messages: None       # the step sets it to save each extra request to model_inputs/
    def __call__(self, messages, tools=None, tool_choice=None):
        while True:
            response = self._call(messages, tools, tool_choice)
            again = self._hold(response)
            if again is None:
                return response
            messages = again
    def _hold(self, response):
        """The request to ask again with when the memory holds this reply's move back, else None."""
        if response.error or self.memory is None or self.last_call is None or len(self.held) >= self.MAX_HOLDS:
            return None
        try:
            note = self.memory.hold_move(self.last_call)
        except Exception:            # a broken hook must not end the batch: the move goes ahead
            note = None
        if not note:
            return None
        call = self.last_call
        self.held.append({"call": {"name": call["name"], "arguments": call["arguments"]}, "note": str(note)})
        self.side_used, self.side_exhausted = 0, False
        messages = self.memory.continue_messages(self.last_messages, [call], [str(note)])
        self.save(f"request_held_{len(self.held)}", messages)
        return messages
    def _call(self, messages, tools=None, tool_choice=None):
        side = self.memory.side_tools() if (tools and self.memory is not None and tool_choice != "none") else []
        limit = max(0, int(self.memory.side_turns())) if side else 0
        if not side or self.side_used >= limit:
            return self._request(messages, tools, tool_choice)
        self.side_offered = True
        names = {t["function"]["name"] for t in side}
        while True:
            response = self._request(messages, list(tools) + side, tool_choice, side_names=names)
            calls = [c for c in (getattr(response, "tool_calls", None) or []) if c["name"] in names]
            if response.error or not calls:
                return response
            self.side_used += 1
            results = [self._handle(call) for call in calls]
            self.side_calls.append({"turn": self.side_used, "calls": calls, "results": results, "with_move": self.last_call is not None})
            messages = self.memory.continue_messages(messages, calls, results)
            self.save(f"request_side_{self.side_used}", messages)
            if self.last_call is not None:                   # a move came with the side calls: follow-ups build on
                self.last_messages = messages                # the post-edit view
                return response
            if self.side_used >= limit:                      # no editing turns left: ask once for a move
                self.side_exhausted = True
                note = self.memory.side_exhausted_note(limit, sorted(names)) if hasattr(self.memory, "side_exhausted_note") else None
                messages = messages + [{"role": "user", "content": note or SIDE_EXHAUSTED_NOTE.format(n=limit, tools=", ".join(sorted(names)))}]
                self.save("request_side_exhausted", messages)
                return self._request(messages, tools, tool_choice)
    def _handle(self, call):
        try:
            return str(self.memory.handle(call))
        except Exception as error:   # a broken memory tool must not end the batch
            return f"ERROR: {call.get('name')} failed ({type(error).__name__}: {str(error)[:300]}). Nothing was changed."
    def _request(self, messages, tools=None, tool_choice=None, side_names=()):
        self.last_messages = messages
        kwargs = {"tools": tools} if tools else {}
        if tools and tool_choice: kwargs["tool_choice"] = tool_choice
        response = self.provider.chat(messages, seed=self.seed, temperature=self.temperature, top_p=self.top_p, **kwargs)
        self.last_reply = {"role": "assistant", "content": response.text or "", "tool_calls": list(getattr(response, "tool_calls", None) or []),
                           "reasoning": getattr(response, "internal_reasoning", None)}
        if self.memory is not None and not response.error:   # the reply as the model wrote it, before any parsing
            self.memory.observe(messages, self.last_reply)
        self.last_call = None
        if tools and getattr(response, "tool_calls", None):
            self.tool_calls += response.tool_calls
            moves = [c for c in response.tool_calls if c["name"] not in side_names]
            if moves or not side_names:
                self.last_call = (moves or response.tool_calls)[0]  # one action per turn; any extra calls are logged, not executed
                response = replace(response, text=tool_call_as_text(self.last_call))
            else:  # only memory tools: no move in this reply (never parsed; kept as the raw output)
                response = replace(response, text=json.dumps({"memory_tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in response.tool_calls],
                                                              "text": response.text or ""}, ensure_ascii=False))
        usage = response.usage or {}
        if self.memory is not None and not response.error:
            try:
                self.memory.observe_call(messages, kwargs.get("tools"), usage)
            except Exception:        # a broken hook must not end the batch
                pass
        for key, value in (("input", usage.get("prompt_tokens")), ("output", usage.get("completion_tokens")), ("reasoning", (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))):
            self.tokens[key] += value or 0
        if isinstance(usage.get("cost"), (int, float)):
            self.cost_usd += float(usage["cost"]); self.cost_seen = True
        if response.response_id:
            self.generation_ids.append(response.response_id)
        if response.response_model:
            self.response_models.append(response.response_model)
        self.attempts += response.attempts; self.retries += response.retries; self.raw.append(response.text or ""); self.finish_reason = response.finish_reason
        if getattr(response, "internal_reasoning", None): self.internal.append(response.internal_reasoning)
        self.api_seconds += getattr(response, "latency_s", 0.0) or 0.0
        self.error_kind = getattr(response, "error_kind", None) if response.error else None
        return response
    def provider_usage(self):
        return {"cost_usd": round(self.cost_usd, 12) if self.cost_seen else None,
                "generation_ids": list(self.generation_ids), "response_models": list(self.response_models)}
    def note_failure(self, error, response, available) -> None:
        """An unusable reply (kept for failures.jsonl, written when the step closes)."""
        self.failures.append({"attempt": len(self.failures), "invalid_type": error, "attempted": attempted_choice(response.text) or None,
                              "menu": list(available), "reply": dict(self.last_reply or {}), "cut_off": response.finish_reason == "length"})
    def base(self, messages):
        """The request to build a follow-up on: the one the latest reply answered (after any memory tool turns)."""
        return self.last_messages if self.last_messages is not None else messages
    def internal_reasoning(self):
        """The model's own thinking in this step's calls (same separator as model_output_raw), or None if none was returned."""
        return "\n---next call---\n".join(self.internal) if self.internal else None
    def termination(self):
        """How an API failure ends the run: context_limit when the request was too large for the model, else api_error."""
        return "context_limit" if self.error_kind == "context_limit" else "api_error"


def new_step(run, step_index, page, choice_set, cfg, shuffle_seed, visited, action="follow"):
    return {"run_id": run.run_id, "step_index": step_index, "timestamp": utc_now(), "dwell_time": None, "action": action,
            "current_page_id": page.page_id if page else None, "current_page_title": page.title if page else "[search]",
            "choice_set": choice_set, "search": None, "intent": None,
            "model_reasoning": None, "model_internal_reasoning": None, "model_note": None, "model_output_raw": None, "parsed_choice": None,
            "invalid_action": {"is_invalid": False, "invalid_type": None, "harness_handling": None},
            "token_counts": {"input": None, "output": None, "reasoning": None},
            "resolution": {"resolve_status": None, "resolve_error": None, "resolved_page_id": None},
            "extras": {"created": page.created if page else None, "modified": page.modified if page else None,
                       "storage_cluster_id": page.storage_cluster_id if page else None, "wikitext_chars": page.wikitext_chars if page else None,
                       "n_links_total": len(page.links) if page else 0, "n_links_offered": len(choice_set),
                       "chosen_position": None, "chosen_original_position": None, "revisit": bool(page) and page.key in visited[:-1],
                       "latency_s": None, "retry_count": 0, "finish_reason": None, "retry_attempts": [],
                       "display_order_policy": cfg.experiment.display_order_policy, "display_shuffle_seed": shuffle_seed,
                       "article_content_was_truncated": False, "tool_order": None,
                       "action_format": action_format(cfg), "tool_calls": [],
                       "provider_usage": {"cost_usd": None, "generation_ids": [], "response_models": []},
                       "candidate_filter_provenance": {"membership_rule": "link resolves (redirects followed) to a loadable page in the selected backend; one row per destination, first link kept; links to the current page dropped", "namespace_exclusions": list(SKIP_NS)}},
            "semantic_cluster_label": list(page.categories) if page else []}


def context_trace(cfg, run):
    """The run's context-embedding trace when context_embedding.enabled, else None. Imported only then: it needs the
    analysis extra (numpy) and calls an embedding API."""
    if not bool((cfg.get("context_embedding") or {}).get("enabled", False)):
        return None
    from gap_attractor.harness.context_embed import make_trace
    return make_trace(cfg, run.dir, (run.doc.get('world') or {}).get('agent_id'))


def embed_hook(run):
    """close_step's embedding call for this run (None when context_embedding is off)."""
    if run.context_trace is None:
        return None
    return lambda step, caller: run.context_trace.record_request(step['step_index'], caller.last_messages, caller.memory)


def close_step(step, caller, started):
    if caller.failures and caller.sink is not None:
        caller.sink(step, caller.failures)
    if caller.embed is not None:
        caller.embed(step, caller)
    """latency_s = time inside HTTP requests to the model (no backoff waits, no harness work);
    dwell_time = the whole step, including page loading and link resolution."""
    step["model_output_raw"] = "\n---next call---\n".join(caller.raw) if caller.raw else None
    step["model_internal_reasoning"] = caller.internal_reasoning()
    step["token_counts"] = dict(caller.tokens)
    step["extras"].update(latency_s=round(caller.api_seconds, 3), retry_count=caller.retries, finish_reason=caller.finish_reason, retry_attempts=caller.attempts, tool_calls=list(caller.tool_calls), provider_usage=caller.provider_usage())
    if caller.side_offered:
        step["extras"].update(side_calls=caller.side_calls, side_turns_exhausted=caller.side_exhausted)
    if caller.held:
        step["extras"]["held_moves"] = caller.held
    step["dwell_time"] = round(time.time() - started, 3)


def attempted_choice(text):
    """What the reply put in "choice", for the feedback message."""
    data = _json_object((text or "").strip()) or {}
    choice = data.get("choice")
    if choice is None:
        match = re.search(r'"choice"\s*:\s*"([^"]+)"', text or "")
        choice = match.group(1) if match else ""
    return str(choice)[:200]


TOOLS_RETRY_FEEDBACK = {
    "unparseable": "Your reply did not call one of the available tools. {retry}",
    "out-of-range": '"{choice}" is not one of the listed titles. Call follow_link with a title exactly as it is listed.',
    "refusal": "Your reply did not choose a page. {retry}",
}


def retry_feedback(cfg, error, text):
    """Feedback on an unusable reply: experiment.retry_feedback in text mode, tools.retry_feedback in tools mode."""
    if action_format(cfg) == "tools":
        templates = {**TOOLS_RETRY_FEEDBACK, **dict(cfg.tools.get("retry_feedback") or {})}
    else:
        templates = {**RETRY_FEEDBACK, **dict(cfg.experiment.get("retry_feedback") or {})}
    retry = cfg.tools.get("retry_prompt", "") if cfg.get("tools") else ""
    message = str(templates.get(error) or templates["unparseable"]).format(choice=attempted_choice(text), retry=retry)
    return message if stop_allowed(cfg) else _OR_STOP.sub("", stop_text(cfg, message))


def unreadable(notes):
    """True if a failed reply said nothing (no text, no tool call) or was cut off at the token limit: a technical
    failure, not the model's choice. A streak of steps with no listed page chosen ends as parse_error if any of its
    failed replies is unreadable, else as no_valid_move (prose, refusals, unlisted pages: the model's own behaviour)."""
    return any(n.get("cut_off") or not ((n["reply"].get("content") or "").strip() or n["reply"].get("tool_calls")) for n in notes)


def retry_or_pick(cfg, caller, messages, failed, available, first_error, first_reasoning, run_seed, step_index, step, save,
                  tools=None, allow_random_fallback=True, retry_instruction=""):
    """After an unusable reply: show the model its reply (its tool call, in tools mode) and what was wrong, and ask
    again (experiment.parse_retries times, default 1). A readable answer (a listed title or STOP) is used and logged as
    'retry'. If every retry fails, a seeded random link is normally opened, logged as 'skip', and later shown to the
    model as a harness note, never as its reply. Search-result selection disables that fallback because a harness-picked
    result would contaminate the search trajectory; the run terminates as a technical parse failure instead.
    `failed` is the unusable ProviderResponse.
    Returns (title | 'STOP', reasoning, parse_failed, api_error_text)."""
    error = first_error
    caller.note_failure(first_error, failed, available)
    for attempt in range(max(1, int(cfg.experiment.get("parse_retries", 1)))):
        feedback = retry_feedback(cfg, error, failed.text)
        if retry_instruction:
            feedback += "\n\n" + retry_instruction
        messages = caller.base(messages) + followup(cfg, caller, failed, feedback)
        save("request_retry" if attempt == 0 else f"request_retry_{attempt + 1}", messages)
        retry = caller(messages, tools)
        caller.retries += 1
        if retry.error:
            return None, None, False, retry.error
        title, reasoning, retry_error = parse_reply(retry.text, available, stop_allowed(cfg))
        if title is not None:
            step["invalid_action"] = {"is_invalid": True, "invalid_type": first_error, "harness_handling": "retry"}
            return title, reasoning or first_reasoning, True, None
        error, failed = retry_error, retry
        caller.note_failure(retry_error, retry, available)
    if not allow_random_fallback:
        step["invalid_action"] = {"is_invalid": True, "invalid_type": first_error or error, "harness_handling": "terminate"}
        return None, first_reasoning, True, None
    step["invalid_action"] = {"is_invalid": True, "invalid_type": first_error or error, "harness_handling": "skip"}
    return random.Random(run_seed + step_index).choice(list(available)), first_reasoning, True, None


def choose(cfg, caller, messages, options, run_seed, step_index, step, save, tools=None, allow_random_fallback=True,
           retry_instruction=""):
    """Ask for a choice among `options` (rows with 'title').
    Returns (title | 'STOP', reasoning, parse_failed, api_error_text)."""
    available = {row["title"]: row for row in options}
    response = caller(messages, tools)
    if response.error:
        return None, None, False, response.error
    title, reasoning, error = parse_reply(response.text, available, stop_allowed(cfg))
    if title is not None:
        return title, reasoning, False, None
    if error == "unparseable" and caller.side_exhausted:   # asked to act after its memory-tool turns, and did not
        error = "side-turns-exhausted"
    return retry_or_pick(cfg, caller, messages, response, available, error, reasoning, run_seed, step_index, step, save,
                         tools, allow_random_fallback, retry_instruction)


def search_start(run, cfg, backend, search, caller, run_seed, save_inputs):
    """Search-first step 0: the model searches (up to max_queries times), then opens one result."""
    started, p = time.time(), cfg.variant.prompts
    step = new_step(run, 0, None, [], cfg, None, [], action="search_start")
    save = (lambda name, messages: save_model_inputs(run.dir, 0, name, messages)) if save_inputs else (lambda *a: None)
    caller.sink = lambda step, notes: failure_log.write(run, backend, None, step, notes)
    caller.embed = embed_hook(run)
    tools_mode = action_format(cfg) == "tools"
    search_tools = tool_specs(cfg, ["search", "stop"]) if tools_mode else None
    choose_tools = tool_specs(cfg, ["follow_link", "stop"]) if tools_mode else None
    rule = cfg.tools.rule if tools_mode else p.search_rule
    messages = [{"role": "system", "content": cfg.experiment.system_prompt + "\n\n" + rule},
                {"role": "user", "content": p.search_prompt}]
    attempts, results, query, mode, n_hits = [], [], None, None, 0
    for attempt in range(int(cfg.variant.search.max_queries)):
        save(f"request_search_{attempt + 1}", messages)
        response = caller(messages, search_tools)
        if response.error:
            step["invalid_action"] = {"is_invalid": True, "invalid_type": "other", "harness_handling": "terminate"}
            step["resolution"].update(resolve_status="fetch_failure", resolve_error=response.error)
            close_step(step, caller, started); run.log(step); return None, None, caller.termination()
        query, reasoning = parse_query(response.text)
        if query is None:
            attempts.append({"query": None, "parse_ok": False, "n_results": 0})
            messages = caller.base(messages) + [{"role": "assistant", "content": response.text or ""},
                                   {"role": "user", "content": cfg.tools.retry_prompt if tools_mode else p.query_retry}]; continue
        results, mode, n_hits = resolve_results(search, backend, query, cfg.variant.search, run_seed, 0)
        attempts.append({"query": query, "reasoning": reasoning, "parse_ok": True, "fts_mode": mode, "n_hits": n_hits, "n_results": len(results)})
        if results: break
        messages = caller.base(messages) + followup(cfg, caller, response, p.no_results.format(query=query))
    step["search"] = {"query": query, "attempts": attempts, "fts_mode": mode, "n_hits": n_hits, "results": results}
    if not results:
        step["resolution"].update(resolve_status="fetch_failure", resolve_error="no search results")
        close_step(step, caller, started); run.log(step); return None, None, "dead_end"
    choose_instruction = search_result_instruction(cfg)
    messages = caller.base(messages) + followup(cfg, caller, response, results_view(query, results), choose_instruction)
    save("request_choose", messages)
    title, reasoning, parse_failed, error = choose(
        cfg, caller, messages, results, run_seed, 0, step, save, choose_tools, allow_random_fallback=False,
        retry_instruction=choose_instruction)
    step["model_reasoning"] = reasoning or None
    if error:
        step["invalid_action"] = {"is_invalid": True, "invalid_type": "other", "harness_handling": "terminate"}
        step["resolution"].update(resolve_status="fetch_failure", resolve_error=error)
        close_step(step, caller, started); run.log(step); return None, None, caller.termination()
    if title is None and step["invalid_action"].get("harness_handling") == "terminate":
        step["resolution"].update(resolve_status="parse_failure",
                                  resolve_error="invalid search-result choice after retry; no random fallback")
        close_step(step, caller, started); run.log(step); return None, None, "parse_error"
    step["parsed_choice"] = title
    if title == "STOP":
        step["resolution"]["resolve_status"] = "success"; close_step(step, caller, started); run.log(step); return None, reasoning, "model_stopped"
    chosen = next(r for r in results if r["title"] == title)
    page = backend.load_page(title)
    step["extras"].update(chosen_position=chosen["shuffled_position"], chosen_original_position=chosen["original_position"])
    unresolved = parse_failed and step["invalid_action"].get("harness_handling") == "skip"
    step["resolution"] = {"resolve_status": "parse_failure" if unresolved else ("success" if page else "fetch_failure"),
                          "resolve_error": None if page else "search result could not be loaded", "resolved_page_id": page.page_id if page else None}
    close_step(step, caller, started); run.log(step)
    item = {"reasoning": reasoning, "choice": title, "page": None, "observation": None, "search_query": query,
            "harness_pick": step["invalid_action"]["harness_handling"] == "skip",
            "results_text": results_view(query, results), "note": reply_note(caller) if memory_wants_note(cfg) else None}
    step["model_note"] = item["note"]
    return page, item, None if page else "dead_end"


def run_episode(dataset, backend, provider, search, run_index, start_page, run_seed, memory=None, run_extra=None):
    """One run, start to end. A multi-agent scheduler drives `episode` step by step instead."""
    steps = episode(dataset, backend, provider, search, run_index, start_page, run_seed, memory, run_extra)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def episode(dataset, backend, provider, search, run_index, start_page, run_seed, memory=None, run_extra=None):
    """One run as a generator. The first next() sets up (loads the start page, or runs a search-first opening) and
    yields None; every further next() takes one step and yields its record (the final step too); the call after the
    final step makes no model call and returns the run's summary as the StopIteration value. A search-first opening
    search (step 0) is part of the setup: it is logged (run.steps[0]) but not yielded.
    `memory`: this agent's memory object (default: a new one from experiment.memory); `run_extra`: extra run.json
    fields (e.g. a world)."""
    cfg, variant = dataset.cfg, variant_of(dataset.cfg)
    config = resolved_config(cfg); config['run_seed'] = run_seed; git = git_provenance(); config['harness_git_commit'] = git['git_commit']; config['harness_git_dirty'] = git['git_dirty']
    config['wikipedia'].update(backend.provenance())
    if search is not None: config.setdefault('search_index', {}).update(search.provenance())
    run = Run(dataset, run_index, start_page, config, run_extra)
    memory = memory if memory is not None else make_memory(cfg)
    run.memory = memory; memory.start(run)
    run.context_trace = context_trace(cfg, run)
    settings = memory.settings()
    mask_spec = link_mask_spec(cfg)
    semantic_topic_map = None
    if mask_spec["mode"] in SEMANTIC_LINK_MASK_MODES:
        semantic_topic_map = _semantic_topic_map(mask_spec["semantic_resources"])
        if semantic_topic_map.hash() != mask_spec["semantic_topic_hash"]:
            raise ValueError("semantic link mask resources changed after configuration validation")
    run.doc['link_mask'] = mask_spec
    if settings: run.doc['memory_settings'] = settings
    run.write()
    save_inputs = bool(cfg.output.get("save_model_inputs", True))
    max_chars = cfg.experiment.article_max_chars
    history = []

    if variant in SEARCH_FIRST_VARIANTS:
        page, first_choice, early_end = search_start(run, cfg, backend, search, Caller(provider, cfg, run_seed, memory), run_seed, save_inputs)
        if early_end: return run.finish(early_end, first_choice)
        history.append(first_choice); memory.record(first_choice, run.steps[-1])
    else:
        page = backend.load_page(start_page)
        if page is None: return run.finish("dead_end")
    run_pre_run_predictions(cfg, provider, run, page, run_seed, save_inputs)
    yield None

    visited, visited_categories = [page.key], [list(page.categories)]
    bad_streak, bad_unreadable, termination, stop_statement = 0, False, "step_limit", None
    intent = intent_mode(cfg)
    for step_index in range(1, int(cfg.experiment.step_limit) + 1):
        ended = memory.should_end()
        if ended:
            termination = str(ended); break
        started, caller = time.time(), Caller(provider, cfg, run_seed, memory)
        choice_set, shuffle_seed, link_counts = build_choice_set(
            page, backend, cfg.experiment.display_order_policy, run_seed, step_index,
            mask_spec=mask_spec, visited_keys=visited, visited_categories=visited_categories,
            topic_map=semantic_topic_map,
        )
        step = new_step(run, step_index, page, choice_set, cfg, shuffle_seed, visited)
        mask_record = link_counts.pop("link_mask")
        step['extras'].update(link_counts)
        step['extras']['candidate_filter_provenance']['link_mask'] = mask_record
        titles = [c["title"] for c in choice_set]
        save = (lambda name, messages, s=step_index: save_model_inputs(run.dir, s, name, messages)) if save_inputs else (lambda *a: None)
        caller.save = save
        caller.sink = lambda step, notes, page=page: failure_log.write(run, backend, page, step, notes)
        caller.embed = embed_hook(run)
        if save_inputs: save_model_inputs(run.dir, step_index, "request", None, page=page, titles=titles)
        if not choice_set and variant not in SEARCH_ANYTIME_VARIANTS:
            step['resolution'].update(resolve_status="fetch_failure", resolve_error="no in-dump links on page")
            close_step(step, caller, started); run.log(step); termination = "dead_end"; break

        max_chars_int = None if max_chars is None else int(max_chars)
        if intent != "none":  # the intent call comes first, from the same state, without links
            record, intent_error, intent_termination = run_intent_call(cfg, provider, run_seed, history, page, max_chars_int, save, search, backend, choice_set, visited, memory)
            if intent_error:
                step['invalid_action'] = {"is_invalid": True, "invalid_type": "other", "harness_handling": "terminate"}
                step['resolution'].update(resolve_status="fetch_failure", resolve_error=intent_error)
                close_step(step, caller, started); run.log(step); termination = intent_termination; break
            step['intent'] = record
            if intent == "walk":  # go wherever the intent points: no link list, no choice call
                step['action'] = "intent_jump"; step['choice_set'] = []; step['extras']['n_links_offered'] = 0
                next_page = backend.load_page(record['mapped_title']) if record['mapped_title'] else None
                step['parsed_choice'] = record['mapped_title']; step['model_reasoning'] = record['why']
                step['resolution'] = {"resolve_status": "success" if next_page else "fetch_failure",
                                      "resolve_error": None if next_page else ("intent could not be mapped to an article" if record['parse_ok'] else "unparseable intent"),
                                      "resolved_page_id": next_page.page_id if next_page else None}
                step['token_counts'] = dict(record['tokens']); step['model_output_raw'] = record['raw']
                step['model_internal_reasoning'] = record.get('internal_reasoning')
                step['extras'].update(latency_s=record['latency_s'], provider_usage=record['provider_usage'])
                step['dwell_time'] = round(time.time() - started, 3)
                run.log(step)
                history.append({"reasoning": f"intent: {record['text']}", "choice": record['mapped_title'], "page": page,
                                "observation": page_without_links(page, max_chars_int), "note": None, "search_query": None, "results_text": None})
                memory.record(history[-1], step)
                if next_page is None: termination = "dead_end"; break
                page = next_page; visited.append(page.key); visited_categories.append(list(page.categories))
                run.yielded = step; yield step; continue

        observation, was_truncated = page_view(page, choice_set, max_chars_int)
        if reveal_budget(cfg) == "remaining":
            observation += "\n\n" + steps_sentence(int(cfg.experiment.step_limit) - step_index + 1, more=True)
        step['extras']['article_content_was_truncated'] = was_truncated
        tools_mode = action_format(cfg) == "tools"
        link_tools = tool_specs(cfg, ["follow_link", "stop"]) if tools_mode else None
        if variant in SEARCH_ANYTIME_VARIANTS:
            order = ["follow", "search"] if (run_seed + step_index) % 2 == 0 else ["search", "follow"]
            step['extras']['tool_order'] = order
            action_tools = tool_specs(cfg, [TOOL_FOR_ACTION[a] for a in order] + ["stop"]) if tools_mode else None
            rule = cfg.tools.rule if tools_mode else action_rule(cfg, order)
        else:
            rule = cfg.tools.rule if tools_mode else stop_text(cfg, cfg.experiment.response_rule)
        messages = memory.messages(rule, history, observation)
        save("request", messages)

        options, error, parse_failed = choice_set, None, False
        if variant in SEARCH_ANYTIME_VARIANTS:
            response = caller(messages, action_tools)
            if response.error:
                error = response.error
            elif parse_action(response.text) == "search":
                query, query_reasoning = parse_query(response.text)
                results, mode, n_hits = resolve_results(search, backend, query or "", cfg.variant.search, run_seed, step_index) if query else ([], "unparseable", 0)
                results, search_mask = apply_search_result_mask(results, mask_spec, visited, run_seed, step_index)
                if search_mask is not None:
                    step['extras']['candidate_filter_provenance']['search_mask'] = search_mask
                    step['extras'].update(
                        n_search_results_eligible_before_mask=len(results) + search_mask["masked_count"],
                        n_search_results_masked=search_mask["masked_count"],
                    )
                step['action'] = "search"
                step['search'] = {"query": query, "reasoning": query_reasoning, "fts_mode": mode, "n_hits": n_hits, "results": results}
                if results:
                    options = results
                    choose_instruction = search_result_instruction(cfg)
                    messages = caller.base(messages) + followup(
                        cfg, caller, response, results_view(query, results), choose_instruction)
                    save("request_choose", messages)
                    title, reasoning, parse_failed, error = choose(
                        cfg, caller, messages, options, run_seed, step_index, step, save, link_tools,
                        allow_random_fallback=False, retry_instruction=choose_instruction)
                else:  # nothing found: fall back to the page's links, told so explicitly
                    no_results = str(cfg.variant.prompts.no_results_follow).replace("{query}", str(query))
                    messages = caller.base(messages) + followup(cfg, caller, response, no_results)
                    save("request_choose", messages)
                    title, reasoning, parse_failed, error = choose(cfg, caller, messages, options, run_seed, step_index, step, save, link_tools)
            elif not choice_set:  # chose to follow, but this page has no links to follow
                title, reasoning = None, None
                step['resolution'].update(resolve_status="fetch_failure", resolve_error="no in-dump links on page")
            else:
                available = {c["title"]: c for c in choice_set}
                title, reasoning, first_error = parse_reply(response.text, available, stop_allowed(cfg))
                if title is None:  # unreadable follow reply: same retry policy as A
                    title, reasoning, parse_failed, error = retry_or_pick(cfg, caller, messages, response, available, first_error, reasoning, run_seed, step_index, step, save, link_tools)
        else:
            title, reasoning, parse_failed, error = choose(cfg, caller, messages, options, run_seed, step_index, step, save, link_tools)

        if error:
            step['invalid_action'] = {"is_invalid": True, "invalid_type": "other", "harness_handling": "terminate"}
            step['resolution'].update(resolve_status="fetch_failure", resolve_error=error)
            close_step(step, caller, started); run.log(step); termination = caller.termination(); break
        if title is None and step["invalid_action"].get("harness_handling") == "terminate":
            step['resolution'].update(resolve_status="parse_failure",
                                      resolve_error="invalid search-result choice after retry; no random fallback")
            close_step(step, caller, started); run.log(step); termination = "parse_error"; break
        if title is None:  # a search-anytime variant chose to follow on a page without links
            close_step(step, caller, started); run.log(step); termination = "dead_end"; break
        step['model_reasoning'] = reasoning or None
        step['parsed_choice'] = title
        if title == "STOP":
            step['resolution']['resolve_status'] = "success"
            close_step(step, caller, started); run.log(step); termination, stop_statement = "model_stopped", reasoning; break
        unresolved = parse_failed and step['invalid_action'].get('harness_handling') == "skip"  # retry also failed -> random pick
        bad_streak = bad_streak + 1 if unresolved else 0
        bad_unreadable = (bad_unreadable or unreadable(caller.failures)) if unresolved else False
        chosen = next(row for row in options if row["title"] == title)
        next_page = backend.load_page(title)
        step['extras'].update(chosen_position=chosen["shuffled_position"], chosen_original_position=chosen["original_position"])
        if step['intent'] is not None:
            step['intent']['choice_matches_intent'] = step['intent']['mapped_title'] is not None and (
                title == step['intent']['mapped_title'] or chosen["page_id"] == step['intent']['mapped_page_id'])
        status, resolution_error = ("parse_failure", "unparseable after retry; random pick") if unresolved else (("fetch_failure", "page could not be retrieved") if next_page is None else ("success", None))
        step['resolution'] = {"resolve_status": status, "resolve_error": resolution_error, "resolved_page_id": next_page.page_id if next_page else None}
        close_step(step, caller, started)
        note = reply_note(caller) if memory_wants_note(cfg) else None
        step['model_note'] = note
        searched = step.get('search')  # (not `search`: that name is the search index)
        run.log(step)
        history.append({"reasoning": step['model_reasoning'], "choice": title, "page": page, "observation": observation, "note": note,
                        "harness_pick": step['invalid_action']['harness_handling'] == "skip",
                        "search_query": searched['query'] if searched else None,
                        "results_text": results_view(searched['query'], searched['results']) if searched and searched['results'] else None})
        memory.record(history[-1], step)
        if bad_streak >= 3: termination = "parse_error" if bad_unreadable else "no_valid_move"; break
        if next_page is None: termination = "dead_end"; break
        page = next_page; visited.append(page.key); visited_categories.append(list(page.categories))
        run.yielded = step; yield step
    if termination == "step_limit" and (ended := memory.should_end()):
        termination = str(ended)   # the target may have been reached on the final allowed step
    if run.steps and run.steps[-1]["step_index"] > 0 and run.steps[-1] is not run.yielded:
        yield run.steps[-1]   # the step that ended the run (STOP, dead end, error) is yielded like any other
    if termination not in ("api_error", "parse_error", "context_limit"):
        run_post_run_readouts(run, cfg, provider, run_seed, save_inputs)
    return run.finish(termination, stop_statement)
