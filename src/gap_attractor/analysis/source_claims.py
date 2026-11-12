"""Validate structured source claims against the pages actually opened in a run."""
import json

from gap_attractor.wikipedia import wiki_key


def parse_stop_sources(statement):
    try:
        payload = json.loads((statement or "").strip())
    except (TypeError, json.JSONDecodeError):
        raise ValueError("model_stop_statement is not a JSON object")
    if not isinstance(payload, dict):
        raise ValueError("model_stop_statement is not a JSON object")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources or any(not isinstance(title, str) or not title.strip() for title in sources):
        raise ValueError("model_stop_statement.sources must be a non-empty list of titles")
    return [title.strip() for title in sources]


def opened_titles(run_doc, steps):
    titles = []
    if run_doc.get("start_page"):
        titles.append(run_doc["start_page"])
    for step in steps:
        if step.get("current_page_title"):
            titles.append(step["current_page_title"])
        choice = step.get("parsed_choice")
        if choice not in (None, "STOP") and (step.get("resolution") or {}).get("resolve_status") == "success":
            titles.append(choice)
    return titles


def validate_claimed_sources(run_doc, steps):
    claims = parse_stop_sources((run_doc.get("outcome") or {}).get("model_stop_statement"))
    opened = opened_titles(run_doc, steps)
    opened_keys = {wiki_key(title) for title in opened}
    missing = [title for title in claims if wiki_key(title) not in opened_keys]
    return claims, opened, missing
