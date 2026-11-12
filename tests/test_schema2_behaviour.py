"""Schema 2.0 behaviour: link resolution, honest retries, reasoning traces, API errors, latency, resume, provenance,
and the SQLite title lookup."""
import bz2
import csv
import json
import sys
import types
from pathlib import Path

import pytest

from gap_attractor.analysis.topics import TopicMap
from gap_attractor.harness.runner import apply_search_result_mask, build_choice_set, run_experiment
from gap_attractor.providers import ProviderResponse, base
from gap_attractor.providers.base import OpenAICompatibleProvider
from gap_attractor.wikipedia import DumpBackend, FixtureBackend, wiki_key
from gap_attractor.wikipedia.backend import PickleLookup, SqliteLookup

from test_fixture_harness import ROOT, config

LINK_CASES = (ROOT / "tests/fixtures/wiki/link_cases.json").as_posix()


def reply(payload, **extra):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ProviderResponse(text, {"prompt_tokens": 1, "completion_tokens": 1}, "stop", 0, None, [], **extra)


class Scripted:
    """A provider that plays back replies (a callable gets the messages) and records every request."""
    def __init__(self, replies):
        self.replies, self.requests, self.calls = list(replies), [], []

    def chat(self, messages, *, seed, **kwargs):
        self.requests.append(messages)
        self.calls.append({"seed": seed, **kwargs})
        item = self.replies.pop(0) if self.replies else reply({"reasoning": "done", "choice": "STOP"})
        return item(messages) if callable(item) else item


def run_with(tmp_path, provider, extra=(), variant="a_hub"):
    module = types.ModuleType("scripted_provider")
    module.make = lambda cfg: provider
    sys.modules["scripted_provider"] = module
    cfg = config(tmp_path, variant=variant, extra=["provider.name=scripted_provider.make", *extra])
    dataset, results = run_experiment(cfg, ROOT)
    run_dir = next(p for p in dataset.root.iterdir() if (p / "steps.jsonl").exists())
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    return dataset, results, steps, run_dir


# --------------------------------------------------------------------------- 1, 2: link resolution
def test_wiki_key_capitalises_the_first_letter_like_mediawiki():
    assert wiki_key("oxygen") == "Oxygen" and wiki_key("silk road") == "Silk_road" and wiki_key(":de:x") == "De:x"
    assert wiki_key("ßeta") == "ßeta"  # upper() would widen the letter; MediaWiki keeps it


def test_choice_set_resolves_lowercase_merges_redirects_and_drops_self_links():
    backend = FixtureBackend(LINK_CASES)
    hub = backend.load_page("Hub")
    rows, _, counts = build_choice_set(hub, backend, "source", 1, 1)
    assert [r["title"] for r in rows] == ["Oxygen", "Carbon", "Nitrogen"]
    assert [r["link_title"] for r in rows] == ["oxygen", "Carbon", "nitrogen"]
    assert [r["original_position"] for r in rows] == [0, 1, 2]
    assert counts == {"n_links_self": 2, "n_links_duplicate": 1, "n_links_unresolved": 1}  # [[oxygen]] = [[Oxygen]] merged by the parser; O2 here
    assert rows[0]["semantic_cluster_label"] == ["Elements"] and rows[0]["page_id"] == 2


def test_redirect_link_is_flagged_and_lands_on_its_target():
    backend = FixtureBackend(LINK_CASES)
    page = backend.load_page("Hub")
    page.links.insert(0, page.links.pop(next(i for i, l in enumerate(page.links) if l.candidate_key == "O2")))
    rows, _, counts = build_choice_set(page, backend, "source", 1, 1)
    assert rows[0]["title"] == "Oxygen" and rows[0]["link_title"] == "O2" and rows[0]["via_redirect"] is True
    assert counts["n_links_duplicate"] == 1


def test_run_logs_link_counts(tmp_path):
    _, _, steps, _ = run_with(tmp_path, Scripted([reply({"reasoning": "r", "choice": "Oxygen"})]),
                              extra=[f"wikipedia.fixture_path={LINK_CASES}", "batch.start_pages=[Hub]", "experiment.step_limit=1"])
    extras = steps[0]["extras"]
    assert extras["n_links_offered"] == 3 and extras["n_links_self"] == 2 and extras["n_links_duplicate"] == 1
    assert steps[0]["parsed_choice"] == "Oxygen"


def _mask_spec(mode, titles=()):
    semantic = mode in ("semantic_visited", "sham_semantic_count")
    topic_map = _test_topic_map() if semantic else None
    return {"mode": mode, "mask_id": f"test-{mode}", "configured_titles": list(titles),
            "configured_keys": [wiki_key(title) for title in titles],
            "semantic_threshold": 0.5 if semantic else None, "semantic_resources": None,
            "semantic_topic_hash": topic_map.hash() if topic_map else None, "sha256": f"sha-{mode}"}


def _test_topic_map():
    return TopicMap(("Science", "Materials", "Other", "unlabeled"),
                    {"Elements": (0,), "Materials": (1,), "Other": (2,)}, {}, frozenset())


def test_visited_link_mask_removes_resolved_visited_destination_and_logs_it():
    backend = FixtureBackend(LINK_CASES)
    rows, _, counts = build_choice_set(backend.load_page("Hub"), backend, "source", 7, 1,
                                      mask_spec=_mask_spec("visited"), visited_keys=["Oxygen"])
    assert [row["title"] for row in rows] == ["Carbon", "Nitrogen"]
    assert [row["original_position"] for row in rows] == [0, 1]
    assert counts["n_links_eligible_before_mask"] == 3 and counts["n_links_masked"] == 1
    mask = counts["link_mask"]
    assert mask["mode"] == "visited" and mask["visited_candidate_count"] == 1
    assert mask["masked_candidates"] == [{"eligible_position": 0, "title": "Oxygen", "link_title": "oxygen",
                                            "via_redirect": False, "page_id": 2, "was_visited": True}]


def test_sham_mask_keeps_visited_link_and_deterministically_matches_removed_count():
    backend = FixtureBackend(LINK_CASES)
    args = (backend.load_page("Hub"), backend, "source", 7, 1)
    first, _, counts = build_choice_set(*args, mask_spec=_mask_spec("sham_visited_count"), visited_keys=["Oxygen"])
    repeat, _, repeated = build_choice_set(*args, mask_spec=_mask_spec("sham_visited_count"), visited_keys=["Oxygen"])
    assert first == repeat and counts["link_mask"] == repeated["link_mask"]
    assert "Oxygen" in [row["title"] for row in first]
    assert len(first) == 2 and counts["n_links_masked"] == 1
    mask = counts["link_mask"]
    assert mask["visited_candidate_count"] == 1 and mask["sham_shortfall"] == 0
    assert mask["masked_candidates"][0]["was_visited"] is False


def _search_rows():
    return [
        {"original_position": 3, "shuffled_position": 0, "title": "Oxygen", "hyperlink": "u:o",
         "page_id": 2, "semantic_cluster_label": ["Elements"], "snippet": "o", "snippet_source": "lead",
         "matched_title": "oxygen", "via_redirect": False},
        {"original_position": 5, "shuffled_position": 1, "title": "Carbon", "hyperlink": "u:c",
         "page_id": 3, "semantic_cluster_label": ["Materials"], "snippet": "c", "snippet_source": "lead",
         "matched_title": "Carbon", "via_redirect": False},
        {"original_position": 8, "shuffled_position": 2, "title": "Nitrogen", "hyperlink": "u:n",
         "page_id": 4, "semantic_cluster_label": ["Other"], "snippet": "n", "snippet_source": "lead",
         "matched_title": "Nitrogen", "via_redirect": False},
    ]


def test_search_aware_visited_mask_removes_revisit_and_preserves_original_search_ranks():
    kept, mask = apply_search_result_mask(_search_rows(), _mask_spec("visited_all"), ["Oxygen"], 7, 2)
    assert [row["title"] for row in kept] == ["Carbon", "Nitrogen"]
    assert [row["original_position"] for row in kept] == [5, 8]
    assert [row["shuffled_position"] for row in kept] == [0, 1]
    assert mask["mode"] == "visited_all" and mask["visited_candidate_count"] == 1
    assert mask["masked_candidates"][0]["title"] == "Oxygen"


def test_search_aware_sham_keeps_revisit_and_matches_removed_count_deterministically():
    spec = _mask_spec("sham_visited_all_count")
    first, mask = apply_search_result_mask(_search_rows(), spec, ["Oxygen"], 7, 2)
    repeat, repeated_mask = apply_search_result_mask(_search_rows(), spec, ["Oxygen"], 7, 2)
    assert first == repeat and mask == repeated_mask
    assert "Oxygen" in [row["title"] for row in first]
    assert len(first) == 2 and mask["masked_count"] == 1 and mask["sham_shortfall"] == 0
    assert mask["masked_candidates"][0]["was_visited"] is False


def test_search_aware_mask_is_logged_and_empty_search_falls_back_to_page_links(tmp_path):
    provider = Scripted([
        reply({"reasoning": "open it", "action": "follow", "choice": "Archive"}),
        reply({"reasoning": "try returning", "action": "search", "query": "Archive"}),
        reply({"reasoning": "search target was masked", "choice": "Bridge"}),
    ])
    _, _, steps, run_dir = run_with(
        tmp_path, provider, variant="c_search_anytime",
        extra=["experiment.step_limit=2", "experiment.link_mask_mode=visited_all",
               "experiment.link_mask_id=test-search-aware"],
    )
    search_step = steps[1]
    assert search_step["action"] == "search" and search_step["search"]["results"] == []
    assert search_step["parsed_choice"] == "Bridge"
    mask = search_step["extras"]["candidate_filter_provenance"]["search_mask"]
    assert mask["visited_candidate_count"] == 1 and mask["masked_count"] == 1
    trajectory = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]
    assert trajectory[1]["transition"] == "link" and trajectory[1]["chosen_title"] == "Bridge"


def test_search_result_choice_never_uses_random_fallback_after_invalid_tool_calls(tmp_path):
    provider = Scripted([
        tool_reply("search", query="Archive"),
        tool_reply("search", "c2", query="Bridge"),
        tool_reply("search", "c3", query="Canal"),
    ])
    _, results, steps, _ = run_with(
        tmp_path, provider, variant="c_search_anytime",
        extra=["experiment.action_format=tools", "experiment.step_limit=1"],
    )
    step = steps[0]
    assert results[0]["termination_reason"] == "parse_error"
    assert results[0]["outcome_category"] == "C-technical"
    assert step["action"] == "search" and step["parsed_choice"] is None
    assert step["invalid_action"]["harness_handling"] == "terminate"
    assert step["resolution"] == {
        "resolve_status": "parse_failure",
        "resolve_error": "invalid search-result choice after retry; no random fallback",
        "resolved_page_id": None,
    }
    initial_instruction = provider.requests[1][-1]
    retry_instruction = provider.requests[2][-1]
    assert initial_instruction["role"] == "tool"
    assert "Call follow_link exactly once" in initial_instruction["content"]
    assert "Do not call search again" in initial_instruction["content"]
    assert retry_instruction["role"] == "tool"
    assert "Call follow_link exactly once" in retry_instruction["content"]
    assert "Do not call search again" in retry_instruction["content"]


def test_visited_static_titles_removes_revisit_and_preregistered_candidate():
    backend = FixtureBackend(LINK_CASES)
    rows, _, counts = build_choice_set(
        backend.load_page("Hub"), backend, "source", 7, 1,
        mask_spec=_mask_spec("visited_static_titles", ["Oxygen"]), visited_keys=["Carbon"],
    )
    assert [row["title"] for row in rows] == ["Nitrogen"]
    mask = counts["link_mask"]
    assert mask["visited_candidate_count"] == 1
    assert mask["static_target_candidate_count"] == 1
    assert mask["static_incremental_target_count"] == 1
    assert [(row["title"], row["was_visited"], row["is_static_target"])
            for row in mask["masked_candidates"]] == [
                ("Oxygen", False, True), ("Carbon", True, False)]


def test_visited_static_sham_preserves_target_and_matches_incremental_count():
    backend = FixtureBackend(LINK_CASES)
    args = (backend.load_page("Hub"), backend, "source", 7, 1)
    spec = _mask_spec("visited_sham_static_count", ["Oxygen"])
    first, _, counts = build_choice_set(*args, mask_spec=spec, visited_keys=["Carbon"])
    repeat, _, repeated = build_choice_set(*args, mask_spec=spec, visited_keys=["Carbon"])
    assert first == repeat and counts["link_mask"] == repeated["link_mask"]
    assert [row["title"] for row in first] == ["Oxygen"]
    mask = counts["link_mask"]
    assert mask["visited_candidate_count"] == 1
    assert mask["static_target_candidate_count"] == 1
    assert mask["static_incremental_target_count"] == 1
    assert mask["sham_shortfall"] == 0
    assert [(row["title"], row["was_visited"], row["is_static_target"])
            for row in mask["masked_candidates"]] == [
                ("Carbon", True, False), ("Nitrogen", False, False)]


def _semantic_backend():
    backend = FixtureBackend(LINK_CASES)
    backend.load_page("Oxygen").categories = ["Elements"]
    backend.load_page("Carbon").categories = ["Materials"]
    backend.load_page("Nitrogen").categories = ["Other"]
    return backend


def test_semantic_mask_removes_exact_revisit_and_topic_neighbour_but_keeps_unrelated():
    backend, topic_map = _semantic_backend(), _test_topic_map()
    rows, _, counts = build_choice_set(
        backend.load_page("Hub"), backend, "source", 7, 1,
        mask_spec=_mask_spec("semantic_visited"), visited_keys=["Carbon"],
        visited_categories=[["Elements"]], topic_map=topic_map,
    )
    assert [row["title"] for row in rows] == ["Nitrogen"]
    mask = counts["link_mask"]
    assert mask["semantic_target_candidate_count"] == 2
    assert [(row["title"], row["was_visited"], row["semantic_match"], row["is_semantic_target"])
            for row in mask["masked_candidates"]] == [
                ("Oxygen", False, True, True), ("Carbon", True, False, True)]


def test_semantic_sham_preserves_targets_and_deterministically_masks_equal_unrelated_count():
    backend, topic_map = _semantic_backend(), _test_topic_map()
    args = (backend.load_page("Hub"), backend, "source", 7, 1)
    kwargs = dict(mask_spec=_mask_spec("sham_semantic_count"), visited_keys=[],
                  visited_categories=[["Elements"]], topic_map=topic_map)
    first, _, counts = build_choice_set(*args, **kwargs)
    repeat, _, repeated = build_choice_set(*args, **kwargs)
    assert first == repeat and counts["link_mask"] == repeated["link_mask"]
    assert "Oxygen" in [row["title"] for row in first]
    assert len(first) == 2 and counts["n_links_masked"] == 1
    mask = counts["link_mask"]
    assert mask["semantic_target_candidate_count"] == 1 and mask["sham_shortfall"] == 0
    assert mask["masked_candidates"][0]["is_semantic_target"] is False


def test_unknown_link_mask_mode_fails_before_a_batch_is_created(tmp_path):
    with pytest.raises(ValueError, match="unknown link_mask_mode"):
        run_with(tmp_path, Scripted([]), extra=["experiment.link_mask_mode=bogus"])


# --------------------------------------------------------------------------- 3: feedback, retry, honest history
def test_out_of_range_reply_gets_specific_feedback_and_its_own_reply_back(tmp_path):
    provider = Scripted([reply({"reasoning": "r1", "choice": "Atlantis"}), reply({"reasoning": "r2", "choice": "Archive"})])
    _, _, steps, run_dir = run_with(tmp_path, provider, extra=["experiment.step_limit=1"])
    retry_request = provider.requests[1]
    assert retry_request[-2] == {"role": "assistant", "content": json.dumps({"reasoning": "r1", "choice": "Atlantis"})}
    assert retry_request[-1]["content"].startswith('"Atlantis" is not one of the listed titles.')
    assert steps[0]["invalid_action"]["harness_handling"] == "retry" and steps[0]["parsed_choice"] == "Archive"
    assert (run_dir / "model_inputs" / "step_001_request_retry.txt").exists()


def test_unparseable_reply_feedback(tmp_path):
    provider = Scripted([reply("I would like to visit the archive"), reply({"reasoning": "r", "choice": "Archive"})])
    run_with(tmp_path, provider, extra=["experiment.step_limit=1"])
    assert provider.requests[1][-1]["content"].startswith("Your reply could not be read")


def test_harness_pick_is_shown_as_a_harness_note_never_as_the_models_reply(tmp_path):
    provider = Scripted([reply("garbage"), reply("still garbage"), reply({"reasoning": "next", "choice": "STOP"})])
    _, _, steps, _ = run_with(tmp_path, provider, extra=["experiment.step_limit=2"])
    picked = steps[0]["parsed_choice"]
    assert steps[0]["invalid_action"]["harness_handling"] == "skip"
    history = provider.requests[2][1:-1]
    assert history == [{"role": "user", "content": f'No listed page was chosen, so "{picked}" was opened at random.'}]
    assert not any(m["role"] == "assistant" for m in provider.requests[2])


@pytest.mark.parametrize("memory", ["pages_no_links", "lead", "sliding_window", "full", "self_notes"])
def test_harness_pick_never_appears_as_an_assistant_turn_in_any_memory_mode(tmp_path, memory):
    provider = Scripted([reply("garbage"), reply("still garbage"), reply({"reasoning": "next", "choice": "STOP", "note": "n"})])
    _, _, steps, _ = run_with(tmp_path, provider, extra=["experiment.step_limit=2", f"experiment.memory={memory}"])
    picked, request = steps[0]["parsed_choice"], provider.requests[2]
    assert not any(m["role"] == "assistant" for m in request)
    assert any(picked in m["content"] and "harness" in m["content"] for m in request if m["role"] == "user")


def test_stop_on_retry_is_respected(tmp_path):
    _, results, steps, _ = run_with(tmp_path, Scripted([reply("garbage"), reply({"reasoning": "enough", "choice": "STOP"})]))
    assert results[0]["termination_reason"] == "model_stopped" and steps[0]["parsed_choice"] == "STOP"


def test_api_error_during_retry_ends_the_run_as_api_error(tmp_path):
    failed = ProviderResponse(None, {}, None, 4, "HTTP 503", [], error_kind="retries_exhausted")
    _, results, steps, _ = run_with(tmp_path, Scripted([reply("garbage"), failed]))
    assert results[0]["termination_reason"] == "api_error" and steps[0]["invalid_action"]["harness_handling"] == "terminate"


def test_three_steps_without_a_listed_page_end_as_no_valid_move(tmp_path):
    provider = Scripted([reply("I'll stop here rather than keep browsing."), reply({"reasoning": "r", "choice": "Atlantis"})] * 3)
    _, results, steps, run_dir = run_with(tmp_path, provider, extra=["experiment.step_limit=10"])
    assert results[0]["termination_reason"] == "no_valid_move" and results[0]["outcome_category"] == "A-normal"
    assert results[0]["status"] == "Completed" and len(steps) == 3
    assert [s["invalid_action"]["harness_handling"] for s in steps] == ["skip"] * 3


@pytest.mark.parametrize("bad", [reply(""), ProviderResponse('{"reasoning": "cut', {}, "length", 0, None, [])], ids=["empty", "cut_off"])
def test_an_empty_or_cut_off_reply_in_the_streak_keeps_parse_error(tmp_path, bad):
    provider = Scripted([reply("prose"), reply("prose"), bad, reply("prose"), reply("prose"), reply("prose")])
    _, results, _, run_dir = run_with(tmp_path, provider, extra=["experiment.step_limit=10"])
    assert results[0]["termination_reason"] == "parse_error" and results[0]["outcome_category"] == "C-technical"
    rows = [json.loads(line) for line in (run_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["cut_off"] for r in rows] == [False, False, bad.finish_reason == "length", False, False, False]


def test_a_valid_move_resets_the_streak(tmp_path):
    first_listed = lambda messages: reply({"reasoning": "r", "choice": messages[-1]["content"].split("## Candidate links")[1].split("\n- ")[1].split("\n")[0]})
    provider = Scripted([reply(""), reply(""), first_listed] + [reply("prose")] * 6)
    _, results, steps, _ = run_with(tmp_path, provider, extra=["experiment.step_limit=10"])
    assert results[0]["termination_reason"] == "no_valid_move" and len(steps) == 5   # skip, move, 3 skips


def test_parse_retries_and_feedback_are_configurable(tmp_path):
    provider = Scripted([reply("a"), reply("b"), reply({"reasoning": "r", "choice": "Archive"})])
    _, _, steps, _ = run_with(tmp_path, provider, extra=["experiment.step_limit=1", "experiment.parse_retries=2",
                                                         "experiment.retry_feedback.unparseable=Try again."])
    assert provider.requests[1][-1]["content"] == "Try again." and provider.requests[2][-1]["content"] == "Try again."
    assert steps[0]["parsed_choice"] == "Archive"


# --------------------------------------------------------------------------- 4, 5, 6: provider replies and errors
def test_internal_reasoning_and_api_latency_are_logged(tmp_path):
    provider = Scripted([reply({"reasoning": "r", "choice": "Archive"}, internal_reasoning="thinking...", latency_s=1.25)])
    _, _, steps, _ = run_with(tmp_path, provider, extra=["experiment.step_limit=1"])
    assert steps[0]["model_internal_reasoning"] == "thinking..."
    assert steps[0]["extras"]["latency_s"] == 1.25 and steps[0]["dwell_time"] >= 0


def test_no_internal_reasoning_is_null(tmp_path):
    _, _, steps, _ = run_with(tmp_path, Scripted([reply({"reasoning": "r", "choice": "Archive"})]), extra=["experiment.step_limit=1"])
    assert steps[0]["model_internal_reasoning"] is None


def test_context_limit_ends_the_run_as_context_limit(tmp_path):
    too_big = ProviderResponse(None, {}, None, 0, "HTTP 413: too large", [], error_kind="context_limit")
    dataset, results, _, run_dir = run_with(tmp_path, Scripted([too_big]))
    assert results[0]["termination_reason"] == "context_limit" and results[0]["outcome_category"] == "B-environment"


class FakeHTTP:
    def __init__(self, status, body=None, headers=None, text=""):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, text or json.dumps(body or {})
    def json(self):
        return self._body


def provider_with(monkeypatch, responses, **kwargs):
    import requests
    sent, waits = [], []
    def post(url, json=None, **_):
        sent.append(json)
        return responses.pop(0)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("gap_attractor.providers.base.time.sleep", waits.append)
    monkeypatch.setenv("TEST_KEY", "k")
    monkeypatch.setitem(base._SUPPORTED, "https://x", {"limited": {"tools", "max_tokens"}})   # no network lookup in tests
    return OpenAICompatibleProvider("openrouter", kwargs.pop("model", "m"), "https://x", "TEST_KEY", **kwargs), sent, waits


OK = {"choices": [{"message": {"content": "hi", "reasoning": "because"}, "finish_reason": "stop"}], "usage": {}}


def test_retry_after_is_honoured_and_reasoning_extracted(monkeypatch):
    provider, sent, waits = provider_with(monkeypatch, [FakeHTTP(429, headers={"Retry-After": "7"}), FakeHTTP(200, OK)],
                                          max_tokens=4000, extra_body={"reasoning": {"effort": "low"}})
    response = provider.chat([{"role": "user", "content": "x"}], seed=1, temperature=0, top_p=1)
    assert response.text == "hi" and response.internal_reasoning == "because" and response.error is None
    assert waits == [7.0] and sent[0]["max_tokens"] == 4000 and sent[0]["reasoning"] == {"effort": "low"}


def test_per_call_request_overrides_are_narrow_and_do_not_mutate_provider_defaults(monkeypatch):
    provider, sent, _ = provider_with(monkeypatch, [FakeHTTP(200, OK), FakeHTTP(200, OK)],
                                      extra_body={"reasoning": {"effort": "high"}})
    provider.chat([{"role": "user", "content": "forecast"}], seed=1, temperature=0, top_p=1,
                  request_overrides={"reasoning": {"effort": "low"}, "response_format": {"type": "json_object"}})
    provider.chat([{"role": "user", "content": "navigate"}], seed=2, temperature=0, top_p=1)
    assert sent[0]["reasoning"] == {"effort": "low"} and sent[0]["response_format"] == {"type": "json_object"}
    assert sent[1]["reasoning"] == {"effort": "high"} and "response_format" not in sent[1]
    with pytest.raises(ValueError, match="unsupported request override"):
        provider.chat([], seed=3, temperature=0, top_p=1, request_overrides={"model": "other"})


def test_per_call_request_overrides_preserve_authentication_header(monkeypatch):
    import requests
    seen = {}
    def post(url, json=None, headers=None, **_):
        seen["headers"] = headers
        return FakeHTTP(200, OK)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setenv("TEST_KEY", "secret-token")
    monkeypatch.setitem(base._SUPPORTED, "https://x", {"m": {"reasoning", "response_format"}})
    provider = OpenAICompatibleProvider("openrouter", "m", "https://x", "TEST_KEY")
    response = provider.chat([], seed=1, temperature=0, top_p=1,
                             request_overrides={"reasoning": {"effort": "low"},
                                                "response_format": {"type": "json_object"}})
    assert response.error is None
    assert seen["headers"]["Authorization"] == "Bearer secret-token"


def test_backoff_is_capped(monkeypatch):
    provider, _, waits = provider_with(monkeypatch, [FakeHTTP(503, headers={"Retry-After": "600"}), FakeHTTP(200, OK)], max_backoff_s=30)
    provider.chat([], seed=1, temperature=0, top_p=1)
    assert waits == [30.0]


@pytest.mark.parametrize("status, text, kind", [(413, "Payload Too Large", "context_limit"),
                                               (400, "This model's maximum context length is 8192 tokens", "context_limit"),
                                               (401, "invalid api key", "client_error")])
def test_non_retryable_errors_fail_at_once(monkeypatch, status, text, kind):
    provider, sent, waits = provider_with(monkeypatch, [FakeHTTP(status, text=text)])
    response = provider.chat([], seed=1, temperature=0, top_p=1)
    assert response.error_kind == kind and len(sent) == 1 and waits == []


def test_in_flight_budget_402_is_retried_but_other_402_is_not(monkeypatch):
    provider, sent, waits = provider_with(monkeypatch, [FakeHTTP(402, text='{"reason":"in_flight_budget_exhausted"}', headers={"Retry-After": "5"}),
                                                        FakeHTTP(200, OK)])
    assert provider.chat([], seed=1, temperature=0, top_p=1).text == "hi" and len(sent) == 2 and waits == [5.0]
    provider, sent, _ = provider_with(monkeypatch, [FakeHTTP(402, text="insufficient credits")])
    assert provider.chat([], seed=1, temperature=0, top_p=1).error_kind == "client_error" and len(sent) == 1


def test_retries_exhausted(monkeypatch):
    provider, sent, _ = provider_with(monkeypatch, [FakeHTTP(500)] * 3, max_retries=2)
    assert provider.chat([], seed=1, temperature=0, top_p=1).error_kind == "retries_exhausted" and len(sent) == 3


def test_openrouter_requests_usage_details(monkeypatch):
    provider, sent, _ = provider_with(monkeypatch, [FakeHTTP(200, OK)])
    provider.chat([{"role": "user", "content": "x"}], seed=3, temperature=0.5, top_p=0.5)
    assert sent[0] == {"model": "m", "messages": [{"role": "user", "content": "x"}], "temperature": 0.5,
                       "top_p": 0.5, "max_tokens": 700, "seed": 3, "usage": {"include": True}}


def test_provider_cost_is_logged_and_exported(tmp_path):
    response = ProviderResponse(
        json.dumps({"reasoning": "r", "choice": "Archive"}),
        {"prompt_tokens": 10, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 1}, "cost": 0.0123},
        "stop", 0, None, [], response_id="gen-1", response_model="provider/model",
    )
    dataset, results, steps, run_dir = run_with(
        tmp_path, Scripted([response]),
        extra=["experiment.step_limit=1", "experiment.tracker_config_id=CFG-1", "experiment.tracker_owner=Yuliya"],
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert steps[0]["extras"]["provider_usage"]["cost_usd"] == pytest.approx(0.0123)
    assert summary["cost_usd"] == pytest.approx(0.0123)
    assert (summary["input_tokens"], summary["output_tokens"], summary["reasoning_tokens"]) == (10, 2, 1)
    assert summary["provider_generation_ids"] == ["gen-1"]
    rows = list(csv.DictReader((dataset.root / "run_log.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["Config ID"] == "CFG-1" and rows[0]["Owner"] == "Yuliya"
    assert float(rows[0]["Cost (USD)"]) == pytest.approx(0.0123)


def test_stateless_pre_run_predictions_are_logged_costed_and_not_shown_to_navigation(tmp_path):
    def prediction(messages):
        self_frame = "your own likely navigation" in messages[-1]["content"]
        return ProviderResponse(
            json.dumps({
                "pages": ["Archive" if self_frame else "Bridge"],
                "topics": ["history"],
                "confidence": 0.7,
            }),
            {
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 1},
                "cost": 0.001,
            },
            "stop", 0, None, [], response_id="prediction-id",
        )

    navigation = ProviderResponse(
        json.dumps({"reasoning": "r", "choice": "Archive"}),
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "completion_tokens_details": {"reasoning_tokens": 1},
            "cost": 0.0123,
        },
        "stop", 0, None, [], response_id="navigation-id",
    )
    provider = Scripted([prediction, prediction, navigation])
    _, _, _, run_dir = run_with(
        tmp_path,
        provider,
        extra=["experiment.step_limit=1", "experiment.pre_run_prediction.mode=self_and_generic"],
    )
    doc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    records = doc["pre_run_prediction"]["records"]
    assert {record["framing"] for record in records} == {"self", "generic"}
    assert all(record["parse_ok"] for record in records)
    assert doc["pre_run_prediction"]["navigation_context_contaminated"] is False
    assert (run_dir / "pre_run_prediction.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["pre_run_prediction_mode"] == "self_and_generic"
    assert summary["valid_pre_run_predictions"] == 2
    assert summary["cost_usd"] == pytest.approx(0.0143)
    assert (summary["input_tokens"], summary["output_tokens"], summary["reasoning_tokens"]) == (18, 8, 3)

    navigation_text = json.dumps(provider.requests[-1])
    assert "preregistered forecast" not in navigation_text
    assert "your own likely navigation" not in navigation_text


def test_self_and_external_predictions_use_separate_models_and_stay_out_of_navigation(tmp_path, monkeypatch):
    self_prediction = ProviderResponse(
        json.dumps({"pages": ["Archive"], "topics": ["history"], "confidence": 0.7}),
        {"prompt_tokens": 4, "completion_tokens": 3, "cost": 0.001},
        "stop", 0, None, [], response_id="self-prediction-id",
    )
    external_prediction = ProviderResponse(
        json.dumps({"pages": ["Bridge"], "topics": ["infrastructure"], "confidence": 0.6}),
        {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.002},
        "stop", 0, None, [], response_id="external-prediction-id",
    )
    navigation = ProviderResponse(
        json.dumps({"reasoning": "r", "choice": "Archive"}),
        {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.01},
        "stop", 0, None, [], response_id="navigation-id",
    )
    target = Scripted([self_prediction, navigation])
    external = Scripted([external_prediction])
    monkeypatch.setattr("gap_attractor.harness.runner._external_prediction_provider", lambda cfg: external)

    _, _, _, run_dir = run_with(
        tmp_path,
        target,
        extra=[
            "experiment.step_limit=1",
            "experiment.pre_run_prediction.mode=self_and_external",
            "experiment.pre_run_prediction.external_model=external/model",
        ],
    )
    doc = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert doc["pre_run_prediction"]["target_model"] == "mock-random"
    assert doc["pre_run_prediction"]["external_forecaster_model"] == "external/model"
    records = {record["framing"]: record for record in doc["pre_run_prediction"]["records"]}
    assert records["self"]["forecaster_model"] == "mock-random"
    assert records["external"]["forecaster_model"] == "external/model"
    assert records["self"]["pages"] == ["Archive"] and records["external"]["pages"] == ["Bridge"]
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["cost_usd"] == pytest.approx(0.013)

    navigation_text = json.dumps(target.requests[-1])
    external_text = json.dumps(external.requests[0])
    assert "independent external forecaster" in external_text
    assert "external/model" not in navigation_text
    assert "preregistered forecast" not in navigation_text


def test_memory_effect_prediction_is_strictly_parsed_and_kept_out_of_navigation(tmp_path, monkeypatch):
    forecast = {
        "higher_unique_pages": "path_only",
        "unique_pages_none": "1-5",
        "unique_pages_path_only": "16-30",
        "higher_revisit_rate": "none",
        "more_likely_to_reach_step_limit": "path_only",
        "p_none_reaches_limit": 0.2,
        "p_path_only_reaches_limit": 0.8,
        "confidence": 0.7,
    }
    target = Scripted([reply(forecast), reply({"reasoning": "r", "choice": "Archive"})])
    external = Scripted([reply({**forecast, "confidence": 0.6})])
    monkeypatch.setattr("gap_attractor.harness.runner._external_prediction_provider", lambda cfg: external)

    _, _, _, run_dir = run_with(
        tmp_path,
        target,
        extra=[
            "experiment.step_limit=1",
            "experiment.pre_run_prediction.mode=self_and_external",
            "experiment.pre_run_prediction.task=memory_effect",
            "experiment.pre_run_prediction.external_model=external/model",
        ],
    )
    doc = json.loads((run_dir / "pre_run_prediction.json").read_text(encoding="utf-8"))
    assert doc["task"] == "memory_effect"
    assert all(record["parse_ok"] for record in doc["records"])
    assert {record["memory_effect"]["higher_unique_pages"] for record in doc["records"]} == {"path_only"}
    assert "Matched conditions: memory=none versus memory=path_only" in json.dumps(target.requests[0])
    assert "Matched conditions: memory=none versus memory=path_only" in json.dumps(external.requests[0])
    assert "memory=none versus memory=path_only" not in json.dumps(target.requests[-1])


def test_invalid_memory_effect_prediction_fails_strict_parser(tmp_path):
    bad = reply({
        "higher_unique_pages": "maybe",
        "unique_pages_none": "about six",
        "unique_pages_path_only": "16-30",
        "higher_revisit_rate": "none",
        "more_likely_to_reach_step_limit": "path_only",
        "p_none_reaches_limit": 2,
        "p_path_only_reaches_limit": 0.8,
        "confidence": 0.7,
    })
    _, _, _, run_dir = run_with(
        tmp_path,
        Scripted([bad, reply({"reasoning": "r", "choice": "Archive"})]),
        extra=[
            "experiment.step_limit=1",
            "experiment.pre_run_prediction.mode=self",
            "experiment.pre_run_prediction.task=memory_effect",
        ],
    )
    doc = json.loads((run_dir / "pre_run_prediction.json").read_text(encoding="utf-8"))
    assert doc["records"][0]["parse_ok"] is False


def test_coarse_single_run_prediction_is_strict_and_request_controls_do_not_touch_navigation(tmp_path, monkeypatch):
    forecast = {
        "first_topic_root": "Philosophy", "unique_pages": "6-15",
        "any_page_revisit": True, "reaches_step_limit": False, "confidence": 0.65,
    }
    target = Scripted([reply(forecast), reply({"reasoning": "r", "choice": "Archive"})])
    external = Scripted([reply({**forecast, "first_topic_root": "History"})])
    monkeypatch.setattr("gap_attractor.harness.runner._external_prediction_provider", lambda cfg: external)
    _, _, _, run_dir = run_with(
        tmp_path, target,
        extra=[
            "experiment.step_limit=1", "experiment.pre_run_prediction.mode=self_and_external",
            "experiment.pre_run_prediction.task=coarse_single_run",
            "experiment.pre_run_prediction.external_model=external/model",
            "experiment.pre_run_prediction.reasoning_effort=low",
            "experiment.pre_run_prediction.structured_output=true",
        ],
    )
    doc = json.loads((run_dir / "pre_run_prediction.json").read_text(encoding="utf-8"))
    assert doc["task"] == "coarse_single_run" and all(record["parse_ok"] for record in doc["records"])
    records = {record["framing"]: record for record in doc["records"]}
    assert records["self"]["coarse_single_run"]["first_topic_root"] == "Philosophy"
    assert records["external"]["coarse_single_run"]["first_topic_root"] == "History"
    assert target.calls[0]["request_overrides"]["reasoning"] == {"effort": "low"}
    assert target.calls[0]["request_overrides"]["response_format"]["type"] == "json_schema"
    assert "request_overrides" not in target.calls[-1]
    assert "first_topic_root" not in json.dumps(target.requests[-1])


def test_invalid_coarse_single_run_prediction_fails_strict_parser(tmp_path):
    bad = reply({
        "first_topic_root": "Whatever", "unique_pages": "about six",
        "any_page_revisit": "yes", "reaches_step_limit": False, "confidence": 2,
    })
    _, _, _, run_dir = run_with(
        tmp_path, Scripted([bad, reply({"reasoning": "r", "choice": "Archive"})]),
        extra=[
            "experiment.step_limit=1", "experiment.pre_run_prediction.mode=self",
            "experiment.pre_run_prediction.task=coarse_single_run",
        ],
    )
    doc = json.loads((run_dir / "pre_run_prediction.json").read_text(encoding="utf-8"))
    assert doc["records"][0]["parse_ok"] is False


def test_unknown_pre_run_prediction_mode_fails_before_a_batch_is_created(tmp_path):
    with pytest.raises(ValueError, match="unknown experiment.pre_run_prediction.mode"):
        run_with(tmp_path, Scripted([]), extra=["experiment.pre_run_prediction.mode=bogus"])


def test_self_and_external_requires_a_distinct_external_model(tmp_path):
    with pytest.raises(ValueError, match="external_model is required"):
        run_with(tmp_path, Scripted([]), extra=["experiment.pre_run_prediction.mode=self_and_external"])
    with pytest.raises(ValueError, match="must differ"):
        run_with(tmp_path, Scripted([]), extra=[
            "experiment.pre_run_prediction.mode=self_and_external",
            "experiment.pre_run_prediction.external_model=mock-random",
        ])


def test_openrouter_leaves_out_sampling_params_the_model_does_not_accept(monkeypatch):
    provider, sent, _ = provider_with(monkeypatch, [FakeHTTP(200, OK)], model="limited")
    response = provider.chat([{"role": "user", "content": "x"}], seed=3, temperature=1, top_p=0.5,
                             tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}])
    assert "temperature" not in sent[0] and "top_p" not in sent[0] and "seed" not in sent[0]
    assert sent[0]["provider"]["require_parameters"] is True and sent[0]["max_tokens"] == 700
    assert response.attempts[0]["dropped_params"] == ["temperature", "top_p", "seed"]
    off, sent_off, _ = provider_with(monkeypatch, [FakeHTTP(200, OK)], model="limited", drop_unsupported_params=False)
    off.chat([{"role": "user", "content": "x"}], seed=3, temperature=1, top_p=0.5)
    assert sent_off[0]["temperature"] == 1 and sent_off[0]["seed"] == 3
    unknown, sent_unknown, _ = provider_with(monkeypatch, [FakeHTTP(200, OK)], model="not-listed")
    response = unknown.chat([{"role": "user", "content": "x"}], seed=3, temperature=1, top_p=0.5)
    assert sent_unknown[0]["temperature"] == 1 and response.attempts[0]["params_unchecked"] is True   # sent, but unknown


# --------------------------------------------------------------------------- 5: resume
def test_resume_reruns_only_cut_off_and_api_error_runs(tmp_path):
    extra = ["batch.start_pages=[Civic Atlas,Archive,Bridge]", "experiment.step_limit=1", "batch.index=7"]
    dataset, _ = run_experiment(config(tmp_path, extra=extra), ROOT)
    runs = sorted(p for p in dataset.root.iterdir() if p.is_dir())
    (runs[1] / "summary.json").unlink()  # run 2 was cut off
    doc = json.loads((runs[2] / "run.json").read_text(encoding="utf-8"))
    doc["outcome"]["termination_reason"] = "api_error"  # run 3 failed on the API
    (runs[2] / "run.json").write_text(json.dumps(doc), encoding="utf-8")
    resumed, results = run_experiment(config(tmp_path, extra=[*extra, "batch.resume=true"]), ROOT)
    assert resumed.root == dataset.root and [r["run_index"] for r in results] == [2, 3]
    assert sorted(p.name for p in (dataset.root / "_superseded").iterdir()) == sorted([runs[1].name, runs[2].name])
    batch = json.loads((dataset.root / "batch.json").read_text(encoding="utf-8"))
    assert batch["resumes"][0]["kept_run_indexes"] == [1] and batch["resumes"][0]["rerun_indexes"] == [2, 3]
    assert json.loads((dataset.root / "dataset_metadata.json").read_text(encoding="utf-8"))["run_summary"]["total_runs"] == 3


def test_resume_refuses_a_different_config(tmp_path):
    extra = ["experiment.step_limit=1", "batch.index=3"]
    dataset, _ = run_experiment(config(tmp_path, extra=extra), ROOT)
    with pytest.raises(ValueError, match="experiment.step_limit"):
        run_experiment(config(tmp_path, extra=["experiment.step_limit=2", "batch.index=3", f"batch.resume={dataset.root.as_posix()}"]), ROOT)


def test_resume_needs_an_existing_batch(tmp_path):
    with pytest.raises(FileNotFoundError, match="batch.resume"):
        run_experiment(config(tmp_path, extra=["batch.index=42", "batch.resume=true"]), ROOT)


# --------------------------------------------------------------------------- 9: provenance
def test_git_provenance_is_logged(tmp_path):
    dataset, _ = run_experiment(config(tmp_path, extra=["experiment.step_limit=1"]), ROOT)
    code = json.loads((dataset.root / "dataset_metadata.json").read_text(encoding="utf-8"))["code"]
    assert set(code) >= {"git_commit", "git_branch", "git_dirty", "git_error"}
    assert code["git_commit"] and code["git_error"] is None and isinstance(code["git_dirty"], bool)
    run_config = json.loads(next(dataset.root.glob("*/config.json")).read_text(encoding="utf-8"))
    assert run_config["harness_git_commit"] == code["git_commit"] and run_config["harness_git_dirty"] == code["git_dirty"]


# --------------------------------------------------------------------------- 1, 7: dump backend
PAGES = [("Oxygen", 10, "Oxygen links to [[carbon]] and [[AT&T]]."), ("Carbon", 11, "Carbon."),
         ("AT&T", 12, "A company."), ("O2", 13, None)]


def synthetic_dump(tmp_path):
    """A two-stream multistream dump plus its index; O2 redirects to Oxygen; Carbon is listed twice in the index."""
    def page_xml(title, page_id, text):
        body = f'<redirect title="Oxygen" /><revision><id>1</id><text>#REDIRECT [[Oxygen]]</text></revision>' if text is None else \
               f"<revision><id>1</id><timestamp>2026-01-01T00:00:00Z</timestamp><text>{text.replace('&', '&amp;')}</text></revision>"
        return f"<page><title>{title.replace('&', '&amp;')}</title><ns>0</ns><id>{page_id}</id>{body}</page>"
    streams = [PAGES[:2], PAGES[2:]]
    blob, index, offset = b"", [], 0
    for stream in streams:
        data = bz2.compress("".join(page_xml(*p) for p in stream).encode())
        index += [f"{offset}:{page_id}:{title.replace('&', '&amp;')}" for title, page_id, _ in stream]  # the index is XML-escaped too
        blob += data; offset += len(data)
    index.append(f"{len(blob)}:99:Carbon")  # a later duplicate line must not win
    (tmp_path / "x.xml.bz2").write_bytes(blob)
    (tmp_path / "x-index.txt.bz2").write_bytes(bz2.compress("\n".join(index).encode()))
    return tmp_path


def test_sqlite_lookup_matches_pickle_lookup(tmp_path):
    dump = synthetic_dump(tmp_path)
    (tmp_path / "p").mkdir()
    pickle_lookup = PickleLookup(tmp_path / "p", dump / "x-index.txt.bz2")
    sqlite_lookup = SqliteLookup(tmp_path / "p", dump / "x-index.txt.bz2")
    for title, _, _ in PAGES:
        assert sqlite_lookup.get(title) == pickle_lookup.get(title)
    assert sqlite_lookup.get("Carbon")[1] == 11  # first index line wins
    assert not (tmp_path / "p" / "title_lookup.sqlite.lock").exists()


@pytest.mark.parametrize("lookup_format", ["sqlite", "pickle"])
def test_dump_backend_resolves_lowercase_redirect_and_ampersand(tmp_path, lookup_format):
    dump = synthetic_dump(tmp_path)
    backend = DumpBackend(dump, tmp_path / "cache", "x.xml.bz2", "x-index.txt.bz2", 10_000_000, lookup_format, page_cache_size=2)
    oxygen = backend.load_page("oxygen")
    assert oxygen.title == "Oxygen" and backend.load_page("O2").title == "Oxygen"
    assert backend.load_page("AT&T").title == "AT&T"
    rows, _, counts = build_choice_set(oxygen, backend, "source", 1, 1)
    assert [r["title"] for r in rows] == ["Carbon", "AT&T"] and counts["n_links_unresolved"] == 0
    assert backend.provenance()["lookup_format"] == lookup_format
    assert len(backend._cache._pages) <= 2


# --------------------------------------------------------------------------- 3.0: tool calling meets honest retries
def tool_reply(name, call_id="c1", **arguments):
    return ProviderResponse(None, {}, "tool_calls", 0, None, [], [{"id": call_id, "name": name, "arguments": arguments}])


def test_invalid_tool_call_gets_feedback_as_a_tool_result(tmp_path):
    provider = Scripted([tool_reply("follow_link", title="Atlantis"), tool_reply("follow_link", "c2", title="Archive")])
    _, _, steps, _ = run_with(tmp_path, provider, extra=["experiment.action_format=tools", "experiment.step_limit=1"])
    assistant, result = provider.requests[1][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "follow_link" and result["role"] == "tool" and result["tool_call_id"] == "c1"
    assert result["content"].startswith('"Atlantis" is not one of the listed titles. Call follow_link')
    assert steps[0]["parsed_choice"] == "Archive" and steps[0]["invalid_action"]["harness_handling"] == "retry"


def test_tools_mode_harness_pick_is_a_note_not_a_tool_call(tmp_path):
    provider = Scripted([tool_reply("follow_link", title="Atlantis"), tool_reply("follow_link", "c2", title="Nowhere"),
                         tool_reply("stop", "c3", reason="done")])
    _, _, steps, _ = run_with(tmp_path, provider, extra=["experiment.action_format=tools", "experiment.step_limit=2"])
    picked = steps[0]["parsed_choice"]
    assert steps[0]["invalid_action"]["harness_handling"] == "skip"
    history = provider.requests[2][1:-1]
    assert history == [{"role": "user", "content": f'No listed page was chosen, so "{picked}" was opened at random.'}]


def test_batch_records_the_start_pages_actually_used(tmp_path):
    txt = tmp_path / "hubs.txt"
    txt.write_text("Archive\nBridge\nCanal\n", encoding="utf-8")
    dataset, _ = run_experiment(config(tmp_path, extra=[f"variant.start_pages_file={txt.as_posix()}", "variant.n_runs=2",
                                                        "variant.runs_per_start=2", "experiment.step_limit=1"]), ROOT)
    batch = json.loads((dataset.root / "batch.json").read_text(encoding="utf-8"))
    assert len(batch["start_pages"]) == 4 and batch["config"]["batch"]["start_page_source"].startswith("variant.start_pages_file")
