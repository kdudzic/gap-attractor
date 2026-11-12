import bz2
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from gap_attractor.harness.runner import display_keys_for_step, page_view, parse_reply, run_experiment
from gap_attractor.providers import MockProvider
from gap_attractor.wikipedia import DumpBackend, FixtureBackend
from gap_attractor.wikipedia.backend import clean_article_text

ROOT = Path(__file__).resolve().parents[1]

# One record per candidate, logged once. Only "title" is shown to the model.
CANDIDATE_FIELDS = {"original_position", "shuffled_position", "title", "hyperlink", "page_id", "semantic_cluster_label"}
# Page links also keep the link as written on the page and whether it reached its destination through a redirect.
LINK_FIELDS = CANDIDATE_FIELDS | {"link_title", "via_redirect"}
# Search results carry the same fields plus what was shown / how they were matched. original_position = search rank.
RESULT_FIELDS = CANDIDATE_FIELDS | {"snippet", "snippet_source", "matched_title", "via_redirect"}
# Copies of the candidate list (or of other step fields) that must not come back.
REMOVED_DUPLICATES = ("available_nodes", "candidate_table", "displayed_candidates",
                      "eligible_candidates_source_order", "choice_set_size", "chosen_page_id")
REMOVED_EXTRAS = ("rendered_request", "raw_links_source_order", "article_source_wikitext",
                  "chosen_candidate_key", "chosen_position_norm")


def config(tmp_path, policy="source", seed=9001, variant="a_hub", extra=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=["provider=mock", "wikipedia=fixture", "experiment=smoke", f"variant={variant}", f"experiment.display_order_policy={policy}", f"experiment.seed={seed}", "experiment.step_limit=3", "batch.start_pages=[Civic Atlas]", f"output.directory={tmp_path.as_posix()}", *extra])
    return cfg


def steps(tmp_path, policy="source", seed=9001, label="run", variant="a_hub", extra=()):
    dataset, _ = run_experiment(config(tmp_path / label, policy, seed, variant, extra), ROOT)
    run_dir = next(path for path in dataset.root.iterdir() if (path / "steps.jsonl").exists())
    return [json.loads(line) for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()], run_dir


def titles(record):
    return [row["title"] for row in record["choice_set"]]


def candidate_rows(page, fixture):
    eligible = [link for link in page.links if fixture.contains(link.candidate_key)]
    return [{"original_position": index, "shuffled_position": index, "title": link.destination,
             "hyperlink": "logging-only", "page_id": None, "semantic_cluster_label": []}
            for index, link in enumerate(eligible)]


def test_display_order_policies_preserve_membership():
    source = ["A", "B", "C", "D"]
    assert display_keys_for_step(source, "source", 9001, 1) == (source, None)
    first, token = display_keys_for_step(source, "seeded_shuffle", 9001, 1)
    again, repeat = display_keys_for_step(source, "seeded_shuffle", 9001, 1)
    other, _ = display_keys_for_step(source, "seeded_shuffle", 9002, 1)
    assert first == again and token == repeat
    assert set(first) == set(source) and len(first) == len(source)
    assert first != other


def test_fixture_resolver_no_five_link_cap_and_provenance(tmp_path):
    records, run_dir = steps(tmp_path, "source")
    first = records[0]
    assert len(first["choice_set"]) == 29 == first["extras"]["n_links_offered"]
    assert set(first["choice_set"][0]) == LINK_FIELDS
    # source order: shown order is the page order
    assert [row["original_position"] for row in first["choice_set"]] == list(range(29))
    assert [row["shuffled_position"] for row in first["choice_set"]] == list(range(29))
    assert first["choice_set"][0]["hyperlink"].startswith("https://en.wikipedia.org/wiki/")
    # the candidate list is logged once; the old copies are gone
    for field in REMOVED_DUPLICATES:
        assert field not in first
    for field in REMOVED_EXTRAS:
        assert field not in first["extras"]
    for field in ("display_order_policy", "display_shuffle_seed", "chosen_position", "chosen_original_position", "retry_attempts", "candidate_filter_provenance", "article_content_was_truncated"):
        assert field in first["extras"]
    chosen = first["choice_set"][first["extras"]["chosen_position"]]
    assert first["resolution"]["resolved_page_id"] == chosen["page_id"]
    for name in ("transcript.md", "pages.csv", "trajectory.jsonl", "summary.json", "metrics.json", "config.json", "run.json"):
        assert (run_dir / name).exists()
    assert (run_dir / "steps" / "step_001.json").exists()


def test_observation_has_full_clean_article_and_separate_candidate_table():
    fixture = FixtureBackend(ROOT / "tests/fixtures/wiki/fixture_pages.json")
    page = fixture.load_page("Civic Atlas")
    observation, truncated = page_view(page, candidate_rows(page, fixture))
    assert not truncated
    assert "synthetic development fixture page" in observation
    assert "[[" not in observation and "https://" not in observation
    assert "## Candidate links (29)" in observation
    assert "\n- Archive" in observation
    # nothing but titles reaches the model
    for logged_only in ("original_position", "shuffled_position", "displayed_position", "page_id", "hyperlink", "logging-only", "candidate_key"):
        assert logged_only not in observation


def test_piped_link_keeps_anchor_text_and_extracts_destination():
    fixture = FixtureBackend(ROOT / "tests/fixtures/wiki/fixture_pages.json")
    page = fixture.load_page("Anchor Text Example")
    link = page.links[0]
    table = [{"original_position": 0, "shuffled_position": 0, "title": link.destination,
              "hyperlink": "logging-only", "page_id": None, "semantic_cluster_label": []}]
    observation, _ = page_view(page, table)
    assert link.candidate_key == "Machine_learning"
    assert link.destination == "Machine learning"
    assert link.anchor_text == "machine learning"
    article = observation.split("## Candidate links", 1)[0]
    assert "machine learning" in article
    assert "[[" not in article and "https://" not in article


def test_realistic_mediawiki_markup_does_not_leak_into_article_text():
    raw = ("Before {{outer|{{inner|value}}}} [[File:Example.jpg|thumb|caption with [[Hidden page]]]] "
           "[[Machine learning|machine learning]] [https://example.test visible label] "
           "<ref name='x'>hidden citation</ref><span>ordinary prose</span> stray ]] {{")
    cleaned, truncated = clean_article_text(raw, None)
    assert not truncated
    assert "machine learning" in cleaned
    assert "visible label" in cleaned
    assert "ordinary prose" in cleaned
    for forbidden in ("[[", "]]", "{{", "}}", "http://", "https://", "<ref", "</ref", "<span", "File:", "Example.jpg"):
        assert forbidden not in cleaned


def test_null_article_limit_never_truncates_available_article_content():
    source = "word " * 10000
    cleaned, truncated = clean_article_text(source, None)
    assert cleaned == source.strip()
    assert not truncated


def test_source_and_shuffle_leave_article_and_membership_invariant(tmp_path, monkeypatch):
    # The prompt is no longer logged, so capture what the model actually receives.
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append(messages[-1]["content"])
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)

    source, _ = steps(tmp_path, "source", 9001, "source")
    source_prompt = sent[0]
    sent.clear()
    shuffled, _ = steps(tmp_path, "seeded_shuffle", 9001, "shuffled")
    shuffled_prompt = sent[0]
    source_first, shuffled_first = source[0], shuffled[0]

    assert source_prompt.split("## Candidate links", 1)[0] == shuffled_prompt.split("## Candidate links", 1)[0]
    assert sorted(row["original_position"] for row in shuffled_first["choice_set"]) == [row["original_position"] for row in source_first["choice_set"]]
    assert titles(shuffled_first) != titles(source_first)
    assert set(titles(shuffled_first)) == set(titles(source_first))
    # the model sees exactly the logged titles, in the logged (shuffled) order, and nothing else
    shown = [line[2:] for line in shuffled_prompt.split("## Candidate links", 1)[1].splitlines() if line.startswith("- ")]
    assert shown == titles(shuffled_first)
    for logged_only in ("original_position", "shuffled_position", "displayed_position", "page_id", "https://", "semantic_cluster_label"):
        assert logged_only not in shuffled_prompt


def test_shuffled_candidate_table_matches_displayed_order_and_choice(tmp_path):
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "table")
    for record in records:
        rows = record["choice_set"]
        assert [row["shuffled_position"] for row in rows] == list(range(len(rows)))
        assert sorted(row["original_position"] for row in rows) == list(range(len(rows)))
        chosen = rows[record["extras"]["chosen_position"]]
        assert chosen["title"] == record["parsed_choice"]
        assert chosen["original_position"] == record["extras"]["chosen_original_position"]
        assert set(chosen) == LINK_FIELDS


def test_dump_filenames_and_compressed_block_guard_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="custom-xml.bz2"):
        DumpBackend(tmp_path, tmp_path / "cache", "custom-xml.bz2", "custom-index.bz2", 1024)
    compressed = tmp_path / "synthetic-block.bz2"
    compressed.write_bytes(bz2.compress(b"x" * 10000))
    backend = object.__new__(DumpBackend)
    backend.xml_path = compressed
    backend.max_compressed_block_bytes = 10
    with pytest.raises(RuntimeError, match="max_compressed_block_bytes=10"):
        backend._stream(0)


def test_seeded_shuffle_is_logged_and_end_to_end_offline(tmp_path):
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "first")
    repeat, _ = steps(tmp_path, "seeded_shuffle", 9001, "repeat")
    different, _ = steps(tmp_path, "seeded_shuffle", 9002, "different")
    assert [titles(item) for item in records] == [titles(item) for item in repeat]
    assert titles(records[0]) != titles(different[0])
    for item in records:
        assert sorted(row["original_position"] for row in item["choice_set"]) == list(range(len(item["choice_set"])))
        assert item["extras"]["display_shuffle_seed"] is not None
        assert item["parsed_choice"] in titles(item)


def test_fixture_and_mock_need_no_external_api():
    fixture = FixtureBackend(ROOT / "tests/fixtures/wiki/fixture_pages.json")
    assert fixture.load_page("Civic Atlas").page_id == 1000
    response = MockProvider().chat([{"role": "user", "content": "- Archive"}], seed=1)
    assert response.error is None and "Archive" in response.text


def test_variant_b_searches_once_then_follows_links(tmp_path, monkeypatch):
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append(messages)
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    records, run_dir = steps(tmp_path, "seeded_shuffle", 9001, "b", "b_search_first", ["variant.n_runs=1"])
    first = records[0]
    assert first["step_index"] == 0 and first["action"] == "search_start"
    assert first["current_page_id"] is None and first["choice_set"] == []
    results = first["search"]["results"]
    assert results and first["search"]["query"]
    assert all(set(r) == RESULT_FIELDS for r in results)
    assert [r["shuffled_position"] for r in results] == list(range(len(results)))
    chosen = results[first["extras"]["chosen_position"]]
    assert chosen["title"] == first["parsed_choice"] == records[1]["current_page_title"]
    assert [r["action"] for r in records[1:]] == ["follow"] * (len(records) - 1)
    assert all(r["search"] is None for r in records[1:])
    assert len(records) == 1 + 3  # search step + step_limit link steps
    # the search prompt shows titles and snippets only
    results_prompt = next(m[-1]["content"] for m in sent if m[-1]["content"].startswith("Search results"))
    for logged_only in ("original_position", "shuffled_position", "page_id", "https://", "matched_title"):
        assert logged_only not in results_prompt
    assert (run_dir / "model_inputs" / "step_000_request_choose.txt").exists()


def test_variant_c_offers_search_or_follow_with_alternating_order(tmp_path):
    records, run_dir = steps(tmp_path, "seeded_shuffle", 9001, "c", "c_search_anytime", ["experiment.step_limit=4"])
    actions = [r["action"] for r in records]
    assert "search" in actions and "follow" in actions
    orders = [tuple(r["extras"]["tool_order"]) for r in records]
    assert set(orders) == {("follow", "search"), ("search", "follow")}
    assert all(a != b for a, b in zip(orders, orders[1:]))  # cyclic shift: alternates every step
    for record in records:
        assert set(record["choice_set"][0]) == LINK_FIELDS  # page links always logged
        if record["action"] == "search":
            results = record["search"]["results"]
            assert results and all(set(r) == RESULT_FIELDS for r in results)
            assert results[record["extras"]["chosen_position"]]["title"] == record["parsed_choice"]
        else:
            assert record["search"] is None
            assert record["choice_set"][record["extras"]["chosen_position"]]["title"] == record["parsed_choice"]
    trajectory = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["transition"] for row in trajectory] == ["search" if a == "search" else "link" for a in actions]


def test_variant_d_searches_first_then_offers_search_or_follow(tmp_path):
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "d", "d_search_first_anytime", ["variant.n_runs=1", "experiment.step_limit=4"])
    opening, later = records[0], records[1:]
    assert opening["step_index"] == 0 and opening["action"] == "search_start"
    assert opening["current_page_id"] is None and opening["search"]["results"]
    assert len(later) == 4  # the opening search does not consume the post-start step limit
    assert "search" in [record["action"] for record in later]
    assert "follow" in [record["action"] for record in later]
    assert all(record["extras"]["tool_order"] in (["follow", "search"], ["search", "follow"]) for record in later)


def test_unknown_variant_fails_loudly(tmp_path):
    cfg = config(tmp_path, variant="a_hub")
    cfg.variant.name = "z_unknown"
    with pytest.raises(ValueError, match="unknown variant"):
        run_experiment(cfg, ROOT)


@pytest.mark.parametrize("mode", ["path_only", "pages_no_links", "lead", "sliding_window", "self_notes", "full"])
def test_memory_modes_build_the_expected_history(tmp_path, monkeypatch, mode):
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append(messages)
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, mode, "a_hub", [f"experiment.memory={mode}", "experiment.memory_window=1", "experiment.step_limit=3"])
    assert len(records) == 3
    third = sent[2]  # the request at step 3 (two earlier steps in memory)
    roles = [m["role"] for m in third]
    assert roles[0] == "system" and roles[-1] == "user"
    current = third[-1]["content"]
    assert current.startswith("# ") or "earlier steps" in current
    if mode == "path_only":  # unchanged default: replies only
        assert roles == ["system", "assistant", "assistant", "user"]
    else:  # every other mode alternates user/assistant
        assert all(a != b for a, b in zip(roles[1:], roles[2:]))
    earlier = "\n".join(m["content"] for m in third[1:-1])
    if mode == "full":
        assert earlier.count("## Candidate links") == 2
    if mode == "pages_no_links":
        assert earlier.count("## Article content") == 2 and "## Candidate links" not in earlier
    if mode == "lead":
        assert "## Candidate links" not in earlier and "## Article content" not in earlier
    if mode == "sliding_window":  # window 1: one earlier page in full, the older one as a path line
        assert earlier.count("## Candidate links") == 1 and "Your earlier steps:" in earlier
    if mode == "self_notes":
        assert roles == ["system", "user"] and "Your earlier steps and notes:" in current and '"note"' in third[0]["content"]
        assert all(r["model_note"] for r in records)
    else:
        assert all(r["model_note"] is None for r in records)


def test_unknown_memory_mode_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown experiment.memory"):
        run_experiment(config(tmp_path, extra=["experiment.memory=everything"]), ROOT)


def record_requests(monkeypatch):
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append(messages)
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    return sent


def test_intent_is_off_by_default(tmp_path):
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "default")
    assert all(r["intent"] is None for r in records)


def test_intent_probe_is_logged_and_never_fed_back(tmp_path, monkeypatch):
    sent = record_requests(monkeypatch)
    plain, _ = steps(tmp_path, "seeded_shuffle", 9001, "plain")
    plain_first_choice = sent[0]
    sent.clear()
    records, run_dir = steps(tmp_path, "seeded_shuffle", 9001, "probe", "a_hub", ["experiment.intent=probe"])
    intent_calls = [m for m in sent if '"intent"' in m[0]["content"]]
    choice_calls = [m for m in sent if '"intent"' not in m[0]["content"]]
    assert len(intent_calls) == len(records) == len(choice_calls)
    # the intent call shows the page WITHOUT links
    assert all("## Candidate links" not in m[-1]["content"] and "## Article content" in m[-1]["content"] for m in intent_calls)
    # the choice call is exactly what it would be without the probe, and never contains an intent
    assert sent.index(choice_calls[0]) == 1 and choice_calls[0] == plain_first_choice
    assert all('"intent"' not in msg["content"] for m in choice_calls for msg in m)
    for record in records:
        intent = record["intent"]
        assert intent["parse_ok"] and intent["text"]
        if intent["in_links"]:
            assert record["choice_set"][intent["intended_shuffled_position"]]["title"] == intent["mapped_title"]
        assert intent["choice_matches_intent"] == (intent["mapped_title"] == record["parsed_choice"])
    assert (run_dir / "model_inputs" / "step_001_request_intent.txt").exists()


def test_intent_walk_in_variant_c_goes_where_the_intent_points(tmp_path):
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "walk", "c_search_anytime", ["experiment.intent=walk", "experiment.step_limit=4"])
    assert records and all(r["action"] == "intent_jump" and r["choice_set"] == [] for r in records)
    for record, following in zip(records, records[1:]):
        assert record["parsed_choice"] == record["intent"]["mapped_title"] == following["current_page_title"]


def test_intent_walk_needs_variant_c(tmp_path):
    with pytest.raises(ValueError, match="needs variant=c_search_anytime"):
        run_experiment(config(tmp_path, extra=["experiment.intent=walk"]), ROOT)


def record_requests_with_tools(monkeypatch):
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append((messages, [t["function"]["name"] for t in kwargs.get("tools") or []]))
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    return sent


def test_action_format_is_text_by_default(tmp_path, monkeypatch):
    sent = record_requests_with_tools(monkeypatch)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "text")
    assert all(not tools for _, tools in sent)
    assert "Reply with JSON only" in sent[0][0][0]["content"]
    assert all(r["extras"]["action_format"] == "text" and r["extras"]["tool_calls"] == [] for r in records)


def test_tools_mode_variant_a_offers_follow_and_stop(tmp_path, monkeypatch):
    sent = record_requests_with_tools(monkeypatch)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "ta", "a_hub", ["experiment.action_format=tools"])
    assert all(tools == ["follow_link", "stop"] for _, tools in sent)
    system = sent[0][0][0]["content"]
    assert "Reply with JSON" not in system and "calling one of the available tools" in system
    for record in records:
        assert record["extras"]["action_format"] == "tools"
        assert record["extras"]["tool_calls"][0]["name"] == "follow_link"
        assert record["parsed_choice"] == record["extras"]["tool_calls"][0]["arguments"]["title"]


def test_tools_mode_variant_b_searches_with_the_search_tool(tmp_path, monkeypatch):
    sent = record_requests_with_tools(monkeypatch)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "tb", "b_search_first", ["experiment.action_format=tools", "variant.n_runs=1"])
    assert sent[0][1] == ["search", "stop"] and sent[1][1] == ["follow_link", "stop"]
    roles = [m["role"] for m in sent[1][0]]
    assert roles[-2:] == ["assistant", "tool"] and sent[1][0][-2]["tool_calls"][0]["function"]["name"] == "search"
    assert records[0]["action"] == "search_start" and records[0]["search"]["results"]
    assert records[0]["parsed_choice"] in [r["title"] for r in records[0]["search"]["results"]]


def test_tools_mode_variant_c_alternates_tool_order(tmp_path, monkeypatch):
    sent = record_requests_with_tools(monkeypatch)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "tc", "c_search_anytime", ["experiment.action_format=tools", "experiment.step_limit=4"])
    action_calls = [tools for _, tools in sent if "search" in tools]
    assert {tuple(t) for t in action_calls} == {("follow_link", "search", "stop"), ("search", "follow_link", "stop")}
    assert all(a != b for a, b in zip(action_calls, action_calls[1:]))
    assert "search" in [r["action"] for r in records]
    for record in records:
        options = record["search"]["results"] if record["action"] == "search" else record["choice_set"]
        assert options[record["extras"]["chosen_position"]]["title"] == record["parsed_choice"]


def test_tools_mode_variant_d_searches_first_then_alternates_actions(tmp_path, monkeypatch):
    sent = record_requests_with_tools(monkeypatch)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "td", "d_search_first_anytime",
                       ["experiment.action_format=tools", "variant.n_runs=1", "experiment.step_limit=4"])
    assert sent[0][1] == ["search", "stop"] and sent[1][1] == ["follow_link", "stop"]
    action_calls = [tools for _, tools in sent[2:] if "search" in tools]
    assert {tuple(tools) for tools in action_calls} == {
        ("follow_link", "search", "stop"), ("search", "follow_link", "stop")}
    assert records[0]["action"] == "search_start"
    assert {record["action"] for record in records[1:]} == {"follow", "search"}


def test_memory_none_sends_only_the_current_page(tmp_path, monkeypatch):
    sent = record_requests(monkeypatch)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "none", "a_hub", ["experiment.memory=none"])
    assert len(records) == 3
    assert all([m["role"] for m in request] == ["system", "user"] for request in sent)


def test_unknown_action_format_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="unknown experiment.action_format"):
        run_experiment(config(tmp_path, extra=["experiment.action_format=xml"]), ROOT)


def test_openrouter_sends_tools_and_reads_tool_calls(monkeypatch):
    import requests
    from gap_attractor.providers.base import OpenAICompatibleProvider
    captured = {}
    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "follow_link", "arguments": '{"title": "Iliad", "reasoning": "x"}'}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    def fake_post(url, json=None, **kwargs):
        captured.update(url=url, payload=json); return FakeResponse()
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    provider = OpenAICompatibleProvider("openrouter", "openai/gpt-oss-120b", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    tools = [{"type": "function", "function": {"name": "follow_link", "description": "d", "parameters": {"type": "object"}}}]
    response = provider.chat([{"role": "user", "content": "hi"}], seed=1, temperature=0, top_p=0.5, tools=tools)
    assert captured["payload"]["tools"] == tools and captured["payload"]["provider"] == {"require_parameters": True}
    assert response.tool_calls == [{"id": "c1", "name": "follow_link", "arguments": {"title": "Iliad", "reasoning": "x"}}]


def test_tools_mode_self_notes_uses_a_note_argument(tmp_path, monkeypatch):
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append(kwargs.get("tools") or [])
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "tn", "a_hub", ["experiment.action_format=tools", "experiment.memory=self_notes"])
    assert all("note" in t["function"]["parameters"]["properties"] for tools in sent for t in tools)
    assert all(r["model_note"] for r in records)


def test_transcript_shows_reasoning_raw_output_intent_tools_and_true_start_page(tmp_path):
    hubs = tmp_path / "hubs.txt"
    hubs.write_text("Civic Atlas\nArchive\n", encoding="utf-8")
    records, run_dir = steps(tmp_path, "seeded_shuffle", 9001, "tr", "a_hub",
                             ["experiment.action_format=tools", "experiment.intent=probe",
                              f"variant.start_pages_file={hubs.as_posix()}", "variant.n_runs=1"])
    text = (run_dir / "transcript.md").read_text(encoding="utf-8")
    start = records[0]["current_page_title"]
    assert f"| Start page | {start} |" in text and "variant.start_pages_file" in text
    assert "**Reasoning:**" in text and "Raw model output" in text
    assert "**Tool called:** `follow_link`" in text
    assert "**Intent (asked before the links were shown):**" in text
    assert "Links shown (in order" in text
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["batch"]["start_pages"] == [start] and "start_pages_file" in config["batch"]["start_page_source"]


def test_transcript_shows_search_query_and_results(tmp_path):
    records, run_dir = steps(tmp_path, "seeded_shuffle", 9001, "trb", "b_search_first", ["variant.n_runs=1"])
    text = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert f"**Search query:** \"{records[0]['search']['query']}\"" in text
    assert "Search results shown (in order)" in text and "search (Variant B starts with a search)" in text


def test_internal_reasoning_is_read_from_the_service_reply():
    from gap_attractor.providers.base import parse_internal_reasoning
    assert parse_internal_reasoning({"content": "x", "reasoning": "I think Iliad is interesting."}) == "I think Iliad is interesting."
    assert parse_internal_reasoning({"reasoning_content": "deepseek style"}) == "deepseek style"
    assert parse_internal_reasoning({"reasoning_details": [{"type": "reasoning.text", "text": "a"}, {"summary": "b"}]}) == "a\nb"
    assert parse_internal_reasoning({"content": "x"}) is None


def test_max_tokens_comes_from_the_provider_config(monkeypatch):
    import requests
    from gap_attractor.providers.base import make_provider
    from omegaconf import OmegaConf
    captured = {}
    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": "{}", "reasoning": "thinking..."}}], "usage": {}}
    monkeypatch.setattr(requests, "post", lambda url, json=None, **kw: (captured.update(payload=json), FakeResponse())[1])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    provider = make_provider(OmegaConf.create({"name": "openrouter", "model": "deepseek/deepseek-v4.1-flash", "max_tokens": 4000}))
    response = provider.chat([{"role": "user", "content": "hi"}], seed=1, temperature=0, top_p=0.5)
    assert captured["payload"]["max_tokens"] == 4000
    assert response.internal_reasoning == "thinking..."


def test_internal_reasoning_is_logged_per_step_and_in_the_transcript(tmp_path, monkeypatch):
    from dataclasses import replace
    original_chat = MockProvider.chat
    def thinking_chat(self, messages, **kwargs):
        return replace(original_chat(self, messages, **kwargs), internal_reasoning="[mock] private thoughts")
    monkeypatch.setattr(MockProvider, "chat", thinking_chat)
    records, run_dir = steps(tmp_path, "seeded_shuffle", 9001, "think", "a_hub", ["experiment.intent=probe"])
    assert all(r["model_internal_reasoning"] == "[mock] private thoughts" for r in records)
    assert all(r["intent"]["internal_reasoning"] == "[mock] private thoughts" for r in records)
    # it is logged, never fed back: later requests must not contain it
    text = (run_dir / "transcript.md").read_text(encoding="utf-8")
    assert "Model's internal reasoning" in text


class CopycatProvider(MockProvider):
    """Behaves like the DeepSeek failure: if the history contains a JSON reply, it copies that format as text
    instead of calling a tool. Otherwise it calls the tool like the normal mock."""
    def chat(self, messages, **kwargs):
        copied = any(m["role"] == "assistant" and (m.get("content") or "").startswith('{"reasoning"') for m in messages)
        if kwargs.get("tools") and kwargs.get("tool_choice") != "none" and copied:
            titles = re.findall(r"^\s*-\s+(.+)$", messages[-1].get("content") or "", flags=re.M)
            return ProviderResponse(json.dumps({"reasoning": "copying", "chose": titles[0] if titles else "STOP"}),
                                    {}, "stop", 0, None, [])
        return super().chat(messages, **kwargs)


def use_provider(monkeypatch, cls):
    import gap_attractor.harness.runner as runner_module
    monkeypatch.setattr(runner_module, "make_provider", lambda cfg: cls())


import re
from gap_attractor.providers.base import ProviderResponse


def test_tools_mode_history_is_tool_calls_so_there_is_nothing_to_copy(tmp_path, monkeypatch):
    use_provider(monkeypatch, CopycatProvider)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "copy", "a_hub", ["experiment.action_format=tools", "experiment.step_limit=4"])
    assert len(records) == 4
    assert all(not r["invalid_action"]["is_invalid"] for r in records)      # no copied JSON, no retries
    assert all(r["extras"]["tool_calls"] for r in records)                  # every step was a real tool call


def test_tools_mode_history_layout(tmp_path, monkeypatch):
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append(messages)
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    steps(tmp_path, "seeded_shuffle", 9001, "layout", "a_hub", ["experiment.action_format=tools", "experiment.step_limit=3"])
    third = sent[2]
    assert [m["role"] for m in third] == ["system", "assistant", "tool", "assistant", "tool", "user"]
    assert all(m["tool_calls"][0]["function"]["name"] == "follow_link" for m in third if m["role"] == "assistant")
    assert all(m["content"].startswith("Opened: ") for m in third if m["role"] == "tool")
    assert not any('"chose"' in (m.get("content") or "") for m in third)


def test_tools_mode_variant_b_history_never_shows_an_unoffered_search_call(tmp_path, monkeypatch):
    sent = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        sent.append((messages, [t["function"]["name"] for t in kwargs.get("tools") or []]))
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    steps(tmp_path, "seeded_shuffle", 9001, "bhist", "b_search_first", ["experiment.action_format=tools", "variant.n_runs=1", "experiment.step_limit=3"])
    for messages, offered in sent[2:]:   # link steps after the opening search
        called = {c["function"]["name"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []
                  if not c["id"].startswith("call_")}  # past actions written by the harness
        assert called <= set(offered)


def test_retry_that_succeeds_does_not_end_the_run(tmp_path, monkeypatch):
    class FirstReplyUnreadable(MockProvider):
        def chat(self, messages, **kwargs):
            last = messages[-1].get("content") or ""
            if "could not be read" not in last and "call one of the available tools" not in last:
                self.calls += 1
                return ProviderResponse("not json at all", {}, "stop", 0, None, [])
            # the retry shows the unreadable reply and then the feedback; answer about the page (like a real model)
            return super().chat(messages[:-2], **kwargs)
    use_provider(monkeypatch, FirstReplyUnreadable)
    records, run_dir = steps(tmp_path, "seeded_shuffle", 9001, "retryok", "a_hub", ["experiment.step_limit=4"])
    assert len(records) == 4 and json.loads((run_dir / "summary.json").read_text())["termination_reason"] == "step_limit"
    assert all(r["invalid_action"]["harness_handling"] == "retry" for r in records)
    assert all(r["resolution"]["resolve_status"] == "success" for r in records)


def test_parse_reply_accepts_chose_as_well_as_choice():
    available = {"Iliad": {}, "Wool": {}}
    assert parse_reply('{"reasoning": "x", "chose": "Iliad"}', available)[0] == "Iliad"
    assert parse_reply('{"reasoning": "x", "choice": "Wool"}', available)[0] == "Wool"


def test_tools_mode_intent_call_answers_in_text_not_with_a_tool(tmp_path, monkeypatch):
    seen = []
    original_chat = MockProvider.chat
    def recording_chat(self, messages, **kwargs):
        seen.append(kwargs.get("tool_choice"))
        return original_chat(self, messages, **kwargs)
    monkeypatch.setattr(MockProvider, "chat", recording_chat)
    records, _ = steps(tmp_path, "seeded_shuffle", 9001, "tint", "a_hub", ["experiment.action_format=tools", "experiment.intent=probe"])
    assert "none" in seen
    assert all(r["intent"]["parse_ok"] for r in records)
