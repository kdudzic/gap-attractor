"""A single agent with a context-file memory, through the normal harness (fixture pages, scripted model)."""
import json

import pytest

import run as entry
from gap_attractor.harness import runner

from helpers import Scripted, call, config, links_in, read_steps, reply, text_of

USAGE = {"prompt_tokens": 10, "completion_tokens": 2}


def run_with(tmp_path, monkeypatch, provider, extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, results = entry.run(config(tmp_path, ["ctx.tokenizer=chars", *extra]))
    return read_steps(dataset) + (results,)


def compact(first, last, summary="Was on Civic Atlas, went on."):
    return lambda messages: reply(call("edit_context", 1, replace_turns=[{"first": first, "last": last, "summary": summary}]))


def test_pages_notices_and_moves_are_the_context(tmp_path, monkeypatch):
    provider = Scripted()
    steps, folder, _ = run_with(tmp_path, monkeypatch, provider)
    third = provider.requests[2]["messages"]
    assert third[0]["role"] == "system" and "## Your context" in third[0]["content"]
    assert [m["role"] for m in third[1:]] == ["user", "assistant", "tool", "user", "assistant", "tool", "user"]
    assert third[1]["content"].startswith("[[TURN 1 role=user]]\n# Civic Atlas")
    assert "[[TURN 2 role=user]]\n## Candidate links" in third[1]["content"]          # the link list is its own turn
    assert "[[TURN 3 role=user]]\n[context: ~" in third[1]["content"]                  # and so is the notice
    assert third[2]["tool_calls"][0]["function"]["name"] == "follow_link"
    assert third[3]["content"].startswith("[[TURN 5 role=user]]\n# ") and "## Candidate links" in third[3]["content"]   # the page is the move's result
    assert provider.requests[0]["tools"] == ["follow_link", "stop", "edit_context"]
    assert all(s["memory"]["tokens_with_page"] > s["memory"]["tokens_before"] for s in steps)
    assert [s["memory"]["links_visible"] for s in steps] == [len(s["choice_set"]) for s in steps]
    assert (folder / "context" / "step_003.txt").exists() and (folder / "context" / "versions.jsonl").exists()
    run = json.loads((folder / "run.json").read_text())
    assert {k: run["memory_settings"][k] for k in ("interface", "budget", "nudges", "headers", "prompt", "max_edit_turns", "gate", "readout")} == \
        {"interface": "tool", "budget": 32768, "nudges": True, "headers": True, "prompt": "neutral", "max_edit_turns": 7, "gate": "fit", "readout": True}
    assert run["memory_settings"]["overflow"] == "wall"
    assert run["memory_settings"]["tokenizer"] == "calibrated" and run["memory_settings"]["count"] == "whole"
    file_text = (folder / "context" / "LIVE_CTX.txt").read_text()
    assert file_text.count("[followed link: ") == 3 and "follow_link:" not in file_text


def test_the_model_compacts_and_keeps_going(tmp_path, monkeypatch):
    provider = Scripted([reply(call("follow_link", 0, title="Archive")), compact(1, 4, "Saw a first page, went on.")])
    steps, folder, _ = run_with(tmp_path, monkeypatch, provider)
    edit = steps[1]["memory"]["edits"][0]
    assert edit["applied"] and edit["tokens_after"] < edit["tokens_before"] and edit["removed_turns"] == 4 and edit["ops"] == ["replace_turns 1..4"]
    again = provider.requests[2]["messages"]                              # the same step, asked again
    assert again[1]["content"].startswith("[[TURN 1 role=user]]\nSaw a first page")
    assert again[-2]["tool_calls"][0]["function"]["name"] == "edit_context" and again[-1]["role"] == "tool"
    assert again[-1]["content"].startswith("[[TURN") and "Applied" in again[-1]["content"]
    assert steps[1]["parsed_choice"] and not steps[1]["invalid_action"]["is_invalid"]
    versions = [json.loads(l) for l in (folder / "context" / "versions.jsonl").read_text().splitlines()]
    edit_version = next(v for v in versions if v["op"] == "edit")
    assert [r["kind"] for r in edit_version["removed"]] == ["observation", "links", "notice", "move"] and edit_version["added"][0]["kind"] == "summary"
    assert steps[1]["memory"]["pages_kept"] == 0.0 and steps[2]["memory"]["pages_kept"] == 0.5
    assert steps[1]["memory"]["path_recall"] == 1.0          # the Archive page itself names Civic Atlas


def test_growing_edits_are_rejected_when_over_target(tmp_path, monkeypatch):
    grow = reply(call("edit_context", 1, replace_turns=[{"first": 3, "last": 3, "summary": "x" * 5000}]))
    steps, _, _ = run_with(tmp_path, monkeypatch, Scripted([grow]), ["ctx.budget_tokens=1000", "ctx.reserve_tokens=100"])
    edit = steps[0]["memory"]["edits"][0]
    assert not edit["applied"] and edit["rejected"] == "gate"


def test_the_current_link_list_cannot_be_edited_until_the_agent_moves(tmp_path, monkeypatch):
    provider = Scripted([compact(2, 2, "gone")])
    steps, _, _ = run_with(tmp_path, monkeypatch, provider)
    edit = steps[0]["memory"]["edits"][0]
    assert not edit["applied"] and edit["rejected"] == "current_page"
    assert "list of links you can follow now" in provider.requests[1]["messages"][-1]["content"]
    assert "edit only turn 1, its text" in provider.requests[1]["messages"][-1]["content"]
    assert steps[0]["parsed_choice"] and not steps[0]["invalid_action"]["is_invalid"]


def test_the_page_just_read_can_be_summarised_at_once(tmp_path, monkeypatch):
    """Review N2: summarising the current page's text (not its links) is the most common compaction."""
    steps, folder, _ = run_with(tmp_path, monkeypatch, Scripted([compact(1, 1, "Civic Atlas: a fixture page.")]))
    assert steps[0]["memory"]["edits"][0]["applied"] and steps[0]["parsed_choice"]
    assert "Civic Atlas: a fixture page." in (folder / "context" / "LIVE_CTX.txt").read_text()


def test_deleting_the_move_before_the_current_page_is_not_mistaken_for_touching_it(tmp_path, monkeypatch):
    drop_move = lambda m: reply(call("edit_context", 1, delete_turns=[{"first": 4, "last": 4}]))   # turn 4 = the move
    steps, _, _ = run_with(tmp_path, monkeypatch, Scripted([reply(call("follow_link", 0, title="Archive")), drop_move]))
    assert steps[1]["memory"]["edits"][0]["applied"]


def test_invalid_edits_do_not_crash_and_change_nothing(tmp_path, monkeypatch):
    bad = [reply(call("edit_context", 1, delete_turns=["delete 1"])), reply(call("edit_context", 2, delete_turns=[{"first": "two", "last": 3}])),
           reply(call("edit_context", 3, replace_turns=[{"first": 1, "last": 2, "summary": "S"}], delete_turns=[{"first": 2, "last": 2}]))]
    steps, _, results = run_with(tmp_path, monkeypatch, Scripted(bad))
    edits = steps[0]["memory"]["edits"]
    assert [e["rejected"] for e in edits] == ["invalid"] * 3 and "overlap" in edits[2]["note"]
    assert results[0]["termination_reason"] == "step_limit"


def test_a_large_page_always_arrives_whole(tmp_path, monkeypatch):
    """Pages are never cut (hard rule): a page larger than the whole budget still arrives in full, with its links."""
    steps, folder, _ = run_with(tmp_path, monkeypatch, Scripted(), ["ctx.budget_tokens=400", "ctx.reserve_tokens=0", *WALL])
    assert not any(s["memory"]["capped"] for s in steps) and steps[0]["memory"]["nudge"] == "over"
    assert [s["memory"]["links_visible"] for s in steps] == [len(s["choice_set"]) for s in steps]


def test_a_retry_after_an_edit_sees_the_edited_context(tmp_path, monkeypatch):
    """Review M3: an edit plus an invalid move in one reply; the retry must show the post-edit file."""
    first = lambda m: reply(call("edit_context", 1, replace_text=[{"turn": 3, "old": "[context: ~", "new": "[SIZE: ~"}]),
                            call("follow_link", 2, title="Not A Link"))
    provider = Scripted([first])
    steps, _, _ = run_with(tmp_path, monkeypatch, provider)
    retry = text_of(provider.requests[1]["messages"])
    assert "[SIZE: ~" in retry and "Applied" in retry and "is not one of the listed titles" in retry
    assert steps[0]["invalid_action"]["invalid_type"] == "out-of-range" and steps[0]["parsed_choice"]


def test_an_edit_during_a_retry_keeps_the_retry_feedback(tmp_path, monkeypatch):
    provider = Scripted([reply(call("follow_link", 1, title="Not A Link")),
                         reply(call("edit_context", 2, delete_turns=[{"first": 3, "last": 3}]))])
    steps, _, _ = run_with(tmp_path, monkeypatch, provider)
    third = text_of(provider.requests[2]["messages"])
    assert "is not one of the listed titles" in third and "Applied" in third


def test_when_editing_turns_run_out_the_model_is_asked_to_move(tmp_path, monkeypatch):
    """Review M2: editing only, max_edit_turns times; then one request with the move tools only, no random link."""
    noop = lambda n: reply(call("edit_context", n, edits=[{"op": "replace_text", "old": "zzz-not-there", "new": "x"}]))
    provider = Scripted([noop(1), noop(2), noop(3)])
    steps, _, _ = run_with(tmp_path, monkeypatch, provider, ["experiment.step_limit=1", "ctx.max_edit_turns=3"])
    assert len(provider.requests) == 4 and provider.requests[3]["tools"] == ["follow_link", "stop"]
    assert "You have used your 3 edit_context calls before this move" in provider.requests[3]["messages"][-1]["content"]
    step = steps[0]
    assert not step["invalid_action"]["is_invalid"] and step["extras"]["side_turns_exhausted"] and len(step["extras"]["side_calls"]) == 3


def test_no_editing_interface(tmp_path, monkeypatch):
    provider = Scripted()
    steps, folder, _ = run_with(tmp_path, monkeypatch, provider, ["ctx_edit=none", "ctx.budget_tokens=1000000"])   # wall, never reached
    assert provider.requests[0]["tools"] == ["follow_link", "stop"]
    first = provider.requests[0]["messages"]
    assert "[[TURN" not in text_of(first) and "edit_context" not in first[0]["content"]   # no headers without an editing tool
    assert all(s["memory"]["nudge"] is None for s in steps)
    assert json.loads((folder / "run.json").read_text())["memory_settings"]["interface"] == "none"


def test_neutral_instructions_and_settings_are_recorded(tmp_path, monkeypatch):
    provider = Scripted()
    steps, folder, _ = run_with(tmp_path, monkeypatch, provider, ["ctx_prompt=neutral"])
    system = provider.requests[0]["messages"][0]["content"]
    assert "edit_context" in system and "what you want to do next" not in system
    assert json.loads((folder / "run.json").read_text())["memory_settings"]["prompt"] == "neutral"


def test_settings_it_cannot_work_with_are_refused(tmp_path):
    with pytest.raises(ValueError, match="action_format=tools"):
        entry.run(config(tmp_path, ["experiment.action_format=text"]))
    with pytest.raises(ValueError, match="supports variant=a_hub and b_search_first"):
        entry.run(config(tmp_path, ["variant=c_search_anytime"]))
    with pytest.raises(ValueError, match="needs ctx.headers"):
        entry.run(config(tmp_path, ["ctx.headers=false"]))


def test_settings_that_would_shorten_the_context_are_refused(tmp_path):
    """Hard rule (23 Oct 2026): the harness never shortens, elides or rewrites anything in an agent's context."""
    with pytest.raises(ValueError, match="shrink_edits is not allowed"):
        entry.run(config(tmp_path, ["ctx.shrink_edits=true"]))
    with pytest.raises(ValueError, match="overflow=soft is not allowed"):
        entry.run(config(tmp_path, ["ctx.overflow=soft"]))


def said(text, *calls, reasoning=None):
    from gap_attractor.providers.base import ProviderResponse
    return ProviderResponse(text, USAGE, "tool_calls", 0, None, [], list(calls), internal_reasoning=reasoning)


def test_verbatim_appends_the_replies_as_written(tmp_path, monkeypatch):
    provider = Scripted([said("Archives sound fun.", call("follow_link", 1, title="Archive", reasoning="old papers"), reasoning="hmm"),
                         said("Let me tidy up.", call("edit_context", 2, delete_turns=[{"first": 3, "last": 3}])),
                         said("Nowhere is good.", call("follow_link", 3, title="Nowhere")),
                         lambda m: reply(call("follow_link", 4, title=links_in(m)[0]))])
    steps, folder, _ = run_with(tmp_path, monkeypatch, provider, ["ctx.append=verbatim", "experiment.step_limit=3"])
    text = (folder / "context" / "LIVE_CTX.txt").read_text()
    assert 'Archives sound fun.\nfollow_link {"title": "Archive", "reasoning": "old papers"}' in text
    assert "hmm" not in text                                                          # no reasoning trace unless asked
    assert 'Let me tidy up.\nedit_context {"delete_turns": [{"first": 3, "last": 3}]}' in text
    assert 'Nowhere is good.\nfollow_link {"title": "Nowhere"}' in text and "is not one of the listed titles" in text  # the failed try stays
    third = provider.requests[-1]["messages"]
    assert any(m.get("content") == "[[TURN 3 role=assistant]]\nArchives sound fun." and m.get("tool_calls") for m in third)
    assert json.loads((folder / "run.json").read_text())["memory_settings"]["append"] == "verbatim"


def test_verbatim_can_keep_reasoning_traces(tmp_path, monkeypatch):
    provider = Scripted([said("", call("follow_link", 1, title="Archive"), reasoning="I want old papers")])
    _, folder, _ = run_with(tmp_path, monkeypatch, provider, ["ctx.append=verbatim", "ctx.append_reasoning=true", "experiment.step_limit=1"])
    assert "[thinking]\nI want old papers\n[/thinking]\nfollow_link" in (folder / "context" / "LIVE_CTX.txt").read_text()


WALL = ["ctx.reserve_tokens=0", "experiment.step_limit=4"]


def fixture_sizes(tmp_path, monkeypatch):
    """Per-step sizes of the fixture run with a budget too large to matter, so the hard-budget tests set their limits
    relative to them (and do not break when the prompts change)."""
    steps, _, _ = run_with(tmp_path / "sizes", monkeypatch, Scripted(), ["ctx.budget_tokens=100000", *WALL])
    return [s["memory"] for s in steps]


def tight(tmp_path, monkeypatch):
    """A limit the first page alone goes over (by about 25 tokens)."""
    return f"ctx.budget_tokens={fixture_sizes(tmp_path, monkeypatch)[0]['tokens_kept_with_page'] - 25}"


def test_hard_budget_a_move_over_the_limit_is_held_then_ends_the_run(tmp_path, monkeypatch):
    provider = Scripted()                                                    # never edits, always moves
    steps, folder, results = run_with(tmp_path, monkeypatch, provider, [tight(tmp_path, monkeypatch), *WALL])
    assert len(steps) == 1 and results[0]["termination_reason"] == "context_budget"
    memory = steps[0]["memory"]
    assert memory["nudge"] == "over" and memory["over_warning"] and not memory["capped"] and memory["over_limit_at_move"]
    assert "Over the limit" in provider.requests[0]["messages"][-1]["content"]      # the page arrived whole, then the warning
    held = steps[0]["extras"]["held_moves"]
    assert len(held) == 3 and len(memory["held_moves"]) == 3 and len(provider.requests) == 4
    answer = provider.requests[1]["messages"][-1]
    assert answer["role"] == "tool" and answer["content"].startswith("[[TURN 5 role=user]]\nNot moved: your context is ~")
    assert "last held move" in provider.requests[3]["messages"][-1]["content"]
    assert provider.requests[1]["tools"] == ["follow_link", "stop", "edit_context"]      # it can edit again
    doc = json.loads((folder / "run.json").read_text())
    assert doc["outcome"]["outcome_category"] == "B-memory" and doc["context_budget_end"]["tokens_kept"] > doc["context_budget_end"]["limit"]
    assert doc["context_budget_end"]["held_moves"] == 3 and doc["memory_settings"]["max_held_moves"] == 3


def test_hard_budget_summarising_the_page_keeps_the_run_going(tmp_path, monkeypatch):
    tidy = lambda m: reply(call("edit_context", 1, replace_turns=[{"first": 1, "last": 1, "summary": "Civic Atlas: a hub."}],
                                delete_turns=[{"first": 3, "last": 3}]))   # the article and the notice; links stay
    steps, _, results = run_with(tmp_path, monkeypatch, Scripted([tidy]), [tight(tmp_path, monkeypatch), *WALL])
    assert steps[0]["memory"]["nudge"] == "over" and steps[0]["memory"]["edits"][0]["applied"] and len(steps) > 1
    assert "held_moves" not in steps[0]["extras"] and steps[0]["memory"]["over_limit_at_move"] is False


def test_hard_budget_a_held_move_then_an_edit_then_the_move(tmp_path, monkeypatch):
    move = reply(call("follow_link", 1, title="Archive"))
    tidy = lambda m: reply(call("edit_context", 2, replace_turns=[{"first": 1, "last": 1, "summary": "Civic Atlas: a hub."}],
                                delete_turns=[{"first": 3, "last": 5}]))   # article; notice, held attempt and its answer
    provider = Scripted([move, tidy, reply(call("follow_link", 3, title="Archive"))])
    steps, folder, results = run_with(tmp_path, monkeypatch, provider, [tight(tmp_path, monkeypatch), *WALL])
    assert len(steps[0]["extras"]["held_moves"]) == 1 and steps[0]["parsed_choice"] == "Archive" and len(steps) > 1
    assert steps[0]["memory"]["edits"][0]["applied"] and steps[0]["memory"]["over_limit_at_move"] is False
    assert (folder / "model_inputs" / "step_001_request_held_1.txt").exists()


def test_hard_budget_crossed_by_the_models_own_turns_warns_at_the_next_page(tmp_path, monkeypatch):
    sizes = fixture_sizes(tmp_path, monkeypatch)        # over only once its own moves and notices have added up
    # a margin over step 2's size that only the next move, page and notice together exceed (step 3, the last page)
    steps, _, results = run_with(tmp_path, monkeypatch, Scripted(), [f"ctx.budget_tokens={sizes[1]['tokens_kept_at_move'] + 40}", *WALL])
    nudges = [s["memory"]["nudge"] for s in steps]
    assert nudges[-1] == "over" and "over" not in nudges[:-1] and all("held_moves" not in s["extras"] for s in steps[:-1])
    assert len(steps[-1]["extras"]["held_moves"]) == 3 and results[0]["termination_reason"] == "context_budget"


def test_a_hard_budget_needs_an_editing_interface_unless_it_cannot_be_reached(tmp_path):
    with pytest.raises(ValueError, match="needs an editing interface"):
        entry.run(config(tmp_path / "x", ["ctx_edit=none", "ctx.overflow=wall"]))


def test_a_stop_over_the_limit_is_held_then_flagged(tmp_path, monkeypatch):
    provider = Scripted([reply(call("stop", n, reasoning="cannot fit")) for n in range(4)])
    steps, folder, results = run_with(tmp_path, monkeypatch, provider, [tight(tmp_path, monkeypatch), *WALL])
    doc = json.loads((folder / "run.json").read_text())
    held = steps[0]["extras"]["held_moves"]
    assert [h["call"]["name"] for h in held] == ["stop"] * 3 and held[0]["note"].startswith("Not stopped:")
    assert results[0]["termination_reason"] == "model_stopped" and doc["stopped_over_limit"]["held_moves"] == 3
    assert "context_budget_end" not in doc
    assert "call follow_link anyway: the move is held" in provider.requests[0]["messages"][0]["content"]


def test_calls_stay_as_made_and_are_written_after_they_have_run(tmp_path, monkeypatch):
    """Nothing the model wrote is shortened (hard rule), and an editing call is written into the file only once it has
    run, followed at once by its result (atomic): the edit it made comes first in the version log."""
    long = ("Civic Atlas, a hub page. " * 20).strip()
    provider = Scripted([said("Summing up.", call("edit_context", 1, replace_turns=[{"first": 1, "last": 1, "summary": long}])),
                         reply(call("follow_link", 2, title="Archive"))])
    steps, folder, _ = run_with(tmp_path, monkeypatch, provider, ["ctx.append=verbatim", "experiment.step_limit=2"])
    text = (folder / "context" / "step_001.txt").read_text()
    assert text.count(long) == 2                         # the summary in turn 1, and the call as made, kept in full
    versions = [json.loads(l) for l in (folder / "context" / "versions.jsonl").read_text().splitlines()]
    ops = [(v["op"], [a["kind"] for a in v.get("added") or []]) for v in versions]
    i = next(k for k, (op, _) in enumerate(ops) if op == "edit")
    assert ops[i + 1] == ("append", ["call"]) and ops[i + 2] == ("append", ["result"])
    assert "shrink" not in [v["op"] for v in versions]
    assert json.loads((folder / "run.json").read_text())["memory_settings"]["shrink_edits"] is False
