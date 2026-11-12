import pytest

from gap_attractor.analysis.source_claims import parse_stop_sources, validate_claimed_sources


def _step(page, choice=None, status="success"):
    return {"current_page_title": page, "parsed_choice": choice,
            "resolution": {"resolve_status": status}}


def test_claimed_sources_must_be_structured_json():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_stop_sources("I used Ozone depletion")


def test_claimed_sources_accept_only_pages_that_were_opened():
    run = {"start_page": "Spreadsheet", "outcome": {"model_stop_statement":
           '{"brief_complete": true, "sources": ["Montreal Protocol", "Ozone depletion"]}'}}
    claims, _, missing = validate_claimed_sources(
        run, [_step("Spreadsheet", "Montreal Protocol"), _step("Montreal Protocol", "Ozone depletion")]
    )
    assert claims == ["Montreal Protocol", "Ozone depletion"] and missing == []


def test_claimed_sources_flag_a_title_mentioned_but_never_opened():
    run = {"start_page": "Spreadsheet", "outcome": {"model_stop_statement":
           '{"brief_complete": true, "sources": ["Montreal Protocol", "Multilateral Fund"]}'}}
    _, _, missing = validate_claimed_sources(run, [_step("Spreadsheet", "Montreal Protocol")])
    assert missing == ["Multilateral Fund"]
