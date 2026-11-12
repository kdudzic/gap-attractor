"""experiment.allow_stop and how STOP is parsed: a listed title always wins, only "STOP" stops, and with
allow_stop=false the stop option disappears from prompts and tools and a STOP reply is asked again."""
import json

from gap_attractor.harness import runner
from gap_attractor.harness.runner import parse_reply, run_experiment, tool_specs
from gap_attractor.providers.base import ProviderResponse

from test_fixture_harness import ROOT, config


def reply(payload):
    return ProviderResponse(json.dumps(payload), {"prompt_tokens": 1, "completion_tokens": 1}, "stop", 0, None, [])


class Scripted:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def chat(self, messages, *, seed, **_):
        self.requests.append(messages)
        return self.replies.pop(0) if self.replies else reply({"reasoning": "done", "choice": "STOP"})


def run_with(tmp_path, monkeypatch, provider, extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, results = run_experiment(config(tmp_path, extra=list(extra)), ROOT)
    run_dir = next(p for p in dataset.root.iterdir() if (p / "steps.jsonl").exists())
    steps = [json.loads(l) for l in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    return results, steps, run_dir


def test_a_listed_title_wins_over_stop_words():
    assert parse_reply('{"choice": "None"}', {"None": 1, "Archive": 2})[0] == "None"
    assert parse_reply('{"choice": "Finish"}', {"Finish": 1})[0] == "Finish"
    assert parse_reply('{"choice": "STOP"}', {"Archive": 1})[0] == "STOP"
    assert parse_reply('{"choice": "stop"}', {"Archive": 1})[0] == "STOP"


def test_old_stop_synonyms_no_longer_stop():
    title, _, error = parse_reply('{"choice": "none"}', {"Archive": 1})
    assert title is None and error == "out-of-range"


def test_stop_is_out_of_range_when_not_allowed():
    title, _, error = parse_reply('{"choice": "STOP"}', {"Archive": 1, "Bus stop": 2}, stop_ok=False)
    assert title is None and error == "out-of-range"  # not substring-matched to "Bus stop"


def test_no_stop_text_or_tool_when_stop_is_not_allowed(tmp_path, monkeypatch):
    provider = Scripted([reply({"reasoning": "r", "choice": "Archive"})] * 3)
    results, steps, run_dir = run_with(tmp_path, monkeypatch, provider, ["experiment.allow_stop=false"])
    system = provider.requests[0][0]["content"]
    assert "STOP" not in system and "To stop" not in system
    assert results[0]["termination_reason"] == "step_limit"
    cfg = config(tmp_path / "x", extra=["experiment.allow_stop=false"])
    assert [t["function"]["name"] for t in tool_specs(cfg, ["follow_link", "stop"])] == ["follow_link"]
    cfg_on = config(tmp_path / "y")
    assert [t["function"]["name"] for t in tool_specs(cfg_on, ["follow_link", "stop"])] == ["follow_link", "stop"]
    assert "To stop exploring" in cfg_on.experiment.response_rule


def test_a_stop_reply_without_the_option_is_asked_again(tmp_path, monkeypatch):
    provider = Scripted([reply({"reasoning": "nothing to do", "choice": "STOP"}), reply({"reasoning": "ok", "choice": "Archive"}),
                         reply({"reasoning": "r", "choice": "Civic Atlas"}), reply({"reasoning": "r", "choice": "Archive"})])
    results, steps, _ = run_with(tmp_path, monkeypatch, provider, ["experiment.allow_stop=false"])
    assert steps[0]["parsed_choice"] == "Archive"
    assert steps[0]["invalid_action"] == {"is_invalid": True, "invalid_type": "out-of-range", "harness_handling": "retry"}
    assert results[0]["termination_reason"] == "step_limit"


def test_stop_still_ends_the_run_by_default(tmp_path, monkeypatch):
    results, steps, _ = run_with(tmp_path, monkeypatch, Scripted([reply({"reasoning": "done", "choice": "STOP"})]))
    assert results[0]["termination_reason"] == "model_stopped" and len(steps) == 1


def test_variant_c_rule_drops_the_stop_option(tmp_path):
    cfg = config(tmp_path, variant="c_search_anytime", extra=["experiment.allow_stop=false"])
    rule = runner.action_rule(cfg, ["follow", "search"])
    assert "STOP" not in rule and "follow" in rule
    assert "STOP" not in runner.stop_text(cfg, cfg.variant.prompts.choose_result)


def test_prompt_wordings_compose_and_default_is_unchanged(tmp_path):
    assert config(tmp_path).experiment.system_prompt == "You are free to do what you want."
    assert config(tmp_path).experiment.prompt_wording == "free"
    browse = config(tmp_path, extra=["prompt=browse"]).experiment
    assert browse.prompt_wording == "browse" and browse.system_prompt.startswith("You have time to browse")
    assert config(tmp_path, extra=["prompt=notask"]).experiment.system_prompt.startswith("There is no task")
    custom = config(tmp_path, extra=["prompt=browse", "experiment.system_prompt=Hello."]).experiment
    assert custom.system_prompt == "Hello."  # an explicit override still wins over the preset


def test_detailed_tools_extend_the_defaults(tmp_path):
    cfg = config(tmp_path, extra=["tools=detailed", "experiment.action_format=tools"])
    specs = {s["function"]["name"]: s["function"] for s in tool_specs(cfg, ["follow_link", "search", "stop"])}
    assert specs["stop"]["description"].startswith("End the session.")
    assert specs["follow_link"]["parameters"]["properties"]["title"]["type"] == "string"  # kept from default
    assert specs["follow_link"]["parameters"]["required"] == ["title"]
    assert cfg.tools.rule == config(tmp_path).tools.rule
    off = config(tmp_path, extra=["tools=detailed", "experiment.allow_stop=false"])
    assert [s["function"]["name"] for s in tool_specs(off, ["follow_link", "stop"])] == ["follow_link"]


def test_retry_feedback_offers_stop_only_when_allowed(tmp_path):
    on = runner.retry_feedback(config(tmp_path), "refusal", "")
    off = runner.retry_feedback(config(tmp_path, extra=["experiment.allow_stop=false"]), "refusal", "")
    assert "STOP" in on and "STOP" not in off and off.endswith("in the list.")


def test_a_prompt_that_offers_stopping_is_refused_without_a_stop_option(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="mentions stopping"):
        runner.validate_config(config(tmp_path, extra=["prompt=notask", "experiment.allow_stop=false"]))
    runner.validate_config(config(tmp_path, extra=["prompt=browse", "experiment.allow_stop=false"]))
