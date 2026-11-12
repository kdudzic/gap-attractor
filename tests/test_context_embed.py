"""context_embedding=steps (harness/context_embed.py): off by default, every step's final request embedded per turn
and as a whole, turns reused by text, a memory's own turns, and a failing embedder never ends a run."""
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("numpy")

from gap_attractor.analysis.embedding import MockEmbedder
from gap_attractor.harness.context_embed import ContextEmbedTrace, load_trace, pieces, trace_summary, turns_from_messages
from gap_attractor.harness.memory import MEMORIES, Memory
from gap_attractor.harness.runner import run_experiment

from test_fixture_harness import ROOT, config

ON = ["context_embedding=steps", "context_embedding.embedder.kind=mock", "+context_embedding.embedder.dims=16"]


def one_run(tmp_path, extra=(), variant="a_hub"):
    dataset, _ = run_experiment(config(tmp_path, variant=variant, extra=list(extra)), ROOT)
    return next(p for p in dataset.root.iterdir() if (p / "steps.jsonl").exists())


def lines_of(run_dir):
    return [json.loads(l) for l in (run_dir / "context_embedding" / "steps.jsonl").read_text().splitlines()]


def test_off_by_default(tmp_path):
    run_dir = one_run(tmp_path)
    assert not (run_dir / "context_embedding").exists()
    assert "context_embedding" not in json.loads((run_dir / "run.json").read_text())


@pytest.mark.parametrize("fmt", ["text", "tools"])
def test_every_step_is_embedded_and_the_system_prompt_is_reused(tmp_path, fmt):
    run_dir = one_run(tmp_path, [*ON, f"experiment.action_format={fmt}"])
    lines = lines_of(run_dir)
    assert [l["step"] for l in lines] == [1, 2, 3] and not any("error" in l for l in lines)
    assert lines[0]["turns"][0]["kind"] == "system"
    assert lines[0]["turns"][0]["key"] == lines[2]["turns"][0]["key"]          # same system text, same vector
    run = json.loads((run_dir / "run.json").read_text())
    assert run["context_embedding"]["steps"] == 3 and run["context_embedding"]["texts_reused"] > 0
    assert "memory_settings" not in run or "context_embedding" not in run["memory_settings"]
    t = load_trace(run_dir)
    assert t["file_mean"].shape == (3, 16) and t["file_direct"].shape == (3, 16)
    assert np.allclose(np.linalg.norm(t["file_direct"], axis=1), 1, atol=1e-2)
    assert trace_summary(run_dir)[0]["drift_start"] < 1e-3


def test_search_first_variant_embeds_the_opening_step_and_skips_steps_without_a_request(tmp_path):
    run_dir = one_run(tmp_path, ON, variant="b_search_first")
    steps = [json.loads(l) for l in (run_dir / "steps.jsonl").read_text().splitlines()]
    with_request = [s["step_index"] for s in steps if s["model_output_raw"] is not None]   # a dead end asks nothing
    assert [l["step"] for l in lines_of(run_dir)] == with_request and with_request[0] == 0


class Labelled(Memory):
    """A memory with turns of its own (as the context-file memory has)."""
    name = "labelled_turns_test"

    def messages(self, rule, history, observation):
        return [{"role": "system", "content": rule}, {"role": "user", "content": observation}]

    def record(self, item, step):
        pass

    def embed_turns(self, messages):
        return [{"id": 7, "kind": "page", "role": "user", "agent": 2, "text": messages[-1]["content"]}]


def test_a_memory_can_give_its_own_turns(tmp_path):
    MEMORIES.register("labelled_turns_test")(Labelled) if "labelled_turns_test" not in MEMORIES.names() else None
    run_dir = one_run(tmp_path, [*ON, "experiment.memory=labelled_turns_test"])
    turns = lines_of(run_dir)[0]["turns"]
    assert [(t["id"], t["kind"], t["author"]) for t in turns] == [(7, "page", 2)]


def test_a_failing_embedder_never_ends_the_run(tmp_path):
    class Broken(MockEmbedder):
        def embed(self, texts):
            raise RuntimeError("HTTP 503")
    cfg = OmegaConf.create({"turns": True, "file": "both", "max_input_chars": 50, "embedder": {"kind": "mock"}})
    trace = ContextEmbedTrace(cfg, tmp_path / "emb", embedder=Broken(OmegaConf.create({"dims": 4})))
    line = trace.record_request(1, [{"role": "user", "content": "x"}])
    assert "HTTP 503" in line["error"] and trace.summary()["errors"] == 1
    assert trace.record_request(2, None) is None                                 # no request in the step: nothing


def test_long_texts_are_embedded_in_pieces_and_tool_calls_become_text(tmp_path):
    text = "\n\n".join(f"paragraph {i} " + "word " * 8 for i in range(10))
    assert all(len(p) <= 50 for p in pieces(text, 50)) and sum(p.count("paragraph") for p in pieces(text, 50)) == 10
    turns = turns_from_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "follow_link", "arguments": '{"title": "Archive"}'}}]}])
    assert turns[0]["text"] == 'follow_link {"title": "Archive"}'
