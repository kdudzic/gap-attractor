"""The harness's context_embedding=steps with a context file: the file's own turns (ids, kinds, authors) are embedded,
unchanged turns keep their vector, summaries get new ones, joint worlds record every author, and backfill works."""
import json

import numpy as np

import embed_contexts
import run as entry
from gap_attractor.harness.context_embed import load_trace, trace_summary
from gap_attractor.harness import runner

from helpers import Scripted, call, config, read_steps, reply

MOCK = ["context_embedding=steps", "context_embedding.embedder.kind=mock", "+context_embedding.embedder.dims=16"]


def run_with(tmp_path, monkeypatch, provider, extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, results = entry.run(config(tmp_path, ["ctx.tokenizer=chars", *extra]))
    return read_steps(dataset)


def lines_of(folder):
    return [json.loads(l) for l in (folder / "context_embedding" / "steps.jsonl").read_text().splitlines()]


def test_off_by_default(tmp_path, monkeypatch):
    steps, folder = run_with(tmp_path, monkeypatch, Scripted())
    run = json.loads((folder / "run.json").read_text())
    assert not (folder / "context_embedding").exists() and "context_embedding" not in run


def test_the_file_turns_are_embedded_every_step_and_reused_when_unchanged(tmp_path, monkeypatch):
    steps, folder = run_with(tmp_path, monkeypatch, Scripted(), MOCK)
    lines = lines_of(folder)
    assert [l["step"] for l in lines] == [1, 2, 3] and not any("error" in l for l in lines)
    assert [t["kind"] for t in lines[0]["turns"]][:2] == ["observation", "links"]     # the file, not the messages
    assert "move" in {t["kind"] for t in lines[1]["turns"]}                            # step 1's move, seen in step 2
    first = [t["key"] for t in lines[0]["turns"]]
    assert first == [t["key"] for t in lines[1]["turns"]][:len(first)]                 # nothing edited: same vectors
    ids = [t["id"] for t in lines[2]["turns"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)                            # stable turn ids
    run = json.loads((folder / "run.json").read_text())
    assert run["context_embedding"]["steps"] == 3 and run["context_embedding"]["texts_reused"] > 0
    assert "context_embedding" not in run["memory_settings"]
    t = load_trace(folder)
    assert t["file_direct"].shape == (3, 16) and np.allclose(np.linalg.norm(t["file_mean"], axis=1), 1, atol=1e-2)


def test_a_summary_gets_a_new_vector_and_the_summarised_turns_leave(tmp_path, monkeypatch):
    compact = lambda messages: reply(call("edit_context", 1, replace_turns=[{"first": 1, "last": 4, "summary": "Saw Civic Atlas."}]))
    provider = Scripted([reply(call("follow_link", 0, title="Archive")), compact])
    steps, folder = run_with(tmp_path, monkeypatch, provider, MOCK)
    lines = lines_of(folder)
    assert "summary" in {t["kind"] for t in lines[1]["turns"]}
    before = {t["key"] for t in lines[0]["turns"]}
    assert len(before & {t["key"] for t in lines[1]["turns"]}) < len(before)
    assert trace_summary(folder)[1]["kept_turns"] < 1.0


def test_joint_worlds_record_every_author(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    cfg = config(tmp_path, ["ctx=joint", "batch.start_pages=[Civic Atlas,Archive,Bridge]", "experiment.step_limit=2",
                            "ctx.tokenizer=chars", "ctx.joint.agent_providers=[{model: m1},{model: m2}]", *MOCK])
    dataset, results = entry.run(cfg)
    runs = [p.parent for p in dataset.root.rglob("run.json")]
    assert len(runs) == 3
    for r in runs:
        lines = lines_of(r)
        agent = json.loads((r / "run.json").read_text())["world"]["agent_id"]
        assert [l["step"] for l in lines] == [1, 2] and {l["agent"] for l in lines} == {agent}
        assert len({t["author"] for l in lines for t in l["turns"]}) >= 2                 # the shared file holds several


def test_backfill_from_snapshots(tmp_path, monkeypatch):
    steps, folder = run_with(tmp_path, monkeypatch, Scripted())                           # a run without embeddings
    embed_contexts.main([str(tmp_path), "embedder.kind=mock", "embedder.dims=16"])
    t = load_trace(folder)
    assert list(t["steps"]) == [1, 2, 3] and t["file_direct"].shape == (3, 16)
    summary = json.loads((folder / "context_embedding" / "summary.json").read_text())
    assert summary["source"].startswith("backfill") and summary["errors"] == 0
    embed_contexts.main([str(tmp_path), "embedder.kind=mock"])                            # second time: skipped
    assert len((folder / "context_embedding" / "steps.jsonl").read_text().splitlines()) == 3
