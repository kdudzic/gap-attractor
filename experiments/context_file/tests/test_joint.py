"""Joint worlds: several agents (and models) over ONE context file, and the controls that take sharing away."""
import json

import run as entry
from gap_attractor.harness import runner

from helpers import Scripted, call, config, links_in, reply, text_of

STARTS = "batch.start_pages=[Civic Atlas,Archive,Bridge]"


def joint(tmp_path, monkeypatch, providers, extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    cfg = config(tmp_path, ["ctx=joint", STARTS, "experiment.step_limit=2", "ctx.tokenizer=chars",
                            "ctx.joint.agent_providers=[{model: m1},{model: m2}]", *extra])
    dataset, results = entry.run(cfg)
    runs = {json.loads(p.read_text())["world"]["agent_id"]: p.parent for p in dataset.root.rglob("run.json")}
    return dataset, results, runs


def steps_of(folder):
    return [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines()]


def versions(dataset, name="versions.jsonl"):
    return [json.loads(l) for l in (dataset.root / "worlds" / "world_001" / name).read_text().splitlines()]


def test_one_file_for_all_agents_and_models(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    dataset, results, runs = joint(tmp_path, monkeypatch, providers)
    world = json.loads((dataset.root / "worlds" / "world_001" / "world.json").read_text())
    assert world["models"] == ["m1", "m2", "m1"] and world["start_pages"] == ["Civic Atlas", "Archive", "Bridge"]
    assert world["ended"] == {"1": {"round": 3, "reason": "step_limit"}, "2": {"round": 3, "reason": "step_limit"},
                              "3": {"round": 3, "reason": "step_limit"}}
    assert len(results) == 3 and all(r["n_steps"] == 2 for r in results)
    docs = [json.loads((runs[k] / "run.json").read_text()) for k in (1, 2, 3)]
    assert [d["config"]["provider"]["model"] for d in docs] == ["m1", "m2", "m1"]
    assert {d["config"]["world"]["name"] for d in docs} == {"joint_labelled"} and docs[0]["world"]["models"] == ["m1", "m2", "m1"]
    assert docs[0]["memory_settings"]["sharing"] == "full" and docs[0]["memory_settings"]["attribution"] == "labelled"
    final = (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()
    assert final.count("[followed link: ") == 6 and final.count("\n# ") + final.startswith("[[TURN 1 role=user agent=") >= 6
    first, second = world["rounds"][0][:2]
    later = next(r for p in providers.values() for r in p.requests if f"you are\nagent {second}." in r["messages"][0]["content"]
                 or f"you are agent {second}." in r["messages"][0]["content"])
    text = text_of(later["messages"][1:])
    assert f"role=user agent={first}]]" in text and f"role=user agent={second}]]" in text and "(you)" not in text
    assert {v["agent"] for v in versions(dataset)} == {1, 2, 3}
    memory = steps_of(runs[1])[0]["memory"]
    assert memory["round"]["round"] == 1 and memory["agent"] == 1 and "named_by_others" in memory
    assert 0 <= memory["menu_named_by_others"] <= 1


def test_a_destructive_edit_is_what_the_next_agent_sees_and_is_attributed_to_its_editor(tmp_path, monkeypatch):
    def wipe(messages):     # agent 1, round 2: everything but its current page (the last page turn and notice)
        n = text_of(messages[1:]).count("[[TURN")
        return reply(call("edit_context", 1, edits=[{"op": "replace_turns", "first": 1, "last": n - 2, "summary": "agent 1 cleared the context"}]))
    m1 = Scripted([reply(call("follow_link", 0, title="Archive")), wipe], model="m1")
    providers = {"m1": m1, "m2": Scripted(model="m2")}
    dataset, _, runs = joint(tmp_path, monkeypatch, providers, ["ctx.joint.order=fixed", "ctx.joint.agent_providers=[{model: m1},{model: m2},{model: m2}]"])
    round2_agent2 = text_of(providers["m2"].requests[2]["messages"])          # agent 2, round 2 (after agent 1's wipe)
    assert "agent 1 cleared the context" in round2_agent2 and "agent=1]]\n# Civic Atlas" not in round2_agent2
    wipe_version = next(v for v in versions(dataset) if v["op"] == "edit")
    assert wipe_version["agent"] == 1 and wipe_version["chars_after"] < wipe_version["chars_before"]
    assert {r["agent"] for r in wipe_version["removed"]} == {1, 2, 3} and wipe_version["added"] == [{"id": wipe_version["added"][0]["id"], "agent": 1, "kind": "summary",
                                                                                         "origin": [r["id"] for r in wipe_version["removed"]]}]
    edit = steps_of(runs[1])[1]["memory"]["edits"][0]
    assert edit["removed_others"] > 0 and set(edit["removed_by_agent"]) == {"1", "2", "3"}
    assert "role=user agent=1]]\nagent 1 cleared the context" in (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()


def test_others_turns_can_be_made_read_only(tmp_path, monkeypatch):
    def wipe(messages):
        return reply(call("edit_context", 1, delete_turns=[{"first": 1, "last": 2}]))   # agent 1's own first page
    def wipe_other(messages):
        return reply(call("edit_context", 2, delete_turns=[{"first": 4, "last": 5}]))   # its own move and agent 2's page
    providers = {"m1": Scripted([reply(call("follow_link", 0, title="Archive")), wipe_other, wipe], model="m1"), "m2": Scripted(model="m2")}
    dataset, _, runs = joint(tmp_path, monkeypatch, providers, ["ctx.joint.order=fixed", "ctx.joint.sharing=readonly_others",
                                                                "ctx.joint.agent_providers=[{model: m1},{model: m2},{model: m2}]"])
    edits = steps_of(runs[1])[1]["memory"]["edits"]
    assert [e["rejected"] for e in edits] == ["others_turns", None] and edits[1]["applied"]
    assert "only the turns you added" in providers["m1"].requests[0]["messages"][0]["content"]
    assert json.loads((runs[1] / "run.json").read_text())["config"]["world"]["name"] == "joint_labelled_readonly_others"


def test_separate_files_under_the_same_scheduler(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    dataset, _, runs = joint(tmp_path, monkeypatch, providers, ["ctx.joint.sharing=separate"])
    world = dataset.root / "worlds" / "world_001"
    assert sorted(p.name for p in world.glob("LIVE_CTX_agent_*.txt")) == ["LIVE_CTX_agent_1.txt", "LIVE_CTX_agent_2.txt", "LIVE_CTX_agent_3.txt"]
    for r in providers["m2"].requests:                                         # agent 2 never sees another page
        moves = sum(1 for m in r["messages"] if m.get("tool_calls"))
        assert text_of(r["messages"][1:]).count("## Article content") == moves + 1
    assert "shared by" not in providers["m2"].requests[0]["messages"][0]["content"]


def test_anonymous_attribution_and_the_fixed_role_control(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    joint(tmp_path / "a", monkeypatch, providers, ["ctx.attribution=none", "ctx.joint.order=fixed"])
    last = providers["m1"].requests[-1]["messages"]                          # agent 3, round 2
    assert "agent=" not in text_of(last[1:]) and "[agent" not in text_of(last[1:])
    assert sum(1 for m in last if m.get("tool_calls")) == 3                  # every agent's move reads as its own tool call
    # (round 1's three moves: a move is written with its page, at its maker's next turn)
    assert "turns do not say which agent" in last[0]["content"]
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    _, _, runs = joint(tmp_path / "b", monkeypatch, providers, ["ctx.attribution=none", "ctx.others_as=user", "ctx.joint.order=fixed"])
    last = providers["m1"].requests[-1]["messages"]
    assert sum(1 for m in last if m.get("tool_calls")) == 1                  # only its own move; others' are user text
    assert json.loads((runs[1] / "run.json").read_text())["memory_settings"]["others_as"] == "user"


def test_a_link_from_another_agents_page_is_out_of_range(tmp_path, monkeypatch):
    seen = {}
    def foreign(messages):          # agent 2 clicks a link of agent 1's page that is not on its own page
        own = set(links_in(messages))
        other = next(t for t in seen["a1"] if t not in own)
        return reply(call("follow_link", 5, title=other))
    def remember(messages):
        seen["a1"] = links_in(messages)
        return reply(call("follow_link", 1, title=seen["a1"][0]))
    providers = {"m1": Scripted([remember], model="m1"), "m2": Scripted([foreign], model="m2")}
    dataset, _, runs = joint(tmp_path, monkeypatch, providers, ["ctx.joint.order=fixed"])
    step = steps_of(runs[2])[0]
    assert step["invalid_action"]["invalid_type"] == "out-of-range" and step["invalid_action"]["harness_handling"] == "retry"
    failure = json.loads((runs[2] / "failures.jsonl").read_text().splitlines()[0])
    assert failure["category"] == "others_menu" and failure["matched"]["source"].startswith("agent 1")


def test_order_is_seeded_and_changes_between_rounds(tmp_path, monkeypatch):
    def rounds(path):
        providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
        dataset, _, _ = joint(path, monkeypatch, providers, ["experiment.step_limit=4"])
        return json.loads((dataset.root / "worlds" / "world_001" / "world.json").read_text())["rounds"]
    a, b = rounds(tmp_path / "a"), rounds(tmp_path / "b")
    assert a == b and len({tuple(r) for r in a[:4]}) > 1


def test_leftover_start_pages_are_recorded(tmp_path, monkeypatch):
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    dataset, _, _ = joint(tmp_path, monkeypatch, providers, ["batch.start_pages=[Civic Atlas,Archive,Bridge,Library]", "experiment.step_limit=1"])
    assert json.loads((dataset.root / "worlds" / "world_001" / "world.json").read_text())["unused_start_pages"] == ["Library"]


def test_every_agent_on_the_default_model(tmp_path, monkeypatch):
    provider = Scripted()
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: provider)
    dataset, results = entry.run(config(tmp_path, ["ctx=joint", STARTS, "experiment.step_limit=1"]))
    assert len(results) == 3 and len({r["model"] for r in results}) == 1


def test_a_joint_batch_is_never_resumed_half_way(tmp_path, monkeypatch):
    import pytest
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    with pytest.raises(ValueError, match="not supported for joint worlds"):
        joint(tmp_path, monkeypatch, providers, ["batch.resume=true"])


def test_moves_are_atomic_each_move_is_followed_by_the_page_it_opened(tmp_path, monkeypatch):
    """23 Oct 2026: a move is written together with the page it opened (at its maker's next turn), so no other
    agent's turn ever comes between them, and the page is sent as the move's tool result."""
    providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
    dataset, _, _ = joint(tmp_path, monkeypatch, providers, ["experiment.step_limit=3"])
    versions = [json.loads(l) for l in (dataset.root / "worlds" / "world_001" / "versions.jsonl").read_text().splitlines()]
    turns = {}
    order = []
    for v in versions:
        for a in v.get("added") or []:
            turns[a["id"]] = a
            order.append(a["id"])
    moves = [i for i, tid in enumerate(order) if turns[tid]["kind"] == "move"]
    assert moves
    for i in moves:
        agent = turns[order[i]]["agent"]
        nxt = turns[order[i + 1]] if i + 1 < len(order) else None
        if nxt is None or nxt["kind"] != "observation" or nxt["agent"] != agent:
            # only a last move, whose session ended before its page opened, has no page after it
            assert not any(turns[t]["agent"] == agent for t in order[i + 1:])
        else:
            assert nxt["kind"] == "observation" and nxt["agent"] == agent
    for p in providers.values():
        for r in p.requests:
            for k, m in enumerate(r["messages"]):
                if m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "follow_link":
                    result = r["messages"][k + 1]
                    assert result["role"] == "tool" and ("## Candidate links" in result["content"] or result["content"].startswith("Opened"))
