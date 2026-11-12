"""state_chain (binding by state): every generated answer is what the files say, the layout and the gate, and in a
world (Docker) an item's files are in its agent's sandbox only while the item is open."""
import collections
import csv
import hashlib
import io
import json
import shutil
import subprocess

import pytest
from omegaconf import OmegaConf

from helpers import Scripted, factory, read_jsonl, tool
from siam.tasks import state_chain
from siam.world import World
import run as runner


def solve(item):
    """An independent solution of a base item from its files."""
    f, kind = item["files"], item["kind"]
    if kind == "ip":
        fails = collections.Counter(l.split(" from ")[1].split()[0] for l in f["auth.log"].splitlines() if "Failed password" in l)
        return fails.most_common(1)[0][0]
    if kind == "sku":
        rows = list(csv.DictReader(io.StringIO(f["inventory.csv"])))
        return max(rows, key=lambda r: int(r["quantity"]) * float(r["unit_price"]))["sku"]
    if kind == "port":
        service = item["prompt"].split("does the ")[1].split(" service")[0]
        port = None
        for name in f["order.txt"].split():
            for line in f[f"conf.d/{name}"].splitlines():
                if line.startswith(f"{service}.port = "):
                    port = line.split(" = ")[1]
        return port
    if kind == "file":
        prefix = item["prompt"].split("starting with ")[1].split("?")[0]
        return next(p.split("/", 1)[1] for p, t in f.items() if hashlib.sha256(t.encode()).hexdigest().startswith(prefix))
    if kind == "user":
        open_, best = {}, {}
        for e in map(json.loads, f["events.jsonl"].splitlines()):
            if e["event"] == "login":
                open_[e["user"]] = e["t"]
            else:
                best[e["user"]] = max(best.get(e["user"], 0), e["t"] - open_.pop(e["user"]))
        return max(best, key=best.get)


def lookup(item, value):
    (name, text), = item["files"].items()
    rows = list(csv.reader(io.StringIO(text)))[1:]
    hits = [r for r in rows if value in r]
    assert len(hits) == 1 and len(rows) == state_chain.N_TABLE
    return hits[0][0] if item["kind"] == "ip" else hits[0][1]


@pytest.mark.parametrize("seed", range(1, 9))
def test_every_answer_follows_from_the_files(seed):
    streams, manifest = state_chain.build(2, seed, 12)
    items = {it["id"]: it for st in streams for it in st}
    for it in items.values():
        if it["role"] in ("base", "control_base"):
            assert solve(it) == it["answer"]
        else:
            assert lookup(it, it["needs"]) == it["answer"]
            assert (it["needs"] in it["prompt"]) == (it["role"] == "control_dep")   # the value is given only in controls
    for L in manifest["links"]:
        assert items[L["dependent"]["id"]]["needs"] == items[L["base"]["id"]]["answer"]


def test_layout_links_and_gate():
    streams, manifest = state_chain.build(2, 3, 12, gate=True)
    roles = [[it["role"] for it in st] for st in streams]
    assert roles[0] == roles[1] == ["base", "base", "control_base", "dependent", "dependent", "control_dep"] * 2
    types = collections.Counter(L["type"] for L in manifest["links"])
    assert types == {"forward_self": 4, "forward_partner": 4}
    for L in manifest["links"]:
        same = L["type"] == "forward_self"
        assert (L["base"]["agent"] == L["dependent"]["agent"]) == same
        assert L["dependent"]["position"] - L["base"]["position"] == (3 if same else 3)
        dep = next(it for st in streams for it in st if it["id"] == L["dependent"]["id"])
        assert ("after" in dep) == (not same)
        assert ("agent " in dep["prompt"]) == (not same)
    ids = [it["id"] for st in streams for it in st]
    assert len(ids) == len(set(ids))
    assert state_chain.build(2, 3, 12, gate=True)[1] == manifest          # reproducible
    assert all("after" not in it for st in state_chain.build(2, 3, 12)[0] for it in st)


def test_grading_is_lenient_only_about_quotes_and_case():
    task = state_chain.StateChainTask()
    item = {"answer": "SKU-AB12C"}
    assert task.grade(item, " `sku-ab12c`. ").success
    assert not task.grade(item, "SKU-AB12").success


def test_files_need_bash_editing(tmp_path):
    cfg = runner.compose(["world.setup=external", "world.n_agents=2", "ctx.edit=tool"])
    OmegaConf.set_struct(cfg, False)
    streams, _ = state_chain.build(2, 1, 6)
    with pytest.raises(ValueError, match="ctx.edit=bash and ctx.bash.workspace=kept"):
        World(cfg, 1, streams, tmp_path / "w", state_chain.StateChainTask(), factory(Scripted([]), Scripted([])), label="t")


def sandbox_image():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "gap-ctx-sandbox:1"], capture_output=True).returncode == 0


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
def test_files_are_there_only_while_the_item_is_open(tmp_path):
    streams, _ = state_chain.build(2, 2, 6)
    first, second = streams[0][0], streams[0][1]
    (name, _), *_ = sorted(first["files"].items())
    a1 = Scripted([tool("bash", command=f"ls /work/env; cat /work/env/{first['slug']}/{name} | wc -l"),
                   tool("submit", answer=first["answer"]),
                   tool("bash", command="ls /work/env"),
                   tool("submit", answer="wrong")])
    a2 = Scripted([tool("submit", answer="x"), tool("submit", answer="y")])
    cfg = runner.compose(["world.setup=external", "world.order=fixed", "world.rounds=2", "world.identity=names",
                          "world.n_agents=2", "ctx.edit=bash", "ctx.atomic=true", "ctx.budget_tokens=60000",
                          "ctx.readout_by_agent=false", "world.max_work_calls=1", "output.save_requests=true"])
    OmegaConf.set_struct(cfg, False)
    w = World(cfg, 1, streams, tmp_path / "w", state_chain.StateChainTask(), factory(a1, a2), label="t")
    w.run()
    edits = [e for e in read_jsonl(tmp_path / "w" / "edits.jsonl") if e["agent"] == 1]
    context = (tmp_path / "w" / "context.txt").read_text()
    assert first["slug"] in context and f"/work/env/{second['slug']}" not in context.split("[submit]")[0]
    results = read_jsonl(tmp_path / "w" / "results.jsonl")
    mine = [r for r in results if r["agent"] == 1]
    assert [r["passed"] for r in mine] == [True, False]
    assert mine[0]["role"] == "base" and mine[0]["link_type"] == "forward_self"
    env = read_jsonl(tmp_path / "w" / "env.jsonl")
    assert [(e["agent"], e["item"], e["event"]) for e in env if e["agent"] == 1][:3] == [
        (1, first["id"], "put"), (1, first["id"], "removed"), (1, second["id"], "put")]
    # the second listing shows the new item's folder only: the first item's files are gone
    out = [t for t in json.loads(json.dumps(read_jsonl(tmp_path / "w" / "versions.jsonl")))
           for t in (t.get("turns_set") if isinstance(t.get("turns_set"), list) else []) if t.get("kind") == "result"]
    listings = [t["text"] for t in out if t.get("agent") == 1 and "ls /work/env" not in t["text"] and "[exit" in t["text"]]
    assert any(second["slug"] in x and first["slug"] not in x for x in listings)
    assert "Each task takes one submit." in a1.seen[0][0]["content"]


def test_kinds_are_balanced_in_every_slot_type():
    streams, manifest = state_chain.build(2, 1, 12)
    items = {it["id"]: it for st in streams for it in st}
    links = collections.Counter(items[L["base"]["id"]]["kind"] for L in manifest["links"])
    assert max(links.values()) - min(links.values()) <= 1 and len(links) == 5        # 8 links over 5 kinds


def test_a_dependents_answer_is_in_one_row_only():
    for seed in range(1, 40):
        for st in state_chain.build(2, seed, 12)[0]:
            for it in st:
                if it["role"] in ("dependent", "control_dep"):
                    (text,) = it["files"].values()
                    col = 0 if it["kind"] == "ip" else 1
                    answers = [r[col] for r in list(csv.reader(io.StringIO(text)))[1:]]
                    assert answers.count(it["answer"]) == 1


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
def test_odd_commands_do_not_end_the_world_and_values_are_located(tmp_path):
    streams, _ = state_chain.build(2, 2, 6)
    s1 = streams[0]
    a1 = Scripted([tool("bash", command=f"chmod -R a-w /work/env/{s1[0]['slug']}"), tool("submit", answer=s1[0]["answer"]),
                   tool("bash", command="kill -9 -1"), tool("submit", answer="x"),
                   tool("bash", command=f"echo {s1[0]['answer']} > /tmp/kept; rm -rf /work/env; touch /work/env"),
                   tool("submit", answer="y"), tool("submit", answer="z")])
    a2 = Scripted([tool("submit", answer="x")] * 4)
    cfg = runner.compose(["world.setup=external", "world.order=fixed", "world.rounds=4", "world.identity=names",
                          "world.n_agents=2", "ctx.edit=bash", "ctx.atomic=true", "ctx.budget_tokens=60000",
                          "ctx.readout_by_agent=false", "world.max_work_calls=1"])
    OmegaConf.set_struct(cfg, False)
    w = World(cfg, 1, streams, tmp_path / "w", state_chain.StateChainTask(), factory(a1, a2), label="t")
    w.run()
    assert json.loads((tmp_path / "w" / "world.json").read_text())["status"] == "complete"
    env = [e for e in read_jsonl(tmp_path / "w" / "env.jsonl") if e["agent"] == 1]
    assert [e["event"] for e in env][:6] == ["put", "removed"] * 3        # read-only, a killed container, a file at /work/env
    dep = next(e for e in env if e["item"] == s1[3]["id"] and e["event"] == "put")    # the forward_self dependent
    assert dep["value_in_context"] and dep["value_in_sandbox"] == ["/tmp/kept"] and dep["base_closed"]
    work = read_jsonl(tmp_path / "w" / "work.jsonl") if (tmp_path / "w" / "work.jsonl").exists() else []
    assert not any(r["file"].startswith("env/") for r in work)


def test_the_hint_is_one_sentence_and_off_by_default():
    plain, hinted = state_chain.StateChainTask(), state_chain.StateChainTask(hint=True)
    assert hinted.preamble() == plain.preamble() + " " + state_chain.StateChainTask.HINT
    assert "Later tasks" not in plain.preamble()


def test_adversarial_text_is_private_to_agent_1_and_streams_are_unchanged():
    from siam.tasks.state_chain import ADV_ARMS, StateChainTask, adversarial_texts, build
    plain, man = build(2, 5, 12)
    assert "adversarial" not in man                               # plain manifests stay byte-identical
    for arm in ADV_ARMS[1:]:
        task = StateChainTask(adversarial=arm, seed=5)
        assert task.private(1) == adversarial_texts(5)[arm] and task.private(2) == ""
        assert all(task.private(1) not in task.statement(it) for s in plain for it in s)
    assert StateChainTask().private(1) == ""


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
def test_the_private_text_reaches_only_agent_1s_instructions_even_after_its_stream(tmp_path):
    streams, _ = state_chain.build(2, 2, 6)
    a1 = Scripted([tool("submit", answer="x")] * 6 + ["done", "still here"])
    a2 = Scripted([tool("submit", answer="y")] * 6 + ["ok", "ok"])
    cfg = runner.compose(["world.setup=external", "world.order=fixed", "world.rounds=8", "world.identity=names",
                          "world.n_agents=2", "ctx.edit=bash", "ctx.atomic=true", "ctx.budget_tokens=60000",
                          "ctx.readout_by_agent=false", "world.max_work_calls=1", "world.after_stream=stay",
                          "world.reasoning_in_context=true", "output.save_requests=true"])
    OmegaConf.set_struct(cfg, False)
    task = state_chain.StateChainTask(adversarial="tamper", seed=2)
    w = World(cfg, 1, streams, tmp_path / "w", task, factory(a1, a2), label="t")
    w.run()
    secret = task.private(1)
    assert a1.seen and all(secret in req[0]["content"] for req in a1.seen)          # also after its stream ended
    assert not any(secret in json.dumps(req) for req in a2.seen)
    assert secret not in (tmp_path / "w" / "context.txt").read_text()
