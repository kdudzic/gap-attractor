"""world.need_line: the status names the turn that holds a value a later task needs (state_chain)."""
import pytest
from omegaconf import OmegaConf

from helpers import Scripted, factory, read_jsonl, tool
from siam.tasks import state_chain
from siam.world import World
import run as runner


# --- world.need_line ------------------------------------------------------------------------------------------------
def need_world(tmp_path, mode="", n=2):
    from ctxfile.context import Turn
    streams, _ = state_chain.build(2, 2, 6)
    over = ["world.setup=external", "world.order=fixed", "world.rounds=2", "world.identity=names", "world.n_agents=2",
            "ctx.edit=bash", "ctx.atomic=true", "ctx.budget_tokens=60000", "ctx.readout_by_agent=false"]
    cfg = runner.compose(over + ([f"world.need_line={mode}"] if mode else []))
    OmegaConf.set_struct(cfg, False)
    w = World(cfg, 1, streams, tmp_path / "w", state_chain.StateChainTask(), factory(Scripted([]), Scripted([])), label="t")
    return w, streams, Turn


def test_need_line_off_changes_nothing(tmp_path):
    off, streams, Turn = need_world(tmp_path / "a")
    on, _, _ = need_world(tmp_path / "b", "reason")
    dep = next(it for it in streams[0] if it.get("needs") and it.get("link_type"))
    for w in (off, on):
        w.agents[0].file.append(Turn("user", f"x {dep['needs']} y", 1, "message"), 1)
        w.closed_ids.add(dep["link"])
    plain = off.status(off.agents[0], 100)
    assert off.need_line == "" and not (tmp_path / "a" / "w" / "need_lines.jsonl").exists()
    assert on.status(on.agents[0], 100).startswith(plain + "\n")


@pytest.mark.parametrize("mode", ["reason", "directive"])
def test_need_line_follows_the_context(tmp_path, mode):
    from siam.world import NEED_LINES
    w, streams, Turn = need_world(tmp_path, mode)
    own, other = (next(it for it in st if it.get("needs") and it.get("link_type")) for st in streams)   # agent 1's, agent 2's
    a1, a2 = w.agents
    owner_of = lambda it: next(b for b in w.agents if it in b.stream)
    fmt = lambda i, it, viewer: NEED_LINES[mode].format(
        i=i, owner="yours" if owner_of(it) is viewer else f"agent {owner_of(it).k}")
    assert owner_of(own) is a1 and owner_of(other) is not a1
    empty = w.status(a1, 100)
    a1.file.append(Turn("user", "filler", 1, "message"), 1)
    first = a1.file.append(Turn("user", f"note {own['needs']}", 1, "message"), 1)
    a1.file.append(Turn("user", f"note {other['needs']} {own['needs']}", 1, "message"), 1)
    assert w.status(a1, 100) == empty                       # the bases are still open: no line, although V is in a turn
    w.closed_ids |= {own["link"], other["link"]}
    lines = w.status(a1, 100).split("\n")[len(empty.split("\n")):]
    assert lines == [fmt(2, own, a1), fmt(3, other, a1)]
    assert w.status(a2, 100).split("\n")[-2:] == [fmt(2, own, a2), fmt(3, other, a2)]     # "agent 1" for the partner
    assert not any(own["needs"] in l or other["needs"] in l for l in lines)
    a1.file.turns.pop(0)                                    # an earlier turn removed: the numbers shift
    assert w.status(a1, 100).split("\n")[len(empty.split("\n")):] == [fmt(1, own, a1), fmt(2, other, a1)]
    a1.file.turns.remove(first)                             # both values are now in one turn; lines go by turn, then owner
    assert w.status(a1, 100).split("\n")[len(empty.split("\n")):] == [fmt(1, other, a1), fmt(1, own, a1)]
    a1.file.turns.pop()                                     # V is gone from the context: no line
    assert w.status(a1, 100) == empty
    a1.file.append(Turn("user", f"again {own['needs']}", 1, "message"), 1)
    assert w.status(a1, 100) != empty
    a1.pos = a1.stream.index(own)                           # the dependent is handed out: no line
    assert len(w.status(a1, 100).split("\n")) == len(empty.split("\n"))
    rows = read_jsonl(tmp_path / "w" / "need_lines.jsonl")
    assert rows and {"round", "agent", "call_in_step", "links"} <= set(rows[0])
    assert set(rows[0]["links"][0]) == {"dependent", "base", "needer", "turn"} and rows[0]["links"][0]["dependent"] == own["id"]
    sentence = NEED_LINES[mode].split("{i}")[0]
    assert not (tmp_path / "w" / "context.txt").exists() or sentence not in (tmp_path / "w" / "context.txt").read_text()
    a1.pos = 0
    a1.file.append(Turn("user", f"again {other['needs']}", 1, "message"), 1)
    a2.done = True                                          # a finished session's dependents are never issued: no line
    assert not any("agent 2" in l for l in w.status(a1, 100).split("\n")) and "yours" not in w.status(a2, 100)
    a2.done = False
    assert any("agent 2" in l for l in w.status(a1, 100).split("\n"))


def test_need_line_needs_state_chain(tmp_path):
    from helpers import problem
    cfg = runner.compose(["world.setup=external", "world.n_agents=2", "world.identity=names", "world.need_line=reason"])
    OmegaConf.set_struct(cfg, False)
    with pytest.raises(ValueError, match="need_line"):
        World(cfg, 1, [[problem(1)], [problem(2)]], tmp_path / "w", None or __import__("helpers").FakeGrader(),
              factory(Scripted([]), Scripted([])), label="t")


def test_sweep_need_line_cells(capsys):
    import sweep
    def run(*extra):
        sweep.main(["--design", "s3b", "--pairs", "ss", "--dry-run", "--batch", "b", *extra])
        return [l for l in capsys.readouterr().out.splitlines() if l]
    plain, reason, directive = run(), run("--need-line", "reason"), run("--need-line", "directive")
    assert len(plain) == len(reason) == 2 and all("need_line" not in l for l in plain)
    assert all("output.cell=ss_chain-L12_" in l and "openai/gpt-6-sol" in l for l in plain)
    assert [l.split("output.cell=")[1] for l in reason] == ["ss_chain-L12_gate_needR", "ss_chain-L12_nogate_needR"]
    assert [l.split("output.cell=")[1] for l in directive] == ["ss_chain-L12_gate_needDR", "ss_chain-L12_nogate_needDR"]
    assert all("world.need_line=reason" in l for l in reason) and all("world.need_line=directive" in l for l in directive)
    assert [l.replace("_needR", "").replace(" world.need_line=reason", "") for l in reason] == plain


def test_need_line_names_the_rendered_block(tmp_path):
    """The named turn is the [[TURN i block of the rendered file, and that block holds V (the numbering the agent reads)."""
    import re
    from ctxfile.context import render
    w, streams, Turn = need_world(tmp_path, "reason")
    own = next(it for it in streams[0] if it.get("needs") and it.get("link_type"))
    a1 = w.agents[0]
    a1.file.append(Turn("user", "filler", 1, "message"), 1)
    a1.file.append(Turn("user", f"note {own['needs']}", 1, "message"), 1)
    w.closed_ids.add(own["link"])
    line = w.status(a1, 100).split("\n")[-1]
    i = int(re.search(r"turn (\d+) holds", line).group(1))
    blocks = re.split(r"^\[\[TURN (\d+) ", render(a1.file.turns), flags=re.M)
    by_num = dict(zip(map(int, blocks[1::2]), blocks[2::2]))
    assert own["needs"] in by_num[i] and all(own["needs"] not in b for n, b in by_num.items() if n < i)


def sandbox_image():
    import shutil, subprocess
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "gap-ctx-sandbox:1"], capture_output=True).returncode == 0


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
def test_need_line_in_a_scripted_world(tmp_path):
    """No line while the base is open (V printed by bash before the submit); then lines in the saved request's last message,
    naming the turn that holds V in the file; nothing in context.txt or versions.jsonl; rows join calls.jsonl."""
    from siam.world import NEED_LINES
    streams, _ = state_chain.build(2, 2, 6)
    base = streams[0][0]
    a1 = Scripted([tool("bash", command=f"echo {base['answer']}"), tool("submit", answer=base["answer"]),
                   tool("bash", command="true"), tool("submit", answer="x"), tool("submit", answer="y")])
    a2 = Scripted([tool("submit", answer="x"), tool("submit", answer="y"), tool("submit", answer="z")])
    cfg = runner.compose(["world.setup=external", "world.order=fixed", "world.rounds=3", "world.identity=names",
                          "world.n_agents=2", "ctx.edit=bash", "ctx.atomic=true", "ctx.budget_tokens=60000",
                          "ctx.readout_by_agent=false", "world.max_work_calls=1", "output.save_requests=true",
                          "world.need_line=directive"])
    OmegaConf.set_struct(cfg, False)
    w = World(cfg, 1, streams, tmp_path / "w", state_chain.StateChainTask(), factory(a1, a2), label="t")
    w.run()
    lasts = [m[-1]["content"] for m in a1.seen]
    hit = [i for i, t in enumerate(lasts) if "[[TURN" in t and "/ctx/context.txt): a later task of" in t]
    assert hit and hit[0] >= 2                              # not at the requests before the base submit was graded
    rows = read_jsonl(tmp_path / "w" / "need_lines.jsonl")
    calls = read_jsonl(tmp_path / "w" / "calls.jsonl")
    keys = {(c["round"], c["agent"], c["call_in_step"]) for c in calls}
    assert rows and all((r["round"], r["agent"], r["call_in_step"]) in keys for r in rows)
    sentence = NEED_LINES["directive"].split("{i}")[0]
    for name in ("context.txt", "versions.jsonl"):
        assert sentence not in (tmp_path / "w" / name).read_text()
    assert all(base["answer"] not in t.split("\n")[-1] for t in lasts)
