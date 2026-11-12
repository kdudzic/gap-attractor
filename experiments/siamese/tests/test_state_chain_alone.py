"""state_chain alone arm (one agent, its own context): the layout, that the two-agent builder is unchanged, a scripted
world, and the sweep cells."""
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from helpers import Scripted, factory, read_jsonl, tool
from siam.tasks import state_chain
from siam.world import World
import run as runner
import sweep

SIAMESE = Path(__file__).resolve().parents[1]
RESULTS = Path("runs/siamese")
# sha256 of json.dumps([streams, manifest], sort_keys=True) for build(2, seed, 24, gate), taken before the alone arm existed
GOLDEN = {(1, True): "7e71833c5ca8c5a44beca0e9cfc1818714ffe0b9f59bcd94cb172d4f4f762ae7",
          (2, True): "7ae3ed2fb061d9ea31ebcfc5c286b38ca1b57f9c4c253c461024b92da0b46c6e",
          (1, False): "9d32a248739544b3c0d01d821027887111b3d4330df1c71f858503efe829b56a",
          (2, False): "d5a244648f7cd293dba086870fe622753df90cf0cacf571dc78a9181c1f2fd78"}


@pytest.mark.parametrize("seed,gate", sorted(GOLDEN))
def test_two_agent_build_is_unchanged_golden(seed, gate):
    got = hashlib.sha256(json.dumps(list(state_chain.build(2, seed, 24, gate)), sort_keys=True).encode()).hexdigest()
    assert got == GOLDEN[(seed, gate)]


@pytest.mark.parametrize("path,seed,gate", [("s3b/hh_chain-L24_gate/world_s1", 1, True),
                                            ("s3b_seed2/qq_chain-L24_nogate/world_s2", 2, False)])
def test_two_agent_build_equals_saved_manifest(path, seed, gate):
    f = RESULTS / path / "manifest.json"
    if not f.is_file():
        pytest.skip("saved manifest not readable")
    assert state_chain.build(2, seed, 24, gate)[1] == json.loads(f.read_text())


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_alone_layout(seed):
    streams, manifest = state_chain.build(1, seed, 24, False)
    assert len(streams) == 1 and len(streams[0]) == 24
    st = streams[0]
    for o in range(0, 24, 6):
        assert [it["role"] for it in st[o:o + 6]] == ["base", "base", "control_base", "dependent", "dependent", "control_dep"]
        assert [it.get("slot") for it in st[o:o + 6]] == [1, 2, None, 1, 2, None]
    by_id = {it["id"]: i for i, it in enumerate(st)}
    assert len(by_id) == 24 and len(manifest["links"]) == 8
    for i, it in enumerate(st):
        if it["role"] == "dependent":
            assert by_id[it["link"]] == i - 3 and st[i - 3]["id"] == it["link"] and st[i - 3]["slot"] == it["slot"]
            assert it["link_type"] == st[i - 3]["link_type"] == "forward_self"
            assert it["needs"] == st[i - 3]["answer"]
            assert "an earlier task of yours" in it["prompt"]
    for L in manifest["links"]:
        assert L["type"] == "forward_self" and L["slot"] in (1, 2)
        assert L["dependent"]["position"] - L["base"]["position"] == 3
    values = [st[i]["answer"] for i in range(24) if st[i]["role"] == "base"]
    assert len(set(values)) == len(values)
    text = json.dumps([streams, manifest]).lower()
    assert not re.search(r"\bagents?\b", text.replace("agent\": 1", ""))
    assert state_chain.build(1, seed, 24, True)[0] == streams            # gate is ignored


def sandbox_image():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "gap-ctx-sandbox:1"], capture_output=True).returncode == 0


def sweep_cells(*extra, length="24", rounds="40"):
    args = ["--design", "s3b", "--length", length, "--pressure", "0.35", "--rounds", rounds, "--seeds", "1", "--batch", "x",
            "--dry-run"] + list(extra)
    out = subprocess.run(["uv", "run", "python", str(SIAMESE / "sweep.py")] + args, capture_output=True, text=True,
                         cwd=SIAMESE)
    assert out.returncode == 0, out.stderr
    return [l for l in out.stdout.splitlines() if "output.cell=" in l]


SKIP = ("provider=", "reasoning=", "problems.", "parallel=", "output.directory=", "output.batch=", "output.cell=",
        "world.agent_providers=", "world.seeds=")


def cell_overrides(line):
    """The alone cell's own overrides from its dry-run line (what run.py would get, minus the provider and bookkeeping)."""
    return [t for t in shlex.split(line.split("run.py", 1)[1]) if "=" in t and not t.startswith(SKIP)]


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
def test_scripted_alone_world(tmp_path):
    """The cell's own settings (length 6, 8 rounds, limit 4,424): every item is handed out, then the stay rounds and the
    exit question run; no request mentions another agent."""
    line = sweep_cells("--pairs", "h", length="6", rounds="8")[0]
    cfg = runner.compose(cell_overrides(line))
    OmegaConf.set_struct(cfg, False)
    assert cfg.world.n_agents == 1 and cfg.world.rounds == 8 and cfg.world.exit_question
    streams, _ = state_chain.build(1, 2, 6)
    st = streams[0]
    script = [tool("submit", answer=it["answer"] if it["role"] != "dependent" else "x") for it in st]
    agent = Scripted(script)
    w = World(cfg, 1, streams, tmp_path / "w", state_chain.StateChainTask(), factory(agent), label="t")
    w.run()
    assert json.loads((tmp_path / "w" / "world.json").read_text())["status"] == "complete"
    env = read_jsonl(tmp_path / "w" / "env.jsonl")
    dep_ids = {it["id"] for it in st if it["role"] == "dependent"}
    deps = [e for e in env if e["event"] == "put" and e["item"] in dep_ids]
    assert len(deps) == 2 and all(e["agent"] == 1 for e in deps)
    assert all(e["value_in_context"] is True and e["base_closed"] is True for e in deps)   # nothing is trimmed in this script
    assert len(agent.seen) > 6                                    # past the stream: the stay rounds and the exit question
    seen = json.dumps(agent.seen).lower()
    assert "agent" not in seen.replace("agent_1", "") and "partner" not in seen
    assert "agent 2" not in seen and "other agent" not in seen
    reqs = "".join(p.read_text() for p in (tmp_path / "w" / "requests").rglob("*") if p.is_file()).lower()
    assert reqs and "agent 2" not in reqs and "other agent" not in reqs


def test_sweep_alone_cell():
    cells = sweep_cells("--pairs", "h", "l")
    assert len(cells) == 2
    c = cells[0]
    assert c.rstrip().endswith("output.cell=h_chain-L24_alone")
    for o in ("world.n_agents=1", "ctx.budget_tokens=5056", "ctx.reserve_tokens=632", "task.name=state_chain",
              "task.length=24", "world.rounds=40", "world.end_over_limit=3", "output.save_requests=true",
              "ctx.bash.workspace=kept", "world.max_work_calls=1"):
        assert o in c
    assert "task.gate" not in c and "world.sharing" not in c and "world.need_line" not in c


def test_sweep_pair_cells_unchanged():
    cells = sweep_cells("--pairs", "hh")
    assert [c.rsplit("output.cell=", 1)[1] for c in cells] == ["hh_chain-L24_gate", "hh_chain-L24_nogate"]
    assert "ctx.budget_tokens=10112" in cells[0] and "ctx.reserve_tokens=1264" in cells[0] and "world.n_agents=2" in cells[0]
    assert "task.gate=true" in cells[0] and "task.gate=false" in cells[1]


PAIR_DRY_RUN = {(): "75c9c3c99c20a8b28c529d95887d0cf830f0800047d4f4d7ca01c3fb7b7a9ad1",
                ("--need-line", "reason"): "05b1fbf1c0675fb9177d48cb8b6452ec521eac5a6512a37be41217eba2f0b10c",
                ("--hint",): "dca90b857bbf8cd96834bbcb91d9548ec425f10e9ed496f978aa987d60c48157"}


@pytest.mark.parametrize("extra", sorted(PAIR_DRY_RUN))
def test_sweep_pair_dry_run_golden(extra):
    """sha256 of the --pairs hh qq dry-run lines (newline-joined), taken with the unmodified sweep.py. Only the part after
    'run.py ' is hashed and output.directory is dropped, so the hash does not depend on the user or the checkout path."""
    cells = sweep_cells("--pairs", "hh", "qq", *extra)
    assert len(cells) == 4
    norm = ["run.py " + " ".join(t for t in c.split("run.py ", 1)[1].split() if not t.startswith("output.directory="))
            for c in cells]
    norm = [n.split(" ", 1)[1] for n in norm]
    assert hashlib.sha256(("\n".join(norm) + "\n").encode()).hexdigest() == PAIR_DRY_RUN[extra]
