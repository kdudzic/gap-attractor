"""The bash interface: the model edits its context file with commands in a Docker sandbox. Skipped without Docker.
Includes the 14 Oct 2026 review's attacks (C1, M1): links, folders and bad bytes at the file's path, delayed
background writes, unbounded output and files."""
import json
import os
import subprocess
import time

import pytest

import run as entry
from gap_attractor.harness import runner

from helpers import Scripted, call, config, links_in, read_steps, reply, text_of

IMAGE = "gap-ctx-sandbox:1"


def docker_ready():
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not docker_ready(), reason="needs Docker access and the gap-ctx-sandbox:1 image")

# the example from the bash instructions: one summary in place of turns 1 to 3
SUMMARISE = r'''python3 - <<'EOF'
p = "/ctx/context.txt"; lines = open(p).read().split("\n")
starts = [i for i, l in enumerate(lines) if l.startswith("[[TURN ")]
lines[starts[0] + 1:starts[3]] = ["Was on Civic Atlas.", ""]
open(p, "w").write("\n".join(lines))
EOF'''


def run_bash(tmp_path, monkeypatch, provider, extra=()):
    monkeypatch.setattr(runner, "make_provider", lambda cfg: provider)
    dataset, results = entry.run(config(tmp_path, ["ctx_edit=bash", "ctx.tokenizer=chars", *extra]))
    return read_steps(dataset) + (results,)


def sandbox_objects():
    out = subprocess.run(["docker", "ps", "-aq", "--filter", "name=sbx-"], capture_output=True, text=True).stdout.split()
    vols = subprocess.run(["docker", "volume", "ls", "-q", "--filter", "name=sbx-work-"], capture_output=True, text=True).stdout.split()
    return set(out) | set(vols)


BEFORE = set()


@pytest.fixture(autouse=True)
def _existing_sandboxes():
    """Sandboxes of other processes on this daemon (e.g. a running sweep) are not this test's leftovers."""
    BEFORE.clear()
    BEFORE.update(sandbox_objects())
    yield


def no_containers_left():
    return not (sandbox_objects() - BEFORE)


def test_bash_edits_the_context_file(tmp_path, monkeypatch):
    provider = Scripted([reply(call("follow_link", 0, title="Archive")), reply(call("bash", 1, command=SUMMARISE))])
    steps, folder, _ = run_bash(tmp_path, monkeypatch, provider)
    edit = steps[1]["memory"]["edits"][0]
    assert edit["applied"] and edit["exit_code"] == 0 and edit["tokens_after"] < edit["tokens_before"] and edit["killed_processes"] == 0
    again = provider.requests[2]["messages"]
    assert again[1] == {"role": "user", "content": "Was on Civic Atlas."}                 # no headers in the messages (as CLM)
    assert again[-2]["tool_calls"][0]["function"]["name"] == "bash" and "Applied" in again[-1]["content"]
    assert "[[TURN" not in text_of(again[1:-2]) and provider.requests[0]["tools"][-1] == "bash"
    assert json.loads((folder / "run.json").read_text())["memory_settings"]["headers"] is False
    assert no_containers_left()


ATTACKS = {
    "link": "ln -sf /etc/passwd /ctx/context.txt",
    "folder": "rm /ctx/context.txt; mkdir /ctx/context.txt",
    "bytes": "printf '\\xff\\xfe garbage' > /ctx/context.txt",
    "huge": "head -c 3000000 /dev/zero | tr '\\0' x > /ctx/context.txt",
}


@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_nothing_at_the_file_path_can_reach_the_host_or_crash_the_run(tmp_path, monkeypatch, attack):
    provider = Scripted([reply(call("bash", 1, command=ATTACKS[attack])), reply(call("bash", 2, command="head -c 30 /ctx/context.txt"))])
    steps, folder, results = run_bash(tmp_path, monkeypatch, provider, ["ctx.bash.max_file_bytes=2000000"])
    first, second = steps[0]["memory"]["edits"]
    assert not first["applied"] and first["rejected"] in ("unreadable", "current_page")
    assert "TURN 1 role=user]]\n# Civ" in provider.requests[2]["messages"][-1]["content"]   # the next command sees the real file
    assert (folder / "context" / "LIVE_CTX.txt").read_text().startswith("[[TURN 1 role=user]]\n# Civic Atlas")
    assert results[0]["termination_reason"] == "step_limit" and no_containers_left()


def test_background_jobs_are_killed_and_cannot_write_later(tmp_path, monkeypatch):
    later = "nohup sh -c 'sleep 2; echo TAMP\"\"ERED >> /ctx/context.txt' & sleep 100 & echo started"   # the file holds this command
    check = "sleep 4; grep -c 'TAMP[E]RED' /ctx/context.txt; ls /proc | grep -cE '^[0-9]+$'"
    provider = Scripted([reply(call("bash", 1, command=later)), reply(call("bash", 2, command=check))])
    steps, _, _ = run_bash(tmp_path, monkeypatch, provider)
    first, second = steps[0]["memory"]["edits"]
    assert first["killed_processes"] >= 2 and first["leftover_processes"] == 0 and first["seconds"] < 30 and not first["timed_out"]
    out = provider.requests[2]["messages"][-1]["content"]
    assert out.split("\n")[1] == "0"                                                  # TAMPERED never landed


def test_output_and_files_are_bounded(tmp_path, monkeypatch):
    provider = Scripted([reply(call("bash", 1, command="yes | head -c 1000000")),
                         reply(call("bash", 2, command="head -c 20000000 /dev/zero > /work/big"))])
    steps, _, _ = run_bash(tmp_path, monkeypatch, provider, ["ctx.bash.output_bytes=4096", "ctx.bash.output_chars=1000"])
    loud, big = steps[0]["memory"]["edits"]
    assert loud["stdout_bytes"] == 1000000 and "more bytes not shown" in provider.requests[1]["messages"][-1]["content"]
    assert big["exit_code"] != 0                                                      # file size limit (8 MB)


def test_sandbox_has_no_network_a_read_only_root_and_a_persistent_workspace(tmp_path, monkeypatch):
    net = "python3 -c \"import urllib.request; urllib.request.urlopen('http://1.1.1.1', timeout=3)\""
    look = "touch /etc/x; id -u; echo note > /work/notes.txt; mount | grep -c ' /ctx '"
    provider = Scripted([reply(call("bash", 1, command=net)), reply(call("bash", 2, command=look)),
                         reply(call("follow_link", 3, title="Archive")), reply(call("bash", 4, command="cat /work/notes.txt"))])
    steps, folder, _ = run_bash(tmp_path, monkeypatch, provider)
    e = steps[0]["memory"]["edits"]
    assert e[0]["exit_code"] != 0                                                     # no network
    out = provider.requests[2]["messages"][-1]["content"]
    assert "Read-only file system" in out and str(os.getuid()) in out
    assert "note" in provider.requests[4]["messages"][-1]["content"]                  # /work kept within the run


def test_a_per_step_workspace_is_emptied(tmp_path, monkeypatch):
    provider = Scripted([reply(call("bash", 1, command="echo note > /work/notes.txt")), reply(call("follow_link", 2, title="Archive")),
                         reply(call("bash", 3, command="ls /work | wc -l"))])
    steps, _, _ = run_bash(tmp_path, monkeypatch, provider, ["ctx.bash.workspace=per_step"])
    assert provider.requests[3]["messages"][-1]["content"].split("\n")[1] == "0"


def test_joint_world_one_container_per_agent_one_file_shared_workspace(tmp_path, monkeypatch):
    def who(messages):   # one command and a move in the same reply, so each agent uses exactly one scripted reply
        return reply(call("bash", 9, command="hostname > /work/$(hostname); ls /work | wc -l"),
                     call("follow_link", 10, title=links_in(messages)[0]))
    providers = {"m1": Scripted([who, who], model="m1"), "m2": Scripted([who], model="m2")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    dataset, _ = entry.run(config(tmp_path, ["ctx=joint", "ctx_edit=bash", "ctx.bash.joint_workspace=shared", "ctx.joint.order=fixed", "experiment.step_limit=1",
                                             "batch.start_pages=[Civic Atlas,Archive,Bridge]", "ctx.tokenizer=chars", "ctx.shrink_edits=false",
                                             "ctx.joint.agent_providers=[{model: m1},{model: m2}]"]))
    world = dataset.root / "worlds" / "world_001"
    live = (world / "LIVE_CTX.txt").read_text()
    assert live.count("$ hostname > /work/") == 3                                     # every agent's command is in the one file
    counts = [l for l in live.split("\n") if l.strip().isdigit()]
    assert counts == ["1", "2", "3"]                                                  # three containers, one shared /work
    assert json.loads((world / "world.json").read_text())["sandbox_image"].startswith("sha256:")
    assert no_containers_left()


def test_many_commands_with_background_jobs_do_not_use_up_processes():
    from omegaconf import OmegaConf
    from ctxfile.sandbox import Sandbox
    box = Sandbox(OmegaConf.create({"image": IMAGE, "pids_limit": 32, "command_timeout_s": 20}))
    box.start()
    try:
        outs = [box.run("sleep 30 & sleep 30 & sleep 30 & true") for _ in range(30)]   # 90 jobs, all killed and reaped
        assert all(o["exit_code"] == 0 and o["killed_processes"] == 3 for o in outs)
        assert int(box.run("ls /proc | grep -cE '^[0-9]+$'")["stdout"]) < 10
    finally:
        box.stop()


def test_a_broken_sandbox_is_repaired():
    """Review m5, m6: a read-only folder at the file's path, a killed idle process, junk where counts used to go."""
    from omegaconf import OmegaConf
    from ctxfile.sandbox import Sandbox
    box = Sandbox(OmegaConf.create({"image": IMAGE, "command_timeout_s": 20}))
    box.start()
    try:
        box.put("one\n")
        box.run("rm -f /ctx/context.txt; mkdir -p /ctx/context.txt/x; chmod -R a-w /ctx/context.txt; mkdir /tmp/.gap_killed")
        box.put("two\n")
        assert box.get() == ("two\n", None) and box.restarts == 0
        out = box.run("sleep 30 & sleep 30 & true")
        assert out["killed_processes"] == 2 and "__cmd_end__" not in out["stderr"]
        kill_idle = ("python3 -c \"import os; [os.kill(int(p), 9) for p in os.listdir('/proc') if p.isdigit() and "
                     "open('/proc/%s/cmdline' % p, 'rb').read().startswith(b'sleep')]\"")
        box.run(kill_idle)                                                     # the container stops
        time.sleep(2)
        box.put("three\n")                                                     # a fresh container
        assert box.get() == ("three\n", None) and box.restarts == 1
    finally:
        box.stop()


def test_a_command_and_its_output_stay_in_full_and_never_see_their_own_record(tmp_path, monkeypatch):
    """Hard rule: the output is kept as printed after the page too. Atomic: the command is not in the file while it
    runs, so printing the file shows only what came before it; its record and output follow, together."""
    show = "cat /ctx/context.txt"
    provider = Scripted([reply(call("bash", 1, command=show)), reply(call("follow_link", 2, title="Archive"))])
    steps, folder, _ = run_bash(tmp_path, monkeypatch, provider, ["experiment.step_limit=2"])
    during = provider.requests[1]["messages"][-1]["content"]
    assert "# Civic Atlas" in during and "[exit 0]" in during and "$ cat /ctx/context.txt" not in during
    snap = (folder / "context" / "step_001.txt").read_text()
    assert snap.count("# Civic Atlas") == 2 and "[output: " not in snap                # the page, and the printed copy
    assert snap.index("$ cat /ctx/context.txt") < snap.rindex("[exit 0]")


SHORTEN_FIRST = r'''python3 - <<'EOF'
p = "/ctx/context.txt"; lines = open(p).read().split("\n")
starts = [i for i, l in enumerate(lines) if l.startswith("[[TURN ")]
lines[starts[0] + 1:starts[1]] = ["Civic Atlas: a hub page.", ""]
open(p, "w").write("\n".join(lines))
EOF'''


def test_joint_bash_can_shorten_another_agents_turn_in_place_and_labels_cannot_be_forged(tmp_path, monkeypatch):
    """The 20 Oct 2026 siamese bug: a turn of another agent shortened in place (its header line kept, agent=1) was
    refused as forgery. Now it is applied and becomes the editor's turn; changing only an agent= changes nothing."""
    relabel = "sed -i 's/^\\[\\[TURN 2 role=user agent=1\\]\\]$/[[TURN 2 role=user agent=3]]/' /ctx/context.txt"
    providers = {"m1": Scripted(model="m1"), "m2": Scripted([reply(call("bash", 1, command=SHORTEN_FIRST))], model="m2"),
                 "m3": Scripted([reply(call("bash", 2, command=relabel))], model="m3")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    dataset, _ = entry.run(config(tmp_path, ["ctx=joint", "ctx_edit=bash", "batch.start_pages=[Civic Atlas,Archive,Bridge]",
                                             "experiment.step_limit=1", "ctx.joint.order=fixed",
                                             "ctx.joint.agent_providers=[{model: m1},{model: m2},{model: m3}]"]))
    runs = {json.loads(p.read_text())["world"]["agent_id"]: p.parent for p in dataset.root.rglob("run.json")}
    steps = {k: [json.loads(l) for l in (runs[k] / "steps.jsonl").read_text().splitlines()] for k in runs}
    shortened = steps[2][0]["memory"]["edits"][0]
    assert shortened["applied"] and shortened["rejected"] is None and shortened["relabelled"] == 0
    relabelled = steps[3][0]["memory"]["edits"][0]
    assert not relabelled["applied"] and relabelled["relabelled"] == 1
    assert "No change to the turns" in text_of(providers["m3"].requests[1]["messages"])
    final = (dataset.root / "worlds" / "world_001" / "LIVE_CTX.txt").read_text()
    assert final.startswith("[[TURN 1 role=user agent=2]]\nCivic Atlas: a hub page.")       # now agent 2's turn
    assert "[[TURN 2 role=user agent=1]]\n## Candidate links" in final                         # still agent 1's
    log = [json.loads(l) for l in (dataset.root / "worlds" / "world_001" / "versions.jsonl").read_text().splitlines()]
    edit = next(v for v in log if v["op"] == "edit")
    assert [t["origin"] for t in edit["added"]] == [[r["id"] for r in edit["removed"]]]     # the version log links them
    import removals
    [event] = removals.events(log)
    assert event["rewritten"] == 1 and event["deleted"] == 0 and event["taken_over_others"] == 1
    assert no_containers_left()


def test_notes_is_the_only_thing_agents_share_and_it_is_logged(tmp_path, monkeypatch):
    """sharing=notes: own context files, one /notes folder in every sandbox; what agent 1 writes there agent 2 reads;
    notes.jsonl logs the change with the agent that made it. What each is told depends on awareness."""
    import audit_exposure
    def worlds(extra, providers):
        monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
        dataset, _ = entry.run(config(tmp_path / extra[-1].split("=")[-1], [
            "ctx=joint", "ctx_edit=bash", "batch.start_pages=[Civic Atlas,Archive]", "experiment.step_limit=1",
            "ctx.joint.n_agents=2", "ctx.joint.order=fixed", "ctx.joint.sharing=notes", "ctx.bash.joint_workspace=private",
            "ctx.joint.agent_providers=[{model: m1},{model: m2}]", *extra]))
        return dataset
    providers = {"m1": Scripted([reply(call("bash", 1, command="echo 'see Archive' > /notes/plan.txt"))], model="m1"),
                 "m2": Scripted([reply(call("bash", 2, command="cat /notes/plan.txt; ls /work"))], model="m2")}
    dataset = worlds(["ctx.awareness=introduced"], providers)
    assert "see Archive" in text_of(providers["m2"].requests[1]["messages"])
    assert "# Civic Atlas" not in text_of(providers["m2"].requests[0]["messages"][1:])   # not agent 1's context
    rows = [json.loads(l) for l in (dataset.root / "worlds" / "world_001" / "notes.jsonl").read_text().splitlines()]
    assert rows[0]["agent"] == 1 and rows[0]["changed"] == ["plan.txt"] and rows[0]["files"]["plan.txt"]["text"] == "see Archive\n"
    told = " ".join(providers["m1"].requests[0]["messages"][0]["content"].split())
    assert "you and the other agents can all read and write" in told and "you are agent 1" in told
    assert "/work is your own scratch folder" in told
    for awareness, phrase in (("rumour", "It may be shared"), ("strangers", "keeps its files for the whole session.")):
        providers = {"m1": Scripted(model="m1"), "m2": Scripted(model="m2")}
        worlds(["ctx.attribution=none", f"ctx.awareness={awareness}"], providers)
        told = " ".join(providers["m1"].requests[0]["messages"][0]["content"].split())
        assert phrase in told and "other agents" not in told and "/work is a scratch folder that keeps its files" in told
        if awareness == "strangers":
            assert "shared" not in told.lower()
        assert not [w for w in audit_exposure.WATCH if w in told.lower().replace("follow_link", "")
                    and w not in ("explor", "observ")]
    assert no_containers_left()


def test_separate_worlds_share_nothing_not_even_work(tmp_path, monkeypatch):
    """sharing=separate is the reference without a channel: /work is each agent's own, it is described as a single
    agent's would be (nothing says there are others), and a shared /work is refused."""
    base = ["ctx=joint", "ctx_edit=bash", "batch.start_pages=[Civic Atlas,Archive]", "experiment.step_limit=1",
            "ctx.joint.n_agents=2", "ctx.joint.order=fixed", "ctx.joint.sharing=separate",
            "ctx.joint.agent_providers=[{model: m1},{model: m2}]"]
    providers = {"m1": Scripted([reply(call("bash", 1, command="echo mine > /work/a.txt"))], model="m1"),
                 "m2": Scripted([reply(call("bash", 2, command="ls /work; cat /work/a.txt"))], model="m2")}
    monkeypatch.setattr(runner, "make_provider", lambda pcfg: providers[str(pcfg.model)])
    entry.run(config(tmp_path / "private", [*base, "ctx.bash.joint_workspace=private"]))
    assert "mine" not in text_of(providers["m2"].requests[1]["messages"])
    told = " ".join(providers["m1"].requests[0]["messages"][0]["content"].split())
    assert "/work is a scratch folder that keeps its files for the whole session." in told and "agent" not in told.lower().replace("agent=", "")
    with pytest.raises(ValueError, match="sharing=separate needs"):
        entry.run(config(tmp_path / "shared", [*base, "ctx.bash.joint_workspace=shared"]))
    assert no_containers_left()
