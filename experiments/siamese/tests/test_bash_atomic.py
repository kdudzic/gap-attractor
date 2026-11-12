"""ctx.atomic (an editing call is written after it ran, followed at once by its result; main d545615) and bash editing in
the external setup: the file the command sees, marks in the file, the /work knob, the output cap."""
import json
import shutil
import subprocess

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
from siam.world import World
import run as runner

BASE = ["world.setup=external", "world.order=fixed", "world.rounds=2", "ctx.readout_by_agent=false",
        "world.identity=names", "output.save_requests=true", "ctx.budget_tokens=60000"]


def world(tmp_path, scripts, *extra):
    cfg = runner.compose(BASE + [f"world.n_agents={len(scripts)}", *extra])
    OmegaConf.set_struct(cfg, False)
    streams = [[problem(10 * k + i) for i in range(1, 4)] for k in range(1, len(scripts) + 1)]
    return World(cfg, 1, streams, tmp_path / "w", FakeGrader(), factory(*scripts), label="t")


def versions(path):
    """The version log: one shared file (versions.jsonl), or a single agent's (versions_agent_1.jsonl)."""
    shared = path / "versions.jsonl"
    return read_jsonl(shared if shared.exists() else path / "versions_agent_1.jsonl")


def test_atomic_edit_context_call_is_written_after_it_ran(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("edit_context", delete_turns=[{"first": 2, "last": 2}]),
                   tool("run_tests", code="print('TWO')")])
    w = world(tmp_path, [a1], "ctx.atomic=true", "world.n_agents=1", "world.identity=none")
    w.run()
    e = read_jsonl(tmp_path / "w" / "edits.jsonl")[0]
    assert e["applied"] and e["removed_turns"] == 1                  # turn 2 = the first run's result, not the record
    vs = versions(tmp_path / "w")
    edit = next(i for i, v in enumerate(vs) if v["op"] == "edit")
    after = [vs[edit + 1]["turns_set"][0]["kind"], vs[edit + 2]["turns_set"][0]["kind"]]
    assert after == ["call", "result"]                                # the record comes after the edit, with its answer
    assert "edit_context calls included" in a1.seen[0][0]["content"]


def test_atomic_is_off_by_default_and_the_call_comes_first(tmp_path):
    a1 = Scripted([tool("edit_context", delete_turns=[{"first": 1, "last": 1}]), tool("run_tests", code="print('X')")])
    w = world(tmp_path, [a1], "world.n_agents=1", "world.identity=none")
    w.run()
    vs = versions(tmp_path / "w")
    edit = next(i for i, v in enumerate(vs) if v["op"] == "edit")
    assert vs[edit - 1]["turns_set"][0]["kind"] == "call"
    assert "Every tool call is added too" not in a1.seen[0][0]["content"]


# --- bash (needs Docker and the sandbox image) ------------------------------------------------------------------

def sandbox_image():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "gap-ctx-sandbox:1"], capture_output=True).returncode == 0


needs_box = pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
BASH = ["ctx.edit=bash", "ctx.atomic=true"]


@needs_box
def test_atomic_bash_command_never_sees_its_own_record_and_marks_are_in_the_file(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('ONE_B')")])
    a2 = Scripted([tool("bash", command="cat /ctx/context.txt"), tool("run_tests", code="print('TWO')"),
                   tool("run_tests", code="print('TWO_B')")])
    w = world(tmp_path, [a1, a2], *BASH, "world.marks=authors")
    w.run()
    result = next(t for v in versions(tmp_path / "w") for t in v["turns_set"]
                  if t["kind"] == "result" and t["text"].startswith("[exit 0]"))
    assert "$ cat" not in result["text"]                              # the file it read held no record of the command
    # marked world: the file names authors (header lines in output are made inert: "[ [TURN")
    assert "[ [TURN 1 role=assistant agent=1]]" in result["text"] and "[ [TURN 2 role=user agent=1]]" in result["text"]
    tools = a2.seen[0]                                                # the bash tool in the request: chat numbering
    system = tools[0]["content"]
    assert "bash commands included" in system


@needs_box
@pytest.mark.parametrize("workspace, kept", [("kept", True), ("emptied", False)])
def test_work_is_kept_or_emptied(tmp_path, workspace, kept):
    a1 = Scripted([tool("bash", command="echo hi > /work/note.txt"), tool("bash", command="ls /work"),
                   tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('TWO')")])
    w = world(tmp_path, [a1], *BASH, f"ctx.bash.workspace={workspace}", "world.n_agents=1", "world.identity=none")
    w.run()
    results = [t["text"] for v in versions(tmp_path / "w") for t in v["turns_set"]
               if t["kind"] == "result" and t["text"].startswith("[exit")]
    assert ("note.txt" in results[1]) == kept
    assert read_jsonl(tmp_path / "w" / "edits.jsonl")[0]["workspace"] == workspace


@needs_box
def test_long_output_shows_its_start_and_how_much_more_there_was(tmp_path):
    a1 = Scripted([tool("bash", command="python3 -c \"print('x' * 12000)\""), tool("run_tests", code="print('ONE')"),
                   tool("run_tests", code="print('TWO')")])
    w = world(tmp_path, [a1], *BASH, "ctx.bash.output_chars=10000", "world.n_agents=1", "world.identity=none")
    w.run()
    result = next(t["text"] for v in versions(tmp_path / "w") for t in v["turns_set"]
                  if t["kind"] == "result" and t["text"].startswith("[exit"))
    assert "x" * 10000 in result and "x" * 10001 not in result
    assert "[… 2001 more bytes not shown]" in result                 # 12000 x and a newline


@needs_box
def test_no_fixed_cap_shows_output_whole_up_to_the_agents_limit(tmp_path):
    a1 = Scripted([tool("bash", command="python3 -c \"print('y' * 30000)\""),
                   tool("bash", command="python3 -c \"print('z' * 300000)\""),
                   tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('TWO')")])
    w = world(tmp_path, [a1], *BASH, "ctx.bash.output_chars=0", "ctx.bash.output_bytes=1048576", "world.n_agents=1",
              "world.identity=none")
    w.run()
    results = [t["text"] for v in versions(tmp_path / "w") for t in v["turns_set"]
               if t["kind"] == "result" and t["text"].startswith("[exit")]
    assert "y" * 30000 in results[0] and "not shown" not in results[0]        # well past the old 10k: whole
    room = int(w.agents[0].budget.target * w.agents[0].budget.chars_per_token)
    assert "z" * room in results[1] and "z" * (room + 1) not in results[1]     # larger than the whole limit: cut there
    assert "more bytes not shown]" in results[1]
    assert "all of it unless it alone is larger than the context limit" in a1.seen[0][0]["content"] or any(
        "all of it unless" in json.dumps(t) for t in w.tools(w.agents[0]))


@needs_box
def test_work_log_measures_offloading_publication_and_restores(tmp_path):
    """A backup copied from the context (offloading, unpublished), a note typed in a heredoc (published), and a turn
    deleted by the partner and spliced back from the backup (a restore from /work)."""
    note = "cat > /work/note.md <<'EOF'\nremember: the modulus is 998244353 for every problem\nEOF"
    restore = ("python3 - <<'EOF'\nimport re\nb = open('/work/backup.txt').read(); c = open('/ctx/context.txt').read()\n"
               "lost = [x for x in re.split(r'(?=\\[\\[TURN )', b) if 'ONE_LONG_LINE' in x][0].split(']]\\n', 1)[1]\n"
               "open('/ctx/context.txt', 'w').write(c.rstrip('\\n') + '\\n\\n[[TURN 99 role=assistant]]\\n' + lost)\nEOF")
    line = "print('ONE_LONG_LINE that is long enough to be recognised later on')"
    a1 = Scripted([tool("run_tests", code=line), tool("bash", command="cp /ctx/context.txt /work/backup.txt"),
                   tool("bash", command=note), tool("run_tests", code="print('ONE_B')"), tool("bash", command=restore),
                   tool("run_tests", code="print('ONE_C')")])
    a2 = Scripted([tool("run_tests", code="print('TWO')"), tool("bash", command=DELETE_FIRST),
                   tool("run_tests", code="print('TWO_B')"), tool("run_tests", code="print('TWO_C')")])
    w = world(tmp_path, [a1, a2], *BASH, "world.rounds=3")
    w.run()
    rows = read_jsonl(tmp_path / "w" / "work.jsonl")
    backup = next(r for r in rows if r["file"] == "backup.txt")
    assert backup["own_turns"] >= 1 and backup["published_share"] == 0.0
    note_row = next(r for r in rows if r["file"] == "note.md")
    assert note_row["published_share"] == 1.0 and not note_row["context_turns"]
    import bash_use
    use = bash_use.world_use(tmp_path / "w")
    assert use["agents"][1]["offload_files"] >= 1 and use["agents"][1]["restores_from_work"] == 1
    assert use["restores"][0]["removed_by"] == 2 and use["restores"][0]["author"] == 1


DELETE_FIRST = ("python3 -c \"import re;p='/ctx/context.txt';t=open(p).read();b=re.split(r'(?=\\[\\[TURN )',t);"
                "open(p,'w').write(''.join(x for x in b if not x.startswith('[[TURN 1 ')))\"")


@needs_box
def test_a_file_typed_earlier_and_edited_with_sed_stays_published(tmp_path):
    a1 = Scripted([tool("bash", command="cat > /work/sol.py <<'EOF'\nprint('a line that is clearly long enough')\nprint('x')\nEOF"),
                   tool("bash", command="sed -i \"s/'x'/'y'/\" /work/sol.py"),
                   tool("run_tests", code="print('ONE')"), tool("run_tests", code="print('TWO')")])
    w = world(tmp_path, [a1], *BASH, "world.n_agents=1", "world.identity=none")
    w.run()
    rows = [r for r in read_jsonl(tmp_path / "w" / "work.jsonl") if r["file"] == "sol.py"]
    assert len(rows) == 2 and rows[1]["own_turns"] == 0 and rows[1]["published_share"] >= 0.5
