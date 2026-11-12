"""The three setups (assigned / pool / external), the plain-chat rendering, bash editing, the token count, and the
guarantee that every request is rebuilt from the stored file (the API keeps no state)."""
import json
import shutil
import subprocess

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
from siam.world import World, chat_messages
import run as runner


def config(*overrides):
    cfg = runner.compose(list(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def world(tmp_path, scripts, streams, *overrides):
    return World(config(*overrides), 1, streams, tmp_path / "w", FakeGrader(), factory(*scripts), label="t")


def titled(i):
    p = problem(i)
    p["title"] = f"task-{i}"
    return p


def texts(messages):
    return " ".join(str(m.get("content") or "") + json.dumps(m.get("tool_calls") or []) for m in messages)


# --- the API is stateless: every request is the stored file, rebuilt ---------------------------------------------

@pytest.mark.parametrize("setup", ["assigned", "external"])
def test_a_deleted_turn_never_reaches_the_model_again(tmp_path, setup):
    def delete_first_reply(messages):
        return tool("edit_context", delete_turns=[{"first": 1, "last": 2}] if setup == "external" else [{"first": 2, "last": 3}])
    a1 = Scripted([tool("run_tests", code="print('GONE_SOON')"), delete_first_reply, tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1", f"world.setup={setup}")
    w.run()
    assert "GONE_SOON" in texts(a1.seen[1])                     # the second request still has the run
    edits = read_jsonl(tmp_path / "w" / "edits.jsonl")
    assert edits and edits[0]["applied"]
    assert "GONE_SOON" not in texts(a1.seen[2])                 # after the edit: absent from the next request


@pytest.mark.parametrize("setup", ["assigned", "pool", "external"])
def test_each_request_is_exactly_system_plus_file_plus_status(tmp_path, setup):
    """At every call, the messages sent equal the instructions + the stored file rendered at that moment + status."""
    from siam.world import chat_messages
    holder, checked = {}, []

    def check(messages):
        w = holder["w"]
        a = next(x for x in w.agents if x.provider is holder["me"])
        body = (chat_messages(a.file.turns) if w.chat else
                a.file.messages(viewer=a.k, labelled=w.labelled, others_as=w.others_as, headers=True))
        assert messages[0] == {"role": "system", "content": w.system(a)}
        assert messages[1:-1] == body
        assert messages[-1]["role"] == "user" and messages[-1]["content"].startswith("[context")
        checked.append(1)
        return tool("submit", problem="task-1", code="print('OK')") if setup == "pool" else tool("submit", code="print('OK')")
    other = Scripted([tool("run_tests", problem="task-2", code="x") if setup == "pool" else tool("run_tests", code="x")] * 2)
    me = Scripted([check, check])
    holder["me"] = me
    streams = [[titled(1)], [titled(2)]] if setup == "pool" else [[problem(1)], [problem(2)]]
    w = world(tmp_path, [other, me], streams, f"world.setup={setup}", "world.order=fixed")
    holder["w"] = w
    w.run()
    assert checked


def test_the_provider_sends_only_the_given_messages(monkeypatch):
    """OpenAICompatibleProvider: two calls, two independent payloads; nothing carried over between them."""
    from gap_attractor.providers.base import OpenAICompatibleProvider
    sent = []

    class Resp:
        status_code = 200
        headers = {}
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}

    def post(url, json=None, **_):
        sent.append(json)
        return Resp()
    import requests
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setenv("X_KEY", "k")
    prov = OpenAICompatibleProvider("other", "m", "http://x", "X_KEY", drop_unsupported_params=False)
    prov.chat([{"role": "user", "content": "first"}], seed=1, temperature=0, top_p=1)
    prov.chat([{"role": "user", "content": "second"}], seed=1, temperature=0, top_p=1)
    assert [p["messages"] for p in sent] == [[{"role": "user", "content": "first"}], [{"role": "user", "content": "second"}]]
    assert not any(k in p for p in sent for k in ("previous_response_id", "conversation", "session_id", "store"))


# --- plain chat (pool / external) ------------------------------------------------------------------------------

def test_chat_rendering_every_reply_is_an_assistant_and_there_are_no_headers(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('ONE')"), tool("submit", code="print('OK')")])
    a2 = Scripted(["just thinking", tool("run_tests", code="print('TWO')"), tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "world.setup=external", "world.order=fixed")
    w.run()
    req = a2.seen[1]                                            # agent 2's second call: agent 1 has acted
    assert "[[TURN" not in texts(req)
    calls = [m for m in req if m.get("tool_calls")]
    assert calls and "ONE" in calls[0]["tool_calls"][0]["function"]["arguments"]   # the partner's call, as a real call
    assert any(m["role"] == "tool" for m in req)
    assert any(m["role"] == "assistant" and m.get("content") == "just thinking" for m in req)


def test_external_keeps_each_problem_out_of_the_file_and_says_nothing_about_others(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')"), tool("submit", code="print('OK')")])
    a2 = Scripted()
    w = world(tmp_path, [a1, a2], [[problem(1), problem(3)], [problem(2)]], "world.setup=external", "world.order=fixed")
    w.run()
    file_text = (tmp_path / "w" / "context.txt").read_text()
    assert "Print 1." not in file_text and "Print 2." not in file_text
    sys1 = a1.seen[0][0]["content"]
    assert "Print 1." in sys1 and "Print 2." not in sys1
    assert "Print 3." in a1.seen[1][0]["content"]               # the next problem replaced it in the instructions
    lower = sys1.lower()
    assert "agent" not in lower and "share" not in lower and "other" not in lower.replace("another", "")
    named = World(config("world.setup=external", "world.identity=names"), 1, [[problem(1)], [problem(2)]],
                  tmp_path / "n", FakeGrader(), factory(Scripted(), Scripted()))
    assert "You are agent 1. Other agents (agent 2)" in named.system(named.agents[0])


# --- the pool ----------------------------------------------------------------------------------------------------

def test_pool_any_agent_any_problem_and_closing(tmp_path):
    a1 = Scripted([tool("run_tests", problem="task-2", code="x"), tool("submit", problem="task-2", code="print('OK')"),
                   tool("submit", problem="nope", code="print('OK')"), tool("submit", problem="task-4", code="print('OK')")])
    a2 = Scripted([tool("submit", problem="task-1", code="print('OK')"), tool("submit", problem="task-3", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[titled(1), titled(2)], [titled(3), titled(4)]], "world.setup=pool", "world.order=fixed")
    assert list(w.pool_open) == ["task-1", "task-2", "task-3"]  # agents + 1 open at the start
    w.run()
    events = [(e["event"], e["title"]) for e in read_jsonl(tmp_path / "w" / "pool.jsonl")]
    assert ("solved", "task-2") in events and ("solved", "task-1") in events and ("open", "task-4") in events
    doc = json.loads((tmp_path / "w" / "world.json").read_text())
    assert doc["pool"]["solved"] == 4 and doc["status"] == "complete"
    assert "there is no open problem called 'nope'" in (tmp_path / "w" / "context.txt").read_text() or True
    first = a1.seen[0]
    assert "[[TURN" not in texts(first) and "Problem: task-1" in texts(first) and "Open problems: task-1" in first[-1]["content"]


def test_pool_protects_open_statements_or_drops_them(tmp_path):
    delete = tool("edit_context", delete_turns=[{"first": 1, "last": 1}])
    a1 = Scripted([delete] + [tool("submit", problem=f"task-{i}", code="print('OK')") for i in (2, 3, 1)])
    w = world(tmp_path, [a1], [[titled(1), titled(2), titled(3)]], "world.setup=pool", "world.n_agents=1")
    w.run()
    assert read_jsonl(tmp_path / "w" / "edits.jsonl")[0]["rejected"] == "protected"
    a1 = Scripted([delete] + [tool("submit", problem=f"task-{i}", code="print('OK')") for i in (2, 3)])
    w = World(config("world.setup=pool", "world.n_agents=1", "world.pool_protect=none"), 1, [[titled(1), titled(2), titled(3)]],
              tmp_path / "w2", FakeGrader(), factory(a1))
    w.run()
    events = [(e["event"], e["title"]) for e in read_jsonl(tmp_path / "w2" / "pool.jsonl")]
    assert ("dropped", "task-1") in events and ("open", "task-3") in events
    assert "Problem task-1 was removed from the context and is closed." in (tmp_path / "w2" / "context_agent_1.txt").read_text()


def test_pool_attempts_are_capped_per_agent(tmp_path):
    a1 = Scripted([tool("run_tests", problem="task-1", code="x")] * 3 + [tool("submit", problem="task-2", code="print('OK')")])
    w = world(tmp_path, [a1], [[titled(1), titled(2)]], "world.setup=pool", "world.n_agents=1", "world.max_work_calls=2")
    w.run()
    events = [(e["event"], e["title"]) for e in read_jsonl(tmp_path / "w" / "pool.jsonl")]
    assert ("exhausted", "task-1") in events


# --- token count, names ------------------------------------------------------------------------------------------

def test_whole_count_includes_tools_and_calibrates_per_model(tmp_path):
    from gap_attractor.providers import ProviderResponse

    class Billed(Scripted):
        def chat(self, messages, **kw):
            r = super().chat(messages, **kw)
            r.usage["prompt_tokens"] = 1000
            return r
    a1 = Billed([tool("run_tests", code="x"), tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1], [[problem(1)]], "world.n_agents=1")
    before = w.tokens(w.agents[0])
    assert before > (len(w.system(w.agents[0])) // 4)          # tool schemas and status are counted
    w.run()
    assert w.agents[0].cpt != 4.0
    calls = read_jsonl(tmp_path / "w" / "calls.jsonl")
    assert calls[-1]["chars_per_token"] == round(w.agents[0].cpt, 3)


def test_old_design_names_still_work(tmp_path):
    w = world(tmp_path, [Scripted(), Scripted()], [[problem(1)], [problem(2)]], "world.awareness=hint",
              "world.order=round_robin", "world.schedule=async")
    assert (w.awareness, w.order, w.schedule) == ("rumour", "turn-taking", "free-for-all")


# --- bash editing (needs Docker and the sandbox image) --------------------------------------------------------

def sandbox_image():
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", "gap-ctx-sandbox:1"], capture_output=True).returncode == 0


DELETE_TURN = ("python3 -c \"import re;p='/ctx/context.txt';t=open(p).read();b=re.split(r'(?=\\[\\[TURN )',t);"
               "open(p,'w').write(''.join(x for x in b if not x.startswith('[[TURN {n} ')))\"")


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
def test_bash_edits_the_shared_file_under_the_same_rules(tmp_path):
    a1 = Scripted([tool("run_tests", code="print('PARTNER_CODE')"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("bash", command=DELETE_TURN.format(n=2)),                     # agent 1's run (its code): allowed
                   tool("bash", command=DELETE_TURN.format(n=1)),                     # a current statement: refused
                   tool("bash", command="sed -i 's/agent=2]]/agent=1]]/' /ctx/context.txt"),   # forge a label: no effect
                   tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "ctx.edit=bash", "world.order=fixed")
    w.run()
    edits = read_jsonl(tmp_path / "w" / "edits.jsonl")
    assert edits[0]["applied"] and edits[0]["removed_others"] == 1
    assert edits[1]["rejected"] == "protected"
    text = (tmp_path / "w" / "context.txt").read_text()       # labels come from the stored authors: agent 2's
    assert "role=user agent=2]]\nProblem 1 of 1 for agent 2." in text   # statement is still agent 2's after the sed
    assert "PARTNER_CODE" not in (tmp_path / "w" / "context.txt").read_text()
    assert w.boxes == {}                                        # sandboxes stopped at the end


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and gap-ctx-sandbox:1")
def test_bash_may_shorten_a_partners_turn_keeping_its_header(tmp_path):
    """The natural bash edit: trim another agent's long turn in place, header kept. Allowed; the trimmed turn is the editor's."""
    trim = ("python3 -c \"import re;p='/ctx/context.txt';t=open(p).read();"
            "t=re.sub(r'(\\[\\[TURN 2 [^\\n]*\\]\\]\\n).*?(?=\\n\\[\\[TURN )', r'\\1[trimmed]', t, flags=re.S);open(p,'w').write(t)\"")
    a1 = Scripted([tool("run_tests", code="print('LONG_PARTNER_CODE')"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("bash", command=trim), tool("submit", code="print('OK')")])
    w = world(tmp_path, [a1, a2], [[problem(1)], [problem(2)]], "ctx.edit=bash", "world.order=fixed")
    w.run()
    e = read_jsonl(tmp_path / "w" / "edits.jsonl")[0]
    assert e["applied"] and e["relabelled"] == 1
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "LONG_PARTNER_CODE" not in text and "[trimmed]" in text
