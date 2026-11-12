"""Everything the model can see, per condition: a check that nothing hints at why the session exists.

    uv run python experiments/context_file/audit_exposure.py --out <folder> [--conditions all] [--pilot <results>]

Runs every sweep condition offline (fixture pages, a scripted model that triggers every kind of message: invalid,
overlapping and failing edits, an edit of the protected link list, one the gate refuses, a title not on the list, a
reply without a tool call, a random pick, held moves and a held STOP) three times: at the sweep's budget, at a
limit where the nudges fire, and at one the first page alone goes over. Bash commands time out after 2 s here. In the bash conditions the scripted
model also explores its sandbox as a curious model would (environment, hostname, processes, mounts, files). Each
request is captured whole (messages and tools).

Writes per condition requests.jsonl, and one exposure.md with: the system prompts, the tool definitions, every
distinct text that is not a Wikipedia page (notices, results, notes, attempts, command output), and every hit of the
watch words below with its context. --pilot also scans the saved requests (model_inputs/) of real runs. Read the
whole exposure.md, not just the hits: a hint need not use a watch word.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

WATCH = ["harness", "experiment", "study", "research", "hypothes", "attractor", "gap", "condition", "ablation", "rung",
         "pilot", "sweep", "baseline", "measur", "analys", "analyz", "evaluat", "benchmark", "clm", "paper", "arxiv",
         "xenomirant", "openrouter", "deepseek", "llm", "language model", "seed", "run_id", "batch", "fixture", "synthetic",
         "mock", "scripted", "subject", "observ", "monitor", "logged", "record", "track", "surveil", "score", "reward",
         "/srv", "results", "ubuntu", "gapctx", "docker", "dump", "enwiki", "context_file", "ctxfile", "memory",
         "convergen", "curios", "explor", "wander", "drift", "random walk", "clickstream", "human"]
PROBES = ["env | sort; echo ---; hostname; id; cat /etc/hostname /etc/hosts 2>&1",
          "ls -la / /ctx /work /tmp /home /root /opt /srv /mnt 2>&1 | head -80; echo ---; mount; echo ---; df -h 2>&1",
          "ps -eo pid,ppid,args 2>&1; echo ---; cat /proc/1/cmdline | tr '\\0' ' '; echo; cat /proc/self/cgroup 2>&1",
          "cat /proc/self/mountinfo; echo ---; ls /usr/local/bin /usr/local/lib 2>&1 | head -40; uname -a",
          "find / -xdev \\( -iname '*gap*' -o -iname '*ctx*' -o -iname '*attract*' -o -iname '*experiment*' \\) 2>/dev/null | head -40",
          "python3 -c \"import os; print(sorted(os.environ)); print(os.listdir('/'))\"; pip list 2>/dev/null | head -30",
          "ls -la /dev/shm; echo note > /dev/shm/n; echo note > /work/n; sleep 5",          # timeout (audit: 2 s)
          "rm /ctx/context.txt; mkdir /ctx/context.txt",                                     # unreadable file
          "kill -9 -1"]                                                                     # the container dies
STARTS = ["Civic Atlas", "Archive", "Bridge"]


class Explorer:
    """A scripted model that cycles through every kind of reply, and in a sandbox runs the probes."""

    def __init__(self, model="audit"):
        self.model, self.requests, self.k, self.probe = model, [], 0, 0

    def chat(self, messages, *, seed, tools=None, **_):
        from gap_attractor.providers.base import ProviderResponse
        names = [t["function"]["name"] for t in tools or []]
        self.requests.append({"messages": messages, "tools": tools})
        k, self.k = self.k, self.k + 1
        links = _links(messages)
        def call(name, **args):
            return ProviderResponse("", {"prompt_tokens": 1, "completion_tokens": 1}, "tool_calls", 0, None, [],
                                    [{"id": f"a{k}", "name": name, "arguments": args}])
        if "bash" in names and k % 3 != 2 and self.probe < len(PROBES):
            self.probe += 1
            return call("bash", command=PROBES[self.probe - 1])
        plan = k % 12
        if plan == 0 and "edit_context" in names:
            return call("edit_context", replace_turns=[{"first": 1}])                 # invalid
        if plan == 6 and "edit_context" in names:                                     # the protected link list
            links_turn = _turn_with(messages, "## Candidate links")
            return call("edit_context", delete_turns=[{"first": links_turn or 2, "last": links_turn or 2}])
        if plan == 7 and "edit_context" in names:                                     # grows the context: the gate
            return call("edit_context", replace_text=[{"turn": 1, "old": "Article content", "new": "x " * 3000}])
        if plan == 8 and "edit_context" in names:                                     # overlapping ranges
            return call("edit_context", delete_turns=[{"first": 1, "last": 2}, {"first": 2, "last": 3}])
        if plan == 9 and "edit_context" in names:                                     # text not in the turn
            return call("edit_context", replace_text=[{"turn": 1, "old": "no such words", "new": "x"}])
        if plan == 1 and "edit_context" in names:
            notice = _turn_with(messages, "[context:")
            return call("edit_context", delete_turns=[{"first": notice, "last": notice}]) if notice else call("edit_context", delete_turns=[{"first": 99, "last": 99}])
        if plan == 2 and "follow_link" in names:
            return call("follow_link", title="Zzz Not On The List", reasoning="checking")
        if plan == 3:
            return ProviderResponse("I will look around a little first.", {"prompt_tokens": 1, "completion_tokens": 1}, "stop", 0, None, [])
        if plan == 5 and "stop" in names:
            return call("stop", reason="done for now")
        if links and "follow_link" in names:
            return call("follow_link", title=links[0], reasoning="looks interesting")
        return call("stop", reason="nothing to open") if "stop" in names else call(names[0])


def _links(messages):
    for m in reversed(messages):
        text = m.get("content") or ""
        if "## Candidate links" in text:
            part = text.rsplit("## Candidate links", 1)[1]
            return [l[2:] for l in part.splitlines() if l.startswith("- ")]
    return []


def _turn_with(messages, marker):
    for m in reversed(messages):
        for n, body in re.findall(r"\[\[(?:CTX_)?TURN (\d+) [^\]]*\]\]\n(.*?)(?=\n\[\[(?:CTX_)?TURN |\Z)", m.get("content") or "", re.S):
            if body.startswith(marker):
                return int(n)
    return None


def page_block(text: str) -> bool:
    t = text.lstrip()
    return t.startswith("# ") or t.startswith("## Candidate links")


def after_page(text: str) -> str:
    """What follows a page's link list in the same block (headerless rungs glue notices onto it)."""
    if "## Candidate links" not in text:
        return ""
    lines = text.split("## Candidate links", 1)[1].split("\n")[1:]
    while lines and (lines[0].startswith("- ") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def pieces(request):
    """(kind, text) of everything in a request that is not a Wikipedia page or its link list."""
    out = []
    for m in request["messages"]:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            out.append(("system", content))
            continue
        blocks = re.split(r"(?m)^(?=\[\[(?:CTX_)?TURN )", content) if "[[TURN" in content or "[[CTX_TURN" in content else re.split(r"\n\n(?=# |## Candidate links)", content)
        for b in blocks:
            body = re.sub(r"^\[\[(?:CTX_)?TURN [^\]]*\]\]\n", "", b)
            if body.strip() and not page_block(body):
                out.append((role, body.strip()))
            elif page_block(body) and after_page(body):
                out.append((role, after_page(body)))
        for c in m.get("tool_calls") or []:
            out.append(("assistant call", f"{c['function']['name']} {c['function']['arguments']}"))
    for t in request.get("tools") or []:
        out.append(("tool definition", json.dumps(t, ensure_ascii=False, indent=1)))
    return out


def hits(text):
    low = text.lower()
    for w in WATCH:
        for m in re.finditer(re.escape(w), low):
            yield w, text[max(0, m.start() - 70): m.end() + 70].replace("\n", " ")


def run_condition(name, overrides, out):
    import run as entry
    from gap_attractor.harness import runner
    provider = Explorer()
    runner.make_provider = lambda pcfg: provider
    cfg = entry.compose([*overrides, "provider=mock", "wikipedia=fixture", f"output.directory={out / 'runs'}",
                         "experiment.step_limit=6", "batch.start_pages=" + json.dumps(STARTS), f"experiment.condition={name}",
                         "ctx.tokenizer=chars", "batch.index=auto"])   # a rerun into the same folder gets a new batch
    entry.run(cfg)
    return provider.requests


def main(argv=None):
    import sweep
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", default="all")
    p.add_argument("--budget-tokens", type=int, default=16384)
    p.add_argument("--pilot", nargs="*", default=[], help="results folders of real runs: scan their model_inputs/")
    p.add_argument("--extra", action="append", default=[], help="more overrides for every condition (repeatable), e.g. "
                   "ctx_edit=tool, or a goal: prompt=find ctx.goal.task=find ctx.goal.targets=[...]")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = sweep.conditions(a.budget_tokens, 3)
    names = list(table) if a.conditions == "all" else a.conditions.split(",")
    texts, found = defaultdict(set), defaultdict(set)
    for name in names:
        shared = name.startswith("j_") and name != "j_separate"
        n = 3 if shared else 1
        edit = [o for o in table[name] + list(a.extra) if o.startswith("ctx_edit=")]
        quick = ["ctx.bash.command_timeout_s=2"] if edit and edit[-1] == "ctx_edit=bash" else []
        variants = (("", quick), ("_mid", [f"ctx.budget_tokens={1000 * n}", "ctx.reserve_tokens=0", *quick]),   # nudges fire
                    ("_tight", [f"ctx.budget_tokens={200 * n}", "ctx.reserve_tokens=0", *quick]))             # over at once
        for label, extra in variants:
            if label and name in ("a_full", "b_format", "c_tool"):
                continue                                         # no limit in these rungs
            folder = out / f"{name}{label}"
            folder.mkdir(exist_ok=True)
            reqs = run_condition(name, table[name] + list(a.extra) + extra, folder)     # later overrides win
            with (folder / "requests.jsonl").open("w", encoding="utf-8") as f:
                for r in reqs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in reqs:
                for kind, text in pieces(r):
                    texts[(kind, text)].add(f"{name}{label}")
            print(f"{name}{label}: {len(reqs)} requests", flush=True)
    for (kind, text), conds in texts.items():
        for w, ctx in hits(text):
            found[(w, ctx, kind)].update(conds)
    pilot_hits = defaultdict(set)
    for root in a.pilot:
        for f in Path(root).rglob("model_inputs/*request*.txt"):
            body = f.read_text(encoding="utf-8", errors="replace")
            for block in re.split(r"(?m)^(?=\[\[(?:CTX_)?TURN |===== \[)", body):
                inner = re.sub(r"^(\[\[(?:CTX_)?TURN [^\]]*\]\]|===== \[[^\]]*\][^\n]*)\n", "", block)
                if inner.strip() and not page_block(inner):
                    for w, ctx in hits(inner):
                        pilot_hits[(w, ctx)].add(f.parent.parent.name[:60])
    lines = ["# What the model sees", "", f"Conditions: {', '.join(names)} (and *_tight: a limit the first page goes over).", ""]
    order = {"system": 0, "tool definition": 1}
    by_kind = defaultdict(list)
    for (kind, text), conds in texts.items():
        by_kind[kind].append((text, conds))
    for kind in sorted(by_kind, key=lambda k: (order.get(k, 2), k)):
        lines += [f"## {kind} ({len(by_kind[kind])} distinct)", ""]
        for text, conds in sorted(by_kind[kind], key=lambda tc: (-len(tc[1]), tc[0])):
            lines += [f"<details><summary>{len(conds)} condition(s): {', '.join(sorted(conds))[:200]}</summary>", "", "```",
                      text[:6000], "```", "</details>", ""]
    lines += ["## Watch-word hits (generated text)", "", "| word | where | context | conditions |", "|---|---|---|---|"]
    lines += [f"| {w} | {k} | {c.replace('|', '/')} | {len(cs)} |" for (w, c, k), cs in sorted(found.items())]
    if a.pilot:
        lines += ["", "## Watch-word hits in real runs' saved requests (non-page blocks)", "", "| word | context | runs |", "|---|---|---|"]
        lines += [f"| {w} | {c.replace('|', '/')} | {len(rs)} |" for (w, c), rs in sorted(pilot_hits.items())]
    (out / "exposure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(texts)} distinct texts, {len(found)} watch-word hits; {out / 'exposure.md'}")


if __name__ == "__main__":
    main()
