"""A scripted model and a fake grader for the world tests."""
from __future__ import annotations

import json
import re

from gap_attractor.providers import ProviderResponse


def problem(i: int) -> dict:
    return {"id": f"p{i}", "title": f"P{i}", "question_content": f"Print {i}.", "platform": "atcoder",
            "difficulty": "medium", "contest_date": "2025-02-01T00:00:00", "starter_code": "",
            "public": [{"input": "", "output": str(i), "testtype": "stdin"}], "private": [], "fn_name": None}


class FakeGrader:
    """Passes a program that contains OK."""
    image_id = "fake"

    def __init__(self):
        self.calls = []

    def grade(self, problem, code, which="all"):
        self.calls.append((problem["id"], which, code))
        ok = "OK" in (code or "")
        meta = {} if ok else {"error_code": -2, "inputs": "", "output": "0", "expected": "1"}
        return {"passed": ok, "n_tests": 1, "n_passed_before_fail": int(ok), "results": [ok], "metadata": meta,
                "seconds": 0.0, "cached": False, "feedback": "" if ok else "The above code is incorrect and got a wrong answer."}

    def close(self):
        pass


def tool(name, **arguments):
    return {"id": f"c_{name}", "name": name, "arguments": arguments}


class Scripted:
    """Replies from a list: each item is a tool call dict, a string (text only), or a function(messages) -> either.
    When the list is exhausted it submits a passing program. Keeps every message list it was sent."""

    def __init__(self, script=(), model="fake/model"):
        self.script, self.model, self.seen = list(script), model, []

    def chat(self, messages, **_):
        self.seen.append(messages)
        item = self.script.pop(0) if self.script else tool("submit", code="print('OK')")
        if callable(item):
            item = item(messages)
        if isinstance(item, str):
            return ProviderResponse(item, {"prompt_tokens": 10, "completion_tokens": 5}, "stop", 0, None, [])
        if "finish" in item:      # {"text": ..., "finish": "length"}: a reply cut off at the limit
            return ProviderResponse(item["text"], {"prompt_tokens": 10, "completion_tokens": item.get("tokens", 8000)},
                                    item["finish"], 0, None, [])
        item = dict(item)
        reasoning = item.pop("reasoning", None)   # {"reasoning": "...", **tool(...)}: the provider's own reasoning text
        return ProviderResponse("thinking aloud", {"prompt_tokens": 10, "completion_tokens": 5}, "tool_calls", 0, None, [],
                                [item], reasoning)


def factory(*scripts):
    """make_provider for World: the k-th agent gets the k-th script."""
    queue = list(scripts)

    def make(pcfg):
        s = queue.pop(0)
        s.model = str(pcfg.model)
        return s
    return make


def turn_numbers(messages, pattern):
    """Numbers of the turns whose header + text match `pattern` (in the messages an agent was sent)."""
    found = []
    for m in messages:
        content = m.get("content") or ""
        for block in re.split(r"(?=\[\[TURN )", content):
            head = re.match(r"\[\[TURN (\d+) ", block)
            if head and re.search(pattern, block):
                found.append(int(head.group(1)))
    return found


def read_jsonl(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []
