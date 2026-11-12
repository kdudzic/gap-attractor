"""Shared test helpers: a scripted provider, tool-call replies and a config composed like run.py's."""
from __future__ import annotations

import json
from pathlib import Path

from gap_attractor.providers.base import ProviderResponse

import run as entry

USAGE = {"prompt_tokens": 10, "completion_tokens": 2}


def call(name, n, **arguments):
    return {"id": f"c{n}", "name": name, "arguments": arguments}


def reply(*calls):
    return ProviderResponse("", USAGE, "tool_calls", 0, None, [], list(calls))


def links_in(messages):
    """Titles of the newest page's candidate links in a request (the newest message that lists links: a user message,
    or the tool result of the move that opened the page)."""
    for m in reversed(messages[1:]):
        content = m.get("content") or ""
        if m["role"] in ("user", "tool") and "## Candidate links" in content:
            part = content.rsplit("## Candidate links", 1)[1]
            return [l[2:] for l in part.splitlines() if l.startswith("- ")]
    return []


def text_of(messages):
    """Everything a request shows, as one string (tool calls included)."""
    out = []
    for m in messages:
        out.append(m.get("content") or "")
        out += [c["function"]["name"] + " " + c["function"]["arguments"] for c in m.get("tool_calls") or []]
    return "\n".join(out)


class Scripted:
    """Replies from a list (callables get the request); when it runs out, follows the first link of the newest page."""
    def __init__(self, replies=(), model="scripted"):
        self.replies, self.requests, self.model = list(replies), [], model

    def chat(self, messages, *, seed, tools=None, **_):
        self.requests.append({"messages": messages, "tools": [t["function"]["name"] for t in tools or []]})
        if self.replies:
            item = self.replies.pop(0)
            return item(messages) if callable(item) else item
        return reply(call("follow_link", 99, title=links_in(messages)[0]))


def config(tmp_path: Path, extra=()):
    return entry.compose([f"output.directory={tmp_path}", "batch.start_pages=[Civic Atlas]", "experiment.step_limit=3", *extra])


def read_steps(dataset):
    folder = next(p for p in sorted(dataset.root.iterdir()) if (p / "steps.jsonl").exists())
    return [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines()], folder
