"""What each candidate model does with each reasoning setting (one small call per model and setting; cents in all).

    uv run --env-file .env python experiments/siamese/probe_reasoning.py [--out <file.json>] [model ...]

For the study's reasoning decisions (D9: thinking off or minimal so the work is visible in replies; "shared minds":
the provider's reasoning written into the shared context, which needs models that return their reasoning *text*).
Each call asks for a tiny program and offers a submit tool, as a world would. Recorded per model and setting:
whether the setting is accepted (error otherwise), reasoning tokens billed, whether reasoning text came back and how
long it is (message.reasoning or reasoning_details), whether the model called the tool, completion tokens, cost.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.8-flash", "qwen/qwen3.8-27b",
          "z-ai/glm-5.3-flash", "xiaomi/mimo-v2.6-pro"]
SETTINGS = {"off": {"enabled": False}, "minimal": {"effort": "minimal"}, "low": {"effort": "low"},
            "medium": {"effort": "medium"}, "high": {"effort": "high"}}
TOOL = {"type": "function", "function": {"name": "submit", "description": "Submit a complete program.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
PROMPT = ("Read N and then N integers from stdin and print how many of them are prime. Think it through, then submit a "
          "complete Python program with the submit tool.")


def call(model: str, reasoning: dict, key: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "tools": [TOOL], "max_tokens": 4000,
            "reasoning": reasoning, "usage": {"include": True}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")[:300]}
    except Exception as e:                      # noqa: BLE001  (a probe: record and go on)
        return {"error": repr(e)[:300]}
    if "error" in d:
        return {"error": str(d["error"])[:300]}
    m, u = d["choices"][0]["message"], d.get("usage") or {}
    text = m.get("reasoning") or "".join(x.get("text") or x.get("summary") or "" for x in m.get("reasoning_details") or [])
    return {"reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "reasoning_text_chars": len(text or ""), "reasoning_head": (text or "")[:200],
            "called_tool": bool(m.get("tool_calls")), "reply_chars": len(m.get("content") or ""),
            "completion_tokens": u.get("completion_tokens"), "cost": u.get("cost"),
            "finish_reason": d["choices"][0].get("finish_reason")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="*", default=MODELS)
    ap.add_argument("--out", default="probe_reasoning.json")
    a = ap.parse_args(argv)
    key = os.environ["OPENROUTER_API_KEY"]
    rows = []
    for model in a.models:
        for name, setting in SETTINGS.items():
            r = dict(model=model, setting=name, **call(model, setting, key))
            rows.append(r)
            print(f"{model:32s} {name:8s} " + (f"ERROR {r['error'][:90]}" if "error" in r else
                  f"reasoning tok {r['reasoning_tokens']}, text {r['reasoning_text_chars']} chars, "
                  f"tool {r['called_tool']}, out {r['completion_tokens']}, ${r['cost']}"), flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
