"""Provider boundary: traversal code only consumes ProviderResponse."""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field

from gap_attractor.registry import PROVIDERS


@dataclass
class ProviderResponse:
    text: str | None
    usage: dict
    finish_reason: str | None
    retries: int
    error: str | None
    attempts: list[dict]
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments": dict}] when tools were offered
    internal_reasoning: str | None = None  # the model's own thinking, if the service returns it (never shown back to the model)
    error_kind: str | None = None    # context_limit | client_error | retries_exhausted | missing_key
    latency_s: float = 0.0           # time spent in HTTP requests only (no backoff waits)
    response_id: str | None = None   # provider generation/request ID when available
    response_model: str | None = None


class MockProvider:
    """Offline stand-in for a model. Picks a random listed title; searches when a search is asked for.

    - Search-first variants B/D (no list shown): replies with a query, cycling through QUERY_TERMS.
    - Search-anytime variants C/D: every second page it searches, otherwise follows.
    """
    name = "mock"
    QUERY_TERMS = ("atlas", "archive", "learning", "topic", "history")

    def __init__(self, model: str = "mock-random", **_: object):
        self.model = model
        self.calls = 0
        self.queries = 0
        self.action_prompts = 0

    def chat(self, messages: list[dict], *, seed: int, tools: list[dict] | None = None, tool_choice: str | None = None,
             **_: object) -> ProviderResponse:
        system, observation = messages[0]["content"] or "", messages[-1].get("content") or ""
        if '"selections"' in system:  # paired post-run readout fixture
            all_text = "\n".join(message.get("content") or "" for message in messages)
            choices = re.findall(r"^\s*-\s+(.+)$", all_text, flags=re.M)
            match = re.search(r"exactly (\d+) distinct", system)
            count = int(match.group(1)) if match else min(10, len(choices))
            payload = {"selections": [{"title": title, "reason": "[mock] selected for the fixture readout."}
                                      for title in choices[:count]],
                       "selection_principle": "[mock] first titles in the controlled fixture list."}
            self.calls += 1
            return self._reply(payload)
        if tools and tool_choice != "none":
            reply = self._tool_reply(observation, [t["function"]["name"] for t in tools], seed)
            if any("note" in (t["function"].get("parameters") or {}).get("properties", {}) for t in tools):
                reply.tool_calls[0]["arguments"]["note"] = f"[mock] note from call {self.calls}."  # self_notes memory
            return reply
        choices = re.findall(r'^\s*"page_title"\s*:\s*"([^"]+)"', observation, flags=re.M)
        if not choices:  # bullet lists: page links ("- Title") and search results ("- Title — snippet")
            choices = [line.split(" — ")[0].strip() for line in re.findall(r"^\s*-\s+(.+)$", observation, flags=re.M)]
        self.calls += 1
        rng = random.Random(seed + self.calls)
        is_results = observation.startswith("Search results")
        if '"intent"' in system:  # intent call: name a page (title-like phrase from the article, else a canned term)
            phrases = [p for p in re.findall(r"\b[A-Z][a-z]+(?: [A-Z0-9][a-z0-9]*)?", observation.split("## Article content", 1)[-1])
                       if p not in {"This", "The", "It", "A", "An", "In"}]
            return self._reply({"intent": rng.choice(phrases) if phrases else self.QUERY_TERMS[self.calls % len(self.QUERY_TERMS)]})
        if '"query"' in system and not choices and not is_results:  # asked to search, nothing to pick from
            query = self.QUERY_TERMS[self.queries % len(self.QUERY_TERMS)]; self.queries += 1
            return self._reply({"reasoning": "[mock] searching.", "query": query})
        if '"action"' in system and not is_results and choices:  # search-anytime: follow or search
            self.action_prompts += 1
            if self.action_prompts % 2 == 0:
                return self._reply({"reasoning": "[mock] searching.", "action": "search", "query": rng.choice(choices).split()[0]})
            return self._reply({"reasoning": "[mock] following.", "action": "follow", "choice": rng.choice(choices)})
        choice = rng.choice(choices) if choices else "STOP"
        payload = {"reasoning": "[mock] exercising the harness.", "choice": choice}
        if '"note"' in system:  # self_notes memory asks for a note
            payload["note"] = f"[mock] note from call {self.calls}."
        return self._reply(payload)

    def _next_action_is_search(self):
        self.action_prompts += 1
        return self.action_prompts % 2 == 0

    def _tool_reply(self, observation, names, seed):
        """Tool-calling mode: search when only search is offered, alternate follow/search when both are, else follow."""
        choices = [line.split(" — ")[0].strip() for line in re.findall(r"^\s*-\s+(.+)$", observation, flags=re.M)]
        self.calls += 1
        rng = random.Random(seed + self.calls)
        if "search" in names and "follow_link" not in names:
            query = self.QUERY_TERMS[self.queries % len(self.QUERY_TERMS)]; self.queries += 1
            name, args = "search", {"query": query, "reasoning": "[mock] searching."}
        elif "search" in names and choices and self._next_action_is_search():
            name, args = "search", {"query": rng.choice(choices).split()[0], "reasoning": "[mock] searching."}
        elif choices and "follow_link" in names:
            name, args = "follow_link", {"title": rng.choice(choices), "reasoning": "[mock] following."}
        else:
            name, args = "stop", {"reason": "[mock] nothing to follow."}
        call = {"id": f"call_{self.calls}", "name": name, "arguments": args}
        return ProviderResponse(None, {"prompt_tokens": 0, "completion_tokens": 0}, "tool_calls", 0, None,
                                [{"attempt": 0, "provider": self.name, "model": self.model, "status": "success"}], [call])

    def _reply(self, payload):
        return ProviderResponse(json.dumps(payload), {"prompt_tokens": 0, "completion_tokens": 0}, "stop", 0, None,
                                [{"attempt": 0, "provider": self.name, "model": self.model, "status": "success"}])


RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504, 529)
CONTEXT_LIMIT_HINTS = ("context length", "context_length", "maximum context", "too many tokens", "prompt is too long",
                       "request too large", "payload too large", "reduce the length")


def retry_after_seconds(response):
    """Seconds asked for by a Retry-After header (number or HTTP date), or None."""
    value = (response.headers or {}).get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        from email.utils import parsedate_to_datetime
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


SAMPLING_PARAMS = ("temperature", "top_p", "seed")
_SUPPORTED: dict[str, dict[str, set]] = {}   # base_url -> model -> supported parameter names (one lookup per process)


def supported_parameters(base_url: str, model: str, timeout: float = 30) -> set | None:
    """OpenRouter's list of request parameters a model accepts (GET /models), or None when unknown."""
    if base_url not in _SUPPORTED:
        import requests
        try:
            data = requests.get(base_url + "/models", timeout=timeout).json()["data"]
            _SUPPORTED[base_url] = {m["id"]: set(m.get("supported_parameters") or []) for m in data}
        except Exception as error:  # no list: send everything, as before
            print(f"[provider] could not read {base_url}/models ({type(error).__name__}); sending all parameters", file=sys.stderr)
            _SUPPORTED[base_url] = {}
    return _SUPPORTED[base_url].get(model)


class OpenAICompatibleProvider:
    """POST /chat/completions with retries.

    Retried (exponential backoff, or the server's Retry-After, capped at max_backoff_s): network errors, timeouts,
    HTTP 408/425/429/5xx, and 200 responses without choices. With `tools`, the actions are offered as OpenAI tool
    definitions and tool calls come back parsed in ProviderResponse.tool_calls. Not retried: a request too large for the model
    (HTTP 413, or 400 mentioning the context length) -> error_kind "context_limit"; any other 4xx -> "client_error".

    OpenRouter (drop_unsupported_params, default on): temperature, top_p and seed are left out of the request when the
    model does not accept them (OpenRouter's /models list), instead of being silently ignored by the service or, with
    tools, making routing fail. Every attempt record names what was left out (`dropped_params`, in step
    extras.retry_attempts), so a run shows when its sampling settings could not apply; `params_unchecked` marks attempts
    whose model was not in the list (or the list could not be read), so nothing was dropped and nothing is known.
    run.json `effective_sampling` sums this up per run.
    """
    def __init__(self, name: str, model: str, base_url: str, api_key_env: str, timeout: int = 120, max_retries: int = 4,
                 max_tokens: int = 700, extra_body: dict | None = None, max_backoff_s: float = 60, drop_unsupported_params: bool | None = None):
        self.name, self.model, self.base_url, self.api_key_env = name, model, base_url.rstrip("/"), api_key_env
        self.drop_unsupported = (name == "openrouter") if drop_unsupported_params is None else bool(drop_unsupported_params)
        self.timeout, self.max_retries, self.max_tokens = timeout, max_retries, int(max_tokens)
        self.extra_body, self.max_backoff_s = dict(extra_body or {}), float(max_backoff_s)

    def _wait(self, attempt, response=None):
        asked = retry_after_seconds(response) if response is not None else None
        wait = min(asked if asked is not None else 2 ** attempt, self.max_backoff_s)
        time.sleep(wait)
        return round(wait, 3)

    def chat(self, messages: list[dict], *, seed: int, temperature: float, top_p: float, max_tokens: int | None = None,
             tools: list[dict] | None = None, tool_choice: str | None = None,
             request_overrides: dict | None = None) -> ProviderResponse:
        import requests
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return ProviderResponse(None, {}, None, 0, f"missing environment variable {self.api_key_env}", [], error_kind="missing_key")
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p,
                   "max_tokens": int(max_tokens or self.max_tokens), "seed": seed, **self.extra_body}
        # Per-call controls are useful for isolated probes (for example a short structured forecast) without changing
        # the later navigation request. Keep this deliberately narrow: callers cannot replace the model, messages,
        # tools, sampling parameters, or provider routing through this hook.
        allowed_overrides = {"reasoning", "reasoning_effort", "include_reasoning", "response_format"}
        for override_key, value in dict(request_overrides or {}).items():
            if override_key not in allowed_overrides:
                raise ValueError(f"unsupported request override {override_key!r}; expected one of {sorted(allowed_overrides)}")
            payload[override_key] = value
        if self.name == "openrouter":
            # OpenRouter only returns provider-accounted cost when explicitly asked.
            payload["usage"] = {"include": True}
        dropped, unchecked = [], False
        if self.drop_unsupported:
            supported = supported_parameters(self.base_url, self.model)
            unchecked = not supported
            if supported:
                dropped = [p for p in SAMPLING_PARAMS if p not in supported]
                for p in dropped:
                    payload.pop(p, None)
        if tools:
            payload["tools"], payload["tool_choice"] = tools, tool_choice or "auto"
            if self.name == "openrouter":  # only route to providers that actually support tool calling
                payload.setdefault("provider", {})["require_parameters"] = True
        attempts, last, api_seconds = [], None, 0.0
        for attempt in range(self.max_retries + 1):
            record = {"attempt": attempt, "provider": self.name, "model": self.model}
            if dropped: record["dropped_params"] = dropped
            if unchecked: record["params_unchecked"] = True   # no /models entry: sent everything, may be ignored
            attempts.append(record)
            started = time.time()
            try:
                response = requests.post(self.base_url + "/chat/completions", json=payload, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            except Exception as error:  # network error / timeout: retryable
                api_seconds += time.time() - started
                last = f"{type(error).__name__}: {error}"
                record["error"] = last
                if attempt < self.max_retries: record["wait_s"] = self._wait(attempt)
                continue
            api_seconds += time.time() - started
            status = record["status_code"] = response.status_code
            if status == 200:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                except Exception as error:  # e.g. a 200 carrying {"error": ...}: retryable
                    last = f"malformed response: {type(error).__name__}: {response.text[:300]}"
                    record["error"] = last
                    if attempt < self.max_retries: record["wait_s"] = self._wait(attempt)
                    continue
                message = choice.get("message") or {}
                return ProviderResponse(message.get("content"), data.get("usage", {}), choice.get("finish_reason"), attempt, None, attempts,
                                        parse_tool_calls(message.get("tool_calls")), parse_internal_reasoning(message),
                                        latency_s=round(api_seconds, 3), response_id=data.get("id"), response_model=data.get("model"))
            body = (response.text or "")[:500]
            last = f"HTTP {status}: {body}"
            record["error"] = body
            if status == 413 or (status == 400 and any(hint in body.lower() for hint in CONTEXT_LIMIT_HINTS)):
                return ProviderResponse(None, {}, None, attempt, last, attempts, error_kind="context_limit", latency_s=round(api_seconds, 3))
            in_flight = status == 402 and "in_flight" in body.lower()   # OpenRouter: the key's in-flight budget is used up by concurrent requests; settles by itself
            if status not in RETRYABLE_STATUS and not in_flight:
                return ProviderResponse(None, {}, None, attempt, last, attempts, error_kind="client_error", latency_s=round(api_seconds, 3))
            if attempt < self.max_retries: record["wait_s"] = self._wait(attempt, response)
        return ProviderResponse(None, {}, None, self.max_retries, last, attempts, error_kind="retries_exhausted", latency_s=round(api_seconds, 3))


OPENAI_COMPATIBLE_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


@PROVIDERS.register("mock")
def _mock(cfg):
    return MockProvider(cfg.model)


def _openai_compatible(name):
    def factory(cfg):
        base, key = OPENAI_COMPATIBLE_DEFAULTS[name]
        extra = cfg.get("extra_body")
        if extra is not None and not isinstance(extra, dict):
            from omegaconf import OmegaConf
            extra = OmegaConf.to_container(extra, resolve=True)
        return OpenAICompatibleProvider(cfg.name, cfg.model, cfg.get("base_url") or base,
                                        cfg.get("api_key_env") or key, cfg.get("timeout", 120), cfg.get("max_retries", 4),
                                        cfg.get("max_tokens", 700), extra, cfg.get("max_backoff_s", 60),
                                        cfg.get("drop_unsupported_params"))
    return factory


for _name in OPENAI_COMPATIBLE_DEFAULTS:
    PROVIDERS.register(_name)(_openai_compatible(_name))


def parse_tool_calls(raw):
    """OpenAI-format tool calls -> [{"id", "name", "arguments": dict}]. Unreadable arguments become {}."""
    calls = []
    for item in raw or []:
        function = item.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except Exception:
            arguments = {}
        calls.append({"id": item.get("id"), "name": function.get("name"), "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def parse_internal_reasoning(message):
    """The model's thinking, where the service returns it: OpenRouter `reasoning` / `reasoning_details`,
    DeepSeek-style `reasoning_content`. None when the model or service hides it (e.g. OpenAI models)."""
    for key in ("reasoning", "reasoning_content"):
        if isinstance(message.get(key), str) and message[key].strip():
            return message[key].strip()
    parts = []
    for detail in message.get("reasoning_details") or []:
        if isinstance(detail, dict):
            text = detail.get("text") or detail.get("summary")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts) or None


def make_provider(cfg):
    """Build the provider named by `provider.name`: a registered name (see PROVIDERS) or a dotted import path
    to a factory taking the provider config. Any object with the `chat(messages, *, seed, temperature, top_p)`
    method returning a ProviderResponse works."""
    return PROVIDERS.get(cfg.name)(cfg)
