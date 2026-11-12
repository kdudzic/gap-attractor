"""The context budget: size estimate, the readout, nudges, the hard limit, and (soft budget) page capping.

The model is shown the LIMIT, the budget minus the reserve (the target; as CLM, which advertises what it enforces):
8 192 - 2 048 = 6 144 tokens by default. Sizes count the whole request (system message included) except the newest
list of links, which the agent cannot edit until it moves. Nudges fire once each as the context passes 25 %, 50 %
and 75 % of the limit (re-armed when it shrinks).
- wall (default, hard): a page arrives whole, even when it takes the context past the limit, and then brings only
  the strongest warning ("over"). The agent may not move while it is over: its move is held (memory.hold_move) and
  it edits again, up to max_held_moves times per step; a move made while still over after that ends the run
  (termination context_budget). CLM instead rolls the newest turns back and asks for a compaction (up to 50
  retries by default); our newest turn is the page the agent has to act on, so it is kept and the move is held.
- soft: a page that does not fit is shortened, cutting the ARTICLE from its end and never the link list. Past
  hard_limit x budget the harness drops the oldest turns and says so in a notice.
Size is characters / chars_per_token unless tiktoken is installed (o200k); chars / 4 matched DeepSeek's reported
prompt tokens to a median ratio of 0.996 on the pilot.
"""
from __future__ import annotations

import math

LINKS_HEADING = "\n## Candidate links ("


class Budget:
    def __init__(self, cfg, tool: str = "edit_context"):
        self.budget = int(cfg.get("budget_tokens", 8192))
        self.reserve = int(cfg.get("reserve_tokens", 2048))
        self.target = self.budget - self.reserve
        self.chars_per_token = float(cfg.get("chars_per_token", 4.0))
        self.hard_limit = float(cfg.get("hard_limit", 1.5))
        self.tiers = [float(x) for x in cfg.get("nudge_tiers", [0.25, 0.5, 0.75])]
        self.urgent_room = float(cfg.get("urgent_room", 0.10))
        self.tool = tool                     # named in the nudges: edit_context | bash; None: no editing (no nudges)
        self.advice = str(cfg.get("prompt_style", "neutral")) == "clm"   # nudges suggest what to do (CLM) or only say the size
        self.fired: set[float] = set()
        self.recent_pages: list[int] = []
        self._encoder, self.tokenizer = None, f"chars/{self.chars_per_token:g}"
        if cfg.get("tokenizer", "auto") in ("auto", "tiktoken"):
            try:
                import tiktoken
                self._encoder, self.tokenizer = tiktoken.get_encoding("o200k_base"), "tiktoken-o200k"
            except Exception:
                if cfg.get("tokenizer") == "tiktoken":
                    raise

    def tokens(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return math.ceil(len(text) / self.chars_per_token)

    def readout(self, tokens: int, show_limit: bool = True) -> str:
        if not show_limit:                    # no budget shown (ctx.readout=false): the size alone
            return f"[context: ~{tokens} tokens]"
        over = " — over the limit" if tokens > self.target else ""
        return f"[context: ~{tokens}/{self.target} tokens{over}]"

    def over(self, tokens: int, calls: int = 3) -> tuple[str, str]:
        """The strongest warning (hard budget): the context is past the limit (a page that arrived whole, or the
        model's own turns). The agent cannot move until it is back under the limit."""
        self.fired = set(self.tiers)          # the lower tiers are past; they re-arm when the context shrinks
        how = "summarise or remove turns (the page you just opened can be summarised too" if self.advice else \
            "shorten or remove turns (the page you just opened can be shortened too"
        return "over", (f"Over the limit: your context is ~{tokens} tokens and may hold {self.target}. You cannot "
                        f"move until it fits: use {self.tool} to {how}; the newest list of links does not count). If "
                        f"you run out of {self.tool} calls first, call follow_link anyway: the move is held and you get "
                        f"{calls} more calls.")

    def held(self, tokens: int, k: int, limit: int, stop: bool = False, calls: int = 3) -> str:
        """The answer to a move (or a STOP) made while over the limit (it is not made). Stopping is named only in the
        answer to a stop the model tried; nothing else the context says mentions it."""
        last = (" This was your last held move: if your context is still over the limit when you next move, the "
                "session ends after that.") if k >= limit else ""
        then = "then you can stop (or move)" if stop else "then call follow_link again"
        return (f"{'Not stopped' if stop else 'Not moved'}: your context is ~{tokens} tokens, over the limit of "
                f"{self.target} (the newest list of links does not count). You have {calls} more {self.tool} calls: "
                f"use them to make it fit, {then}.{last}")

    def rearm(self, tokens: int) -> None:
        """Tiers the context has fallen back below fire again (called with the size before a new page, so a
        compaction in the last step re-arms them even though no nudge was evaluated in between)."""
        self.fired = {t for t in self.fired if tokens / self.target >= t}

    def nudge(self, tokens: int, pages_per_round: int = 1, urgent: bool = True) -> tuple[str | None, str | None]:
        """(level, text) for this turn: tiers (fractions of the limit) fire once each and re-arm when the context falls
        back below them; 'urgent' (soft budget only) whenever the room left is less than what arrives before this
        agent's next turn: the mean of its last 3 pages x pages_per_round (the number of agents sharing the file), or
        urgent_room x the limit."""
        if self.tool is None:
            return None, None
        share = tokens / self.target
        self.fired = {t for t in self.fired if share >= t}
        room = self.target - tokens
        recent = self.recent_pages[-3:]
        if urgent and room < max(self.urgent_room * self.target, pages_per_round * sum(recent) / len(recent) if recent else 0):
            return "urgent", (f"Context at ~{tokens}/{self.target} tokens: the next page may not fit. Use {self.tool} "
                              "before you move to make room.")
        due = [t for t in self.tiers if share >= t and t not in self.fired]
        if not due:
            return None, None
        tier = max(due)
        self.fired.update(due)
        if not self.advice:                   # neutral: the size only, no suggestion what to do
            part = {0.25: "a quarter", 0.5: "half", 0.75: "three quarters"}.get(tier, f"{tier:.0%}")
            return f"{tier * 100:g}", f"Your context is about {part} full (~{tokens}/{self.target} tokens)."
        if tier <= 0.25:
            return "25", f"Context at ~{tokens}/{self.target} tokens. No action needed yet."
        if tier <= 0.5:
            return "50", (f"Context at ~{tokens}/{self.target} tokens. When convenient, use {self.tool} to summarise "
                          "turns you no longer need in full.")
        return "75", (f"Context at ~{tokens}/{self.target} tokens. Use {self.tool} now to replace settled turns with "
                      "short summaries.")

    def cap_page(self, observation: str, room_tokens: int) -> tuple[str, dict]:
        """Shorten a page observation to about room_tokens by cutting its article text from the end. The title and
        the whole link list are always kept (even if that is over the room). Returns (text, info)."""
        cut = observation.rfind(LINKS_HEADING)
        article, links = (observation[:cut], observation[cut:]) if cut >= 0 else (observation, "")
        info = {"capped": False, "article_chars": len(article), "article_chars_kept": len(article)}
        if self.tokens(observation) <= room_tokens:
            return observation, info
        marker = "\n\n[… the rest of this article is omitted: your context budget is full …]\n"
        room_chars = int(max(0, room_tokens - self.tokens(links)) * self.chars_per_token) - len(marker)
        title_end = article.find("\n") + 1 if "\n" in article else len(article)
        keep = max(title_end, min(len(article), room_chars))
        info.update(capped=True, article_chars_kept=keep)
        return article[:keep].rstrip() + marker + links, info
