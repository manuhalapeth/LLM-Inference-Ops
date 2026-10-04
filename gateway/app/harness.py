"""Harnesses: checks that stop bad requests before they use GPU time.

Once a request reaches vLLM it gets compute whether it deserves it or not, so
every check here runs in the gateway first. Each check either passes or
returns a Rejection with an HTTP status and a reason that is logged, counted
in metrics and recorded on the trace.
"""

import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass
class Rejection:
    status: int
    reason: str
    message: str
    retry_after_s: float | None = None


# Phrases that try to override the application's instructions. A simple
# first line of defence: cheap, explainable, and easy to extend. It catches
# the common phrasings, not a determined attacker (see design_choices).
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|messages)",
    r"\bdisregard\s+(all\s+|any\s+)?(the\s+|your\s+)?(previous|prior|above|earlier|system)\s+(instructions|prompts|rules|messages)",
    r"\bforget\s+(everything|all)\s+(you\s+were\s+told|your\s+instructions|previous\s+instructions)",
    r"\b(reveal|show|print|repeat|output|leak|tell\s+me)\s+(me\s+)?(your|the)\s+(full\s+|entire\s+|exact\s+)?(system\s+prompt|hidden\s+instructions|initial\s+instructions|system\s+message)",
    r"\byou\s+are\s+now\s+(dan|in\s+developer\s+mode|jailbroken|unrestricted)\b",
    r"\b(pretend|act\s+as\s+if)\s+you\s+(have\s+no|are\s+free\s+of|don'?t\s+have)\s+(rules|restrictions|guidelines|filters)",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


class TokenCounter:
    """Counts prompt tokens the way vLLM will, before the request is sent.

    Qwen's chat template adds 5 tokens per message, 3 for the assistant turn,
    and a 21 token default system prompt when the request has none. Checked
    against vLLM's own counts on real requests (exact match).
    """

    PER_MESSAGE = 5
    ASSISTANT_PREFIX = 3
    DEFAULT_SYSTEM_PROMPT = 21

    def __init__(self, tokenizer_path: str):
        self.tokenizer = None
        if tokenizer_path != "none" and Path(tokenizer_path).exists():
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)

    @property
    def exact(self) -> bool:
        return self.tokenizer is not None

    def _count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        return math.ceil(len(text) / 4)  # rough fallback

    def chat(self, messages: list[dict]) -> int:
        tokens = sum(self._count(_text(m.get("content"))) for m in messages)
        tokens += self.PER_MESSAGE * len(messages) + self.ASSISTANT_PREFIX
        if not any(m.get("role") == "system" for m in messages):
            tokens += self.DEFAULT_SYSTEM_PROMPT
        return tokens

    def completion(self, prompt: str) -> int:
        return self._count(prompt)


def _text(content) -> str:
    """Message content is a string, or a list of parts in the OpenAI format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class RateLimiter:
    """Per user token buckets: one for requests, one for tokens, per minute."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm, self.tpm = rpm, tpm
        self._buckets: dict[str, list[float]] = {}  # user -> [requests, tokens, last_refill]
        self._lock = threading.Lock()

    def check(self, user: str, tokens: int, now: float | None = None) -> Rejection | None:
        now = time.monotonic() if now is None else now
        with self._lock:
            req, tok, last = self._buckets.get(user, [float(self.rpm), float(self.tpm), now])
            elapsed = now - last
            req = min(self.rpm, req + elapsed * self.rpm / 60)
            tok = min(self.tpm, tok + elapsed * self.tpm / 60)
            if req < 1 or tok < tokens:
                wait = max((1 - req) * 60 / self.rpm, (tokens - tok) * 60 / self.tpm, 0)
                self._buckets[user] = [req, tok, now]
                which = "requests" if req < 1 else "tokens"
                return Rejection(429, "rate_limited",
                                 f"user {user!r} is over its {which} per minute limit",
                                 retry_after_s=round(wait, 1))
            self._buckets[user] = [req - 1, tok - tokens, now]
        return None


@dataclass
class Decision:
    """Outcome of the harness checks for one request."""
    rejection: Rejection | None
    prompt_tokens: int | None = None
    body: dict | None = None  # possibly rewritten (default max_tokens)


class Harness:
    def __init__(self, settings: Settings, counter: TokenCounter | None = None):
        self.s = settings
        self.counter = counter or TokenCounter(settings.tokenizer_path)
        self.rate_limiter = RateLimiter(settings.rate_limit_rpm, settings.rate_limit_tpm)

    def check(self, path: str, body: dict | None, user: str) -> Decision:
        """Run every check in order; the first failure wins."""
        if not isinstance(body, dict):
            return Decision(Rejection(400, "invalid_request", "request body must be a JSON object"))

        if path.endswith("/chat/completions"):
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return Decision(Rejection(400, "invalid_request", "'messages' must be a non-empty list"))
            untrusted = " ".join(_text(m.get("content")) for m in messages if m.get("role") != "system")
            prompt_tokens = self.counter.chat(messages)
        elif path.endswith("/completions"):
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                return Decision(Rejection(400, "invalid_request", "'prompt' must be a non-empty string"))
            untrusted = prompt
            prompt_tokens = self.counter.completion(prompt)
        else:
            return Decision(None, body=body)

        if _INJECTION.search(untrusted):
            return Decision(Rejection(400, "prompt_injection",
                                      "request looks like an attempt to override the system instructions"),
                            prompt_tokens)

        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        if max_tokens is None:
            body = {**body, "max_tokens": self.s.max_output_tokens}
            max_tokens = self.s.max_output_tokens
        elif not isinstance(max_tokens, int) or max_tokens < 1:
            return Decision(Rejection(400, "invalid_request", "'max_tokens' must be a positive integer"), prompt_tokens)
        elif max_tokens > self.s.max_output_tokens:
            return Decision(Rejection(400, "max_tokens_too_large",
                                      f"max_tokens {max_tokens} is over the limit of {self.s.max_output_tokens}"),
                            prompt_tokens)

        if prompt_tokens > self.s.max_prompt_tokens:
            return Decision(Rejection(413, "prompt_too_long",
                                      f"prompt is {prompt_tokens} tokens, over the limit of {self.s.max_prompt_tokens}"),
                            prompt_tokens)
        if prompt_tokens + max_tokens > self.s.context_limit:
            return Decision(Rejection(413, "prompt_too_long",
                                      f"prompt ({prompt_tokens}) + max_tokens ({max_tokens}) exceeds the "
                                      f"{self.s.context_limit} token context"),
                            prompt_tokens)

        rejection = self.rate_limiter.check(user, prompt_tokens + max_tokens)
        return Decision(rejection, prompt_tokens, body)
