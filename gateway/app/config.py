"""Gateway settings, read once from the environment."""

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    upstream_url: str = os.getenv("UPSTREAM_URL", "http://nginx:80")

    # Harnesses: checks that run before a request is allowed to reach the GPU.
    harnesses_enabled: bool = _bool("HARNESSES_ENABLED", True)
    # Prompt tokens a single request may send (the model's context is 8,192).
    max_prompt_tokens: int = int(os.getenv("MAX_PROMPT_TOKENS", "6000"))
    # Output tokens a single request may ask for. Also the default when a
    # request doesn't say, so nothing can generate until the context is full.
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
    context_limit: int = int(os.getenv("CONTEXT_LIMIT", "8192"))
    # Per user (x-user-id header) request and token budgets, per minute.
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "60"))
    rate_limit_tpm: int = int(os.getenv("RATE_LIMIT_TPM", "200000"))
    # Requests the gateway forwards at once before shedding load with 503.
    max_in_flight: int = int(os.getenv("MAX_IN_FLIGHT", "256"))
    # Per request time limit; clients can ask for less with x-timeout-s.
    default_timeout_s: float = float(os.getenv("DEFAULT_TIMEOUT_S", "120"))
    max_timeout_s: float = float(os.getenv("MAX_TIMEOUT_S", "600"))

    # Tokenizer used to count prompt tokens before forwarding. "none" falls
    # back to an estimate (used in tests).
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "/srv/tokenizer.json")
