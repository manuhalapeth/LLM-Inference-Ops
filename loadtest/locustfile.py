"""Locust load test: more and more concurrent users until one GPU breaks.

Closed loop: every simulated user sends its next request as soon as the last
one finishes, so the number of users is the number of requests in flight.
The user count goes up in steps (LOAD_STEPS, each held STEP_SECONDS).

Every request streams, so the client measures time to first token, time
between tokens and total time. Each request is also written to
$RESULTS_DIR/requests_<pid>.jsonl for analysis afterwards
(loadtest/analyze_load.py).

Runs in the official Locust image on the compose network:

    docker compose --profile loadtest run --rm -p 127.0.0.1:8089:8089 locust
"""

import json
import os
import random
import time
import uuid
from pathlib import Path

import requests
from locust import LoadTestShape, User, constant, events, task
from locust.runners import WorkerRunner

HOST = os.getenv("GATEWAY_URL", "http://gateway:8080")
MODEL = os.getenv("SERVED_MODEL_NAME", "llm")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/results"))
STEPS = [int(x) for x in os.getenv("LOAD_STEPS", "1,2,4,8,16,32,64,96,128,192,256,384,512").split(",")]
STEP_SECONDS = int(os.getenv("STEP_SECONDS", "60"))
# "mix": independent requests (Phases 4 and 5). "sessions": every user holds a
# conversation about a long document, so each request repeats that user's
# growing prefix (what sticky routing and a shared KV store are for).
LOAD_MODE = os.getenv("LOAD_MODE", "mix")
SESSION_TURNS = int(os.getenv("SESSION_TURNS", "6"))
FOLLOW_UPS = [
    "What was the root cause, in one sentence?",
    "List the follow-up actions as short bullet points.",
    "Which team should own each follow-up action?",
    "What metric would have caught this earliest?",
    "Write a two sentence status update for customers.",
    "What is the single most important lesson here?",
]

# A chat product's mix: mostly short exchanges, some long documents, some long answers.
CATEGORY_WEIGHTS = {"short_qa": 0.35, "multi_turn": 0.25, "summarization": 0.25, "long_generation": 0.15}

PROMPTS = json.loads((Path(__file__).parent / "prompts.json").read_text())
BY_CATEGORY = {c: [p for p in PROMPTS if p["category"] == c] for c in CATEGORY_WEIGHTS}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_log = None


def _record(row: dict) -> None:
    global _log
    if _log is None:  # one file per process (Locust forks with --processes)
        _log = open(RESULTS_DIR / f"requests_{os.getpid()}.jsonl", "a", buffering=1)
    _log.write(json.dumps(row) + "\n")


@events.test_start.add_listener
def _on_start(environment, **_):
    if not isinstance(environment.runner, WorkerRunner):  # only the master writes the run's timing
        (RESULTS_DIR / "run_meta.json").write_text(json.dumps(
            {"start": time.time(), "steps": STEPS, "step_seconds": STEP_SECONDS,
             "category_weights": CATEGORY_WEIGHTS}) + "\n")


def unique_messages(prompt: dict) -> list[dict]:
    """The prompt's messages with a unique tag at the start of the first user turn.

    Without it, 24 prompts sent thousands of times would be served from the
    prefix cache, hiding the real load. System prompts stay identical, as they
    would in a real application, so they are still shared.
    """
    messages = [dict(m) for m in prompt["messages"]]
    first_user = next(m for m in messages if m["role"] == "user")
    first_user["content"] = f"(request {uuid.uuid4().hex[:12]}) {first_user['content']}"
    return messages


class ChatUser(User):
    wait_time = constant(0)

    def on_start(self):
        self.session = requests.Session()
        self.user_id = f"locust-{uuid.uuid4().hex[:8]}"
        self.history, self.turn = [], 0

    def next_request(self) -> tuple[str, str, list[dict], int]:
        """category, prompt id, messages, max_tokens for this user's next request."""
        if LOAD_MODE != "sessions":
            category = random.choices(list(CATEGORY_WEIGHTS), weights=list(CATEGORY_WEIGHTS.values()))[0]
            prompt = random.choice(BY_CATEGORY[category])
            return category, prompt["id"], unique_messages(prompt), prompt["max_tokens"]
        if not self.history or self.turn >= SESSION_TURNS:
            prompt = random.choice(BY_CATEGORY["summarization"])
            self.history, self.turn = unique_messages(prompt), 0  # a new conversation, unique to this user
            return "session_open", prompt["id"], self.history, 256
        self.history = self.history + [{"role": "user", "content": FOLLOW_UPS[(self.turn - 1) % len(FOLLOW_UPS)]}]
        return "session_follow_up", f"turn_{self.turn}", self.history, 128

    @task
    def chat(self):
        category, prompt_id, messages, max_tokens = self.next_request()
        body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        row = {"t_start": time.time(), "category": category, "prompt_id": prompt_id}
        answer = []
        start = time.perf_counter()
        first_token, usage, error, status = None, {}, None, None
        try:
            with self.session.post(f"{HOST}/v1/chat/completions", json=body, stream=True, timeout=600,
                                   headers={"x-user-id": self.user_id}) as resp:
                status = resp.status_code
                if status != 200:
                    error = resp.text[:200]
                else:
                    for line in resp.iter_lines():
                        if not line.startswith(b"data: ") or line == b"data: [DONE]":
                            continue
                        event = json.loads(line[6:])
                        if "error" in event:
                            error = str(event["error"])[:200]
                        if event.get("usage"):
                            usage = event["usage"]
                        for c in event.get("choices", []):
                            if c.get("delta", {}).get("content"):
                                answer.append(c["delta"]["content"])
                                if first_token is None:
                                    first_token = time.perf_counter()
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
        end = time.perf_counter()

        out = usage.get("completion_tokens", 0)
        row.update({
            "status": status, "error": error,
            "ttft_s": first_token - start if first_token else None,
            "e2e_s": end - start,
            "prompt_tokens": usage.get("prompt_tokens"), "output_tokens": out,
            "itl_s": (end - first_token) / (out - 1) if first_token and out > 1 else None,
        })
        _record(row)
        if LOAD_MODE == "sessions":
            if error is None and status == 200 and answer:
                self.history = self.history + [{"role": "assistant", "content": "".join(answer)}]
                self.turn += 1
            else:
                self.history = []  # start over after a failed turn

        # Also report to Locust's own stats, so its web UI shows TTFT and total time live.
        ok = error is None and status == 200
        events.request.fire(request_type="TOTAL", name=category, response_time=row["e2e_s"] * 1000,
                            response_length=out, exception=None if ok else Exception(error or status), context={})
        if row["ttft_s"] is not None:
            events.request.fire(request_type="TTFT", name=category, response_time=row["ttft_s"] * 1000,
                                response_length=0, exception=None, context={})


class Steps(LoadTestShape):
    """Hold each user count in STEPS for STEP_SECONDS, then stop."""

    def tick(self):
        step = int(self.get_run_time() // STEP_SECONDS)
        if step >= len(STEPS):
            return None
        return STEPS[step], max(STEPS[step], 1)  # reach each step almost at once
