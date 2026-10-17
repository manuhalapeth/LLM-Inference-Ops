# Stand-in for a vLLM OpenAI server: same uvicorn idle keep-alive as vLLM
# (VLLM_HTTP_TIMEOUT_KEEP_ALIVE, default 5 s), streams tokens as SSE.
import asyncio, json, os, socket, time
import uvicorn
from fastapi import FastAPI

app = FastAPI()
NAME = socket.gethostname()
# BUSY_MS > 0 blocks the event loop in slices of that many ms, like an API server that is
# busy detokenizing and streaming hundreds of responses. 0 (the default) means idle.
BUSY_MS = float(os.environ.get("BUSY_MS", "0"))


@app.on_event("startup")
async def start_busy_loop():
    if BUSY_MS > 0:
        async def busy():
            while True:
                time.sleep(BUSY_MS / 1000)
                await asyncio.sleep(0.001)
        asyncio.get_running_loop().create_task(busy())

@app.get("/health")
async def health():
    return {"server": NAME}

@app.post("/v1/chat/completions")
async def chat(body: dict):
    from fastapi.responses import StreamingResponse
    n = int(body.get("max_tokens", 16))
    delay = float(body.get("token_delay", 0.02))
    if body.get("stream"):
        # Same chunk shape as vLLM's /v1/chat/completions stream (about 270 bytes each).
        async def gen():
            created = int(time.time())
            for i in range(n):
                await asyncio.sleep(delay)
                chunk = {"id": "chatcmpl-8f3c2a1b9d7e4f60a1b2c3d4e5f60718", "object": "chat.completion.chunk",
                         "created": created, "model": "Qwen/Qwen2.5-7B-Instruct",
                         "choices": [{"index": 0, "delta": {"content": " tok"}, "logprobs": None,
                                      "finish_reason": "length" if i == n - 1 else None}],
                         "server": NAME}
                yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    await asyncio.sleep(delay * n)
    return {"server": NAME, "choices": [{"message": {"content": "tok " * n}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning",
                timeout_keep_alive=int(os.environ.get("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5")))
