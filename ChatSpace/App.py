"""FastAPI chat server for the minillm SFT model (Docker Space, free CPU tier).

Serves the JS chat page at `/` and streams generated text from `POST /api/chat` as
server-sent events. Run locally:
    MODEL_REPO=artifacts/hf_model uvicorn App:app --app-dir ChatSpace --port 7860
"""

import json
import os
import threading
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_REPO = os.environ.get("MODEL_REPO", "your-username/minillm-125m-chat")
MAX_CONTEXT = 1024
STATIC_DIR = Path(__file__).resolve().parent / "Static"

torch.set_num_threads(max(1, os.cpu_count() or 1))
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, dtype=torch.float32)
model.eval()
STOP_IDS = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

# The free tier has 2 vCPUs; one generation at a time keeps latency predictable.
generation_lock = threading.Lock()

app = FastAPI(title="minillm chat")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    system_prompt: str = Field(default="", max_length=2000)
    max_new_tokens: int = Field(default=256, ge=16, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=1.5)


class CancelOnDisconnect(StoppingCriteria):
    """Stops generation when the client goes away, so the CPU is freed for the next user."""

    def __init__(self, cancelled: threading.Event):
        self.cancelled = cancelled

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancelled.is_set()


def build_input_ids(request: ChatRequest) -> torch.Tensor:
    turns = [{"role": message.role, "content": message.content} for message in request.messages]
    system = [{"role": "system", "content": request.system_prompt}] if request.system_prompt.strip() else []
    budget = MAX_CONTEXT - request.max_new_tokens
    # Drop the oldest exchanges until the prompt fits in the 1024-token window.
    while True:
        prompt = tokenizer.apply_chat_template(system + turns, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        if input_ids.shape[1] <= budget or len(turns) == 1:
            return input_ids[:, -budget:]
        turns = turns[2:] if len(turns) > 2 else turns[-1:]


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def stream_reply(request: ChatRequest):
    input_ids = build_input_ids(request)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
    cancelled = threading.Event()
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        streamer=streamer,
        max_new_tokens=request.max_new_tokens,
        do_sample=request.temperature > 0,
        temperature=max(request.temperature, 1e-5),
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        eos_token_id=STOP_IDS,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([CancelOnDisconnect(cancelled)]),
    )

    def run_generation() -> None:
        with generation_lock:
            model.generate(**generation_kwargs)

    worker = threading.Thread(target=run_generation, daemon=True)
    worker.start()
    try:
        yield sse("start", {"queued": generation_lock.locked()})
        for text in streamer:
            if text:
                yield sse("token", {"text": text})
        yield sse("done", {})
    except Exception as error:  # e.g. streamer timeout while queued behind another user
        yield sse("error", {"message": str(error)})
    finally:
        # Runs when the generator is closed, including on client disconnect.
        cancelled.set()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "ChatPage.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_REPO, "busy": generation_lock.locked()}


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="The last message must come from the user")
    # A sync generator is iterated in Starlette's threadpool, so the event loop never blocks.
    return StreamingResponse(
        stream_reply(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
