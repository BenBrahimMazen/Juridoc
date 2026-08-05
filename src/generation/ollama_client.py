"""
Minimal Ollama HTTP client (httpx). Sync calls — FastAPI runs them in a
threadpool via `def` endpoints, so no async ceremony is needed.
"""
from __future__ import annotations

import json

import httpx

import config


def _base() -> str:
    return config.OLLAMA_HOST.rstrip("/")


def is_reachable(timeout: float = 3.0) -> bool:
    try:
        with httpx.Client(timeout=timeout) as c:
            c.get(f"{_base()}/api/tags")
        return True
    except Exception:
        return False


def list_models(timeout: float = 5.0) -> list[str]:
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{_base()}/api/tags")
            r.raise_for_status()
            data = r.json()
        return [m.get("name", "") or m.get("model", "") for m in data.get("models", [])]
    except Exception:
        return []


def model_available(model: str | None = None, timeout: float = 5.0) -> bool:
    model = model or config.LLM_MODEL
    avail = list_models(timeout=timeout)
    # Ollama tags can appear as "qwen2.5:7b" or "qwen2.5:7b-instruct-q4_K_M".
    # Match on the model family prefix to be lenient.
    fam = model.split(":")[0].lower()
    return any(m.lower().startswith(fam) for m in avail)


def chat_with_stats(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> tuple[str, dict]:
    """Single chat completion, streamed internally.

    Returns ``(content, stats)`` where ``stats`` holds Ollama's response counters:
    ``prompt_eval_count`` / ``eval_count`` (tokens) and the four nanosecond
    durations (``prompt_eval_duration`` etc.). Used by the diagnostic script to
    tell prompt-eval cost from generation cost.

    We stream the response even though the public API returns the full string at
    once. On the CPU-only target box a full answer can take several minutes to
    generate; with a non-streaming POST the server sends nothing until it is done,
    so any client read-timeout shorter than the full generation time kills the
    call. Streaming keeps the connection alive token-by-token — httpx's read
    timeout then only needs to exceed the gap between tokens (~0.3 s), not the
    whole generation. The final NDJSON line still carries the aggregate stats.
    """
    model = model or config.LLM_MODEL
    payload = _chat_payload(system, user, model, temperature)
    parts: list[str] = []
    final: dict = {}
    with httpx.Client(timeout=config.LLM_TIMEOUT_S) as c:
        with c.stream("POST", f"{_base()}/api/chat", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message") or {}
                if msg.get("content"):
                    parts.append(msg["content"])
                if data.get("done"):
                    final = data
    content = "".join(parts).strip()
    stats = {
        "model": model,
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_duration_ns": final.get("prompt_eval_duration"),
        "eval_count": final.get("eval_count"),
        "eval_duration_ns": final.get("eval_duration"),
        "total_duration_ns": final.get("total_duration"),
        "load_duration_ns": final.get("load_duration"),
    }
    return content, stats


def chat(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Single chat completion. Returns the assistant message text (streamed internally)."""
    content, _ = chat_with_stats(
        system, user, model=model, temperature=temperature
    )
    return content


def _chat_payload(system: str, user: str, model: str, temperature: float | None) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        # "think" is left false (llama3.1 has no thinking mode and 400s if true).
        "think": config.LLM_THINK,
        "options": {
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
            "num_thread": config.LLM_NUM_THREAD,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "keep_alive": config.LLM_KEEP_ALIVE,
        },
    }


def chat_stream(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
):
    """Generator yielding assistant content deltas as Ollama streams them.

    Same payload/options as :func:`chat_with_stats`, but yields each ``content``
    delta so a streaming endpoint (e.g. Server-Sent Events) can render a
    multi-minute answer token-by-token instead of blocking. Streaming also keeps
    the HTTP read-timeout from biting during generation (the timeout only needs to
    cover load + prompt-eval, not the whole answer).
    """
    model = model or config.LLM_MODEL
    payload = _chat_payload(system, user, model, temperature)
    with httpx.Client(timeout=config.LLM_TIMEOUT_S) as c:
        with c.stream("POST", f"{_base()}/api/chat", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = (data.get("message") or {}).get("content")
                if delta:
                    yield delta
