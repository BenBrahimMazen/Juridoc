"""
Production FastAPI service for the Tunisian Legal RAG.

Endpoints
---------
GET  /                 – service info (landing)
GET  /health           – readiness: Ollama, model, index, per-domain counts
GET  /domains          – the 17 legal domains + labels + indexed chunk counts
POST /query            – one-shot JSON answer (blocks until the full answer is ready)
POST /query/stream     – Server-Sent Events: emits meta → tokens → done (use this
                         from a UI; a full answer can take several minutes)
POST /ingest           – (re-)ingest a file / a domain / all

Design notes
------------
- Endpoints are plain ``def``; FastAPI runs them in its threadpool, so the
  synchronous Ollama call does not block the event loop.
- CORS is enabled (configurable via ``TLR_API_CORS``) so an Angular/SPA front-end
  on another origin can call the API directly.
- On startup the LLM is best-effort warmed into RAM (a background thread) so the
  first query is not hit by the multi-minute cold-load penalty.
- Auth / sessions / rate-limiting are intentionally out of scope — the Spring Boot
  layer is expected to own those concerns.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config
from src.api import schemas
from src.generation import answerer, ollama_client, prompts
from src.ingestion import pipeline
from src.retrieval import retriever
from src.routing import router as router_mod
from src.routing.domains import DOMAINS, LABELS
from src.routing.router import RouteDecision
from src.vectorstore import store

VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# App + middleware
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Best-effort: load the model into RAM so the first real query is warm.
    if config.API_WARM_ON_STARTUP and ollama_client.is_reachable():
        def _warm() -> None:
            try:
                ollama_client.chat(prompts.SYSTEM_PROMPT, "Prêt.")
            except Exception:
                pass  # warming is optional; never fail startup over it
        threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(
    title="Juridoc — Tunisian Legal RAG",
    description="Offline local-first RAG for Tunisian legal Q&A across 17 legal domains.",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _domain_scores(decision: RouteDecision) -> list[schemas.DomainScore]:
    return [
        schemas.DomainScore(code=code, label=LABELS.get(code, code), score=score)
        for code, score in decision.top_domains
    ]


def _resolve(question: str, req: schemas.QueryRequest):
    """Route (or apply a user override) and retrieve. Returns (boost, detected,
    reason, decision_or_None, chunks)."""
    if req.domain:
        return req.domain, req.domain, "user_override", None, retriever.retrieve(question, req.domain, top_k=req.top_k)
    decision = router_mod.route(question)
    chunks = retriever.retrieve(question, decision, top_k=req.top_k)
    return decision, decision.primary_domain, decision.reason, decision, chunks


def _sse(event: str | None, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return (f"event: {event}\n" if event else "") + f"data: {payload}\n\n"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict:
    return {
        "service": "Juridoc — Tunisian Legal RAG",
        "version": VERSION,
        "indexed_chunks": store.count(),
        "domains": len(DOMAINS),
        "llm_model": config.LLM_MODEL,
        "ollama_reachable": ollama_client.is_reachable(),
        "endpoints": ["/health", "/domains", "/query", "/query/stream", "/ingest", "/docs"],
    }


@app.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    reachable = ollama_client.is_reachable()
    model_ok = ollama_client.model_available() if reachable else False
    messages: list[str] = []
    status = "ok"
    if not reachable:
        status = "degraded"
        messages.append("Ollama server not reachable — start it (ollama serve / tray app).")
    elif not model_ok:
        status = "degraded"
        messages.append(f"LLM model '{config.LLM_MODEL}' not pulled. Run: ollama pull {config.LLM_MODEL}")
    indexed = store.count()
    if indexed == 0:
        status = "degraded"
        messages.append("Vector store is empty — run ingestion (POST /ingest or scripts/run_ingestion.py).")

    by_domain = {d: store.count(d) for d in DOMAINS if store.count(d) > 0}
    return schemas.HealthResponse(
        status=status,
        ollama_reachable=reachable,
        llm_model=config.LLM_MODEL,
        llm_model_available=model_ok,
        embedding_model=config.EMBEDDING_MODEL,
        collection=config.COLLECTION_NAME,
        indexed_chunks=indexed,
        by_domain=by_domain,
        messages=messages,
    )


@app.get("/domains", response_model=list[schemas.DomainInfo])
def domains() -> list[schemas.DomainInfo]:
    return [schemas.DomainInfo(code=d, label=LABELS[d], indexed_chunks=store.count(d)) for d in DOMAINS]


@app.post("/query", response_model=schemas.QueryResponse)
def query(req: schemas.QueryRequest) -> schemas.QueryResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    boost, detected, reason, decision, chunks = _resolve(question, req)
    started = time.perf_counter()
    llm_enabled = ollama_client.is_reachable()
    payload = answerer.generate_answer(question, chunks, llm_enabled=llm_enabled)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return schemas.QueryResponse(
        answer=payload["answer"],
        domain_detected=detected,
        domains=_domain_scores(decision) if decision else [
            schemas.DomainScore(code=detected, label=LABELS.get(detected, detected), score=1.0)
        ],
        confidence=decision.confidence if decision else None,
        citations=[schemas.Citation(**c) for c in payload["citations"]],
        confidence_note=payload.get("confidence_note"),
        route_reason=reason,
        n_chunks_retrieved=len(chunks),
        model=config.LLM_MODEL,
        elapsed_ms=elapsed_ms,
    )


@app.post("/query/stream")
def query_stream(req: schemas.QueryRequest) -> StreamingResponse:
    """Server-Sent Events stream of an answer.

    Event sequence: ``meta`` (routing + retrieved sources) → many ``token``
    (answer deltas, render live) → ``done`` (citations + confidence note).
    ``error`` is emitted instead of ``done`` on failure. Comment lines
    (``: keep-alive``) keep the connection alive during the long prompt-evaluation
    that precedes the first token.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    boost, detected, reason, decision, chunks = _resolve(question, req)
    llm_enabled = ollama_client.is_reachable()

    def gen():
        sources = [
            {
                "code_name": c["metadata"].get("code_name", ""),
                "article_number": c["metadata"].get("article_number", ""),
                "domain": c["metadata"].get("domain", ""),
                "similarity": round(c.get("similarity", 0.0), 4),
            }
            for c in chunks
        ]
        yield _sse("meta", {
            "domain_detected": detected,
            "reason": reason,
            "confidence": decision.confidence if decision else None,
            "domains": [{"code": c, "label": LABELS.get(c, c), "score": s}
                        for c, s in (decision.top_domains if decision else [(detected, 1.0)])],
            "sources": sources,
            "n_chunks": len(chunks),
            "model": config.LLM_MODEL,
            "llm_enabled": llm_enabled,
        })

        # No grounding -> honest canned message, no LLM call.
        if not chunks:
            ans = ("Les documents indexés ne contiennent pas d'élément suffisant "
                   "pour répondre à cette question.")
            yield _sse("token", ans)
            yield _sse("done", {"answer": ans, "citations": [],
                                "confidence_note": prompts.INSUFFICIENT_NOTE})
            return

        if not llm_enabled:
            ans = answerer._fallback_summary(question, chunks)
            yield _sse("token", ans)
            fin = answerer.finalize(chunks, ans, llm_ok=False)
            yield _sse("done", {"answer": ans, **fin})
            return

        # Stream the LLM in a worker thread so we can emit keep-alive comments
        # while it is busy with model-load + prompt-evaluation (no tokens yet).
        context = answerer.build_context(chunks)
        user = prompts.USER_HEADER.format(question=question, context=context)
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                for delta in ollama_client.chat_stream(prompts.SYSTEM_PROMPT, user):
                    q.put(("t", delta))
            except Exception as e:  # noqa: BLE001
                q.put(("e", f"{type(e).__name__}: {e}"))
            q.put(("x", None))

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        parts: list[str] = []
        while True:
            try:
                kind, val = q.get(timeout=10)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if kind == "t":
                parts.append(val)
                yield _sse("token", val)
            elif kind == "e":
                yield _sse("error", {"message": val})
                break
            else:  # done
                answer = "".join(parts).strip()
                fin = answerer.finalize(chunks, answer, llm_ok=True)
                yield _sse("done", {"answer": answer, **fin})
                break
        t.join(timeout=1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/ingest", response_model=schemas.IngestResult)
def ingest(req: schemas.IngestRequest = schemas.IngestRequest()) -> schemas.IngestResult:
    try:
        if req.file:
            target = None
            for d in config.source_dirs():
                cand = d / req.file
                if cand.exists():
                    target = cand
                    break
            if not target:
                raise HTTPException(status_code=404, detail=f"file not found: {req.file}")
            results = [pipeline.ingest_file(target)]
        elif req.domain:
            results = pipeline.ingest_domain(req.domain)
        else:
            results = pipeline.ingest_all()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return schemas.IngestResult(status="ok", results=results, summary=pipeline.summary(results))
