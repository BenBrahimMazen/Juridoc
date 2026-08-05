"""
Central configuration for the Tunisian Legal RAG service.

Everything is overridable via environment variables so the same code runs on
another machine without edits. Defaults are tuned for the target box:
Windows, CPU-only, ~17-20 GB RAM, fully offline at runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent

# Source PDF roots. Domain is now assigned PER-FILE via src/ingestion/file_meta.py
# (not per-folder), so here we just enumerate every folder that may hold source
# PDFs. Any that exist are scanned; drop new PDFs into pdfs/new.
_SOURCE_DIR_CANDIDATES: list[Path] = [
    BASE_DIR / "pdfs" / "Finance & Banking",
    BASE_DIR / "pdfs" / "General legal",
    BASE_DIR / "pdfs" / "new",
    BASE_DIR / "data" / "raw" / "finance_banking",
    BASE_DIR / "data" / "raw" / "general_legal",
]


def source_dirs() -> list[Path]:
    """Existing directories that may hold source PDFs (scanned regardless of domain)."""
    return [p for p in _SOURCE_DIR_CANDIDATES if p.exists()]


# Back-compat: old 2-domain folder map (a couple of helpers still reference it).
_RAW_DIRS: dict[str, list[Path]] = {
    "finance_banking": [
        BASE_DIR / "pdfs" / "Finance & Banking",
        BASE_DIR / "data" / "raw" / "finance_banking",
    ],
    "general_legal": [
        BASE_DIR / "pdfs" / "General legal",
        BASE_DIR / "data" / "raw" / "general_legal",
    ],
}


def raw_dirs(domain: str) -> list[Path]:
    """Existing source directories for a given (legacy) domain."""
    return [p for p in _RAW_DIRS.get(domain, []) if p.exists()]


def all_raw_dirs() -> dict[str, list[Path]]:
    return {d: raw_dirs(d) for d in _RAW_DIRS}


# ChromaDB persistent storage (file-based, no server process).
CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma"

# Optional: dump cleaned intermediate text per PDF for debugging the parser.
PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
# One collection with a `domain` metadata field + filtered queries.
# Simpler than two collections and makes the "ambiguous -> search both" case
# trivial (no filter). See README for rationale.
COLLECTION_NAME: str = os.environ.get("TLR_COLLECTION", "tunisian_legal")
DISTANCE_METRIC: str = "cosine"  # e5 embeddings are normalized -> cosine is correct

# --------------------------------------------------------------------------- #
# Embedding model (local, multilingual/French-capable)
# --------------------------------------------------------------------------- #
# intfloat/multilingual-e5-base: strong French, ~1.1 GB, fast enough on CPU,
# 768-dim. Upgrade path for better retrieval quality: "intfloat/multilingual-e5-large".
# IMPORTANT: e5 requires an "E5-style" prefix ("query: " / "passage: ") on inputs.
EMBEDDING_MODEL: str = os.environ.get("TLR_EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE: int = int(os.environ.get("TLR_EMBED_BATCH", "32"))

# --------------------------------------------------------------------------- #
# LLM (Ollama, local)
# --------------------------------------------------------------------------- #
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Default LLM: llama3.1 (8B, Q4_K_M, ~4.9 GB). Chosen over qwen3 on this box after
# empirical testing: on this Ollama build qwen3's "thinking" mode CANNOT be turned
# off — `think:false` dumps an English chain-of-thought into the answer content and
# the `/no_think` switch is ignored — so qwen3 either pollutes the answer or (with
# think=true) burns a huge token budget on a reasoning trace before answering. llama3.1
# has no thinking mode: it answers directly in clean formal French with a tiny token
# budget and cites sources, which is exactly what this grounded persona needs.
# Override with TLR_LLM_MODEL (e.g. qwen3:4b) if you want to re-evaluate.
LLM_MODEL: str = os.environ.get("TLR_LLM_MODEL", "llama3.1:latest")
# think=false is required for llama3.1 (it 400s on think=true). Leave 0 unless using
# a model that needs reasoning routed to a separate field.
LLM_THINK: bool = os.environ.get("TLR_LLM_THINK", "0") == "1"
LLM_TEMPERATURE: float = float(os.environ.get("TLR_LLM_TEMP", "0.1"))
# Context window. A full prompt is ~1,100 tokens (persona ~505 + tiered retrieved
# context ~500 + question); 4096 gives comfortable headroom against verbose articles
# while keeping the KV-cache RAM footprint modest.
LLM_NUM_CTX: int = int(os.environ.get("TLR_LLM_CTX", "4096"))
# Generous: the target box can take minutes for a cold model load + large prompt
# eval before the first token streams. Streaming keeps this from biting during
# generation; this only needs to cover load + prompt-eval.
LLM_TIMEOUT_S: float = float(os.environ.get("TLR_LLM_TIMEOUT", "600"))
LLM_NUM_PREDICT: int = int(os.environ.get("TLR_LLM_MAX_TOKENS", "400"))
# Keep the model resident in RAM this long after each call. The target box pays a
# large cold-load penalty (minutes for an 8B model read from disk) before the first
# token; on a machine that also sleeps and evicts models, a long keep_alive lets a
# warm-up call keep the model hot for the real query. Set 0 to unload immediately.
LLM_KEEP_ALIVE: str = os.environ.get("TLR_LLM_KEEP_ALIVE", "30m")
# Threads for Ollama inference. On this 4-physical / 8-logical-core box, 8 threads
# outperform 4 for prompt-evaluation (the latency bottleneck): prompt-eval runs at
# ~9 tok/s with 8 vs ~6 tok/s with 4, because prompt-eval is a batched matmul that
# benefits from all logical cores. Generation (~2–3 tok/s) is memory-bound and
# largely thread-insensitive. Override via env if your CPU differs.
LLM_NUM_THREAD: int = int(os.environ.get("TLR_LLM_NUM_THREAD", "8"))

# --------------------------------------------------------------------------- #
# Retrieval / routing
# --------------------------------------------------------------------------- #
TOP_K: int = int(os.environ.get("TLR_TOP_K", "3"))
# Hybrid retrieval: over-fetch POOL_FACTOR*top_k candidates, then re-rank by
# ALPHA*embedding_sim + (1-ALPHA)*keyword_overlap. ALPHA=0.6 keeps embedding
# dominant while letting exact-term matches (e.g. COC "validité du contrat")
# fix topical near-misses.
HYBRID_ALPHA: float = float(os.environ.get("TLR_HYBRID_ALPHA", "0.6"))
HYBRID_POOL_FACTOR: int = int(os.environ.get("TLR_POOL_FACTOR", "3"))
# Multi-domain SOFT-BOOST retrieval (no hard domain exclusion): retrieval searches
# the whole corpus, then chunks whose primary/secondary domain is among the
# router's top-predicted domains get a score bump added to the hybrid score.
# top-1 domain -> DOMAIN_BOOST, top-2 -> DOMAIN_BOOST_SECOND. Tiered so the
# strongest prediction dominates without excluding anything.
DOMAIN_BOOST: float = float(os.environ.get("TLR_DOMAIN_BOOST", "0.10"))
DOMAIN_BOOST_SECOND: float = float(os.environ.get("TLR_DOMAIN_BOOST_SECOND", "0.05"))
# Router: how many ranked (domain, score) pairs to return; confidence = top1/(top1+top2).
ROUTER_TOP_N: int = int(os.environ.get("TLR_ROUTER_TOP_N", "5"))
# Router keyword-vs-embedding blend for per-domain scoring (0=embedding only, 1=keyword only).
ROUTER_KW_WEIGHT: float = float(os.environ.get("TLR_ROUTER_KW_WEIGHT", "0.5"))
# Router marks "low confidence" when the top1 margin is below this (top1/(top1+top2)).
ROUTER_LOW_CONF_THRESHOLD: float = float(os.environ.get("TLR_ROUTER_LOW_CONF", "0.55"))
# Legacy binary ambiguity band (unused by the new 17-domain router; kept for back-compat).
AMBIGUOUS_THRESHOLD: float = float(os.environ.get("TLR_AMBIG_THRESH", "0.55"))

# Chunking guardrails (per-article; only used when a single article is very long).
SUBCHUNK_MAX_CHARS: int = int(os.environ.get("TLR_SUBCHUNK_MAX", "1400"))
SUBCHUNK_MIN_CHARS: int = int(os.environ.get("TLR_SUBCHUNK_MIN", "200"))

# OCR fallback for scanned/image-only PDFs (rapidocr-onnxruntime). Only triggers
# on pages with no text layer. Set TLR_OCR=0 to disable.
OCR_ENABLED: bool = os.environ.get("TLR_OCR", "1") == "1"

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
API_HOST: str = os.environ.get("TLR_API_HOST", "127.0.0.1")
API_PORT: int = int(os.environ.get("TLR_API_PORT", "8000"))
# CORS: origins allowed to call the API (e.g. an Angular dev server). Comma-list.
# "*" allows all (fine for a local/private deployment; restrict in production).
API_CORS_ORIGINS: list[str] = [
    o.strip() for o in os.environ.get("TLR_API_CORS", "*").split(",") if o.strip()
]
# Best-effort: warm (load) the LLM into RAM on API startup so the first query is
# not hit by the multi-minute cold-load penalty. Set 0 to disable.
API_WARM_ON_STARTUP: bool = os.environ.get("TLR_API_WARM", "1") == "1"
