"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question, in French")
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain override (one of the 17 domain codes); if omitted, the router decides.",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=30)


class Citation(BaseModel):
    code_name: str
    article_number: str
    source_file: str = ""
    law_reference: str = ""
    excerpt: str
    similarity: float = 0.0


class DomainScore(BaseModel):
    code: str
    label: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    domain_detected: str
    domains: list[DomainScore] = Field(
        default_factory=list,
        description="Router's ranked top domains (code, label, score).",
    )
    confidence: Optional[float] = Field(
        default=None, description="Router confidence (top1/(top1+top2)); lower = more ambiguous."
    )
    citations: list[Citation]
    confidence_note: Optional[str] = None
    route_reason: str = ""
    n_chunks_retrieved: int = 0
    model: str = ""
    elapsed_ms: int = 0


class DomainInfo(BaseModel):
    code: str
    label: str
    indexed_chunks: int


class IngestRequest(BaseModel):
    domain: Optional[str] = Field(
        default=None,
        description="Optional: ingest one 17-domain bucket (e.g. droit_penal).",
    )
    file: Optional[str] = Field(
        default=None,
        description="Optional: ingest a single source_file name only.",
    )


class IngestResult(BaseModel):
    status: str = "ok"
    results: list[dict] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    ollama_reachable: bool
    llm_model: str
    llm_model_available: bool
    embedding_model: str
    collection: str
    indexed_chunks: int
    by_domain: dict[str, int]
    messages: list[str] = Field(default_factory=list)
