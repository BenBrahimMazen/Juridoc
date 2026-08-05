"""
ChromaDB access layer.

Design choice (documented in README): ONE collection with a `domain` metadata
field, queried with `where={"domain": ...}` filters. This is simpler than two
collections and makes the "ambiguous -> search both domains" case a plain query
with no filter. Retrieval quality is identical (HNSW over the same vectors).

Ingestion idempotency: before upserting a source PDF we delete every existing
chunk whose `source_file` matches, then insert fresh. Re-running ingestion never
duplicates entries, and re-ingesting an updated PDF cleanly replaces its chunks.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import threading
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings

import config
from src.embeddings.embedder import E5EmbeddingFunction

_client = None
_collection = None
_ef = None
_lock = threading.Lock()


def _now_iso() -> str:
    return _dt.date.today().isoformat()


def _get_ef() -> E5EmbeddingFunction:
    global _ef
    if _ef is None:
        _ef = E5EmbeddingFunction()
    return _ef


def get_client() -> chromadb.api.ClientAPI:
    global _client
    if _client is None:
        config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


def get_collection() -> chromadb.api.Collection:
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                _collection = get_client().get_or_create_collection(
                    name=config.COLLECTION_NAME,
                    embedding_function=_get_ef(),
                    metadata={"hnsw:space": config.DISTANCE_METRIC},
                )
    return _collection


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def _make_id(source_file: str, article_number: str, chunk_index: int,
             text: str) -> str:
    h = hashlib.md5(text[:200].encode("utf-8")).hexdigest()[:8]
    safe_art = (article_number or "preamble").replace(" ", "_")
    return f"{source_file}::{safe_art}::{chunk_index:04d}::{h}"


def replace_source_chunks(
    source_file: str,
    chunks: Iterable[dict[str, Any]],
) -> int:
    """Delete all existing chunks for `source_file`, then upsert new ones.

    Each chunk dict: {text, code_name, article_number, domain, law_reference?}.
    Returns the number of chunks written.
    """
    coll = get_collection()
    # Clean slate for this file (handles changed structure / removed articles).
    try:
        coll.delete(where={"source_file": source_file})
    except Exception:
        # Collection may be empty; delete with a where-clause on an empty
        # collection can raise in some Chroma versions — ignore.
        pass

    items = list(chunks)
    if not items:
        return 0

    ids, texts, metas = [], [], []
    today = _now_iso()
    for i, c in enumerate(items):
        ids.append(_make_id(source_file, c["article_number"], i, c["text"]))
        texts.append(c["text"])
        metas.append({
            "code_name": c["code_name"],
            "article_number": c["article_number"],
            "domain": c["domain"],
            "domain_secondary": c.get("domain_secondary") or "",
            "source_file": source_file,
            "law_reference": c.get("law_reference") or "",
            "ingestion_date": today,
        })
    # Chroma upserts in batches internally; pass everything at once.
    coll.upsert(ids=ids, documents=texts, metadatas=metas)
    return len(ids)


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def query(
    query_text: str,
    domain: str | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Semantic search. `domain` None or "ambiguous" -> search both domains."""
    top_k = top_k or config.TOP_K
    where = None
    if domain and domain != "ambiguous":
        where = {"domain": domain}

    coll = get_collection()
    res = coll.query(
        query_embeddings=[_get_ef().embed_query(query_text)],
        n_results=top_k,
        where=where,
    )
    out: list[dict[str, Any]] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "text": doc,
            "metadata": meta,
            "distance": float(dist),
            # cosine distance -> similarity in [0,1] (e5 vectors are normalized)
            "similarity": max(0.0, 1.0 - float(dist) / 2.0),
        })
    return out


def count(domain: str | None = None) -> int:
    coll = get_collection()
    if not domain:
        return coll.count()
    # chromadb's Collection.count() ignores the `where` filter (1.5.x), so count
    # via a filtered get() (ids only) instead.
    try:
        res = coll.get(where={"domain": domain}, include=[])
        return len((res or {}).get("ids", []) or [])
    except Exception:
        return coll.count()


def list_source_files() -> list[str]:
    """Distinct source_file values currently indexed."""
    coll = get_collection()
    try:
        res = coll.get(include=["metadatas"])
        files = sorted({m.get("source_file", "") for m in (res.get("metadatas") or [])})
        return [f for f in files if f]
    except Exception:
        return []
