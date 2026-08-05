"""
Embedding model wrapper around sentence-transformers.

Uses intfloat/multilingual-e5-base by default. E5 models REQUIRE an input
prefix ("query: " for search queries, "passage: " for documents); forgetting it
silently degrades retrieval, so this wrapper applies the prefixes for you.

The model is lazily loaded once (singleton) — loading ~1 GB on first use is slow
and we never want two copies in memory. ChromaDB is given an instance of
:class:`E5EmbeddingFunction` so embeddings are computed consistently at ingest
and query time.
"""
from __future__ import annotations

import threading
from typing import Sequence

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

import config

_lock = threading.Lock()
_model = None  # cached SentenceTransformer


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(
                    config.EMBEDDING_MODEL,
                    device="cpu",
                )
    return _model


class E5EmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function using E5 query/passage prefixes."""

    def __init__(self) -> None:
        self._is_query = True  # default to query mode; override via embed_documents

    # --- ChromaDB hooks ---
    def __call__(self, input: Documents) -> Embeddings:
        # Chroma calls this for stored documents -> passage mode.
        return self.embed_documents(list(input))

    def name(self) -> str:  # pragma: no cover
        return "multilingual-e5-base"

    # --- direct API used by retriever/router ---
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = _get_model()
        prefixed = [
            (f"passage: {t}" if not t.startswith(("query:", "passage:")) else t)
            for t in texts
        ]
        vecs = model.encode(
            prefixed,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        model = _get_model()
        prefixed = text if text.startswith(("query:", "passage:")) else f"query: {text}"
        vec = model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vec.tolist()


# Module-level convenience singleton for code that isn't ChromaDB.
def get_embedding_function() -> E5EmbeddingFunction:
    return E5EmbeddingFunction()
