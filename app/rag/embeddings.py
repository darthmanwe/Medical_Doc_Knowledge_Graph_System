"""Shared embedding module — singleton wrapper around sentence-transformers.

Uses all-MiniLM-L6-v2 (384 dimensions) by default.
Called by: ingestion pipeline, vector RAG, graph RAG (entity-first seed).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    """Lazily load the sentence-transformers model (downloads ~90 MB on first run)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s' …", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info(
            "Embedding model loaded (dim=%d).",
            _model.get_sentence_embedding_dimension(),
        )
    return _model


def warm_load() -> None:
    """Force-load the model at startup so the first request isn't slow."""
    _load_model()


def embed_text(text: str) -> list[float]:
    """Embed a single text string → list of floats."""
    model = _load_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of texts → list of float-vectors."""
    if not texts:
        return []
    model = _load_model()
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))
