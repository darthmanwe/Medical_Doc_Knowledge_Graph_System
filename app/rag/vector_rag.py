"""Vector-only RAG baseline.

Uses ONLY ChromaDB for retrieval (no graph context).
Same embeddings model, same LLM, same prompt template as graph RAG
— only the retrieval strategy differs, ensuring a fair comparison.
"""

from __future__ import annotations

import logging
import time

import chromadb

from app.config import settings
from app.models.schema import Citation, ContextBundle, QueryResponse
from app.rag.embeddings import embed_text
from app.rag.llm_client import generate

logger = logging.getLogger(__name__)

_chroma_client: chromadb.ClientAPI | None = None
_collection = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


SYSTEM_PROMPT = """You are a medical knowledge assistant. Answer the question based ONLY on the provided context chunks.

RULES:
1. Only state facts that are directly supported by the provided context.
2. If the context doesn't contain enough information, say so explicitly.
3. Cite which chunk supports each claim using [Chunk N] notation.
4. Be precise with medical terminology.
5. Expand all medical abbreviations in your answer.
"""


def vector_rag_query(
    question: str,
    top_k: int = 5,
) -> QueryResponse:
    """Execute a vector-only RAG query against ChromaDB."""
    t_retrieval = time.time()

    collection = _get_collection()
    query_embedding = embed_text(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieval_ms = (time.time() - t_retrieval) * 1000

    # Build context from retrieved chunks
    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    context_parts: list[str] = []
    for i, (doc, meta, dist) in enumerate(zip(chunks, metadatas, distances), 1):
        section = meta.get("section", "unknown") if meta else "unknown"
        context_parts.append(f"[Chunk {i}] (Section: {section}, similarity: {1-dist:.3f})\n{doc}")

    context_text = "\n\n".join(context_parts)

    # Generate answer
    t_gen = time.time()
    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context above, citing [Chunk N] for each claim."
    )

    answer = generate(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    generation_ms = (time.time() - t_gen) * 1000

    # Build minimal citations (no graph provenance available)
    citations: list[Citation] = []
    for i, (doc, meta) in enumerate(zip(chunks, metadatas)):
        citations.append(Citation(
            entity_name=f"Chunk {i+1}",
            source_text=doc[:200] if doc else "",
            section=meta.get("section", "unknown") if meta else "unknown",
            source_file=meta.get("doc_id", "unknown") if meta else "unknown",
            confidence=1 - distances[i] if distances else 0.0,
            extraction_method="vector_similarity",
        ))

    return QueryResponse(
        question=question,
        strategy="vector",
        answer=answer,
        citations=citations,
        context_bundle=ContextBundle(raw_chunks=chunks),
        retrieval_time_ms=round(retrieval_ms, 2),
        generation_time_ms=round(generation_ms, 2),
    )
