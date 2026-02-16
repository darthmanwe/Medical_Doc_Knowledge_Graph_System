"""Graph-backed RAG using the full retrieval engine.

Uses ONLY Neo4j (vector index + graph expansion) for retrieval.
Mirrors data2.ai's approach: "every answer reveals the logic behind it."

Pipeline:
  1. Entity-first retrieval via Neo4j vector index → seed entities
  2. K-hop neighborhood expansion → subgraph context (adaptive depth)
  3. Relationship-constrained filtering → clinically relevant paths
  4. Path-based reasoning → explicit reasoning chains
  5. Provenance linking → citations back to source chunks
  6. Re-rank all context elements by query relevance → pruned bundle
  7. Structured context bundle → LLM prompt with citation markers
"""

from __future__ import annotations

import logging
import time

from app.models.schema import QueryResponse
from app.rag.llm_client import generate
from app.retrieval.context_builder import (
    build_context,
    format_context_for_prompt,
    rerank_context_bundle,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a medical knowledge assistant backed by a Knowledge Graph.\n"
    "Answer ONLY from the provided context. Be concise and factual.\n"
    "Rules:\n"
    "1. State only facts supported by the context.\n"
    "2. Cite sources: reference the section and source file from Provenance.\n"
    "3. When reasoning across entities, show the path: A -[REL]-> B.\n"
    "4. If the context is insufficient, say so.\n"
    "5. Expand medical abbreviations on first use."
)


def graph_rag_query(
    question: str,
    top_k: int = 5,
    max_hops: int = 3,
) -> QueryResponse:
    """Execute a graph-backed RAG query using the full retrieval engine."""

    # Step 1-5: Build context bundle using all retrieval patterns
    t_retrieval = time.time()
    context_bundle = build_context(query=question, top_k=top_k, max_hops=max_hops)

    # Step 6: Re-rank context elements by query relevance
    context_bundle = rerank_context_bundle(question, context_bundle)
    retrieval_ms = (time.time() - t_retrieval) * 1000

    # Step 7: Format context for LLM prompt
    formatted_context = format_context_for_prompt(context_bundle)

    # Step 8: Generate answer
    t_gen = time.time()

    user_message = (
        f"{formatted_context}\n\n---\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely using the context above. Cite sources."
    )

    answer = generate(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1024,
    )
    generation_ms = (time.time() - t_gen) * 1000

    return QueryResponse(
        question=question,
        strategy="graph",
        answer=answer,
        citations=context_bundle.citations,
        context_bundle=context_bundle,
        retrieval_time_ms=round(retrieval_ms, 2),
        generation_time_ms=round(generation_ms, 2),
    )
