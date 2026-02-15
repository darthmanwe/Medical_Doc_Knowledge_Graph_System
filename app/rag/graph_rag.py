"""Graph-backed RAG using the full retrieval engine.

Uses ONLY Neo4j (vector index + graph expansion) for retrieval.
Mirrors data2.ai's approach: "every answer reveals the logic behind it."

Pipeline:
  1. Entity-first retrieval via Neo4j vector index → seed entities
  2. K-hop neighborhood expansion → subgraph context
  3. Relationship-constrained filtering → clinically relevant paths
  4. Path-based reasoning → explicit reasoning chains
  5. Provenance linking → citations back to source chunks
  6. Structured context bundle → LLM prompt with citation markers
"""

from __future__ import annotations

import logging
import time

from app.models.schema import QueryResponse
from app.rag.llm_client import generate
from app.retrieval.context_builder import build_context, format_context_for_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical knowledge assistant powered by a Knowledge Graph.
You have access to structured medical data including entities, relationships, reasoning paths, and provenance citations.

RULES:
1. Only state facts that are directly supported by the provided context.
2. Use the Knowledge Graph structure to reason about relationships between medical concepts.
3. Cite your sources using the provenance citations provided — reference the specific section and source file.
4. When reasoning across multiple entities, explain the path: Entity A -[RELATIONSHIP]-> Entity B.
5. If the context doesn't contain enough information, say so explicitly.
6. Expand all medical abbreviations in your answer.
7. Distinguish between facts extracted from the document vs inferences from graph relationships.
"""


def graph_rag_query(
    question: str,
    top_k: int = 5,
    max_hops: int = 3,
) -> QueryResponse:
    """Execute a graph-backed RAG query using the full retrieval engine."""

    # Step 1-5: Build context bundle using all retrieval patterns
    t_retrieval = time.time()
    context_bundle = build_context(query=question, top_k=top_k, max_hops=max_hops)
    retrieval_ms = (time.time() - t_retrieval) * 1000

    # Step 6: Format context for LLM prompt
    formatted_context = format_context_for_prompt(context_bundle)

    # Step 7: Generate answer
    t_gen = time.time()

    user_message = (
        f"## Knowledge Graph Context\n\n"
        f"{formatted_context}\n\n"
        f"---\n\n"
        f"## Question\n{question}\n\n"
        f"Answer based on the Knowledge Graph context above. "
        f"Cite specific source sections and entity relationships. "
        f"Explain your reasoning path through the graph when applicable."
    )

    answer = generate(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=1500,
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
