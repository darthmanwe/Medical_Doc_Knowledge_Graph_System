"""Entity-first retrieval pattern.

Strategy: embed the query → vector similarity search on SourceChunk
embeddings in Neo4j → follow SOURCED_FROM edges to discover seed entities.

This is the entry point for all graph-backed retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.graph.connection import sync_session
from app.models.schema import GraphNode
from app.rag.embeddings import embed_text
from app.retrieval.utils import sanitize_properties

logger = logging.getLogger(__name__)

VECTOR_SEED_QUERY = """
CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $query_embedding)
YIELD node AS chunk, score
WHERE score >= $threshold
OPTIONAL MATCH (entity)-[sf:SOURCED_FROM]->(chunk)
RETURN chunk, score,
       CASE WHEN entity IS NOT NULL
            THEN {element_id: elementId(entity), labels: labels(entity), properties: properties(entity)}
            ELSE null
       END AS entity_data
ORDER BY score DESC
"""


def entity_first_retrieval(
    query: str,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
) -> tuple[list[GraphNode], list[dict[str, Any]]]:
    """Find seed entities by embedding-similarity on source chunks.

    Returns:
        (seed_entities, matched_chunks)
        where each chunk dict has {chunk_id, text, section, score}.
    """
    top_k = top_k or settings.retrieval_top_k
    threshold = threshold or settings.retrieval_score_threshold

    query_embedding = embed_text(query)
    seen_entity_ids: set[str] = set()
    seed_entities: list[GraphNode] = []
    matched_chunks: list[dict[str, Any]] = []

    with sync_session() as session:
        result = session.run(
            VECTOR_SEED_QUERY,
            top_k=top_k,
            query_embedding=query_embedding,
            threshold=threshold,
        )

        for record in result:
            chunk_node = record["chunk"]
            score = record["score"]
            entity_data = record["entity_data"]

            matched_chunks.append({
                "chunk_id": chunk_node.get("chunk_id"),
                "text": chunk_node.get("text"),
                "section": chunk_node.get("section"),
                "score": score,
            })

            if entity_data and entity_data["element_id"] not in seen_entity_ids:
                seen_entity_ids.add(entity_data["element_id"])
                seed_entities.append(
                    GraphNode(
                        element_id=entity_data["element_id"],
                        labels=entity_data["labels"],
                        properties=sanitize_properties(entity_data.get("properties") or {}),
                    )
                )

    logger.info(
        "Entity-first retrieval: %d seed entities from %d matched chunks.",
        len(seed_entities), len(matched_chunks),
    )
    return seed_entities, matched_chunks
