"""Provenance / source-of-truth linking.

Traces every entity back to its source document chunks with full citation
metadata: section, source file, extraction confidence, extraction method.

This is the critical explainability layer aligned with data2.ai's eXAI philosophy:
"every answer reveals the logic behind it — showing exactly how and why the data connects."
"""

from __future__ import annotations

import logging

from app.graph.connection import sync_session
from app.models.schema import Citation

logger = logging.getLogger(__name__)

PROVENANCE_QUERY = """
MATCH (entity)-[sf:SOURCED_FROM]->(chunk:SourceChunk)-[bt:BELONGS_TO]->(doc:Document)
WHERE elementId(entity) IN $entity_ids
RETURN
    entity.name AS entity_name,
    chunk.text AS source_text,
    chunk.section AS section,
    chunk.chunk_id AS chunk_id,
    doc.source_file AS source_file,
    doc.doc_type AS doc_type,
    sf.confidence AS extraction_confidence,
    sf.extraction_method AS extraction_method
ORDER BY sf.confidence DESC
"""

PROVENANCE_BY_NAME_QUERY = """
MATCH (entity)-[sf:SOURCED_FROM]->(chunk:SourceChunk)-[bt:BELONGS_TO]->(doc:Document)
WHERE entity.name IN $entity_names
RETURN
    entity.name AS entity_name,
    labels(entity) AS entity_labels,
    chunk.text AS source_text,
    chunk.section AS section,
    chunk.chunk_id AS chunk_id,
    doc.source_file AS source_file,
    sf.confidence AS extraction_confidence,
    sf.extraction_method AS extraction_method
ORDER BY sf.confidence DESC
"""

# Trace a specific answer back to all supporting evidence
FULL_TRACE_QUERY = """
MATCH (entity)-[sf:SOURCED_FROM]->(chunk:SourceChunk)-[:BELONGS_TO]->(doc:Document)
WHERE elementId(entity) IN $entity_ids
OPTIONAL MATCH (chunk)-[:NEXT]->(next_chunk:SourceChunk)
RETURN
    entity.name AS entity_name,
    labels(entity) AS entity_labels,
    chunk.text AS source_text,
    chunk.section AS section,
    doc.source_file AS source_file,
    sf.confidence AS confidence,
    sf.extraction_method AS method,
    next_chunk.text AS continuation_text
ORDER BY doc.source_file, chunk.start_char
"""


def get_citations_by_ids(entity_ids: list[str]) -> list[Citation]:
    """Get provenance citations for entities by element ID."""
    if not entity_ids:
        return []

    citations: list[Citation] = []
    with sync_session() as session:
        result = session.run(PROVENANCE_QUERY, entity_ids=entity_ids)
        for record in result:
            citations.append(Citation(
                entity_name=record["entity_name"],
                source_text=record["source_text"],
                section=record["section"],
                source_file=record["source_file"],
                confidence=record["extraction_confidence"] or 0.0,
                extraction_method=record["extraction_method"] or "unknown",
            ))

    logger.info("Retrieved %d provenance citations.", len(citations))
    return citations


def get_citations_by_names(entity_names: list[str]) -> list[Citation]:
    """Get provenance citations for entities by name."""
    if not entity_names:
        return []

    citations: list[Citation] = []
    with sync_session() as session:
        result = session.run(PROVENANCE_BY_NAME_QUERY, entity_names=entity_names)
        for record in result:
            citations.append(Citation(
                entity_name=record["entity_name"],
                source_text=record["source_text"],
                section=record["section"],
                source_file=record["source_file"],
                confidence=record["extraction_confidence"] or 0.0,
                extraction_method=record["extraction_method"] or "unknown",
            ))

    logger.info("Retrieved %d provenance citations by name.", len(citations))
    return citations


def get_full_evidence_trace(entity_ids: list[str]) -> list[dict]:
    """Get full evidence trace including continuation chunks for richer context."""
    if not entity_ids:
        return []

    traces: list[dict] = []
    with sync_session() as session:
        result = session.run(FULL_TRACE_QUERY, entity_ids=entity_ids)
        for record in result:
            traces.append({
                "entity_name": record["entity_name"],
                "entity_labels": record["entity_labels"],
                "source_text": record["source_text"],
                "section": record["section"],
                "source_file": record["source_file"],
                "confidence": record["confidence"],
                "method": record["method"],
                "continuation_text": record["continuation_text"],
            })

    return traces
