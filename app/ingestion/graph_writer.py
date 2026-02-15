"""Idempotent Neo4j graph writer with provenance edges.

Writes entities, relationships, chunks, and documents to Neo4j using
per-label MERGE queries (safe Cypher — no dynamic labels).
Dual-writes chunk embeddings to ChromaDB for vector-only RAG baseline.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from app.config import settings
from app.graph.connection import sync_session
from app.graph import queries as Q
from app.models.schema import (
    EntityLabel,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    TextChunk,
)

logger = logging.getLogger(__name__)

# ── ChromaDB client (lazy singleton) ──────────────────────────────────────────

_chroma_client: chromadb.ClientAPI | None = None
_chroma_collection = None


def _get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _chroma_collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


# ── Entity upsert dispatch ─────────────────────────────────────────────────────

_LABEL_TO_QUERY: dict[EntityLabel, str] = {
    EntityLabel.CONDITION: Q.UPSERT_CONDITION,
    EntityLabel.SYMPTOM: Q.UPSERT_SYMPTOM,
    EntityLabel.MEDICATION: Q.UPSERT_MEDICATION,
    EntityLabel.PROCEDURE: Q.UPSERT_PROCEDURE,
    EntityLabel.VITAL: Q.UPSERT_VITAL,
    EntityLabel.RISK_FACTOR: Q.UPSERT_RISK_FACTOR,
}

# Relationship type → Cypher query template
_REL_QUERIES: dict[str, str] = {
    "HAS_CONDITION": Q.LINK_PATIENT_CONDITION,
    "EXHIBITS_SYMPTOM": Q.LINK_PATIENT_SYMPTOM,
    "TAKES_MEDICATION": Q.LINK_PATIENT_MEDICATION,
    "HAS_VITAL": Q.LINK_PATIENT_VITAL,
    "HAS_RISK_FACTOR": Q.LINK_PATIENT_RISK_FACTOR,
    "SCHEDULED_FOR": Q.LINK_PATIENT_PROCEDURE,
    "MANIFESTS_AS": Q.LINK_CONDITION_SYMPTOM,
    "TREATED_WITH": Q.LINK_CONDITION_MEDICATION,
}


def write_document(doc_id: str, doc_type: str, source_file: str) -> None:
    """Create/update a Document node."""
    with sync_session() as session:
        session.run(
            Q.UPSERT_DOCUMENT,
            doc_id=doc_id,
            doc_type=doc_type,
            source_file=source_file,
        )
    logger.debug("Document node upserted: %s", doc_id)


def write_chunks(chunks: list[TextChunk]) -> int:
    """Write SourceChunk nodes to Neo4j and dual-write embeddings to ChromaDB.

    Returns the number of chunks written.
    """
    collection = _get_chroma_collection()
    written = 0

    with sync_session() as session:
        prev_chunk_id: str | None = None

        for chunk in chunks:
            session.run(
                Q.UPSERT_CHUNK,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                section=chunk.section.value,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                embedding=chunk.embedding or [],
                doc_id=chunk.doc_id,
            )

            # Linked-list ordering
            if prev_chunk_id is not None:
                session.run(
                    Q.LINK_CHUNK_SEQUENCE,
                    prev_chunk_id=prev_chunk_id,
                    curr_chunk_id=chunk.chunk_id,
                )
            prev_chunk_id = chunk.chunk_id
            written += 1

    # Dual-write to ChromaDB
    ids = [c.chunk_id for c in chunks if c.embedding]
    embeds = [c.embedding for c in chunks if c.embedding]
    docs = [c.text for c in chunks if c.embedding]
    metas = [{"section": c.section.value, "doc_id": c.doc_id} for c in chunks if c.embedding]

    if ids:
        collection.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
        logger.info("ChromaDB: upserted %d chunk embeddings.", len(ids))

    return written


def write_patient(patient_props: dict[str, Any]) -> None:
    """Write a Patient node from parsed demographics."""
    with sync_session() as session:
        session.run(Q.UPSERT_PATIENT, **patient_props)
    logger.info("Patient node upserted: %s", patient_props.get("name"))


def write_entities(
    extraction_results: list[ExtractionResult],
    patient_number: str,
) -> int:
    """Write extracted entities to Neo4j with provenance edges.

    Returns the number of entities written.
    """
    written = 0
    with sync_session() as session:
        for result in extraction_results:
            for entity in result.entities:
                if entity.label == EntityLabel.PATIENT:
                    continue  # Patient written separately from demographics

                params = _entity_to_params(entity)
                query = _LABEL_TO_QUERY.get(entity.label)
                if not query:
                    logger.warning("No upsert query for label %s", entity.label)
                    continue

                session.run(query, **params)

                # Provenance edge
                session.run(
                    Q.LINK_ENTITY_SOURCE,
                    entity_name=entity.name,
                    entity_label=entity.label.value,
                    chunk_id=entity.source_chunk_id,
                    confidence=entity.confidence,
                    extraction_method=entity.extraction_method,
                )
                written += 1

    logger.info("Wrote %d entities to Neo4j.", written)
    return written


def write_relationships(
    extraction_results: list[ExtractionResult],
    patient_number: str,
) -> int:
    """Write extracted relationships to Neo4j. Returns count written."""
    written = 0
    with sync_session() as session:
        for result in extraction_results:
            for rel in result.relationships:
                rel_type = rel.relationship_type.value
                query = _REL_QUERIES.get(rel_type)
                if not query:
                    logger.warning("No query for relationship type %s", rel_type)
                    continue

                params = _relationship_to_params(rel, patient_number)
                try:
                    session.run(query, **params)
                    written += 1
                except Exception as exc:
                    logger.warning(
                        "Relationship write failed (%s -> %s): %s",
                        rel.source_name, rel.target_name, exc,
                    )

    logger.info("Wrote %d relationships to Neo4j.", written)
    return written


# ── Param builders ─────────────────────────────────────────────────────────────


def _entity_to_params(entity: ExtractedEntity) -> dict[str, Any]:
    """Convert an ExtractedEntity to Neo4j query parameters."""
    props = entity.properties or {}

    if entity.label == EntityLabel.CONDITION:
        return {
            "name": entity.name,
            "status": props.get("status", "active"),
            "severity": props.get("severity", ""),
        }
    elif entity.label == EntityLabel.SYMPTOM:
        return {
            "name": entity.name,
            "description": props.get("description", ""),
            "frequency": props.get("frequency", ""),
            "duration": props.get("duration", ""),
            "quality": props.get("quality", ""),
        }
    elif entity.label == EntityLabel.MEDICATION:
        return {
            "name": entity.name,
            "dosage": props.get("dosage", ""),
            "route": props.get("route", ""),
            "instruction": props.get("instruction", ""),
        }
    elif entity.label == EntityLabel.PROCEDURE:
        return {
            "name": entity.name,
            "type": props.get("type", ""),
            "status": props.get("status", ""),
        }
    elif entity.label == EntityLabel.VITAL:
        return {
            "vital_id": f"{entity.name}_{props.get('value', '')}",
            "type": entity.name,
            "value": props.get("value", ""),
            "unit": props.get("unit", ""),
        }
    elif entity.label == EntityLabel.RISK_FACTOR:
        return {
            "name": entity.name,
            "source": props.get("source", "patient"),
        }
    else:
        return {"name": entity.name}


def _relationship_to_params(
    rel: ExtractedRelationship, patient_number: str
) -> dict[str, Any]:
    """Convert an ExtractedRelationship to Neo4j query parameters."""
    rel_type = rel.relationship_type.value
    base: dict[str, Any] = {"confidence": rel.confidence}

    if rel_type in ("HAS_CONDITION", "EXHIBITS_SYMPTOM", "TAKES_MEDICATION",
                     "HAS_VITAL", "HAS_RISK_FACTOR", "SCHEDULED_FOR"):
        base["patient_number"] = patient_number

    if rel_type == "HAS_CONDITION":
        base["condition_name"] = rel.target_name
    elif rel_type == "EXHIBITS_SYMPTOM":
        base["symptom_name"] = rel.target_name
    elif rel_type == "TAKES_MEDICATION":
        base["medication_name"] = rel.target_name
        base["adherence_status"] = rel.properties.get("adherence_status", "")
    elif rel_type == "HAS_VITAL":
        base["vital_id"] = f"{rel.target_name}_{rel.properties.get('value', '')}"
    elif rel_type == "HAS_RISK_FACTOR":
        base["risk_name"] = rel.target_name
    elif rel_type == "SCHEDULED_FOR":
        base["procedure_name"] = rel.target_name
    elif rel_type == "MANIFESTS_AS":
        base["condition_name"] = rel.source_name
        base["symptom_name"] = rel.target_name
    elif rel_type == "TREATED_WITH":
        base["condition_name"] = rel.source_name
        base["medication_name"] = rel.target_name

    return base
