"""Ingestion pipeline orchestrator.

Two paths:
  1. Structured JSON (demographics) → Patient node + provenance chunk
  2. Unstructured SOAP notes → chunk → extract → resolve → write

Both paths produce (:SourceChunk)-[:BELONGS_TO]->(:Document) provenance.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from app.ingestion.chunker import (
    chunk_soap_notes,
    load_demographics_json,
    parse_demographics,
)
from app.ingestion.extractor import extract_from_chunks
from app.ingestion.entity_resolver import resolve_entities
from app.ingestion.graph_writer import (
    write_chunks,
    write_document,
    write_entities,
    write_patient,
    write_relationships,
)
from app.models.schema import IngestResponse
from app.rag.embeddings import embed_batch

logger = logging.getLogger(__name__)


def run_ingestion(
    soap_path: str = "Task_Files/soap_notes.txt",
    demographics_path: str = "Task_Files/demographics.json",
) -> IngestResponse:
    """Execute the full ingestion pipeline end-to-end."""
    t0 = time.time()
    total_nodes = 0
    total_rels = 0
    total_chunks = 0

    # ── 1. Demographics (structured path) ──────────────────────────────────────
    logger.info("=== Ingesting demographics from %s ===", demographics_path)
    demo_doc_id = "demographics_001"
    write_document(demo_doc_id, "demographics_json", demographics_path)

    demo_chunks, demo_data = load_demographics_json(demographics_path, demo_doc_id)

    # Embed and write demographics chunk
    demo_texts = [c.text for c in demo_chunks]
    demo_embeddings = embed_batch(demo_texts)
    for chunk, emb in zip(demo_chunks, demo_embeddings):
        chunk.embedding = emb
    total_chunks += write_chunks(demo_chunks)

    # Write Patient node
    patient_props = parse_demographics(demo_data)
    write_patient(patient_props)
    total_nodes += 1

    # Provenance for patient → demographics chunk
    from app.graph.connection import sync_session
    from app.graph.queries import LINK_ENTITY_SOURCE
    with sync_session() as session:
        session.run(
            LINK_ENTITY_SOURCE,
            entity_name=patient_props["name"],
            entity_label="Patient",
            chunk_id=demo_chunks[0].chunk_id,
            confidence=1.0,
            extraction_method="structured_json",
        )

    patient_number = patient_props["patient_number"]
    logger.info("Patient node created: %s (%s)", patient_props["name"], patient_number)

    # ── 2. SOAP notes (unstructured NLP path) ──────────────────────────────────
    logger.info("=== Ingesting SOAP notes from %s ===", soap_path)
    soap_doc_id = "soap_notes_001"
    write_document(soap_doc_id, "soap_notes", soap_path)

    # Read file
    soap_text = Path(soap_path).read_text(encoding="utf-8")

    # Chunk
    chunks = chunk_soap_notes(soap_text, soap_doc_id)
    logger.info("Chunked SOAP notes into %d chunks.", len(chunks))

    # Embed chunks
    chunk_texts = [c.text for c in chunks]
    embeddings = embed_batch(chunk_texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    # Write chunks to Neo4j + ChromaDB
    total_chunks += write_chunks(chunks)

    # Extract entities and relationships
    logger.info("Running LLM entity extraction on %d chunks…", len(chunks))
    extraction_results = extract_from_chunks(chunks)

    # Resolve entities (fuzzy + semantic dedup)
    logger.info("Resolving entities…")
    resolved_results = resolve_entities(extraction_results)

    # Write entities
    total_nodes += write_entities(resolved_results, patient_number)

    # Write relationships
    total_rels += write_relationships(resolved_results, patient_number)

    elapsed = time.time() - t0
    logger.info(
        "Ingestion complete: %d nodes, %d relationships, %d chunks in %.1fs",
        total_nodes, total_rels, total_chunks, elapsed,
    )

    return IngestResponse(
        status="success",
        nodes_created=total_nodes,
        relationships_created=total_rels,
        chunks_indexed=total_chunks,
        duration_seconds=round(elapsed, 2),
    )
