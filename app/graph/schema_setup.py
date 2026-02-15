"""Create Neo4j constraints, indexes, and vector indexes at startup.

Runs via the *sync* driver so it blocks until schema is ready before
the async FastAPI app starts serving requests.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.graph.connection import sync_session

logger = logging.getLogger(__name__)

# ── Constraint & index DDL statements ──────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_number IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:SourceChunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT medication_name IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT procedure_name IF NOT EXISTS FOR (p:Procedure) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT vital_id IF NOT EXISTS FOR (v:Vital) REQUIRE v.vital_id IS UNIQUE",
    "CREATE CONSTRAINT risk_factor_name IF NOT EXISTS FOR (r:RiskFactor) REQUIRE r.name IS UNIQUE",
]

INDEXES = [
    # Full-text index for hybrid search on chunk text
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:SourceChunk) ON EACH [c.text]",
    # Composite index for faster entity-first lookups
    "CREATE INDEX patient_name IF NOT EXISTS FOR (p:Patient) ON (p.name)",
]

VECTOR_INDEX = (
    "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
    "FOR (c:SourceChunk) ON (c.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}"
)


def apply_schema() -> None:
    """Apply all constraints, indexes, and the vector index."""
    with sync_session() as session:
        for stmt in CONSTRAINTS:
            try:
                session.run(stmt)
                logger.debug("Applied: %s", stmt[:80])
            except Exception as exc:
                logger.warning("Constraint skipped (%s): %s", exc, stmt[:80])

        for stmt in INDEXES:
            try:
                session.run(stmt)
                logger.debug("Applied: %s", stmt[:80])
            except Exception as exc:
                logger.warning("Index skipped (%s): %s", exc, stmt[:80])

        try:
            session.run(VECTOR_INDEX, dim=settings.embedding_dim)
            logger.info(
                "Vector index 'chunk_embedding' ensured (dim=%d).",
                settings.embedding_dim,
            )
        except Exception as exc:
            logger.warning("Vector index skipped (%s)", exc)

    logger.info("Neo4j schema setup complete.")
