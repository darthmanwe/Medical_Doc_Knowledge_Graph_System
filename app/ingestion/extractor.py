"""LLM-based entity and relationship extractor using Claude tool_use.

Schema-guided: the extraction prompt constrains Claude to only emit
entity types and relationship types defined in our graph schema.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.schema import (
    EntityLabel,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    RelationshipType,
    TextChunk,
)
from app.rag.llm_client import extract_structured

logger = logging.getLogger(__name__)

# ── Tool schema for Claude ─────────────────────────────────────────────────────

EXTRACTION_TOOL = {
    "name": "extract_medical_entities",
    "description": (
        "Extract medical entities and relationships from a clinical text chunk. "
        "Return structured JSON with entities and relationships arrays."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "List of medical entities found in the text.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Canonical name of the entity (expand abbreviations: HTN→Hypertension, SOB→Shortness of Breath, etc.)",
                        },
                        "label": {
                            "type": "string",
                            "enum": [e.value for e in EntityLabel],
                            "description": "Entity type label.",
                        },
                        "properties": {
                            "type": "object",
                            "description": "Additional properties depending on entity type.",
                            "properties": {
                                "status": {"type": "string"},
                                "severity": {"type": "string"},
                                "description": {"type": "string"},
                                "frequency": {"type": "string"},
                                "duration": {"type": "string"},
                                "quality": {"type": "string"},
                                "dosage": {"type": "string"},
                                "route": {"type": "string"},
                                "instruction": {"type": "string"},
                                "type": {"type": "string"},
                                "value": {"type": "string"},
                                "unit": {"type": "string"},
                                "source": {"type": "string"},
                            },
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score 0.0-1.0.",
                        },
                    },
                    "required": ["name", "label", "confidence"],
                },
            },
            "relationships": {
                "type": "array",
                "description": "Relationships between extracted entities.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_name": {"type": "string"},
                        "source_label": {
                            "type": "string",
                            "enum": [e.value for e in EntityLabel],
                        },
                        "target_name": {"type": "string"},
                        "target_label": {
                            "type": "string",
                            "enum": [e.value for e in EntityLabel],
                        },
                        "relationship_type": {
                            "type": "string",
                            "enum": [r.value for r in RelationshipType if r not in (
                                RelationshipType.SOURCED_FROM,
                                RelationshipType.BELONGS_TO,
                                RelationshipType.NEXT,
                            )],
                        },
                        "properties": {
                            "type": "object",
                            "description": "Optional relationship properties (e.g., adherence_status).",
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "source_name", "source_label",
                        "target_name", "target_label",
                        "relationship_type", "confidence",
                    ],
                },
            },
        },
        "required": ["entities", "relationships"],
    },
}

SYSTEM_PROMPT = """You are a medical NLP extraction engine. Given a clinical text chunk from a SOAP note, extract all medical entities and their relationships.

RULES:
1. Expand ALL medical abbreviations to their canonical forms:
   - HTN → Hypertension, BP → Blood Pressure, HR → Heart Rate, RR → Respiratory Rate,
   - SpO2 → Oxygen Saturation, SOB → Shortness of Breath, MI → Myocardial Infarction,
   - GERD → Gastroesophageal Reflux Disease, LE → Lower Extremity, ECG → Electrocardiogram,
   - CV → Cardiovascular, CTA → Clear to Auscultation, RRR → Regular Rate and Rhythm,
   - N/V → Nausea/Vomiting, SL → Sublingual, EMS → Emergency Medical Services,
   - f/u → Follow-up, hx → History, NS ST-T → Non-Specific ST-T wave
2. Each entity must have the correct label from: Patient, Condition, Symptom, Medication, Procedure, Vital, RiskFactor
3. Create relationships between entities where clinically appropriate
4. Assign confidence scores based on how explicitly the information appears in the text
5. A patient reference like "Peter Fern (62M)" should be extracted as a Patient entity
6. Vitals (BP, HR, RR, SpO2) should each be separate Vital entities with value and unit
7. Risk factors (age, family history, etc.) should be RiskFactor entities with source property
"""


def extract_from_chunk(chunk: TextChunk) -> ExtractionResult:
    """Extract entities and relationships from a single text chunk."""
    user_msg = (
        f"SOAP Section: {chunk.section.value}\n"
        f"Text:\n{chunk.text}\n\n"
        "Extract all medical entities and relationships from this text."
    )

    try:
        raw = extract_structured(
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_medical_entities"},
        )
    except Exception as exc:
        logger.error("Extraction failed for chunk %s: %s", chunk.chunk_id, exc)
        return ExtractionResult(chunk_id=chunk.chunk_id)

    entities = _parse_entities(raw.get("entities", []), chunk.chunk_id)
    relationships = _parse_relationships(raw.get("relationships", []), chunk.chunk_id)

    logger.info(
        "Chunk %s (%s): extracted %d entities, %d relationships",
        chunk.chunk_id[:8], chunk.section.value, len(entities), len(relationships),
    )

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        entities=entities,
        relationships=relationships,
    )


def extract_from_chunks(chunks: list[TextChunk]) -> list[ExtractionResult]:
    """Extract entities from all chunks sequentially."""
    results: list[ExtractionResult] = []
    for chunk in chunks:
        if not chunk.text.strip():
            continue
        result = extract_from_chunk(chunk)
        results.append(result)
    return results


# ── Internal parsers ───────────────────────────────────────────────────────────


def _parse_entities(raw_entities: list[dict], chunk_id: str) -> list[ExtractedEntity]:
    """Parse raw extraction output into validated ExtractedEntity objects."""
    entities: list[ExtractedEntity] = []
    for raw in raw_entities:
        try:
            entity = ExtractedEntity(
                name=raw["name"],
                label=EntityLabel(raw["label"]),
                properties=raw.get("properties", {}),
                confidence=min(max(raw.get("confidence", 0.5), 0.0), 1.0),
                source_chunk_id=chunk_id,
                extraction_method="llm_claude",
            )
            entities.append(entity)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed entity %s: %s", raw, exc)
    return entities


def _parse_relationships(
    raw_rels: list[dict], chunk_id: str
) -> list[ExtractedRelationship]:
    """Parse raw relationship output into validated objects."""
    relationships: list[ExtractedRelationship] = []
    for raw in raw_rels:
        try:
            rel = ExtractedRelationship(
                source_name=raw["source_name"],
                source_label=EntityLabel(raw["source_label"]),
                target_name=raw["target_name"],
                target_label=EntityLabel(raw["target_label"]),
                relationship_type=RelationshipType(raw["relationship_type"]),
                properties=raw.get("properties", {}),
                confidence=min(max(raw.get("confidence", 0.5), 0.0), 1.0),
                source_chunk_id=chunk_id,
            )
            relationships.append(rel)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed relationship %s: %s", raw, exc)
    return relationships
