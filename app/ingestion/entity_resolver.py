"""Two-pass entity resolution: fuzzy string match + embedding-based semantic similarity.

Ensures abbreviation variants (HTN / Hypertension) and near-duplicates are
merged under a single canonical name before graph write.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from rapidfuzz import fuzz

from app.models.schema import ExtractionResult, ExtractedEntity, ExtractedRelationship
from app.rag.embeddings import cosine_similarity, embed_batch

logger = logging.getLogger(__name__)

# Known medical synonym map (abbreviation → canonical)
MEDICAL_SYNONYMS: dict[str, str] = {
    "htn": "Hypertension",
    "high blood pressure": "Hypertension",
    "bp": "Blood Pressure",
    "hr": "Heart Rate",
    "rr": "Respiratory Rate",
    "spo2": "Oxygen Saturation",
    "sob": "Shortness of Breath",
    "mi": "Myocardial Infarction",
    "heart attack": "Myocardial Infarction",
    "gerd": "Gastroesophageal Reflux Disease",
    "acid reflux": "Gastroesophageal Reflux Disease",
    "ecg": "Electrocardiogram",
    "ekg": "Electrocardiogram",
    "le edema": "Lower Extremity Edema",
    "ntg": "Nitroglycerin",
    "nitroglycerin sl": "Nitroglycerin Sublingual",
    "f/u": "Follow-up",
    "cta": "Clear to Auscultation",
    "rrr": "Regular Rate and Rhythm",
}

# Thresholds
FUZZY_THRESHOLD = 88          # rapidfuzz token_sort_ratio threshold
SEMANTIC_THRESHOLD = 0.85     # cosine similarity threshold for embedding match


def resolve_entities(results: list[ExtractionResult]) -> list[ExtractionResult]:
    """Run two-pass resolution across all extraction results.

    Pass 1: Exact / fuzzy string matching via rapidfuzz.
    Pass 2: Embedding-based semantic similarity for remaining entities.
    """
    # Collect all unique entity names across results
    all_entities: list[ExtractedEntity] = []
    for r in results:
        all_entities.extend(r.entities)

    if not all_entities:
        return results

    # Build canonical name mapping
    name_map = _build_name_map(all_entities)

    # Apply mapping to all results
    resolved = []
    for result in results:
        new_entities = [_remap_entity(e, name_map) for e in result.entities]
        new_rels = [_remap_relationship(r, name_map) for r in result.relationships]

        # Deduplicate entities within the same chunk
        seen: dict[tuple[str, str], ExtractedEntity] = {}
        deduped: list[ExtractedEntity] = []
        for e in new_entities:
            key = (e.name, e.label.value)
            if key in seen:
                if e.confidence > seen[key].confidence:
                    seen[key] = e
                    deduped = [x for x in deduped if (x.name, x.label.value) != key]
                    deduped.append(e)
            else:
                seen[key] = e
                deduped.append(e)

        resolved.append(
            ExtractionResult(
                chunk_id=result.chunk_id,
                entities=deduped,
                relationships=new_rels,
            )
        )

    total_remaps = sum(1 for v in name_map.values() if v != v)
    logger.info(
        "Entity resolution: %d unique names, %d remappings applied.",
        len(name_map),
        sum(1 for k, v in name_map.items() if k != v),
    )

    return resolved


def _build_name_map(entities: list[ExtractedEntity]) -> dict[str, str]:
    """Build a mapping from original entity names to canonical names."""
    unique_names = list({e.name for e in entities})
    name_map: dict[str, str] = {n: n for n in unique_names}

    # Pass 1: Known synonyms + fuzzy match
    for name in unique_names:
        lower = name.lower().strip()

        # Check known synonym map first
        if lower in MEDICAL_SYNONYMS:
            name_map[name] = MEDICAL_SYNONYMS[lower]
            continue

        # Fuzzy match against all other names
        for other in unique_names:
            if other == name:
                continue
            score = fuzz.token_sort_ratio(name.lower(), other.lower())
            if score >= FUZZY_THRESHOLD:
                # Keep the longer / more descriptive name as canonical
                canonical = name if len(name) >= len(other) else other
                name_map[name] = canonical
                name_map[other] = canonical

    # Pass 2: Embedding-based semantic match for remaining distinct names
    canonical_names = list(set(name_map.values()))
    if len(canonical_names) > 1:
        embeddings = embed_batch(canonical_names)
        for i, name_a in enumerate(canonical_names):
            for j, name_b in enumerate(canonical_names):
                if j <= i:
                    continue
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= SEMANTIC_THRESHOLD:
                    canonical = name_a if len(name_a) >= len(name_b) else name_b
                    # Update all entries pointing to name_a or name_b
                    for k, v in name_map.items():
                        if v == name_a or v == name_b:
                            name_map[k] = canonical

    return name_map


def _remap_entity(entity: ExtractedEntity, name_map: dict[str, str]) -> ExtractedEntity:
    """Apply name mapping to an entity."""
    canonical = name_map.get(entity.name, entity.name)
    if canonical != entity.name:
        return entity.model_copy(update={"name": canonical})
    return entity


def _remap_relationship(
    rel: ExtractedRelationship, name_map: dict[str, str]
) -> ExtractedRelationship:
    """Apply name mapping to relationship source/target names."""
    updates: dict = {}
    if rel.source_name in name_map and name_map[rel.source_name] != rel.source_name:
        updates["source_name"] = name_map[rel.source_name]
    if rel.target_name in name_map and name_map[rel.target_name] != rel.target_name:
        updates["target_name"] = name_map[rel.target_name]
    if updates:
        return rel.model_copy(update=updates)
    return rel
