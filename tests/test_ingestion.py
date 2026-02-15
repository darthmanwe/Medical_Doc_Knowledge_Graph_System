"""Tests for the ingestion pipeline components."""

import json
from pathlib import Path

import pytest

from app.ingestion.chunker import (
    chunk_soap_notes,
    load_demographics_json,
    parse_demographics,
)
from app.models.schema import SOAPSection


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_SOAP = """Subjective:
Peter Fern (62M) returns for f/u of intermittent exertional chest tightness. Reports 
episodes 2–3x/week when walking uphill or carrying groceries.

Objective:
BP 152/88, HR 78, RR 16, SpO2 97%. Appears well, no acute distress.

Assessment:
1. Likely stable angina given exertional pattern and resolution w/ rest.
2. Hypertension — borderline control; adherence inconsistent.
3. GERD — chronic but unlikely to explain exertional symptoms.

Plan:
- Continue antihypertensives; reinforce daily adherence.
- Provide nitroglycerin SL for exertional discomfort.
"""

SAMPLE_DEMOGRAPHICS = {
    "patient_name": "Peter Julius Fern",
    "dob": "1960-04-15",
    "patient_number": "XXX34-6565-890",
    "health_card_number": "9696178816",
    "phone_home": "613-6565-890",
    "phone_mobile": "647-666-8888",
    "email": "peter.fern@email.com",
    "address": {
        "street": "45 Maple Ave",
        "city": "Toronto",
        "province": "ON",
        "postal_code": "K7L 3V8",
        "country": "Canada",
    },
}


# ── Chunker tests ──────────────────────────────────────────────────────────────

class TestSOAPChunker:
    def test_splits_into_four_sections(self):
        chunks = chunk_soap_notes(SAMPLE_SOAP, "test_doc")
        sections = {c.section for c in chunks}
        assert SOAPSection.SUBJECTIVE in sections
        assert SOAPSection.OBJECTIVE in sections
        assert SOAPSection.ASSESSMENT in sections
        assert SOAPSection.PLAN in sections

    def test_chunk_ids_are_unique(self):
        chunks = chunk_soap_notes(SAMPLE_SOAP, "test_doc")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunks_have_text(self):
        chunks = chunk_soap_notes(SAMPLE_SOAP, "test_doc")
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0, f"Chunk {chunk.chunk_id} has empty text"

    def test_assessment_splits_on_numbered_items(self):
        chunks = chunk_soap_notes(SAMPLE_SOAP, "test_doc")
        assessment_chunks = [c for c in chunks if c.section == SOAPSection.ASSESSMENT]
        # Should have at least 2 sub-chunks for the 3 numbered items
        assert len(assessment_chunks) >= 2

    def test_chunks_retain_doc_id(self):
        chunks = chunk_soap_notes(SAMPLE_SOAP, "test_doc")
        for chunk in chunks:
            assert chunk.doc_id == "test_doc"

    def test_character_offsets_are_valid(self):
        chunks = chunk_soap_notes(SAMPLE_SOAP, "test_doc")
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char


# ── Demographics loader tests ──────────────────────────────────────────────────

class TestDemographicsLoader:
    def test_parse_demographics(self):
        result = parse_demographics(SAMPLE_DEMOGRAPHICS)
        assert result["name"] == "Peter Julius Fern"
        assert result["patient_number"] == "XXX34-6565-890"
        assert result["health_card"] == "9696178816"
        assert result["address_city"] == "Toronto"
        assert result["address_postal"] == "K7L 3V8"

    def test_parse_demographics_all_fields(self):
        result = parse_demographics(SAMPLE_DEMOGRAPHICS)
        expected_keys = {
            "patient_number", "name", "dob", "health_card",
            "phone_home", "phone_mobile", "email",
            "address_street", "address_city", "address_province",
            "address_postal", "address_country",
        }
        assert set(result.keys()) == expected_keys

    def test_load_demographics_json(self, tmp_path):
        json_file = tmp_path / "demo.json"
        json_file.write_text(json.dumps(SAMPLE_DEMOGRAPHICS))
        chunks, data = load_demographics_json(str(json_file), "demo_doc")
        assert len(chunks) == 1
        assert chunks[0].section == SOAPSection.DEMOGRAPHICS
        assert data["patient_name"] == "Peter Julius Fern"


# ── Entity resolver tests (unit-level) ─────────────────────────────────────────

class TestEntityResolver:
    def test_medical_synonyms_mapping(self):
        from app.ingestion.entity_resolver import MEDICAL_SYNONYMS
        assert MEDICAL_SYNONYMS["htn"] == "Hypertension"
        assert MEDICAL_SYNONYMS["mi"] == "Myocardial Infarction"
        assert MEDICAL_SYNONYMS["gerd"] == "Gastroesophageal Reflux Disease"
