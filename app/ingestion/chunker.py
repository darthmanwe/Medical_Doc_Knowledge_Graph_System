"""SOAP-section-aware text chunker.

Parses medical SOAP notes into section-labelled chunks.
Also provides a structured JSON loader for demographics data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.models.schema import SOAPSection, TextChunk

# Section header patterns (case-insensitive, handles trailing colons/spaces)
_SECTION_PATTERNS: list[tuple[SOAPSection, re.Pattern]] = [
    (SOAPSection.SUBJECTIVE, re.compile(r"^Subjective\s*:", re.IGNORECASE)),
    (SOAPSection.OBJECTIVE, re.compile(r"^Objective\s*:", re.IGNORECASE)),
    (SOAPSection.ASSESSMENT, re.compile(r"^Assessment\s*:", re.IGNORECASE)),
    (SOAPSection.PLAN, re.compile(r"^Plan\s*:", re.IGNORECASE)),
]


def _make_chunk_id(doc_id: str, section: str, idx: int) -> str:
    """Deterministic chunk ID from document, section, and index."""
    raw = f"{doc_id}::{section}::{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── SOAP chunker ───────────────────────────────────────────────────────────────


def chunk_soap_notes(
    text: str,
    doc_id: str,
    max_chunk_chars: int = 600,
    overlap_chars: int = 80,
) -> list[TextChunk]:
    """Split SOAP note text into section-aware chunks.

    Strategy:
      1. Split text into SOAP sections by header.
      2. Within each section, apply sliding-window chunking if the
         section exceeds *max_chunk_chars*.
      3. In the Assessment section, additionally split on numbered items.
    """
    sections = _split_into_sections(text)
    chunks: list[TextChunk] = []
    global_idx = 0

    for section_label, section_text, sec_start in sections:
        if section_label == SOAPSection.ASSESSMENT:
            sub_texts = _split_assessment_items(section_text, sec_start)
        else:
            sub_texts = [(section_text, sec_start)]

        for sub_text, sub_start in sub_texts:
            windows = _sliding_window(sub_text, max_chunk_chars, overlap_chars)
            for win_text, win_offset in windows:
                abs_start = sub_start + win_offset
                chunk = TextChunk(
                    chunk_id=_make_chunk_id(doc_id, section_label.value, global_idx),
                    text=win_text.strip(),
                    section=section_label,
                    start_char=abs_start,
                    end_char=abs_start + len(win_text),
                    doc_id=doc_id,
                )
                chunks.append(chunk)
                global_idx += 1

    return chunks


def _split_into_sections(text: str) -> list[tuple[SOAPSection, str, int]]:
    """Return [(section_label, section_body_text, start_char), ...]."""
    boundaries: list[tuple[int, SOAPSection]] = []

    for line_match in re.finditer(r"^(.+)$", text, re.MULTILINE):
        line = line_match.group(1)
        for sec, pat in _SECTION_PATTERNS:
            if pat.match(line):
                header_end = line_match.start() + pat.match(line).end()
                boundaries.append((header_end, sec))
                break

    if not boundaries:
        return [(SOAPSection.SUBJECTIVE, text, 0)]

    sections: list[tuple[SOAPSection, str, int]] = []
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] - len(label.value) - 2 if i + 1 < len(boundaries) else len(text)
        # Find the actual start of next section header for cleaner end boundary
        if i + 1 < len(boundaries):
            next_header_start = text.rfind("\n", 0, boundaries[i + 1][0])
            if next_header_start > start:
                end = next_header_start
        body = text[start:end].strip()
        sections.append((label, body, start))

    return sections


def _split_assessment_items(text: str, base_offset: int) -> list[tuple[str, int]]:
    """Split Assessment section on numbered items (e.g., '1. ...', '2. ...')."""
    items: list[tuple[str, int]] = []
    pattern = re.compile(r"(?:^|\n)\s*(\d+)\.\s+", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if len(matches) <= 1:
        return [(text, base_offset)]

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        items.append((text[start:end].strip(), base_offset + start))

    return items


def _sliding_window(
    text: str, max_chars: int, overlap: int
) -> list[tuple[str, int]]:
    """Sliding window over text → list of (window_text, offset)."""
    if len(text) <= max_chars:
        return [(text, 0)]

    windows: list[tuple[str, int]] = []
    step = max(max_chars - overlap, 1)
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        windows.append((text[pos:end], pos))
        if end == len(text):
            break
        pos += step
    return windows


# ── Demographics JSON loader ───────────────────────────────────────────────────


def load_demographics_json(path: str | Path, doc_id: str) -> tuple[list[TextChunk], dict]:
    """Load a demographics JSON file as a single SourceChunk.

    Structured data doesn't need NLP chunking — the whole JSON is one chunk
    so provenance traces back to the raw file.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    chunk = TextChunk(
        chunk_id=_make_chunk_id(doc_id, "Demographics", 0),
        text=raw,
        section=SOAPSection.DEMOGRAPHICS,
        start_char=0,
        end_char=len(raw),
        doc_id=doc_id,
    )
    return [chunk], data


def parse_demographics(data: dict) -> dict:
    """Flatten demographics JSON into Patient node properties."""
    address = data.get("address", {})
    return {
        "patient_number": data.get("patient_number", ""),
        "name": data.get("patient_name", ""),
        "dob": data.get("dob", ""),
        "health_card": data.get("health_card_number", ""),
        "phone_home": data.get("phone_home", ""),
        "phone_mobile": data.get("phone_mobile", ""),
        "email": data.get("email", ""),
        "address_street": address.get("street", ""),
        "address_city": address.get("city", ""),
        "address_province": address.get("province", ""),
        "address_postal": address.get("postal_code", ""),
        "address_country": address.get("country", ""),
    }
