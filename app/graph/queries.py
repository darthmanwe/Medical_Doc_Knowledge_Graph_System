"""Reusable Cypher query templates.

Each query is a plain string with ``$param`` placeholders ready for
``session.run(QUERY, param=value)`` usage.
"""

# ── Ingestion ──────────────────────────────────────────────────────────────────

UPSERT_DOCUMENT = """
MERGE (d:Document {doc_id: $doc_id})
ON CREATE SET d.doc_type      = $doc_type,
              d.source_file   = $source_file,
              d.ingested_at   = datetime()
ON MATCH  SET d.updated_at    = datetime()
RETURN d
"""

UPSERT_CHUNK = """
MERGE (c:SourceChunk {chunk_id: $chunk_id})
ON CREATE SET c.text       = $text,
              c.section    = $section,
              c.start_char = $start_char,
              c.end_char   = $end_char,
              c.embedding  = $embedding,
              c.created_at = datetime()
ON MATCH  SET c.updated_at = datetime()
WITH c
MATCH (d:Document {doc_id: $doc_id})
MERGE (c)-[:BELONGS_TO]->(d)
RETURN c
"""

LINK_CHUNK_SEQUENCE = """
MATCH (a:SourceChunk {chunk_id: $prev_chunk_id})
MATCH (b:SourceChunk {chunk_id: $curr_chunk_id})
MERGE (a)-[:NEXT]->(b)
"""

# ── Per-label entity upserts (safe Cypher, no dynamic labels) ──────────────────

UPSERT_PATIENT = """
MERGE (p:Patient {patient_number: $patient_number})
ON CREATE SET p.name              = $name,
              p.dob               = $dob,
              p.health_card       = $health_card,
              p.phone_home        = $phone_home,
              p.phone_mobile      = $phone_mobile,
              p.email             = $email,
              p.address_street    = $address_street,
              p.address_city      = $address_city,
              p.address_province  = $address_province,
              p.address_postal    = $address_postal,
              p.address_country   = $address_country,
              p.created_at        = datetime()
ON MATCH  SET p.updated_at        = datetime()
RETURN p
"""

UPSERT_CONDITION = """
MERGE (c:Condition {name: $name})
ON CREATE SET c.status     = $status,
              c.severity   = $severity,
              c.created_at = datetime()
ON MATCH  SET c.updated_at = datetime()
RETURN c
"""

UPSERT_SYMPTOM = """
MERGE (s:Symptom {name: $name})
ON CREATE SET s.description = $description,
              s.frequency   = $frequency,
              s.duration    = $duration,
              s.quality     = $quality,
              s.created_at  = datetime()
ON MATCH  SET s.updated_at  = datetime()
RETURN s
"""

UPSERT_MEDICATION = """
MERGE (m:Medication {name: $name})
ON CREATE SET m.dosage      = $dosage,
              m.route       = $route,
              m.instruction = $instruction,
              m.created_at  = datetime()
ON MATCH  SET m.updated_at  = datetime()
RETURN m
"""

UPSERT_PROCEDURE = """
MERGE (pr:Procedure {name: $name})
ON CREATE SET pr.type       = $type,
              pr.status     = $status,
              pr.created_at = datetime()
ON MATCH  SET pr.updated_at = datetime()
RETURN pr
"""

UPSERT_VITAL = """
MERGE (v:Vital {vital_id: $vital_id})
ON CREATE SET v.type       = $type,
              v.value      = $value,
              v.unit       = $unit,
              v.created_at = datetime()
ON MATCH  SET v.updated_at = datetime()
RETURN v
"""

UPSERT_RISK_FACTOR = """
MERGE (r:RiskFactor {name: $name})
ON CREATE SET r.source     = $source,
              r.created_at = datetime()
ON MATCH  SET r.updated_at = datetime()
RETURN r
"""

# ── Relationship creation ──────────────────────────────────────────────────────

LINK_PATIENT_CONDITION = """
MATCH (p:Patient {patient_number: $patient_number})
MATCH (c:Condition {name: $condition_name})
MERGE (p)-[r:HAS_CONDITION]->(c)
SET r.confidence = $confidence
"""

LINK_PATIENT_SYMPTOM = """
MATCH (p:Patient {patient_number: $patient_number})
MATCH (s:Symptom {name: $symptom_name})
MERGE (p)-[r:EXHIBITS_SYMPTOM]->(s)
SET r.confidence = $confidence
"""

LINK_PATIENT_MEDICATION = """
MATCH (p:Patient {patient_number: $patient_number})
MATCH (m:Medication {name: $medication_name})
MERGE (p)-[r:TAKES_MEDICATION]->(m)
SET r.adherence_status = $adherence_status,
    r.confidence       = $confidence
"""

LINK_PATIENT_VITAL = """
MATCH (p:Patient {patient_number: $patient_number})
MATCH (v:Vital {vital_id: $vital_id})
MERGE (p)-[r:HAS_VITAL]->(v)
"""

LINK_PATIENT_RISK_FACTOR = """
MATCH (p:Patient {patient_number: $patient_number})
MATCH (r:RiskFactor {name: $risk_name})
MERGE (p)-[rel:HAS_RISK_FACTOR]->(r)
"""

LINK_PATIENT_PROCEDURE = """
MATCH (p:Patient {patient_number: $patient_number})
MATCH (pr:Procedure {name: $procedure_name})
MERGE (p)-[r:SCHEDULED_FOR]->(pr)
"""

LINK_CONDITION_SYMPTOM = """
MATCH (c:Condition {name: $condition_name})
MATCH (s:Symptom {name: $symptom_name})
MERGE (c)-[r:MANIFESTS_AS]->(s)
SET r.confidence = $confidence
"""

LINK_CONDITION_MEDICATION = """
MATCH (c:Condition {name: $condition_name})
MATCH (m:Medication {name: $medication_name})
MERGE (c)-[r:TREATED_WITH]->(m)
SET r.confidence = $confidence
"""

# ── Provenance edge ────────────────────────────────────────────────────────────

LINK_ENTITY_SOURCE = """
MATCH (e {name: $entity_name})
WHERE $entity_label IN labels(e)
MATCH (sc:SourceChunk {chunk_id: $chunk_id})
MERGE (e)-[r:SOURCED_FROM]->(sc)
SET r.confidence        = $confidence,
    r.extraction_method = $extraction_method
"""

# ── Graph statistics ───────────────────────────────────────────────────────────

GRAPH_STATS_NODES_RELS = """
MATCH (n) WITH count(n) AS node_count
OPTIONAL MATCH ()-[r]->()
WITH node_count, count(r) AS rel_count
RETURN node_count, rel_count
"""

GRAPH_STATS_LABELS = """
MATCH (n) UNWIND labels(n) AS lbl
RETURN lbl AS label, count(*) AS cnt ORDER BY cnt DESC
"""
