"""Pydantic models for the entire system: graph entities, API contracts, extraction output, evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════════════

class SOAPSection(str, Enum):
    SUBJECTIVE = "Subjective"
    OBJECTIVE = "Objective"
    ASSESSMENT = "Assessment"
    PLAN = "Plan"
    DEMOGRAPHICS = "Demographics"


class EntityLabel(str, Enum):
    PATIENT = "Patient"
    CONDITION = "Condition"
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    PROCEDURE = "Procedure"
    VITAL = "Vital"
    RISK_FACTOR = "RiskFactor"


class RelationshipType(str, Enum):
    HAS_CONDITION = "HAS_CONDITION"
    EXHIBITS_SYMPTOM = "EXHIBITS_SYMPTOM"
    TAKES_MEDICATION = "TAKES_MEDICATION"
    MANIFESTS_AS = "MANIFESTS_AS"
    TREATED_WITH = "TREATED_WITH"
    HAS_VITAL = "HAS_VITAL"
    HAS_RISK_FACTOR = "HAS_RISK_FACTOR"
    SCHEDULED_FOR = "SCHEDULED_FOR"
    SOURCED_FROM = "SOURCED_FROM"
    BELONGS_TO = "BELONGS_TO"
    NEXT = "NEXT"


class QuestionCategory(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    PROVENANCE = "provenance"
    RELATIONSHIP = "relationship"
    CROSS_REFERENCE = "cross_reference"


# ═══════════════════════════════════════════════════════════════════════════════
#  Ingestion / Extraction Models
# ═══════════════════════════════════════════════════════════════════════════════

class TextChunk(BaseModel):
    """A section-aware text chunk from a document."""
    chunk_id: str
    text: str
    section: SOAPSection
    start_char: int
    end_char: int
    doc_id: str
    embedding: Optional[list[float]] = None


class ExtractedEntity(BaseModel):
    """An entity extracted by the NLP / LLM pipeline."""
    name: str
    label: EntityLabel
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_id: str
    extraction_method: str = "llm"


class ExtractedRelationship(BaseModel):
    """A relationship extracted between two entities."""
    source_name: str
    source_label: EntityLabel
    target_name: str
    target_label: EntityLabel
    relationship_type: RelationshipType
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunk_id: str


class ExtractionResult(BaseModel):
    """Full output of entity/relationship extraction for one chunk."""
    chunk_id: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Retrieval Models
# ═══════════════════════════════════════════════════════════════════════════════

class GraphNode(BaseModel):
    """A node returned from Neo4j."""
    element_id: str
    labels: list[str]
    properties: dict[str, Any]


class GraphRelationship(BaseModel):
    """A relationship returned from Neo4j."""
    element_id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: dict[str, Any]


class Citation(BaseModel):
    """Provenance citation linking an entity to its source text."""
    entity_name: str
    source_text: str
    section: str
    source_file: str
    confidence: float
    extraction_method: str


class ContextBundle(BaseModel):
    """Structured context assembled for LLM prompting."""
    seed_entities: list[GraphNode] = Field(default_factory=list)
    neighborhood_nodes: list[GraphNode] = Field(default_factory=list)
    neighborhood_edges: list[GraphRelationship] = Field(default_factory=list)
    reasoning_paths: list[list[str]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    raw_chunks: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  API Request / Response Models
# ═══════════════════════════════════════════════════════════════════════════════

class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""
    soap_notes_path: str = "Task_Files/soap_notes.txt"
    demographics_path: str = "Task_Files/demographics.json"


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""
    status: str
    nodes_created: int
    relationships_created: int
    chunks_indexed: int
    duration_seconds: float


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    question: str
    strategy: str = "graph"  # "graph" | "vector" | "both"
    max_hops: int = 3
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""
    question: str
    strategy: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    context_bundle: Optional[ContextBundle] = None
    retrieval_time_ms: float
    generation_time_ms: float


class ExploreResponse(BaseModel):
    """Response from the /graph/explore endpoint."""
    center_node: GraphNode
    nodes: list[GraphNode]
    edges: list[GraphRelationship]
    hops: int


class GraphSchemaResponse(BaseModel):
    """Response from the /graph/schema endpoint."""
    node_count: int
    relationship_count: int
    label_counts: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    neo4j_connected: bool
    anthropic_ok: bool
    embedding_model_loaded: bool
    apoc_extended: bool


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation Models
# ═══════════════════════════════════════════════════════════════════════════════

class GoldStandardQA(BaseModel):
    """A test question with its expected answer for evaluation."""
    question: str
    expected_answer: str
    expected_source_sections: list[str]
    category: QuestionCategory


class MetricScores(BaseModel):
    """Evaluation metric scores for a single question."""
    faithfulness: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    answer_correctness: float = Field(ge=0.0, le=1.0)
    citation_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class EvaluationRecord(BaseModel):
    """Evaluation result for a single question under one strategy."""
    question: str
    category: QuestionCategory
    strategy: str
    answer: str
    expected_answer: str
    scores: MetricScores
    retrieval_time_ms: float
    generation_time_ms: float


class EvaluationReport(BaseModel):
    """Full comparison report."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    records: list[EvaluationRecord] = Field(default_factory=list)
    vector_avg_scores: Optional[MetricScores] = None
    graph_avg_scores: Optional[MetricScores] = None
    summary: str = ""
