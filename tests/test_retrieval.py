"""Tests for retrieval module components.

These tests validate query templates and data structures without requiring
a live Neo4j connection (unit-level validation).
"""

import pytest

from app.models.schema import (
    Citation,
    ContextBundle,
    GraphNode,
    GraphRelationship,
)


class TestGraphModels:
    """Validate Pydantic model construction for graph data."""

    def test_graph_node_creation(self):
        node = GraphNode(
            element_id="4:abc:123",
            labels=["Condition"],
            properties={"name": "Hypertension", "status": "active"},
        )
        assert "Condition" in node.labels
        assert node.properties["name"] == "Hypertension"

    def test_graph_relationship_creation(self):
        rel = GraphRelationship(
            element_id="5:abc:456",
            type="HAS_CONDITION",
            start_node_id="4:abc:1",
            end_node_id="4:abc:2",
            properties={"confidence": 0.95},
        )
        assert rel.type == "HAS_CONDITION"

    def test_citation_creation(self):
        cite = Citation(
            entity_name="Hypertension",
            source_text="BP 152/88 borderline control",
            section="Assessment",
            source_file="Task_Files/soap_notes.txt",
            confidence=0.9,
            extraction_method="llm_claude",
        )
        assert cite.confidence == 0.9
        assert cite.section == "Assessment"

    def test_context_bundle_empty(self):
        bundle = ContextBundle()
        assert bundle.seed_entities == []
        assert bundle.citations == []
        assert bundle.raw_chunks == []

    def test_context_bundle_populated(self):
        node = GraphNode(
            element_id="4:abc:1",
            labels=["Patient"],
            properties={"name": "Peter Fern"},
        )
        cite = Citation(
            entity_name="Stable Angina",
            source_text="Likely stable angina given exertional pattern",
            section="Assessment",
            source_file="soap_notes.txt",
            confidence=0.85,
            extraction_method="llm_claude",
        )
        bundle = ContextBundle(
            seed_entities=[node],
            citations=[cite],
            raw_chunks=["some chunk text"],
        )
        assert len(bundle.seed_entities) == 1
        assert len(bundle.citations) == 1


class TestCypherQueries:
    """Validate that our Cypher query templates are well-formed strings."""

    def test_vector_seed_query_is_parameterized(self):
        from app.retrieval.entity_first import VECTOR_SEED_QUERY
        assert "$top_k" in VECTOR_SEED_QUERY
        assert "$query_embedding" in VECTOR_SEED_QUERY
        assert "$threshold" in VECTOR_SEED_QUERY

    def test_khop_query_has_params(self):
        from app.retrieval.k_hop_expansion import CYPHER_KHOP
        assert "$seed_ids" in CYPHER_KHOP
        assert "$max_hops" in CYPHER_KHOP

    def test_relationship_constrained_query(self):
        from app.retrieval.relationship_filter import RELATIONSHIP_CONSTRAINED_QUERY
        assert "HAS_CONDITION" in RELATIONSHIP_CONSTRAINED_QUERY
        assert "TREATED_WITH" in RELATIONSHIP_CONSTRAINED_QUERY
        assert "$seed_ids" in RELATIONSHIP_CONSTRAINED_QUERY

    def test_shortest_path_query(self):
        from app.retrieval.path_reasoning import SHORTEST_PATH_QUERY
        assert "shortestPath" in SHORTEST_PATH_QUERY
        assert "$entity_a_id" in SHORTEST_PATH_QUERY
        assert "$entity_b_id" in SHORTEST_PATH_QUERY

    def test_provenance_query(self):
        from app.retrieval.provenance import PROVENANCE_QUERY
        assert "SOURCED_FROM" in PROVENANCE_QUERY
        assert "BELONGS_TO" in PROVENANCE_QUERY
        assert "$entity_ids" in PROVENANCE_QUERY


class TestContextFormatting:
    """Test the context builder formatting logic."""

    def test_format_empty_bundle(self):
        from app.retrieval.context_builder import format_context_for_prompt
        bundle = ContextBundle()
        result = format_context_for_prompt(bundle)
        assert isinstance(result, str)

    def test_format_with_chunks(self):
        from app.retrieval.context_builder import format_context_for_prompt
        bundle = ContextBundle(raw_chunks=["BP 152/88", "Stable angina"])
        result = format_context_for_prompt(bundle)
        assert "Source Text" in result
        assert "152/88" in result
