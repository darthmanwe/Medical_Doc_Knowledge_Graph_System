"""Structured context bundle assembler for LLM prompting.

Combines all retrieval signals into a unified ContextBundle:
  - Seed entities (from entity-first retrieval)
  - K-hop neighborhood (expanded subgraph)
  - Relationship-constrained context
  - Path-based reasoning chains
  - Provenance citations back to source text

The bundle is then serialized into a structured prompt section.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.schema import (
    Citation,
    ContextBundle,
    GraphNode,
    GraphRelationship,
)
from app.retrieval.entity_first import entity_first_retrieval
from app.retrieval.k_hop_expansion import expand_k_hop
from app.retrieval.relationship_filter import (
    get_condition_chains,
    relationship_constrained_expansion,
)
from app.retrieval.path_reasoning import find_paths_from_seeds
from app.retrieval.provenance import get_citations_by_ids

logger = logging.getLogger(__name__)


def build_context(
    query: str,
    *,
    top_k: int = 5,
    max_hops: int = 3,
) -> ContextBundle:
    """Assemble a full context bundle for a query using all retrieval patterns.

    Pipeline:
      1. Entity-first retrieval → seed entities + matched chunks
      2. K-hop expansion → neighborhood subgraph
      3. Relationship-constrained filtering → clinically relevant subset
      4. Path-based reasoning → explicit reasoning chains
      5. Provenance linking → source citations
    """
    # Step 1: Entity-first retrieval
    seed_entities, matched_chunks = entity_first_retrieval(
        query, top_k=top_k,
    )

    seed_ids = [e.element_id for e in seed_entities]
    raw_chunks = [c["text"] for c in matched_chunks if c.get("text")]

    if not seed_ids:
        logger.warning("No seed entities found for query: %s", query[:80])
        return ContextBundle(raw_chunks=raw_chunks)

    # Step 2: K-hop expansion
    neighborhood_nodes, neighborhood_edges = expand_k_hop(
        seed_ids, max_hops=max_hops,
    )

    # Step 3: Relationship-constrained expansion
    rel_nodes, rel_edges = relationship_constrained_expansion(seed_ids)

    # Merge relationship-constrained results into neighborhood
    existing_node_ids = {n.element_id for n in neighborhood_nodes}
    for node in rel_nodes:
        if node.element_id not in existing_node_ids:
            neighborhood_nodes.append(node)
            existing_node_ids.add(node.element_id)

    existing_edge_ids = {e.element_id for e in neighborhood_edges}
    for edge in rel_edges:
        if edge.element_id not in existing_edge_ids:
            neighborhood_edges.append(edge)
            existing_edge_ids.add(edge.element_id)

    # Step 4: Path-based reasoning
    reasoning_paths = _extract_reasoning_paths(seed_ids, neighborhood_nodes)

    # Step 5: Provenance citations
    all_entity_ids = seed_ids + [n.element_id for n in neighborhood_nodes]
    # Deduplicate
    all_entity_ids = list(set(all_entity_ids))
    citations = get_citations_by_ids(all_entity_ids)

    bundle = ContextBundle(
        seed_entities=seed_entities,
        neighborhood_nodes=neighborhood_nodes,
        neighborhood_edges=neighborhood_edges,
        reasoning_paths=reasoning_paths,
        citations=citations,
        raw_chunks=raw_chunks,
    )

    logger.info(
        "Context bundle: %d seeds, %d nodes, %d edges, %d paths, %d citations, %d chunks.",
        len(bundle.seed_entities),
        len(bundle.neighborhood_nodes),
        len(bundle.neighborhood_edges),
        len(bundle.reasoning_paths),
        len(bundle.citations),
        len(bundle.raw_chunks),
    )
    return bundle


def _extract_reasoning_paths(
    seed_ids: list[str],
    neighborhood_nodes: list[GraphNode],
) -> list[list[str]]:
    """Build reasoning paths from seeds to interesting targets."""
    paths: list[list[str]] = []

    # Find paths to Condition and Medication nodes in the neighborhood
    interesting_labels = {"Condition", "Medication", "Procedure"}
    targets = [
        n for n in neighborhood_nodes
        if any(lbl in interesting_labels for lbl in n.labels)
        and n.properties.get("name")
    ]

    for target in targets[:5]:  # Limit to avoid excessive queries
        found = find_paths_from_seeds(seed_ids, target.properties["name"])
        for p in found:
            chain = p.get("entity_chain", [])
            rel_chain = p.get("relationship_chain", [])
            if chain:
                readable = []
                for i, entity_name in enumerate(chain):
                    readable.append(str(entity_name))
                    if i < len(rel_chain):
                        readable.append(f"-[{rel_chain[i]}]->")
                paths.append(readable)

    return paths


def format_context_for_prompt(bundle: ContextBundle) -> str:
    """Serialize a ContextBundle into a structured text block for LLM prompting."""
    sections: list[str] = []

    # Source chunks
    if bundle.raw_chunks:
        sections.append("## Source Text Chunks")
        for i, chunk in enumerate(bundle.raw_chunks, 1):
            sections.append(f"[Chunk {i}]: {chunk}")

    # Entities (cap to 10 most relevant seeds to limit prompt size)
    if bundle.seed_entities:
        sections.append("\n## Key Entities")
        for entity in bundle.seed_entities[:10]:
            labels = ", ".join(entity.labels)
            name = entity.properties.get("name", "unknown")
            sections.append(f"- ({labels}) {name}: {_format_props(entity.properties)}")

    # Neighborhood (cap to 15)
    if bundle.neighborhood_nodes:
        sections.append("\n## Connected Entities (Knowledge Graph Neighborhood)")
        for node in bundle.neighborhood_nodes[:15]:
            labels = ", ".join(node.labels)
            name = node.properties.get("name", "unknown")
            sections.append(f"- ({labels}) {name}: {_format_props(node.properties)}")

    # Relationships (cap to 15, use entity names where available)
    if bundle.neighborhood_edges:
        node_id_name = {
            n.element_id: n.properties.get("name", n.element_id[:8])
            for n in (bundle.seed_entities + bundle.neighborhood_nodes)
        }
        sections.append("\n## Relationships")
        for edge in bundle.neighborhood_edges[:15]:
            src = node_id_name.get(edge.start_node_id, edge.start_node_id[:8])
            tgt = node_id_name.get(edge.end_node_id, edge.end_node_id[:8])
            sections.append(f"- {src} -[{edge.type}]-> {tgt}")

    # Reasoning paths
    if bundle.reasoning_paths:
        sections.append("\n## Reasoning Paths (Entity Chains)")
        for path in bundle.reasoning_paths[:10]:
            sections.append(f"  Path: {' '.join(path)}")

    # Citations (cap to 15 highest-confidence for prompt conciseness)
    if bundle.citations:
        sorted_cites = sorted(bundle.citations, key=lambda c: c.confidence, reverse=True)
        sections.append("\n## Source Citations (Provenance)")
        for c in sorted_cites[:15]:
            sections.append(
                f"- Entity '{c.entity_name}' from [{c.section}] in {c.source_file} "
                f"(confidence: {c.confidence:.2f}): \"{c.source_text[:120]}…\""
            )

    return "\n".join(sections)


def _format_props(props: dict) -> str:
    """Format entity properties for prompt display, excluding embeddings."""
    filtered = {
        k: v for k, v in props.items()
        if k not in ("embedding", "created_at", "updated_at") and v
    }
    if not filtered:
        return ""
    return ", ".join(f"{k}={v}" for k, v in filtered.items())
