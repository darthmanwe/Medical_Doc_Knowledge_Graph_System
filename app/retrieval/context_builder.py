"""Structured context bundle assembler for LLM prompting.

Combines all retrieval signals into a unified ContextBundle:
  - Seed entities (from entity-first retrieval)
  - K-hop neighborhood (expanded subgraph)
  - Relationship-constrained context
  - Path-based reasoning chains
  - Provenance citations back to source text

Includes relevance re-ranking to prune context elements that are not
semantically related to the query, improving context precision.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
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

# ── Query complexity keywords ──────────────────────────────────────────────────

_COMPLEX_KEYWORDS = {
    "how", "why", "relate", "relationship", "explain", "connection",
    "between", "compare", "conditions", "medications", "treatment",
    "chain", "path", "multiple", "risk factors",
}


def classify_query_complexity(query: str) -> str:
    """Classify a query as 'simple' or 'complex' using keyword heuristics.

    Simple: single-entity factual lookups (blood pressure, age, name).
    Complex: multi-hop reasoning, relationship, or explanation questions.
    """
    lower = query.lower()
    # Check for complex keywords
    if any(kw in lower for kw in _COMPLEX_KEYWORDS):
        return "complex"
    # Check for question words that imply reasoning
    if lower.startswith(("how ", "why ")):
        return "complex"
    return "simple"


def build_context(
    query: str,
    *,
    top_k: int = 5,
    max_hops: int = 3,
) -> ContextBundle:
    """Assemble a full context bundle for a query using all retrieval patterns.

    Pipeline:
      1. Entity-first retrieval → seed entities + matched chunks
      2. K-hop expansion → neighborhood subgraph (adaptive depth)
      3. Relationship-constrained filtering → clinically relevant subset
      4. Path-based reasoning → explicit reasoning chains
      5. Provenance linking → source citations
    """
    # Adaptive retrieval depth based on query complexity
    complexity = classify_query_complexity(query)
    if complexity == "simple":
        effective_hops = 1
        effective_top_k = min(top_k, 3)
    else:
        effective_hops = min(max_hops, 2)
        effective_top_k = top_k

    logger.info("Query complexity: %s → hops=%d, top_k=%d", complexity, effective_hops, effective_top_k)

    # Step 1: Entity-first retrieval
    seed_entities, matched_chunks = entity_first_retrieval(
        query, top_k=effective_top_k,
    )

    seed_ids = [e.element_id for e in seed_entities]
    # Deduplicate chunks by chunk_id (each chunk can appear multiple times
    # due to multiple SOURCED_FROM edges), preserving order by relevance score.
    seen_chunk_ids: set[str | None] = set()
    raw_chunks: list[str] = []
    for c in matched_chunks:
        cid = c.get("chunk_id")
        if c.get("text") and cid not in seen_chunk_ids:
            raw_chunks.append(c["text"])
            seen_chunk_ids.add(cid)

    if not seed_ids:
        logger.warning("No seed entities found for query: %s", query[:80])
        return ContextBundle(raw_chunks=raw_chunks)

    # Step 2: K-hop expansion (adaptive depth)
    neighborhood_nodes, neighborhood_edges = expand_k_hop(
        seed_ids, max_hops=effective_hops,
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

    # Step 4: Path-based reasoning (only for complex queries)
    if complexity == "complex":
        reasoning_paths = _extract_reasoning_paths(seed_ids, neighborhood_nodes)
    else:
        reasoning_paths = []

    # Step 5: Provenance citations
    all_entity_ids = seed_ids + [n.element_id for n in neighborhood_nodes]
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
        "Context bundle (pre-rerank): %d seeds, %d nodes, %d edges, %d paths, %d citations, %d chunks.",
        len(bundle.seed_entities),
        len(bundle.neighborhood_nodes),
        len(bundle.neighborhood_edges),
        len(bundle.reasoning_paths),
        len(bundle.citations),
        len(bundle.raw_chunks),
    )
    return bundle


# ── Relevance re-ranking ──────────────────────────────────────────────────────


def rerank_context_bundle(
    query: str,
    bundle: ContextBundle,
    *,
    threshold: float | None = None,
    max_seeds: int = 8,
    max_nodes: int = 10,
    max_citations: int = 10,
    max_paths: int = 5,
) -> ContextBundle:
    """Re-rank all context elements by semantic relevance to the query.

    Scores each element using cosine similarity between the query embedding
    and the element's text representation. Prunes elements below the threshold.

    Automatically tightens caps for simple queries to boost context precision.
    """
    from app.rag.embeddings import embed_text, embed_batch, cosine_similarity

    threshold = threshold or settings.rerank_threshold
    query_emb = embed_text(query)

    # Tighter caps for simple queries to maximize context precision
    complexity = classify_query_complexity(query)
    if complexity == "simple":
        max_seeds = min(max_seeds, 3)
        max_nodes = min(max_nodes, 3)
        max_citations = min(max_citations, 3)
        max_paths = 0
        threshold = max(threshold, 0.35)  # higher bar for simple queries
    else:
        # Complex queries need more graph context for multi-hop reasoning
        max_seeds = min(max_seeds, 6)
        max_nodes = min(max_nodes, 8)
        max_citations = min(max_citations, 8)

    # ── Score and filter seed entities ──────────────────────────────────────
    scored_seeds = []
    if bundle.seed_entities:
        seed_texts = [_node_text(e) for e in bundle.seed_entities]
        seed_embs = embed_batch(seed_texts)
        for entity, emb in zip(bundle.seed_entities, seed_embs):
            score = cosine_similarity(query_emb, emb)
            if score >= threshold:
                scored_seeds.append((score, entity))
        scored_seeds.sort(key=lambda x: x[0], reverse=True)

    filtered_seeds = [e for _, e in scored_seeds[:max_seeds]]

    # ── Score and filter neighborhood nodes ─────────────────────────────────
    scored_nodes = []
    if bundle.neighborhood_nodes:
        node_texts = [_node_text(n) for n in bundle.neighborhood_nodes]
        node_embs = embed_batch(node_texts)
        for node, emb in zip(bundle.neighborhood_nodes, node_embs):
            score = cosine_similarity(query_emb, emb)
            if score >= threshold:
                scored_nodes.append((score, node))
        scored_nodes.sort(key=lambda x: x[0], reverse=True)

    filtered_nodes = [n for _, n in scored_nodes[:max_nodes]]

    # ── Filter edges to only those connecting kept nodes ────────────────────
    kept_ids = {e.element_id for e in filtered_seeds} | {n.element_id for n in filtered_nodes}
    filtered_edges = [
        edge for edge in bundle.neighborhood_edges
        if edge.start_node_id in kept_ids or edge.end_node_id in kept_ids
    ][:max_nodes]  # cap edges too

    # ── Score and filter citations ──────────────────────────────────────────
    scored_citations = []
    if bundle.citations:
        cite_texts = [f"{c.entity_name} {c.source_text[:100]}" for c in bundle.citations]
        cite_embs = embed_batch(cite_texts)
        for citation, emb in zip(bundle.citations, cite_embs):
            score = cosine_similarity(query_emb, emb)
            if score >= threshold:
                scored_citations.append((score, citation))
        scored_citations.sort(key=lambda x: x[0], reverse=True)

    filtered_citations = [c for _, c in scored_citations[:max_citations]]

    # ── Filter reasoning paths to those containing relevant entity names ────
    relevant_names = {
        e.properties.get("name", "").lower()
        for e in filtered_seeds + filtered_nodes
        if e.properties.get("name")
    }
    filtered_paths = []
    for path in bundle.reasoning_paths:
        path_names = {s.lower() for s in path if not s.startswith("-[")}
        if path_names & relevant_names:
            filtered_paths.append(path)
    filtered_paths = filtered_paths[:max_paths]

    # Raw chunks come from vector search and are already relevance-ranked.
    # Keep them all — they are the primary evidence for the answer.
    reranked = ContextBundle(
        seed_entities=filtered_seeds,
        neighborhood_nodes=filtered_nodes,
        neighborhood_edges=filtered_edges,
        reasoning_paths=filtered_paths,
        citations=filtered_citations,
        raw_chunks=bundle.raw_chunks,
    )

    logger.info(
        "Context bundle (post-rerank): %d seeds, %d nodes, %d edges, %d paths, %d citations.",
        len(reranked.seed_entities),
        len(reranked.neighborhood_nodes),
        len(reranked.neighborhood_edges),
        len(reranked.reasoning_paths),
        len(reranked.citations),
    )
    return reranked


def _node_text(node: GraphNode) -> str:
    """Build a text representation of a graph node for embedding."""
    name = node.properties.get("name", "")
    labels = " ".join(node.labels)
    # Include key property values (skip internal ones)
    extras = []
    for k, v in node.properties.items():
        if k not in ("name", "embedding", "created_at", "updated_at", "element_id") and v:
            extras.append(str(v))
    extra_str = " ".join(extras[:3])  # limit to 3 extra properties
    return f"{labels} {name} {extra_str}".strip()


# ── Reasoning paths ───────────────────────────────────────────────────────────


def _extract_reasoning_paths(
    seed_ids: list[str],
    neighborhood_nodes: list[GraphNode],
) -> list[list[str]]:
    """Build reasoning paths from seeds to interesting targets."""
    paths: list[list[str]] = []

    interesting_labels = {"Condition", "Medication", "Procedure"}
    targets = [
        n for n in neighborhood_nodes
        if any(lbl in interesting_labels for lbl in n.labels)
        and n.properties.get("name")
    ]

    for target in targets[:5]:
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


# ── Prompt formatting ─────────────────────────────────────────────────────────


def format_context_for_prompt(bundle: ContextBundle) -> str:
    """Serialize a ContextBundle into a focused text block for LLM prompting.

    Three sections to minimize redundancy:
      1. Source Text — raw evidence chunks
      2. Graph Context — entities + relationships (merged, no duplication)
      3. Provenance — top citations for audit trail
    """
    sections: list[str] = []

    # Section 1: Source text chunks (the evidence)
    if bundle.raw_chunks:
        sections.append("## Source Text")
        for i, chunk in enumerate(bundle.raw_chunks, 1):
            sections.append(f"[Chunk {i}]: {chunk}")

    # Section 2: Graph context (merged entities + relationships)
    all_entities = bundle.seed_entities + bundle.neighborhood_nodes
    # Deduplicate by element_id
    seen = set()
    unique_entities = []
    for e in all_entities:
        if e.element_id not in seen:
            seen.add(e.element_id)
            unique_entities.append(e)

    if unique_entities:
        sections.append("\n## Graph Context")
        for entity in unique_entities[:15]:
            labels = ", ".join(entity.labels)
            name = entity.properties.get("name", "unknown")
            sections.append(f"- ({labels}) {name}: {_format_props(entity.properties)}")

        # Add relationships inline
        if bundle.neighborhood_edges:
            node_id_name = {
                n.element_id: n.properties.get("name", n.element_id[:8])
                for n in unique_entities
            }
            for edge in bundle.neighborhood_edges[:10]:
                src = node_id_name.get(edge.start_node_id, edge.start_node_id[:8])
                tgt = node_id_name.get(edge.end_node_id, edge.end_node_id[:8])
                sections.append(f"  {src} -[{edge.type}]-> {tgt}")

    # Add reasoning paths (if any)
    if bundle.reasoning_paths:
        sections.append("\n## Reasoning Paths")
        for path in bundle.reasoning_paths[:5]:
            sections.append(f"  {' '.join(path)}")

    # Section 3: Provenance citations
    if bundle.citations:
        sections.append("\n## Provenance")
        for c in bundle.citations[:10]:
            sections.append(
                f"- '{c.entity_name}' from [{c.section}] in {c.source_file} "
                f"(confidence: {c.confidence:.2f}): \"{c.source_text[:100]}\""
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
