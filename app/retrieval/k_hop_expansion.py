"""K-hop neighborhood expansion.

Primary: Pure Cypher variable-length path (works without APOC Extended).
Enhanced: APOC Extended apoc.neighbors.byhop (auto-detected at startup).
"""

from __future__ import annotations

import logging

from app.config import settings
from app.graph.connection import has_apoc_extended, sync_session
from app.models.schema import GraphNode, GraphRelationship
from app.retrieval.utils import sanitize_properties

logger = logging.getLogger(__name__)

# ── Pure Cypher (always works) ─────────────────────────────────────────────────

CYPHER_KHOP = """
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH path = (seed)-[*1..$max_hops]-(neighbor)
WITH seed, neighbor, path
WHERE elementId(neighbor) <> elementId(seed)
WITH seed, neighbor, path,
     length(path) AS hops,
     [n IN nodes(path) | {
        element_id: elementId(n),
        labels: labels(n),
        properties: properties(n)
     }] AS path_nodes,
     [r IN relationships(path) | {
        element_id: elementId(r),
        type: type(r),
        start_node_id: elementId(startNode(r)),
        end_node_id: elementId(endNode(r)),
        properties: properties(r)
     }] AS path_rels
RETURN DISTINCT
    elementId(neighbor) AS neighbor_id,
    labels(neighbor) AS neighbor_labels,
    properties(neighbor) AS neighbor_props,
    hops,
    path_nodes,
    path_rels
ORDER BY hops
LIMIT 100
"""

# ── APOC Extended version ─────────────────────────────────────────────────────

APOC_KHOP = """
MATCH (seed) WHERE elementId(seed) IN $seed_ids
CALL apoc.neighbors.byhop(seed, '', $max_hops) YIELD nodes
UNWIND range(0, size(nodes)-1) AS hop_idx
UNWIND nodes[hop_idx] AS neighbor
WITH seed, neighbor, hop_idx
WHERE elementId(neighbor) <> elementId(seed)
RETURN DISTINCT
    elementId(neighbor) AS neighbor_id,
    labels(neighbor) AS neighbor_labels,
    properties(neighbor) AS neighbor_props,
    hop_idx + 1 AS hops
ORDER BY hops
LIMIT 100
"""


def expand_k_hop(
    seed_entity_ids: list[str],
    max_hops: int | None = None,
) -> tuple[list[GraphNode], list[GraphRelationship]]:
    """Expand k hops from seed entities, returning neighborhood nodes and edges.

    Auto-selects APOC Extended or pure Cypher based on startup detection.
    """
    max_hops = max_hops or settings.retrieval_max_hops
    if not seed_entity_ids:
        return [], []

    use_apoc = has_apoc_extended()
    query = APOC_KHOP if use_apoc else CYPHER_KHOP
    logger.debug(
        "K-hop expansion: %d seeds, %d hops, strategy=%s",
        len(seed_entity_ids), max_hops, "apoc" if use_apoc else "cypher",
    )

    nodes: list[GraphNode] = []
    edges: list[GraphRelationship] = []
    seen_node_ids: set[str] = set()
    seen_edge_ids: set[str] = set()

    with sync_session() as session:
        result = session.run(
            query,
            seed_ids=seed_entity_ids,
            max_hops=max_hops,
        )

        for record in result:
            nid = record["neighbor_id"]
            if nid not in seen_node_ids:
                seen_node_ids.add(nid)
                nodes.append(GraphNode(
                    element_id=nid,
                    labels=record["neighbor_labels"],
                    properties=sanitize_properties(record["neighbor_props"]),
                ))

            # Extract edges from path (Cypher version only)
            if not use_apoc and "path_rels" in record.keys():
                for rel_data in record["path_rels"]:
                    eid = rel_data["element_id"]
                    if eid not in seen_edge_ids:
                        seen_edge_ids.add(eid)
                        edges.append(GraphRelationship(
                            element_id=eid,
                            type=rel_data["type"],
                            start_node_id=rel_data["start_node_id"],
                            end_node_id=rel_data["end_node_id"],
                            properties=sanitize_properties(rel_data.get("properties", {})),
                        ))

    logger.info(
        "K-hop expansion: %d neighbor nodes, %d edges discovered.",
        len(nodes), len(edges),
    )
    return nodes, edges


