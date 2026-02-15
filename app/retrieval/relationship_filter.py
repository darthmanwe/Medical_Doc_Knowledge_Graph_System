"""Relationship-constrained expansion.

Expands only along clinically relevant relationship types,
filtering the subgraph to entities connected by specific edge patterns.
This constrains the neighborhood to medically meaningful paths.
"""

from __future__ import annotations

import logging

from app.graph.connection import sync_session
from app.models.schema import GraphNode, GraphRelationship
from app.retrieval.utils import sanitize_properties

logger = logging.getLogger(__name__)

# Clinically relevant traversal patterns
RELATIONSHIP_CONSTRAINED_QUERY = """
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH path = (seed)-[:HAS_CONDITION|TREATED_WITH|MANIFESTS_AS|TAKES_MEDICATION|EXHIBITS_SYMPTOM|HAS_RISK_FACTOR|HAS_VITAL|SCHEDULED_FOR*1..3]-(target)
WHERE target <> seed
  AND any(node IN nodes(path) WHERE
      node:Condition OR node:Medication OR node:Symptom
      OR node:Vital OR node:RiskFactor OR node:Procedure)
WITH DISTINCT target, path,
     [n IN nodes(path) | {
        element_id: elementId(n),
        labels: labels(n),
        name: n.name
     }] AS node_chain,
     [r IN relationships(path) | {
        element_id: elementId(r),
        type: type(r),
        start_node_id: elementId(startNode(r)),
        end_node_id: elementId(endNode(r)),
        properties: properties(r)
     }] AS rel_chain
RETURN
    elementId(target) AS target_id,
    labels(target) AS target_labels,
    properties(target) AS target_props,
    node_chain,
    rel_chain
ORDER BY length(path)
LIMIT 50
"""

# Condition-focused expansion (for symptom → condition → treatment chains)
CONDITION_CHAIN_QUERY = """
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH (seed)-[:HAS_CONDITION|EXHIBITS_SYMPTOM*1..2]-(cond:Condition)
OPTIONAL MATCH (cond)-[:TREATED_WITH]->(med:Medication)
OPTIONAL MATCH (cond)-[:MANIFESTS_AS]->(sym:Symptom)
RETURN DISTINCT
    elementId(cond) AS condition_id,
    cond.name AS condition_name,
    properties(cond) AS condition_props,
    collect(DISTINCT {name: med.name, props: properties(med)}) AS medications,
    collect(DISTINCT {name: sym.name, props: properties(sym)}) AS symptoms
"""


def relationship_constrained_expansion(
    seed_entity_ids: list[str],
) -> tuple[list[GraphNode], list[GraphRelationship]]:
    """Expand along clinically relevant edges only.

    Returns filtered neighborhood nodes and edges.
    """
    if not seed_entity_ids:
        return [], []

    nodes: list[GraphNode] = []
    edges: list[GraphRelationship] = []
    seen_nids: set[str] = set()
    seen_eids: set[str] = set()

    with sync_session() as session:
        result = session.run(
            RELATIONSHIP_CONSTRAINED_QUERY,
            seed_ids=seed_entity_ids,
        )

        for record in result:
            tid = record["target_id"]
            if tid not in seen_nids:
                seen_nids.add(tid)
                nodes.append(GraphNode(
                    element_id=tid,
                    labels=record["target_labels"],
                    properties=sanitize_properties(record["target_props"] or {}),
                ))

            for rel_data in record["rel_chain"]:
                eid = rel_data["element_id"]
                if eid not in seen_eids:
                    seen_eids.add(eid)
                    edges.append(GraphRelationship(
                        element_id=eid,
                        type=rel_data["type"],
                        start_node_id=rel_data["start_node_id"],
                        end_node_id=rel_data["end_node_id"],
                        properties=sanitize_properties(rel_data.get("properties") or {}),
                    ))

    logger.info(
        "Relationship-constrained expansion: %d nodes, %d edges.",
        len(nodes), len(edges),
    )
    return nodes, edges


def get_condition_chains(
    seed_entity_ids: list[str],
) -> list[dict]:
    """Get condition → medication → symptom chains from seeds."""
    if not seed_entity_ids:
        return []

    chains: list[dict] = []
    with sync_session() as session:
        result = session.run(CONDITION_CHAIN_QUERY, seed_ids=seed_entity_ids)
        for record in result:
            chains.append({
                "condition_id": record["condition_id"],
                "condition_name": record["condition_name"],
                "condition_props": record["condition_props"],
                "medications": [m for m in record["medications"] if m.get("name")],
                "symptoms": [s for s in record["symptoms"] if s.get("name")],
            })

    logger.info("Found %d condition chains.", len(chains))
    return chains
