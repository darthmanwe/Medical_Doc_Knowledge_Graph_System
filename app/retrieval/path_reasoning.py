"""Path-based reasoning — shortest-path and all-paths queries between entities.

Enables multi-hop reasoning by surfacing the explicit chain of entities
and relationships connecting two concepts in the graph.
"""

from __future__ import annotations

import logging

from app.graph.connection import sync_session

logger = logging.getLogger(__name__)

SHORTEST_PATH_QUERY = """
MATCH (a) WHERE elementId(a) = $entity_a_id
MATCH (b) WHERE elementId(b) = $entity_b_id
WITH a, b WHERE elementId(a) <> elementId(b)
MATCH path = shortestPath((a)-[*..6]-(b))
RETURN
    [n IN nodes(path) | n.name] AS entity_chain,
    [r IN relationships(path) | type(r)] AS relationship_chain,
    length(path) AS hops,
    [n IN nodes(path) | {
        element_id: elementId(n),
        labels: labels(n),
        name: n.name
    }] AS detailed_nodes
"""

ALL_PATHS_QUERY = """
MATCH (a) WHERE elementId(a) = $entity_a_id
MATCH (b) WHERE elementId(b) = $entity_b_id
WITH a, b WHERE elementId(a) <> elementId(b)
MATCH path = (a)-[*..4]-(b)
WITH path, length(path) AS hops
ORDER BY hops
LIMIT $max_paths
RETURN
    [n IN nodes(path) | n.name] AS entity_chain,
    [r IN relationships(path) | type(r)] AS relationship_chain,
    hops
"""

# Find paths from any seed to a named entity
SEED_TO_NAMED_QUERY = """
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH (target) WHERE target.name = $target_name
WITH seed, target WHERE elementId(seed) <> elementId(target)
MATCH path = shortestPath((seed)-[*..5]-(target))
RETURN
    elementId(seed) AS seed_id,
    seed.name AS seed_name,
    [n IN nodes(path) | n.name] AS entity_chain,
    [r IN relationships(path) | type(r)] AS relationship_chain,
    length(path) AS hops
ORDER BY hops
LIMIT 5
"""


def find_shortest_path(
    entity_a_id: str,
    entity_b_id: str,
) -> dict | None:
    """Find the shortest path between two entities by element ID.

    Returns dict with entity_chain, relationship_chain, hops, detailed_nodes
    or None if no path exists.
    """
    if entity_a_id == entity_b_id:
        return None

    with sync_session() as session:
        result = session.run(
            SHORTEST_PATH_QUERY,
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
        )
        record = result.single()

    if record is None:
        logger.info("No path found between %s and %s.", entity_a_id, entity_b_id)
        return None

    path = {
        "entity_chain": record["entity_chain"],
        "relationship_chain": record["relationship_chain"],
        "hops": record["hops"],
        "detailed_nodes": record["detailed_nodes"],
    }
    logger.info(
        "Path found: %s (%d hops)",
        " → ".join(str(n) for n in path["entity_chain"]),
        path["hops"],
    )
    return path


def find_all_paths(
    entity_a_id: str,
    entity_b_id: str,
    max_paths: int = 5,
) -> list[dict]:
    """Find all paths (up to max_paths) between two entities."""
    if entity_a_id == entity_b_id:
        return []

    paths: list[dict] = []
    with sync_session() as session:
        result = session.run(
            ALL_PATHS_QUERY,
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            max_paths=max_paths,
        )
        for record in result:
            paths.append({
                "entity_chain": record["entity_chain"],
                "relationship_chain": record["relationship_chain"],
                "hops": record["hops"],
            })

    logger.info("Found %d paths between entities.", len(paths))
    return paths


def find_paths_from_seeds(
    seed_entity_ids: list[str],
    target_name: str,
) -> list[dict]:
    """Find shortest paths from seed entities to a named target."""
    paths: list[dict] = []
    with sync_session() as session:
        result = session.run(
            SEED_TO_NAMED_QUERY,
            seed_ids=seed_entity_ids,
            target_name=target_name,
        )
        for record in result:
            paths.append({
                "seed_id": record["seed_id"],
                "seed_name": record["seed_name"],
                "entity_chain": record["entity_chain"],
                "relationship_chain": record["relationship_chain"],
                "hops": record["hops"],
            })

    logger.info(
        "Found %d paths from seeds to '%s'.", len(paths), target_name,
    )
    return paths
