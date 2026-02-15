"""FastAPI service — entrypoint for the Medical Document Knowledge Graph System.

Lifespan startup:
  1. Apply Neo4j schema (constraints + indexes) via sync driver
  2. Warm-load embedding model
  3. Verify Anthropic API key
  4. Detect APOC Extended availability
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.graph.connection import (
    close_drivers,
    detect_apoc_extended,
    get_sync_driver,
    has_apoc_extended,
)
from app.graph.schema_setup import apply_schema
from app.models.schema import (
    EvaluationReport,
    ExploreResponse,
    GraphNode,
    GraphSchemaResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.rag.embeddings import warm_load as warm_embeddings
from app.rag.llm_client import verify_api_key

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── State flags ────────────────────────────────────────────────────────────────
_state = {
    "neo4j_ok": False,
    "anthropic_ok": False,
    "embedding_ok": False,
    "apoc_extended": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("=== Starting Medical KG RAG System ===")

    # 1. Neo4j schema
    try:
        get_sync_driver()
        apply_schema()
        _state["neo4j_ok"] = True
        logger.info("Neo4j schema applied successfully.")
    except Exception as exc:
        logger.error("Neo4j startup failed: %s", exc)

    # 2. Embedding model
    try:
        warm_embeddings()
        _state["embedding_ok"] = True
    except Exception as exc:
        logger.error("Embedding model load failed: %s", exc)

    # 3. Anthropic API
    _state["anthropic_ok"] = verify_api_key()

    # 4. APOC detection
    if _state["neo4j_ok"]:
        _state["apoc_extended"] = detect_apoc_extended()

    logger.info("Startup state: %s", _state)

    yield

    # Shutdown
    await close_drivers()
    logger.info("=== System shutdown complete ===")


app = FastAPI(
    title="Medical Document Knowledge Graph System",
    description=(
        "Neo4j Knowledge Graph-backed RAG system for medical document understanding. "
        "Implements entity-first retrieval, k-hop expansion, relationship-constrained "
        "filtering, path-based reasoning, and provenance linking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check including Neo4j, Anthropic, and embedding model status."""
    return HealthResponse(
        status="healthy" if all(_state.values()) else "degraded",
        neo4j_connected=_state["neo4j_ok"],
        anthropic_ok=_state["anthropic_ok"],
        embedding_model_loaded=_state["embedding_ok"],
        apoc_extended=_state["apoc_extended"],
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest medical documents (SOAP notes + demographics) into the knowledge graph."""
    from app.ingestion.pipeline import run_ingestion

    try:
        result = run_ingestion(
            soap_path=request.soap_notes_path,
            demographics_path=request.demographics_path,
        )
        return result
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(request: QueryRequest):
    """Ask a question — returns answer with citations and retrieval metadata."""
    from app.rag.graph_rag import graph_rag_query
    from app.rag.vector_rag import vector_rag_query

    try:
        if request.strategy == "vector":
            return vector_rag_query(request.question, top_k=request.top_k)
        elif request.strategy == "graph":
            return graph_rag_query(
                request.question,
                top_k=request.top_k,
                max_hops=request.max_hops,
            )
        elif request.strategy == "both":
            v_resp = vector_rag_query(request.question, top_k=request.top_k)
            g_resp = graph_rag_query(
                request.question,
                top_k=request.top_k,
                max_hops=request.max_hops,
            )
            # Return graph response but include both in metadata
            g_resp.answer = (
                f"## Graph-Backed Answer\n{g_resp.answer}\n\n"
                f"## Vector-Only Answer (Baseline)\n{v_resp.answer}"
            )
            return g_resp
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/explore/{entity_name}", response_model=ExploreResponse)
async def explore_entity(entity_name: str, hops: int = 2):
    """Return k-hop subgraph around a named entity."""
    from app.graph.connection import sync_session
    from app.retrieval.k_hop_expansion import expand_k_hop

    try:
        with sync_session() as session:
            result = session.run(
                "MATCH (n) WHERE n.name = $name RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props LIMIT 1",
                name=entity_name,
            )
            record = result.single()
            if record is None:
                raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found.")

            center = GraphNode(
                element_id=record["eid"],
                labels=record["labels"],
                properties={k: v for k, v in record["props"].items() if k != "embedding"},
            )

        nodes, edges = expand_k_hop([center.element_id], max_hops=hops)

        return ExploreResponse(
            center_node=center,
            nodes=nodes,
            edges=edges,
            hops=hops,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Explore failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/graph/schema", response_model=GraphSchemaResponse)
async def get_graph_schema():
    """Return current graph schema statistics."""
    from app.graph.connection import sync_session
    from app.graph.queries import GRAPH_STATS_LABELS, GRAPH_STATS_NODES_RELS

    try:
        with sync_session() as session:
            nodes_rels = session.run(GRAPH_STATS_NODES_RELS)
            nr_record = nodes_rels.single()
            labels_result = session.run(GRAPH_STATS_LABELS)
            label_counts = [{"label": r["label"], "count": r["cnt"]} for r in labels_result]

        return GraphSchemaResponse(
            node_count=nr_record["node_count"] if nr_record else 0,
            relationship_count=nr_record["rel_count"] if nr_record else 0,
            label_counts=label_counts,
        )
    except Exception as exc:
        logger.error("Schema query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate", response_model=EvaluationReport)
async def run_evaluation_endpoint():
    """Run the full evaluation harness comparing vector-only vs graph-backed RAG."""
    from app.evaluation.harness import run_evaluation

    try:
        report = run_evaluation()
        return report
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
