# Neo4j Knowledge Graph for Grounded RAG

### Medical Document Understanding via Property Graph Retrieval, Provenance-Linked Citations, and Explainable AI

A complete Knowledge Graph-backed RAG system that ingests medical documents into a Neo4j property graph with full provenance traceability, implements five expert Cypher retrieval patterns, and evaluates graph-backed RAG against vector-only RAG on factual grounding and explainability.

Aim is to create a proveable comparison for Graph-back RAG versus Vector-only RAG in a traceable manner.

**Built with:** Python · Neo4j 5.x · FastAPI · Anthropic Claude · sentence-transformers · ChromaDB

---

## Highlights

| Metric | Value |
|--------|-------|
| **Faithfulness (both strategies)** | **>0.96** — <4% of generated claims are unsupported by retrieved context |
| **Graph Context Precision** | **0.78** — doubled from 0.49 via semantic re-ranking + adaptive retrieval depth |
| **Context Recall** | **0.86-0.91** — both strategies capture >86% of ground-truth facts |
| **Graph Generation Latency** | **~3.5s** — down from 12.8s (73% faster) after re-ranking + prompt optimization |
| **Provenance Citations** | **8-10 per graph query** — focused via semantic re-ranking (was 34 pre-optimization) |
| **Knowledge Graph** | **48 nodes, 72 relationships** across 9 entity types with full citation chains |
| **Test Coverage** | **29/29 tests passed** — ingestion, retrieval (5 Cypher patterns), and evaluation |
| **Evaluation Harness** | **11 gold-standard questions**, 5 categories, 5 RAGAS-style metrics, LLM-as-judge |

---

## Skills Demonstrated

### Graph RAG (Neo4j + Cypher)
- Designed a **medical property graph schema** (Patient, Condition, Symptom, Medication, Procedure, Vital, RiskFactor) with typed relationships and provenance edges
- Wrote **5 composable Cypher retrieval patterns**: entity-first vector seeding, k-hop neighborhood expansion (APOC + pure Cypher fallback), relationship-constrained traversal, shortestPath reasoning chains, and provenance linking
- Built **structured context bundles** that give the LLM entities, relationships, reasoning paths, and source citations — not just raw text chunks
- Implemented **SOURCED_FROM provenance** with confidence scores and extraction method tracking, enabling full audit trails from answer → entity → source chunk → document

### Vector RAG (ChromaDB + Embeddings)
- Built a **pure vector baseline** using ChromaDB with cosine similarity (all-MiniLM-L6-v2, 384-dim) for fair comparison
- Implemented **dual-write embeddings** at ingestion — chunks indexed in both Neo4j's native vector index and ChromaDB
- Achieved **0.99 faithfulness and 0.90 context precision** on vector-only retrieval — strong baseline that validates embedding quality

### NLP & Ingestion
- **LLM-based entity extraction** using Claude's tool_use API with schema-constrained JSON output (7 entity types, 11 relationship types)
- **Two-pass entity resolution**: rapidfuzz fuzzy matching (88% threshold) + embedding-based semantic similarity (0.85 cosine threshold)
- **Section-aware SOAP chunking** with assessment sub-splitting, character offsets, and document provenance

### Evaluation & Measurement
- **RAGAS-style evaluation harness** with LLM-as-judge scoring (faithfulness, context precision, context recall, answer correctness, citation accuracy)
- **Fair comparison design**: same embeddings model, same LLM, same prompt structure — only the retrieval strategy differs
- Identified the **precision-vs-explainability trade-off** and closed 70% of the gap: graph RAG context precision improved from 0.49 to 0.78 (vs vector's 0.90) via semantic re-ranking, adaptive retrieval depth, and chunk deduplication

---

## Evaluation Results (Real Run — 11 Questions)

### Aggregate Scores

| Metric | Vector RAG | Graph RAG | Insight |
|--------|------------|-----------|---------|
| **Faithfulness** | 0.99 | **0.96** | Both excellent — <4% unsupported claims |
| **Context Precision** | 0.90 | **0.78** | Gap narrowed from 0.41 to 0.12 after re-ranking |
| **Context Recall** | 0.91 | 0.86 | Both capture >86% of ground-truth facts |
| **Answer Correctness** | 0.83 | **0.80** | Near-parity (3% gap, was 8%) |
| **Citation Accuracy** | n/a | **0.57** | Graph-only: improved from 0.41 via focused provenance |

### Per-Category Performance

| Category | Vector Faithfulness | Graph Faithfulness | Vector Correctness | Graph Correctness |
|----------|--------------------|--------------------|--------------------|--------------------|
| Single-Hop (3 Qs) | 1.00 | 0.90 | 0.94 | 0.90 |
| Multi-Hop (2 Qs) | 0.95 | **1.00** | 0.87 | **0.87** |
| Relationship (2 Qs) | 1.00 | 0.95 | 0.85 | 0.77 |
| Cross-Reference (2 Qs) | 1.00 | 1.00 | 0.92 | 0.90 |
| Provenance (2 Qs) | 1.00 | 1.00 | 0.55 | 0.54 |

### The Trade-off

**Graph RAG now matches or beats vector on multi-hop reasoning** (1.00 faithfulness and context precision for multi-hop queries). The remaining gap is on simple single-hop and cross-reference queries where graph expansion adds little value — as expected.

Graph RAG's differentiator is explainability: entity chains, reasoning paths, and provenance citations that let an analyst *prove* an answer, not just read one. After optimization (semantic re-ranking, adaptive depth, chunk deduplication), the precision trade-off is now just 12% — while retaining full provenance traceability.

Full results: **[outputs/results.md](outputs/results.md)** · Evaluation data: **[outputs/evaluation_report.json](outputs/evaluation_report.json)**

---

## Architecture

```
Medical Documents (SOAP notes + Demographics JSON)
        │
        ▼
┌─────────────────────────────────────────────┐
│            INGESTION PIPELINE                │
│  Section-Aware Chunker → Claude Extraction   │
│  Entity Resolution (fuzzy + semantic)        │
│  Graph Writer + Dual Embedding Index         │
└────────┬──────────────────────┬──────────────┘
         │                      │
   Neo4j Property          ChromaDB Vector
   Graph (48 nodes,        Store (8 chunks,
    72 relationships)       384-dim embeddings)
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌──────────────────┐
│   Graph RAG     │    │   Vector RAG     │
│   5-stage       │    │   Single-stage   │
│   Cypher        │    │   cosine search  │
│   retrieval     │    │   (baseline)     │
└────────┬────────┘    └────────┬─────────┘
         │                      │
         ▼                      ▼
    Claude Sonnet (answer generation + citations)
         │
         ▼
┌─────────────────────────────────────────────┐
│          EVALUATION HARNESS                  │
│  11 gold-standard questions · 5 categories   │
│  LLM-as-judge · RAGAS-style metrics          │
│  Side-by-side vector vs graph comparison      │
└──────────────────────────────────────────────┘
```

---

## Retrieval Patterns (Expert Cypher)

| # | Pattern | What It Does | Cypher Technique |
|---|---------|--------------|------------------|
| 1 | **Entity-First Retrieval** | Embed query → vector search → follow SOURCED_FROM edges to seed entities | `db.index.vector.queryNodes` + pattern match |
| 2 | **K-Hop Expansion** | Expand seeds 1-3 hops to collect subgraph neighborhood | `[*1..$max_hops]` or `apoc.neighbors.byhop` |
| 3 | **Relationship-Constrained** | Traverse only clinically relevant edge types | `[:HAS_CONDITION\|TREATED_WITH\|MANIFESTS_AS*1..3]` |
| 4 | **Path-Based Reasoning** | Find shortest paths between entities for multi-hop chains | `shortestPath((a)-[*..6]-(b))` with same-node guard |
| 5 | **Provenance Linking** | Trace every entity back to source text with confidence | `(e)-[:SOURCED_FROM]->(chunk)-[:BELONGS_TO]->(doc)` |

These are assembled into a **structured context bundle** containing: seed entities, neighborhood nodes, relationship edges, reasoning paths, provenance citations, and raw source chunks — all serialized into a single LLM prompt.

---

## Graph Schema

```
(:Patient)-[:HAS_CONDITION]->(:Condition)-[:TREATED_WITH]->(:Medication)
(:Patient)-[:EXHIBITS_SYMPTOM]->(:Symptom)
(:Condition)-[:MANIFESTS_AS]->(:Symptom)
(:Patient)-[:HAS_VITAL]->(:Vital)
(:Patient)-[:HAS_RISK_FACTOR]->(:RiskFactor)
(:Patient)-[:SCHEDULED_FOR]->(:Procedure)
(:Entity)-[:SOURCED_FROM {confidence, method}]->(:SourceChunk)-[:BELONGS_TO]->(:Document)
(:SourceChunk)-[:NEXT]->(:SourceChunk)
```

**9 node labels** · **11 relationship types** · **Provenance on every extracted entity**

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/Medical_Doc_Knowledge_Graph_System.git
cd Medical_Doc_Knowledge_Graph_System
pip install -r requirements.txt

# 2. Configure
cp .env.example .env   # Add your ANTHROPIC_API_KEY and NEO4J_PASSWORD

# 3. Start Neo4j
docker-compose up -d   # Wait ~30s for Neo4j to be ready

# 4. Run tests
pytest tests/ -v       # 29 passed

# 5. Start the app
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 6. Ingest documents
curl -X POST http://localhost:8000/ingest

# 7. Query (graph-backed RAG)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What conditions explain the exertional symptoms?", "strategy": "graph"}'

# 8. Query (vector-only baseline)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the patient blood pressure?", "strategy": "vector"}'

# 9. Run evaluation harness
curl -X POST http://localhost:8000/evaluate

# 10. Explore the graph
curl http://localhost:8000/graph/explore/Hypertension?hops=2
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Ingest documents into the knowledge graph |
| `POST` | `/query` | Ask questions (`strategy`: graph / vector / both) |
| `GET` | `/graph/explore/{name}` | Explore k-hop subgraph around an entity |
| `GET` | `/graph/schema` | Graph statistics (node/relationship counts) |
| `POST` | `/evaluate` | Run side-by-side evaluation harness |
| `GET` | `/health` | System health (Neo4j, Anthropic, embeddings, APOC) |

---

## Project Structure

```
Medical_Doc_Knowledge_Graph_System/
├── app/
│   ├── main.py                  # FastAPI entrypoint + lifespan
│   ├── config.py                # Pydantic settings from .env
│   ├── models/schema.py         # All Pydantic models
│   ├── graph/
│   │   ├── connection.py        # Neo4j driver management
│   │   ├── schema_setup.py      # Constraint/index DDL
│   │   └── queries.py           # Reusable Cypher templates
│   ├── ingestion/
│   │   ├── chunker.py           # SOAP section-aware text splitter
│   │   ├── extractor.py         # Claude tool_use entity extraction
│   │   ├── entity_resolver.py   # Fuzzy + semantic dedup
│   │   ├── graph_writer.py      # Neo4j upsert + ChromaDB dual-write
│   │   └── pipeline.py          # Orchestrator
│   ├── retrieval/
│   │   ├── entity_first.py      # Vector seed → graph entities
│   │   ├── k_hop_expansion.py   # Neighborhood expansion (APOC + Cypher)
│   │   ├── relationship_filter.py # Clinically-constrained traversal
│   │   ├── path_reasoning.py    # Shortest-path reasoning chains
│   │   ├── provenance.py        # Source-of-truth citations
│   │   ├── context_builder.py   # Structured context bundle assembler
│   │   └── utils.py             # Neo4j type sanitization
│   ├── rag/
│   │   ├── embeddings.py        # sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── llm_client.py        # Anthropic Claude wrapper + retry
│   │   ├── vector_rag.py        # ChromaDB-only baseline
│   │   └── graph_rag.py         # Full graph-backed RAG
│   └── evaluation/
│       ├── questions.py          # 11 gold-standard questions (5 categories)
│       ├── metrics.py            # RAGAS-style LLM-as-judge metrics
│       ├── harness.py            # Side-by-side evaluation runner
│       └── report.py             # Markdown report generator
├── tests/                        # 29 unit tests
├── outputs/                      # Results, evaluation data, charts
│   ├── results.md                # Full results write-up
│   ├── evaluation_report.json    # Complete evaluation (22 records)
│   └── test_results.txt          # Pytest log
├── Task_Files/                   # Input documents (SOAP notes + demographics)
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── study_packet.md               # Interview prep (15 sections + glossary)
└── README.md
```

---

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Neo4j 5.26 native vector index** | Eliminates need for external vector DB in graph RAG path; single query language (Cypher) for both vector and graph retrieval |
| **APOC Extended auto-detection** | APOC `neighbors.byhop` for k-hop when available; pure Cypher variable-length path fallback ensures portability |
| **Claude Sonnet (tool_use API)** | Schema-constrained structured extraction — no regex parsing, no hallucinated entity types |
| **all-MiniLM-L6-v2 (384-dim)** | CPU-only local embeddings; zero API cost, ~5ms per query |
| **ChromaDB for vector baseline** | Isolated vector store ensures fair comparison — vector RAG has zero graph context |
| **Dual-write embeddings** | Chunks indexed in both Neo4j and ChromaDB at ingestion time; both strategies use the same embeddings |
| **Per-label Cypher MERGE** | Dynamic-label MERGE is illegal in Cypher; `apoc.merge.node()` used when APOC available |
| **Exponential backoff on API calls** | Handles Anthropic rate limits gracefully with tenacity retry |

---

## License

MIT
