# Neo4j Knowledge Graph for Grounded RAG

### Medical Document Understanding via Property Graph Retrieval, Provenance-Linked Citations, and Explainable AI

This system ingests medical documents into a Neo4j property graph with full provenance traceability, implements five composable Cypher retrieval patterns, and evaluates graph-backed RAG against vector-only RAG on factual grounding and explainability. The goal is to produce a provable, side-by-side comparison of graph-backed retrieval versus vector-only retrieval in a fully traceable manner.

**Built with:** Python, Neo4j 5.x, FastAPI, Anthropic Claude, sentence-transformers, ChromaDB

---

## Highlights

| Metric | Value |
|--------|-------|
| **Faithfulness (both strategies)** | **>0.96** with fewer than 4% of generated claims unsupported by retrieved context |
| **Graph Context Precision** | **0.78**, doubled from 0.49 through semantic re-ranking and adaptive retrieval depth |
| **Context Recall** | **0.86 to 0.91** across both strategies, capturing over 86% of ground-truth facts |
| **Graph Generation Latency** | **~3.5s**, reduced from 12.8s (73% faster) after re-ranking and prompt optimization |
| **Provenance Citations** | **8 to 10 per graph query**, focused through semantic re-ranking (previously 34 before optimization) |
| **Knowledge Graph** | **48 nodes and 72 relationships** across 9 entity types with full citation chains |
| **Test Coverage** | **29 of 29 tests passed** covering ingestion, retrieval (5 Cypher patterns), and evaluation |
| **Evaluation Harness** | **11 gold-standard questions** across 5 categories, scored with 5 RAGAS-style metrics using LLM-as-judge |

---

## Skills Demonstrated

### Graph RAG (Neo4j + Cypher)
- Designed a medical property graph schema with seven node types (Patient, Condition, Symptom, Medication, Procedure, Vital, RiskFactor), typed relationships, and provenance edges that form complete citation chains
- Wrote five composable Cypher retrieval patterns: entity-first vector seeding, k-hop neighborhood expansion with APOC and a pure Cypher fallback, relationship-constrained traversal, shortestPath reasoning chains, and provenance linking
- Built structured context bundles that provide the LLM with entities, relationships, reasoning paths, and source citations rather than raw text chunks alone
- Implemented SOURCED_FROM provenance edges carrying confidence scores and extraction method metadata, enabling full audit trails from generated answer to extracted entity to source chunk to original document

### Vector RAG (ChromaDB + Embeddings)
- Built a pure vector baseline using ChromaDB with cosine similarity over all-MiniLM-L6-v2 embeddings (384 dimensions) to serve as a fair comparison against the graph-backed approach
- Implemented dual-write embeddings at ingestion time so that chunks are indexed in both Neo4j's native vector index and ChromaDB simultaneously
- Achieved 0.99 faithfulness and 0.90 context precision on vector-only retrieval, establishing a strong baseline that validates the quality of the underlying embeddings

### NLP and Ingestion
- Performed LLM-based entity extraction using Claude's tool_use API with schema-constrained JSON output covering 7 entity types and 11 relationship types
- Applied two-pass entity resolution combining rapidfuzz fuzzy string matching at an 88% threshold with embedding-based semantic similarity at a 0.85 cosine threshold to merge duplicate and variant entity references
- Implemented section-aware SOAP chunking with assessment sub-splitting, character offset tracking, and document provenance metadata

### Evaluation and Measurement
- Built a RAGAS-style evaluation harness using LLM-as-judge scoring across five metrics: faithfulness, context precision, context recall, answer correctness, and citation accuracy
- Ensured a fair comparison design where both strategies share the same embeddings model, the same LLM, and the same prompt structure, with only the retrieval strategy differing between runs
- Identified the precision-versus-explainability trade-off and closed 70% of the gap by improving graph RAG context precision from 0.49 to 0.78 (compared to vector's 0.90) through semantic re-ranking, adaptive retrieval depth, and chunk deduplication

---

## Evaluation Results (Real Run, 11 Questions)

### Aggregate Scores

| Metric | Vector RAG | Graph RAG | Insight |
|--------|------------|-----------|---------|
| **Faithfulness** | 0.99 | **0.96** | Both strategies score above 0.96, with fewer than 4% unsupported claims |
| **Context Precision** | 0.90 | **0.78** | The gap narrowed from 0.41 to 0.12 after re-ranking was introduced |
| **Context Recall** | 0.91 | 0.86 | Both strategies capture over 86% of ground-truth facts |
| **Answer Correctness** | 0.83 | **0.80** | Near parity at a 3% gap, down from 8% before optimization |
| **Citation Accuracy** | n/a | **0.57** | Graph-only metric that improved from 0.41 through focused provenance filtering |

### Per-Category Performance

| Category | Vector Faithfulness | Graph Faithfulness | Vector Correctness | Graph Correctness |
|----------|--------------------|--------------------|--------------------|--------------------|
| Single-Hop (3 Qs) | 1.00 | 0.90 | 0.94 | 0.90 |
| Multi-Hop (2 Qs) | 0.95 | **1.00** | 0.87 | **0.87** |
| Relationship (2 Qs) | 1.00 | 0.95 | 0.85 | 0.77 |
| Cross-Reference (2 Qs) | 1.00 | 1.00 | 0.92 | 0.90 |
| Provenance (2 Qs) | 1.00 | 1.00 | 0.55 | 0.54 |

### The Trade-off

Graph RAG now matches or beats vector on multi-hop reasoning, scoring 1.00 for both faithfulness and context precision on multi-hop queries. The remaining gap appears on simple single-hop and cross-reference queries where graph expansion adds little value, which is the expected behavior.

The core differentiator of graph RAG is explainability. Entity chains, reasoning paths, and provenance citations allow an analyst to prove an answer rather than simply read one. After optimization through semantic re-ranking, adaptive depth, and chunk deduplication, the precision trade-off has narrowed to just 12% while retaining full provenance traceability.

Full results: **[outputs/results.md](outputs/results.md)** | Evaluation data: **[outputs/evaluation_report.json](outputs/evaluation_report.json)**

---

## Architecture

```
Medical Documents (SOAP notes + Demographics JSON)
        |
        v
+---------------------------------------------+
|            INGESTION PIPELINE                |
|  Section-Aware Chunker -> Claude Extraction  |
|  Entity Resolution (fuzzy + semantic)        |
|  Graph Writer + Dual Embedding Index         |
+--------+----------------------+--------------+
         |                      |
   Neo4j Property          ChromaDB Vector
   Graph (48 nodes,        Store (8 chunks,
    72 relationships)       384-dim embeddings)
         |                      |
         v                      v
+-----------------+    +------------------+
|   Graph RAG     |    |   Vector RAG     |
|   5-stage       |    |   Single-stage   |
|   Cypher        |    |   cosine search  |
|   retrieval     |    |   (baseline)     |
+--------+--------+    +--------+---------+
         |                      |
         v                      v
    Claude Sonnet (answer generation + citations)
         |
         v
+---------------------------------------------+
|          EVALUATION HARNESS                  |
|  11 gold-standard questions, 5 categories    |
|  LLM-as-judge with RAGAS-style metrics       |
|  Side-by-side vector vs graph comparison     |
+----------------------------------------------+
```

---

## Retrieval Patterns (Expert Cypher)

| # | Pattern | What It Does | Cypher Technique |
|---|---------|--------------|------------------|
| 1 | **Entity-First Retrieval** | Embeds the query, runs vector search, and follows SOURCED_FROM edges to discover seed entities | `db.index.vector.queryNodes` combined with pattern matching |
| 2 | **K-Hop Expansion** | Expands seed entities by 1 to 3 hops to collect the surrounding subgraph neighborhood | `[*1..$max_hops]` variable-length paths or `apoc.neighbors.byhop` |
| 3 | **Relationship-Constrained** | Traverses only clinically relevant edge types to filter noise from the expansion | `[:HAS_CONDITION\|TREATED_WITH\|MANIFESTS_AS*1..3]` |
| 4 | **Path-Based Reasoning** | Finds shortest paths between entities to surface multi-hop reasoning chains | `shortestPath((a)-[*..6]-(b))` with a same-node guard |
| 5 | **Provenance Linking** | Traces every entity back to its source text with an associated confidence score | `(e)-[:SOURCED_FROM]->(chunk)-[:BELONGS_TO]->(doc)` |

These five patterns are assembled into a structured context bundle containing seed entities, neighborhood nodes, relationship edges, reasoning paths, provenance citations, and raw source chunks. The bundle is then serialized into a single LLM prompt.

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

The schema uses 9 node labels and 11 relationship types. Every extracted entity carries provenance metadata linking it back to its source text.

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
| `POST` | `/ingest` | Ingests documents into the knowledge graph |
| `POST` | `/query` | Accepts a question with a strategy parameter (graph, vector, or both) |
| `GET` | `/graph/explore/{name}` | Explores the k-hop subgraph around a named entity |
| `GET` | `/graph/schema` | Returns graph statistics including node and relationship counts |
| `POST` | `/evaluate` | Runs the side-by-side evaluation harness |
| `GET` | `/health` | Reports system health for Neo4j, Anthropic, embeddings, and APOC |

---

## Project Structure

```
Medical_Doc_Knowledge_Graph_System/
├── app/
│   ├── main.py                  # FastAPI entrypoint and lifespan management
│   ├── config.py                # Pydantic settings loaded from .env
│   ├── models/schema.py         # All Pydantic models
│   ├── graph/
│   │   ├── connection.py        # Neo4j driver management
│   │   ├── schema_setup.py      # Constraint and index DDL
│   │   └── queries.py           # Reusable Cypher templates
│   ├── ingestion/
│   │   ├── chunker.py           # SOAP section-aware text splitter
│   │   ├── extractor.py         # Claude tool_use entity extraction
│   │   ├── entity_resolver.py   # Fuzzy and semantic deduplication
│   │   ├── graph_writer.py      # Neo4j upsert with ChromaDB dual-write
│   │   └── pipeline.py          # Ingestion orchestrator
│   ├── retrieval/
│   │   ├── entity_first.py      # Vector seed to graph entity mapping
│   │   ├── k_hop_expansion.py   # Neighborhood expansion via APOC and Cypher
│   │   ├── relationship_filter.py # Clinically-constrained traversal
│   │   ├── path_reasoning.py    # Shortest-path reasoning chains
│   │   ├── provenance.py        # Source-of-truth citation linking
│   │   ├── context_builder.py   # Structured context bundle assembler with re-ranking
│   │   └── utils.py             # Neo4j type sanitization utilities
│   ├── rag/
│   │   ├── embeddings.py        # sentence-transformers (all-MiniLM-L6-v2)
│   │   ├── llm_client.py        # Anthropic Claude wrapper with retry logic
│   │   ├── vector_rag.py        # ChromaDB-only baseline
│   │   └── graph_rag.py         # Full graph-backed RAG pipeline
│   └── evaluation/
│       ├── questions.py          # 11 gold-standard questions across 5 categories
│       ├── metrics.py            # RAGAS-style LLM-as-judge metrics
│       ├── harness.py            # Side-by-side evaluation runner
│       └── report.py             # Markdown report generator
├── tests/                        # 29 unit tests
├── outputs/                      # Results, evaluation data, and charts
│   ├── results.md                # Full results write-up
│   ├── evaluation_report.json    # Complete evaluation with 22 records
│   └── test_results.txt          # Pytest log
├── Task_Files/                   # Input documents (SOAP notes and demographics)
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Neo4j 5.26 native vector index** | Eliminates the need for an external vector database in the graph RAG path, allowing a single query language (Cypher) for both vector and graph retrieval |
| **APOC Extended auto-detection** | Uses APOC `neighbors.byhop` for k-hop expansion when available, with a pure Cypher variable-length path fallback to ensure portability across environments |
| **Claude Sonnet (tool_use API)** | Provides schema-constrained structured extraction without regex parsing and without risk of hallucinated entity types |
| **all-MiniLM-L6-v2 (384-dim)** | Runs entirely on CPU with zero API cost and approximately 5ms latency per query embedding |
| **ChromaDB for vector baseline** | Serves as an isolated vector store to ensure a fair comparison where vector RAG has zero graph context |
| **Dual-write embeddings** | Indexes chunks in both Neo4j and ChromaDB at ingestion time so that both strategies operate on identical embeddings |
| **Per-label Cypher MERGE** | Works around the Cypher limitation that dynamic-label MERGE is illegal by using `apoc.merge.node()` when APOC is available |
| **Exponential backoff on API calls** | Handles Anthropic rate limits gracefully using the tenacity retry library |

---

## License

MIT
