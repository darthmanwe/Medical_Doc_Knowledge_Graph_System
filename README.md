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

## Ablation Study

Each optimization was applied incrementally and evaluated independently to isolate its contribution to overall graph RAG performance. The table below reports the aggregate metric after each stage was added on top of the previous ones.

| Stage | Change | Faithfulness | Ctx Precision | Ctx Recall | Correctness | Gen Latency |
|-------|--------|-------------|---------------|------------|-------------|-------------|
| **Baseline** | 5-stage retrieval, no filtering | 0.94 | 0.49 | 0.91 | 0.75 | 12.8s |
| **+ Prompt capping** | Cap seeds to 10, nodes to 15, citations to 15 | 0.94 | 0.49 | 0.91 | 0.75 | 7.0s |
| **+ Chunk deduplication** | Deduplicate raw chunks by chunk_id | 0.95 | 0.58 | 0.86 | 0.78 | 5.5s |
| **+ Adaptive depth** | 1-hop for simple queries, 2-hop for complex | 0.95 | 0.65 | 0.86 | 0.79 | 4.2s |
| **+ Semantic re-ranking** | Cosine similarity scoring and threshold pruning | **0.96** | **0.78** | 0.86 | **0.80** | **3.5s** |

Key observations from the ablation:

- **Chunk deduplication** was the single highest-impact fix for context precision (+9 points). The entity-first retrieval pattern returns one row per (chunk, entity) pair through the SOURCED_FROM fan-out, which inflated the context with duplicate text that the LLM-as-judge correctly scored as redundant.
- **Adaptive depth** improved precision by another 7 points by routing simple factual queries through a 1-hop expansion instead of the full 3-hop traversal, which prevented tangential entities from entering the context.
- **Semantic re-ranking** provided the final 13-point precision lift. Embedding each context element against the query and pruning below threshold removed graph neighbors that were topologically close but semantically irrelevant.
- **Context recall dropped by 5 points** (0.91 to 0.86) as a consequence of more aggressive pruning. This is the expected precision-recall trade-off, and the 0.86 recall remains within 5% of the vector baseline.

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
| **all-MiniLM-L6-v2 (384-dim)** | Selected over domain-specific alternatives after considering latency, cost, and context constraints (see model selection rationale below) |
| **ChromaDB for vector baseline** | Serves as an isolated vector store to ensure a fair comparison where vector RAG has zero graph context |
| **Dual-write embeddings** | Indexes chunks in both Neo4j and ChromaDB at ingestion time so that both strategies operate on identical embeddings |
| **Per-label Cypher MERGE** | Works around the Cypher limitation that dynamic-label MERGE is illegal by using `apoc.merge.node()` when APOC is available |
| **Exponential backoff on API calls** | Handles Anthropic rate limits gracefully using the tenacity retry library |
| **Bessel-corrected std dev in evaluation** | Reports sample standard deviation (ddof=1) alongside means to quantify variance across the small (n=11) question set |
| **70/30 LLM-embedding blend for correctness** | Combines LLM-as-judge semantic assessment with embedding cosine similarity to reduce single-evaluator bias on answer correctness scoring |

### Embedding Model Selection Rationale

The choice of `all-MiniLM-L6-v2` over domain-specific medical embedding models (BioBERT, ClinicalBERT, PubMedBERT) was deliberate and based on four factors:

1. **Retrieval task profile.** The system retrieves short clinical text chunks (average 300 characters) against natural language queries. General-purpose sentence embeddings perform well on short-text semantic similarity, which is the primary retrieval signal. Domain-specific models like BioBERT are optimized for biomedical named entity recognition and relation extraction at the token level, not sentence-level retrieval.

2. **Inference latency constraint.** MiniLM-L6 produces a 384-dimensional embedding in approximately 5ms on CPU. ClinicalBERT (768-dim, 12 layers) runs at approximately 25ms per embedding on the same hardware. Since the semantic re-ranking layer embeds 20 to 40 context elements per query, this difference compounds to 100ms versus 500ms of additional retrieval latency.

3. **Normalized embedding quality.** With L2-normalized embeddings and cosine similarity, MiniLM-L6 achieves competitive performance on STS benchmarks (Spearman rho = 0.82 on STS-B) while using half the dimensions of BERT-base models. The vector baseline achieved 0.99 faithfulness and 0.90 context precision, confirming that embedding quality is not the bottleneck.

4. **Fair comparison design.** Both retrieval strategies share the same embedding model. Using a general-purpose model avoids introducing a confound where one strategy might benefit disproportionately from domain-specific embeddings.

For a production deployment in a clinical setting with specialized terminology (radiology reports, pathology, genomics), a domain-adapted model would likely improve recall on rare medical terms. This is documented as a future improvement.

---

## Limitations and Future Work

This section documents known limitations and areas where additional engineering effort would strengthen the system. Senior ML practitioners should read these as scoped claims rather than unqualified results.

### Statistical Power

The evaluation harness uses 11 gold-standard questions. While the question set is stratified across 5 categories to cover different reasoning patterns, this sample size limits statistical confidence. The evaluation now reports standard deviation alongside mean scores, but formal significance testing (paired bootstrap, Wilcoxon signed-rank) would require a larger question set (n >= 30) to produce reliable p-values.

### Evaluation Methodology

The LLM-as-judge approach introduces evaluator bias that is correlated with the generator model (both are Claude Sonnet). An ideal evaluation would use a separate evaluator model or include human annotation as ground truth. The citation accuracy metric uses heuristic string overlap rather than ground-truth citation labels, which underestimates accuracy for paraphrased citations.

### Embedding Model Coverage

The system uses a single general-purpose embedding model. A production system serving clinical NLP workloads would benefit from a model comparison study across domain-specific alternatives (PubMedBERT, BioLORD, ClinicalBERT) evaluated on retrieval-specific metrics such as MRR@K and NDCG@K against a labeled relevance corpus.

### Hyperparameter Sensitivity

Key thresholds (re-ranking cosine threshold of 0.25, entity resolution fuzzy threshold of 88, semantic similarity threshold of 0.85, and chunking parameters of 600-character max with 80-character overlap) were empirically tuned on the development set. No systematic hyperparameter search (grid, Bayesian, or sensitivity analysis) was performed. These thresholds may not generalize to other document types or medical specialties without re-tuning.

### Scalability Validation

All benchmarks were measured on a single-patient graph with 48 nodes. Retrieval latency and context precision behavior at 10K to 1M node scale has not been empirically validated. The architectural mitigations (adaptive depth, relationship-constrained traversal, and re-ranking) are designed to scale, but profiling under load is necessary before production deployment.

### Future Improvements

| Area | Improvement | Expected Impact |
|------|-------------|-----------------|
| **Retrieval metrics** | Add MRR@K and NDCG@K using labeled relevance judgments | More granular retrieval quality measurement |
| **Domain embeddings** | Compare PubMedBERT and BioLORD against MiniLM-L6 on medical retrieval | Better recall on specialized terminology |
| **Hybrid search** | Combine sparse (BM25) and dense (embedding) retrieval with reciprocal rank fusion | Improved recall for exact-match medical codes |
| **Confidence calibration** | Calibrate extraction confidence scores against human labels | Reliable provenance confidence thresholds |
| **Experiment tracking** | Integrate MLflow or Weights and Biases for hyperparameter and metric versioning | Reproducible experiment comparison |
| **Streaming generation** | Use Anthropic streaming API to reduce perceived latency from 3.5s to under 1s | Better interactive user experience |

---

## License

MIT
