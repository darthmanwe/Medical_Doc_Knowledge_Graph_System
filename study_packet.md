# Knowledge Graph RAG System — Interview Study Guide

**Prepared for: data2.ai (Data Squared USA Inc.) interview**
**Role focus: Neo4j + Knowledge Graphs + Explainable AI (eXAI)**

---

## Table of Contents

1. [data2.ai Company Profile & What They Value](#1-data2ai-company-profile)
2. [Neo4j Fundamentals](#2-neo4j-fundamentals)
3. [Cypher Query Language Deep-Dive](#3-cypher-deep-dive)
4. [Property Graph Model vs Other Graph Models](#4-property-graph-model)
5. [Knowledge Graph Construction Patterns](#5-knowledge-graph-construction)
6. [Entity Extraction & Resolution](#6-entity-extraction--resolution)
7. [GraphRAG vs Traditional RAG](#7-graphrag-vs-traditional-rag)
8. [Retrieval Patterns (with Cypher Examples)](#8-retrieval-patterns)
9. [Provenance & Explainability (eXAI)](#9-provenance--explainability)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [This Project — Architecture Walkthrough](#11-this-project-walkthrough)
12. [Real-World Debugging & Fixes](#12-real-world-debugging--fixes)
13. [Trade-offs & Latency Analysis](#13-trade-offs--latency-analysis)
14. [Scaling to Production](#14-scaling-to-production)
15. [Interview Q&A Prep](#15-interview-qa-prep)

---

## 1. data2.ai Company Profile

### What They Are
- **Service-Disabled Veteran-Owned Small Business (SDVOSB)**
- Pioneering **Explainable AI (eXAI)** analytics
- Flagship product: **reView** — a patented, hallucination-resistant AI platform
- Built on **Neo4j AuraDB** knowledge graphs
- Clients: Defense/intelligence agencies, oil & gas, energy, financial services

**Source:** [Neo4j Customer Story — Data²](https://neo4j.com/customer-stories/data%C2%B2/)

### Core Technology Stack (from public sources)
- **Neo4j AuraDB** — graph database backbone
- **Knowledge Graphs** — property graph model for connected data intelligence
- **GraphRAG** — graph-grounded retrieval-augmented generation
- **LLMs** (Anthropic, others) — for question answering + entity extraction
- **Cloud-agnostic deployment** — on-prem, air-gapped, or cloud

### Their Architecture Pattern (Connect → Enrich → Reason → Visualize)

**Source:** [data2.ai Product Page](https://www.data2.ai/product)

1. **Connect** — Graph Database, Relationship Discovery, Multi-Format Ingestion, Platform-Agnostic Integration, Zero-Trust Security
2. **Enrich** — NLP (they call it "National Language Processing"), Evolving Algorithms, AI-Entity Extraction, Sub-Second Queries
3. **Reason** — Explainable Results, Multi-Agent Intelligence, Advanced Reasoning, Hallucination-Resistant, Persona-Based Access
4. **Visualize** — Interactive Dashboards, Graph Explorer, Chain-of-Custody, Real-Time Analytics

### Key data2.ai Concepts to Reference in Interview

- **Evidence Graphs vs Question Graphs** — data2 maintains separate graphs to prevent LLMs from ingesting incorrect data
- **Arctic Loader** — their custom component for loading tabular data + documents into knowledge graphs
- **LLM-powered fact extractor** — extracts entities from unstructured sources (drilling reports, maintenance logs, etc.)
- **Subgraph-level access control** — Neo4j enables role-based security at the graph level
- **"The knowledge graph is a dynamic, evolving brain"** — quote from their CRO Jeff Dalgliesh

### Their Stated Metrics
- **99.99% accuracy** with graph-based AI verification
- **95% faster time-to-insight** (weeks → hours)
- **70% lower analytics costs**
- **50% analyst workload freed up** by reView
- **$40M+ in customer savings**

### Why They Chose Neo4j Over Alternatives

From their CRO Jeff Dalgliesh:
> "We discovered that with Neo4j, we didn't need to worry about all that mess behind the scenes. Graph technology helped us focus on building our AI capabilities without getting bogged down in database administration and scaling."

> "Triple stores tend to be more focused on ontological purity than solving real-world problems. They're academic, with a rigid insistence on precision. But when you're trying to rapidly make sense of a messy domain, you need a flexible model that helps you solve business problems quickly."

---

## 2. Neo4j Fundamentals

### What Neo4j Is
- **Native graph database** — stores data as nodes and relationships (not tables or documents)
- **ACID-compliant** — full transactional guarantees
- **Index-free adjacency** — each node physically points to its neighbors, making traversals O(1) per hop regardless of graph size (unlike JOINs which degrade with scale)

### Key Concepts

| Concept | Definition |
|---------|-----------|
| **Node** | An entity (person, concept, thing). Has labels and properties. |
| **Relationship** | A directed, typed connection between two nodes. Has properties. |
| **Label** | A tag on a node that groups it (e.g., `:Patient`, `:Condition`). Nodes can have multiple labels. |
| **Property** | A key-value pair stored on nodes or relationships. |
| **Property Graph Model** | The data model Neo4j uses — nodes, relationships, labels, properties. |

### Neo4j vs Relational Databases

| Aspect | Relational DB | Neo4j |
|--------|--------------|-------|
| Data model | Tables, rows, columns | Nodes, relationships, properties |
| Relationships | Implicit via foreign keys + JOINs | First-class, explicit, named, directed |
| Schema | Fixed upfront | Flexible, evolve as needed |
| Traversal cost | O(log n) per JOIN | O(1) per hop (index-free adjacency) |
| Multi-hop queries | Expensive self-joins | Natural pattern matching |

### Neo4j Editions
- **Community** — free, single-instance
- **Enterprise** — clustering, security, performance features
- **AuraDB** — fully managed cloud service (what data2 uses)

### APOC (Awesome Procedures on Cypher)
A library of 450+ procedures and functions extending Cypher:
- **APOC Core** — bundled with Neo4j, basic utilities
- **APOC Extended** — additional procedures like `apoc.neighbors.byhop` for k-hop traversal

### GDS (Graph Data Science)
Neo4j's library for graph algorithms:
- **Community Detection** — Louvain, Label Propagation
- **Centrality** — PageRank, Betweenness, Degree
- **Similarity** — Node Similarity, Jaccard
- **Path Finding** — Dijkstra, A*, Yen's K-Shortest
- **Embeddings** — Node2Vec, FastRP, GraphSAGE

---

## 3. Cypher Deep-Dive

### Core Pattern Matching

```cypher
-- Find all conditions a patient has
MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition)
WHERE p.name = "Peter Fern"
RETURN c.name, c.status

-- Find conditions treated with a specific medication
MATCH (c:Condition)-[:TREATED_WITH]->(m:Medication)
WHERE m.name = "Nitroglycerin"
RETURN c.name, m.dosage

-- Multi-hop: patient → condition → medication chain
MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition)-[:TREATED_WITH]->(m:Medication)
RETURN p.name, c.name AS condition, m.name AS medication
```

### Variable-Length Paths

```cypher
-- Find all nodes within 3 hops of a patient
MATCH path = (p:Patient {name: "Peter Fern"})-[*1..3]-(connected)
RETURN DISTINCT connected, length(path) AS distance

-- Find shortest path between two entities
MATCH path = shortestPath(
    (a:Condition {name: "Stable Angina"})-[*..6]-(b:Medication {name: "Nitroglycerin"})
)
RETURN [n IN nodes(path) | n.name] AS chain
```

### MERGE (Idempotent Upsert)

```cypher
-- Create or update — never duplicates
MERGE (c:Condition {name: "Hypertension"})
ON CREATE SET c.status = "active", c.created_at = datetime()
ON MATCH SET c.updated_at = datetime()
RETURN c
```

**CRITICAL PITFALL:** Cypher does NOT support dynamic labels in MERGE:
```cypher
-- ILLEGAL: MERGE (e:{variable_label} {name: $name})
-- USE INSTEAD: apoc.merge.node() or per-label queries from application code
CALL apoc.merge.node(["Condition"], {name: "Hypertension"}, {status: "active"})
YIELD node RETURN node
```

### Vector Index Queries (Neo4j 5.11+)

```cypher
-- Create vector index
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:SourceChunk) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}

-- Query by vector similarity
CALL db.index.vector.queryNodes('chunk_embedding', 5, $query_embedding)
YIELD node, score
WHERE score > 0.35
RETURN node.text, score
```

### Full-Text Search

```cypher
CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS
FOR (c:SourceChunk) ON EACH [c.text]

-- Query
CALL db.index.fulltext.queryNodes('chunk_text', 'chest pain angina')
YIELD node, score
RETURN node.text, score
```

### Aggregation & Collection

```cypher
-- Collect all medications per condition
MATCH (c:Condition)-[:TREATED_WITH]->(m:Medication)
RETURN c.name, collect(m.name) AS medications

-- Count relationships by type
MATCH ()-[r]->()
RETURN type(r) AS rel_type, count(r) AS count
ORDER BY count DESC
```

### List Comprehensions & UNWIND

```cypher
-- Transform node properties inline
MATCH path = (p:Patient)-[*1..2]-(n)
RETURN [node IN nodes(path) | {name: node.name, labels: labels(node)}] AS chain

-- Batch insert with UNWIND
UNWIND $entities AS entity
MERGE (e:Condition {name: entity.name})
SET e += entity.properties
```

---

## 4. Property Graph Model

### Property Graph vs RDF Triple Store

| Aspect | Property Graph (Neo4j) | RDF Triple Store |
|--------|----------------------|------------------|
| Model | Nodes + relationships with properties | Subject-predicate-object triples |
| Schema | Flexible, evolve over time | Ontology-driven, rigid |
| Properties on edges | Native support | Must be reified (workaround) |
| Query language | Cypher | SPARQL |
| Performance | Optimized for traversals | Optimized for inference |
| Real-world fit | Pragmatic, solves business problems fast | Academic, ontological purity |

**data2 explicitly chose Neo4j over triple stores** because:
> "Triple stores tend to be more focused on ontological purity than solving real-world problems." — Jeff Dalgliesh, data2 CRO

### When to Use a Knowledge Graph

- **Connected data** — entities with complex, multi-hop relationships
- **Evolving schema** — domain models that change over time
- **Reasoning** — need to traverse relationships to derive insights
- **Provenance tracking** — trace facts back to sources
- **Recommendation** — similarity and influence patterns
- **Fraud detection** — hidden patterns in transaction networks

---

## 5. Knowledge Graph Construction

### Pipeline (what we built in this project)

```
Raw Text → Chunk → Extract Entities → Resolve Entities → Write to Graph
                                                        → Embed Chunks
                                                        → Link Provenance
```

### Chunking Strategy
- **Section-aware** — parse document structure (SOAP: Subjective/Objective/Assessment/Plan)
- **Sliding window** — overlap between chunks preserves context at boundaries
- **Sub-section splitting** — numbered lists in Assessment get separate chunks
- Each chunk retains: section label, character offsets, parent document reference

### Entity Extraction Approaches

| Method | Pros | Cons |
|--------|------|------|
| **LLM-based** (what we use) | High accuracy, understands context, handles abbreviations | Expensive per token, latency |
| **spaCy / scispaCy NER** | Fast, free, good for standard entities | Misses domain-specific concepts, no relationships |
| **Regex / rule-based** | Deterministic, no model needed | Brittle, doesn't generalize |
| **Hybrid** (best practice) | LLM primary + NER secondary for validation | More complex pipeline |

### Schema-Guided Extraction
The extraction prompt constrains the LLM to only emit entity types and relationship types defined in our schema. This prevents hallucinated entity types.

### Entity Resolution (Deduplication)
Medical text is full of abbreviation variants:
- HTN = Hypertension = High Blood Pressure
- MI = Myocardial Infarction = Heart Attack
- GERD = Gastroesophageal Reflux Disease = Acid Reflux

**Two-pass resolution:**
1. **Fuzzy string matching** (rapidfuzz, Levenshtein distance) — catches typos and minor variants
2. **Embedding-based semantic matching** — catches synonyms that are spelled differently

---

## 6. Entity Extraction & Resolution

### Anthropic Claude Tool Use for Structured Extraction

Claude's `tool_use` API forces structured JSON output — no regex parsing needed:

```python
tools = [{
    "name": "extract_medical_entities",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "properties": {
                        "name": {"type": "string"},
                        "label": {"enum": ["Condition", "Symptom", ...]},
                        "confidence": {"type": "number"},
                    }
                }
            },
            "relationships": { ... }
        }
    }
}]
```

This is much more reliable than asking the LLM to output JSON in freeform text.

### Medical Abbreviation Expansion
Critical for knowledge graphs — the extraction prompt explicitly instructs:
- HTN → Hypertension
- BP → Blood Pressure  
- MI → Myocardial Infarction
- SOB → Shortness of Breath
- etc.

Without this, the graph would have duplicate nodes like "HTN" and "Hypertension".

---

## 7. GraphRAG vs Traditional RAG

### Traditional RAG
```
Query → Embed → Vector Search (top-k chunks) → Concatenate → LLM → Answer
```
- Simple, well-understood
- Works for single-hop factual questions
- **Fails at:** multi-hop reasoning, relationship queries, provenance

### GraphRAG
```
Query → Embed → Vector Seed → Entity-First Retrieval → K-Hop Expansion
      → Relationship Filtering → Path Reasoning → Provenance Linking
      → Structured Context Bundle → LLM → Answer with Citations
```
- Leverages graph structure for relationship-aware context
- Handles multi-hop reasoning naturally
- **Provenance built in** — every fact traces to source

### When GraphRAG Wins (from research: arxiv.org/abs/2502.11371)
- **Multi-hop questions** — "What medications treat the conditions causing the patient's symptoms?"
- **Relationship queries** — "How does medication adherence relate to blood pressure control?"
- **Provenance questions** — "Which section of the note mentions family history?"
- **Cross-entity reasoning** — connecting information spread across different document sections

### When Vector RAG is Sufficient
- **Single-hop factual** — "What is the patient's blood pressure?"
- **Direct text retrieval** — "What does the Assessment section say?"
- **Simple lookups** — the answer is in one chunk

### data2.ai's Approach
Their reView platform uses **separate Evidence and Question graphs:**
- **Evidence Graph** — stores extracted facts, entities, relationships from source data
- **Question Graph** — stores user questions and AI-generated answers (stored back into the graph)
- This separation prevents the LLM from ingesting its own previous (potentially incorrect) answers as facts

---

## 8. Retrieval Patterns

### Pattern 1: Entity-First Retrieval
**The entry point for all graph-backed retrieval.**

```cypher
CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $query_embedding)
YIELD node AS chunk, score
WHERE score >= $threshold
MATCH (entity)-[:SOURCED_FROM]->(chunk)
RETURN entity, chunk, score
ORDER BY score DESC
```

**How it works:** Embed the query → find similar chunks via vector index → follow SOURCED_FROM edges backward to discover which entities were extracted from those chunks. These become "seed entities" for further expansion.

### Pattern 2: K-Hop Neighborhood Expansion
**Collects the subgraph context around seed entities.**

```cypher
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH path = (seed)-[*1..3]-(neighbor)
WHERE neighbor <> seed
RETURN DISTINCT neighbor, length(path) AS hops
ORDER BY hops
```

**How it works:** From seed entities, traverse all relationships up to K hops. This surfaces related entities that may not appear in the original text chunks but are connected in the graph.

### Pattern 3: Relationship-Constrained Expansion
**Filters to clinically meaningful paths only.**

```cypher
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH path = (seed)-[:HAS_CONDITION|TREATED_WITH|MANIFESTS_AS*1..3]-(target)
WHERE any(node IN nodes(path) WHERE node:Condition OR node:Medication OR node:Symptom)
RETURN path
```

**How it works:** Instead of blindly expanding in all directions, traverse only along relationship types that are clinically relevant. This prevents noise from unrelated entity connections.

### Pattern 4: Path-Based Reasoning
**Enables multi-hop reasoning by surfacing explicit chains.**

```cypher
MATCH path = shortestPath((a)-[*..6]-(b))
WHERE elementId(a) = $entity_a_id AND elementId(b) = $entity_b_id
RETURN
    [n IN nodes(path) | n.name] AS entity_chain,
    [r IN relationships(path) | type(r)] AS relationship_chain,
    length(path) AS hops
```

**How it works:** Find the shortest path between two entities. The path itself (entity chain + relationship chain) becomes part of the LLM context, enabling explicit reasoning like:
`Patient -[HAS_CONDITION]-> Stable Angina -[TREATED_WITH]-> Nitroglycerin`

### Pattern 5: Provenance / Source-of-Truth Linking
**The explainability layer — critical for data2.ai alignment.**

```cypher
MATCH (entity)-[sf:SOURCED_FROM]->(chunk:SourceChunk)-[:BELONGS_TO]->(doc:Document)
WHERE elementId(entity) IN $entity_ids
RETURN entity.name, chunk.text, chunk.section, doc.source_file, sf.confidence
```

**How it works:** Every entity has a `SOURCED_FROM` edge to the SourceChunk it was extracted from. Following the chain to the Document node gives a full citation: entity → source text → section → file. This is how you prove to a user (or auditor) *exactly* where a fact came from.

---

## 9. Provenance & Explainability (eXAI)

### What is Explainable AI?
AI where every output can be:
- **Traced** — back to specific source data
- **Verified** — by a human examining the evidence chain
- **Defended** — under regulatory scrutiny or audit

### data2.ai's eXAI Philosophy
> "Analysts need to be able to dissect exactly how the AI reached a particular conclusion or recommendation." — Eric Costantini, data2 CBO

Their platform delivers:
- **Visual lineage mapping** — traces every AI-generated answer back to exact source data with record-level citations
- **Hallucination resistance** — by grounding answers in graph-verified facts rather than LLM memory
- **Chain-of-custody** — complete audit trail for every decision

### How Our Project Implements eXAI

1. **Provenance Edges**: `(:Entity)-[:SOURCED_FROM {confidence, extraction_method}]->(:SourceChunk)-[:BELONGS_TO]->(:Document)`

2. **Confidence Scores**: Every extracted entity carries a confidence score (0.0-1.0) indicating how explicitly the information appeared in the source text.

3. **Extraction Method Tracking**: Each SOURCED_FROM edge records whether the entity was extracted by `llm_claude`, `structured_json`, or `spacy_ner`.

4. **Citation in LLM Responses**: The graph-backed RAG prompt includes provenance citations, and the LLM is instructed to cite specific sections and source files.

5. **Evidence Separation**: Source chunks (evidence) are separate from generated answers (which are transient API responses), preventing circular reasoning.

### Why This Matters for data2.ai's Clients

- **Defense/Intelligence**: Analysts must justify conclusions to leadership. "The AI said so" is not acceptable.
- **Oil & Gas**: Regulatory compliance requires traceable decision chains for safety-critical operations.
- **Financial**: Audit requirements demand provable logic behind risk assessments and portfolio decisions.

---

## 10. Evaluation Methodology

### RAGAS Framework (Retrieval-Augmented Generation Assessment)
**Source:** [arxiv.org/abs/2502.11371](https://arxiv.org/abs/2502.11371) — "RAG vs. GraphRAG: A Systematic Evaluation and Key Insights"

| Metric | What It Measures | How We Compute It |
|--------|-----------------|-------------------|
| **Faithfulness** | Are claims in the answer supported by context? | LLM-as-judge scores claim-by-claim support |
| **Context Precision** | Is retrieved context relevant to the question? | LLM-as-judge evaluates relevance of each chunk |
| **Context Recall** | Are ground-truth facts present in retrieved context? | LLM-as-judge checks coverage of expected answer |
| **Answer Correctness** | Does the answer match the expected answer? | 70% LLM judge + 30% embedding cosine similarity |
| **Citation Accuracy** | Do provenance links map to actual evidence? | Entity name / source text overlap analysis |

### LLM-as-Judge Pattern
Uses a separate Claude call with a structured evaluation tool to objectively score each metric. The judge receives: question, generated answer, expected answer, retrieved context — and must score each metric 0.0-1.0 with reasoning.

### Fair Comparison Design
- **Vector RAG** uses ONLY ChromaDB — no graph context
- **Graph RAG** uses ONLY Neo4j — vector index + graph expansion
- **Same embeddings model** (all-MiniLM-L6-v2) for both
- **Same LLM** (Claude 3.5 Sonnet) for both
- **Same prompt template structure** for both
- **Only the retrieval strategy differs**

---

## 11. This Project Walkthrough

### What We Built

**Neo4j Knowledge Graph for Grounded RAG** — a complete system that:

1. **Ingests** medical documents (SOAP notes + demographics) into a Neo4j property graph using an NLP extraction pipeline with entity resolution
2. **Implements** five expert Cypher retrieval patterns: entity-first, k-hop expansion, relationship-constrained filtering, path-based reasoning, and provenance linking
3. **Compares** vector-only RAG vs graph-backed RAG on factual consistency and context precision

### Input Data
- `Task_Files/soap_notes.txt` — Clinical SOAP note for patient Peter Fern (62M) with stable angina, hypertension, GERD
- `Task_Files/demographics.json` — Structured patient demographics (name, DOB, health card, address)

### Graph Produced
After ingestion, the Neo4j graph contains:
- **Patient** node (Peter Julius Fern) with all demographics properties
- **Condition** nodes (Stable Angina, Hypertension, GERD, Hyperlipidemia)
- **Symptom** nodes (Exertional Chest Tightness, etc.)
- **Medication** nodes (Antihypertensives, Nitroglycerin Sublingual, etc.)
- **Procedure** nodes (Treadmill Myocardial Perfusion Scan, Fasting Lipid Panel, etc.)
- **Vital** nodes (Blood Pressure 152/88, Heart Rate 78, etc.)
- **RiskFactor** nodes (Age 62, Family History MI at 58, Hypertension)
- **Document** and **SourceChunk** nodes with full provenance chains

### API Endpoints
- `POST /ingest` — run the full ingestion pipeline
- `POST /query` — ask questions with strategy selection (graph/vector/both)
- `GET /graph/explore/{name}` — visualize k-hop neighborhood
- `POST /evaluate` — run side-by-side evaluation harness

---

## 12. Real-World Debugging & Fixes

During development and integration testing, five production-grade issues were discovered and resolved. Each illustrates a common pitfall when working with Neo4j, Cypher, and graph-backed API systems.

---

### Fix 1: Neo4j 5 Cypher — `WHERE` After Multi-MATCH Requires `WITH`

**Symptom:** K-hop neighborhood expansion returned `SyntaxError: Invalid input 'WHERE'` on the Cypher query.

**Root cause:** In Neo4j 5, after a second `MATCH` clause (or `UNWIND`), you cannot immediately follow with `WHERE` — you must first pipe the variables through a `WITH` clause.

**Broken Cypher:**
```cypher
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH path = (seed)-[*1..3]-(neighbor)
WHERE elementId(neighbor) <> elementId(seed)   -- ❌ Invalid in Neo4j 5
RETURN ...
```

**Fixed Cypher:**
```cypher
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH path = (seed)-[*1..3]-(neighbor)
WITH seed, neighbor, path                       -- ✅ Pipeline variables first
WHERE elementId(neighbor) <> elementId(seed)
RETURN ...
```

**Why this matters:** This is a common trap when migrating from Neo4j 4 to 5 or writing Cypher with multiple `MATCH` clauses. The fix applies to both pure Cypher and APOC-based queries (`UNWIND ... WITH ... WHERE`).

**File:** `app/retrieval/k_hop_expansion.py` — both `CYPHER_KHOP` and `APOC_KHOP` queries.

---

### Fix 2: Neo4j 5 Inequality Operator — `<>` Not `!=`

**Symptom:** After fixing the `WITH` issue, the query returned `Unknown operation '!='`.

**Root cause:** Neo4j 5's Cypher uses `<>` for "not equal to" (following the SQL standard). The `!=` operator is not supported.

**Fix:** Replace all `!=` with `<>` in Cypher queries:
```cypher
-- ❌ WHERE elementId(neighbor) != elementId(seed)
-- ✅ WHERE elementId(neighbor) <> elementId(seed)
```

**Why this matters:** If you're used to writing Python or JavaScript, `!=` is muscle memory. Cypher follows the SQL convention. Additionally, we use `elementId()` for comparison instead of bare node references, because `elementId()` returns a stable string for comparison whereas `seed <> neighbor` may fail if nodes are referenced across different `MATCH` scopes.

**File:** `app/retrieval/k_hop_expansion.py`

---

### Fix 3: JSON Serialization — Neo4j `DateTime` Not Serializable

**Symptom:** Graph RAG queries returned `500 Internal Server Error` from the FastAPI endpoint. Vector RAG worked fine. Running the same `graph_rag_query()` in a Python REPL succeeded — but calling `json.dumps()` on the response raised `TypeError: Object of type DateTime is not JSON serializable`.

**Root cause:** Neo4j's Python driver returns `neo4j.time.DateTime` objects for temporal properties (e.g., `created_at`). These survive through Pydantic's `dict[str, Any]` fields in `GraphNode.properties` without complaint, but fail when FastAPI serializes the response to JSON.

**Fix:** Created `app/retrieval/utils.py` with a `sanitize_properties()` function that:
- Removes `embedding` and binary values (these are large and not useful in API responses)
- Converts `neo4j.time.DateTime` / `neo4j.time.Date` to ISO strings via `.iso_format()`
- Converts Python `datetime` to `.isoformat()`
- Recursively sanitizes nested dicts and lists

Applied `sanitize_properties()` everywhere we build `GraphNode` or `GraphRelationship` from Neo4j records:
- `app/retrieval/entity_first.py` — seed entities from vector search
- `app/retrieval/k_hop_expansion.py` — neighborhood nodes and edges
- `app/retrieval/relationship_filter.py` — filtered target nodes and edges

**Why this matters:** This is a classic issue when combining Neo4j's native types with JSON-based APIs. The fix is systematic — a single sanitizer used everywhere — rather than ad-hoc conversions sprinkled through the codebase.

**Key learning:** When a Pydantic model accepts `dict[str, Any]`, it will happily store non-JSON-serializable values. The error only surfaces at serialization time (FastAPI response). Direct Python testing misses this unless you explicitly call `json.dumps()` on the result.

---

### Fix 4: `shortestPath` Same-Node Error

**Symptom:** 6 out of 11 evaluation harness queries failed for graph RAG with: `Neo.DatabaseError.Statement.ExecutionFailed: The shortest path algorithm does not work when the start and end nodes are the same.`

**Root cause:** The `SEED_TO_NAMED_QUERY` in `path_reasoning.py` finds paths from seed entities to named target entities. When a seed entity happens to have the same name as the target (e.g., seed "Hypertension" → target "Hypertension"), Neo4j's `shortestPath()` algorithm throws an exception rather than returning an empty result.

This affected queries like "What is the patient's age and sex?" where the Patient entity was both a seed and a target in path reasoning.

**Fix (two layers):**

**Layer 1 — Cypher guard:** Added `WITH seed, target WHERE elementId(seed) <> elementId(target)` before every `shortestPath()` call. This filters out same-node pairs at the database level.

```cypher
-- Before (crashes):
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH (target) WHERE target.name = $target_name
MATCH path = shortestPath((seed)-[*..5]-(target))

-- After (safe):
MATCH (seed) WHERE elementId(seed) IN $seed_ids
MATCH (target) WHERE target.name = $target_name
WITH seed, target WHERE elementId(seed) <> elementId(target)  -- ✅ Guard
MATCH path = shortestPath((seed)-[*..5]-(target))
```

**Layer 2 — Python guard:** Added early-return checks in `find_shortest_path()` and `find_all_paths()`:
```python
if entity_a_id == entity_b_id:
    return None  # Same node — no meaningful path
```

**Why this matters:** The Neo4j `shortestPath()` same-node limitation is documented but easy to miss. When building graph RAG systems where seeds and targets come from the same entity pool, this collision is inevitable. The defense-in-depth approach (Cypher `WHERE` + Python check) ensures robustness.

**Impact:** Graph RAG evaluation went from 6/11 errors to 0/11 errors. Average faithfulness improved from 0.42 → 0.94.

**Files:** `app/retrieval/path_reasoning.py` — `SHORTEST_PATH_QUERY`, `ALL_PATHS_QUERY`, `SEED_TO_NAMED_QUERY`, plus Python functions `find_shortest_path()` and `find_all_paths()`.

---

### Fix 5: Graph RAG Prompt Optimization — 12.8s → 7.0s Generation

**Symptom:** Graph RAG returned correct answers but took 12.8 seconds for generation (vs 2.5s for vector RAG). The retrieval was fast (~100ms); the bottleneck was LLM processing of a massive prompt context.

**Root cause:** The `format_context_for_prompt()` function was dumping the entire context bundle into the LLM prompt without limits:
- All 30 seed entities (full property sets)
- All 20+ neighborhood nodes
- All 34 provenance citations (each with 150-char source text)
- All reasoning paths

This produced a prompt with thousands of tokens, and Claude's processing time scales with input token count.

**Fix:** Applied caps in `app/retrieval/context_builder.py:format_context_for_prompt()`:

| Context Section | Before | After |
|----------------|--------|-------|
| Seed entities | All (~30) | Top 10 |
| Neighborhood nodes | 20 | 15 |
| Relationships | 20 (with truncated element IDs) | 15 (with human-readable entity names) |
| Provenance citations | All (~34) | Top 15 (sorted by confidence desc) |
| Source text preview | 150 chars | 120 chars |

Also set `max_tokens=1500` for graph RAG generation (down from default 4096) since well-cited answers don't need 4K tokens.

Additionally, improved relationship display from:
```
- [HAS_CONDITION] 4:abc1234… → 4:def5678…   (opaque element IDs)
```
to:
```
- Peter Fern -[HAS_CONDITION]-> Stable Angina   (human-readable names)
```

**Result:** Generation time dropped from **12.8s → 7.0s** (45% faster) with no loss in answer quality or faithfulness.

**Why this matters:** In production RAG systems, prompt engineering isn't just about *what* to include — it's about *how much*. More context doesn't always mean better answers. The LLM needs enough evidence to ground its response, but flooding it with 34 citations when 15 suffice just adds latency and cost without improving quality.

---

### Summary — Fixes at a Glance

| # | Issue | Root Cause | Fix | Impact |
|---|-------|-----------|-----|--------|
| 1 | `WHERE` after multi-MATCH | Neo4j 5 Cypher syntax requires `WITH` pipeline | Add `WITH ... WHERE` | K-hop queries execute |
| 2 | `!=` operator unknown | Neo4j uses `<>` (SQL convention) | Replace `!=` → `<>` | K-hop queries execute |
| 3 | `DateTime` not JSON-serializable | Neo4j driver returns native time types | `sanitize_properties()` util | Graph endpoint returns 200 |
| 4 | shortestPath same-node crash | Seeds can match targets in path reasoning | Cypher `WHERE` guard + Python check | 6 eval errors → 0 |
| 5 | 12.8s graph generation | Prompt too large (30 seeds, 34 citations) | Cap context + reduce max_tokens | 45% faster (7.0s) |

---

## 13. Trade-offs & Latency Analysis

### The Core Trade-off: Precision vs Explainability

Our evaluation revealed a clear trade-off between the two retrieval strategies:

| Metric | Vector RAG | Graph RAG | What it means |
|--------|------------|-----------|---------------|
| Faithfulness | **0.99** | 0.94 | Both ground answers well; <7% unsupported claims |
| Context Precision | **0.90** | 0.49 | Graph retrieves 2x the context, half is tangential |
| Context Recall | 0.91 | 0.91 | Both capture 91% of ground-truth facts (identical) |
| Answer Correctness | **0.83** | 0.75 | Vector answers are slightly more accurate overall |
| Citation Accuracy | n/a | 0.41 | Graph-only: ~40% of provenance citations directly support answer |

**The honest story:** Vector RAG wins on raw accuracy metrics. Graph RAG's value is *not* in beating vector on precision — it's in providing an explainability layer that vector can't:

1. **Entity chains** — "Patient -[HAS_CONDITION]-> Stable Angina -[TREATED_WITH]-> Nitroglycerin" makes reasoning auditable
2. **Provenance traceability** — every fact traces through SOURCED_FROM → SourceChunk → Document with confidence scores
3. **Multi-hop context** — the expanded neighborhood captures related entities the vector search wouldn't surface on its own

For data2.ai's use case (defense, energy, financial — where *why* matters as much as *what*), this trade-off is the right one. An analyst reviewing intelligence doesn't just need the answer; they need the evidence chain to defend it.

### Why Graph RAG Has Lower Context Precision

Context precision (0.49) is the weakest graph metric. Here's exactly why:

The 5-stage retrieval pipeline produces a rich context bundle:
- **30 seed entities** from vector search (many share SOURCED_FROM edges with matched chunks)
- **22 neighborhood nodes** from k-hop expansion (3 hops captures tangential entities)
- **21 relationship edges** from constrained traversal
- **25 reasoning paths** from shortestPath queries
- **34 provenance citations** linking entities back to source text

For a single-hop question like "What is the patient's blood pressure?", only a small subset of this context is strictly needed (the Vital node with BP 152/88 and its source chunk). But the graph retrieves the *entire neighborhood* — including symptoms, conditions, and medications that are connected but not relevant to the specific question.

**This is by design.** The expanded context:
- Makes multi-hop questions answerable without additional retrieval passes
- Provides the LLM with enough relational structure to explain *why* (not just *what*)
- Enables citation accuracy as a separate explainability metric

**The production tuning lever:** Adjusting `retrieval_max_hops` (currently 3) and the prompt caps (currently 10 seeds, 15 nodes, 15 citations) directly trades precision for recall/explainability. For simple factual questions, 1-hop expansion with 5 citations would score higher precision. For complex analytical questions, 3 hops with full citations is necessary.

### Latency Breakdown: Where Time Is Actually Spent

```
                          Vector RAG          Graph RAG
                          ──────────          ─────────
Embedding query:           ~5 ms               ~5 ms        (all-MiniLM-L6-v2, CPU)
Vector search:            ~190 ms (ChromaDB)   ~50 ms (Neo4j vector index)
Graph expansion:            n/a                ~15 ms (APOC k-hop, 3 hops)
Relationship filter:        n/a                ~20 ms (constrained Cypher)
Path reasoning:             n/a                ~15 ms (shortestPath × 5 targets)
Provenance linking:         n/a                ~5 ms  (SOURCED_FROM traversal)
                          ──────────          ─────────
Total retrieval:          ~200 ms              ~110 ms
                          ──────────          ─────────
LLM generation (Claude):  ~2.5 s              ~7.0 s
                          ──────────          ─────────
Total end-to-end:         ~2.7 s              ~7.1 s
```

**Key observations:**

1. **Graph retrieval is faster than vector** (~110ms vs ~200ms). Neo4j's native vector index is faster than ChromaDB for our dataset size, and the additional graph traversal stages add only ~55ms combined. Index-free adjacency means each hop is O(1).

2. **~95% of graph RAG latency is LLM generation** (7.0s out of 7.1s). The retrieval pipeline is not the bottleneck — Claude processing the structured context prompt is. This is input-token-bound: the graph context bundle is ~3-5x larger than the vector chunk list.

3. **The 7.0s generation is acceptable for analytical workloads** (an analyst reviewing evidence), but would be too slow for a real-time chatbot. For interactive use, you'd either:
   - Use a faster model (Claude Haiku: ~1-2s) with some quality loss
   - Implement streaming (first tokens arrive in ~500ms, perceived latency drops)
   - Pre-compute and cache common subgraph patterns

4. **Vector search is the bottleneck in vector RAG** (~190ms of ~200ms retrieval). ChromaDB is a file-based store; switching to a managed vector DB (Pinecone, Weaviate) or using Neo4j's vector index for both strategies would equalize this.

### Where Latency Would Change at Scale

| Component | Current (48 nodes) | 10K nodes | 1M nodes | Mitigation |
|-----------|-------------------|-----------|----------|------------|
| Vector search | ~50ms (Neo4j) | ~50ms | ~80ms | HNSW index scales logarithmically |
| K-hop expansion | ~15ms | ~50ms | ~200ms+ | Limit hops to 2; use relationship-type filters to prune early |
| Path reasoning | ~15ms | ~100ms | ~1s+ | Pre-compute common paths; use GDS cached graph projections |
| Provenance | ~5ms | ~10ms | ~20ms | Index-backed, scales linearly |
| LLM generation | ~7s | ~7s | ~7s | Scales with prompt size (cap context), not graph size |

The graph traversal stages are the only components that degrade with scale. The mitigation is always the same: constrain the traversal scope (fewer hops, stricter relationship filters, lower `LIMIT` caps).

### Trade-off Summary

| Dimension | Vector RAG | Graph RAG |
|-----------|------------|-----------|
| **Accuracy** | Higher (0.83 correctness) | Lower (0.75 correctness) |
| **Explainability** | None (chunks only) | Full (entities + paths + provenance) |
| **Latency** | Fast (2.7s total) | Slower (7.1s total) |
| **Retrieval speed** | ~200ms | ~110ms (faster) |
| **Hallucination resistance** | Excellent (0.99) | Excellent (0.94) |
| **Audit readiness** | Low (no citation chain) | High (SOURCED_FROM + confidence) |
| **Multi-hop ability** | LLM-only inference | Explicit graph traversal |
| **Cost per query** | Lower (smaller prompt) | Higher (~3-5x more input tokens) |

**Bottom line:** If you only need an accurate answer, use vector RAG. If you need to *prove* the answer — trace it to source data, show the reasoning path, and defend it under scrutiny — use graph RAG. data2.ai's clients (defense, energy, financial) need the latter.

---

## 14. Scaling to Production

### Current State vs Production Requirements

| Dimension | Current (Demo) | Production Target |
|-----------|---------------|-------------------|
| Graph size | 48 nodes, 72 rels | 100K-10M+ nodes |
| Documents | 2 files | 10K-100K+ files |
| Concurrent users | 1 | 50-500+ |
| Ingestion throughput | ~0.6 nodes/sec | 100+ nodes/sec |
| Query latency (P95) | ~7s | <3s |
| Availability | Single-process | 99.9%+ |

### Scaling Axis 1: Ingestion Throughput

**Current bottleneck:** LLM extraction is sequential — one Claude call per chunk, ~5-8s each.

**Fixes:**
- **Async worker pool** — process chunks in parallel with `asyncio.gather()` or Celery workers. 8 concurrent extractions × 5s each = 8 chunks per 5s instead of 1.
- **Batch graph writes** — replace per-entity `MERGE` with `UNWIND $batch MERGE (e:Condition {name: row.name}) SET e += row.props`. One Cypher call per batch of 50-100 entities instead of one per entity.
- **Streaming ingestion** — decouple extraction from graph write. Extract → message queue (Redis/Kafka) → graph writer consumer. This lets extraction and writing scale independently.
- **Cheaper extraction for bulk** — use Claude Haiku or a fine-tuned local model (Llama 3) for bulk extraction, reserve Sonnet for high-value or ambiguous documents.

**Target:** 100+ nodes/sec is achievable with 8 parallel extractors + batch writes.

### Scaling Axis 2: Neo4j Graph Performance

**Current:** Single Docker container, no clustering, 48 nodes.

**Fixes for 100K-1M+ nodes:**

| Strategy | What | When to Use |
|----------|------|-------------|
| **AuraDB Professional** | Managed Neo4j with auto-scaling, backups, monitoring | Default choice for cloud deployment |
| **Read replicas** | Neo4j Enterprise/Aura supports read replicas for query scaling | When read throughput > single instance capacity |
| **Graph projections (GDS)** | Project a subgraph into memory for fast algorithm execution | When path reasoning or community detection becomes slow |
| **Composite indexes** | Add indexes on frequently filtered properties (e.g., `entity.name + label`) | When `MATCH (e:Condition {name: $name})` slows down |
| **Relationship indexes** | Neo4j 5.7+ supports relationship property indexes | When filtering on SOURCED_FROM.confidence or edge properties |

**Cypher optimizations at scale:**
```cypher
-- Current: unbounded variable-length path (expensive at scale)
MATCH path = (seed)-[*1..3]-(neighbor)

-- Production: constrain relationship types in the pattern
MATCH path = (seed)-[:HAS_CONDITION|MANIFESTS_AS|TREATED_WITH*1..2]-(neighbor)

-- Even better: use APOC expand with limits
CALL apoc.path.subgraphNodes(seed, {
    maxLevel: 2,
    relationshipFilter: "HAS_CONDITION|MANIFESTS_AS|TREATED_WITH",
    limit: 50
}) YIELD node
```

### Scaling Axis 3: Query Latency

**Current bottleneck breakdown:**
- Retrieval: 110ms (fine at any scale with proper indexes)
- LLM generation: 7.0s (95% of total latency)

**Fixes:**

**Reduce LLM latency:**
- **Streaming responses** — use Anthropic's streaming API (`stream=True`). First tokens arrive in ~300-500ms; user sees output immediately while generation continues. Perceived latency drops from 7s to <1s.
- **Model selection** — Claude Haiku for simple queries (~500ms generation), Sonnet for complex multi-hop only. Route based on question complexity classification.
- **Context budget** — dynamically adjust context caps based on query type:
  - Single-hop: 5 seeds, 5 citations, max_tokens=512 → ~2s
  - Multi-hop: 10 seeds, 15 citations, max_tokens=1500 → ~5s
  - Full analytical: 15 seeds, 20 citations, max_tokens=2000 → ~7s

**Reduce retrieval latency at scale:**
- **Pre-computed neighborhood cache** — for frequently queried entities, cache their 2-hop neighborhood in Redis. TTL = graph update interval.
- **Graph projections** — use GDS to create in-memory graph projections for hot subgraphs. Path reasoning on projected graphs is 10-100x faster than on-disk.
- **Query result caching** — cache (query_embedding_hash → context_bundle) for repeat or near-duplicate questions. Invalidate on graph write.

### Scaling Axis 4: Reliability & Operations

| Concern | Solution |
|---------|----------|
| **High availability** | Neo4j Aura (managed) or Enterprise causal clustering (3+ core servers) |
| **Zero-downtime deploys** | Blue-green deployment with health checks; FastAPI behind load balancer |
| **Monitoring** | Prometheus + Grafana for: query latency p50/p95/p99, Neo4j heap/page cache, LLM token usage |
| **Alerting** | Alert on: retrieval >500ms, generation >15s, Neo4j connection failures, evaluation faithfulness drop |
| **Data freshness** | Incremental ingestion — detect new/changed documents, re-extract only affected chunks, merge into existing graph |
| **Schema evolution** | Neo4j's flexible schema means new entity types / relationship types can be added without migration. Add new labels and relationship types alongside existing ones. |

### Scaling Axis 5: Cost Control

| Component | Current Cost Driver | Production Optimization |
|-----------|-------------------|------------------------|
| LLM extraction | Claude Sonnet per chunk (~$0.003/chunk) | Batch with Haiku ($0.0003/chunk) for routine docs; Sonnet for ambiguous |
| LLM generation | Claude Sonnet per query (~$0.01/query) | Haiku for simple queries; streaming to reduce timeout retries |
| LLM evaluation | Claude Sonnet per judge call (~$0.005/eval) | Run evaluation on schedule (nightly), not per-query |
| Neo4j | Docker (free) | AuraDB Free tier for dev; Professional ($65/mo) for production |
| Embeddings | CPU inference (free, ~5ms/query) | Stay on CPU — GPU is unnecessary for all-MiniLM-L6-v2 |

**Estimated production cost at 1K queries/day:**
- LLM generation: ~$10/day (mix of Haiku + Sonnet)
- Neo4j AuraDB: ~$2/day
- Compute (FastAPI on 2-core VM): ~$1/day
- **Total: ~$13/day (~$400/month)**

### Architecture Diagram — Production

```
                    ┌─────────────┐
                    │  Load       │
                    │  Balancer   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ FastAPI  │ │ FastAPI  │ │ FastAPI  │  (stateless app replicas)
         │ Worker 1 │ │ Worker 2 │ │ Worker 3 │
         └────┬─────┘ └────┬─────┘ └────┬─────┘
              │            │            │
              ▼            ▼            ▼
    ┌──────────────────────────────────────────┐
    │           Redis Cache Layer               │
    │  (query cache, neighborhood cache, TTL)   │
    └──────────────┬───────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
  ┌──────────────┐    ┌──────────────┐
  │  Neo4j Aura  │    │  Anthropic   │
  │  (graph +    │    │  Claude API  │
  │  vector idx) │    │  (Haiku +    │
  │              │    │   Sonnet)    │
  └──────────────┘    └──────────────┘
         │
  ┌──────┴───────┐
  │  Kafka /     │
  │  Redis Queue │  (async ingestion pipeline)
  │              │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Extraction  │
  │  Workers     │  (parallel LLM extraction + graph write)
  └──────────────┘
```

### What Changes from Demo → Production (Summary)

| Area | Demo | Production |
|------|------|------------|
| App server | Single `uvicorn` process | Multiple replicas behind load balancer |
| Neo4j | Docker container | AuraDB or Enterprise causal cluster |
| Ingestion | Synchronous, sequential | Async workers + message queue |
| LLM calls | All Sonnet, no streaming | Haiku/Sonnet routing + streaming |
| Caching | None | Redis for query results + subgraph neighborhoods |
| Monitoring | Log files | Prometheus/Grafana + alerting |
| Context sizing | Fixed caps | Dynamic caps based on query complexity |

---

## 15. Interview Q&A Prep

### "Why did you use Neo4j instead of a relational database?"
> Medical data is inherently connected — patients have conditions, conditions manifest as symptoms, symptoms are treated with medications. These are multi-hop relationships that require expensive self-joins in SQL but are natural pattern matches in Cypher. Neo4j's index-free adjacency means each hop is O(1) regardless of dataset size. Additionally, provenance tracking (linking facts to source text) is naturally expressed as graph edges.

### "Explain your graph schema design."
> I designed a property graph with labeled nodes for each medical concept (Patient, Condition, Symptom, Medication, Vital, etc.) and typed relationships between them (HAS_CONDITION, TREATED_WITH, MANIFESTS_AS). The critical addition is the provenance layer: every extracted entity has a SOURCED_FROM edge to the SourceChunk it came from, with confidence score and extraction method. SourceChunks link to Documents via BELONGS_TO, creating a full citation chain. This mirrors data2.ai's approach of tracing every insight to its source data.

### "How does your retrieval engine work?"
> It uses five composable patterns. First, entity-first retrieval embeds the query and finds similar source chunks via Neo4j's native vector index, then follows SOURCED_FROM edges backward to discover seed entities. Second, k-hop expansion traverses variable-length paths from seeds to collect the neighborhood subgraph. Third, relationship-constrained filtering narrows to clinically relevant edge types. Fourth, path-based reasoning finds explicit entity chains for multi-hop questions. Fifth, provenance linking traces all entities back to source text for citations. These are assembled into a structured context bundle for LLM prompting.

### "How do you handle medical abbreviations and entity dedup?"
> Two-pass entity resolution. Pass 1 uses rapidfuzz (token sort ratio ≥ 88) for fuzzy string matching plus a known medical synonym map (HTN→Hypertension, MI→Myocardial Infarction). Pass 2 uses sentence-transformer embeddings with cosine similarity ≥ 0.85 to catch semantic synonyms that are spelled differently. The canonical (longer, more descriptive) name is kept, and all references are remapped before graph write.

### "What's the difference between your vector RAG and graph RAG?"
> Vector RAG is a pure baseline: embed query → cosine similarity on ChromaDB → top-k chunks → LLM. It has zero graph context. Graph RAG starts the same way but uses the vector results as seeds into the knowledge graph, then expands via k-hop traversal, filters by relationship type, finds reasoning paths, and adds provenance citations. The graph version can answer multi-hop questions like "What conditions explain the patient's exertional symptoms given their risk factors?" because it can traverse Patient → Condition → Symptom → RiskFactor chains.

### "How do you evaluate your system?"
> I built a RAGAS-style evaluation harness with 11 gold-standard questions across five categories: single-hop factual, multi-hop reasoning, provenance tracing, relationship queries, and cross-reference. Each question runs through both strategies. I use an LLM-as-judge pattern (separate Claude call with structured tool output) to score faithfulness, context precision, context recall, and answer correctness. Citation accuracy is computed only for graph RAG by checking whether provenance links overlap with the expected answer.
>
> **Actual results (11 questions, real run):** Vector RAG scored 0.99 faithfulness / 0.90 context precision / 0.91 context recall / 0.83 correctness. Graph RAG scored 0.94 faithfulness / 0.49 context precision / 0.91 context recall / 0.75 correctness / 0.41 citation accuracy. Both achieve high faithfulness (>0.93), meaning minimal hallucination. The graph's lower context precision (0.49 vs 0.90) is the expected trade-off — the expanded subgraph includes neighborhood context that's useful for explainability but not all directly answering the question. Context recall is identical (0.91), confirming the graph captures the same ground-truth facts while adding relational structure.

### "How does this relate to data2.ai's reView platform?"
> This project directly mirrors reView's architecture. Their Connect→Enrich→Reason→Visualize pipeline maps to my Ingest→Extract→Retrieve→Query flow. My provenance edges (SOURCED_FROM with confidence scores) implement the same traceability that lets reView show "the proof behind every conclusion." The entity extraction pipeline is analogous to their Arctic Loader + LLM fact extractor. The evaluation harness demonstrates the same principle they advocate: explainable AI means every decision is verifiable and defensible, not a black-box output.

### "Tell me about a technical challenge you solved."
> I'll give two examples — one Cypher-level and one system-level.
>
> **Cypher challenge:** During integration testing, 6 out of 11 evaluation queries crashed with a Neo4j `shortestPath` error. The issue: when building reasoning paths from seed entities to target entities, some seeds and targets resolved to the *same* node (e.g., "Hypertension" was both a seed from vector retrieval and a target in the path query). Neo4j throws an exception on `shortestPath` when start and end nodes are identical. I fixed it with defense-in-depth: a Cypher-level `WITH seed, target WHERE elementId(seed) <> elementId(target)` guard before every `shortestPath` call, plus a Python-level early return when IDs match. That took graph RAG from 45% failure rate to zero errors.
>
> **System challenge:** Neo4j's Python driver returns `neo4j.time.DateTime` objects for temporal properties. These survive through Pydantic's `dict[str, Any]` without complaint — but when FastAPI serializes the response to JSON, it crashes. The subtle part: running the same code in a Python REPL works perfectly. The error only surfaces when the full HTTP response chain serializes. I built a `sanitize_properties()` utility that converts Neo4j time types to ISO strings and applied it at every graph-to-Pydantic boundary. This taught me that Pydantic's `Any` type is a serialization time bomb — it validates on input but defers type checking to output.
>
> I also had a classic Neo4j 5 migration trap: `WHERE` immediately after a second `MATCH` is invalid — you need a `WITH` pipeline step first. And `!=` doesn't exist in Cypher; it uses `<>` (SQL convention). These are small but critical differences that can waste hours if you don't know them.
>
> *(Full details with code examples in [Section 12: Real-World Debugging & Fixes](#12-real-world-debugging--fixes).)*

### "How would you scale this to millions of documents?"
> Five axes. (1) **Ingestion parallelism** — async worker pool for parallel LLM extraction (8 concurrent = 8x throughput), batch `UNWIND` writes instead of per-entity MERGE, and a message queue (Kafka/Redis) to decouple extraction from graph writing. (2) **Neo4j scaling** — AuraDB Professional for managed auto-scaling, read replicas for query throughput, composite + relationship indexes for filtered lookups. (3) **Query latency** — streaming LLM responses (perceived latency drops from 7s to <1s), model routing (Haiku for simple queries, Sonnet for complex), dynamic context budgets based on query complexity, and Redis caching for repeat queries and pre-computed subgraph neighborhoods. (4) **Cypher optimization** — at scale, replace unbounded variable-length paths with relationship-type-constrained patterns and `apoc.path.subgraphNodes` with explicit limits. Pre-compute hot subgraphs using GDS graph projections for 10-100x faster path reasoning. (5) **Cost control** — Haiku for routine extraction ($0.0003/chunk vs $0.003/chunk Sonnet), CPU-only embeddings (all-MiniLM-L6-v2 doesn't need GPU), and scheduled evaluation runs instead of per-query judging. Estimated production cost at 1K queries/day: ~$400/month.
>
> *(Full architecture diagrams and component-level scaling strategies in [Section 14: Scaling to Production](#14-scaling-to-production).)*

---

## Key Terms Glossary

| Term | Definition |
|------|-----------|
| **Property Graph** | Data model where nodes and relationships carry key-value properties |
| **Cypher** | Neo4j's declarative graph query language |
| **APOC** | "Awesome Procedures on Cypher" — extension library for Neo4j |
| **GDS** | Graph Data Science — Neo4j's graph algorithm library |
| **GraphRAG** | Retrieval-augmented generation using knowledge graph context |
| **Entity Resolution** | Merging duplicate/variant entity references into one canonical form |
| **Provenance** | The origin/lineage of a piece of data — traceability to source |
| **eXAI** | Explainable AI — AI outputs that can be traced, verified, defended |
| **Index-Free Adjacency** | Neo4j storage where each node physically points to neighbors |
| **MERGE** | Cypher's idempotent upsert — create if not exists, update if exists |
| **Vector Index** | Neo4j 5.11+ feature for approximate nearest neighbor search on embeddings |
| **SOURCED_FROM** | Our provenance edge type linking entities to source text chunks |
| **SourceChunk** | A section of text from a document, stored as a graph node with embeddings |
| **K-Hop Expansion** | Traversing K relationship steps from a starting node to collect neighborhood |
| **Entity-First Retrieval** | Using vector similarity to find relevant graph entities as retrieval seeds |
| **Relationship-Constrained** | Filtering graph traversal to specific, meaningful relationship types |
| **Path-Based Reasoning** | Using shortest/all paths between entities for multi-hop inference |
| **RAGAS** | Retrieval-Augmented Generation Assessment — evaluation framework |
| **LLM-as-Judge** | Using an LLM to evaluate quality of another LLM's outputs |
| **Faithfulness** | Metric: fraction of generated claims supported by retrieved context |
| **Context Precision** | Metric: fraction of retrieved context that is relevant |
| **Context Recall** | Metric: fraction of needed facts covered by retrieved context |
| **AuraDB** | Neo4j's fully managed cloud graph database service |
| **Arctic Loader** | data2.ai's component for loading data into knowledge graphs |
| **reView** | data2.ai's flagship explainable AI analytics platform |
| **Evidence Graph** | data2 pattern: separate graph for verified facts/evidence |
| **Question Graph** | data2 pattern: separate graph for user questions + AI answers |
| **Chain-of-Custody** | Complete audit trail showing how data moved and transformed |

---

*Last updated: February 2026*
*System repository: Medical_Doc_Knowledge_Graph_System*
