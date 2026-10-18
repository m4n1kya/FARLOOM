# FARLOOM — Technical Knowledge Intelligence Engine

<p align="center">
  <strong>Hybrid GraphRAG · Dense Vectors · Okapi BM25 · Neo4j AuraDB · Reciprocal Rank Fusion</strong>
</p>

<p align="center">
  <a href="https://farloom.onrender.com"><img src="https://img.shields.io/badge/Live_Demo-farloom.onrender.com-000000?style=for-the-badge&logo=render&logoColor=white" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Tests-104_Passed-000000?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/Python-3.12+-000000?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Neo4j-AuraDB_Ready-000000?style=for-the-badge&logo=neo4j&logoColor=white" alt="Neo4j AuraDB">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-000000?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/License-MIT-000000?style=for-the-badge" alt="License">
</p>

---

### [Launch Live Demo Workspace](https://farloom.onrender.com)

FARLOOM is an enterprise-grade technical knowledge intelligence engine that ingests documentation, synthesizes a multi-tenant knowledge graph, performs hybrid retrieval across three distinct evidence streams, and answers complex multi-hop technical questions with verifiable provenance.

FARLOOM fuses **dense vector search (ChromaDB)**, **sparse keyword matching (Okapi BM25)**, and a **dynamic knowledge graph (Neo4j AuraDB / NetworkX)** into a unified retrieval engine governed by **Reciprocal Rank Fusion (RRF)**.

---

## Key Highlights

- **Hybrid Evidence Fusion**: Integrates vector proximity, lexical frequency, and graph topology. Outperforms standalone vector search on exact identifiers and multi-hop entity reasoning.
- **Dual Knowledge Graph Backends**: Supports zero-dependency in-memory **NetworkX** graphs alongside enterprise cloud-native **Neo4j AuraDB** with multi-tenant Cypher queries and schema constraints.
- **OLED Knowledge Universe Visualizer**: High-performance HTML5 Canvas interface featuring cosmic force-directed physics, cluster nebulae, photon pulse animations, and shortest-path relational tracing.
- **Isolated Knowledge Spaces**: Multi-tenant workspace architecture allowing users to isolate corpora, vector collections, and graph subgraphs with zero cross-space leakage.
- **Complete Document Lifecycle**: Drag-and-drop document parser (.pdf, .md, .txt) with heading-aware chunkers, metadata extraction, and atomic cascade deletion across SQLite, ChromaDB, and Neo4j.
- **Empirical Evaluation Benchmark**: Built-in benchmark suite evaluating Hit@k, MRR, Context Precision, and Faithfulness against gold-standard QA datasets.
- **Zero API Key Requirement**: Runs 100% offline out-of-the-box with deterministic heuristics, while offering plug-and-play support for Google Gemini and OpenAI models.

---

## Architecture

```
Documents (PDF / Markdown / TXT)
         │
         ▼
    ┌─────────────┐
    │   Ingest    │  load → chunk → embed → extract entities + triples
    └──────┬──────┘
           │
     ┌─────┼──────────────────────────────────┐
     ▼     ▼                                  ▼
  ChromaDB SQLite                    NetworkX / Neo4j AuraDB
  (vectors) (chunks + metadata)       (knowledge graph backend)
           │
           ▼
    ┌─────────────────────────────────┐
    │      Hybrid Retriever           │
    │  dense │  bm25  │  graph expand │
    │    └───┴────────┴───┘           │
    │    Reciprocal Rank Fusion (RRF) │
    └──────────────┬──────────────────┘
                   │
                   ▼
           ┌───────────────┐
           │AnswerSynthesizer│  FakeLLM / GeminiLLM / OpenAILLM
           └───────┬───────┘
                   │
                   ▼
         AskResponse { answer, citations, debug }
```

---

## Knowledge Graph Backends

FARLOOM supports dual knowledge graph storage backends selected explicitly via configuration:

| Backend | Configuration | Characteristics |
|---|---|---|
| **NetworkX** (Default) | `GRAPH_BACKEND=networkx` | Fast, in-memory graph hydrated from SQLite. 100% offline with zero external dependencies. |
| **Neo4j / AuraDB** | `GRAPH_BACKEND=neo4j` | Persistent distributed graph database, official `neo4j` Python driver, parameterized Cypher queries, multi-space tenancy, and cloud persistence. |

### Connecting to Neo4j AuraDB (Free Tier)

1. **Create Free Database**:
   Visit [Neo4j AuraDB Console](https://console.neo4j.io) and create an **AuraDB Free** instance.
2. **Download Credentials**:
   Save your connection URI (e.g. `neo4j+s://<instance-id>.databases.neo4j.io`) and password generated during instance setup.
3. **Configure Environment Variables**:
   In your `.env` file or hosting secrets:
   ```bash
   GRAPH_BACKEND=neo4j
   NEO4J_URI=neo4j+s://<instance-id>.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_secure_password
   NEO4J_DATABASE=neo4j
   ```
4. **Automatic Schema & Verification**:
   When started in Neo4j mode, FARLOOM automatically:
   - Verifies driver connectivity on startup and reports state in `/health` and `/status`.
   - Idempotently creates uniqueness constraints: `CREATE CONSTRAINT entity_uid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE`.
   - Creates composite indexes on `(space_id, key)` for multi-tenant query performance.
   - Syncs existing triples and chunk provenance without duplicates using parameterized `MERGE`.

---

## Retrieval Modes

| Mode | Legs | Best for |
|------|------|----------|
| `dense` | Vector search (ChromaDB) | Semantic similarity and conceptual matching |
| `bm25` | Okapi BM25 | Exact terms, identifiers, function names, version strings |
| `hybrid` | dense + BM25 via RRF | General purpose robust text retrieval |
| `hybrid_graph` | dense + BM25 + graph expansion | Complex multi-hop technical reasoning and organizational links |

### How Retrieval Works

#### 1. Dense (Vector) Retrieval
Feature-hashing embeddings (512-dim, offline) stored in ChromaDB. Cosine similarity query returns semantically related passages without network latency.

#### 2. BM25 Retrieval
Okapi BM25 over tokenized chunk text. Excels at exact package names, API identifiers, version strings, and error codes that semantic similarity misses.

#### 3. Graph Expansion
1. Named entities in the query are matched to graph nodes (longest-match, case-insensitive).
2. Bounded BFS from seed nodes (configurable depth + node limit).
3. Each visited entity votes for its source chunks with weight `1 / (1 + hop) / sqrt(mention_count)`.
4. Chunks ranked by closeness + focus score to prevent hub contamination.

#### 4. Reciprocal Rank Fusion (RRF)
```
RRF_score(chunk) = Σ  1 / (k + rank_in_leg)
                  legs
```
Default `k = 60`. Confidence is normalized against the mode's theoretical ceiling.

---

## Measured Evaluation Benchmark

Evaluation performed using `graphrag eval` over the bundled benchmark gold QA datasets (k=5):

### Bundled Benchmark Corpus (`data/corpus`, 26 questions)

| Retrieval Mode | Hit@5 | MRR | Context Precision@5 | Faithfulness | Answer Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| `dense` | 0.846 | 0.772 | 0.171 | 1.000 | 26/26 |
| `bm25` | 0.923 | 0.846 | 0.244 | 1.000 | 26/26 |
| `hybrid` | 0.885 | 0.804 | 0.177 | 1.000 | 26/26 |
| **`hybrid_graph`** | **1.000** | **0.868** | **0.200** | **1.000** | **26/26** |

### Technical Documentation Corpus (`data/tech_corpus`, 8 questions)

| Retrieval Mode | Hit@5 | MRR | Context Precision@5 | Faithfulness | Answer Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| `dense` | 0.750 | 0.594 | 0.200 | 1.000 | 8/8 |
| `bm25` | 0.750 | 0.656 | 0.181 | 1.000 | 8/8 |
| `hybrid` | 0.750 | 0.667 | 0.200 | 1.000 | 8/8 |
| **`hybrid_graph`** | **0.750** | **0.667** | **0.175** | **1.000** | **8/8** |

---

## Installation & Local Development

### Prerequisites
- Python 3.12+
- Git

### Setup

```powershell
# Clone the repository
git clone https://github.com/m4n1kya/FARLOOM.git
cd FARLOOM

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install package in editable mode with development tools
pip install -e ".[dev]"

# Configure environment
Copy-Item .env.example .env

# Launch development server
python -m uvicorn graphrag_knowledge_engine.api:create_app --factory --host 127.0.0.1 --port 8000 --reload
```

Visit `http://localhost:8000` to interact with the workspace.

---

## Environment Variables

| Variable | Default | Description |
|:---|:---:|:---|
| `GRAPH_BACKEND` | `networkx` | Graph engine: `networkx` (in-memory) or `neo4j` (persistent) |
| `NEO4J_URI` | — | Connection URI for Neo4j AuraDB (`neo4j+s://...`) |
| `NEO4J_USERNAME` | `neo4j` | Neo4j database username |
| `NEO4J_PASSWORD` | — | Neo4j database password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `GRAPHRAG_PROVIDER` | `fake` | LLM backend: `fake` (offline extractive), `gemini`, or `openai` |
| `GEMINI_API_KEY` | — | Google Gemini API key (free tier available at AI Studio) |
| `GRAPHRAG_GEMINI_MODEL` | `gemini-2.0-flash-lite` | Target Gemini model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GRAPHRAG_DATA_DIR` | `artifacts/runtime` | Storage path for SQLite databases and ChromaDB collections |
| `GRAPHRAG_MIN_CONFIDENCE` | `0.4` | Strict retrieval confidence threshold before refusal |

---

## API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Service health status and database connectivity |
| `GET` | `/status` | Active provider, graph backend, space directory, and demo-mode status |
| `GET` | `/spaces` | List all isolated Knowledge Spaces with document & graph metrics |
| `POST` | `/spaces` | Provision a new isolated Knowledge Space |
| `DELETE` | `/spaces/{space_id}` | Delete a Knowledge Space and purge its indexed data |
| `GET` | `/docs/list` | List all ingested documents in the target space |
| `POST` | `/upload` | Multipart upload for `.pdf`, `.md`, `.txt` files into target space |
| `DELETE` | `/docs/{doc_id}` | Cascade-delete a document from SQLite, ChromaDB, and Neo4j |
| `GET` | `/ask?q=&mode=` | Hybrid GraphRAG answer synthesis with citations |
| `GET` | `/graph/overview` | Curated hub subgraph for visualization |
| `GET` | `/graph/neighbors` | Ego-network neighborhood traversal |
| `GET` | `/graph/path` | Shortest semantic path tracing between two entities |
| `POST` | `/eval/run` | Execute benchmark evaluation across retrieval modes |
| `GET` | `/docs` | Interactive OpenAPI / Swagger UI |

---

## Testing & Quality Assurance

```powershell
# Run the complete test suite (104 tests)
pytest

# Verbose output
pytest -v

# Code quality and formatting check
ruff check .

# Static type verification
mypy src
```

---

## Deployment

### Render (Live Web Service)
A turn-key [`render.yaml`](render.yaml) is included in the repository.

1. Connect `m4n1kya/FARLOOM` to Render as a **Blueprint**.
2. Render provisions a Python 3.12 Web Service automatically.
3. Configure your `NEO4J_URI` and secrets in the Render Environment dashboard.

### Docker Container
```powershell
# Build Docker image
docker build -t farloom:latest .

# Run container
docker run -p 8000:7860 -e GRAPH_BACKEND=neo4j -e NEO4J_URI=neo4j+s://... farloom:latest
```

---

## Author & License

- **Author**: Manikya N. ([@m4n1kya](https://github.com/m4n1kya))
- **License**: MIT License — see [LICENSE](LICENSE) for details.
