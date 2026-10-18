# FARLOOM — Technical Knowledge Intelligence Engine

> **Weave Knowledge. Reveal Connections.**

FARLOOM is an AI-powered technical knowledge system that ingests documentation,
constructs a knowledge graph, performs hybrid retrieval across multiple evidence
streams, and answers complex questions with traceable citations.

FARLOOM weaves vector search, BM25 keyword matching, and a knowledge graph
into a unified retrieval engine with an integrated web workspace.

---

## Architecture

```
Documents (PDF / Markdown / TXT)
         │
         ▼
    ┌─────────────┐
    │   Ingest    │  load → chunk → embed → extract entities+triples
    └──────┬──────┘
           │
     ┌─────┼──────────────────────┐
     ▼     ▼                      ▼
  ChromaDB SQLite    NetworkX / Neo4j AuraDB
  (vectors) (chunks) (knowledge graph backend)
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
   - Verifies driver connectivity on startup and reports state in `/health/ready` and `/status`.
   - Idempotently creates uniqueness constraints: `CREATE CONSTRAINT entity_uid IF NOT EXISTS FOR (e:Entity) REQUIRE e.uid IS UNIQUE`.
   - Creates composite indexes on `(space_id, key)` for multi-tenant query performance.
   - Syncs existing triples and chunk provenance without duplicates using parameterized `MERGE`.

### AuraDB Inactivity and Resumption

Free-tier Neo4j AuraDB instances automatically pause after periods of inactivity. FARLOOM includes built-in detection for paused instances:
- System diagnostics report instance status and suggest resuming via the console.
- The web UI displays an interactive notice with a direct link to `https://console.neo4j.io`.
- Inability to reach Neo4j returns a clear 503 on `/health/ready` rather than failing silently or presenting a false healthy state.


## Retrieval Modes

| Mode | Legs | Best for |
|------|------|----------|
| `dense` | Vector search (ChromaDB) | Semantic similarity |
| `bm25` | Okapi BM25 | Exact terms, identifiers, version strings |
| `hybrid` | dense + BM25 via RRF | General purpose |
| `hybrid_graph` | dense + BM25 + graph expansion | Multi-hop reasoning |

## How Retrieval Works

### Dense (vector) retrieval
Feature-hashing embeddings (512-dim, offline) stored in ChromaDB.
Cosine similarity query returns semantically related passages.

### BM25 retrieval
Okapi BM25 over tokenized chunk text. Excels at exact package names,
API identifiers, version strings, and error codes that semantic similarity misses.

### Graph expansion
1. Named entities in the query are matched to graph nodes (longest-match, case-insensitive).
2. Bounded BFS from seed nodes (configurable depth + node limit).
3. Each visited entity votes for its source chunks with weight `1/(1+hop)/sqrt(mention_count)`.
4. Chunks ranked by closeness + focus score.

### Reciprocal Rank Fusion (RRF)
```
RRF_score(chunk) = Σ  1 / (k + rank_in_leg)
                  legs
```
Default `k = 60`. Confidence is normalized against the mode's theoretical ceiling.

---

## Installation (Windows PowerShell)

```powershell
# Clone and enter the project
git clone https://github.com/m4n1kya/FARLOOM.git
cd FARLOOM

# Create a Python 3.12 virtual environment
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e ".[dev]"
```

## Local Development

```powershell
# Copy example environment
Copy-Item .env.example .env

# Start the server (offline demo mode by default)
.venv\Scripts\python.exe -m uvicorn graphrag_knowledge_engine.api:app `
    --host 127.0.0.1 --port 8000 --reload

# Visit http://localhost:8000
```

The bundled corpus is auto-ingested on first startup if the index is empty.

## No-Key Demonstration Mode

No API key is required. With `GRAPHRAG_PROVIDER=fake`:

- Ingestion, chunking, embedding, BM25 indexing, and graph construction all work.
- Answers are **extractive**: scored sentences from retrieved passages, with `[n]` citations.
- All four retrieval modes work and can be compared.
- Evaluation (`/eval/run`) runs and produces real metrics.
- Graph explorer shows real extracted entities and relationships.

**Status shown in the dashboard:** *Offline demonstration mode — no generative model configured.*

## Optional: Gemini Free Tier (Real LLM)

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Set in `.env`:
   ```
   GRAPHRAG_PROVIDER=gemini
   GEMINI_API_KEY=your_key_here
   GRAPHRAG_GEMINI_MODEL=gemini-2.0-flash-lite
   ```
3. Restart the server.

If the Gemini API is unavailable, the engine falls back to extractive answers with
a clear disclosure message — the application never silently returns a fake LLM response.

## Optional: OpenAI

```
GRAPHRAG_PROVIDER=openai
OPENAI_API_KEY=your_key_here
GRAPHRAG_OPENAI_EMBED_MODEL=text-embedding-3-small
GRAPHRAG_OPENAI_CHAT_MODEL=gpt-4o-mini
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHRAG_PROVIDER` | `fake` | `fake`, `gemini`, or `openai` |
| `GRAPHRAG_DATA_DIR` | `artifacts/runtime` | Runtime index directory |
| `GRAPHRAG_MIN_CONFIDENCE` | `0.4` | Refusal threshold (0–1) |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GRAPHRAG_GEMINI_MODEL` | `gemini-2.0-flash-lite` | Gemini model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `GRAPHRAG_OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `GRAPHRAG_OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `GRAPHRAG_HOST` | `127.0.0.1` | Bind host for `graphrag serve` |
| `GRAPHRAG_PORT` | `8000` | Bind port |

## Example Questions (bundled corpus)

Original corpus (Aurora Handbook, Helios Spec, Meridian Policy):
- "What rate limit does Beacon API enforce per customer agent?"
- "Who owns the Meridian Policy?"
- "Which engineering organization is led from the campus hosting the primary Pulse Engine cluster?"
- "How long are Pulse Engine telemetry archives retained?"

Technical documentation corpus (Python, FastAPI, ChromaDB, NetworkX):
- "What is the difference between pip and uv?"
- "How does ChromaDB handle cosine distance vs similarity?"
- "Which FastAPI handler replaces on_event startup hooks?"
- "What does BM25 excel at that vector search fails on?"
- "How does FARLOOM's graph expansion scoring work?"

## Running Tests

```powershell
# All tests
.venv\Scripts\pytest -q

# With verbose output
.venv\Scripts\pytest -v

# Single test file
.venv\Scripts\pytest tests/test_farloom_api.py -v
```

## Lint and Type Checking

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
```

## Evaluation

```powershell
# Via CLI
.venv\Scripts\graphrag eval `
    --corpus data/corpus `
    --gold data/gold/qa_pairs.json `
    --k 5

# Via demo script
.venv\Scripts\python scripts/demo.py
```

### Measured Evaluation Results (fake provider, k=5)

#### Bundled Benchmark Corpus (`data/corpus`, 26 questions)

| config | hit@5 | MRR | ctx_precision@5 | faithfulness | answered |
|--------|-------|-----|-----------------|--------------|----------|
| dense | 0.846 | 0.772 | 0.171 | 1.000 | 26/26 |
| bm25 | 0.923 | 0.846 | 0.244 | 1.000 | 26/26 |
| hybrid | 0.885 | 0.804 | 0.177 | 1.000 | 26/26 |
| **hybrid_graph** | **1.000** | **0.868** | **0.200** | **1.000** | **26/26** |

*Measured using `graphrag eval --corpus data/corpus --gold data/gold/qa_pairs.json --k 5`.*

#### Technical Documentation Corpus (`data/tech_corpus`, 8 questions)

| config | hit@5 | MRR | ctx_precision@5 | faithfulness | answered |
|--------|-------|-----|-----------------|--------------|----------|
| dense | 0.750 | 0.594 | 0.200 | 1.000 | 8/8 |
| bm25 | 0.750 | 0.656 | 0.181 | 1.000 | 8/8 |
| hybrid | 0.750 | 0.667 | 0.200 | 1.000 | 8/8 |
| **hybrid_graph** | **0.750** | **0.667** | **0.175** | **1.000** | **8/8** |

*Measured using `graphrag eval --corpus data/tech_corpus --gold data/gold/tech_qa_pairs.json --k 5`.*

## CLI Commands

```powershell
# Ingest a document
.venv\Scripts\graphrag ingest data/corpus/aurora_handbook.md --strategy heading

# Ask a question
.venv\Scripts\graphrag ask "What is Helios Sync?" --mode hybrid_graph

# Explore graph neighbors
.venv\Scripts\graphrag neighbors "Helios Sync" --depth 2

# Run evaluation
.venv\Scripts\graphrag eval --corpus data/corpus --gold data/gold/qa_pairs.json --k 5

# Start API server
.venv\Scripts\graphrag serve --host 127.0.0.1 --port 8000
```

## Deployment (Render Free Tier)

`render.yaml` is pre-configured for Render's free web service tier.

1. Push to your GitHub repository.
2. In Render: New → Web Service → connect your `FARLOOM` repo.
3. Render auto-detects `render.yaml`.
4. Optionally set `GEMINI_API_KEY` in Render's Environment → Secrets.
5. Deploy.

### Free-Tier Limitations

- **Ephemeral filesystem**: The runtime index (`/tmp/farloom-runtime`) resets on restart.
  The bundled corpus is automatically re-ingested on each cold start.
- **Spin-down**: Free Render services sleep after 15 minutes of inactivity.
  First request after sleep takes ~30 seconds (index rebuild).
- **No persistent uploads**: Files uploaded through the API are not retained after restart.
- **CPU only**: Inference uses the fake (offline) or Gemini free-tier provider.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (returns process & graph status) |
| `GET` | `/health/ready` | Readiness check (fails with 503 if graph backend is unreachable) |
| `GET` | `/status` | Active provider, graph backend, space directory, and demo-mode status |
| `GET` | `/spaces` | List all isolated Knowledge Spaces |
| `POST` | `/spaces` | Create a new isolated Knowledge Space |
| `DELETE`| `/spaces/{space_id}` | Delete an isolated Knowledge Space and its index |
| `POST` | `/spaces/{space_id}/upload` | Upload and ingest a document into a Knowledge Space |
| `POST` | `/upload` | Upload and ingest a document with new or target space selection |
| `POST` | `/ingest` | Ingest a local filesystem path |
| `GET` | `/ask?q=&mode=` | Hybrid GraphRAG answer synthesis with citations |
| `GET` | `/graph/neighbors?entity=&depth=` | Ego-network graph neighbors |
| `GET` | `/graph/stats` | Entity/edge counts and graph nodes |
| `GET` | `/graph/entities?q=&limit=` | Search entity labels |
| `GET` | `/docs/list` | List ingested documents in the target space |
| `GET` | `/chunks/count` | Total indexed chunk count in the target space |
| `POST` | `/eval/run` | Run evaluation benchmark |
| `GET` | `/docs` | OpenAPI / Swagger interactive UI |


## Known Limitations

- **Fake embeddings**: The offline FakeEmbedder uses feature hashing, not a pre-trained
  language model. Dense retrieval quality is lower than OpenAI/Gemini embeddings.
- **Graph extraction**: FakeLLM uses capitalized noun-phrase heuristics, not NER.
  Multi-word entities may be split at line boundaries in some documents.
- **Gemini embeddings**: No free embedding API is available; Gemini mode uses FakeEmbedder
  for the dense leg. BM25 and graph legs are unaffected.
- **File upload persistence**: Not implemented in the public version.
- **Concurrent writes**: Single-writer SQLite; not suitable for high-concurrency production.

## License

MIT License. See [LICENSE](LICENSE) for the full text.

