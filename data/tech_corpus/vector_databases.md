# Vector Databases and ChromaDB — Technical Reference

## Vector Databases

A vector database stores high-dimensional embedding vectors and supports
approximate nearest-neighbor (ANN) search. It is the foundation of
retrieval-augmented generation (RAG) systems.

Common vector databases:
- **ChromaDB** — open source, embedded or client-server, Python-native
- **Pinecone** — managed cloud service
- **Weaviate** — open source, graph + vector hybrid
- **Qdrant** — open source, Rust-based, high performance
- **pgvector** — PostgreSQL extension for vector similarity

## ChromaDB

ChromaDB is an open-source embedding database designed for AI applications.
It stores embeddings, documents, and metadata together in a single collection.

### Installation

```bash
pip install chromadb
```

### Key Concepts

- **Collection**: A named group of embeddings (equivalent to a table).
- **Embedding**: A fixed-dimension float vector representing semantic meaning.
- **Distance function**: Cosine (default for most NLP), L2 (Euclidean), or inner product.
- **Metadata**: Arbitrary key-value pairs attached to each embedding.

### Embedded Mode (no server)

```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_data",
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)
```

`PersistentClient` persists data to disk. `EphemeralClient` stores data only in memory.

### Adding Embeddings

```python
collection.upsert(
    ids=["chunk_001", "chunk_002"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["Document text 1", "Document text 2"],
    metadatas=[{"source": "doc_a.md"}, {"source": "doc_b.md"}],
)
```

`upsert` INSERT-OR-REPLACE: if an ID already exists, it is overwritten.

### Querying

```python
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
    include=["documents", "distances", "metadatas"],
)
```

ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite).
Cosine similarity = 1 - cosine_distance.

### Embedding Functions

ChromaDB includes built-in embedding functions (e.g., `SentenceTransformerEmbeddingFunction`),
but using them downloads ONNX models automatically. For offline environments,
provide embeddings explicitly to bypass the built-in embedder.

```python
# Explicit embeddings: no model download
collection.upsert(ids=ids, embeddings=my_embeddings)
```

## HNSW Index

ChromaDB uses HNSW (Hierarchical Navigable Small World) for ANN search.
HNSW parameters:

- `hnsw:space` — distance metric: `cosine`, `l2`, `ip`
- `hnsw:ef_construction` — construction quality (default 100)
- `hnsw:M` — graph connectivity (default 16)

Higher `M` and `ef_construction` improve recall at the cost of memory and build time.

## Embedding Models

### Sentence Transformers

`all-MiniLM-L6-v2` is a popular free model producing 384-dim embeddings.
`all-mpnet-base-v2` produces 768-dim embeddings with better quality.

### OpenAI Embeddings

`text-embedding-3-small` produces 1536-dim embeddings (configurable via `dimensions` param).
`text-embedding-3-large` produces 3072-dim embeddings.
These REQUIRE an OpenAI API key and incur per-token costs.

### Embedding Compatibility

**Critical rule**: embeddings stored in an index MUST use the same model as
query embeddings. Mixing embedding models produces meaningless similarity scores.
When changing the embedding model, the index must be rebuilt completely.

## BM25 — Sparse Retrieval

BM25 (Best Match 25) is a probabilistic ranking function for exact keyword search.
Unlike vector search, BM25 does not require embeddings and works offline.

```python
from rank_bm25 import BM25Okapi

corpus = [["fast", "car"], ["slow", "car"], ["fast", "train"]]
bm25 = BM25Okapi(corpus)
query = ["fast"]
scores = bm25.get_scores(query)
```

BM25 EXCELS at: exact identifiers, package names, API names, version strings,
error codes, and technical terms where semantic similarity fails.

BM25 FAILS at: synonyms, paraphrasing, and out-of-vocabulary terms.

## Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from multiple retrieval legs without requiring
comparable score scales:

```
RRF_score(d) = sum(1 / (k + rank_in_leg(d)))
```

where `k` is a constant (typically 60) and the sum is over all legs.

Benefits:
- Insensitive to score normalization differences between legs
- Stable across query types
- Simple to implement

RRF is used in FARLOOM to combine dense vector, BM25, and graph expansion legs.

## Hybrid Retrieval Modes

| Mode | Legs | Use Case |
|------|------|----------|
| `dense` | Vector search only | Semantic similarity |
| `bm25` | BM25 only | Exact keyword match |
| `hybrid` | Dense + BM25 | General purpose |
| `hybrid_graph` | Dense + BM25 + Graph | Multi-hop reasoning |
