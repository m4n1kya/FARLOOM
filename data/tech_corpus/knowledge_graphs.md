# Knowledge Graphs and NetworkX — Technical Reference

## Knowledge Graphs

A knowledge graph represents information as entities (nodes) and typed
relationships (edges), forming a directed graph:

```
Entity A --[RELATION]--> Entity B
```

Examples:
- `FastAPI --[DEPENDS_ON]--> Starlette`
- `ChromaDB --[USES]--> HNSW`
- `uv --[REPLACES]--> pip`
- `Pydantic v2 --[INCOMPATIBLE_WITH]--> Pydantic v1`

Knowledge graphs enable multi-hop reasoning: to answer "What does FastAPI
depend on?" the system traverses edges outward from the FastAPI node.

## Graph-Enhanced RAG

In GraphRAG, the knowledge graph augments standard vector retrieval:

1. **Entity detection**: Identify entities named in the query.
2. **Graph expansion**: BFS from seed entities to neighboring nodes.
3. **Chunk scoring**: Rank chunks by their entity mention provenance.
4. **Fusion**: Combine graph-ranked chunks with dense and BM25 results via RRF.

The graph leg DISCOVERS evidence that semantic similarity might miss. For example,
"Which framework replaced Tornado in the Aurora project?" requires knowing that
Aurora USES FastAPI, and FastAPI REPLACED Tornado — two hops.

## Entity Extraction

Named Entity Recognition (NER) detects entities in text. Approaches:

- **Rule-based**: Capitalized noun phrases (fast, offline, low recall)
- **Statistical**: CRF or spaCy models (better recall, requires model download)
- **LLM-based**: GPT/Gemini prompted to extract entities as JSON (best recall, requires API)

FARLOOM's FakeLLM uses rule-based capitalized noun-phrase detection:
capitalized words that are not sentence-initial stopwords become entity candidates.

## Triple Extraction

A triple is a (subject, relation, object) fact:

```
(FastAPI, depends_on, Starlette)
(ChromaDB, uses, HNSW)
```

Triples are extracted per sentence using entity co-occurrence: for each pair of
entities in a sentence, the text between them becomes the relation label.

## NetworkX

NetworkX is a Python library for creating, analyzing, and visualizing graphs.

### Installation

```bash
pip install networkx
```

### Graph Types

- `Graph` — undirected, simple edges
- `DiGraph` — directed, simple edges
- `MultiGraph` — undirected, multiple edges between nodes
- `MultiDiGraph` — directed, multiple edges (used by FARLOOM for triple storage)

### Basic Operations

```python
import networkx as nx

G = nx.MultiDiGraph()

# Add nodes
G.add_node("FastAPI", label="FastAPI")
G.add_node("Starlette", label="Starlette")

# Add edges
G.add_edge("FastAPI", "Starlette", relation="depends_on", chunk_ids={"chunk_001"})

# Traverse
for neighbor in G.neighbors("FastAPI"):
    print(neighbor)

# BFS
from collections import deque
queue = deque(["FastAPI"])
visited = {"FastAPI"}
while queue:
    node = queue.popleft()
    for n in G.neighbors(node):
        if n not in visited:
            visited.add(n)
            queue.append(n)
```

### Graph Properties

```python
G.number_of_nodes()   # entity count
G.number_of_edges()   # relationship count
G.degree("FastAPI")   # number of edges connected to a node
nx.is_directed(G)     # True for DiGraph / MultiDiGraph
```

### Cycle Detection

```python
# Check for cycles
has_cycle = not nx.is_directed_acyclic_graph(G)

# Safe traversal with depth limit
depth = {}
queue = deque([seed])
depth[seed] = 0
while queue:
    node = queue.popleft()
    if depth[node] >= max_depth:
        continue
    for neighbor in G.neighbors(node):
        if neighbor not in depth:
            depth[neighbor] = depth[node] + 1
            queue.append(neighbor)
```

Depth limiting PREVENTS unbounded traversal in cyclic graphs.

## Graph Traversal Scoring

FARLOOM scores chunks retrieved via graph expansion:

```
weight(entity) = 1 / (1 + hop) / sqrt(mention_count)
```

- `1 / (1 + hop)`: Closer entities get higher weight (hop=0 → weight=1.0).
- `sqrt(mention_count)` IDF-style discount: entities mentioned in many chunks
  are weaker signals than rare entities.

Final chunk score: `closeness + focus`
- `closeness`: maximum single entity vote the chunk received
- `focus`: total vote mass / number of connected entities mentioning the chunk

This scoring prefers chunks that are specifically about the seed entities
over "hub" chunks that mention many unrelated entities.

## Entity Normalization

Entity normalization prevents duplicate nodes for the same entity:

```python
def normalize(name: str) -> str:
    return " ".join(name.casefold().split())

normalize("FastAPI") == normalize("fastapi")  # True
```

FARLOOM stores the casefolded form as the node key and the original display label
as a node attribute, so "FastAPI" and "fastapi" map to the same graph node.

## Knowledge Graph vs Vector Search

| Property | Vector Search | Knowledge Graph |
|----------|---------------|-----------------|
| Query type | Semantic similarity | Entity relationships |
| Evidence | Embedding proximity | Graph path traversal |
| Multi-hop | No | Yes |
| Offline | Yes (with FakeEmbedder) | Yes (NetworkX) |
| Entity linking | Implicit | Explicit |
| Dependency questions | Poor | Excellent |
