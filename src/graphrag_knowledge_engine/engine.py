"""Engine: wires providers, stores, retrieval, and answering together.

``create_engine`` picks providers from config: deterministic fakes by default
(fully offline), OpenAI implementations when ``provider="openai"``, or Gemini
(free tier) when ``provider="gemini"``. The rest of the pipeline is identical
either way - that symmetry is the point.
"""

from __future__ import annotations

from pathlib import Path

from graphrag_knowledge_engine.answer import AnswerSynthesizer
from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.ingest.chunkers import get_chunker
from graphrag_knowledge_engine.ingest.loaders import load_document
from graphrag_knowledge_engine.models import (
    AskResponse,
    ChunkStrategy,
    Document,
    IngestResponse,
    NeighborsResponse,
    RetrievalMode,
    Triple,
)
from graphrag_knowledge_engine.providers.embeddings import (
    EmbeddingProvider,
    FakeEmbedder,
    OpenAIEmbedder,
)
from graphrag_knowledge_engine.providers.llm import FakeLLM, LLMProvider, OpenAILLM
from graphrag_knowledge_engine.retrieval.bm25 import Bm25Index
from graphrag_knowledge_engine.retrieval.retriever import HybridRetriever
from graphrag_knowledge_engine.store.base import BaseGraphStore
from graphrag_knowledge_engine.store.documents import DocumentStore
from graphrag_knowledge_engine.store.factory import create_graph_store
from graphrag_knowledge_engine.store.vectors import VectorIndex


class Engine:
    def __init__(
        self,
        config: EngineConfig,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        space_id: str = "default",
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.llm = llm
        self.space_id = space_id
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = DocumentStore(config.data_dir / "store.sqlite")
        self.vectors = VectorIndex(config.data_dir / "chroma")
        self.graph: BaseGraphStore = create_graph_store(
            config=config,
            space_id=space_id,
            triples=self.store.all_triples(),
            mentions=self.store.all_mentions(),
        )
        self._bm25 = self._build_bm25()
        self._retriever = self._build_retriever()
        self._answerer = AnswerSynthesizer(config, llm, self.store)

    def _build_bm25(self) -> Bm25Index:
        return Bm25Index([(c.chunk_id, c.text) for c in self.store.all_chunks()])

    def _build_retriever(self) -> HybridRetriever:
        return HybridRetriever(self.config, self.embedder, self.vectors, self._bm25, self.graph)

    # -- ingestion -----------------------------------------------------------

    def ingest_path(self, path: Path, strategy: ChunkStrategy = "recursive") -> IngestResponse:
        loaded = load_document(path)
        chunker = get_chunker(
            strategy, chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap
        )
        chunks = chunker.chunk(loaded)
        doc = Document(
            doc_id=loaded.doc_id,
            source=loaded.source,
            title=loaded.title,
            fmt=loaded.fmt,
            num_chunks=len(chunks),
        )
        self.store.upsert_document(doc, chunks)

        # Dense leg: embed and upsert into ChromaDB.
        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        self.vectors.add([c.chunk_id for c in chunks], embeddings)

        # Graph leg: extract entity mentions and triples per chunk, persist both.
        triples: list[Triple] = []
        mentions: list[tuple[str, str]] = []
        for chunk in chunks:
            for entity in self.llm.extract_entities(chunk.text):
                mentions.append((entity, chunk.chunk_id))
            for subject, relation, obj in self.llm.extract_triples(chunk.text):
                triples.append(
                    Triple(
                        subject=subject,
                        relation=relation,
                        obj=obj,
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                    )
                )
        self.store.add_triples(triples)
        self.store.add_mentions(mentions)
        self.graph.add_triples(triples)
        self.graph.add_mentions(mentions)

        # Sparse leg: rebuild BM25 over the updated corpus.
        self._bm25 = self._build_bm25()
        self._retriever = self._build_retriever()

        return IngestResponse(
            doc_id=doc.doc_id,
            source=doc.source,
            num_chunks=len(chunks),
            num_triples=len(triples),
            strategy=strategy,
        )

    def delete_document(self, doc_id_or_source: str) -> bool:
        """Delete a document and atomically clean its chunks, vectors, and graph elements."""
        doc = self.store.get_document(doc_id_or_source)
        if not doc:
            return False

        chunk_ids = self.store.get_chunk_ids_for_doc(doc.doc_id)
        # 1. Cascade-delete from SQLite store
        self.store.delete_document(doc.doc_id)
        # 2. Delete vectors from ChromaDB
        self.vectors.delete(chunk_ids)
        # 3. Clean graph edges & orphaned nodes
        self.graph.remove_chunks(chunk_ids)
        # 4. Rebuild BM25 sparse index & hybrid retriever
        self._bm25 = self._build_bm25()
        self._retriever = self._build_retriever()
        return True

    # -- querying ------------------------------------------------------------

    def ask(self, question: str, mode: RetrievalMode = "hybrid_graph") -> AskResponse:
        retrieval = self._retriever.retrieve(question, mode)
        return self._answerer.answer(question, retrieval)

    def neighbors(self, entity: str, depth: int = 1) -> NeighborsResponse:
        return NeighborsResponse(
            entity=entity,
            found=self.graph.has_entity(entity),
            edges=self.graph.neighbors(entity, depth=depth),
        )

    def close(self) -> None:
        self.store.close()
        self.graph.close()


def create_engine(config: EngineConfig, space_id: str = "default") -> Engine:
    """Construct an engine with providers chosen by config."""
    embedder: EmbeddingProvider
    llm: LLMProvider
    if config.provider == "openai":
        embedder = OpenAIEmbedder(
            api_key=config.openai_api_key,
            model=config.openai_embed_model,
            base_url=config.openai_base_url,
        )
        llm = OpenAILLM(
            api_key=config.openai_api_key,
            model=config.openai_chat_model,
            base_url=config.openai_base_url,
        )
    elif config.provider == "gemini":
        from graphrag_knowledge_engine.providers.gemini import GeminiLLM
        # Gemini uses FakeEmbedder for now (no free embedding API).
        # Retrieval still works via BM25 + graph legs.
        embedder = FakeEmbedder(dim=config.embed_dim)
        llm = GeminiLLM(
            api_key=config.gemini_api_key,
            model=config.gemini_chat_model,
        )
    else:
        embedder = FakeEmbedder(dim=config.embed_dim)
        llm = FakeLLM()
    return Engine(config, embedder, llm, space_id=space_id)

