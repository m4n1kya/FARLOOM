"""Answer synthesis with inline citations and low-confidence refusal.

The answerer takes the fused retrieval result, refuses outright when the
normalized retrieval confidence is under ``min_confidence`` (or nothing was
retrieved), and otherwise hands the top chunks to the LLM as numbered context
blocks. The
LLM cites blocks inline as [1], [2], ...; the response maps each tag back to
its chunk, source file, section, and page.
"""

from __future__ import annotations

from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.models import (
    AskResponse,
    Citation,
    RetrievalDebug,
    TaggedContext,
)
from graphrag_knowledge_engine.providers.llm import LLMProvider
from graphrag_knowledge_engine.store.documents import DocumentStore

REFUSAL_MESSAGE = (
    "I can't answer that confidently: retrieval found no sufficiently relevant "
    "context in the indexed documents."
)


class AnswerSynthesizer:
    def __init__(self, config: EngineConfig, llm: LLMProvider, store: DocumentStore) -> None:
        self._config = config
        self._llm = llm
        self._store = store

    def answer(self, question: str, retrieval: RetrievalDebug) -> AskResponse:
        if not retrieval.fused or retrieval.confidence < self._config.min_confidence:
            return AskResponse(
                question=question,
                answer=REFUSAL_MESSAGE,
                refused=True,
                citations=[],
                debug=retrieval,
            )

        top = retrieval.fused[: self._config.context_size]
        chunks = self._store.get_chunks([c.chunk_id for c in top])
        contexts: list[TaggedContext] = []
        for candidate in top:
            chunk = chunks.get(candidate.chunk_id)
            if chunk is None:
                continue
            contexts.append(
                TaggedContext(
                    tag=len(contexts) + 1,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    section=chunk.section,
                    page=chunk.page,
                )
            )

        answer_text = self._llm.answer(question, contexts)
        cited_tags = {c.tag for c in contexts if f"[{c.tag}]" in answer_text}
        citations = [
            Citation(
                tag=c.tag,
                chunk_id=c.chunk_id,
                source=c.source,
                section=c.section,
                page=c.page,
                snippet=c.text[:160],
            )
            for c in contexts
            if c.tag in cited_tags
        ]
        return AskResponse(
            question=question,
            answer=answer_text,
            refused=False,
            citations=citations,
            debug=retrieval,
        )
