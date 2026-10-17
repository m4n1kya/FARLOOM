"""Answer faithfulness: LLM-judged claim support against cited context.

The answer is split into claim sentences (citation markers stripped), and the
LLM judges each claim against the concatenated text of the chunks the answer
actually cited. Faithfulness = supported claims / total claims. With the
FakeLLM judge this degrades to deterministic content-word containment, which
keeps the whole eval harness offline.
"""

from __future__ import annotations

import re

from graphrag_knowledge_engine.models import AskResponse
from graphrag_knowledge_engine.providers.llm import LLMProvider
from graphrag_knowledge_engine.store.documents import DocumentStore

_CITATION_RE = re.compile(r"\s*\[\d+\]")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def split_claims(answer: str) -> list[str]:
    """Sentences of the answer with [n] markers removed."""
    cleaned = _CITATION_RE.sub("", answer)
    return [s.strip() for s in _SENTENCE_RE.findall(cleaned) if s.strip()]


def score_faithfulness(
    response: AskResponse, llm: LLMProvider, store: DocumentStore
) -> float:
    """Fraction of answer claims the judge finds supported by cited chunks."""
    if response.refused or not response.citations:
        return 0.0
    chunk_ids = [c.chunk_id for c in response.citations]
    chunks = store.get_chunks(chunk_ids)
    evidence = "\n".join(chunks[cid].text for cid in chunk_ids if cid in chunks)
    claims = split_claims(response.answer)
    if not claims or not evidence:
        return 0.0
    supported = sum(1 for claim in claims if llm.judge_claim(claim, evidence))
    return supported / len(claims)
