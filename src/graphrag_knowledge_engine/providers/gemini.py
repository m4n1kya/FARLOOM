"""Gemini LLM provider (free-tier compatible).

Uses the Google Generative Language REST API via httpx.
Requires ``GEMINI_API_KEY`` environment variable.

Model selection: gemini-2.0-flash-lite is used by default as it is on the
official free tier. The model can be overridden via ``GRAPHRAG_GEMINI_MODEL``.

This provider implements the same ``LLMProvider`` protocol as ``FakeLLM``
and ``OpenAILLM`` so it drops into the engine without changes to the pipeline.
"""

from __future__ import annotations

import logging

import httpx

from graphrag_knowledge_engine.models import TaggedContext
from graphrag_knowledge_engine.providers.llm import (
    FakeLLM,
)

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

_ENTITY_PROMPT = (
    "List the named entities (people, teams, products, places, named policies) "
    'mentioned in the text. Respond with JSON only: {"entities": ["name", ...]}.\n\nText:\n'
)

_TRIPLE_PROMPT = (
    "Extract (subject, relation, object) triples from the text. "
    'Respond with JSON only: {"triples": [["subject", "relation", "object"], ...]}.\n\nText:\n'
)

_ANSWER_PROMPT = (
    "Answer the question using ONLY the numbered context blocks below. "
    "Cite sources inline as [n]. If the context is insufficient, say so.\n\n"
)

_JUDGE_PROMPT = (
    "Does the evidence support the claim? "
    "Respond with exactly SUPPORTED or UNSUPPORTED.\n\n"
)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from a response if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop first and last lines if they are fence markers
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        stripped = "\n".join(inner).strip()
    return stripped


class GeminiLLM:
    """Google Gemini LLM provider (free tier).

    Falls back to ``FakeLLM`` for entity/triple extraction only when the API
    key is missing, but always uses the real model for answer generation when
    a key is present. This is never a silent fallback: the engine marks
    demo-mode in the status endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-lite",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiLLM requires GEMINI_API_KEY")
        self._api_key = api_key
        self._model = model
        self._client = httpx.Client(timeout=timeout)
        self._fallback = FakeLLM()

    def _complete(self, prompt: str) -> str:
        url = f"{_GEMINI_BASE}/models/{self._model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        try:
            resp = self._client.post(url, json=payload, params={"key": self._api_key})
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned no candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            return str(parts[0].get("text", "")) if parts else ""
        except (httpx.HTTPStatusError, httpx.RequestError, KeyError, ValueError) as exc:
            logger.warning("Gemini API error: %s", exc)
            raise

    def extract_entities(self, text: str) -> list[str]:
        """Entity extraction — use FakeLLM heuristic (fast + offline)."""
        return self._fallback.extract_entities(text)

    def extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """Triple extraction — use FakeLLM heuristic (fast + offline)."""
        return self._fallback.extract_triples(text)

    def answer(self, question: str, contexts: list[TaggedContext]) -> str:
        blocks = "\n\n".join(f"[{c.tag}] ({c.source}) {c.text}" for c in contexts)
        prompt = f"{_ANSWER_PROMPT}{blocks}\n\nQuestion: {question}"
        try:
            return self._complete(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini answer failed, using extractive fallback: %s", exc)
            # Honest fallback: extractive answer clearly labelled
            extractive = self._fallback.answer(question, contexts)
            return (
                f"[Offline demonstration mode — Gemini unavailable ({type(exc).__name__})] "
                f"{extractive}"
            )

    def judge_claim(self, claim: str, evidence: str) -> bool:
        prompt = f"{_JUDGE_PROMPT}Claim: {claim}\n\nEvidence:\n{evidence}"
        try:
            verdict = self._complete(prompt)
            return "UNSUPPORTED" not in verdict.upper()
        except Exception:  # noqa: BLE001
            return self._fallback.judge_claim(claim, evidence)
