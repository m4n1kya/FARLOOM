"""Tests for the Gemini LLM provider (mocked HTTP responses)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag_knowledge_engine.models import TaggedContext
from graphrag_knowledge_engine.providers.gemini import GeminiLLM, _strip_fences


def test_strip_fences_no_fences() -> None:
    assert _strip_fences("hello world") == "hello world"


def test_strip_fences_with_json_fence() -> None:
    raw = "```json\n{\"entities\": []}\n```"
    assert _strip_fences(raw) == '{"entities": []}'


def test_gemini_requires_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLLM(api_key="")


def test_gemini_extract_entities_uses_fallback() -> None:
    """Entity extraction falls back to FakeLLM offline heuristic."""
    llm = GeminiLLM.__new__(GeminiLLM)
    from graphrag_knowledge_engine.providers.llm import FakeLLM
    llm._fallback = FakeLLM()
    llm._api_key = "dummy"
    llm._model = "gemini-2.0-flash-lite"

    import httpx
    llm._client = httpx.Client()

    entities = llm.extract_entities("Aurora Dynamics is headquartered in Portland.")
    assert "Aurora Dynamics" in entities


def test_gemini_answer_success() -> None:
    """Answer method calls Gemini API and returns the model response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "The answer is [1]."}]}}
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    llm = GeminiLLM.__new__(GeminiLLM)
    from graphrag_knowledge_engine.providers.llm import FakeLLM
    llm._fallback = FakeLLM()
    llm._api_key = "test-key"
    llm._model = "gemini-2.0-flash-lite"
    llm._client = MagicMock()
    llm._client.post.return_value = mock_resp

    contexts = [
        TaggedContext(
            tag=1,
            chunk_id="c1",
            text="Aurora Dynamics is in Portland.",
            source="aurora_handbook.md",
        )
    ]
    result = llm.answer("Where is Aurora Dynamics?", contexts)
    assert result == "The answer is [1]."


def test_gemini_answer_fallback_on_api_error() -> None:
    """On API failure, falls back to extractive answer with disclosure."""
    import httpx
    llm = GeminiLLM.__new__(GeminiLLM)
    from graphrag_knowledge_engine.providers.llm import FakeLLM
    llm._fallback = FakeLLM()
    llm._api_key = "test-key"
    llm._model = "gemini-2.0-flash-lite"
    llm._client = MagicMock()
    llm._client.post.side_effect = httpx.RequestError("timeout")

    contexts = [
        TaggedContext(
            tag=1,
            chunk_id="c1",
            text="Aurora Dynamics is in Portland.",
            source="aurora_handbook.md",
        )
    ]
    result = llm.answer("Where is Aurora Dynamics?", contexts)
    assert "Offline demonstration mode" in result
    assert "RequestError" in result


def test_gemini_judge_claim_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "SUPPORTED"}]}}]
    }
    mock_resp.raise_for_status = MagicMock()

    llm = GeminiLLM.__new__(GeminiLLM)
    from graphrag_knowledge_engine.providers.llm import FakeLLM
    llm._fallback = FakeLLM()
    llm._api_key = "test-key"
    llm._model = "gemini-2.0-flash-lite"
    llm._client = MagicMock()
    llm._client.post.return_value = mock_resp

    result = llm.judge_claim("Aurora is in Portland", "Aurora Dynamics is headquartered in Portland.")
    assert result is True


def test_gemini_judge_claim_fallback() -> None:
    """Falls back to FakeLLM judge on API error."""
    import httpx
    llm = GeminiLLM.__new__(GeminiLLM)
    from graphrag_knowledge_engine.providers.llm import FakeLLM
    llm._fallback = FakeLLM()
    llm._api_key = "test-key"
    llm._model = "gemini-2.0-flash-lite"
    llm._client = MagicMock()
    llm._client.post.side_effect = httpx.RequestError("timeout")

    # FakeLLM judge: "Portland" is in evidence text
    result = llm.judge_claim("Portland", "Aurora Dynamics is headquartered in Portland.")
    assert isinstance(result, bool)
