"""LLM provider boundary.

The protocol is *semantic*, not stringly-typed: callers ask for triples, an
answer, or a claim verdict, and each implementation owns its prompting (or, in
the fake's case, a deterministic algorithm). That keeps prompt templates out of
the core pipeline and makes the fake a drop-in replacement for tests, demos,
and the eval harness.

``FakeLLM`` behaviours (all deterministic, all offline):

* ``extract_entities`` - capitalized noun-phrase detection over unwrapped
  paragraphs. Hard-wrapped lines are joined per paragraph first so a name
  split across lines ("Pulse\\nEngine") still counts as one mention. Single
  capitalized words at the start of a sentence are ignored - that is
  ordinary sentence case, not a name - which keeps the graph free of noise
  nodes like "Expired" or "Staff".
* ``extract_triples`` - entity co-occurrence: within each sentence, every
  ordered pair of detected entities yields a triple whose relation is the
  cleaned lowercase text between the two mentions (falling back to
  ``related_to``).
* ``answer`` - extractive: scores context sentences by content-word overlap
  with the question and stitches the best ones together with [n] citations.
* ``judge_claim`` - claim content-words containment against the evidence text
  (>= 0.5 of claim tokens present => supported).
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from graphrag_knowledge_engine.models import TaggedContext
from graphrag_knowledge_engine.providers.embeddings import STOPWORDS, content_tokens

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+)*\b")
_RELATION_TOKEN_RE = re.compile(r"[a-z]+")

# Words that frequently start sentences but are not entities on their own.
_ENTITY_STOPWORDS = frozenset(
    {
        "The", "This", "That", "These", "Those", "It", "If", "In", "On", "At", "For",
        "And", "But", "Or", "All", "Any", "Each", "Every", "When", "Where", "While",
        "After", "Before", "During", "See", "Note", "Our", "We", "You", "Your", "They",
        "As", "By", "To", "From", "With", "Within", "Under", "Over", "No", "Not", "Do",
        "What", "Which", "Who", "How", "Why",
    }
)


def content_words(text: str) -> set[str]:
    """Tokens minus stopwords - the overlap unit used by the fake LLM."""
    return set(content_tokens(text))


def split_sentences(text: str) -> list[str]:
    """Sentences, with hard line-wraps unwrapped one paragraph at a time.

    Markdown prose is often wrapped at ~80 columns, which would otherwise cut
    sentences (and entity names) at line boundaries. Lines are joined within
    each blank-line-delimited paragraph before sentence splitting.
    """
    sentences: list[str] = []
    for block in _PARAGRAPH_RE.split(text):
        joined = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if joined:
            sentences.extend(_SENTENCE_RE.findall(joined))
    return sentences


def _sentence_mentions(sentence: str) -> list[tuple[int, int, str]]:
    """Entity mentions in one sentence as (start, end, cleaned_name).

    A single capitalized word at the very start of a sentence is ambiguous -
    it may simply be sentence case ("Expired data is purged...") - so it is
    only kept when the mention spans multiple words ("Pulse Engine is...").
    Heading-like lines (no terminal punctuation, e.g. "About the Company")
    are title case rather than prose, so only multi-word mentions count there.
    """
    lead = len(sentence) - len(sentence.lstrip())
    heading_like = not sentence.rstrip().endswith((".", "!", "?"))
    mentions: list[tuple[int, int, str]] = []
    for m in _ENTITY_RE.finditer(sentence):
        name = _clean_entity(m.group(0))
        if name is None:
            continue
        if " " not in name and (heading_like or m.start() == lead):
            continue
        mentions.append((m.start(), m.end(), name))
    return mentions


def extract_entities(text: str) -> list[str]:
    """Detect capitalized noun-phrase entities, preserving first-seen order."""
    seen: dict[str, None] = {}
    for sentence in split_sentences(text):
        for _start, _end, name in _sentence_mentions(sentence):
            if name not in seen:
                seen[name] = None
    return list(seen)


class LLMProvider(Protocol):
    """Semantic LLM interface: entities, triples, grounded answers, claim verdicts."""

    def extract_entities(self, text: str) -> list[str]: ...

    def extract_triples(self, text: str) -> list[tuple[str, str, str]]: ...

    def answer(self, question: str, contexts: list[TaggedContext]) -> str: ...

    def judge_claim(self, claim: str, evidence: str) -> bool: ...


class FakeLLM:
    """Deterministic stand-in used by tests, `make demo`, and `make eval`."""

    def extract_entities(self, text: str) -> list[str]:
        return extract_entities(text)

    def extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for sentence in split_sentences(text):
            mentions = _sentence_mentions(sentence)
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    s_start, s_end, subject = mentions[i]
                    o_start, _o_end, obj = mentions[j]
                    if subject == obj:
                        continue
                    relation = _relation_between(sentence[s_end:o_start])
                    triple = (subject, relation, obj)
                    if triple not in seen:
                        seen.add(triple)
                        triples.append(triple)
        return triples

    def answer(self, question: str, contexts: list[TaggedContext]) -> str:
        q_words = content_words(question)
        scored: list[tuple[float, int, int, str, int]] = []
        for ctx in contexts:
            for s_idx, sentence in enumerate(split_sentences(ctx.text)):
                sent = sentence.strip()
                words = content_words(sent)
                if not words:
                    continue
                overlap = len(q_words & words)
                if overlap == 0:
                    continue
                score = overlap / (1.0 + 0.05 * len(words))
                scored.append((score, ctx.tag, s_idx, sent, ctx.tag))
        if not scored:
            return "I cannot answer this from the provided context."
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        parts: list[str] = []
        used_tags: list[int] = []
        for _score, _tag, _s_idx, sent, tag in scored[:2]:
            parts.append(f"{sent.rstrip('.')} [{tag}].")
            if tag not in used_tags:
                used_tags.append(tag)
        return " ".join(parts)

    def judge_claim(self, claim: str, evidence: str) -> bool:
        claim_words = content_words(claim)
        if not claim_words:
            return True
        evidence_words = content_words(evidence)
        return len(claim_words & evidence_words) / len(claim_words) >= 0.5


def _clean_entity(raw: str) -> str | None:
    words = raw.strip().split()
    while words and words[0] in _ENTITY_STOPWORDS:
        words = words[1:]
    cleaned = " ".join(words)
    return cleaned if len(cleaned) > 2 else None


def _relation_between(between: str) -> str:
    tokens = [t for t in _RELATION_TOKEN_RE.findall(between.lower()) if t not in STOPWORDS]
    if not tokens or len(tokens) > 3:
        return "related_to"
    return "_".join(tokens)


_ENTITY_PROMPT = (
    "List the named entities (people, teams, products, places, named policies) "
    'mentioned in the text. Respond with JSON: {"entities": ["name", ...]}.\n\nText:\n'
)

_TRIPLE_PROMPT = (
    "Extract (subject, relation, object) triples from the text. "
    'Respond with JSON: {"triples": [["subject", "relation", "object"], ...]}.\n\nText:\n'
)

_ANSWER_PROMPT = (
    "Answer the question using ONLY the numbered context blocks. "
    "Cite sources inline as [n]. If the context is insufficient, say so.\n\n"
)

_JUDGE_PROMPT = (
    "Does the evidence support the claim? Respond with exactly SUPPORTED or UNSUPPORTED.\n\n"
)


class OpenAILLM:
    """Real LLM provider backed by the OpenAI chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAILLM requires an API key")
        self._model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def _complete(self, prompt: str) -> str:
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])

    def extract_entities(self, text: str) -> list[str]:
        raw = self._complete(_ENTITY_PROMPT + text)
        try:
            payload = json.loads(raw)
            return [str(e) for e in payload.get("entities", []) if str(e).strip()]
        except (json.JSONDecodeError, AttributeError, TypeError):
            return []

    def extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        raw = self._complete(_TRIPLE_PROMPT + text)
        try:
            payload = json.loads(raw)
            return [
                (str(t[0]), str(t[1]), str(t[2]))
                for t in payload.get("triples", [])
                if isinstance(t, list) and len(t) == 3
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def answer(self, question: str, contexts: list[TaggedContext]) -> str:
        blocks = "\n\n".join(f"[{c.tag}] ({c.source}) {c.text}" for c in contexts)
        return self._complete(f"{_ANSWER_PROMPT}{blocks}\n\nQuestion: {question}")

    def judge_claim(self, claim: str, evidence: str) -> bool:
        verdict = self._complete(f"{_JUDGE_PROMPT}Claim: {claim}\n\nEvidence:\n{evidence}")
        return "UNSUPPORTED" not in verdict.upper()
