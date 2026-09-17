"""The :class:`Case`, one query/context/answer example with canonical spans."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .spans import Span, chunk_spans, sentence_spans


@dataclass
class Case:
    """A single example: a query, a retrieved context, and an answer to check.

    ``sent_spans`` are character spans over ``answer`` and ``chunk_spans`` are
    character spans over ``context``. Build one with :meth:`from_texts`.
    """

    query: str
    context: str
    answer: str
    sent_spans: List[Span] = field(default_factory=list)
    chunk_spans: List[Span] = field(default_factory=list)

    @classmethod
    def from_texts(cls, context: str, answer: str, query: str = "", k_chunks: int = 5) -> "Case":
        """Build a case, segmenting the answer into sentences and the context into ``k_chunks``."""
        context = (context or "").strip()
        answer = (answer or "").strip()
        return cls(
            query=(query or "").strip(),
            context=context,
            answer=answer,
            sent_spans=sentence_spans(answer),
            chunk_spans=chunk_spans(context, k_chunks),
        )

    @property
    def n_sentences(self) -> int:
        return len(self.sent_spans)

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_spans)

    def sentence(self, j: int) -> str:
        s, e = self.sent_spans[j]
        return self.answer[s:e]

    def chunk(self, i: int) -> str:
        s, e = self.chunk_spans[i]
        return self.context[s:e]
