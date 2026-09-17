"""Character-span segmentation of answers into sentences and contexts into chunks.

All spans are ``(start, end)`` character offsets, end exclusive, over the original
text, so the same span denotes the same text for every consumer.
"""
from __future__ import annotations

import math
import re
from typing import List, Tuple

Span = Tuple[int, int]

_SENT_END = re.compile(r"[.!?]+(?:\s+|$)")


def sentence_spans(text: str) -> List[Span]:
    """Return character spans of the sentences in ``text``.

    Each returned span covers non-empty content and the spans tile the text in
    order. A trailing fragment without terminal punctuation is included.
    """
    spans: List[Span] = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def chunk_spans(context: str, k: int) -> List[Span]:
    """Return character spans of up to ``k`` chunks over ``context``.

    Chunks group whole sentences into ``k`` roughly equal pieces and are returned
    as spans into the original context, so ``context[s:e]`` recovers each chunk.
    """
    if not context:
        return []
    sents = sentence_spans(context)
    if not sents:
        return [(0, len(context))]
    if len(sents) <= k:
        return sents
    per = int(math.ceil(len(sents) / k))
    groups: List[Span] = []
    for i in range(0, len(sents), per):
        s = sents[i][0]
        e = sents[min(i + per, len(sents)) - 1][1]
        groups.append((s, e))
    return groups
