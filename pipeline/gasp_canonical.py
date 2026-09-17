# -*- coding: utf-8 -*-
"""
Canonical preprocessing for GASP.

One reference record per case is built here and consumed unchanged by every method
(sensitivity scorer, NLI baseline, explanation evaluation). Sentence and chunk
boundaries are stored in CHARACTER offsets over the ORIGINAL context and answer, so
"chunk i" and "sentence j" denote exactly the same text in every method. No method
re-splits or re-tokenizes the context to define chunks.

Model-window truncation is a separate, per-scorer concern: a scorer records the
character range of context/answer it actually fed the model (retained span), which
lets the label-alignment audit report what was scored versus labeled. The
canonical chunk/sentence spans themselves never depend on any model tokenizer.
"""
import re
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional

Span = Tuple[int, int]  # (start_char, end_char), end exclusive


def sentence_spans(text: str) -> List[Span]:
    """Character spans of sentences in `text`, covering non-empty content."""
    spans: List[Span] = []
    start = 0
    for m in re.finditer(r'[.!?]+(?:\s+|$)', text):
        end = m.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def chunk_spans(context: str, k: int) -> List[Span]:
    """
    Character spans of up to k chunks over `context`. Chunks group whole sentences
    (same policy as the original split), but are returned as spans into the ORIGINAL
    context so every consumer slices identical text via context[s:e].
    """
    ctx = context
    sent = sentence_spans(ctx)
    if not sent:
        return [(0, len(ctx))] if ctx else []
    if len(sent) <= k:
        return sent
    import math
    per = int(math.ceil(len(sent) / k))
    groups: List[Span] = []
    for i in range(0, len(sent), per):
        s = sent[i][0]
        e = sent[min(i + per, len(sent)) - 1][1]
        groups.append((s, e))
    return groups


@dataclass
class Case:
    """One canonical case. All spans are character offsets into `context`/`answer`."""
    case_id: str            # unique per (source_id, answer_id)
    source_id: str          # RAGTruth source_id / meeting id
    answer_id: str          # per-answer id
    dataset: str            # ragtruth | tofueval | ragbench
    task: str               # e.g. Summary, Data2txt, QA
    query: str
    context: str            # full, untruncated
    answer: str             # full, untruncated
    sent_spans: List[Span] = field(default_factory=list)   # over answer
    chunk_spans: List[Span] = field(default_factory=list)  # over context
    # hallucination spans over answer (char), each: (start, end, type in {conflict,baseless})
    halluc_spans: List[Tuple[int, int, str]] = field(default_factory=list)

    def chunk_text(self, i: int) -> str:
        s, e = self.chunk_spans[i]
        return self.context[s:e]

    def sent_text(self, j: int) -> str:
        s, e = self.sent_spans[j]
        return self.answer[s:e]

    def sent_label(self, j: int) -> Tuple[int, Optional[str]]:
        """Label a sentence from hallucination spans that overlap ITS char range."""
        s, e = self.sent_spans[j]
        t = None
        for (hs, he, ht) in self.halluc_spans:
            if not (he <= s or hs >= e):      # overlap
                if ht == "conflict":
                    return 1, "conflict"
                t = "baseless"
        return (1, t) if t is not None else (0, None)


def build_case(*, case_id, source_id, answer_id, dataset, task, query, context, answer,
               k_chunks, halluc_spans=None) -> Optional[Case]:
    context = (context or "").strip()
    answer = (answer or "").strip()
    if not context or not answer:
        return None
    return Case(
        case_id=str(case_id), source_id=str(source_id), answer_id=str(answer_id),
        dataset=dataset, task=task, query=(query or "").strip(),
        context=context, answer=answer,
        sent_spans=sentence_spans(answer),
        chunk_spans=chunk_spans(context, k_chunks),
        halluc_spans=list(halluc_spans or []),
    )


def to_record(case: Case) -> dict:
    return asdict(case)


def from_record(d: dict) -> Case:
    return Case(
        case_id=d["case_id"], source_id=d["source_id"], answer_id=d["answer_id"],
        dataset=d["dataset"], task=d["task"], query=d["query"],
        context=d["context"], answer=d["answer"],
        sent_spans=[tuple(x) for x in d["sent_spans"]],
        chunk_spans=[tuple(x) for x in d["chunk_spans"]],
        halluc_spans=[tuple(x) for x in d["halluc_spans"]],
    )
