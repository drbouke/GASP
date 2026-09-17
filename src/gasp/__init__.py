"""GASP, a span-level detector of ungrounded content in retrieval-augmented generation.

GASP scores each answer sentence by its grounding sensitivity, the change in the
sentence likelihood when the retrieved context is perturbed, and returns the chunk
that best supports each sentence.
"""
from __future__ import annotations

from .case import Case
from .detector import GASP, Detection, SentenceResult
from .io import read_items, write_records
from .metrics import evaluate, pr_auc, roc_auc, threshold_metrics
from .scoring import Scorer, SentenceScore
from .spans import chunk_spans, sentence_spans

__all__ = [
    "GASP",
    "Detection",
    "SentenceResult",
    "Scorer",
    "SentenceScore",
    "Case",
    "sentence_spans",
    "chunk_spans",
    "evaluate",
    "roc_auc",
    "pr_auc",
    "threshold_metrics",
    "read_items",
    "write_records",
    "__version__",
]

__version__ = "0.2.1"
