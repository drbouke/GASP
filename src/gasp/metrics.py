"""Evaluation metrics for grounding detection, when reference labels are available.

Scores are oriented so that a higher score means more likely unsupported. Labels are
``1`` for an unsupported (hallucinated) span and ``0`` for a grounded one.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, computed from rank statistics."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the precision-recall curve (average precision by trapezoid)."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if (y == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / (y.sum())
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in NumPy 2.0
    return float(trapezoid(precision, recall))


def threshold_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    """Precision, recall, F1, and Matthews correlation at a decision threshold."""
    y = np.asarray(labels, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "mcc": mcc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(labels: Sequence[int], scores: Sequence[float], threshold: float | None = None) -> Dict[str, float]:
    """Full metric suite: ROC-AUC, PR-AUC, and, if a threshold is given, its point metrics."""
    out: Dict[str, float] = {
        "n": int(len(labels)),
        "positives": int(np.asarray(labels).sum()),
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
    }
    if threshold is not None:
        out.update(threshold_metrics(labels, scores, threshold))
    return out
