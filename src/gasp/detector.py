"""The high-level :class:`GASP` detector and its result types."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from .case import Case
from .scoring import Scorer, SentenceScore

# features available as the sensitivity scalar; "gap" needs only the two-pass variant
_SENSITIVITY_FEATURES = {"max_drop", "gap", "mean_drop", "top2_drop", "max_jsd", "jsd_noctx"}
_TWO_PASS_FEATURES = {"gap", "jsd_noctx"}


@dataclass
class SentenceResult:
    """The detector's verdict for one answer sentence.

    ``sensitivity`` is the grounding sensitivity: higher means the sentence depends
    more on the retrieved evidence and is more likely grounded, lower means it barely
    reacts to removing evidence and is more likely unsupported. ``flagged`` is set
    only when a threshold is supplied.
    """

    index: int
    text: str
    sensitivity: float
    supporting_chunk_index: int
    supporting_chunk: Optional[str]
    features: Dict[str, float]
    flagged: Optional[bool] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Detection:
    """The per-sentence results for one answer."""

    sentences: List[SentenceResult]

    def __iter__(self):
        return iter(self.sentences)

    def __len__(self) -> int:
        return len(self.sentences)

    def flagged(self) -> List[SentenceResult]:
        """The sentences flagged as likely unsupported (requires a threshold)."""
        return [s for s in self.sentences if s.flagged]

    def to_records(self) -> List[Dict]:
        """The results as a list of flat dictionaries, one per sentence."""
        return [s.to_dict() for s in self.sentences]

    def summary(self) -> Dict[str, float]:
        """Per-answer summary: sentence count, flagged count, and mean sensitivity."""
        n = len(self.sentences)
        sens = [s.sensitivity for s in self.sentences]
        n_flag = sum(1 for s in self.sentences if s.flagged)
        return {
            "n_sentences": n,
            "n_flagged": n_flag,
            "mean_sensitivity": float(sum(sens) / n) if n else 0.0,
            "min_sensitivity": float(min(sens)) if n else 0.0,
        }


class GASP:
    """Grounding-Aware Sensitivity by Perturbation, a span-level RAG grounding detector.

    Parameters
    ----------
    model_id:
        Any Hugging Face causal language model used as the scorer. It need not be the
        model that produced the answer, and it may be small and run on CPU.
    k_chunks:
        Number of context chunks for the leave-one-out perturbations.
    threshold:
        Optional sensitivity threshold below which a sentence is flagged.
    economical:
        If True, run only the full-context and no-context passes (the two-pass variant),
        which is much faster but provides no per-chunk attribution.
    sensitivity_feature:
        Which grounding feature to report as ``sensitivity`` (default ``max_drop``; use
        ``gap`` with ``economical=True``).
    max_ctx_tokens, max_ans_tokens, device, dtype:
        Passed to the underlying scorer.

    Example
    -------
    >>> from gasp import GASP
    >>> det = GASP("Qwen/Qwen2.5-1.5B-Instruct", k_chunks=5, threshold=0.5)
    >>> result = det.detect(context=ctx, answer=ans, query=question)
    >>> for s in result.flagged():
    ...     print(s.text)
    """

    def __init__(
        self,
        model_id: str,
        k_chunks: int = 5,
        threshold: Optional[float] = None,
        economical: bool = False,
        sensitivity_feature: str = "max_drop",
        **scorer_kwargs,
    ) -> None:
        if sensitivity_feature not in _SENSITIVITY_FEATURES:
            raise ValueError(
                f"sensitivity_feature must be one of {sorted(_SENSITIVITY_FEATURES)}"
            )
        if economical and sensitivity_feature not in _TWO_PASS_FEATURES:
            sensitivity_feature = "gap"
        self.scorer = Scorer(model_id, **scorer_kwargs)
        self.k_chunks = k_chunks
        self.threshold = threshold
        self.economical = economical
        self.sensitivity_feature = sensitivity_feature

    def detect(
        self,
        context: str,
        answer: str,
        query: str = "",
        threshold: Optional[float] = None,
    ) -> Detection:
        """Score every sentence of ``answer`` against ``context`` for grounding.

        If a ``threshold`` is given here or on the detector, sentences whose sensitivity
        falls below it are marked ``flagged``. Thresholds are corpus dependent and are
        best calibrated on held-out data.
        """
        thr = threshold if threshold is not None else self.threshold
        case = Case.from_texts(context, answer, query, k_chunks=self.k_chunks)
        results: List[SentenceResult] = []
        for sc in self.scorer.score(case, leave_one_out=not self.economical):
            feats = _features(sc)
            sensitivity = feats[self.sensitivity_feature]
            has_attr = (not self.economical) and 0 <= sc.attributed_chunk < case.n_chunks
            results.append(
                SentenceResult(
                    index=sc.index,
                    text=sc.text,
                    sensitivity=sensitivity,
                    supporting_chunk_index=sc.attributed_chunk if has_attr else -1,
                    supporting_chunk=case.chunk(sc.attributed_chunk) if has_attr else None,
                    features=feats,
                    flagged=(sensitivity < thr) if thr is not None else None,
                )
            )
        return Detection(results)

    def detect_batch(
        self,
        items: Iterable[Dict[str, str]],
        threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Detect over many answers.

        ``items`` is an iterable of dicts, each with ``context`` and ``answer`` and an
        optional ``query``. Returns one :class:`Detection` per item.
        """
        out: List[Detection] = []
        for it in items:
            out.append(
                self.detect(
                    context=it["context"],
                    answer=it["answer"],
                    query=it.get("query", ""),
                    threshold=threshold,
                )
            )
        return out


def _features(sc: SentenceScore) -> Dict[str, float]:
    return {
        "gap": sc.gap,
        "jsd_noctx": sc.jsd_noctx,
        "max_drop": sc.max_drop,
        "mean_drop": sc.mean_drop,
        "top2_drop": sc.top2_drop,
        "max_jsd": sc.max_jsd,
        "n_tokens": float(sc.n_tokens),
    }
