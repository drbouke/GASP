"""The grounding-sensitivity scorer.

Holds a fixed answer and re-scores it under the full context, under no context, and
with each context chunk removed in turn, reading the log-likelihood drops and
Jensen-Shannon divergences at the answer tokens and aggregating them per sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .case import Case


@dataclass
class SentenceScore:
    """Per-sentence grounding features and the attributed supporting chunk."""

    index: int
    text: str
    n_tokens: int
    gap: float                 # mean token log-lik drop, full context vs none
    jsd_noctx: float           # mean token JSD, full context vs none
    max_drop: float            # largest per-chunk leave-one-out mean drop
    mean_drop: float           # mean over chunks of the leave-one-out drop
    top2_drop: float           # mean of the two largest per-chunk drops
    max_jsd: float             # largest per-chunk leave-one-out JSD
    chunk_drops: List[float] = field(default_factory=list)
    attributed_chunk: int = -1  # index of the chunk whose removal most lowers likelihood


class Scorer:
    """Load a causal language model and score answers by grounding sensitivity."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_ctx_tokens: int = 1800,
        max_ans_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError(
                "gasp scoring needs PyTorch and transformers. "
                "Install them with: pip install gasp-rag[torch]"
            ) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dtype is not None:
            torch_dtype = td[dtype]
        else:
            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.model.to(self.device).eval()
        torch.set_grad_enabled(False)
        self.model_id = model_id
        self.max_ctx_tokens = max_ctx_tokens
        self.max_ans_tokens = max_ans_tokens

    def _prompt(self, context: str, query: str):
        text = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: "
        return self._torch.tensor(self.tokenizer(text).input_ids)

    def _token_logprobs(self, prompt_ids, answer_ids):
        torch = self._torch
        ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(self.device)
        n_prompt, n_ans = prompt_ids.numel(), answer_ids.numel()
        logits = self.model(ids).logits[0][n_prompt - 1:n_prompt - 1 + n_ans].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        target = logprobs[torch.arange(n_ans, device=self.device), answer_ids.to(self.device)]
        return target, logprobs.exp(), logprobs

    def _jsd(self, p_full, logp_full, p_other):
        torch = self._torch
        eps = 1e-8
        m = 0.5 * (p_full + p_other)
        log_m = (m + eps).log()
        kl_p = (p_full * (logp_full - log_m)).sum(-1)
        kl_q = (p_other * ((p_other + eps).log() - log_m)).sum(-1)
        return (0.5 * kl_p + 0.5 * kl_q).clamp_(min=0.0, max=float(np.log(2)) + 1e-4)

    def score(self, case: Case, leave_one_out: bool = True) -> List[SentenceScore]:
        """Score every answer sentence of ``case`` and return its grounding features.

        With ``leave_one_out=False`` only the full-context and no-context passes run
        (the economical two-pass variant), so the ``gap`` feature is available but the
        per-chunk drops and the attribution are not.
        """
        torch = self._torch
        enc = self.tokenizer(case.answer, return_offsets_mapping=True, add_special_tokens=False)
        answer_ids = torch.tensor(enc["input_ids"][: self.max_ans_tokens])
        offsets = enc["offset_mapping"][: self.max_ans_tokens]
        n_ans = answer_ids.numel()
        if n_ans < 3:
            return []

        cenc = self.tokenizer(case.context, add_special_tokens=False, return_offsets_mapping=True)
        c_offsets = cenc["offset_mapping"]
        keep = self.max_ctx_tokens
        ctx_ret_chars = (
            c_offsets[keep - 1][1] if len(c_offsets) >= keep and keep > 0 else len(case.context)
        )
        ctx = case.context[:ctx_ret_chars]

        tlp_full, p_full, logp_full = self._token_logprobs(self._prompt(ctx, case.query), answer_ids)
        tlp_noc, p_noc, _ = self._token_logprobs(self._prompt("", case.query), answer_ids)
        gap_tok = (tlp_full - tlp_noc).cpu().numpy()
        jsd_noc_tok = self._jsd(p_full, logp_full, p_noc).cpu().numpy()

        n_chunks = case.n_chunks
        drop_tok = np.full((n_chunks, n_ans), np.nan)
        jsd_tok = np.full((n_chunks, n_ans), np.nan)
        if leave_one_out:
            for ci, (cs, ce) in enumerate(case.chunk_spans):
                if cs >= ctx_ret_chars:
                    continue
                removed = ctx[:cs] + ctx[min(ce, ctx_ret_chars):]
                tlp, p_removed, _ = self._token_logprobs(self._prompt(removed, case.query), answer_ids)
                drop_tok[ci] = (tlp_full - tlp).cpu().numpy()
                jsd_tok[ci] = self._jsd(p_full, logp_full, p_removed).cpu().numpy()

        scores: List[SentenceScore] = []
        for j, (s, e) in enumerate(case.sent_spans):
            tk = np.array([t for t in range(n_ans) if s <= offsets[t][0] < e])
            if tk.size < 3:
                continue
            per_chunk = [float(np.nanmean(drop_tok[c][tk])) for c in range(n_chunks)]
            per_chunk = [d for d in per_chunk if d == d]  # drop NaN chunks
            per_jsd = [float(np.nanmean(jsd_tok[c][tk])) for c in range(n_chunks)]
            per_jsd = [d for d in per_jsd if d == d]
            arr = np.array(per_chunk) if per_chunk else np.array([0.0])
            top2 = float(np.mean(np.sort(arr)[-2:])) if arr.size >= 2 else float(arr.max())
            attributed = int(np.argmax(per_chunk)) if per_chunk else -1
            scores.append(
                SentenceScore(
                    index=j,
                    text=case.sentence(j),
                    n_tokens=int(tk.size),
                    gap=float(np.mean(gap_tok[tk])),
                    jsd_noctx=float(np.nanmean(jsd_noc_tok[tk])),
                    max_drop=float(arr.max()),
                    mean_drop=float(arr.mean()),
                    top2_drop=top2,
                    max_jsd=float(max(per_jsd)) if per_jsd else 0.0,
                    chunk_drops=per_chunk,
                    attributed_chunk=attributed,
                )
            )
        return scores
