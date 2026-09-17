# -*- coding: utf-8 -*-
"""
ContextCite baseline (Cohen-Wang et al., 2024), applied as a grounding detector.

For each case we run N random context-chunk ablations, score the fixed answer under
each ablated context, and fit a linear surrogate predicting each sentence's mean
log-probability from the binary chunk-presence vector. The surrogate weights are the
per-chunk attributions; the detector score for a sentence is the largest positive
attribution (how strongly some context chunk raises the model's probability of the
sentence). Grounded sentences have high attribution; hallucinated ones near zero.
This is an empirical adaptation of ContextCite into a detector, named as such.

Consumes the canonical cases.jsonl (identical chunks/sentences as GASP).
Usage (GPU): python run_contextcite.py --cases <cases.jsonl> --model Qwen/Qwen2.5-7B-Instruct
             --out contextcite.csv --n_ablations 32
"""
import json, argparse
import numpy as np, pandas as pd
from gasp_canonical import from_record


class LM:
    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(self.device).eval()
        torch.set_grad_enabled(False)

    def answer_tok_lp(self, context, query, answer_ids):
        torch = self.torch
        pid = torch.tensor(self.tok(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: ").input_ids)
        ids = torch.cat([pid, answer_ids]).unsqueeze(0).to(self.device)
        P, A = pid.numel(), answer_ids.numel()
        logits = self.model(ids).logits[0][P - 1:P - 1 + A].float()
        lp = torch.log_softmax(logits, -1)
        return lp[torch.arange(A, device=self.device), answer_ids.to(self.device)].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="contextcite.csv")
    ap.add_argument("--n_ablations", type=int, default=32)
    ap.add_argument("--max_ans", type=int, default=256)
    a = ap.parse_args()
    import torch
    lm = LM(a.model)
    rng = np.random.default_rng(0)
    cases = [from_record(json.loads(l)) for l in open(a.cases, encoding="utf-8")]
    print(f"[contextcite] {len(cases)} cases, N={a.n_ablations}, model={a.model}")
    rows = []
    for ci, c in enumerate(cases):
        K = len(c.chunk_spans)
        if K < 2:
            continue
        enc = lm.tok(c.answer, return_offsets_mapping=True, add_special_tokens=False)
        aid = torch.tensor(enc["input_ids"][:a.max_ans]); offs = enc["offset_mapping"][:a.max_ans]
        A = aid.numel()
        if A < 8:
            continue
        # token index set per sentence
        sent_tok = []
        for (s, e) in c.sent_spans:
            tk = [t for t in range(A) if offs[t][0] >= s and offs[t][0] < e]
            sent_tok.append(tk)
        # N ablations: random chunk masks; score answer per token
        design = np.zeros((a.n_ablations, K)); Ytok = np.zeros((a.n_ablations, A))
        try:
            for n in range(a.n_ablations):
                mask = rng.integers(0, 2, size=K)
                if mask.sum() == 0:
                    mask[rng.integers(0, K)] = 1
                ctx = "".join(c.context[s:e] for k, (s, e) in enumerate(c.chunk_spans) if mask[k])
                Ytok[n] = lm.answer_tok_lp(ctx, c.query, aid)
                design[n] = mask
        except torch.cuda.OutOfMemoryError:
            # a few extreme-length cases exceed GPU memory; skip and free, keep every
            # scored case on identical footing across datasets
            torch.cuda.empty_cache()
            print(f"  skipped case {c.case_id} (OOM, seq too long)")
            continue
        # ridge surrogate per sentence: y ~ design
        XtX = design.T @ design + 1e-2 * np.eye(K)
        for j, tk in enumerate(sent_tok):
            if len(tk) < 3:
                continue
            y = Ytok[:, tk].mean(axis=1)
            w = np.linalg.solve(XtX, design.T @ (y - y.mean()))
            rows.append(dict(case_id=c.case_id, source_id=c.source_id, sent_idx=j,
                             label=c.sent_label(j)[0],
                             contextcite_max=float(np.max(w)),
                             contextcite_sumpos=float(np.sum(np.clip(w, 0, None)))))
        if (ci + 1) % 25 == 0:
            print(f"  {ci+1}/{len(cases)}")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[contextcite] wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
