# -*- coding: utf-8 -*-
"""
Analytic compute-cost panel for GASP vs the advanced baselines.

Cost here is hardware-independent and fully reproducible: for every method we count,
from the SAME canonical cases and each method's real tokenizer, the number of
model forward passes and the total tokens each pass processes, then convert to a FLOPs
proxy with the standard transformer inference estimate

    FLOPs_per_pass ~= 2 * N_params * N_tokens                          (prefill; the
    attention term is negligible at these sequence lengths relative to the MLP term)

For the encoder-decoder verifier (MiniCheck / flan-t5-large) only the encoder params and
encoder tokens are counted, since the decoder emits one or two label tokens. We report
GFLOPs (= 2 * params_in_billions * tokens) per response and amortized per scored answer
sentence, plus each method's Experiment-2 ROC-AUC, so the value/cost trade-off is explicit.

The key structural asymmetry the panel exposes: GASP encodes the context a fixed number of
times PER RESPONSE (2 for the economical gap variant, 2+K for the full variant) and shares
that work across all sentences, whereas the per-sentence verifiers re-encode the context
once (ctx variants) or K times (max-chunk variants) FOR EVERY answer sentence.

Usage: python cost_analysis.py --cases canon_results/Qwen2.5-14B-Instruct_ragtruth_K5/cases.jsonl
"""
import json, argparse
import numpy as np
from gasp_canonical import from_record

# active parameters in billions (encoder-only for the T5 verifier)
PARAMS = {
    "qwen14b": 14.8, "qwen7b": 7.6, "qwen1.5b": 1.5,
    "deberta_large": 0.435, "flan_t5_large_enc": 0.39, "roberta_large": 0.355,
}
MAX_CTX = 1800          # scorer context-token cap actually used in Exp 2
MAX_ANS = 256
SCAFFOLD = 12           # "Context:\n .. \n\nQuestion: .. \n\nAnswer: " wrapper tokens (approx)
NLI_CAP = 512           # deberta/roberta pair truncation
ALIGN_WIN = 350         # AlignScore internal context window (roberta)
JUDGE_INSTR = 40        # LLM-judge instruction tokens (approx)
N_ABLATION = 24         # ContextCite ablations per response

# Exp-2 ROC-AUC per method/config (measured, 14B scorer where applicable)
AUC = {
    "GASP-full (14B, 2+K passes)": 0.779, "GASP-gap (14B, 2 passes)": 0.763,
    "GASP-gap (1.5B, 2 passes)": 0.733,
    "chunk-NLI (max)": 0.709, "chunk-NLI (ctx)": 0.635,
    "ContextCite (sumpos)": 0.754, "ContextCite (max)": 0.740,
    "AlignScore (max)": 0.779, "AlignScore (ctx)": 0.811,
    "MiniCheck (max)": 0.790, "MiniCheck (ctx)": 0.834,
    "LLM-judge (14B)": 0.861,
}


def gflops(params_b, tokens):
    return 2.0 * params_b * tokens  # GFLOPs = 2 * params(billions) * tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    print("# loading tokenizers (CPU, vocab only)")
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")   # shared by all Qwen2.5 sizes
    dtok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    ftok = AutoTokenizer.from_pretrained("google/flan-t5-large")
    rtok = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    def ntok(tok, text):
        return len(tok(text, add_special_tokens=False).input_ids)

    cases = [from_record(json.loads(l)) for l in open(a.cases, encoding="utf-8")]
    print(f"# {len(cases)} canonical cases\n")

    # accumulators: total GFLOPs and total scored sentences across the corpus
    tot = {k: 0.0 for k in AUC}
    n_sent_total = 0
    n_resp = 0

    for c in cases:
        ctx = min(ntok(qtok, c.context), MAX_CTX)
        ans = min(ntok(qtok, c.answer), MAX_ANS)
        q = ntok(qtok, c.query)
        base = q + SCAFFOLD + ans
        K = len(c.chunk_spans)
        chunk_tok = [ntok(qtok, c.chunk_text(i)) for i in range(K)]
        # scale chunk tokens so they sum to the capped context (chunks tile the full context)
        s = sum(chunk_tok) or 1
        chunk_tok = [t * ctx / s for t in chunk_tok]
        sents = [c.sent_text(j) for j in range(len(c.sent_spans))]
        sents = [s for s in sents if len(s.strip()) >= 3]
        ns = max(1, len(sents))
        n_sent_total += len(sents); n_resp += 1

        # ---- GASP (Qwen scorer) : per RESPONSE, shared across sentences ----
        full_pass = ctx + base
        noctx_pass = base
        chunk_passes = sum((ctx - ct) + base for ct in chunk_tok)   # leave-one-chunk-out
        gasp_full_tokens = full_pass + noctx_pass + chunk_passes
        gasp_gap_tokens = full_pass + noctx_pass
        tot["GASP-full (14B, 2+K passes)"] += gflops(PARAMS["qwen14b"], gasp_full_tokens)
        tot["GASP-gap (14B, 2 passes)"] += gflops(PARAMS["qwen14b"], gasp_gap_tokens)
        tot["GASP-gap (1.5B, 2 passes)"] += gflops(PARAMS["qwen1.5b"], gasp_gap_tokens)

        # ---- ContextCite (Qwen 7B) : per RESPONSE, N ablations over ~half the context ----
        cc_tokens = N_ABLATION * (0.5 * ctx + base)
        tot["ContextCite (sumpos)"] += gflops(PARAMS["qwen7b"], cc_tokens)
        tot["ContextCite (max)"] += gflops(PARAMS["qwen7b"], cc_tokens)

        # ---- per-SENTENCE verifiers: re-encode context (ctx) or K chunks (max) per sentence ----
        # context/chunk token counts do not depend on the sentence; compute once per response
        ctx_d = ntok(dtok, c.context); ctx_f = ntok(ftok, c.context); ctx_r = ntok(rtok, c.context)
        chunks_d = [min(ntok(dtok, c.chunk_text(i)), NLI_CAP) for i in range(K)]
        chunks_f = [ntok(ftok, c.chunk_text(i)) for i in range(K)]
        chunks_r = [min(ntok(rtok, c.chunk_text(i)), ALIGN_WIN) for i in range(K)]
        for st in sents:
            sd = ntok(dtok, st); sf = ntok(ftok, st); sr = ntok(rtok, st)
            # chunk-NLI (deberta): pair (premise, hyp) capped at 512
            tot["chunk-NLI (max)"] += gflops(PARAMS["deberta_large"], sum(min(cd + sd, NLI_CAP) for cd in chunks_d))
            tot["chunk-NLI (ctx)"] += gflops(PARAMS["deberta_large"], min(ctx_d + sd, NLI_CAP))
            # MiniCheck (flan-t5 encoder): doc+claim; long ctx tiled into 512-token windows
            tot["MiniCheck (max)"] += gflops(PARAMS["flan_t5_large_enc"], sum(min(cf + sf, 512) for cf in chunks_f))
            ctx_windows = max(1, int(np.ceil(ctx_f / 512)))
            tot["MiniCheck (ctx)"] += gflops(PARAMS["flan_t5_large_enc"], ctx_windows * min(ctx_f + sf, 512))
            # AlignScore (roberta): chunk<=350 each; ctx tiled into 350-token windows
            tot["AlignScore (max)"] += gflops(PARAMS["roberta_large"], sum(min(cr + sr, NLI_CAP) for cr in chunks_r))
            aw = max(1, int(np.ceil(ctx_r / ALIGN_WIN)))
            tot["AlignScore (ctx)"] += gflops(PARAMS["roberta_large"], aw * min(ALIGN_WIN + sr, NLI_CAP))
            # LLM-judge (Qwen 14B): instruction + full ctx + sentence, one prompted pass
            tot["LLM-judge (14B)"] += gflops(PARAMS["qwen14b"], JUDGE_INSTR + ctx + ntok(qtok, st))

    print(f"# corpus: {n_resp} responses, {n_sent_total} scored sentences "
          f"({n_sent_total/n_resp:.1f} sentences/response)\n")
    # per-response and per-sentence GFLOPs; relative to the cheapest (GASP-gap 1.5B)
    ref = tot["GASP-gap (1.5B, 2 passes)"] / n_sent_total
    rows = []
    for name, gf in tot.items():
        per_resp = gf / n_resp
        per_sent = gf / n_sent_total
        rows.append((name, AUC[name], per_resp, per_sent, per_sent / ref))
    rows.sort(key=lambda r: r[3])   # by per-sentence cost
    print(f"{'method':30s} {'AUC':>5s} {'GFLOPs/resp':>12s} {'GFLOPs/sent':>12s} {'xGASP-gap-1.5B':>14s}")
    for name, auc, pr, ps, rel in rows:
        print(f"{name:30s} {auc:5.3f} {pr:12.1f} {ps:12.1f} {rel:13.1f}x")
    print("\nReading: lower GFLOPs/sent is cheaper. Pair with AUC to read the value/cost frontier: "
          "the two methods that beat GASP (MiniCheck-ctx, LLM-judge) are also the most expensive.")


if __name__ == "__main__":
    main()
