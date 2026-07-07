# -*- coding: utf-8 -*-
"""
Quantifies truncation. For the exact responses scored by each model (stored
orig_idx), reports the fraction of contexts truncated at 700 tokens, the fraction
of answers truncated at 200 tokens, and the fraction of annotated hallucination
spans that fall within the kept answer tokens. Character offsets in
hallucination_labels index the output text.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, json
import numpy as np, pandas as pd

BASE = str(RESULTS)
MODELS = [("Qwen2.5-0.5B-Instruct_K5", "Qwen/Qwen2.5-0.5B-Instruct"),
          ("Qwen2.5-1.5B-Instruct_K5", "Qwen/Qwen2.5-1.5B-Instruct"),
          ("SmolLM2-1.7B-Instruct_K5", "HuggingFaceTB/SmolLM2-1.7B-Instruct")]
MAX_CTX, MAX_ANS = 700, 200

def main():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    lines = ["# Truncation analysis (context 700 tok, answer 200 tok)\n"]
    for tag, model in MODELS:
        rp = os.path.join(BASE, tag, "response.csv")
        if not os.path.exists(rp):
            continue
        rdf = pd.read_csv(rp)
        tok = AutoTokenizer.from_pretrained(model)
        ctx_trunc, ans_trunc, ctx_lens, ans_lens = 0, 0, [], []
        spans_total, spans_in_kept = 0, 0
        for _, r in rdf.iterrows():
            ex = ds[int(r["orig_idx"])]
            ctx_ids = tok(ex["context"].strip(), add_special_tokens=False)["input_ids"]
            ans_ids = tok(ex["output"].strip(), add_special_tokens=False)["input_ids"]
            ctx_lens.append(len(ctx_ids)); ans_lens.append(len(ans_ids))
            if len(ctx_ids) > MAX_CTX: ctx_trunc += 1
            if len(ans_ids) > MAX_ANS: ans_trunc += 1
            kept_ans_text = tok.decode(ans_ids[:MAX_ANS])
            kept_len = len(kept_ans_text)
            labs = json.loads(ex["hallucination_labels"]) if ex["hallucination_labels"].strip() else []
            for sp in labs:
                spans_total += 1
                if int(sp.get("start", 0)) < kept_len:
                    spans_in_kept += 1
        n = len(rdf)
        lines.append(f"## {tag}  (n={n} responses, {spans_total} annotated spans)")
        lines.append(f"- contexts truncated at {MAX_CTX} tok: {ctx_trunc}/{n} = {ctx_trunc/n:.1%}")
        lines.append(f"- context tokens: median {int(np.median(ctx_lens))}, 90th pct {int(np.percentile(ctx_lens,90))}, max {max(ctx_lens)}")
        lines.append(f"- answers truncated at {MAX_ANS} tok: {ans_trunc}/{n} = {ans_trunc/n:.1%}")
        lines.append(f"- answer tokens: median {int(np.median(ans_lens))}, 90th pct {int(np.percentile(ans_lens,90))}, max {max(ans_lens)}")
        lines.append(f"- annotated hallucination spans starting within kept answer: {spans_in_kept}/{spans_total} = {spans_in_kept/max(spans_total,1):.1%}\n")
    out = os.path.join(BASE, "TRUNCATION_STATS.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("written to", out)

if __name__ == "__main__":
    main()
