# -*- coding: utf-8 -*-
"""
Max-chunk NLI baseline. For each answer sentence, the context is split into the
same K=5 chunks used by GASP and the sentence is scored as its maximum entailment
over the chunks (chunk = premise, sentence = hypothesis) from a large NLI cross-
encoder. Higher max-entailment means better supported, so the hallucination score
is its negative. Reports span-level AUC with a grouped bootstrap 95% CI.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, re, json
import numpy as np, pandas as pd

BASE = str(RESULTS)
TAG = "Qwen2.5-1.5B-Instruct_K5"
NLI = "cross-encoder/nli-deberta-v3-large"
MAX_CTX = 700

def split_chunks(context, k):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', context.strip()) if s.strip()]
    if len(sents) <= k: return sents if sents else [context]
    per = int(np.ceil(len(sents) / k))
    return [" ".join(sents[i:i+per]) for i in range(0, len(sents), per)]

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI).to(dev).eval()
    torch.set_grad_enabled(False)
    ent_idx = next(i for i, l in model.config.id2label.items() if "entail" in str(l).lower())

    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    sdf = pd.read_csv(os.path.join(BASE, TAG, "sentence.csv"))
    ctx_chunks = {}
    def chunks_for(oi):
        if oi not in ctx_chunks:
            c = ds[int(oi)]["context"].strip()
            ids = tok(c, add_special_tokens=False, truncation=True, max_length=MAX_CTX)["input_ids"]
            ctx_chunks[oi] = split_chunks(tok.decode(ids), 5)
        return ctx_chunks[oi]

    def entail(premise, hypothesis):
        x = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(dev)
        return float(torch.softmax(model(**x).logits[0], -1)[ent_idx].cpu())

    vals = []
    for j, r in sdf.iterrows():
        chs = chunks_for(r["orig_idx"]); sent = str(r["sent_text"])
        try:
            vals.append(max(entail(c, sent) for c in chs))
        except Exception:
            vals.append(np.nan)
        if (j + 1) % 400 == 0: print(f"  {j+1}/{len(sdf)}")
    sdf["nli_maxchunk"] = vals
    sdf.to_csv(os.path.join(BASE, TAG, "sentence_nli_maxchunk.csv"), index=False)

    d = sdf.dropna(subset=["nli_maxchunk", "label"])
    sc = -d["nli_maxchunk"].to_numpy(); y = d["label"].to_numpy()
    auc = roc_auc_score(y, sc)
    rng = np.random.default_rng(0); uniq = d["resp_id"].unique()
    gmap = {g: np.where(d["resp_id"].to_numpy() == g)[0] for g in uniq}; aucs = []
    for _ in range(1000):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gmap[g] for g in gs]); yy = y[idx]
        if len(np.unique(yy)) < 2: continue
        aucs.append(roc_auc_score(yy, sc[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"\nMax-chunk NLI (deberta-v3-large) span AUC: {auc:.3f}  [{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    main()
