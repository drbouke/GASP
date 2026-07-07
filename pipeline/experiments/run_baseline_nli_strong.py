# -*- coding: utf-8 -*-
"""
Whole-context NLI baseline. Computes entailment of each answer sentence (and each
response) against the retrieved context using a large NLI cross-encoder. Adds an
nli_large column to a copy of the sentence/response CSVs and reports AUC with a
bootstrap 95% CI at the sentence level (grouped by response).
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, json, argparse
import numpy as np, pandas as pd

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="Qwen2.5-1.5B-Instruct_K5")
    p.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-large")
    p.add_argument("--max_ctx_tokens", type=int, default=700)
    p.add_argument("--base", default=str(RESULTS))
    return p.parse_args()

def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(dev).eval()
    torch.set_grad_enabled(False)
    id2 = model.config.id2label
    ent_idx = next(i for i, l in id2.items() if "entail" in str(l).lower())
    print("NLI:", args.nli_model, "entail idx", ent_idx)

    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    ctx_cache = {}
    def context_for(oi):
        if oi not in ctx_cache:
            c = ds[int(oi)]["context"].strip()
            ids = tok(c, add_special_tokens=False, truncation=True, max_length=args.max_ctx_tokens)["input_ids"]
            ctx_cache[oi] = tok.decode(ids)
        return ctx_cache[oi]

    def entail(premise, hypothesis):
        x = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(dev)
        return float(torch.softmax(model(**x).logits[0], -1)[ent_idx].cpu())

    sdf = pd.read_csv(os.path.join(args.base, args.tag, "sentence.csv"))
    vals = []
    for j, r in sdf.iterrows():
        try:
            vals.append(entail(context_for(r["orig_idx"]), str(r["sent_text"])))
        except Exception:
            vals.append(np.nan)
        if (j + 1) % 400 == 0:
            print(f"  {j+1}/{len(sdf)}")
    sdf["nli_large"] = vals
    sdf.to_csv(os.path.join(args.base, args.tag, "sentence_nli_large.csv"), index=False)

    d = sdf.dropna(subset=["nli_large", "label"])
    # higher entailment => less hallucinated, so use negative as the hallucination score
    auc = roc_auc_score(d["label"], -d["nli_large"])
    rng = np.random.default_rng(0); uniq = d["resp_id"].unique()
    gmap = {g: np.where(d["resp_id"].to_numpy() == g)[0] for g in uniq}
    yl = d["label"].to_numpy(); sc = -d["nli_large"].to_numpy()
    aucs = []
    for _ in range(1000):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gmap[g] for g in gs]); yy = yl[idx]
        if len(np.unique(yy)) < 2: continue
        aucs.append(roc_auc_score(yy, sc[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"\nWhole-context NLI (deberta-v3-large) sentence-level AUC: {auc:.3f}  [{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    main()
