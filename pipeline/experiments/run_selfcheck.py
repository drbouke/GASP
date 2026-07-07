# -*- coding: utf-8 -*-
"""
SelfCheckGPT-style self-consistency baseline. For each response we sample N
stochastic regenerations of the answer given (context, query), then score each
answer sentence by its average contradiction against the samples using an NLI
model (SelfCheckGPT-NLI). Higher inconsistency => more likely hallucinated.
Reports sentence-level AUC with a bootstrap 95% CI grouped by response.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, argparse
import numpy as np, pandas as pd

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--src_tag", default="Qwen2.5-1.5B-Instruct_K5")
    p.add_argument("--nli_model", default="cross-encoder/nli-deberta-v3-small")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--new_tokens", type=int, default=160)
    p.add_argument("--max_ctx_tokens", type=int, default=700)
    p.add_argument("--base", default=str(RESULTS))
    return p.parse_args()

def main():
    args = get_args()
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForSequenceClassification, set_seed)
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    ntok = AutoTokenizer.from_pretrained(args.nli_model)
    nli = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(dev).eval()
    torch.set_grad_enabled(False)
    con_idx = next(i for i, l in nli.config.id2label.items() if "contrad" in str(l).lower())

    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    sdf = pd.read_csv(os.path.join(args.base, args.src_tag, "sentence.csv"))
    rmap = {int(oi): None for oi in sdf["orig_idx"].unique()}

    def gen_samples(context, query):
        cids = tok(context, add_special_tokens=False).input_ids[:args.max_ctx_tokens]
        context = tok.decode(cids)
        ids = tok(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: ", return_tensors="pt").input_ids.to(dev)
        # batch the N samples in one generate call (num_return_sequences) for a large speedup
        o = lm.generate(ids, max_new_tokens=args.new_tokens, do_sample=True,
                        temperature=1.0, top_p=0.95, num_return_sequences=args.n_samples,
                        pad_token_id=tok.eos_token_id)
        return [tok.decode(o[i][ids.shape[1]:], skip_special_tokens=True) for i in range(args.n_samples)]

    def contradiction(premise, hypothesis):
        x = ntok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(dev)
        return float(torch.softmax(nli(**x).logits[0], -1)[con_idx].cpu())

    set_seed(42)
    print(f"generating samples for {len(rmap)} responses (batched N={args.n_samples})...")
    for gi, oi in enumerate(list(rmap.keys())):
        ex = ds[oi]
        rmap[oi] = gen_samples(ex["context"].strip(), (ex["query"] or "").strip())
        if (gi + 1) % 50 == 0: print(f"  generated {gi+1}/{len(rmap)}")
    scores = []
    for j, r in sdf.iterrows():
        samples = rmap[int(r["orig_idx"])]
        sent = str(r["sent_text"])
        c = np.mean([contradiction(s, sent) for s in samples]) if samples else np.nan
        scores.append(c)
        if (j + 1) % 400 == 0: print(f"  scored {j+1}/{len(sdf)}")
    sdf["selfcheck"] = scores
    sdf.to_csv(os.path.join(args.base, args.src_tag, "sentence_selfcheck.csv"), index=False)

    d = sdf.dropna(subset=["selfcheck", "label"])
    auc = roc_auc_score(d["label"], d["selfcheck"])
    rng = np.random.default_rng(0); uniq = d["resp_id"].unique()
    gmap = {g: np.where(d["resp_id"].to_numpy() == g)[0] for g in uniq}
    yl = d["label"].to_numpy(); sc = d["selfcheck"].to_numpy(); aucs = []
    for _ in range(1000):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gmap[g] for g in gs]); yy = yl[idx]
        if len(np.unique(yy)) < 2: continue
        aucs.append(roc_auc_score(yy, sc[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"\nSelfCheckGPT-NLI self-consistency sentence-level AUC: {auc:.3f}  [{lo:.3f}, {hi:.3f}]")

if __name__ == "__main__":
    main()
