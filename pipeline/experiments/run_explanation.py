# -*- coding: utf-8 -*-
"""
Proxy evaluation of GASP's chunk attribution. RAGTruth annotates hallucinated
spans but not the gold supporting passage for grounded spans, so for each grounded
sentence the attributed chunk is taken as the one with the maximum leave-one-out
likelihood drop (the explanation GASP returns). Using a large NLI cross-encoder,
the attributed chunk is tested for entailing the sentence more than a random chunk
and more than the highest lexical-overlap chunk, and the agreement rate between the
drop-based attribution and NLI-based support is reported. Best-chunk entailment is
also compared between grounded and hallucinated sentences.
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

def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    from scipy.stats import wilcoxon, mannwhitneyu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI).to(dev).eval()
    torch.set_grad_enabled(False)
    ent_idx = next(i for i, l in model.config.id2label.items() if "entail" in str(l).lower())

    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    sdf = pd.read_csv(os.path.join(BASE, TAG, "sentence.csv"))
    sdf["cd"] = sdf["chunk_drops"].apply(json.loads)
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

    rng = np.random.default_rng(0)
    rows = []
    for j, r in sdf.iterrows():
        chs = chunks_for(r["orig_idx"]); sent = str(r["sent_text"])
        cd = r["cd"][:len(chs)]
        if len(chs) < 3 or len(cd) < 3: continue
        katt = int(np.argmax(cd))                      # attributed = max likelihood drop
        cand = [i for i in range(len(chs)) if i != katt]
        krand = int(rng.choice(cand))
        klex = max(cand, key=lambda i: jaccard(chs[i], sent))
        try:
            ent = [entail(c, sent) for c in chs]
        except Exception:
            continue
        katt_nli = int(np.argmax(ent))
        rows.append(dict(resp_id=r["resp_id"], label=int(r["label"]), e_att=ent[katt], e_rand=ent[krand],
                         e_lex=ent[klex], e_best=max(ent), agree=int(katt == katt_nli)))
        if (j + 1) % 400 == 0: print(f"  {j+1}/{len(sdf)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "explanation_eval.csv"), index=False)
    g = df[df.label == 0]  # grounded spans: the attribution should mean something
    L = ["# Explanation-quality proxy (Qwen2.5-1.5B, large NLI)\n",
         f"Grounded spans evaluated: {len(g)} (of {len(df)} total).",
         "\n## Entailment of the drop-attributed chunk vs controls (grounded spans)",
         f"- attributed (max drop) chunk entails span: mean {g.e_att.mean():.3f}",
         f"- random chunk: mean {g.e_rand.mean():.3f}",
         f"- highest lexical-overlap chunk: mean {g.e_lex.mean():.3f}",
         f"- best possible chunk (oracle max entail): mean {g.e_best.mean():.3f}"]
    try:
        _, p1 = wilcoxon(g.e_att, g.e_rand, alternative="greater")
        _, p2 = wilcoxon(g.e_att, g.e_lex, alternative="greater")
        L.append(f"- paired Wilcoxon attributed > random: p = {p1:.2e}")
        L.append(f"- paired Wilcoxon attributed > lexical-overlap: p = {p2:.2e}")
    except Exception as e:
        L.append(f"- wilcoxon: {e}")
    L.append(f"\n## Attribution agreement with NLI-chosen support (grounded spans)")
    L.append(f"- drop-attributed chunk == NLI-max-entail chunk: {g.agree.mean():.1%}  (chance ~ 1/K = 20%)")
    L.append(f"\n## Best-chunk entailment separates grounded from hallucinated")
    gg, hh = df[df.label == 0].e_best, df[df.label == 1].e_best
    try:
        _, p3 = mannwhitneyu(gg, hh, alternative="greater")
        L.append(f"- grounded best-entail {gg.mean():.3f} vs hallucinated {hh.mean():.3f}, Mann-Whitney p = {p3:.2e}")
    except Exception as e:
        L.append(f"- mannwhitney: {e}")
    # Response-level aggregation (sentences within a response are not independent)
    gr = g.groupby("resp_id").agg(e_att=("e_att", "mean"), e_rand=("e_rand", "mean"),
                                  e_lex=("e_lex", "mean"), agree=("agree", "mean")).reset_index()
    L.append(f"\n## Response-level aggregation ({len(gr)} responses with grounded spans)")
    L.append(f"- attributed {gr.e_att.mean():.3f} vs random {gr.e_rand.mean():.3f} vs lexical {gr.e_lex.mean():.3f}")
    L.append(f"- response-level agreement mean {gr.agree.mean():.1%}")
    try:
        _, pr1 = wilcoxon(gr.e_att, gr.e_rand, alternative="greater")
        _, pr2 = wilcoxon(gr.e_att, gr.e_lex, alternative="greater")
        L.append(f"- paired Wilcoxon (per response) attributed > random: p = {pr1:.2e}")
        L.append(f"- paired Wilcoxon (per response) attributed > lexical: p = {pr2:.2e}")
    except Exception as e:
        L.append(f"- response wilcoxon: {e}")
    out = os.path.join(BASE, "EXPLANATION_EVAL.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("written to", out)

if __name__ == "__main__":
    main()
