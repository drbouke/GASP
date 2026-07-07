# -*- coding: utf-8 -*-
"""
Attractor analysis over 200 Qwen2.5-1.5B responses across early, middle, and late
layers.
  - correlation dimension per layer, with two references: the no-context
    trajectory and an iid-Gaussian cloud of the same size and dimension;
  - movement controls: the hidden-state shift from removing the most-influential-
    by-likelihood chunk is compared to a random chunk, a length-matched chunk, the
    highest lexical-overlap chunk, and the first chunk;
  - paired response-level statistics (Wilcoxon signed-rank).
Correlation dimension uses the Grassberger-Procaccia estimator (Euclidean metric,
scaling region between the 5th and 60th percentile of pairwise distances).
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, re, json
import numpy as np, pandas as pd

BASE = str(RESULTS)
SRC = "Qwen2.5-1.5B-Instruct_K5"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N, MAX_CTX, MAX_ANS = 200, 700, 160

def split_chunks(context, k):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', context.strip()) if s.strip()]
    if len(sents) <= k: return sents if sents else [context]
    per = int(np.ceil(len(sents) / k))
    return [" ".join(sents[i:i+per]) for i in range(0, len(sents), per)]

def corr_dim_cloud(X, lo_p=5, hi_p=60):
    from scipy.spatial.distance import pdist
    d = pdist(X.astype(np.float64)); d = d[d > 0]
    if d.size < 50: return np.nan
    lo, hi = np.percentile(d, lo_p), np.percentile(d, hi_p)
    if not (hi > lo > 0): return np.nan
    rs = np.logspace(np.log10(lo), np.log10(hi), 15)
    C = np.array([np.mean(d < r) for r in rs]); m = C > 0
    if m.sum() < 4: return np.nan
    return float(np.polyfit(np.log(rs[m]), np.log(C[m]), 1)[0])

def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)

def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, output_hidden_states=True).to(dev).eval()
    torch.set_grad_enabled(False)
    nl = model.config.num_hidden_layers
    layers = {"early": nl // 4, "mid": nl // 2, "late": 3 * nl // 4}
    MIDL = layers["mid"]
    print("layers", layers, "hidden", model.config.hidden_size)

    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    rdf = pd.read_csv(os.path.join(BASE, SRC, "response.csv"))
    rdf["cd"] = rdf["chunk_drops"].apply(json.loads)
    rng = np.random.default_rng(7)
    pos = rdf[rdf.label == 1].sample(min(N // 2, (rdf.label == 1).sum()), random_state=7)
    neg = rdf[rdf.label == 0].sample(min(N // 2, (rdf.label == 0).sum()), random_state=7)
    sub = pd.concat([pos, neg]).reset_index(drop=True)

    def hidden(context, query, answer):
        cids = tok(context, add_special_tokens=False).input_ids[:MAX_CTX]
        context = tok.decode(cids)
        aid = torch.tensor(tok(answer, add_special_tokens=False).input_ids[:MAX_ANS])
        pid = torch.tensor(tok(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: ").input_ids)
        ids = torch.cat([pid, aid]).unsqueeze(0).to(dev)
        P, A = pid.numel(), aid.numel()
        hs = model(ids).hidden_states  # tuple over layers
        return {name: hs[li][0][P:P+A].float().cpu().numpy() for name, li in layers.items()}

    rows, npoints = [], []
    for j, r in sub.iterrows():
        oi = int(r["orig_idx"]); ex = ds[oi]
        ctx = ex["context"].strip(); query = (ex["query"] or "").strip(); ans = ex["output"].strip()
        chunks = split_chunks(tok.decode(tok(ctx, add_special_tokens=False).input_ids[:MAX_CTX]), 5)
        if len(chunks) < 3: continue
        cd = r["cd"][:len(chunks)]
        ktop = int(np.argmax(cd))
        clen = [len(tok(c, add_special_tokens=False).input_ids) for c in chunks]
        # length-matched to ktop (excluding ktop)
        cand = [i for i in range(len(chunks)) if i != ktop]
        klen = min(cand, key=lambda i: abs(clen[i] - clen[ktop]))
        klex = max(cand, key=lambda i: jaccard(chunks[i], ans))
        krand = int(rng.choice(cand))
        kfirst = 0 if ktop != 0 else 1
        def rm(k): return " ".join(chunks[:k] + chunks[k+1:])
        try:
            Hf = hidden(ctx, query, ans)
            A = Hf["mid"].shape[0]
            if A < 16: continue
            npoints.append(A)
            Hn = hidden("", query, ans)
            def mv(kidx):
                Hk = hidden(rm(kidx), query, ans)
                a = min(A, Hk["mid"].shape[0])
                return float(np.mean(np.linalg.norm(Hf["mid"][:a] - Hk["mid"][:a], axis=1)))
            row = dict(label=int(r["label"]), drop_max=float(np.max(cd)),
                       cd_early=corr_dim_cloud(Hf["early"]), cd_mid=corr_dim_cloud(Hf["mid"]),
                       cd_late=corr_dim_cloud(Hf["late"]), cd_noctx_mid=corr_dim_cloud(Hn["mid"]),
                       move_top=mv(ktop), move_rand=mv(krand), move_len=mv(klen),
                       move_lex=mv(klex), move_first=mv(kfirst))
            rows.append(row)
        except Exception as e:
            print("skip", repr(e)[:60])
        if (j + 1) % 25 == 0: print(f"  {j+1}/{len(sub)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "attractor_controls.csv"), index=False)
    # iid-Gaussian high-dim control: same median #points, same hidden dim
    npt = int(np.median(npoints)) if npoints else 128
    gcd = corr_dim_cloud(rng.standard_normal((max(npt, 60), model.config.hidden_size)))

    L = ["# Attractor controls (Qwen2.5-1.5B)\n",
         f"Sample: {len(df)} responses; points per trajectory median {npt}. "
         f"Grassberger-Procaccia: Euclidean metric, scaling region 5th-60th percentile of pairwise distances, 15 radii.\n",
         "## Correlation dimension by layer (finite and stable => low-dimensional attractor)",
         f"- early (layer {layers['early']}): {df.cd_early.mean():.2f}",
         f"- mid (layer {layers['mid']}): {df.cd_mid.mean():.2f}",
         f"- late (layer {layers['late']}): {df.cd_late.mean():.2f}",
         f"- no-context trajectory (mid): {df.cd_noctx_mid.mean():.2f}",
         f"- iid-Gaussian control ({model.config.hidden_size}-dim, {max(npt,60)} pts): {gcd:.2f}  (must be high)",
         "\n## Trajectory movement under chunk removal (mid layer), response-level means",
         f"- most-influential (by likelihood): {df.move_top.mean():.3f}",
         f"- random chunk: {df.move_rand.mean():.3f}",
         f"- length-matched chunk: {df.move_len.mean():.3f}",
         f"- highest lexical-overlap chunk: {df.move_lex.mean():.3f}",
         f"- first chunk: {df.move_first.mean():.3f}"]
    def paired(a, b, name):
        try:
            s, p = wilcoxon(df[a], df[b], alternative="greater")
            L.append(f"- paired Wilcoxon {a} > {b}: p = {p:.2e}")
        except Exception as e:
            L.append(f"- {name}: {e}")
    L.append("\n## Paired response-level tests (Wilcoxon signed-rank)")
    for b in ["move_rand", "move_len", "move_lex", "move_first"]:
        paired("move_top", b, b)
    rho, pr = spearmanr(df.move_top, df.drop_max)
    L.append(f"\nSpearman(move_top, likelihood drop_max), response-level: {rho:.3f} (p={pr:.2e})")
    g, h = df[df.label == 0].move_top, df[df.label == 1].move_top
    u, p2 = mannwhitneyu(g, h, alternative="greater")
    L.append(f"move_top grounded {g.mean():.3f} vs hallucinated {h.mean():.3f}, Mann-Whitney p = {p2:.2e}")
    out = os.path.join(BASE, "ATTRACTOR_CONTROLS.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("written to", out)

if __name__ == "__main__":
    main()
