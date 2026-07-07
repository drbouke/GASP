# -*- coding: utf-8 -*-
"""
Empirical attractor estimation for the strengthened theory. For a sample of
responses it extracts the answer's hidden-state trajectory (mid layer) under the
full context and measures:
  (1) the correlation dimension of the trajectory (finite => a low-dimensional
      attractor, not space-filling),
  (2) the movement of the trajectory when the most-influential context chunk is
      removed versus a random chunk (per-token hidden-state shift), which is the
      empirical counterpart of moving the invariant measure,
  (3) whether that movement is larger for grounded than for hallucinated
      responses and whether it tracks the grounding-stability signal (drop_max).
This links the measured JSD/likelihood signal to a movement of the attractor.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, re, json, argparse
import numpy as np, pandas as pd

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--src_tag", default="Qwen2.5-1.5B-Instruct_K5")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--max_ctx_tokens", type=int, default=700)
    p.add_argument("--max_ans_tokens", type=int, default=160)
    p.add_argument("--base", default=str(RESULTS))
    return p.parse_args()

def split_chunks(context, k):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', context.strip()) if s.strip()]
    if len(sents) <= k: return sents if sents else [context]
    per = int(np.ceil(len(sents) / k))
    return [" ".join(sents[i:i+per]) for i in range(0, len(sents), per)]

def corr_dim_cloud(X):
    from scipy.spatial.distance import pdist
    d = pdist(X.astype(np.float64)); d = d[d > 0]
    if d.size < 50: return np.nan
    lo, hi = np.percentile(d, 5), np.percentile(d, 60)
    if not (hi > lo > 0): return np.nan
    rs = np.logspace(np.log10(lo), np.log10(hi), 15)
    C = np.array([np.mean(d < r) for r in rs]); m = C > 0
    if m.sum() < 4: return np.nan
    return float(np.polyfit(np.log(rs[m]), np.log(C[m]), 1)[0])

def main():
    args = get_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from scipy.stats import spearmanr, mannwhitneyu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, output_hidden_states=True).to(dev).eval()
    torch.set_grad_enabled(False)
    layer = (model.config.num_hidden_layers + 1) // 2
    print("layer", layer)

    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    rdf = pd.read_csv(os.path.join(args.base, args.src_tag, "response.csv"))
    rdf["cd"] = rdf["chunk_drops"].apply(json.loads)
    # balanced sample
    rng = np.random.default_rng(1)
    pos = rdf[rdf.label == 1].sample(min(args.n//2, (rdf.label==1).sum()), random_state=1)
    neg = rdf[rdf.label == 0].sample(min(args.n//2, (rdf.label==0).sum()), random_state=1)
    sub = pd.concat([pos, neg]).reset_index(drop=True)

    def hidden(context, query, answer):
        cids = tok(context, add_special_tokens=False).input_ids[:args.max_ctx_tokens]
        context = tok.decode(cids)
        aid = torch.tensor(tok(answer, add_special_tokens=False).input_ids[:args.max_ans_tokens])
        pid = torch.tensor(tok(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: ").input_ids)
        ids = torch.cat([pid, aid]).unsqueeze(0).to(dev)
        P, A = pid.numel(), aid.numel()
        hs = model(ids).hidden_states[layer][0][P:P+A].float().cpu().numpy()
        return hs  # (A, H) answer-token hidden states

    rows = []
    for j, r in sub.iterrows():
        oi = int(r["orig_idx"]); ex = ds[oi]
        ctx = ex["context"].strip(); query = (ex["query"] or "").strip(); ans = ex["output"].strip()
        chunks = split_chunks(tok.decode(tok(ctx, add_special_tokens=False).input_ids[:args.max_ctx_tokens]), 5)
        if len(chunks) < 2: continue
        cd = r["cd"]; ktop = int(np.argmax(cd[:len(chunks)]))
        krand = int(rng.integers(0, len(chunks)))
        while krand == ktop and len(chunks) > 1: krand = int(rng.integers(0, len(chunks)))
        try:
            h_full = hidden(ctx, query, ans)
            if h_full.shape[0] < 16: continue
            h_top = hidden(" ".join(chunks[:ktop]+chunks[ktop+1:]), query, ans)
            h_rnd = hidden(" ".join(chunks[:krand]+chunks[krand+1:]), query, ans)
            A = min(h_full.shape[0], h_top.shape[0], h_rnd.shape[0])
            move_top = float(np.mean(np.linalg.norm(h_full[:A]-h_top[:A], axis=1)))
            move_rnd = float(np.mean(np.linalg.norm(h_full[:A]-h_rnd[:A], axis=1)))
            rows.append(dict(label=int(r["label"]), corr_dim=corr_dim_cloud(h_full),
                             move_top=move_top, move_rand=move_rnd, drop_max=float(np.max(cd))))
        except Exception as e:
            print("skip", repr(e)[:60])
        if (j+1) % 20 == 0: print(f"  {j+1}/{len(sub)}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.base, "attractor.csv"), index=False)
    cd = df["corr_dim"].dropna()
    print(f"\nAttractor correlation dimension: mean {cd.mean():.2f}, median {cd.median():.2f} (hidden dim {model.config.hidden_size})")
    print(f"Trajectory movement: remove-most-influential {df.move_top.mean():.3f} vs remove-random {df.move_rand.mean():.3f}")
    try:
        u, p = mannwhitneyu(df.move_top, df.move_rand, alternative='greater')
        print(f"  move_top > move_rand: Mann-Whitney p = {p:.2e}")
        rho, pr = spearmanr(df.move_top, df.drop_max)
        print(f"Spearman(move_top, grounding drop_max) = {rho:.3f} (p={pr:.2e})")
        g = df[df.label==0].move_top; h = df[df.label==1].move_top
        u2, p2 = mannwhitneyu(g, h, alternative='greater')
        print(f"move_top grounded {g.mean():.3f} vs hallucinated {h.mean():.3f}, p = {p2:.2e}")
    except Exception as e:
        print("stats skipped:", e)

if __name__ == "__main__":
    main()
