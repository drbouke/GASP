# -*- coding: utf-8 -*-
"""
Attractor visualization (paper Figure 9). The answer-token hidden-state trajectory
of a representative grounded response scored by Qwen2.5-1.5B, under the full
context, removal of the most-influential chunk, and removal of a length-matched
control chunk. Produces fig6_attractor_traj.png: (a) a 2D PCA projection of the
three trajectories and (b) the per-token hidden-state shift from the full-context
trajectory. Requires a CUDA device; downloads the model and RAGTruth on first use.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS, FIGURES
import os, re, json
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "Qwen2.5-1.5B-Instruct_K5"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAXCTX, MAXANS = 700, 160
OI = 975  # representative grounded response (movement ~9.1 vs ~3.7, close to sample means)


def split_chunks(context, k):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', context.strip()) if s.strip()]
    if len(sents) <= k:
        return sents if sents else [context]
    per = int(np.ceil(len(sents) / k))
    return [" ".join(sents[i:i + per]) for i in range(0, len(sents), per)]


dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, output_hidden_states=True).to(dev).eval()
torch.set_grad_enabled(False)
LAYER = (model.config.num_hidden_layers + 1) // 2
ds = load_dataset("wandb/RAGTruth-processed", split="train")
rdf = pd.read_csv(os.path.join(str(RESULTS), TAG, "response.csv"))
rdf["cd"] = rdf["chunk_drops"].apply(json.loads)
r = rdf[rdf.orig_idx == OI].iloc[0]
ex = ds[OI]
ctx = ex["context"].strip()
q = (ex["query"] or "").strip()
ans = ex["output"].strip()


def hidden(context):
    cids = tok(context, add_special_tokens=False).input_ids[:MAXCTX]
    context = tok.decode(cids)
    aid = torch.tensor(tok(ans, add_special_tokens=False).input_ids[:MAXANS])
    pid = torch.tensor(tok(f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer: ").input_ids)
    ids = torch.cat([pid, aid]).unsqueeze(0).to(dev)
    P, A = pid.numel(), aid.numel()
    return model(ids).hidden_states[LAYER][0][P:P + A].float().cpu().numpy()


chunks = split_chunks(tok.decode(tok(ctx, add_special_tokens=False).input_ids[:MAXCTX]), 5)
cd = r["cd"][:len(chunks)]
ktop = int(np.argmax(cd))
tlen = len(tok(chunks[ktop], add_special_tokens=False).input_ids)
others = [i for i in range(len(chunks)) if i != ktop]
kctrl = min(others, key=lambda i: abs(len(tok(chunks[i], add_special_tokens=False).input_ids) - tlen))
hf = hidden(ctx)
ht = hidden(" ".join(chunks[:ktop] + chunks[ktop + 1:]))
hc = hidden(" ".join(chunks[:kctrl] + chunks[kctrl + 1:]))
A = min(len(hf), len(ht), len(hc))
hf, ht, hc = hf[:A], ht[:A], hc[:A]
dtop = np.linalg.norm(hf - ht, axis=1)
dctrl = np.linalg.norm(hf - hc, axis=1)
print(f"A={A} ktop=#{ktop} kctrl=#{kctrl} move_top={dtop.mean():.2f} move_ctrl={dctrl.mean():.2f}")
pca = PCA(n_components=2).fit(np.vstack([hf, ht, hc]))
ev = pca.explained_variance_ratio_ * 100
Zf, Zt, Zc = pca.transform(hf), pca.transform(ht), pca.transform(hc)

NAVY, GREEN, RED = "#1f3b73", "#2e8b3d", "#c1272d"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
ax1.plot(Zf[:, 0], Zf[:, 1], "-", color=NAVY, lw=1.3, alpha=0.5, label="Full context", zorder=2)
ax1.plot(Zc[:, 0], Zc[:, 1], "-", color=GREEN, lw=1.3, alpha=0.5, label="Length-matched control removed", zorder=3)
ax1.plot(Zt[:, 0], Zt[:, 1], "-", color=RED, lw=1.5, alpha=0.95, label="Most influential chunk removed", zorder=5)
ax1.scatter(Zt[:, 0], Zt[:, 1], s=9, color=RED, alpha=0.9, zorder=6, edgecolors="none")
ax1.scatter([Zf[0, 0]], [Zf[0, 1]], s=120, marker="*", color="black", zorder=9)
ax1.annotate("start", (Zf[0, 0], Zf[0, 1]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax1.set_xlabel(f"PC1 ({ev[0]:.0f}%)")
ax1.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
ax1.set_title("(a) Answer-token trajectory (2D PCA)", fontsize=10)
ax1.legend(fontsize=7.5, loc="best")
t = np.arange(A)
ax2.plot(t, dtop, color=RED, lw=1.8, label=f"Influential removed (mean {dtop.mean():.1f})")
ax2.plot(t, dctrl, color=GREEN, lw=1.8, label=f"Control removed (mean {dctrl.mean():.1f})")
ax2.fill_between(t, dctrl, dtop, where=(dtop >= dctrl), color=RED, alpha=0.08)
ax2.axhline(dtop.mean(), color=RED, ls=":", lw=1, alpha=0.7)
ax2.axhline(dctrl.mean(), color=GREEN, ls=":", lw=1, alpha=0.7)
ax2.set_xlabel("Answer token index")
ax2.set_ylabel("Hidden-state shift from full context")
ax2.set_title("(b) Per-token displacement", fontsize=10)
ax2.legend(fontsize=8, loc="best")
plt.tight_layout()
os.makedirs(str(FIGURES), exist_ok=True)
out = os.path.join(str(FIGURES), "fig6_attractor_traj.png")
plt.savefig(out, dpi=200)
print("saved", out)
