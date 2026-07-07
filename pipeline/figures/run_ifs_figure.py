# -*- coding: utf-8 -*-
"""
Grounding-as-attractor-movement figure (paper Figure 8). Per-token log-probability
of the two case-study spans, with and without the retrieved context, under
Qwen2.5-1.5B. The grounded span's tokens collapse when the context is removed,
while the hallucinated span is almost unchanged; the mean gap between the curves is
the grounding-sensitivity signal. Produces fig7_ifs_motivation.png. Requires a CUDA
device; downloads the model and RAGTruth on first use.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS, FIGURES
import os, re
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = "Qwen2.5-1.5B-Instruct_K5"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAXCTX, MAXANS = 700, 200

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
torch.set_grad_enabled(False)
ds = load_dataset("wandb/RAGTruth-processed", split="train")
sdf = pd.read_csv(os.path.join(str(RESULTS), TAG, "sentence.csv"))


def tok_logprobs(context, query, answer):
    cids = tok(context, add_special_tokens=False).input_ids[:MAXCTX] if context is not None else None
    enc = tok(answer, return_offsets_mapping=True, add_special_tokens=False)
    aid = torch.tensor(enc["input_ids"][:MAXANS])
    offs = enc["offset_mapping"][:MAXANS]
    pref = f"Question: {query}\n\nAnswer: " if context is None else f"Context:\n{tok.decode(cids)}\n\nQuestion: {query}\n\nAnswer: "
    pid = torch.tensor(tok(pref).input_ids)
    ids = torch.cat([pid, aid]).unsqueeze(0).to(dev)
    P, A = pid.numel(), aid.numel()
    lp = torch.log_softmax(model(ids).logits[0][P - 1:P - 1 + A].float(), dim=-1)
    tlp = lp[torch.arange(A), aid.to(dev)].cpu().numpy()
    return aid.numpy(), offs, tlp


def span_tokens(offs, aid, tlpF, tlpN, answer, key):
    s = answer.find(key)
    e = s + len(key)
    idx = [i for i, (a, b) in enumerate(offs) if a >= s and a < e]
    toks = [tok.decode([int(aid[i])]).strip() for i in idx]
    return toks, np.array([tlpF[i] for i in idx]), np.array([tlpN[i] for i in idx])


picks = {"grounded": "sizzling rice soup and imperial shrimp", "halluc": "vegetarian, gluten-free, and vegan"}
data = {}
for kind, key in picks.items():
    row = sdf[sdf.sent_text.str.contains(re.escape(key))].iloc[0]
    ex = ds[int(row.orig_idx)]
    ctx = ex["context"].strip()
    q = (ex["query"] or "").strip()
    ans = ex["output"].strip()
    aid, offs, tlpF = tok_logprobs(ctx, q, ans)
    _, _, tlpN = tok_logprobs(None, q, ans)
    toks, lpF, lpN = span_tokens(offs, aid, tlpF, tlpN, ans, row.sent_text if row.sent_text in ans else key)
    data[kind] = (toks, lpF, lpN)
    print(kind, "gap=%.2f" % float(np.mean(lpF - lpN)), "ntok", len(toks))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
titles = {"grounded": "(a) Grounded span", "halluc": "(b) Hallucinated span"}
NAVY, RED = "#1f3b73", "#c1272d"
for ax, kind in zip(axes, ["grounded", "halluc"]):
    toks, lpF, lpN = data[kind]
    x = np.arange(len(toks))
    ax.plot(x, lpF, "-o", color=NAVY, lw=2, ms=4, label="With retrieved context")
    ax.plot(x, lpN, "--o", color=RED, lw=2, ms=4, label="Without context (prior)")
    ax.fill_between(x, lpN, lpF, where=(lpF >= lpN), color=NAVY, alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=7)
    ax.set_title(titles[kind] + f"  (mean gap {np.mean(lpF - lpN):+.2f} nats)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("token log-probability")
plt.tight_layout()
os.makedirs(str(FIGURES), exist_ok=True)
out = os.path.join(str(FIGURES), "fig7_ifs_motivation.png")
plt.savefig(out, dpi=200)
print("saved", out)
