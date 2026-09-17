# -*- coding: utf-8 -*-
"""
Generate the paper figures into results/figures/.

Four figures on RAGTruth with the 14B scorer:
  fig_feature_auc.png     single-feature span-level ROC-AUC of each grounding feature.
  fig_grounding_dist.png  distribution of the max leave-one-out drop, grounded vs unsupported.
  fig_transfer.png        span-level GASP+base AUC across the three benchmarks.
  fig_cost_accuracy.png   compute (GFLOPs/sentence) vs span-level AUC per method.

The distribution figure reads canon_results/<tag>/sentence.csv (produced by run_gasp.py);
pass --data to point at the canon_results directory. The AUC and cost values are the numbers
reported in the paper's tables and are listed below so the plots are reproducible without the
full score files.

Usage: python make_paper_figures.py [--data canon_results] [--out ../results/figures]
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1f4e79"; GREY = "#7f8c8d"; ORANGE = "#c0392b"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "font.family": "DejaVu Sans"})

# single-feature span-level AUC (14B, RAGTruth)
FEATURES = [("gap", 0.745), ("top-2 drop", 0.711), ("max drop", 0.709), ("mean drop", 0.704),
            ("perplexity", 0.657), ("JSD no-ctx", 0.648), ("max JSD", 0.613)]
GROUNDING = {"gap", "top-2 drop", "max drop", "mean drop", "JSD no-ctx", "max JSD"}
# span-level GASP+base per benchmark
TRANSFER = [("RAGTruth", 0.773), ("TofuEval", 0.711), ("RAGBench", 0.634)]
# (method, span-AUC, GFLOPs/sentence, is_gasp, label-dx-factor, label-dy)
COST = [("chunk-NLI (max)", 0.721, 638, False, 1.10, +0.006),
        ("AlignScore (ctx)", 0.791, 608, False, 0.60, +0.006),
        ("MiniCheck (ctx)", 0.831, 717, False, 1.10, 0.0),
        ("MiniCheck (max)", 0.792, 645, False, 1.10, -0.010),
        ("LLM-judge (14B)", 0.853, 20376, False, 0.62, +0.006),
        ("ContextCite", 0.736, 26103, False, 0.42, -0.004),
        ("GASP-gap (1.5B)", 0.717, 429, True, 1.12, -0.010),
        ("GASP-gap (14B)", 0.745, 4236, True, 1.12, +0.004),
        ("GASP-full (14B)", 0.773, 17897, True, 0.55, +0.007)]


def fig_feature_auc(out):
    names = [f[0] for f in FEATURES]; vals = [f[1] for f in FEATURES]
    cols = [NAVY if n in GROUNDING else GREY for n in names]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.66)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0.5, 0.8)
    ax.axvline(0.5, color="k", lw=0.8, ls=":")
    for yi, v in zip(y, vals):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("single-feature span-level ROC-AUC")
    ax.set_title("Grounding features (navy) vs perplexity (grey), 14B scorer")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_feature_auc.png"), dpi=200); plt.close(fig)


def fig_grounding_dist(out, data, tag="RT_14B"):
    import pandas as pd
    path = os.path.join(data, tag, "sentence.csv")
    if not os.path.exists(path):
        print(f"  skip fig_grounding_dist: {path} not found")
        return
    df = pd.read_csv(path)

    def max_drop(js):
        v = [x for x in json.loads(js) if x is not None]
        return float(np.max(v)) if v else np.nan
    df["max_drop"] = df["chunk_drops"].map(max_drop)
    md = df.dropna(subset=["max_drop"])
    g = md[md["label"] == 0]["max_drop"].clip(-0.5, 3)
    u = md[md["label"] == 1]["max_drop"].clip(-0.5, 3)
    fig, ax = plt.subplots(figsize=(6.2, 3.2)); bins = np.linspace(-0.5, 3, 40)
    ax.hist(g, bins=bins, color=NAVY, alpha=0.6, density=True, label="grounded")
    ax.hist(u, bins=bins, color=ORANGE, alpha=0.6, density=True, label="unsupported")
    ax.set_xlabel("grounding sensitivity (max log-likelihood drop on context removal)")
    ax.set_ylabel("density"); ax.legend()
    ax.set_title("Grounded spans react more to evidence removal")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_grounding_dist.png"), dpi=200); plt.close(fig)


def fig_transfer(out):
    fig, ax = plt.subplots(figsize=(5.2, 3.2)); x = np.arange(len(TRANSFER))
    ax.bar(x, [t[1] for t in TRANSFER], color=[NAVY, NAVY, GREY], width=0.55)
    for xi, t in zip(x, TRANSFER):
        ax.text(xi, t[1] + 0.006, f"{t[1]:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([t[0] for t in TRANSFER]); ax.set_ylim(0.5, 0.85)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_ylabel("span-level ROC-AUC (GASP+base, 14B)")
    ax.set_title("Transfer across benchmarks")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_transfer.png"), dpi=200); plt.close(fig)


def fig_cost_accuracy(out):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, auc, fl, isg, dx, dy in COST:
        ax.scatter(fl, auc, s=72, marker="D" if isg else "o",
                   color=NAVY if isg else GREY, zorder=3, edgecolor="k", linewidth=0.4)
        ax.annotate(name, (fl, auc), xytext=(fl * dx, auc + dy), fontsize=8,
                    color=NAVY if isg else "black", fontweight="bold" if isg else "normal")
    ax.set_xscale("log"); ax.set_xlim(3e2, 6e4); ax.set_ylim(0.70, 0.87)
    ax.set_xlabel("compute per sentence (GFLOPs, log scale)")
    ax.set_ylabel("span-level ROC-AUC")
    ax.set_title("Cost against accuracy on RAGTruth (GASP = navy diamonds)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_cost_accuracy.png"), dpi=200); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "canon_results"))
    ap.add_argument("--out", default=os.path.join(here, "..", "results", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_feature_auc(a.out)
    fig_grounding_dist(a.out, a.data)
    fig_transfer(a.out)
    fig_cost_accuracy(a.out)
    print("wrote figures to", a.out)


if __name__ == "__main__":
    main()
