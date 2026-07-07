# -*- coding: utf-8 -*-
"""
Regenerates the paper's headline figures from the scored results, so figures and
tables come from one consistent, leakage-clean experiment with three scorers.
Produces fig1_response_auc, fig2_sentence_auc, fig3_roc, fig4_dist, fig5_imp as
PNGs in the paper's figure directory.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS, FIGURES
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

BASE = str(RESULTS)
OUT = str(FIGURES)
RNG = np.random.default_rng(0)
GROUND = ["gap", "jsd_noctx", "drop_max", "jsd_loo_max"]
BASECOLS = ["mean_surprisal", "length"]
MODELS = [("Qwen2.5-0.5B-Instruct_K5", "Qwen2.5-0.5B"),
          ("Qwen2.5-1.5B-Instruct_K5", "Qwen2.5-1.5B"),
          ("SmolLM2-1.7B-Instruct_K5", "SmolLM2-1.7B")]
PRIMARY, ACCENT, GREY = "#1f4e79", "#c0504d", "#7f7f7f"
PALETTE = ["#9dc3e6", "#1f4e79", "#c55a11"]  # 0.5B, 1.5B, SmolLM2

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

def clf(seed=42):
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)

def add_features(df):
    cd = df["chunk_drops"].apply(json.loads); cj = df["chunk_jsds"].apply(json.loads)
    df["drop_max"] = cd.apply(lambda v: float(np.max(v)) if v else np.nan)
    df["jsd_loo_max"] = cj.apply(lambda v: float(np.max(v)) if v else np.nan)
    df["length"] = df["n_ans"] if "n_ans" in df.columns else df["n_tok"]
    return df

def oof(df, cols, grouped):
    d = df.dropna(subset=cols + ["label"]).reset_index(drop=True)
    y = d["label"].to_numpy()
    if grouped:
        cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d[cols], y, d["resp_id"]))
    else:
        cv = list(StratifiedKFold(5, shuffle=True, random_state=42).split(d[cols], y))
    p = cross_val_predict(clf(), d[cols].to_numpy(), y, cv=cv, method="predict_proba")[:, 1]
    return p, y, d["resp_id"].to_numpy(), d

def ci(y, p, groups=None, B=1000):
    aucs = []
    if groups is not None:
        uniq = np.unique(groups); gmap = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(B):
            gs = RNG.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([gmap[g] for g in gs]); yy = y[idx]
            if len(np.unique(yy)) < 2: continue
            aucs.append(roc_auc_score(yy, p[idx]))
    else:
        n = len(y)
        for _ in range(B):
            idx = RNG.integers(0, n, n); yy = y[idx]
            if len(np.unique(yy)) < 2: continue
            aucs.append(roc_auc_score(yy, p[idx]))
    a = np.array(aucs)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)

def load(tag):
    r = add_features(pd.read_csv(os.path.join(BASE, tag, "response.csv")))
    s = add_features(pd.read_csv(os.path.join(BASE, tag, "sentence.csv")))
    return r, s

def threshold_oof(df, grouped):
    d = df.dropna(subset=GROUND + ["label"]).reset_index(drop=True)
    y = d["label"].to_numpy(); oof = np.zeros(len(d))
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d[GROUND], y, d["resp_id"]) if grouped
          else StratifiedKFold(5, shuffle=True, random_state=42).split(d[GROUND], y))
    for tr, te in cv:
        mu = d[GROUND].iloc[tr].mean(); sd = d[GROUND].iloc[tr].std() + 1e-9
        oof[te] = ((d[GROUND].iloc[te] - mu) / sd).sum(axis=1).to_numpy()
    return y, -oof, (d["resp_id"].to_numpy() if grouped else None)

FEATSETS = [("Perplexity", "clf", ["mean_surprisal"]), ("Length", "clf", ["length"]),
            ("GASP-thr", "thr", None), ("GASP-trn", "clf", GROUND), ("GASP+base", "clf", GROUND + BASECOLS)]

def bars(level_grouped, fname, title):
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    x = np.arange(len(FEATSETS)); w = 0.26
    for mi, (tag, name) in enumerate(MODELS):
        r, s = load(tag); df = s if level_grouped else r
        means, los, his = [], [], []
        for label, mode, cols in FEATSETS:
            if mode == "thr":
                y, p, g = threshold_oof(df, level_grouped)
            else:
                p, y, g, _ = oof(df, cols, level_grouped); g = g if level_grouped else None
            m, lo, hi = ci(y, p, g)
            means.append(m); los.append(m - lo); his.append(hi - m)
        ax.bar(x + (mi - 1) * w, means, w, yerr=[los, his], capsize=3,
               label=name, color=PALETTE[mi], edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, ls="--", c=GREY, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in FEATSETS])
    ax.set_ylabel("AUC"); ax.set_ylim(0.45, 0.82); ax.set_title(title)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)
    print("wrote", fname)

def roc_fig():
    # One panel per scorer. GASP grounding and perplexity are scorer-dependent;
    # the whole-context NLI baseline is scorer-independent, drawn as a shared reference.
    nli_fpr = nli_tpr = nli_auc = None
    nli_path = os.path.join(BASE, "Qwen2.5-1.5B-Instruct_K5", "sentence_nli_large.csv")
    if os.path.exists(nli_path):
        n = pd.read_csv(nli_path).dropna(subset=["nli_large", "label"])
        nli_fpr, nli_tpr, _ = roc_curve(n["label"], -n["nli_large"])
        nli_auc = roc_auc_score(n["label"], -n["nli_large"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (tag, name) in zip(axes, MODELS):
        _, s = load(tag)
        pg, yg, gg, dg = oof(s, GROUND, True)
        fpr, tpr, _ = roc_curve(yg, pg); ax.plot(fpr, tpr, c=PRIMARY, lw=2,
            label=f"GASP grounding (AUC {roc_auc_score(yg, pg):.2f})")
        pp, yp, gp, _ = oof(s, ["mean_surprisal"], True)
        fpr, tpr, _ = roc_curve(yp, pp); ax.plot(fpr, tpr, c=ACCENT, lw=1.6,
            label=f"Perplexity (AUC {roc_auc_score(yp, pp):.2f})")
        if nli_fpr is not None:
            ax.plot(nli_fpr, nli_tpr, c=GREY, lw=1.5, ls=":", label=f"NLI-large (AUC {nli_auc:.2f})")
        ax.plot([0, 1], [0, 1], ls="--", c=GREY, lw=1)
        ax.set_xlabel("False positive rate"); ax.set_title(name)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_roc_sentence.png")); plt.close(fig); print("wrote fig4_roc_sentence.png")

def dist_fig():
    # One panel per scorer, shared axes for comparison.
    data, xmax = [], 0.0
    for tag, name in MODELS:
        _, s = load(tag); d = s.dropna(subset=["drop_max", "label"])
        data.append((d, name)); xmax = max(xmax, float(d["drop_max"].quantile(0.99)))
    bins = np.linspace(0, xmax, 40)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True, sharex=True)
    for ax, (d, name) in zip(axes, data):
        ax.hist(d[d.label == 0]["drop_max"], bins=bins, density=True, alpha=0.6, color=PRIMARY, label="grounded")
        ax.hist(d[d.label == 1]["drop_max"], bins=bins, density=True, alpha=0.6, color=ACCENT, label="hallucinated")
        ax.set_xlabel("max leave-one-out likelihood drop"); ax.set_title(name)
    axes[0].set_ylabel("density"); axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_grounding_distribution.png")); plt.close(fig); print("wrote fig3_grounding_distribution.png")

def imp_fig():
    # One panel per scorer, shared feature order (by mean gain) so labels align.
    # Color separates the four grounding features from the two baseline features.
    from matplotlib.patches import Patch
    cols = GROUND + BASECOLS
    disp = np.array(["gap", r"jsd$_{\varnothing}$", "drop", r"jsd$_{\mathrm{loo}}$", "perplexity", "length"])
    is_ground = np.array([True, True, True, True, False, False])
    GCOL, BCOL = "#1f4e79", "#d9822b"   # grounding feature (navy), baseline (amber)
    imps = []
    for tag, name in MODELS:
        _, s = load(tag); d = s.dropna(subset=cols + ["label"])
        m = clf().fit(d[cols].to_numpy(), d["label"].to_numpy())
        v = m.feature_importances_.astype(float); imps.append(v / v.sum())
    order = np.argsort(np.mean(imps, axis=0))
    xmax = max(float(v.max()) for v in imps)
    ypos = np.arange(len(order))
    colors = [GCOL if is_ground[i] else BCOL for i in order]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), sharey=True)
    for ax, (tag, name), v in zip(axes, MODELS, imps):
        vals = v[order]
        ax.barh(ypos, vals, color=colors, edgecolor="white", linewidth=0.7, height=0.72)
        for yp, val in zip(ypos, vals):
            ax.text(val + xmax * 0.02, yp, f"{val:.2f}", va="center", ha="left", fontsize=8.5, color="#404040")
        ax.set_yticks(ypos)
        ax.set_xlabel("relative importance (gain)"); ax.set_title(name)
        ax.set_xlim(0, xmax * 1.24)
        ax.grid(axis="x", color="0.9", lw=0.7); ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_yticklabels(disp[order])
    handles = [Patch(facecolor=GCOL, label="grounding feature"), Patch(facecolor=BCOL, label="baseline")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "fig5_feature_importance.png")); plt.close(fig); print("wrote fig5_feature_importance.png")

if __name__ == "__main__":
    bars(False, "fig1_response_auc.png", "Response-level AUC")
    bars(True, "fig2_sentence_auc.png", "Span-level AUC (leakage-clean)")
    roc_fig(); dist_fig(); imp_fig()
    print("all figures done")
