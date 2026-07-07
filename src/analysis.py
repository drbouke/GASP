# -*- coding: utf-8 -*-
"""
Offline analysis for GASP. Consumes results/<tag>/{response,sentence}.csv
(with per-chunk drop/JSD vectors) and produces:
  - feature ablation (single features and groups),
  - aggregation study (max vs mean vs top-2),
  - grounding AUC with bootstrap 95% CI (grouped by response at span level),
  - significance vs perplexity (paired bootstrap),
  - threshold-only (no-classifier) variant,
  - chunking sensitivity (Qwen1.5B across K).
Writes ANALYSIS.md and per-table CSVs. CPU only.
"""

from config import RESULTS
import os, json, argparse
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)

def clf(seed=42):
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)

def add_features(df):
    cd = df["chunk_drops"].apply(json.loads); cj = df["chunk_jsds"].apply(json.loads)
    df["drop_max"] = cd.apply(lambda v: float(np.max(v)) if v else np.nan)
    df["drop_mean"] = cd.apply(lambda v: float(np.mean(v)) if v else np.nan)
    df["drop_top2"] = cd.apply(lambda v: float(np.sum(sorted(v)[-2:])) if v else np.nan)
    df["jsd_loo_max"] = cj.apply(lambda v: float(np.max(v)) if v else np.nan)
    df["jsd_loo_mean"] = cj.apply(lambda v: float(np.mean(v)) if v else np.nan)
    df["jsd_loo_top2"] = cj.apply(lambda v: float(np.sum(sorted(v)[-2:])) if v else np.nan)
    df["length"] = df["n_ans"] if "n_ans" in df.columns else df["n_tok"]
    return df

def oof_proba(df, cols, grouped):
    d = df.dropna(subset=cols + ["label"]).reset_index(drop=True)
    y = d["label"].to_numpy()
    if len(np.unique(y)) < 2 or len(d) < 25:
        return None, None, None
    if grouped:
        cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d[cols], y, d["resp_id"]))
    else:
        cv = list(StratifiedKFold(5, shuffle=True, random_state=42).split(d[cols], y))
    p = cross_val_predict(clf(), d[cols].to_numpy(), y, cv=cv, method="predict_proba")[:, 1]
    return p, y, d["resp_id"].to_numpy()

def boot_auc_ci(y, p, groups=None, B=1000):
    aucs = []
    n = len(y)
    if groups is not None:
        uniq = np.unique(groups)
        gmap = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(B):
            gs = RNG.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([gmap[g] for g in gs])
            yy, pp = y[idx], p[idx]
            if len(np.unique(yy)) < 2: continue
            aucs.append(roc_auc_score(yy, pp))
    else:
        for _ in range(B):
            idx = RNG.integers(0, n, n)
            yy, pp = y[idx], p[idx]
            if len(np.unique(yy)) < 2: continue
            aucs.append(roc_auc_score(yy, pp))
    a = np.array(aucs)
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def boot_diff(y, p1, p2, groups=None, B=1000):
    """Paired bootstrap of AUC(p1) - AUC(p2). Returns mean diff, CI, and one-sided p (H0: diff<=0)."""
    diffs = []
    if groups is not None:
        uniq = np.unique(groups); gmap = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(B):
            gs = RNG.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([gmap[g] for g in gs])
            yy = y[idx]
            if len(np.unique(yy)) < 2: continue
            diffs.append(roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p2[idx]))
    else:
        n = len(y)
        for _ in range(B):
            idx = RNG.integers(0, n, n); yy = y[idx]
            if len(np.unique(yy)) < 2: continue
            diffs.append(roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p2[idx]))
    d = np.array(diffs)
    p_val = float(np.mean(d <= 0))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), p_val

GROUND = ["gap", "jsd_noctx", "drop_max", "jsd_loo_max"]

def auc_ci_for(df, cols, grouped, lines, label):
    p, y, g = oof_proba(df, cols, grouped)
    if p is None:
        lines.append(f"- {label}: n/a"); return None
    m, lo, hi = boot_auc_ci(y, p, g if grouped else None)
    lines.append(f"- {label}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")
    return (p, y, g)

def analyze_model(base, tag, lines):
    rp = os.path.join(base, tag, "response.csv"); sp = os.path.join(base, tag, "sentence.csv")
    if not os.path.exists(sp): return
    rdf = add_features(pd.read_csv(rp)); sdf = add_features(pd.read_csv(sp))
    lines.append(f"\n## {tag}  (resp={len(rdf)}, sent={len(sdf)}, halluc sent={int(sdf.label.sum())})")

    for level, df, grouped in [("Response", rdf, False), ("Sentence (grouped)", sdf, True)]:
        lines.append(f"\n### {level}-level AUC with 95% bootstrap CI")
        auc_ci_for(df, ["mean_surprisal"], grouped, lines, "perplexity")
        auc_ci_for(df, ["length"], grouped, lines, "length")
        gr = auc_ci_for(df, GROUND, grouped, lines, "grounding (GASP)")
        auc_ci_for(df, GROUND + ["mean_surprisal", "length"], grouped, lines, "grounding + baselines")
        # ablation: single features
        lines.append(f"\n#### {level} ablation (single features)")
        for c in GROUND:
            auc_ci_for(df, [c], grouped, lines, c)
        lines.append(f"\n#### {level} ablation (groups)")
        auc_ci_for(df, ["gap", "jsd_noctx"], grouped, lines, "no-context only")
        auc_ci_for(df, ["drop_max", "jsd_loo_max"], grouped, lines, "leave-one-out only")
        auc_ci_for(df, ["gap", "drop_max"], grouped, lines, "likelihood only")
        auc_ci_for(df, ["jsd_noctx", "jsd_loo_max"], grouped, lines, "divergence only")
        # significance vs perplexity
        pg, yg, gg = oof_proba(df, GROUND, grouped)
        pp, yp, gp = oof_proba(df, ["mean_surprisal"], grouped)
        if pg is not None and pp is not None and len(pg) == len(pp):
            md, lo, hi, pv = boot_diff(yg, pg, pp, gg if grouped else None)
            lines.append(f"\n{level} grounding vs perplexity: AUC diff = {md:+.3f} [{lo:+.3f}, {hi:+.3f}], bootstrap p = {pv:.3f}")

    # aggregation study (sentence level)
    lines.append(f"\n### Aggregation study (sentence, grouped): grounding with different leave-one-out aggregation")
    for agg, cols in [("max", ["gap","jsd_noctx","drop_max","jsd_loo_max"]),
                      ("mean", ["gap","jsd_noctx","drop_mean","jsd_loo_mean"]),
                      ("top-2 sum", ["gap","jsd_noctx","drop_top2","jsd_loo_top2"])]:
        auc_ci_for(sdf, cols, True, lines, f"aggregation = {agg}")

    # threshold-only (no classifier): AUC of a single raw feature and of a standardized sum
    lines.append(f"\n### Threshold-only variant (no trained classifier), sentence level")
    for c in ["drop_max", "jsd_loo_max", "gap"]:
        d = sdf.dropna(subset=[c, "label"])
        lines.append(f"- raw {c} AUC: {roc_auc_score(d['label'], d[c]):.3f}")
    d = sdf.dropna(subset=GROUND + ["label"]).copy()
    z = (d[GROUND] - d[GROUND].mean()) / (d[GROUND].std() + 1e-9)
    lines.append(f"- standardized sum of the four features AUC: {roc_auc_score(d['label'], z.sum(axis=1)):.3f}")

def chunking(base, lines):
    lines.append("\n## Chunking sensitivity (Qwen2.5-1.5B, sentence level, grounding AUC with CI)")
    for K in [3, 5, 10]:
        tag = f"Qwen2.5-1.5B-Instruct_K{K}"
        sp = os.path.join(base, tag, "sentence.csv")
        if not os.path.exists(sp):
            lines.append(f"- K={K}: (missing)"); continue
        sdf = add_features(pd.read_csv(sp))
        auc_ci_for(sdf, GROUND, True, lines, f"K = {K}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default=str(RESULTS))
    args = ap.parse_args()
    lines = ["# GASP analysis\n"]
    for tag in ["Qwen2.5-1.5B-Instruct_K5", "Qwen2.5-0.5B-Instruct_K5", "SmolLM2-1.7B-Instruct_K5"]:
        analyze_model(args.base, tag, lines)
    chunking(args.base, lines)
    out = os.path.join(args.base, "ANALYSIS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("\nwritten to", out)

if __name__ == "__main__":
    main()
