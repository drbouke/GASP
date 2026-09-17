# -*- coding: utf-8 -*-
"""
Analysis of the grounding signal against simple baselines.

Consumes canonical sentence.csv from run_gasp.py and asks whether the grounding-sensitivity
signal adds value over the simple baselines under a source-level split.

- split dev/test by source_id (no source appears in both).
- report each RAW feature's ranking AUC directly (no classifier), AND trained
  classifiers on {perplexity}, {length}, {perplexity+length COMBINED}, {GASP},
  {GASP+base}; the decisive contrast is GASP+base vs the combined base.
- source-level paired bootstrap CI on the DIFFERENCE between methods; two-sided.
  Inconclusive -> "no difference found", not a positive result.

Usage: python analyze_gasp.py canon_results/<TAG>/sentence.csv [--level span|response]
"""
import sys, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RAW_FEATS = ["max_drop", "mean_drop", "top2_drop", "gap", "jsd_noctx", "max_jsd",
             "mean_surprisal", "n_tok"]
GASP_FEATS = ["max_drop", "mean_drop", "top2_drop", "gap", "jsd_noctx", "max_jsd"]
BASE_FEATS = ["mean_surprisal", "n_tok"]


def _valid(lst):
    return np.array([x for x in lst if x is not None], dtype=float)


def add_features(df):
    def feats(js):
        v = _valid(json.loads(js))
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        top2 = np.mean(np.sort(v)[-2:]) if v.size >= 2 else float(v.max())
        return float(np.max(v)), float(np.mean(v)), float(top2)
    md, mn, t2 = zip(*df["chunk_drops"].map(feats))
    df["max_drop"], df["mean_drop"], df["top2_drop"] = md, mn, t2
    df["max_jsd"] = df["chunk_jsds"].map(lambda js: float(np.max(_valid(json.loads(js)))) if _valid(json.loads(js)).size else np.nan)
    return df


def source_split(df, frac_test=0.4, seed=0):
    """split by source_id; no source in both dev and test."""
    srcs = df["source_id"].unique()
    rng = np.random.default_rng(seed); rng.shuffle(srcs)
    n_test = max(1, int(round(len(srcs) * frac_test)))
    test_src = set(srcs[:n_test])
    te = df[df["source_id"].isin(test_src)].copy()
    dv = df[~df["source_id"].isin(test_src)].copy()
    assert not (set(dv["source_id"]) & set(te["source_id"])), "source leak"
    return dv, te


def raw_auc(dev, test, feat, y):
    """direction chosen on dev, ranking AUC on test (no classifier)."""
    d = dev.dropna(subset=[feat]); t = test.dropna(subset=[feat])
    if d[y].nunique() < 2 or t[y].nunique() < 2:
        return np.nan, np.nan
    sign = 1.0 if roc_auc_score(d[y], d[feat]) >= 0.5 else -1.0
    s = sign * t[feat].values
    return roc_auc_score(t[y], s), average_precision_score(t[y], s)


def clf_auc(dev, test, feats, y):
    d = dev.dropna(subset=feats); t = test.dropna(subset=feats)
    if d[y].nunique() < 2 or t[y].nunique() < 2 or len(d) < 10:
        return np.nan, np.nan, None
    sc = StandardScaler().fit(d[feats])
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(d[feats]), d[y])
    p = m.predict_proba(sc.transform(t[feats]))[:, 1]
    return roc_auc_score(t[y], p), average_precision_score(t[y], p), pd.Series(p, index=t.index)


def paired_bootstrap_diff(test, y, score_a, score_b, n=2000, seed=0):
    """source-level paired bootstrap CI on AUC(a)-AUC(b); two-sided."""
    if score_a is None or score_b is None:
        return None
    t = test.loc[score_a.index]
    srcs = t["source_id"].unique(); rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(srcs, size=len(srcs), replace=True)
        idx = t.index[t["source_id"].isin(pick)]
        yy = t.loc[idx, y]
        if yy.nunique() < 2:
            continue
        diffs.append(roc_auc_score(yy, score_a.loc[idx]) - roc_auc_score(yy, score_b.loc[idx]))
    if len(diffs) < 50:
        return None
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi),
               sig=bool(lo > 0 or hi < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--level", choices=["span", "response"], default="span")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = add_features(pd.read_csv(a.csv))
    y = "label"
    if a.level == "response":
        # response-level: max grounding-suspicion per case (label = any hallucinated span)
        g = df.groupby("case_id")
        df = g.agg({**{f: "max" for f in RAW_FEATS}, "source_id": "first", y: "max"}).reset_index()

    dev, test = source_split(df, seed=a.seed)
    print(f"# GASP analysis — {a.csv}  [{a.level}]")
    print(f"dev: {len(dev)} rows / {dev['source_id'].nunique()} sources | "
          f"test: {len(test)} rows / {test['source_id'].nunique()} sources | "
          f"test positives: {int(test[y].sum())}/{len(test)}")

    print("\n## Raw single-feature ranking AUC (no classifier; direction fixed on dev)")
    for f in RAW_FEATS:
        auc, ap_ = raw_auc(dev, test, f, y)
        print(f"  {f:14s}  AUC {auc:.3f}  AUPRC {ap_:.3f}" if auc == auc else f"  {f:14s}  n/a")

    print("\n## Trained classifiers (dev-fit, test-eval)")
    sets = {"base_ppl": ["mean_surprisal"], "base_len": ["n_tok"],
            "base_combined": BASE_FEATS, "gasp": GASP_FEATS,
            "gasp+base": GASP_FEATS + BASE_FEATS}
    scores = {}
    for name, feats in sets.items():
        auc, ap_, p = clf_auc(dev, test, feats, y)
        scores[name] = p
        print(f"  {name:14s}  AUC {auc:.3f}  AUPRC {ap_:.3f}" if auc == auc else f"  {name:14s}  n/a")

    print("\n## Decisive paired comparisons (source-level bootstrap, two-sided 95% CI)")
    for a_name, b_name in [("gasp+base", "base_combined"), ("gasp", "base_combined")]:
        r = paired_bootstrap_diff(test, y, scores.get(a_name), scores.get(b_name))
        if r is None:
            print(f"  {a_name} - {b_name}: insufficient data")
        else:
            verdict = "DIFFERENT" if r["sig"] else "no difference found"
            print(f"  {a_name} - {b_name}: dAUC {r['mean']:+.3f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  -> {verdict}")

    print("\nReading: value exists only if a GASP set clearly beats base_combined AND the paired CI "
          "excludes 0. Overlapping/negative CI after the fixes => reposition, do not pile on experiments.")


if __name__ == "__main__":
    main()
