# -*- coding: utf-8 -*-
"""
GASP vs the advanced baselines on the SAME test sentences.

Merges a GASP scorer's sentence.csv with every available baseline CSV (chunk-NLI,
MiniCheck, AlignScore, ContextCite, LLM-judge) on (case_id, sent_idx), fits GASP
(gasp+base) on the source-level dev split, and compares its test AUC against each
baseline score column (direction fixed on dev) with a source-level paired bootstrap
CI (two-sided). Inconclusive -> "no difference found".

Usage:
  python compare_baselines.py --gasp canon_results/<TAG>/sentence.csv \
      --baselines canon_results/baselines_nli.csv canon_results/minicheck.csv \
                  canon_results/alignscore.csv canon_results/contextcite.csv \
                  canon_results/llmjudge.csv
"""
import argparse, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from analyze_gasp import (add_features, source_split, raw_auc, clf_auc,
                          paired_bootstrap_diff, GASP_FEATS, BASE_FEATS)

# every candidate baseline score column produced by the runners
BASELINE_COLS = ["nli_maxchunk", "nli_fullctx", "minicheck_max", "minicheck_ctx",
                 "alignscore_max", "alignscore_ctx", "contextcite_max", "contextcite_sumpos",
                 "llmjudge_pyes"]
KEYS = ["case_id", "sent_idx"]


def load_baselines(paths):
    frames = []
    for p in paths:
        for fp in glob.glob(p):
            df = pd.read_csv(fp)
            keep = KEYS + [c for c in df.columns if c in BASELINE_COLS]
            if len(keep) > len(KEYS):
                frames.append(df[keep])
                print(f"  loaded {fp}: {[c for c in keep if c not in KEYS]}")
    if not frames:
        raise SystemExit("no baseline score columns found in the given files")
    b = frames[0]
    for f in frames[1:]:
        b = b.merge(f, on=KEYS, how="outer")
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gasp", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    g = add_features(pd.read_csv(a.gasp))
    print("# loading baselines")
    b = load_baselines(a.baselines)
    df = g.merge(b, on=KEYS, how="inner")
    y = "label"
    have = [c for c in BASELINE_COLS if c in df]
    df = df.dropna(subset=GASP_FEATS)
    dev, test = source_split(df, seed=a.seed)

    print(f"\n# GASP vs advanced baselines  [{a.gasp}]")
    print(f"merged sentences: {len(df)} | test {len(test)} / {test['source_id'].nunique()} sources | "
          f"positives {int(test[y].sum())}")

    # GASP (gasp+base) trained on dev
    _, _, gasp_scores = clf_auc(dev, test, GASP_FEATS + BASE_FEATS, y)
    gasp_auc = roc_auc_score(test.loc[gasp_scores.index, y], gasp_scores)
    print(f"\nGASP (gasp+base): AUC {gasp_auc:.3f}  AUPRC "
          f"{average_precision_score(test.loc[gasp_scores.index, y], gasp_scores):.3f}")

    print("\n## Baselines (raw score, direction fixed on dev) and paired diff vs GASP")
    print(f"  {'baseline':18s} {'AUC':>6s} {'AUPRC':>6s}   paired dAUC (GASP - baseline)")
    for col in have:
        sub = df.dropna(subset=[col])
        d2, t2 = source_split(sub, seed=a.seed)
        if t2[y].nunique() < 2 or d2.dropna(subset=[col])[y].nunique() < 2:
            print(f"  {col:18s}  (insufficient label spread on its non-null rows)")
            continue
        auc, ap_ = raw_auc(d2, t2, col, y)
        sign = 1.0 if roc_auc_score(d2[y], d2[col]) >= 0.5 else -1.0
        # align on the intersection of this baseline's test rows and GASP's scored rows
        idx = t2.index.intersection(gasp_scores.index)
        if len(idx) < 20:
            print(f"  {col:18s} AUC {auc:.3f}  AUPRC {ap_:.3f}   (too few shared test rows)")
            continue
        bscore = pd.Series(sign * t2.loc[idx, col].values, index=idx)
        r = paired_bootstrap_diff(test.loc[idx], y, gasp_scores.loc[idx], bscore)
        line = f"  {col:18s} AUC {auc:.3f}  AUPRC {ap_:.3f}"
        if r:
            verdict = "GASP better" if (r['sig'] and r['mean'] > 0) else \
                      ("baseline better" if (r['sig'] and r['mean'] < 0) else "no difference found")
            line += f"   dAUC {r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] -> {verdict}"
        print(line)

    print("\nReading: GASP wins a column only where dAUC>0 AND the CI excludes 0. A near-tie where "
          "GASP needs no extra trained verifier and is cheaper is also a valid outcome per the plan.")


if __name__ == "__main__":
    main()
