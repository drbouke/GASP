"""
Answer-level context-sensitivity baseline (ConSens-style) against sentence-level grounding sensitivity.

ConSens (Vankov et al., 2025) scores a whole answer by log(PPL without context / PPL with context),
which equals the answer's mean per-token log-likelihood gap between the full and the empty context.
This script computes that answer-level gap from the canonical sentence file as the token-weighted mean
of the sentence gaps of each answer (sentences shorter than a few tokens are not scored, so the value
is computed over the scored tokens; the content-word filter of ConSens is not applied), and compares:

  span level      each sentence scored by its own gap  vs  by its answer's gap
  response level  each answer scored by its answer gap  vs  by its lowest sentence gap

on the same source-level test split as analyze_gasp.py, with a source-level paired bootstrap.

  python consens_baseline.py canon_results/RT_14B/sentence.csv [--seed 0]
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_gasp import source_split, paired_bootstrap_diff


def oriented(dev, test, col, y="label"):
    """Score oriented so that higher means more suspicious; direction fixed on the dev partition."""
    d = dev.dropna(subset=[col])
    sign = 1.0 if roc_auc_score(d[y], d[col]) >= 0.5 else -1.0
    return pd.Series(sign * test[col].values, index=test.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.csv).dropna(subset=["gap", "n_tok"])
    w = df["gap"] * df["n_tok"]
    ans = (w.groupby(df["case_id"]).sum() / df["n_tok"].groupby(df["case_id"]).sum()).rename("answer_gap")
    df = df.join(ans, on="case_id")

    dev, test = source_split(df, seed=a.seed)
    s_sent, s_ans = oriented(dev, test, "gap"), oriented(dev, test, "answer_gap")
    auc_sent, auc_ans = roc_auc_score(test["label"], s_sent), roc_auc_score(test["label"], s_ans)
    r = paired_bootstrap_diff(test, "label", s_sent, s_ans, seed=a.seed)
    print(f"# {a.csv}")
    print(f"test: {len(test)} sentences, {test['source_id'].nunique()} sources, {int(test['label'].sum())} positives")
    print(f"span level      sentence gap AUC {auc_sent:.3f} | answer gap (ConSens-style) AUC {auc_ans:.3f} | "
          f"diff {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {'sig' if r['sig'] else 'ns'}")

    g = df.groupby("case_id").agg(answer_gap=("answer_gap", "first"), min_sent_gap=("gap", "min"),
                                  source_id=("source_id", "first"), label=("label", "max")).reset_index()
    gdev, gtest = source_split(g, seed=a.seed)
    r_ans, r_min = oriented(gdev, gtest, "answer_gap"), oriented(gdev, gtest, "min_sent_gap")
    auc_a, auc_m = roc_auc_score(gtest["label"], r_ans), roc_auc_score(gtest["label"], r_min)
    rr = paired_bootstrap_diff(gtest, "label", r_min, r_ans, seed=a.seed)
    print(f"response level  answer gap (ConSens-style) AUC {auc_a:.3f} | lowest sentence gap AUC {auc_m:.3f} | "
          f"diff {rr['mean']:+.3f} [{rr['lo']:+.3f}, {rr['hi']:+.3f}] {'sig' if rr['sig'] else 'ns'}"
          f"  ({len(gtest)} test answers, {int(gtest['label'].sum())} positive)")


if __name__ == "__main__":
    main()
