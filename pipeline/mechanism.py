# -*- coding: utf-8 -*-
"""
Mechanism and complementarity analysis for GASP, from existing scores.

Three questions the accuracy table alone cannot answer, all computed from the canonical
sentence scores + baseline CSVs (no new compute):

  (1) Response level. Aggregate each method's sentence scores to a per-response suspicion
      (max over the response's sentences); label = the response contains any unsupported
      span. GASP's gap is a naturally response-level signal, so this tests whether it is
      more competitive when the task is "does this answer contain a problem at all".

  (2) Score correlation. Spearman correlation between GASP's composite score and each
      baseline. Low correlation means GASP measures something different from a trained
      entailment/fact-checking verifier, which is the precondition for complementarity.

  (3) Complementary value. Does adding GASP's features on top of the STRONGEST baselines
      (the two that beat GASP alone: MiniCheck-ctx and the LLM-judge) raise AUC? A trained
      classifier on [baseline] vs [baseline + GASP] with a source-level paired bootstrap CI.
      A positive gain whose CI excludes 0 means GASP contributes signal the verifier lacks.

Usage: python mechanism.py --gasp canon_results/<TAG>/sentence.csv \
         --baselines canon_results/baselines_nli.csv canon_results/contextcite.csv \
                     canon_results/llmjudge.csv canon_results/alignscore.csv \
                     canon_results/minicheck.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from analyze_gasp import (add_features, source_split, clf_auc, paired_bootstrap_diff,
                          GASP_FEATS, BASE_FEATS)
from compare_baselines import load_baselines, BASELINE_COLS, KEYS


def resp_scores(test, score, agg_label):
    """Aggregate sentence scores + labels to response level (max suspicion / any-positive)."""
    df = pd.DataFrame({"case_id": test["case_id"].values, "source_id": test["source_id"].values,
                       "s": score.values, "y": test["label"].values}, index=score.index)
    g = df.groupby("case_id")
    r = g.agg(source_id=("source_id", "first"), s=("s", "max"), y=("y", "max")).reset_index()
    return r


def paired_resp_bootstrap(r_a, r_b, n=2000, seed=0):
    """Paired source-level bootstrap on response-level AUC(a)-AUC(b). r_a,r_b aligned by case."""
    m = r_a.merge(r_b, on=["case_id", "source_id"], suffixes=("_a", "_b"))
    srcs = m["source_id"].unique(); rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        pick = rng.choice(srcs, size=len(srcs), replace=True)
        sub = m[m["source_id"].isin(pick)]
        if sub["y_a"].nunique() < 2:
            continue
        diffs.append(roc_auc_score(sub["y_a"], sub["s_a"]) - roc_auc_score(sub["y_b"], sub["s_b"]))
    if len(diffs) < 50:
        return None
    d = np.array(diffs); lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gasp", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    g = add_features(pd.read_csv(a.gasp))
    b = load_baselines(a.baselines)
    df = g.merge(b, on=KEYS, how="inner").dropna(subset=GASP_FEATS)
    have = [c for c in BASELINE_COLS if c in df]
    y = "label"
    dev, test = source_split(df, seed=a.seed)
    print(f"# Mechanism / complementarity  [{a.gasp}]")
    print(f"merged {len(df)} sentences | dev {len(dev)} / test {len(test)} "
          f"({test['source_id'].nunique()} test sources, {int(test[y].sum())} positive spans)\n")

    # GASP composite (gasp+base) test scores, reused throughout
    _, _, gasp_s = clf_auc(dev, test, GASP_FEATS + BASE_FEATS, y)

    # ---------- (1) response level ----------
    print("## (1) Response level (label = response has any unsupported span; max-pool suspicion)")
    r_gasp = resp_scores(test, gasp_s, y)
    print(f"   responses in test: {len(r_gasp)} | positive responses: {int(r_gasp['y'].sum())}")
    print(f"   {'method':20s} {'resp-AUC':>8s}   paired dAUC (GASP - baseline)")
    gasp_auc = roc_auc_score(r_gasp["y"], r_gasp["s"])
    print(f"   {'GASP (gasp+base)':20s} {gasp_auc:8.3f}")
    for col in have:
        sub = df.dropna(subset=[col])
        d2, t2 = source_split(sub, seed=a.seed)
        sign = 1.0 if roc_auc_score(d2[y], d2[col]) >= 0.5 else -1.0
        bscore = pd.Series(sign * t2[col].values, index=t2.index)
        r_b = resp_scores(t2, bscore, y)
        auc_b = roc_auc_score(r_b["y"], r_b["s"])
        rr = paired_resp_bootstrap(r_gasp, r_b, seed=a.seed)
        verdict = ""
        if rr:
            verdict = ("GASP better" if rr["sig"] and rr["mean"] > 0 else
                       "baseline better" if rr["sig"] and rr["mean"] < 0 else "no difference found")
            verdict = f"dAUC {rr['mean']:+.3f} [{rr['lo']:+.3f},{rr['hi']:+.3f}] -> {verdict}"
        print(f"   {col:20s} {auc_b:8.3f}   {verdict}")

    # ---------- (2) score correlation ----------
    print("\n## (2) Spearman correlation of the GASP composite with each baseline (test spans)")
    for col in have:
        sub = test.dropna(subset=[col])
        rho = spearmanr(gasp_s.loc[sub.index], sub[col]).correlation
        print(f"   rho(GASP, {col:18s}) = {rho:+.3f}")
    print("   (low |rho| => GASP measures something a trained verifier does not)")

    # ---------- (3) complementary value on top of the strongest baselines ----------
    print("\n## (3) Does GASP add signal on top of the strongest baselines? "
          "[base] vs [base + GASP], paired source-level bootstrap")
    rng = np.random.default_rng(a.seed)
    for col in [c for c in ["minicheck_ctx", "llmjudge_pyes", "alignscore_ctx", "minicheck_max"] if c in df]:
        sub = df.dropna(subset=[col] + GASP_FEATS).copy()
        # noise control: same count of random features as GASP, to separate signal from capacity
        noise = [f"_noise{i}" for i in range(len(GASP_FEATS))]
        for nfc in noise:
            sub[nfc] = rng.standard_normal(len(sub))
        d2, t2 = source_split(sub, seed=a.seed)
        _, _, s_base = clf_auc(d2, t2, [col] + BASE_FEATS, y)
        _, _, s_both = clf_auc(d2, t2, [col] + BASE_FEATS + GASP_FEATS, y)
        _, _, s_noise = clf_auc(d2, t2, [col] + BASE_FEATS + noise, y)
        if s_base is None or s_both is None:
            print(f"   {col:16s}: insufficient data"); continue
        idx = s_base.index
        auc_base = roc_auc_score(t2.loc[idx, y], s_base)
        auc_both = roc_auc_score(t2.loc[idx, y], s_both)
        rr = paired_bootstrap_diff(t2.loc[idx], y, s_both, s_base)
        rn = paired_bootstrap_diff(t2.loc[idx], y, s_noise, s_base)
        tag = ("GASP adds signal" if rr and rr["sig"] and rr["mean"] > 0 else
               "no added value" if rr else "n/a")
        cis = f"[{rr['lo']:+.3f},{rr['hi']:+.3f}]" if rr else ""
        ncis = f"[{rn['lo']:+.3f},{rn['hi']:+.3f}]" if rn else ""
        print(f"   {col:16s}: base {auc_base:.3f} -> +GASP {auc_both:.3f}  dAUC {rr['mean']:+.3f} {cis} "
              f"-> {tag}   (noise control dAUC {rn['mean']:+.3f} {ncis})")

    print("\nReading: a positive response-level result, low correlation, and a significant +GASP gain on "
          "top of MiniCheck/LLM-judge would establish GASP as a complementary training-free signal, not a "
          "redundant weaker detector.")


if __name__ == "__main__":
    main()
