# -*- coding: utf-8 -*-
"""
Integration / complementarity analysis.

For each strong verifier, compares the verifier alone against a logistic classifier that
adds the grounding features, on the same test spans, and against controls that add cheap
or random features instead. All AUCs are computed on one common sample per verifier, with
the baseline direction fixed on the development split:

  raw            baseline's own ranking AUC on the test split
  +GASP          logistic reg. on [baseline + 6 GASP sensitivity features]
  +simple        control: logistic reg. on [baseline + perplexity + length]
  +noise         control: logistic reg. on [baseline + 6 N(0,1) features]

Reports the paired source-level bootstrap CI of (+GASP - raw) and of (+GASP - +simple),
plus per-task rows when the cases carry a task label. A gain clears the +simple control
only when its interval excludes zero.

Usage: python integration_analysis.py --gasp <sentence.csv> --baselines <csv...> [--seed 0]
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from analyze_gasp import (add_features, source_split, clf_auc, raw_auc,
                          paired_bootstrap_diff, GASP_FEATS, BASE_FEATS)
from compare_baselines import load_baselines, BASELINE_COLS, KEYS

# strongest verifiers, in the order the paper discusses them
STRONG = ["llmjudge_pyes", "minicheck_ctx", "minicheck_max", "alignscore_ctx"]


def raw_ranker(dev, test, col, y):
    """Baseline as a plain ranker: sign fixed on dev, raw score on test.
    Returns (test-AUC, aligned score Series) so it can enter a paired bootstrap."""
    d = dev.dropna(subset=[col]); t = test.dropna(subset=[col])
    if d[y].nunique() < 2 or t[y].nunique() < 2:
        return np.nan, None
    sign = 1.0 if roc_auc_score(d[y], d[col]) >= 0.5 else -1.0
    s = pd.Series(sign * t[col].values, index=t.index)
    return roc_auc_score(t[y], s), s


def one_baseline(df, col, y, seed, task_col=None):
    """Full panel for a single strong baseline on its own common sample."""
    sub = df.dropna(subset=[col] + GASP_FEATS).copy()
    if sub[y].nunique() < 2 or len(sub) < 40:
        return None
    rng = np.random.default_rng(seed)
    noise = [f"_z{i}" for i in range(len(GASP_FEATS))]
    for c in noise:
        sub[c] = rng.standard_normal(len(sub))
    d, t = source_split(sub, seed=seed)

    raw_test_auc, raw_s = raw_ranker(d, t, col, y)                 # the honest reference
    _, _, s_gasp = clf_auc(d, t, [col] + GASP_FEATS, y)
    _, _, s_simple = clf_auc(d, t, [col] + BASE_FEATS, y)
    _, _, s_noise = clf_auc(d, t, [col] + noise, y)
    idx = s_gasp.index
    yy = t.loc[idx, y]

    def auc(s):
        return roc_auc_score(yy, s.loc[idx]) if s is not None else np.nan

    out = dict(
        col=col, n=len(sub), n_test=len(t), pos_test=int(t[y].sum()),
        raw=raw_test_auc, gasp=auc(s_gasp), simple=auc(s_simple), noise=auc(s_noise),
        # decisive: does [baseline+GASP] beat the baseline's OWN ranking on the same ids?
        d_vs_raw=paired_bootstrap_diff(t.loc[idx], y, s_gasp, raw_s.loc[idx]),
        # does GASP beat the cheap simple-feature model?
        d_vs_simple=paired_bootstrap_diff(t.loc[idx], y, s_gasp, s_simple),
        d_vs_noise=paired_bootstrap_diff(t.loc[idx], y, s_gasp, s_noise),
    )
    if task_col and task_col in t:
        rows = []
        for task, g in t.loc[idx].groupby(task_col):
            if g[y].nunique() < 2:
                continue
            gi = g.index
            rows.append((task, len(g), int(g[y].sum()),
                         roc_auc_score(g[y], raw_s.loc[gi]),
                         roc_auc_score(g[y], s_gasp.loc[gi])))
        out["per_task"] = rows
    return out


def fmt_ci(r):
    if not r:
        return "n/a"
    tag = "sig" if r["sig"] else "ns"
    return f"{r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] {tag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gasp", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task-col", default="task")
    a = ap.parse_args()

    g = add_features(pd.read_csv(a.gasp))
    b = load_baselines(a.baselines)
    df = g.merge(b, on=KEYS, how="inner").dropna(subset=GASP_FEATS)
    have = [c for c in STRONG if c in df] + [c for c in BASELINE_COLS if c in df and c not in STRONG]
    y = "label"
    task_col = a.task_col if a.task_col in df else None

    print(f"# Integration analysis  [{a.gasp}]")
    print(f"merged {len(df)} sentences | {df['source_id'].nunique()} sources | "
          f"{int(df[y].sum())} positive spans | task_col={'yes' if task_col else 'no'}\n")

    print("## Strongest verifier alone (raw) vs +GASP, with controls, on one common sample")
    print(f"{'baseline':16s} {'n_test':>6s} {'raw':>6s} {'+GASP':>6s} {'+simple':>7s} {'+noise':>6s}"
          f"   {'+GASP - raw':>22s}")
    panels = []
    for col in have:
        r = one_baseline(df, col, y, a.seed, task_col)
        if r is None:
            continue
        panels.append(r)
        print(f"{r['col']:16s} {r['n_test']:6d} {r['raw']:6.3f} {r['gasp']:6.3f} "
              f"{r['simple']:7.3f} {r['noise']:6.3f}   {fmt_ci(r['d_vs_raw']):>22s}")

    print("\n## Paired deltas of +GASP against each reference (source-level bootstrap 95% CI)")
    for r in panels:
        print(f"  {r['col']:16s}  vs raw   {fmt_ci(r['d_vs_raw'])}")
        print(f"  {'':16s}  vs simple {fmt_ci(r['d_vs_simple'])}")
        print(f"  {'':16s}  vs noise  {fmt_ci(r['d_vs_noise'])}")

    if task_col:
        print("\n## Per-task raw vs +GASP (same test ids)")
        for r in panels:
            if not r.get("per_task"):
                continue
            print(f"  {r['col']}")
            for task, n, pos, raw, gasp in r["per_task"]:
                print(f"     {str(task):14s} n={n:4d} pos={pos:3d}  raw {raw:.3f} -> +GASP {gasp:.3f}")

    print("\n## Spearman correlation of GASP composite with each baseline (test spans)")
    dev, test = source_split(df, seed=a.seed)
    _, _, gasp_s = clf_auc(dev, test, GASP_FEATS + BASE_FEATS, y)
    for col in have:
        sub = test.dropna(subset=[col])
        rho = spearmanr(gasp_s.loc[sub.index], sub[col]).correlation
        print(f"  rho(GASP, {col:16s}) = {rho:+.3f}")

    print("\nReading: a gain is real only if (+GASP - raw) clears 0 and also clears the +simple "
          "control. If +GASP merely matches +simple, cheap features explain it, not sensitivity.")


if __name__ == "__main__":
    main()
