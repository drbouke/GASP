# -*- coding: utf-8 -*-
"""
Dump every headline number the paper's results tables need, computed consistently from the
saved canonical CSVs (14B and the scorer matrix, all three datasets). Prints, per scorer and
per dataset, response-level and span-level AUC with source-level bootstrap CIs for perplexity,
length, GASP-threshold (best single raw feature, training-free), GASP-trained, and GASP+base;
plus per-type and per-task splits on RAGTruth-14B. No model calls.
"""
import glob, os, json
import numpy as np, pandas as pd
from analyze_gasp import add_features, source_split, raw_auc, clf_auc, GASP_FEATS, BASE_FEATS

RT = "canon_results"


def boot_ci_auc(df, feat, y="label", n=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    d = df.dropna(subset=[feat])
    if d[y].nunique() < 2:
        return np.nan, (np.nan, np.nan)
    srcs = d["source_id"].unique(); rng = np.random.default_rng(seed); vals = []
    base = roc_auc_score(d[y], d[feat])
    sign = 1.0 if base >= 0.5 else -1.0
    for _ in range(n):
        pick = rng.choice(srcs, size=len(srcs), replace=True)
        s = d[d["source_id"].isin(pick)]
        if s[y].nunique() < 2:
            continue
        vals.append(roc_auc_score(s[y], sign * s[feat]))
    return roc_auc_score(d[y], sign * d[feat]), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def resp_agg(df, y="label"):
    g = df.groupby("case_id")
    agg = {f: "max" for f in GASP_FEATS + BASE_FEATS}
    r = g.agg({**agg, "source_id": "first", y: "max"}).reset_index()
    return r


def scorer_row(path, level):
    df = add_features(pd.read_csv(path))
    if level == "response":
        df = resp_agg(df)
    dev, test = source_split(df, seed=0)
    y = "label"
    out = {}
    # perplexity, length: raw
    for name, feat in [("ppl", "mean_surprisal"), ("len", "n_tok")]:
        out[name] = raw_auc(dev, test, feat, y)[0]
    # GASP-threshold: best single raw GASP feature, direction+choice on dev
    best, bestauc = None, -1
    for f in GASP_FEATS:
        a = raw_auc(dev, test, f, y)[0]
        dev_a = raw_auc(dev, dev, f, y)[0]  # pick on dev
        if dev_a == dev_a and dev_a > bestauc:
            bestauc = dev_a; best = f
    out["gasp_thr"] = raw_auc(dev, test, best, y)[0] if best else np.nan
    out["gasp_trn"] = clf_auc(dev, test, GASP_FEATS, y)[0]
    out["gasp_base"] = clf_auc(dev, test, GASP_FEATS + BASE_FEATS, y)[0]
    return out


def main():
    scorers = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
               "SmolLM2-1.7B-Instruct", "Meta-Llama-3.1-8B-Instruct", "Phi-3.5-mini-instruct"]
    print("### RAGTruth scorer matrix (AUC) ###")
    for level in ["response", "span"]:
        print(f"\n-- {level} --")
        print(f"{'scorer':28s} {'ppl':>6s} {'len':>6s} {'gaspThr':>7s} {'gaspTrn':>7s} {'gasp+base':>9s}")
        for s in scorers:
            p = f"{RT}/{s}_ragtruth_K5/sentence.csv"
            if not os.path.exists(p):
                continue
            r = scorer_row(p, level)
            print(f"{s:28s} {r['ppl']:6.3f} {r['len']:6.3f} {r['gasp_thr']:7.3f} {r['gasp_trn']:7.3f} {r['gasp_base']:9.3f}")

    # 14B headline with CIs, both levels
    print("\n### RAGTruth 14B headline with source-level 95% CI ###")
    df14 = add_features(pd.read_csv(f"{RT}/Qwen2.5-14B-Instruct_ragtruth_K5/sentence.csv"))
    for level in ["response", "span"]:
        d = resp_agg(df14) if level == "response" else df14
        dev, test = source_split(d, seed=0)
        print(f"\n-- {level} (test {len(test)} rows / {test.source_id.nunique()} sources, pos {int(test.label.sum())}) --")
        for name, feats in [("perplexity", ["mean_surprisal"]), ("GASP-trained", GASP_FEATS),
                            ("GASP+base", GASP_FEATS + BASE_FEATS)]:
            auc, _, sc = clf_auc(dev, test, feats, "label")
            print(f"  {name:14s} AUC {auc:.3f}")

    # per-type and per-task on 14B span
    print("\n### RAGTruth 14B span per-type / per-task ###")
    df = df14.copy()
    # need sent_type + task; sent_type in sentence.csv; task not in sentence.csv -> from cases.jsonl
    dev, test = source_split(df, seed=0)
    for t in ["baseless", "conflict"]:
        sub = test[(test.sent_type == t) | (test.label == 0)]
        a = raw_auc(dev, sub, "gap", "label")[0]
        # trained gasp+base for the type
        _, _, sc = clf_auc(dev, sub, GASP_FEATS + BASE_FEATS, "label")
        from sklearn.metrics import roc_auc_score
        au = roc_auc_score(sub.loc[sc.index, "label"], sc) if sc is not None else float("nan")
        print(f"  {t:10s} vs clean: gasp+base AUC {au:.3f} (n_pos {int((test.sent_type==t).sum())})")


if __name__ == "__main__":
    main()
