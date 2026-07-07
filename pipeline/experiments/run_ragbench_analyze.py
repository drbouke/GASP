# -*- coding: utf-8 -*-
"""Analysis for the RAGBench third benchmark (multi-domain RAG). Same as the
TofuEval analysis: grounding detection AUC with bootstrap CIs at response and span
level, threshold-only variant, significance vs perplexity, per scorer."""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os
import numpy as np, pandas as pd
from analysis import add_features, oof_proba, boot_auc_ci, boot_diff, GROUND
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

BASE = str(RESULTS)
TAGS = ["ragbench_Qwen2.5-0.5B-Instruct", "ragbench_Qwen2.5-1.5B-Instruct", "ragbench_SmolLM2-1.7B-Instruct"]

def threshold_only(d, grouped):
    y = d["label"].to_numpy(); oof = np.zeros(len(d))
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d[GROUND], y, d["resp_id"]) if grouped
          else StratifiedKFold(5, shuffle=True, random_state=42).split(d[GROUND], y))
    for tr, te in cv:
        mu = d[GROUND].iloc[tr].mean(); sd = d[GROUND].iloc[tr].std() + 1e-9
        oof[te] = ((d[GROUND].iloc[te] - mu) / sd).sum(axis=1).to_numpy()
    return boot_auc_ci(y, -oof, d["resp_id"].to_numpy() if grouped else None)

def main():
    lines = ["# RAGBench third benchmark (multi-domain RAG, grounding detection)\n"]
    for tag in TAGS:
        sp = os.path.join(BASE, tag, "sentence.csv"); rp = os.path.join(BASE, tag, "response.csv")
        if not os.path.exists(sp):
            lines.append(f"## {tag}: (missing)\n"); continue
        rdf = add_features(pd.read_csv(rp)); sdf = add_features(pd.read_csv(sp))
        doms = ", ".join(f"{k}:{v}" for k, v in rdf["domain"].value_counts().items()) if "domain" in rdf.columns else ""
        lines.append(f"## {tag}  (responses={len(rdf)}, sentences={len(sdf)}, halluc_sent={int(sdf.label.sum())})")
        lines.append(f"domains: {doms}")
        for level, df, grouped in [("Response", rdf, False), ("Sentence", sdf, True)]:
            lines.append(f"\n### {level}-level AUC [95% CI]")
            for name, cols in [("perplexity", ["mean_surprisal"]), ("length", ["length"]),
                               ("grounding (GASP-trained)", GROUND),
                               ("grounding + baselines", GROUND + ["mean_surprisal", "length"])]:
                p, y, g = oof_proba(df, cols, grouped)
                if p is None:
                    lines.append(f"- {name}: n/a"); continue
                m, lo, hi = boot_auc_ci(y, p, g if grouped else None)
                lines.append(f"- {name}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
            d = df.dropna(subset=GROUND + ["label"]).reset_index(drop=True)
            mt, lot, hit = threshold_only(d, grouped)
            lines.append(f"- GASP-threshold (default): {mt:.3f} [{lot:.3f}, {hit:.3f}]")
            pg, yg, gg = oof_proba(df, GROUND, grouped); pp, yp, gp = oof_proba(df, ["mean_surprisal"], grouped)
            if pg is not None and pp is not None and len(pg) == len(pp):
                md, lo, hi, pv = boot_diff(yg, pg, pp, gg if grouped else None)
                lines.append(f"- grounding vs perplexity: diff {md:+.3f} [{lo:+.3f}, {hi:+.3f}], p={pv:.3f}")
        lines.append("")
    out = os.path.join(BASE, "RAGBENCH_RESULTS.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("written to", out)

if __name__ == "__main__":
    main()
