# -*- coding: utf-8 -*-
"""
Evaluates the threshold-only (no-classifier) grounding-sensitivity variant under
grouped 5-fold cross-validation, standardizing the features on each training fold
(mean and standard deviation fit on the train folds only) before summing them into
the score. Reports span-level AUC with a grouped bootstrap 95% CI.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os
import numpy as np, pandas as pd
from analysis import add_features, boot_auc_ci, oof_proba, GROUND
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

BASE = str(RESULTS)
TAGS = ["Qwen2.5-0.5B-Instruct_K5", "Qwen2.5-1.5B-Instruct_K5", "SmolLM2-1.7B-Instruct_K5"]

def threshold_only_cv(d, cols):
    """Per-fold standardized negated sum, out-of-fold, grouped by response."""
    y = d["label"].to_numpy(); groups = d["resp_id"].to_numpy()
    oof = np.zeros(len(d))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d[cols], y, groups):
        mu = d[cols].iloc[tr].mean(); sd = d[cols].iloc[tr].std() + 1e-9
        oof[te] = ((d[cols].iloc[te] - mu) / sd).sum(axis=1).to_numpy()
    return y, -oof, groups  # negate: features are larger for grounded spans

lines = ["# Threshold-only, corrected protocol (grouped CV, per-fold standardization)\n"]
for tag in TAGS:
    sp = os.path.join(BASE, tag, "sentence.csv")
    if not os.path.exists(sp):
        continue
    d = add_features(pd.read_csv(sp)).dropna(subset=GROUND + ["label"]).reset_index(drop=True)
    # trained GASP, out-of-fold grouped CV
    p, y, g = oof_proba(d, GROUND, True)
    mt, lot, hit = boot_auc_ci(y, p, g)
    # threshold-only, corrected (per-fold standardization, out-of-fold)
    yy, sc, gg = threshold_only_cv(d, GROUND)
    mth, loth, hith = boot_auc_ci(yy, sc, gg)
    # old (buggy) number: global standardization, no CV, whole data
    z = (d[GROUND] - d[GROUND].mean()) / (d[GROUND].std() + 1e-9)
    old = roc_auc_score(d["label"], -z.sum(axis=1))
    lines.append(f"## {tag}")
    lines.append(f"- trained GASP (grouped CV):            {mt:.3f}  [{lot:.3f}, {hit:.3f}]")
    lines.append(f"- threshold-only (grouped CV, per-fold): {mth:.3f}  [{loth:.3f}, {hith:.3f}]")
    lines.append(f"- threshold-only (OLD: global std, no CV, whole data): {old:.3f}\n")

out = os.path.join(BASE, "THRESHOLD_FIX.md")
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("written to", out)
