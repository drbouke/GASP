# -*- coding: utf-8 -*-
"""Response-level threshold-only (negated standardized sum), stratified 5-fold OOF,
matching the response-level protocol, so the default detector can be shown at both levels."""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os
import numpy as np, pandas as pd
from analysis import add_features, boot_auc_ci, GROUND
from sklearn.model_selection import StratifiedKFold

BASE = str(RESULTS)
for tag in ["Qwen2.5-0.5B-Instruct_K5", "Qwen2.5-1.5B-Instruct_K5", "SmolLM2-1.7B-Instruct_K5"]:
    rp = os.path.join(BASE, tag, "response.csv")
    d = add_features(pd.read_csv(rp)).dropna(subset=GROUND + ["label"]).reset_index(drop=True)
    y = d["label"].to_numpy(); oof = np.zeros(len(d))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(d[GROUND], y):
        mu = d[GROUND].iloc[tr].mean(); sd = d[GROUND].iloc[tr].std() + 1e-9
        oof[te] = ((d[GROUND].iloc[te] - mu) / sd).sum(axis=1).to_numpy()
    m, lo, hi = boot_auc_ci(y, -oof, None)
    print(f"{tag}: response threshold-only {m:.3f} [{lo:.3f}, {hi:.3f}]")
