# -*- coding: utf-8 -*-
"""
Reports grounding-sensitivity detection AUC separately for each RAGTruth task type
(Summary vs Data2txt), using out-of-fold grouped cross-validation and the GASP
grounding features, across the three scored model tags.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os
import numpy as np, pandas as pd
from analysis import add_features, oof_proba, boot_auc_ci, GROUND

BASE = str(RESULTS)
TAGS = ["Qwen2.5-0.5B-Instruct_K5", "Qwen2.5-1.5B-Instruct_K5", "SmolLM2-1.7B-Instruct_K5"]

def main():
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    lines = ["# Per-task detection (grounding GASP, leakage-clean grouped CV)\n"]
    for tag in TAGS:
        sp = os.path.join(BASE, tag, "sentence.csv"); rp = os.path.join(BASE, tag, "response.csv")
        if not os.path.exists(sp):
            continue
        lines.append(f"## {tag}")
        for level, path, grouped in [("Response", rp, False), ("Sentence", sp, True)]:
            df = add_features(pd.read_csv(path))
            idxs = df["orig_idx"].unique()
            tmap = {int(i): ds[int(i)]["task_type"] for i in idxs}
            df["task"] = df["orig_idx"].map(lambda i: tmap[int(i)])
            for task in ["Summary", "Data2txt"]:
                sub = df[df["task"] == task].reset_index(drop=True)
                if sub["label"].nunique() < 2 or len(sub) < 40:
                    lines.append(f"- {level} / {task}: n/a (n={len(sub)})"); continue
                p, y, g = oof_proba(sub, GROUND, grouped)
                if p is None:
                    lines.append(f"- {level} / {task}: n/a"); continue
                m, lo, hi = boot_auc_ci(y, p, g if grouped else None)
                pos = int(sub["label"].sum())
                lines.append(f"- {level} / {task}: {m:.3f}  [{lo:.3f}, {hi:.3f}]  (n={len(sub)}, positives={pos})")
        lines.append("")
    out = os.path.join(BASE, "PER_TASK.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("written to", out)

if __name__ == "__main__":
    main()
