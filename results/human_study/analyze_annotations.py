# -*- coding: utf-8 -*-
"""
Reproduce the attribution human-study numbers reported in the paper from the three
annotator files in Reports/. Prints per-annotator support rates, the majority-vote
support rate, and inter-annotator agreement (mean pairwise exact agreement and Fleiss'
kappa on the four levels and on the collapsed supported-versus-not distinction).

  python analyze_annotations.py
"""
import os
import pandas as pd, numpy as np
from collections import Counter

R = os.path.join(os.path.dirname(__file__), "Reports")
LVL = ["A", "B", "C", "D"]; NAMES = ["fully", "partly", "topical", "not"]


def norm(s):
    return str(s).strip().upper()[:1]


def fleiss_kappa(P, n):
    N = P.shape[0]
    Pi = ((P ** 2).sum(1) - n) / (n * (n - 1))
    pj = P.sum(0) / (N * n)
    Pe = (pj ** 2).sum()
    return (Pi.mean() - Pe) / (1 - Pe)


def main():
    dfs = [pd.read_csv(os.path.join(R, f"annotation{i}.csv"), encoding="utf-8-sig") for i in (1, 2, 3)]
    G = [d["annotator_grade"].map(norm) for d in dfs]
    n_items = len(G[0])
    print(f"# attribution human study: {n_items} grounded sentences, 3 annotators\n")
    print("per-annotator support rate (fully or partly):")
    for i, g in enumerate(G):
        c = Counter(g)
        print(f"  A{i+1}: " + ", ".join(f"{NAMES[j]} {100*c.get(LVL[j],0)/n_items:4.1f}%" for j in range(4)) +
              f"  | supported {100*(c.get('A',0)+c.get('B',0))/n_items:.1f}%")

    M = pd.DataFrame({"a1": G[0].values, "a2": G[1].values, "a3": G[2].values})

    def majority(r):
        m = Counter([r.a1, r.a2, r.a3]).most_common()
        return m[0][0] if m[0][1] >= 2 else "tie"
    M["maj"] = M.apply(majority, axis=1)
    sup = 100 * ((M["maj"] == "A") | (M["maj"] == "B")).mean()
    cm = Counter(M["maj"])
    print(f"\nmajority vote: " + ", ".join(f"{k}:{cm.get(k,0)}" for k in LVL + ['tie']) +
          f"  | supported {sup:.1f}%")

    pairs = [("a1", "a2"), ("a1", "a3"), ("a2", "a3")]
    pa = np.mean([(M[a] == M[b]).mean() for a, b in pairs])
    P4 = np.array([[Counter([r.a1, r.a2, r.a3]).get(c, 0) for c in LVL] for _, r in M.iterrows()], float)
    Mb = M.replace({"A": "S", "B": "S", "C": "U", "D": "U"})
    pab = np.mean([(Mb[a] == Mb[b]).mean() for a, b in pairs])
    P2 = np.array([[Counter([r.a1, r.a2, r.a3]).get(c, 0) for c in ("S", "U")] for _, r in Mb.iterrows()], float)
    print(f"\ninter-annotator agreement:")
    print(f"  4-level: mean pairwise exact {100*pa:.1f}%, Fleiss kappa {fleiss_kappa(P4,3):.3f}")
    print(f"  supported/not: mean pairwise {100*pab:.1f}%, Fleiss kappa {fleiss_kappa(P2,3):.3f}")


if __name__ == "__main__":
    main()
