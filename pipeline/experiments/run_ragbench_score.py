# -*- coding: utf-8 -*-
"""
Third benchmark: RAGBench (multi-domain, RAG-native). Pools the test splits of
several domains, maps each example to the GASP format (question -> query, joined
documents -> context, response -> answer), and scores it with the same pipeline.
Response-level label = not adherence_score (False adherence => a response with at
least one unsupported sentence). Sentence-level label = sentence key in
unsupported_response_sentence_keys. Balanced sampling of hallucinated vs grounded
responses across domains. Same response.csv / sentence.csv schema as before.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, json, argparse
import numpy as np, pandas as pd
from scorer import LM, split_chunks

DOMAINS = ["pubmedqa", "finqa", "covidqa", "hotpotqa", "techqa", "delucionqa"]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n_per_class", type=int, default=400)
    p.add_argument("--k_chunks", type=int, default=5)
    p.add_argument("--max_ctx_tokens", type=int, default=1000)
    p.add_argument("--max_ans_tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="")
    p.add_argument("--outroot", default=str(RESULTS))
    return p.parse_args()

def load_pool():
    from datasets import load_dataset
    rows = []
    for dom in DOMAINS:
        try:
            ds = load_dataset("rungalileo/ragbench", dom, split="test")
        except Exception as e:
            print("  skip domain", dom, repr(e)[:60]); continue
        for e in ds:
            docs = e.get("documents") or []
            resp = (e.get("response") or "").strip()
            rsents = e.get("response_sentences") or []
            if not docs or not resp or not rsents:
                continue
            unsup = set(e.get("unsupported_response_sentence_keys") or [])
            adh = e.get("adherence_score")
            resp_label = 0 if adh is True else 1  # not adherent => hallucination
            rows.append(dict(domain=dom, query=(e.get("question") or "").strip(),
                             context="\n".join(str(d) for d in docs),
                             sents=[(str(k), str(t)) for k, t in rsents],
                             unsup=unsup, resp_label=resp_label))
    return rows

def main():
    args = get_args()
    tag = args.tag or ("ragbench_" + args.model.split("/")[-1])
    outdir = os.path.join(args.outroot, tag); os.makedirs(outdir, exist_ok=True)

    print("[1/3] pooling RAGBench domains")
    pool = load_pool()
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pool))
    pos, neg = [], []
    for i in idx:
        r = pool[int(i)]
        b = pos if r["resp_label"] == 1 else neg
        if len(b) < args.n_per_class:
            b.append(r)
        if len(pos) >= args.n_per_class and len(neg) >= args.n_per_class:
            break
    n = min(len(pos), len(neg)); sample = pos[:n] + neg[:n]
    print(f"      balanced: {n} hallucinated, {n} grounded ({len(pool)} pooled)")

    print(f"[2/3] loading {args.model}")
    lm = LM(args.model)

    print(f"[3/3] scoring (K={args.k_chunks})")
    resp_rows, sent_rows = [], []
    for jj, r in enumerate(sample):
        answer, spans, labels = "", [], []
        for key, txt in r["sents"]:
            st = len(answer); answer += (txt + " "); spans.append((st, st + len(txt)))
            labels.append(1 if key in r["unsup"] else 0)
        answer = answer.strip()
        try:
            sc = lm.score(r["context"], r["query"], answer, args.k_chunks, args.max_ctx_tokens, args.max_ans_tokens)
            if sc is None:
                continue
        except Exception as e:
            print("  skip", repr(e)[:70]); continue
        K = sc["n_chunks"]; offs = sc["offsets"]
        resp_rows.append(dict(model=tag, resp_id=jj, orig_idx=jj, domain=r["domain"],
            mean_surprisal=float(-np.mean(sc["tlp_full"])), n_ans=sc["n"],
            gap=float(np.mean(sc["gap_tok"])), jsd_noctx=float(np.mean(sc["jsd_noc_tok"])),
            chunk_drops=json.dumps([float(np.mean(sc["drop_tok"][k])) for k in range(K)]),
            chunk_jsds=json.dumps([float(np.mean(sc["jsd_tok"][k])) for k in range(K)]),
            label=int(r["resp_label"])))
        for (s, e), lab in zip(spans, labels):
            tk = [t for t in range(sc["n"]) if offs[t][0] >= s and offs[t][0] < e]
            if len(tk) < 3:
                continue
            tk = np.array(tk)
            sent_rows.append(dict(model=tag, resp_id=jj, orig_idx=jj, domain=r["domain"],
                mean_surprisal=float(-np.mean(sc["tlp_full"][tk])), n_tok=int(len(tk)),
                gap=float(np.mean(sc["gap_tok"][tk])), jsd_noctx=float(np.mean(sc["jsd_noc_tok"][tk])),
                chunk_drops=json.dumps([float(np.mean(sc["drop_tok"][k][tk])) for k in range(K)]),
                chunk_jsds=json.dumps([float(np.mean(sc["jsd_tok"][k][tk])) for k in range(K)]),
                sent_type="none", label=int(lab), sent_text=answer[s:e].strip()[:600]))
        if (jj + 1) % 100 == 0:
            print(f"      {jj+1}/{len(sample)} done, {len(sent_rows)} sentences")
    pd.DataFrame(resp_rows).to_csv(os.path.join(outdir, "response.csv"), index=False)
    pd.DataFrame(sent_rows).to_csv(os.path.join(outdir, "sentence.csv"), index=False)
    print(f"      saved {len(resp_rows)} responses, {len(sent_rows)} sentences to {outdir}")

if __name__ == "__main__":
    main()
