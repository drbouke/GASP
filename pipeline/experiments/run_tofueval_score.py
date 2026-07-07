# -*- coding: utf-8 -*-
"""
Second benchmark: TofuEval (MeetingBank domain), sentence-level factual
consistency for topic-focused summaries. Reuses the GASP scoring harness. Each
TofuEval summary (doc_id, topic, model) is reconstructed from its labeled
sentences; the MeetingBank transcript is the context, the topic is the query, and
each sentence carries a binary label (sent_label 'no' = hallucination). Writes the
same response.csv / sentence.csv schema as the RAGTruth runs so analyze_rev works.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
from config import RESULTS
import os, re, json, argparse
import numpy as np, pandas as pd
from scorer import LM, split_chunks

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--k_chunks", type=int, default=5)
    p.add_argument("--max_ctx_tokens", type=int, default=700)
    p.add_argument("--max_ans_tokens", type=int, default=200)
    p.add_argument("--tag", default="")
    p.add_argument("--outroot", default=str(RESULTS))
    p.add_argument("--datadir", default=os.path.join(os.path.dirname(__file__), "tofueval_data"))
    return p.parse_args()

def build_transcript_map():
    from datasets import load_dataset
    mb = load_dataset("huuuyeah/meetingbank")
    m = {}
    for split in mb.keys():
        for r in mb[split]:
            for key in ("id", "uid"):
                v = r.get(key)
                if v:
                    m[str(v)] = r["transcript"]
    return m

def main():
    args = get_args()
    tag = args.tag or ("tofueval_" + args.model.split("/")[-1])
    outdir = os.path.join(args.outroot, tag); os.makedirs(outdir, exist_ok=True)

    dev = pd.read_csv(os.path.join(args.datadir, "meetingbank_factual_eval_dev.csv"))
    test = pd.read_csv(os.path.join(args.datadir, "meetingbank_factual_eval_test.csv"))
    tf = pd.concat([dev, test], ignore_index=True)
    print(f"[1/3] loading transcripts")
    tmap = build_transcript_map()

    print(f"[2/3] loading {args.model}")
    lm = LM(args.model)

    print(f"[3/3] scoring {tf.groupby(['doc_id','topic','model_name']).ngroups} summaries (K={args.k_chunks})")
    resp_rows, sent_rows = [], []
    groups = list(tf.groupby(["doc_id", "topic", "model_name"]))
    for gi, ((doc_id, topic, model_name), g) in enumerate(groups):
        if str(doc_id) not in tmap:
            continue
        g = g.sort_values("sent_idx")
        sents = [str(s).strip() for s in g["summ_sent"].tolist()]
        labels = [1 if str(l).strip().lower() == "no" else 0 for l in g["sent_label"].tolist()]
        # reconstruct answer and per-sentence char spans
        answer, spans = "", []
        for s in sents:
            st = len(answer); answer += (s + " "); spans.append((st, st + len(s)))
        answer = answer.strip()
        context = tmap[str(doc_id)]; query = str(topic).strip()
        try:
            sc = lm.score(context, query, answer, args.k_chunks, args.max_ctx_tokens, args.max_ans_tokens)
            if sc is None:
                continue
        except Exception as e:
            print("  skip", repr(e)[:70]); continue
        K = sc["n_chunks"]; offs = sc["offsets"]
        resp_rows.append(dict(model=tag, resp_id=gi, orig_idx=gi,
            mean_surprisal=float(-np.mean(sc["tlp_full"])), n_ans=sc["n"],
            gap=float(np.mean(sc["gap_tok"])), jsd_noctx=float(np.mean(sc["jsd_noc_tok"])),
            chunk_drops=json.dumps([float(np.mean(sc["drop_tok"][k])) for k in range(K)]),
            chunk_jsds=json.dumps([float(np.mean(sc["jsd_tok"][k])) for k in range(K)]),
            label=int(max(labels)), doc_id=str(doc_id), summ_model=model_name))
        for (s, e), lab in zip(spans, labels):
            tk = [t for t in range(sc["n"]) if offs[t][0] >= s and offs[t][0] < e]
            if len(tk) < 3:
                continue
            tk = np.array(tk)
            sent_rows.append(dict(model=tag, resp_id=gi, orig_idx=gi,
                mean_surprisal=float(-np.mean(sc["tlp_full"][tk])), n_tok=int(len(tk)),
                gap=float(np.mean(sc["gap_tok"][tk])), jsd_noctx=float(np.mean(sc["jsd_noc_tok"][tk])),
                chunk_drops=json.dumps([float(np.mean(sc["drop_tok"][k][tk])) for k in range(K)]),
                chunk_jsds=json.dumps([float(np.mean(sc["jsd_tok"][k][tk])) for k in range(K)]),
                sent_type="none", label=int(lab), sent_text=answer[s:e].strip()[:600]))
        if (gi + 1) % 100 == 0:
            print(f"      {gi+1}/{len(groups)} done, {len(sent_rows)} sentences")
    pd.DataFrame(resp_rows).to_csv(os.path.join(outdir, "response.csv"), index=False)
    pd.DataFrame(sent_rows).to_csv(os.path.join(outdir, "sentence.csv"), index=False)
    print(f"      saved {len(resp_rows)} summaries, {len(sent_rows)} sentences to {outdir}")

if __name__ == "__main__":
    main()
