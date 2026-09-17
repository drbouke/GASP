# -*- coding: utf-8 -*-
"""
Graded attribution-quality evaluation for GASP's explanation.

For each grounded answer sentence, GASP returns the chunk whose removal most lowers the
sentence likelihood as the candidate supporting passage. This script grades that attributed
chunk, and a random control chunk from the same case, into four support levels with an
instruction-tuned judge, reading a calibrated distribution over the four options from the
next-token logits. It reports how often the GASP-attributed chunk is graded fully or partly
supporting versus the random control, an automatic proxy for a human localization study.

Consumes the canonical cases.jsonl. GPU:
  python run_attribution_eval.py --cases <cases.jsonl> --model Qwen/Qwen2.5-14B-Instruct \
      --n 600 --out attribution.csv
"""
import json, argparse
import numpy as np, pandas as pd
from gasp_canonical import from_record

LEVELS = [("A", "fully"), ("B", "partly"), ("C", "topically"), ("D", "not")]


class Judge:
    def __init__(self, model_id, max_ctx=900):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(self.device).eval()
        torch.set_grad_enabled(False); self.max_ctx = max_ctx
        self.opt_ids = [self._first(l) for l, _ in LEVELS]

    def _first(self, w):
        for cand in (w, " " + w):
            t = self.tok(cand, add_special_tokens=False).input_ids
            if t:
                return t[0]
        return self.tok(w, add_special_tokens=False).input_ids[0]

    def grade(self, passage, sentence):
        torch = self.torch
        pids = self.tok(passage, add_special_tokens=False).input_ids[:self.max_ctx]
        passage = self.tok.decode(pids)
        msg = [{"role": "user", "content":
                "Judge how well the passage supports the statement.\n\n"
                f"Passage:\n{passage}\n\nStatement: {sentence}\n\n"
                "Choose one:\n(A) fully supports it\n(B) partly supports it\n"
                "(C) only topically related\n(D) does not support it\n\nAnswer with a single letter."}]
        try:
            ids = self.tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = self.tok(msg[0]["content"] + "\nAnswer:", return_tensors="pt").input_ids
        logits = self.model(ids.to(self.device)).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        probs = torch.softmax(torch.stack([lp[i] for i in self.opt_ids]), 0).cpu().numpy()
        return int(np.argmax(probs)), probs  # 0=fully..3=not


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--sentences", required=True, help="sentence.csv with per-chunk drops")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="attribution.csv")
    a = ap.parse_args()
    j = Judge(a.model)
    cmap = {c.case_id: c for c in (from_record(json.loads(l)) for l in open(a.cases, encoding="utf-8"))}
    df = pd.read_csv(a.sentences)
    df = df[df.label == 0]  # grounded sentences only
    rng = np.random.default_rng(0)

    def attributed(js):
        v = [(-1e9 if x is None else x) for x in json.loads(js)]
        return int(np.argmax(v)) if v else -1
    df = df.copy(); df["att"] = df["chunk_drops"].map(attributed)
    df = df[df.att >= 0]
    df = df.sample(min(a.n, len(df)), random_state=0)
    print(f"[attr] grading {len(df)} grounded sentences, model={a.model}")
    rows = []
    for idx, (_, r) in enumerate(df.iterrows()):
        c = cmap.get(r.case_id)
        if c is None:
            continue
        K = len(c.chunk_spans)
        if K < 2 or r.att >= K:
            continue
        sent = c.sent_text(int(r.sent_idx)).strip()
        if len(sent) < 8:
            continue
        ctrl = int(rng.integers(0, K))
        while ctrl == r.att and K > 1:
            ctrl = int(rng.integers(0, K))
        ga, pa = j.grade(c.chunk_text(int(r.att)), sent)
        gc, pc = j.grade(c.chunk_text(ctrl), sent)
        rows.append(dict(case_id=r.case_id, sent_idx=int(r.sent_idx), att_grade=ga, ctrl_grade=gc,
                         att_support=float(pa[0] + pa[1]), ctrl_support=float(pc[0] + pc[1])))
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(df)}")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[attr] wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
