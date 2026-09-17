# -*- coding: utf-8 -*-
"""
LLM-judge (RAGAS-style faithfulness) baseline.

Prompts an instruct model, per answer sentence, to decide whether the sentence is
fully supported by the retrieved context, and reads a calibrated P(Yes) from the
next-token logits over {Yes, No} rather than a hard label. Higher = more supported
(less likely hallucinated). Uses one of our OWN open models (no external API).

Consumes the canonical cases.jsonl. GPU:
  python run_llmjudge.py --cases <cases.jsonl> --model Qwen/Qwen2.5-14B-Instruct --out llmjudge.csv
"""
import json, argparse
import numpy as np, pandas as pd
from gasp_canonical import from_record


class Judge:
    def __init__(self, model_id, max_ctx_tokens=1800):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(self.device).eval()
        torch.set_grad_enabled(False)
        self.max_ctx = max_ctx_tokens
        # first-token ids for Yes / No variants
        self.yes_ids = self._first_ids(["Yes", " Yes", "yes"])
        self.no_ids = self._first_ids(["No", " No", "no"])

    def _first_ids(self, words):
        ids = set()
        for w in words:
            t = self.tok(w, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        return list(ids)

    def p_yes(self, context, sentence):
        torch = self.torch
        cids = self.tok(context, add_special_tokens=False).input_ids[:self.max_ctx]
        ctx = self.tok.decode(cids)
        msg = [{"role": "user", "content":
                "You check whether a statement is fully supported by the given context.\n\n"
                f"Context:\n{ctx}\n\nStatement: {sentence}\n\n"
                "Is the statement fully supported by the context? Answer with a single word, Yes or No."}]
        try:
            ids = self.tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = self.tok(msg[0]["content"] + "\nAnswer:", return_tensors="pt").input_ids
        ids = ids.to(self.device)
        logits = self.model(ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        y = torch.logsumexp(lp[self.yes_ids], 0)
        n = torch.logsumexp(lp[self.no_ids], 0)
        return float(torch.softmax(torch.stack([y, n]), 0)[0].cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--out", default="llmjudge.csv")
    a = ap.parse_args()
    j = Judge(a.model)
    cases = [from_record(json.loads(l)) for l in open(a.cases, encoding="utf-8")]
    print(f"[llmjudge] {len(cases)} cases, model={a.model}")
    rows = []
    for ci, c in enumerate(cases):
        for k in range(len(c.sent_spans)):
            sent = c.sent_text(k).strip()
            if len(sent) < 3:
                continue
            rows.append(dict(case_id=c.case_id, source_id=c.source_id, sent_idx=k,
                             label=c.sent_label(k)[0], llmjudge_pyes=j.p_yes(c.context, sent)))
        if (ci + 1) % 25 == 0:
            print(f"  {ci+1}/{len(cases)}")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[llmjudge] wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
