# -*- coding: utf-8 -*-
"""
Advanced-baseline runner for GASP (chunk-level NLI + MiniCheck).

Consumes the canonical cases.jsonl (identical chunks/sentences/labels as the GASP
scorer) and scores each answer sentence with the nearest strong
grounding verifiers, so the comparison is apples-to-apples on the SAME inputs:

  nli_maxchunk : max entailment prob of the sentence over the canonical chunks
  nli_fullctx: entailment prob of the sentence given the full context
  minicheck_max: max MiniCheck support prob over the canonical chunks   (if available)
  minicheck_ctx: MiniCheck support prob given the full context           (if available)

Higher = more supported (less likely hallucinated); direction is fixed on dev in the
comparison. Output: baselines.csv keyed by (case_id, sent_idx) for joining to the GASP
sentence features. Run on GPU: python run_baselines.py --cases <cases.jsonl> --out baselines.csv
"""
import os, json, argparse
import numpy as np, pandas as pd
from gasp_canonical import from_record


def load_nli(model_id="cross-encoder/nli-deberta-v3-large"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(dev).eval()
    torch.set_grad_enabled(False)
    ent = next(i for i, l in model.config.id2label.items() if "entail" in str(l).lower())

    def entail(premise, hyp):
        x = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=512).to(dev)
        return float(torch.softmax(model(**x).logits[0], -1)[ent].cpu())
    return entail


def load_minicheck():
    try:
        try:
            from minicheck.minicheck import MiniCheck
        except ModuleNotFoundError:
            from minicheck import MiniCheck
        mc = MiniCheck(model_name="flan-t5-large", enable_prefix_caching=False)

        def score(doc, claim):
            _, prob, _, _ = mc.score(docs=[doc], claims=[claim])
            return float(prob[0])
        print("[baselines] MiniCheck flan-t5-large loaded")
        return score
    except Exception as e:
        print("[baselines] MiniCheck unavailable, skipping:", repr(e)[:120])
        return None


def load_alignscore():
    try:
        import os
        from alignscore import AlignScore
        ckpt = os.path.join(os.path.dirname(__file__), "AlignScore-large.ckpt")
        if not os.path.exists(ckpt):
            import urllib.request
            print("[baselines] downloading AlignScore-large.ckpt ...")
            urllib.request.urlretrieve(
                "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt", ckpt)
        sc = AlignScore(model="roberta-large", batch_size=16, device="cuda",
                        ckpt_path=ckpt, evaluation_mode="nli_sp")

        def score(doc, claim):
            return float(sc.score(contexts=[doc], claims=[claim])[0])
        print("[baselines] AlignScore-large loaded")
        return score
    except Exception as e:
        print("[baselines] AlignScore unavailable, skipping:", repr(e)[:120])
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="baselines.csv")
    ap.add_argument("--only", choices=["all", "nli", "minicheck", "alignscore"], default="all",
                    help="restrict to one verifier so conflicting deps run in isolated envs")
    a = ap.parse_args()

    entail = load_nli() if a.only in ("all", "nli") else None
    mc = load_minicheck() if a.only in ("all", "minicheck") else None
    al = load_alignscore() if a.only in ("all", "alignscore") else None
    if entail is None and mc is None and al is None:
        raise SystemExit(f"[baselines] nothing to run for --only {a.only}")

    cases = [from_record(json.loads(l)) for l in open(a.cases, encoding="utf-8")]
    print(f"[baselines] {len(cases)} canonical cases | only={a.only}")
    rows = []
    for ci, c in enumerate(cases):
        chunks = [c.chunk_text(i) for i in range(len(c.chunk_spans))]
        if not chunks:
            continue
        for j in range(len(c.sent_spans)):
            sent = c.sent_text(j).strip()
            if len(sent) < 3:
                continue
            lab = c.sent_label(j)[0]
            row = dict(case_id=c.case_id, source_id=c.source_id, sent_idx=j, label=lab)
            if entail is not None:
                row["nli_maxchunk"] = max(entail(ch, sent) for ch in chunks)
                row["nli_fullctx"] = entail(c.context, sent)
            if mc is not None:
                row["minicheck_max"] = max(mc(ch, sent) for ch in chunks)
                row["minicheck_ctx"] = mc(c.context, sent)
            if al is not None:
                row["alignscore_max"] = max(al(ch, sent) for ch in chunks)
                row["alignscore_ctx"] = al(c.context, sent)
            rows.append(row)
        if (ci + 1) % 50 == 0:
            print(f"  {ci+1}/{len(cases)} cases")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[baselines] wrote {len(rows)} sentence rows to {a.out}")


if __name__ == "__main__":
    main()
