# -*- coding: utf-8 -*-
"""
Extended scoring harness for GASP.

For each RAGTruth response it re-scores the fixed answer under the full context,
no context, and each leave-one-out chunk removal, and stores the PER-CHUNK
log-likelihood drops and JSDs (as JSON lists) plus the no-context features, at
both response and sentence level. Storing the per-chunk vectors lets the offline
analysis derive:
  - single-feature and group ablations,
  - max vs mean vs top-2 aggregation,
  - the threshold-only (no classifier) variant,
without re-running the model. Chunking sensitivity is obtained by re-running with
different --k_chunks. Cross-family robustness by re-running with a different --model.

Also stores sentence text, answer text, and the RAGTruth row index so the
additional baselines (large NLI, self-consistency) can be computed later.
"""

from config import RESULTS
import os, re, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n_per_class", type=int, default=200)
    p.add_argument("--tasks", type=str, default="Summary,Data2txt")
    p.add_argument("--k_chunks", type=int, default=5)
    p.add_argument("--max_ctx_tokens", type=int, default=700)
    p.add_argument("--max_ans_tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--outroot", type=str, default=str(RESULTS))
    return p.parse_args()

def sentence_spans(text):
    spans, start = [], 0
    for m in re.finditer(r'[.!?]+(?:\s+|$)', text):
        end = m.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans

def split_chunks(context, k):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', context.strip()) if s.strip()]
    if len(sents) <= k:
        return sents if sents else [context]
    per = int(np.ceil(len(sents) / k))
    return [" ".join(sents[i:i+per]) for i in range(0, len(sents), per)]

class LM:
    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt).to(self.device)
        self.model.eval(); torch.set_grad_enabled(False)
        print(f"      LM {self.device}: {model_id}")

    def _dist(self, prompt_ids, aid):
        torch = self.torch
        ids = torch.cat([prompt_ids, aid]).unsqueeze(0).to(self.device)
        P, A = prompt_ids.numel(), aid.numel()
        logits = self.model(ids).logits[0][P-1:P-1+A].float()
        lp = torch.log_softmax(logits, dim=-1)
        aid_d = aid.to(self.device)
        tlp = lp[torch.arange(A, device=self.device), aid_d]
        return tlp, lp.exp(), lp

    def _jsd(self, p_full, lp_full, p_q):
        eps = 1e-8
        m = 0.5 * (p_full + p_q); lm = (m + eps).log()
        kl_p = (p_full * (lp_full - lm)).sum(-1)
        lq = (p_q + eps).log(); kl_q = (p_q * (lq - lm)).sum(-1)
        return 0.5 * kl_p + 0.5 * kl_q

    def score(self, context, query, answer, k_chunks, max_ctx, max_ans):
        torch = self.torch
        cids = self.tok(context, add_special_tokens=False).input_ids[:max_ctx]
        context = self.tok.decode(cids)
        enc = self.tok(answer, return_offsets_mapping=True, add_special_tokens=False)
        aid = torch.tensor(enc["input_ids"][:max_ans]); offs = enc["offset_mapping"][:max_ans]
        A = aid.numel()
        if A < 8:
            return None
        pid_full = torch.tensor(self.tok(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: ").input_ids)
        tlp_full, p_full, lp_full = self._dist(pid_full, aid)
        tlp_full_c = tlp_full.cpu().numpy()
        pid_noc = torch.tensor(self.tok(f"Question: {query}\n\nAnswer: ").input_ids)
        tlp_noc, p_noc, _ = self._dist(pid_noc, aid)
        gap_tok = (tlp_full - tlp_noc).cpu().numpy()
        jsd_noc_tok = self._jsd(p_full, lp_full, p_noc).cpu().numpy()
        del p_noc
        chunks = split_chunks(context, k_chunks)
        drop_tok = np.full((len(chunks), A), np.nan)
        jsd_tok = np.full((len(chunks), A), np.nan)
        for ci in range(len(chunks)):
            rem = " ".join(chunks[:ci] + chunks[ci+1:])
            pid = torch.tensor(self.tok(f"Context:\n{rem}\n\nQuestion: {query}\n\nAnswer: ").input_ids)
            tlp, p_l, _ = self._dist(pid, aid)
            drop_tok[ci] = (tlp_full - tlp).cpu().numpy()
            jsd_tok[ci] = self._jsd(p_full, lp_full, p_l).cpu().numpy()
            del p_l
        del p_full, lp_full
        return dict(offsets=offs, n=A, tlp_full=tlp_full_c, gap_tok=gap_tok,
                    jsd_noc_tok=jsd_noc_tok, drop_tok=drop_tok, jsd_tok=jsd_tok, n_chunks=len(chunks))

def main():
    args = get_args()
    tag = args.tag or (args.model.split("/")[-1] + f"_K{args.k_chunks}")
    outdir = os.path.join(args.outroot, tag)
    os.makedirs(outdir, exist_ok=True)
    import pandas as pd
    from datasets import load_dataset

    print(f"[1/3] loading {args.model}")
    lm = LM(args.model)

    print(f"[2/3] sampling RAGTruth (tasks={args.tasks})")
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    keep = set(t.strip() for t in args.tasks.split(","))
    rng = np.random.default_rng(args.seed); idx = rng.permutation(len(ds))
    pos, neg = [], []
    for i in idx:
        i = int(i); r = ds[i]
        if r["task_type"] not in keep: continue
        if not (r["output"] or "").strip() or not (r["context"] or "").strip(): continue
        lab = r["hallucination_labels_processed"]
        b = pos if (lab["evident_conflict"] + lab["baseless_info"]) > 0 else neg
        if len(b) >= args.n_per_class: continue
        b.append(i)
        if len(pos) >= args.n_per_class and len(neg) >= args.n_per_class: break
    n = min(len(pos), len(neg)); rows_idx = pos[:n] + neg[:n]
    print(f"      balanced: {n} hallucinated, {n} clean")

    print(f"[3/3] scoring (K={args.k_chunks})")
    resp_rows, sent_rows = [], []
    for jj, oi in enumerate(rows_idx):
        r = ds[oi]; ans = r["output"].strip(); query = (r["query"] or "").strip()
        try:
            sc = lm.score(r["context"].strip(), query, ans, args.k_chunks, args.max_ctx_tokens, args.max_ans_tokens)
            if sc is None: continue
        except Exception as e:
            print("   skip:", repr(e)[:70]); continue
        spans_h = json.loads(r["hallucination_labels"]) if r["hallucination_labels"].strip() else []
        def htype(s, e):
            t = None
            for h in spans_h:
                if not (e <= h["start"] or s >= h["end"]):
                    if "conflict" in h["label_type"].lower(): return "conflict"
                    t = "baseless"
            return t
        lab = r["hallucination_labels_processed"]
        K = sc["n_chunks"]
        # response level (mean over all tokens)
        resp_rows.append(dict(model=tag, resp_id=jj, orig_idx=oi,
            mean_surprisal=float(-np.mean(sc["tlp_full"])), n_ans=sc["n"],
            gap=float(np.mean(sc["gap_tok"])), jsd_noctx=float(np.mean(sc["jsd_noc_tok"])),
            chunk_drops=json.dumps([float(np.mean(sc["drop_tok"][k])) for k in range(K)]),
            chunk_jsds=json.dumps([float(np.mean(sc["jsd_tok"][k])) for k in range(K)]),
            has_conflict=int(lab["evident_conflict"]>0), has_baseless=int(lab["baseless_info"]>0),
            label=int((lab["evident_conflict"]+lab["baseless_info"])>0),
            answer=ans[:1200]))
        # sentence level
        offs = sc["offsets"]
        for (s, e) in sentence_spans(ans):
            tk = [t for t in range(sc["n"]) if offs[t][0] >= s and offs[t][0] < e]
            if len(tk) < 3: continue
            tk = np.array(tk); lt = htype(s, e)
            sent_rows.append(dict(model=tag, resp_id=jj, orig_idx=oi,
                mean_surprisal=float(-np.mean(sc["tlp_full"][tk])), n_tok=int(len(tk)),
                gap=float(np.mean(sc["gap_tok"][tk])), jsd_noctx=float(np.mean(sc["jsd_noc_tok"][tk])),
                chunk_drops=json.dumps([float(np.mean(sc["drop_tok"][k][tk])) for k in range(K)]),
                chunk_jsds=json.dumps([float(np.mean(sc["jsd_tok"][k][tk])) for k in range(K)]),
                sent_type=(lt or "none"), label=int(lt is not None), sent_text=ans[s:e].strip()[:600]))
        if (jj + 1) % 50 == 0: print(f"      {jj+1}/{len(rows_idx)} done")
    pd.DataFrame(resp_rows).to_csv(os.path.join(outdir, "response.csv"), index=False)
    pd.DataFrame(sent_rows).to_csv(os.path.join(outdir, "sentence.csv"), index=False)
    print(f"      saved {len(resp_rows)} responses, {len(sent_rows)} sentences to {outdir}")

if __name__ == "__main__":
    main()
