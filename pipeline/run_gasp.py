# -*- coding: utf-8 -*-
"""
Unified canonical GASP scorer.

Every method downstream consumes the SAME canonical chunks/sentences: chunk i is the
character span context[s:e] defined once by gasp_canonical, never re-tokenized. The
scorer:
  - builds a canonical Case per example (full context/answer, char-span chunks and
    sentences, source_id, hallucination spans);
  - scores full context / no context / per-canonical-chunk leave-one-out, holding the
    answer fixed; a chunk that falls outside the model's retained context window is
    recorded as NaN and counted, not silently mis-indexed;
  - emits source_id and the retained context/answer fractions for the truncation audit
    and for source-level splitting;
  - writes cases.jsonl (canonical records), response.csv, sentence.csv, audit.csv.

Advanced models: pass any HF causal LM via --model (e.g. Qwen2.5-7B-Instruct,
Meta-Llama-3.1-8B-Instruct, Mistral-7B-Instruct). Larger contexts via --max_ctx_tokens.
"""
import os, json, argparse, warnings
import numpy as np
from gasp_canonical import build_case, to_record, sentence_spans
warnings.filterwarnings("ignore")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", type=str, default="ragtruth")
    p.add_argument("--tasks", type=str, default="Summary,Data2txt,QA")
    p.add_argument("--n_per_class", type=int, default=300)
    p.add_argument("--k_chunks", type=int, default=5)
    p.add_argument("--max_ctx_tokens", type=int, default=1800)
    p.add_argument("--max_ans_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--outroot", type=str, default=os.path.join(os.path.dirname(__file__), "canon_results"))
    p.add_argument("--datadir", type=str, default=os.path.join(os.path.dirname(__file__), "tofueval_data"))
    p.add_argument("--max_cases", type=int, default=0, help="cap total cases (0 = no cap); used for validation")
    return p.parse_args()


class LM:
    def __init__(self, model_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, device_map=None).to(self.device)
        self.model.eval(); torch.set_grad_enabled(False)
        print(f"      LM {self.device}: {model_id}")

    def _dist(self, prompt_ids, aid):
        torch = self.torch
        ids = torch.cat([prompt_ids, aid]).unsqueeze(0).to(self.device)
        P, A = prompt_ids.numel(), aid.numel()
        logits = self.model(ids).logits[0][P - 1:P - 1 + A].float()
        lp = torch.log_softmax(logits, dim=-1)
        aid_d = aid.to(self.device)
        tlp = lp[torch.arange(A, device=self.device), aid_d]
        return tlp, lp.exp(), lp

    def _jsd(self, p_full, lp_full, p_q):
        # JSD in nats, bounded in [0, ln 2]; guarded so downstream can assert bounds
        eps = 1e-8
        m = 0.5 * (p_full + p_q); lm = (m + eps).log()
        kl_p = (p_full * (lp_full - lm)).sum(-1)
        lq = (p_q + eps).log(); kl_q = (p_q * (lq - lm)).sum(-1)
        j = 0.5 * kl_p + 0.5 * kl_q
        return j.clamp_(min=0.0, max=float(np.log(2)) + 1e-4)

    def score(self, case, max_ctx, max_ans):
        """Score a canonical Case. Chunk indices are the case's canonical chunks."""
        torch = self.torch
        # answer tokens with char offsets; truncate to window, record retained answer chars
        enc = self.tok(case.answer, return_offsets_mapping=True, add_special_tokens=False)
        aid = torch.tensor(enc["input_ids"][:max_ans])
        offs = enc["offset_mapping"][:max_ans]
        A = aid.numel()
        if A < 8:
            return None
        ans_ret_chars = offs[-1][1] if offs else 0            # chars of answer actually scored

        # context: truncate to window BUT keep canonical chunk spans; record retained char range
        cids = self.tok(case.context, add_special_tokens=False, return_offsets_mapping=True)
        ids_full = cids["input_ids"]; coffs = cids["offset_mapping"]
        keep = ids_full[:max_ctx]
        ctx_ret_chars = coffs[max_ctx - 1][1] if len(coffs) >= max_ctx and max_ctx > 0 else len(case.context)
        ctx_used = case.context[:ctx_ret_chars]

        def prompt(ctx):
            return torch.tensor(self.tok(f"Context:\n{ctx}\n\nQuestion: {case.query}\n\nAnswer: ").input_ids)

        tlp_full, p_full, lp_full = self._dist(prompt(ctx_used), aid)
        tlp_full_c = tlp_full.cpu().numpy()
        tlp_noc, p_noc, _ = self._dist(prompt(""), aid)
        gap_tok = (tlp_full - tlp_noc).cpu().numpy()
        jsd_noc_tok = self._jsd(p_full, lp_full, p_noc).cpu().numpy()
        del p_noc

        Kc = len(case.chunk_spans)
        drop_tok = np.full((Kc, A), np.nan)
        jsd_tok = np.full((Kc, A), np.nan)
        chunk_in_window = []
        for ci, (cs, ce) in enumerate(case.chunk_spans):
            if cs >= ctx_ret_chars:            # chunk fully outside the retained window
                chunk_in_window.append(0)
                continue
            chunk_in_window.append(1)
            # remove this canonical chunk from the retained context by char slicing
            rem = ctx_used[:cs] + ctx_used[min(ce, ctx_ret_chars):]
            tlp, p_l, _ = self._dist(prompt(rem), aid)
            drop_tok[ci] = (tlp_full - tlp).cpu().numpy()
            jsd_tok[ci] = self._jsd(p_full, lp_full, p_l).cpu().numpy()
            del p_l
        del p_full, lp_full
        return dict(offsets=offs, n=A, tlp_full=tlp_full_c, gap_tok=gap_tok,
                    jsd_noc_tok=jsd_noc_tok, drop_tok=drop_tok, jsd_tok=jsd_tok,
                    ctx_ret_chars=int(ctx_ret_chars), ans_ret_chars=int(ans_ret_chars),
                    chunk_in_window=chunk_in_window)


def load_ragtruth_cases(tasks, n_per_class, k_chunks, seed):
    from datasets import load_dataset
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    keep = set(t.strip() for t in tasks.split(","))
    rng = np.random.default_rng(seed); idx = rng.permutation(len(ds))
    pos, neg = [], []
    for i in idx:
        i = int(i); r = ds[i]
        if r["task_type"] not in keep:
            continue
        if not (r["output"] or "").strip() or not (r["context"] or "").strip():
            continue
        lab = r["hallucination_labels_processed"]
        is_pos = (lab["evident_conflict"] + lab["baseless_info"]) > 0
        b = pos if is_pos else neg
        if len(b) >= n_per_class:
            continue
        # hallucination spans (char) over the answer
        hs = []
        if (r["hallucination_labels"] or "").strip():
            for h in json.loads(r["hallucination_labels"]):
                t = "conflict" if "conflict" in h["label_type"].lower() else "baseless"
                hs.append((int(h["start"]), int(h["end"]), t))
        # source_id: RAGTruth provides source_id; fall back to source_info id, else row idx
        sid = r.get("source_id") or r.get("source_info_id") or f"row{i}"
        case = build_case(case_id=f"{sid}::{i}", source_id=str(sid), answer_id=str(i),
                          dataset="ragtruth", task=r["task_type"], query=r.get("query", ""),
                          context=r["context"], answer=r["output"], k_chunks=k_chunks,
                          halluc_spans=hs)
        if case is None:
            continue
        b.append((case, r))
        if len(pos) >= n_per_class and len(neg) >= n_per_class:
            break
    n = min(len(pos), len(neg))
    return [c for c, _ in (pos[:n] + neg[:n])]


def _spans_from_sentences(sents, labels, htype="baseless"):
    """Concatenate labeled sentences into one answer; return (answer, halluc_spans)
    where a hallucinated sentence contributes its char range as a halluc span."""
    answer, hspans = "", []
    for s, lab in zip(sents, labels):
        st = len(answer)
        answer += s + " "
        if lab:
            hspans.append((st, st + len(s), htype))
    return answer.strip(), hspans


def load_tofueval_cases(datadir, k_chunks):
    """TofuEval (MeetingBank): topic-focused summaries with per-sentence factual labels.
    Context = transcript, query = topic, source_id = doc_id (summaries of one meeting
    share the transcript, so the source-level split keeps them together)."""
    from datasets import load_dataset
    import pandas as pd
    dev = pd.read_csv(os.path.join(datadir, "meetingbank_factual_eval_dev.csv"))
    test = pd.read_csv(os.path.join(datadir, "meetingbank_factual_eval_test.csv"))
    tf = pd.concat([dev, test], ignore_index=True)
    mb = load_dataset("huuuyeah/meetingbank")
    tmap = {}
    for split in mb.keys():
        for r in mb[split]:
            for key in ("id", "uid"):
                v = r.get(key)
                if v:
                    tmap[str(v)] = r["transcript"]
    cases = []
    for (doc_id, topic, model_name), g in tf.groupby(["doc_id", "topic", "model_name"]):
        if str(doc_id) not in tmap:
            continue
        g = g.sort_values("sent_idx")
        sents = [str(s).strip() for s in g["summ_sent"].tolist()]
        labels = [1 if str(l).strip().lower() == "no" else 0 for l in g["sent_label"].tolist()]
        answer, hspans = _spans_from_sentences(sents, labels)
        case = build_case(case_id=f"{doc_id}::{topic}::{model_name}", source_id=str(doc_id),
                          answer_id=f"{topic}::{model_name}", dataset="tofueval", task="Summary",
                          query=str(topic).strip(), context=tmap[str(doc_id)], answer=answer,
                          k_chunks=k_chunks, halluc_spans=hspans)
        if case is not None:
            cases.append(case)
    np.random.default_rng(42).shuffle(cases)   # diversify source_ids before any --max_cases cap
    return cases


def load_ragbench_cases(n_per_class, k_chunks, seed):
    """RAGBench (multi-domain RAG): question -> query, joined documents -> context,
    response -> answer; sentence label = key in unsupported_response_sentence_keys.
    source_id = hash of the documents so responses over the same evidence group."""
    from datasets import load_dataset
    import hashlib
    domains = ["pubmedqa", "finqa", "covidqa", "hotpotqa", "techqa", "delucionqa"]
    pool = []
    for dom in domains:
        try:
            ds = load_dataset("rungalileo/ragbench", dom, split="test")
        except Exception as e:
            print("  skip domain", dom, repr(e)[:60]); continue
        for e in ds:
            docs = e.get("documents") or []
            rsents = e.get("response_sentences") or []
            if not docs or not (e.get("response") or "").strip() or not rsents:
                continue
            unsup = set(e.get("unsupported_response_sentence_keys") or [])
            ctx = "\n".join(str(d) for d in docs)
            sid = dom + "_" + hashlib.md5(ctx.encode("utf-8")).hexdigest()[:12]
            pool.append(dict(domain=dom, query=(e.get("question") or "").strip(), context=ctx,
                             sents=[str(t) for _, t in rsents],
                             labels=[1 if str(k) in unsup else 0 for k, _ in rsents],
                             resp_label=0 if e.get("adherence_score") is True else 1, source_id=sid))
    rng = np.random.default_rng(seed); idx = rng.permutation(len(pool))
    pos, neg = [], []
    for i in idx:
        r = pool[int(i)]
        b = pos if r["resp_label"] == 1 else neg
        if len(b) < n_per_class:
            b.append(r)
        if len(pos) >= n_per_class and len(neg) >= n_per_class:
            break
    n = min(len(pos), len(neg)); sample = pos[:n] + neg[:n]
    print(f"      RAGBench balanced: {n}/class from {len(pool)} pooled")
    cases = []
    for j, r in enumerate(sample):
        answer, hspans = _spans_from_sentences(r["sents"], r["labels"])
        case = build_case(case_id=f"{r['source_id']}::{j}", source_id=r["source_id"], answer_id=str(j),
                          dataset="ragbench", task=r["domain"], query=r["query"], context=r["context"],
                          answer=answer, k_chunks=k_chunks, halluc_spans=hspans)
        if case is not None:
            cases.append(case)
    return cases


def main():
    args = get_args()
    tag = args.tag or (args.model.split("/")[-1] + f"_{args.dataset}_K{args.k_chunks}")
    outdir = os.path.join(args.outroot, tag); os.makedirs(outdir, exist_ok=True)
    import pandas as pd

    print(f"[1/3] loading {args.model}")
    lm = LM(args.model)
    print(f"[2/3] building canonical cases ({args.dataset}, tasks={args.tasks})")
    if args.dataset == "ragtruth":
        cases = load_ragtruth_cases(args.tasks, args.n_per_class, args.k_chunks, args.seed)
    elif args.dataset == "tofueval":
        cases = load_tofueval_cases(args.datadir, args.k_chunks)
    elif args.dataset == "ragbench":
        cases = load_ragbench_cases(args.n_per_class, args.k_chunks, args.seed)
    else:
        raise SystemExit(f"dataset {args.dataset} not wired yet (add a loader)")
    if args.max_cases and len(cases) > args.max_cases:
        cases = cases[:args.max_cases]
    print(f"      {len(cases)} balanced cases")

    print("[3/3] scoring")
    cf = open(os.path.join(outdir, "cases.jsonl"), "w", encoding="utf-8")
    resp_rows, sent_rows, audit_rows = [], [], []
    for jj, case in enumerate(cases):
        try:
            sc = lm.score(case, args.max_ctx_tokens, args.max_ans_tokens)
            if sc is None:
                continue
        except Exception as e:
            print("   skip:", repr(e)[:70]); continue
        cf.write(json.dumps(to_record(case)) + "\n")
        K = len(case.chunk_spans)
        lab = 1 if any(True for _ in case.halluc_spans) else 0
        # audit: what was scored vs available
        n_sent = len(case.sent_spans)
        n_sent_scored = sum(1 for (s, e) in case.sent_spans if s < sc["ans_ret_chars"])
        halluc_outside = sum(1 for (hs, he, _) in case.halluc_spans if hs >= sc["ans_ret_chars"])
        audit_rows.append(dict(case_id=case.case_id, source_id=case.source_id, task=case.task,
            ctx_chars=len(case.context), ctx_ret_frac=round(sc["ctx_ret_chars"]/max(len(case.context),1), 3),
            ans_chars=len(case.answer), ans_ret_frac=round(sc["ans_ret_chars"]/max(len(case.answer),1), 3),
            n_chunks=K, chunks_in_window=int(sum(sc["chunk_in_window"])),
            n_sent=n_sent, n_sent_scored=n_sent_scored, halluc_spans_outside_scored=halluc_outside))
        resp_rows.append(dict(model=tag, case_id=case.case_id, source_id=case.source_id, task=case.task,
            mean_surprisal=float(-np.mean(sc["tlp_full"])), n_ans=sc["n"],
            gap=float(np.mean(sc["gap_tok"])), jsd_noctx=float(np.nanmean(sc["jsd_noc_tok"])),
            chunk_drops=json.dumps([None if not sc["chunk_in_window"][k] else float(np.nanmean(sc["drop_tok"][k])) for k in range(K)]),
            chunk_jsds=json.dumps([None if not sc["chunk_in_window"][k] else float(np.nanmean(sc["jsd_tok"][k])) for k in range(K)]),
            label=lab))
        offs = sc["offsets"]
        for j_s, (s, e) in enumerate(case.sent_spans):
            tk = [t for t in range(sc["n"]) if offs[t][0] >= s and offs[t][0] < e]
            slab, stype = case.sent_label(j_s)
            fully_scored = int(e <= sc["ans_ret_chars"])
            if len(tk) < 3:
                continue
            tk = np.array(tk)
            sent_rows.append(dict(model=tag, case_id=case.case_id, source_id=case.source_id, sent_idx=j_s,
                mean_surprisal=float(-np.mean(sc["tlp_full"][tk])), n_tok=int(len(tk)),
                gap=float(np.mean(sc["gap_tok"][tk])), jsd_noctx=float(np.nanmean(sc["jsd_noc_tok"][tk])),
                chunk_drops=json.dumps([None if not sc["chunk_in_window"][k] else float(np.nanmean(sc["drop_tok"][k][tk])) for k in range(K)]),
                chunk_jsds=json.dumps([None if not sc["chunk_in_window"][k] else float(np.nanmean(sc["jsd_tok"][k][tk])) for k in range(K)]),
                sent_type=stype or "none", label=slab, fully_scored=fully_scored,
                sent_text=case.sent_text(j_s)[:600]))
        if (jj + 1) % 50 == 0:
            print(f"      {jj+1}/{len(cases)} done")
    cf.close()
    pd.DataFrame(resp_rows).to_csv(os.path.join(outdir, "response.csv"), index=False)
    pd.DataFrame(sent_rows).to_csv(os.path.join(outdir, "sentence.csv"), index=False)
    pd.DataFrame(audit_rows).to_csv(os.path.join(outdir, "audit.csv"), index=False)
    print(f"      saved {len(resp_rows)} responses, {len(sent_rows)} sentences, audit to {outdir}")


if __name__ == "__main__":
    main()
