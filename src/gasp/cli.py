"""Command-line interface: ``gasp detect`` and ``gasp eval``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from . import __version__
from .io import read_items, write_records
from .metrics import evaluate


def _detect(args: argparse.Namespace) -> int:
    from .detector import GASP  # imported here so `gasp eval` needs no torch

    items = read_items(args.input)
    det = GASP(
        args.model,
        k_chunks=args.k_chunks,
        threshold=args.threshold,
        economical=args.economical,
        sensitivity_feature=args.feature,
        max_ctx_tokens=args.max_ctx_tokens,
        max_ans_tokens=args.max_ans_tokens,
        device=args.device,
    )
    records: List[dict] = []
    for item_id, it in enumerate(items):
        result = det.detect(it["context"], it["answer"], it.get("query", ""))
        for s in result:
            rec = {"item_id": item_id, "sent_index": s.index, "sentence": s.text,
                   "sensitivity": s.sensitivity, "flagged": s.flagged,
                   "supporting_chunk": s.supporting_chunk}
            rec.update({f"feat_{k}": v for k, v in s.features.items()})
            if "label" in it:
                rec["label"] = it["label"]
            records.append(rec)
    write_records(records, args.output)
    print(f"wrote {len(records)} sentence results from {len(items)} answers to {args.output}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    import csv
    from pathlib import Path

    p = Path(args.input)
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    labels, scores = [], []
    for r in rows:
        if r.get(args.label_col) in (None, ""):
            continue
        labels.append(int(float(r[args.label_col])))
        # lower sensitivity means more suspicious, so the suspicion score is negated
        scores.append(-float(r[args.score_col]))
    # --threshold is a sensitivity threshold (flag if sensitivity < threshold); on the
    # negated suspicion score that becomes suspicion >= -threshold
    thr = None if args.threshold is None else -args.threshold
    m = evaluate(labels, scores, threshold=thr)
    for k, v in m.items():
        print(f"{k:12s} {v:.4f}" if isinstance(v, float) else f"{k:12s} {v}")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gasp", description="Grounding-sensitivity detection for RAG.")
    ap.add_argument("--version", action="version", version=f"gasp {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="score answers in a .jsonl/.csv file")
    d.add_argument("--input", required=True, help=".jsonl/.csv with context, answer, [query]")
    d.add_argument("--output", required=True, help=".jsonl/.csv for per-sentence results")
    d.add_argument("--model", required=True, help="Hugging Face scorer model id")
    d.add_argument("--k-chunks", type=int, default=5, dest="k_chunks")
    d.add_argument("--threshold", type=float, default=None)
    d.add_argument("--economical", action="store_true", help="two-pass variant, no attribution")
    d.add_argument("--feature", default="max_drop", help="sensitivity feature to report")
    d.add_argument("--max-ctx-tokens", type=int, default=1800, dest="max_ctx_tokens")
    d.add_argument("--max-ans-tokens", type=int, default=256, dest="max_ans_tokens")
    d.add_argument("--device", default=None, help="cpu or cuda")
    d.set_defaults(func=_detect)

    e = sub.add_parser("eval", help="compute metrics from a labeled results file")
    e.add_argument("--input", required=True, help=".jsonl/.csv with a label and a score column")
    e.add_argument("--label-col", default="label", dest="label_col")
    e.add_argument("--score-col", default="sensitivity", dest="score_col")
    e.add_argument("--threshold", type=float, default=None, help="suspicion threshold for point metrics")
    e.set_defaults(func=_eval)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
