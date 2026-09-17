"""Read a file of RAG items and write detection results, as JSONL or CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def read_items(path: str) -> List[Dict[str, str]]:
    """Read items from a ``.jsonl`` or ``.csv`` file.

    Each item must have ``context`` and ``answer`` columns or keys; ``query`` is
    optional. Any extra fields (for example a ``label``) are carried through.
    """
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif p.suffix.lower() == ".csv":
        with p.open(encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
    else:
        raise ValueError("input must be a .jsonl or .csv file")
    for it in items:
        if "context" not in it or "answer" not in it:
            raise ValueError("each item needs a 'context' and an 'answer' field")
    return items


def write_records(records: List[Dict], path: str) -> None:
    """Write result records to ``.jsonl`` or ``.csv``."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    elif p.suffix.lower() == ".csv":
        if not records:
            p.write_text("", encoding="utf-8")
            return
        fields = list(records[0].keys())
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
    else:
        raise ValueError("output must be a .jsonl or .csv file")
