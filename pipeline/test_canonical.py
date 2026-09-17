# -*- coding: utf-8 -*-
"""
Closure tests for the canonical preprocessing.

These guard the core invariant: "chunk i" and "sentence j" denote the SAME text in
every method. Any mismatch fails here and must halt an experiment rather than pass
silently. CPU-only; run: python test_canonical.py
"""
import json
from gasp_canonical import (build_case, to_record, from_record, sentence_spans,
                            chunk_spans)

CTX = ("Aspirin lowers fever. It also thins the blood. Patients on warfarin should avoid it. "
       "The recommended dose is 325 mg. Overdose can cause tinnitus. Store below 25 C.")
ANS = ("Aspirin reduces fever and thins blood. The usual dose is 325 mg. "
       "It cures diabetes permanently.")


def _case(k=5):
    # hallucination span over the third answer sentence ("It cures diabetes...")
    s3 = sentence_spans(ANS)[2]
    return build_case(case_id="src1::ans1", source_id="src1", answer_id="ans1",
                      dataset="ragtruth", task="Summary", query="What does aspirin do?",
                      context=CTX, answer=ANS, k_chunks=k,
                      halluc_spans=[(s3[0], s3[1], "baseless")])


def test_chunk_text_identity_across_consumers():
    """Two independent 'methods' that reference chunk i must see identical text."""
    c = _case()
    # method A (e.g. sensitivity): builds chunk list by slicing canonical spans
    method_a = [c.context[s:e] for (s, e) in c.chunk_spans]
    # method B (e.g. NLI baseline): uses the accessor
    method_b = [c.chunk_text(i) for i in range(len(c.chunk_spans))]
    assert method_a == method_b, "chunk text differs between consumers"
    # and every chunk is a real substring of the original context (no re-tokenized drift)
    for t in method_a:
        assert t in c.context
    print(f"  ok: {len(method_a)} chunks identical across consumers")


def test_chunk_count_leq_k():
    for k in (3, 5, 8):
        c = _case(k)
        assert len(c.chunk_spans) <= k, f"more than {k} chunks"
    print("  ok: chunk count <= k for k in {3,5,8}")


def test_sentence_spans_cover_and_nonoverlap():
    sp = sentence_spans(ANS)
    for i in range(1, len(sp)):
        assert sp[i][0] >= sp[i - 1][1], "sentence spans overlap"
    joined = "".join(ANS[s:e] for (s, e) in sp).replace(" ", "")
    assert joined == ANS.replace(" ", ""), "sentence spans miss content"
    print(f"  ok: {len(sp)} sentence spans cover the answer, no overlap")


def test_label_from_overlap():
    """a sentence label comes from spans overlapping ITS char range only."""
    c = _case()
    labels = [c.sent_label(j)[0] for j in range(len(c.sent_spans))]
    assert labels[0] == 0 and labels[1] == 0, "grounded sentences mislabeled"
    assert labels[2] == 1, "hallucinated sentence not labeled"
    print(f"  ok: sentence labels align to overlapping spans: {labels}")


def test_persistence_roundtrip_preserves_text():
    """Saving/loading the canonical record must not change any chunk/sentence text."""
    c = _case()
    d = json.loads(json.dumps(to_record(c)))
    c2 = from_record(d)
    assert [c.chunk_text(i) for i in range(len(c.chunk_spans))] == \
           [c2.chunk_text(i) for i in range(len(c2.chunk_spans))]
    assert [c.sent_text(j) for j in range(len(c.sent_spans))] == \
           [c2.sent_text(j) for j in range(len(c2.sent_spans))]
    print("  ok: JSON round-trip preserves chunk and sentence text")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} canonical closure tests")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print("ALL CANONICAL CLOSURE TESTS PASSED")
