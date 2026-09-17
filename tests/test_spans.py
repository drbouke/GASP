from gasp import chunk_spans, sentence_spans


def test_sentence_spans_tile_and_recover():
    text = "First sentence. Second one! Third? tail without period"
    spans = sentence_spans(text)
    assert len(spans) == 4
    # every span recovers non-empty text and spans are ordered and non-overlapping
    prev_end = 0
    for s, e in spans:
        assert s >= prev_end
        assert text[s:e].strip()
        prev_end = e


def test_chunk_spans_respects_k_and_recovers_text():
    context = " ".join(f"Sentence number {i}." for i in range(12))
    spans = chunk_spans(context, k=5)
    assert 1 <= len(spans) <= 5
    for s, e in spans:
        assert context[s:e].strip()


def test_chunk_spans_fewer_sentences_than_k():
    context = "Only one sentence here."
    spans = chunk_spans(context, k=5)
    assert len(spans) == 1


def test_empty_inputs():
    assert sentence_spans("") == []
    assert chunk_spans("", 5) == []
