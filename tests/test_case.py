from gasp import Case


def test_case_from_texts_segments_and_recovers():
    context = "Alpha fact one. Beta fact two. Gamma fact three. Delta fact four."
    answer = "The report restates alpha. It also mentions delta."
    case = Case.from_texts(context, answer, query="summarize", k_chunks=3)

    assert case.n_sentences == 2
    assert 1 <= case.n_chunks <= 3
    # accessors recover the exact spanned text
    for j in range(case.n_sentences):
        assert case.sentence(j).strip()
        assert case.sentence(j) == answer[case.sent_spans[j][0]:case.sent_spans[j][1]]
    for i in range(case.n_chunks):
        assert case.chunk(i) == context[case.chunk_spans[i][0]:case.chunk_spans[i][1]]


def test_case_strips_and_handles_query_default():
    case = Case.from_texts("  a sentence.  ", "  an answer.  ")
    assert case.context == "a sentence."
    assert case.answer == "an answer."
    assert case.query == ""
