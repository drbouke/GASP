import math

from gasp import evaluate, roc_auc, threshold_metrics


def test_roc_auc_perfect_and_reversed():
    labels = [0, 0, 1, 1]
    assert roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_roc_auc_ties_give_half():
    assert abs(roc_auc([0, 1], [0.5, 0.5]) - 0.5) < 1e-9


def test_threshold_metrics_counts():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.4, 0.6, 0.1]  # threshold 0.5 -> predict [1,0,1,0]
    m = threshold_metrics(labels, scores, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert abs(m["precision"] - 0.5) < 1e-9
    assert abs(m["recall"] - 0.5) < 1e-9


def test_evaluate_suite_keys():
    m = evaluate([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], threshold=0.5)
    for key in ("n", "positives", "roc_auc", "pr_auc", "precision", "recall", "f1", "mcc"):
        assert key in m
    assert not math.isnan(m["roc_auc"])
