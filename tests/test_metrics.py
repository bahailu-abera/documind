
from evaluation.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_finds_all_relevant_chunks():
    assert recall_at_k([3, 7, 2], {3, 2}, 3) == 1.0


def test_recall_at_k_finds_only_some_relevant_chunks():
    assert recall_at_k([3, 7], {3, 2}, 2) == 0.5


def test_recall_at_k_returns_zero_when_none_found():
    assert recall_at_k([4, 5], {1, 2}, 2) == 0.0


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank([8, 3, 2], {3, 2}) == 0.5


def test_reciprocal_rank_returns_zero_when_none_found():
    assert reciprocal_rank([8, 9], {3}) == 0.0
