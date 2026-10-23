
def recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """Fraction of known relevant chunks retrieved in the top k."""
    if not relevant_ids:
        return 0.0

    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[int],
    relevant_ids: set[int],
) -> float:
    """Reciprocal rank of the first relevant result, or 0 if absent."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0
