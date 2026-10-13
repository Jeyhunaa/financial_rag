from rag import fuse_ranked_candidates


def test_weighted_fusion_protects_strong_dense_candidate():
    dense = [10, 11, 12]
    lexical = [20, 21, 22]
    weighted = fuse_ranked_candidates(dense, lexical, dense_weight=3)
    balanced = fuse_ranked_candidates(dense, lexical, dense_weight=1)
    assert weighted[10] > weighted[20]
    assert balanced[10] == balanced[20]


def test_fusion_deduplicates_and_rejects_invalid_weights():
    import pytest
    assert fuse_ranked_candidates([1, 1], [1])[1] == 4 / 61
    with pytest.raises(ValueError):
        fuse_ranked_candidates([1], [2], dense_weight=0)
