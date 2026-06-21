import pytest

from agastya.stages.evidence.merkle import leaf_hash, merkle_root, verify_leaf


def test_leaf_hash_is_deterministic():
    assert leaf_hash(b"abc") == leaf_hash(b"abc")


def test_leaf_hash_changes_with_input():
    assert leaf_hash(b"abc") != leaf_hash(b"abd")


def test_root_of_single_leaf_is_leaf():
    h = leaf_hash(b"only")
    assert merkle_root([b"only"]) == h


def test_root_stable_for_two_leaves():
    root = merkle_root([b"a", b"b"])
    assert root == merkle_root([b"a", b"b"])
    assert root != merkle_root([b"b", b"a"])


def test_verify_leaf_membership():
    leaves = [b"a", b"b", b"c"]
    root = merkle_root(leaves)
    assert verify_leaf(b"b", leaves, root) is True
    assert verify_leaf(b"z", leaves, root) is False


def test_empty_leaves_rejected():
    with pytest.raises(ValueError):
        merkle_root([])
