"""Tests for dataset-generation seed handling (no solver needed)."""

from build_dataset import _split_rng


def test_split_rngs_produce_disjoint_seed_streams():
    # With a shared base seed, the per-split rngs must not replay each other's
    # per-sample seeds — otherwise val/test become subsets of train.
    streams = {
        split: set(_split_rng(42, split).integers(0, 2**31, size=1000).tolist())
        for split in ("train", "val", "test")
    }
    assert streams["train"].isdisjoint(streams["val"])
    assert streams["train"].isdisjoint(streams["test"])
    assert streams["val"].isdisjoint(streams["test"])


def test_split_rng_is_deterministic():
    a = _split_rng(7, "train").integers(0, 2**31, size=100)
    b = _split_rng(7, "train").integers(0, 2**31, size=100)
    assert (a == b).all()


def test_split_rng_differs_by_base_seed():
    a = _split_rng(1, "test").integers(0, 2**31, size=100)
    b = _split_rng(2, "test").integers(0, 2**31, size=100)
    assert (a != b).any()
