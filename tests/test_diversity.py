"""Tests for pairwise perceptual-diversity aggregation (krea2_explorations.diversity).

The metric model (DreamSim/LPIPS) is injected as ``dist_fn`` so this stays dependency-light and
runs under the project venv. Here we exercise the pure aggregation with a trivial scalar distance;
the real harness passes a cosine distance over precomputed perceptual embeddings.
"""
import pytest

from krea2_explorations.diversity import (
    diversity_table,
    pairwise_distances,
    pairwise_diversity,
)


def _absdiff(a, b):
    return abs(a - b)


def test_pairwise_distances_enumerates_upper_triangle():
    # pairs (1,2)=1, (1,5)=4, (2,5)=3 in row-major order
    assert pairwise_distances([1, 2, 5], _absdiff) == [1, 4, 3]


def test_pairwise_diversity_is_mean_of_pairs():
    assert pairwise_diversity([1, 2, 5], _absdiff) == pytest.approx((1 + 4 + 3) / 3)


def test_two_items_is_the_single_distance():
    assert pairwise_diversity([3, 7], _absdiff) == pytest.approx(4)


def test_identical_items_have_zero_diversity():
    assert pairwise_diversity([9, 9, 9, 9], _absdiff) == 0.0


def test_order_independent():
    assert pairwise_diversity([1, 2, 5], _absdiff) == pytest.approx(
        pairwise_diversity([5, 1, 2], _absdiff)
    )


def test_fewer_than_two_items_raises():
    with pytest.raises(ValueError):
        pairwise_diversity([1], _absdiff)
    with pytest.raises(ValueError):
        pairwise_distances([], _absdiff)


def test_diversity_table_maps_each_arm_to_its_mean():
    table = diversity_table({"baseline": [9, 9, 9], "diverse": [0, 5, 10]}, _absdiff)
    assert table["baseline"] == 0.0
    assert table["diverse"] == pytest.approx((5 + 10 + 5) / 3)


def test_diversity_table_arm_needs_two_items():
    with pytest.raises(ValueError):
        diversity_table({"x": [1]}, _absdiff)
