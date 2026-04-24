import pytest

from xtc.utils.math import (
    estimate_count_prob_smooth,
    estimate_unique_prob_good_turing,
    estimate_richness_chao1,
)

RTOL, ATOL = 1e-3, 1e-6

PROB_SMOOTH_TESTS = [
    (0, 0, 1, {}, "zero sample smooth to 1"),
    (0, 1, 1/2, {}, "zero count smooth to 0.5"),
    (1, 1, 1, {}, "1/1 is 1"),
    (1, 10, 2/11, {}, "1/10 smooth to 0.2"),
    (1, 100, 2/101, {}, "1/10 smooth to 0.02"),
    (1, 10, 2/11, {}, "1/10 smooth to 0.2"),
    (10, 10, 1, {}, "10/10 is 1"),
    (10, 100, 11/101, {}, "10/100 smooth to 0.1"),
    (10, 1000, 11/1001, {}, "10/1000 smooth to 0.01"),
    (0, 0, 0, dict(alpha=0), "unsmoothed 0 is 0"),
    (0, 1, 0, dict(alpha=0), "unsmoothed 0 is 0"),
    (0, 0, 1, dict(alpha=2), "alpha 2 0 smooth to 1"),
    (0, 1, 2/3, dict(alpha=2), "alpha 2 1 smooth to 2/3"),
]

@pytest.mark.parametrize(
    "count, samples, expected, kwargs, msg",
    PROB_SMOOTH_TESTS,
)
def test_prob_smooth(count: int, samples: int, expected: int, kwargs: dict, msg: str):
    estimated = estimate_count_prob_smooth(count, samples, **kwargs)
    assert abs(estimated-expected) <= RTOL * expected + ATOL, f"unexpected: {msg}"


GOOD_TURING_TESTS = [
    (0, 0, 1, "emty sample smooth to 1"),
    (1, 0, 1/2, "no once smooth to 0.50"),
    (10, 1, 2/11, "1 once approximately 0.2"),
    (100, 1, 2/101, "1 once approximately 0.02"),
    (1000, 1, 2/1001, "no once approximately 0.002"),
    (10, 10, 11/11, "ten once is 1"),
    (100, 10, 11/101, "ten once approximately 0.1"),
    (1000, 10, 11/1001, "ten once approximately 0.01"),
]

@pytest.mark.parametrize(
    "samples, once, expected, msg",
    GOOD_TURING_TESTS,
)
def test_good_turing(samples: int, once: int, expected: int, msg: str):
    estimated = estimate_unique_prob_good_turing(samples, once)
    assert abs(estimated-expected) <= RTOL * expected + ATOL, f"unexpected: {msg}"


CHAO1_TESTS = [
    (0, 0, 0, 0, "empty returns empty"),
    (1, 0, 0, 1, "no once returns same"),
    (100, 0, 0, 100, "no once returns same"),
    (1, 1, 0, 1, "1 once estimates to +0"),
    (2, 1, 1, 2, "1 once, 1 twice estimates to +0"),
    (2, 2, 0, 2+1, "2 once estimates to +1"),
    (3, 2, 1, 3+0.5, "2 once, 1 twice estimates to +0.5"),
    (9, 8, 1, 9+14, "8 once, 1 twice estimates to +14"),
    (100, 8, 1, 100+14, "do not depend on draw size"),
]

@pytest.mark.parametrize(
    "uniques, once, twice, expected, msg",
    CHAO1_TESTS,
)
def test_chao1(uniques: int, once: int, twice: int, expected: int, msg: str):
    estimated = estimate_richness_chao1(uniques, once, twice)
    assert estimated == expected, f"unexpected: {msg}"
