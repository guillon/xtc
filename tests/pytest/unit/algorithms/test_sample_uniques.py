import pytest
from collections.abc import Iterator, Generator
from typing import TypeVar
from itertools import zip_longest, islice
from functools import partial

from xtc.utils.algorithms import sample_uniques

Sample = TypeVar("Sample")

def draw_from_iter(it: Iterator[Sample], num: int) -> Generator[Sample]:
    draws = (s for s, _ in zip_longest(islice(it, num), range(num)) if s is not None)
    return draws

SAMPLER_TESTS = [
    ([], 100, 0, {},
     "empty sampler sould return empty"),
    ([None] * 100 + [1], 100, 1, {},
     "unfrequent sample, should return as prob > 1/1000"),
    ([None] * 100 + [1], 100, 0, dict(prob=1/100),
     "unfrequent sample, should return empty as prob < 1/100"),
    ([None] * 5000 + [1], 100, 0, {},
     "very unfrequent sample, should return empty as prob < 1/1000"),
    ([None] * 5000 + [1], 100, 1, dict(prob=1/10000),
     "very unfrequent sample, should return as prob > 1/10000"),
    ([None] * 100 + [1] + [None] * 100 + [2] + [None] * 100 + [3], 100, 3, {},
    "should return 3 elements"),
    ([1] * 101 + [None] * 100 + [2] + [None] * 100 + [3], 100, 3, {},
    "should return 3 elements"),
    ([None] * 100 + [1] + [None] * 100 + [1] + [None] * 100 + [2], 100, 2, {},
    "should return 2 elements"),
    ([None] * 100 + [1], 1, 1, {},
     "unfrequent 1 sample sould return"),
    ([None] * 100 + [1] + [None] * 100 + [2], 2, 2, {},
     "unfrequent 2 samples sould return"),
]

@pytest.mark.parametrize(
    "draws, requested_len, expected_len, kwargs, message",
    SAMPLER_TESTS,
)
def test_sample_uniques(
        draws: list[Sample],
        requested_len: int,
        expected_len: int,
        kwargs: dict,
        message: str,
):
    sampler = partial(draw_from_iter, iter(draws))
    samples = sample_uniques(sampler, requested_len, **kwargs)
    assert len(samples) == expected_len, f"unexpected: {samples}: {message}"
