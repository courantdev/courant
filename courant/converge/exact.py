"""Exact window sums (TRD CONV-2, replacing M1-D-5's blocked reduction).

Every reduction over more than two floats in this package is `math.fsum`, which returns the
correctly rounded value of the exact sum, so its result depends only on the multiset of
addends and never on their order.

The stopping-point search needs the sum over thousands of overlapping windows. M1 proposed
caching one fsum per block of 256 and reducing each window over its head elements, cached
blocks and tail elements. That is deterministic, but not equal to fsum over the window,
because each block sum is itself rounded, and it costs O(w/256 + 512) per window.

This module holds every prefix sum exactly instead, as Shewchuk's non-overlapping partials
(the representation `math.fsum` uses internally). The sum over `[a, b)` is then
`fsum(P_b + (-P_a))`, the correctly rounded value of an exact difference. That is equal,
bit for bit, to `fsum(values[a:b])`, and costs O(p) per window, where p is the partials
count, typically 1 to 4. The test suite asserts the equality on random and adversarial
arrays.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence

SCHEME = "exact-prefix-partials"


def _grow(partials: Sequence[float], x: float) -> list[float]:
    """`partials` plus `x`, exactly, as non-overlapping partials (Shewchuk 1997)."""
    out: list[float] = []
    for y in partials:
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        lo = y - (hi - x)
        if lo:
            out.append(lo)
        x = hi
    out.append(x)
    return out


class ExactSums:
    """Correctly rounded sums over any window of one finite addend array."""

    __slots__ = ("_n", "_flat", "_start")

    def __init__(self, addends: Sequence[float]) -> None:
        flat = array("d")
        start = array("q", [0])
        partials: list[float] = []
        for x in addends:
            if not math.isfinite(x):
                raise ValueError("non-finite addend: filter before summing (CONV-9)")
            partials = _grow(partials, float(x))
            flat.extend(partials)
            start.append(len(flat))
        self._n = len(start) - 1
        self._flat = flat
        self._start = start

    def __len__(self) -> int:
        return self._n

    def _prefix(self, k: int) -> array:
        if k == 0:
            return array("d")
        return self._flat[self._start[k - 1]:self._start[k]]

    def window(self, a: int, b: int) -> float:
        """fsum(addends[a:b]), exactly, in O(partials)."""
        if a < 0 or b > self._n:
            raise IndexError(f"window [{a}, {b}) outside 0..{self._n}")
        if b <= a:
            return 0.0
        parts = list(self._prefix(b))
        parts.extend(-v for v in self._prefix(a))
        return math.fsum(parts)

    def total(self) -> float:
        return self.window(0, self._n)


def mean(values: Sequence[float]) -> float:
    """The fsum mean of a non-empty sequence."""
    return math.fsum(values) / len(values)


def variance(values: Sequence[float], ddof: int = 1) -> float:
    """Two-pass variance by fsum: the mean first, then the squared deviations."""
    n = len(values)
    if n - ddof <= 0:
        return 0.0
    m = mean(values)
    return math.fsum((v - m) * (v - m) for v in values) / (n - ddof)


def icbrt(n: int) -> int:
    """The integer cube root, floor(n ** (1/3)), by binary search (M1-D-3)."""
    if n < 0:
        raise ValueError("negative")
    lo, hi = 0, 1
    while hi * hi * hi <= n:
        hi = hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid * mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return lo
