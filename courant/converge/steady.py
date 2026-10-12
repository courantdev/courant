"""The steady branch: fixed point, limit cycle, residual orders (TRD CONV-6, M1 §5.1.5-5.1.7, 5.1.14).

A steady solver's answer is a fixed point. This module asks three questions of one monitor:
1. Has it stopped moving, and stayed stopped? That is the drift criterion over trailing
   windows.
2. From where? That is the stopping-point search.
3. Is it in a limit cycle that a trailing-window slope would call settled? That is the
   detrended autocorrelation test.

Residuals get their own, weaker questions. They describe the iteration, not the answer.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass

from courant.converge.exact import ExactSums, mean

WINDOW_FLOOR = 200          # M1-F-1: below this, no window rule applies
DRIFT_FACTOR = 0.2          # M1-F-3: the drift across a window must be < 0.2 * tol
ACF_THRESHOLD = 0.5         # M1-U-2
ITERATION_LAG_MAX = 3       # CONV-6c: periods of 2 or 3 iterations are the iteration, not the flow
ACF_EXACT_MAX = 2048        # CONV-6b: longer windows use the two-level autocorrelation
COARSE_DIVISOR = 64         # M1-X-2: coarse stride is w // 64
ORDERS_CAP = 30             # M1-G-1
PLATEAU_DRIFT = 0.2         # M1-P-2


def window_width(n: int) -> int:
    """w = min(n, max(200, ceil(n / 10)))."""
    return min(n, max(WINDOW_FLOOR, -(-n // 10)))


class Drift:
    """Absolute drift |slope| * (w - 1) of trailing windows of width `w` (M1-F-2, CONV-6a).

    The values are centred on the final value, `y' = y - c`, so the cancellation in the
    closed-form slope is smallest where the decision is made. A window-local abscissa cannot
    be cached, so `S_xy` uses the global-index sum `S_iy` and the exact identity
    `S_xy = S_iy - a * S_y`. Both sums are exact window sums (`exact.py`).
    """

    def __init__(self, values: Sequence[float], w: int) -> None:
        n = len(values)
        if w < 2 or w > n:
            raise ValueError(f"window {w} invalid for {n} samples")
        c = values[n - 1]
        centred = array("d", (v - c for v in values))
        self.n = n
        self.w = w
        self._y = ExactSums(centred)
        self._iy = ExactSums(array("d", (i * centred[i] for i in range(n))))
        self._sx = w * (w - 1) // 2
        sxx = (w - 1) * w * (2 * w - 1) // 6
        self._den = w * sxx - self._sx * self._sx
        self.evaluations = 0

    def slope(self, e: int) -> float:
        """OLS slope per sample over the window ending at index `e` (inclusive)."""
        a = e - self.w + 1
        sy = self._y.window(a, e + 1)
        sxy = self._iy.window(a, e + 1) - a * sy
        return (self.w * sxy - self._sx * sy) / self._den

    def drift(self, e: int) -> float:
        self.evaluations = self.evaluations + 1
        return abs(self.slope(e)) * (self.w - 1)


def unsettled(drift: float, tau: float) -> bool:
    """A window fails the criterion. An exactly zero drift always passes, so a constant
    tail with no usable scale (tau = 0) is still settled."""
    return drift > 0.0 and drift >= tau


@dataclass(frozen=True, slots=True)
class Search:
    index: int | None      # the stopping point, or None when the last window is unsettled
    evaluations: int
    decimated: bool
    resolution: int        # in samples


def stopping_index(drift: Drift, tau: float, budget: int) -> Search:
    """The earliest index from which every trailing window is settled (M1-F-5, M1-X-2).

    A coarse backward pass at stride w // 64 finds the last failing grid point. The
    exhaustive pass then checks every skipped window after it, from the end backwards,
    so the first failure it meets is the true maximum. If the exhaustive pass would exceed
    `budget`, the coarse answer is reported, rounded up to the next grid point, which
    understates the saving rather than overstating it.
    """
    n, w = drift.n, drift.w
    last, first = n - 1, w - 1
    if unsettled(drift.drift(last), tau):
        return Search(None, drift.evaluations, False, 1)
    sigma = max(1, w // COARSE_DIVISOR)
    fail = None
    e = last - sigma
    while e >= first:
        if unsettled(drift.drift(e), tau):
            fail = e
            break
        e = e - sigma
    lo = fail if fail is not None else first - 1
    gaps = (last - lo) - 1 - (last - lo - 1) // sigma
    if drift.evaluations + gaps > budget:
        if fail is not None:
            index = min(fail + sigma, last)
        else:
            index = last - ((last - first) // sigma) * sigma
        return Search(index, drift.evaluations, True, sigma)
    for e in range(last - 1, lo, -1):
        if (last - e) % sigma == 0:
            continue                      # a coarse point, already known to pass
        if unsettled(drift.drift(e), tau):
            return Search(e + 1, drift.evaluations, False, 1)
    return Search(lo + 1, drift.evaluations, False, 1)


# ------------------------------------------------------------------ the limit-cycle test
@dataclass(frozen=True, slots=True)
class Oscillation:
    tested: bool
    lag: int | None = None
    acf: float | None = None
    amplitude: float | None = None
    n_periods: int | None = None
    fires: bool = False
    cycle_mean: float | None = None
    reason: str | None = None


def detrend(values: Sequence[float], slope: float) -> list[float]:
    """The trailing window minus its fitted line (M1-U-1)."""
    w = len(values)
    m = mean(values)
    intercept = m - slope * ((w - 1) / 2)
    return [values[t] - (intercept + slope * t) for t in range(w)]


def _acf(z: Sequence[float], lmax: int) -> list[float] | None:
    """Biased autocorrelation rho(0..lmax+1), divisor w at every lag (M1-U-1)."""
    w = len(z)
    zbar = mean(z)
    d = [v - zbar for v in z]
    g0 = math.fsum(v * v for v in d) / w
    if g0 <= 0.0:
        return None
    top = min(lmax + 1, w - 1)
    rho = [1.0]
    rho.extend(math.fsum(d[t] * d[t + lag] for t in range(w - lag)) / w / g0
               for lag in range(1, top + 1))
    return rho


def _best_period(rho: list[float], lmax: int) -> tuple[int, float] | None:
    best = None
    for lag in range(2, min(lmax, len(rho) - 2) + 1):
        r = rho[lag]
        if r > ACF_THRESHOLD and r > rho[lag - 1] and r >= rho[lag + 1]:
            if best is None or r > best[1]:
                best = (lag, r)
    return best


def _block_means(z: Sequence[float], size: int) -> list[float]:
    k = len(z) // size
    start = len(z) - k * size
    return [mean(z[start + j * size:start + (j + 1) * size]) for j in range(k)]


def oscillation(z: Sequence[float], limit: float) -> Oscillation:
    """The limit-cycle test on a detrended window `z` (M1-U-2, U-3).

    `limit` is the amplitude a cycle must exceed to matter: tol for an absolute monitor,
    tol * R for a relative one. When the whole window's peak-to-peak is already below it,
    no cycle in it can fire, and the autocorrelation is not computed (CONV-6b).
    """
    w = len(z)
    if w < 16:
        return Oscillation(False, reason="window too short")
    if max(z) - min(z) <= limit:
        return Oscillation(False, reason="peak-to-peak below tolerance")
    candidates: list[tuple[int, float]] = []
    exact = z[w - ACF_EXACT_MAX:] if w > ACF_EXACT_MAX else z
    lmax = len(exact) // 4
    rho = _acf(exact, lmax)
    if rho is not None:
        found = _best_period(rho, lmax)
        if found:
            candidates.append(found)
    if w > ACF_EXACT_MAX:
        size = -(-w // ACF_EXACT_MAX)
        coarse = _block_means(z, size)
        lmax_c = len(coarse) // 4
        rho_c = _acf(coarse, lmax_c)
        if rho_c is not None:
            found = _best_period(rho_c, lmax_c)
            if found and found[0] * size > ACF_EXACT_MAX // 4:
                candidates.append((found[0] * size, found[1]))
    if not candidates:
        return Oscillation(True, fires=False, reason="no autocorrelation peak above 0.5")
    lag, r = sorted(candidates, key=lambda c: (-c[1], c[0]))[0]
    periods = w // lag
    if periods < 1:
        return Oscillation(True, fires=False, reason="period longer than the window")
    tail = z[w - periods * lag:]
    amplitude = max(tail) - min(tail)
    return Oscillation(True, lag=lag, acf=r, amplitude=amplitude, n_periods=periods,
                       fires=amplitude > limit)


# ------------------------------------------------------------------ residuals
@dataclass(frozen=True, slots=True)
class ResidualSummary:
    field: str
    n: int
    r_ref: float | None
    r_final: float | None
    r_min: float | None
    r_min_index: int | None
    ratio: float | None
    orders_dropped_floor: int | None
    orders_capped: bool
    plateaued: bool | None       # None when the series is too short for a window
    drift_rel: float | None


def orders_dropped(r_ref: float, r_final: float) -> tuple[int, bool]:
    """max{j : r_final * 10**j <= r_ref}, by repeated multiplication, no logarithm."""
    j = 0
    v = r_final
    while j < ORDERS_CAP and v * 10.0 <= r_ref:
        v = v * 10.0
        j = j + 1
    return j, j >= ORDERS_CAP


def residual_summary(field: str, values: Sequence[float]) -> ResidualSummary:
    """Orders dropped (Celik), the minimum, and the plateau test (M1-P-1, P-2, G-1)."""
    finite = [v for v in values if math.isfinite(v) and v > 0.0]
    n = len(finite)
    if n < 2:
        return ResidualSummary(field, n, None, finite[-1] if finite else None, None, None,
                               None, None, False, None, None)
    r_ref, r_final = finite[1], finite[-1]
    r_min = min(finite)
    r_min_index = finite.index(r_min)
    orders, capped = orders_dropped(r_ref, r_final)
    plateaued, drift_rel = None, None
    if n >= WINDOW_FLOOR:
        w = window_width(n)
        d = Drift(finite, w)
        drift_rel = d.drift(n - 1) / r_final
        plateaued = drift_rel < PLATEAU_DRIFT
    return ResidualSummary(field, n, r_ref, r_final, r_min, r_min_index, r_ref / r_final,
                           orders, capped, plateaued, drift_rel)
