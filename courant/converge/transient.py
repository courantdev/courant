"""The transient branch: is a time-average known well enough? (TRD CONV-7, M1 §5.1.8-5.1.10)

A time-averaged quantity has a mean, the mean has a confidence interval, and the run has
converged when the interval is narrow enough. The series is autocorrelated, so the naive
standard error can be wrong by a factor of ten or a hundred. The procedure, per monitor:

1. Remove the initial transient by MSER-5 (White 1997; White, Cobb & Spratt 2000).
2. Grow non-overlapping batches until the batch means are nearly independent (Fishman's
   doubling).
3. Test what is left for stationarity, two ways (Geweke 1992; von Neumann on batch means).
4. Form a Student-t interval on k - 1 degrees of freedom and compare its half-width with the
   tolerance.

Stated deviations from M1, each measured before it was made (RRD RM-15, evidence log):
- **CONV-7b. The batch size is fixed, not searched.** It is `b = floor(sqrt(n))` (Flegal &
  Jones 2010), with a first-order AR(1) correction `(1 + rho1) / (1 - rho1)` on the variance
  of the mean when the batch means are still positively correlated. If `rho1 >= 0.5`, the
  run is too short for the correlation present. M1's sequential doubling, accepting at
  `|rho1| < 0.10`, covered the true mean of AR(1) series in 0.919 (rho 0.5) and 0.899
  (rho 0.9) of 200 fixtures at n = 10,000. This rule covers 0.960, 0.945 and 0.965 at rho
  0.5, 0.9 and 0.97. The doubling search stopped on whichever batch size happened to show
  low sample correlation, which biases the variance down. A batch size fixed in advance
  cannot do that, and is one candidate rather than a sequence, which is the stronger form
  of M1-V-3's reproducibility argument.
- **CONV-7c. The stopping point uses prefix semantics,** the run as it was at step e. It is
  found by a coarse pass, a bisection and a verification sweep within a fixed evaluation
  count. The resolution is reported.
- **CONV-7e. MSER-5 searches the first half** for its minimum (d <= floor(nb/2)). The tail
  guard fires when that minimum sits at the half's upper boundary. Over the full domain, the
  statistic is unstable on its last few batches, and the minimum landed at the very end on
  9 of 80 clean series with a short transient (tau 50 and 200). It does so on 0 of 80 in the
  first half. A transient the guard misses is still caught downstream, by the stationarity
  tests on what remains.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from courant.converge.exact import ExactSums, icbrt, mean, variance
from courant.converge.tables import t975

MSER_BATCH = 5
BATCH_MIN = 20
RHO1_TOO_CORRELATED = 0.5
Z_CRIT = 1.96
GEWEKE_A = 0.10
GEWEKE_B = 0.50
LOW_ESS = 50.0
SKEW_LIMIT = 2.0
SPECTRAL_RATIO = 2.0
SEARCH_COARSE = 16
SEARCH_VERIFY = 8
SEARCH_ROUNDS = 2

METHOD_MSER = "MSER-5 (White 1997; White, Cobb & Spratt 2000)"
METHOD_BATCH = ("non-overlapping batch means, batch size floor(sqrt(n)) (Flegal & Jones 2010), "
                "first-order AR(1) variance correction; Student-t interval on k-1 degrees of freedom")
METHOD_STATIONARITY = ("Geweke (1992) with a Bartlett lag-truncated S(0); von Neumann ratio "
                       "on batch means with the AR(1) mean correction (approximation)")


# ------------------------------------------------------------------ MSER-5
@dataclass(frozen=True, slots=True)
class Mser:
    insufficient: bool
    n_batches: int
    tail_dropped: int
    d_star: int | None = None
    recomputations: int = 0
    in_tail: bool = False

    @property
    def truncate_at(self) -> int:
        return MSER_BATCH * (self.d_star or 0)


def mser5(y: Sequence[float]) -> Mser:
    """The truncation point d* minimising MSER over batch means of five (M1-T-1 to T-6, CONV-7e)."""
    n = len(y)
    nb = n // MSER_BATCH
    dropped = n - MSER_BATCH * nb
    if nb < 2 * BATCH_MIN:
        return Mser(True, nb, dropped)
    z = [math.fsum(y[MSER_BATCH * j:MSER_BATCH * (j + 1)]) / MSER_BATCH for j in range(nb)]
    c = z[-1]
    zc = [v - c for v in z]                 # MSER is shift-invariant; centring limits cancellation
    s_sum = ExactSums(zc)
    q_sum = ExactSums([v * v for v in zc])
    best_d, best = 0, math.inf
    recomputed = 0
    half = nb // 2
    for d in range(half + 1):
        m = nb - d
        s = s_sum.window(d, nb)
        ss = q_sum.window(d, nb) - s * s / m
        if ss < 0.0:
            mu = s / m
            ss = math.fsum((zc[i] - mu) * (zc[i] - mu) for i in range(d, nb))
            recomputed = recomputed + 1
        value = ss / (m * m)
        if value < best:
            best_d, best = d, value
    return Mser(False, nb, dropped, best_d, recomputed, best_d >= half - 1)


# ------------------------------------------------------------------ batch means
@dataclass(frozen=True, slots=True)
class Batches:
    accepted: bool
    n: int                       # the truncated length these were formed on
    size: int                    # batch size m
    count: int                   # batch count k
    rho1: float
    means: tuple[float, ...] = ()
    mean: float | None = None
    half_width: float | None = None
    se: float | None = None
    ess: float | None = None
    skew: float | None = None
    dropped: int = 0


def _rho1(dev: Sequence[float]) -> float:
    den = math.fsum(d * d for d in dev)
    if den <= 0.0:
        return 0.0
    return math.fsum(dev[j] * dev[j + 1] for j in range(len(dev) - 1)) / den


def batch_means(y: Sequence[float]) -> Batches:
    """Batches of floor(sqrt(n)), the AR(1)-corrected t-interval, and the ESS (CONV-7b)."""
    n = len(y)
    m = max(1, math.isqrt(n))
    k = n // m
    if k < BATCH_MIN:
        return Batches(False, n, m, k, math.nan)
    sums = ExactSums(y)
    b = [sums.window(j * m, (j + 1) * m) / m for j in range(k)]
    bbar = mean(b)
    dev = [v - bbar for v in b]
    rho = _rho1(dev)
    if rho >= RHO1_TOO_CORRELATED:
        return Batches(False, n, m, k, rho, tuple(b), bbar)
    ss = math.fsum(d * d for d in dev)
    s2b = ss / (k - 1)
    inflation = (1.0 + rho) / (1.0 - rho) if rho > 0.0 else 1.0
    se = math.sqrt(s2b * inflation / k)
    half = t975(k - 1) * se
    s2y = variance(y)
    ess = (n * s2y / (m * s2b * inflation)) if s2b > 0.0 else None
    m2 = ss / k
    skew = (math.fsum(d * d * d for d in dev) / k) / (m2 * math.sqrt(m2)) if m2 > 0.0 else 0.0
    return Batches(True, n, m, k, rho, tuple(b), bbar, half, se, ess, skew, n - k * m)


# ------------------------------------------------------------------ stationarity
def s0_bartlett(x: Sequence[float]) -> float:
    """Spectral density at zero, Bartlett weights 1 - l/(m+1), m = floor(n^(1/3)) (M1-Z-2)."""
    n = len(x)
    if n < 2:
        return 0.0
    lags = icbrt(n)
    xbar = mean(x)
    d = [v - xbar for v in x]
    g = [math.fsum(d[t] * d[t + lag] for t in range(n - lag)) / n for lag in range(lags + 1)]
    return g[0] + 2.0 * math.fsum((1.0 - lag / (lags + 1)) * g[lag] for lag in range(1, lags + 1))


@dataclass(frozen=True, slots=True)
class Stationarity:
    stationary: bool
    geweke_z: float
    vn_z: float
    vn: float
    n_a: int
    n_b: int
    which_failed: str | None
    oscillatory_batches: bool


def stationarity(y: Sequence[float], batches: Batches) -> Stationarity:
    """Geweke's z on the truncated series and von Neumann on its batch means (M1-Z-1 to Z-4)."""
    n = len(y)
    n_a, n_b = int(GEWEKE_A * n), int(GEWEKE_B * n)
    a, b = y[:n_a], y[n - n_b:]
    den = math.sqrt(s0_bartlett(a) / n_a + s0_bartlett(b) / n_b) if n_a > 1 and n_b > 1 else 0.0
    z = (mean(a) - mean(b)) / den if den > 0.0 else 0.0
    bm = batches.means
    k = len(bm)
    bbar = mean(bm)
    vden = math.fsum((v - bbar) * (v - bbar) for v in bm)
    vn = (math.fsum((bm[i] - bm[i + 1]) * (bm[i] - bm[i + 1]) for i in range(k - 1)) / vden
          if vden > 0.0 else 2.0)
    mu = 2.0 * (1.0 - batches.rho1)
    sigma = math.sqrt(4.0 * (k - 2) / ((k + 1) * (k - 1)))
    zvn = (vn - mu) / sigma
    g_ok, v_ok = abs(z) < Z_CRIT, zvn >= -Z_CRIT
    which = None
    if not g_ok and not v_ok:
        which = "Both Geweke's test and the von Neumann test failed."
    elif not g_ok:
        which = "Geweke's test failed; the von Neumann test passed."
    elif not v_ok:
        which = "The von Neumann test failed; Geweke's test passed."
    return Stationarity(g_ok and v_ok, z, zvn, vn, n_a, n_b, which, zvn > Z_CRIT)


# ------------------------------------------------------------------ one assessment
@dataclass(frozen=True, slots=True)
class TransientResult:
    verdict: str        # CONVERGED | INSUFFICIENT_DATA | MONOTONE_DRIFT | NOT_STATIONARY | CI_TOO_WIDE
    stage: str
    mser: Mser
    batches: Batches | None = None
    stationarity: Stationarity | None = None
    within: bool = False


def assess_prefix(y: Sequence[float], limit_abs: float | None, tol_rel: float | None,
                  scale: float | None, drift_settled=None) -> TransientResult:
    """Steps 2 to 6 of M1-Z-6 on one series.

    `limit_abs` is the absolute tolerance, or None for a relative criterion against `scale`.
    `drift_settled` is a callable answering whether the last window has stopped moving. It is
    consulted only when MSER's tail guard fires, and separates MONOTONE_DRIFT from
    TRUNCATION_IN_TAIL (M1-T-7).
    """
    ms = mser5(y)
    if ms.insufficient:
        return TransientResult("INSUFFICIENT_DATA", "mser", ms)
    if ms.in_tail:
        moving = drift_settled is not None and drift_settled() is False
        verdict = "MONOTONE_DRIFT" if moving else "INSUFFICIENT_DATA"
        return TransientResult(verdict, "mser_tail", ms)
    yt = y[ms.truncate_at:MSER_BATCH * ms.n_batches]
    bt = batch_means(yt)
    if not bt.accepted:
        return TransientResult("INSUFFICIENT_DATA", "batching", ms, bt)
    st = stationarity(yt, bt)
    if not st.stationary:
        return TransientResult("NOT_STATIONARY", "stationarity", ms, bt, st)
    if limit_abs is not None:
        within = bt.half_width <= limit_abs
    elif scale:
        within = bt.half_width / scale <= tol_rel
    else:
        within = bt.half_width == 0.0
    return TransientResult("CONVERGED" if within else "CI_TOO_WIDE", "criterion", ms, bt, st,
                           within)


@dataclass(frozen=True, slots=True)
class TransientSearch:
    index: int | None       # last sample of the earliest converged prefix
    evaluations: int
    resolution: int         # samples
    non_monotone_rounds: int


def stopping_prefix(y: Sequence[float], ok) -> TransientSearch:
    """The earliest prefix from which every longer prefix passes `ok` (CONV-7c).

    Candidates are prefix lengths on MSER's batch grid. The search is a coarse backward pass
    at a sixteenth of the run, a bisection inside the failing bracket, and a verification
    sweep of eight candidates repeated at most twice. Every unevaluated candidate can only
    move the answer later, so the reported point is conservative.
    """
    nb = len(y) // MSER_BATCH
    lo_j = 2 * BATCH_MIN
    evals = 0
    cache: dict[int, bool] = {}

    def passes(j: int) -> bool:
        nonlocal evals
        if j not in cache:
            evals = evals + 1
            cache[j] = ok(y[:MSER_BATCH * j])
        return cache[j]

    if nb < lo_j or not passes(nb):
        return TransientSearch(None, evals, MSER_BATCH, 0)
    sigma = max(1, (nb - lo_j) // SEARCH_COARSE)
    fail = None
    j = nb - sigma
    while j >= lo_j:
        if not passes(j):
            fail = j
            break
        j = j - sigma
    if fail is None:
        best, resolution = (j + sigma if j < lo_j else lo_j), sigma * MSER_BATCH
        best = max(best, lo_j)
    else:
        lo, hi = fail, min(fail + sigma, nb)    # lo fails, hi passes
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if passes(mid):
                hi = mid
            else:
                lo = mid
        best, resolution = hi, MSER_BATCH
    rounds = 0
    while rounds < SEARCH_ROUNDS:
        span = nb - best
        probes = sorted({best + (span * i) // SEARCH_VERIFY for i in range(1, SEARCH_VERIFY)})
        failing = [p for p in probes if not passes(p)]
        if not failing:
            break
        best = max(failing) + 1
        rounds = rounds + 1
    return TransientSearch(MSER_BATCH * best - 1, evals, resolution, rounds)
