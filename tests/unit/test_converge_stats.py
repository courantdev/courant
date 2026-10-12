"""The transient branch against theory, at laptop scale (TRD TST-10, M1 §5.1.18).

M1 asks for 1,000 fixtures per case at n = 100,000. That is a cloud job. These tests use
100 fixtures at n = 10,000, so each floor sits three binomial standard deviations below
nominal at that count: a pass is evidence, and a failure is a signal rather than noise.
"""

from __future__ import annotations

import math
import random

import pytest

from courant.converge import transient as tr
from courant.converge.assess import Clock, MonitorSeries, Options, RunMeta, assess
from courant.converge.exact import mean

N = 10_000
FIXTURES = 100


def ar1(n: int, rho: float, mu: float = 5.0, sigma: float = 1.0, seed: int = 0) -> list[float]:
    """Stationary from the first sample: x0 is drawn from the marginal."""
    rng = random.Random(seed)
    s = math.sqrt(1.0 - rho * rho)
    x = rng.gauss(0.0, 1.0)
    out = []
    for _ in range(n):
        out.append(mu + sigma * x)
        x = rho * x + s * rng.gauss(0.0, 1.0)
    return out


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.97])
def test_batch_means_intervals_cover_the_true_mean(rho):
    covered = 0
    for seed in range(FIXTURES):
        b = tr.batch_means(ar1(N, rho, seed=1000 * int(rho * 100) + seed))
        assert b.accepted
        covered = covered + (abs(b.mean - 5.0) <= b.half_width)
    # p = 0.95 over 100 fixtures: sd 0.0218, three sd below nominal is 0.885
    assert covered >= 88, f"coverage {covered} of {FIXTURES} at rho={rho}"


def test_effective_sample_size_matches_theory():
    rho = 0.5
    ess = sorted(tr.batch_means(ar1(N, rho, seed=s)).ess for s in range(40))
    median = ess[len(ess) // 2]
    theory = N * (1 - rho) / (1 + rho)
    assert abs(median - theory) / theory < 0.25, (median, theory)


def test_geweke_false_rejection_rate_on_stationary_series():
    rejections = 0
    for seed in range(FIXTURES):
        y = ar1(N, 0.5, seed=5000 + seed)
        b = tr.batch_means(y)
        rejections = rejections + (abs(tr.stationarity(y, b).geweke_z) >= tr.Z_CRIT)
    assert rejections <= 12, f"{rejections} of {FIXTURES} rejected (nominal 5)"


def test_geweke_rejects_a_mean_shift():
    rejections = 0
    for seed in range(40):
        y = ar1(N, 0.5, seed=7000 + seed)
        y = [v + (0.25 if i > N // 2 else 0.0) for i, v in enumerate(y)]
        b = tr.batch_means(y)
        rejections = rejections + (not tr.stationarity(y, b).stationary)
    assert rejections >= 36


def test_von_neumann_on_independent_batch_means_centres_on_two():
    rng = random.Random(11)
    vns = []
    for _ in range(300):
        bm = tuple(rng.gauss(0, 1) for _ in range(50))
        b = tr.Batches(True, 50, 1, 50, 0.0, bm, mean(bm))
        vns.append(tr.stationarity(list(bm), b).vn)
    assert abs(mean(vns) - 2.0) < 0.05


@pytest.mark.parametrize("tau", [50, 200])
def test_mser5_recovers_a_constructed_transient(tau):
    hits = 0
    for seed in range(20):
        rng = random.Random(seed + tau)
        y = [1.0 + 10.0 * math.exp(-i / tau) + rng.gauss(0, 1) for i in range(N // 2)]
        ms = tr.mser5(y)
        hits = hits + (tau <= ms.truncate_at <= 6 * tau)
    assert hits >= 18


def test_mser_suffix_form_agrees_with_two_pass():
    rng = random.Random(4)
    y = [100.0 + 5.0 * math.exp(-i / 80) + rng.gauss(0, 0.1) for i in range(3000)]
    fast = tr.mser5(y)
    nb = len(y) // 5
    z = [math.fsum(y[5 * j:5 * j + 5]) / 5 for j in range(nb)]
    best = min(range(nb - 1), key=lambda d: (math.fsum((v - mean(z[d:])) ** 2 for v in z[d:])
                                            / (nb - d) ** 2, d))
    assert fast.d_star == best


# ------------------------------------------------------------------ end to end
def _mon(values):
    return MonitorSeries(id="avg/volFieldValue:volAverage(p)", column="volAverage(p)",
                         source="postProcessing/avg/0/volFieldValue.dat",
                         times=[0.001 * (i + 1) for i in range(len(values))], values=values)


def _meta():
    return RunMeta(regime="transient", regime_basis="config", ranks=8, ranks_basis="log_nprocs")


def test_a_declared_time_average_converges_with_a_stopping_point():
    y = ar1(8000, 0.5, mu=1.0, sigma=0.01, seed=21)
    conv = assess([_mon(y)], {}, Clock(), _meta(), Options(transient_average=True))
    assert conv.status == "PASS", [f.code for f in conv.findings]
    row = conv.data["monitors"]["avg/volFieldValue:volAverage(p)"]
    assert row["verdict"] == "CONVERGED" and row["stop_index"] < 7999
    assert row["delta_ok"] is True
    assert conv.data["waste"]["wasted_fraction"] > 0.5


def test_an_undeclared_transient_is_reported_but_not_graded():
    y = ar1(8000, 0.5, mu=1.0, sigma=0.01, seed=21)
    conv = assess([_mon(y)], {}, Clock(), _meta())
    assert conv.status == "NOT_ASSESSABLE"
    assert conv.counted is False
    assert any(f.code == "CONV.TRANSIENT_NO_INTENT" for f in conv.findings)
    assert conv.data["monitors"]["avg/volFieldValue:volAverage(p)"]["verdict"] == "CONVERGED"


def test_a_trend_is_not_stationary():
    y = [v + 2e-5 * i for i, v in enumerate(ar1(8000, 0.5, mu=1.0, sigma=0.01, seed=3))]
    conv = assess([_mon(y)], {}, Clock(), _meta(), Options(transient_average=True))
    assert conv.status == "FAIL"
    assert any(f.code in ("CONV.NOT_STATIONARY", "CONV.MONOTONE_DRIFT") for f in conv.findings)


def test_noise_against_a_tight_tolerance_is_too_wide():
    y = ar1(4000, 0.9, mu=1.0, sigma=0.5, seed=8)
    conv = assess([_mon(y)], {}, Clock(), _meta(),
                  Options(transient_average=True, tol_abs={"volAverage(p)": 1e-4}))
    assert conv.status == "FAIL"
    assert any(f.code == "CONV.CI_TOO_WIDE" for f in conv.findings)


def test_a_transient_that_has_not_finished_is_never_a_pass():
    for seed in range(10):
        rng = random.Random(seed)
        n = 5000
        y = [1.0 + 10.0 * math.exp(-i / (0.7 * n / 3)) + rng.gauss(0, 1) for i in range(n)]
        conv = assess([_mon(y)], {}, Clock(), _meta(), Options(transient_average=True))
        assert conv.status != "PASS", seed


def test_mser_minimum_is_searched_in_the_first_half():
    rng = random.Random(9)
    y = [1.0 + 10.0 * math.exp(-i / 50) + rng.gauss(0, 1) for i in range(5000)]
    ms = tr.mser5(y)
    assert ms.d_star <= ms.n_batches // 2 and not ms.in_tail


def test_a_short_transient_is_insufficient():
    conv = assess([_mon(ar1(150, 0.5))], {}, Clock(), _meta(), Options(transient_average=True))
    assert conv.status == "WARN"
    assert any(f.code == "CONV.INSUFFICIENT_DATA" for f in conv.findings)
