"""Convergence arithmetic: purity, exactness, the steady branch, degenerate rows (TRD TST-10)."""

from __future__ import annotations

import ast
import hashlib
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from courant.converge import sentences
from courant.converge.assess import Clock, MonitorSeries, Options, RunMeta, assess, json_safe
from courant.converge.exact import ExactSums, icbrt
from courant.converge.scale import is_coefficient, tolerance
from courant.converge.steady import (
    detrend,
    oscillation,
    orders_dropped,
    window_width,
)
from courant.converge.tables import NORMAL_975, t975

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "courant" / "converge"
PURE = ("exact.py", "tables.py", "scale.py", "steady.py", "transient.py", "assess.py", "sentences.py")
#: Pinned so the shipped t table cannot drift (M1-D-6). Regenerate with scripts/gen_t_table.py.
T_TABLE_SHA256 = "f1bbe77e46a2ff427b0a201ee9d101fb7982e2c690cabede5b472c8750af53c5"


def _mon(values, times=None, mid="probe/0/p:p", column="p", **kw) -> MonitorSeries:
    times = times or [float(i + 1) for i in range(len(values))]
    return MonitorSeries(id=mid, column=column, source="postProcessing/x.dat", times=times,
                         values=values, **kw)


def _clock(n, dt_wall=0.5):
    return Clock(times=[float(i + 1) for i in range(n)],
                 clock_s=[float(int((i + 1) * dt_wall)) for i in range(n)],
                 exec_s=[(i + 1) * dt_wall for i in range(n)])


def _steady_meta(**kw):
    return RunMeta(regime="steady", regime_basis="config", ranks=4, ranks_basis="log_nprocs", **kw)


def _approach(n=2000, tau=100.0, base=1.0, amp=1.0, noise=0.0, seed=1):
    rng = random.Random(seed)
    return [base + amp * math.exp(-i / tau) + noise * rng.gauss(0.0, 1.0) for i in range(n)]


# ------------------------------------------------------------------ purity (CONV-2)
FORBIDDEN_IMPORTS = {"os", "pathlib", "io", "subprocess", "time", "datetime", "random",
                     "numpy", "scipy", "statistics"}
FORBIDDEN_MATH = {"log", "log10", "log2", "log1p", "exp", "expm1", "pow"}


@pytest.mark.parametrize("name", PURE)
def test_the_arithmetic_is_pure(name):
    tree = ast.parse((PKG / name).read_text())
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [a.name for a in node.names if a.name.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                problems.append(node.module)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "sum":
                problems.append(f"sum() at line {node.lineno}")
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id == "math" and f.attr in FORBIDDEN_MATH:
                problems.append(f"math.{f.attr} at line {node.lineno}")
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
            problems.append(f"augmented accumulation at line {node.lineno}")
    assert not problems, f"{name}: {problems}"


# ------------------------------------------------------------------ exactness
def test_window_sums_equal_fsum_bit_for_bit():
    rng = random.Random(7)
    xs = [rng.uniform(-1, 1) * 10.0 ** rng.randint(-12, 12) for _ in range(4000)]
    xs[100:110] = [1e16, 1.0, -1e16, 3.0, 1e-16, -1.0, 2.0 ** 60, -(2.0 ** 60), 0.1, 0.2]
    es = ExactSums(xs)
    for _ in range(3000):
        a, b = sorted(rng.sample(range(len(xs) + 1), 2))
        assert es.window(a, b) == math.fsum(xs[a:b])
    assert es.total() == math.fsum(xs)
    assert es.window(5, 5) == 0.0


def test_exact_sums_refuse_non_finite():
    with pytest.raises(ValueError):
        ExactSums([1.0, math.nan])


def test_integer_cube_root():
    for n in range(0, 20000):
        r = icbrt(n)
        assert r ** 3 <= n < (r + 1) ** 3
    for n in (10 ** 6, 10 ** 9 - 1, 10 ** 9, 2 ** 63, 10 ** 30 + 7):
        r = icbrt(n)
        assert r ** 3 <= n < (r + 1) ** 3


def test_the_t_table_is_pinned_and_correct():
    digest = hashlib.sha256((PKG / "tables.py").read_bytes()).hexdigest()
    assert digest == T_TABLE_SHA256, "tables.py changed: regenerate deliberately and re-pin"
    # M1-D-6's spot values, to the precision a reviewer checks by hand.
    assert abs(t975(19) - 2.093024054408263) < 1e-12
    assert abs(t975(29) - 2.045229642132703) < 1e-12
    assert t975(1001) == NORMAL_975
    with pytest.raises(ValueError):
        t975(0)


# ------------------------------------------------------------------ tolerances
@pytest.mark.parametrize("col,coeff", [("Cd", True), ("Cl", True), ("CmPitch", True),
                                       ("Cd(f)", True), ("Cs", True), ("myCoeff", True),
                                       ("p", False), ("total_x", False), ("areaAverage(p)", False)])
def test_coefficients_are_judged_in_counts(col, coeff):
    assert is_coefficient(col) is coeff
    tol = tolerance("obj/f:" + col, col, {}, {})
    assert tol.branch == ("absolute" if coeff else "relative")
    assert tol.value == (1e-4 if coeff else 0.01)


def test_two_tolerances_for_one_monitor_is_an_error():
    with pytest.raises(ValueError):
        tolerance("obj/f:Cd", "Cd", {"Cd": 1e-4}, {"Cd": 0.01})


# ------------------------------------------------------------------ steady branch
def test_window_rule():
    assert window_width(250) == 200
    assert window_width(2000) == 200
    assert window_width(5001) == 501
    assert window_width(150) == 150


def test_a_settling_series_has_a_stopping_point_and_a_waste_figure():
    vals = _approach()
    conv = assess([_mon(vals)], {}, _clock(len(vals)), _steady_meta())
    assert conv.status == "PASS"
    d = conv.data
    stop = d["converged_at_step"]
    assert 600 < stop < 1100
    w = d["waste"]
    assert w["wasted_steps"] == 2000 - stop
    assert 0.4 < w["wasted_fraction"] < 0.7
    assert w["wasted_wall_clock_s"] is not None and w["wasted_core_hours"] is not None
    assert w["wasted_core_hours"] == pytest.approx(4 * w["wasted_wall_clock_s"] / 3600)
    assert conv.findings[0].code == "CONV.ANSWER_UNCHANGED"      # M1-A-2: the delta first
    mrow = d["monitors"]["probe/0/p:p"]
    assert mrow["delta_ok"] is True and mrow["delta"] <= 0.01 * mrow["scale"]["value"]


def test_the_stopping_point_is_prefix_stable():
    vals = _approach(n=3000)
    full = assess([_mon(vals)], {}, Clock(), _steady_meta()).data["converged_at_step"]
    for cut in (full + 300, full + 900, 2600):
        part = vals[:cut]
        # the window grows with n, so only cuts that keep w = 200 are comparable
        if window_width(cut) == window_width(3000):
            got = assess([_mon(part)], {}, Clock(), _steady_meta()).data["converged_at_step"]
            assert got == full


def test_a_drifting_series_is_not_converged_and_says_so_in_fixed_words():
    vals = [1.0 + 1e-4 * i for i in range(2000)]
    conv = assess([_mon(vals)], {}, Clock(), _steady_meta())
    assert conv.status == "FAIL"
    f = next(f for f in conv.findings if f.code == "CONV.NOT_CONVERGED")
    assert f.severity == "FAIL"
    assert f.message.startswith("This run did not reach a fixed point. probe/0/p:p was still moving")
    assert conv.data["waste"] is None and conv.data["converged_at_step"] is None


@pytest.mark.parametrize("period", [8, 25, 120])
def test_a_limit_cycle_is_unsteady_under_a_steady_solver(period):
    rng = random.Random(period)
    n = max(4000, 50 * period)       # the window (n/10) must hold four cycles (M1-U-2)
    vals = [2.0 + 0.05 * math.sin(2 * math.pi * i / period) + 1e-4 * rng.gauss(0, 1)
            for i in range(n)]
    conv = assess([_mon(vals)], {}, _clock(n), _steady_meta())
    assert conv.status == "FAIL"
    f = next(f for f in conv.findings if f.code == "CONV.UNSTEADY_STEADY")
    assert abs(f.value["lag"] - period) <= 1
    assert f.message.startswith("This run used a steady solver on a flow that is not steady.")
    assert conv.data["waste"] is None                       # M1-U-5: no saving on a limit cycle


def test_a_quantised_converged_monitor_is_not_a_limit_cycle():
    vals = [float(f"{v:.4g}") for v in _approach(n=3000, base=0.3, amp=0.1, tau=150)]
    conv = assess([_mon(vals, mid="fc/coefficient:Cd", column="Cd")], {}, Clock(), _steady_meta())
    assert not any(f.code == "CONV.UNSTEADY_STEADY" for f in conv.findings)
    assert conv.status == "PASS"


def test_oscillation_search_recovers_the_period_on_long_windows():
    rng = random.Random(3)
    z = [math.sin(2 * math.pi * i / 700) + 0.01 * rng.gauss(0, 1) for i in range(6000)]
    osc = oscillation(detrend(z, 0.0), 0.01)
    assert osc.fires and abs(osc.lag - 700) <= 700 // 10


def test_constant_series():
    conv = assess([_mon([0.5] * 400)], {}, Clock(), _steady_meta())
    assert conv.status == "PASS"
    assert any(f.code == "CONV.CONSTANT_SERIES" for f in conv.findings)


def test_a_short_series_is_insufficient_not_converged():
    conv = assess([_mon(_approach(n=150))], {}, Clock(), _steady_meta())
    assert conv.status == "WARN"
    f = next(f for f in conv.findings if f.code == "CONV.INSUFFICIENT_DATA")
    assert f.severity == "WARN" and f.value["n_required"] == 200
    assert f.message.startswith("There is not enough data to separate")


def test_a_few_early_nans_are_dropped_and_reported():
    vals = _approach()
    vals[10] = math.nan
    conv = assess([_mon(vals)], {}, Clock(), _steady_meta())
    f = next(f for f in conv.findings if f.code == "CONV.NON_FINITE")
    assert f.severity == "WARN" and f.value["count"] == 1
    assert conv.status == "PASS"


def test_a_late_nan_fails_the_run():
    vals = _approach()
    vals[-5] = math.inf
    conv = assess([_mon(vals)], {}, Clock(), _steady_meta())
    assert conv.status == "FAIL"
    assert next(f for f in conv.findings if f.code == "CONV.NON_FINITE").severity == "FAIL"


def test_a_diverging_series():
    vals = [1.0 + 0.01 * math.exp(i / 60.0) for i in range(1500)]
    conv = assess([_mon(vals)], {}, Clock(), _steady_meta())
    assert conv.status == "FAIL"
    assert any(f.code == "CONV.DIVERGED" for f in conv.findings)


def test_a_zero_mean_noisy_monitor_abstains_rather_than_fails():
    rng = random.Random(5)
    vals = [1e-12 * rng.gauss(0, 1) for _ in range(1000)]
    alone = assess([_mon(vals)], {}, Clock(), _steady_meta())
    row = alone.data["monitors"]["probe/0/p:p"]
    assert row["verdict"] == "NOT_ASSESSABLE" and row["not_assessable_reason"] == "ZERO_MEAN_SCALE"
    assert alone.status == "NOT_ASSESSABLE"          # M1-O-3: every monitor abstained
    assert any(f.code == "CONV.ZERO_MEAN_SCALE" and f.severity == "WARN" for f in alone.findings)
    paired = assess([_mon(vals), _mon(_approach(n=1000), mid="q/f:p")], {}, Clock(), _steady_meta())
    assert paired.status == "WARN"                   # the other monitor converged; one abstained


def test_a_component_near_zero_is_measured_against_its_vector():
    rng = random.Random(2)
    n = 1200
    fx = _approach(n=n, base=10.0, amp=2.0, tau=80)
    fy = [1e-9 * rng.gauss(0, 1) for _ in range(n)]
    mons = [_mon(fx, mid="forces/force:total_x", column="total_x", vector="forces/force:total"),
            _mon(fy, mid="forces/force:total_y", column="total_y", vector="forces/force:total")]
    conv = assess(mons, {}, Clock(), _steady_meta())
    row = conv.data["monitors"]["forces/force:total_y"]
    assert row["scale"]["basis"] == "vector_magnitude"
    assert row["verdict"] == "CONVERGED"
    assert conv.status == "PASS"


def test_phase_fractions_are_excluded():
    vals = [0.5 + 0.1 * math.sin(i / 3.0) for i in range(1000)]
    conv = assess([_mon(vals, mid="vol/volFieldValue:volAverage(alpha.water)",
                        column="volAverage(alpha.water)")], {}, Clock(), _steady_meta())
    row = conv.data["monitors"]["vol/volFieldValue:volAverage(alpha.water)"]
    assert row["verdict"] == "EXCLUDED" and row["counts"] is False
    assert conv.status == "NOT_ASSESSABLE"


def test_no_monitors():
    conv = assess([], {"p": [1.0, 0.1, 0.01]}, Clock(),
                  RunMeta(regime="steady", paths_searched=("postProcessing",)))
    assert conv.status == "NOT_ASSESSABLE"
    f = next(f for f in conv.findings if f.code == "CONV.NO_MONITORS")
    assert f.message == sentences.NOT_ASSESSABLE_NO_MONITORS.format(paths="postProcessing")


def test_a_killed_run_reports_no_waste():
    vals = _approach()
    conv = assess([_mon(vals)], {}, _clock(len(vals)),
                  _steady_meta(ended_by="signal", last_complete_step=2000))
    assert conv.status == "WARN"
    assert conv.data["waste"] is None
    assert any(f.code == "CONV.RUN_KILLED" for f in conv.findings)


def test_the_budget_decimates_conservatively():
    vals = _approach(n=20000, tau=1000)
    exact = assess([_mon(vals)], {}, Clock(), _steady_meta())
    cheap = assess([_mon(vals)], {}, Clock(), _steady_meta(), Options(eval_budget=200))
    assert exact.status == "PASS" and cheap.status == "WARN"
    assert cheap.data["budget"]["decimated"] is True
    assert cheap.data["converged_at_step"] >= exact.data["converged_at_step"]
    assert any(f.code == "CONV.BUDGET_DECIMATED" for f in cheap.findings)


def test_the_monitor_cap():
    mons = [_mon(_approach(n=300), mid=f"m{i:02d}/f:p") for i in range(5)]
    conv = assess(mons, {}, Clock(), _steady_meta(), Options(max_monitors=3))
    assert conv.data["monitors_dropped"] == ["m03/f:p", "m04/f:p"]
    assert any(f.code == "CONV.MONITOR_CAP" for f in conv.findings)


def test_the_binding_monitor_is_the_last_to_settle():
    a = _approach(tau=50)
    b = _approach(tau=150)
    conv = assess([_mon(a, mid="a/f:p"), _mon(b, mid="b/f:p")], {}, Clock(), _steady_meta())
    by = conv.data["converged_at_step_by_monitor"]
    assert conv.data["binding_monitor"] == "b/f:p"
    assert conv.data["converged_at_step"] == max(by.values())


# ------------------------------------------------------------------ residuals
def test_celik_orders_by_integer_counting():
    assert orders_dropped(1e-1, 1e-5) == (4, False)
    assert orders_dropped(1.0, 2e-3) == (2, False)
    assert orders_dropped(1.0, 1e-40) == (30, True)


def test_celik_finding_severity():
    good = {"Ux": [1.0, 0.5] + [10.0 ** (-k / 50) for k in range(300)]}
    bad = {"p": [1.0, 0.5] + [0.5 * 10.0 ** (-k / 300) for k in range(300)]}
    f_good = next(f for f in assess([], good, Clock(), _steady_meta()).findings
                  if f.code == "CONV.CELIK_RESIDUAL_ORDERS")
    f_bad = next(f for f in assess([], bad, Clock(), _steady_meta()).findings
                 if f.code == "CONV.CELIK_RESIDUAL_ORDERS")
    assert f_good.severity == "INFO" and f_bad.severity == "WARN"


# ------------------------------------------------------------------ determinism
def test_identical_inputs_identical_output():
    vals = _approach(noise=1e-7)
    a = assess([_mon(vals)], {"p": vals}, _clock(len(vals)), _steady_meta()).as_dict()
    b = assess([_mon(list(vals))], {"p": list(vals)}, _clock(len(vals)), _steady_meta()).as_dict()
    assert a == b


def test_hash_seed_does_not_change_the_output():
    code = ("import json,math;from courant.converge.assess import *;"
            "v=[1+math.exp(-i/90) for i in range(900)];"
            "m=[MonitorSeries(id=k,column='p',source='s',times=[float(i+1) for i in range(900)],values=v)"
            " for k in ('b/f:p','a/f:p','c/f:p')];"
            "print(json.dumps(assess(m,{'p':v},Clock(),RunMeta(regime='steady')).as_dict(),sort_keys=True))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}, cwd=ROOT).stdout
            for seed in ("0", "12345")}
    assert len(outs) == 1


def test_non_finite_values_are_null_with_a_sibling():
    out = json_safe({"a": math.nan, "b": math.inf, "c": 1.0, "d": [math.nan, 2.0]})
    assert out == {"a": None, "a_non_finite": "nan", "b": None, "b_non_finite": "inf", "c": 1.0,
                   "d": [None, 2.0]}


def test_every_fixed_sentence_is_a_constant():
    assert set(sentences.ALL) >= {"NOT_CONVERGED", "NO_STOPPING_POINT_UNSTEADY", "NOT_STATIONARY",
                                  "INSUFFICIENT_DATA", "RUN_KILLED", "NOT_ASSESSABLE_NO_MONITORS",
                                  "DECIMATED", "TRANSIENT_NO_INTENT", "ZERO_MEAN_SCALE"}
