"""One run's convergence: monitors, residuals and clock in; status, findings and data out.

TRD CONV-1, CONV-10, CONV-11. A pure function: `series.py` reads the files, and this module
only does arithmetic on what it is handed. A monitor that raises degrades to NOT_ASSESSABLE
with the exception type only (M1-E-2). One bad monitor never costs the run its assessment.
"""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from courant.converge import sentences
from courant.converge import steady as st
from courant.converge import transient as tr
from courant.converge.exact import SCHEME, mean
from courant.converge.scale import Tolerance, is_coefficient, reference_scale, tolerance

PASS, WARN, FAIL, NOT_ASSESSABLE = "PASS", "WARN", "FAIL", "NOT_ASSESSABLE"

#: Monitor verdicts that make the module FAIL (M1-O-3).
_FAILING = frozenset({"UNSTEADY_UNDER_STEADY_SOLVER", "NOT_CONVERGED", "NOT_STATIONARY",
                      "DIVERGED", "CI_TOO_WIDE", "MONOTONE_DRIFT"})
_PHASE_FRACTION = re.compile(r"(^|[(:/])alpha\.")
NON_FINITE_FRACTION = 0.01      # E-3
DIVERGED_RATIO = 1.0e6          # E-4
EXCEED_RATIO = 1.0e3            # E-4: first index reported

#: The order findings are emitted in (M1-O-5), so the output is byte-stable.
ORDER = (
    "CONV.ANSWER_UNCHANGED", "CONV.ANSWER_MOVED", "CONV.WASTE", "CONV.CORE_HOURS_NOT_COMPUTED",
    "CONV.ALLOCATION", "CONV.ALLOCATION_NOT_COMPUTED", "CONV.REGIME_INFERRED",
    "CONV.TRANSIENT_NO_INTENT", "CONV.STEADY_FIXED_POINT", "CONV.NOT_CONVERGED",
    "CONV.UNSTEADY_STEADY", "CONV.ITERATION_OSCILLATION", "CONV.SMALL_OSCILLATION", "CONV.RESIDUAL_PLATEAU",
    "CONV.RESIDUAL_FALLING", "CONV.CELIK_RESIDUAL_ORDERS", "CONV.TRANSIENT_STATIONARY",
    "CONV.NOT_STATIONARY", "CONV.STATIONARITY_DISAGREE", "CONV.OSCILLATORY_BATCHES",
    "CONV.CI_TOO_WIDE", "CONV.LOW_ESS", "CONV.BATCH_SKEW", "CONV.VARIANCE_DISAGREE",
    "CONV.ZERO_MEAN_SCALE", "CONV.SIGN_CHANGE", "CONV.TRUNCATION_IN_TAIL",
    "CONV.MONOTONE_DRIFT", "CONV.INSUFFICIENT_DATA", "CONV.CONSTANT_SERIES",
    "CONV.NON_FINITE", "CONV.DIVERGED", "CONV.RESTART_OVERLAP", "CONV.MONITOR_UNREADABLE",
    "CONV.RUN_KILLED", "CONV.NO_MONITORS", "CONV.MONITOR_CAP", "CONV.BUDGET_DECIMATED",
    "CONV.NON_MONOTONE_CRITERION", "CONV.INTERNAL_ERROR",
)
_RANK = {code: i for i, code in enumerate(ORDER)}

METHOD_DRIFT = "Trailing-window OLS drift (closed form, exact window sums)"
METHOD_ACF = "Detrended autocorrelation, biased estimator; lag search 2..w/4"
METHOD_CELIK = "Celik, Ghia, Roache & Freitas (2008), J. Fluids Eng. 130(7) 078001, Preliminaries"
METHOD_SCALE = "Courant reference-scale ladder"


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class MonitorSeries:
    id: str                              # "<object>/<file stem>:<column>"
    column: str
    source: str                          # case-relative path(s) the samples came from
    times: Sequence[float]
    values: Sequence[float]
    vector: str | None = None            # the parent vector, for an _x/_y/_z column
    reader_findings: tuple = ()          # Findings from the reader (restarts, bad rows)
    not_assessable: str | None = None    # set by the reader when the samples are unusable


@dataclass(frozen=True)
class Clock:
    """Per solver step, from the log: simulated time, ClockTime and ExecutionTime."""
    times: Sequence[float] = ()
    clock_s: Sequence[float | None] = ()
    exec_s: Sequence[float | None] = ()


@dataclass(frozen=True)
class RunMeta:
    regime: str | None = None                  # "steady" | "transient" | None
    regime_basis: str = "unknown"              # config | solver | inferred | unknown
    regime_conflict: Mapping | None = None
    ranks: int | None = None
    ranks_basis: str = "unknown"               # log_nprocs | serial | user | unknown
    ended_by: str | None = None                # "signal" | "crash" | None
    last_complete_step: int | None = None
    paths_searched: tuple[str, ...] = ()
    monitors_ignored: tuple[str, ...] = ()
    monitors_unselected: tuple[str, ...] = ()
    selection: str = "all"                     # default_totals | user_patterns | all


@dataclass(frozen=True)
class Options:
    tol_abs: Mapping[str, float] = field(default_factory=dict)
    tol_rel: Mapping[str, float] = field(default_factory=dict)
    ref_scale: Mapping[str, float] = field(default_factory=dict)
    transient_average: bool = False
    monitors: tuple[str, ...] = ()             # glob patterns over monitor ids or columns
    max_monitors: int = 64
    eval_budget: int = 2_000_000          # CONV-6: exact sums make an evaluation O(1); M1 sized 120,000 for blocked sums
    cost_per_core_hour: float | None = None
    ccf_ghz: float | None = None

    def __post_init__(self) -> None:
        both = sorted(set(self.tol_abs) & set(self.tol_rel))
        if both:
            raise ValueError(f"both an absolute and a relative tolerance were given for: "
                             f"{', '.join(both)} (CONV-8: two criteria are not merged)")
        for table in (self.tol_abs, self.tol_rel, self.ref_scale):
            bad = [k for k, v in table.items() if not (isinstance(v, (int, float)) and v > 0)]
            if bad:
                raise ValueError(f"tolerances and scales must be positive: {', '.join(sorted(bad))}")

    def used(self) -> dict:
        return {"tol_abs": dict(sorted(self.tol_abs.items())),
                "tol_rel": dict(sorted(self.tol_rel.items())),
                "ref_scale": dict(sorted(self.ref_scale.items())),
                "transient_average": self.transient_average,
                "monitors": list(self.monitors),
                "max_monitors": self.max_monitors, "eval_budget": self.eval_budget,
                "cost_per_core_hour": self.cost_per_core_hour, "ccf_ghz": self.ccf_ghz,
                "drift_factor": st.DRIFT_FACTOR, "window_floor": st.WINDOW_FLOOR}


# ------------------------------------------------------------------ outputs
@dataclass(frozen=True)
class Finding:
    code: str
    severity: str                       # INFO | WARN | FAIL
    monitor: str | None
    method: str
    threshold: Mapping
    value: Mapping
    message: str | None = None

    def as_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "monitor": self.monitor,
                "method": self.method, "threshold": dict(self.threshold),
                "value": dict(self.value), "message": self.message}


@dataclass(frozen=True)
class Convergence:
    status: str
    regime: str | None
    findings: tuple[Finding, ...]
    data: Mapping
    inputs: tuple = ()

    def as_dict(self) -> dict:
        return {"status": self.status, "regime": self.regime,
                "findings": [f.as_dict() for f in self.findings],
                "data": json_safe(self.data),
                "inputs": [i if isinstance(i, dict) else i.model_dump() for i in self.inputs]}

    @property
    def counted(self) -> bool:
        """Whether this status may enter the verdict (CONV-7a, CONV-12)."""
        return bool(self.data.get("enters_verdict"))


def json_safe(obj):
    """Non-finite floats become null plus a `<key>_non_finite` sibling (M1-E-1)."""
    if isinstance(obj, Mapping):
        out = {}
        for k in sorted(obj):
            v = obj[k]
            if isinstance(v, float) and not math.isfinite(v):
                out[k] = None
                out[f"{k}_non_finite"] = "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
            else:
                out[k] = json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [None if isinstance(v, float) and not math.isfinite(v) else json_safe(v)
                for v in obj]
    return obj


def _fmt(x: float | None) -> str:
    if x is None:
        return "not available"
    return f"{x:.6g}"


# ------------------------------------------------------------------ steps and clocks
def _steps(times: Sequence[float], clock: Clock, regime: str | None) -> tuple[list[int], str]:
    """The solver step of every monitor sample, and how it was decided."""
    ct = list(clock.times)
    if ct and times and ct[0] - 1e-9 <= times[0] and times[-1] <= ct[-1] + 1e-9 * max(1.0, abs(ct[-1])):
        out = []
        for t in times:
            i = bisect.bisect_left(ct, t)
            if i == len(ct) or (i > 0 and abs(ct[i - 1] - t) <= abs(ct[i] - t)):
                i = i - 1
            out.append(i + 1)
        return out, "log_time"
    if regime == "steady" and times and all(float(t).is_integer() for t in times):
        return [int(t) for t in times], "time_as_iteration"
    return list(range(1, len(times) + 1)), "sample_index"


def _clock_at(series: Sequence[float | None], step: int) -> float | None:
    """The clock at a 1-based step, linearly interpolated across missing entries."""
    i = step - 1
    if not series or i < 0 or i >= len(series):
        return None
    if series[i] is not None:
        return series[i]
    lo = next((j for j in range(i - 1, -1, -1) if series[j] is not None), None)
    hi = next((j for j in range(i + 1, len(series)) if series[j] is not None), None)
    if lo is None or hi is None:
        return None
    return series[lo] + (series[hi] - series[lo]) * (i - lo) / (hi - lo)


# ------------------------------------------------------------------ one monitor
@dataclass
class _Mon:
    series: MonitorSeries
    verdict: str = NOT_ASSESSABLE
    counts: bool = True
    advisory_reason: str | None = None
    row: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    stop_step: int | None = None
    stop_time: float | None = None
    first_step: int | None = None
    final_step: int | None = None
    steps: list = field(default_factory=list)
    step_basis: str = "sample_index"
    decimated: bool = False
    evaluations: int = 0
    resolution: int = 1


def _clean(mon: _Mon) -> tuple[list[float], list[float]] | None:
    """Drop non-finite samples (E-3). None when too many or too late to assess."""
    s = mon.series
    bad = [i for i, v in enumerate(s.values) if not math.isfinite(v)]
    if not bad:
        return list(s.times), list(s.values)
    n = len(s.values)
    late = any(i >= n - max(1, n // 10) for i in bad)
    severe = len(bad) > NON_FINITE_FRACTION * n or late
    first = bad[0]
    kind = "nan" if math.isnan(s.values[first]) else "inf"
    mon.findings.append(Finding(
        "CONV.NON_FINITE", FAIL if severe else WARN, s.id, "finite-value filter",
        {"fraction": NON_FINITE_FRACTION},
        {"series": s.id, "kind": kind, "count": len(bad), "of": n, "first_index": first,
         "first_time": s.times[first], "in_final_tenth": late}))
    mon.row["non_finite"] = {"count": len(bad), "of": n, "first_index": first}
    if severe:
        mon.verdict = NOT_ASSESSABLE
        mon.row["not_assessable_reason"] = "NON_FINITE"
        return None
    keep = [i for i in range(n) if math.isfinite(s.values[i])]
    return [s.times[i] for i in keep], [s.values[i] for i in keep]


def _scale_and_tol(mon: _Mon, window: Sequence[float], vector_mag: float | None,
                   options: Options):
    s = mon.series
    tol = tolerance(s.id, s.column, options.tol_abs, options.tol_rel)
    user = options.ref_scale.get(s.id, options.ref_scale.get(s.column))
    sc = reference_scale(window, user=user, vector_magnitude=vector_mag)
    mon.row["tolerance"] = {"branch": tol.branch, "value": tol.value, "source": tol.source}
    mon.row["scale"] = {"value": sc.value, "basis": sc.basis, "mean": sc.mean, "stdev": sc.stdev}
    if sc.basis == "stdev":
        mon.findings.append(Finding(
            "CONV.ZERO_MEAN_SCALE", "INFO", s.id, METHOD_SCALE, {},
            {"monitor": s.id, "mean": sc.mean, "stdev": sc.stdev, "scale_used": sc.value,
             "scale_basis": sc.basis}))
    if sc.basis == "mean" and window and min(window) < 0.0 < max(window):
        crossings = sum_crossings(window)
        mon.findings.append(Finding(
            "CONV.SIGN_CHANGE", WARN, s.id, METHOD_SCALE, {},
            {"monitor": s.id, "crossings": crossings, "window": len(window)}))
    return tol, sc


def sum_crossings(values: Sequence[float]) -> int:
    return len([i for i in range(1, len(values)) if (values[i - 1] < 0.0) != (values[i] < 0.0)])


def _limit(tol: Tolerance, scale_value: float | None) -> float:
    """The absolute size of the tolerance: tol itself, or tol * R."""
    if tol.branch == "absolute":
        return tol.value
    return tol.value * scale_value if scale_value else 0.0


def _delta(mon: _Mon, tol: Tolerance, scale_value: float | None, at_stop: float,
           at_end: float) -> None:
    s = mon.series
    delta = abs(at_stop - at_end)
    frac = delta / scale_value if scale_value else (0.0 if delta == 0.0 else math.inf)
    counts = delta * 1.0e4 if is_coefficient(s.column) else None
    ok = delta <= tol.value if tol.branch == "absolute" else frac <= tol.value
    mon.row.update({"value_at_stop": at_stop, "value_at_end": at_end, "delta": delta,
                    "delta_fraction": frac if scale_value else None, "delta_counts": counts,
                    "delta_ok": ok})
    if not scale_value:
        mon.row["delta_fraction_non_finite"] = "undefined_zero_scale"
    value = {"monitor": s.id, "value_at_stop": at_stop, "value_at_end": at_end, "delta": delta,
             "delta_fraction": frac if scale_value else None, "delta_counts": counts}
    if ok:
        mon.findings.append(Finding("CONV.ANSWER_UNCHANGED", "INFO", s.id,
                                    "Courant answer-unchanged delta",
                                    {"tol": tol.value, "branch": tol.branch}, value))
    else:
        mon.verdict = "NOT_CONVERGED"
        mon.findings.append(Finding("CONV.ANSWER_MOVED", FAIL, s.id,
                                    "Courant answer-unchanged delta",
                                    {"tol": tol.value, "branch": tol.branch}, value,
                                    "The answer moved by more than its tolerance after the "
                                    "stopping point, so it was not a stopping point."))


def _steady(mon: _Mon, times, values, vector_mag, options: Options, budget: int) -> None:
    s = mon.series
    n = len(values)
    mon.row["n"] = n
    if n < st.WINDOW_FLOOR:
        mon.verdict = "INSUFFICIENT_DATA"
        mon.findings.append(Finding(
            "CONV.INSUFFICIENT_DATA", WARN, s.id, METHOD_DRIFT, {"n_required": st.WINDOW_FLOOR},
            {"monitor": s.id, "stage": "window", "n": n, "n_required": st.WINDOW_FLOOR},
            sentences.INSUFFICIENT_DATA.format(
                monitor=s.id, detail=f"It has {n} samples and the window rule needs "
                                     f"{st.WINDOW_FLOOR}.")))
        return
    w = st.window_width(n)
    window = values[n - w:]
    tol, sc = _scale_and_tol(mon, window, vector_mag, options)
    limit = _limit(tol, sc.value)
    mon.row["window"] = w
    if max(values) == min(values):
        mon.findings.append(Finding("CONV.CONSTANT_SERIES", "INFO", s.id, "constant series",
                                    {}, {"monitor": s.id, "value": values[0], "n": n}))
        _converged(mon, w - 1, values, tol, sc.value)
        return
    drift = st.Drift(values, w)
    last_drift = drift.drift(n - 1)
    mon.row["drift_last"] = last_drift
    mon.row["drift_last_rel"] = last_drift / sc.value if sc.value else None
    tau = st.DRIFT_FACTOR * limit
    osc = st.oscillation(st.detrend(window, drift.slope(n - 1)), limit)
    mon.row["oscillation"] = {"tested": osc.tested, "lag": osc.lag, "acf": osc.acf,
                              "amplitude": osc.amplitude, "n_periods": osc.n_periods,
                              "fires": osc.fires, "reason": osc.reason}
    smooth = 1
    if osc.tested and osc.lag is not None and osc.lag <= st.ITERATION_LAG_MAX:
        # CONV-6c: a period of two or three iterations is the steady iteration alternating,
        # not a flow that is unsteady. It is judged by half its amplitude, the distance from
        # the cycle mean to any sample, and values are read as means over one period.
        tail = window[w - osc.n_periods * osc.lag:]
        cycle_mean = mean(tail)
        half = osc.amplitude / 2.0
        rel = osc.amplitude / sc.value if sc.value else None
        value = {"monitor": s.id, "lag": osc.lag, "acf": osc.acf, "amplitude": osc.amplitude,
                 "amplitude_rel": rel, "half_amplitude": half, "cycle_mean": cycle_mean}
        if half > limit:
            mon.verdict = "NOT_CONVERGED"
            mon.findings.append(Finding(
                "CONV.ITERATION_OSCILLATION", FAIL, s.id, METHOD_ACF,
                {"acf": st.ACF_THRESHOLD, "half_amplitude": limit}, value,
                sentences.ITERATION_OSCILLATION.format(
                    monitor=s.id, lag=osc.lag, amplitude=_fmt(osc.amplitude),
                    cycle_mean=_fmt(cycle_mean), half=_fmt(half), tol=_fmt(limit))))
            return
        mon.findings.append(Finding("CONV.ITERATION_OSCILLATION", "INFO", s.id, METHOD_ACF,
                                    {"acf": st.ACF_THRESHOLD, "half_amplitude": limit}, value))
        smooth = osc.lag
    elif osc.fires:
        tail = window[w - osc.n_periods * osc.lag:]
        cycle_mean = mean(tail)
        rel = osc.amplitude / sc.value if sc.value else None
        mon.verdict = "UNSTEADY_UNDER_STEADY_SOLVER"
        mon.findings.append(Finding(
            "CONV.UNSTEADY_STEADY", FAIL, s.id, METHOD_ACF,
            {"acf": st.ACF_THRESHOLD, "amplitude": limit},
            {"monitor": s.id, "lag": osc.lag, "acf": osc.acf, "amplitude": osc.amplitude,
             "amplitude_rel": rel, "period_samples": osc.lag, "n_periods_in_window": osc.n_periods,
             "cycle_mean": cycle_mean},
            sentences.NO_STOPPING_POINT_UNSTEADY.format(
                period_steps=osc.lag, amplitude=_fmt(osc.amplitude), amplitude_rel=_fmt(rel),
                n_periods=osc.n_periods, cycle_mean=_fmt(cycle_mean))))
        return
    elif osc.tested and osc.lag is not None:
        mon.findings.append(Finding(
            "CONV.SMALL_OSCILLATION", "INFO", s.id, METHOD_ACF,
            {"acf": st.ACF_THRESHOLD, "amplitude": limit},
            {"monitor": s.id, "lag": osc.lag, "acf": osc.acf, "amplitude": osc.amplitude}))
    search = st.stopping_index(drift, tau, budget)
    mon.evaluations = search.evaluations
    if search.index is None:
        r_first = reference_scale(values[:w]).value
        if r_first and abs(values[-1]) > DIVERGED_RATIO * r_first:
            first_exceed = next(i for i, v in enumerate(values) if abs(v) > EXCEED_RATIO * r_first)
            mon.verdict = "DIVERGED"
            mon.findings.append(Finding(
                "CONV.DIVERGED", FAIL, s.id, "growth against the first window's scale",
                {"ratio": DIVERGED_RATIO},
                {"monitor": s.id, "first_exceed_index": first_exceed, "final_value": values[-1],
                 "reference_scale": r_first}))
            return
        if sc.basis == "stdev" and s.id not in options.ref_scale and s.column not in options.ref_scale:
            mon.verdict = NOT_ASSESSABLE
            mon.row["not_assessable_reason"] = "ZERO_MEAN_SCALE"
            mon.findings.append(Finding("CONV.ZERO_MEAN_SCALE", WARN, s.id, METHOD_SCALE, {},
                                        {"monitor": s.id, "stdev": sc.stdev},
                                        sentences.ZERO_MEAN_SCALE.format(monitor=s.id)))
            return
        mon.verdict = "NOT_CONVERGED"
        rel = last_drift / sc.value if sc.value else None
        mon.findings.append(Finding(
            "CONV.NOT_CONVERGED", FAIL, s.id, METHOD_DRIFT,
            {"drift": tau, "drift_factor": st.DRIFT_FACTOR, "tol": tol.value, "branch": tol.branch},
            {"monitor": s.id, "window": w, "drift": last_drift, "drift_rel": rel,
             "final_value": values[-1]},
            sentences.NOT_CONVERGED.format(monitor=s.id, drift_rel=_fmt(rel), window=w,
                                           tol=_fmt(st.DRIFT_FACTOR * tol.value))))
        return
    mon.decimated, mon.resolution = search.decimated, search.resolution
    _converged(mon, search.index, values, tol, sc.value, smooth)
    mon.findings.append(Finding(
        "CONV.STEADY_FIXED_POINT", "INFO", s.id, METHOD_DRIFT,
        {"drift": tau, "drift_factor": st.DRIFT_FACTOR, "tol": tol.value, "branch": tol.branch},
        {"monitor": s.id, "window": w, "drift_last": last_drift,
         "converged_at_index": search.index, "evaluations": search.evaluations,
         "resolution": search.resolution}))


def _converged(mon: _Mon, index: int, values, tol: Tolerance, scale_value, smooth: int = 1) -> None:
    mon.verdict = "CONVERGED"
    mon.row["stop_index"] = index
    mon.stop_step = mon.steps[index]
    if smooth > 1:                      # an iteration oscillation: read one period's mean
        at_stop = mean(values[max(0, index - smooth + 1):index + 1])
        at_end = mean(values[len(values) - smooth:])
        mon.row["values_read_as"] = f"mean over {smooth} samples (one period of the iteration oscillation)"
    else:
        at_stop, at_end = values[index], values[-1]
    _delta(mon, tol, scale_value, at_stop, at_end)
    i = max(range(len(values)), key=lambda k: abs(values[k] - values[-1]))
    mon.row["max_abs_deviation_index"] = i


def _transient(mon: _Mon, times, values, options: Options) -> None:
    s = mon.series
    n = len(values)
    mon.row["n"] = n
    if n < st.WINDOW_FLOOR:
        mon.verdict = "INSUFFICIENT_DATA"
        mon.findings.append(Finding(
            "CONV.INSUFFICIENT_DATA", WARN, s.id, tr.METHOD_MSER, {"n_required": st.WINDOW_FLOOR},
            {"monitor": s.id, "stage": "mser", "n": n, "n_required": st.WINDOW_FLOOR},
            sentences.INSUFFICIENT_DATA.format(
                monitor=s.id, detail=f"It has {n} samples and MSER-5 needs {st.WINDOW_FLOOR}.")))
        return
    if max(values) == min(values):
        tol, sc = _scale_and_tol(mon, values, None, options)
        mon.findings.append(Finding("CONV.CONSTANT_SERIES", "INFO", s.id, "constant series",
                                    {}, {"monitor": s.id, "value": values[0], "n": n}))
        _converged(mon, st.WINDOW_FLOOR - 1, values, tol, sc.value)
        return
    full = tr.mser5(values)
    trunc = values[full.truncate_at:tr.MSER_BATCH * full.n_batches] if not full.insufficient else values
    tol, sc = _scale_and_tol(mon, trunc, None, options)
    limit_abs = tol.value if tol.branch == "absolute" else None

    def settled_tail(vs=values) -> bool:
        w = st.window_width(len(vs))
        d = st.Drift(vs, w)
        return not st.unsettled(d.drift(len(vs) - 1), st.DRIFT_FACTOR * _limit(tol, sc.value))

    res = tr.assess_prefix(values, limit_abs, tol.value, sc.value, settled_tail)
    ms = res.mser
    mon.row["transient"] = {"mser_d_star": ms.d_star, "mser_n_batches": ms.n_batches,
                            "mser_tail_dropped": ms.tail_dropped,
                            "mser_two_pass_recomputations": ms.recomputations,
                            "truncation_index": ms.truncate_at if not ms.insufficient else None}
    if res.verdict in ("INSUFFICIENT_DATA", "MONOTONE_DRIFT"):
        mon.verdict = res.verdict
        if res.stage == "mser_tail":
            code = "CONV.MONOTONE_DRIFT" if res.verdict == "MONOTONE_DRIFT" else "CONV.TRUNCATION_IN_TAIL"
            mon.findings.append(Finding(
                code, FAIL if code == "CONV.MONOTONE_DRIFT" else WARN, s.id,
                tr.METHOD_MSER + "; tail guard: second half (Courant)", {"fraction": 0.5},
                {"monitor": s.id, "d_star": ms.d_star, "n_batches": ms.n_batches,
                 "post_transient_samples": tr.MSER_BATCH * (ms.n_batches - (ms.d_star or 0))}))
        else:
            detail = ("Its batch means are still strongly correlated (lag-1 correlation at least "
                      "0.5), or fewer than 20 batches remain after the transient."
                      if res.stage == "batching" else
                      "MSER-5 needs at least 40 batch means of five samples.")
            mon.findings.append(Finding(
                "CONV.INSUFFICIENT_DATA", WARN, s.id, tr.METHOD_BATCH,
                {"batches_required": tr.BATCH_MIN},
                {"monitor": s.id, "stage": res.stage, "n": n},
                sentences.INSUFFICIENT_DATA.format(monitor=s.id, detail=detail)))
        return
    bt, sty = res.batches, res.stationarity
    mon.row["transient"].update({
        "batch_size": bt.size, "batch_count": bt.count, "rho1": bt.rho1, "mean": bt.mean,
        "half_width": bt.half_width, "half_width_rel": bt.half_width / sc.value if sc.value else None,
        "ess": bt.ess, "skew": bt.skew, "se": bt.se,
        "geweke_z": sty.geweke_z, "vn_z": sty.vn_z})
    stat_value = {"monitor": s.id, "geweke_z": sty.geweke_z, "vn_z": sty.vn_z, "n_a": sty.n_a,
                  "n_b": sty.n_b}
    if sty.oscillatory_batches:
        mon.findings.append(Finding("CONV.OSCILLATORY_BATCHES", WARN, s.id, tr.METHOD_STATIONARITY,
                                    {"z": tr.Z_CRIT}, {"monitor": s.id, "vn_z": sty.vn_z,
                                                       "batch_size": bt.size}))
    if bt.skew is not None and abs(bt.skew) > tr.SKEW_LIMIT:
        mon.findings.append(Finding("CONV.BATCH_SKEW", WARN, s.id, tr.METHOD_BATCH,
                                    {"skew": tr.SKEW_LIMIT},
                                    {"monitor": s.id, "skew": bt.skew, "k": bt.count}))
    if res.verdict == "NOT_STATIONARY":
        mon.verdict = "NOT_STATIONARY"
        mon.findings.append(Finding("CONV.NOT_STATIONARY", FAIL, s.id, tr.METHOD_STATIONARITY,
                                    {"z": tr.Z_CRIT}, {**stat_value, "which_failed": sty.which_failed},
                                    sentences.NOT_STATIONARY.format(monitor=s.id,
                                                                    which_failed=sty.which_failed)))
        if (abs(sty.geweke_z) < tr.Z_CRIT) != (sty.vn_z >= -tr.Z_CRIT):
            mon.findings.append(Finding("CONV.STATIONARITY_DISAGREE", "INFO", s.id,
                                        tr.METHOD_STATIONARITY, {"z": tr.Z_CRIT},
                                        {**stat_value, "which_failed": sty.which_failed}))
        return
    mon.findings.append(Finding("CONV.TRANSIENT_STATIONARY", "INFO", s.id, tr.METHOD_STATIONARITY,
                                {"z": tr.Z_CRIT}, stat_value))
    yt = values[ms.truncate_at:tr.MSER_BATCH * ms.n_batches]
    se_spec = math.sqrt(max(0.0, tr.s0_bartlett(yt)) / len(yt))
    mon.row["transient"]["spectral_se"] = se_spec
    if bt.se and se_spec and max(se_spec / bt.se, bt.se / se_spec) > tr.SPECTRAL_RATIO:
        mon.findings.append(Finding("CONV.VARIANCE_DISAGREE", WARN, s.id,
                                    "Batch-means vs Bartlett spectral standard error",
                                    {"ratio": tr.SPECTRAL_RATIO},
                                    {"monitor": s.id, "se_batch": bt.se, "se_spectral": se_spec}))
    if bt.ess is not None and bt.ess < tr.LOW_ESS:
        mon.findings.append(Finding("CONV.LOW_ESS", WARN, s.id, tr.METHOD_BATCH, {"ess": tr.LOW_ESS},
                                    {"monitor": s.id, "ess": bt.ess, "n_truncated": bt.n,
                                     "batch_size": bt.size}))
    if res.verdict == "CI_TOO_WIDE":
        mon.verdict = "CI_TOO_WIDE"
        mon.findings.append(Finding(
            "CONV.CI_TOO_WIDE", FAIL, s.id, tr.METHOD_BATCH, {"tol": tol.value, "branch": tol.branch},
            {"monitor": s.id, "mean": bt.mean, "half_width": bt.half_width, "k": bt.count,
             "batch_size": bt.size, "ess": bt.ess}))
        return

    def ok(prefix) -> bool:
        return tr.assess_prefix(prefix, limit_abs, tol.value, sc.value).verdict == "CONVERGED"

    search = tr.stopping_prefix(values, ok)
    mon.evaluations = search.evaluations
    if search.index is None:
        mon.verdict = "NOT_CONVERGED"
        return
    mon.resolution = search.resolution
    if search.non_monotone_rounds:
        mon.findings.append(Finding("CONV.NON_MONOTONE_CRITERION", "INFO", s.id, tr.METHOD_BATCH,
                                    {"rounds": tr.SEARCH_ROUNDS},
                                    {"monitor": s.id, "rounds_used": search.non_monotone_rounds}))
    prefix = tr.assess_prefix(values[:search.index + 1], limit_abs, tol.value, sc.value)
    at_stop = prefix.batches.mean if prefix.batches and prefix.batches.mean is not None else bt.mean
    mon.verdict = "CONVERGED"
    mon.row["stop_index"] = search.index
    mon.stop_step = mon.steps[search.index]
    _delta(mon, tol, sc.value, at_stop, bt.mean)


# ------------------------------------------------------------------ the module
def assess(monitors: Sequence[MonitorSeries], residuals: Mapping[str, Sequence[float]],
           clock: Clock, meta: RunMeta, options: Options = Options(), inputs: tuple = ()) -> Convergence:
    findings: list[Finding] = []
    regime = meta.regime
    if meta.regime_basis == "inferred":
        findings.append(Finding("CONV.REGIME_INFERRED", "INFO", None, "Courant regime inference",
                                {}, {"regime": regime, "basis": meta.regime_basis}))

    ordered = sorted(monitors, key=lambda m: m.id)
    dropped = [m.id for m in ordered[options.max_monitors:]]
    ordered = ordered[:options.max_monitors]
    if dropped:
        findings.append(Finding("CONV.MONITOR_CAP", WARN, None, "monitor cap",
                                {"max_monitors": options.max_monitors},
                                {"discovered": len(monitors), "assessed": len(ordered)}))

    transient_advisory = regime == "transient" and not options.transient_average
    if transient_advisory and ordered:
        findings.append(Finding("CONV.TRANSIENT_NO_INTENT", "INFO", None, "CONV-7a", {}, {},
                                sentences.TRANSIENT_NO_INTENT))

    # vector magnitudes over each sample, for components near zero (scale.py)
    vectors: dict[str, list[MonitorSeries]] = {}
    for m in ordered:
        if m.vector:
            vectors.setdefault(m.vector, []).append(m)

    def magnitude(m: MonitorSeries, w: int) -> float | None:
        sib = vectors.get(m.vector or "", [])
        if len(sib) < 2 or len({len(x.values) for x in sib}) != 1:
            return None
        n = len(sib[0].values)
        mags = [math.sqrt(math.fsum(x.values[i] * x.values[i] for x in sib)) for i in range(n - w, n)]
        mags = [v for v in mags if math.isfinite(v)]
        return mean(mags) if mags else None

    mons: list[_Mon] = []
    n_assessed = len([m for m in ordered if not _PHASE_FRACTION.search(m.id)])
    budget = max(1, options.eval_budget // max(1, n_assessed))
    for m in ordered:
        mon = _Mon(m)
        mon.findings.extend(m.reader_findings)
        mons.append(mon)
        if _PHASE_FRACTION.search(m.id):
            mon.verdict, mon.counts, mon.advisory_reason = "EXCLUDED", False, "PHASE_FRACTION"
            continue
        if m.not_assessable:
            mon.row["not_assessable_reason"] = m.not_assessable
            continue
        try:
            cleaned = _clean(mon)
            if cleaned is None:
                continue
            times, values = cleaned
            mon.steps, mon.step_basis = _steps(times, clock, regime)
            if mon.steps:
                mon.first_step, mon.final_step = mon.steps[0], mon.steps[-1]
            if regime == "transient":
                if transient_advisory:
                    mon.counts, mon.advisory_reason = False, "TRANSIENT_NO_INTENT"
                _transient(mon, times, values, options)
            else:
                w = st.window_width(len(values))
                _steady(mon, times, values, magnitude(m, min(w, len(values))), options, budget)
        except Exception as exc:  # noqa: BLE001 - M1-E-2: one monitor never costs the run
            mon.verdict = NOT_ASSESSABLE
            mon.row["not_assessable_reason"] = "INTERNAL_ERROR"
            mon.findings.append(Finding("CONV.INTERNAL_ERROR", WARN, m.id, "M1-E-2", {},
                                        {"monitor": m.id, "exception": type(exc).__name__}))

    for mon in mons:
        findings.extend(mon.findings)

    counting = [m for m in mons if m.counts]
    assessed = [m for m in counting if m.verdict != NOT_ASSESSABLE]
    hard_non_finite = any(f.code == "CONV.NON_FINITE" and f.severity == FAIL
                          for m in counting for f in m.findings)
    if not ordered:
        findings.append(Finding("CONV.NO_MONITORS", WARN, None, "monitor discovery", {},
                                {"paths_searched": list(meta.paths_searched)},
                                sentences.NOT_ASSESSABLE_NO_MONITORS.format(
                                    paths=", ".join(meta.paths_searched) or "none")))
    if not assessed and not hard_non_finite:
        status = NOT_ASSESSABLE
    elif hard_non_finite or any(m.verdict in _FAILING for m in assessed) or \
            any(m.row.get("delta_ok") is False for m in assessed):
        status = FAIL
    elif any(m.verdict in (NOT_ASSESSABLE, "INSUFFICIENT_DATA") for m in counting) or \
            any(m.decimated for m in counting) or meta.ended_by:
        status = WARN
    else:
        status = PASS

    # the global stopping point: the last quantity to settle (M1-F-6)
    stops = {m.series.id: m.stop_step for m in counting if m.stop_step is not None}
    all_stopped = bool(counting) and len(stops) == len(counting)
    # M1-F-6: the run has stopped only when its last quantity has. A monitor without a
    # stopping point means the run has none; the per-monitor points are still reported.
    binding = max(sorted(stops), key=lambda k: stops[k]) if stops and all_stopped else None
    converged_at = stops[binding] if binding else None
    resolution = max([m.resolution for m in counting if m.stop_step is not None] or [1])
    if any(m.decimated for m in counting):
        findings.append(Finding("CONV.BUDGET_DECIMATED", WARN, None, "evaluation budget",
                                {"eval_budget": options.eval_budget},
                                {"evaluations_used": sum_int(m.evaluations for m in mons),
                                 "step_resolution": resolution},
                                sentences.DECIMATED.format(step_resolution=resolution)))

    waste = None
    if status in (PASS, WARN) and all_stopped and binding and not meta.ended_by:
        bmon = next(m for m in counting if m.series.id == binding)
        waste = _waste(bmon, converged_at, clock, meta, options, findings)
    if meta.ended_by:
        sentence = sentences.RUN_KILLED if meta.ended_by == "signal" else sentences.RUN_CRASHED
        findings.append(Finding("CONV.RUN_KILLED", WARN, None, "events", {},
                                {"ended_by": meta.ended_by,
                                 "last_complete_step": meta.last_complete_step},
                                sentence.format(last_complete_step=meta.last_complete_step)))

    res_table, res_findings = _residuals(residuals, status)
    findings.extend(res_findings)

    findings.sort(key=lambda f: (_RANK.get(f.code, len(ORDER)), f.monitor or ""))
    data = {
        "regime": regime, "regime_basis": meta.regime_basis,
        "regime_conflict": dict(meta.regime_conflict) if meta.regime_conflict else None,
        "enters_verdict": status in (PASS, WARN, FAIL) and not transient_advisory and bool(assessed),
        "converged_at_step": converged_at, "step_resolution": resolution if converged_at else None,
        "binding_monitor": binding,
        "converged_at_step_by_monitor": dict(sorted(stops.items())),
        "monitors": {m.series.id: {**m.row, "verdict": m.verdict, "counts": m.counts,
                                   "advisory_reason": m.advisory_reason, "source": m.series.source,
                                   "first_step": m.first_step, "final_step": m.final_step,
                                   "step_basis": m.step_basis, "stop_step": m.stop_step}
                     for m in mons},
        "residuals": res_table,
        "waste": waste,
        "monitors_dropped": dropped,
        "monitors_ignored": list(meta.monitors_ignored),
        "monitors_unselected": list(meta.monitors_unselected),
        "monitor_selection": meta.selection,
        "paths_searched": list(meta.paths_searched),
        "budget": {"evaluations_used": sum_int(m.evaluations for m in mons),
                   "evaluation_budget": options.eval_budget,
                   "decimated": any(m.decimated for m in mons)},
        "reduction": {"scheme": SCHEME},
        "options_used": options.used(),
        "ranks": meta.ranks, "ranks_basis": meta.ranks_basis,
    }
    return Convergence(status=status, regime=regime, findings=tuple(findings), data=data,
                       inputs=tuple(inputs))


def sum_int(values) -> int:
    """Integer totals only; floats are reduced by fsum elsewhere."""
    total = 0
    for v in values:
        total = total + int(v)
    return total


def _waste(bmon: _Mon, stop_step: int, clock: Clock, meta: RunMeta, options: Options,
           findings: list) -> dict:
    first, final = bmon.first_step, bmon.final_step
    wasted_steps = final - stop_step
    span = final - first
    wall = cpu = None
    if bmon.step_basis == "log_time":
        c_end, c_stop = _clock_at(clock.clock_s, final), _clock_at(clock.clock_s, stop_step)
        e_end, e_stop = _clock_at(clock.exec_s, final), _clock_at(clock.exec_s, stop_step)
        wall = (c_end - c_stop) if c_end is not None and c_stop is not None else None
        cpu = (e_end - e_stop) if e_end is not None and e_stop is not None else None
    total_wall = None
    if bmon.step_basis == "log_time":
        c_first = _clock_at(clock.clock_s, first)
        c_last = _clock_at(clock.clock_s, final)
        total_wall = (c_last - c_first) if c_first is not None and c_last is not None else None
    waste = {"converged_at_step": stop_step, "first_step": first, "final_step": final,
             "wasted_steps": wasted_steps, "steps_total": span,
             "wasted_fraction": (wasted_steps / span) if span > 0 else 0.0,
             "wasted_wall_clock_s": wall, "wall_clock_total_s": total_wall,
             "wasted_cpu_s": cpu, "cpu_basis": "ExecutionTime of rank 0",
             "ranks": meta.ranks, "ranks_basis": meta.ranks_basis,
             "wasted_core_hours": None, "wasted_cost": None, "cost_per_core_hour": options.cost_per_core_hour,
             "wasted_auh": None, "ccf_ghz": options.ccf_ghz}
    missing = []
    if wall is None:
        missing.append("the log carried no wall clock for the stopping point and the end")
    if not meta.ranks:
        missing.append("the core count is not known")
    if not missing:
        waste["wasted_core_hours"] = meta.ranks * wall / 3600.0
        if options.cost_per_core_hour is not None:
            waste["wasted_cost"] = waste["wasted_core_hours"] * options.cost_per_core_hour
        if options.ccf_ghz is not None:
            waste["wasted_auh"] = meta.ranks * wall * options.ccf_ghz / 3600.0
            findings.append(Finding("CONV.ALLOCATION", "INFO", None,
                                    "FIA 2026 Sporting Regulations Section F Art. F4.4.4 (iss. 06, 27 Feb 2026)",
                                    {}, {"ncu": meta.ranks, "nss": wall, "ccf_ghz": options.ccf_ghz,
                                         "wasted_auh": waste["wasted_auh"],
                                         "wasted_mauh": waste["wasted_auh"] / 1.0e6},
                                    sentences.ALLOCATION_DISCLAIMER))
    else:
        findings.append(Finding("CONV.CORE_HOURS_NOT_COMPUTED", "INFO", None, "CONV-10", {},
                                {"missing": missing},
                                sentences.CORE_HOURS_NOT_COMPUTED.format(missing="; ".join(missing))))
    findings.append(Finding("CONV.WASTE", "INFO", None, "Courant waste figures", {},
                            {k: waste[k] for k in ("wasted_steps", "steps_total", "wasted_fraction",
                                                   "wasted_wall_clock_s", "wasted_cpu_s",
                                                   "wasted_core_hours")}))
    return waste


def _residuals(residuals: Mapping[str, Sequence[float]], status: str):
    table, findings = {}, []
    by_field = {}
    for name in sorted(residuals):
        if _PHASE_FRACTION.search(name):
            continue
        r = st.residual_summary(name, residuals[name])
        table[name] = {"n": r.n, "r_ref": r.r_ref, "r_final": r.r_final, "r_min": r.r_min,
                       "r_min_index": r.r_min_index, "ratio": r.ratio,
                       "orders_dropped_floor": r.orders_dropped_floor,
                       "orders_capped": r.orders_capped, "plateaued": r.plateaued,
                       "drift_rel": r.drift_rel}
        if r.orders_dropped_floor is not None:
            by_field[name] = r.orders_dropped_floor
        if r.plateaued is True:
            findings.append(Finding("CONV.RESIDUAL_PLATEAU", FAIL if status == FAIL else "INFO",
                                    None, "Trailing-window OLS drift on residuals",
                                    {"drift_rel": st.PLATEAU_DRIFT},
                                    {"field": name, "r_final": r.r_final, "drift_rel": r.drift_rel}))
        elif r.plateaued is False:
            findings.append(Finding("CONV.RESIDUAL_FALLING", "INFO", None,
                                    "Trailing-window OLS drift on residuals",
                                    {"drift_rel": st.PLATEAU_DRIFT},
                                    {"field": name, "r_final": r.r_final, "drift_rel": r.drift_rel}))
    if by_field:
        minimum = min(by_field.values())
        sev = "INFO" if minimum >= 3 else WARN
        msg = None
        if 3 <= minimum < 4:
            msg = "Every field dropped at least three orders; Celik et al. prefer four."
        elif minimum < 3:
            msg = "At least one field dropped fewer than three orders of magnitude."
        findings.append(Finding("CONV.CELIK_RESIDUAL_ORDERS", sev, None, METHOD_CELIK,
                                {"orders": 3, "preferred_orders": 4},
                                {"by_field": dict(sorted(by_field.items())), "minimum": minimum}, msg))
    return table, findings
