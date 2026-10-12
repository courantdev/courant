"""Verification checks — deterministic physics audit (Tech Arch §7).

Two tiers:
  * event/residual-based — run offline on the Watcher's history (divergence,
    convergence quality, continuity/mass balance, turbulence bounds),
  * field-based — need solution fields (conservation flux, y+, regime, plausibility);
    they run live and SKIP (not fail) when fields are unavailable.

Pure functions over {events, config, fields}. No LLM. Each returns a
VerificationResult whose severity feeds the Trust Grade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from courant.case.config import CaseConfig
from courant.events import Event, EventKind
from courant.events.diverged import divergences
from courant.verification.fields import FieldData
from courant.verification.convergence import (
    RESIDUAL_LOOSE,
    RESIDUAL_TARGET,
    solve_converged,
    worst_final_residual,
)
from courant.verification.grade import VerificationResult, VSeverity

OK, MINOR, MAJOR, CRIT = (VSeverity.OK, VSeverity.MINOR, VSeverity.MAJOR,
                          VSeverity.CRITICAL)


@dataclass
class VerifyContext:
    run_id: str
    events: list[Event]
    config: CaseConfig | None = None
    fields: FieldData | None = None

    def of(self, kind: EventKind) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def last_residual(self) -> Event | None:
        residuals = self.of(EventKind.RESIDUAL)
        return residuals[-1] if residuals else None


def _result(code, name, severity, passed, **kw) -> VerificationResult:
    return VerificationResult(check_code=code, check_name=name, severity=severity,
                              passed=passed, **kw)


def _skip(code, name, why) -> VerificationResult:
    return VerificationResult(check_code=code, check_name=name, severity=OK,
                              passed=True, skipped=True, detail=why)


class VerificationCheck(Protocol):
    code: str
    name: str

    def run(self, ctx: VerifyContext) -> VerificationResult:
        ...


# ---------------------------------------------------------------------------
# Event / residual-based checks (offline-capable)
# ---------------------------------------------------------------------------

class StabilityCheck:
    code, name = "stability.divergence", "Solution stability (no NaN/Inf)"

    def run(self, ctx: VerifyContext) -> VerificationResult:
        nan = divergences(ctx.events)
        if nan:
            return _result(self.code, self.name, CRIT, False,
                           detail=f"residual reached NaN/Inf on {nan[0].payload.field}",
                           evidence={"field": nan[0].payload.field})
        return _result(self.code, self.name, OK, True, detail="no NaN/Inf residuals")


class CrashCheck:
    code, name = "completion.crash", "Run completed without crashing"

    def run(self, ctx: VerifyContext) -> VerificationResult:
        crashes = ctx.of(EventKind.CRASH)
        if crashes:
            return _result(self.code, self.name, CRIT, False,
                           detail=f"solver crashed (tag={crashes[0].payload.tag})")
        return _result(self.code, self.name, OK, True, detail="clean exit")


class ConvergenceCheck:
    code, name = "convergence.residual", "Residuals converged below tolerance"
    # Shared with the fix loop via `courant.verification.convergence`. Two copies
    # of a threshold are two thresholds, and this pair is what decides whether a
    # fix worked — the fix loop used to answer that question with "did the
    # process exit 0", which is how a run that reached Co=4.6e+30 was reported
    # as fixed.
    TARGET, LOOSE = RESIDUAL_TARGET, RESIDUAL_LOOSE

    def run(self, ctx: VerifyContext) -> VerificationResult:
        last = ctx.last_residual()
        if last is None:
            return _skip(self.code, self.name, "no residual history")
        # `worst_final_residual`, not a fourth hand-rolled max().
        #
        # This one dropped NaN and nothing else. `convergence.worst_final_residual`
        # also drops infinities and **phase fractions** — and the omission had a
        # live cost: an unmodified `interFoam/damBreak` was graded **C**, because
        # `alpha.water` sits at 1.08e-02 while `p_rgh` settles to 5.3e-04. The
        # initial residual of a volume fraction measures how far the interface
        # MOVED, not how far the solve is from satisfying its equations, and on
        # a free-surface run the interface moving is the entire point. Every
        # customer doing VOF would have been told their correct run failed.
        #
        # That reasoning was already written down, in the module this now
        # imports from. It just was not called from here.
        worst = worst_final_residual(ctx.events)
        if worst is None:
            return _skip(self.code, self.name, "no residual values")

        # **A transient run is judged on stability, by the same function the fix
        # loop uses.** Sixth instance of one fact with two implementations, and
        # the same fact as the fifth: `solve_converged` learned that a
        # time-marching solve's final residual is a function of its timestep, so
        # a residual bar is a steady-state criterion applied where its
        # assumption does not hold — and this check kept its own copy of the old
        # rule.
        #
        # The cost was Courant disagreeing with itself inside one report. On
        # `pisoFoam/RAS/cavity` a CFL repair brought the Courant number to a
        # stable 0.8, the fix loop called it converged, and this graded it C for
        # a residual of 2.22e-02 — a number that is high only because the
        # repaired timestep is 3.4x coarser than the tutorial's own.
        #
        # So it asks rather than re-deciding. The steady branch below is
        # untouched, and a run with no config still gets the residual bar.
        if ctx.config is not None and not ctx.config.is_steady():
            stable, why = solve_converged(ctx.events, config=ctx.config)
            return _result(
                self.code, self.name,
                OK if stable else MAJOR, stable, value=worst,
                detail=(f"transient solve judged on stability rather than a "
                        f"residual threshold — {why} (final worst residual "
                        f"{worst:.2e})"))

        if worst <= self.TARGET:
            sev, passed = OK, True
        elif worst <= self.LOOSE:
            sev, passed = MINOR, True
        else:
            sev, passed = MAJOR, False
        return _result(self.code, self.name, sev, passed, value=worst,
                       threshold=self.TARGET,
                       detail=f"final worst residual {worst:.2e} (target {self.TARGET:.0e})")


class ContinuityCheck:
    """Reports the cumulative continuity error. Deliberately does not grade on it.

    **IT USED TO, AND IT GRADED TWO OF OPENFOAM'S OWN TUTORIALS DOWN.**
    `simpleFoam/pitzDaily` finishes with a cumulative continuity error of
    **1.51**, which against a 1e-03 threshold is MAJOR — so Courant handed a C
    to a case the OpenFOAM project ships as correct. Worse, the repaired run in
    `pitzDaily::unbounded_scheme` reached 1.14 — *better than the tutorial* —
    and was failed for it.

    The number is dimensional and cumulative. It is a running sum in units of
    volumetric flow, so its scale is set by the geometry and the inlet velocity
    and by how many iterations have been run. Measured across this corpus, the
    final cumulative error on healthy tutorials spans **4e-18 (icoFoam/cavity)
    to 1.5 (pitzDaily)** — eighteen orders of magnitude, with nothing wrong in
    any of them. No constant separates those, and picking one that happens to
    fit the cases we have is fitting the corpus rather than the physics.

    This is the same conclusion already reached for `CONV.MASS_IMBALANCE` in the
    taxonomy, and for the same reason; see `docs/COVERAGE.md`. It simply had not
    been applied here, where it was actively costing grades.

    **What does grade conservation is `MassFluxCheck`**, which divides the net
    boundary flux by the inflow and is therefore dimensionless — a real
    imbalance shows up there as a fraction, on any case, at any scale. It needs
    the solution fields, which is the honest cost: without them Courant reports
    the continuity number and does not pretend to judge it.
    """

    code, name = "conservation.continuity", "Global continuity (mass) balance"

    def run(self, ctx: VerifyContext) -> VerificationResult:
        last = ctx.last_residual()
        if last is None or last.payload.continuity is None:
            return _skip(self.code, self.name, "no continuity data")
        cum = last.payload.continuity.cumulative
        if cum is None:
            return _skip(self.code, self.name, "no cumulative continuity error")
        mag = abs(cum)
        # The one judgement that survives being scale-free: a continuity error
        # that is not a number at all means the solve stopped being arithmetic.
        if math.isnan(mag) or math.isinf(mag):
            return _result(self.code, self.name, MAJOR, False, value=None,
                           detail="cumulative continuity error is not finite")
        return _result(self.code, self.name, OK, True, value=mag,
                       detail=f"cumulative continuity error {cum:.2e} "
                              f"(reported, not graded — see MassFluxCheck)")


class TurbulenceBoundsCheck:
    code, name = "bounds.turbulence", "Turbulence quantities not over-clipped"
    TURB = {"k", "epsilon", "omega", "nut"}
    MANY = 10

    def run(self, ctx: VerifyContext) -> VerificationResult:
        # The one definition (verification/bounding.py): only a field that went negative
        # counts, and startup clipping is not judged as persistent. Counting every bounding
        # line made the unmodified T3A tutorial grade C, for 1,001 floor clips at exactly 0.
        from courant.verification.bounding import bounding_by_field, floor_clips_by_field

        by = {f: v for f, v in bounding_by_field(ctx.events).items() if f in self.TURB}
        n = sum(c for c, _ in by.values())
        floors = sum(c for f, c in floor_clips_by_field(ctx.events).items() if f in self.TURB)
        if n == 0:
            detail = "no turbulence clipping" if not floors else (
                f"no negative excursion ({floors} floor clips at or within round-off of zero)")
            return _result(self.code, self.name, OK, True, detail=detail)
        persistent = sorted(f for f, (_, settled) in by.items() if not settled)
        sev = MAJOR if (persistent and n >= self.MANY) else MINOR
        fields = sorted(by)
        return _result(self.code, self.name, sev, sev is not MAJOR, value=float(n),
                       detail=f"{n} bounding excursions on {fields}"
                              + (f", still clipping late on {persistent}" if persistent else
                                 ", all during startup"),
                       evidence={"fields": fields, "persistent": persistent})


# ---------------------------------------------------------------------------
# Field-based checks (live; skip when fields unavailable)
# ---------------------------------------------------------------------------

class MassFluxCheck:
    code, name = "conservation.mass_flux", "Net boundary mass flux ~ 0"
    LOOSE = 1e-2

    def run(self, ctx: VerifyContext) -> VerificationResult:
        if ctx.fields is None or not ctx.fields.patch_fluxes:
            return _skip(self.code, self.name, "no field flux data")
        fluxes = ctx.fields.patch_fluxes
        net = sum(fluxes.values())
        inflow = sum(-f for f in fluxes.values() if f < 0) or 1.0
        rel = abs(net) / inflow
        if rel <= 1e-3:
            sev, passed = OK, True
        elif rel <= self.LOOSE:
            sev, passed = MINOR, True
        else:
            sev, passed = MAJOR, False
        return _result(self.code, self.name, sev, passed, value=rel, threshold=1e-3,
                       detail=f"net/inflow flux imbalance {rel:.2e}")


class YPlusCheck:
    code, name = "nearwall.yplus", "Near-wall y+ compatible with wall treatment"

    def run(self, ctx: VerifyContext) -> VerificationResult:
        if ctx.fields is None or ctx.fields.yplus is None:
            return _skip(self.code, self.name, "no y+ field")
        yplus = np.asarray(ctx.fields.yplus)
        in_log_layer = float(np.mean((yplus >= 30) & (yplus <= 300)))
        if in_log_layer >= 0.8:
            sev, passed = OK, True
        elif in_log_layer >= 0.5:
            sev, passed = MINOR, True
        else:
            sev, passed = MAJOR, False
        return _result(self.code, self.name, sev, passed, value=in_log_layer,
                       threshold=0.8,
                       detail=f"{in_log_layer:.0%} of wall faces in y+ [30,300]")


class RegimeCheck:
    code, name = "regime.reynolds", "Flow regime matches turbulence modeling"
    RE_TURBULENT = 2300.0

    def run(self, ctx: VerifyContext) -> VerificationResult:
        f = ctx.fields
        if f is None or None in (f.inlet_speed, f.length_scale, f.nu) or f.nu == 0:
            return _skip(self.code, self.name, "insufficient data for Reynolds number")
        re = f.inlet_speed * f.length_scale / f.nu
        laminar = ctx.config is not None and ctx.config.is_laminar()
        if laminar and re > self.RE_TURBULENT:
            return _result(self.code, self.name, MAJOR, False, value=re,
                           threshold=self.RE_TURBULENT,
                           detail=f"laminar model at Re={re:.0f} (turbulent regime)")
        if not laminar and re < 1000:
            return _result(self.code, self.name, MINOR, True, value=re,
                           detail=f"turbulence model at low Re={re:.0f}")
        return _result(self.code, self.name, OK, True, value=re,
                       detail=f"Re={re:.0f} consistent with modeling")


class PlausibilityCheck:
    code, name = "bounds.plausibility", "Field values physically plausible"

    def run(self, ctx: VerifyContext) -> VerificationResult:
        f = ctx.fields
        if f is None or not f.internal:
            return _skip(self.code, self.name, "no internal fields")
        # A nan or inf anywhere in a solution field is a blown-up result, as a non-finite
        # residual is (courant.events.diverged). Checked first: `np.min` of an array holding
        # nan is nan, `nan < 0` is False, and a k field of nan read as "k non-negative".
        for name in sorted(f.internal):
            arr = np.asarray(f.internal[name], dtype=float)
            bad = int(arr.size - np.count_nonzero(np.isfinite(arr)))
            if bad:
                return _result(self.code, self.name, MAJOR, False,
                               detail=f"non-finite {name} ({bad} of {arr.size} values "
                                      f"nan or inf)")
        for name in ("k", "epsilon", "omega", "T"):
            arr = f.internal.get(name)
            if arr is not None and np.size(arr) and np.min(np.asarray(arr)) < 0:
                return _result(self.code, self.name, MAJOR, False,
                               value=float(np.min(arr)),
                               detail=f"negative {name} (min {float(np.min(arr)):.2e})")
        # THE VELOCITY-SPIKE TEST IS NOT RUN, AND SAYING SO IS THE POINT.
        #
        # It compared `max|U|` against `f.inlet_speed` — which is the 95th
        # percentile of *that same array*. The ratio of a distribution's maximum
        # to its own 95th percentile sits near 1 whatever the values are, so the
        # 10x threshold could not be reached. Measured on a decomposed cavity run
        # that had diverged: max|U| 14.7 against a lid moving at 1 m/s, a true
        # overspeed of 14.7x, and the check computed 1.27 and passed.
        #
        # A real reference scale is a *boundary* condition, not a statistic of
        # the answer, and deriving one is not a one-line change: the obvious
        # source is `0/U`, and on a lid-driven cavity, a dam break or a heated
        # room that field is uniformly zero, so the naive version divides by
        # nothing on three of the seven certified tutorials. That work needs its
        # own evidence and its own certification run.
        #
        # Until then this check verifies what it can actually verify, and its
        # detail line says which. An affirmative "fields within plausible
        # bounds", printed on the strength of a comparison that cannot fail, is
        # exactly the false reassurance this product exists to remove.
        checked = [n for n in ("k", "epsilon", "omega", "T")
                   if n in f.internal and np.size(f.internal[n])]
        if not checked:
            return _skip(self.code, self.name,
                         "no boundable scalar fields; velocity scale needs a "
                         "boundary reference Courant does not read yet")
        return _result(self.code, self.name, OK, True,
                       detail=f"{', '.join(checked)} non-negative "
                              f"(velocity scale not checked — needs a boundary "
                              f"reference)")


DEFAULT_CHECKS: list[VerificationCheck] = [
    StabilityCheck(), CrashCheck(), ConvergenceCheck(), ContinuityCheck(),
    TurbulenceBoundsCheck(), MassFluxCheck(), YPlusCheck(), RegimeCheck(),
    PlausibilityCheck(),
]


def run_checks(ctx: VerifyContext,
               checks: list[VerificationCheck] | None = None) -> list[VerificationResult]:
    return [c.run(ctx) for c in (checks or DEFAULT_CHECKS)]
