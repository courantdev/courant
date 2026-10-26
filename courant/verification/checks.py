"""Verification checks — deterministic physics audit (Tech Arch §7).

Two tiers:
  * event/residual-based — run offline on the Watcher's history (divergence,
    convergence quality, continuity/mass balance, turbulence bounds),
  * field-based — need solution fields (conservation flux, y+, regime, plausibility);
    they run live and SKIP (not fail) when fields are unavailable. The y+ and regime checks
    have no live input at all (`YPLUS_NOT_IMPLEMENTED`, `REYNOLDS_NOT_IMPLEMENTED`).

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
    judge_residuals,
    judged_on_the_bar,
    solve_converged,
    stall_sentence,
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
    #: `--checks` (TRD GATE-13): "next" also runs what `courant.verify.checks.NEXT` holds back.
    checks: str = "default"
    #: The solver that ran, as the verdict names it (the log's `Exec` line, else the run's
    #: record; `courant.verify.pipeline._solver`). `config.solver` is controlDict's
    #: `application`, which a copied case may not have run. None: not known beyond `config`.
    solver: str | None = None
    #: TRD VER-16b: the velocity components the latest U shows to be zero to machine precision
    #: (`convergence.zero_components`), left out of the residual bar, as `CheckContext` has them.
    zero_components: tuple[str, ...] = ()

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
        #
        # "Transient" means time-accurate here: under `--checks next` a local-time-stepping run
        # marches to a steady state and is judged on its residuals (TRD VER-19), and a v2406
        # steady solver whose `ddtSchemes` was not read is steady (VER-21). One function decides
        # it for every asker of the bar (`judged_on_the_bar`: `residual.not_converged` and
        # VER-16b's exclusion too; the review of T-WEDGE-MESH).
        if not judged_on_the_bar(ctx.config, ctx.checks):
            stable, why = solve_converged(ctx.events, config=ctx.config)
            return _result(
                self.code, self.name,
                OK if stable else MAJOR, stable, value=worst,
                detail=(f"transient solve judged on stability rather than a "
                        f"residual threshold — {why} (final worst residual "
                        f"{worst:.2e})"))

        # The steady bar is per field and decided once, in `judge_residuals`, which
        # `ResidualAnalyzer` also asks (TRD VER-16; RRD RM-35, S's decision of 13 Oct 2026):
        # each field against the `residualControl` tolerance its solver reads, the flat TARGET
        # where there is none, and the stall detector beside it. On the flat bar this check and
        # `residual.not_converged` already shared one threshold (OK here exactly when no
        # finding there); asking one function keeps that true per field. MAJOR stays where it
        # was, above LOOSE, because no declared tolerance moves a bar past LOOSE.
        judged = judge_residuals(ctx.events, ctx.config, ctx.checks,
                                 exclude=ctx.zero_components)
        # The judgement's own worst: the solve each field is read at is decided there (the
        # first, under VER-16a). Under the default it is `worst_final_residual`'s number.
        worst = judged.worst
        if judged.worst > self.LOOSE:
            sev, passed = MAJOR, False
        elif judged.over or judged.stall is not None:
            sev, passed = MINOR, True
        else:
            sev, passed = OK, True
        if not judged.per_field:
            detail = f"final worst residual {worst:.2e} (target {self.TARGET:.0e})"
            value, threshold = worst, self.TARGET
        else:
            value, threshold = judged.residual, judged.bar
            detail = (f"final residual {judged.residual:.2e} on {judged.field}, "
                      f"{'above' if judged.over else 'within'} {judged.bar_words()}"
                      + ("" if judged.over else "; every field within its own bar"))
        if judged.stall is not None:
            detail += f"; stalled: {stall_sentence(judged.stall)}"
        return _result(self.code, self.name, sev, passed, value=value,
                       threshold=threshold, detail=detail)


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

def _balance_not_owed(fields: FieldData, config: CaseConfig | None,
                      ran: str | None = None, checks: str = "default") -> str | None:
    """Why this run's boundary flux need not balance, or None when it must (TRD VER-18).

    Net flux out of a fixed domain is minus the rate its contents change. That is zero for a
    divergence-free volume flux at every step (an incompressible solver,
    `courant.case.solvers.SOLENOIDAL_SOLVERS`), and for any flux once the run is steady. A
    transient compressible run's mass flux balances the change of mass inside, which is not
    read. Which solver's run it is comes from `ran`, the solver the verdict names, and else
    from controlDict's `application`; when they name different solvers the case was set up for
    another one, and nothing is judged. Data built without a source (`flux_source` None)
    carries no such facts and is judged as given. Steady is `CaseConfig.is_steady` under
    `checks`, as the residual bar asks it in the same verdict (under `next` the solver fallback
    knows v2406's own steady solvers; the final review of 0.6.0 found this asking the default)."""
    from courant.case.solvers import keeps_volume_flux_solenoidal, same_solver

    src = fields.flux_source
    if src is None:
        return None
    if config is None:
        return (f"{src.file} was read, and the case configuration was not, so whether the run "
                f"must balance (steady, or a divergence-free volume flux) is unknown")
    if ran and config.solver and not same_solver(ran, config.solver):
        return (f"the solver that ran is {ran} and system/controlDict names {config.solver}: "
                f"on a case set up for another solver, whether {src.file} must balance is not "
                f"judged")
    solver = ran or config.solver
    if config.is_steady(checks):
        return None
    if src.kind == "mass":
        return (f"{src.file} holds a mass flux and the run is transient: the flux through its "
                f"boundary balances the change of mass inside the domain, which is not read")
    if not keeps_volume_flux_solenoidal(solver):
        return (f"{src.file} holds the volume flux of a transient run of "
                f"{solver or 'an unnamed solver'}, which is not known to keep it "
                f"divergence-free")
    return None


class MassFluxCheck:
    """The boundary of a run that must conserve mass carries as much out as in (TRD VER-18).

    The measure is the net flux through the domain's boundary over the flux into it, from the
    `phi` the solver wrote (`courant.verification._fluid_reader`). A compressible solver writes
    a mass flux, an incompressible one a volume flux; the ratio is the same in either unit, so
    the bars hold for both. With no inflow above round-off, the outflow is the denominator: all
    of it is imbalance. A boundary whose flux in and out are both round-off next to the largest
    face flux in the file is a closed domain: nothing crosses it, so nothing is judged. A
    non-finite flux on any face, interior or boundary, is MAJOR.

    It reads only what `courant verify --checks next` reads (GATE-13): under the default the
    reader leaves `patch_fluxes` empty and this check skips with v0.5.0's words.
    """

    code, name = "conservation.mass_flux", "Net boundary mass flux ~ 0"
    TARGET, LOOSE = 1e-3, 1e-2
    #: A boundary whose inflow and outflow are both at most this fraction of the largest face
    #: flux in the file carries none: a wall's flux is zero to about 1e-16 of the flow in double
    #: precision, and a real inlet carries a fraction of the flow far above this.
    ROUND_OFF = 1e-9

    def run(self, ctx: VerifyContext) -> VerificationResult:
        f = ctx.fields
        if f is None or not f.patch_fluxes:
            return _skip(self.code, self.name,
                         (f.flux_note if f is not None else None) or "no field flux data")
        why = _balance_not_owed(f, ctx.config, ctx.solver, ctx.checks)
        if why:
            return _skip(self.code, self.name, why)
        src = f.flux_source
        evidence = {"input": f"{src.file} (boundary face fluxes)"} if src is not None else {}
        unit = {"mass": " kg/s", "volume": " m^3/s"}.get(getattr(src, "kind", None), "")
        net = sum(f.patch_fluxes.values())
        if src is not None:
            inflow, outflow, largest = src.inflow, src.outflow, src.largest
        else:
            inflow = sum(-v for v in f.patch_fluxes.values() if v < 0)
            outflow = sum(v for v in f.patch_fluxes.values() if v > 0)
            largest = 0.0
        if not all(math.isfinite(x) for x in (net, inflow, outflow)):
            return _result(self.code, self.name, MAJOR, False, evidence=evidence,
                           detail="the boundary flux is not finite (nan or inf in phi)")
        if not math.isfinite(largest):          # an interior face's: it is the round-off scale
            return _result(self.code, self.name, MAJOR, False, evidence=evidence,
                           detail=f"an interior face flux in {src.file} is not finite "
                                  f"(nan or inf)")
        # Closed only when both directions are round-off; otherwise the denominator is the
        # inflow, or the outflow when no real flux comes in (a wall's round-off is not inflow).
        tiny = self.ROUND_OFF * largest
        scale, through = (inflow, "inflow") if inflow > tiny else (outflow, "outflow")
        if inflow <= tiny and outflow <= tiny:
            where = f" in {src.file}" if src is not None else ""
            return _skip(self.code, self.name,
                         f"no flow crosses the boundary{where} (a closed domain): there is no "
                         f"balance to judge")
        rel = abs(net) / scale
        if rel <= self.TARGET:
            sev, passed, bar = OK, True, self.TARGET
        elif rel <= self.LOOSE:
            sev, passed, bar = MINOR, True, self.TARGET
        else:
            sev, passed, bar = MAJOR, False, self.LOOSE
        detail = f"net/inflow flux imbalance {rel:.2e}"
        if src is not None:
            detail = (f"net boundary flux {net:.3g}{unit} against an {through} of "
                      f"{scale:.3g}{unit}: out of balance by {rel:.2e} of the {through} "
                      f"({'above' if rel > bar else 'within'} {bar:.0e})")
        return _result(self.code, self.name, sev, passed, value=rel, threshold=bar,
                       detail=detail, evidence=evidence)


#: TRD VER-7c (hard-proofing audit, 21 Oct 2026): no reader in the product sets `FieldData.yplus`
#: or `FieldData.length_scale`; only the unit tests supply them. So these two checks skip on
#: every real case, and their skip lines said "no y+ field" on a pitzDaily run that had written
#: `282/yPlus`, as if the case lacked it. Each now says it is not implemented. Feeding them is a
#: new check behind `--checks next` (GATE-13), pre-registered first: fed that run's 473 wall-face
#: values (0.24 to 26.3), the flat [30, 300] band below calls the unmodified tutorial MAJOR, where
#: docs/specs/spec-m3.md R02/R03 judge each patch by its wall treatment and the solver's yPlusLam.
YPLUS_NOT_IMPLEMENTED = ("not implemented (Courant reads no y+, neither a yPlus field nor the "
                         "yPlus function object's output)")
REYNOLDS_NOT_IMPLEMENTED = ("not implemented (Courant derives no length scale for a Reynolds "
                            "number from the case)")


class YPlusCheck:
    code, name = "nearwall.yplus", "Near-wall y+ compatible with wall treatment"

    def run(self, ctx: VerifyContext) -> VerificationResult:
        if ctx.fields is None or ctx.fields.yplus is None:
            return _skip(self.code, self.name, YPLUS_NOT_IMPLEMENTED)
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
        if f is None or f.length_scale is None:
            return _skip(self.code, self.name, REYNOLDS_NOT_IMPLEMENTED)
        if None in (f.inlet_speed, f.nu) or f.nu == 0:
            return _skip(self.code, self.name, "insufficient data for Reynolds number")
        re = f.inlet_speed * f.length_scale / f.nu
        laminar = ctx.config is not None and ctx.config.is_laminar(ctx.checks)
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


#: TRD VER-7b, behind `--checks next` (`fields.unread`): not one of `DEFAULT_CHECKS`, whose
#: results every caller lists; the gate adds it to its own report under next.
FIELDS_UNREAD = ("fields.unread", "Solution fields at the time judged read")


def fields_unread(read: FieldData | None) -> VerificationResult | None:
    """A minor check (grade B at most) when a field the run wrote at the time the field reader
    judged exists and was not read, or a decomposed case's field was read on some ranks only
    (`FieldData.unread`, `FieldData.partial`); None otherwise. The field checks skip, or judge
    part of the domain, so an accept at grade A would say more than was checked (the review of
    21 Oct 2026: a field cut mid-write was accepted at grade A in both modes)."""
    if read is None or read.source_time is None or not (read.unread or read.partial):
        return None
    from courant.verify.sentences import fields_unread_reason

    names = [*read.unread, *(name for name, _, _ in read.partial)]
    return _result(*FIELDS_UNREAD, MINOR, False,
                   detail=fields_unread_reason(read.source_time, read.unread, read.partial),
                   evidence={"input": f"time directory {read.source_time} ({', '.join(names)})"})
