"""Deterministic checkers — Layer 1 of the Diagnosis Engine (Tech Arch §5.1).

Pure functions over a ``CheckContext`` = {config, mesh, events, log}. Each returns
typed Findings with evidence pointers. NO LLM. This layer establishes the hard
facts; the LLM (Layer 2) reasons over them and must cite them.

Checkers fire only on clear signals and attach ``suggests`` (taxonomy codes the
fact is evidence for). They deliberately do not try to be a full classifier — the
LLM does final ranking over the whole Evidence Bundle. Being conservative here
keeps the facts trustworthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from courant.case.config import CaseConfig
from courant.case.schemes import unbounded_convection_schemes
from courant.case.mesh import MeshQuality
from courant.diagnosis.findings import Finding, FindingSet, FindingSeverity
from courant.events import Event, EventKind
from courant.events.diverged import diverged, is_divergence, is_residual_climb
from courant.verification.convergence import (
    RESIDUAL_TARGET,
    courant_limit,
    worst_final_residual,
)

CRIT = FindingSeverity.CRITICAL
WARN = FindingSeverity.WARN
INFO = FindingSeverity.INFO


@dataclass
class CheckContext:
    run_id: str
    events: list[Event]
    config: CaseConfig | None = None
    mesh: MeshQuality | None = None
    log_text: str | None = None

    def events_of(self, kind: EventKind) -> list[Event]:
        return [e for e in self.events if e.kind == kind]


class Checker(Protocol):
    name: str

    def run(self, ctx: CheckContext) -> list[Finding]:
        ...


# ---------------------------------------------------------------------------
# Event-based checkers
# ---------------------------------------------------------------------------


class CourantChecker:
    name = "cfl"
    #: Kept for callers that name it. The limit a run is judged against comes from
    #: `courant_limit(config)` (VER-13), which returns this for a fixed timestep.
    STABILITY_LIMIT = 1.0
    #: A first-step Courant number above this is the initial field, not the
    #: timestep. Five times the stability limit; the corpus separates 0.003 from
    #: 500, so the exact value is not load-bearing and is deliberately not tuned.
    ABSURD_AT_START = 5.0

    def run(self, ctx: CheckContext) -> list[Finding]:
        residuals = ctx.events_of(EventKind.RESIDUAL)
        cos = [(r.payload.courant_max, r.payload.time)
               for r in residuals if r.payload.courant_max is not None]
        # Did the Courant series start with the run? "The first number we saw"
        # and "the number on the first step" are the same thing for every real
        # transient solver, which prints it every step — but not for a log that
        # begins reporting it late, and reading the two as equivalent made
        # `cfl.absurd_at_start` fire on a run whose Courant number climbed
        # normally from a section of log that simply had no Courant lines in it.
        starts_at_run_start = bool(residuals) and (
            residuals[0].payload.courant_max is not None)
        if not cos:
            return []
        max_co, at_time = max(cos, key=lambda t: t[0])
        # VER-13: the limit that applies to *this* run, from the one definition.
        limit, basis = courant_limit(ctx.config)
        if max_co <= limit:
            return []

        # **Did it recover?** The peak alone is not the question.
        #
        # `buoyantPimpleFoam/hotRoom` — an unmodified OpenFOAM tutorial — spikes
        # to Co 25.7 on its second timestep as a 600 K floor patch relaxes into
        # a 300 K room, then decays monotonically to 0.63 and sits there for the
        # remaining 990 steps, converging cleanly. Scored on the bare maximum
        # that is a CRITICAL finding on a case the OpenFOAM project ships as
        # correct — the most expensive mistake this product can make, and it
        # fires on a customer who did nothing wrong.
        #
        # A startup transient that decays is not a stability failure: the solve
        # demonstrably survived it. What matters is whether the run is *still*
        # over the limit once it has settled, which is also exactly what a real
        # blow-up looks like — on the real diverged cavity the tail is where the
        # Courant number is worst, not the head.
        #
        # So severity comes from the tail and the peak is reported alongside it.
        # Deliberately not a smaller change: raising the CRITICAL threshold or
        # ignoring the first N steps would both have silenced this case and both
        # would be arbitrary, where "did it come back down" is the physical
        # question.
        sustained = max(c for c, _ in cos[-max(1, len(cos) // 5):])
        recovered = sustained <= limit
        out = [Finding(
            id="cfl.max", checker=self.name, title="Courant number exceeds stability limit",
            severity=(INFO if recovered else CRIT if max_co > 5 * limit else WARN),
            detail=(f"max Co={max_co:.3g} at t={at_time} exceeds "
                    + (f"limit {limit}" if limit == self.STABILITY_LIMIT else
                       f"{limit:.3g}, {basis} with the controller's slack")
                    + (f", but the run settled back to {sustained:.3g} — a "
                       "startup transient, not a sustained violation"
                       if recovered else
                       f"; still {sustained:.3g} over the final fifth of the run")),
            value=max_co, threshold=limit, location=str(at_time),
            # A transient that recovered explains nothing about a run that then
            # went wrong for another reason, so it stops pointing at a cause.
            suggests=[] if recovered else ["NUM.CFL_VIOLATION"],
        )]

        # **Absurd on the very first step — the initial field, not the timestep.**
        #
        # This is what `icoFoam/cavity` with a corrupted `0/U` actually does:
        # `Courant Number mean: 475 max: 500` on step one, then a floating-point
        # exception inside the linear solver, 58 lines of log, no residuals at
        # all. The knowledge base looked for `residual.nan` and the pattern
        # `Initial residual = (?:inf|nan)`, neither of which OpenFOAM ever
        # prints here — the signature was written from imagination rather than
        # from output, so the scenario was diagnosed NUM.CFL_VIOLATION and the
        # real cause was unreachable.
        #
        # The distinction is physical and the corpus separates on it completely.
        # A CFL violation starts from a sane initial condition and climbs as the
        # solution develops: every other case in the corpus, healthy and blown-up
        # alike, opens at Co 0 (planarContraction at 0.003). A bad initial field
        # is already absurd before the timestep has had any opportunity to
        # matter. 500 against 0.003.
        #
        # Deliberately the FIRST reading and not a maximum over the first few:
        # `buoyantPimpleFoam/hotRoom` opens at 0 and spikes to 25.7 on step two
        # as a 600 K patch relaxes into the room, and that tutorial is correct.
        # Reading "was it wrong before we started" is the only formulation that
        # separates the two.
        first_co = cos[0][0]
        if starts_at_run_start and first_co > self.ABSURD_AT_START:
            out.append(Finding(
                id="cfl.absurd_at_start", checker=self.name,
                title="Courant number already absurd on the first step",
                severity=CRIT, value=first_co, threshold=self.ABSURD_AT_START,
                detail=(f"the first reported Courant number is {first_co:.3g}, "
                        "before the timestep has advanced the solution at all — "
                        "the initial field is already unphysical"),
                suggests=["NUM.BAD_INITIAL_FIELD"],
            ))

        # Courant climbing over time without adaptive dt -> also a control problem
        climbing = max_co > first_co * 3
        adaptive_off = _adaptive_is_off(ctx.config)
        if climbing and (adaptive_off or ctx.config is None):
            # *Which* cause this points at depends on the solver. Only a solver
            # that reads `adjustTimeStep` can be missing it; on icoFoam,
            # pisoFoam or simpleFoam a climbing Courant number means the fixed
            # timestep is simply too large.
            #
            # Getting this wrong ranked NUM.MISSING_ADJUSTABLE_DT (score 9)
            # above NUM.CFL_VIOLATION (score 6) on a real pisoFoam blow-up — a
            # diagnosis the solver cannot have, whose canonical fix is the
            # invalid `adjustTimeStep` advice already removed from the proposer.
            # The same solver-awareness, one layer up.
            solver = ctx.config.solver if ctx.config is not None else None
            adaptive_possible = _honours_adaptive_dt(solver)
            climbing_suggests = (["NUM.MISSING_ADJUSTABLE_DT"] if adaptive_possible
                                 else ["NUM.CFL_VIOLATION"])
            out.append(Finding(
                id="cfl.climbing", checker=self.name,
                title=("Courant number climbing without adaptive timestep"
                       if adaptive_possible
                       else "Courant number climbing on a fixed timestep"),
                # Severity from the tail, like `cfl.max` above. A climb that came
                # back down is the startup transient `cfl.max` already calls INFO.
                # Warning on it here while calling it recovered there was one
                # fact with two answers inside one checker. It raised a WARN on
                # the unmodified hotRoom tutorial (RRD RM-01 v1).
                severity=INFO if recovered else WARN,
                detail=(f"Co rose {first_co:.3g} -> {max_co:.3g}; "
                        + ("adjustTimeStep off/unknown" if adaptive_possible
                           else f"{solver or 'this solver'} has no adaptive timestep")),
                # `value` is the *first* reading, not the peak — cfl.max already
                # carries the peak, and the early one is the more useful number:
                # it describes the flow before divergence corrupted it, so it is
                # what a timestep reduction should be computed from. The fix
                # proposer reads it. Structured rather than left in `detail`,
                # because a consumer that has to parse prose is a consumer that
                # will one day parse it wrongly.
                value=first_co, threshold=limit,
                suggests=[] if recovered else climbing_suggests,
            ))
        return out


class ResidualAnalyzer:
    name = "residual"
    #: Imported, not restated. This was a literal `1e-3` sitting beside
    #: `RESIDUAL_TARGET = 1e-3` in `courant/verification/convergence.py`, which
    #: is the module that exists specifically so this number has one home.
    NOT_CONVERGED = RESIDUAL_TARGET

    def run(self, ctx: CheckContext) -> list[Finding]:
        out: list[Finding] = []
        divs = ctx.events_of(EventKind.DIVERGENCE)
        stalls = ctx.events_of(EventKind.STALL)
        residuals = ctx.events_of(EventKind.RESIDUAL)

        nan_divs = [d for d in divs if is_divergence(d)]
        if nan_divs:
            d = nan_divs[0]
            out.append(Finding(
                id="residual.nan", checker=self.name, title="Residual reached NaN/Inf",
                severity=CRIT, field=d.payload.field, log_offset=d.log_offset,
                detail=f"field {d.payload.field} residual became "
                       f"{'NaN' if d.payload.is_nan else 'Inf'}",
            ))
        slope_divs = [d for d in divs if d.payload.slope is not None]
        if slope_divs:
            d = max(slope_divs, key=lambda e: e.payload.slope)
            out.append(Finding(
                id="residual.growth", checker=self.name, title="Residuals growing monotonically",
                severity=WARN, field=d.payload.field, value=d.payload.slope,
                detail=f"log10-residual slope {d.payload.slope:.2f}/step on {d.payload.field}",
                # Only where over-relaxation is actually a candidate.
                #
                # Growing residuals *are* divergence — naming a cause for them
                # unconditionally makes that cause the default answer for every
                # blow-up. `NUM.OVER_RELAXATION` topped three unrelated
                # pitzDaily faults on this suggestion alone, on a case whose
                # relaxation factors are OpenFOAM's own 0.7.
                #
                # Under-relaxation is what damps the outer loop of a *steady*
                # solver, so it is a real candidate when the loop exists and the
                # factors are high — and not otherwise.
                suggests=_growth_suggests(ctx.config),
            ))
        if any(s.payload.reason == "residual_plateau" for s in stalls):
            out.append(Finding(
                id="residual.plateau", checker=self.name,
                title="Residuals plateaued above tolerance", severity=WARN,
                detail="monitored residual flat while above convergence criterion",
                suggests=["CONV.RESIDUAL_PLATEAU"],
            ))
        elif residuals and not nan_divs and not ctx.events_of(EventKind.CRASH):   # ADP-5
            # A completed-ish run whose final residual is still high — silent
            # non-convergence.
            #
            # **Not on a run that crashed.** "The residuals were above tolerance"
            # is true of every crash ever, and saying it there is restating the
            # failure rather than explaining it. Reported unconditionally it put
            # `CONV.RESIDUAL_PLATEAU` at the top of a missing dictionary, a
            # negative-volume mesh and a wrong turbulence model in one
            # certification run — beating, in the mesh case, a CRITICAL
            # `mesh.negative_volume` that says precisely what is wrong.
            # `worst_final_residual`, not a fourth hand-rolled max().
            #
            # Three places computed this and all three excluded different
            # things. The one in `convergence.py` drops NaN, infinity and phase
            # fractions; this one dropped nothing. On a multiphase run that
            # meant reading `alpha.water` — whose initial residual measures how
            # far the interface MOVED, not how far the solve is from converging,
            # and which never settles because the interface moving is the entire
            # point of the simulation. Every healthy VOF run would have been
            # told its residuals were above tolerance.
            worst = worst_final_residual(ctx.events)
            # **Not on a transient run whose case we can read.** A time-marching
            # solve is judged on stability, not on a residual bar: its final
            # initial-residual is a function of the timestep. That is the single
            # definition in `courant/verification/convergence.py::solve_converged`,
            # and the Courant checks above already cover stability. Applying the
            # 1e-3 bar here anyway was the eighth instance of one fact with two
            # implementations. It raised a WARN on 4 of the 7 unmodified
            # tutorials, all transient, and the verifier's first certification
            # (RRD RM-01 v0) is what found it.
            transient = ctx.config is not None and not ctx.config.is_steady()
            if worst is not None and worst > self.NOT_CONVERGED and not transient:
                out.append(Finding(
                    id="residual.not_converged", checker=self.name,
                    title="Final residual above convergence criterion", severity=WARN,
                    value=worst, threshold=self.NOT_CONVERGED,
                    detail=f"final worst residual {worst:.3g} > {self.NOT_CONVERGED}",
                    suggests=_stall_suggests(ctx.config),
                ))
        return out


#: A linear solver told to stop this far from solved is not solving the
#: equation. OpenFOAM's own tutorials sit at 1e-06/0.05 for pressure; the
#: loosest anything ships with is 1e-04.
LOOSE_TOLERANCE = 1e-4
LOOSE_REL_TOL = 0.2


#: Above this a relaxation factor is doing little to damp the outer loop. Below
#: it, blaming one for a divergence is blaming a number the engineer took from
#: OpenFOAM's own tutorials.
RELAXATION_SUSPECT = 0.8

#: A relaxation factor of 1.0 applies the full correction every iteration —
#: there is no under-relaxation left. Anything below it is a choice OpenFOAM's
#: own tutorials make routinely (`pitzDaily` ships 0.9), so it is not evidence
#: of anything.
RELAXATION_UNDAMPED = 1.0


def _growth_suggests(config) -> list[str]:
    """Whether over-relaxation is a candidate for these growing residuals."""
    if config is None or not config.is_steady():
        return []
    factors = config.relaxation_factors or {}
    hot = any(v is not None and v >= RELAXATION_SUSPECT for v in factors.values())
    return ["NUM.OVER_RELAXATION"] if hot else []


def _stall_suggests(config) -> list[str]:
    """What a run that finished above tolerance most likely means.

    Both `NUM.SOLVER_TOLERANCE` and `CONV.RESIDUAL_PLATEAU` list
    `residual.not_converged` in their signatures, so they scored identically and
    the `suggests` tiebreak decided every one of them — always for the plateau,
    including on a case whose pressure solver had been told to stop at 0.1.

    We can do better than a constant, because the case says which it is. A
    linear solver that stops far from solved *is* the explanation; a properly
    converged inner solve that still leaves the outer residual flat is a
    plateau. Reading the tolerance is the difference between guessing and
    knowing, and it is right there in `fvSolution`.
    """
    solvers = getattr(config, "linear_solvers", None) or {}
    loose = any(
        (s.tolerance is not None and s.tolerance > LOOSE_TOLERANCE)
        or (s.rel_tol is not None and s.rel_tol > LOOSE_REL_TOL)
        for s in solvers.values()
    )
    return ["NUM.SOLVER_TOLERANCE"] if loose else ["CONV.RESIDUAL_PLATEAU"]


from courant.verification.bounding import (  # noqa: E402 - the one definition (bounding.py)
    bounding_by_field,
)


def _bounding_by_field(ctx: CheckContext) -> dict[str, tuple[int, bool]]:
    """Per field: excursions, and whether they all happened during startup.

    Delegates to `courant.verification.bounding`, the one place this is answered. The grade's
    `TurbulenceBoundsCheck` asks the same function, so they cannot disagree.
    """
    return bounding_by_field(ctx.events)


class BoundingChecker:
    name = "bounding"
    TURB_FIELDS = {"k", "epsilon", "omega", "nut"}

    def run(self, ctx: CheckContext) -> list[Finding]:
        by_field = _bounding_by_field(ctx)
        if not by_field:
            return []
        out: list[Finding] = []
        for f, (c, settled) in sorted(by_field.items(),
                                      key=lambda kv: kv[1][0], reverse=True):
            out.append(Finding(
                # THE ID CARRIES THE DISTINCTION, because a knowledge base that
                # keys on `bounding.k` is a third place this question gets
                # answered, and it answered it differently. Startup relaxation
                # and a field still being clipped at the end are different facts
                # and do not get to share a name.
                id=f"bounding.{f}.startup" if settled else f"bounding.{f}",
                checker=self.name, title=f"Field {f} bounded {c}x",
                severity=INFO if settled else (WARN if c >= 3 else INFO),
                field=f, value=float(c),
                detail=(f"{f} clipped {c} times, all while the solve was still "
                        "settling from its initial field — normal startup "
                        "relaxation, not a fault"
                        if settled else
                        f"{f} still being clipped late in the solve ({c} times "
                        "in total)"),
                # Startup relaxation is evidence for nothing. A finding that
                # explains a healthy run explains nothing at all.
                suggests=[] if settled else self._suggests_for(f, ctx.config),
            ))
        return out

    def _suggests_for(self, field: str, config) -> list[str]:
        """What clipping *this* field, on *this* case, is evidence for.

        `["BC.TURB_INLET", "NUM.OVER_RELAXATION"]` for every turbulence field
        regardless of the case was the standing weakness in this file, and it
        cost real rankings: two INFO bounding findings put `BC.TURB_INLET` at
        the top of three unrelated pitzDaily faults, including one whose fault
        was *literally* an unbounded scheme on `div(phi,k)`.

        The case answers the question. If the field's own convection term is
        discretised without bounding, that is the mechanism — central
        differencing on a positive-definite quantity overshoots below zero and
        OpenFOAM clips it. Only when the discretisation is already bounded does
        the inlet condition become the interesting hypothesis, because then
        something else put the field out of range.
        """
        if field not in self.TURB_FIELDS:
            return []
        schemes = getattr(config, "div_schemes", None) or {}
        unbounded = unbounded_convection_schemes(schemes)
        if any(term.lower().endswith(f"{field.lower()})") for term in unbounded):
            return ["NUM.UNBOUNDED_SCHEME"]
        return ["BC.TURB_INLET", "NUM.OVER_RELAXATION"]


#: The scheme vocabulary now lives in `courant/case/schemes.py`.
#:
#: It used to live here, and a second, older copy of the same rule lived in the
#: knowledge base's `unbounded_div` config predicate. When this one was corrected
#: to read `limitedLinear` as bounded, that one was not — so the KB went on
#: scoring `NUM.UNBOUNDED_SCHEME` on every RAS case and put it in the wrong
#: answer of eight of twelve failing certification scenarios. One definition now.


def _adaptive_is_off(config) -> bool:
    """Whether this case is running on a fixed timestep.

    `adjust_timestep is False` is not the same question, and using it cost a
    whole certification run. A solver that cannot adapt has no reason to carry
    an `adjustTimeStep` entry, so `pisoFoam/RAS/cavity` — and most real
    fixed-timestep cases — simply omit it and parse as **None**. The gate never
    fired on exactly the population it was written for.

    The symptom was baffling until the config was printed: making these findings
    solver-aware moved The Bench's top-1 from 67% to 73% and changed nothing at
    all on real cases, because the synthetic fixtures *do* write
    `adjustTimeStep no` explicitly. A fixture diverging from the world.

    So: explicitly off, or absent on a solver that could never have honoured it.
    Absent on a solver that *does* read it is left alone — there the default
    matters and guessing would be a different bug.
    """
    if config is None:
        return False
    if config.adjust_timestep is False:
        return True
    return (config.adjust_timestep is None
            and not _honours_adaptive_dt(config.solver))


def _honours_adaptive_dt(solver: str | None) -> bool:
    """Whether this solver reads `adjustTimeStep` at all.

    Imported lazily to keep `courant.case` out of this module's import graph.
    """
    from courant.case.solvers import honours_adjust_time_step

    return honours_adjust_time_step(solver)


def _unbounded_scheme_symptom(ctx: CheckContext) -> str | None:
    """What in *this run* an unbounded convection scheme could actually explain.

    An unbounded scheme is a **predisposition**, not a fault. `div(phi,U) Gauss
    linear` is what the OpenFOAM project itself ships for `icoFoam/cavity`, and
    it is the right choice there: pure central differencing on a laminar Re=10
    lid-driven cavity. Reported unconditionally, the finding scored
    `NUM.UNBOUNDED_SCHEME` at 6.0 on every case that had one — enough to
    displace the true cause in eight of the twelve scenarios that failed the
    first full certification matrix, including a Courant blow-up, three mesh
    faults and a wrong turbulence model.

    So it is reported only alongside the symptom it actually produces: a
    transported field being **bounded** (clipped) during the solve. That is the
    textbook consequence of an unbounded discretisation overshooting, and it is
    specific — OpenFOAM prints `bounding k, min: -0.003 max: 0.4` precisely when
    a scheme has driven a quantity outside its physical range.

    Two things are deliberately *not* symptoms:

    * **A climbing Courant number.** That is a timestep problem at any
      discretisation, and treating it as scheme evidence is exactly how the
      false ranking arose in the first place.
    * **A residual going non-finite.** This was tried, and it is too weak:
      *every* blow-up ends with a NaN somewhere, so it re-admitted the finding
      on three mesh faults and a bad initial field, where the scheme is
      blameless. Being the last thing to break is not being the cause.

    Same precedent as `scheme.fixed_dt` below — say it only where it is
    actionable. Returns the evidence, so the finding can quote it.
    """
    bounded_fields = sorted(f for f, (_, settled) in _bounding_by_field(ctx).items()
                            if not settled and f != "?")
    if bounded_fields:
        return f"fields still being bounded late in the solve: {', '.join(bounded_fields)}"
    return None


_FATAL_SIGNATURES = [
    # `Entry 'method' not found in dictionary "system/fvSchemes/wallDist"` is one
    # of the most common errors OpenFOAM emits and Courant read nothing from it.
    # It is what a real case does when a chosen model needs configuration the
    # case does not carry — swapping kEpsilon for kOmegaSST asks for a wall
    # distance method, and the run dies naming the exact missing entry. The
    # diagnosis was *empty*: `crash.fatal` fired, no KB entry references it, and
    # the answer to "what went wrong" was nothing at all, on a log that says
    # precisely what went wrong.
    (re.compile(r"Entry '[^']+' not found in dictionary", re.I),
     "ENV.MISSING_DICT", "a required dictionary entry is missing"),
    (re.compile(r"cannot find file|No such file|Cannot open file", re.I),
     "ENV.MISSING_DICT", "missing file/dictionary"),
    (re.compile(r"numberOfSubdomains|processor directories", re.I),
     "ENV.DECOMP_MISMATCH", "decomposition mismatch"),
    # Read from what `blockMesh` and `checkMesh` actually print in v2406, not
    # from what "negative volume" sounds like it should be. On the inverted
    # hexahedron the certification pipeline injects, the real output is
    #
    #     zero or negative pyramid volume: -0.0166667 for face 4
    #     --> FOAM FATAL IO ERROR: (openfoam-2406)
    #     Block hex (0 1 2 3 4 5 6 7) ... is inside-out
    #
    # and the pattern `negative volume` matches neither line — "negative
    # *pyramid* volume" has a word in between, and the fatal error says
    # "inside-out". So the one signature that names this fault could not fire on
    # the fault. `tests/fixtures/real_openfoam/blockMesh_negative_volume.log` is
    # that log, committed, so the next edit here is checked against the solver
    # rather than against a memory of it.
    (re.compile(r"negative (?:cell |pyramid )?volume|inside-out", re.I),
     "MESH.NEGATIVE_VOLUME", "negative cell volume"),
]

#: Reverse flow is **not** a fatal signature and was removed from the list above.
#:
#: OpenFOAM says "reverse flow" in routine places. `pisoFoam/RAS/cavity` — a
#: healthy tutorial — contains
#:
#:     setFlow1:
#:         operating mode: function
#:         reverse flow direction at time: 1
#:
#: which is the `setFlow` function object *announcing its own configuration*.
#: Matching it produced `crash.reversed_outflow` at CRITICAL on a perfectly good
#: run, and because it is critical it then outranked the true cause: a CFL
#: blow-up on that case was diagnosed as a boundary-condition problem.
#:
#: Genuine reverse flow at an outlet is still worth reporting — it is a real
#: taxonomy entry — but it is a *recurring* condition, not a one-line
#: announcement, and it is a warning rather than a crash. Hence the count.
_RE_REVERSE_FLOW = re.compile(r"reverse flow", re.I)

#: Below this, reverse flow is noise: a function-object banner, or a single
#: transient at startup that the solver rode out.
REVERSE_FLOW_MIN_OCCURRENCES = 5


class CrashSignatureChecker:
    name = "crash"

    def run(self, ctx: CheckContext) -> list[Finding]:
        crashes = ctx.events_of(EventKind.CRASH)
        log = ctx.log_text or ""
        out: list[Finding] = []

        # Only look for *why it crashed* when it actually crashed. These patterns
        # name fatal conditions, and scanning the whole log unconditionally meant
        # a healthy run could be handed a CRITICAL crash finding — which is both
        # wrong and, being critical, loud enough to outrank the real cause.
        if crashes:
            for rx, code, label in _FATAL_SIGNATURES:
                if rx.search(log):
                    out.append(Finding(
                        id=f"crash.{code.split('.')[-1].lower()}", checker=self.name,
                        title=f"Fatal signature: {label}", severity=CRIT,
                        detail=f"log matched {label!r}", suggests=[code],
                    ))

        # Reverse flow, counted rather than matched once — see the note above.
        reversals = len(_RE_REVERSE_FLOW.findall(log))
        if reversals >= REVERSE_FLOW_MIN_OCCURRENCES:
            out.append(Finding(
                id="flow.reversed_outflow", checker=self.name,
                title="Persistent reverse flow at a boundary", severity=WARN,
                value=float(reversals),
                detail=f"reverse flow reported {reversals} times",
                suggests=["BC.REVERSED_OUTFLOW"],
            ))
        if crashes and not out:
            c = crashes[0]
            out.append(Finding(
                id="crash.fatal", checker=self.name, title="Solver crashed",
                severity=CRIT, log_offset=c.log_offset,
                detail=f"solver exited abnormally (tag={c.payload.tag})",
            ))
        # dedupe by id (a signature may match twice)
        seen: dict[str, Finding] = {}
        for f in out:
            seen.setdefault(f.id, f)
        return list(seen.values())


# ---------------------------------------------------------------------------
# Static (config / mesh) checkers
# ---------------------------------------------------------------------------


class MeshChecker:
    name = "mesh"

    def run(self, ctx: CheckContext) -> list[Finding]:
        m = ctx.mesh
        if m is None:
            return []
        out: list[Finding] = []
        if m.n_negative_volume_cells and m.n_negative_volume_cells > 0:
            out.append(Finding(
                id="mesh.negative_volume", checker=self.name, title="Negative cell volumes",
                severity=CRIT, value=float(m.n_negative_volume_cells),
                detail=f"{m.n_negative_volume_cells} cell(s) with negative volume",
                suggests=["MESH.NEGATIVE_VOLUME"],
            ))
        if m.max_non_orthogonality is not None and m.max_non_orthogonality > 70:
            corr = ctx.config.n_non_orthogonal_correctors if ctx.config else None
            out.append(Finding(
                id="mesh.non_orthogonality", checker=self.name,
                title="High mesh non-orthogonality", severity=WARN,
                value=m.max_non_orthogonality, threshold=70.0,
                detail=f"max non-orthogonality {m.max_non_orthogonality:.1f} deg, "
                       f"nNonOrthogonalCorrectors={corr}",
                suggests=["MESH.SKEWNESS"],
            ))
        if m.max_skewness is not None and m.max_skewness > 4:
            out.append(Finding(
                id="mesh.skewness", checker=self.name, title="High mesh skewness",
                severity=WARN, value=m.max_skewness, threshold=4.0,
                detail=f"max skewness {m.max_skewness:.2f}", suggests=["MESH.SKEWNESS"],
            ))
        if m.max_aspect_ratio is not None and m.max_aspect_ratio > 1000:
            out.append(Finding(
                id="mesh.aspect_ratio", checker=self.name, title="Extreme aspect ratio",
                severity=WARN, value=m.max_aspect_ratio, threshold=1000.0,
                detail=f"max aspect ratio {m.max_aspect_ratio:.0f}",
                suggests=["MESH.ASPECT_RATIO"],
            ))
        return out


class SchemeToleranceChecker:
    name = "scheme"

    def run(self, ctx: CheckContext) -> list[Finding]:
        c = ctx.config
        if c is None:
            return []
        out: list[Finding] = []

        # A linear solver told to stop far from solved — read from the case,
        # not from the log.
        #
        # **This is the silent-error class, and nothing else covered it.** On
        # `pisoFoam/RAS/cavity` with the pressure solver set to `tolerance 0.1`
        # the run completes, reaches a final residual of 8.55e-03 and is graded
        # A. Every log-based checker is therefore quiet and correct to be quiet
        # — the run really did converge — so the orchestrator's
        # `_physics_is_suspect` gate returned before diagnosing and Courant said
        # nothing at all about a pressure equation being solved to one part in
        # ten. The product claims it finds "failures *and* physically wrong
        # results"; a wrong result that converges is exactly the second half.
        #
        # Safe against the tutorials by a clear margin, measured rather than
        # assumed: across all seven healthy cases in the corpus the loosest
        # tolerance anyone ships is **1e-05** and the loosest relTol is **0.1**,
        # against thresholds of 1e-04 and 0.2. The two faulted cases sit at 0.1
        # and 0.01.
        solvers = getattr(c, "linear_solvers", None) or {}
        slack = {
            field: s for field, s in solvers.items()
            if (s.tolerance is not None and s.tolerance > LOOSE_TOLERANCE)
            or (s.rel_tol is not None and s.rel_tol > LOOSE_REL_TOL)
        }
        if slack:
            worst_field, worst = max(
                slack.items(), key=lambda kv: kv[1].tolerance or 0.0)
            out.append(Finding(
                id="solver.loose_tolerance", checker=self.name,
                title="Linear solver tolerance far looser than any tutorial ships",
                severity=WARN, field=worst_field, value=worst.tolerance,
                threshold=LOOSE_TOLERANCE,
                detail=(f"{', '.join(sorted(slack))} solved to "
                        f"tolerance {worst.tolerance!r} / relTol "
                        f"{worst.rel_tol!r}; OpenFOAM's own tutorials use "
                        f"1e-06 to 1e-05"),
                suggests=["NUM.SOLVER_TOLERANCE"],
            ))
        # Relaxation with nothing left to damp — and **only on a steady solver**.
        #
        # The threshold was 0.9 with no solver gate, and both halves were wrong
        # against OpenFOAM's own tutorials. `simpleFoam/pitzDaily` ships
        # `equations { U 0.9; ".*" 0.9; }` and converges perfectly well;
        # `interFoam/damBreak` ships `".*" 1.0`, which is harmless because a
        # transient PIMPLE loop with `nOuterCorrectors 1` has no outer iteration
        # to relax. So this fired on two unmodified tutorials, and because
        # `NUM.OVER_RELAXATION` then scored 6.0 — finding, config predicate and
        # suggestion together — it topped three unrelated pitzDaily faults whose
        # relaxation was exactly what the tutorial shipped.
        #
        # 0.9 is a number OpenFOAM's own authors choose. **1.0 is the one that
        # means something**: it removes the damping entirely, which is precisely
        # what the `over_relaxation` fault injects and what an engineer does by
        # accident. And on a transient solve the setting is inert, so saying
        # anything about it there is noise.
        hot = ({f: v for f, v in c.relaxation_factors.items()
                if v is not None and v >= RELAXATION_UNDAMPED}
               if c.is_steady() else {})
        if hot:
            worst_f, worst_v = max(hot.items(), key=lambda kv: kv[1])
            out.append(Finding(
                id="scheme.relaxation", checker=self.name,
                title="Relaxation factor(s) too high", severity=WARN,
                field=worst_f, value=worst_v, threshold=0.9,
                detail=f"relaxationFactors { {k: round(v,3) for k,v in hot.items()} }",
                suggests=["NUM.OVER_RELAXATION"],
            ))
        # unbounded divergence scheme on a convection term
        unbounded = unbounded_convection_schemes(c.div_schemes)
        symptom = _unbounded_scheme_symptom(ctx)
        if unbounded and symptom:
            out.append(Finding(
                id="scheme.unbounded_div", checker=self.name,
                title="Unbounded divergence scheme on transported field", severity=WARN,
                detail=f"div schemes without bounding: {unbounded}; {symptom}",
                suggests=["NUM.UNBOUNDED_SCHEME"],
            ))
        # Fixed timestep with no adaptivity — but only worth saying for a solver
        # that *could* have adapted. icoFoam, pisoFoam and simpleFoam have no
        # other mode, so reporting it there is noise that also mis-suggests
        # NUM.MISSING_ADJUSTABLE_DT, a cause those solvers cannot have.
        if (_adaptive_is_off(c) and not c.is_steady()
                and c.delta_t is not None and _honours_adaptive_dt(c.solver)):
            out.append(Finding(
                id="scheme.fixed_dt", checker=self.name,
                title="Fixed timestep without adjustTimeStep", severity=INFO,
                value=c.delta_t, detail=f"deltaT={c.delta_t}, adjustTimeStep off",
                suggests=["NUM.MISSING_ADJUSTABLE_DT"],
            ))
        return out


class BoundaryConfigChecker:
    name = "bc"

    def run(self, ctx: CheckContext) -> list[Finding]:
        c = ctx.config
        if c is None:
            return []
        # over-constrained p-U: patches fixing both p and U (aggregated into one finding)
        bad = [patch for patch, fields in c.boundary.items()
               if fields.get("p") == "fixedValue" and fields.get("U") == "fixedValue"]
        if not bad:
            return []
        return [Finding(
            id="bc.overconstrained", checker=self.name,
            title="Over-constrained pressure-velocity coupling", severity=WARN,
            location=",".join(bad),
            detail=f"patches fix both p and U (fixedValue/fixedValue): {bad}",
            suggests=["BC.OVERCONSTRAINED_PU"],
        )]


class PhysicsChecker:
    name = "physics"

    def run(self, ctx: CheckContext) -> list[Finding]:
        c = ctx.config
        if c is None:
            return []
        out: list[Finding] = []
        nu = c.transport.get("nu")
        if nu is not None and (nu <= 0 or nu > 1e-2 or nu < 1e-8):
            out.append(Finding(
                id="physics.nu", checker=self.name, title="Suspicious viscosity value",
                severity=WARN, value=nu,
                detail=f"nu={nu:g} m^2/s is outside typical fluid range [1e-8, 1e-2]",
                suggests=["PHYS.TRANSPORT_UNITS"],
            ))
        # A laminar model on a run that will not settle.
        #
        # Two signatures, because turbulence modelled as laminar fails in two
        # ways and the checker only knew one. It can blow up — residuals growing
        # with a slope — and that was covered. Far more often it simply *never
        # converges*: the flow is genuinely turbulent, the laminar equations have
        # no steady solution for it, and the residuals sit above tolerance
        # forever without ever misbehaving badly enough to look like divergence.
        #
        # Missing the second one had a specific cost. `PHYS.LAMINAR_TURBULENT`
        # was left scoring on its config predicate alone, with no Finding behind
        # it — so on the one fixture where it ranked first it cited no evidence,
        # which is the Phase-1 definition of done ("every diagnosis cites at
        # least one real Finding") failing quietly rather than loudly.
        # ...and the discriminator that stops it firing on every laminar case
        # that ever failed: **there is no Courant violation**.
        #
        # A laminar solve that went unstable and a laminar solve whose timestep
        # was too large look identical in the residuals — both grow, both may
        # crash. What separates them is whether the flow outran the timestep.
        # If the Courant number breached its stability limit, the timestep is
        # the explanation and the turbulence model is a bystander. If it did
        # not, then the numerics were fine and the physics is the remaining
        # candidate.
        #
        # That is what makes this a finding rather than an observation about
        # `simulationType`, and without it the check would fire on every
        # `icoFoam` blow-up in the corpus.
        oscillating = any(is_residual_climb(e) for e in ctx.events_of(EventKind.DIVERGENCE))
        stalled_high = _finished_above_tolerance(ctx)
        died = bool(ctx.events_of(EventKind.CRASH))
        if (c.is_laminar() and (oscillating or stalled_high or died)
                and not _courant_breached(ctx)):
            out.append(Finding(
                id="physics.laminar_unstable", checker=self.name,
                title="Laminar model with unstable residuals",
                # WARN when the run failed outright with the timestep in range:
                # a laminar model on a turbulent flow is then the likely reason,
                # not a footnote. INFO when it is only oscillating, which many
                # things cause.
                severity=WARN if (stalled_high or died) else INFO,
                detail="laminar turbulence type, Courant number within its "
                       "stability limit, and the solve "
                       + ("terminated abnormally" if died
                          else "never reached tolerance" if stalled_high
                          else "oscillates/grows"),
                suggests=["PHYS.LAMINAR_TURBULENT"],
            ))
        return out


def _finished_above_tolerance(ctx: CheckContext) -> bool:
    """Whether the run ran to the end with its residuals still above tolerance.

    Same condition `residual.not_converged` uses, read from the events directly
    because checkers do not see each other's findings.
    """
    if ctx.events_of(EventKind.CRASH) or diverged(ctx.events):         # ADP-5: one definition
        return False
    residuals = ctx.events_of(EventKind.RESIDUAL)
    if not residuals:
        return False
    worst = worst_final_residual(ctx.events)
    return worst is not None and worst > ResidualAnalyzer.NOT_CONVERGED


def _courant_breached(ctx: CheckContext) -> bool:
    """Whether any timestep exceeded the limit that applies to this run (VER-13)."""
    limit, _ = courant_limit(ctx.config)
    return any(
        e.payload.courant_max is not None and e.payload.courant_max > limit
        for e in ctx.events_of(EventKind.RESIDUAL)
    )


class CaseIntegrityChecker:
    """Say when we could not read the case, rather than reasoning as if we had.

    Every other config checker starts `if c is None: return []` or reads a field
    that is `None`, so a case whose `controlDict` failed to parse produces
    *silence* — indistinguishable from a case that is fine. That is the worst
    available outcome: the engineer gets a diagnosis built on a config we never
    actually read, with nothing anywhere saying so.

    Reachable on real cases. `foamDictionary -set` rewrites a dictionary with
    its `#include` directives expanded, and OpenFOAM's own `pisoFoam/RAS/cavity`
    expands into `fields ( U { mean on; ... } );` — a list containing an inline
    dictionary, which foamlib cannot parse. Any case an engineer has edited with
    OpenFOAM's own tool can arrive in this state.

    WARN, not CRITICAL: the run itself may be perfectly healthy, and a healthy
    tutorial must never produce a CRITICAL finding. What is wrong is our reading
    of it, and that is what the finding says.
    """

    name = "case"

    def run(self, ctx: CheckContext) -> list[Finding]:
        unreadable = getattr(ctx.config, "unreadable", None) if ctx.config else None
        if not unreadable:
            return []
        return [Finding(
            id="case.unreadable_dict", checker=self.name,
            title="Case dictionary could not be parsed", severity=WARN,
            value=float(len(unreadable)),
            detail="; ".join(f"{f}: {why}" for f, why in sorted(unreadable.items()))
                   + " — findings that depend on these files are unavailable, "
                     "not absent",
        )]


DEFAULT_CHECKERS: list[Checker] = [
    CourantChecker(), ResidualAnalyzer(), BoundingChecker(), CrashSignatureChecker(),
    MeshChecker(), SchemeToleranceChecker(), BoundaryConfigChecker(), PhysicsChecker(),
    CaseIntegrityChecker(),
]


def run_all_checkers(ctx: CheckContext,
                     checkers: list[Checker] | None = None) -> FindingSet:
    """Run every checker in deterministic order; collect id-unique Findings."""
    fs = FindingSet()
    for checker in (checkers or DEFAULT_CHECKERS):
        for finding in checker.run(ctx):
            if finding.id not in fs.ids:
                fs.add(finding)
    return fs
