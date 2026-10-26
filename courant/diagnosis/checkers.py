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

from courant.case.config import PRESSURE_FIXING, CaseConfig
from courant.case.schemes import unbounded_convection_schemes
from courant.case.mesh import SKEW_THRESHOLD, MeshQuality
from courant.case.viscosity import NU_MAX, NU_MIN, Viscosity, viscosities
from courant.case.regime import (
    DNS_SPACING,
    RA_TRANSITION,
    buoyant_regime,
    solver_read_gravity,
)
from courant.diagnosis.findings import Finding, FindingSet, FindingSeverity
from courant.events import Event, EventKind
from courant.events.diverged import diverged, is_divergence, is_residual_climb
from courant.verification.convergence import (
    RESIDUAL_TARGET,
    courant_limit,
    judge_residuals,
    judged_on_the_bar,
    stall_sentence,
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
    #: `--checks` (TRD GATE-13): "next" also runs what `courant.verify.checks.NEXT` holds back
    #: for one minor release (VER-16's bar per field and stall, VER-17's regime check).
    checks: str = "default"
    #: TRD VER-16b: the velocity components the latest U shows to be zero to machine precision
    #: (`convergence.zero_components`), left out of the residual bar. () without fields.
    zero_components: tuple[str, ...] = ()

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
        # **Not on a declared steady solve, under `--checks next`** (TRD VER-13a, `cfl.steady`):
        # its Courant number is pseudo-time. The one number scalarTransportFoam prints before a
        # steadyState loop gave a healthy heated pipe two CRITICALs (an internal evaluation, 22
        # Oct 2026; the committed fixture's is 95.2, `tests/fixtures/real_openfoam/wedge_pipe/
        # energy`). Under the default every run is judged as v0.5.0 judged it; under next so are
        # local time stepping, every transient run, a case whose `ddtSchemes` was not read whole
        # by the case reader, a run whose log does not show the default region from its start
        # and one whose log says the solver read `fvSchemes` again. The file is read as it is
        # when verified: an edit after the run leaves nothing in the log, and is not checked.
        if steady_courant(ctx.config, ctx.log_text, ctx.checks) is not None:
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
        adaptive_off = _adaptive_is_off(ctx.config, ctx.checks)
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
            adaptive_possible = _honours_adaptive_dt(solver, ctx.checks)
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


#: The one `ddtSchemes` entry VER-13a takes as no time derivative, as v2406 writes it.
STEADY_DDT = "steadyState"


def steady_courant(config, log_text: str | None = None, checks: str = "default") -> str | None:
    """Why a Courant number the log prints decides nothing on this run, or None (TRD VER-13a).

    **Under `--checks next` only** (`cfl.steady` in `courant.verify.checks.NEXT`): under the
    default this is None for every run, so every `cfl.*` check and `physics.laminar_unstable`'s
    Courant condition are v0.5.0's (GATE-13; every review of VER-13a found a new false accept,
    and its one measured benefit is one false alarm).

    Only the case's own declaration, read from its own `system/fvSchemes` by the case reader,
    every entry of `ddtSchemes`, where foamlib's reading of the file is the solver's
    (`CaseConfig.ddt_read_whole`, `_foam_reader._ddt_read_whole`: no directive, macro,
    slash-scoped keyword or `select` in the file, no `constant/regionProperties`, `ddtSchemes`
    and each of its keywords written once); not a configuration a sidecar, a benchmark's
    `case_facts.json` or an MCP `case_json` states (`CaseConfig.from_view`).

    The file as it is when verified: nothing shows that the solver read this content, since
    v2406 prints no scheme it reads. All a log can show of it is the solver reading a `fvSchemes`
    again during the run (`watcher.parser.reread_fvschemes`, v2406's `Re-reading object
    fvSchemes`), and such a log is judged: its steps before that line ran on the file as it was
    then (the review of 5e2a8d5, a pimpleFoam log of Euler steps at a sustained Co 8 beside a
    steady file). A file edited after the run, or after the last step the log holds, leaves no
    line in it, and is read as edited: that is not checked (TRD VER-13a).

    And only on a run whose log shows the case's default region from its start
    (`watcher.parser.default_region_run`): the log holds the run's command line and its start, and
    neither names a region. A solver run with
    `-region <name>` reads `system/<name>/fvSchemes` (schemesLookup.C's `IOobject(dictName,
    obr.time().system(), obr, ...)`, whose path is `instance()/db_.dbDir()`, IOobject.C, and a
    region's `dbDir()` is its name, polyMesh.C), and pimpleFoam prints "Create mesh for time"
    whatever its region (createDynamicFvMesh.H), so the command line is what shows it (the review
    of hp6/final: `pimpleFoam -region fluid` beside a stale steady top-level `fvSchemes` was
    spared). Nor on a multi-region run: chtMultiRegionFoam builds its regions' meshes alone
    (createMeshes.H, fluid/createFluidMeshes.H), so a case that names regions
    (`CaseConfig.regions`, which only a view sets) or a log that shows them
    (`watcher.parser.names_regions`: its region headers, its `Region: <name> Courant Number`
    lines, which are the only Courant numbers a CHT log prints) is judged; the case reader
    answers no to a case holding `constant/regionProperties`.

    The declaration: `ddtSchemes default steadyState`, with every other entry of
    `ddtSchemes` it writes steadyState too, since v2406 looks a field's own `ddt(T)` up before
    the default (`schemesLookup::lookupDetail::lookup`, src/OpenFOAM/matrices/schemes/
    schemesLookupDetail.C). `steadyStateDdtScheme::fvmDdt` then returns an empty matrix (src/
    finiteVolume/finiteVolume/ddtSchemes/steadyStateDdtScheme/steadyStateDdtScheme.C), so no
    equation holds the time step, and the number `CourantNo.H` prints, 0.5 sum|phi|/V deltaT
    (src/finiteVolume/cfdTools/incompressible/CourantNo.H), is the flux against a pseudo-step.

    Not the solver tables, which `CaseConfig.is_steady` falls back on when `fvSchemes` was not
    read: they name controlDict's `application`, which a copied case may not have run, and no
    solver in them prints a Courant number in v2406 (no file in the directory of any solver in
    `STEADY_SOLVERS` or `V2406_STEADY` under applications/solvers names `CourantNo`, read in the
    pinned image on 25 Oct 2026, but chtMultiRegionSimpleFoam's own
    `fluid/compressibleMultiRegionCourantNo.H`, which none of its files includes; pimpleFoam.C
    and scalarTransportFoam.C include `CourantNo.H`). A log that prints one on such a case was
    written by another solver, so its number is judged: sparing it was a false accept (the review
    of T-WEDGE-MESH, a pimpleFoam log on a case copied from simpleFoam). A local-time-stepping
    case (`localEuler`), every transient case, a case whose `ddtSchemes` was not read or not read
    whole and a run with no configuration are judged as before.
    """
    from courant.verify.checks import enabled
    from courant.watcher.parser import default_region_run, names_regions, reread_fvschemes

    if not enabled("cfl.steady", checks):
        return None
    if config is None or not getattr(config, "ddt_read_whole", None):
        return None
    if (config.ddt_scheme or "").split() != [STEADY_DDT]:
        return None
    if any((scheme or "").split() != [STEADY_DDT]
           for scheme in (getattr(config, "ddt_terms", None) or {}).values()):
        return None
    if getattr(config, "regions", None) or names_regions(log_text) or \
            not default_region_run(log_text) or reread_fvschemes(log_text):
        return None
    return f"ddtSchemes default {STEADY_DDT}"


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
                suggests=_growth_suggests(ctx.config, ctx.checks),
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
            judged = judge_residuals(ctx.events, ctx.config, ctx.checks,
                                     exclude=ctx.zero_components)
            # **Not on a transient run whose case we can read.** A time-marching
            # solve is judged on stability, not on a residual bar: its final
            # initial-residual is a function of the timestep. That is the single
            # definition in `courant/verification/convergence.py::solve_converged`,
            # and the Courant checks above already cover stability. Applying the
            # 1e-3 bar here anyway was the eighth instance of one fact with two
            # implementations. It raised a WARN on 4 of the 7 unmodified
            # tutorials, all transient, and the verifier's first certification
            # (RRD RM-01 v0) is what found it.
            #
            # **The bar is per field** (RRD RM-35, S's decision of 13 Oct 2026): each field
            # against its declared `residualControl` tolerance on a steady run, the flat
            # NOT_CONVERGED where the case declares none. `judge_residuals` decides it, and
            # `ConvergenceCheck` asks the same function, so this finding fires exactly when
            # that check is not OK.
            # Time-accurate, as `ConvergenceCheck` and VER-16b's exclusion ask it: one function,
            # `judged_on_the_bar` (TRD VER-21, VER-19 under `next`; the review of T-WEDGE-MESH).
            transient = not judged_on_the_bar(ctx.config, ctx.checks)
            if judged is not None and judged.over and not transient:
                if judged.per_field:
                    detail = (f"final residual {judged.residual:.3g} on {judged.field} > "
                              f"{judged.bar_words()}")
                    value, threshold = judged.residual, judged.bar
                else:
                    detail = f"final worst residual {judged.worst:.3g} > {self.NOT_CONVERGED}"
                    value, threshold = judged.worst, self.NOT_CONVERGED
                out.append(Finding(
                    id="residual.not_converged", checker=self.name,
                    title="Final residual above convergence criterion", severity=WARN,
                    field=judged.field if judged.per_field else None,
                    value=value, threshold=threshold, detail=detail,
                    suggests=_stall_suggests(ctx.config, ctx.log_text),
                ))
            # **A stalled history under heavy under-relaxation** (`convergence.Stall`): the
            # solver's own "converged" can be spurious when relaxation deflates the residual
            # it compares. Its own finding, so the reason says what was seen, and it is
            # evidence for a plateau, which is what it is.
            if judged is not None and judged.stall is not None and not transient:
                s = judged.stall
                out.append(Finding(
                    id="residual.stalled", checker=self.name,
                    title="Residual history stalled", severity=WARN,
                    field=s.field, value=s.residual, threshold=s.bar,
                    detail=stall_sentence(s),
                    suggests=["CONV.RESIDUAL_PLATEAU"],
                ))
        return out


#: A linear solver told to stop this far from solved is not solving the
#: equation. OpenFOAM's own tutorials sit at 1e-06/0.05 for pressure; the
#: loosest anything ships with is 1e-04.
LOOSE_TOLERANCE = 1e-4
LOOSE_REL_TOL = 0.2

#: A solve as v2406 prints it (`SolverPerformance::print`, LduMatrix/SolverPerformance.C):
#: `<solver>:  Solving for <name>`, then `, Initial residual = ...` or, for a singular matrix,
#: `:  solution singularity`. <name> is a `word`, the field's (with a component's name after it
#: on each component of a vector or tensor), and a word may hold a ':' (a cloud's fields are
#: `IOobject::scopedName(cloud, ...)`, `kinematicCloud:alpha`, MPPIC's Implicit.C, the separator
#: IOobject.C's `scopeSeparator`) or a ',', so the name runs to what follows it. A region header
#: (`Solving for fluid region <name>`) and MULES' `MULES: Solving for <name>` have neither after
#: the word, so they are not read as solves.
_SOLVED = re.compile(r"Solving for (\S+?)(?:, Initial residual = |:  solution singularity)")
#: The line every v2406 solver prints before it reads its case (createTime.H: `Info<< "Create
#: time\n"`): a log that holds it holds the run from its start, and so any solve made before the
#: time loop (interFoam's pcorr, initCorrectPhi.H).
_CREATE_TIME = re.compile(r"^Create time[ \t\r]*$", re.MULTILINE)
#: Solves a log shows whatever SolverPerformance's switch is: viewFactor.C prints its `qr` solve
#: unconditionally (`solverPerf.print(...)`, no log level).
_PRINTED_ANYWAY = frozenset({"qr"})
#: GAMG's coarsest level, which GAMGSolverSolve.C prints under GAMG's own debug switch
#: (`(log_ >= 2) || debug`), and at 2 that switch prints GAMG's every cycle of a field
#: (`(log_ >= 2) || (debug >= 2)`) whatever SolverPerformance's is: its line says the log's
#: solves need not have printed at SolverPerformance's level.
_GAMG_DEBUG = "coarsestLevelCorr"


def _solved_names(log_text: str | None) -> set[str] | None:
    """The names the log shows solved, as printed (`_SOLVED`), or None when it shows no solve at
    all (no log, as in `courant preflight`, a log whose solves are silenced, a window of output
    with none in it): nothing can then be said not to have been solved."""
    names = {m.group(1) for m in _SOLVED.finditer(log_text or "")}
    return names or None


def _solves_print(solvers: dict, printed: set[str]) -> bool:
    """Whether the log proves that a solve under a solver entry with no `log` of its own prints a
    line: SolverPerformance's debug switch is on. A solve prints when
    `solverControls.getOrDefault<int>("log", solverPerformance::debug)` is not 0
    (fvScalarMatrix.C, fvMatrixSolve.C, faMatrixSolve.C), and the switch is one for every type
    (SolverPerformance.H: `ClassName("SolverPerformance")`, etc/controlDict: `SolverPerformance
    1`), which a case can turn off (`DebugSwitches`). It is on when the log shows a solve made at
    its level: no entry declares a non-zero `log` (its own level), the log shows a solve other
    than one printed whatever the switch (`_PRINTED_ANYWAY`), and GAMG's debug switch, which
    prints GAMG's cycles whatever SolverPerformance's is, is not raised (`_GAMG_DEBUG`). The
    review of 25 Oct 2026: with `SolverPerformance 0` and `log 1` on p alone the log shows p
    alone, and that says nothing of what else was solved."""
    if _GAMG_DEBUG in printed:
        return False
    if any(getattr(s, "log", None) not in (None, 0) for s in solvers.values()):
        return False
    return bool(printed - _PRINTED_ANYWAY)


def _unsolved_pcorr(solvers: dict, log_text: str | None) -> set[str]:
    """The solver entries of a `pcorr` the run never solved, by their keys.

    v2406 solves pcorr only in `CorrectPhi` (CorrectPhi.C, and the overset and cavitating
    solvers' own correctPhi.H, every file of its source and solvers that names pcorr), on a field
    named `pcorr` looked up as `pcorr` or `pcorrFinal` (`pcorr.select(...)`); pimpleFoam calls it
    only when the mesh moved or changed and `correctPhi` is on (pimpleFoam.C: `if
    (mesh.changing()) { ... if (correctPhi) { #include "correctPhi.H"`; createDyMControls.H:
    `correctPhi` defaults to `mesh.dynamic()`), interFoam once before its time loop on any mesh
    (initCorrectPhi.H). An entry is left out only when all of these hold:
    - only a pcorr lookup reads it: every name its key matches begins with `pcorr`
      (`wordre.literal_prefix`; v2406 looks no other such name up). An entry any other name may
      read (`".*"`, `"(p|pcorr)"`, `"rho.*|pcorr.*"`) is judged, whatever the log shows: a
      solver may look an entry up by a name no line prints (reactingFoam's species, solved as
      `YiEqn.solve("Yi")` and printed as `Solving for CH4`, YEqn.H; the review of 25 Oct 2026);
    - the log holds the run from its start (`_CREATE_TIME`) and shows solves, none of a name
      beginning `pcorr`;
    - a pcorr solve under the entry would have printed: its own `log` is not 0 and is set, or it
      is unset and the log proves SolverPerformance's switch on (`_solves_print`)."""
    from courant.case import wordre

    printed = _solved_names(log_text)
    if printed is None or not _CREATE_TIME.search(log_text or ""):
        return set()
    if any(name.startswith("pcorr") for name in printed):
        return set()
    level_on = _solves_print(solvers, printed)
    out = set()
    for key, spec in solvers.items():
        pattern = len(key) >= 2 and key[0] == key[-1] == '"'
        if not (wordre.literal_prefix(key[1:-1]) if pattern else key).startswith("pcorr"):
            continue
        log = getattr(spec, "log", None)
        if log == 0 or (log is None and not level_on):
            continue
        out.add(key)
    return out


def used_solvers(config, log_text: str | None) -> dict:
    """The `fvSolution` solver entries this run used, by their keys: every entry but those of a
    `pcorr` the log shows the run never solved (`_unsolved_pcorr`). One definition, for
    `solver.loose_tolerance` and the cause a residual left above its bar suggests
    (`_stall_suggests`).

    A linear solver's tolerance says how far a field is solved only where the field is solved,
    and pcorr is solved only when the mesh moves (or once, at interFoam's start): on a static
    mesh its entry, written in many tutorials, is never read (an internal evaluation on rebuilt runs, 22
    Oct 2026, rejected four healthy conformal Couette controls on it). Where it is solved its
    tolerance is judged as before, and so is every other entry, as in v0.5.0: which entry a
    solve read is not in its line (a lookup by another name, a solve silenced), so the log
    proves an entry unused only where the one field that reads it is never solved."""
    solvers = getattr(config, "linear_solvers", None) or {}
    unsolved = _unsolved_pcorr(solvers, log_text)
    return {k: s for k, s in solvers.items() if k not in unsolved}


def loose_solvers(config, log_text: str | None) -> dict:
    """The used solver entries (`used_solvers`) told to stop far from solved."""
    return {field: s for field, s in used_solvers(config, log_text).items()
            if (s.tolerance is not None and s.tolerance > LOOSE_TOLERANCE)
            or (s.rel_tol is not None and s.rel_tol > LOOSE_REL_TOL)}


#: Above this a relaxation factor is doing little to damp the outer loop. Below
#: it, blaming one for a divergence is blaming a number the engineer took from
#: OpenFOAM's own tutorials.
RELAXATION_SUSPECT = 0.8

#: A relaxation factor of 1.0 applies the full correction every iteration —
#: there is no under-relaxation left. Anything below it is a choice OpenFOAM's
#: own tutorials make routinely (`pitzDaily` ships 0.9), so it is not evidence
#: of anything.
RELAXATION_UNDAMPED = 1.0


def _growth_suggests(config, checks: str = "default") -> list[str]:
    """Whether over-relaxation is a candidate for these growing residuals: on a steady case, as
    `scheme.relaxation` asks it (`is_steady` under `checks`; v2406's solver tables under `next`,
    TRD VER-21)."""
    if config is None or not config.is_steady(checks):
        return []
    factors = config.relaxation_factors or {}
    hot = any(v is not None and v >= RELAXATION_SUSPECT for v in factors.values())
    return ["NUM.OVER_RELAXATION"] if hot else []


def _stall_suggests(config, log_text: str | None = None) -> list[str]:
    """What a run that finished above tolerance most likely means.

    Both `NUM.SOLVER_TOLERANCE` and `CONV.RESIDUAL_PLATEAU` list
    `residual.not_converged` in their signatures, so they scored identically and
    the `suggests` tiebreak decided every one of them — always for the plateau,
    including on a case whose pressure solver had been told to stop at 0.1.

    We can do better than a constant, because the case says which it is. A
    linear solver that stops far from solved *is* the explanation; a properly
    converged inner solve that still leaves the outer residual flat is a
    plateau. Reading the tolerance is the difference between guessing and
    knowing, and it is right there in `fvSolution`, for a solver entry the run used
    (`used_solvers`).
    """
    return ["NUM.SOLVER_TOLERANCE"] if loose_solvers(config, log_text) else \
        ["CONV.RESIDUAL_PLATEAU"]


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


def _adaptive_is_off(config, checks: str = "default") -> bool:
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

    **Under `--checks next` (`solvers.v2406`, TRD VER-21) the default is read, not
    guessed:** v2406 reads it as `getOrDefault("adjustTimeStep", false)`
    (`createTimeControls.H`, `readTimeControls.H`), so absent is off on every solver, as
    `courant_limit` already takes it. Without that, completing the adaptive table would have
    silenced `cfl.climbing` on every newly listed solver whose case omits the entry, which is
    a fixed timestep.
    """
    if config is None:
        return False
    if config.adjust_timestep is False:
        return True
    from courant.verify.checks import enabled

    if config.adjust_timestep is None and enabled("solvers.v2406", checks):
        return True
    return (config.adjust_timestep is None
            and not _honours_adaptive_dt(config.solver, checks))


def _honours_adaptive_dt(solver: str | None, checks: str = "default") -> bool:
    """Whether this solver reads `adjustTimeStep` at all (v2406's own table under `next`).

    Imported lazily to keep `courant.case` out of this module's import graph.
    """
    from courant.case.solvers import honours_adjust_time_step

    return honours_adjust_time_step(solver, checks)


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
                detail=f"max non-orthogonality {shown(m.max_non_orthogonality, 1)} deg, "
                       f"nNonOrthogonalCorrectors={corr}",
                suggests=["MESH.SKEWNESS"],
            ))
        # VER-7d: on faces with an area, when checkMesh's maximum is a zero-area face's quotient
        # by ROOTVSMALL on an `empty` patch (`MeshQuality.skewness`); "not assessed" names them.
        skew = m.skewness()
        if skew is not None and skew > SKEW_THRESHOLD:
            out.append(Finding(
                id="mesh.skewness", checker=self.name, title="High mesh skewness",
                severity=WARN, value=skew, threshold=SKEW_THRESHOLD,
                detail=f"max skewness {shown(skew, 2)}", suggests=["MESH.SKEWNESS"],
            ))
        if m.max_aspect_ratio is not None and m.max_aspect_ratio > 1000:
            out.append(Finding(
                id="mesh.aspect_ratio", checker=self.name, title="Extreme aspect ratio",
                severity=WARN, value=m.max_aspect_ratio, threshold=1000.0,
                detail=f"max aspect ratio {shown(m.max_aspect_ratio, 0)}",
                suggests=["MESH.ASPECT_RATIO"],
            ))
        return out


#: Up to this magnitude a metric prints in fixed notation with its own places, as it always has;
#: past it, in three significant figures. A zero-area face gave checkMesh a skewness of
#: 2.5475503e+146 on the committed wedge pipe, which `:.2f` printed as a number of 147 digits
#: (TRD VER-7d).
FIXED_BELOW = 1e4


def shown(value: float, places: int) -> str:
    """A metric for a finding's sentence, in a few significant figures whatever its size."""
    return f"{value:.{places}f}" if abs(value) < FIXED_BELOW else f"{value:.3g}"


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
        #
        # **Only a solver entry the run used** (`used_solvers`, TRD VER-5a): a tolerance says how
        # far a field is solved where the field is solved, and the log says which were. A
        # `pcorr` entry on a static mesh is never read (pimpleFoam solves pcorr only after the
        # mesh moved, with `correctPhi` on), and it rejected four healthy Couette controls of
        # an internal evaluation on rebuilt runs. With no solve in the log (`courant preflight`, which has
        # no run) every entry counts, as before.
        slack = loose_solvers(c, ctx.log_text)
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
        # Steady by construction only: a local-time-stepping case has a pseudo time derivative
        # to damp it, and v2406's LTS tutorials ship `".*" 1` (TRD VER-19).
        hot = ({f: v for f, v in c.relaxation_factors.items()
                if v is not None and v >= RELAXATION_UNDAMPED}
               if c.is_steady(ctx.checks) else {})
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
        # Not on a local-time-stepping case under `next` (VER-19): its step is per cell, set from
        # maxCo, whatever `adjustTimeStep` says.
        if (_adaptive_is_off(c, ctx.checks) and not c.marches_to_steady_state(ctx.checks)
                and c.delta_t is not None and _honours_adaptive_dt(c.solver, ctx.checks)):
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
        # over-constrained p-U: patches fixing both p and U (aggregated into one finding). The
        # pressure is the one the solver solves, p_rgh included, under `--checks next` (PRE-5).
        bad = c.overconstrained_patches(ctx.checks)
        if not bad:
            return []
        if bad == c.overconstrained_patches():          # v0.5.0's finding, word for word
            detail = f"patches fix both p and U (fixedValue/fixedValue): {bad}"
        else:
            pairs = {patch: next(f"{p} {c.boundary[patch][p]}" for p in ("p_rgh", "p")
                                 if c.boundary[patch].get(p) in PRESSURE_FIXING)
                     for patch in bad}
            detail = ("patches fix both the solved pressure and U: "
                      + ", ".join(f"{patch} ({pair}, U fixedValue)"
                                  for patch, pair in pairs.items()))
        return [Finding(
            id="bc.overconstrained", checker=self.name,
            title="Over-constrained pressure-velocity coupling", severity=WARN,
            location=",".join(bad),
            detail=detail,
            suggests=["BC.OVERCONSTRAINED_PU"],
        )]


class PhysicsChecker:
    name = "physics"

    def run(self, ctx: CheckContext) -> list[Finding]:
        c = ctx.config
        if c is None:
            return []
        out: list[Finding] = []
        # The top-level `nu` as written; under `--checks next` also each phase's `nu`, and a
        # `mu` divided by the density the case's equation of state gives (PRE-6,
        # `courant.case.viscosity`, the one definition). One finding, naming each out of range.
        bad = [v for v in viscosities(c, ctx.checks) if v.out_of_range]
        if bad:
            out.append(Finding(
                id="physics.nu", checker=self.name, title="Suspicious viscosity value",
                severity=WARN, value=_nu_value(bad[0]),
                detail="; ".join(_nu_words(v) for v in bad),
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
        #
        # **The Courant condition is v0.5.0's** (`_courant_within`): no step above the run's
        # limit, a Courant number read or not. Asking a steady case for one read on a time step
        # (VER-5a's first form, 25 Oct 2026) changed no verdict on any measured run, only reasons,
        # and took the finding from the legacy bench's laminar case, so it was not kept; on a
        # time-marching run it accepted a laminar LTS run that never settled (the review of 25
        # Oct 2026). Under `--checks next` (VER-13a, `cfl.steady`), on a case whose own
        # `fvSchemes` declares `ddtSchemes default steadyState` (`steady_courant`), the number
        # read is not held against the limit: it decides nothing of the run, so it does not rule
        # the laminar model out either, and the finding says it was not judged.
        #
        # Not changed, and open (H-LAMINAR-TRANSIENT, S): on a time-marching run "never reached
        # tolerance" is the flat bar the gate's own convergence check does not apply to such a
        # run (`residual.not_converged`'s transient exclusion, TRD VER-21), and a periodic laminar
        # flow ends above it at a large enough step (the second look's cylinder at Re 100,
        # rejected on this finding alone). Excluding it as `residual.not_converged` does would
        # accept the certification corpus's hotRoom `laminar_on_turbulent` fault and 28
        # Courant-10K hotRoom twins this finding alone rejects, and no reading of the residual
        # history separates them from the cylinder (TRD VER-5a, the record of 25 Oct 2026).
        oscillating = any(is_residual_climb(e) for e in ctx.events_of(EventKind.DIVERGENCE))
        stalled_high = _finished_above_tolerance(ctx)
        died = bool(ctx.events_of(EventKind.CRASH))
        if (c.is_laminar(ctx.checks) and (oscillating or stalled_high or died)
                and _courant_within(ctx)):
            out.append(Finding(
                id="physics.laminar_unstable", checker=self.name,
                title="Laminar model with unstable residuals",
                # WARN when the run failed outright with the timestep in range:
                # a laminar model on a turbulent flow is then the likely reason,
                # not a footnote. INFO when it is only oscillating, which many
                # things cause.
                severity=WARN if (stalled_high or died) else INFO,
                detail=_laminar_words(c) + ", " + _courant_words(c, ctx.log_text, ctx.checks)
                       + ", and the solve "
                       + ("terminated abnormally" if died
                          else "never reached tolerance" if stalled_high
                          else "oscillates/grows"),
                suggests=["PHYS.LAMINAR_TURBULENT"],
            ))
        # A laminar model on a case heated from below that its own numbers call turbulent, on a
        # mesh that cannot resolve that turbulence directly (TRD VER-17): neither modelled nor
        # resolved, whatever the residuals do. The estimate is `courant.case.regime`'s, and both
        # of its numbers are bounds that understate the case against the run. It does not reach
        # the 3 hotRoom `laminar_on_turbulent` twins RM-35 version 2 accepted blind (044, 083,
        # 092): their heater and their cold floor share a level, so no length is provable.
        # Behind `--checks next` until 0.6.0 (GATE-13).
        from courant.verify.checks import enabled

        if (enabled("physics.laminar_regime", ctx.checks) and c.is_laminar(ctx.checks)
                and solver_read_gravity(ctx.log_text)):
            regime = buoyant_regime(c, ctx.mesh)
            if regime is not None and regime.turbulent and regime.unresolved:
                t_cold, t_hot = regime.temperatures
                out.append(Finding(
                    id="physics.laminar_regime", checker=self.name,
                    title="Laminar model on a turbulent buoyant flow the mesh cannot resolve",
                    severity=WARN, value=regime.rayleigh, threshold=RA_TRANSITION,
                    detail=(f"laminar turbulence type; walls held at {t_hot:g} K lie at least "
                            f"{regime.height:.3g} m below walls held at {t_cold:g} K, so the "
                            f"Rayleigh number is at least {regime.rayleigh:.2g}, above the "
                            f"{RA_TRANSITION:.0e} at which natural "
                            f"convection turns turbulent; and the mean cell, "
                            f"{regime.cell:.3g} m, is {regime.cell / regime.kolmogorov:.3g} "
                            f"times the largest the Kolmogorov scale can be "
                            f"({regime.kolmogorov:.2g} m), where a direct simulation needs at "
                            f"most {DNS_SPACING:.3g} times"),
                    suggests=["PHYS.LAMINAR_TURBULENT"],
                ))
        return out


def _nu_value(v: Viscosity) -> float:
    """The `nu` a `physics.nu` finding carries: the bound nearer the range, as it understates."""
    return v.lo if v.source == "nu" or v.hi >= NU_MIN else v.hi


def _nu_words(v: Viscosity) -> str:
    """One viscosity out of range, for `physics.nu`'s sentence. The top-level `nu`'s words are
    v0.5.0's exactly. A phase is named by its pseudonym (`Viscosity.alias`, `phase1`): its name
    is the case author's, and a finding reaches the Evidence Bundle."""
    if v.source == "mu":
        spread = f"{v.lo:.3g}" if v.lo == v.hi else f"{v.lo:.3g} to {v.hi:.3g}"
        return (f"mu={v.mu:g} Pa s gives nu {spread} m^2/s ({v.basis}), "
                f"outside typical fluid range [{NU_MIN:g}, {NU_MAX:g}]")
    if v.phase is not None:
        alias = v.alias or "a phase"
        return (f"{alias} nu={v.lo:g} m^2/s (constant/transportProperties, {alias} by its "
                "place in `phases`) is outside typical fluid range [1e-8, 1e-2]")
    return f"nu={v.lo:g} m^2/s is outside typical fluid range [1e-8, 1e-2]"


def _laminar_words(c: CaseConfig) -> str:
    """What made the run laminar, for a finding's sentence: `simulationType laminar` (v0.5.0's
    words), or a model whose `turbulence` switch is off (VER-17a, under `--checks next`)."""
    if (c.turbulence_type or "").strip().lower() == "laminar":
        return "laminar turbulence type"
    return (f"{c.turbulence_type} {c.turbulence_model or 'model'} with its turbulence switch "
            "off (no model equation solved)")


def _courant_words(c: CaseConfig, log_text: str | None = None, checks: str = "default") -> str:
    """What `physics.laminar_unstable` says of the Courant number, as `_courant_within` judged it:
    v0.5.0's words, or, under `--checks next` where VER-13a holds no number against the limit
    (`steady_courant`), that it was not judged, which is what the verdict's "not assessed" line
    says of it too. Under the default, v0.5.0's words on every run."""
    spared = steady_courant(c, log_text, checks)
    return ("Courant number within its stability limit" if spared is None
            else f"Courant number not judged ({spared})")


def _finished_above_tolerance(ctx: CheckContext) -> bool:
    """Whether the run ran to the end with its residuals still above tolerance.

    The bar `residual.not_converged` uses (`judge_residuals`), read from the events directly
    because checkers do not see each other's findings, but without its transient exclusion: on a
    time-marching run this is the evidence the hotRoom `laminar_on_turbulent` fault is rejected
    on, and whether it stays evidence there is S's decision (H-LAMINAR-TRANSIENT, the comment at
    the call).
    """
    if ctx.events_of(EventKind.CRASH) or diverged(ctx.events):         # ADP-5: one definition
        return False
    residuals = ctx.events_of(EventKind.RESIDUAL)
    if not residuals:
        return False
    judged = judge_residuals(ctx.events, ctx.config, ctx.checks,  # the same bar
                             exclude=ctx.zero_components)
    return judged is not None and bool(judged.over)


def _courant_within(ctx: CheckContext) -> bool:
    """Whether the run's Courant number stayed within the limit that applies to this run
    (VER-13), as `physics.laminar_unstable` says it did.

    Under the default checks v0.5.0's test, exactly, on every run (its `not
    _courant_breached(ctx)`: no step above the limit).
    Under `--checks next` (VER-13a, `cfl.steady`; T-WEDGE-MESH): on a case that declares
    `ddtSchemes default steadyState` (`steady_courant`), the number is a pseudo-step's and
    decides nothing, so it is not held against the limit here either: a number that decides
    nothing of the run explains nothing of it, and does not silence the finding (one fact, one
    answer; the review of T-WEDGE-MESH).

    Elsewhere v0.5.0's test stands: no step above the limit, a line before the loop included,
    and a Courant number need not have been read. A steady run with no Courant number keeps the
    finding too: asking for one there (VER-5a's first form, 25 Oct 2026) changed no verdict on any
    measured run, only reasons, and took the finding from the legacy bench's laminar case, so
    v0.5.0's test was kept, which can only add a finding. A local-time-stepping solver prints none
    inside its loop (rhoPimpleFoam.C, interFoam.C: `if (LTS) { #include "setRDeltaT.H" } else
    { #include "CourantNo.H" ...`), and the default parser does not read every solver's line
    (rhoCentralFoam's `Mean and max Courant Numbers`, centralCourantNo.H). Under the default
    such a run is transient, so no residual bar holds it, and this finding is what rejects one
    that never settled (the review of 25 Oct 2026: asking for a Courant number there accepted a
    laminar LTS run stopped far from steady, and decided H-LAMINAR-TRANSIENT, which is S's)."""
    residuals = ctx.events_of(EventKind.RESIDUAL)
    if steady_courant(ctx.config, ctx.log_text, ctx.checks) is None:   # VER-13a, under next
        limit, _ = courant_limit(ctx.config)
        if any(e.payload.courant_max is not None and e.payload.courant_max > limit
               for e in residuals):
            return False
    return True


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
        out: list[Finding] = []
        unreadable = getattr(ctx.config, "unreadable", None) if ctx.config else None
        if unreadable:
            out.append(Finding(
                id="case.unreadable_dict", checker=self.name,
                title="Case dictionary could not be parsed", severity=WARN,
                value=float(len(unreadable)),
                detail="; ".join(f"{f}: {why}" for f, why in sorted(unreadable.items()))
                       + " — findings that depend on these files are unavailable, "
                         "not absent",
            ))
        # TRD VER-3b: a value the dictionary writes through a `$macro`, an `#include` or `#calc`
        # that the reader did not follow. Parsed, so not `case.unreadable_dict`; not read, so
        # not absent either. The detail names the file, the key path (a phase by its place in
        # `phases`) and how the value is written, never the value written
        # (`CaseConfig.unread_keys`): a Finding reaches the Evidence Bundle a model and the cloud
        # see. What the case wrote is said locally only (`unread_values_sentence`). A reader
        # limitation (`policy.READER_LIMITATIONS`): it decides no verdict, in either mode.
        unread = getattr(ctx.config, "unread_values", None) if ctx.config else None
        if unread:
            keys = getattr(ctx.config, "unread_keys", None) or {}
            out.append(Finding(
                id="case.unread_value", checker=self.name,
                title=UNREAD_VALUES_TITLE, severity=WARN,
                value=float(len(unread)),
                detail="; ".join(f"{f}: {keys.get(f) or 'a value'}" for f in sorted(unread))
                       + UNREAD_VALUES_TAIL,
            ))
        return out


UNREAD_VALUES_TITLE = "Case values were not read as the solver read them"
UNREAD_VALUES_TAIL = " — findings that depend on them are unavailable, not absent"


def unread_values_sentence(config) -> str | None:
    """`courant verify`'s "not assessed" and `courant preflight`'s "not checked" sentence for the
    values the case reader did not read as the solver read them (TRD VER-3b), quoting what the
    case wrote (`CaseConfig.unread_values`). Local output only: the finding `case.unread_value`
    says the same in fixed vocabulary."""
    unread = getattr(config, "unread_values", None) if config is not None else None
    if not unread:
        return None
    return (UNREAD_VALUES_TITLE + ": "
            + "; ".join(f"{f}: {why}" for f, why in sorted(unread.items())) + UNREAD_VALUES_TAIL)


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
