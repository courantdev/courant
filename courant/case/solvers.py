"""What each OpenFOAM solver can actually do.

Domain knowledge, kept in one place because getting it wrong is invisible.
`adjustTimeStep` is the case in point: it is read from `controlDict` only by
solvers that include `setDeltaT.H`. Every other solver **silently ignores it** —
no warning, no error, no entry in the log.

That produced a real and embarrassing failure. Given a lid-driven cavity
diverging on the Courant number, Courant proposed "enable adaptive timestepping
capped at maxCo 0.8" to `icoFoam`, which is a fixed-timestep PISO solver. The
edit applied cleanly, `foamDictionary` confirmed the values were written, and the
solve crashed exactly as before. Measured in the container:

    adjustTimeStep yes; maxCo 0.8;  → icoFoam
    "deltaT = " lines in the log:   0        (the timestep never adapted)
    Courant Number max:             66.8, 113.8, …   (against maxCo 0.8)

Two costs. One of a three-attempt budget is spent on a fix that cannot work, and
any CFD engineer reading the report sees us tell icoFoam to do something icoFoam
does not do — which is the kind of error that ends an evaluation.

The lists below are read from OpenFOAM v2406's own sources rather than recalled:

    grep -rl setDeltaT.H $WM_PROJECT_DIR/applications/solvers --include='*.C'

58 solvers include it. Rather than carry all 58, the rule is expressed as the
family it actually tracks — the PIMPLE/multiphase/reacting transient solvers —
with the common fixed-timestep solvers named explicitly, because those are the
ones a wrong answer costs us.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Solvers that read `adjustTimeStep`, verified against v2406 sources.
ADAPTIVE_TIMESTEP_SOLVERS = frozenset({
    "pimpleFoam", "rhoPimpleFoam", "rhoPimpleAdiabaticFoam", "SRFPimpleFoam",
    "buoyantPimpleFoam", "buoyantBoussinesqPimpleFoam",
    "interFoam", "interIsoFoam", "interMixingFoam", "interPhaseChangeFoam",
    "compressibleInterFoam", "multiphaseInterFoam", "multiphaseEulerFoam",
    "twoPhaseEulerFoam", "twoLiquidMixingFoam", "driftFluxFoam",
    "cavitatingFoam", "potentialFreeSurfaceFoam",
    "reactingFoam", "rhoReactingFoam", "rhoReactingBuoyantFoam",
    "sprayFoam", "engineFoam", "XiFoam", "XiEngineFoam", "fireFoam",
    "PDRFoam", "chemFoam", "DPMFoam", "MPPICInterFoam", "rhoCentralFoam",
    "chtMultiRegionFoam",
    # Overset. `over*DyMFoam` wrap a PIMPLE loop and read `adjustTimeStep` the
    # same way; an unknown solver name defaulted to "no adaptivity", which makes
    # the timestep fix propose the wrong repair on a family that does support it.
    "overPimpleDyMFoam", "overRhoPimpleDyMFoam", "overInterDyMFoam",
    "overBuoyantPimpleDyMFoam", "overPotentialFoam",
    # The OpenFOAM.org fork. v11 replaced the individual solver executables with
    # one `foamRun -solver <module>` binary, so an entire fork of the user base
    # arrives under a name that was in no capability set — and `foamRun` does
    # read `adjustTimeStep`, so the timestep repair was proposing a fixed step
    # to every .org user.
    #
    # Steadiness is NOT decided here for these: a `foamRun` case is steady or
    # transient according to its `ddtSchemes`, and `CaseConfig.is_steady` already
    # prefers the declared scheme over the solver name. That is the correct
    # precedence and it is why this only needed the adaptive half.
    "foamRun", "foamMultiRun",
})

#: Solvers that ignore it. Named explicitly because these are the ones the
#: taxonomy's timestep failures actually show up on, so a wrong answer here is
#: the expensive kind.
FIXED_TIMESTEP_SOLVERS = frozenset({
    "icoFoam", "pisoFoam", "simpleFoam", "potentialFoam", "sonicFoam",
    # Steady compressible. `rhoSimpleFoam` was in neither set, so `is_steady`
    # returned False for it and every steady-only check was silenced — the
    # relaxation finding among them, on the most common compressible steady
    # solver there is. Under-relaxation is the single failure mode a practising
    # CFD consultant named first when asked what breaks most often.
    "rhoSimpleFoam", "rhoPorousSimpleFoam",
    "buoyantSimpleFoam", "laplacianFoam", "scalarTransportFoam",
    "boundaryFoam", "SRFSimpleFoam", "adjointShapeOptimizationFoam",
    "porousSimpleFoam", "shallowWaterFoam", "electrostaticFoam",
})

#: Steady-state solvers. `deltaT` is a relaxation pseudo-step, and the Courant
#: number is not a stability criterion — so a CFL "fix" that shrinks deltaT is
#: not just useless here, it slows convergence.
STEADY_SOLVERS = frozenset({
    "simpleFoam", "potentialFoam", "buoyantSimpleFoam", "porousSimpleFoam",
    "rhoSimpleFoam", "rhoPorousSimpleFoam",
    "SRFSimpleFoam", "adjointShapeOptimizationFoam", "laplacianFoam",
})


# ------------------------------------------------------------------ v2406's own list (GATE-13)
#
# The sets above are v0.5.0's: a hand-picked subset, and an internal evaluation (20 Oct 2026)
# ran a solver in none of them, `buoyantBoussinesqSimpleFoam`, whose `SIMPLE/residualControl`
# VER-16 therefore could not read. The sets below complete them for ESI OpenFOAM v2406, read on
# 20 Oct 2026 from the pinned image's own sources (`courant.config.OPENFOAM_IMAGE`), not
# recalled. The facts are a generated file, `tests/fixtures/solvers/v2406.json`
# (`scripts/v2406_solvers.py`), and `tests/unit/test_solvers_v2406.py` holds every table below
# to it: every executable a `Make/files` under `applications/solvers` builds, and in its main
# `.C` file (or the one it `#include`s whole: MPPICFoam is DPMFoam.C, MPPICDyMFoam is
# DPMDyMFoam.C, reactingHeterogenousParcelFoam is reactingParcelFoam.C, simpleSprayFoam is
# simpleReactingParcelFoam.C) the control it builds and the time-step file it includes:
#
#   simpleControl + `simple.loop()`        the run ends when SIMPLE/residualControl is met
#   setDeltaT.H, setMultiRegionDeltaT.H    reads `adjustTimeStep` (each such file in v2406 does)
#   no fvm::ddt in its equations           steady by construction
#
# Solvers from `modules/` (OpenQBMM's pbeFoam and the like) are not in `applications/solvers` and
# are in no set. Every name ends in "Foam", so `_base` already resolved each one; what changes is
# what the tables say about it, and only under `--checks next` (`solvers.v2406`, TRD VER-21):
# under the default the sets above decide, as in v0.5.0.

#: Steady by construction and missing above. All but two loop on `simple.loop()`
#: (`V2406_SIMPLE_LOOP`): `chtMultiRegionSimpleFoam` reads each region's `SIMPLE` dictionary for
#: its correctors and has no `residualControl` at all, and `overPotentialFoam` has no time loop.
V2406_STEADY = frozenset({
    "buoyantBoussinesqSimpleFoam", "overSimpleFoam", "overRhoSimpleFoam",
    "simpleReactingParcelFoam", "simpleCoalParcelFoam", "simpleSprayFoam",
    "chtMultiRegionSimpleFoam", "overPotentialFoam",
})

#: The steady solvers whose run ends on `SIMPLE/residualControl` (`simpleControl::loop`), beyond
#: `STEADY_SOLVERS`. `residual_control_loop` reads them in either mode: it is consulted only by
#: the bar per field, which is itself behind `--checks next` (`residual.control`, VER-16).
V2406_SIMPLE_LOOP = frozenset({
    "buoyantBoussinesqSimpleFoam", "overSimpleFoam", "overRhoSimpleFoam",
    "simpleReactingParcelFoam", "simpleCoalParcelFoam", "simpleSprayFoam",
})

#: Transient solvers that loop on `simple.loop()` too, so their run also ends when
#: `SIMPLE/residualControl` is met (`residual_control_loop`). Each solves a time derivative.
V2406_TRANSIENT_SIMPLE_LOOP = frozenset({
    "scalarTransportFoam", "thermoFoam", "overLaplacianDyMFoam", "solidFoam",
})

#: In `STEADY_SOLVERS` with no `simple.loop()`: v2406's `potentialFoam.C` has no time loop (it
#: iterates `potentialFlow.correctNonOrthogonal()` once through), so no `residualControl` ends it.
V2406_NO_SIMPLE_LOOP = frozenset({"potentialFoam"})

#: In `STEADY_SOLVERS` and wrong there: v2406's `laplacianFoam.C` solves `fvm::ddt(T)` (its
#: tutorial, flange, marches in time with Euler). Not steady under `--checks next`.
V2406_NOT_STEADY = frozenset({"laplacianFoam"})

#: Read `adjustTimeStep` (their main file includes setDeltaT.H or setMultiRegionDeltaT.H) and are
#: missing from `ADAPTIVE_TIMESTEP_SOLVERS`.
V2406_ADAPTIVE = frozenset({
    "XiDyMFoam", "coldEngineFoam", "sonicDyMFoam", "liquidFilmFoam",
    "chtMultiRegionTwoPhaseEulerFoam", "solidFoam",
    "DPMDyMFoam", "MPPICFoam", "MPPICDyMFoam", "coalChemistryFoam", "kinematicParcelFoam",
    "reactingParcelFoam", "reactingHeterogenousParcelFoam", "sprayDyMFoam",
    "cavitatingDyMFoam", "compressibleInterDyMFoam", "compressibleInterFilmFoam",
    "compressibleInterIsoFoam", "overCompressibleInterDyMFoam",
    "compressibleMultiphaseInterFoam", "icoReactingMultiphaseInterFoam",
    "interCondensatingEvaporatingFoam", "interPhaseChangeDyMFoam",
    "overInterPhaseChangeDyMFoam", "potentialFreeSurfaceDyMFoam",
    "reactingMultiphaseEulerFoam", "reactingTwoPhaseEulerFoam",
})

#: In `ADAPTIVE_TIMESTEP_SOLVERS` and wrong there: v2406's `overPotentialFoam.C` builds a
#: `pisoControl` for its non-orthogonal correctors, has no time loop and includes no setDeltaT.H.
V2406_NOT_ADAPTIVE = frozenset({"overPotentialFoam"})

#: Ignore `adjustTimeStep` (no setDeltaT.H, no setMultiRegionDeltaT.H) and are missing from
#: `FIXED_TIMESTEP_SOLVERS`; named so `is_known` knows them.
V2406_FIXED = frozenset({
    "dnsFoam", "acousticFoam", "overLaplacianDyMFoam", "overPotentialFoam", "overRhoSimpleFoam",
    "sonicLiquidFoam", "dsmcFoam", "mdEquilibrationFoam", "mdFoam", "magneticFoam", "mhdFoam",
    "financialFoam", "sphereSurfactantFoam", "surfactantFoam", "buoyantBoussinesqSimpleFoam",
    "chtMultiRegionSimpleFoam", "thermoFoam", "adjointOptimisationFoam", "nonNewtonianIcoFoam",
    "overSimpleFoam", "icoUncoupledKinematicParcelFoam", "icoUncoupledKinematicParcelDyMFoam",
    "simpleReactingParcelFoam", "simpleCoalParcelFoam", "simpleSprayFoam",
    "uncoupledKinematicParcelFoam", "uncoupledKinematicParcelDyMFoam", "solidDisplacementFoam",
    "solidEquilibriumDisplacementFoam",
})


def _v2406(checks: str) -> bool:
    from courant.verify.checks import enabled

    return enabled("solvers.v2406", checks)


def _steady_set(checks: str) -> frozenset[str]:
    if _v2406(checks):
        return (STEADY_SOLVERS | V2406_STEADY) - V2406_NOT_STEADY
    return STEADY_SOLVERS


def _adaptive_set(checks: str) -> frozenset[str]:
    if _v2406(checks):
        return (ADAPTIVE_TIMESTEP_SOLVERS | V2406_ADAPTIVE) - V2406_NOT_ADAPTIVE
    return ADAPTIVE_TIMESTEP_SOLVERS


def _known_set(checks: str) -> frozenset[str]:
    known = ADAPTIVE_TIMESTEP_SOLVERS | FIXED_TIMESTEP_SOLVERS | STEADY_SOLVERS
    return known | V2406_STEADY | V2406_ADAPTIVE | V2406_FIXED if _v2406(checks) else known


#: Solvers whose volumetric face flux `phi` is divergence-free at every step, steady or
#: transient: each solves a pressure (or `p_rgh`) equation that is continuity for a fluid, or a
#: mixture of phases, of constant density (the Boussinesq solvers' density varies only in the
#: buoyancy term). On a fixed mesh the net flux through the boundary of such a run is zero to
#: the pressure solver's tolerance (TRD VER-18). Not here: a solver whose volumetric flux carries
#: an expansion (compressible phases: `compressibleInterFoam`; phase change: `cavitatingFoam`,
#: `interPhaseChangeFoam`; a dispersed phase's volume fraction: `DPMFoam`, the Euler-Euler
#: solvers), a free surface the mesh does not follow (`potentialFreeSurfaceFoam`), and every
#: solver that writes a mass flux, whose boundary balances only at a steady state. A listed
#: solver's flux is divergence-free only when the case adds no mass or volume itself: from
#: OpenFOAM.org 9 phase change is interFoam with the `VoFCavitation` fvModel, and `massSource`
#: and `volumeSource` add to continuity on any solver. The case's `fvModels` and `fvOptions`
#: decide that, not this table (`courant.verification._fluid_reader`, VER-18).
SOLENOIDAL_SOLVERS = frozenset({
    "icoFoam", "nonNewtonianIcoFoam", "pisoFoam", "pimpleFoam", "SRFPimpleFoam",
    "simpleFoam", "SRFSimpleFoam", "porousSimpleFoam", "adjointShapeOptimizationFoam",
    "buoyantBoussinesqSimpleFoam", "buoyantBoussinesqPimpleFoam",
    "interFoam", "interIsoFoam", "interMixingFoam", "multiphaseInterFoam",
})


def keeps_volume_flux_solenoidal(solver: str | None) -> bool:
    """Whether this solver's volumetric `phi` is divergence-free at every step (VER-18). An
    unknown solver gets False: a balance it need not keep is not judged."""
    return bool(solver) and _base(solver) in SOLENOIDAL_SOLVERS


#: The solvers whose run ends on a `PIMPLE/residualControl` met, OpenFOAM.org's `foamRun` and
#: `foamMultiRun` (v11 on): their PIMPLE `residualControl` holds one scalar per field and is the
#: run's convergence criterion, while the outer loop's is `outerCorrectorResidualControl`.
#: ESI's PIMPLE `residualControl` (a dictionary per field, `tolerance` and `relTol`) only ends
#: the outer-corrector loop inside a time step, so it is no run criterion.
PIMPLE_CONVERGENCE_SOLVERS = frozenset({"foamRun", "foamMultiRun"})


def is_multi_run(solver: str | None) -> bool:
    """Whether this is OpenFOAM.org's `foamMultiRun`, whose log names each line's region in a
    prefix (`courant.watcher.parser`, TRD VER-20)."""
    return bool(solver) and _base(solver) == "foamMultiRun"


def residual_control_loop(solver: str | None) -> str | None:
    """Which `fvSolution` dictionary holds this solver's run-level `residualControl`.

    `"SIMPLE"` for the solvers that loop on `simple.loop()` (`simpleControl::criteriaSatisfied`
    ends the run), the transient ones among them; `"PIMPLE"` for OpenFOAM.org's `foamRun`
    family; None for everything else: a transient ESI solver's PIMPLE `residualControl` is a
    tolerance on the outer loop, potentialFoam has no time loop, and an unknown solver's is not
    known to be read at all. Read by the solver's algorithm, never by which dictionaries the
    file happens to contain: a `PIMPLE` block left in a simpleFoam case is never read, and a
    stale `SIMPLE` block in a foamRun case is not either. Asked only by VER-16's bar per field
    (behind `--checks next`, and only on a run that marches to a steady state), so it takes
    v2406's reading in either mode.
    """
    if not solver:
        return None
    base = _base(solver)
    if ((base in STEADY_SOLVERS and base not in V2406_NO_SIMPLE_LOOP)
            or base in V2406_SIMPLE_LOOP or base in V2406_TRANSIENT_SIMPLE_LOOP):
        return "SIMPLE"
    if base in PIMPLE_CONVERGENCE_SOLVERS:
        return "PIMPLE"
    return None


def honours_adjust_time_step(solver: str | None, checks: str = "default") -> bool:
    """Whether setting `adjustTimeStep` on this solver does anything.

    Unknown solvers get ``False``. That is the conservative direction: proposing
    a fix that silently does nothing wastes an attempt from a small budget and
    misinforms the report, whereas declining to propose it only means the loop
    reaches for the timestep reduction instead — which works on every solver.

    Under `--checks next` (`solvers.v2406`) the table is v2406's own list (`V2406_ADAPTIVE`,
    less `V2406_NOT_ADAPTIVE`); under the default, v0.5.0's.
    """
    if not solver:
        return False
    return _base(solver) in _adaptive_set(checks)


def is_steady(solver: str | None, checks: str = "default") -> bool:
    """Whether this solver is steady by construction (no time derivative in its equations).

    Under `--checks next` (`solvers.v2406`) with v2406's own steady solvers (`V2406_STEADY`);
    under the default, v0.5.0's `STEADY_SOLVERS`. A case's declared `ddtSchemes` comes first
    (`CaseConfig.is_steady`); this is the fallback when it was not read.
    """
    if not solver:
        return False
    return _base(solver) in _steady_set(checks)


#: Parallel launchers. A production OpenFOAM run is never invoked bare — it is
#: `mpirun -np 64 <solver> -parallel`, or the scheduler's equivalent.
_LAUNCHERS = frozenset({
    "mpirun", "mpiexec", "orterun", "srun", "aprun", "ibrun", "jsrun",
    "prun", "poe", "runjob", "lamexec", "oshrun", "mpirun.openmpi",
    "mpirun.mpich", "time", "nohup", "stdbuf",
})


def _base(solver: str) -> str:
    """The solver executable's name, with its decoration removed.

    Decoration means a path, a launcher, the launcher's flags, and the solver's
    own flags: `mpirun --allow-run-as-root -np 8 rhoSimpleFoam -parallel` is a
    `rhoSimpleFoam` run.

    THIS USED TO TAKE THE FIRST TOKEN, while its docstring claimed to strip the
    MPI wrapper. It returned `mpirun` for every parallel command, so on every
    parallel run — which is every industrial run — the solver was unrecognised:
    `is_steady` said False, silencing every steady-only check including
    relaxation, and `honours_adjust_time_step` said False, which makes the
    timestep repair propose a fixed step on a solver that supports adaptivity.

    A known solver name anywhere in the command wins outright, because it is
    exact and survives any flag spelling — `-hostfile hosts` would otherwise
    offer `hosts` as the executable. Only when nothing is recognised does this
    fall back to position: the first token that is not a launcher, not a flag,
    and not a flag's numeric value.
    """
    tokens = [tok.rsplit("/", 1)[-1] for tok in solver.strip().split()]
    if not tokens:
        return ""

    # Every table, v2406's included, in either mode: which token is the solver is a fact about
    # the command line, and each v2406 name ends in "Foam", which the rule below already takes.
    known = _known_set("next")
    for tok in tokens:
        if tok in known:
            return tok

    # Naming convention, for a solver we do not have in the table. Nearly every
    # OpenFOAM solver ends in "Foam", and this is what rescues a custom or
    # in-house solver from being mistaken for a launcher flag's value —
    # `-hostfile hosts myCustomFoam` would otherwise resolve to `hosts`.
    for tok in tokens:
        if tok.endswith("Foam") and tok not in _LAUNCHERS:
            return tok

    for tok in tokens:
        if tok in _LAUNCHERS or tok.startswith("-") or "=" in tok:
            continue  # a launcher, its flag, or `-x VAR=value`
        # `-np 4` — the count trailing a launcher flag, not an executable.
        try:
            float(tok)
        except ValueError:
            return tok
    return tokens[0]


def same_solver(a: str, b: str) -> bool:
    """Whether two commands run the same solver executable, decoration removed (`_base`)."""
    return _base(a) == _base(b)


def is_known(solver: str | None, checks: str = "default") -> bool:
    """Whether the capability table names this solver (after stripping any MPI wrapper); under
    `--checks next` (`solvers.v2406`), v2406's own list included."""
    if not solver:
        return False
    return _base(solver) in _known_set(checks)


# ---------------------------------------------------------------- the log's Exec lines
#: An OpenFOAM banner's `Exec   : <program> <arguments>` line.
_EXEC_LINE = re.compile(r"^Exec[ \t]*:[ \t]*(\S+)([^\n]*)$", re.M)
#: `nProcs : N` in a banner: printed by a parallel run only.
_NPROCS = re.compile(r"^nProcs[ \t]*:[ \t]*(\d{1,9})", re.M)
#: How far past its Exec line a banner's `nProcs` is looked for (a banner is about 12 lines).
_BANNER = 2000


@dataclass(frozen=True)
class Exec:
    """One `Exec :` line of a log: the program, without its directory, its arguments, and the
    `nProcs` its own banner printed (None when the banner prints none, as a serial run's)."""

    program: str
    args: tuple[str, ...] = ()
    nprocs: int | None = None

    @property
    def parallel(self) -> bool:
        """`-parallel` among its arguments, or its banner's `nProcs` above 1."""
        return "-parallel" in self.args or (self.nprocs or 0) > 1

    def option(self, name: str) -> str | None:
        """The value after `-<name>` among its arguments (`foamRun -solver incompressibleFluid`)."""
        for i, a in enumerate(self.args[:-1]):
            if a == f"-{name}":
                return self.args[i + 1]
        return None


def exec_lines(log_text: str | None) -> list[Exec]:
    """Every `Exec :` line of the log, in order. A log written by several applications
    (`(blockMesh; icoFoam) > log.run`) has one per application."""
    text = log_text or ""
    found = list(_EXEC_LINE.finditer(text))
    out = []
    for i, m in enumerate(found):
        stop = found[i + 1].start() if i + 1 < len(found) else len(text)
        banner = _NPROCS.search(text, m.end(), min(stop, m.end() + _BANNER))
        out.append(Exec(m.group(1).rsplit("/", 1)[-1], tuple(m.group(2).split()),
                        int(banner.group(1)) if banner else None))
    return out


def names_a_solver(program: str | None) -> bool:
    """Whether `program` is a solver: one the capability tables name or, by OpenFOAM's naming of
    its solvers, a program whose name ends in `Foam` (a custom or in-house solver, as `_base`
    reads one). A utility (blockMesh, decomposePar, postProcess, setFields) is not."""
    if not program:
        return False
    known = ADAPTIVE_TIMESTEP_SOLVERS | FIXED_TIMESTEP_SOLVERS | STEADY_SOLVERS
    return program in known or (program.endswith("Foam") and program not in _LAUNCHERS)


def solver_exec(log_text: str | None) -> Exec | None:
    """The log's solver: its last `Exec :` line that names a solver (`names_a_solver`). A log
    written by several applications names each (`(blockMesh; icoFoam) > log.run`): its first is
    blockMesh, and the solver is the last solver among them. None when no Exec line names one (a
    utility's log, a log cut at its head, `-noBanner`): which solver ran is then not known.

    The one reading of which program ran: conformance's solver item and the verdict's
    `run.solver` (`courant.spec.as_run`, `courant.verify.pipeline`), and the log's parallel
    signal (`courant.case.mesh.ran_parallel`)."""
    return next((e for e in reversed(exec_lines(log_text)) if names_a_solver(e.program)), None)
