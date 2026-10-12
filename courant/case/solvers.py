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


def honours_adjust_time_step(solver: str | None) -> bool:
    """Whether setting `adjustTimeStep` on this solver does anything.

    Unknown solvers get ``False``. That is the conservative direction: proposing
    a fix that silently does nothing wastes an attempt from a small budget and
    misinforms the report, whereas declining to propose it only means the loop
    reaches for the timestep reduction instead — which works on every solver.
    """
    if not solver:
        return False
    return _base(solver) in ADAPTIVE_TIMESTEP_SOLVERS


def is_steady(solver: str | None) -> bool:
    if not solver:
        return False
    return _base(solver) in STEADY_SOLVERS


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

    known = ADAPTIVE_TIMESTEP_SOLVERS | FIXED_TIMESTEP_SOLVERS | STEADY_SOLVERS
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


def is_known(solver: str | None) -> bool:
    """Whether the capability table names this solver (after stripping any MPI wrapper)."""
    if not solver:
        return False
    return _base(solver) in (ADAPTIVE_TIMESTEP_SOLVERS | FIXED_TIMESTEP_SOLVERS | STEADY_SOLVERS)
