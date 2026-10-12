"""KB v1 — the 25 hand-authored taxonomy entries (Tech Arch §6).

Each entry ties a taxonomy code to: the checker Findings that indicate it, log
regexes, config predicates, and canonical fixes ordered by cost tier. Grown from
real runs later; hand-authored first for quality.
"""

from __future__ import annotations

from courant.diagnosis.kb import (
    CanonicalFix,
    KBEntry,
    KBSignature,
    KnowledgeBase,
)
from courant.taxonomy import EscalationTier as T


def _e(code, *, finding_ids=None, log_patterns=None, config_predicates=None,
       fixes=None) -> KBEntry:
    return KBEntry(
        code=code,
        signature=KBSignature(
            finding_ids=finding_ids or [],
            log_patterns=log_patterns or [],
            config_predicates=config_predicates or [],
        ),
        canonical_fixes=fixes or [],
    )


def _fix(desc, tier, hint="") -> CanonicalFix:
    return CanonicalFix(description=desc, tier=tier, diff_hint=hint)


_ENTRIES = [
    # --- Numerics ---
    _e("NUM.CFL_VIOLATION",
       finding_ids=["cfl.max"],
       # **Three digits, not one.** This matched any Courant number at or above
       # 1, which is true of two healthy tutorials in this corpus:
       # `buoyantPimpleFoam/hotRoom` spikes to 25.7 as a 600 K floor patch
       # relaxes into a 300 K room, and `interFoam/damBreak` runs to 1.24 under
       # adaptive control. Both converge. So the pattern handed two points to a
       # CFL diagnosis on runs where `CourantChecker` had already looked at the
       # tail, seen the excursion recover, and deliberately withheld them.
       #
       # Deleting it outright was the first attempt and it was worse: the
       # pattern is genuine corroboration on a real blow-up, and without it
       # confidence on the end-to-end diverging case fell from 0.67 to 0.57,
       # under the fix loop's gate — so a case Courant used to repair went
       # unrepaired. The finding does the nuanced grading; the pattern should
       # only speak where there is nothing to grade. A Courant number in the
       # hundreds is not a startup transient in any case, at any settings.
       log_patterns=[r"Courant Number mean:.*max:\s*(?:\d{3,}|\d+\.?\d*e\+\d+)"],
       fixes=[_fix("Reduce deltaT in controlDict", T.NUMERICS,
                   "controlDict: deltaT -> deltaT/4"),
              _fix("Enable adjustTimeStep with maxCo < 1", T.NUMERICS,
                   "controlDict: adjustTimeStep yes; maxCo 0.8")]),
    _e("NUM.MISSING_ADJUSTABLE_DT",
       finding_ids=["cfl.climbing", "scheme.fixed_dt"],
       config_predicates=["fixed_dt_no_adaptivity"],
       fixes=[_fix("Turn on adjustTimeStep and cap maxCo", T.NUMERICS,
                   "controlDict: adjustTimeStep yes; maxCo 0.8")]),
    _e("NUM.OVER_RELAXATION",
       finding_ids=["scheme.relaxation", "residual.growth"],
       config_predicates=["relaxation_high"],
       fixes=[_fix("Lower relaxationFactors (p=0.3, U=0.7)", T.NUMERICS,
                   "fvSolution.relaxationFactors: p 0.3; U 0.7")]),
    # No `config_predicates=["unbounded_div"]`, deliberately.
    #
    # The predicate scored 2.0 from the *configuration alone* — from the mere
    # presence of `Gauss linear` on a convection term, which is what OpenFOAM
    # ships for `icoFoam/cavity` and is correct there. So even after the checker
    # learned to stay quiet without a symptom, this code went on outranking the
    # true cause on a bad initial field, an extreme aspect ratio and a negative
    # cell volume: three faults where the scheme is blameless and 2.0 was enough
    # to win.
    #
    # The judgement about whether an unbounded scheme is *implicated* lives in
    # `_unbounded_scheme_symptom`, which weighs the run. A KB predicate that
    # re-derives the same conclusion from config alone is not a second opinion,
    # it is the first one with the reasoning removed.
    _e("NUM.UNBOUNDED_SCHEME",
       finding_ids=["scheme.unbounded_div"],
       log_patterns=[r"bounding (?:U|Ux|Uy|Uz)"],
       fixes=[_fix("Switch to a bounded convection scheme", T.SCHEMES,
                   "fvSchemes.divSchemes: div(phi,U) bounded Gauss linearUpwind grad(U)")]),
    _e("NUM.SOLVER_TOLERANCE",
       finding_ids=["residual.not_converged"],
       fixes=[_fix("Tighten linear-solver tolerance / relTol", T.NUMERICS,
                   "fvSolution.solvers.p: tolerance 1e-7; relTol 0.01")]),
    # `residual.nan` and `Initial residual = inf|nan` are both gone: OpenFOAM
    # prints NEITHER on this fault. The real run — `icoFoam/cavity` with a
    # corrupted `0/U` — reports `Courant Number mean: 475 max: 500` on step one
    # and then dies in the linear solver with a floating-point exception, 58
    # lines of log and no residual line anywhere. The old signature was written
    # from what a bad initial field *ought* to look like, so this code could
    # never be emitted and the scenario was diagnosed NUM.CFL_VIOLATION.
    #
    # `cfl.absurd_at_start` is the signature the solver actually produces.
    _e("NUM.BAD_INITIAL_FIELD",
       finding_ids=["cfl.absurd_at_start"],
       fixes=[_fix("Initialize with potentialFoam before the solve", T.NUMERICS,
                   "run: potentialFoam -initialiseUBCs")]),

    # --- Mesh ---
    _e("MESH.SKEWNESS",
       finding_ids=["mesh.non_orthogonality", "mesh.skewness"],
       log_patterns=[r"non-orthogonality Max:\s*(?:[7-9]\d|\d{3})", r"skewness\s*=?\s*[4-9]"],
       fixes=[_fix("Add nNonOrthogonalCorrectors", T.SCHEMES,
                   "fvSolution.SIMPLE: nNonOrthogonalCorrectors 2"),
              _fix("Improve mesh orthogonality", T.MESH)]),
    _e("MESH.NEGATIVE_VOLUME",
       finding_ids=["mesh.negative_volume", "crash.negative_volume"],
       log_patterns=[r"negative volume"],
       fixes=[_fix("Repair/regenerate the offending mesh region", T.MESH)]),
    _e("MESH.YPLUS_RESOLUTION",
       finding_ids=["bounding.nut"],
       log_patterns=[r"y\+.*(?:below|above)"],
       fixes=[_fix("Match near-wall spacing to the wall-function y+ range", T.MESH)]),
    _e("MESH.ASPECT_RATIO",
       finding_ids=["mesh.aspect_ratio"],
       fixes=[_fix("Reduce extreme aspect-ratio cells", T.MESH)]),
    _e("MESH.LEAKY_SNAPPY",
       log_patterns=[r"Cannot find a cell|leak path|leak detected"],
       fixes=[_fix("Fix locationInMesh / close the surface in snappyHexMeshDict", T.MESH)]),

    # --- Boundary ---
    _e("BC.OVERCONSTRAINED_PU",
       finding_ids=["bc.overconstrained"],
       config_predicates=["overconstrained_pu"],
       fixes=[_fix("Use fixedValue U with zeroGradient p (single reference)", T.NUMERICS,
                   "0/p boundary: outlet { type zeroGradient; }")]),
    # THE BARE `reverse flow` PATTERN IS GONE, and this is the second half of a
    # fix that was only half applied. `pisoFoam/RAS/cavity` — an unmodified
    # tutorial — contains the line `setFlow1: ... reverse flow direction at
    # time: 1`, a function object announcing its own configuration. The checker
    # learned that: it counts occurrences and requires several before saying
    # anything, and consults fatal signatures only when the run actually
    # crashed. The knowledge base kept matching the string once, so
    # BC.REVERSED_OUTFLOW went on being scored on the healthy baseline and
    # ranked second on `laminar_on_turbulent`, whose fault is not a boundary
    # condition at all.
    #
    # `flow.reversed_outflow` is the counted, graded version of the same
    # evidence, which is what this should have keyed on from the start.
    _e("BC.REVERSED_OUTFLOW",
       finding_ids=["crash.reversed_outflow", "flow.reversed_outflow"],
       fixes=[_fix("Use inletOutlet at the outlet for U and turbulence", T.NUMERICS,
                   "0/U outlet { type inletOutlet; inletValue uniform (0 0 0); }")]),
    # NO `finding_ids`, and that is the fix rather than an omission.
    #
    # This keyed on `bounding.k`, `bounding.epsilon` and `bounding.omega`, which
    # made it the THIRD place in the codebase deciding what clipping a
    # turbulence field means — and it disagreed with the other two. The checker
    # reads the case: if that field's own convection term is discretised without
    # bounding, the scheme is the mechanism and it says so. The knowledge base
    # ignored that judgement and credited the inlet anyway, three points per
    # finding, twice over — so on `simpleFoam/pitzDaily::unbounded_scheme`,
    # whose fault is literally an unbounded scheme on `div(phi,k)`,
    # BC.TURB_INLET scored 6.0 against NUM.UNBOUNDED_SCHEME's 4.0 and won.
    #
    # It also ranked first on the *unmodified* pitzDaily tutorial.
    #
    # The checker's `suggests` is now the only route here, so this code is
    # reached exactly when the case says the discretisation is innocent.
    _e("BC.TURB_INLET",
       fixes=[_fix("Recompute k, epsilon/omega from intensity + length scale", T.NUMERICS,
                   "0/k, 0/epsilon inlet values from I and L")]),
    _e("BC.WALL_FUNCTION",
       log_patterns=[r"wall function|nutkWallFunction|kqRWallFunction"],
       fixes=[_fix("Match wall functions to the turbulence model and y+", T.NUMERICS)]),

    # --- Physics ---
    _e("PHYS.TURB_REGIME",
       fixes=[_fix("Choose a turbulence model suited to the Reynolds regime", T.SCHEMES)]),
    _e("PHYS.LAMINAR_TURBULENT",
       finding_ids=["physics.laminar_unstable"],
       config_predicates=["laminar"],
       fixes=[_fix("Switch to a RAS model (e.g. kOmegaSST) for the turbulent regime",
                   T.SCHEMES, "momentumTransport: simulationType RAS")]),
    _e("PHYS.TRANSPORT_UNITS",
       finding_ids=["physics.nu"],
       config_predicates=["nu_out_of_range"],
       fixes=[_fix("Correct nu in transportProperties (units m^2/s)", T.NUMERICS,
                   "constant/transportProperties: nu [0 2 -1 0 0 0 0] 1e-5")]),
    # `thermophysical` alone is gone. **Every** compressible and buoyant solver
    # prints `Reading thermophysical properties` on startup, so that single word
    # scored this code 2.0 on every healthy run of two whole solver families —
    # a case with no CHT region anywhere in it. Same class as the `trapFpe`
    # banner and the `reverse flow` function-object announcement: a routine
    # line read as a fault signature.
    #
    # What is left names an actual inconsistency rather than the subject it is
    # about.
    _e("PHYS.CHT_THERMO",
       log_patterns=[r"inconsistent thermo|thermo\.compressible",
                     r"thermophysical properties.*(?:mismatch|inconsistent)",
                     r"different thermo(?:physical)? (?:model|type)"],
       fixes=[_fix("Reconcile thermophysical properties across CHT regions", T.NUMERICS)]),

    # --- Convergence ---
    _e("CONV.RESIDUAL_PLATEAU",
       finding_ids=["residual.plateau", "residual.not_converged"],
       fixes=[_fix("Improve conditioning (relaxation/mesh) or raise iterations", T.NUMERICS)]),
    _e("CONV.OSCILLATION",
       finding_ids=["residual.growth"],
       fixes=[_fix("Use a more diffusive/bounded scheme to damp oscillation", T.SCHEMES)]),
    # `finding_ids` was empty here, so this code could only ever match a log
    # regex — and once uncited hypotheses were ranked below cited ones, that made
    # it structurally undiagnosable. `continuity.not_falling` is its first real
    # detector: OpenFOAM prints the local continuity error every step, the parser
    # always captured it, and nothing read it.
    # `CONV.MASS_IMBALANCE` has **no detector**, and that is now a measured
    # conclusion rather than an oversight.
    #
    # The log pattern that used to be here was exactly backwards:
    # `continuity errors.*cumulative = [1-9]\d*\.?\d*e-0?[0-3]` is unanchored,
    # so `cumulative = 1.05734e-18` — a *healthy* icoFoam cavity — matched on the
    # `e-1` prefix, while `cumulative = 4.35275` on the genuinely broken case did
    # not match at all. It fired on correct runs and stayed quiet on wrong ones.
    # Deleted rather than repaired: the quantity is dimensional and spans eight
    # orders of magnitude across four *correct* tutorials, so no numeric
    # threshold in a regex can separate anything.
    #
    # A trend-based checker was then built and removed. See
    # `certification/README.md` for the measurement that killed it — on the
    # series the parser actually produces, healthy `pisoFoam/RAS/cavity` and the
    # same case with its pressure solver crippled end at 9.35e-10 and 9.28e-10.
    # Identical to two significant figures. PISO's final corrector enforces
    # continuity whatever the tolerance, so the quantity carries no information
    # about that fault and no threshold over it ever could.
    _e("CONV.MASS_IMBALANCE",
       fixes=[_fix("Tighten pressure solver; check boundary flux balance", T.NUMERICS)]),
    _e("CONV.MESH_DEPENDENCE",
       fixes=[_fix("Run a mesh-independence study; refine until QoI stabilizes", T.MESH)]),

    # --- Environment ---
    _e("ENV.DECOMP_MISMATCH",
       finding_ids=["crash.decomp_mismatch"],
       log_patterns=[r"numberOfSubdomains|processor directories"],
       fixes=[_fix("Match numberOfSubdomains to MPI ranks; re-decompose", T.NUMERICS,
                   "system/decomposeParDict: numberOfSubdomains <nprocs>")]),
    _e("ENV.MISSING_DICT",
       finding_ids=["crash.missing_dict"],
       log_patterns=[r"cannot find file|No such file|Cannot open file"],
       fixes=[_fix("Provide the missing dictionary/field file", T.NUMERICS)]),
]


def build_default_kb() -> KnowledgeBase:
    return KnowledgeBase(_ENTRIES)
