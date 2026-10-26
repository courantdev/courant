"""Failure taxonomy — the shared vocabulary of CFD failure modes (Tech Arch §6).

This is the top-25 MVP subset (~80% real-world coverage). It is referenced by:
  * benchmark ground truth (what a case's true root cause is),
  * the Knowledge Base (signatures + canonical fixes per code),
  * the Diagnosis Engine (hypotheses must map to a taxonomy code),
  * evaluation scoring (is the true code in the model's top-2?).

Every code is stable and namespaced by class. Escalation tier orders the fix
loop's cheapest-first retry strategy: numerics -> schemes -> mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureClass(str, Enum):
    NUMERICS = "numerics"           # stability / timestep / relaxation / solver
    MESH = "mesh"                   # quality, resolution, topology
    BOUNDARY = "boundary"           # boundary-condition setup
    PHYSICS = "physics"             # model / regime / properties
    CONVERGENCE = "convergence"     # silent errors: plateau, imbalance, oscillation
    ENVIRONMENT = "environment"     # decomposition, missing dicts, resources


class EscalationTier(str, Enum):
    NUMERICS = "numerics"           # cheapest fixes: dt, relaxation, tolerances
    SCHEMES = "schemes"             # fvSchemes / fvSolution changes
    MESH = "mesh"                   # mesh refine/repair (most expensive)


@dataclass(frozen=True)
class TaxonomyEntry:
    code: str
    title: str
    failure_class: FailureClass
    tier: EscalationTier
    summary: str
    #: **Can the product actually emit this code?**
    #:
    #: Three of these could not, and nothing said so. A reachability audit over
    #: the knowledge base, the checkers and 31 real solver logs found codes with
    #: no route to them of any kind — no finding any checker emits, no log
    #: pattern the solver ever prints, no config predicate, no suggestion. They
    #: were in the taxonomy, counted in "25 failure modes", and unreachable.
    #:
    #: A vocabulary is a promise. Listing a failure mode says the product can
    #: name it, and a code that cannot be emitted is a claim that cannot be
    #: backed — so it is marked here rather than quietly counted.
    #:
    #: `why_undetectable` must be set when this is False, and a test enforces
    #: both directions: nothing claims detectability it does not have, and
    #: nothing disclaims it without a reason.
    detectable: bool = True
    why_undetectable: str = ""

    def __post_init__(self) -> None:
        if self.detectable and self.why_undetectable:
            raise ValueError(f"{self.code}: detectable but carries a reason")
        if not self.detectable and not self.why_undetectable:
            raise ValueError(f"{self.code}: undetectable without a reason")


# The 25 canonical failure modes (Tech Arch §6, "top-25 MVP subset").
_ENTRIES: list[TaxonomyEntry] = [
    # --- Numerics / stability ---
    TaxonomyEntry("NUM.CFL_VIOLATION", "Courant number too high",
                  FailureClass.NUMERICS, EscalationTier.NUMERICS,
                  "Timestep too large for the mesh/velocity; Co exceeds stability limit."),
    TaxonomyEntry("NUM.MISSING_ADJUSTABLE_DT", "Fixed dt without adjustableRunTime",
                  FailureClass.NUMERICS, EscalationTier.NUMERICS,
                  "No adaptive timestep control; Co spikes on transients."),
    TaxonomyEntry("NUM.OVER_RELAXATION", "Over-aggressive relaxation factors",
                  FailureClass.NUMERICS, EscalationTier.NUMERICS,
                  "Under-relaxation factors too high; SIMPLE loop destabilizes."),
    TaxonomyEntry("NUM.UNBOUNDED_SCHEME", "Unbounded divergence scheme",
                  FailureClass.NUMERICS, EscalationTier.SCHEMES,
                  "Non-bounded div scheme on a tet-dominant mesh causes oscillation."),
    TaxonomyEntry("NUM.SOLVER_TOLERANCE", "Linear-solver tolerance mismatch",
                  FailureClass.NUMERICS, EscalationTier.NUMERICS,
                  "Linear solver tolerance too loose/tight for the field coupling."),
    TaxonomyEntry("NUM.BAD_INITIAL_FIELD", "Bad initial field (no potentialFoam)",
                  FailureClass.NUMERICS, EscalationTier.NUMERICS,
                  "Uninitialized/poor initial field; first steps diverge without potentialFoam."),

    # --- Mesh quality ---
    TaxonomyEntry("MESH.SKEWNESS", "High skewness / non-orthogonality",
                  FailureClass.MESH, EscalationTier.MESH,
                  "Skewness or non-orthogonality above solver limits without correctors."),
    TaxonomyEntry("MESH.NEGATIVE_VOLUME", "Negative cell volumes",
                  FailureClass.MESH, EscalationTier.MESH,
                  "checkMesh reports negative volumes; mesh is invalid."),
    TaxonomyEntry("MESH.YPLUS_RESOLUTION", "Poor boundary-layer resolution (y+)",
                  FailureClass.MESH, EscalationTier.MESH,
                  "Near-wall y+ incompatible with the chosen wall treatment."),
    TaxonomyEntry("MESH.ASPECT_RATIO", "Extreme aspect ratio",
                  FailureClass.MESH, EscalationTier.MESH,
                  "Aspect-ratio extremes degrade conditioning and stability."),
    TaxonomyEntry("MESH.LEAKY_SNAPPY", "Leaky snappyHexMesh",
                  FailureClass.MESH, EscalationTier.MESH,
                  "snappyHexMesh failed to close the domain; leaked cells."),

    # --- Boundary conditions ---
    TaxonomyEntry("BC.OVERCONSTRAINED_PU", "Over/under-constrained p-U",
                  FailureClass.BOUNDARY, EscalationTier.NUMERICS,
                  "Pressure-velocity coupling over/under specified at boundaries."),
    TaxonomyEntry("BC.REVERSED_OUTFLOW", "Reversed outlet flow (missing inletOutlet)",
                  FailureClass.BOUNDARY, EscalationTier.NUMERICS,
                  "Backflow at an outlet lacking inletOutlet destabilizes the solve."),
    TaxonomyEntry("BC.TURB_INLET", "Inconsistent turbulence inlet values",
                  FailureClass.BOUNDARY, EscalationTier.NUMERICS,
                  "k/epsilon/omega inlet values inconsistent with the flow regime."),
    TaxonomyEntry("BC.WALL_FUNCTION", "Wrong wall functions for the model",
                  FailureClass.BOUNDARY, EscalationTier.NUMERICS,
                  "Wall-function choice incompatible with turbulence model / y+."),

    # --- Physics setup ---
    TaxonomyEntry("PHYS.TURB_REGIME", "Turbulence model unsuited to regime",
                  FailureClass.PHYSICS, EscalationTier.SCHEMES,
                  "Turbulence model inappropriate for the actual Reynolds regime.",
                  detectable=False,
                  why_undetectable=
                  "Whether kEpsilon or kOmegaSST suits a flow is a modelling "
                  "judgement about a regime the log does not report. It needs a "
                  "Reynolds number, which needs a length scale, which is a "
                  "property of the geometry rather than of anything the solver "
                  "prints. PHYS.LAMINAR_TURBULENT covers what the case and the "
                  "log can show: a laminar setup on a run that will not settle "
                  "(physics.laminar_unstable), and, under --checks next, a "
                  "buoyant laminar case whose own numbers make it turbulent "
                  "(TRD VER-17). A laminar run of a turbulent flow that "
                  "converges is not judged on its regime by default (the "
                  "Reynolds-number check is not implemented, TRD VER-7c), and "
                  "this entry is what is left over."),
    TaxonomyEntry("PHYS.LAMINAR_TURBULENT", "Laminar/turbulent mismatch",
                  FailureClass.PHYSICS, EscalationTier.SCHEMES,
                  "Laminar setup for a turbulent regime (or vice versa)."),
    TaxonomyEntry("PHYS.TRANSPORT_UNITS", "Wrong transport properties / units",
                  FailureClass.PHYSICS, EscalationTier.NUMERICS,
                  "Viscosity/density/units wrong, shifting the effective regime."),
    TaxonomyEntry("PHYS.CHT_THERMO", "CHT thermophysical inconsistency",
                  FailureClass.PHYSICS, EscalationTier.NUMERICS,
                  "Conjugate heat transfer thermophysical properties inconsistent across regions."),

    # --- Convergence quality (silent errors) ---
    TaxonomyEntry("CONV.RESIDUAL_PLATEAU", "Residual plateau above tolerance",
                  FailureClass.CONVERGENCE, EscalationTier.NUMERICS,
                  "Residuals flatten above the convergence criterion; not converged."),
    TaxonomyEntry("CONV.OSCILLATION", "Oscillating quantities of interest",
                  FailureClass.CONVERGENCE, EscalationTier.SCHEMES,
                  "Monitored quantities oscillate rather than settle."),
    TaxonomyEntry("CONV.MASS_IMBALANCE", "Mass/energy imbalance",
                  FailureClass.CONVERGENCE, EscalationTier.NUMERICS,
                  "Global mass or energy not conserved within tolerance.",
                  detectable=False,
                  why_undetectable=
                  "A symptom, not a cause, and the number is dimensional. "
                  "OpenFOAM prints `sum local` and `cumulative` continuity "
                  "errors in units of volumetric flow, so any threshold is "
                  "case-specific: measured on this corpus a healthy pitzDaily "
                  "ends at 3.0e-03 and a healthy planarContraction at 2.1e-11, "
                  "eight orders apart with nothing wrong in either. The fault "
                  "that produces an imbalance is diagnosed by its cause — the "
                  "loose pressure tolerance — which is both correct and "
                  "actionable, where 'mass does not balance' is neither. The "
                  "same reasoning applies to grading: `ContinuityCheck` in "
                  "courant/verification/checks.py reports this number and "
                  "deliberately does not deduct on it, after an absolute "
                  "threshold gave two unmodified OpenFOAM tutorials a C."),
    TaxonomyEntry("CONV.MESH_DEPENDENCE", "Not mesh-independent",
                  FailureClass.CONVERGENCE, EscalationTier.MESH,
                  "Key quantities still change with mesh refinement.",
                  detectable=False,
                  why_undetectable=
                  "Not a property of a run. Mesh independence is a property of "
                  "a *sequence* of runs on successively refined meshes, and "
                  "Courant watches one solve. Nothing in a single log, however "
                  "carefully read, can establish it. Detecting this means "
                  "running the case again on a finer mesh, which is a different "
                  "product behaviour and not a checker."),

    # --- Environment / mechanical ---
    TaxonomyEntry("ENV.DECOMP_MISMATCH", "Decomposition mismatch",
                  FailureClass.ENVIRONMENT, EscalationTier.NUMERICS,
                  "nprocs disagrees with decomposeParDict; parallel run fails."),
    TaxonomyEntry("ENV.MISSING_DICT", "Missing fields / dictionaries",
                  FailureClass.ENVIRONMENT, EscalationTier.NUMERICS,
                  "A required field file or dictionary entry is absent."),
]

TAXONOMY: dict[str, TaxonomyEntry] = {e.code: e for e in _ENTRIES}


def get(code: str) -> TaxonomyEntry:
    if code not in TAXONOMY:
        raise KeyError(f"Unknown taxonomy code: {code!r}")
    return TAXONOMY[code]


def all_codes() -> list[str]:
    return list(TAXONOMY.keys())


def by_class(failure_class: FailureClass) -> list[TaxonomyEntry]:
    return [e for e in TAXONOMY.values() if e.failure_class is failure_class]


def detectable_codes() -> list[str]:
    """The codes the product claims it can name, and is measured against.

    Coverage is reported against this rather than against the full 25. Three
    entries stay in the taxonomy because they are real failure modes an engineer
    will recognise and a report may need to *discuss* — but they are not
    something Courant will ever put at the top of a diagnosis, and counting them
    as covered would be claiming a capability that does not exist.
    """
    return [c for c, e in TAXONOMY.items() if e.detectable]


def undetectable() -> list[TaxonomyEntry]:
    """...and the ones it does not, each with the reason."""
    return [e for e in TAXONOMY.values() if not e.detectable]
