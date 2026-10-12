"""CaseConfig — a structured, diagnosis-relevant view of an OpenFOAM case.

The full case is a tree of dictionaries (controlDict, fvSchemes, fvSolution, the
0/ boundary fields, constant/ properties). Diagnosis only needs a curated subset:
timestep control, relaxation, schemes, linear-solver tolerances, turbulence model,
transport properties, and boundary-condition types. That subset is this model.

Two constructors:
  * ``from_dict`` — build from a plain dict (benchmark ``case_facts.json``; replay).
  * ``from_foam`` — parse a real case directory with foamlib (live path; lazy import).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LinearSolverSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    solver: str | None = None            # GAMG, PCG, smoothSolver, ...
    tolerance: float | None = None
    rel_tol: float | None = None


class CaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    solver: str | None = None

    # controlDict
    delta_t: float | None = None
    end_time: float | None = None
    adjust_timestep: bool | None = None
    max_co: float | None = None
    write_control: str | None = None
    write_interval: float | None = None

    # fvSchemes
    ddt_scheme: str | None = None                        # steadyState, Euler, backward
    div_schemes: dict[str, str] = Field(default_factory=dict)   # term -> scheme

    # fvSolution
    relaxation_factors: dict[str, float] = Field(default_factory=dict)  # field -> factor
    #: Which sub-dictionary each relaxation key came from: ``"fields"``,
    #: ``"equations"``, or ``""`` for a flat mapping.
    #:
    #: `relaxation_factors` deliberately flattens the two, because every checker
    #: and KB predicate only cares *how high* a factor is. A **fix** has to write
    #: the value back, and for that the scope is the difference between a valid
    #: edit and a dry-run rejection — `simpleFoam/pitzDaily` has `equations` and
    #: no `fields`, and guessing wrong cost the certification run a repair.
    relaxation_scopes: dict[str, str] = Field(default_factory=dict)

    #: Case files that exist but could not be parsed, file -> reason.
    #:
    #: Absence of a value and inability to read it are different facts and used
    #: to be the same one. A `foamDictionary`-rewritten `controlDict` yielded a
    #: config with no solver, no timestep and no write control, silently, and
    #: every checker downstream behaved as though the engineer had omitted them.
    unreadable: dict[str, str] = Field(default_factory=dict)
    linear_solvers: dict[str, LinearSolverSpec] = Field(default_factory=dict)
    n_non_orthogonal_correctors: int | None = None
    n_correctors: int | None = None
    #: PIMPLE's outer iterations. None reads as OpenFOAM's default of 1 (PISO mode); only
    #: with more than one may a run legitimately sit above Courant 1 (VER-13).
    n_outer_correctors: int | None = None

    # turbulence
    turbulence_type: str | None = None                   # laminar | RAS | LES
    turbulence_model: str | None = None                  # kEpsilon, kOmegaSST, ...

    # transport / thermophysical
    transport: dict[str, float] = Field(default_factory=dict)   # nu, etc.

    # boundary conditions: patch -> field -> bc type
    boundary: dict[str, dict[str, str]] = Field(default_factory=dict)

    # multi-region (CHT)
    regions: list[str] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "CaseConfig":
        d = dict(data)
        if "linear_solvers" in d and d["linear_solvers"]:
            d["linear_solvers"] = {
                k: (v if isinstance(v, LinearSolverSpec)
                    else LinearSolverSpec(field=k, **v))
                for k, v in d["linear_solvers"].items()
            }
        return cls.model_validate(d)

    @classmethod
    def from_foam(cls, case_dir: str) -> "CaseConfig":  # pragma: no cover - needs foamlib+case
        """Parse a real OpenFOAM case with foamlib (live path)."""
        from courant.case._foam_reader import read_case_config

        return read_case_config(case_dir)

    # --- convenience predicates used by checkers ---
    def is_steady(self) -> bool:
        """Whether this case marches to a steady state.

        Two sources, because either alone is wrong on real cases. `ddtSchemes
        default steadyState` is the direct statement and takes precedence — a
        case can say so explicitly, and a `simpleFoam` case with an explicit
        transient scheme is not steady whatever its solver name suggests.

        But `fvSchemes` is a file we may not have been able to read, and
        `simpleFoam` is steady *by construction*: it has no time derivative to
        discretise. Reading only the scheme meant such a case was treated as
        transient, which silences every steady-only check — the relaxation
        finding among them, on the one solver family where relaxation is the
        thing that matters.
        """
        declared = (self.ddt_scheme or "").strip().lower()
        if declared:
            return declared == "steadystate"

        from courant.case.solvers import is_steady as solver_is_steady

        return solver_is_steady(self.solver)

    def is_laminar(self) -> bool:
        return (self.turbulence_type or "").strip().lower() == "laminar"

    def boundary_types_for(self, field: str) -> dict[str, str]:
        """patch -> bc type for a given field, across all patches."""
        return {
            patch: fields[field]
            for patch, fields in self.boundary.items()
            if field in fields
        }
