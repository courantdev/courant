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

import re

from pydantic import BaseModel, ConfigDict, Field

#: Boundary conditions that fix a pressure's value on a patch, the solved `p_rgh`'s included
#: (TRD PRE-5): with a fixed velocity on the same patch the coupling is over-constrained.
PRESSURE_FIXING = frozenset({"fixedValue", "totalPressure", "prghPressure", "prghTotalPressure"})
#: The solvers that solve `p_rgh` (`CaseConfig.solves_p_rgh`): interFoam's family
#: (compressibleInterFoam, interIsoFoam, multiphaseInterFoam, interPhaseChangeFoam, ...), the
#: buoyant solvers (rhoReactingBuoyantFoam among them), driftFluxFoam, the Euler multiphase
#: solvers, chtMultiRegionFoam, fireFoam and twoLiquidMixingFoam, ESI's and OpenFOAM.org's names
#: alike.
_P_RGH_SOLVERS = re.compile(r"(?i)inter\w*Foam|buoyant|driftFluxFoam|EulerFoam|"
                            r"chtMultiRegion|fireFoam|twoLiquidMixingFoam")


class LinearSolverSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    solver: str | None = None            # GAMG, PCG, smoothSolver, ...
    tolerance: float | None = None
    rel_tol: float | None = None
    #: The entry's `log`, as written (None when it is not). v2406 prints a solve's "Solving for"
    #: line only when it is not 0 (`fvScalarMatrix.C`, `fvMatrixSolve.C`:
    #: `solverControls.getOrDefault<int>("log", solverPerformance::debug)`), so a field solved
    #: under `log 0` leaves no line in the log (`solver.loose_tolerance` then counts the entry).
    log: float | None = None


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
    #: TRD VER-13a, read under `--checks next` only (`cfl.steady`): `ddtSchemes`' entries other
    #: than `default` (term -> scheme as written), read when `default` is: v2406 looks a field's
    #: own `ddt(T)` up before the default (`schemesLookup::lookupDetail::lookup`), so `default
    #: steadyState` is the whole case's declaration only when every one of them is steadyState
    #: too (`checkers.steady_courant`). {} under the default, which does not read them.
    ddt_terms: dict[str, str | None] = Field(default_factory=dict)
    #: TRD VER-13a, read under `--checks next` only: True when foamlib's reading of `ddtSchemes`
    #: in the case's own `system/fvSchemes` is the solver's, every entry (`_foam_reader.
    #: _ddt_read_whole`: no directive, macro, slash-scoped keyword or `select` in the file, no
    #: `constant/regionProperties`, the block and each of its keywords written once, and the
    #: file's keywords as `courant.case.dictscan` reads them), so that `ddt_scheme` with
    #: `ddt_terms` is the case's declaration; None otherwise. Only `read_case_config` sets it:
    #: `from_dict` never takes it, and `from_view` keeps it only for this run's own reading of
    #: the case directory, since a sidecar, a benchmark's `case_facts.json` or an MCP
    #: `case_json` states what `ddtSchemes` holds, which is no evidence (the review of
    #: hp6/integrate). A VER-13a spare rests on it (`checkers.steady_courant`). It says how the
    #: file reads as it is when verified, not that the solver read that content (the review of
    #: 5e2a8d5: the log can show only a read again, `watcher.parser.reread_fvschemes`).
    ddt_read_whole: bool | None = None
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
    #: The dictionary v2406's `solution::read` relaxes equations from (`eqnRelaxDict_`), key ->
    #: factor in the order foamlib gives: `relaxationFactors/equations`; with neither `fields`
    #: nor `equations`, the flat `relaxationFactors` (its backwards compatibility); empty
    #: otherwise. Its own map, because the flattened one lets `fields { U 0.9; }` overwrite
    #: `equations { U 0.05; }` and `relaxation_factors` stays v0.5.0's (GATE-13). A factor this
    #: reader cannot read (a `table`) is None: an entry OpenFOAM finds, whose factor is not known.
    #: None when the reader that wrote the configuration did not read it (a sidecar from before
    #: 0.6.0), and `convergence.equation_relaxation` reads the flattened map. Set only beside a
    #: non-empty `relaxation_factors`, so it changes no `nothing_read`; not in
    #: `redact.CONFIG_FIELDS_VERBATIM`, so it stays on this machine.
    relaxation_equations: dict[str, float | None] | None = None

    #: Case files that exist but could not be parsed, file -> reason.
    #:
    #: Absence of a value and inability to read it are different facts and used
    #: to be the same one. A `foamDictionary`-rewritten `controlDict` yielded a
    #: config with no solver, no timestep and no write control, silently, and
    #: every checker downstream behaved as though the engineer had omitted them.
    unreadable: dict[str, str] = Field(default_factory=dict)
    #: Values a parsed dictionary writes through a directive the reader did not follow, so did
    #: not read as the solver did (TRD VER-3b): `$nuVal`, an `#include` that may set them,
    #: `#calc`. File -> what and why, quoting what the case wrote (bounded). Stays local: the
    #: verdict's "not assessed" says it; `redact.sanitize_config_view` lets no field it does not
    #: name cross, and the Evidence Bundle's `config_view` leaves it out.
    unread_values: dict[str, str] = Field(default_factory=dict)
    #: The same values, file -> the key paths and how each is written (`macro`, `directive`,
    #: `#inputMode`), never what was written: what the finding `case.unread_value` says, so what
    #: reaches a diagnoser's Evidence Bundle (the review of 21 Oct 2026). A key path is
    #: OpenFOAM's keywords and the entry names the case writes, a phase name in it written as
    #: its place in `phases` (`phase_pseudonym`; the final review of 0.6.0).
    unread_keys: dict[str, str] = Field(default_factory=dict)
    #: The dictionaries every OpenFOAM solver reads (`_foam_reader.REQUIRED`) that the case
    #: directory does not hold, in the order read. A file that exists and cannot be read is
    #: `unreadable`, never here: absent and unavailable are two facts (VER-7, VER-14).
    missing: list[str] = Field(default_factory=list)
    linear_solvers: dict[str, LinearSolverSpec] = Field(default_factory=dict)
    n_non_orthogonal_correctors: int | None = None
    n_correctors: int | None = None
    #: PIMPLE's outer iterations. None reads as OpenFOAM's default of 1 (PISO mode); only
    #: with more than one may a run legitimately sit above Courant 1 (VER-13).
    n_outer_correctors: int | None = None
    #: `residualControl` as the case writes it, per loop dictionary: `{"SIMPLE": {...},
    #: "PIMPLE": {...}}`, each field key (as written; a quoted key is a regular expression, as
    #: OpenFOAM reads it) to its absolute tolerance, in declaration order. SIMPLE's entries are
    #: scalars or a dictionary's `tolerance`; PIMPLE's are its scalar entries only (OpenFOAM.org's
    #: run criterion), never ESI's per-field outer-loop dictionaries. Which one the solver reads
    #: is `courant.case.solvers.residual_control_loop`; the gate's bar follows that one on a
    #: steady run (`courant.verification.convergence.residual_bar`).
    residual_control: dict[str, dict[str, float]] = Field(default_factory=dict)

    # turbulence
    turbulence_type: str | None = None                   # laminar | RAS | LES
    turbulence_model: str | None = None                  # kEpsilon, kOmegaSST, ...
    #: The RAS or LES block's `turbulence` switch as written: False when it is off, None when it
    #: is not written (OpenFOAM's default, on). `is_laminar` asks it under `--checks next`.
    turbulence_switch: bool | None = None

    # transport / thermophysical
    transport: dict[str, float] = Field(default_factory=dict)   # nu, etc.
    #: Each Newtonian phase's `nu` in `constant/transportProperties` (interFoam's `phases (water
    #: air);` and a subdictionary per phase), by phase name; None for a single-phase case. The
    #: names are the case author's, so it does not cross the trust boundary (`redact`). TRD PRE-6.
    phase_nu: dict[str, float] | None = None
    #: The phases `constant/transportProperties` lists, in order (`phases (water air);`); None
    #: for a single-phase case. The case author's words, local like `phase_nu`: a finding names
    #: a phase by its place here (`phase_pseudonym`), as a patch crosses as `patch1` (`redact`).
    phases: list[str] | None = None

    # buoyancy (TRD VER-17): what a buoyant solver read, for the regime check
    #: `constant/g`'s value, m/s^2.
    gravity: tuple[float, float, float] | None = None
    #: The lowest and highest temperature any `fixedValue` wall patch of `0/T` holds (a per-face
    #: list included, as `setFields` writes it): the temperatures the run's walls are held at.
    #: The internal field is an initial condition, and an inlet or open patch is not a wall, so
    #: neither is one of them. Read without the mesh: a wall is a patch whose type in
    #: `constant/polyMesh/boundary` is `wall`; with no mesh, every `fixedValue` patch counts.
    fixed_temperature_range: tuple[float, float] | None = None
    #: The bounding boxes ((min x, y, z), (max x, y, z)) of the centres of the wall faces held at
    #: the highest and at the lowest of those temperatures, from `constant/polyMesh`'s boundary
    #: faces. None when the mesh files cannot be read or the values do not match the faces.
    hot_faces_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    cold_faces_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    #: `thermoType/equationOfState` (`perfectGas`, `rhoConst`, ...).
    equation_of_state: str | None = None
    #: The absolute pressure the state is evaluated at, Pa: `0/p`'s internal field when it is
    #: uniform, else `pRef`.
    pressure: float | None = None

    # boundary conditions: patch -> field -> bc type
    boundary: dict[str, dict[str, str]] = Field(default_factory=dict)

    #: Whether `constant/dynamicMeshDict` moves or changes the mesh, by the keyword that says so
    #: (`courant.case.motion.MeshMotion.by`: "dynamicFvMesh", "mover", "topoChanger",
    #: "refiner"), "unreadable" when the file exists and could not be read, None for a fixed
    #: mesh. Fixed words only. TRD VER-5a: a moving or sliding mesh's balance is not assessed.
    mesh_motion: str | None = None

    # multi-region (CHT)
    regions: list[str] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "CaseConfig":
        """A configuration from a plain dictionary. `ddt_read_whole` is never taken from one: it
        says the case reader read `system/fvSchemes` itself, which a dictionary can only state
        (TRD VER-13a; `read_case_config` sets it on what this returns)."""
        d = {k: v for k, v in data.items() if k != "ddt_read_whole"}
        if "linear_solvers" in d and d["linear_solvers"]:
            d["linear_solvers"] = {
                k: (v if isinstance(v, LinearSolverSpec)
                    else LinearSolverSpec(field=k, **v))
                for k, v in d["linear_solvers"].items()
            }
        return cls.model_validate(d)

    @classmethod
    def from_view(cls, view: dict | None, *, read_here: bool = False) -> "CaseConfig | None":
        """A captured configuration view as a configuration, or None when there is none or it
        does not validate. `read_here`: the view is this run's own reading of the case directory
        (`ingest.from_case_dir`). From any other view (a corpus sidecar, an MCP `case_json`, a
        stored record) `ddt_read_whole` is dropped, as `MeshQuality.from_view` drops VER-7d's
        computed fields: the view states what `ddtSchemes` holds, and VER-13a spares a Courant
        number only on the reader's own reading of it (the review of hp6/integrate)."""
        if not view:
            return None
        try:
            config = cls.model_validate(view)
        except Exception:  # noqa: BLE001 — not a configuration view: none, as with no view
            return None
        if read_here or not config.ddt_read_whole:
            return config
        return config.model_copy(update={"ddt_read_whole": None})

    @classmethod
    def from_foam(cls, case_dir: str, *, buoyancy: bool = True, directives: bool = False,
                  steady_ddt: bool = False) -> "CaseConfig":  # pragma: no cover - needs a case
        """Parse a real OpenFOAM case with foamlib (live path). `buoyancy=False` leaves out
        `constant/g`, and so the wall faces read from the mesh files (TRD VER-17), as the gate
        does under the default `--checks` (GATE-13). `directives=True` reads a dictionary whose
        value a `$macro` or an `#include` writes as the solver reads it (TRD VER-3b), as the gate
        does under `--checks next`; without it such a value is reported in `unread_values` (a
        dictionary foamlib cannot parse at all, in `unreadable`). `steady_ddt=True` reads
        `ddt_terms` and `ddt_read_whole` (TRD VER-13a), as the gate does under `--checks next`
        (`cfl.steady`); without it neither is read."""
        from courant.case._foam_reader import read_case_config

        return read_case_config(case_dir, buoyancy=buoyancy, directives=directives,
                                steady_ddt=steady_ddt)

    def nothing_read(self) -> bool:
        """Whether the reader found no setting at all: every field is unset but the two that say
        what it could not read (`missing`, `unreadable`) and the list of phases (`phases`), which
        names the phases and sets nothing a check reads: a directory holding only `phases (a b);`
        with no Newtonian phase was assessed, where 0.5.0 said NOT_ASSESSABLE (the review of the
        0.6.0 hard-proofing). Such a configuration is no configuration, as a bare log's is
        (VER-7, PRE-1). `mesh_motion` alone is none either: a directory holding only a
        `constant/dynamicMeshDict` says nothing a check reads but that line (TRD VER-5a)."""
        return not self.model_dump(exclude_defaults=True,
                                   exclude={"missing", "unreadable", "phases", "mesh_motion"})

    def interface_patches(self) -> list[str]:
        """The patches a boundary condition of the case makes a non-conformal interface
        (`courant.case.motion.is_interface_type`: `cyclicAMI`, `cyclicACMI`, ...), in the order
        `boundary` lists them. The names are the case author's; the line that reports them
        (`courant.verify.sentences.moving_mesh`) names none."""
        from courant.case.motion import is_interface_type

        return [patch for patch, fields in self.boundary.items()
                if any(is_interface_type(t) for t in fields.values())]

    # --- convenience predicates used by checkers ---
    def write_period(self) -> float | None:
        """The simulated time between two writes, when the case states it (TRD VER-7b):
        `writeInterval` under `runTime` or `adjustableRunTime`, `writeInterval` steps of a fixed
        `deltaT` under `timeStep`. None otherwise (an adjusted step, `cpuTime`, `clockTime`)."""
        if self.write_interval is None or self.write_interval <= 0:
            return None
        if self.write_control in ("runTime", "adjustableRunTime", "adjustable"):
            return self.write_interval
        if self.write_control == "timeStep" and not self.adjust_timestep and self.delta_t:
            return self.write_interval * self.delta_t
        return None

    def is_steady(self, checks: str = "default") -> bool:
        """Whether this case is steady: it has no time derivative to march.

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

        `checks` is `--checks` (GATE-13): under `next` the solver fallback knows v2406's own
        steady solvers (`courant.case.solvers.V2406_STEADY`, TRD VER-21). A local-time-stepping
        case (`localEuler`) is not steady here, whatever the mode: it has a time derivative, a
        pseudo one, and that is what the checks asking this need to know (relaxation is not the
        only damping it has). Where it ended up is `marches_to_steady_state`'s question.
        """
        declared = (self.ddt_scheme or "").strip().lower()
        if declared:
            return declared == "steadystate"

        from courant.case.solvers import is_steady as solver_is_steady

        return solver_is_steady(self.solver, checks)

    def is_local_euler(self) -> bool:
        """Whether this case runs local time stepping (LTS), by OpenFOAM's own test
        (`fv::localEulerDdt::enabled`, v2406): the first word of `ddtSchemes/default` is
        `localEuler`, exactly. Each cell then takes its own time step, set from `maxCo` and the
        PIMPLE dictionary's limits, so the run's time is a pseudo-time and only its end state
        is a result."""
        words = (self.ddt_scheme or "").split()
        return bool(words) and words[0] == "localEuler"

    def marches_to_steady_state(self, checks: str = "default") -> bool:
        """Whether only where this run ended up is its answer, so it is judged on its final
        residuals (TRD VER-19): a steady case, and under `--checks next` (`case.local_euler`) a
        local-time-stepping one, whose pseudo-time steps march to a steady state and are not
        time-accurate. The checks that ask this are the ones about the end state: the residual
        bar (`ConvergenceCheck`'s steady branch, `residual.not_converged`, VER-16's bar per
        field), the fixed-timestep note, and the convergence block's regime. Relaxation and the
        stall detector ask `is_steady`: LTS has a pseudo time derivative, its tutorials ship
        `".*" 1` relaxation, and its final outer corrector (every solve, with the one corrector
        those tutorials use) relaxes with the `<field>Final` factors.

        Under the default an LTS case is transient, as in v0.5.0: judged on stability, which
        for LTS means almost nothing (an LTS solver prints no Courant number), so how far its
        final residuals had fallen never entered a default verdict.
        """
        if self.is_steady(checks):
            return True
        from courant.verify.checks import enabled

        return enabled("case.local_euler", checks) and self.is_local_euler()

    def is_laminar(self, checks: str = "default") -> bool:
        """Whether no turbulence model is solved: `simulationType laminar`, and under
        `case.turbulence_switch` (TRD VER-17a, behind `--checks next`) a RAS or LES case whose
        `turbulence` switch is off. OpenFOAM reads that switch as
        `getOrDefault<Switch>("turbulence", true)` (`RASModel.C`), and a model's `correct()`
        returns at once when it is off: no equation of the model is solved and the eddy viscosity
        stays at its initial field, so the run models no turbulence."""
        if (self.turbulence_type or "").strip().lower() == "laminar":
            return True
        from courant.verify.checks import enabled

        return (self.turbulence_switch is False
                and enabled("case.turbulence_switch", checks)
                and (self.turbulence_type or "").strip().upper() in ("RAS", "LES"))

    def overconstrained_patches(self, checks: str = "default") -> list[str]:
        """The patches whose boundary conditions fix both the pressure and the velocity, in the
        order `boundary` lists them. One definition, for `bc.overconstrained` and the knowledge
        base's `overconstrained_pu`.

        v0.5.0 asked a field named `p` held `fixedValue`. Under `bc.solved_pressure` (TRD PRE-5,
        behind `--checks next`) it asks the pressure the solver solves as well, `p_rgh` on
        interFoam and the buoyant solvers, whose `0/p` is `calculated` or absent, and counts every
        condition that fixes a pressure's value (`PRESSURE_FIXING`)."""
        from courant.verify.checks import enabled

        if not enabled("bc.solved_pressure", checks):
            return [patch for patch, fields in self.boundary.items()
                    if fields.get("p") == "fixedValue" and fields.get("U") == "fixedValue"]
        pressures = self.solved_pressures(checks)
        return [patch for patch, fields in self.boundary.items()
                if fields.get("U") == "fixedValue"
                and any(fields.get(p) in PRESSURE_FIXING for p in pressures)]

    def solves_p_rgh(self) -> bool:
        """Whether the case's solver solves `p_rgh` (the pressure less its hydrostatic part):
        interFoam and its family, the buoyant solvers, driftFluxFoam, the Euler multiphase
        solvers, chtMultiRegionFoam and fireFoam (`_P_RGH_SOLVERS`), when the case has a
        `0/p_rgh`. A solver the reader does not name (no `application`, or OpenFOAM.org's
        `foamRun`, whose solver module it does not read) is taken at the case's word: its own
        `0/p_rgh`. A stray `0/p_rgh` on a solver that solves `p` is not its pressure."""
        if not any("p_rgh" in fields for fields in (self.boundary or {}).values()):
            return False
        if self.solver is None or self.solver in ("foamRun", "foamMultiRun"):
            return True
        return _P_RGH_SOLVERS.search(self.solver) is not None

    def solved_pressures(self, checks: str = "default") -> tuple[str, ...]:
        """The pressure fields `bc.overconstrained` asks (`overconstrained_patches`), one
        definition for it and for what `courant.verify.preflight.setup_not_checked` says it did
        not ask: `p` (v0.5.0's rule), and under `bc.solved_pressure` (TRD PRE-5, `--checks
        next`) `p_rgh` as well on a case whose solver solves it (`solves_p_rgh`)."""
        from courant.verify.checks import enabled

        if enabled("bc.solved_pressure", checks) and self.solves_p_rgh():
            return ("p", "p_rgh")
        return ("p",)

    def boundary_types_for(self, field: str) -> dict[str, str]:
        """patch -> bc type for a given field, across all patches."""
        return {
            patch: fields[field]
            for patch, fields in self.boundary.items()
            if field in fields
        }


def phase_pseudonym(phases, name: str) -> str:
    """How a finding names the phase `name`: `phase1`, `phase2`, ... by its place in `phases`, the
    list `constant/transportProperties` writes; "a phase" when it is not listed. A phase name is
    the case author's word, and a finding reaches the Evidence Bundle, where pattern scrubbing
    cannot recognise it (the final review of the 0.6.0 integration, 21 Oct 2026). One rule for
    every finding that names a phase (`case.unread_value`, `physics.nu`)."""
    listed = [str(p) for p in phases or ()]
    return f"phase{listed.index(name) + 1}" if name in listed else "a phase"
