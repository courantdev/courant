"""Preflight: the checks no timestep can change, before the solver spends anything (TRD PRE-1 to PRE-3).

A viscosity in the wrong units, a solver told to stop at a tolerance of 0.1, relaxation switched
off on a steady loop, a boundary that fixes both pressure and velocity: these are in the case
before the first timestep, and every core-hour the run then spends is spent on a result that
cannot be trusted. Preflight reads the case as it stands and says so. It needs no log.

It runs the existing checkers on the configuration and mesh metrics alone, and keeps only the
findings that are fully determined by them (`SETUP_FINDINGS`). Their thresholds are the
checkers' own, set before this module existed. Everything that needs the run (residuals, the
Courant number, conservation, bounding) is listed under "not checked before the run".

Read-only, deterministic, offline: the same invariants as `courant verify` (INV-1 to INV-8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from courant.verify.schema import InputFile, Reason, ToolInfo, canonical_json, manifest, num

SCHEMA = "courant.preflight/1"
PreflightValue = Literal["SETUP_FAULT", "NO_SETUP_FAULT_FOUND", "NOT_ASSESSABLE"]

#: PRE-2: the findings a timestep cannot change. Each is computed from the case alone.
SETUP_FINDINGS = frozenset({
    "physics.nu", "solver.loose_tolerance", "scheme.relaxation", "bc.overconstrained",
    "mesh.negative_volume", "mesh.non_orthogonality", "mesh.skewness", "mesh.aspect_ratio",
})

#: The default's sentence. v0.5.0's said "residuals, the Courant number, conservation, bounding"
#: are checked once the run exists, which overstated: under the default no check judges
#: conservation, the log's continuity error is reported and graded only when it is not finite,
#: and the boundary flux balance is not read (TRD VER-18). Corrected in 0.6.0 (S, 22 Oct 2026,
#: H-PREFLIGHT-CONSERVATION): every default preflight document changes in this sentence alone, a
#: named GATE-13 change, and no verdict moves.
NEEDS_THE_RUN = ("Run behaviour (residuals, the Courant number, bounding) is checked once the "
                 "run exists: courant verify or courant watch. Conservation is not judged by "
                 "default: the continuity error the log prints is reported, and graded only when "
                 "it is not finite; the balance of the boundary flux is read only by courant "
                 "verify --checks next.")
#: The sentence under `--checks next`, where the boundary flux balance is read (VER-18).
NEEDS_THE_RUN_NEXT = ("Run behaviour (residuals, the Courant number, bounding) is checked once "
                      "the run exists: courant verify or courant watch. Conservation is judged "
                      "only by courant verify --checks next, as the balance of the boundary flux "
                      "in the latest time's phi, on a run that must balance (steady, or an "
                      "incompressible solver's) and whose phi it can read; the continuity error "
                      "the log prints is reported, and graded only when it is not finite.")
NO_MESH = "Mesh quality was not checked: no checkMesh output was found."
#: The mesh files were read for the cell aspect ratio only (TRD VER-7a).
MESH_PARTIAL = ("Mesh quality was read from constant/polyMesh for the cell aspect ratio only: no "
                "checkMesh output was found, so skewness, non-orthogonality and negative volumes "
                "were not checked.")
NO_CONFIG = "No readable case configuration was found (system/, constant/)."


def needs_the_run(checks: str = "default") -> str:
    """What the run is needed for, in the words true of this `--checks` mode."""
    from courant.verify.checks import enabled

    return NEEDS_THE_RUN_NEXT if enabled("conservation.mass_flux.phi", checks) else NEEDS_THE_RUN


class PreflightRun(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    solver: str | None
    inputs: tuple[InputFile, ...]


class Preflight(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_: str = Field(default=SCHEMA, alias="schema")
    tool: ToolInfo
    run: PreflightRun
    verdict: PreflightValue
    reasons: tuple[Reason, ...]
    probable_causes: tuple[str, ...]
    not_checked: tuple[str, ...]
    next_action: str | None

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode="json", by_alias=True))


def _checkers():
    from courant.diagnosis.checkers import (
        BoundaryConfigChecker,
        CaseIntegrityChecker,
        MeshChecker,
        PhysicsChecker,
        SchemeToleranceChecker,
    )

    # CaseIntegrityChecker says which dictionary could not be read; it is never a setup fault.
    return [MeshChecker(), SchemeToleranceChecker(), BoundaryConfigChecker(), PhysicsChecker(),
            CaseIntegrityChecker()]


def _found(config, mesh, checks: str) -> list:
    from courant.diagnosis.checkers import CheckContext, run_all_checkers

    if config is None:
        return []
    return list(run_all_checkers(CheckContext(run_id="preflight", events=[], config=config,
                                              mesh=mesh, log_text="", checks=checks),
                                 checkers=_checkers()))


def _setup(found: list) -> list:
    return [f for f in found if f.id in SETUP_FINDINGS and f.severity.value != "info"]


def setup_not_checked(config, checks: str = "default") -> list[str]:
    """What the units and boundary rules read none of on this case, in a sentence each (TRD
    PRE-7; the hard-proofing audit, 21 Oct 2026): `courant preflight`'s `not_checked` and `courant
    verify`'s `not_assessed` say it, where they were silent. Reporting, in both modes; no finding.

    - The viscosities `physics.nu` reads are `courant.case.viscosity.viscosities`' (the one
      definition). By default it reads the top-level `nu` only, so the ones only
      `physics.nu.solved` (PRE-6, `--checks next`) reads, a `mu` and each phase's `nu`, are named.
      When even `next` reads none, that is said in both modes.
    - By default `bc.overconstrained` asks `p` (v0.5.0's rule); a case whose solver solves
      `p_rgh`, which `bc.solved_pressure` (PRE-5, `--checks next`) asks, is named. Which
      pressures it asks is `CaseConfig.solved_pressures`, the one definition."""
    from courant.case.viscosity import viscosities
    from courant.verify import sentences

    if config is None:
        return []
    out: list[str] = []
    every = viscosities(config, "next")
    if not every:
        out.append(sentences.VISCOSITY_NONE)
    else:
        read = viscosities(config, checks)
        unread = [v for v in every if v not in read]
        parts = []
        if any(v.source == "mu" for v in unread):
            parts.append("mu in constant/thermophysicalProperties")
        phases = [v.phase for v in unread if v.phase is not None]
        if phases:
            parts.append(f"the nu of phases {', '.join(phases)} in constant/transportProperties")
        if parts:
            out.append(sentences.viscosity_next_only(parts))
    if set(config.solved_pressures("next")) - set(config.solved_pressures(checks)):
        out.append(sentences.PRESSURE_P_RGH)
    return out


def setup_findings(config, mesh, checks: str = "default") -> list:
    """The setup-determined findings at WARN or above, for a config and mesh (PRE-2)."""
    return _setup(_found(config, mesh, checks))


def preflight_views(run_id: str, config_view: dict | None, mesh_view: dict | None,
                    inputs: tuple[InputFile, ...] = (), solver: str | None = None, *,
                    checks: str = "default", computed_here: bool = False) -> Preflight:
    """The preflight on captured views. `checks` is `--checks` (GATE-13): under the default, a
    mesh view computed from the mesh files (VER-7a) is not read, as v0.5.0 did not compute it.
    `computed_here`: the mesh view is `preflight_path`'s own reading of the case directory, so
    VER-7d's fields computed from `constant/polyMesh` stand; from any other view they are dropped
    (`MeshQuality.from_view`), as `courant verify` drops them; and the configuration view is its
    own reading too, so VER-13a's `ddt_read_whole` stands, where a view from anywhere else (an
    MCP `case_json`, a sidecar) states it and it is dropped (`CaseConfig.from_view`), as
    `courant verify` drops it (the review of hp6/final)."""
    from courant import __version__
    from courant.case.config import CaseConfig
    from courant.case.mesh import MeshQuality
    from courant.verify.checks import enabled
    from courant.verify.pipeline import _finding_input, _kb, limitations
    from courant.verify.sentences import not_found, skewness_zero_area

    config = CaseConfig.from_view(config_view, read_here=computed_here)
    mesh = MeshQuality.from_view(mesh_view, computed_here=computed_here)
    if mesh is not None and mesh.source is not None and \
            not enabled("mesh.aspect_ratio.polymesh", checks):
        mesh = mesh_view = None
    run = PreflightRun(id=run_id, solver=solver or getattr(config, "solver", None), inputs=inputs)
    tool = ToolInfo(version=__version__, checks="next" if checks == "next" else None)
    every = _found(config, mesh, checks)
    unread = limitations(every, config)         # what could not be read, as verify says
    if config is None or config.nothing_read():   # PRE-1: no case dictionary was read at all
        return Preflight(tool=tool, run=run, verdict="NOT_ASSESSABLE", reasons=(),
                         probable_causes=(), not_checked=(NO_CONFIG, *unread, needs_the_run(checks)),
                         next_action="Run preflight on a case directory with system/ and constant/.")
    found = _setup(every)
    reasons = tuple(Reason(
        code=f.id, kind="finding", severity="critical" if f.severity.value == "critical" else "warn",
        summary=f.title + (f": {f.detail}" if f.detail else ""),
        input=_finding_input(f, "case", mesh_view), method=f"checker {f.checker} (setup only)",
        threshold=num(f.threshold), value=num(f.value)) for f in found)
    causes = tuple(dict.fromkeys(code for f in found for code in f.suggests))
    action = None
    for code in causes:
        entry = _kb().get(code)
        if entry is not None and entry.canonical_fixes:
            action = f"Suggested, not applied: {entry.canonical_fixes[0].description}"
            break
    not_checked = ([needs_the_run(checks)]
                   + ([not_found(config.missing)] if config.missing else [])
                   + unread + ([NO_MESH] if mesh is None else
                               [MESH_PARTIAL] if mesh.source is not None else []))
    skew_set_aside = skewness_zero_area(mesh)       # VER-7d, as verify says it
    if skew_set_aside:
        not_checked.append(skew_set_aside)
    not_checked += setup_not_checked(config, checks)
    verdict: PreflightValue = "SETUP_FAULT" if reasons else "NO_SETUP_FAULT_FOUND"
    if reasons and action is None:
        action = "Correct the setup above before spending compute on this run."
    return Preflight(tool=tool, run=run, verdict=verdict, reasons=reasons,
                     probable_causes=causes, not_checked=tuple(not_checked), next_action=action)


def preflight_path(path: str | Path, case_json: str | Path | None = None, *,
                   checks: str = "default") -> Preflight:
    """A case directory (read with foamlib), or a corpus sidecar's captured views. With
    `checks="next"` (GATE-13) a case with no checkMesh log has its aspect ratio computed from
    `constant/polyMesh` (VER-7a)."""
    from courant.verify.checks import enabled
    from courant.verify.ingest import InputError, read_sidecar

    enabled("", checks)                                 # refuse an unknown mode first

    path = Path(path)
    if case_json is not None:
        meta = read_sidecar(case_json)
        return preflight_views(Path(case_json).name.replace(".case.json", ""),
                               meta.get("config") or meta.get("config_view"),
                               meta.get("mesh") or meta.get("mesh_view"),
                               (InputFile.of(Path(case_json), Path(case_json).name),),
                               meta.get("solver"), checks=checks)
    if not path.is_dir():
        raise InputError(f"preflight needs a case directory (or --case-json): {path}")
    from courant.case.config import CaseConfig
    from courant.case.mesh import read_mesh_quality
    from courant.case.reads import recording

    with recording(path) as reads:                      # VER-14: the verifier's own recorder
        try:
            config_view = CaseConfig.from_foam(
                str(path), buoyancy=enabled("physics.laminar_regime", checks),
                directives=enabled("case.directives", checks),
            ).model_dump(mode="json")
        except Exception:                               # noqa: BLE001
            config_view = None
        # The aspect ratio from the mesh files (VER-7a), behind --checks next until 0.6.0.
        mesh = read_mesh_quality(str(path),
                                 polymesh=enabled("mesh.aspect_ratio.polymesh", checks))
    return preflight_views(path.name, config_view,
                           mesh.model_dump(mode="json") if mesh is not None else None,
                           manifest(path, reads.paths()), checks=checks, computed_here=True)


def render_text(p: Preflight) -> str:
    out = [f"PREFLIGHT {p.verdict}", f"RUN       {p.run.id}" + (f" · {p.run.solver}" if p.run.solver else "")]
    if p.reasons:
        out += ["", "SETUP FAULTS (found before any timestep)"]
        for i, r in enumerate(p.reasons, 1):
            out.append(f"  {i:<2} {r.code} — {r.summary}")
            out.append(f"     read    {r.input}")
            if r.value is not None or r.threshold is not None:
                out.append(f"     value   {r.value or '—'}   threshold {r.threshold or '—'}")
    if p.probable_causes:
        out += ["", "PROBABLE CAUSES"] + [f"  {c}" for c in p.probable_causes]
    out += ["", "NOT CHECKED BEFORE THE RUN"] + [f"  - {s}" for s in p.not_checked]
    if p.next_action:
        out += ["", f"NEXT  {p.next_action}"]
    return "\n".join(out) + "\n"
