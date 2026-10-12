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

NEEDS_THE_RUN = ("Run behaviour (residuals, the Courant number, conservation, bounding) is "
                 "checked once the run exists: courant verify or courant watch.")
NO_MESH = "Mesh quality was not checked: no checkMesh output was found."
NO_CONFIG = "No readable case configuration was found (system/, constant/)."


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
        MeshChecker,
        PhysicsChecker,
        SchemeToleranceChecker,
    )

    return [MeshChecker(), SchemeToleranceChecker(), BoundaryConfigChecker(), PhysicsChecker()]


def setup_findings(config, mesh) -> list:
    """The setup-determined findings at WARN or above, for a config and mesh (PRE-2)."""
    from courant.diagnosis.checkers import CheckContext, run_all_checkers

    if config is None:
        return []
    found = run_all_checkers(CheckContext(run_id="preflight", events=[], config=config,
                                          mesh=mesh, log_text=""), checkers=_checkers())
    return [f for f in found if f.id in SETUP_FINDINGS and f.severity.value != "info"]


def preflight_views(run_id: str, config_view: dict | None, mesh_view: dict | None,
                    inputs: tuple[InputFile, ...] = (), solver: str | None = None) -> Preflight:
    from courant import __version__
    from courant.case.config import CaseConfig
    from courant.case.mesh import MeshQuality
    from courant.verify.pipeline import _finding_input, _kb, _validated

    config = _validated(CaseConfig, config_view)
    mesh = _validated(MeshQuality, mesh_view)
    run = PreflightRun(id=run_id, solver=solver or getattr(config, "solver", None), inputs=inputs)
    tool = ToolInfo(version=__version__)
    if config is None:
        return Preflight(tool=tool, run=run, verdict="NOT_ASSESSABLE", reasons=(),
                         probable_causes=(), not_checked=(NO_CONFIG, NEEDS_THE_RUN),
                         next_action="Run preflight on a case directory with system/ and constant/.")
    found = setup_findings(config, mesh)
    reasons = tuple(Reason(
        code=f.id, kind="finding", severity="critical" if f.severity.value == "critical" else "warn",
        summary=f.title + (f": {f.detail}" if f.detail else ""),
        input=_finding_input(f, "case"), method=f"checker {f.checker} (setup only)",
        threshold=num(f.threshold), value=num(f.value)) for f in found)
    causes = tuple(dict.fromkeys(code for f in found for code in f.suggests))
    action = None
    for code in causes:
        entry = _kb().get(code)
        if entry is not None and entry.canonical_fixes:
            action = f"Suggested, not applied: {entry.canonical_fixes[0].description}"
            break
    not_checked = [NEEDS_THE_RUN] + ([NO_MESH] if mesh is None else [])
    verdict: PreflightValue = "SETUP_FAULT" if reasons else "NO_SETUP_FAULT_FOUND"
    if reasons and action is None:
        action = "Correct the setup above before spending compute on this run."
    return Preflight(tool=tool, run=run, verdict=verdict, reasons=reasons,
                     probable_causes=causes, not_checked=tuple(not_checked), next_action=action)


def preflight_path(path: str | Path, case_json: str | Path | None = None) -> Preflight:
    """A case directory (read with foamlib), or a corpus sidecar's captured views."""
    import json

    from courant.verify.ingest import InputError

    path = Path(path)
    if case_json is not None:
        meta = json.loads(Path(case_json).read_text())
        return preflight_views(Path(case_json).name.replace(".case.json", ""),
                               meta.get("config") or meta.get("config_view"),
                               meta.get("mesh") or meta.get("mesh_view"),
                               (InputFile.of(Path(case_json), Path(case_json).name),),
                               meta.get("solver"))
    if not path.is_dir():
        raise InputError(f"preflight needs a case directory (or --case-json): {path}")
    from courant.case.config import CaseConfig
    from courant.case.mesh import read_mesh_quality
    from courant.case.reads import recording

    with recording(path) as reads:                      # VER-14: the verifier's own recorder
        try:
            config_view = CaseConfig.from_foam(str(path)).model_dump(mode="json")
        except Exception:                               # noqa: BLE001
            config_view = None
        mesh = read_mesh_quality(str(path))
    return preflight_views(path.name, config_view,
                           mesh.model_dump(mode="json") if mesh is not None else None,
                           manifest(path, reads.paths()))


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
