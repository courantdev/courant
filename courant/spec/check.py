"""Deck-versus-spec conformance, and a specification drafted from a case (PRD 2.0 P2-SPEC-2).

Conformance answers one question deterministically: is this case the problem that was asked?
RRD RM-18 measured why it matters: on 20 real runs with the right physics and the wrong
input, the reference-free verdict caught 8 (every one a numerical consequence of the fault);
against the task specification, 20 of 20, with 0 of 7 false nonconformances on the untouched
tutorials.
"""

from __future__ import annotations

import math
from pathlib import Path

from courant.case import wordre
from courant.spec import deck
from courant.spec.model import SPEC_SCHEMA, Spec, SpecItem


# ---------------------------------------------------------------- comparison
def _fmt(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(_num(x) for x in v) + ")"
    return _num(v)


def _num(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() and abs(x) < 1e15 else repr(x)


def _close(spec: float, found: float, rel: float, abs_tol: float | None) -> bool:
    tol = max(abs_tol or 0.0, rel * abs(spec))
    if tol == 0.0:
        return found == spec
    return abs(found - spec) <= tol


def bounded_quantity(item: SpecItem, found) -> float | None:
    """SPEC-12: the one number a bound is compared with. A scalar is itself; a vector, its
    magnitude; a blockMeshDict's hex cell counts, the total number of cells. None for text."""
    if found is None or isinstance(found, str):
        return None
    if item.blocks == "cells":
        if isinstance(found, (int, float)):             # the solved mesh's count (SPEC-13)
            return float(found)
        counts = [float(x) for x in found]
        if len(counts) % 3:
            return None
        return float(sum(counts[i] * counts[i + 1] * counts[i + 2]
                         for i in range(0, len(counts), 3)))
    if isinstance(found, (list, tuple)):
        return math.sqrt(sum(float(x) ** 2 for x in found))
    return float(found)


def within(item: SpecItem, found) -> bool | None:
    """SPEC-12: whether the found quantity lies inside the item's bound (inclusive)."""
    q = bounded_quantity(item, found)
    if q is None:
        return None
    return (item.min is None or q >= item.min) and (item.max is None or q <= item.max)


def _bound(item: SpecItem) -> str:
    if item.min is not None and item.max is not None:
        return f"in [{_num(item.min)}, {_num(item.max)}]"
    return f"≥ {_num(item.min)}" if item.min is not None else f"≤ {_num(item.max)}"


def conforms(item: SpecItem, found, *, rel: float, angle_deg: float) -> bool | None:
    """Whether `found` meets the item. None when there is nothing to compare."""
    if item.bounded:
        return within(item, found)
    want = item.sha256 if item.sha256 is not None else item.value
    if want is None or found is None:
        return None
    if item.exact:
        if isinstance(want, str) or isinstance(found, str):
            return str(want).lower() == str(found).lower()
        return [float(x) for x in want] == [float(x) for x in found]
    if isinstance(want, str) or isinstance(found, str):
        return str(want) == str(found)
    want_l = list(want) if isinstance(want, (list, tuple)) else None
    found_l = list(found) if isinstance(found, (list, tuple)) else None
    if want_l is None and found_l is None:
        return _close(float(want), float(found), rel, item.abs_tol)
    if want_l is None or found_l is None or len(want_l) != len(found_l):
        return False
    if len(want_l) == 3:                                # a vector: magnitude, then direction
        ns = math.sqrt(sum(x * x for x in want_l))
        nf = math.sqrt(sum(x * x for x in found_l))
        if not _close(ns, nf, rel, item.abs_tol):
            return False
        if ns == 0.0 or nf == 0.0:
            return ns == nf or nf <= (item.abs_tol or 0.0)
        cos = sum(a * b for a, b in zip(want_l, found_l)) / (ns * nf)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) <= angle_deg
    return all(_close(float(a), float(b), rel, item.abs_tol) for a, b in zip(want_l, found_l))


def _dims_match(item: SpecItem, found_dims) -> bool | None:
    if item.dimensions is None or found_dims is None:
        return None

    def pad(d):
        d = [float(x) for x in d]
        return d + [0.0] * (7 - len(d))

    return pad(item.dimensions) == pad(found_dims)


def _specified(item: SpecItem) -> str | None:
    if item.bounded:
        return _bound(item)
    return _fmt(item.sha256 if item.sha256 is not None else item.value)


def _tolerance(item: SpecItem, rel: float, angle_deg: float) -> str:
    if item.bounded:
        return "bound"
    if item.exact:
        return "exact"
    if isinstance(item.value, str):
        return "exact"
    parts = [f"{100 * rel:g}%"]
    if item.abs_tol:
        parts.append(f"or {_num(item.abs_tol)}")
    if isinstance(item.value, (list, tuple)) and len(item.value) == 3:
        parts.append(f"{angle_deg:g}°")
    return ", ".join(parts)


# ---------------------------------------------------------------- the check
@wordre.bounded                                     # the case's keyword patterns (SPEC-2)
def check_conformance(spec: Spec | dict, case_dir: str | Path | None, *, spec_sha256: str,
                      locked_sha256: str | None = None, checks: str = "default",
                      run=None):
    """The conformance block for one case. Never raises on a case it cannot read.

    `checks` is `--checks` (GATE-13). Under `next` each value is also checked against what ran
    (`conformance.as_run`, TRD SPEC-13, `courant.spec.as_run`): `run` is the solver's log
    (`as_run.Run`), when there is one, and the case's own mesh files are read for the mesh
    items. Under the default the deck alone decides, as in v0.5.0."""
    from courant.spec.impossible import impossibilities
    from courant.verify.checks import enabled
    from courant.verify.schema import ConformanceBlock, ConformanceItem

    as_run = None
    if enabled("conformance.as_run", checks) and case_dir is not None:
        from courant.spec.as_run import AsRun

        as_run = AsRun(case_dir, run)
    if isinstance(spec, dict):
        spec = Spec.model_validate(spec)
    invalid = impossibilities(spec)
    if invalid:
        # VER-5.11: no case can conform to an impossible specification, so the case is not
        # compared to it (and not read for it).
        items = tuple(ConformanceItem(
            name=item.name, file=item.file, location=item.location,
            specified=_specified(item), found=None, conforms=None,
            tolerance=_tolerance(item, spec.rel_tol, spec.angle_tol_deg),
            modelling_choice=item.modelling_choice,
            note="not compared: the specification is impossible in itself")
            for item in spec.items)
        return ConformanceBlock(
            spec_schema=SPEC_SCHEMA, spec_sha256=spec_sha256, locked_sha256=locked_sha256,
            task=spec.task, items=items, checked=0, nonconformant=0, unreadable=len(items),
            invalid=invalid)
    items: list[ConformanceItem] = []
    for item in spec.items:
        rel = item.rel_tol if item.rel_tol is not None else spec.rel_tol
        angle = item.angle_tol_deg if item.angle_tol_deg is not None else spec.angle_tol_deg
        if case_dir is None:
            found = deck.Found(None, item.file, "only a log was given; the case directory is needed")
        else:
            found = deck.read(case_dir, item)
            if as_run is not None:
                found = as_run.read(item, found, rel=rel)
        ok = conforms(item, found.value, rel=rel, angle_deg=angle)
        if ok is None and found.missing:                 # absent from a file read in full (VER-15)
            ok = False
        note = found.note
        shown = found.value
        basis = found.basis
        if item.bounded and found.faces is not None:
            # SPEC-12 on a per-face list: every face must lie inside the bound, never the mode.
            qs = [bounded_quantity(item, f) for f in found.faces]
            if any(q is None for q in qs):
                ok, note = None, "a per-face list whose values a bound cannot be applied to"
            else:
                lo, hi = min(qs), max(qs)
                ok = all(within(item, f) for f in found.faces)
                shown = lo if lo == hi else f"{_num(lo)} to {_num(hi)}"
                basis = f"every one of {len(qs)} face values ({_num(lo)} to {_num(hi)})"
                if not ok:
                    note = f"face values {_num(lo)} to {_num(hi)} in the case, {_bound(item)} specified"
        elif item.bounded and found.value is not None:
            q = bounded_quantity(item, found.value)
            shown = q if q is not None else found.value
            if ok is None:
                note = "a bound applies to a number; the case holds text"
            elif ok is False:
                note = f"{_fmt(q)} in the case, {_bound(item)} specified"
        dims_ok = _dims_match(item, found.dimensions)
        if ok is not None and dims_ok is False:
            ok = False
            note = (f"dimensions {_fmt(found.dimensions)} in the case, "
                    f"{_fmt(item.dimensions)} specified")
        items.append(ConformanceItem(
            name=item.name, file=found.file, location=item.location,
            specified=_specified(item),
            found=_fmt(shown), conforms=ok, tolerance=_tolerance(item, rel, angle),
            modelling_choice=item.modelling_choice, note=note, basis=basis))
    return ConformanceBlock(
        spec_schema=SPEC_SCHEMA, spec_sha256=spec_sha256, locked_sha256=locked_sha256,
        task=spec.task, items=tuple(items),
        checked=sum(1 for i in items if i.conforms is not None),
        nonconformant=sum(1 for i in items if i.conforms is False),
        unreadable=sum(1 for i in items if i.conforms is None))


# ---------------------------------------------------------------- drafting a spec
#: Boundary conditions whose parameters are part of the problem, and the entry that holds
#: each. A wall function's `value`, or a `calculated` patch's, is an initial guess, not a
#: boundary condition anyone asked for, and is left out. The one table of which entry a type
#: applies: the drafter drafts these, and `courant.spec.as_run` (SPEC-13, under --checks next)
#: compares a patch item only on an entry its type applies.
BC_PARAMETERS = {
    "fixedValue": ("value",),
    "movingWallVelocity": ("value",),
    "inletOutlet": ("inletValue",),
    "outletInlet": ("outletValue",),
    "totalPressure": ("p0",),
    "prghPressure": ("p",),
    "prghTotalPressure": ("p0",),
    "fixedGradient": ("gradient",),
    "flowRateInletVelocity": ("volumetricFlowRate", "massFlowRate"),
    "turbulentIntensityKineticEnergyInlet": ("intensity",),
    "turbulentMixingLengthDissipationRateInlet": ("mixingLength",),
    "turbulentMixingLengthFrequencyInlet": ("mixingLength",),
}

#: (file, block path, key, name, modelling choice) for the properties a task names.
_PROPERTIES = (
    ("constant/transportProperties", (), "nu", "nu", False),
    ("constant/physicalProperties", (), "nu", "nu", False),
    ("constant/thermophysicalProperties", ("mixture", "transport"), "mu", "mu", False),
    ("constant/thermophysicalProperties", ("mixture", "thermodynamics"), "Cp", "Cp", False),
    ("constant/thermophysicalProperties", ("mixture", "specie"), "molWeight", "molWeight", False),
    ("constant/transportProperties", (), "sigma", "surface tension", False),
    ("constant/g", (), "value", "gravity", False),
    ("system/controlDict", (), "application", "solver", True),
    ("system/controlDict", (), "endTime", "endTime", False),
    ("system/blockMeshDict", (), "scale", "mesh scale", False),
    ("system/blockMeshDict", (), "convertToMeters", "mesh scale", False),
    ("constant/turbulenceProperties", (), "simulationType", "turbulence regime", True),
    ("constant/turbulenceProperties", ("RAS",), "RASModel", "turbulence model", True),
    ("constant/turbulenceProperties", ("LES",), "LESModel", "turbulence model", True),
    ("constant/momentumTransport", (), "simulationType", "turbulence regime", True),
    ("constant/momentumTransport", ("RAS",), "model", "turbulence model", True),
    ("constant/momentumTransport", ("LES",), "model", "turbulence model", True),
)

#: Drafted under `--checks next` only (SPEC-13; the default drafts as v0.5.0 did, GATE-13), and
#: only for a case whose `application` is OpenFOAM.org's foamRun: every such case runs the same
#: program, and what solves is its module (`solver`, or `-solver` on the Exec line).
_MODULE = ("system/controlDict", (), "solver", "solver module", True)

_PHASES_FILE = "constant/transportProperties"
_TEXT_KEYS = {"application", "simulationType", "RASModel", "LESModel", "model", "solver"}


def _probe(case: Path, item: SpecItem):
    found = deck.read(case, item)
    return found.value


@wordre.bounded                                     # the case's keyword patterns (SPEC-2)
def draft_spec(case_dir: str | Path, *, task: str | None = None,
               include_mesh: bool = True, checks: str = "default") -> Spec:
    """A specification of the problem this case sets up: its boundary-condition parameters,
    material properties, body forces, run length, geometry scale and modelling choices.

    Meant to be generated from the *reference* case a task is written against, reviewed, and
    then locked before any agent runs (`courant spec hash`). Only values the reader can
    actually read are included, so a draft checked against its own case, in the mode it was
    drafted for (`checks`, GATE-13), conforms, with one exception under `next`: a value under
    `0/` read at a start time `courant spec check` cannot know. A run reference with
    `startFrom latestTime` and several time directories (pisoFoam/RAS/cavity ships it) has no log
    there to say where it started, so its `0/` items are not checked (VER-5.9) until a log is
    given.
    """
    from courant.spec.as_run import AsRun, names_the_mesh
    from courant.verify.checks import enabled

    case = Path(case_dir)
    items: list[SpecItem] = []
    names: set[str] = set()
    #: SPEC-13, under `conformance.as_run` (--checks next): blockMeshDict's cells and scale are
    #: drafted only when the case's own mesh is the one blockMeshDict describes; on a mesh
    #: refined, scaled or snapped after blockMesh they name a mesh nobody solved on, and the draft
    #: would not conform to its own case under next. The default drafts as v0.5.0 did.
    solved = AsRun(case) if enabled("conformance.as_run", checks) else None

    def add(**kw) -> None:
        if kw["name"] in names:
            return
        try:
            item = SpecItem(**kw)
        except ValueError:
            return
        if _probe(case, item) is None:
            return
        if solved is not None and names_the_mesh(item):
            written = deck.read(case, item)
            if solved.read(item, written, rel=0.0).value != written.value:
                return
        names.add(item.name)
        items.append(item)

    # Boundary-condition parameters. Fields are listed from 0/ and 0.orig/; each value is read
    # the way conformance will read it, so one rewritten by the run (setFields) comes from
    # 0.orig/ and one that cannot be read at all is left out of the draft.
    fields = sorted({p.name for d in ("0", "0.orig") if (case / d).is_dir()
                     for p in (case / d).iterdir()
                     if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".orig")})
    for name in fields:
        src = "0" if (case / "0" / name).is_file() else "0.orig"
        text, _ = deck._read_text(case, f"{src}/{name}")
        try:
            bf = deck.block(text or "", "boundaryField")
        except deck.KeywordNotEvaluated:
            bf = None                       # no draft item from a block that cannot be told
        if bf is None:
            continue
        for patch, body in deck.blocks(bf):
            bc_type = deck.entry(body, "type")
            for key in BC_PARAMETERS.get(bc_type or "", ()):
                if deck.entry(body, key) is None:
                    continue
                label = patch if not any(c in patch for c in "()|*.[") else f'"{patch}"'
                item_kw = {"name": f"{name} at {label}" + ("" if key == "value" else f" ({key})"),
                           "file": f"0/{name}", "patch": patch,
                           "key": None if key == "value" else key}
                found = deck.read(case, SpecItem(**item_kw, value=0.0))
                if found.value is None:
                    continue
                add(**item_kw, value=found.value)

    # Properties, forces, run control, modelling choices.
    properties = _PROPERTIES
    if solved is not None and _probe(case, SpecItem(name="a", file=_MODULE[0], key="application",
                                                    value="x")) == "foamRun":
        properties = (*_PROPERTIES, _MODULE)
    for file, path, key, name, choice in properties:
        if not (case / file).is_file():
            continue
        is_text = key in _TEXT_KEYS
        probe = SpecItem(name=name, file=file, path=path, key=key,
                         value="x" if is_text else 0.0)
        found = deck.read(case, probe)
        if found.value is None:
            continue
        dims = found.dimensions if found.dimensions else None
        add(name=name, file=file, path=path, key=key, value=found.value,
            dimensions=dims, modelling_choice=choice)

    # Per-phase properties (multiphase cases name their phases).
    phases_raw = deck.entry(deck._read_text(case, _PHASES_FILE)[0] or "", "phases")
    if phases_raw:
        for phase in phases_raw.strip("() ").split():
            for key in ("nu", "rho"):
                probe = SpecItem(name=f"{phase} {key}", file=_PHASES_FILE, path=(phase,),
                                 key=key, value=0.0)
                found = deck.read(case, probe)
                if found.value is not None:
                    add(name=f"{phase} {key}", file=_PHASES_FILE, path=(phase,), key=key,
                        value=found.value, dimensions=found.dimensions or None)

    # The mesh: cell counts are a modelling choice; surface geometry is identity.
    if include_mesh and (case / "system/blockMeshDict").is_file():
        cells = _probe(case, SpecItem(name="c", file="system/blockMeshDict", blocks="cells",
                                      value=(0.0,)))
        if cells:
            add(name="mesh cells", file="system/blockMeshDict", blocks="cells",
                value=tuple(cells), modelling_choice=True)
    tri = case / "constant" / "triSurface"
    if tri.is_dir():
        for surf in sorted(p for p in tri.iterdir()
                           if p.suffix.lower() in (".stl", ".obj", ".vtk", ".ftr", ".gz")):
            rel = f"constant/triSurface/{surf.name}"
            add(name=f"geometry {surf.name}", file=rel,
                sha256=deck.read(case, SpecItem(name="g", file=rel, sha256="0")).value)

    if not items:
        raise ValueError(f"nothing in {case} could be read into a specification")
    return Spec(schema=SPEC_SCHEMA, task=task, items=tuple(items))
