"""The heat balance a steady run's own `wallHeatFlux` output shows (TRD VER-22).

Read only under `courant verify --checks next` (`conservation.energy.wallheatflux`, GATE-13); the
pipeline calls `read_heat`, and `courant.verification.checks.energy_balance` grades what it
returns. Nothing here rebuilds a heat flux from `T`, a conductivity or the mesh: the integrals
are OpenFOAM's own.

**What v2406's `wallHeatFlux` writes** (src/functionObjects/field/wallHeatFlux/wallHeatFlux.C,
read in the pinned image): a table headed `Time patch min max integral`, one row per wall patch
at each execution, `integral = gSum(magSf*hfp)` with `hfp = alpha*snGrad(he)`, less `qr` where
the run has one, so positive into the region. Its patches are the wall patches only
(`findPatchIDs<wallPolyPatch>`). The table is
`postProcessing/[<region>/]<name>/<startTime>/wallHeatFlux*.dat` (`writeFile::baseFileDir` adds
the region for any but the default one); a file of that name is taken for one only when its
header names those five columns.

**Which balances are owed.** The regions and their kinds are `constant/regionProperties`'s
(`courant.case._foam_reader.read_region_properties`); with none the case is one region,
OpenFOAM's `region0`. Each region's patches are the boundary file of the mesh the solver read
for it (`courant.case.mesh.solver_mesh`, the one rule, with its `region`: a parallel run's
`processor<N>/constant/[<region>/]polyMesh`, every rank's patches together; a mesh it cannot
name is said and not judged), read by `courant.case.polymesh.read_boundary`. A region whose rows
name a patch that is not a wall of that boundary is not judged: the boundary read is not the
mesh the heat flux was written on. A region closed to flow (every patch with faces `empty`,
coupled, `symmetry`, `symmetryPlane`, `wedge`, or a wall type `WALL_TYPES`) owes |net| over its
heat in, or over its heat out, the denominator chosen by the boundary balance's own rule
(`MassFluxCheck.denominator`). A `mappedWall` patch and the patch its `sampleRegion` and
`samplePatch` name, a wall of that region, owe |q_a + q_b| over the larger of |q_a| and |q_b|,
held to the same rule with the larger side as the heat in. Each row is the one at the log's last
time (`courant.spec.as_run._same_time`).

**Round-off, and heat that crosses a patch both ways** (`MassFluxCheck.round_off`, the one test).
The scale is never the sums judged: an integral is round-off next to the largest |integral| in
every row the case wrote at that time, every region's, and a face's heat flux next to the largest
|min| or |max| in those rows (VER-18's scale is likewise the largest face flux in its file, not
the boundary's sums). A patch whose `min` is below zero and `max` above, neither round-off,
carries heat both in and out across its faces: its integral is a net, so the heat in read is less
than the heat that crosses and the measure is more than the balance's. A balance with such a
patch is judged only where its measure is graded OK; any other is not judged, and said. So a
passive region whose heat enters and leaves through one patch is never failed on its residual.

**Not judged, and why** (`HeatReading.sentences`, never a traceback): a run that is not steady
or that another solver ran (`checks.run_as_set_up`, the boundary flux's own test; for a
multi-region case the solver table and each region's own `system/<region>/fvSchemes`, never the
case's, which its solver does not read: `_regions_steady`); a source or a radiation model where
the solver looks for one, in the case or a region (`courant.case.reynolds.source_file`,
`_SOURCE_LINE`, `_radiates`); a solid whose transport is anisotropic, or whose
thermophysicalProperties cannot be read whole (wallHeatFlux reports a scalar conductivity:
`_anisotropic`); a regionProperties or a boundary file that cannot be read, or a mesh directory
that cannot be searched; a region open to flow (the heat its flow carries is not read); a wall
patch with no row at the last time; a row that does not parse; two different rows for one patch
at one time; a mapped wall whose other side its entry does not name by `sampleRegion` and
`samplePatch` (a `coupleGroup`, or `mappedWallVariableThickness`). A patch with no faces is
never a scale nor a patch both ways (its row is `1e+300 -1e+300 0`). A run that wrote no table gets nothing:
no reading, no sentence.

Every file is opened through `courant.case.reads` (`may_read`, `reading`), so it joins the
verdict's manifest (VER-14) and is confined under the MCP tools (MCP-3).

**The case's words.** Every region, patch and case path a sentence or the check's reason cites
is written between backticks (`_q`), so the support bundle's pseudonymiser
(`courant.verify.support`, SEC-6) maps each one (`cited`), whether or not a file of it is in the
manifest. A name the pseudonymiser maps is mapped wherever it appears in the verdict.
"""

from __future__ import annotations

import gzip
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from courant.case.mesh import is_time
from courant.case.reads import may_read, plain_or_gz, reading
from courant.case.reynolds import _CONSTRAINT

#: OpenFOAM's `polyMesh::defaultRegion`: the one region of a case with no regionProperties.
DEFAULT_REGION = "region0"
#: The patch types v2406's `wallHeatFlux` reports: `wallPolyPatch` and the classes deriving
#: from it, found in the pinned image (opencfd/openfoam-default:2406) with
#:   grep -rlE --include=*.H "public[[:space:]]+(Foam::)?(mappedWallPolyPatch|wallPolyPatch)\b" \
#:       /usr/lib/openfoam/openfoam2406
#: (mappedWallPolyPatch.H, mappedVariableThicknessWallPolyPatch.H) and their `TypeName`s.
WALL_TYPES = frozenset({"wall", "mappedWall", "mappedWallVariableThickness"})
#: The patch type whose `sampleRegion` and `samplePatch` name the other side of a region
#: interface (VER-22 (b)).
INTERFACE_TYPE = "mappedWall"
#: Patch types through which no heat is carried by a flow and none is reported: the direction a
#: 2-D case does not solve, mirrors and a wedge's sides. Coupled types are `_fluid_reader._COUPLED`.
_CLOSED_TYPES = _CONSTRAINT                 # `courant.case.reynolds`'s, the one list
#: The columns of v2406's table, as its header names them (`writeFileHeader`).
COLUMNS = ("Time", "patch", "min", "max", "integral")
_TABLE_GLOBS = ("wallHeatFlux*.dat", "wallHeatFlux*.dat.gz")
#: A case-authored word in a sentence or reason of this reader's: between backticks (`_q`).
_QUOTED = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class Balance:
    """One balance judged: a region's (`key` its name) or an interface's (`key` its two patch
    names), its measure, and the words and files its reason cites."""

    key: str | tuple[str, str]
    measure: float
    #: "the case" / "region plate" / "the interface plate_to_air (plate) and air_to_plate (air)".
    where: str
    time: float
    #: The integrals the measure is made of, in words.
    sums: str
    #: What the measure is a fraction of ("the heat in", "the larger side").
    scale_name: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class HeatReading:
    """What `read_heat` read and judged owed. Empty (no rows, no sentence) for a run that wrote
    no `wallHeatFlux` table."""

    #: {region: {patch: the integral at the log's last time}}, every row read whole.
    rows: dict[str, dict[str, float]] = field(default_factory=dict)
    #: {region: its table, relative to the case} (the first, where a region has several).
    files: dict[str, str] = field(default_factory=dict)
    balances: tuple[Balance, ...] = ()
    #: Under "not assessed": what was not judged, and why.
    sentences: tuple[str, ...] = ()

    @property
    def measures(self) -> dict:
        return {b.key: b.measure for b in self.balances}


def _say(why: str) -> str:
    from courant.verification.checks import ENERGY

    return f"{ENERGY[1]}: {why}"


def _q(word: str) -> str:
    """A case-authored word (a region, a patch, a path under the case) as a sentence cites it:
    between backticks, so `cited` finds it (a backtick in it is written as a quote)."""
    return "`" + str(word).replace("`", "'") + "`"


def cited(text: str) -> list[str]:
    """The case-authored words `text`, a sentence or reason of this reader's, cites (`_q`), in
    order: what the support bundle's pseudonymiser maps (`courant.verify.support`, SEC-6)."""
    return [m.group(1) for m in _QUOTED.finditer(text or "")]


def _region_dir(region: str) -> str:
    return "" if region == DEFAULT_REGION else f"{region}/"


def _named(region: str) -> str:
    return "the case" if region == DEFAULT_REGION else f"region {_q(region)}"


def _listed(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


# ------------------------------------------------------------------------------ the tables
def _tables(case: Path) -> list[tuple[str, str]]:
    """(region directory or None, the table relative to the case) for every file named as a
    `wallHeatFlux` table under `postProcessing/[<region>/]<name>/<startTime>/`, sorted."""
    base = case / "postProcessing"
    if not may_read(base) or not os.path.isdir(base):    # MCP-3: listed, so confined like a read
        return []
    found = []
    for pattern in _TABLE_GLOBS:
        for depth, prefix in ((4, "*/*/"), (5, "*/*/*/")):
            for p in base.glob(prefix + pattern):
                rel = p.relative_to(case).as_posix()
                parts = rel.split("/")
                # os.path.isfile, never Path.is_file: a directory that can be listed and not
                # searched is no table, where Python 3.11's raises PermissionError.
                if len(parts) != depth or not is_time(parts[-2]) or not os.path.isfile(p):
                    continue
                found.append((parts[1] if depth == 5 else None, rel))
    return sorted(set(found), key=lambda x: x[1])


def _lines(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


class _TooLarge(ValueError):
    """The heat tables hold more text than one reading reads (`courant.converge.series.MAX_BYTES`,
    the bound every other postProcessing table is read under), decompressed."""


def _read_table(case: Path, rel: str, last: float, budget: list[int]):
    """(is a v2406 wallHeatFlux table, {patch: [(the integral, min, max) as read]} at `last`,
    the patches of the rows at `last` that do not parse (None for a row whose time does not)).
    Raises `OSError` when it cannot be read, `_TooLarge` past `budget[0]` characters (every
    table of the reading drawing on one budget), and anything a broken compressed file raises."""
    from courant.spec.as_run import _same_time

    rows: dict[str, list[tuple[float, float, float]]] = {}
    bad: list[str | None] = []
    header = False
    path = case / rel
    with reading(path), _lines(path) as fh:
        while True:
            line = fh.readline(budget[0] + 1)
            if not line:
                break
            budget[0] -= len(line)
            if budget[0] < 0:
                raise _TooLarge(rel)
            text = line.strip()
            if not text:
                continue
            if text.startswith("#"):
                header = header or tuple(text[1:].split()) == COLUMNS
                continue
            if not header:
                return False, {}, []
            parts = text.split()
            try:
                t = float(parts[0])
            except (ValueError, IndexError):
                bad.append(None)
                continue
            if not _same_time(t, last):
                continue
            try:
                if len(parts) != len(COLUMNS):
                    raise ValueError
                lo, hi, value = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                bad.append(parts[1] if len(parts) > 1 else None)
                continue
            rows.setdefault(parts[1], []).append((value, lo, hi))
    return header, rows, bad


# ------------------------------------------------------------------------------ the reading
def read_heat(case_dir: str, last: float | None, config, solver: str | None,
              checks: str = "default", log_text: str | None = None) -> HeatReading:
    """The heat balances a steady run's `wallHeatFlux` output shows (module docstring): the rows
    at `last` (the log's last time), the balances owed and their measures, and a sentence for
    each that is not judged. `config` is the case's (`CaseConfig`), `solver` the solver that ran
    (the verdict's), `checks` the verdict's mode, `log_text` the solver's log, where it says it
    built a source or a radiation model, and whether it ran in parallel and from which time (the
    mesh it read, `courant.case.mesh.solver_mesh`)."""
    if not case_dir:
        return HeatReading()
    case = Path(case_dir)
    tables = _tables(case)
    if not tables:
        return HeatReading()
    if last is None or not math.isfinite(last):
        return HeatReading(sentences=(_say(
            "the run wrote a wallHeatFlux table and the log shows no time step to read it at, so "
            "the heat balance is not judged"),))
    from courant.case._foam_reader import (REGION_PROPERTIES, multi_region,
                                           read_region_properties)

    # A multi-region case (`multi_region`, the one predicate: a solver that reads regions and a
    # regionProperties): its solver reads each region's schemes, never system/fvSchemes, so
    # steadiness is the regions' (`_regions_steady`). A stray regionProperties beside a
    # single-region solver is not read.
    multi = multi_region(case, solver or getattr(config, "solver", None))
    why = _not_owed(case, config, solver, checks, log_text, multi)
    if why is not None:
        return HeatReading(sentences=(_say(why),))

    try:
        kinds = read_region_properties(str(case)) if multi else None
    except Exception:                                   # noqa: BLE001 - said, never raised
        return HeatReading(sentences=(_say(
            f"{REGION_PROPERTIES.replace(os.sep, '/')}, which names the case's regions, could "
            f"not be read whole, so the heat balance is not judged"),))
    regions = [DEFAULT_REGION] if kinds is None else [r for names in kinds.values() for r in names]
    if kinds is None:
        # One region (a solidFoam run among them): refused only where its file reads whole and
        # names an anisotropic transport; a fluid's thermo is not asked to read whole.
        why = _anisotropic(case, DEFAULT_REGION, strict=False)
        if why is not None:
            return HeatReading(sentences=(_say(why),))
    if kinds is not None:
        why = _regions_steady(case, regions, solver or config.solver, checks)
        if why is not None:
            return HeatReading(sentences=(_say(why),))
        for region in regions:
            why = _source_or_radiation(case, region, None)
            if why is not None:
                return HeatReading(sentences=(_say(why),))
        for region in kinds.get("solid", ()):
            why = _anisotropic(case, region)
            if why is not None:
                return HeatReading(sentences=(_say(why),))
    return _judge(case, tables, regions, kinds is not None, last, log_text)


def _not_owed(case: Path, config, solver, checks, log_text, multi: bool = False) -> str | None:
    """Why the run as a whole owes no heat balance that is judged, or None. A multi-region
    case's steadiness is not asked here (`_regions_steady`): its system/fvSchemes is not read by
    its solver."""
    from courant.verification.checks import run_as_set_up

    if config is None:
        return ("the run wrote a wallHeatFlux table and the case configuration was not read, so "
                "whether its heat must balance (a steady run) is unknown")
    other, steady = run_as_set_up(config, solver, checks)
    if other:
        return (f"the solver that ran is {solver} and system/controlDict names {config.solver}: "
                f"on a case set up for another solver, the heat balance is not judged")
    if not steady and not multi:
        return ("the run is not steady: the heat stored inside changes, and it is not read, so "
                "the heat balance of its walls is not judged")
    if getattr(config, "mesh_motion", None) is not None:
        return ("the mesh moves (constant/dynamicMeshDict): the heat balance of its walls is not "
                "judged")
    # A multi-region solver reads each region's sources and radiation (`constant/<region>/`,
    # `system/<region>/`, asked per region by `read_heat`), never the case's own files: only
    # its log is asked here (T-FOLLOWUP-MINORS, T-ENERGY-BALANCE's minors).
    return _source_or_radiation(case, None, log_text, files=not multi)


def _regions_steady(case: Path, regions: list[str], solver: str | None,
                    checks: str) -> str | None:
    """Why a multi-region run is not judged steady, or None. v2406's chtMultiRegion solvers read
    each region's `system/<region>/fvSchemes` and never the case's own (`_foam_reader.
    _ddt_read_whole` refuses it for the same reason). Steady only when the solver that ran is
    steady by construction (`courant.case.solvers.is_steady` under `checks`) and every region's
    file, read whole (no directive or macro), declares `ddtSchemes { default steadyState; }`."""
    from courant.case.solvers import is_steady
    from courant.spec import deck

    if not is_steady(solver, checks):
        return (f"the solver that ran, {solver or 'not named'}, is not a steady one: the heat "
                f"stored inside the regions may change, and it is not read, so the heat balance "
                f"is not judged")
    for region in regions:
        rel = f"system/{region}/fvSchemes"
        try:
            text, why = deck._read_text(case, rel)
        except OSError:
            text, why = None, f"{rel} could not be read"
        if text is None:
            return (f"{_q(rel)}, the schemes the solver read for {_named(region)}, could not be "
                    f"read ({why.removeprefix(rel).strip()}), so whether the run is steady is "
                    f"not known and the heat balance is not judged")
        try:
            body = deck.block(text, "ddtSchemes") if deck._complete(text) else None
        except ValueError:                              # a quoted key not evaluated
            body = None
        if body is None and not (deck._complete(text) and "ddtSchemes" not in text):
            return (f"the ddtSchemes of {_q(rel)} is not read whole (a directive, a macro or a "
                    f"quoted pattern): whether {_named(region)} is steady is not known, and the "
                    f"heat balance is not judged")
        declared = (deck.entry(body or "", "default") or "").strip()
        if declared.lower() != "steadystate":
            return (f"{_q(rel)} declares ddtSchemes default {declared or 'nothing'}, not "
                    f"steadyState: the regions' schemes do not all say the run is steady, so the "
                    f"heat balance is not judged")
    return None


#: A solid transport v2406 solves with a tensor conductivity (`thermo.isotropic()` false,
#: solveSolid.H `taniAlpha()`): constAnIso and its kin, all named `...AnIso`.
_ANISOTROPIC = re.compile(r"aniso", re.IGNORECASE)


def _anisotropic(case: Path, region: str, strict: bool = True) -> str | None:
    """Why solid `region`'s heat flux rows cannot be taken for the heat it conducted, or None.
    v2406's wallHeatFlux writes a solid's `alpha()*snGrad(he)`, a scalar conductivity; a solid
    whose `thermoType` names an anisotropic transport is solved with a tensor one, so its rows
    are not the flux the solver conserved. A file not read whole cannot show it is isotropic, and
    is refused when `strict` (a region regionProperties calls solid); else only an anisotropic
    transport read whole is (a one-region case, whose thermo may be a fluid's)."""
    from courant.spec import deck

    rel = f"constant/{_region_dir(region)}thermophysicalProperties"
    tail = "so the heat balance is not judged"
    try:
        text, why = deck._read_text(case, rel)
        raw = None
        if text is not None and deck._complete(text):
            raw = deck.block(text, "thermoType") or deck.entry(text, "thermoType")
    except (OSError, ValueError):
        text, why, raw = None, f"{rel} could not be read", None
    if not strict and (raw is None or not _ANISOTROPIC.search(raw)):
        return None
    if text is None:
        return (f"{_q(rel)}, the thermophysical model of solid {_named(region)}, could not be "
                f"read ({why.removeprefix(rel).strip()}): whether its conductivity is a scalar, "
                f"the one v2406's wallHeatFlux reports, is not known, {tail}")
    if raw is None:
        return (f"{_q(rel)} names no thermoType the reader reads whole (a directive or a macro, "
                f"or none): whether the conductivity of solid {_named(region)} is a scalar, the "
                f"one v2406's wallHeatFlux reports, is not known, {tail}")
    if _ANISOTROPIC.search(raw):
        return (f"{_q(rel)} gives solid {_named(region)} an anisotropic transport: the solver "
                f"conducts its heat with a tensor conductivity and v2406's wallHeatFlux reports "
                f"a scalar one, so its rows are not the heat it conducted, {tail}")
    return None


def _source_or_radiation(case: Path, region: str | None, log_text: str | None, *,
                         files: bool = True) -> str | None:
    """A source or a radiation model where the solver looks for one, in the case (`region`
    None, and the solver's log) or in a region's own directories. `files=False`: the log alone,
    for a multi-region case, whose solver reads no case-level source or radiation file."""
    from courant.case.reynolds import _SOURCE_LINE, _Budget, _radiates, source_file

    if log_text and _SOURCE_LINE.search(log_text[:200_000]):
        return ("the solver's log says it built a source (an MRF zone, fvOptions, an SRF frame or "
                "a moving mesh), which may put heat into the domain that its walls do not carry, "
                "so the heat balance is not judged")
    found = source_file(case, (), region) if files else None
    if found is not None:
        return (f"{_q(found)} is where the solver looks for a source, which may put heat into the "
                f"domain that its walls do not carry, so the heat balance is not judged")
    if _radiates(case, (), log_text, _Budget(), region, files=files):
        where = f"constant/{_region_dir(region or DEFAULT_REGION)}radiationProperties"
        return (f"a radiation model other than none may heat the run ({_q(where)} or the "
                f"solver's log), and its heat is not in the walls' conduction alone, so the heat "
                f"balance is not judged")
    return None


def _start(log_text: str | None) -> float | None:
    """The time the solver read its mesh at, from its log (`courant.spec.as_run.Run.start_name`,
    the one reading of it), or None when the log does not say."""
    if not log_text:
        return None
    from courant.spec.as_run import Run

    name = Run(log_text, "", (), exit_code=None).start_name
    try:
        return float(name) if name is not None else None
    except ValueError:
        return None


def _unsearched(named: str, error: str | None, tail: str) -> str:
    return (f"which mesh the solver read for {named} is not known (a directory of it could not "
            f"be searched: {error or 'an error'}), {tail}")


def _boundary(case: Path, region: str, log_text: str | None,
              start: float | None) -> tuple[tuple | None, list[str], str | None]:
    """(the patches of the mesh the solver read for `region`, every rank's together; the boundary
    files read, relative to the case; None) or (None, those files, why it is not judged). The
    mesh is `courant.case.mesh.solver_mesh`'s, the one rule; a patch is a rank's sum of faces."""
    from courant.case.mesh import solver_mesh
    from courant.case.polymesh import read_boundary

    named = _named(region)
    tail = f"so the heat balance of {named} and of its interfaces is not judged"
    try:
        found = solver_mesh(case, log_text, start=start,
                            region=None if region == DEFAULT_REGION else region)
    except OSError as exc:                              # a directory listed and not searched
        return None, [], _unsearched(named, exc.strerror, tail)
    if found.unsearchable:      # the same, said by `solver_mesh` since T-ZERO-UNLISTABLE
        return None, [], _unsearched(named, found.unsearchable, tail)
    if found.why:
        return None, [], (f"which mesh the solver read for {named} ({_q(found.where)}) is not "
                          f"known ({found.why}), {tail}")
    if not found.dirs:
        return None, [], f"{named} has no mesh ({_q(found.where)}), {tail}"
    merged: dict[str, object] = {}
    rels: list[str] = []
    for d in found.dirs:
        path = Path(plain_or_gz(str(d / "boundary")))      # the field reader's rule
        rel = path.relative_to(case).as_posix()
        rels.append(rel)
        try:
            patches = read_boundary(path)
        except Exception:                               # noqa: BLE001 - said, never raised
            return None, rels, f"{_q(rel)} could not be read, {tail}"
        for p in patches:
            seen = merged.get(p.name)
            if seen is None:
                merged[p.name] = p
            elif seen.type != p.type:
                return None, rels, (f"patch {_q(p.name)} is of type {seen.type} on one rank and "
                                    f"{p.type} on another ({_listed(_q(r) for r in rels)}), "
                                    f"{tail}")
            else:
                merged[p.name] = replace(seen, n=seen.n + p.n)
    return tuple(merged.values()), rels, None


def _judge(case: Path, tables, regions: list[str], multi: bool, last: float,
           log_text: str | None = None) -> HeatReading:
    from courant.verification._fluid_reader import _COUPLED
    from courant.verification.checks import MassFluxCheck

    said: list[str] = []
    at = f"{last:g}"
    # ---- the rows at the last time, every table of a region merged
    found: dict[str, dict[str, list[tuple[float, float, float, str]]]] = {}
    files: dict[str, list[str]] = {}
    broken: dict[str, set[str]] = {}            # region: patches whose rows cannot be taken
    blind: set[str] = set()                     # regions with a table or a row not read
    from courant.converge.series import MAX_BYTES

    budget = [MAX_BYTES]                        # one bound for every table of the reading
    for region_dir, rel in tables:
        region = region_dir if multi else (DEFAULT_REGION if region_dir is None else None)
        if region is None or region not in regions:
            said.append(_say(f"{_q(rel)} is not a table of a region the case has, so it was not "
                             f"read"))
            continue
        try:
            is_table, rows, bad = _read_table(case, rel, last, budget)
        except _TooLarge:
            said.append(_say(f"the heat tables together are past the {MAX_BYTES // 2**20} MiB "
                             f"one reading reads, so {_q(rel)} was not read whole and the heat "
                             f"balance of {_named(region)} is not judged"))
            blind.add(region)
            files.setdefault(region, []).append(rel)
            budget[0] = 0
            continue
        except Exception:                               # noqa: BLE001 - said, never raised
            said.append(_say(f"{_q(rel)} could not be read, so the heat balance of "
                             f"{_named(region)} is not judged"))
            blind.add(region)
            files.setdefault(region, []).append(rel)
            continue
        if not is_table:
            said.append(_say(f"{_q(rel)} is not in the columns v2406's wallHeatFlux writes "
                             f"({' '.join(COLUMNS)}), so it was not read"))
            continue
        files.setdefault(region, []).append(rel)
        for patch, values in rows.items():
            for v, lo, hi in values:
                found.setdefault(region, {}).setdefault(patch, []).append((v, lo, hi, rel))
        for patch in bad:
            if patch is None:
                blind.add(region)
                said.append(_say(f"a row of {_q(rel)} does not parse (neither its time nor its "
                                 f"patch can be read), so the heat balance of {_named(region)} "
                                 f"is not judged"))
            else:
                broken.setdefault(region, set()).add(patch)
                said.append(_say(f"the row of wall patch {_q(patch)} at time {at} in {_q(rel)} "
                                 f"does not parse, so the heat balance of {_named(region)} is "
                                 f"not judged"))
    rows: dict[str, dict[str, float]] = {}
    spans: dict[str, dict[str, tuple[float, float]]] = {}    # each patch's (min, max), W/m^2
    for region, patches in found.items():
        for patch, seen in patches.items():
            distinct = sorted({repr(v) for v, *_ in seen})
            if len(distinct) > 1:
                broken.setdefault(region, set()).add(patch)
                said.append(_say(
                    f"two rows of wall patch {_q(patch)} at time {at} give different integrals "
                    f"({_listed(distinct)}, in {_listed(_q(r) for r in sorted({s[3] for s in seen}))}"
                    f"), so the heat balance of {_named(region)} is not judged"))
            elif patch not in broken.get(region, ()):
                rows.setdefault(region, {})[patch] = seen[0][0]
                spans.setdefault(region, {})[patch] = (min(s[1] for s in seen),
                                                       max(s[2] for s in seen))
    # ---- each region's patches: the boundary of the mesh the solver read
    start = _start(log_text)
    boundaries: dict[str, tuple] = {}
    boundary_files: dict[str, list[str]] = {}
    for region in regions:
        if region not in files:
            if multi:
                said.append(_say(f"{_named(region)} wrote no wallHeatFlux table, so its heat "
                                 f"balance is not judged"))
            continue
        patches, rels, why = _boundary(case, region, log_text, start)
        if why is not None:
            said.append(_say(why))
            continue
        boundaries[region], boundary_files[region] = patches, rels
    # ---- the scales round-off is judged against: every row the case wrote at the last time,
    # every region's, so never only the sums a balance is made of (`MassFluxCheck.round_off`).
    # A patch with no faces is left out: v2406 still writes its row, and gMin and gMax of an
    # empty field are pTraits<scalar>::max and -max (`1e+300 -1e+300 0`), min above max.
    empty = {(r, p.name) for r, ps in boundaries.items() for p in ps if not p.n}
    every = [s for region, patches in found.items() for patch, seen in patches.items()
             if (region, patch) not in empty for s in seen if not s[1] > s[2]]
    largest = max((abs(s[0]) for s in every if math.isfinite(s[0])), default=0.0)
    face = max((abs(x) for s in every for x in s[1:3] if math.isfinite(x)), default=0.0)
    balances: list[Balance] = []
    walls_of: dict[str, set[str]] = {}
    for region in regions:
        if region not in boundaries:
            continue
        patches = boundaries[region]
        walls_of[region] = {p.name for p in patches if p.type in WALL_TYPES}
        if region in blind:
            continue
        # A row of a patch the boundary read has no wall of: not the mesh the run wrote it on.
        strangers = sorted(set(found.get(region, {})) - walls_of[region])
        if strangers:
            blind.add(region)
            many = len(strangers) > 1
            said.append(_say(
                f"{_listed(_q(n) for n in strangers)} {'have' if many else 'has'} rows in "
                f"{_listed(_q(f) for f in files[region])} and {'are' if many else 'is'} not a "
                f"wall patch of {_listed(_q(f) for f in boundary_files[region])}, the mesh read "
                f"for {_named(region)}: the run wrote its heat flux on another boundary, so the "
                f"heat balance of {_named(region)} and of its interfaces is not judged"))
            continue
        opening = [p.name for p in patches if p.n and p.type not in WALL_TYPES
                   and p.type not in _CLOSED_TYPES and not any(c in p.type for c in _COUPLED)]
        if opening:
            said.append(_say(
                f"{_named(region)} is open to flow through {_listed(_q(n) for n in opening)}: "
                f"the heat the flow carries through {'it' if len(opening) == 1 else 'them'} is "
                f"not read, so its heat balance is not judged (only the heat through its walls "
                f"was read)"))
            continue
        walls = [p.name for p in patches if p.type in WALL_TYPES]
        have = rows.get(region, {})
        missing = [w for w in walls if w not in have and w not in broken.get(region, ())]
        if missing:
            said.append(_say(
                f"wall patch{'es' if len(missing) > 1 else ''} "
                f"{_listed(_q(n) for n in missing)} of {_named(region)} "
                f"{'have' if len(missing) > 1 else 'has'} no row at time {at} in "
                f"{_listed(_q(f) for f in files[region])}, so its heat balance is not judged"))
            continue
        if broken.get(region):
            continue                                    # said where the row was read
        if not walls:
            continue
        b = _region_balance(region, [(w, have[w]) for w in walls], last, files[region], largest)
        if b is None:
            said.append(_say(f"no heat crosses the walls of {_named(region)} at time {at} (its "
                             f"heat in and its heat out are round-off next to the largest "
                             f"integral the case wrote at that time), so there is no heat "
                             f"balance to judge"))
            continue
        why = _both_ways(b, [(w, spans[region][w]) for w in walls], face, at)
        if why is not None:
            said.append(_say(why))
        else:
            balances.append(b)
    # ---- the interfaces
    done: set[frozenset] = set()
    unresolved: list[tuple[str, object]] = []
    for region in regions:
        if region not in boundaries:
            continue
        for p in boundaries[region]:
            if p.type not in WALL_TYPES or not p.type.startswith("mapped"):
                continue
            if p.type != INTERFACE_TYPE or not p.sample_patch:
                if p.n:
                    unresolved.append((region, p))          # said below, unless its other side named it
                continue
            other = p.sample_region or region
            pair = frozenset({(region, p.name), (other, p.sample_patch)})
            if pair in done:
                continue
            done.add(pair)
            where = (f"the interface {_q(p.name)} ({_named(region)}) and {_q(p.sample_patch)} "
                     f"({_named(other)})")
            if other not in regions:
                said.append(_say(f"{where} is not judged: the case lists no region {_q(other)}"))
                continue
            if other not in files or other not in boundaries:
                continue                                # said above, of the region
            if p.sample_patch not in walls_of.get(other, ()):
                said.append(_say(f"{where} is not judged: {_q(p.sample_patch)} is not a wall "
                                 f"patch of {_named(other)}"))
                continue
            sides = [(region, p.name), (other, p.sample_patch)]
            if any(r in blind or n in broken.get(r, ()) for r, n in sides):
                continue                                # said where the row was read
            absent = [n for r, n in sides if n not in rows.get(r, {})]
            if absent:
                said.append(_say(f"{where} is not judged: {_listed(_q(n) for n in absent)} "
                                 f"{'have' if len(absent) > 1 else 'has'} no row at time {at}"))
                continue
            qa, qb = rows[region][p.name], rows[other][p.sample_patch]
            larger = max(abs(qa), abs(qb))
            # VER-22 (b): the boundary balance's round-off rule, the larger side as the heat in
            # (`MassFluxCheck.denominator`), against the case's largest integral.
            if math.isfinite(qa + qb) and MassFluxCheck.denominator(larger, 0.0, largest) is None:
                said.append(_say(f"no heat crosses {where} at time {at} (both sides are "
                                 f"round-off next to the largest integral the case wrote at that "
                                 f"time), so there is no heat balance to judge"))
                continue
            measure = abs(qa + qb) / larger if math.isfinite(qa + qb) else math.nan
            key = tuple(n for _, n in sorted(sides))
            b = Balance(
                key=key, measure=measure, where=where, time=last,
                sums=(f"{_q(p.name)} carries {qa:.4g} W and {_q(p.sample_patch)} {qb:.4g} W, "
                      f"positive into each side"),
                scale_name="the larger side",
                files=tuple(dict.fromkeys(files[region] + files[other])))
            why = _both_ways(b, [(n, spans[r][n]) for r, n in sides], face, at)
            if why is not None:
                said.append(_say(why))
            else:
                balances.append(b)
    paired = {side for pair in done for side in pair}
    for region, p in unresolved:
        if (region, p.name) in paired:
            continue
        if p.type == INTERFACE_TYPE:
            said.append(_say(
                f"the interface {_q(p.name)} ({_named(region)}) is not judged: its boundary entry "
                f"does not name its other side by sampleRegion and samplePatch (a coupleGroup, "
                f"which is not resolved here)"))
        else:
            said.append(_say(
                f"the interface {_q(p.name)} ({_named(region)}) is not judged: it is a {p.type} "
                f"patch (a baffle with a thickness), which is not resolved here"))
    return HeatReading(rows=rows, files={r: f[0] for r, f in files.items()},
                       balances=tuple(balances), sentences=tuple(dict.fromkeys(said)))


def _region_balance(region: str, values: list[tuple[str, float]], last: float,
                    files: list[str], largest: float) -> Balance | None:
    """VER-22 (a): |net| over the heat in, or over the heat out, the denominator chosen by the
    boundary balance's own rule (`MassFluxCheck.denominator`), its scale `largest`, the largest
    |integral| in every row the case wrote at `last` (every region's, so not one the region's
    own sums set); None for walls no heat crosses. A non-finite integral gives a non-finite
    measure (MAJOR in the check)."""
    from courant.verification.checks import MassFluxCheck

    ints = [v for _, v in values]
    where = _named(region)
    if not all(math.isfinite(v) for v in ints):
        return Balance(key=region, measure=math.nan, where=where, time=last, sums="",
                       scale_name="the heat in", files=tuple(files))
    net = sum(ints)
    heat_in = sum(v for v in ints if v > 0)
    heat_out = -sum(v for v in ints if v < 0)
    over = MassFluxCheck.denominator(heat_in, heat_out, largest)
    if over is None:
        return None
    scale, through = over[0], f"the heat {over[1]}"
    return Balance(key=region, measure=abs(net) / scale, where=f"the walls of {where}",
                   time=last,
                   sums=(f"net wall heat flux {net:.4g} W over its {len(values)} wall "
                         f"patches against {through} of {scale:.4g} W"),
                   scale_name=through, files=tuple(files))


def _both_ways(b: Balance, spans: list[tuple[str, tuple[float, float]]], face: float,
               at: str) -> str | None:
    """Why balance `b` is not judged, or None (VER-22). A patch whose row's `min` is below zero
    and `max` above, neither round-off next to `face` (the largest face heat flux, |min| or
    |max|, in every row the case wrote at the time: `MassFluxCheck.round_off`), carries heat both
    into and out of its side across its faces; a non-finite `min` or `max` cannot show it does
    not. Its integral is then a net, so the heat in (or the larger side) read from the integrals
    is less than the heat that crosses, and the measure more than the balance's: a measure graded
    OK stands, any other cannot be told from these rows and is not judged. A non-finite measure
    always stands (MAJOR in the check)."""
    from courant.verification.checks import MassFluxCheck
    from courant.verification.grade import VSeverity

    if not math.isfinite(b.measure) or MassFluxCheck.grade(b.measure)[0] is VSeverity.OK:
        return None
    both = [(n, lo, hi) for n, (lo, hi) in spans
            if not (math.isfinite(lo) and math.isfinite(hi))
            or (lo < 0 < hi and not MassFluxCheck.round_off(lo, face)
                and not MassFluxCheck.round_off(hi, face))]
    if not both:
        return None
    many = len(both) > 1
    ranges = "; ".join(f"{_q(n)} from {lo:.4g} to {hi:.4g} W/m^2" for n, lo, hi in both)
    return (f"wall patch{'es' if many else ''} {_listed(_q(n) for n, _, _ in both)} "
            f"{'carry' if many else 'carries'} heat both in and out across "
            f"{'their' if many else 'its'} faces at time {at} ({ranges}), so "
            f"{'their integrals are nets' if many else 'its integral is a net'} and "
            f"{b.scale_name} read from the integrals is less than the heat that crosses: the heat "
            f"balance of {b.where} as read, {b.measure:.2e} of {b.scale_name} "
            f"({MassFluxCheck.against(b.measure, MassFluxCheck.TARGET)}), is one a balance within "
            f"the bar can also show, so it is not judged")


__all__ = ["COLUMNS", "DEFAULT_REGION", "WALL_TYPES", "Balance", "HeatReading", "cited",
           "read_heat"]
