"""MeshQuality — structured checkMesh metrics for diagnosis (Tech Arch §5.1).

The mesh checker "wraps checkMesh; extracts skewness, non-orthogonality, aspect
ratio, negative volumes into structured Findings." This model is that structured
extraction. Constructors: ``from_dict`` (replay case-facts) and ``from_checkmesh``
(parse real checkMesh stdout, live path).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class MeshQuality(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_cells: int | None = None
    max_non_orthogonality: float | None = None       # degrees
    average_non_orthogonality: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None
    n_negative_volume_cells: int = 0
    mesh_ok: bool | None = None                      # checkMesh "Mesh OK." verdict
    #: None: checkMesh's own output. `POLYMESH_SOURCE`: computed from `constant/polyMesh` by
    #: `polymesh_quality`, which gives `n_cells` and `max_aspect_ratio` only (and the extent
    #: below).
    source: str | None = None
    #: The mesh's extent, for the buoyant regime check (TRD VER-17): checkMesh's "Overall domain
    #: bounding box" (min, max), its "Total volume", and its solution directions (1 where
    #: solved, 0 where an `empty` patch removes the direction), or the same from the mesh files.
    bounding_box: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    total_volume: float | None = None
    solution_directions: tuple[int, int, int] | None = None
    #: checkMesh's face-area check failed ("Zero or negative face area detected"), and the
    #: number of faces it wrote to its `zeroAreaFaces` set when it printed one (TRD VER-7d).
    zero_area_faces: bool = False
    n_zero_area_faces: int | None = None
    #: checkMesh's count of faces above its skewness threshold, 4 (0 when it printed "OK").
    n_skew_faces: int | None = None
    #: TRD VER-7d: the largest skewness of the faces that have an area, computed from
    #: `constant/polyMesh` as v2406 computes it, when checkMesh's maximum is a zero-area face's
    #: (`skewness_of_faces_with_area`); and the `empty` patches those zero-area faces lie on.
    #: None: checkMesh's maximum stands (every mesh with no zero-area face). Computed from the
    #: case's files in the run that reads them, never taken from a view (`from_view`).
    skewness_with_area: float | None = None
    zero_area_patches: tuple[str, ...] = ()

    def zero_area_set_aside(self) -> bool:
        """Whether VER-7d's reading is whole: the faces with an area judged, and every number its
        "not assessed" sentence names present (checkMesh's maximum and count of zero-area faces,
        the patches they lie on). `skewness` and `sentences.skewness_zero_area` both ask this,
        so the value judged is checkMesh's maximum wherever the verdict does not say why not."""
        return (self.skewness_with_area is not None and bool(self.zero_area_patches)
                and self.n_zero_area_faces is not None and self.max_skewness is not None)

    def skewness(self) -> float | None:
        """The skewness `mesh.skewness` judges: checkMesh's maximum, or, where that maximum is a
        zero-area face's quotient by ROOTVSMALL on an `empty` patch, the largest on the faces
        with an area (VER-7d, `zero_area_set_aside`)."""
        return self.skewness_with_area if self.zero_area_set_aside() else self.max_skewness

    def stated(self) -> "MeshQuality":
        """This reading without what VER-7d computes from the mesh files: what a view can state
        (checkMesh's own numbers), so checkMesh's maximum is judged as printed."""
        if self.skewness_with_area is None and not self.zero_area_patches:
            return self
        return self.model_copy(update={"skewness_with_area": None, "zero_area_patches": ()})

    @classmethod
    def from_view(cls, view: dict | None, *, computed_here: bool = False) -> "MeshQuality | None":
        """A captured mesh view as a reading, or None when there is none or it does not validate.

        `computed_here`: the view is this run's own reading of the case directory
        (`ingest.from_case_dir`, `preflight_path` on a directory), which computed VER-7d's fields
        from `constant/polyMesh`. Any other view (a corpus sidecar, an MCP `case_json`, a stored
        record) cannot show what the mesh files hold, so those fields are dropped (`stated`) and
        checkMesh's maximum is judged, as 0.5.0 judged it: 0.5.0 had no such fields and ignored
        them. A sidecar holding part of them ended `courant verify` in a traceback (the review of
        T-WEDGE-MESH)."""
        if not view:
            return None
        try:
            mesh = cls.model_validate(view)
        except Exception:  # noqa: BLE001 — not a mesh view: no mesh, as with none
            return None
        return mesh if computed_here else mesh.stated()

    @classmethod
    def from_dict(cls, data: dict) -> "MeshQuality":
        return cls.model_validate(data)

    @classmethod
    def from_checkmesh(cls, text: str) -> "MeshQuality":
        """Parse the metrics out of checkMesh stdout."""

        # The first group that actually captured, not group(1).
        #
        # With an alternation, whichever branch matches at the *earliest*
        # position wins — and for `aspect ratio…|Max aspect ratio…` against real
        # output, that is the second branch, because "Max" starts before
        # "aspect". So group(1) was None and `float(None)` raised TypeError on
        # every real checkMesh log. `read_mesh_quality` catches and returns None,
        # so the symptom was silent: mesh metrics simply never arrived, on the
        # agent path as well as the live one, and all five MESH.* codes were
        # undiagnosable even when a checkMesh log was sitting right there.
        def _first(match: "re.Match | None") -> str | None:
            if match is None:
                return None
            return next((g for g in match.groups() if g is not None), None)

        def _f(pattern: str) -> float | None:
            captured = _first(re.search(pattern, text))
            return float(captured) if captured is not None else None

        def _i(pattern: str) -> int | None:
            captured = _first(re.search(pattern, text))
            return int(captured) if captured is not None else None

        non_ortho_max = _f(r"non-orthogonality Max:\s*([\d.eE+-]+)")
        non_ortho_avg = _f(r"non-orthogonality Max:\s*[\d.eE+-]+\s*average:\s*([\d.eE+-]+)")
        # `[=:]` — and this is the second time this line has been wrong.
        #
        # checkMesh prints the metric one way when the mesh is fine and another
        # way when it is not:
        #
        #     Max aspect ratio = 1 OK.
        #     ***High aspect ratio cells found, Max aspect ratio: 2000, ...
        #
        # The failing form uses a **colon**. With `=?` the value was readable
        # only on a healthy mesh and unreadable on exactly the meshes worth
        # reporting — so `MESH.ASPECT_RATIO` could never be diagnosed, and the
        # scenario for it came back with an empty diagnosis while `checkMesh`
        # was printing the number in as many words.
        #
        # A perfect inversion, and the same shape as the `CONV.MASS_IMBALANCE`
        # log pattern that fired on healthy runs and missed broken ones. The
        # lesson both times: the failing output is a *different string* from the
        # passing one, and a pattern derived from a healthy run has never seen
        # the case it exists for.
        skew = _f(r"[Mm]ax skewness\s*[=:]?\s*([\d.eE+-]+)")
        aspect = _f(r"Max aspect ratio\s*[=:]?\s*([\d.eE+-]+)|"
                    r"aspect ratio\s*[=:]?\s*([\d.eE+-]+)")
        n_cells = _i(r"cells:\s*(\d+)")
        neg = negative_volume_cells(text)
        skew_faces = _SKEW_FACES.search(text)
        zero_set = _ZERO_AREA_SET.search(text)
        mesh_ok = "Mesh OK." in text or "Mesh OK" in text
        num = r"([-+\d.eE]+)"
        box = re.search(r"Overall domain bounding box \(" + " ".join([num] * 3) + r"\) \("
                        + " ".join([num] * 3) + r"\)", text)
        bounding_box = None
        if box:
            v = [float(x) for x in box.groups()]
            bounding_box = (tuple(v[:3]), tuple(v[3:]))
        directions = re.search(
            r"Mesh has \d solution \(non-empty\) directions \(([01]) ([01]) ([01])\)", text)
        volume = re.search(r"Total volume = ([-+\d.eE]+?)\.?\s", text)

        return cls(
            n_cells=n_cells,
            max_non_orthogonality=non_ortho_max,
            average_non_orthogonality=non_ortho_avg,
            max_skewness=skew,
            max_aspect_ratio=aspect,
            n_negative_volume_cells=neg,
            mesh_ok=mesh_ok,
            bounding_box=bounding_box,
            total_volume=float(volume.group(1)) if volume else None,
            solution_directions=(tuple(int(x) for x in directions.groups())
                                 if directions else None),
            zero_area_faces=_ZERO_AREA in text,
            n_zero_area_faces=int(zero_set.group(1)) if zero_set else None,
            n_skew_faces=(int(skew_faces.group(1)) if skew_faces.group(1) else 0)
            if skew_faces else None,
        )


# ---------------------------------------------------------------- v2406's own words (TRD VER-7d)
#: A cell of zero or negative volume, in the only words v2406's checkMesh says it:
#: `primitiveMesh::checkCellVolumes` (src/OpenFOAM/meshes/primitiveMesh/primitiveMeshCheck/
#: primitiveMeshCheck.C) prints " ***Zero or negative cell volume detected.  Minimum negative
#: volume: <v>, Number of negative volume cells: <n>" when a cell's volume is below VSMALL, and
#: `checkGeometry` (applications/utilities/mesh/manipulation/checkMesh/checkGeometry.C) then
#: "  <<Writing <n> zero volume cells to set zeroVolumeCells". v0.5.0 counted one negative cell
#: wherever "negative volume" appeared, which includes checkGeometry's line on the tetrahedra a
#: face decomposes into (" ***Error in face tets: <n> faces with low quality or negative volume
#: decomposition tets.", polyMeshTetDecomposition::checkFaceTets under -allGeometry): a quality
#: measure of the decomposition, not a cell's volume. A healthy axisymmetric pipe (a rebuilt run
#: of an internal evaluation, 22 Oct 2026) was rejected on it as a CRITICAL "1 cell(s) with
#: negative volume" next to checkMesh's own "Cell volumes OK."
_NEGATIVE_CELLS = re.compile(r"Zero or negative cell volume detected\.[^\n]*")
_NEGATIVE_COUNT = re.compile(r"Number of negative volume cells:\s*(\d+)")
_ZERO_VOLUME_SET = re.compile(r"<<Writing\s+(\d+)\s+zero volume cells")
#: `primitiveMesh::checkFaceAreas` (primitiveMeshCheck.C): " ***Zero or negative face area
#: detected.  Minimum area: <a>" when a face's |area| is below VSMALL, and checkGeometry's
#: "  <<Writing <n> zero area faces to set zeroAreaFaces".
_ZERO_AREA = "Zero or negative face area detected"
_ZERO_AREA_SET = re.compile(r"<<Writing\s+(\d+)\s+zero area faces")
#: `polyMesh::checkFaceSkewness` (src/OpenFOAM/meshes/polyMesh/polyMeshCheck/polyMeshCheck.C):
#: " ***Max skewness = <s>, <n> highly skew faces detected ..." or "    Max skewness = <s> OK.".
_SKEW_FACES = re.compile(r"Max skewness = [-+\d.eE]+(?:, (\d+) highly skew faces| OK)")
#: `primitiveMesh::skewThreshold_` (primitiveMeshCheck.C), the bar `mesh.skewness` uses too.
SKEW_THRESHOLD = 4.0


def negative_volume_cells(text: str) -> int:
    """How many cells checkMesh reports with a zero or negative volume: the count on v2406's
    own line, else the count it wrote to its set, else one if the line is there uncounted."""
    line = _NEGATIVE_CELLS.search(text)
    if line is None:
        return 0
    count = _NEGATIVE_COUNT.search(line.group(0)) or _ZERO_VOLUME_SET.search(text)
    return max(1, int(count.group(1))) if count else 1


#: Where a checkMesh log conventionally lands. `log.checkMesh` is what
#: OpenFOAM's own `runApplication` wrapper writes and what `DockerRunner` writes
#: for a pre-command; `checkMesh.log` is the other common hand-rolled spelling.
CHECKMESH_LOG_NAMES = ("log.checkMesh", "checkMesh.log")


#: `MeshQuality.source` of metrics computed from the mesh files rather than read from checkMesh.
POLYMESH_SOURCE = "constant/polyMesh"
#: Above these the mesh is not read for metrics, so the gate stays quick and small in memory on a
#: user's mesh: cost bounds, not physics thresholds. Measured on the founder's laptop (13 Oct
#: 2026): 500,750 faces (25 MB of files) took 1.2 s and 363 MB peak; 1,997,499 took 5.3 s and
#: 2.4 GB, so the bound is 500,000. The bytes are checked on disk and the faces against both the
#: owner header's note and the faces list's own count, before anything is parsed (a note is
#: text anyone can write). Such a mesh is reported exactly as a mesh with no checkMesh log is.
POLYMESH_MAX_FACES = 500_000
POLYMESH_MAX_BYTES = 32 * 2**20

_NPROCS = re.compile(r"^nProcs[ \t]*:[ \t]*(\d{1,9})", re.M)
#: A collated decomposition (`-fileHandler collated`): one `processors<N>/` (or
#: `processors<N>_<first>-<last>/`) holding every rank, which the reader does not read.
_COLLATED = re.compile(r"processors\d+(_\d+-\d+)?")
#: An uncollated rank directory, as decomposePar writes it.
_RANK = re.compile(r"processor(\d{1,9})")


def ran_parallel(log_text: str | None) -> bool | None:
    """Whether the run was parallel, on either of the log's two signals: a banner's `nProcs`
    above 1 (in the first 20,000 characters, where a banner sits, or in the solver's own banner),
    or `-parallel` on the solver's Exec line (`solvers.solver_exec`, the one Exec reading). False
    when the log names its solver and neither signal is there; None when the log does not say
    (no log, or no Exec line naming a solver and no `nProcs`)."""
    if not log_text:
        return None
    from courant.case.solvers import solver_exec

    ex = solver_exec(log_text)
    m = _NPROCS.search(log_text[:20000])
    if (ex is not None and ex.parallel) or (m is not None and int(m.group(1)) > 1):
        return True
    return False if ex is not None else None


@dataclass(frozen=True)
class SolverMesh:
    """The mesh the solver read: its polyMesh directories (one, or one per rank of a parallel
    run, in rank order), or why they cannot be named. Neither (`dirs` empty, `why` None): the
    case holds no mesh files at all."""

    dirs: tuple[Path, ...] = ()
    why: str | None = None
    #: How a verdict names it: "constant/polyMesh", "0.1/polyMesh", "processor*/constant/polyMesh".
    where: str = "constant/polyMesh"
    #: The case holds rank directories with a mesh, or a collated decomposition, whether or not
    #: the solver read them.
    decomposed: bool = False


def time_dirs(root: Path) -> list[str]:
    """The finite numeric time directory names directly under `root`, ascending."""
    out = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for d in entries:
        try:
            t = float(d.name)
        except ValueError:
            continue
        if d.is_dir() and math.isfinite(t):
            out.append(d.name)
    return sorted(out, key=float)


def _instance(root: Path, start: float | None) -> tuple[Path | None, str | None]:
    """(the polyMesh directory under `root` a run starting at `start` read, None) or (None, why
    not). As OpenFOAM's `findInstance` looks it up: the latest time directory at or before the
    start that holds a `polyMesh/`, else `constant/polyMesh`. A time directory whose `polyMesh/`
    holds no `owner` (a moving mesh writes `points` alone) splits the mesh over two instances,
    which the reader does not assemble; with the start not known, any time `polyMesh/` leaves the
    instance not known. (None, None): no mesh under `root`."""
    timed = [n for n in time_dirs(root) if (root / n / "polyMesh").is_dir()]
    if timed and start is None:
        return None, f"time {timed[0]} carries its own polyMesh, and the run's start is not known"
    before = [n for n in timed if float(n) <= start + 1e-12] if timed else []
    if before:
        d = root / before[-1] / "polyMesh"
        if _has_owner(d):
            return d, None
        return None, (f"{d.relative_to(root.parent if root.name.startswith('processor') else root)}"
                      " holds no owner list (a moved mesh's points alone), which the reader does "
                      "not assemble with constant/polyMesh")
    d = root / "constant" / "polyMesh"
    return (d, None) if _has_owner(d) else (None, None)


def _has_owner(d: Path) -> bool:
    """Whether `d` holds a polyMesh, written plain or compressed (`owner.gz`)."""
    return (d / "owner").is_file() or (d / "owner.gz").is_file()


def rank_dirs(case: Path) -> list[Path]:
    """The case's uncollated rank directories (`processor0`, `processor1`, ...), in rank order."""
    try:
        ranks = [d for d in case.iterdir() if d.is_dir() and _RANK.fullmatch(d.name)]
    except OSError:
        return []
    return sorted(ranks, key=lambda d: int(d.name[len("processor"):]))


def solver_mesh(case_dir: str | Path, log_text: str | None = None, *,
                start: float | None = None) -> SolverMesh:
    """The mesh the solver read, or why it cannot be named: the one rule, for the mesh metrics
    (`solver_polymesh`, VER-7a) and for conformance's mesh items (`courant.spec.as_run`,
    SPEC-13). `log_text` is the run's log, `start` the time it started from (None: not known).

    - A multi-region case (`constant/regionProperties`) reads `constant/<region>/polyMesh`, one
      per region: not named.
    - A parallel run (`ran_parallel`, either signal) read its ranks: `processor<N>/`, each at its
      instance (`_instance`), all of them present and, when its banner gives `nProcs`, as many.
      A collated decomposition (`processors<N>/`), no rank directories, a rank without a mesh,
      or a parallel signal while the solver's own Exec line says serial (a log of several
      applications): not named.
    - A serial run read the case's own mesh at its instance; rank directories left beside it are
      not what it read.
    - With no word of either (no log, or a log naming no solver): the case's own mesh, when
      there is no log at all and it has one (`courant spec check`, the drafter); else the ranks,
      when the case holds those alone; else, decomposed beside its own mesh, not named.
    """
    from courant.case.solvers import solver_exec

    case = Path(case_dir)
    ranks = [r for r in rank_dirs(case) if (r / "constant" / "polyMesh").is_dir()
             or any((r / n / "polyMesh").is_dir() for n in time_dirs(r))]
    try:
        collated = sorted(d.name for d in case.iterdir()
                          if d.is_dir() and _COLLATED.fullmatch(d.name))
    except OSError:
        collated = []
    decomposed = bool(collated) or any((r / "constant" / "polyMesh").is_dir()
                                       for r in rank_dirs(case))
    if (case / "constant" / "regionProperties").is_file():
        return SolverMesh(why="a multi-region case (constant/regionProperties): each region reads "
                              "its own constant/<region>/polyMesh, which the reader does not read",
                          decomposed=decomposed)
    parallel = ran_parallel(log_text)
    ex = solver_exec(log_text) if log_text else None

    def own() -> SolverMesh:
        d, why = _instance(case, start)
        where = d.relative_to(case).as_posix() if d is not None else "constant/polyMesh"
        return SolverMesh(dirs=(d,) if d is not None else (), why=why, where=where,
                          decomposed=decomposed)

    def by_rank() -> SolverMesh:
        where = "processor*/constant/polyMesh"
        if collated:
            return SolverMesh(why=f"the case is decomposed collated ({collated[0]}/, "
                                  "-fileHandler collated), which the reader does not read",
                              where=f"{collated[0]}/", decomposed=True)
        if not ranks:
            return SolverMesh(why="the run was parallel, and the case holds no processor "
                                  "directory with a mesh", where=where, decomposed=decomposed)
        numbers = [int(r.name[len("processor"):]) for r in ranks]
        n = ex.nprocs if ex is not None and ex.nprocs else None
        if numbers != list(range(len(ranks))) or (n is not None and n != len(ranks)):
            said = f"ran on {n} ranks" if n else "ran in parallel"
            return SolverMesh(why=f"the run {said}, and the case holds rank directories "
                                  f"{', '.join(r.name for r in ranks)}", where=where,
                              decomposed=True)
        dirs = []
        for r in ranks:
            d, why = _instance(r, start)
            if d is None:
                return SolverMesh(why=f"{r.name}: {why or 'no mesh'}", where=where,
                                  decomposed=True)
            dirs.append(d)
        rel = {d.relative_to(r).as_posix() for d, r in zip(dirs, ranks)}
        return SolverMesh(dirs=tuple(dirs), where="processor*/" + (rel.pop() if len(rel) == 1
                                                                   else "<time>/polyMesh"),
                          decomposed=True)

    if parallel:
        if ex is not None and not ex.parallel:
            return SolverMesh(why=f"the log shows a parallel run (nProcs above 1), and its solver's "
                                  f"Exec line ({ex.program}) does not: which mesh the solver read "
                                  "is not known", decomposed=decomposed)
        return by_rank()
    if parallel is False:
        return own()
    mine = own()
    if not decomposed or (log_text is None and (mine.dirs or mine.why)):
        return mine
    if not (mine.dirs or mine.why):
        return by_rank()
    return SolverMesh(why="the case is decomposed beside its own mesh, and the log does not say "
                          "whether the solver read its ranks", decomposed=True)


def _not_what_the_solver_read(case: Path, log_text: str | None) -> str | None:
    """Why `constant/polyMesh` may not be the mesh the metrics should read, or None: the mesh the
    solver read is not that one (`solver_mesh`, the one rule: parallel on either signal,
    multi-region, a time directory carrying its own polyMesh, the start not being known here),
    or the case is decomposed at all. The metrics keep v0.5.0's narrower reading (GATE-13): any
    decomposition leaves them unread, even beside a serial run's own mesh."""
    found = solver_mesh(case, log_text)
    if found.why:
        return found.why
    if found.decomposed:
        return "the case is decomposed"
    if found.dirs != (case / "constant" / "polyMesh",):
        return "no constant/polyMesh"
    return None


def solver_polymesh(case_dir: str, *, log_text: str | None = None, wedge: bool = False):
    """`constant/polyMesh` as `courant.case.polymesh` reads it, or None: missing, not the mesh the
    solver read (`_not_what_the_solver_read`), above the cost bounds, binary or malformed, or a
    wedge mesh (read with `wedge`: VER-7d's skewness, which does not need the mesh's geometric
    directions). The one gate on reading the mesh files: `polymesh_quality` (VER-7a), the case
    reader's held wall temperatures (VER-17) and `skewness_of_faces_with_area` (VER-7d) ask it.
    Notes each file it read (VER-14)."""
    from courant.case.polymesh import FILES, read_polymesh
    from courant.case.reads import refuse_outside

    case = Path(case_dir)
    d = case / "constant" / "polyMesh"
    if not all((d / n).is_file() for n in FILES) or _not_what_the_solver_read(case, log_text):
        return None
    try:
        for n in FILES:                 # MCP-3: sized and read only inside the roots
            refuse_outside(d / n)
        if sum((d / n).stat().st_size for n in FILES) > POLYMESH_MAX_BYTES:
            return None
        with open(d / "owner", "rb") as fh:
            m = re.search(rb"nFaces:[ \t]*(\d{1,12})", fh.read(4096))
        if m and int(m.group(1)) > POLYMESH_MAX_FACES:
            return None
        mesh = read_polymesh(case, max_faces=POLYMESH_MAX_FACES)   # notes each file (VER-14)
    except Exception:  # noqa: BLE001 — binary, partial or malformed: no metrics, as no log
        return None
    if len(mesh.faces) > POLYMESH_MAX_FACES or \
            (not wedge and any(p.type == "wedge" for p in mesh.boundary)):
        return None
    return mesh


#: The patch types whose faces `boundaryFaceSkewness` judges in v2406 (`polyMeshTools::
#: faceSkewness` uses it on every patch that is not `coupled()`): the plain and the constraint
#: types that couple nothing. Any other type (a cyclic, AMI, processor or mapped patch, or one
#: this list does not name) leaves checkMesh's maximum standing.
_UNCOUPLED = frozenset({"patch", "wall", "empty", "wedge", "symmetry", "symmetryPlane"})
#: OpenFOAM's `VSMALL` in double precision: `checkFaceAreas` calls a face of |area| below it
#: zero (doubleScalar.H).
VSMALL = 1.0e-300


def skewness_of_faces_with_area(case_dir: str, quality: "MeshQuality", *,
                                log_text: str | None = None) -> "MeshQuality":
    """`quality` with VER-7d's reading added when it applies, else `quality` itself.

    checkMesh reports faces of zero area and a maximum skewness above 4. v2406 computes a zero-
    area face's skewness as a distance over its floor ROOTVSMALL (`Mesh.face_skewness`), so that
    maximum (2.5475503e+146 on the committed axisymmetric pipe, `tests/fixtures/real_openfoam/
    wedge_pipe`, whose collapsed axis is an `empty` patch of zero-area faces) may say nothing
    of any face that is discretised. This recomputes every
    face's skewness from `constant/polyMesh` and, when the mesh files are the mesh checkMesh
    described and every zero-area face lies on an `empty` patch, which the solver does not
    discretise (`emptyFvPatch::size()` is 0, src/finiteVolume/fvMesh/fvPatches/constraint/empty/
    emptyFvPatch.H), sets the largest skewness of the faces with an area. Evidence that the files
    are what checkMesh read, all of it required: the same number of cells, of zero-area faces
    (|area| below VSMALL, `checkFaceAreas`) and of faces above the threshold of 4. Nothing
    applies, and checkMesh's maximum stands, for a mesh with a negative-volume cell, a patch type
    outside `_UNCOUPLED`, a zero-area face off an `empty` patch, a count that differs, or mesh
    files `solver_polymesh` does not read."""
    import numpy as np

    if not (quality.zero_area_faces and quality.max_skewness is not None
            and quality.max_skewness > SKEW_THRESHOLD and quality.n_skew_faces is not None
            and quality.n_zero_area_faces is not None and quality.n_cells is not None
            and quality.n_negative_volume_cells == 0 and quality.source is None):
        return quality
    try:
        mesh = solver_polymesh(case_dir, log_text=log_text, wedge=True)
    except Exception:  # noqa: BLE001 — a directory that cannot be searched: v0.5.0's reading
        # `Path.is_file` raises on EACCES (Python 3.11 ignores ENOENT, ENOTDIR, EBADF and ELOOP
        # alone), and `read_mesh_quality`, which never raises, ended `courant verify` and
        # `courant preflight` in an internal error where v0.5.0 gave a verdict (the review of
        # hp6/integrate). checkMesh's maximum stands.
        return quality
    if mesh is None or any(p.type not in _UNCOUPLED for p in mesh.boundary):
        return quality
    try:
        with np.errstate(all="ignore"):
            skew = mesh.face_skewness()
            zero = np.linalg.norm(mesh.face_areas, axis=1) < VSMALL
    except Exception:  # noqa: BLE001 — degenerate beyond what the formula takes: v0.5.0's reading
        return quality
    on_empty = np.zeros(len(mesh.faces), dtype=bool)
    patches = []
    for p in mesh.boundary:
        if p.type == "empty" and p.n:
            on_empty[p.start:p.start + p.n] = True
            if zero[p.start:p.start + p.n].any():
                patches.append(p.name)
    if (mesh.n_cells != quality.n_cells or int(zero.sum()) != quality.n_zero_area_faces
            or int((skew > SKEW_THRESHOLD).sum()) != quality.n_skew_faces
            or not np.isfinite(skew[~zero]).all() or (zero & ~on_empty).any()):
        return quality
    rest = skew[~zero]
    return quality.model_copy(update={
        "skewness_with_area": float(rest.max()) if rest.size else 0.0,
        "zero_area_patches": tuple(patches)})


def polymesh_quality(case_dir: str, *, log_text: str | None = None) -> "MeshQuality | None":
    """checkMesh's cell aspect ratio, from `constant/polyMesh` as the solver read it, or None.

    The definition is checkMesh's (`primitiveMeshTools::cellClosedness`, called by
    `checkClosedCells` with `mesh.geometricD()`): per cell, the sums of the component magnitudes
    of its face area vectors; over the solved directions, the largest over the smallest; in
    3-D, also one sixth of their total over V^(2/3), whichever is larger. A direction is
    unsolved when the `empty` patches' summed |area| components, normalised, exceed 1e-6 in it
    (`polyMesh::calcDirections`). The mesh and its geometry are `courant.case.polymesh`'s.

    None (not checked, as with no checkMesh log) for: a mesh that may not be the one the solver
    read (`_not_what_the_solver_read`); a wedge mesh (its geometric directions depend on the
    wedge's centre normal); a binary or unreadable one; one above the cost bounds.

    Found missing (RRD RM-35, runs c10k.icoFoam_cavity.019 and .075): a 2000 x 1 cavity whose
    cells are 2000 times longer than tall was accepted because nobody had run checkMesh.
    """
    import numpy as np

    from courant.case.polymesh import ROOTVSMALL

    mesh = solver_polymesh(case_dir, log_text=log_text)
    if mesh is None:
        return None
    try:
        mag = np.abs(mesh.face_areas)
        ni = mesh.n_internal
        sum_mag = np.zeros((mesh.n_cells, 3))
        np.add.at(sum_mag, mesh.owner, mag)
        np.add.at(sum_mag, mesh.neighbour, mag[:ni])
        solved = np.ones(3, dtype=bool)
        empty = np.zeros(3)
        for p in mesh.boundary:
            if p.type == "empty" and p.n:
                empty += mag[p.start:p.start + p.n].sum(axis=0)
        if np.linalg.norm(empty) > 0:
            solved = empty / np.linalg.norm(empty) <= 1e-6
        if not solved.any() or not mesh.n_cells:
            return None
        cmp = sum_mag[:, solved]
        ratio = cmp.max(axis=1) / (cmp.min(axis=1) + ROOTVSMALL)
        if solved.all():
            vol = np.maximum(ROOTVSMALL, mesh.cell_volumes)
            ratio = np.maximum(ratio, sum_mag.sum(axis=1) / 6.0 / vol ** (2.0 / 3.0))
        lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
        extent = {"bounding_box": (tuple(float(x) for x in lo), tuple(float(x) for x in hi)),
                  "total_volume": float(mesh.cell_volumes.sum()),
                  "solution_directions": tuple(int(x) for x in solved)}
    except Exception:  # noqa: BLE001 — degenerate geometry: no metrics rather than a traceback
        return None
    return MeshQuality(n_cells=int(mesh.n_cells), max_aspect_ratio=float(ratio.max()),
                       source=POLYMESH_SOURCE, **extent)


def read_mesh_quality(case_dir: str, *, polymesh: bool = False,
                      log_text: str | None = None) -> "MeshQuality | None":
    """Mesh metrics from a checkMesh log already sitting in the case, or None.

    With `polymesh`, a case with no checkMesh log gets the aspect ratio computed from its mesh
    files (`polymesh_quality`; `log_text`, the run's log, says whether the run was parallel).
    The gate asks for it (`verify.ingest`, `verify.preflight`); the watcher does not
    (`internal.session`). A checkMesh log, when there is one, always wins, because it is the
    solver project's own computation of every metric.

    Shared by the two paths that need it so there is one answer to "where does
    the checkMesh log live". They differ in who *runs* checkMesh, not in where
    they look for it:

      * `courant watch` never runs it. That is the customer's cluster, their
        queue and possibly their 50M-cell mesh, and a watcher that silently
        starts a heavy job is a watcher nobody installs twice.
      * `run_live_case` does, because the sandbox is ours and the run is already
        happening inside it.

    Never raises. A partial or truncated checkMesh log is common, and losing
    mesh metrics must not cost the rest of the diagnosis.
    """

    from courant.case.reads import reading

    for name in CHECKMESH_LOG_NAMES:
        candidate = Path(case_dir) / name
        if not candidate.is_file():
            continue
        try:
            with reading(candidate):    # VER-14: read, so listed, whether or not it parses
                text = candidate.read_text(errors="replace")
            quality = MeshQuality.from_checkmesh(text)
        except Exception:  # noqa: BLE001 — a partial log is not a failure
            return None
        # VER-7d: a zero-area face's skewness is not a measurement. Reads the mesh files (and
        # lists them, VER-14) only for a log that reports zero-area faces above the threshold.
        return skewness_of_faces_with_area(case_dir, quality, log_text=log_text)
    return polymesh_quality(case_dir, log_text=log_text) if polymesh else None
