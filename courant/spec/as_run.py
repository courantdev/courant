"""Conformance against what ran, not only the deck (TRD SPEC-13).

The deck reader (`courant.spec.deck`) reads a value where the specification names it. For most
items that file is what the solver read. For these it is not, and the hard-proofing audit
(21 Oct 2026) measured each on real v2406 runs, every one accepted against its unedited
`courant spec init` draft:

- **solver** names `system/controlDict`'s `application`, which no solver reads. What ran is the
  log's `Exec` line: the cavity deck run with pimpleFoam conformed to "solver icoFoam", and the
  pisoFoam/RAS deck run with icoFoam to "solver pisoFoam", in a verdict whose own `run.solver`
  said otherwise.
- **endTime** names `controlDict`'s `endTime`, but the solver also read `stopAt`, and the log
  shows where the run ended: a cavity with `stopAt nextWrite` ended at t = 0.1 of 0.5 and
  conformed to "endTime 0.5".
- **turbulence model** names `RAS/RASModel`, but the solver also read the model's `turbulence`
  switch (`getOrDefault<Switch>("turbulence", true)`, RASModel.C): off, no equation of the model
  is solved. The log names the model it selected.
- **mesh cells** and **mesh scale** name `system/blockMeshDict`, which the solver never reads:
  it reads `constant/polyMesh`. A mesh refined after blockMesh (`refineMesh`) or scaled
  (`transformPoints -scale 0.9`) conformed to both.
- **every value under `0/`** (a boundary value, an initial field) is read from `0/`, but a run
  restarted from a later time (`startFrom latestTime` after editing that time) read its fields
  there: a cavity whose `0.1/U` held a 2 m/s lid ran from 0.1 and conformed to the 1 m/s of
  `0/U`.
- **a boundary value** is the patch's `value` (or the entry the item names) whatever the patch's
  type, but the solver applies only the entries its type reads: a cavity lid switched to
  `type zeroGradient;` with its `value uniform (1 0 0)` left behind never moved (every Courant
  number and residual of the run 0) and conformed to "U at movingWall (1 0 0)".

Under `conformance.as_run` (behind `--checks next`, GATE-13: it turns accepts into rejects and
into insufficient), `AsRun.read` takes the deck reader's value for an item and checks it
against what ran:

- the solver is the program of the log's last `Exec` line that names a solver
  (`solvers.solver_exec`: a log of several applications, `(blockMesh; icoFoam) > log.run`, ran
  icoFoam). A log with no such line cannot say what ran, so the item is not checked (VER-5.9):
  the deck's `application` never stands in for it. With no log at all (`courant spec check`)
  the deck decides, and the item's basis says so. On OpenFOAM.org's `foamRun` the program is
  the same for every case and what solves is its module: a `solver module` item (controlDict's
  `solver`) is the Exec line's `-solver` argument when it has one, else controlDict's `solver`
  (what foamRun reads); `foamMultiRun`, which reads `regionSolvers`, and any other program leave
  it not checked;
- the end time is where the log's last time step was, when the log shows the solver finished
  (`End`) somewhere other than the deck's `endTime` and nothing it printed says it stopped on its
  own criterion (`... solution converged in N iterations`, a `runTimeControl` function object);
  a `stopAt` other than `endTime` is itself a different end (`end_against`, which the verdict's
  not-checked line for a run without such an item reads too);
- a model whose switch is off is "off", and a model the log selected that the deck does not
  name is the log's;
- a value under `0/` is read at the time the run started from: the log's `Create mesh for
  time = X`, else `controlDict`'s `startFrom`/`startTime` (`latestTime` after a run names its
  end, so without a log it is not known). From a later time, a boundary value's file is that
  time directory's, in the case or in `processor*/` (the ranks first when the run was parallel,
  as the mesh is); when it is in neither, or the start is not known, the item is not checked
  (VER-5.9). An initial field (or a file's hash) is not checked on a restart: the solver read an
  earlier run's state there;
- a boundary value is read through the patch's type, in the file it was read from: a type that
  applies the item's entry (`check.BC_PARAMETERS`) is compared as before, `noSlip` applies
  (0 0 0), a type that applies no value of its own (zeroGradient, slip, a gradient, a
  constraint) does not conform to a `value` item, and any other type is not checked (VER-5.9);
- the mesh is the one the solver read, as `courant.case.mesh.solver_mesh` names it (the one
  rule, the mesh metrics' too): the case's own (`constant/polyMesh`, or the time directory at
  or before the start holding one), or for a parallel run (either of the log's signals) each
  rank's (`processor<N>/`), summed. A multi-region case, a collated decomposition
  (`processors<N>/`), a parallel signal the solver's own Exec line contradicts, a rank
  missing, or a decomposed case beside its own mesh with no log saying which ran leaves the
  mesh items not checked (VER-5.9);
- the cell count is the solved mesh's, counted from its lists as OpenFOAM counts them, and its
  split per block is counted along each hex block's straight edges between its corner
  vertices (blockMesh puts nx + 1 points on such an edge), at the scale the solved mesh is at.
  A split whose cells are the mesh's is what was solved, and is compared (a mesh refined after
  blockMesh is found at its refined counts and does not conform). When it cannot be counted
  (curved edges, named vertices, a mesh moved or rebuilt, or edges whose counts do not give the
  mesh's cells, as a snapped or locally refined mesh's do not), only the total is compared and
  the basis says so: blocks that give it are the deck's, or the specification's when only those
  do (a blockMeshDict edited after meshing); blocks that give neither describe some other mesh,
  so the item is not checked (VER-5.9). A bound on the cell count is compared with the solved
  mesh's;
- the scale is the one at which every block vertex of blockMeshDict is a point of the solved
  mesh (blockMesh puts every block's corners in the mesh) and the mesh's extent is the
  vertices' (with their curved edges' points): the deck's, the specification's, or the solved
  mesh's own extent over the vertices' (a mesh scaled after blockMesh), so a mesh scaled 0.9 is
  found at 0.9 times the deck's scale, and one scaled up 2 (whose points include every vertex)
  at twice it, and neither conforms. When no single scale does it (a mesh moved, snapped or
  rebuilt, a curved edge bulging past its points), the item is not checked.

A mesh that is there but that the reader cannot read whole (binary, past the size bound summed
over every rank before any is read, compressed, any rank of a decomposed one) leaves the mesh
items not checked (VER-5.9), never compared on blockMeshDict alone. A case with no mesh files at
all is compared with blockMeshDict only when there is no log (`courant spec check` before
meshing): with a log the solver read some mesh, and it is not in the case, so the items are not
checked. Read-only, deterministic, no subprocess.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

from courant.case import reads
from courant.spec import deck

_CREATE_MESH = re.compile(r"^Create mesh for time = (\S+)", re.MULTILINE)
_SELECTED = re.compile(r"^Selecting (RAS|LES) (?:turbulence |momentum transport )?model (\S+)",
                       re.MULTILINE)
#: What a solver prints when it ends a run on a criterion of the case's own before `endTime`:
#: residualControl met (`simpleControl`, `pimpleControl`: "SIMPLE solution converged in 285
#: iterations"), or a `runTimeControl` function object's conditions met.
_OWN_STOP = re.compile(r"(?im)^.*(?:\bsolution converged in \d+ iterations|"
                       r"\bstopping calculation\b).*$")
_RUN_TIME_CONTROL = re.compile(r"\btype\s+runTimeControl\s*;")

_CONTROL = "system/controlDict"
_TURBULENCE_FILES = ("constant/turbulenceProperties", "constant/momentumTransport")
_MODEL_KEYS = ("RASModel", "LESModel", "model")
_SCALE_KEYS = ("scale", "convertToMeters")
#: The words OpenFOAM's `Switch` reads as false (`Switch.C`).
_OFF = frozenset({"false", "off", "no", "n", "f", "none", "0"})
#: Patch types that apply no value of their own: the patch value is the interior's
#: (zeroGradient, whose dictionary constructor reads no `value`, `zeroGradientFvPatchField.C`),
#: follows from a gradient (fixedGradient) or from a slip or geometric constraint. A `value`
#: entry written under one is ignored by the solver.
_NO_VALUE = frozenset({"zeroGradient", "fixedGradient", "slip", "symmetry", "symmetryPlane",
                       "empty", "wedge", "cyclic", "cyclicAMI", "cyclicACMI", "cyclicSlip",
                       "processor", "processorCyclic"})
#: Patch types that apply a fixed value of their own, whatever their `value` entry.
_APPLIES_ZERO = {"noSlip": [0.0, 0.0, 0.0]}
_WORD = re.compile(r"[A-Za-z_][\w:]*")
#: What each `stopAt` other than `endTime` does (`Time::stopAtControlNames`).
_STOP_AT = {"writenow": "at once, after writing", "nowritenow": "at once, without writing",
            "nextwrite": "at its next write", "clocktime": "at a wall-clock time",
            "cputime": "at a CPU time"}


def names_the_mesh(item) -> bool:
    """Whether an item names blockMeshDict's cells or scale, the two `AsRun` checks against the
    mesh the solver read."""
    return item.blocks == "cells" or (item.file.endswith("blockMeshDict") and not item.path
                                      and item.key in _SCALE_KEYS)


#: The foamRun family (OpenFOAM.org v11 on): one program for every case, the solver a module.
_MODULE_RUNNERS = frozenset({"foamRun", "foamMultiRun"})
#: Why a mesh item with a log is not checked when the case holds no mesh files at all.
_NOT_IN_CASE = ("the mesh the solver read is not in the case (no constant/polyMesh, no time "
                "directory's polyMesh, no rank's), so it is not known")


def executable(log_text: str | None) -> str | None:
    """The solver the log ran, without its directory (`solvers.solver_exec`, the one reading of
    which program ran), or None."""
    from courant.case.solvers import solver_exec

    ex = solver_exec(log_text)
    return ex.program if ex is not None else None


@dataclass(frozen=True)
class Run:
    """The solver's own record of a run: its log, the parser's events from it, and the exit code
    the verdict read (`exitcode.verdict_exit_code`: the caller's, else the log's End under the
    checks in force, VER-3a under --checks next; None, unknown). The pipeline passes the code it
    decided VER-5.16 on, so the endTime item and the line a run that ended short carries read
    the end it read: there is no second reading of `End` here (the final review of 0.6.0, 21 Oct
    2026, found `ended` inferring its own from any `End`, a utility's included)."""

    log_text: str
    log_name: str
    events: tuple = ()
    exit_code: int | None = field(kw_only=True)

    @cached_property
    def exec(self):
        """The solver's Exec line (`solvers.solver_exec`), or None."""
        from courant.case.solvers import solver_exec

        return solver_exec(self.log_text)

    @cached_property
    def executable(self) -> str | None:
        return self.exec.program if self.exec is not None else None

    @cached_property
    def parallel(self) -> bool:
        """Whether the run was parallel, on either of the log's signals (`mesh.ran_parallel`)."""
        from courant.case.mesh import ran_parallel

        return bool(ran_parallel(self.log_text))

    @property
    def ended(self) -> bool:
        """Whether the run finished: exit code 0, as the verdict read it (`exit_code`)."""
        return self.exit_code == 0

    @cached_property
    def _steps(self) -> list:
        from courant.events import EventKind

        return [e.payload for e in self.events if e.kind is EventKind.RESIDUAL
                and getattr(e.payload, "time", None) is not None]

    @cached_property
    def last_time(self) -> float | None:
        return float(self._steps[-1].time) if self._steps else None

    @cached_property
    def last_delta_t(self) -> float | None:
        for p in reversed(self._steps):
            if getattr(p, "delta_t", None):
                return float(p.delta_t)
        return None

    @cached_property
    def own_stop(self) -> str | None:
        m = _OWN_STOP.search(self.log_text)
        return m.group(0).strip() if m else None

    @cached_property
    def start_name(self) -> str | None:
        """The time the solver read its mesh and fields at, as it printed it (`Create mesh for
        time = 0.1`), or None when the log does not say."""
        m = _CREATE_MESH.search(self.log_text)
        return m.group(1) if m and _number(m.group(1)) is not None else None

    @cached_property
    def selected_model(self) -> tuple[str, str] | None:
        m = _SELECTED.search(self.log_text)
        return (m.group(1), m.group(2)) if m else None


@dataclass(frozen=True)
class SolvedMesh:
    """The mesh the solver read, as far as the reader could read it."""

    where: str                          # e.g. "constant/polyMesh"
    cells: int | None = None
    cells_basis: str = ""
    points: object | None = None        # an (N, 3) array
    note: str | None = None             # why something was not read


def _number(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _times(root: Path) -> list[str]:
    """The finite numeric time directory names directly under `root`, ascending."""
    from courant.case.mesh import time_dirs

    return time_dirs(root)


def _same_time(a: float, b: float) -> bool:
    """Whether two times are one: equal within 1e-9 of the larger in magnitude (or absolutely,
    within 1e-9, below 1). It is not OpenFOAM's `timeName` comparison (6 significant digits by
    default): two times one `timeName` writes alike may differ by more than this, and are then
    two times here."""
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


class AsRun:
    """`deck.read`'s value for an item, checked against what ran. One per conformance check."""

    def __init__(self, case_dir: str | Path, run: Run | None = None):
        self.case = Path(case_dir)
        self.run = run

    # ---------------------------------------------------------------- dispatch
    def read(self, item, found: deck.Found, *, rel: float) -> deck.Found:
        if item.file.startswith("0/"):
            try:
                return self._at_start(item, found)
            except Exception as exc:                    # noqa: BLE001 - never the deck's 0/ then
                return deck.Found(None, item.file, note=f"the time the run started from could "
                                                        f"not be read: {reads.reason(exc)}")
        try:
            if item.blocks == "cells":
                return self._cells(item, found)
            if item.sha256 is not None or item.patch is not None:
                return found
            if item.file == _CONTROL and not item.path and item.key == "application":
                return self._solver(found)
            if item.file == _CONTROL and not item.path and item.key == "solver":
                return self._module(found)
            if item.file == _CONTROL and not item.path and item.key == "endTime":
                return self._end_time(found, rel)
            if (item.file in _TURBULENCE_FILES and len(item.path) == 1
                    and item.path[0] in ("RAS", "LES") and item.key in _MODEL_KEYS):
                return self._model(item, found)
            if item.file.endswith("blockMeshDict") and not item.path and item.key in _SCALE_KEYS:
                return self._scale(item, found)
        except Exception as exc:                        # noqa: BLE001 - a reader limit, not a verdict
            if names_the_mesh(item):
                # blockMeshDict is not what the solver read: never compared on it instead.
                return deck.Found(None, item.file, note=f"the mesh the solver read could not be "
                                                        f"read: {reads.reason(exc)}")
            return replace(found, basis=_join(found.basis, f"not compared with what ran: "
                                                           f"{reads.reason(exc)}"))
        return found

    # ---------------------------------------------------------------- the time the run started
    @cached_property
    def start(self) -> tuple[str | None, str]:
        """(the time the run read its fields at, as named; None when it is not known) and how
        that is known, or why not. The log's `Create mesh for time` is what the solver did and
        wins; without it, `controlDict`'s `startFrom` (OpenFOAM's default `startTime`) and
        `startTime` (default 0)."""
        run = self.run
        if run is not None and run.start_name is not None:
            return run.start_name, f"{run.log_name}: 'Create mesh for time = {run.start_name}'"
        text, _ = deck._read_text(self.case, _CONTROL)
        start_from = (deck.entry(text or "", "startFrom") or "startTime").strip().strip('"')
        how = f"{_CONTROL} startFrom {start_from}"
        if start_from == "startTime":
            raw = deck.entry(text or "", "startTime")
            t = _number(raw) if raw is not None else 0.0
            return (None, f"{how}, startTime is not a number") if t is None else \
                (_fmt(t), f"{how}, startTime {_fmt(t)}")
        times = sorted({n for root in self._roots() for n in _times(root)}, key=float)
        if start_from == "firstTime":
            return (times[0], how) if times else ("0", how)
        if start_from == "latestTime":
            if len(times) <= 1:
                return (times[0] if times else "0"), how
            return None, (f"{how}: after a run, the latest time is where it ended, not where it "
                          "started, and no log says which time it read")
        # not one of OpenFOAM's three words, so the case's own text: never quoted
        return None, f"{_CONTROL} startFrom is not firstTime, startTime or latestTime"

    def _roots(self) -> list[Path]:
        """Where the solver read its time directories: the case, then its rank directories
        (`mesh.rank_dirs`); the ranks first when the run was parallel (`Run.parallel`, either of
        the log's signals), as the mesh is read from them."""
        from courant.case.mesh import rank_dirs

        ranks = rank_dirs(self.case)
        if self.run is not None and self.run.parallel:
            return [*ranks, self.case]
        return [self.case, *ranks]

    def _at_start(self, item, found: deck.Found) -> deck.Found:
        """A value under `0/`, read at the time the run started from."""
        name, how = self.start
        if name is None:
            return deck.Found(None, item.file, note=f"the time the run started from is not "
                                                    f"known ({how}), so the value it read is not")
        t = float(name)
        if t == 0.0:
            return self._applied(item, found)
        if item.patch is None:
            # An initial field (`internalField`) or a whole file's hash: at a later start time
            # the solver read the state some earlier run left, not the initial condition the item
            # names, so neither the later file nor `0/` says what this run began from.
            return deck.Found(None, item.file,
                              note=f"the run started from t = {name} ({how}): the initial state "
                                   f"it read there is not {item.file}'s, so this item is not "
                                   "what it read")
        rest = item.file[2:]
        for root in self._roots():
            directory = next((d for d in _times(root) if _same_time(float(d), t)), None)
            if directory is None or not (root / directory / rest).is_file():
                continue
            rel = (root / directory / rest).relative_to(self.case).as_posix()
            try:
                got = deck._read_one(self.case, rel, item)
            except Exception as exc:                    # noqa: BLE001 - a reader limit
                return deck.Found(None, rel, note=f"unreadable: {reads.reason(exc)}")
            return self._applied(item, replace(got, basis=_join(
                got.basis, f"the run started from t = {name} ({how}), so {rel} is what it read")))
        return deck.Found(None, item.file,
                          note=f"the run started from t = {name} ({how}), and no {name}/{rest} "
                               "is in the case or its processor directories")

    # ---------------------------------------------------------------- the patch's type
    def _applied(self, item, found: deck.Found) -> deck.Found:
        """A boundary value as the patch's type applies it, in the file the value was read from.

        The deck reader reads the entry an item names (`value` by default) whatever the patch's
        type; the solver applies only the entries its type reads. A type that applies the
        item's entry (`check.BC_PARAMETERS`, the drafter's table) is compared as before;
        `noSlip` applies (0 0 0); a type that applies no value of its own (`_NO_VALUE`) does not
        conform to a `value` item; any other type is not checked (VER-5.9)."""
        if item.patch is None or item.sha256 is not None:
            return found
        try:
            text, _ = deck._read_text(self.case, found.file)
            bf = deck.block(text or "", "boundaryField")
            body = deck.block(bf, item.patch) if bf is not None else None
            raw = deck.entry(body, "type") if body is not None else None
        except Exception as exc:                        # noqa: BLE001 - a reader limit
            return deck.Found(None, found.file, note=f"the type of {item.patch} in {found.file} "
                                                     f"could not be read: {reads.reason(exc)}")
        if body is None:
            return found                                # no such patch: the deck's note stands
        from courant.spec.check import BC_PARAMETERS

        key = item.key or "value"
        kind = (raw or "").strip().strip('"')
        if not _WORD.fullmatch(kind):
            return deck.Found(None, found.file,
                              note=f"the type of {item.patch} in {found.file} is not a "
                                   "literal word, so which of its entries the solver applies is "
                                   "not known")       # never quoted: the case's text (deck)
        where = f"{item.patch} is {kind} in {found.file}"
        if key in BC_PARAMETERS.get(kind, ()):
            return found
        if key == "value" and kind in _APPLIES_ZERO:
            return deck.Found(_APPLIES_ZERO[kind], found.file, dimensions=found.dimensions,
                              basis=_join(found.basis, f"{kind} applies (0 0 0), whatever its "
                                                       "value entry says"))
        if key == "value" and kind in _NO_VALUE:
            return deck.Found(f"type {kind}", found.file,
                              note=f"{where}, which applies no value of its own (the patch value "
                                   "follows from the interior, a gradient or a constraint), so "
                                   "its value entry is not what the solver applied")
        if kind in _NO_VALUE or kind in _APPLIES_ZERO:
            # Another entry (an outlet's inletValue) under a type that reads none: the task's
            # condition is not the solver's, but a zero-gradient outlet may be near enough to an
            # inletOutlet one where the flow only leaves, so it is not checked, not rejected.
            return deck.Found(None, found.file,
                              note=f"{where}, which reads no {key} entry, so the condition this "
                                   "item names is not one the solver applied")
        if kind in BC_PARAMETERS:
            return deck.Found(None, found.file,
                              note=f"{where}, which applies its {' and '.join(BC_PARAMETERS[kind])}"
                                   f", not its {key} entry, so the value it applied is not the "
                                   "one this item names")
        return deck.Found(None, found.file,
                          note=f"{where}: which of its entries the solver applies is not known to "
                               f"the reader, so its {key} entry is not taken for it")

    # ---------------------------------------------------------------- the solver
    def _solver(self, found: deck.Found) -> deck.Found:
        run = self.run
        if run is None:
            return replace(found, basis=_join(found.basis, "not compared with what ran: no "
                                                           "solver log was given"))
        exe = run.executable
        if exe is None:
            return deck.Found(None, run.log_name,
                              note=f"{run.log_name} has no 'Exec :' line naming a solver, so the "
                                   f"program that ran is not known; {_CONTROL}'s application, "
                                   "which no solver reads, does not stand in for it")
        if exe == found.value:
            basis = f"{run.log_name}'s Exec line names the same program"
            if exe in _MODULE_RUNNERS:
                # OpenFOAM.org v11 on: every case runs foamRun; what solves is its module.
                basis += (f"; on {exe} what solves is the solver module, which this item does "
                          "not name (a 'solver module' item, controlDict's solver, does)")
            return replace(found, basis=_join(found.basis, basis))
        named = f"system/controlDict names {found.value}" if found.value else \
            "system/controlDict names no application"
        return deck.Found(exe, run.log_name,
                          basis=f"the log's Exec line, the program that ran; {named}")

    def _module(self, found: deck.Found) -> deck.Found:
        """controlDict's `solver`, the module OpenFOAM.org's foamRun solves with: the Exec line's
        `-solver` argument when it has one (it overrides controlDict's, `foamRun.C`), else
        controlDict's `solver`, which foamRun reads. Not checked when the program that ran is
        not foamRun: foamMultiRun reads `regionSolvers`, and any other solver no module."""
        run = self.run
        if run is None:
            return replace(found, basis=_join(found.basis, "not compared with what ran: no "
                                                           "solver log was given"))
        ex = run.exec
        if ex is None:
            return deck.Found(None, run.log_name,
                              note=f"{run.log_name} has no 'Exec :' line naming a solver, so the "
                                   "module that ran is not known")
        if ex.program != "foamRun":
            reads = ("the modules its controlDict's regionSolvers names"
                     if ex.program == "foamMultiRun" else "no solver module")
            return deck.Found(None, run.log_name,
                              note=f"{ex.program} ran ({run.log_name}), which runs {reads}, not "
                                   f"{_CONTROL}'s solver entry")
        given = ex.option("solver")
        if given is None:
            return replace(found, basis=_join(found.basis, f"{run.log_name}'s Exec line gives "
                                                           "foamRun no -solver, so it read this"))
        if given == found.value:
            return replace(found, basis=_join(found.basis, f"{run.log_name}'s Exec line gives "
                                                           "foamRun the same -solver"))
        named = f"{_CONTROL} names {found.value}" if found.value else \
            f"{_CONTROL} names no solver"
        return deck.Found(given, run.log_name,
                          basis=f"the Exec line's -solver {given}, which foamRun ran whatever "
                                f"controlDict says; {named}")

    # ---------------------------------------------------------------- the end time
    def _end_time(self, found: deck.Found, rel: float) -> deck.Found:
        end = _number(found.value)
        if end is None:
            return found
        text, _ = deck._read_text(self.case, _CONTROL)
        stop = (deck.entry(text or "", "stopAt") or "endTime").strip().strip('"')
        if stop.lower() != "endtime":
            how = _STOP_AT.get(stop.lower(), f"as '{stop}' says")
            return deck.Found(f"stopAt {stop}", _CONTROL,
                              note=f"system/controlDict sets stopAt {stop}: the solver stops "
                                   f"{how}, not at its endTime {_fmt(end)}")
        how, reached, basis = end_against(self.run, end, rel=rel, control_text=text or "")
        if how in ("unknown", "reached"):
            return found
        if how == "own":
            return replace(found, basis=basis)
        return deck.Found(reached, self.run.log_name,
                          note=f"the run ended at t = {_fmt(reached)} ({self.run.log_name}), "
                               f"{how} its endTime {_fmt(end)}")

    # ---------------------------------------------------------------- the turbulence model
    def _model(self, item, found: deck.Found) -> deck.Found:
        if not isinstance(found.value, str):
            return found
        text, _ = deck._read_text(self.case, found.file)
        body = deck.block(text or "", item.path[0])
        switch = deck.entry(body, "turbulence") if body is not None else None
        if switch is not None and switch.strip().strip('"').lower() in _OFF:
            return deck.Found(f"{found.value}, turbulence {switch}", found.file,
                              note=f"{item.path[0]}/turbulence is {switch} in {found.file}: no "
                                   f"equation of {found.value} is solved, and the eddy "
                                   "viscosity stays at its initial field")
        logged = self.run.selected_model if self.run else None
        if logged and logged[0] == item.path[0] and logged[1] != found.value:
            return deck.Found(logged[1], self.run.log_name,
                              basis=f"the {logged[0]} model the log selected; {found.file} "
                                    f"names {found.value}")
        return found

    # ---------------------------------------------------------------- the mesh
    @cached_property
    def mesh(self) -> SolvedMesh | None:
        """The mesh the solver read (`courant.case.mesh.solver_mesh`, the one rule, the mesh
        metrics' too), or None when the case holds no mesh files at all.

        A mesh that cannot be named (a multi-region case, a collated decomposition, a rank
        missing, ...) or read whole (binary, past the size bound summed over every rank before
        any is read, compressed, any rank of it) comes back with no count or points and a note:
        the mesh items are then not checked, never compared on blockMeshDict alone."""
        from courant.case import polymesh
        from courant.case.mesh import POLYMESH_MAX_BYTES, solver_mesh

        found = solver_mesh(self.case, self.run.log_text if self.run is not None else None,
                            start=_number(self.start[0]))
        if found.why:
            return SolvedMesh(found.where, note=found.why)
        if not found.dirs:
            return None
        for d in found.dirs:
            if not (d / "owner").is_file():
                packed = ", ".join(sorted(p.name for p in d.iterdir() if p.name.endswith(".gz")))
                return SolvedMesh(found.where, note=f"{d.relative_to(self.case).as_posix()} is "
                                                    f"written compressed ({packed}), which the "
                                                    "reader does not read")
        # The cost bound is on all that would be read, every rank of it, before any is parsed.
        size = sum((d / n).stat().st_size for d in found.dirs for n in polymesh.FILES
                   if (d / n).is_file())
        if size > POLYMESH_MAX_BYTES:
            over = f" over its {len(found.dirs)} ranks" if len(found.dirs) > 1 else ""
            return SolvedMesh(found.where, note=f"{found.where} is {size / 2**20:.0f} MiB{over}, "
                                                f"past the reader's "
                                                f"{POLYMESH_MAX_BYTES // 2**20} MiB bound")
        parts = [self._read_mesh(d) for d in found.dirs]
        return parts[0] if len(parts) == 1 else self._summed(parts, found.where)

    def _read_mesh(self, d: Path) -> SolvedMesh:
        from courant.case import polymesh

        where = d.relative_to(self.case).as_posix()
        lists = "its owner and neighbour lists"
        try:
            owner = polymesh._read(d / "owner", lambda p: polymesh._labels(polymesh._body(p)[0]))
            neighbour = polymesh._read(d / "neighbour",
                                       lambda p: polymesh._labels(polymesh._body(p)[0]))
            cells = polymesh.cell_count(owner, neighbour, polymesh.noted_cells(d / "owner"))
        except polymesh.UnsupportedFormat as exc:       # "owner is binary: convert it ..."
            return SolvedMesh(where, note=f"{where}/{exc}")
        except (ValueError, OSError) as exc:
            return SolvedMesh(where, note=f"{where} could not be read: {reads.reason(exc)}")
        try:
            points = polymesh._read(d / "points", polymesh._points)
        except polymesh.UnsupportedFormat as exc:
            return SolvedMesh(where, cells=cells, cells_basis=lists, note=f"{where}/{exc}")
        except (ValueError, OSError) as exc:
            return SolvedMesh(where, cells=cells, cells_basis=lists,
                              note=f"{where}/points could not be read: {reads.reason(exc)}")
        return SolvedMesh(where, cells=cells, cells_basis=lists, points=points)

    @staticmethod
    def _summed(parts: list[SolvedMesh], where: str) -> SolvedMesh:
        """A decomposed mesh: its ranks' cells summed and their points pooled, or the first
        rank's reason it was not read whole."""
        import numpy as np

        unread = next((p for p in parts if p.cells is None), None)
        if unread is not None:
            return SolvedMesh(where, note=unread.note)
        basis = f"its owner and neighbour lists, summed over its {len(parts)} ranks"
        cells = sum(p.cells for p in parts)
        unread = next((p for p in parts if p.points is None), None)
        if unread is not None:
            return SolvedMesh(where, cells=cells, cells_basis=basis, note=unread.note)
        return SolvedMesh(where, cells=cells, cells_basis=basis,
                          points=np.concatenate([np.asarray(p.points) for p in parts]))

    def _no_mesh(self, found: deck.Found) -> deck.Found:
        """A mesh item when the case holds no mesh files at all: compared with blockMeshDict
        when there is no log (`courant spec check` before meshing); with a log the solver read
        some mesh, and it is not in the case."""
        if self.run is None:
            return found
        return deck.Found(None, "constant/polyMesh", note=_NOT_IN_CASE)

    def _cells(self, item, found: deck.Found) -> deck.Found:
        mesh = self.mesh
        if mesh is None:
            return self._no_mesh(found)
        if mesh.cells is None:
            return deck.Found(None, mesh.where,
                              note=f"the cell count of {mesh.where}, the mesh the solver read, is "
                                   f"not known: {mesh.note}")
        said = f"{mesh.where}, the mesh the solver read ({mesh.cells_basis})"
        if item.bounded:
            return deck.Found(float(mesh.cells), mesh.where,
                              basis=f"the cell count of {said}")
        if found.value is None:
            return found
        written = [float(x) for x in found.value]
        blocks = _cell_total(written)
        asked = ([float(x) for x in item.value] if isinstance(item.value, (list, tuple))
                 else None)
        split, why = self._split(found, mesh)
        if split is not None and _cell_total(split) != mesh.cells:
            why = (f"the counts along its block edges, {_fmt(split)}, give "
                   f"{_cell_total(split)} cells, not its {mesh.cells} (a mesh refined in part, "
                   "snapped or rebuilt)")
            split = None
        if split is not None:
            # Each block's counts as the solver's mesh has them, whatever blockMeshDict says now.
            counted = "counted along each block's edges"
            if split == written:
                return replace(found, basis=_join(found.basis, f"{said} has the {blocks} cells "
                                                               f"these blocks give, {counted}"))
            whose = ("the specified blocks give" if split == asked
                     else f"of blocks {_fmt(split)}")
            return deck.Found(split, mesh.where,
                              basis=f"{said} has the {mesh.cells} cells {whose}, {counted}; "
                                    f"{found.file}'s blocks give {_fmt(written)}")
        only = f"only the total was compared, its split per block not being known: {why}"
        if blocks == mesh.cells:
            return replace(found, basis=_join(found.basis, f"{said} has the {blocks} cells "
                                                           f"these blocks give; {only}"))
        if asked and _cell_total(asked) == mesh.cells:
            # blockMeshDict edited after meshing: the solver read the specified count.
            return deck.Found(asked, mesh.where,
                              basis=f"{said} has the {mesh.cells} cells the specified blocks "
                                    f"give; {only}; {found.file}'s blocks give {blocks}")
        return deck.Found(None, mesh.where,
                          note=f"{said} has {mesh.cells} cells; {found.file}'s blocks give "
                               f"{blocks}, so it is not the mesh this item names ({why})")

    def _split(self, found: deck.Found, mesh: SolvedMesh) -> tuple[list[float] | None, str]:
        """(each hex block's cell counts on the solved mesh, '') or (None, why not): counted along
        a straight edge per direction of each block, between its corner vertices placed where
        the solved mesh has them (`_placed`)."""
        if mesh.points is None:
            return None, f"its points were not read ({mesh.note})"
        text, _ = deck._read_text(self.case, found.file)
        text = text or ""
        points = _array(mesh.points)
        placed, why = _placed(text, points, [_deck_scale(text)])
        if placed is None:
            return None, why
        return _block_counts(text, placed, points, len(found.value) // 3)

    def _scale(self, item, found: deck.Found) -> deck.Found:
        """The scale at which blockMeshDict's vertices are points of the solved mesh, and its
        extent theirs: the deck's, the specification's (blockMeshDict edited after meshing), or
        the solved mesh's own extent over the vertices' (a mesh scaled after blockMesh)."""
        mesh = self.mesh
        scale = found.value
        if scale is None:
            return found
        if mesh is None:
            return self._no_mesh(found)
        if mesh.points is None:
            return deck.Found(None, mesh.where,
                              note=f"the scale of {mesh.where}, the mesh the solver read, is not "
                                   f"known: {mesh.note}")
        text, _ = deck._read_text(self.case, found.file)
        vertices, targets, why = _geometry(text or "")
        if vertices is None:
            return replace(found, basis=_join(found.basis, f"not compared with {mesh.where}: "
                                                           f"{why}"))
        s_deck = _vec(scale)
        if s_deck is None:
            return found
        points = _array(mesh.points)
        said = f"{mesh.where}, the mesh the solver read"
        n = len(vertices)
        fits = f"is a point of {said}, and its extent is theirs"
        if _fits(vertices, targets, s_deck, points):
            return replace(found, basis=_join(found.basis, f"every one of {n} block vertices "
                                                           f"at this scale {fits}"))
        asked = _vec(item.value) if item.value is not None and not item.bounded else None
        for s in (asked, _own_scale(targets, points, s_deck)):
            if s is None or not _fits(vertices, targets, s, points):
                continue
            value = (float(s[0]) if not isinstance(scale, (list, tuple))
                     and _uniform(s) else [float(x) for x in s])
            return deck.Found(value, mesh.where,
                              basis=f"every one of {n} block vertices at {_fmt(value)} {fits}; "
                                    f"{found.file} says {_fmt(scale)}")
        missing = _not_points(vertices * s_deck, points)
        if missing:
            how = f"{missing} of {n} are not mesh points at its {_fmt(scale)}"
        else:
            import numpy as np

            how = (f"at its {_fmt(scale)} each is a mesh point, but the mesh spans "
                   f"{_fmt([float(x) for x in np.ptp(points, axis=0)])} where they span "
                   f"{_fmt([float(x) for x in np.ptp(targets * s_deck, axis=0)])}: a curved "
                   "edge bulging past its points, or a mesh grown past them")
        return deck.Found(None, mesh.where,
                          note=f"no single scale puts {found.file}'s {n} block vertices on "
                               f"{said} ({how}), so the scale it was solved at is not known")


def end_against(run: Run | None, end: float, *, rel: float,
                control_text: str = "") -> tuple[str, float | None, str | None]:
    """Where the run ended against `end`, the endTime it was given: (how, the time its last step
    reached, a basis). One definition, for the endTime item (SPEC-13) and the verdict's
    not-checked line on a run without one (`sentences.ENDED_SHORT`).

    - "unknown": the run is not known to have finished (`Run.ended`: exit code 0 as the verdict
      read it, the caller's or the log's `End`, whose End VER-3a decides under --checks next),
      or no step carries a time;
    - "reached": the last step is within one time step, or `rel`, of `end`;
    - "own": short of it, on a criterion of the case's own: the solver printed its own stop
      (`... solution converged in N iterations`, `stopping calculation`), or `control_text`
      (the case's controlDict) holds a `runTimeControl` function object;
    - "before" or "past": anywhere else."""
    if run is None or not run.ended or run.last_time is None:
        return "unknown", None, None
    reached = run.last_time
    if abs(reached - end) <= max(rel * abs(end), run.last_delta_t or 0.0):
        return "reached", reached, None
    if reached < end and run.own_stop:
        return "own", reached, (f"the run ended at {_fmt(reached)} on its own criterion "
                                f"({run.log_name}: '{run.own_stop[:80]}')")
    if reached < end and _RUN_TIME_CONTROL.search(control_text):
        return "own", reached, (f"the run ended at {_fmt(reached)}; the case's runTimeControl "
                                "function object may end it")
    return ("before" if reached < end else "past"), reached, None


def _cell_total(counts) -> int:
    """The cells a list of hex blocks' counts (nx ny nz, ...) gives."""
    c = [float(x) for x in counts]
    return int(sum(c[i] * c[i + 1] * c[i + 2] for i in range(0, len(c) - 2, 3)))


def _vertices(text: str):
    """(blockMeshDict's vertices as an (N, 3) array, None) or (None, why not)."""
    import numpy as np

    raw = deck.entry(text, "vertices")
    if raw is None:
        return None, "no literal vertices list"
    for word, why in (("transform", "it transforms its vertices"),
                      ("prescale", "it prescales its vertices")):
        if deck.entry(text, word) is not None or deck.block(text, word) is not None:
            return None, why
    body = re.sub(r"\bname\s+\w+", " ", raw)            # ESI's named vertices
    if re.search(r"[$#]|\bproject\b|[A-Za-z_]{2,}", re.sub(r"[eE][-+]?\d", "0", body)):
        return None, "its vertices are not all literal numbers"
    triples = re.findall(r"\(([^()]*)\)", body)
    try:
        rows = [[float(x) for x in t.split()] for t in triples]
    except ValueError:
        return None, "its vertices are not all literal numbers"
    if not rows or any(len(r) != 3 for r in rows):
        return None, "its vertices are not all literal 3-vectors"
    return np.asarray(rows, dtype=float), None


def _not_points(targets, points) -> int:
    """How many of `targets` are not points of the mesh, to the precision a mesh is written at
    (OpenFOAM's default `writePrecision 6`, so 1e-5 of the coordinates' size)."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if not len(pts):
        return len(targets)
    size = max(float(np.max(np.abs(targets))), float(np.ptp(targets, axis=0).max()), 1e-300)
    tol = 1e-5 * size
    order = np.argsort(pts[:, 0], kind="stable")
    xs = pts[order, 0]
    missing = 0
    for v in targets:
        lo, hi = np.searchsorted(xs, v[0] - tol), np.searchsorted(xs, v[0] + tol, side="right")
        near = pts[order[lo:hi]]
        if not len(near) or not (np.abs(near - v) <= tol).all(axis=1).any():
            missing += 1
    return missing


def _tol(targets) -> float:
    """The precision a mesh is written at, for `targets` (`_not_points`'s)."""
    import numpy as np

    t = np.asarray(targets, dtype=float)
    return 1e-5 * max(float(np.max(np.abs(t))), float(np.ptp(t, axis=0).max()), 1e-300)


def _array(points):
    import numpy as np

    return np.asarray(points, dtype=float)


def _vec(v):
    """A scale as a 3-vector (a scalar applies to all three axes), or None."""
    import numpy as np

    try:
        s = np.asarray(v if isinstance(v, (list, tuple)) else [v] * 3, dtype=float)
    except (TypeError, ValueError):
        return None
    return s if s.shape == (3,) and np.isfinite(s).all() else None


def _uniform(s) -> bool:
    import numpy as np

    return bool(np.allclose(s, s[0], rtol=1e-6, atol=0.0))


def _deck_scale(text: str):
    """blockMeshDict's own scale (`scale`, else `convertToMeters`, else 1), or None when it is
    not a literal number or 3-vector."""
    for key in _SCALE_KEYS:
        raw = deck.entry(text, key)
        if raw is not None:
            value, _, _ = deck.parse_value(raw)
            return _vec(value)
    return _vec(1.0)


def _edges(text: str):
    """blockMeshDict's `edges` entries as (kind, first vertex, last vertex, raw), or None when
    the list is not all literal (a macro, a directive, a vertex named rather than numbered)."""
    raw = deck.entry(text, "edges")
    if raw is None:
        return []
    body = raw.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    if re.search(r"[$#]", body):
        return None
    starts = list(re.finditer(r"\b([A-Za-z]\w*)\s+(\d+)\s+(\d+)\b", body))
    words = re.findall(r"\b[A-Za-z]\w*\b", re.sub(r"[eE][-+]?\d", "0", body))
    if len(words) != len(starts) + sum(1 for w in words if w == "origin"):
        return None                                 # a word that does not start an edge
    out = []
    for k, m in enumerate(starts):
        end = starts[k + 1].start() if k + 1 < len(starts) else len(body)
        out.append((m.group(1), int(m.group(2)), int(m.group(3)), body[m.end():end]))
    return out


def _geometry(text: str):
    """(blockMeshDict's vertices, the points that bound its extent, None) or (None, None, why
    not). A straight-edged deck's extent is its vertices'; a curved edge's points (an arc's
    point, a polyLine's or spline's) are on the mesh's boundary too, and are added. An arc's
    `origin` (ESI) is its centre, not a point of it, and is not. Edges that cannot be read as
    literal points add none, so the extent asked of the mesh is its vertices' alone (stricter,
    never looser)."""
    import numpy as np

    vertices, why = _vertices(text)
    if vertices is None:
        return None, None, why
    extra = []
    for kind, _a, _b, rest in _edges(text) or ():
        if kind in ("project", "projectCurve"):
            continue
        rest = re.sub(r"\borigin\s*(?:[-+\d.eE]+\s*)?\([^()]*\)", " ", rest)
        for t in re.findall(r"\(([^()]*)\)", rest):
            try:
                row = [float(x) for x in t.split()]
            except ValueError:
                continue
            if len(row) == 3 and all(math.isfinite(x) for x in row):
                extra.append(row)
    targets = np.vstack([vertices, np.asarray(extra, dtype=float)]) if extra else vertices
    return vertices, targets, None


def _same_extent(targets, points) -> bool:
    """Whether the mesh's bounding box is that of `targets`, to the precision of `_not_points`."""
    import numpy as np

    tol = _tol(targets)
    t, p = np.asarray(targets, dtype=float), np.asarray(points, dtype=float)
    return bool(np.allclose(p.min(axis=0), t.min(axis=0), rtol=0.0, atol=tol)
                and np.allclose(p.max(axis=0), t.max(axis=0), rtol=0.0, atol=tol))


def _fits(vertices, targets, s, points) -> bool:
    """Whether the deck at scale `s` is the solved mesh's: every block vertex a mesh point (a
    mesh scaled up by a whole factor still holds every original vertex) and the mesh's extent
    the deck's (which it does not)."""
    return not _not_points(vertices * s, points) and _same_extent(targets * s, points)


def _own_scale(targets, points, fallback):
    """The solved mesh's extent over the deck's, per axis, as a mesh is written (6 digits)."""
    import numpy as np

    extent_t, extent_p = np.ptp(targets, axis=0), np.ptp(points, axis=0)
    own = np.where(extent_t > 0, extent_p / np.where(extent_t > 0, extent_t, 1.0), fallback)
    return np.asarray([float(f"{x:.6g}") for x in own])


def _placed(text: str, points, candidates):
    """(blockMeshDict's vertices where the solved mesh has them, None) or (None, why not): at the
    first of `candidates`, or the mesh's own extent over the deck's, at which `_fits` holds."""
    vertices, targets, why = _geometry(text)
    if vertices is None:
        return None, why
    for s in [*candidates, _own_scale(targets, points, candidates[0] if candidates[0]
                                      is not None else _vec(1.0))]:
        if s is not None and _fits(vertices, targets, s, points):
            return vertices * s, None
    return None, "blockMeshDict's vertices are not the mesh's at any one scale"


#: A hex block: its eight vertex labels and its cell counts (as `deck._HEX` reads them).
_HEX_BLOCK = re.compile(r"\bhex\s*\(([^)]*)\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)")
#: A hex block's edges along each of its local directions, as vertex positions in its list
#: (OpenFOAM's hex model: x1 from 0 to 1, x2 from 0 to 3, x3 from 0 to 4).
_HEX_AXES = (((0, 1), (3, 2), (7, 6), (4, 5)),
             ((0, 3), (1, 2), (5, 6), (4, 7)),
             ((0, 4), (1, 5), (2, 6), (3, 7)))


def _block_counts(text: str, placed, points, n_blocks: int) -> tuple[list[float] | None, str]:
    """(each hex block's (nx ny nz) on the solved mesh, '') or (None, why not), counted along one
    straight, uncollapsed edge per direction of each block: blockMesh puts n + 1 points on a
    straight edge of n cells, whatever its grading, and a refined mesh has its refined count
    there. Curved edges are skipped (their points are off the chord); a deck that merges patch
    pairs or projects faces is not counted."""
    for word in ("mergePatchPairs", "faces"):
        raw = deck.entry(text, word)
        if raw is not None and raw.strip().strip("()").strip():
            return None, f"blockMeshDict's {word} is not empty"
    edges = _edges(text)
    if edges is None:
        return None, "blockMeshDict's edges are not all literal, so which are straight is not known"
    curved = {frozenset((a, b)) for kind, a, b, _ in edges if kind != "line"}
    hexes = list(_HEX_BLOCK.finditer(text))
    if len(hexes) != n_blocks:
        return None, "its hex blocks could not all be read with their vertices"
    tol = _tol(placed)
    out: list[float] = []
    for k, m in enumerate(hexes):
        try:
            labels = [int(x) for x in m.group(1).split()]
        except ValueError:
            return None, "a hex block names its vertices rather than numbering them"
        if len(labels) != 8 or max(labels) >= len(placed) or min(labels) < 0:
            return None, f"hex block {k} does not list 8 of its vertices"
        for axis in _HEX_AXES:
            n = None
            for i, j in axis:
                a, b = labels[i], labels[j]
                if a == b or frozenset((a, b)) in curved:
                    continue
                n = _edge_cells(placed[a], placed[b], points, tol)
                if n is not None:
                    break
            if n is None:
                return None, f"hex block {k} has no straight edge along one of its directions"
            out.append(float(n))
    return out, ""


def _edge_cells(a, b, points, tol: float) -> int | None:
    """The cells along the straight segment a-b: the distinct mesh points on it, less one (a
    decomposed mesh's pooled ranks repeat their shared points). None for a degenerate edge."""
    import numpy as np

    u = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    length = float(np.sqrt(u @ u))
    if length <= tol:
        return None
    p = np.asarray(points, dtype=float)
    box = (p >= np.minimum(a, b) - tol).all(axis=1) & (p <= np.maximum(a, b) + tol).all(axis=1)
    q = p[box] - a
    t = q @ u / length                                  # distance along the edge
    off = np.linalg.norm(q - np.outer(t / length, u), axis=1)
    ts = np.sort(t[off <= tol])
    if len(ts) < 2:
        return None
    return int(np.count_nonzero(np.diff(ts) > tol))


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return "(" + " ".join(_fmt(x) for x in v) + ")"
    x = float(v)
    return str(int(x)) if x.is_integer() and abs(x) < 1e15 else f"{x:g}"


def _join(a: str | None, b: str) -> str:
    return f"{a}; {b}" if a else b


__all__ = ["AsRun", "Run", "SolvedMesh", "end_against", "executable", "names_the_mesh"]
