"""Read solution fields from a solved case (live verification path).

The fields are parsed with foamlib, the library the case reader already parses `0/`, `constant/`
and `system/` with, and a dependency of every install (pyproject.toml `dependencies`, the release
lock, the binary). Until 11 Oct 2026 they were parsed with fluidfoam, which only the `sandbox`
extra installed and the binary excluded: a development checkout read fields, CI and the shipped
binary never did, and the verdict said "fields not read" there. fluidfoam also read a binary
field wrongly (a byte of the data taken for a delimiter), returned nothing for a `.gz` field and
failed on a one-line header; foamlib reads all three.

Best-effort: reads the internal fields the field-based checks use (p, U, k, epsilon,
omega, T, nut) plus nu, from the latest time directory. Anything missing is simply
absent — the corresponding checks skip rather than fail.

PARALLEL CASES — the reason this file is not three lines shorter.
================================================================
A decomposed run writes its results into `processor0/`, `processor1/`, … and leaves
the case root holding only `0/`, `constant/` and `system/`. `find_time_dirs` looks
at the root, so on every parallel case it returned `["0"]` and this function read
`0/U` — **the initial condition** — and handed it to the verification suite as the
result.

That is worse than failing. Measured on a real 4-rank `icoFoam` run that diverged to
a Courant number of 1580 with a cumulative continuity error of 2.65e+27, the
field-based checks reported *"fields within plausible bounds"*, because the initial
condition is perfectly plausible. The run was graded D only because the log-based
checks caught it. In the silent-error class — a run that completes and converges but
is physically wrong, which is half of what this product claims to find — the log is
clean by definition and these checks are the only thing left. They were reading t=0.

So two rules here, in order:

1. **A decomposed case is read from its `processor*/` directories**, concatenated.
   For the checks that use fields — bounds, plausibility, y+, a velocity scale — the
   union of the subdomains is the domain, and concatenation is exactly right.
2. **If the real result cannot be read, return no fields at all.** Never fall back to
   `0/`. A check that skips is honest; a check that passes on the initial condition
   is a false "physics-OK", and that is the one invariant this product says is
   structurally zero.

THE BOUNDARY FLUX (TRD VER-18), read only when asked (`fluxes=True`, which only
`courant verify --checks next` asks, GATE-13). Until 20 Oct 2026 nothing filled
`FieldData.patch_fluxes`, so `MassFluxCheck` skipped on every case, and a compressible nozzle
run whose boundary mass flux was out of balance was accepted, grade A (one rebuilt run, an
internal evaluation of 20 Oct 2026, not a measurement). It is read from the latest time's
`phi`, the face flux the solver itself computed and wrote, never rebuilt from `U`: rebuilding
it would judge an interpolation the solver never made. Per patch, signed as OpenFOAM signs it
(out of the domain positive), summed over the faces that bound the domain:

  * `empty` patches carry no flux (the direction a 2-D or 1-D case does not solve);
  * coupled patches (`cyclic`, `cyclicAMI`, `cyclicACMI`, `processor`, ... any type whose name
    holds `cyclic` or `processor`) are interior faces split in two: what leaves one side enters
    the other, so neither side is the boundary, and counting their inflow would inflate the
    balance's denominator;
  * `symmetry`, `symmetryPlane` and `wedge` patches are counted. Their faces bound the
    computational domain, the solver's own continuity sums them, and their flux is zero, or
    cancels between a wedge's two sides, when the run is right.

A compressible solver's `phi` is a mass flux (kg/s), an incompressible one's a volume flux
(m^3/s); the check judges net over inflow, so the units cancel, and the kind is kept only to
say which balance holds when (`MassFluxCheck`). An `overset` patch, a moving mesh, a decomposed
run, a binary or unreadable `phi`: not read, and `flux_note` says why. So is a case whose own
`fvModels` or `fvOptions` declares a source not known to act on momentum, energy or turbulence
alone (`_NO_CONTINUITY_SOURCE`): a mass or volume source, or phase change (from OpenFOAM.org 9
interFoam cavitates through the `VoFCavitation` fvModel), puts flux into the domain that its
boundary does not carry. The file is noted for the manifest once read, parsed or not (VER-14).
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

from courant.case.reads import may_read, reading
from courant.verification.fields import FieldData, FluxSource

#: Scalar fields the field-based checks look for.
_SCALARS = ("p", "k", "epsilon", "omega", "T", "nut")

#: `phi`'s dimensions as OpenFOAM writes them, (mass length time temperature moles current
#: luminous intensity), and the flux each one is.
_FLUX_KINDS = {(1, 0, -1, 0, 0, 0, 0): "mass", (0, 3, -1, 0, 0, 0, 0): "volume"}
#: A patch type holding one of these names couples two sets of interior faces (VER-18).
_COUPLED = ("cyclic", "processor")
#: Where a case declares its own sources, as the solver reads them: OpenFOAM.org's
#: `constant/fvModels` (9 on), and `fvOptions` in `constant/` or `system/` (ESI; OpenFOAM.org to
#: 8). Each is read plain or `.gz`. `fvConstraints` is not read.
_SOURCE_FILES = (("constant", "fvModels"), ("constant", "fvOptions"), ("system", "fvOptions"))
#: fvModels and fvOptions types that act on momentum, energy or turbulence alone, in either
#: fork's spelling, so leave the continuity the solver keeps as it was (VER-18). Any other type
#: may not: a mass or volume source (`massSource`, `volumeSource`), phase change
#: (`VoFCavitation`), clouds and films, and a semi-implicit or coded source, which may act on
#: any field, continuity's included. With one declared, the balance is not judged.
_NO_CONTINUITY_SOURCE = frozenset({
    "explicitPorositySource", "interRegionExplicitPorositySource",
    "actuationDiskSource", "radialActuationDiskSource", "rotorDisk", "rotorDiskSource",
    "meanVelocityForce", "pressureGradientExplicitSource",
    "directionalPressureGradientExplicitSource", "accelerationSource",
    "tabulatedAccelerationSource", "buoyancyForce", "buoyancyEnergy", "buoyancyTurbSource",
    "radiation", "heatTransfer", "interRegionHeatTransfer", "heatExchangerSource",
    "effectivenessHeatExchangerSource", "viscousDissipation", "jouleHeatingSource",
    "solidificationMeltingSource", "VoFSolidificationMeltingSource", "VoFTurbulenceDamping",
    "turbulenceDamping", "limitTemperature", "limitVelocity", "fixedTemperatureConstraint",
    "velocityDampingConstraint",
})


def _numeric_time_dirs(root: str) -> list[str]:
    """Numeric time directory names directly under `root`, ascending."""
    out: list[str] = []
    if not may_read(root) or not os.path.isdir(root):   # MCP-3: listed, so confined like a read
        return out
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        try:
            float(name)
        except ValueError:
            continue
        out.append(name)
    return sorted(out, key=float)


def processor_dirs(case_dir: str) -> list[str]:
    """Where a decomposed run put its results, in rank order.

    Two layouts, because OpenFOAM has two file handlers:

      * `uncollated` (the default) — `processor0/`, `processor1/`, … one per rank.
      * `collated` — a single `processors4/` (or `processors4_0-3/` for grouped
        writing) holding one file per field with every rank's data inside it.
        Increasingly the choice on large runs, because it stops a 4096-rank job
        from creating 4096 directories.

    Both are returned. A collated directory is a *directory of time
    directories* just like a rank directory, so the caller's logic is unchanged;
    whether the field files inside it can be parsed is a separate question, and
    one that fails safely — an unreadable field is simply absent.
    """
    ranked: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(case_dir, "processor[0-9]*")):
        if not os.path.isdir(path):
            continue
        suffix = os.path.basename(path)[len("processor"):]
        if suffix.isdigit():
            ranked.append((int(suffix), path))
    if ranked:
        return [path for _, path in sorted(ranked)]

    collated = [p for p in glob.glob(os.path.join(case_dir, "processors[0-9]*"))
                if os.path.isdir(p)]
    return sorted(collated)


def _field_file(case: str, time: str, name: str) -> str:
    """The file a field is read from at `time`: the plain file, else its `.gz`."""
    path = os.path.join(case, time, name)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return path + ".gz"
    return path


def _binary_as_read(arch) -> None:
    """Raise unless a binary field's bytes are what foamlib reads: little-endian 64-bit
    scalars, on a little-endian host. foamlib ignores the header's `arch`, reads the host's byte
    order and guesses the scalar size from the byte count, so a big-endian or single-precision
    field would come back as other numbers. Such a field is left unread, never guessed. No
    `arch` is OpenFOAM's native default (LSB, 64-bit scalars on every platform it ships for).
    The label size does not matter here: a field's counts are written in ASCII."""
    if sys.byteorder != "little":
        raise ValueError(f"binary field on a {sys.byteorder}-endian host")
    if arch is None:
        return
    parts = [p.strip() for p in str(arch).strip().strip('"').split(";") if p.strip()]
    order = [p for p in parts if "=" not in p]
    sizes = dict(p.split("=", 1) for p in parts if "=" in p)
    if order != ["LSB"] or sizes.get("scalar", "64") != "64" or not set(sizes) <= {"label", "scalar"}:
        raise ValueError(f"binary field written as {arch!r}; read only as LSB;scalar=64")


def _internal(path: str, vector: bool) -> np.ndarray | None:
    """The internal field of one field file, as per-cell values: shape (N,) for a scalar,
    (N, 3) for a vector, N = 1 for a `uniform` value. None when the file holds no field of that
    shape (a collated `decomposedBlockData`, a tensor, a region directory's stray file)."""
    from foamlib import FoamFieldFile

    f = FoamFieldFile(path)
    if str(f.format) == "binary":
        _binary_as_read(f["FoamFile"].get("arch"))
    values = np.asarray(f.internal_field, dtype=float)
    if vector:
        if values.shape == (3,):
            values = values.reshape(1, 3)
        return values if values.ndim == 2 and values.shape[1] == 3 else None
    values = np.atleast_1d(values)
    return values if values.ndim == 1 else None


#: A field file that is not there: a missing field. Any other error opening one (no permission,
#: an I/O error, a path outside a confinement) is a file the run wrote that was not read.
_ABSENT = (FileNotFoundError, IsADirectoryError, NotADirectoryError)


def _read_one(case: str, time: str, failed: list[str] | None = None) -> dict[str, np.ndarray]:
    """Every readable field in one case-shaped directory at one time. A field whose file exists
    and was not read (cut mid-write, a collated block, a binary layout not read, a file that
    could not be opened) is appended to `failed` (TRD VER-7b): the run wrote it, and "no fields
    were found" would be false."""
    out: dict[str, np.ndarray] = {}
    # VER-14: `reading` notes a field file once it has been read, parsed or not, and leaves out
    # one that could not be opened (absent, unreadable, purged), which is not listed.
    for name, vector in (*((s, False) for s in _SCALARS), ("U", True)):
        path = _field_file(case, time, name)
        try:
            with reading(path):
                values = _internal(path, vector)
        except _ABSENT:          # not there: absent or purged, a missing field
            continue
        except Exception:  # noqa: BLE001 — there and not read: not opened, or not a field
            if failed is not None and name not in failed:
                failed.append(name)
            continue
        if values is not None:
            out[name] = values
        elif failed is not None and name not in failed:     # read: no field of that shape
            failed.append(name)                               # (a collated block)
    return out


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Union the per-rank subdomains into one field set."""
    merged: dict[str, np.ndarray] = {}
    for name in set().union(*(p.keys() for p in parts)) if parts else ():
        arrays = [p[name] for p in parts if name in p and np.size(p[name])]
        if not arrays:
            continue
        try:
            merged[name] = np.concatenate(arrays, axis=0)
        except ValueError:  # ragged ranks — a partial write; drop the field
            continue
    return merged


class _NotRead(ValueError):
    """The boundary flux cannot be summed from what was read; the message says why."""


def _moving_mesh(case: str) -> str | None:
    """Why the mesh may move, from `constant/dynamicMeshDict`, or None for a fixed mesh. On a
    moving mesh the solver may write its flux relative to the mesh's motion, and the net flux
    through a moving boundary then balances the change of the domain's volume, which is not
    read, so it is not judged. ESI names the mesh type in `dynamicFvMesh` (`staticFvMesh` is a
    fixed mesh); OpenFOAM.org in `mover`, `topoChanger` and `refiner` sub-dictionaries. The one
    reading of it is `courant.case.motion.read_mesh_motion`, the case reader's too."""
    from courant.case.motion import read_mesh_motion

    motion = read_mesh_motion(case)
    if motion is None:
        return None
    if motion.by == "unreadable":
        return "constant/dynamicMeshDict could not be read, so whether the mesh moves is unknown"
    if motion.by == "dynamicFvMesh":
        return f"the mesh is dynamic (constant/dynamicMeshDict: dynamicFvMesh {motion.kind})"
    return f"the mesh is dynamic (constant/dynamicMeshDict: {motion.by})"


def _flux_sources(case: str) -> str | None:
    """Why the case's own sources may put flux into the domain, from its `fvModels` and
    `fvOptions` (`_SOURCE_FILES`), or None when every source it declares acts on momentum,
    energy or turbulence alone (`_NO_CONTINUITY_SOURCE`). With a mass, volume or phase-change
    source the boundary need not balance, steady or not, so it is not judged. A file that
    cannot be read is not guessed at. Each entry counts as declared: whether a fork honours an
    `active` switch is not read here."""
    from collections.abc import Mapping

    from foamlib import FoamFile

    for where, name in _SOURCE_FILES:
        path = _field_file(case, where, name)
        if not os.path.isfile(path):
            continue
        shown = os.path.relpath(path, case).replace(os.sep, "/")
        try:
            with reading(path):
                d = FoamFile(path)
                declared = [(str(k), v.get("type")) for k, v in d.items()
                            if isinstance(v, Mapping)]      # `libs (...)` is not an entry
        except Exception:                                   # noqa: BLE001
            return (f"{shown} could not be read, so whether it declares a mass, volume or "
                    f"phase-change source is unknown")
        for key, kind in declared:
            if kind is None or str(kind) not in _NO_CONTINUITY_SOURCE:
                of = f"of type {kind}" if kind is not None else "with no type"
                return (f"{shown} declares {key}, {of}, which is not known to act on momentum, "
                        f"energy or turbulence alone: a mass, volume or phase-change source "
                        f"puts flux into the domain that its boundary does not carry, so the "
                        f"balance is not judged")
    return None


def _face_counts(case: str, time: str, shown: str) -> dict[str, int]:
    """Each patch's face count, from the mesh at `time` (`<time>/polyMesh/boundary`) or else
    `constant/polyMesh/boundary`: what a `uniform` value is multiplied by."""
    from courant.case.polymesh import read_boundary

    rel = f"{time}/polyMesh/boundary"
    if not os.path.isfile(os.path.join(case, rel)):
        rel = "constant/polyMesh/boundary"
    try:
        return {p.name: p.n for p in read_boundary(os.path.join(case, rel))}
    except Exception as exc:                            # noqa: BLE001
        raise _NotRead(f"{shown} holds a uniform boundary value and {rel}, which says how many "
                       f"faces it covers, could not be read") from exc


def _boundary_flux(case: str, time: str) -> tuple[dict[str, float], FluxSource]:
    """The signed flux through each patch that bounds the domain, from `<time>/phi` (VER-18).
    Raises `_NotRead` with the reason when it cannot be summed as written."""
    from foamlib import FoamFieldFile

    path = _field_file(case, time, "phi")
    shown = os.path.relpath(path, case).replace(os.sep, "/")
    if not os.path.isfile(path):
        raise _NotRead(f"no {time}/phi (the face flux at the latest time) was found")
    try:
        with reading(path):
            f = FoamFieldFile(path)
            if str(f.format) != "ascii":
                raise _NotRead(f"{shown} is written {f.format}; only an ASCII phi is read")
            cls = str(f["FoamFile"].get("class", ""))
            if cls != "surfaceScalarField":
                raise _NotRead(f"{shown} is a {cls or 'file of no class'}, not a face flux")
            try:
                dims = tuple(float(x) for x in f.dimensions)
            except (TypeError, ValueError):
                dims = ()
            kind = _FLUX_KINDS.get(dims)
            if kind is None:
                raise _NotRead(f"{shown} has dimensions {list(dims) or 'none'}: neither a mass "
                               f"flux nor a volume flux")
            internal = np.abs(np.atleast_1d(np.asarray(f.internal_field, dtype=float)))
            patches = {str(k): dict(v) for k, v in f.boundary_field.items()}
    except _NotRead:
        raise
    except Exception as exc:                            # noqa: BLE001
        raise _NotRead(f"{shown} could not be read") from exc

    net: dict[str, float] = {}
    inflow = outflow = 0.0
    largest = float(internal.max(initial=0.0))
    counts: dict[str, int] | None = None
    for name, patch in patches.items():
        ptype = str(patch.get("type", ""))
        if ptype == "empty" or any(c in ptype.lower() for c in _COUPLED):
            continue
        if ptype == "overset":
            raise _NotRead(f"patch {name} of {shown} is overset: flux passes between the meshes "
                           f"through interpolated cells, so the boundary is not a balance")
        raw = patch.get("value")
        try:
            if raw is None:                             # asarray(None) would be a nan
                raise ValueError("no value")
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise _NotRead(f"patch {name} of {shown} holds no flux values") from exc
        if values.ndim == 0:                            # `uniform v`: v on each of n faces
            v, n = float(values), 0
            if v != 0.0:
                counts = counts if counts is not None else _face_counts(case, time, shown)
                n = counts.get(name, -1)
                if n < 0:
                    raise _NotRead(f"patch {name} of {shown} has no face count in the mesh's "
                                   f"boundary file")
            # Arithmetic, not `np.full(n, v)`: a boundary file's count is text, and a count of
            # 10**12 must not become an allocation.
            total = v * n
            net[name] = total
            inflow += max(-total, 0.0)
            outflow += max(total, 0.0)
            largest = max(largest, abs(v) if n else 0.0)
            continue
        if values.ndim != 1:
            raise _NotRead(f"patch {name} of {shown} does not hold one flux per face")
        net[name] = float(values.sum())
        inflow += float(-values[values < 0].sum())
        outflow += float(values[values > 0].sum())
        largest = max(largest, float(np.abs(values).max(initial=0.0)))
    if not net:
        raise _NotRead(f"every patch of {shown} is empty or coupled: no face of it bounds the "
                       f"domain")
    return net, FluxSource(file=shown, kind=kind, inflow=inflow, outflow=outflow, largest=largest)


def after(time: str | float, last: float) -> bool:
    """Whether a time directory is later than the log's last time step (TRD VER-7b): later, and
    not the same time by the one rule conformance compares times with
    (`courant.spec.as_run._same_time`)."""
    from courant.spec.as_run import _same_time

    t = float(time)
    return t > last and not _same_time(t, last)


def _ranks(procs: list[str], time: str) -> tuple[dict[str, np.ndarray], tuple[str, ...],
                                                 tuple[tuple[str, int, int], ...]]:
    """The fields of every rank at `time`, unioned (`_concat`), with what was not read whole:
    (the fields, the names no rank read and one or more could not, (name, ranks read, ranks)
    for a field some ranks read and others did not). The checks run on the ranks read, so a
    sentence that says they were not run would be false (TRD VER-7b)."""
    parts: list[dict[str, np.ndarray]] = []
    failed: set[str] = set()
    for p in procs:
        failed_here: list[str] = []
        parts.append(_read_one(p, time, failed_here))
        failed.update(failed_here)
    merged = _concat(parts)
    on = {n: sum(1 for part in parts if n in part and np.size(part[n])) for n in merged}
    partial = tuple((n, on[n], len(parts)) for n in sorted(merged) if on[n] < len(parts))
    unread = tuple(n for n in (*_SCALARS, "U") if n in failed and n not in merged)
    return merged, unread, partial


def read_fields(case_dir: str, time: str | None = None, *, fluxes: bool = False,
                until: float | None = None) -> FieldData:
    """The solution at the latest time (or `time`). With `until`, the log's last time step (TRD
    VER-7b; only `courant verify --checks next` passes it, GATE-13): a time directory later than
    it is not the run the log records, so it is not read (`after_log` names it), and the ranks
    are read when they went further than a root reconstructed partway (`reconstructPar -time`).
    Without it the choice is v0.5.0's, and `later_ranks` says when the ranks went further.

    With `fluxes`, also the boundary flux from the `phi` of the time read (TRD VER-18): only
    `courant verify --checks next` asks for it (GATE-13), so under the default `phi` is not
    opened and the manifest is v0.5.0's."""
    procs = processor_dirs(case_dir)
    root_times = _numeric_time_dirs(case_dir)
    # Rank 0's clock: every rank writes the same times.
    proc_times = _numeric_time_dirs(procs[0]) if procs else []
    after_log: tuple[str, ...] = ()
    if until is not None:
        after_log = tuple(sorted({t for t in (*root_times, *proc_times) if after(t, until)},
                                 key=float))
        root_times = [t for t in root_times if not after(t, until)]
        proc_times = [t for t in proc_times if not after(t, until)]
    solved_at_root = [t for t in root_times if float(t) > 0.0]
    ranks_ahead = bool(solved_at_root and proc_times
                       and float(proc_times[-1]) > float(solved_at_root[-1]))

    internal: dict[str, np.ndarray] = {}
    unread: list[str] | tuple[str, ...] = []
    partial: tuple[tuple[str, int, int], ...] = ()
    latest = time
    later_ranks = None
    from_ranks = False

    if (solved_at_root or not procs) and not (until is not None and ranks_ahead):
        # Serial, or a parallel case the engineer has already reconstructed.
        latest = time or (root_times[-1] if root_times else None)
        if latest is not None and (solved_at_root or not procs):
            internal = _read_one(case_dir, latest, unread)
        if ranks_ahead and time is None:
            later_ranks = proc_times[-1]
    else:
        # Decomposed and not reconstructed (or, with `until`, reconstructed short of where the
        # ranks went): the answer is in the ranks.
        from_ranks = True
        latest = time or (proc_times[-1] if proc_times else None)
        if latest is not None and float(latest) > 0.0:
            internal, unread, partial = _ranks(procs, latest)

    # Rule 2. Reading `0/` and calling it the result is how a diverged run gets
    # told its fields are plausible.
    if latest is None or float(latest) <= 0.0:
        return FieldData(time=0.0, internal={}, nu=None, inlet_speed=None, after_log=after_log)

    nu = None
    try:
        from foamlib import FoamFile

        from courant.case._foam_reader import _scalar

        transport = os.path.join(case_dir, "constant/transportProperties")
        with reading(transport):                # VER-14: read for nu, whether or not it has one
            nu = _scalar(FoamFile(transport)["nu"])
    except Exception:  # noqa: BLE001
        pass

    inlet_speed = None
    if "U" in internal:
        u = internal["U"]
        mag = np.linalg.norm(u, axis=1) if u.ndim == 2 else np.abs(u)
        if mag.size:
            inlet_speed = float(np.percentile(mag, 95))

    patch_fluxes: dict[str, float] = {}
    source = note = None
    if fluxes:
        if from_ranks:                  # the fields were read from processor*/
            done = ("was not reconstructed" if not solved_at_root
                    else "was reconstructed only partway")
            note = (f"the run is decomposed and {done}: its face flux is split "
                    "across processor*/, and is not read")
        else:
            note = _moving_mesh(case_dir) or _flux_sources(case_dir)
            if note is None:
                try:
                    patch_fluxes, source = _boundary_flux(case_dir, latest)
                except _NotRead as exc:
                    note = str(exc)

    try:
        t = float(latest)
    except ValueError:
        t = 0.0
    return FieldData(time=t, internal=internal, nu=nu, inlet_speed=inlet_speed,
                     patch_fluxes=patch_fluxes, flux_source=source, flux_note=note,
                     source_time=latest, unread=tuple(unread), partial=partial,
                     later_ranks=later_ranks, after_log=after_log)
