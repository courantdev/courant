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
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

from courant.case.reads import reading
from courant.verification.fields import FieldData

#: Scalar fields the field-based checks look for.
_SCALARS = ("p", "k", "epsilon", "omega", "T", "nut")


def _numeric_time_dirs(root: str) -> list[str]:
    """Numeric time directory names directly under `root`, ascending."""
    out: list[str] = []
    if not os.path.isdir(root):
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


def _read_one(case: str, time: str) -> dict[str, np.ndarray]:
    """Every readable field in one case-shaped directory at one time."""
    out: dict[str, np.ndarray] = {}
    # VER-14: `reading` notes a field file once it has been read, parsed or not, and leaves out
    # one that could not be opened (absent, unreadable, purged), which is simply a missing field.
    for name, vector in (*((s, False) for s in _SCALARS), ("U", True)):
        path = _field_file(case, time, name)
        try:
            with reading(path):
                values = _internal(path, vector)
        except Exception:  # noqa: BLE001 — field not present, or not parseable, in this case
            continue
        if values is not None:
            out[name] = values
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


def read_fields(case_dir: str, time: str | None = None) -> FieldData:
    procs = processor_dirs(case_dir)
    root_times = _numeric_time_dirs(case_dir)
    solved_at_root = [t for t in root_times if float(t) > 0.0]

    internal: dict[str, np.ndarray] = {}
    latest = time

    if solved_at_root or not procs:
        # Serial, or a parallel case the engineer has already reconstructed.
        latest = time or (root_times[-1] if root_times else None)
        if latest is not None and (solved_at_root or not procs):
            internal = _read_one(case_dir, latest)
    else:
        # Decomposed and not reconstructed: the answer is in the ranks.
        # Use rank 0's clock — every rank writes the same times.
        proc_times = _numeric_time_dirs(procs[0])
        latest = time or (proc_times[-1] if proc_times else None)
        if latest is not None and float(latest) > 0.0:
            internal = _concat([_read_one(p, latest) for p in procs])

    # Rule 2. Reading `0/` and calling it the result is how a diverged run gets
    # told its fields are plausible.
    if latest is None or float(latest) <= 0.0:
        return FieldData(time=0.0, internal={}, nu=None, inlet_speed=None)

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

    try:
        t = float(latest)
    except ValueError:
        t = 0.0
    return FieldData(time=t, internal=internal, nu=nu, inlet_speed=inlet_speed)
