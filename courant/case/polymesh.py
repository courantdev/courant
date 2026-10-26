"""OpenFOAM's polyMesh: the ASCII reader, the writer and the geometry OpenFOAM builds from it.

One implementation (TRD FLD-1, FLD-2, VER-7a), public so the gate can use it: the gate's cell
aspect ratio (`courant.case.mesh.polymesh_quality`), the field checks (the private field
package re-exports `Mesh`, `Patch`, `read_polymesh` and `write_polymesh` from here) and the
re-solve's cell count (through `cell_count`).

The layout is OpenFOAM's: internal faces first, each owned by its lower-numbered cell, boundary
faces grouped by patch after them. Geometry follows OpenFOAM's own construction:
- face area vectors and centres from triangles fanned about the points' average
  (`primitiveMesh::makeFaceCentresAndAreas`); a face of zero area keeps that average as its
  centre and has zero area, as OpenFOAM's does;
- cell volumes and centres from pyramids on each face, apex at the average of the cell's face
  centres (`primitiveMesh::makeCellCentresAndVols`).

The reader takes the files as an agent or a user left them, so every step is linear in the
file: comments are stripped with `str.find`, lists are delimited by a vectorised bracket depth
and split on their brackets, and no regular expression that can backtrack meets the file body.
Binary meshes are refused with a clear message rather than half-read. Each file is noted for
the verdict's manifest once its bytes are read, parsed or not (VER-14, `reads.reading`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from courant.case.reads import reading, refuse_outside

#: OpenFOAM's `ROOTVSMALL` in double precision.
ROOTVSMALL = 1.0e-150
FILES = ("points", "faces", "owner", "neighbour", "boundary")
#: Faces per vectorised step of the face geometry.
_CHUNK = 65536


class UnsupportedFormat(ValueError):
    pass


# ------------------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class Patch:
    name: str
    type: str
    start: int
    n: int


def cell_count(owner: np.ndarray, neighbour: np.ndarray, noted: int | None = None) -> int:
    """The number of cells as OpenFOAM derives it (`polyMesh::initMesh`): the highest cell label
    in the owner and neighbour lists, plus one. The `nCells` note in a header is text anyone can
    write, so it is never the count; a note that disagrees with the lists is refused."""
    top = int(max(np.max(owner, initial=-1), np.max(neighbour, initial=-1))) + 1
    if noted is not None and noted != top:
        raise ValueError(f"the owner header notes {noted} cells and the lists hold {top}")
    return top


@dataclass(frozen=True)
class Mesh:
    points: np.ndarray                  # (P, 3)
    faces: tuple[np.ndarray, ...]       # F faces, each an array of point indices
    owner: np.ndarray                   # (F,)
    neighbour: np.ndarray               # (F_internal,)
    boundary: tuple[Patch, ...] = field(default_factory=tuple)
    n_cells: int = 0

    def __post_init__(self):
        n = cell_count(self.owner, self.neighbour)
        if not self.n_cells:
            object.__setattr__(self, "n_cells", n)
        if len(self.owner) != len(self.faces):
            raise ValueError("owner must have one entry per face")
        if len(self.neighbour) > len(self.faces):
            raise ValueError("more neighbours than faces")

    @property
    def n_internal(self) -> int:
        return len(self.neighbour)

    @cached_property
    def _face_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """(centres, areas), vectorised by face size."""
        faces = self.faces
        centres = np.empty((len(faces), 3))
        areas = np.empty((len(faces), 3))
        sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
        chunks = [(n, idx[k:k + _CHUNK]) for n in np.unique(sizes)
                  for idx in (np.nonzero(sizes == n)[0],) for k in range(0, len(idx), _CHUNK)]
        for n, idx in chunks:                     # in chunks, so the temporaries stay small
            p = self.points[np.stack([np.asarray(faces[i]) for i in idx])]       # (m, n, 3)
            if n == 3:
                areas[idx] = 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
                centres[idx] = p.mean(axis=1)
                continue
            est = p.mean(axis=1)
            nxt = np.roll(p, -1, axis=1)
            tri_n = np.cross(nxt - p, est[:, None] - p)      # 2 x triangle area vectors
            tri_c = (p + nxt + est[:, None]) / 3.0
            sum_n = tri_n.sum(axis=1)
            mag = np.linalg.norm(sum_n, axis=1)
            real = mag >= ROOTVSMALL
            unit = np.zeros_like(sum_n)
            unit[real] = sum_n[real] / mag[real, None]
            w = np.einsum("mnk,mk->mn", tri_n, unit)
            c = est.copy()
            c[real] = (w[real, :, None] * tri_c[real]).sum(axis=1) / w[real].sum(axis=1)[:, None]
            centres[idx] = c
            areas[idx] = np.where(real[:, None], 0.5 * sum_n, 0.0)
        return centres, areas

    @property
    def face_centres(self) -> np.ndarray:
        return self._face_geometry[0]

    @property
    def face_areas(self) -> np.ndarray:
        """Area vectors, pointing out of the owner cell."""
        return self._face_geometry[1]

    @cached_property
    def _cell_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        fc, fa = self.face_centres, self.face_areas
        n = self.n_cells
        est = np.zeros((n, 3))
        count = np.zeros(n)
        np.add.at(est, self.owner, fc)
        np.add.at(count, self.owner, 1)
        ni = self.n_internal
        np.add.at(est, self.neighbour, fc[:ni])
        np.add.at(count, self.neighbour, 1)
        est /= np.maximum(count, 1)[:, None]

        vol = np.zeros(n)
        cen = np.zeros((n, 3))
        # owner side: area vector points out of the cell
        pv = np.einsum("ij,ij->i", fa, fc - est[self.owner]) / 3.0
        np.add.at(vol, self.owner, pv)
        np.add.at(cen, self.owner, pv[:, None] * (0.75 * fc + 0.25 * est[self.owner]))
        # neighbour side: area vector points into the cell
        nv = np.einsum("ij,ij->i", fa[:ni], est[self.neighbour] - fc[:ni]) / 3.0
        np.add.at(vol, self.neighbour, nv)
        np.add.at(cen, self.neighbour, nv[:, None] * (0.75 * fc[:ni] + 0.25 * est[self.neighbour]))
        with np.errstate(invalid="ignore", divide="ignore"):
            cen = np.where(vol[:, None] > 0, cen / vol[:, None], est)
        return cen, vol

    @property
    def cell_centres(self) -> np.ndarray:
        return self._cell_geometry[0]

    @property
    def cell_volumes(self) -> np.ndarray:
        return self._cell_geometry[1]

    def patch(self, name: str) -> Patch:
        for p in self.boundary:
            if p.name == name:
                return p
        raise KeyError(name)

    def interpolation_weights(self) -> np.ndarray:
        """OpenFOAM linear weights for internal faces: the owner's share of the face value."""
        ni = self.n_internal
        fc, fa = self.face_centres[:ni], self.face_areas[:ni]
        cp = self.cell_centres[self.owner[:ni]]
        cn = self.cell_centres[self.neighbour]
        dn = np.abs(np.einsum("ij,ij->i", cn - fc, fa))
        dp = np.abs(np.einsum("ij,ij->i", fc - cp, fa))
        total = dn + dp
        return np.where(total > 0, dn / np.where(total > 0, total, 1), 0.5)

    def face_skewness(self) -> np.ndarray:
        """checkMesh's skewness of every face, as v2406 computes it (TRD VER-7d).

        `polyMeshTools::faceSkewness` (src/OpenFOAM/meshes/polyMesh/polyMeshCheck/
        polyMeshTools.C), which `polyMesh::checkFaceSkewness` calls: an internal face by
        `primitiveMeshTools::faceSkewness`, a boundary face of a patch that is not coupled by
        `primitiveMeshTools::boundaryFaceSkewness` (src/OpenFOAM/meshes/primitiveMesh/
        primitiveMeshCheck/primitiveMeshTools.C). With Cpf the vector from the owner's centre to
        the face centre and d the one to the neighbour's centre (on a boundary face, n (n . Cpf)
        with n = normalised(A), which is zero for |A| < ROOTVSMALL, VectorSpaceI.H):

            sv = Cpf - ((A . Cpf) / ((A . d) + ROOTVSMALL)) d
            fd = max(c |d| + ROOTVSMALL, max over the face's points |sv/(|sv| + ROOTVSMALL) . (p - Cf)|)
            skewness = |sv| / fd,     c = 0.2 inside, 0.4 on a boundary

        On a face of zero area d is zero and its points have no extent across Cpf, so fd is
        the floor ROOTVSMALL and the skewness is |Cpf| / 1e-150: a distance over the guard
        against division by zero, not a measure of the face. A coupled patch's faces use the
        neighbour across the coupling, which this does not read: callers refuse such a mesh."""
        fc, fa, cc = self.face_centres, self.face_areas, self.cell_centres
        own, nei, ni = self.owner, self.neighbour, self.n_internal
        nf = len(self.faces)
        cpf = fc - cc[own]
        d = np.empty((nf, 3))
        d[:ni] = cc[nei] - cc[own[:ni]]
        mag = np.linalg.norm(fa[ni:], axis=1)
        normal = np.zeros((nf - ni, 3))
        real = mag >= ROOTVSMALL
        normal[real] = fa[ni:][real] / mag[real, None]
        d[ni:] = normal * np.einsum("ij,ij->i", normal, cpf[ni:])[:, None]
        coef = np.full(nf, 0.4)
        coef[:ni] = 0.2
        along = np.einsum("ij,ij->i", fa, cpf) / (np.einsum("ij,ij->i", fa, d) + ROOTVSMALL)
        sv = cpf - along[:, None] * d
        mag_sv = np.linalg.norm(sv, axis=1)
        sv_hat = sv / (mag_sv + ROOTVSMALL)[:, None]
        fd = coef * np.linalg.norm(d, axis=1) + ROOTVSMALL
        sizes = np.fromiter((len(f) for f in self.faces), dtype=np.int64, count=nf)
        for n in np.unique(sizes):
            idx = np.nonzero(sizes == n)[0]
            for k in range(0, len(idx), _CHUNK):      # in chunks, as `_face_geometry` does
                part = idx[k:k + _CHUNK]
                p = self.points[np.stack([np.asarray(self.faces[i]) for i in part])]
                reach = np.abs(np.einsum("mk,mpk->mp", sv_hat[part], p - fc[part][:, None]))
                fd[part] = np.maximum(fd[part], reach.max(axis=1))
        return mag_sv / fd


# ------------------------------------------------------------------------------ reading
def strip_comments(text: str) -> str:
    """`//` and `/* */` comments removed, in one forward pass (no regular expression)."""
    out, i, n = [], 0, len(text)
    a = b = -2                       # the next `//` and `/*` at or after i; -1 when there is none
    while i < n:
        # Each search runs again only once i has passed its last answer, so the whole pass is
        # linear: re-searching from every i is quadratic on "/*/*/*..." with no `//` in it.
        if a != -1 and a < i:
            a = text.find("//", i)
        if b != -1 and b < i:
            b = text.find("/*", i)
        j = min(x for x in (a, b, n) if x >= 0)
        out.append(text[i:j])
        if j == n:
            break
        if j == a:
            k = text.find("\n", j)
            i = n if k < 0 else k
        else:
            k = text.find("*/", j + 2)
            i = n if k < 0 else k + 2
    return "".join(out)


_HEADER_ENTRY = re.compile(r"(\w{1,64})[ \t]+([^;\n]{0,256});")


def _body(path: Path) -> tuple[str, dict]:
    """(the text after the FoamFile header, the header's entries). Comments removed; binary
    refused."""
    text = strip_comments(path.read_text(errors="replace"))
    header: dict = {}
    k = text.find("FoamFile")
    if k >= 0:
        a, b = text.find("{", k), text.find("}", k)
        if 0 <= a < b:
            header = {key: val.strip().strip('"')
                      for key, val in _HEADER_ENTRY.findall(text[a + 1:b])}
            text = text[b + 1:]
    if header.get("format", "ascii") != "ascii":
        raise UnsupportedFormat(f"{path.name} is {header.get('format')}: convert it with "
                                "`foamFormatConvert` (binary polyMesh is not read in this version)")
    return text, header


def _outer_list(text: str, start: int = 0) -> tuple[int, str, int]:
    """(the count written before it, the contents, the index after it) of the first top-level
    `N ( ... )` list at or after `start`. Raises on a list that is not closed."""
    a = text.find("(", start)
    if a < 0:
        raise ValueError("no list")
    count = text[start:a].split()
    if not count or not count[-1].isdigit():
        raise ValueError("a list without its size")
    tail = np.frombuffer(text[a:].encode("ascii", errors="replace"), dtype=np.uint8)
    at = np.flatnonzero((tail == 40) | (tail == 41))         # every bracket, nothing else
    depth = np.cumsum(np.where(tail[at] == 40, 1, -1))
    closed = np.flatnonzero(depth == 0)
    if not closed.size:
        raise ValueError("unbalanced parentheses in polyMesh list")
    end = a + int(at[closed[0]])
    return int(count[-1]), text[a + 1:end], end + 1


def _labels(text: str) -> np.ndarray:
    n, inner, _ = _outer_list(text)
    out, inside, _ = _numbers(inner, np.int64)
    if inside.any() or len(out) != n:
        raise ValueError(f"a list of {n} holds {len(out)} labels")
    return out


def _read(path: Path, parse):
    with reading(path):
        return parse(path)


_BLANK = str.maketrans("()", "  ")


def _numbers(inner: str, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(every number of a list's contents in order, whether each lies inside an item's brackets,
    how many numbers each item holds), once the brackets are checked to open and close in turn
    (no nesting, nothing left open). Vectorised: no Python object per number or per character,
    so the cost is linear and the memory a few arrays (a token list took gigabytes at the
    cost bound)."""
    import warnings

    raw = np.frombuffer(inner.encode("ascii", errors="replace"), dtype=np.uint8)
    opens, closes = np.flatnonzero(raw == 40), np.flatnonzero(raw == 41)
    if len(opens) != len(closes) or (len(opens) and (
            (opens >= closes).any() or (opens[1:] <= closes[:-1]).any())):
        raise ValueError("unbalanced or nested items")
    word = ~np.isin(raw, np.frombuffer(b" \t\r\n\f\v()", dtype=np.uint8))
    starts = np.flatnonzero(word & ~np.concatenate(([False], word[:-1])))
    if not len(starts):                                 # an empty list (no internal faces)
        return np.zeros(0, dtype=dtype), np.zeros(0, dtype=bool), np.zeros(len(opens), int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # a token numpy cannot read raises
        values = np.fromstring(inner.translate(_BLANK), dtype=dtype, sep=" ")
    if len(values) != len(starts):
        raise ValueError("a token that is not a number")
    item = np.searchsorted(opens, starts, side="right") - 1
    inside = (item >= 0) & (starts < closes[np.maximum(item, 0)]) if len(opens) else \
        np.zeros(len(starts), dtype=bool)
    counts = np.bincount(item[inside], minlength=len(opens))
    return values, inside, counts


def _points(path: Path, max_n: int | None = None) -> np.ndarray:
    n, inner, _ = _outer_list(_body(path)[0])
    if max_n is not None and n > max_n:
        raise ValueError(f"points: {n} points, above the bound of {max_n}")
    if inner.count("(") != n:                           # before parsing: the count is cheap
        raise ValueError(f"points: a list of {n} holds {inner.count('(')}")
    values, inside, counts = _numbers(inner, float)
    if len(counts) != n or not inside.all() or (counts != 3).any():
        raise ValueError("points: the list does not hold its count of 3-vectors")
    return values.reshape(n, 3)


def _faces(path: Path, max_faces: int | None = None) -> tuple[np.ndarray, ...]:
    text, header = _body(path)
    if header.get("class") == "faceCompactList":
        offsets_n, offsets_text, after = _outer_list(text)
        if max_faces is not None and offsets_n - 1 > max_faces:
            raise ValueError(f"faces: {offsets_n - 1} faces, above the bound of {max_faces}")
        flat_n, flat_text, _ = _outer_list(text, after)
        offsets, i1, _ = _numbers(offsets_text, np.int64)
        flat, i2, _ = _numbers(flat_text, np.int64)
        if (i1.any() or i2.any() or len(offsets) != offsets_n or len(flat) != flat_n
                or len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != len(flat)
                or (np.diff(offsets) < 3).any()):
            raise ValueError("faces: the compact lists disagree")
        return tuple(np.split(flat, offsets[1:-1]))
    n, inner, _ = _outer_list(text)
    if max_faces is not None and n > max_faces:
        raise ValueError(f"faces: {n} faces, above the bound of {max_faces}")
    if inner.count("(") != n:                           # before parsing: the count is cheap
        raise ValueError(f"faces: a list of {n} holds {inner.count('(')}")
    values, inside, counts = _numbers(inner, np.int64)  # size(labels) size(labels) ...
    sizes = values[~inside]
    if len(counts) != n or len(sizes) != n or (sizes != counts).any() or (counts < 3).any():
        raise ValueError("faces: an item's size does not match its labels")
    return tuple(np.split(values[inside], np.cumsum(counts)[:-1]))


def _boundary(path: Path) -> tuple[Patch, ...]:
    _, inner, _ = _outer_list(_body(path)[0])
    out = []
    for chunk in inner.split("}")[:-1]:
        name, sep, block = chunk.partition("{")
        if not sep or not name.split():
            raise ValueError("boundary: a patch without a name")
        kv = dict(_HEADER_ENTRY.findall(block))
        out.append(Patch(name=name.split()[-1], type=kv.get("type", "patch").strip(),
                         start=int(kv["startFace"]), n=int(kv["nFaces"])))
    return tuple(out)


def read_boundary(path: str | Path) -> tuple[Patch, ...]:
    """One `polyMesh/boundary` file's patches, in order (the boundary flux's face counts, TRD
    VER-18). Noted for the manifest once its bytes are read (VER-14)."""
    return _read(Path(path), _boundary)


def noted_cells(path: Path) -> int | None:
    """The `nCells` an owner file's header notes, from its first 4 KiB, or None."""
    refuse_outside(path)                # MCP-3
    with open(path, "rb") as fh:
        m = re.search(rb"nCells:[ \t]*(\d{1,12})", fh.read(4096))
    return int(m.group(1)) if m else None


def read_polymesh(case_dir: str | Path, *, max_faces: int | None = None) -> Mesh:
    """`constant/polyMesh` of a case. Raises `UnsupportedFormat` on a binary file and
    `ValueError` (or `OSError`) on anything it cannot read whole, or on more than `max_faces`
    faces (checked from the faces list's own count, before it is parsed)."""
    d = Path(case_dir) / "constant" / "polyMesh"
    # A real mesh has fewer points than faces (a hex mesh about a third as many); twice the faces
    # bound leaves room for any of them and still bounds the parse.
    points = _read(d / "points", lambda p: _points(p, 2 * max_faces if max_faces else None))
    faces = _read(d / "faces", lambda p: _faces(p, max_faces))
    owner = _read(d / "owner", lambda p: _labels(_body(p)[0]))
    neighbour = _read(d / "neighbour", lambda p: _labels(_body(p)[0]))
    boundary = _read(d / "boundary", _boundary)
    n_cells = cell_count(owner, neighbour, noted_cells(d / "owner"))
    return Mesh(points=points, faces=faces, owner=owner, neighbour=neighbour,
                boundary=boundary, n_cells=n_cells)


# ------------------------------------------------------------------------------ writing
def _header(cls: str, obj: str, note: str = "") -> str:
    n = f'    note        "{note}";\n' if note else ""
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n{n}    location    \"constant/polyMesh\";\n"
            f"    object      {obj};\n}}\n\n")


def write_polymesh(mesh: Mesh, case_dir: str | Path) -> Path:
    """ASCII writer, used to make fixtures. Never pointed at a user's case."""
    d = Path(case_dir) / "constant" / "polyMesh"
    d.mkdir(parents=True, exist_ok=True)
    (d / "points").write_text(_header("vectorField", "points") + f"{len(mesh.points)}\n(\n"
                              + "".join(f"({float(x)!r} {float(y)!r} {float(z)!r})\n"
                                        for x, y, z in mesh.points) + ")\n")
    (d / "faces").write_text(_header("faceList", "faces") + f"{len(mesh.faces)}\n(\n"
                             + "".join(f"{len(f)}({' '.join(str(int(v)) for v in f)})\n"
                                       for f in mesh.faces) + ")\n")
    note = (f"nPoints:{len(mesh.points)}  nCells:{mesh.n_cells}  nFaces:{len(mesh.faces)}  "
            f"nInternalFaces:{mesh.n_internal}")
    (d / "owner").write_text(_header("labelList", "owner", note) + f"{len(mesh.owner)}\n(\n"
                             + "\n".join(str(int(v)) for v in mesh.owner) + "\n)\n")
    (d / "neighbour").write_text(_header("labelList", "neighbour", note)
                                 + f"{len(mesh.neighbour)}\n(\n"
                                 + "\n".join(str(int(v)) for v in mesh.neighbour) + "\n)\n")
    (d / "boundary").write_text(
        _header("polyBoundaryMesh", "boundary") + f"{len(mesh.boundary)}\n(\n"
        + "".join(f"    {p.name}\n    {{\n        type            {p.type};\n"
                  f"        nFaces          {p.n};\n        startFace       {p.start};\n    }}\n"
                  for p in mesh.boundary) + ")\n")
    return d
