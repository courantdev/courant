"""The heat a steady flow carries through a region's openings (TRD VER-22 (c), T-ENERGY-FLOW).

Read only under `courant verify --checks next` (`conservation.energy.flow`, GATE-13), by
`courant.verification._heat_reader` for a region open to flow; the region is then judged on the
heat through its walls (its `wallHeatFlux` rows) plus what `read_openings` returns.

**What the solver balanced** (ESI's source, read in a GitHub copy of v2206; the pinned v2406 image
was not reachable when this was written): the fluid
energy equation of `buoyantSimpleFoam` and `chtMultiRegionSimpleFoam` is
`fvm::div(phi, he) + fvc::div(phi, K) - fvm::laplacian(alphaEff, he) == rho*(U&g) + Sh +
fvOptions`. Summed over the region, at a converged steady state, what crosses each boundary face
is the convective `phi_f he_f`, with the patch's own value of `he`, and the conductive
`alphaEff_f snGrad(he)_f |Sf|`. `wallHeatFlux` writes the second on the walls, where `phi` is
zero. On an opening this reads both:

- `he` is the thermo's: with `thermo hConst` and `equationOfState perfectGas`,
  `hs = Cp (T - Tref) + Hsref` (`hConstThermoI.H`; `perfectGas::H` and `::Cp` are 0), linear in
  `T` with slope `Cp`. So only differences of `T` enter: the convective part is taken against
  `T_ref`, the mass-weighted temperature of every face through which flow enters the region
  (`phi < 0`), and the thermo's datum (`Tref`, `Hsref`, `Hf`) never enters. What the openings do
  not balance in mass is VER-18's to judge, not this reading's: carried at `T_ref`, it adds no
  heat.
- `alphaEff` of a laminar region is the thermo's `alpha`, `mu/Pr` for `transport const`
  (`constTransportI.H`, `alphah = mu*rPr`), the same on every face.
- `snGrad(he) = Cp dc (T_f - T_P)`, with `T_f` the face value the case wrote (a `zeroGradient`
  face's is its cell's), `T_P` the cell's, and `dc = 1/(n . (Cf - C_P))`, the patch's
  `deltaCoeffs` (`1/|delta|`, `delta = n (n . (Cf - C_P))`, `fvPatch::delta`), for a face whose
  cell lies behind it (a face with `n . (Cf - C_P) <= 0` is not judged): the same for every basic
  patch type, and a non-coupled patch has no non-orthogonal correction. The cell centres are the
  `basic` geometry scheme's (`polymesh.Mesh`); a region whose `fvSchemes` selects another
  `geometry` is not judged.

- The kinetic energy the flow carries, `fvc::div(phi, K)` with `K = |U|^2/2`, crosses each face
  as `phi_f K_f`, with the face `U` the case wrote. The buoyancy work `rho*(U&g)` is a volume
  term; with the mass flux divergence-free it is `div(phi (g . x))`, so it crosses each face as
  `phi_f (g . Cf)` (to the run's discretisation and continuity error; the review of 4 Nov 2026
  found a 1 m vertical channel with a 0.5 K rise out by 2% without it). `g` is the case's
  `constant/g` (a region holding its own `g` beside it is not judged).

So the energy a face carries into the region is `-phi_f (e_f - e_ref) + (mu/Pr) Cp dc_f (T_f -
T_P) |Sf|`, with `e = Cp T + |U|^2/2 - g . Cf` and `e_ref` its mass-weighted value over the faces
the flow enters by, in W, positive into the region as `wallHeatFlux`'s rows are.

**Not judged, and why** (a sentence, never a traceback): a mesh not read from one directory
(a decomposed run); a mesh, a thermophysicalProperties or a turbulenceProperties that cannot be
read whole; a region that is not laminar; a thermo other than `hConst`, an energy other than an
enthalpy, an equation of state other than `perfectGas`, a transport other than `const`, a
mixture other than `pureMixture`, or a `Cp`, `mu` or `Pr` that is not a positive number; no
time directory at the log's last time; a `phi` that is not an ASCII mass flux; a `T` or `U` that
is not an ASCII field; no `constant/g`, or a region's own `g` beside it; an `fvSchemes` that
selects a `geometry` other than `basic` or is not read whole; a mesh naming more cells than
faces, or an opening's faces outside its boundary faces; an opening's face values that are not
one per face; a face of zero area or whose cell does not lie behind it. Every file is opened through `courant.case.reads` (VER-14, MCP-3), and
every case name a sentence cites is between backticks (`_heat_reader._q`).
"""

from __future__ import annotations

import math
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from courant.case.reads import kind, may_read, plain_or_gz, reading, reason, unknown

#: The thermo this reading reads, by `thermoType` entry: the one whose `he` is linear in `T`
#: with the slope `Cp` and whose `alphaEff` is one number (module docstring).
_THERMO = {"thermo": ("hConst",), "equationOfState": ("perfectGas",),
           "energy": ("sensibleEnthalpy", "absoluteEnthalpy"), "transport": ("const",),
           "mixture": ("pureMixture",)}


@dataclass(frozen=True)
class Openings:
    """The heat carried into a region through each of its openings at the time read."""

    #: {opening: the heat into the region through it, W}.
    heat: dict[str, float]
    #: The heat into the region through the openings' faces that carry heat in, and out
    #: through those that carry it out, W, each at least 0: read face by face, so no opening's
    #: net hides heat crossing it both ways (`_heat_reader._both_ways` asks the walls only).
    heat_in: float
    heat_out: float
    #: The largest |phi_f e_f| over the openings' faces, W: the magnitude of the numbers
    #: multiplied, against which a heat is round-off (`MassFluxCheck.round_off`); never a sum
    #: judged.
    scale: float
    #: The mass-weighted temperature of the flow in, K.
    t_ref: float
    #: The kinetic energy and the buoyancy work carried in, W (included in `heat`), for the reason.
    mechanical: float
    #: The files read, relative to the case.
    files: tuple[str, ...]


def _number(raw: str | None) -> float | None:
    try:
        value = float(str(raw).strip()) if raw is not None else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) and value > 0 else None


def _thermo(case: Path, rel: str, q) -> tuple[float, float, float] | str:
    """(Cp, mu, Pr) of a region's `thermophysicalProperties` at `rel`, or why it is not read."""
    from courant.spec import deck

    try:
        text, why = deck._read_text(case, rel)
    except OSError:
        text, why = None, f"{rel} could not be read"
    if text is None:
        return f"{q(rel)} could not be read ({why.removeprefix(rel).strip()})"
    if not deck._complete(text):
        return f"{q(rel)} is not read whole (a directive or a macro)"
    try:
        kind = deck.block(text, "thermoType")
        mixture = deck.block(text, "mixture")
        thermo = deck.block(mixture, "thermodynamics") if mixture is not None else None
        transport = deck.block(mixture, "transport") if mixture is not None else None
    except ValueError:                                      # a quoted key not evaluated
        return f"{q(rel)} is not read whole (a quoted pattern)"
    if kind is None:
        return f"{q(rel)} names no thermoType block"
    for key, allowed in _THERMO.items():
        value = (deck.entry(kind, key) or "").strip()
        if value not in allowed:
            return (f"{q(rel)} gives the {key} {q(value) if value else 'nothing'}, where this "
                    f"reading reads "
                    f"{' or '.join(allowed)} only")
    cp = _number(deck.entry(thermo, "Cp")) if thermo is not None else None
    mu = _number(deck.entry(transport, "mu")) if transport is not None else None
    pr = _number(deck.entry(transport, "Pr")) if transport is not None else None
    if cp is None or mu is None or pr is None:
        return f"{q(rel)} does not give a positive Cp, mu and Pr under mixture"
    return cp, mu, pr


def _laminar(case: Path, rel: str, q) -> str | None:
    """Why a region's `turbulenceProperties` (v2406's name) is not a laminar one, or None."""
    from courant.spec import deck

    try:
        text, why = deck._read_text(case, rel)
    except OSError:
        text, why = None, f"{rel} could not be read"
    if text is None:
        return f"{q(rel)} could not be read ({why.removeprefix(rel).strip()})"
    if not deck._complete(text):
        return f"{q(rel)} is not read whole (a directive or a macro)"
    sim = (deck.entry(text, "simulationType") or "").strip()
    if sim != "laminar":
        return (f"{q(rel)} gives simulationType {q(sim) if sim else 'nothing'}, not laminar: the "
                f"turbulent "
                f"conductivity at the openings is not read")
    return None


def _geometry(case: Path, rel: str, q) -> str | None:
    """Why the cell centres the solver used may not be `polymesh.Mesh`'s, or None: `rel` (the
    region's `fvSchemes`) selects a `geometry` scheme other than `basic` (ESI's
    `fvGeometryScheme`, from v2006), or cannot be read whole to show it does not."""
    from courant.spec import deck

    try:
        text, why = deck._read_text(case, rel)
    except OSError:
        text, why = None, f"{rel} could not be read"
    if text is None:
        return f"{q(rel)} could not be read ({why.removeprefix(rel).strip()})"
    if not deck._complete(text):
        return f"{q(rel)} is not read whole (a directive or a macro)"
    try:
        body = deck.block(text, "geometry")
    except ValueError:
        return f"{q(rel)} is not read whole (a quoted pattern)"
    kind_ = (deck.entry(body, "type") or "basic").strip() if body is not None else "basic"
    if kind_ != "basic":
        return (f"{q(rel)} selects the geometry scheme {q(kind_)}, whose cell centres are not "
                f"the ones read here")
    return None


def _gravity(case: Path, region_dir: str, q) -> tuple[np.ndarray, str] | str:
    """(g, the file it was read from): the case's `constant/g`, which the committed v2406
    chtMultiRegionSimpleFoam solve read with no region file beside it. A region's own
    `constant/<region>/g` beside it is a file this reading cannot tell the solver ignored, so
    such a region is not judged (the re-review of 4 Nov 2026)."""
    from foamlib import FoamFile

    from courant.case import foamguard

    rel = "constant/g"
    if region_dir:
        own = f"constant/{region_dir}g"
        if not may_read(case / own):
            return f"{q(own)} is outside the roots this reading may read"
        if kind(case / own, stat.S_ISREG) is not False:
            return (f"the region holds its own {q(own)} beside {q(rel)}, and which of them the "
                    f"solver read is not known here")
    path = case / rel
    if not may_read(path):
        return f"{q(rel)} is outside the roots this reading may read"
    found = kind(path, stat.S_ISREG)
    if found is None:
        return f"{q(rel)} could not be looked for"
    if not found:
        return f"the case holds no {q(rel)}"
    try:
        with reading(path):
            foamguard.check_path(path)
            g = np.asarray(FoamFile(path)["value"], dtype=float)
    except Exception:                                   # noqa: BLE001 - said, never raised
        return f"{q(rel)} could not be read"
    if g.shape != (3,) or not np.isfinite(g).all():
        return f"{q(rel)} does not hold one finite vector"
    return g, rel


def _time_dir(case: Path, last: float) -> str | None:
    from courant.case.mesh import time_dirs
    from courant.spec.as_run import _same_time

    found = [t for t in time_dirs(case) if _same_time(float(t), last)]
    return found[0] if len(found) == 1 else None


def _field(path: Path, cls: str):
    """(the internal field, {patch: its entry}) of an ASCII field file of class `cls`, its
    dimensions, read through `reading`. Raises `ValueError` with the reason."""
    from foamlib import FoamFieldFile

    from courant.case import foamguard

    with reading(path):
        foamguard.check_path(path)      # a field cut mid-write: refused, never parsed for ever
        f = FoamFieldFile(path)
        if str(f.format) != "ascii":
            raise ValueError("is not written in ASCII, the only format read")
        if str(f["FoamFile"].get("class", "")) != cls:
            raise ValueError(f"is not a {cls}")
        try:
            dims = tuple(float(x) for x in f.dimensions)
        except (TypeError, ValueError):
            dims = ()
        return f.internal_field, {str(k): dict(v) for k, v in f.boundary_field.items()}, dims


def _per_face(raw, n: int, width: int = 0) -> np.ndarray | None:
    """`raw` as `n` face (or cell) values, scalars or (`width` 3) vectors: a `uniform` value
    repeated, or a list of `n`."""
    if raw is None:                                     # asarray(None) would be a nan
        return None
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    one = (width,) if width else ()
    if values.shape == one:
        return np.broadcast_to(values, (n, *one)).copy()
    return values if values.shape == (n, *one) else None


def read_openings(case: Path, region: str, openings, mesh_dir: Path, last: float,
                  region_dir: str, q) -> Openings | str:
    """The heat carried into `region` through each patch of `openings` (`polymesh.Patch`es of
    the boundary `_heat_reader` read), at the log's last time `last`, or why it is not read (a
    clause the caller words into its sentence). `mesh_dir` is the one polyMesh directory the
    solver read for the region; `region_dir` is "" or "<region>/"; `q` quotes a case word."""
    from courant.case.mesh import POLYMESH_MAX_BYTES, POLYMESH_MAX_FACES, polymesh_bytes
    from courant.case.polymesh import FILES, read_polymesh_dir
    from courant.verification._fluid_reader import _FLUX_KINDS

    thermo_rel = f"constant/{region_dir}thermophysicalProperties"
    turbulence_rel = f"constant/{region_dir}turbulenceProperties"
    # OpenFOAM.org's layout: a region's thermo is its `physicalProperties`, so a
    # thermophysicalProperties beside it is not what the solver read and is not opened
    # (`_foam_reader._read_region_thermo`'s rule, T-MULTI-REGION-2).
    physical_rel = f"constant/{region_dir}physicalProperties"
    if not may_read(case / physical_rel):
        return f"{q(physical_rel)} is outside the roots this reading may read"
    if kind(case / physical_rel, stat.S_ISREG) is not False:
        return (f"the region holds {q(physical_rel)}, OpenFOAM.org's layout, which this "
                f"reading does not read")
    props = _thermo(case, thermo_rel, q)
    if isinstance(props, str):
        return props
    cp, mu, pr = props
    why = _laminar(case, turbulence_rel, q)
    if why is not None:
        return why
    why = _geometry(case, f"system/{region_dir}fvSchemes", q)
    if why is not None:
        return why
    gravity = _gravity(case, region_dir, q)
    if isinstance(gravity, str):
        return gravity
    g, g_rel = gravity
    time = _time_dir(case, last)
    if time is None:
        return f"the case holds no one time directory at the log's last time, {last:g}"
    # ---- the mesh the solver read, every file bounded (`courant.case.mesh`'s bounds)
    mesh_rel = mesh_dir.relative_to(case).as_posix()
    try:
        # MCP-3: each file asked of the confinement before it is looked for or sized, as
        # `courant.case.mesh.solver_polymesh` asks it ("sized and read only inside the roots").
        if not all(may_read(mesh_dir / n) and may_read(f"{mesh_dir / n}.gz") for n in FILES):
            return f"the mesh {q(mesh_rel)} is outside the roots this reading may read"
        files = [Path(plain_or_gz(str(mesh_dir / n))) for n in FILES]
        if polymesh_bytes(files) > POLYMESH_MAX_BYTES:
            return f"the mesh {q(mesh_rel)} is past the {POLYMESH_MAX_BYTES // 2**20} MiB read"
        mesh = read_polymesh_dir(mesh_dir, max_faces=POLYMESH_MAX_FACES)
    except Exception:                                   # noqa: BLE001 - said, never raised
        return f"the mesh {q(mesh_rel)} could not be read whole"
    # Every cell owns a face, so a real mesh has no more cells than faces: a label past that is
    # case text that would size the cell arrays below (the cell centres, a `uniform` T).
    if mesh.n_cells > len(mesh.owner):
        return f"the mesh {q(mesh_rel)} names more cells than it has faces"
    for p in openings:
        if p.start < mesh.n_internal or p.start + p.n > len(mesh.faces):
            return (f"the faces of opening {q(p.name)} that the boundary of {q(mesh_rel)} names "
                    f"are not boundary faces of its mesh")
    # ---- the fields at that time
    read: dict[str, tuple] = {}
    for name, cls in (("phi", "surfaceScalarField"), ("T", "volScalarField"),
                      ("U", "volVectorField")):
        plain = case / time / region_dir / name
        if not (may_read(plain) and may_read(f"{plain}.gz")):    # MCP-3: before it is looked for
            return f"{q(f'{time}/{region_dir}{name}')} is outside the roots this reading may read"
        path = Path(plain_or_gz(str(plain)))
        rel = path.relative_to(case).as_posix()
        found = kind(path, stat.S_ISREG)
        if found is None:
            return (f"{q(f'{time}/{region_dir}{name}')} could not be looked for (a directory on "
                    f"the way could not be searched: {reason(unknown(path))})")
        if not found:
            return f"there is no {q(f'{time}/{region_dir}{name}')} at the log's last time"
        if path.stat().st_size > POLYMESH_MAX_BYTES:
            return f"{q(rel)} is past the {POLYMESH_MAX_BYTES // 2**20} MiB read"
        try:
            read[name] = (rel, *_field(path, cls))
        except Exception as exc:                        # noqa: BLE001 - said, never raised
            detail = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(
                ("is ", "is not")) else "could not be read"
            return f"{q(rel)} {detail}"
    phi_rel, _, phi_patches, phi_dims = read["phi"]
    if _FLUX_KINDS.get(phi_dims) != "mass":
        return f"{q(phi_rel)} is not a mass flux, so the enthalpy it carries is not read"
    t_rel, t_internal, t_patches, _ = read["T"]
    t_cells = _per_face(t_internal, mesh.n_cells)
    if t_cells is None:
        return f"{q(t_rel)} does not hold one temperature per cell of {q(mesh_rel)}"
    u_rel, u_internal, u_patches, _ = read["U"]
    u_cells = _per_face(u_internal, mesh.n_cells, 3)
    if u_cells is None:
        return f"{q(u_rel)} does not hold one velocity per cell of {q(mesh_rel)}"
    # ---- each face of each opening
    centres, areas, cells = mesh.face_centres, mesh.face_areas, mesh.cell_centres
    faces: dict[str, tuple] = {}
    for p in openings:
        idx = np.arange(p.start, p.start + p.n)
        phi = _per_face((phi_patches.get(p.name) or {}).get("value"), p.n)
        owner_t = t_cells[mesh.owner[idx]]
        t_face = _face_values(t_patches.get(p.name) or {}, p.n, owner_t)
        u_face = _face_values(u_patches.get(p.name) or {}, p.n, u_cells[mesh.owner[idx]], 3)
        if phi is None:
            return f"{q(phi_rel)} does not hold one flux per face of opening {q(p.name)}"
        if t_face is None:
            return f"{q(t_rel)} does not hold one value per face of opening {q(p.name)}"
        if u_face is None:
            return f"{q(u_rel)} does not hold one value per face of opening {q(p.name)}"
        mag = np.linalg.norm(areas[idx], axis=1)
        if not (mag > 0).all():
            return f"opening {q(p.name)} has a face of no area in {q(mesh_rel)}"
        normal = np.einsum("ij,ij->i", areas[idx] / mag[:, None],
                           centres[idx] - cells[mesh.owner[idx]])
        # The patch's `deltaCoeffs`, 1/|delta|: a face whose cell is not behind it is not read.
        if not (normal > 0).all():
            return (f"opening {q(p.name)} has a face whose cell does not lie behind it in "
                    f"{q(mesh_rel)}")
        mechanical = 0.5 * np.einsum("ij,ij->i", u_face, u_face) - centres[idx] @ g
        faces[p.name] = (phi, t_face, owner_t, mag, 1.0 / normal, mechanical)
    values = [x for v in faces.values() for part in (v[0], v[1], v[2], v[5]) for x in part]
    if not all(math.isfinite(x) for x in values):
        return (f"{q(phi_rel)}, {q(t_rel)} or {q(u_rel)} holds a value that is not finite on an "
                f"opening")
    phi_all = np.concatenate([v[0] for v in faces.values()]) if faces else np.zeros(0)
    e_all = np.concatenate([cp * v[1] + v[5] for v in faces.values()]) if faces else np.zeros(0)
    t_all = np.concatenate([v[1] for v in faces.values()]) if faces else np.zeros(0)
    flow = phi_all < 0 if (phi_all < 0).any() else phi_all > 0
    if flow.any():                                      # the flow in, else (none in) the flow out
        w = phi_all[flow] / phi_all[flow].sum()
        e_ref, t_ref = float(w @ e_all[flow]), float(w @ t_all[flow])
    else:
        e_ref = t_ref = 0.0                             # no flow: the convective part is zero
    heat: dict[str, float] = {}
    heat_in = heat_out = scale = carried_mech = 0.0
    for name, (phi, t_face, owner_t, mag, dc, mech) in faces.items():
        e_face = cp * t_face + mech
        q_face = -phi * (e_face - e_ref) + (mu / pr) * cp * dc * (t_face - owner_t) * mag
        heat[name] = float(q_face.sum())
        heat_in += float(q_face[q_face > 0].sum())
        heat_out -= float(q_face[q_face < 0].sum())
        carried_mech += float((-phi * mech).sum())
        scale = max(scale, float((np.abs(phi) * np.abs(e_face)).max(initial=0.0)))
    return Openings(heat=heat, heat_in=heat_in, heat_out=heat_out, scale=scale, t_ref=t_ref,
                    mechanical=carried_mech,
                    files=(phi_rel, t_rel, u_rel, g_rel, f"{mesh_rel}/boundary", thermo_rel,
                           turbulence_rel))


def _face_values(entry: dict, n: int, cells: np.ndarray, width: int = 0) -> np.ndarray | None:
    """A patch's face values as the case wrote them (`value`), or its cells' for a
    `zeroGradient` patch, which writes none; None otherwise."""
    if "value" in entry:
        return _per_face(entry["value"], n, width)
    if str(entry.get("type", "")) == "zeroGradient":
        return cells
    return None


__all__ = ["Openings", "read_openings"]
