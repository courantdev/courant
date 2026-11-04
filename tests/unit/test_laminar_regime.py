"""T-LAMINAR-REGIME (TRD VER-5b): `physics.laminar_unstable` defers to the case's own regime numbers
on a time-marching laminar run, behind `--checks next` (S's decision on H-LAMINAR-TRANSIENT, 26
Oct 2026), and its review of the same day.

The finding takes residuals that end above the flat 1e-3 bar as evidence that a laminar model is
on a turbulent flow. On a time-marching run that bar is not the gate's convergence test (VER-21),
and a periodic laminar flow ends above it at a large enough step: a rebuilt laminar cylinder at
Re 100 (a private evaluation's, 22 Oct 2026) was rejected on this finding alone. Under `next` the
finding weighs the case's numbers first (`courant.case.regime.regime_numbers`): the Reynolds
number the case gives unambiguously (`courant.case.reynolds`), and physics.laminar_regime's
Rayleigh number for a case heated from below, a bound only where buoyancy alone moves the fluid.
Every number below its flow's laminar bound sets the residual history aside and the verdict says
why; a number above its turbulent bound is cited; otherwise nothing changes. No number bounds a
flow something else drives. Under the default the finding is 0.6.0's on every run.

Narrowed by the final review (of 8249ee4, 26 Oct 2026): a Reynolds number is given for flow past
one bluff body only, never for a duct (no laminar bound holds for every 2-D geometry: a baffle's
jet turns unsteady below Armaly's step bound); its speed bound is the larger of the inlet's and
the start field's largest |U| (a faster start field is carried for many flow-through times); and
buoyancy alone refuses a radiation model, which ESI's buoyant solvers build in the solver.

The runs: a laminar cylinder at Re 100 written from scratch on v2406's `cylinder2D` settings and
solved on the pinned image over two ranks (`tests/fixtures/real_openfoam/laminar_cylinder`);
v2406-shaped synthetic case directories written here (`courant.case.polymesh.write_polymesh`)
for each reading the Reynolds number gives or refuses, and for each faulted case the review found
moving toward accept (a mixed-convection cavity and a split-floor heater as full verdicts, a 3-D
nozzle and tee, a second opening, a held inlet or wall, an MRF zone and each other source, a
plate aligned with the flow, 60,000 hull points), and each its second review found (of dde0ecb:
the cylinder spun behind a `$` entry or a `wall` group entry, serially and over two ranks, as
full verdicts; a moving body or a held wall pressure behind a `$` entry, a moving body behind a
group entry; the cavity's sides heated behind either), and each its final review found (of
8249ee4: a 2-D channel narrowed by a baffle, a body that is not bluff, a start field faster than
the inlet, the cylinder started 100 times faster as a full verdict, serially and over two ranks,
and the still cavity under a radiation model, as a full verdict); the committed wedge pipe; and
the certification corpus's two `laminar_on_turbulent` faults, which stay rejected.
"""

from __future__ import annotations

import gzip
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from courant.case.config import CaseConfig
from courant.case.mesh import MeshQuality
from courant.case.polymesh import Mesh, Patch, write_polymesh
from courant.case.regime import (
    RA_ONSET,
    RA_TRANSITION,
    RE_EXTERNAL_LAMINAR,
    RE_EXTERNAL_TURBULENT,
    regime_numbers,
)
from courant.case.reynolds import Reynolds, _widths, reynolds_number
from courant.diagnosis.checkers import (
    LAMINAR_REGIME,
    CheckContext,
    PhysicsChecker,
    laminar_spared_sentence,
)
from courant.events import CrashPayload, Event, EventKind, ResidualPayload
from courant.spec.as_run import Run

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
REAL = ROOT / "tests" / "fixtures" / "real_openfoam"


# ------------------------------------------------------------------ synthetic v2406-shaped cases
def _structured(xs, ys, zs, solid=frozenset(), side=None, body="body"):
    """A hexahedral mesh on the grid xs x ys x zs, the cells in `solid` left out (a body), each
    boundary face on the patch `side(name, i, j, k)` names for a domain side (`xlo` ... `zhi`), or
    on `body` beside a solid cell. Faces as OpenFOAM orders them: internal first (owner the lower
    cell, normal towards the higher), then boundary faces by patch, each normal outward."""
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
    cells = {}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if (i, j, k) not in solid:
                    cells[(i, j, k)] = len(cells)

    def p(i, j, k):
        return (k * (ny + 1) + j) * (nx + 1) + i

    points = np.array([(xs[i], ys[j], zs[k]) for k in range(nz + 1) for j in range(ny + 1)
                       for i in range(nx + 1)], dtype=float)
    corners = {  # +x, +y, +z oriented quads at the lower corner (i, j, k) of each direction
        0: lambda i, j, k: [p(i, j, k), p(i, j + 1, k), p(i, j + 1, k + 1), p(i, j, k + 1)],
        1: lambda i, j, k: [p(i, j, k), p(i, j, k + 1), p(i + 1, j, k + 1), p(i + 1, j, k)],
        2: lambda i, j, k: [p(i, j, k), p(i + 1, j, k), p(i + 1, j + 1, k), p(i, j + 1, k)],
    }
    names = ("x", "y", "z")
    internal, boundary = [], {}
    for d in range(3):
        n = (nx, ny, nz)
        for k in range(nz + (d == 2)):
            for j in range(ny + (d == 1)):
                for i in range(nx + (d == 0)):
                    lo = (i - (d == 0), j - (d == 1), k - (d == 2))
                    hi = (i, j, k)
                    a, b = cells.get(lo), cells.get(hi)
                    quad = corners[d](i, j, k)
                    if a is not None and b is not None:
                        internal.append((a, b, quad))
                        continue
                    if a is None and b is None:
                        continue
                    at = (i, j, k)[d]
                    if at == 0:
                        name = side(names[d] + "lo", i, j, k)
                    elif at == n[d]:
                        name = side(names[d] + "hi", i, j, k)
                    else:
                        name = body if isinstance(body, str) else body(lo if a is None else hi)
                    owner = a if a is not None else b
                    boundary.setdefault(name, []).append(
                        (owner, quad if a is not None else quad[::-1]))
    internal.sort(key=lambda t: (t[0], t[1]))
    faces = [np.array(q) for _, _, q in internal]
    owner = [a for a, _, _ in internal]
    neighbour = [b for _, b, _ in internal]
    patches = []
    for name, fs in boundary.items():
        patches.append((name, len(faces), len(fs)))
        faces += [np.array(q) for _, q in fs]
        owner += [o for o, _ in fs]
    return points, faces, np.array(owner), np.array(neighbour), patches


LOG = """\
Exec   : {solver}
nProcs : 1

Create time

Create mesh for time = 0

{models}
Time = 0.01

smoothSolver:  Solving for Ux, Initial residual = 0.5, Final residual = 1e-06, No Iterations 2
End
"""
MODELS = ("Selecting incompressible transport model Newtonian\n"
          "Selecting turbulence model type laminar\nSelecting laminar stress model Stokes\n")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


HEADER = 'FoamFile { version 2.0; format ascii; class %s; object %s; }\n'
#: The constraint types, whose condition is the patch's own type (fvPatchFieldNew.C).
CONSTRAINT = ("empty", "wedge", "symmetry", "symmetryPlane")


def _entries(patches, types, given: dict[str, str], default) -> str:
    return "".join(f"    {name}\n    {{\n        {given.get(name, default(name, types[name]))}\n"
                   "    }\n" for name, _, _ in patches)


def _p_default(name: str, kind: str) -> str:
    """v2406's usual pressure pairing: fixed at the outlet, free (zeroGradient) elsewhere."""
    if kind in CONSTRAINT:
        return f"type {kind};"
    return "type fixedValue; value uniform 0;" if name == "outlet" else "type zeroGradient;"


def _case(tmp: Path, grid, types: dict[str, str], u: dict[str, str], *, nu: float = 1e-3,
          solver: str = "pimpleFoam", models: str = MODELS, u_text: str | None = None,
          p: dict[str, str] | None = None, log: str | None = None):
    """A case directory: the mesh, `0/U` (one entry per patch, `u`), `0/p` (`p`, else fixed at
    the outlet and free elsewhere), the transport and the turbulence dictionaries, controlDict,
    and a solver log in v2406's shape. Returns the case, its configuration as the reader reads
    it, and its run."""
    points, faces, owner, neighbour, patches = grid
    boundary = tuple(Patch(name=n, type=types[n], start=s, n=c) for n, s, c in patches)
    write_polymesh(Mesh(points=points, faces=tuple(faces), owner=owner, neighbour=neighbour,
                        boundary=boundary), tmp)
    header = HEADER
    entries = _entries(patches, types, u, lambda name, kind: f"type {kind};")
    _write(tmp / "0" / "U", u_text if u_text is not None else (
        header % ("volVectorField", "U") + "dimensions [0 1 -1 0 0 0 0];\n"
        "internalField uniform (0 0 0);\nboundaryField\n{\n" + entries + "}\n"))
    _write(tmp / "0" / "p", header % ("volScalarField", "p") + "dimensions [0 2 -2 0 0 0 0];\n"
           "internalField uniform 0;\nboundaryField\n{\n"
           + _entries(patches, types, p or {}, _p_default) + "}\n")
    _write(tmp / "constant" / "transportProperties", header % ("dictionary", "transportProperties")
           + f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {nu};\n")
    _write(tmp / "constant" / "turbulenceProperties",
           header % ("dictionary", "turbulenceProperties") + "simulationType laminar;\n")
    _write(tmp / "system" / "controlDict", header % ("dictionary", "controlDict")
           + f"application {solver};\nstartFrom startTime;\nstartTime 0;\nendTime 1;\n"
           "deltaT 0.01;\nwriteControl timeStep;\nwriteInterval 100;\n")
    log = log if log is not None else LOG.format(solver=solver, models=models)
    _write(tmp / f"log.{solver}", log)
    return tmp, CaseConfig.from_foam(str(tmp)), Run(log, f"log.{solver}", (), exit_code=0)


def _re(case_cfg_run) -> Reynolds | None:
    return reynolds_number(*case_cfg_run)


SLAB = (0.0, 0.01)
#: A 2-D channel 1 m long and 0.1 m high, walls top and bottom, one cell thick.
CHANNEL = dict(xs=np.linspace(0, 1, 11), ys=np.linspace(0, 0.1, 5), zs=SLAB)
CHANNEL_SIDES = {"xlo": "inlet", "xhi": "outlet", "ylo": "lower", "yhi": "upper",
                 "zlo": "frontAndBack", "zhi": "frontAndBack"}
CHANNEL_TYPES = {"inlet": "patch", "outlet": "patch", "lower": "wall", "upper": "wall",
                 "frontAndBack": "empty"}
INLET = "type fixedValue; value uniform (0.5 0 0);"
OPEN = {"outlet": "type zeroGradient;", "lower": "type noSlip;", "upper": "type noSlip;"}


def _channel(sides=None, **kw):
    sides = {**CHANNEL_SIDES, **(sides or {})}
    return _structured(**{**CHANNEL, **kw}, side=lambda s, i, j, k: sides[s])


#: A 2-D box 2 m by 1 m with a 0.2 m square body at its centre, symmetry planes top and bottom.
BOX = dict(xs=np.linspace(0, 2, 21), ys=np.linspace(0, 1, 11), zs=SLAB)
SQUARE = frozenset((i, j, 0) for i in (6, 7) for j in (4, 5))
BOX_SIDES = {"xlo": "inlet", "xhi": "outlet", "ylo": "sides", "yhi": "sides",
             "zlo": "frontAndBack", "zhi": "frontAndBack"}
BOX_TYPES = {"inlet": "patch", "outlet": "patch", "sides": "symmetryPlane", "body": "wall",
             "frontAndBack": "empty"}
STREAM = "type fixedValue; value uniform (1 0 0);"


def _box(solid=SQUARE, body="body", sides=None, **kw):
    sides = {**BOX_SIDES, **(sides or {})}
    return _structured(**{**BOX, **kw}, solid=solid, side=lambda s, i, j, k: sides[s],
                       body=body)


def _external(tmp, grid=None, types=None, u=None, **kw):
    grid = grid if grid is not None else _box()
    return _case(tmp, grid, {**BOX_TYPES, **(types or {})},
                 {"inlet": STREAM, "outlet": "type zeroGradient;", "body": "type noSlip;",
                  **(u or {})}, **kw)


# ------------------------------------------------------------------ no duct flow gives a number
#: A fin on the lower wall halfway along the channel: 3 of its 4 cells high (a 4:1 contraction).
FIN = frozenset((5, j, 0) for j in range(3))
#: The final review's baffled channel (of 8249ee4): 10 mm high and 0.1 m long, a baffle joined to
#: its upper wall leaving a 1 mm gap above the lower one (one wall piece, touching the inlet).
BAFFLED = dict(xs=np.linspace(0, 0.1, 21), ys=np.linspace(0, 0.01, 11), zs=SLAB)
BAFFLE = frozenset((5, j, 0) for j in range(1, 10))


def _duct3d(solid=frozenset()):
    """A square duct 1 m long, 0.1 m wide and high, walls on its four sides, in 3-D."""
    sides = {**CHANNEL_SIDES, "zlo": "lower", "zhi": "upper"}
    return _structured(xs=np.linspace(0, 1, 11), ys=np.linspace(0, 0.1, 5),
                       zs=np.linspace(0, 0.1, 5), solid=solid, side=lambda s, i, j, k: sides[s])


#: A 2:1 contraction of the duct over x in [0.4, 0.7]: only its 2 x 2 central cells stay open.
THROAT = frozenset((i, j, k) for i in (4, 5, 6) for j in range(4) for k in range(4)
                   if not (j in (1, 2) and k in (1, 2)))
#: A second opening at the end of the channel's upper wall, beside the outlet.
CORNER_SIDES = lambda s, i, j, k: "branch" if s == "yhi" and i == 9 else CHANNEL_SIDES[s]  # noqa: E731
#: The 3-D duct with a branch opening through the middle of its upper wall: the first review's
#: tee, its branch held at 50 m^2/s^2 above the outlet.
TEE_SIDES = lambda s, i, j, k: ("branch" if s == "yhi" and 4 <= i < 6 and 1 <= k < 3  # noqa: E731
                                else {**CHANNEL_SIDES, "zlo": "lower", "zhi": "upper"}[s])


@pytest.mark.parametrize("duct", ["channel", "narrowed", "baffled", "mirror", "duct3d",
                                  "nozzle3d", "tee", "second_opening"])
def test_no_duct_flow_gives_a_number(tmp_path, duct):
    """The final review (of 8249ee4, 26 Oct 2026). In a 2-D duct continuity carries the inlet's
    flow rate per unit depth through every section, so the inlet's number is every section's; but
    no laminar bound is shown for every 2-D geometry it may describe. Armaly's 1200 was measured
    behind one backward-facing step, and the review's baffle (Re 1000 on the inlet's hydraulic
    diameter, 0.05 m/s and nu 1e-6) makes a planar jet through its 1 mm gap, slot Re 500, which
    turns unsteady well below it. At 8249ee4 the 2-D channel, narrowed or baffled, gave a number
    below 1200; no duct gives one now, in 2-D, in 3-D, through a contraction, beside a mirror or
    with a second opening."""
    u = {"inlet": INLET, **OPEN}
    types = dict(CHANNEL_TYPES)
    p: dict[str, str] = {}
    nu = 1e-3
    if duct == "channel":
        grid = _channel()
    elif duct == "narrowed":
        grid = _structured(**CHANNEL, solid=FIN, side=lambda s, i, j, k: CHANNEL_SIDES[s])
        types["body"], u["body"] = "wall", "type noSlip;"
    elif duct == "baffled":
        grid = _structured(**BAFFLED, solid=BAFFLE, side=lambda s, i, j, k: CHANNEL_SIDES[s])
        types["body"], u["body"] = "wall", "type noSlip;"
        u["inlet"], nu = "type fixedValue; value uniform (0.05 0 0);", 1e-6
    elif duct == "mirror":
        grid = _channel(sides={"yhi": "mirror"})
        types["mirror"] = "symmetryPlane"
    elif duct in ("duct3d", "nozzle3d"):
        grid = _duct3d(THROAT if duct == "nozzle3d" else frozenset())
        types["body"], u["body"] = "wall", "type noSlip;"
    elif duct == "tee":
        grid = _structured(xs=np.linspace(0, 1, 11), ys=np.linspace(0, 0.1, 5),
                           zs=np.linspace(0, 0.1, 5), side=TEE_SIDES)
        types["branch"] = "patch"
        u["branch"], p["branch"] = "type zeroGradient;", "type fixedValue; value uniform 50;"
    else:
        grid = _structured(**CHANNEL, side=CORNER_SIDES)
        types["branch"], u["branch"] = "patch", "type zeroGradient;"
    assert _re(_case(tmp_path, grid, types, u, p=p, nu=nu)) is None


def test_the_wedge_pipe_gives_none(tmp_path):
    """The committed laminar pipe on a 5 degree wedge: a duct, which gives no number."""
    case = tmp_path / "flow"
    _unpack(REAL / "wedge_pipe" / "flow", case)
    log = gzip.open(REAL / "wedge_pipe" / "flow" / "log.simpleFoam.gz", "rt").read()
    assert reynolds_number(case, CaseConfig.from_foam(str(case)),
                           Run(log, "log.simpleFoam", (), exit_code=0)) is None


# ------------------------------------------------------------------ what else drives the flow
@pytest.mark.parametrize("change", ["second_held_opening", "inlet_held", "wall_held"])
def test_a_flow_the_pressure_may_drive_gives_none(tmp_path, change):
    """pimpleFoam's flux is `phi = phiHbyA - pEqn.flux()` (pEqn.H), whose boundary part vanishes
    only where the pressure is free (zeroGradient, or fixedFluxPressure, which sets its gradient
    so it does): there a fixed velocity is the flux, elsewhere the pressure drives it. A second
    opening held at a pressure, or a fixed velocity, on the inlet or a wall, with a fixed
    pressure. Each gave a number at 36c2a6d."""
    if change == "second_held_opening":
        case = _external(tmp_path, nu=2e-3, types={"sides": "patch"},
                         u={"sides": "type zeroGradient;"},
                         p={"sides": "type fixedValue; value uniform 0;"})
    elif change == "inlet_held":
        case = _external(tmp_path, nu=2e-3, p={"inlet": "type fixedValue; value uniform 1;"})
    else:
        case = _external(tmp_path, nu=2e-3, p={"body": "type fixedValue; value uniform 0;"})
    assert _re(case) is None


def test_the_pressure_pairings_v2406_uses_give_a_number(tmp_path):
    """fixedFluxPressure is as free as zeroGradient (constrainPressure sets its gradient so the
    flux is the velocity's), and one opening held at a pressure is the outlet."""
    assert _re(_external(tmp_path / "a", nu=2e-3)) is not None
    assert _re(_external(tmp_path / "b", nu=2e-3, p={
        "inlet": "type fixedFluxPressure; value uniform 0;",
        "body": "type fixedFluxPressure; value uniform 0;"})) is not None


MRF = HEADER % ("dictionary", "MRFProperties") + (
    "MRF1 { cellZone rotor; active yes; origin (1 0.5 0); axis (0 0 1); omega 100; }\n")


@pytest.mark.parametrize("driver", [
    "constant/MRFProperties", "constant/SRFProperties", "constant/fvOptions", "system/fvOptions",
    "constant/fvModels", "system/fvConstraints", "constant/dynamicMeshDict",
    "log: Creating MRF zone list from MRFProperties",
    "log: Creating finite volume options from \"system/fvOptions\"",
    "log: Selecting dynamicFvMesh dynamicMotionSolverFvMesh", "log: Reading g"])
def test_anything_else_that_drives_the_flow_gives_none(tmp_path, driver):
    """An MRF zone sets its walls to Omega x r (MRFZone::correctBoundaryVelocity, MRFZone.C): the
    review's rotor of 0.1 m at 100 rad/s in a 0.01 m/s stream (nu 1e-5) has Re 100 on the
    stream and about 5e4 at its tip. An SRF frame, an fvOption or fvModel (a momentum source),
    an fvConstraint (OpenFOAM.org's meanVelocityForce), a moving mesh, or buoyancy (a log that
    read `constant/g`: buoyantBoussinesqPimpleFoam reads `nu` and prints Newtonian too) moves
    the fluid with no inlet's speed. Each gives none: its file where the solver would look
    (IOMRFZoneList.C, fvOptions.C: `constant`, then `system`; dynamicFvMeshNew.C), or the
    solver's own line saying it took one."""
    case, cfg, run = _external(tmp_path, nu=2e-3)
    base = _re((case, cfg, run))
    assert base is not None and base.upper < RE_EXTERNAL_LAMINAR
    if driver.startswith("log: "):
        text = run.log_text.replace("Create mesh for time = 0\n",
                                    f"Create mesh for time = 0\n\n{driver[5:]}\n")
        run = Run(text, run.log_name, (), exit_code=0)
    else:
        body = ("dynamicFvMesh dynamicMotionSolverFvMesh;\n" if "dynamicMesh" in driver
                else "")
        _write(case / driver, MRF if "MRF" in driver else
               HEADER % ("dictionary", driver.split("/")[1]) + body)
        cfg = CaseConfig.from_foam(str(case))
    assert reynolds_number(case, cfg, run) is None


# ------------------------------------------------------------------ the Reynolds number: one bluff body
def test_one_body_in_the_stream_is_bounded_on_its_largest_extent(tmp_path):
    """The upper bound is on the body's largest extent (the square's diagonal), which no length
    of it exceeds; the lower bound on its width across the flow (the side). A square is bluff:
    its diagonal is sqrt 2 times its side."""
    re = _re(_external(tmp_path, nu=2e-3))
    assert re.length == pytest.approx(0.2 * math.sqrt(2))
    assert re.upper == pytest.approx(100.0 * math.sqrt(2)) and re.lower == pytest.approx(100.0)


#: A plate 1 m long and 1e-4 m thick, aligned with the stream, in the middle of the box.
PLATE_BOX = dict(xs=np.linspace(0, 2, 21), ys=np.array([0.0, 0.25, 0.5, 0.5001, 0.75, 1.0]),
                 zs=SLAB)
PLATE = frozenset((i, 2, 0) for i in range(5, 15))
#: A body 0.6 m along the stream and 0.2 m across it; and one 0.2 m along and 0.6 m across.
LONG = frozenset((i, j, 0) for i in range(6, 12) for j in (4, 5))
WIDE = frozenset((i, j, 0) for i in (6, 7) for j in range(2, 8))
#: An inflow whose direction turns by 5 degrees in its first second (a `uniformFixedValue` table).
TURNING = ("type uniformFixedValue; uniformValue table ((0 (0.9962 0.0872 0)) "
           "(1 (0.9962 0.0872 0)) (2 (1 0 0)) (100 (1 0 0)));")


@pytest.mark.parametrize("body", ["plate", "long", "wide_turning"])
def test_a_body_that_is_not_bluff_gives_no_number(tmp_path, body):
    """Roshko's bounds are a bluff body's wake's, so only a bluff body gives a number (the final
    review, of 8249ee4): one no longer in any direction than twice its width across the flow
    (`_BLUFF`). The first review's plate aligned with the flow (1.4 m/s, nu 1e-6: Re 140 on its
    thickness, 1.4e6 on its length) and a body three times as long as it is wide give none. With
    a direction that turns in time the body must be bluff across any direction the flow may take:
    its largest extent at most twice its least width. The wide body is bluff across a fixed
    stream, and gives none across a turning one. At 8249ee4 each gave an upper bound."""
    if body == "plate":
        grid = _structured(**PLATE_BOX, solid=PLATE, side=lambda s, i, j, k: BOX_SIDES[s])
        assert _re(_external(tmp_path, grid=grid, nu=1e-6, u={
            "inlet": "type fixedValue; value uniform (1.4 0 0);"})) is None
        return
    if body == "long":
        assert _re(_external(tmp_path, grid=_box(solid=LONG), nu=2e-3)) is None
        return
    fixed = _re(_external(tmp_path / "fixed", grid=_box(solid=WIDE), nu=2e-3))
    assert fixed.lower == pytest.approx(0.6 / 2e-3)
    assert fixed.upper == pytest.approx(math.hypot(0.2, 0.6) / 2e-3)
    assert _re(_external(tmp_path / "turning", grid=_box(solid=WIDE), nu=2e-3,
                         u={"inlet": TURNING})) is None


def test_a_flow_at_an_angle_takes_the_bodys_width_across_it(tmp_path):
    """The square's width across a flow at 45 degrees in the solved plane is its diagonal."""
    s = 1 / math.sqrt(2)
    re = _re(_external(tmp_path, nu=2e-3,
                       u={"inlet": f"type fixedValue; value uniform ({s!r} {s!r} 0);"}))
    assert re.length == pytest.approx(0.2 * math.sqrt(2))
    assert re.lower == pytest.approx(re.upper)


def test_a_velocity_that_changes_direction_in_time_takes_the_bodys_largest_width(tmp_path):
    """A `uniformFixedValue` table: the largest speed of its entries, and with its direction
    turning, the body's largest width in the solved plane (the square's diagonal); no lower
    bound. The square is bluff across any direction: its diagonal is sqrt 2 times its least
    width."""
    re = _re(_external(tmp_path, u={"inlet": TURNING}, nu=2e-3))
    assert re.lower is None
    assert re.length == pytest.approx(0.2 * math.sqrt(2))
    assert re.speed == pytest.approx(math.hypot(0.9962, 0.0872))


def test_decomposepars_table_form_reads_as_the_inline_one(tmp_path):
    """decomposePar writes a table as `uniformValue table;` and `uniformValueCoeffs { values N
    (...); }` (TableBase::writeData), which Table.C reads from its coefficients' `values`."""
    inline = _re(_external(tmp_path / "a", nu=2e-3, u={"inlet": (
        "type uniformFixedValue; uniformValue table ((0 (2 0 0)) (5 (1 0 0)));")}))
    coeffs = _re(_external(tmp_path / "b", nu=2e-3, u={"inlet": (
        "type uniformFixedValue; uniformValue table; uniformValueCoeffs { values 2 "
        "((0 (2 0 0)) (5 (1 0 0))); }")}))
    assert inline == coeffs and inline.speed == pytest.approx(2.0) and inline.lower is None


@pytest.mark.parametrize("form", ["(1 0 0)", "constant (1 0 0)", "uniform (1 0 0)"])
def test_a_constant_uniform_value_is_one_velocity(tmp_path, form):
    re = _re(_external(tmp_path, nu=2e-3, u={
        "inlet": f"type uniformFixedValue; uniformValue {form};"}))
    assert re.lower == pytest.approx(100.0) and re.upper == pytest.approx(100.0 * math.sqrt(2))


def test_a_body_in_three_dimensions_takes_its_projected_widths(tmp_path):
    """In 3-D the lower bound is on the smallest width of the body's projection on the plane
    normal to the flow; the upper on a bound of its largest extent, the projection's diameter and
    its extent along the flow at right angles (the cube's: its side times sqrt(3))."""
    zs = np.linspace(0, 1, 11)
    cube = frozenset((i, j, k) for i in (6, 7) for j in (4, 5) for k in (4, 5))
    sides = {**BOX_SIDES, "zlo": "sides", "zhi": "sides"}
    grid = _structured(**{**BOX, "zs": zs}, solid=cube, side=lambda s, i, j, k: sides[s])
    re = _re(_external(tmp_path, grid=grid, nu=2e-3))
    assert re.lower == pytest.approx(100.0)
    assert re.upper == pytest.approx(100.0 * math.sqrt(3))


@pytest.mark.parametrize("change", ["two_bodies", "body_on_the_side", "moving_body",
                                    "nonuniform_inlet", "pressure_outlet", "slip_wall",
                                    "velocity_along_the_empty_direction"])
def test_an_external_flow_with_any_doubt_gives_none(tmp_path, change):
    kw: dict = {}
    if change == "two_bodies":
        kw["grid"] = _box(solid=SQUARE | {(14, 4, 0), (14, 5, 0)})
    elif change == "body_on_the_side":
        kw["grid"] = _box(solid=frozenset({(6, 0, 0), (7, 0, 0)}))
    elif change == "moving_body":
        kw["u"] = {"body": "type fixedValue; value uniform (0 0.1 0);"}
    elif change == "nonuniform_inlet":
        kw["u"] = {"inlet": "type fixedValue; value nonuniform List<vector> 10(" +
                   " ".join(["(1 0 0)"] * 10) + ");"}
    elif change == "pressure_outlet":
        kw["u"] = {"outlet": "type pressureInletOutletVelocity; value uniform (0 0 0);"}
    elif change == "slip_wall":
        kw["u"] = {"body": "type slip;"}
    else:
        kw["u"] = {"inlet": "type fixedValue; value uniform (1 0 0.1);"}
    assert _re(_external(tmp_path, nu=2e-3, **kw)) is None


#: The box's cells: 20 x 10, the square's 4 left out.
BOX_CELLS = 20 * 10 - len(SQUARE)


def _box_u(internal: str | None, fmt: str = "ascii") -> str:
    """The box's `0/U`, its inlet at 1 m/s, with `internal` as its internalField (None: none)."""
    line = "" if internal is None else f"internalField {internal};\n"
    return (f"FoamFile {{ version 2.0; format {fmt}; class volVectorField; object U; }}\n"
            "dimensions [0 1 -1 0 0 0 0];\n" + line + "boundaryField\n{\n"
            f"    inlet {{ {STREAM} }}\n    outlet {{ type zeroGradient; }}\n"
            "    sides { type symmetryPlane; }\n    body { type noSlip; }\n"
            "    frontAndBack { type empty; }\n}\n")


def _listed(*vectors: str, n: int = BOX_CELLS, count: int | None = None) -> str:
    """`nonuniform List<vector>` as v2406 writes it (Field::writeEntry, Field.C): `vectors` first,
    the rest of the n values the free stream's; `count` the length it states (n by default)."""
    values = [*vectors, *["(1 0 0)"] * (n - len(vectors))]
    return (f"nonuniform List<vector> \n{n if count is None else count}\n(\n"
            + "\n".join(values) + "\n)\n")


@pytest.mark.parametrize("internal, speed", [
    ("uniform (0 0 0)", 1.0), ("uniform (3 0 0)", 3.0), (_listed("(0 40 0)"), 40.0),
    (_listed("(-0.6 0.8 0)"), 1.0)], ids=["at rest", "uniform faster", "one cell faster",
                                          "as fast"])
def test_the_speed_bound_is_the_larger_of_the_inlets_and_the_start_fields(tmp_path, internal,
                                                                          speed):
    """The final review (of 8249ee4): the upper bound took the inlet's speed alone, though the
    field the run starts from may move faster (a wake an earlier run left, or a field mapped
    from a turbulent run), and is carried for many flow-through times. Its speed is the larger of
    the inlet's |U| and the start field's largest |U|, `uniform` or one value per cell; the lower
    bound stays the free stream's, the inlet's."""
    re = _re(_external(tmp_path, nu=2e-3, u_text=_box_u(internal)))
    assert re.speed == pytest.approx(speed)
    assert re.upper == pytest.approx(speed * 0.2 * math.sqrt(2) / 2e-3)
    assert re.lower == pytest.approx(100.0)


@pytest.mark.parametrize("internal, fmt", [
    (None, "ascii"),
    ("uniform (1 0)", "ascii"),
    (f"nonuniform List<vector> {BOX_CELLS}{{(1 0 0)}}", "ascii"),
    (_listed().replace("List<vector> ", ""), "ascii"),
    (_listed().replace("List<vector>", "List<scalar>"), "ascii"),
    (_listed(n=BOX_CELLS - 1), "ascii"),
    (_listed(count=BOX_CELLS + 1), "ascii"),
    (_listed("(1 0 x)"), "ascii"),
    (_listed("(1 0)"), "ascii"),
    (_listed("(nan 0 0)"), "ascii"),
    (_listed("((1 0 0))"), "ascii"),
    (_listed() + " extra", "ascii"),
    (_listed(), "binary"),
], ids=["absent", "not a vector", "a uniform list", "no List<vector>", "scalars",
        "not one per cell", "not its count", "not a number", "two components", "not finite",
        "nested", "something after it", "binary"])
def test_a_start_field_the_reader_cannot_read_whole_gives_no_number(tmp_path, internal, fmt):
    """v2406 reads an internalField as `uniform` and one value, or `nonuniform` and a list whose
    length must be the mesh's cells (Field::assign, Field.C: "Size ... is not equal to the
    expected length"). Read here: `uniform (x y z)`, or `nonuniform List<vector> N (...)` of N
    finite vectors, N the cells, in an ascii file. Any other form (a list written `N{...}`, a
    binary list) gives no number, as a value the reader cannot bound."""
    case = _external(tmp_path, nu=2e-3, u_text=_box_u(internal, fmt))
    assert _re(case) is None
    assert _re(_external(tmp_path / "control", nu=2e-3, u_text=_box_u(_listed()))) is not None


def test_the_viscosity_is_the_one_the_solver_says_it_took(tmp_path):
    """Newtonian by its own line (viscosityModelNew.C), and a laminar stress model of Stokes
    (laminarModel.C); icoFoam reads `nu` itself and prints neither."""
    assert _re(_external(tmp_path / "a", nu=2e-3)) is not None
    cross = MODELS.replace("Newtonian", "CrossPowerLaw")
    assert _re(_external(tmp_path / "b", nu=2e-3, models=cross)) is None
    maxwell = MODELS.replace("Stokes", "Maxwell")
    assert _re(_external(tmp_path / "c", nu=2e-3, models=maxwell)) is None
    assert _re(_external(tmp_path / "d", nu=2e-3, models="")) is None
    assert _re(_external(tmp_path / "e", nu=2e-3, solver="icoFoam", models="")) is not None


def test_a_macro_from_the_same_file_is_read_and_a_directive_or_a_merge_is_not(tmp_path):
    header = 'FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n'

    def u_file(inlet: str, extra: str = "") -> str:
        return (header + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 0 0);\n"
                + extra + "boundaryField\n{\n"
                f"    inlet {{ {inlet} }}\n    outlet {{ type zeroGradient; }}\n"
                "    sides { type symmetryPlane; }\n    body { type noSlip; }\n"
                "    frontAndBack { type empty; }\n}\n")

    got = _re(_external(tmp_path / "a", nu=2e-3,
                        u_text=u_file("type fixedValue; value $internalField;")))
    assert got is not None and got.lower == pytest.approx(100.0)
    assert _re(_external(tmp_path / "b", nu=2e-3, u_text=u_file(
        "type fixedValue; value uniform (1 0 0);", '#include "elsewhere"\n'))) is None
    assert _re(_external(tmp_path / "c", nu=2e-3, u_text=u_file(
        "type zeroGradient; $other;", "other { type fixedValue; value uniform (9 0 0); }\n"
    ))) is None


def test_a_value_the_case_writes_is_read_in_linear_time():
    """The case writes these values: a run of digits that a number pattern could split two ways
    would cost the square of its length. They are tokenised in one pass instead."""
    import time

    from courant.case.reynolds import _table, _vector

    t = time.perf_counter()
    assert _vector("(" + "1" * 90_000 + " 0)") is None
    assert _table("(" * 40_000 + "(0 " + "9" * 40_000) is None
    assert _table("(" + "(0 (1 0 0)) " * 5_000 + ")") is not None
    assert time.perf_counter() - t < 2.0


@pytest.mark.parametrize("field", ["U", "p"])
def test_a_field_written_with_macros_to_hold_the_reader_gives_no_number_at_once(tmp_path, field):
    """The review of b1fea1c: each `$` the reader resolves cost a pass over the whole file, so an
    internalField of 100,000 `$a` held `courant verify --checks next` for hours. More `$` than
    `_MAX_MACROS` in the files one reading reads gives no number, before any is resolved."""
    import time

    case, cfg, run = _external(tmp_path, nu=2e-3)
    assert _re((case, cfg, run)) is not None
    path = case / "0" / field
    text = path.read_text()
    start = text.index("internalField")
    path.write_text(text[:start] + "a 0;\ninternalField uniform " + " ".join(["$a"] * 100_000)
                    + text[text.index(";", start):])
    t = time.perf_counter()
    assert reynolds_number(case, CaseConfig.from_foam(str(case)), run) is None
    assert time.perf_counter() - t < 5.0


def _macro_p(case: Path) -> None:
    """The case's `0/p` with two `$`, both resolved: `internalField uniform $p0` and its outlet's
    `value $internalField`."""
    path = case / "0" / "p"
    path.write_text(path.read_text().replace("internalField uniform 0;",
                                             "p0 0;\ninternalField uniform $p0;")
                    .replace("type fixedValue; value uniform 0;",
                             "type fixedValue; value $internalField;"))


def test_a_macro_is_read_only_within_the_readings_bounds(tmp_path):
    """A `$` is resolved by passes over the file it is in: at most `_MAX_MACROS` of them in all
    the field files one reading reads (here `0/U` and `0/p`), in a file of at most
    `_MAX_MACRO_FILE` bytes on disk, comments and all. A longer file is read when it holds no
    `$`: a solver writes its fields expanded, a restart's per-cell lists among them."""
    from courant.case.reynolds import _MAX_MACRO_FILE, _MAX_MACROS

    padding = f'padding "{"x" * _MAX_MACRO_FILE}";\n'

    def u(n_a: int, extra: str = "") -> str:
        internal = "uniform " + "$a " * n_a + "$Uin"
        text = _box_u(internal).replace("dimensions", "a ;\nUin (1 0 0);\n" + extra
                                        + "dimensions")
        return text.replace(STREAM, "type fixedValue; value $internalField;")

    long = _box_u("uniform (1 0 0)").replace("dimensions", padding + "dimensions")
    comment = f'/* {"x" * _MAX_MACRO_FILE} */\n'              # stripped before it is read
    for name, text, p_macros, read in [
            ("short", u(0), False, True),
            ("long", long, False, True),
            ("long_macro", u(0, padding), False, False),
            ("long_comment_macro", u(0, comment), False, False),
            ("at", u(_MAX_MACROS - 2), False, True),             # 8 in 0/U
            ("past", u(_MAX_MACROS - 1), False, False),          # 9 in 0/U
            ("at_both", u(_MAX_MACROS - 4), True, True),         # 6 in 0/U, 2 in 0/p
            ("past_both", u(_MAX_MACROS - 3), True, False)]:     # 7 and 2
        case, cfg, run = _external(tmp_path / name, nu=2e-3, u_text=text)
        if p_macros:
            _macro_p(case)
        got = reynolds_number(case, CaseConfig.from_foam(str(case)), run)
        assert (got is not None) is read, name
        if read:
            assert got.lower == pytest.approx(100.0), name


def _split_box(tmp, rows: int, *, extra: str = "", front_and_back: bool = False,
               outlet: str = "type zeroGradient;", inlet: str | None = None, top: str = ""):
    """The box with each face of its inlet and of its outlet its own patch (`rows` of each), its
    lower and upper sides two symmetryPlane patches, and, with `front_and_back`, its empty sides
    two patches: 2 * rows + 4 patches, or 2 * rows + 5. In `0/U`: `extra`, entries added to the
    boundaryField; `outlet`, each outlet's condition, written as one pattern entry; `inlet`, where
    given, each inlet's, written so too; `top`, entries before the boundaryField."""
    sides = {**BOX_SIDES, "yhi": "top", **({"zlo": "front", "zhi": "back"} if front_and_back
                                            else {})}

    def split(s, i, j, k):
        return {"xlo": f"in{j}", "xhi": f"out{j}"}.get(s) or sides[s]

    grid = _structured(**{**BOX, "ys": np.linspace(0, 1, rows + 1)}, solid=frozenset(
        (i, j, 0) for i in (6, 7) for j in range(rows // 2 - 1, rows // 2 + 1)), side=split)
    names = [n for n, _, _ in grid[4]]
    kinds = {"top": "symmetryPlane", "front": "empty", "back": "empty"}
    types = {n: BOX_TYPES.get(n) or kinds.get(n, "patch") for n in names}
    case, _, run = _case(tmp, grid, types, {n: STREAM for n in names if n.startswith("in")}
                         | {"body": "type noSlip;"}, nu=2e-3)
    u = case / "0" / "U"
    head, body, tail = _boundary_field(u.read_text())
    shared = ("\n    out",) + (("\n    in",) if inlet is not None else ())
    body = "".join(block for block in re.split(r"(?=\n    \w)", body)
                   if not block.startswith(shared))           # one pattern entry each
    if inlet is not None:
        body += f'    "in.*" {{ {inlet} }}\n'
    start = head.rindex("boundaryField")
    u.write_text(head[:start] + top + head[start:] + body + f'    "out.*" {{ {outlet} }}\n'
                 + extra + tail)
    return case, CaseConfig.from_foam(str(case)), run


def test_a_mesh_of_many_patches_or_a_boundary_field_of_many_entries_gives_no_number(tmp_path,
                                                                                     monkeypatch):
    """The review of b1fea1c: each patch's block was looked up by a pass over the boundaryField,
    so 20,000 one-face patches held the reader. A rank's mesh of more than `_MAX_PATCHES` patches
    (its constraint patches counted), or a boundaryField of more than `_MAX_ENTRIES` `{}` blocks,
    gives no number, where the same case gives one with the bound raised. A long boundaryField with few entries is read: a restart writes each
    patch's values as a per-face list."""
    from courant.case.reynolds import _MAX_ENTRIES, _MAX_PATCHES

    rows = (_MAX_PATCHES - 4) // 2                  # 2 * 14 + 4 = 32 patches
    assert _re(_split_box(tmp_path / "at", rows)) is not None
    past = _split_box(tmp_path / "past", rows, front_and_back=True)                 # 33
    assert _re(past) is None
    from courant.case import reynolds

    monkeypatch.setattr(reynolds, "_MAX_PATCHES", _MAX_PATCHES + 1)
    assert _re(past) is not None
    monkeypatch.undo()
    extra = "".join(f"    unused{i} {{ type zeroGradient; }}\n" for i in range(_MAX_ENTRIES))
    assert _re(_split_box(tmp_path / "many", 4, extra=extra)) is None             # 9 + 64
    few = "".join(f"    unused{i} {{ type zeroGradient; }}\n" for i in range(_MAX_ENTRIES - 16))
    assert _re(_split_box(tmp_path / "few", 4, extra=few)) is not None            # 9 + 48
    assert _re(_split_box(tmp_path / "long", 4, outlet="type zeroGradient;" + " " * 300_000)
               ) is not None


def test_a_block_every_patch_shares_is_read_once(tmp_path, monkeypatch):
    """The review of 0d4930f: a pattern block every patch takes, its value's macros passes over the
    file, was resolved again for each patch (20 passes a patch with four `$internalField` of four
    `$x`: 34 s on one rank, 97 s on three). Each distinct block is read once: here 13 inlets share
    `"in.*"`, whose `value $internalField` is `uniform $u`."""
    import time

    from courant.spec import deck

    calls = []
    resolve = deck._resolve_macros
    monkeypatch.setattr(deck, "_resolve_macros",
                        lambda *a, **k: calls.append(1) or resolve(*a, **k))
    words = f'padding "{"x" * 90_000}";\nu (1 0 0);\n'
    case, cfg, run = _split_box(tmp_path, 13, inlet="type fixedValue; value $internalField;",
                                top=words)
    u = case / "0" / "U"
    u.write_text(u.read_text().replace("internalField uniform (0 0 0);",
                                       "internalField uniform $u;"))
    cfg = CaseConfig.from_foam(str(case))
    t = time.perf_counter()
    got = reynolds_number(case, cfg, run)
    assert time.perf_counter() - t < 3.0
    assert got is not None and got.lower == pytest.approx(1.0 * (2 / 13) / 2e-3)    # 2 cells
    # U's internalField (2 calls), the shared block once (3), p's internalField (1); read once a
    # patch, 13 inlets would make it 42
    assert len(calls) == 6


def test_past_the_readings_rank_or_byte_bound_there_is_no_number(tmp_path, monkeypatch):
    """`_MAX_RANKS` ranks, and `_MAX_READ` bytes on disk in all the files one reading reads (every
    rank's, `U` and `p` together): past either, no number."""
    from courant.case import reynolds

    case = _cylinder(tmp_path)                      # two ranks
    log = (case / "log.pimpleFoam").read_text()
    run = Run(log, "log.pimpleFoam", (), exit_code=0)
    cfg = CaseConfig.from_foam(str(case))
    assert reynolds_number(case, cfg, run) is not None
    monkeypatch.setattr(reynolds, "_MAX_RANKS", 1)
    assert reynolds_number(case, cfg, run) is None
    monkeypatch.undo()
    size = sum((case / f"processor{r}" / "0" / f).stat().st_size for r in (0, 1) for f in "Up")
    monkeypatch.setattr(reynolds, "_MAX_READ", size)
    assert reynolds_number(case, cfg, run) is not None
    monkeypatch.setattr(reynolds, "_MAX_READ", size - 1)
    assert reynolds_number(case, cfg, run) is None


def test_the_radiation_dictionary_is_charged_to_the_readings_budget(tmp_path, monkeypatch):
    """The review of c9c2eae: `constant/radiationProperties` was read whole outside any bound. It
    is charged to the reading's bytes with `0/U` and `0/T`; past them it is not buoyancy alone."""
    from courant.case import reynolds

    case, run = _cavity(tmp_path, extra={"constant/radiationProperties": _radiation("none")})
    assert _alone(case, run)
    size = sum((case / rel).stat().st_size
               for rel in ("constant/radiationProperties", "0/U", "0/T"))
    monkeypatch.setattr(reynolds, "_MAX_READ", size)
    assert _alone(case, run)
    monkeypatch.setattr(reynolds, "_MAX_READ", size - 1)
    assert not _alone(case, run)


def test_a_file_past_the_budget_is_never_opened(tmp_path, monkeypatch):
    """The review of c9c2eae: the budget was charged after the file was read and its comments
    stripped, so a field file of a gigabyte of comment was read whole. Each file is sized on
    disk first: one past the budget gives no number, and is never opened."""
    from courant.case.reynolds import _MAX_READ
    from courant.spec import deck

    case, cfg, run = _external(tmp_path, nu=2e-3)
    u = case / "0" / "U"
    u.write_text(u.read_text().replace("dimensions", f'/* {"x" * _MAX_READ} */\ndimensions'))
    opened = []
    read = deck._read_text
    monkeypatch.setattr(deck, "_read_text", lambda c, rel: opened.append(rel) or read(c, rel))
    assert reynolds_number(case, cfg, run) is None
    assert "0/U" not in opened


def test_a_reading_at_its_byte_bound_takes_seconds(tmp_path, monkeypatch):
    """The review of c9c2eae: at 16,000,000 characters a reading of distinct blocks took about
    37 s. Each of the 14 inlet blocks of `0/U` here is its own (a `tag` entry apart), padded to
    about 400,000 bytes in all, so each is read (16 conditions: the inlets, the body and the
    outlets' one pattern). The reading is linear in its bytes: 3.3 s on the laptop scaled to
    4,000,000 bytes, 13 s to the old 16,000,000 (26 Oct 2026). The bound is pinned, and the
    scaled time held under 20 s, so a slower runner does not fail it and a cost per byte four
    times the laptop's does."""
    import time

    from courant.case import reynolds
    from courant.case.reynolds import _MAX_READ

    calls = []
    condition = reynolds._condition
    monkeypatch.setattr(reynolds, "_condition", lambda b, r: calls.append(1) or condition(b, r))
    case, cfg, run = _split_box(tmp_path, 14)
    u = case / "0" / "U"
    head, *rest = u.read_text().split(STREAM)
    pad = ";" * (400_000 // len(rest))
    u.write_text(head + "".join(f"{STREAM} tag_{i} 1;{pad}{tail}" for i, tail in enumerate(rest)))
    size = u.stat().st_size + (case / "0" / "p").stat().st_size
    t = time.perf_counter()
    assert reynolds_number(case, cfg, run) is not None
    elapsed = time.perf_counter() - t
    assert len(calls) == 16
    assert _MAX_READ <= 4_000_000
    assert elapsed * _MAX_READ / size < 20.0


def test_a_finite_cylinder_across_a_3d_flow_is_not_bluff_on_the_ratio(tmp_path):
    """A square prism across the flow, its span three times its side (a finite cylinder's
    shape): its extent is sqrt(11) times its side, more than `_BLUFF` times its width across the
    flow, so it gives no number, where the cube gives one
    (`test_a_body_in_three_dimensions_takes_its_projected_widths`)."""
    zs = np.linspace(0, 1, 11)
    prism = frozenset((i, j, k) for i in (6, 7) for j in (4, 5) for k in range(2, 8))
    sides = {**BOX_SIDES, "zlo": "sides", "zhi": "sides"}
    grid = _structured(**{**BOX, "zs": zs}, solid=prism, side=lambda s, i, j, k: sides[s])
    assert _re(_external(tmp_path, grid=grid, nu=2e-3)) is None


def test_widths_of_a_point_set():
    square = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)], dtype=float)
    assert _widths(square) == pytest.approx((1.0, math.sqrt(2)))
    ring = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64,
                                                                     endpoint=False)])
    least, most = _widths(ring)
    assert most == pytest.approx(2.0) and 2 * math.cos(math.pi / 64) <= least + 1e-12 <= 2.0


def _brute_widths(xy):
    """Every pair's distance, and every direction's width over a fine set (the least width is
    attained with one caliper on a hull edge, so it is also each edge's farthest point)."""
    xy = np.asarray(xy, dtype=float)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)).max()
    least = math.inf
    for a in range(len(xy)):
        for b in range(len(xy)):
            e = xy[b] - xy[a]
            n = np.linalg.norm(e)
            if n == 0:
                continue
            dist = (e[0] * (xy[:, 1] - xy[a, 1]) - e[1] * (xy[:, 0] - xy[a, 0])) / n
            if dist.min() >= -1e-12:                         # every point on one side: a hull edge
                least = min(least, float(dist.max()))
    return least, float(d)


@pytest.mark.parametrize("seed", range(6))
def test_widths_match_every_pair_and_every_edge(seed):
    rng = np.random.default_rng(seed)
    xy = rng.normal(size=(40, 2)) * rng.uniform(0.1, 3.0, size=2)
    assert _widths(xy) == pytest.approx(_brute_widths(xy), rel=1e-9)


def test_widths_are_linear_in_the_hull():
    """The review of 26 Oct 2026: for each hull vertex, a norm over all of them, so a cylinder
    meshed with 60,000 wall faces (every wall point on the hull) took about 3.6e9 operations on
    `--checks next`. Rotating calipers walk the hull once."""
    import time

    a = np.linspace(0, 2 * math.pi, 60_000, endpoint=False)
    ring = np.column_stack([np.cos(a), np.sin(a)]) * 0.06
    t = time.perf_counter()
    least, most = _widths(ring)
    assert time.perf_counter() - t < 2.0
    assert most == pytest.approx(0.12) and least == pytest.approx(0.12, rel=1e-8)


def test_a_case_without_its_files_gives_none(tmp_path):
    case, cfg, run = _external(tmp_path, nu=2e-3)
    assert reynolds_number(None, cfg, run) is None
    (case / "constant" / "polyMesh" / "faces").unlink()
    assert reynolds_number(case, cfg, run) is None


# ------------------------------------------------------------------ the decision
def _residual(seq, time, res, courant=0.5):
    return Event(run_id="r", seq=seq, kind=EventKind.RESIDUAL,
                 payload=ResidualPayload(time=time, initial_residuals=res, courant_max=courant))


HIGH = [_residual(i, 0.1 * i, {"Ux": 3e-3, "p": 8e-3}) for i in range(1, 4)]
CRASH = Event(run_id="r", seq=9, kind=EventKind.CRASH, payload=CrashPayload(signal="SIGFPE"))
TRANSIENT = CaseConfig(solver="pimpleFoam", ddt_scheme="Euler", turbulence_type="laminar")
STEADY = CaseConfig(solver="simpleFoam", ddt_scheme="steadyState", turbulence_type="laminar")


def _number(upper, lower):
    return Reynolds(upper=upper, lower=lower, speed=1.0, length=0.1, nu=1e-3)


def _run(config=TRANSIENT, events=HIGH, reynolds=None, checks="next", **kw):
    calls = []

    def read():
        calls.append(1)
        return reynolds

    ctx = CheckContext(run_id="r", events=list(events), config=config, checks=checks,
                       read_reynolds=read, **kw)
    found = [f for f in PhysicsChecker().run(ctx) if f.id == "physics.laminar_unstable"]
    return found, ctx, calls


def test_below_the_laminar_bound_the_residual_history_is_set_aside_and_said():
    found, ctx, calls = _run(reynolds=_number(100.0, None))
    assert found == [] and calls == [1]
    said = laminar_spared_sentence(ctx.laminar)
    assert "Reynolds number is at most 100" in said and "below 150" in said
    assert "Roshko 1954" in said and "end above the flat 1e-3 bar" in said


def test_under_the_default_nothing_is_read_and_the_finding_is_060s():
    found, ctx, calls = _run(reynolds=_number(100.0, 100.0), checks="default")
    [f] = found
    assert calls == [] and ctx.laminar.numbers == () and laminar_spared_sentence(ctx.laminar) is None
    assert f.detail == ("laminar turbulence type, Courant number within its stability limit, and "
                        "the solve never reached tolerance")
    assert f.value is None and f.threshold is None


def test_above_the_turbulent_bound_the_finding_cites_the_lower_bound():
    [f], ctx, _ = _run(reynolds=_number(5e5, 4e5))
    assert f.detail.endswith("; the case's Reynolds number is at least 4e+05 (flow past one "
                             "bluff body, on its extent across the flow), above 300, where a "
                             "circular cylinder's wake is irregular (Roshko 1954)")
    assert (f.value, f.threshold) == (4e5, RE_EXTERNAL_TURBULENT)
    assert laminar_spared_sentence(ctx.laminar) is None


@pytest.mark.parametrize("number", [_number(200.0, 200.0), _number(400.0, None)])
def test_between_the_bounds_or_without_a_lower_bound_nothing_changes(number):
    [f], ctx, _ = _run(reynolds=number)
    [g], _, _ = _run(reynolds=number, checks="default")
    assert f == g and laminar_spared_sentence(ctx.laminar) is None


def test_with_no_number_nothing_changes():
    [f], _, calls = _run(reynolds=None)
    [g], _, _ = _run(checks="default")
    assert f == g and calls == [1]


@pytest.mark.parametrize("config, events", [
    (STEADY, HIGH),                                       # judged on the bar: VER-5b is moot
    (TRANSIENT, [_residual(1, 0.1, {"Ux": 1e-5, "p": 1e-5})]),       # nothing to set aside
    (TRANSIENT, [_residual(1, 0.1, {"Ux": 0.5}, courant=9.0)]),      # a Courant breach
    (TRANSIENT.model_copy(update={"turbulence_type": "RAS"}), HIGH),  # not laminar
])
def test_the_files_are_read_only_for_a_transient_laminar_run_the_finding_would_reject(config,
                                                                                    events):
    found, ctx, calls = _run(config=config, events=events, reynolds=_number(100.0, 100.0))
    assert calls == [] and found == _run(config=config, events=events, checks="default")[0]


def test_a_crash_is_not_residual_history_and_keeps_the_finding():
    [f], ctx, _ = _run(events=[*HIGH, CRASH], reynolds=_number(100.0, 100.0))
    assert "terminated abnormally" in f.detail and laminar_spared_sentence(ctx.laminar) is None


def test_every_number_must_be_below_its_bound():
    numbers = regime_numbers(None, _number(100.0, 100.0))
    assert [n.below for n in numbers] == [True]
    [re] = regime_numbers(None, _number(149.9, None))
    assert (re.laminar, re.turbulent) == (RE_EXTERNAL_LAMINAR, RE_EXTERNAL_TURBULENT)
    assert re.flow == "flow past one bluff body" and not regime_numbers(None, _number(150, 0))[0].below


# ------------------------------------------------------------------ the Rayleigh number (VER-17's)
READ_G = "Reading g\n"
AIR = {"mu": 1.8e-5, "Pr": 0.7, "molWeight": 28.9}


def _layer(height, t_cold, t_hot, cell):
    box = ((0.0, 0.0, 0.0), (height, height, height))
    floor = ((0.0, 0.0, 0.0), (height, 0.0, height))
    ceiling = ((0.0, height, 0.0), (height, height, height))
    cfg = TRANSIENT.model_copy(update={
        "solver": "buoyantPimpleFoam", "gravity": (0.0, -9.81, 0.0), "transport": dict(AIR),
        "equation_of_state": "perfectGas", "pressure": 1e5,
        "fixed_temperature_range": (t_cold, t_hot), "hot_faces_box": floor,
        "cold_faces_box": ceiling})
    n = round(height / cell) ** 3
    mesh = MeshQuality(n_cells=n, bounding_box=box, total_volume=height ** 3,
                       solution_directions=(1, 1, 1))
    return cfg, mesh


def test_a_still_layer_heated_from_below_sets_the_residuals_aside_on_its_upper_bound():
    """2 mm of air, 1 K across it, and nothing but buoyancy moving it (the case's files say so,
    `buoyancy_alone`, asked with the temperatures the estimate read): physics.laminar_regime's
    reading bounds the Rayleigh number above at about 0.9 (the domain's height, the cold gas's
    viscosity, |g| dT / T_cold), far below the 1708 at which such a layer begins to convect."""
    cfg, mesh = _layer(0.002, 300.0, 301.0, 0.0005)
    asked = []
    found, ctx, _ = _run(config=cfg, mesh=mesh, log_text=READ_G,
                         read_still=lambda temps: asked.append(temps) or True)
    [ra] = ctx.laminar.numbers
    assert ra.name == "Rayleigh" and ra.upper < RA_ONSET and ra.lower <= ra.upper
    assert found == [] and asked == [(300.0, 301.0)]
    assert "Rayleigh number is at most" in laminar_spared_sentence(ctx.laminar)


@pytest.mark.parametrize("still", [False, None], ids=["something else moves it", "no case"])
def test_a_rayleigh_number_spares_nothing_unless_buoyancy_alone_moves_the_fluid(still):
    """Below 1708 buoyancy drives no convection; it says nothing of a flow something else drives
    (the review of 26 Oct 2026: a lid-driven cavity over a warm floor, Re about 13,000, Ra about
    100). Where the case's files do not show that nothing else moves the fluid, or there are no
    files (a log and a sidecar), the Rayleigh number has no upper bound, and the finding stands."""
    cfg, mesh = _layer(0.002, 300.0, 301.0, 0.0005)
    kw = {} if still is None else {"read_still": lambda temps: still}
    [f], ctx, _ = _run(config=cfg, mesh=mesh, log_text=READ_G, **kw)
    [ra] = ctx.laminar.numbers
    assert ra.upper == math.inf and not ra.below and ctx.laminar.spared == ()
    assert "never reached tolerance" in f.detail and laminar_spared_sentence(ctx.laminar) is None


def test_a_turbulent_layer_is_cited_with_physics_laminar_regimes_lower_bound(monkeypatch):
    """The Rayleigh number is physics.laminar_regime's, estimated once for both findings."""
    from courant.diagnosis import checkers

    real, estimates = checkers.buoyant_regime, []
    monkeypatch.setattr(checkers, "buoyant_regime",
                        lambda *a: estimates.append(1) or real(*a))
    cfg, mesh = _layer(5.0, 290.0, 320.0, 0.5)
    ctx = CheckContext(run_id="r", events=list(HIGH), config=cfg, checks="next", mesh=mesh,
                       log_text=READ_G)
    found = {f.id: f for f in PhysicsChecker().run(ctx)}
    f = found["physics.laminar_unstable"]
    [ra] = ctx.laminar.numbers
    assert ra.lower == ctx.buoyant.rayleigh > RA_TRANSITION
    assert f.threshold == RA_TRANSITION and "Rayleigh number is at least" in f.detail
    assert "physics.laminar_regime" in found and estimates == [1]


def test_without_reading_g_there_is_no_rayleigh_number():
    cfg, mesh = _layer(0.002, 300.0, 301.0, 0.0005)
    [f], ctx, _ = _run(config=cfg, mesh=mesh, log_text="")
    assert ctx.laminar.numbers == () and "never reached tolerance" in f.detail


# ------------------------------------------------------------------ a buoyant cavity, as files
#: A 2-D square cavity 1 cm wide, 8 x 8 cells: a floor held at 301 K, a lid at 300 K above it,
#: adiabatic sides, air at 1 bar (buoyantPimpleFoam's hotRoom properties).
CAVITY = dict(xs=np.linspace(0, 0.01, 9), ys=np.linspace(0, 0.01, 9), zs=(0.0, 0.001))
CAVITY_SIDES = {"xlo": "sides", "xhi": "sides", "ylo": "floor", "yhi": "lid",
                "zlo": "frontAndBack", "zhi": "frontAndBack"}
CAVITY_TYPES = {"floor": "wall", "lid": "wall", "sides": "wall", "frontAndBack": "empty"}
CAVITY_U = {"floor": "type noSlip;", "lid": "type noSlip;", "sides": "type noSlip;"}
CAVITY_T = {"floor": "type fixedValue; value uniform 301;",
            "lid": "type fixedValue; value uniform 300;", "sides": "type zeroGradient;"}
THERMO = HEADER % ("dictionary", "thermophysicalProperties") + """\
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}
mixture
{
    specie { molWeight 28.9; }
    thermodynamics { Cp 1000; Hf 0; }
    transport { mu 1.8e-05; Pr 0.7; }
}
"""
BUOYANT_LOG = """\
Exec   : buoyantPimpleFoam
nProcs : 1

Create time

Create mesh for time = 0

PIMPLE: Operating solver in PISO mode

Reading thermophysical properties

Reading field U

Reading/calculating face flux field phi

Creating turbulence model

Selecting turbulence model type laminar
Selecting laminar stress model Stokes

Reading g

Reading field p_rgh

No MRF models present

Radiation model not active: radiationProperties not found
Selecting radiationModel none
No finite volume options present

Starting time loop

{steps}End
"""
BUOYANT_STEP = """\
Courant Number mean: 0.01 max: 0.2
deltaT = 0.01
Time = {t:.2f}

PIMPLE: iteration 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
DILUPBiCGStab:  Solving for Ux, Initial residual = 0.005, Final residual = 1e-08, No Iterations 2
DILUPBiCGStab:  Solving for Uy, Initial residual = 0.005, Final residual = 1e-08, No Iterations 2
DILUPBiCGStab:  Solving for h, Initial residual = 0.004, Final residual = 1e-08, No Iterations 2
GAMG:  Solving for p_rgh, Initial residual = 0.006, Final residual = 1e-08, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 1e-09, global = 1e-11, cumulative = 1e-11
ExecutionTime = {t:.2f} s  ClockTime = 0 s

"""


def _cavity(tmp: Path, *, u=None, t=None, types=None, sides=None, t_internal="uniform 300",
            u_internal="uniform (0 0 0)", extra=None):
    """A buoyantPimpleFoam case directory written from scratch in v2406's shape: the mesh, `0/U`,
    `0/T`, `0/p`, `0/p_rgh`, `constant/g`, the thermophysical and turbulence dictionaries,
    controlDict, fvSchemes (Euler), fvSolution, and a log whose residuals end above the flat bar
    at a Courant number of 0.2 (a synthetic log in the solver's shape)."""
    named = {**CAVITY_SIDES, **({} if callable(sides) else sides or {})}
    side = sides if callable(sides) else (lambda s, i, j, k: named[s])
    types = {**CAVITY_TYPES, **(types or {})}
    points, faces, owner, neighbour, patches = _structured(**CAVITY, side=side)
    boundary = tuple(Patch(name=n, type=types[n], start=s, n=c) for n, s, c in patches)
    write_polymesh(Mesh(points=points, faces=tuple(faces), owner=owner, neighbour=neighbour,
                        boundary=boundary), tmp)
    own = lambda name, kind: f"type {kind};"  # noqa: E731
    wall_p = lambda name, kind: (f"type {kind};" if kind in CONSTRAINT  # noqa: E731
                                 else "type calculated; value uniform 100000;")
    wall_prgh = lambda name, kind: (f"type {kind};" if kind in CONSTRAINT  # noqa: E731
                                    else "type fixedFluxPressure; value uniform 100000;")
    fields = {
        "U": ("volVectorField", "[0 1 -1 0 0 0 0]", u_internal,
              _entries(patches, types, {**CAVITY_U, **(u or {})}, own)),
        "T": ("volScalarField", "[0 0 0 1 0 0 0]", t_internal,
              _entries(patches, types, {**CAVITY_T, **(t or {})}, own)),
        "p": ("volScalarField", "[1 -1 -2 0 0 0 0]", "uniform 100000",
              _entries(patches, types, {}, wall_p)),
        "p_rgh": ("volScalarField", "[1 -1 -2 0 0 0 0]", "uniform 100000",
                  _entries(patches, types, {}, wall_prgh)),
    }
    for name, (cls, dims, internal, entries) in fields.items():
        _write(tmp / "0" / name, HEADER % (cls, name) + f"dimensions {dims};\n"
               f"internalField {internal};\nboundaryField\n{{\n{entries}}}\n")
    _write(tmp / "constant" / "g", HEADER % ("uniformDimensionedVectorField", "g")
           + "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n")
    _write(tmp / "constant" / "thermophysicalProperties", THERMO)
    _write(tmp / "constant" / "turbulenceProperties",
           HEADER % ("dictionary", "turbulenceProperties") + "simulationType laminar;\n")
    _write(tmp / "system" / "controlDict", HEADER % ("dictionary", "controlDict")
           + "application buoyantPimpleFoam;\nstartFrom startTime;\nstartTime 0;\n"
           "stopAt endTime;\nendTime 0.05;\ndeltaT 0.01;\nwriteControl timeStep;\n"
           "writeInterval 100;\n")
    _write(tmp / "system" / "fvSchemes", HEADER % ("dictionary", "fvSchemes")
           + "ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\n"
           "divSchemes { default none; div(phi,U) Gauss linear; div(phi,h) Gauss linear; "
           "div(phi,K) Gauss linear; div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear; }\n"
           "laplacianSchemes { default Gauss linear corrected; }\n"
           "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
    _write(tmp / "system" / "fvSolution", HEADER % ("dictionary", "fvSolution")
           + "solvers\n{\n    \"rho.*\" { solver diagonal; }\n"
           "    p_rgh { solver GAMG; smoother DIC; tolerance 1e-8; relTol 0.01; }\n"
           "    p_rghFinal { $p_rgh; relTol 0; }\n"
           "    \"(U|h)\" { solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0.1; }\n"
           "    \"(U|h)Final\" { $U; relTol 0; }\n}\n"
           "PIMPLE { momentumPredictor yes; nOuterCorrectors 1; nCorrectors 2; "
           "nNonOrthogonalCorrectors 0; }\n")
    for rel, text in (extra or {}).items():
        _write(tmp / rel, text)
    log = BUOYANT_LOG.format(steps="".join(BUOYANT_STEP.format(t=0.01 * i) for i in range(1, 6)))
    _write(tmp / "log.buoyantPimpleFoam", log)
    return tmp, Run(log, "log.buoyantPimpleFoam", (), exit_code=0)


#: The review's heat-flux variant: half the floor held at 300.000001 K, the other half a
#: fixedGradient heater, which `fixed_temperature_range` (fixedValue walls only) does not count.
HEATER = dict(
    sides=lambda s, i, j, k: "heater" if s == "ylo" and i >= 4 else CAVITY_SIDES[s],
    types={"heater": "wall"}, u={"heater": "type noSlip;"},
    t={"floor": "type fixedValue; value uniform 300.000001;",
       "heater": "type fixedGradient; gradient uniform 1000;"})


def _alone(case: Path, run: Run) -> bool:
    from courant.case.reynolds import buoyancy_alone

    cfg = CaseConfig.from_foam(str(case), buoyancy=True)
    return buoyancy_alone(case, cfg, run, cfg.fixed_temperature_range)


def test_a_closed_cavity_heated_below_is_moved_by_buoyancy_alone(tmp_path):
    """Every patch a still wall or empty, the fluid at rest at the start and within the held
    temperatures, the walls held (fixedValue) or adiabatic (zeroGradient: gradientEnergy sets
    the energy's gradient from T's, gradientEnergyFvPatchScalarField.C), no MRF, fvOption or
    moving mesh."""
    case, run = _cavity(tmp_path)
    assert CaseConfig.from_foam(str(case), buoyancy=True).fixed_temperature_range == (300, 301)
    assert _alone(case, run)


@pytest.mark.parametrize("change", [
    "moving_lid", "inlet", "heat_flux_heater", "external_heat_flux", "lumped_mass_lid",
    "hot_interior", "nonuniform_interior", "moving_interior", "symmetry_top", "mrf",
    "fv_option_heat_source", "moving_mesh"])
def test_anything_else_that_moves_or_heats_the_fluid_is_not_buoyancy_alone(tmp_path, change):
    """The review of 26 Oct 2026: a moving lid (the mixed-convection cavity) or an inlet drives the
    flow whatever the Rayleigh number; a heater that is not a held wall (fixedGradient,
    externalWallHeatFlux, lumpedMassWallTemperature), a hotter interior or an fvOption heat
    source heats the fluid past the temperatures the bound counts; a symmetry plane (a mirror of
    the domain along g) or a moving mesh, an MRF zone or an fvOption moves it past the bound."""
    kw: dict = {}
    if change == "moving_lid":
        kw["u"] = {"lid": "type fixedValue; value uniform (20 0 0);"}
    elif change == "inlet":
        kw.update(sides={"xlo": "inlet"}, types={"inlet": "patch"},
                  u={"inlet": "type fixedValue; value uniform (0.01 0 0);"},
                  t={"inlet": "type fixedValue; value uniform 300;"})
    elif change == "heat_flux_heater":
        kw.update(HEATER)
    elif change == "external_heat_flux":
        kw["t"] = {"floor": "type externalWallHeatFlux; mode power; Q 10; kappaMethod "
                            "fluidThermo; value uniform 300;"}
    elif change == "lumped_mass_lid":
        kw["t"] = {"lid": "type lumpedMassWallTemperature; kappaMethod fluidThermo; mass 1000; "
                          "Cp 4100; value uniform 300;"}
    elif change == "hot_interior":
        kw["t_internal"] = "uniform 310"
    elif change == "nonuniform_interior":
        kw["t_internal"] = "nonuniform List<scalar> 64(" + " ".join(["300.5"] * 64) + ")"
    elif change == "moving_interior":
        kw["u_internal"] = "uniform (0.5 0 0)"
    elif change == "symmetry_top":
        kw.update(types={"lid": "symmetryPlane"})
        kw["t"] = {"lid": "type symmetryPlane;"}
        kw["u"] = {"lid": "type symmetryPlane;"}
    elif change == "mrf":
        kw["extra"] = {"constant/MRFProperties": MRF}
    elif change == "fv_option_heat_source":
        kw["extra"] = {"system/fvOptions": HEADER % ("dictionary", "fvOptions") + (
            "heater { type scalarSemiImplicitSource; selectionMode all; volumeMode absolute; "
            "sources { h (10 0); } }\n")}
    else:
        kw["extra"] = {"constant/dynamicMeshDict": HEADER % ("dictionary", "dynamicMeshDict")
                       + "dynamicFvMesh dynamicMotionSolverFvMesh;\n"}
    case, run = _cavity(tmp_path, **kw)
    assert not _alone(case, run)


def _radiation(model: str) -> str:
    """`constant/radiationProperties` with `radiationModel` set to `model` (v2406's
    radiationModelNew.C reads that entry; the coefficients are the model's)."""
    return HEADER % ("dictionary", "radiationProperties") + f"radiationModel {model};\n"


SOLAR = _radiation("solarLoad") + (
    "solarLoadCoeffs { sunDirectionModel sunDirConstant; sunDirection (0 -1 0); "
    "sunLoadModel sunLoadConstant; directSolarRad 1000; diffuseSolarRad 0; "
    "useReflectedRays false; spectralDistribution (1); }\n")


@pytest.mark.parametrize("change", ["P1", "fvDOM", "solarLoad", "macro", "directive",
                                    "no_entry", "unreadable", "log"])
def test_a_radiation_model_is_not_buoyancy_alone(tmp_path, change):
    """The final review (of 8249ee4): ESI's buoyant solvers build their radiation model in the
    solver, not as an fvOption. buoyantPimpleFoam's and buoyantSimpleFoam's createFields.H
    include createRadiationModel.H (`radiation::radiationModel::New(thermo.T())`),
    buoyantBoussinesqPimpleFoam's createIncompressibleRadiationModel.H; `New` reads
    `radiationModel` from `constant/radiationProperties` and prints "Selecting radiationModel"
    (radiationModelNew.C), and the energy equation takes its source (EEqn.H `radiation->Sh(thermo,
    he)`, TEqn.H `radiation->ST(rhoCpRef, T)`). A model heats the fluid and the walls past the
    held temperatures the Rayleigh number counts (a solarLoad from outside). A file whose
    `radiationModel` is anything but `none` as written (a macro, a directive, no such entry, a
    file that cannot be read), or a log line naming another model, is not buoyancy alone."""
    extra = {"P1": _radiation("P1"), "fvDOM": _radiation("fvDOM"), "solarLoad": SOLAR,
             "macro": _radiation("$model") + "model none;\n",
             "directive": '#include "solar"\n' + _radiation("none"),
             "no_entry": HEADER % ("dictionary", "radiationProperties") + "solverFreq 1;\n"}
    case, run = _cavity(tmp_path, extra={"constant/radiationProperties": extra[change]}
                        if change in extra else None)
    assert _alone(case, run) is (change in ("unreadable", "log"))
    if change == "unreadable":
        (case / "constant" / "radiationProperties").mkdir(parents=True)
    elif change == "log":
        run = Run(run.log_text.replace("Selecting radiationModel none",
                                       "Selecting radiationModel P1"), run.log_name, (),
                  exit_code=0)
    assert not _alone(case, run)


@pytest.mark.parametrize("written", [None, "none"])
def test_no_radiation_model_is_still_buoyancy_alone(tmp_path, written):
    """Without the file v2406 prints "Radiation model not active: radiationProperties not found"
    and "Selecting radiationModel none" (radiationModelNew.C), as the cavity's log does; `none`
    (noRadiation) adds no source. Either way the cavity is moved by buoyancy alone."""
    extra = None if written is None else {"constant/radiationProperties":
                                          _radiation(written)}
    case, run = _cavity(tmp_path, extra=extra)
    assert "Selecting radiationModel none" in run.log_text and _alone(case, run)


def test_the_still_cavity_under_a_solar_load_stays_rejected_under_next(tmp_path):
    """The final review's buoyant case, as a full verdict: the still cavity (its Rayleigh number's
    bound below 1708, set aside under `--checks next` at 8249ee4) with a solarLoad radiation
    model, which heats it from outside the held walls. It is rejected on
    physics.laminar_unstable, as by default, and the verdict lists the file it read."""
    from courant.verify import verify_path

    case, _ = _cavity(tmp_path, extra={"constant/radiationProperties": SOLAR})
    v, d = verify_path(case, checks="next"), verify_path(case, checks="default")
    assert (v.verdict, v.rule) == (d.verdict, d.rule) == ("reject", "VER-5.4")
    assert "physics.laminar_unstable" in [r.code for r in v.reasons]
    assert not any("not raised" in s for s in v.not_assessed)
    assert "constant/radiationProperties" in {f.path for f in v.run.inputs}


def test_the_mixed_convection_cavity_stays_rejected_under_next(tmp_path):
    """The review's blocking case, as a case directory: the cavity's lid moving at 20 m/s over the
    floor held 1 K warmer (Re about 13,000 on the lid; the Rayleigh number's bound about 100,
    below 1708), its residuals ending above the flat bar. At 36c2a6d it moved from reject
    (VER-5.4) to accept under `--checks next`; it is rejected on physics.laminar_unstable, as by
    default. The same cavity with the lid at rest is set aside on its Rayleigh number."""
    from courant.verify import verify_path

    moving, _ = _cavity(tmp_path / "moving", u={"lid": "type fixedValue; value uniform (20 0 0);"})
    for checks in ("default", "next"):
        v = verify_path(moving, checks=checks)
        assert v.verdict == "reject" and v.rule == "VER-5.4", (checks, v.verdict, v.rule)
        assert "physics.laminar_unstable" in [r.code for r in v.reasons]
        assert not any("not raised" in s for s in v.not_assessed)
    still, _ = _cavity(tmp_path / "still")
    v = verify_path(still, checks="next")
    assert "physics.laminar_unstable" not in [r.code for r in v.reasons]
    [said] = [s for s in v.not_assessed if "not raised" in s]
    assert "its Rayleigh number is at most" in said and "below 1707.76" in said
    heater, _ = _cavity(tmp_path / "heater", **HEATER)
    v = verify_path(heater, checks="next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.4")
    assert "physics.laminar_unstable" in [r.code for r in v.reasons]


# ------------------------------------------------------------------ the laminar cylinder, solved
CYLINDER = REAL / "laminar_cylinder" / "run"
D, U0, NU = 0.12, 0.012984, 1.558e-5


def _cylinder(tmp: Path, *, serial: bool = False) -> Path:
    """The fixture's run as a case directory, its files unzipped. `serial`: its ranks removed and
    its log's banner made a serial run's (`Exec   : pimpleFoam`, `nProcs : 1`), so the gate reads
    the case's own mesh and `0/U` (a synthetic edit of the banner alone)."""
    case = tmp / "cylinder"
    _unpack(CYLINDER, case)
    if serial:
        for rank in ("processor0", "processor1"):
            shutil.rmtree(case / rank)
        log = case / "log.pimpleFoam"
        log.write_text(log.read_text().replace("Exec   : pimpleFoam -parallel",
                                                "Exec   : pimpleFoam")
                       .replace("nProcs : 2", "nProcs : 1"))
    return case


def _verify(case: Path, checks: str):
    from courant.verify import verify_path

    return verify_path(case, checks=checks)


def test_the_cylinder_is_rejected_by_default_on_the_laminar_finding_alone(tmp_path):
    v = _verify(_cylinder(tmp_path), "default")
    assert (v.verdict, v.rule) == ("reject", "VER-5.4")
    assert [r.code for r in v.reasons] == ["physics.laminar_unstable"]
    assert not any("not raised" in s for s in v.not_assessed)


def test_under_next_the_cylinder_at_re_100_is_accepted_and_the_verdict_says_why(tmp_path):
    from courant.verify import sentences

    v = _verify(_cylinder(tmp_path), "next")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")
    assert v.reasons == ()
    [said] = [s for s in v.not_assessed if "not raised" in s]
    assert "Reynolds number is at most 100 (flow past one bluff body, on its largest extent)" in said
    assert "below 150" in said
    assert f"Flow regime matches turbulence modeling: {sentences.REYNOLDS_NOT_FED}" in \
        v.not_assessed
    read = {f.path for f in v.run.inputs}
    assert {"processor0/0/U", "processor1/0/U", "processor0/0/p", "processor1/0/p",
            "processor0/constant/polyMesh/points",
            "processor1/constant/polyMesh/faces"} <= read


def test_with_the_behaviour_off_next_rejects_the_cylinder_as_the_default_does(tmp_path,
                                                                              monkeypatch):
    from courant.verify import checks

    real = checks.enabled
    monkeypatch.setattr(checks, "enabled", lambda name, mode="default":
                        False if name == LAMINAR_REGIME else real(name, mode))
    v = _verify(_cylinder(tmp_path), "next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.4")
    assert [r.code for r in v.reasons] == ["physics.laminar_unstable"]


@pytest.mark.parametrize("serial", [False, True], ids=["two ranks", "serial"])
def test_the_cylinders_reynolds_number_from_the_mesh_the_solver_read(tmp_path, serial):
    """Over two ranks the ranks' meshes and `0/U` (as decomposePar wrote its table) are read, and
    joined by their points; serially the case's own. Both give the same number: the largest
    speed of the inflow's table, the cylinder's diameter in the solved plane (the flow turns in
    its first 75 s), nu."""
    case = _cylinder(tmp_path, serial=serial)
    log = (case / "log.pimpleFoam").read_text()
    re = reynolds_number(case, CaseConfig.from_foam(str(case)),
                         Run(log, "log.pimpleFoam", (), exit_code=0))
    assert re.lower is None
    assert re.length == pytest.approx(D, rel=1e-9)
    assert re.speed == pytest.approx(U0, rel=1e-7)
    assert re.upper == pytest.approx(U0 * D / NU, rel=1e-6)


def _edit(case: Path, rel: str, old: str, new: str) -> None:
    p = case / rel
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new))


def _steady_inflow(case: Path) -> None:
    """The inflow made constant (a synthetic edit of `0/U`; the log is the run's)."""
    text = (case / "0" / "U").read_text()
    start = text.index("type            uniformFixedValue;")
    end = text.index(");", start) + 2
    (case / "0" / "U").write_text(text[:start] + f"type fixedValue; value uniform ({U0} 0 0);"
                                  + text[end:])


def test_a_turbulent_reynolds_number_is_cited_and_the_verdict_stands(tmp_path):
    """The same run with nu 1000 times smaller and a constant inflow (synthetic edits of
    `constant/transportProperties` and `0/U`; the log is the run's): Re 1e5, above 300, is cited
    under next, with its inputs; the default says what it said."""
    case = _cylinder(tmp_path, serial=True)
    _edit(case, "constant/transportProperties", f"nu              {NU};",
          f"nu              {NU / 1000};")
    _steady_inflow(case)
    v = _verify(case, "next")
    d = _verify(case, "default")
    assert (v.verdict, v.rule) == (d.verdict, d.rule) == ("reject", "VER-5.4")
    [r] = [r for r in v.reasons if r.code == "physics.laminar_unstable"]
    assert "the case's Reynolds number is at least 1e+05" in r.summary
    assert "above 300, where a circular cylinder's wake is irregular" in r.summary
    assert r.input.startswith("the solver log (its start, transport and laminar stress models)")
    [rd] = [r for r in d.reasons if r.code == "physics.laminar_unstable"]
    assert "Reynolds" not in rd.summary


def _start_field(case: Path, internal: str) -> None:
    """Every `0/U` the case holds (the case's, each rank's) started from `internal` instead of
    the free stream (a synthetic edit; the log is the run's)."""
    old = f"internalField   uniform ({U0} 0 0);"
    for path in [case / "0" / "U", *sorted(case.glob("processor*/0/U"))]:
        _edit(case, path.relative_to(case).as_posix(), old, f"internalField   {internal};")


@pytest.mark.parametrize("serial", [False, True], ids=["two ranks", "serial"])
def test_the_cylinder_started_from_a_faster_field_stays_rejected_under_next(tmp_path, serial):
    """The final review (of 8249ee4): the speed bound read the inlet alone, so a run started from
    a field faster than its inlet (an earlier run's wake, the inlet lowered for a restart) was
    bounded on the inlet. The cylinder's start field at 100 times its free stream (a synthetic
    edit of every `0/U`): at 8249ee4 Re 100, accepted under `--checks next`; its bound is now on
    the start field's speed, Re about 1e4, and the run is rejected as by default."""
    case = _cylinder(tmp_path, serial=serial)
    _start_field(case, f"uniform ({100 * U0!r} 0 0)")
    log = (case / "log.pimpleFoam").read_text()
    re = reynolds_number(case, CaseConfig.from_foam(str(case)),
                         Run(log, "log.pimpleFoam", (), exit_code=0))
    assert re.speed == pytest.approx(100 * U0) and re.upper == pytest.approx(100 * U0 * D / NU)
    v, d = _verify(case, "next"), _verify(case, "default")
    assert (v.verdict, v.rule) == (d.verdict, d.rule) == ("reject", "VER-5.4")
    assert "physics.laminar_unstable" in [r.code for r in v.reasons]
    assert not any("not raised" in s for s in v.not_assessed)


def test_the_cylinder_started_from_its_own_wake_is_bounded_on_its_largest_speed(tmp_path):
    """The run's own field at t = 600, as v2406 wrote it (`nonuniform List<vector>`, one value
    per cell), copied into `0/U` (synthetic): the bound's speed is that field's largest |U|,
    about 1.3 times the free stream, Re about 129, still below 150; at 100 times that field, Re
    about 1.3e4, nothing is set aside."""
    case = _cylinder(tmp_path, serial=True)
    text = (case / "600" / "U").read_text()
    start = text.index("internalField")
    raw = text[start + len("internalField"):text.index(";", start)].strip()
    values = np.array([[float(x) for x in item.split()]
                       for item in raw[raw.index("(") + 1:raw.rindex(")")].replace("(", " ")
                       .split(")") if item.strip()])
    assert values.shape == (1920, 3)
    fastest = float(np.linalg.norm(values, axis=1).max())
    _start_field(case, raw)
    log = (case / "log.pimpleFoam").read_text()
    run = Run(log, "log.pimpleFoam", (), exit_code=0)
    re = reynolds_number(case, CaseConfig.from_foam(str(case)), run)
    assert re.speed == pytest.approx(fastest, rel=1e-12) and fastest > 1.25 * U0
    assert RE_EXTERNAL_LAMINAR > re.upper == pytest.approx(fastest * D / NU)
    v = _verify(case, "next")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")
    scaled = " ".join(f"({x * 100!r} {y * 100!r} {z * 100!r})" for x, y, z in values.tolist())
    _edit(case, "0/U", raw, f"nonuniform List<vector> 1920 ( {scaled} )")
    re = reynolds_number(case, CaseConfig.from_foam(str(case)), run)
    assert re.upper == pytest.approx(100 * fastest * D / NU)
    assert _verify(case, "next").rule == "VER-5.4"


def test_between_the_bounds_the_cylinder_is_judged_as_before(tmp_path):
    """nu halved (Re 200, between 150 and 300; a synthetic edit): nothing is set aside."""
    case = _cylinder(tmp_path, serial=True)
    _edit(case, "constant/transportProperties", f"nu              {NU};",
          f"nu              {NU / 2};")
    v = _verify(case, "next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.4")
    assert [r.code for r in v.reasons] == ["physics.laminar_unstable"]
    assert not any("not raised" in s for s in v.not_assessed)


# ------------------------------------------------------------------ what a field file may hide
#: The review of dde0ecb (26 Oct 2026). A `$name;` written as an entry expands the dictionary it
#: names where it stands (`dictionary::substituteScopedKeyword`, merged under the default
#: `#inputMode merge`), so it may replace a patch's condition after the patch's own block; and
#: v2406 gives a patch without an entry of its own a group's entry before any pattern's
#: (`GeometricBoundaryField::readField`: the patch's name, then a literal key naming one of its
#: groups, `inGroups` and the `wall` group every wall joins, then a pattern). The reader followed
#: neither: it read the patch's literal block, or the pattern, as written.
SPIN = "type rotatingWallVelocity; origin (0 0 0); axis (0 0 1); omega 1000;"


def _boundary_field(text: str) -> tuple[str, str, str]:
    """(the text up to and with `boundaryField {`, its body, the text from its closing brace)."""
    open_ = text.index("{", text.index("boundaryField"))
    depth = 0
    for i in range(open_, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[:open_ + 1], text[open_ + 1:i], text[i:]
    raise AssertionError("boundaryField is not closed")


def _macro(text: str, patch: str, condition: str, where: str) -> str:
    """`text` with a dictionary holding `patch { condition }` defined before its boundaryField and
    expanded by a `$` entry after the patch's own block: as the last entry of the boundaryField
    (`where` "boundaryField"), or as a top-level entry after it ("top")."""
    head, body, tail = _boundary_field(text)
    start = head.rindex("boundaryField")
    if where == "boundaryField":
        return (head[:start] + f"over {{ {patch} {{ {condition} }} }}\n" + head[start:] + body
                + "    $over;\n" + tail + "\n")
    return (head[:start] + f"over {{ boundaryField {{ {patch} {{ {condition} }} }} }}\n"
            + head[start:] + body + tail + "\n$over;\n")


@pytest.mark.parametrize("where", ["boundaryField", "top"])
@pytest.mark.parametrize("field, condition", [
    ("U", "type fixedValue; value uniform (0 5 0);"),   # the body moving across the stream
    ("p", "type fixedValue; value uniform 0;"),         # its pressure held: the flux not U's
], ids=["U", "p"])
def test_a_macro_that_is_an_entry_gives_no_number(tmp_path, field, condition, where):
    """A `$` the reader does not resolve itself (one inside `internalField` or a condition's
    `value`, `inletValue` or `uniformValue`, from the same file) may set any patch's condition:
    the file gives no number, wherever in it the macro stands."""
    case, cfg, run = _external(tmp_path, nu=2e-3)
    assert _re((case, cfg, run)) is not None
    path = case / "0" / field
    path.write_text(_macro(path.read_text(), "body", condition, where))
    assert reynolds_number(case, CaseConfig.from_foam(str(case)), run) is None


def _with_groups(case: Path, patch: str, groups: str) -> None:
    """`patch`'s entry in `constant/polyMesh/boundary` given `inGroups groups;`."""
    entry = f"    {patch}\n    {{\n        type            wall;\n"
    _edit(case, "constant/polyMesh/boundary", entry, entry + f"        inGroups        {groups};\n")


@pytest.mark.parametrize("group", ["wall", "inGroups"])
@pytest.mark.parametrize("field, own, condition", [
    ("U", "type noSlip;", "type fixedValue; value uniform (0 5 0);"),   # the body moving
    ("p", "type zeroGradient;", "type fixedValue; value uniform 0;"),   # its pressure held
], ids=["U", "p"])
def test_a_group_entry_v2406_applies_before_a_pattern_gives_no_number(tmp_path, group, field,
                                                                      own, condition):
    """The body without an entry of its own, under a pattern: v2406 gives it a literal key that
    names one of its groups first (the `wall` group every wall joins, or one its `inGroups`
    names), so the pattern's condition is not the one the solver read, and the file gives no
    number. Without such a key the pattern is read, as v2406's third step reads it; with the
    body's own entry, that entry, as its first."""
    key = "wall" if group == "wall" else "bodies"
    entry = f"    body\n    {{\n        {own}\n    }}\n"
    grouped = f"    {key}\n    {{\n        {condition}\n    }}\n"
    pattern = f'    "b.*"\n    {{\n        {own}\n    }}\n'
    for name, written, given in (("pattern", pattern, True), ("group", grouped + pattern, False),
                                 ("own", grouped + pattern + entry, True)):
        case, _, run = _external(tmp_path / name, nu=2e-3)
        if group == "inGroups":
            _with_groups(case, "body", "1(bodies)")
        _edit(case, f"0/{field}", entry, written)
        got = reynolds_number(case, CaseConfig.from_foam(str(case)), run)
        assert (got is not None) is given, name


@pytest.mark.parametrize("trick", ["macro", "group"])
def test_a_heater_behind_a_macro_or_a_group_is_not_buoyancy_alone(tmp_path, trick):
    """The review's buoyant form: the sides heated by a `fixedGradient` the reader did not see,
    through a `$` entry after their own adiabatic block, or a `wall` group entry before the
    pattern `"side.*"` (the floor and the lid each with its own held entry)."""
    case, run = _cavity(tmp_path)
    assert _alone(case, run)
    heater = "type fixedGradient; gradient uniform 10000;"
    path = case / "0" / "T"
    if trick == "macro":
        path.write_text(_macro(path.read_text(), "sides", heater, "boundaryField"))
    else:
        _edit(case, "0/T", "    sides\n    {\n        type zeroGradient;\n    }\n",
              f"    wall\n    {{\n        {heater}\n    }}\n"
              '    "side.*"\n    {\n        type zeroGradient;\n    }\n')
    assert not _alone(case, run)


def _spun(case: Path, trick: str) -> None:
    """The cylinder spun at 1000 rad/s (about 60 m/s at its surface, Re about 5e5) in every `0/U`
    the case holds, the case's and each rank's, behind a `$` entry after its noSlip block or a
    `wall` group entry before a pattern (synthetic edits; the log is the run's)."""
    own = "    cylinder\n    {\n        type            noSlip;\n    }\n"
    for path in [case / "0" / "U", *sorted(case.glob("processor*/0/U"))]:
        text = path.read_text()
        assert own in text
        if trick == "macro":
            path.write_text(_macro(text, "cylinder", SPIN, "boundaryField"))
        else:
            path.write_text(text.replace(own, f"    wall\n    {{\n        {SPIN}\n    }}\n"
                                              '    ".*"\n    {\n        type noSlip;\n    }\n'))


@pytest.mark.parametrize("serial", [False, True], ids=["two ranks", "serial"])
@pytest.mark.parametrize("trick", ["macro", "group"])
def test_the_cylinder_spun_behind_a_macro_or_a_group_stays_rejected_under_next(tmp_path, trick,
                                                                               serial):
    """The review's blocking case, as a full verdict: at dde0ecb the reader read the cylinder as
    still, gave Re 100 below 150, and the run moved from reject (VER-5.4, as at d32bd45 under
    `--checks next`) to accept. It gives no number, and the verdict is the default's."""
    case = _cylinder(tmp_path, serial=serial)
    _spun(case, trick)
    log = (case / "log.pimpleFoam").read_text()
    assert reynolds_number(case, CaseConfig.from_foam(str(case)),
                           Run(log, "log.pimpleFoam", (), exit_code=0)) is None
    v, d = _verify(case, "next"), _verify(case, "default")
    assert (v.verdict, v.rule) == (d.verdict, d.rule) == ("reject", "VER-5.4")
    assert "physics.laminar_unstable" in [r.code for r in v.reasons]
    assert not any("not raised" in s for s in v.not_assessed)


# ------------------------------------------------------------------ the corpus faults
@pytest.mark.parametrize("sid, rule", [
    ("buoyantPimpleFoam_hotRoom__laminar_on_turbulent", "VER-5.4"),
    ("pisoFoam_RAS_cavity__laminar_on_turbulent", "VER-5.2"),
])
def test_the_corpus_laminar_on_turbulent_faults_stay_rejected_under_next(sid, rule):
    """Their sidecars hold no mesh files, no velocity values and no gravity: no regime number,
    so nothing changes (the hard requirement of T-LAMINAR-REGIME). By construction: a log with a
    sidecar never reaches VER-5b's sparing path (no Reynolds reader, no buoyancy-alone reader),
    so this replay, as Courant-10K's, cannot show the rule judging a faulted case; the case
    directories above do (the review of 26 Oct 2026)."""
    from courant.verify.ingest import from_log
    from courant.verify.pipeline import _reynolds_reader, _still_reader

    inputs = from_log(CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json").model_copy(
        update={"checks": "next"})
    assert _reynolds_reader(inputs, None, object()) is None
    assert _still_reader(inputs, None, object()) is None
    from courant.verify import verify_path

    a = verify_path(CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json")
    b = verify_path(CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json", checks="next")
    assert (a.verdict, a.rule) == (b.verdict, b.rule) == ("reject", rule)
    assert "physics.laminar_unstable" in [r.code for r in b.reasons]
    assert not any("not raised" in s for s in b.not_assessed)


def test_the_name_is_behind_the_flag_for_070():
    from courant.verify.checks import NEXT, enabled

    assert NEXT[LAMINAR_REGIME] == "0.7.0"
    assert not enabled(LAMINAR_REGIME) and enabled(LAMINAR_REGIME, "next")


# ------------------------------------------------------------------ helpers
def _unpack(src: Path, dst: Path) -> None:
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".gz":
            with gzip.open(f, "rb") as s, open(out.with_suffix(""), "wb") as d:
                shutil.copyfileobj(s, d)
        else:
            shutil.copyfile(f, out)
