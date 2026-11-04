"""The Reynolds number a case gives unambiguously, and whether buoyancy alone moves its fluid, from
the files its solver read (TRD VER-5b).

`physics.laminar_unstable` (VER-5a) takes a time-marching laminar run whose residuals end above the
flat bar as evidence that the flow is turbulent. Under `--checks next`
(`physics.laminar_unstable.regime`) it first asks the case's own regime numbers
(`courant.case.regime.regime_numbers`): the Reynolds number read here, and physics.laminar_regime's
Rayleigh number for a case heated from below, which can set a run aside only where nothing but
buoyancy moves its fluid (`buoyancy_alone`). A number is given only where the case leaves no
choice of length or speed; anything else gives none.

**One flow: past one bluff body.** The case's `wall` patches form one connected surface (their
faces joined through shared points) that touches no patch but an `empty` one, so it is a body
inside the domain; every patch that fixes the velocity fixes the same uniform one, and the flow
enters through at least one of them; and the body is bluff (`_BLUFF`): no longer, in any
direction, than twice its width across the flow, or, where the flow's direction changes in time,
than twice its least width, so across any direction the flow may take. ``Re = |U| L / nu``, ``L``
from the body's wall faces' points in the mesh the solver read: its largest extent for the upper
bound, its width across the flow for the lower. Roshko's bounds are a bluff body's wake's, so a
body that is not bluff gives none (a plate aligned with the flow has Re 140 on its thickness and
1.4e6 on its length, the review of 26 Oct 2026).

**No duct flow** (the final review, of 8249ee4, 26 Oct 2026). In a 2-D duct continuity carries the
inlet's flow rate per unit depth through every section, so the inlet's number is every section's;
but no laminar bound is shown for every geometry it may describe. Armaly's 1200 was measured
behind one backward-facing step, and a baffle that leaves a gap makes a planar jet, which turns
unsteady at a slot number (half the section's) well below it. In 3-D or on a wedge a contraction
raises the number as well. A flow through a duct gives none.

**Nothing else drives the flow** (`_driven_otherwise`, `_pressure_paths`), or no number: no
solver log line saying it read `constant/g` (`readGravitationalAcceleration.H`: buoyancy is the
Rayleigh number's, and buoyantBoussinesqPimpleFoam reads `nu` and prints a Newtonian model too,
its createFields.H); no moving mesh (`CaseConfig.mesh_motion`, dynamicFvMeshNew.C); no MRF zone
(MRFZone::correctBoundaryVelocity, MRFZone.C, sets the walls inside one to ``Omega x r``), SRF
frame, fvOption, fvModel or fvConstraint (a momentum source), its file present where the solver
would look (IOMRFZoneList.C: `constant/MRFProperties`; fvOptions.C: `constant/fvOptions`, then
`system/fvOptions`) or its line in the log; every patch that fixes the velocity (an inlet, a wall)
leaving the pressure free (`zeroGradient`, `fixedFluxPressure`), so its flux is its velocity's:
pimpleFoam's flux is ``phi = phiHbyA - pEqn.flux()`` (pEqn.H), whose boundary part vanishes only
there (fixedFluxPressure sets its gradient so that it does, constrainPressure); and, past the
inlets, at most one opening held at a pressure: a second one held at another pressure drives the
flow through it, at a speed no inlet bounds.

**Two bounds, each conservative for what it decides** (`Reynolds.upper`, `Reynolds.lower`): the
upper bound alone may set a run's residual history aside (below the laminar bound), and the lower
bound alone may be cited as turbulent (above the turbulent bound).

- The speed, upper: the larger of the inlet's and the start field's. The inlet's: a `fixedValue`
  inlet's ``|U|``; a `uniformFixedValue` one's largest ``|U|`` over its inline `table`'s entries
  (a linear or step interpolation between entries, clamped or repeated outside them, never
  leaves their convex hull, so no time of the run sees a larger inflow). The start field's: the
  largest ``|U|`` of `<start>/U`'s internalField in every root the solver read
  (`_start_speed`): a field faster than the inlet (an earlier run's wake, a field mapped from
  another run) is carried through the domain for many flow-through times, so it bounds the speed
  as well (the final review). It is read as v2406 writes it (Field::writeEntry, Field.C):
  `uniform (x y z)`, or `nonuniform List<vector> N (...)` with N the root's cells, which
  Field::assign requires, in an ascii file; any other form gives no number. Lower: the inlet's
  speed, the free stream's, given only for a velocity that does not change in time.
- The length, upper: the body's largest extent; in 2-D its points' diameter in the solved plane;
  in 3-D (one flow direction only) ``sqrt(d^2 + a^2)``, d the diameter of its projection on the
  plane normal to the flow and a its extent along the flow, which no distance between two of its
  points exceeds. Lower: with one flow direction, the body's width across it (in 2-D
  perpendicular to the flow in the solved plane; in 3-D the least width of the projection); with
  a direction that changes in time, none.
- ``nu``: the case's one Newtonian kinematic viscosity (`constant/transportProperties` `nu`), on
  a solver that says it took it so: icoFoam reads `nu` itself; another solver's log must print
  `Selecting incompressible transport model Newtonian` and, when it prints a laminar stress model,
  `Stokes`.

**Buoyancy alone** (`buoyancy_alone`): the Rayleigh number's bound below its onset says only that
buoyancy drives no convection, so it sets a run aside only where nothing else moves or heats the
fluid: no log line, moving mesh, MRF zone, SRF frame or fvOption, fvModel or fvConstraint, as
above (an fvOption is also a heat source); every patch a `wall` or an `empty` one (an opening, an
inlet, or a mirror, which extends the domain past the extent the bound is taken over, gives
none); every wall still (`noSlip` or a zero `fixedValue`: no lid moves); the fluid at rest at the
start (`internalField uniform (0 0 0)`); every wall's temperature held (`fixedValue`, uniform,
within the temperatures the bound was taken between) or adiabatic (`zeroGradient`: the energy's
condition is then `gradientEnergy`, basicThermo::heBoundaryTypes in basicThermo.C, whose gradient
is Cp times T's normal gradient, zero there, gradientEnergyFvPatchScalarField.C), so no heater
the bound does not count (a `fixedGradient`, `externalWallHeatFlux` or
`lumpedMassWallTemperature` one); the interior's initial temperature uniform within them; and no
radiation model (`_radiates`, the final review). ESI's buoyant solvers build one in the solver,
not as an fvOption: buoyantPimpleFoam's and buoyantSimpleFoam's createFields.H include
createRadiationModel.H, buoyantBoussinesqPimpleFoam's and buoyantBoussinesqSimpleFoam's
createIncompressibleRadiationModel.H, each `radiation::radiationModel::New(T)`, which reads
`radiationModel` from `constant/radiationProperties` and prints "Selecting radiationModel"
(radiationModelNew.C); the energy equation takes its source (EEqn.H `radiation->Sh(thermo, he)`,
TEqn.H `radiation->ST(rhoCpRef, T)`), which heats the fluid past the held temperatures the bound
counts (a solarLoad from outside). A `constant/radiationProperties`, in the case or a root the
solver read, whose `radiationModel` is anything but `none` as written (a directive, a macro, no
such entry, a file that cannot be read), or a log line naming another model, gives none.

**What is read.** The mesh the solver read (`courant.case.mesh.solver_mesh`, the one rule: the
case's own `constant/polyMesh`, or every rank's for a parallel run), within the cost bounds of
`solver_polymesh`; and the fields at the time the run started from
(`courant.spec.as_run.AsRun.start`), in the same roots: `<start>/U` and `<start>/p` for the Reynolds
number, `<start>/U` and `<start>/T` for buoyancy alone, each read as written and in full, and
`constant/radiationProperties`, where there is one, for buoyancy alone. A directive anywhere in a
field gives no number, and so does a `$` macro anywhere but in a value this reader resolves from the
same file (`internalField`, a condition's `value`, `inletValue` or `uniformValue`): a `$name;` entry
expands the dictionary it names where it stands, in the boundaryField or at the top of the file, and
may replace a patch's condition after the patch's own block (`_resolved_macros`). A patch's
condition is its own entry's, or a pattern's: where v2406 would first give it a group's (a literal
key naming one of its groups, from `inGroups` in `polyMesh/boundary` or the `wall` group every wall
joins; `GeometricBoundaryField::readField`), none (`_boundary`; both the review of dde0ecb, 26 Oct
2026). One reading's work on these files, the radiation dictionaries among them, is bounded: each is
sized on disk before it is opened; each field file is scanned once for its blocks, each distinct
block in it read once; more than 128 ranks, a rank's mesh of more than 32 patches (its processor
patches and empty shares not counted, its constraint patches counted), a boundaryField of more than
64 `{}` blocks, more than 8 `$` in all the files the reading reads, a `$` in a file of more than
100,000 bytes, or more than 4,000,000 bytes in all of them gives no number (`_MAX_RANKS`; the
reviews of b1fea1c, 0d4930f and c9c2eae). Each file read is noted for the manifest (VER-14). A
source's file is looked for, not read. A decomposed mesh is joined by its points' coordinates, which
decomposePar copies from the case's.

Names of patches are the case author's: nothing here reaches a verdict but numbers.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

#: The patch types this reader knows (`polyMesh/boundary`). Any other (a cyclic, an AMI, a
#: mapped patch) gives no number: what crosses it is not known here.
_KNOWN = frozenset({"patch", "wall", "empty", "wedge", "symmetry", "symmetryPlane", "processor"})
#: The constraint types: a patch of one takes its type as its condition (fvPatchFieldNew.C).
_CONSTRAINT = frozenset({"empty", "wedge", "symmetry", "symmetryPlane"})
#: The velocity conditions of a `patch` patch that set no speed of their own: the patch value is
#: the interior's (zeroGradient), or its tangential part with no normal one (slip, a
#: basicSymmetry condition); and `inletOutlet` with a zero `inletValue` (`_condition`) is zero
#: where the flow would enter (inletOutletFvPatchField.C: `refValue()` is `inletValue`,
#: `valueFraction() = neg(phip)`), so a reversed flow brings no speed in.
_NO_SPEED = frozenset({"zeroGradient", "slip"})
#: A stationary wall's velocity conditions.
_STILL_WALL = frozenset({"noSlip"})

#: A value's tokens: a bracket, or a run of anything else but blanks and brackets. One character
#: class a match, so the scan is linear in the text: the case writes these values, and a number
#: pattern with two ways to split a run of digits backtracks in the square of its length.
_TOKEN = re.compile(r"[()]|[^\s()]+")
#: A table longer than this is not read (one lookup's bound, as `deck` bounds its reads).
_MAX_TABLE = 100_000
#: The bounds on one reading's work on the case's files (`_Budget`, `_roots_and_meshes`,
#: `_boundary`; the reviews of b1fea1c and 0d4930f, 26 Oct 2026). Each `$` resolved is a pass
#: over the file it is in and each patch's block was a pass over the boundaryField, so 100,000
#: macros or 20,000 one-face patches held the reader for hours, and a pattern block shared by
#: every patch, resolved once a patch on every rank, for minutes within per-file bounds; and a
#: reading at 16,000,000 characters took about 37 s, with `constant/radiationProperties` read
#: whole outside any bound (the review of c9c2eae). Now each file is sized on disk before it is
#: opened, scanned once for its blocks, each distinct block read once in it, and no `$` costs a
#: pass where there is none; past any of these the reading gives no number: more ranks than
#: `_MAX_RANKS`; a rank's mesh of more patches than `_MAX_PATCHES` (its processor patches and its
#: empty shares of a patch not counted); a boundaryField of more `{}` blocks than `_MAX_ENTRIES`;
#: more `$` than `_MAX_MACROS` in all the files one reading reads (a solver, decomposePar among
#: them, writes its fields expanded, so `$` is in a hand-written serial case's files); a `$` in a
#: file of more than `_MAX_MACRO_FILE` bytes; more than `_MAX_READ` bytes in all of them.
_MAX_RANKS = 128
_MAX_PATCHES = 32
_MAX_ENTRIES = 64
_MAX_MACROS = 8
_MAX_MACRO_FILE = 100_000
_MAX_READ = 4_000_000


class _Budget:
    """What one reading (`_reynolds`, `_buoyancy_alone`) may still read across every rank's field
    files and the radiation dictionaries it reads: bytes on disk (`_MAX_READ`) and `$`
    (`_MAX_MACROS`). Made once a reading; `read` charges each file to it."""

    def __init__(self) -> None:
        self.bytes, self.macros = _MAX_READ, _MAX_MACROS

    def read(self, case: Path, rel: str) -> str | None:
        """`rel`'s text as `deck._read_text` reads it (noted for the manifest, its comments
        stripped), sized on disk and charged before it is opened, or None: outside the roots
        (MCP-3), not a regular file, gone or unreadable, past the bytes left, past the `$` left,
        or a `$` in a file of more than `_MAX_MACRO_FILE` bytes. A file not opened is not noted
        for the manifest (VER-14)."""
        from courant.case.reads import may_read
        from courant.spec import deck

        path = case / rel
        if not may_read(path) or not path.is_file():
            return None
        try:
            size = path.stat().st_size
        except OSError:
            return None                         # gone since it was looked for
        self.bytes -= size
        if self.bytes < 0:
            return None
        text, _ = deck._read_text(case, rel)
        if text is None:
            return None
        n = text.count("$")
        self.macros -= n
        return text if self.macros >= 0 and (not n or size <= _MAX_MACRO_FILE) else None
#: What a solver of the incompressible family prints when it builds its viscosity
#: (viscosityModelNew.C: "Selecting incompressible transport model " << modelType) and its
#: laminar stress model (laminarModel.C: "Selecting laminar stress model " << modelType).
_TRANSPORT_MODEL = re.compile(r"^Selecting incompressible transport model (\S+)[ \t]*$",
                              re.MULTILINE)
_LAMINAR_MODEL = re.compile(r"^Selecting laminar stress model (\S+)[ \t]*$", re.MULTILINE)
#: Within this fraction of the inlet's |U| A, a patch's net flux is no flux: a side boundary
#: whose fixed velocity runs along it.
_TANGENTIAL = 1e-6
#: The pressure conditions that leave a patch's flux to its velocity: pimpleFoam's flux is
#: ``phi = phiHbyA - pEqn.flux()`` (pimpleFoam/pEqn.H), and the boundary part of the pressure
#: equation's flux is zero on a `zeroGradient` patch (no gradient coefficient) and is set to the
#: velocity's on a `fixedFluxPressure` one (its gradient chosen so, `constrainPressure`); with a
#: fixed velocity `constrainHbyA` makes ``phiHbyA`` that velocity's flux there. Any other
#: condition (a `fixedValue`, `totalPressure`, ... pressure) lets the pressure drive the flux.
_FREE_P = frozenset({"zeroGradient", "fixedFluxPressure"})
#: What else moves (or heats) a fluid, where a solver looks for it: IOMRFZoneList.C
#: (`"MRFProperties" + solverName` in `constant`), SRFModelNew.C (`constant/SRFProperties`),
#: fvOptions.C (`constant/fvOptions`, else `system/fvOptions`); and the same sources under
#: OpenFOAM.org's names, fvModels and fvConstraints (its meanVelocityForce is a constraint), for
#: a case written for that fork. Looked for, never read: any of them gives no number.
_SOURCES = ("constant/MRFProperties", "constant/SRFProperties", "constant/fvOptions",
            "system/fvOptions", "constant/fvModels", "system/fvModels",
            "constant/fvConstraints", "system/fvConstraints")
#: The solver's own lines for the same: IOMRFZoneList.C ("Creating MRF zone list from "),
#: fvOptions.C ("Creating finite volume options from "), SRFModelNew.C ("Selecting SRFModel "),
#: dynamicFvMeshNew.C ("Selecting dynamicFvMesh " << modelType; `staticFvMesh` does not move).
_SOURCE_LINE = re.compile(r"^(?:Creating MRF zone list from |Creating finite volume options from "
                          r"|Selecting SRFModel |Selecting dynamicFvMesh (?!staticFvMesh[ \t]*$))",
                          re.MULTILINE)
#: Where an ESI buoyant solver's radiation model is chosen (radiationModelNew.C: an IOobject
#: "radiationProperties" in `T.time().constant()`, its `radiationModel` entry, `none` without
#: the file), and the line it prints for it ("Selecting radiationModel " << modelType); `none`
#: is noRadiation, whose Rp and Ru are zero (noRadiation.C), so its source `Sh` is none
#: (radiationModel.C) (`_radiates`).
_RADIATION_FILE = "constant/radiationProperties"
_RADIATION_LINE = "Selecting radiationModel"
#: A body no longer, in any direction, than this many times its width across the flow is bluff:
#: a circle 1, a square at most sqrt(2), a cube or a sphere in 3-D at most sqrt(3) on this
#: reader's bound of its extent. Roshko's bounds are a bluff body's wake's; a longer body (a
#: plate, an airfoil) turns on its length, so only a bluff body gives a number (the final review,
#: of 8249ee4). Where the flow's direction changes in time, its width across the flow is taken
#: at its least, over every direction in the solved plane. A finite cylinder across a 3-D flow,
#: its span three diameters, is not bluff on it (its extent about 3.3 diameters); a flat plate
#: across the flow is, and whether the cylinder's bounds hold for its wake is not shown here
#: (TRD VER-5b, Limits; the review of b1fea1c).
_BLUFF = 2.0
#: The entries of a condition whose `$macro` the reader resolves; one elsewhere may set the
#: type or merge entries, and gives no number.
_U_MACROS = ("value", "inletValue", "uniformValue")
_SCALAR_MACROS = ("value",)


@dataclass(frozen=True)
class Reynolds:
    """A case's Reynolds number for flow past one bluff body, as two bounds (module docstring)."""

    upper: float
    #: None when the velocity changes in time (its direction or its size).
    lower: float | None
    #: The upper bound's inputs: the largest speed (the inlet's or the start field's, the
    #: larger), m/s, its length, m, and nu, m^2/s.
    speed: float
    length: float
    nu: float


def _ok(*xs) -> bool:
    try:
        return all(math.isfinite(x) and x > 0 for x in xs)
    except TypeError:
        return False


# ------------------------------------------------------------------------------ the viscosity
def _kinematic_nu(config, log_text: str | None) -> float | None:
    """The case's one Newtonian kinematic viscosity, as the solver says it took it, or None."""
    if config is None or config.equation_of_state is not None or config.phases \
            or config.phase_nu or config.regions:
        return None
    nu = (config.transport or {}).get("nu")
    if not _ok(nu) or not log_text:
        return None
    from courant.case.solvers import solver_exec

    ex = solver_exec(log_text)
    if ex is None:
        return None
    if ex.program == "icoFoam":         # icoFoam/createFields.H reads `nu` from transportProperties
        return nu
    models = _TRANSPORT_MODEL.findall(log_text[:200_000])
    if models != ["Newtonian"]:
        return None
    stress = _LAMINAR_MODEL.findall(log_text[:200_000])
    return nu if stress in ([], ["Stokes"]) else None


# ------------------------------------------------------------------------------ the velocity
@dataclass(frozen=True)
class _Velocity:
    """A patch's fixed velocity: its uniform value, or every entry of its inline table."""

    vectors: tuple[tuple[float, float, float], ...]
    constant: bool


def _number(token: str) -> float | None:
    try:
        x = float(token)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _vector_at(tokens: list[str], i: int) -> tuple[tuple[float, float, float] | None, int]:
    """(the vector `( x y z )` at token i, the token after it), or (None, i)."""
    if tokens[i:i + 1] != ["("] or tokens[i + 4:i + 5] != [")"]:
        return None, i
    xs = [_number(t) for t in tokens[i + 1:i + 4]]
    if len(xs) != 3 or any(x is None for x in xs):
        return None, i
    return (xs[0], xs[1], xs[2]), i + 5


def _vector(raw: str | None) -> tuple[float, float, float] | None:
    """`(x y z)` and nothing else."""
    text = (raw or "").strip()
    if len(text) > _MAX_TABLE:
        return None
    tokens = _TOKEN.findall(text)
    v, end = _vector_at(tokens, 0)
    return v if v is not None and end == len(tokens) else None


def _uniform_vector(raw: str | None) -> tuple[float, float, float] | None:
    """`uniform (x y z)`, as a `fixedValue` patch writes its value."""
    text = (raw or "").strip()
    if not text.startswith("uniform"):
        return None
    return _vector(text[len("uniform"):])


def _table(raw: str | None) -> tuple[tuple[float, float, float], ...] | None:
    """A table's (time vector) entries, as `List<Tuple2<scalar, vector>>` reads them: `(entries)`
    or `N (entries)` with N their number. Any other form (a `N{...}` list, a file): None."""
    text = (raw or "").strip()
    if len(text) > _MAX_TABLE:
        return None
    tokens = _TOKEN.findall(text)
    count = None
    if tokens and tokens[0] != "(":
        if not tokens[0].isdigit():
            return None
        count, tokens = int(tokens[0]), tokens[1:]
    if tokens[:1] != ["("] or tokens[-1:] != [")"]:
        return None
    entries, i, last = [], 1, len(tokens) - 1
    while i < last:                             # each entry: ( t ( x y z ) )
        if tokens[i] != "(" or i + 2 >= last or _number(tokens[i + 1]) is None:
            return None
        v, i = _vector_at(tokens, i + 2)
        if v is None or tokens[i:i + 1] != [")"]:
            return None
        entries.append(v)
        i += 1
    if not entries or i != last or (count is not None and count != len(entries)):
        return None
    return tuple(entries)


def _function1(body: str, resolve) -> _Velocity | None:
    """A `uniformFixedValue` patch's `uniformValue`, as v2406 reads it: `PatchFunction1<vector>::
    New` (src/meshTools/PatchFunction1/PatchFunction1/PatchFunction1New.C) takes a value that is
    not a word, or `uniform`/`constant` and a value, as one value for every face and time
    (`ConstantField`); another word names a `Function1` (`UniformValueField`, through
    `Function1::New`, Function1New.C), its coefficients in the entry's own dictionary
    (`uniformValue { type table; ... }`), else in `uniformValueCoeffs`, else the patch's own
    (`optionalSubDict`). Read here: the constant forms, and `table` (Table.C: inline, `uniformValue
    table (...)`, or its coefficients' `values`, as decomposePar writes it) interpolated `linear`,
    TableBase.C's default. Anything else (a file, another interpolation, another type): None.
    `resolve` substitutes a `$macro` in an entry's value, or gives None."""
    from courant.spec import deck

    raw = resolve(deck.entry(body, "uniformValue"))
    coeffs = None
    if raw is None:
        if deck.entry(body, "uniformValue") is not None:
            return None                         # a macro that did not resolve
        coeffs = deck.block(body, "uniformValue")
        if coeffs is None:
            return None
        word, rest = (deck.entry(coeffs, "type") or "").strip(), None
    else:
        text = raw.strip()
        if text.startswith("("):
            v = _vector(text)
            return None if v is None else _Velocity((v,), True)
        word, _, rest = text.partition(" ")
        rest = rest.strip() or None
        coeffs = deck.block(body, "uniformValueCoeffs")
        if coeffs is None:
            coeffs = body                       # `optionalSubDict`: the patch's own entries
    if word in ("constant", "uniform"):
        v = _vector(rest if rest is not None else resolve(deck.entry(coeffs, "value")))
        return None if v is None else _Velocity((v,), True)
    if word != "table" or deck.entry(coeffs, "file") is not None:
        return None
    if (deck.entry(coeffs, "interpolationScheme") or "linear").strip() != "linear":
        return None
    entries = _table(rest if rest is not None else resolve(deck.entry(coeffs, "values")))
    if entries is None:
        return None
    distinct = tuple(dict.fromkeys(entries))
    return _Velocity(distinct, len(distinct) == 1)


def _condition(body: str, resolve) -> tuple[str, _Velocity | None] | None:
    """(the velocity condition's type, its fixed velocity when it fixes one), or None when it is
    not read: a type outside the ones this reader knows, or a value it cannot read whole.
    `resolve` substitutes a `$macro` in a value (`deck`'s rule), or gives None."""
    from courant.spec import deck

    kind = (deck.entry(body, "type") or "").strip()
    if kind == "fixedValue":
        v = _uniform_vector(resolve(deck.entry(body, "value")))
        return None if v is None else (kind, _Velocity((v,), True))
    if kind == "uniformFixedValue":
        vel = _function1(body, resolve)
        return None if vel is None else (kind, vel)
    if kind == "inletOutlet":
        v = _uniform_vector(resolve(deck.entry(body, "inletValue")))
        return (kind, None) if v == (0.0, 0.0, 0.0) else None
    if kind in _NO_SPEED or kind in _STILL_WALL:
        return kind, None
    return None


def _zero(vel: _Velocity | None) -> bool:
    return vel is not None and all(x == 0.0 for v in vel.vectors for x in v)


def _pressure(body: str, resolve) -> tuple[str, None] | None:
    """A pressure condition's type (`_FREE_P` decides what it does to the flux), or None."""
    from courant.spec import deck

    kind = (deck.entry(body, "type") or "").strip()
    return (kind, None) if kind else None


def _uniform_scalar(raw: str | None) -> float | None:
    """`uniform x`, as a `fixedValue` patch or a uniform `internalField` writes it."""
    tokens = _TOKEN.findall((raw or "").strip()[:_MAX_TABLE])
    if len(tokens) != 2 or tokens[0] != "uniform":
        return None
    return _number(tokens[1])


def _temperature(body: str, resolve) -> tuple[str, float | None] | None:
    """(a temperature condition's type, its value when it is a uniform `fixedValue`), or None."""
    from courant.spec import deck

    kind = (deck.entry(body, "type") or "").strip()
    if not kind:
        return None
    value = _uniform_scalar(resolve(deck.entry(body, "value"))) if kind == "fixedValue" else None
    return kind, value


# ------------------------------------------------------------------------------ the mesh
@dataclass(frozen=True)
class _Surface:
    """The domain's boundary, joined over every rank: each face's global point ids, its area
    vector (outward), its patch's name and type; and the points."""

    points: object                      # (P, 3) array
    faces: tuple                        # arrays of global point ids
    areas: object                       # (F, 3) array
    patch: tuple[str, ...]
    kind: dict[str, str]


def _surface(meshes) -> _Surface | None:
    import numpy as np

    kinds: dict[str, str] = {}
    coords, faces, areas, names = [], [], [], []
    offset = 0
    for mesh in meshes:
        local: dict[int, int] = {}
        for p in mesh.boundary:
            if p.type not in _KNOWN:
                return None
            if p.type == "processor":
                continue
            if kinds.setdefault(p.name, p.type) != p.type:
                return None                     # one name, two types across ranks
            for i in range(p.start, p.start + p.n):
                f = np.asarray(mesh.faces[i])
                for v in f.tolist():
                    if v not in local:
                        local[v] = offset + len(local)
                faces.append(np.array([local[v] for v in f.tolist()], dtype=np.int64))
                names.append(p.name)
            areas.append(mesh.face_areas[p.start:p.start + p.n])
        order = sorted(local, key=local.get)
        coords.append(mesh.points[np.asarray(order, dtype=np.int64)] if order
                      else np.zeros((0, 3)))
        offset += len(local)
    if not faces:
        return None
    pts = np.concatenate(coords)
    # One point written by two ranks is one point: decomposePar copies each from the case's mesh.
    unique, inverse = np.unique(pts, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    present = set(names)                        # a patch with no face on any rank bounds nothing
    return _Surface(points=unique, faces=tuple(inverse[f] for f in faces),
                    areas=np.concatenate(areas), patch=tuple(names),
                    kind={n: k for n, k in kinds.items() if n in present})


def _solved(surface: _Surface):
    """The solved directions, as v2406 decides them (`courant.case.mesh.solved_directions`),
    from the `empty` faces of every rank."""
    import numpy as np

    from courant.case.mesh import solved_directions

    empty = np.zeros(3)
    for i, name in enumerate(surface.patch):
        if surface.kind[name] == "empty":
            empty += np.abs(surface.areas[i])
    return solved_directions(empty)


def _pieces(faces) -> dict[int, int]:
    """{point: the piece it belongs to} for a set of faces joined through shared points."""
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in faces:
        ids = [int(v) for v in f]
        for v in ids:
            parent.setdefault(v, v)
        root = find(ids[0])
        for v in ids[1:]:
            r = find(v)
            if r != root:
                parent[r] = root
    return {v: find(v) for v in parent}


# ------------------------------------------------------------------------------ widths
def _hull(xy):
    """The convex hull of 2-D points, counter-clockwise (Andrew's monotone chain)."""
    import numpy as np

    pts = sorted(set(map(tuple, np.asarray(xy, dtype=float).tolist())))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _widths(xy) -> tuple[float, float]:
    """(the smallest, the largest) width of a 2-D point set over every direction: the hull's
    least distance between parallel supporting lines (one of them through a hull edge), and its
    diameter. Rotating calipers (Shamos 1978; Toussaint, "Solving geometric problems with the
    rotating calipers", 1983): for each hull edge in turn the vertex farthest from it, which
    only moves forward around the hull, so the walk is linear in the hull's vertices. Every
    antipodal pair is an edge's end and its farthest vertex, so the diameter is among those."""
    import numpy as np

    h = _hull(xy)
    if len(h) < 3:
        span = float(np.linalg.norm(h[-1] - h[0])) if len(h) == 2 else 0.0
        return 0.0, span
    pts = [(float(x), float(y)) for x, y in h.tolist()]
    n = len(pts)

    def area2(a, b, c) -> float:                    # twice the triangle's area, CCW positive
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    least, diameter, j = math.inf, 0.0, 1
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        while area2(a, b, pts[(j + 1) % n]) > area2(a, b, pts[j]):
            j = (j + 1) % n
        c = pts[j]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length > 0:
            least = min(least, area2(a, b, c) / length)
        diameter = max(diameter, math.hypot(c[0] - a[0], c[1] - a[1]),
                       math.hypot(c[0] - b[0], c[1] - b[1]))
    return (least if math.isfinite(least) else 0.0), diameter


# ------------------------------------------------------------------------------ the number
def reynolds_number(case_dir: str | Path | None, config, run) -> Reynolds | None:
    """The case's Reynolds number (module docstring), or None. Never raises: a file it cannot
    read gives no number, as an ambiguous case does."""
    if not case_dir or config is None or run is None:
        return None
    try:
        return _reynolds(Path(case_dir), config, run)
    except Exception:  # noqa: BLE001 - a reader limit gives no number, never a traceback
        return None


def buoyancy_alone(case_dir: str | Path | None, config, run,
                   temperatures: tuple[float, float] | None) -> bool:
    """Whether nothing but buoyancy moves the case's fluid and nothing but its held walls heat it
    (no source and no radiation model), within `temperatures` (the coldest and the hottest the
    Rayleigh number was taken between): the module docstring's "Buoyancy alone". False whenever
    that is not shown, a file that cannot be read included. Never raises."""
    if not case_dir or config is None or run is None or not temperatures:
        return False
    try:
        return _buoyancy_alone(Path(case_dir), config, run, temperatures)
    except Exception:  # noqa: BLE001 - not shown, so not alone
        return False


def _roots_and_meshes(case: Path, run, start: str):
    """(the roots the solver read its mesh and fields from, their meshes, each mesh's
    {patch: the groups it may be in}), or None."""
    from courant.case import polymesh
    from courant.case.mesh import POLYMESH_MAX_BYTES, POLYMESH_MAX_FACES, solver_mesh
    from courant.case.reads import refuse_outside

    found = solver_mesh(case, run.log_text, start=float(start))
    if found.why or not found.dirs:
        return None
    roots = [d.parent.parent for d in found.dirs]
    if len(roots) > _MAX_RANKS:
        return None                             # past the work bounds, before any is read
    if any(d != r / "constant" / "polyMesh" for d, r in zip(found.dirs, roots)):
        return None                             # a mesh at a time instance: not read here
    files = [d / n for d in found.dirs for n in polymesh.FILES]
    if not all(f.is_file() for f in files):
        return None
    for f in files:
        refuse_outside(f)                       # MCP-3: sized and read only inside the roots
    if sum(f.stat().st_size for f in files) > POLYMESH_MAX_BYTES:
        return None
    meshes = [polymesh.read_polymesh(r, max_faces=POLYMESH_MAX_FACES) for r in roots]
    if sum(len(m.faces) for m in meshes) > POLYMESH_MAX_FACES:
        return None
    groups = []
    for d, mesh in zip(found.dirs, meshes):
        got = polymesh._read(d / "boundary", _patch_groups)
        if got is None or [n for n, _ in got] != [p.name for p in mesh.boundary]:
            return None
        groups.append({n: words | {p.type} for (n, words), p in zip(got, mesh.boundary)})
    return roots, meshes, groups


def _patch_groups(path: Path) -> list[tuple[str, frozenset[str]]] | None:
    """Every patch of a `polyMesh/boundary` file, in order, with every word of its `inGroups`
    entries (`inGroups 1(wall);`, `inGroups List<word> 1(wall);`): the groups v2406 puts it in
    (patchIdentifier.C reads `inGroups`), or more, the count and `List<word>` taken as words too.
    The caller adds the patch's type, the group a wall (wallPolyPatch.C) or a constraint patch
    (polyPatch.C) joins whatever the file says. None where an entry may be hidden or misread: a
    directive, a macro or a quoted key in the list (`readIfPresent` matches a pattern), or
    `inGroups` written other than as an entry of its own. Split on `;` and blanks, so linear in
    the file's length (the mesh's cost bound holds it)."""
    from courant.case import polymesh

    _, inner, _ = polymesh._outer_list(polymesh._body(path)[0])
    if any(c in inner for c in "#$\""):
        return None
    out = []
    for chunk in inner.split("}")[:-1]:
        name, sep, block = chunk.partition("{")
        if not sep or not name.split():
            return None
        *entries, rest = block.split(";")
        if "inGroups" in rest:
            return None                         # not ended by its `;`
        words: set[str] = set()
        for entry in entries:
            if "inGroups" not in entry:
                continue
            tokens = entry.split()
            if tokens[0] != "inGroups" or entry.count("inGroups") != 1:
                return None
            words.update(_GROUP_WORD.findall(" ".join(tokens[1:])))
        out.append((name.split()[-1], frozenset(words)))
    return out


#: A word of an `inGroups` list: a run of anything but blanks and brackets (one class, linear).
_GROUP_WORD = re.compile(r"[^\s()]+")


def _driven_otherwise(case: Path, roots, config, log_text: str | None) -> bool:
    """Whether something besides the boundary conditions may move the fluid (or heat it): a moving
    mesh (`CaseConfig.mesh_motion`, which reads `constant/dynamicMeshDict` as dynamicFvMeshNew.C
    does), a source's file where the solver would look (`_SOURCES`, in the case and in every root
    the solver read; a file that cannot be looked at counts), or the solver's own line for one
    (`_SOURCE_LINE`)."""
    if config.mesh_motion is not None:
        return True
    if log_text and _SOURCE_LINE.search(log_text[:200_000]):
        return True
    return source_file(case, roots) is not None


def source_file(case: Path, roots=(), region: str | None = None) -> str | None:
    """The first of `_SOURCES` present (a link or a file that cannot be looked at counts) in the
    case or in a root the solver read, as `<root>/<rel>` relative to the case; None when there is
    none. Looked for, never read. With `region`, where a multi-region solver looks for that
    region's: the same files under `constant/<region>/` and `system/<region>/` (TRD VER-22)."""
    rels = _SOURCES if not region else tuple(
        f"{top}/{region}/{name}" for top, name in (rel.split("/", 1) for rel in _SOURCES))
    for base in dict.fromkeys([case, *roots]):
        for rel in rels:
            if os.path.lexists(base / rel):
                return (base / rel).relative_to(case).as_posix()
    return None


def _radiates(case: Path, roots, log_text: str | None, budget: _Budget,
              region: str | None = None, *, files: bool = True) -> bool:
    """Whether a radiation model may heat the fluid (module docstring, "Buoyancy alone"): a line
    of the solver's log naming a model other than `none` (`_RADIATION_LINE`, anywhere in the
    log), or a `constant/radiationProperties` in the case or in a root the solver read whose
    `radiationModel` is anything but `none` as written: a directive or a macro in the file, no
    such entry, or a file that cannot be read or is past the reading's `budget` counts. A file
    read is noted for the manifest (VER-14); one past the budget is never opened, so it is not.
    With `region`, the file is the region's, `constant/<region>/radiationProperties`, where a
    multi-region solver reads it (TRD VER-22). `files=False`: the log alone."""
    from courant.spec import deck

    text = log_text or ""
    at = text.find(_RADIATION_LINE)
    while at >= 0:
        if at == 0 or text[at - 1] == "\n":
            end = text.find("\n", at)
            if text[at + len(_RADIATION_LINE):end if end >= 0 else len(text)].split() != ["none"]:
                return True
        at = text.find(_RADIATION_LINE, at + 1)
    if not files:
        return False
    rel = _RADIATION_FILE if not region else f"constant/{region}/radiationProperties"
    for base in dict.fromkeys([case, *roots]):
        path = base / rel
        if not os.path.lexists(path):
            continue
        got = budget.read(case, path.relative_to(case).as_posix())
        if got is None or "#" in got or "$" in got or deck.entry(got, "radiationModel") != "none":
            return True
    return False


def _ascii(text: str, found=None) -> bool:
    """Whether a field file is written in ascii (its `FoamFile` header's `format`, ascii when it
    has none): a `nonuniform` list in a binary file is not text, and is not read here. `found`:
    the text's top-level blocks, where the caller has them (`deck.block`)."""
    from courant.spec import deck

    header = deck.block(text, "FoamFile", found)
    return header is None or deck.entry(header, "format") in (None, "ascii")


def _boundary(case: Path, roots, meshes, groups, start: str, field: str, condition, macros,
              budget: _Budget):
    """({patch: condition(body)}, every root's raw `internalField`) for one field at the start,
    the conditions the same in every root, or None: a root with no such field, a directive in
    it, a file past the reading's `budget` or a mesh or a boundaryField past the work bounds
    (`_MAX_PATCHES`), a `$` macro anywhere but in the values this reader resolves itself
    (`_resolved_macros`),
    a patch whose block v2406 may take from one of its groups (`groups`, `_roots_and_meshes`), or
    a condition `condition` cannot read (None). A root's `internalField` is None where it has
    none, or where it is a `nonuniform` list in a binary file (`_ascii`). A constraint patch's
    condition is its own type: v2406 refuses any other ("inconsistent patch and patchField
    types", fvPatchFieldNew.C), so a run that ran had it.

    A patch's block, as v2406 finds it (`GeometricBoundaryField::readField`): the entry whose
    unquoted key is its name; else one whose unquoted key names one of its groups (the last
    written); else the entry `dictionary::findDict` finds by its name (its text, quoted or not,
    then the patterns), which is `deck.block`'s lookup. The group step is not followed here: a
    patch with no unquoted key of its own and an unquoted key naming one of its groups gives
    none (the review of dde0ecb: a `wall` entry spinning a cylinder that a pattern `".*"` read as
    `noSlip`)."""
    from courant.spec import deck
    from courant.spec.as_run import _same_time, _times

    t = float(start)
    out: dict[str, tuple] = {}
    internal: list[str | None] = []
    for root, mesh, member in zip(roots, meshes, groups):
        directory = next((d for d in _times(root) if _same_time(float(d), t)), None)
        if directory is None:
            return None
        rel = (root / directory / field).relative_to(case).as_posix()
        text = budget.read(case, rel)           # past the reading's bounds: None (`_MAX_RANKS`)
        if text is None or "#" in text:
            return None                         # a directive the reader does not follow
        if sum(1 for p in mesh.boundary if p.type != "processor" and p.n) > _MAX_PATCHES:
            return None
        top = deck._quoted_blocks(text)         # the file scanned once for its blocks
        bf = deck.block(text, "boundaryField", top)
        if bf is None:
            return None
        keys = deck._quoted_blocks(bf)
        if len(keys) > _MAX_ENTRIES:
            return None
        raw = deck.entry(text, "internalField")
        held = text.count("$")                  # none: none in any span of it, so no pass
        if held and held != _resolved_macros(raw, keys, macros):
            return None                         # a `$name;` entry may set any patch's block
        named = {k for k, quoted, _ in keys if not quoted}
        patterned = {k for k, quoted, _ in keys if quoted}
        path = case / rel

        def resolve(raw, text=text, path=path):
            """A value's `$macro`s from the same file (`deck._resolve_macros`), or None."""
            return None if raw is None else deck._resolve_macros(raw, text, path, case)

        value = resolve(raw) if raw is not None else None
        if value is not None and value.startswith("nonuniform") and not _ascii(text, top):
            value = None                        # a binary list: not text
        internal.append(value)
        read: dict[str, tuple | None] = {}      # each distinct block in it read once
        for p in mesh.boundary:
            if p.type == "processor" or not p.n:
                continue                        # a rank's empty share of a patch applies nowhere
            if p.type in _CONSTRAINT:
                got = (p.type, None)
            else:
                own = p.name in named and p.name not in patterned
                if not own and named & member[p.name]:
                    return None                 # v2406 may give it a group's block
                body = deck.block(bf, p.name, keys)
                if body is None:
                    return None
                if body not in read:
                    held = body.count("$")
                    read[body] = None if held and held != sum(
                        (deck.entry(body, k) or "").count("$") for k in macros
                    ) else condition(body, resolve)     # else a macro may set the type or merge
                got = read[body]
            if got is None or out.setdefault(p.name, got) != got:
                return None
    return out, internal


def _resolved_macros(internal: str | None, keys, macros) -> int:
    """How many of a field file's `$` this reader resolves itself: those in its `internalField`
    (`internal`, as `deck.entry` reads it) and, in each block of its boundaryField (`keys`, every
    writing), in the entries `macros` names, each entry's last definition as `deck.entry` reads it.
    Each is a different span of the file, so the count is at most the file's; any other `$` is one
    the reader does not follow. A `$name;` written as an entry expands the dictionary it names where
    it stands (`dictionary::substituteScopedKeyword`, merged under the default `#inputMode merge`),
    in the boundaryField or at the top of the file, and so may replace a patch's condition after the
    patch's own block (the review of dde0ecb, 26 Oct 2026)."""
    from courant.spec import deck

    n = (internal or "").count("$")
    for _, _, body in keys:
        n += sum((deck.entry(body, k) or "").count("$") for k in macros)
    return n


#: A per-cell vector list as v2406 writes an internalField (Field::writeEntry, Field.C:
#: `nonuniform` and `List<vector>` with its length, then the list).
_NONUNIFORM_VECTORS = re.compile(r"nonuniform\s+List<vector>\s*(?=\d)")


def _start_speed(raws, meshes) -> float | None:
    """The largest |U| of the field the run starts from (module docstring, "The speed"): every
    root's `<start>/U` internalField (`raws`, as `_boundary` gives them), as v2406 reads one
    (Field::assign, Field.C: `uniform` and one value, or `nonuniform` and a list whose length
    must be the mesh's cells, else "Size ... is not equal to the expected length"). Read here:
    `uniform (x y z)`, or `nonuniform List<vector> N (...)` of N finite 3-vectors, N the root's
    cells, nothing after it. Any other form (none, a list written `N{...}` or without its type,
    a binary list, `_boundary`) gives None. Vectorised (`polymesh._numbers`), so linear in the
    list's length."""
    import numpy as np

    from courant.case import polymesh

    fastest = 0.0
    for raw, mesh in zip(raws, meshes, strict=True):
        text = (raw or "").strip()
        if text.startswith("uniform"):
            v = _vector(text[len("uniform"):])
            if v is None:
                return None
            fastest = max(fastest, math.sqrt(sum(x * x for x in v)))
            continue
        m = _NONUNIFORM_VECTORS.match(text)
        if m is None:
            return None
        try:
            n, inner, end = polymesh._outer_list(text, m.end())
            if text[end:].strip() or text[m.end():text.index("(", m.end())].split() != [str(n)]:
                return None                     # something after the list, or before its count
            if n != mesh.n_cells or inner.count("(") != n:
                return None                     # not one value per cell (Field::assign)
            values, inside, counts = polymesh._numbers(inner, float)
        except (ValueError, Warning):          # `_numbers` raises a numpy warning as an error
            return None                         # not a list of numbers, or nested
        if len(counts) != n or not inside.all() or (counts != 3).any() \
                or not np.isfinite(values).all():
            return None
        if n:
            fastest = max(fastest, float(np.sqrt((values.reshape(n, 3) ** 2).sum(axis=1)).max()))
    return fastest


def _reynolds(case: Path, config, run) -> Reynolds | None:
    import numpy as np

    from courant.case.regime import solver_read_gravity
    from courant.spec.as_run import AsRun

    nu = _kinematic_nu(config, run.log_text)
    if nu is None or solver_read_gravity(run.log_text):
        return None                             # buoyancy may move it: the Rayleigh number's
    # A cheap first look, before any mesh is read: some patch's own velocity condition fixes one.
    fixing = {"fixedValue", "uniformFixedValue"}
    if not any(fields.get("U") in fixing for fields in (config.boundary or {}).values()):
        return None
    start, _ = AsRun(case, run).start
    if start is None:
        return None
    got = _roots_and_meshes(case, run, start)
    if got is None:
        return None
    roots, meshes, groups = got
    if _driven_otherwise(case, roots, config, run.log_text):
        return None
    surface = _surface(meshes)
    if surface is None:
        return None
    budget = _Budget()
    velocity = _boundary(case, roots, meshes, groups, start, "U", _condition, _U_MACROS, budget)
    pressure = _boundary(case, roots, meshes, groups, start, "p", _pressure, _SCALAR_MACROS,
                         budget)
    if velocity is None or pressure is None:
        return None
    conditions, pressure = velocity[0], {n: kind for n, (kind, _) in pressure[0].items()}
    start_speed = _start_speed(velocity[1], meshes)
    if start_speed is None:
        return None                             # a start field the reader cannot bound
    if set(conditions) != set(surface.kind) or set(pressure) != set(surface.kind):
        return None
    kinds = surface.kind
    walls = [n for n, k in kinds.items() if k == "wall"]
    for name in walls:                          # a stationary wall, or none of this
        kind, vel = conditions[name]
        if not (kind in _STILL_WALL or (kind == "fixedValue" and _zero(vel))):
            return None
    fixed: dict[str, _Velocity] = {}
    for name, k in kinds.items():
        kind, vel = conditions[name]
        if k == "patch":
            if vel is not None and not _zero(vel):
                fixed[name] = vel
            elif kind not in _NO_SPEED | {"inletOutlet"}:
                return None                     # a no-slip surface not typed wall, or a zero inlet
    if not fixed:
        return None
    held = _pressure_paths(kinds, conditions, pressure, fixed)
    if held is None or len(held) > 1:
        return None                             # a second opening held at a pressure
    patch = np.asarray(surface.patch)
    flux: dict[str, tuple[float, float]] = {}   # name -> (least, greatest) net flux over entries
    area: dict[str, float] = {}
    for name, vel in fixed.items():
        s = surface.areas[patch == name]
        total = s.sum(axis=0)
        area[name] = float(np.linalg.norm(s, axis=1).sum())
        q = [float(np.dot(v, total)) for v in vel.vectors]
        flux[name] = (min(q), max(q))
    inlets = []
    for name, vel in fixed.items():
        lo, hi = flux[name]
        scale = _TANGENTIAL * max(np.linalg.norm(v) for v in vel.vectors) * area[name]
        if hi < -scale:
            inlets.append(name)
        elif lo < -scale:
            return None                         # in at one time, out or along at another
    if not inlets:
        return None
    wall_faces = [f for f, n in zip(surface.faces, surface.patch) if kinds[n] == "wall"]
    if not wall_faces or "wedge" in kinds.values():
        return None
    return _external(surface, kinds, fixed, _solved(surface), wall_faces, nu, start_speed)


def _pressure_paths(kinds, conditions, pressure, fixed) -> list[str] | None:
    """The openings held at a pressure, or None where a fixed velocity is not its flux.

    Every patch that fixes the velocity (an inlet, a side, a wall) must leave the pressure free
    (`_FREE_P`): elsewhere the pressure drives its flux. An opening is a `patch` patch that fixes
    no velocity, through which the flow may pass: one whose velocity is `zeroGradient` or
    `inletOutlet` (mass crosses an `inletOutlet` patch held at a pressure whichever way, with its
    zero `inletValue` only where it enters), or a `slip` one held at a pressure (its velocity has
    no normal part, but the pressure's flux does). Held: its pressure not free."""
    for name, k in kinds.items():
        if k in ("wall", "patch") and (name in fixed or k == "wall") \
                and pressure[name] not in _FREE_P:
            return None
    held = []
    for name, k in kinds.items():
        if k != "patch" or name in fixed:
            continue
        free = pressure[name] in _FREE_P
        if conditions[name][0] == "slip" and free:
            continue                            # a surface the flow slides along
        if not free:
            held.append(name)
    return held


def _external(surface, kinds, fixed, solved, wall_faces, nu, start_speed) -> Reynolds | None:
    """Flow past one bluff body (module docstring), or None."""
    import numpy as np

    if len(set(fixed.values())) != 1:
        return None
    wall_points = set(int(v) for f in wall_faces for v in f)
    touching = set()
    for f, n in zip(surface.faces, surface.patch):
        if kinds[n] not in ("wall", "empty"):
            touching.update(int(v) for v in f)
    if wall_points & touching or len(set(_pieces(wall_faces).values())) != 1:
        return None
    vel = next(iter(fixed.values()))
    dims = int(np.sum(solved))
    has_empty = "empty" in kinds.values()
    if (dims == 2) != has_empty or dims < 2:
        return None                             # an empty patch off a coordinate axis, or 1-D
    vectors = np.asarray(vel.vectors, dtype=float)
    if np.any(np.abs(vectors[:, ~solved]) > 1e-9 * np.linalg.norm(vectors, axis=1)[:, None]):
        return None                             # a velocity along the direction not solved
    speeds = np.linalg.norm(vectors, axis=1)
    units = vectors / speeds[:, None]
    one_direction = bool(np.all(np.abs(units - units[0]) <= 1e-12))
    body = surface.points[sorted(wall_points)][:, solved]
    across: float | None = None                 # the lower bound's length
    if dims == 2:
        narrowest, extent = _widths(body)       # its least width and largest extent in the plane
        if one_direction:
            u = units[0][solved]
            proj = body @ np.array([-u[1], u[0]])
            across = narrowest = float(proj.max() - proj.min())
    else:
        if not one_direction:
            return None
        u = units[0]
        a = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(a) < 0.5:
            a = np.cross(u, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(u, a)
        across, wide = _widths(np.column_stack([body @ a, body @ b]))
        narrowest = across
        along = body @ u
        extent = math.hypot(wide, float(along.max() - along.min()))
    if not (narrowest > 0 and extent <= _BLUFF * narrowest):
        return None                             # not bluff: Roshko's bounds are a bluff body's
    speed = max(float(speeds.max()), start_speed)
    upper = speed * extent / nu
    lower = float(speeds[0]) * across / nu if vel.constant and across is not None else None
    if not _ok(upper) or (lower is not None and not _ok(lower)):
        return None
    return Reynolds(upper=upper, lower=lower, speed=speed, length=extent, nu=nu)


# ------------------------------------------------------------------------------ buoyancy alone
def _within(x: float | None, lo: float, hi: float) -> bool:
    """`x` in [lo, hi], within the last digits a written value may lose."""
    if x is None or not math.isfinite(x):
        return False
    tol = 1e-12 * max(1.0, abs(lo), abs(hi))
    return lo - tol <= x <= hi + tol


def _buoyancy_alone(case: Path, config, run, temperatures) -> bool:
    from courant.spec.as_run import AsRun

    t_cold, t_hot = (float(t) for t in temperatures)
    if not t_cold <= t_hot:
        return False
    start, _ = AsRun(case, run).start
    if start is None:
        return False
    got = _roots_and_meshes(case, run, start)
    if got is None:
        return False
    roots, meshes, groups = got
    budget = _Budget()
    if _driven_otherwise(case, roots, config, run.log_text) or \
            _radiates(case, roots, run.log_text, budget):
        return False
    kinds: dict[str, str] = {}
    for mesh in meshes:
        for p in mesh.boundary:
            if p.type == "processor" or not p.n:
                continue
            if p.type not in ("wall", "empty") or kinds.setdefault(p.name, p.type) != p.type:
                return False                    # an opening, an inlet or a mirror of the domain
    if "wall" not in kinds.values():
        return False
    velocity = _boundary(case, roots, meshes, groups, start, "U", _condition, _U_MACROS, budget)
    temperature = _boundary(case, roots, meshes, groups, start, "T", _temperature, _SCALAR_MACROS,
                            budget)
    if velocity is None or temperature is None:
        return False
    (u, u_internal), (t, t_internal) = velocity, temperature
    if set(u) != set(kinds) or set(t) != set(kinds):
        return False
    if any(_uniform_vector(raw) != (0.0, 0.0, 0.0) for raw in u_internal):
        return False                            # moving at the start
    if not all(_within(_uniform_scalar(raw), t_cold, t_hot) for raw in t_internal):
        return False                            # hotter or colder inside than the walls
    for name, k in kinds.items():
        if k != "wall":
            continue
        kind, vel = u[name]
        if not (kind in _STILL_WALL or (kind == "fixedValue" and _zero(vel))):
            return False                        # a moving wall (a lid)
        kind, value = t[name]
        if kind == "zeroGradient":
            continue                            # adiabatic
        if kind != "fixedValue" or not _within(value, t_cold, t_hot):
            return False                        # a heater the bound does not count
    return True


__all__ = ["Reynolds", "buoyancy_alone", "reynolds_number"]
