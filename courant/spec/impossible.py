"""An impossible specification (TRD SPEC-9, VER-5.11).

A specification can be valid JSON, valid `courant.spec/1`, and still ask for something no case
can be: a negative viscosity, a viscosity written in the dimensions of a velocity, a direction of
zero length, a bound whose minimum lies above its maximum. No run can conform to it, so a
verdict against it says nothing about the run: the gate answers `insufficient` with
`spec_invalid` and the reason, and does not compare the case (VER-15). A spec check is not a
physics check (PRD 2.0 §12, question 4).

What this is not: a specification that names a patch or a key the case does not have is not
impossible. That is the case not being the problem that was asked (a nonconformance, VER-5.7).

The quantities below are the ones a `courant.spec/1` names in OpenFOAM's own vocabulary: the
entry's key (`nu`, `mu`, `rho`, ...) or, for a boundary value, the field file it sits in (`0/T`).
An item whose quantity is not in the table is checked only for what needs no quantity (bounds,
cell counts). Deterministic and pure: no file is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath

from courant.spec.model import Spec, SpecItem


@dataclass(frozen=True)
class Quantity:
    what: str
    #: OpenFOAM's dimension set [mass length time temperature moles current luminosity]; None
    #: when the quantity is written in more than one (pressure: kinematic or in Pa).
    dims: tuple[tuple[float, ...], ...]
    #: "positive": > 0; "nonneg": >= 0; None: either sign is physical.
    sign: str | None


_P_KIN = (0, 2, -2, 0, 0, 0, 0)
_P_PA = (1, -1, -2, 0, 0, 0, 0)

#: Keys and field names, as OpenFOAM spells them.
QUANTITIES: dict[str, Quantity] = {
    "nu": Quantity("kinematic viscosity", ((0, 2, -1, 0, 0, 0, 0),), "nonneg"),
    "mu": Quantity("dynamic viscosity", ((1, -1, -1, 0, 0, 0, 0),), "nonneg"),
    "rho": Quantity("density", ((1, -3, 0, 0, 0, 0, 0),), "positive"),
    "Cp": Quantity("specific heat capacity", ((0, 2, -2, -1, 0, 0, 0),), "positive"),
    "Cv": Quantity("specific heat capacity", ((0, 2, -2, -1, 0, 0, 0),), "positive"),
    "molWeight": Quantity("molar mass", (), "positive"),
    "sigma": Quantity("surface tension", ((1, 0, -2, 0, 0, 0, 0),), "nonneg"),
    "T": Quantity("absolute temperature", ((0, 0, 0, 1, 0, 0, 0),), "positive"),
    "k": Quantity("turbulent kinetic energy", ((0, 2, -2, 0, 0, 0, 0),), "nonneg"),
    "epsilon": Quantity("turbulent dissipation rate", ((0, 2, -3, 0, 0, 0, 0),), "nonneg"),
    "omega": Quantity("specific dissipation rate", ((0, 0, -1, 0, 0, 0, 0),), "nonneg"),
    "nut": Quantity("turbulent viscosity", ((0, 2, -1, 0, 0, 0, 0),), "nonneg"),
    "nuTilda": Quantity("modified turbulent viscosity", ((0, 2, -1, 0, 0, 0, 0),), "nonneg"),
    "U": Quantity("velocity", ((0, 1, -1, 0, 0, 0, 0),), None),
    "g": Quantity("gravitational acceleration", ((0, 1, -2, 0, 0, 0, 0),), None),
    "p": Quantity("pressure", (_P_KIN, _P_PA), None),
    "p_rgh": Quantity("pressure", (_P_KIN, _P_PA), None),
    "endTime": Quantity("end time", (), "nonneg"),
    "deltaT": Quantity("time step", (), "positive"),
    "scale": Quantity("mesh scale", (), "positive"),
    "convertToMeters": Quantity("mesh scale", (), "positive"),
    "intensity": Quantity("turbulence intensity", (), "nonneg"),
    "mixingLength": Quantity("mixing length", (), "positive"),
}

#: Entries that hold a direction: a zero-length one points nowhere.
DIRECTION_KEYS = frozenset({"direction", "flowDir", "axis", "normal", "n", "e1", "e2", "e3"})

#: The entries of a boundary condition that hold the field's own quantity. Any other entry (a heat
#: flux `q` or `Q`, a `gradient`, `refGradient` or `uniformGradient`, a flow rate) holds something
#: else, and no rule of the field applies to it.
_FIELD_VALUE_KEYS = frozenset({None, "value", "inletValue", "outletValue", "refValue", "p0",
                               "p", "T0"})
#: Boundary-condition parameters with their own rule.
_BC_PARAMETERS = frozenset({"intensity", "mixingLength"})
#: Field files named by the quantity they hold.
_FIELDS = frozenset({"U", "p", "p_rgh", "T", "k", "epsilon", "omega", "nut", "nuTilda"})

#: The files that hold each property (a phase suffix, `.water`, is allowed). A key of the same
#: name elsewhere (an MRF zone's or a dynamic mesh's `omega`, a function object's `nu`) is a
#: different quantity and is not judged.
_PROPERTY_FILES: dict[str, frozenset[str]] = {
    "nu": frozenset({"transportProperties", "physicalProperties", "momentumTransport",
                     "turbulenceProperties"}),
    "mu": frozenset({"thermophysicalProperties", "physicalProperties", "transportProperties"}),
    "rho": frozenset({"transportProperties", "thermophysicalProperties", "physicalProperties"}),
    "Cp": frozenset({"thermophysicalProperties", "physicalProperties"}),
    "Cv": frozenset({"thermophysicalProperties", "physicalProperties"}),
    "molWeight": frozenset({"thermophysicalProperties", "physicalProperties"}),
    "sigma": frozenset({"transportProperties", "phaseProperties", "physicalProperties"}),
    "g": frozenset({"g"}),
    "endTime": frozenset({"controlDict"}),
    "deltaT": frozenset({"controlDict"}),
    "scale": frozenset({"blockMeshDict"}),
    "convertToMeters": frozenset({"blockMeshDict"}),
}

#: Solvers for which `T` is a thermodynamic (absolute) temperature. For any other, or when the
#: specification names no solver, `T` may be a passive scalar (scalarTransportFoam, which
#: `courant spec init` drafts too) and its sign is not judged.
_THERMO_SOLVERS = ("buoyant", "rho", "chtMultiRegion", "sonic", "thermoFoam", "reacting",
                   "XiFoam", "fireFoam", "compressible", "solidFoam")


def _base(file: str) -> str:
    return PurePosixPath(file).name.split(".", 1)[0]


def quantity_key(item: SpecItem) -> str | None:
    """The OpenFOAM name of the quantity the item holds, or None when no rule here applies: the
    field's name for a boundary entry that holds the field's value (`value`, `inletValue`, `p0`,
    ...); a property's key, only in a file that holds that property; a direction key; a boundary
    parameter with its own rule. None for hashes and cell counts."""
    if item.sha256 is not None or item.blocks is not None:
        return None
    if item.key in DIRECTION_KEYS:
        return item.key
    if item.patch is not None:
        if item.key in _BC_PARAMETERS:
            return item.key
        field = PurePosixPath(item.file).name
        return field if item.key in _FIELD_VALUE_KEYS and field in _FIELDS else None
    key = _base(item.file) if item.key == "value" else item.key   # constant/g: `value (..)`
    return key if _base(item.file) in _PROPERTY_FILES.get(key or "", ()) else None


def _thermo(spec: Spec) -> bool:
    solver = next((i.value for i in spec.items
                   if i.key == "application" and isinstance(i.value, str)), None)
    return bool(solver) and str(solver).startswith(_THERMO_SOLVERS)


def _pad(d) -> tuple[float, ...]:
    d = [float(x) for x in d]
    return tuple(d + [0.0] * (7 - len(d)))


def _numbers(value) -> list[float] | None:
    if value is None or isinstance(value, str):
        return None
    return [float(x) for x in (value if isinstance(value, (list, tuple)) else (value,))]


def item_problems(item: SpecItem, *, thermo: bool = False) -> list[str]:
    """Why no case can meet this item. Empty when one can. `thermo`: the specification names a
    solver for which `T` is an absolute temperature."""
    out: list[str] = []
    name = f"'{item.name}'"
    if item.min is not None and item.max is not None and item.min > item.max:
        out.append(f"{name}: its bound cannot be met (min {item.min:g} is above max "
                   f"{item.max:g})")
    nums = _numbers(item.value)
    if any(not math.isfinite(x) for x in (nums or [])) or any(
            b is not None and not math.isfinite(b) for b in (item.min, item.max)):
        out.append(f"{name}: a value that is not a finite number")
        return out
    if item.blocks == "cells":
        if nums is not None and any(x < 1 or not x.is_integer() for x in nums):
            out.append(f"{name}: a cell count must be a whole number of at least 1")
        if item.max is not None and item.max < 1:
            out.append(f"{name}: a mesh has at least one cell (max {item.max:g})")
        return out

    key = quantity_key(item)
    if key in DIRECTION_KEYS and nums is not None and len(nums) == 3 \
            and math.sqrt(sum(x * x for x in nums)) == 0.0:
        out.append(f"{name}: a direction of zero length")
    if key == "T" and not thermo:
        return out
    q = QUANTITIES.get(key or "")
    if q is None:
        return out
    if item.dimensions is not None and q.dims and _pad(item.dimensions) not in {
            _pad(d) for d in q.dims}:
        want = " or ".join("[" + " ".join(f"{x:g}" for x in d) + "]" for d in q.dims)
        got = "[" + " ".join(f"{x:g}" for x in item.dimensions) + "]"
        out.append(f"{name}: dimensions {got} do not fit a {q.what} ({want})")
    if q.sign is not None:
        scalar = nums is not None and len(nums) == 1
        lowest = nums[0] if scalar else None
        if lowest is not None and (lowest < 0 or (q.sign == "positive" and lowest == 0)):
            out.append(f"{name}: a {q.what} of {lowest:g} is impossible "
                       f"({'above zero' if q.sign == 'positive' else 'zero or more'} required)")
        if item.max is not None and (item.max < 0 or (q.sign == "positive" and item.max == 0)):
            out.append(f"{name}: a {q.what} bounded at max {item.max:g} is impossible")
    return out


def impossibilities(spec: Spec | dict) -> tuple[str, ...]:
    """SPEC-9: every reason the specification is impossible in itself, in item order. Empty
    when a case could conform to it."""
    if isinstance(spec, dict):
        spec = Spec.model_validate(spec)
    thermo = _thermo(spec)
    return tuple(p for item in spec.items for p in item_problems(item, thermo=thermo))
