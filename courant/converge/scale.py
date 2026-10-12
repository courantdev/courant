"""Tolerances and the reference-scale ladder (TRD CONV-8, M1-C-1 to C-6).

Force and moment coefficients are judged in counts: one count is 1e-4 in a coefficient, which
is a tolerance an aerodynamicist recognises. Everything else is judged relatively, at 1% of a
reference scale, and the scale is chosen by a ladder whose rung is always recorded.

Two stated deviations from M1:
- **Coefficient names match by prefix** (`^C[dlmsyn]`), not by word boundary, so `CmPitch` and
  `Cd(f)` count as coefficients. Under `\\b` they would have been judged at 1%.
- **A vector component is measured against its vector's mean magnitude**, before its own
  mean. On the first real run (pitzDaily, a 2-D case) the z-components of the wall force
  were round-off, 1e-21 to 1e-24. Judged against their own means, they were "not converged"
  and "in a limit cycle", which is absurd. A component's tolerance is a share of the force
  it is a component of: for the dominant component the two scales coincide, and for a minor
  one the vector's scale is the meaningful one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from courant.converge.exact import mean, variance

TOL_ABS_COEFFICIENT = 1.0e-4
TOL_REL_DEFAULT = 0.01

_COEFFICIENT = re.compile(r"^C[dlmsyn]")


def is_coefficient(column: str) -> bool:
    return bool(_COEFFICIENT.match(column)) or "coeff" in column.lower()


@dataclass(frozen=True, slots=True)
class Tolerance:
    branch: str          # "absolute" | "relative"
    value: float
    source: str          # "default" | "user"


def _lookup(table: Mapping[str, float], monitor: str, column: str) -> float | None:
    if monitor in table:
        return table[monitor]
    return table.get(column)


def tolerance(monitor: str, column: str, tol_abs: Mapping[str, float],
              tol_rel: Mapping[str, float]) -> Tolerance:
    """The criterion for one monitor. Both kinds supplied for one monitor is an error."""
    a = _lookup(tol_abs, monitor, column)
    r = _lookup(tol_rel, monitor, column)
    if a is not None and r is not None:
        raise ValueError(f"both an absolute and a relative tolerance were given for {monitor}")
    if a is not None:
        return Tolerance("absolute", float(a), "user")
    if r is not None:
        return Tolerance("relative", float(r), "user")
    if is_coefficient(column):
        return Tolerance("absolute", TOL_ABS_COEFFICIENT, "default")
    return Tolerance("relative", TOL_REL_DEFAULT, "default")


@dataclass(frozen=True, slots=True)
class Scale:
    value: float | None   # None only for the constant rung
    basis: str            # user | mean | vector_magnitude | stdev | constant
    mean: float
    stdev: float


def reference_scale(values: Sequence[float], *, user: float | None = None,
                    vector_magnitude: float | None = None) -> Scale:
    """R_q for one monitor over the values it is judged on."""
    m = mean(values) if values else 0.0
    sd = math.sqrt(variance(values)) if len(values) > 1 else 0.0
    if user is not None and user > 0:
        return Scale(float(user), "user", m, sd)
    if vector_magnitude is not None and vector_magnitude > 0.0:
        return Scale(vector_magnitude, "vector_magnitude", m, sd)
    if abs(m) >= sd and m != 0.0:
        return Scale(abs(m), "mean", m, sd)
    if sd > 0.0:
        return Scale(sd, "stdev", m, sd)
    return Scale(None, "constant", m, sd)
