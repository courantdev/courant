"""The buoyant regime of a case heated from below, from what its solver read (TRD VER-17).

A laminar run claims to resolve every scale of the motion it computes: no model stands in for
the ones it does not. On a case whose walls are held hotter below than above (Rayleigh-Benard
convection, in any geometry) that claim can be checked against the case's own numbers, with no
reference and no specification:

- **Is the flow turbulent?** The Rayleigh number over the distance along gravity between the
  hottest and the coldest held walls, ``Ra = g' d^3 / (nu alpha)``, against ``RA_TRANSITION``.
- **Could the mesh resolve it anyway?** The mean cell size against Groetzbach's criterion for a
  direct simulation, ``Delta <= pi eta`` (``DNS_SPACING``), with eta the Kolmogorov scale.

Both are estimated **conservatively**, so each can only understate the case against the run:
the Rayleigh number is a lower bound and the Kolmogorov scale an upper bound.

- **Heated from below, or nothing.** Every face held at the hottest temperature lies below every
  face held at the coldest, measured along ``g`` (the corners of the two faces' bounding boxes).
  A case heated from above is stably stratified, and one whose hot and cold walls share a level
  (OpenFOAM's hotRoom: a heater set into a cooler floor, no cold wall above it) has no distance
  the temperature difference provably acts across. Neither gets an estimate.
- **The length.** Ra uses ``d``, the least separation between the hot and the cold faces: Ra grows
  as ``d^3``, so the least gives a lower bound. The Kolmogorov bound grows with the length, as
  ``D^(1/4)`` (below), so it uses ``D``, the greatest separation.
- ``g'`` is the density contrast at uniform pressure, ``|g| (1 - T_cold / T_hot)`` for a perfect
  gas, which is below ``|g| beta dT`` with ``beta = 1 / T_cold`` (``|g| beta dT`` with the case's
  ``beta`` for a Boussinesq solver, one that read no equation of state). ``|g|`` is the part of
  gravity in the solved directions: a component along an ``empty`` direction moves nothing.
- ``nu = mu / rho`` at the hot temperature (the lightest gas, so the largest ``nu``), and
  ``alpha = nu / Pr``.
- ``eta <= D Pr^(1/2) Ra_D^(-1/4) = Pr^(1/2) D^(1/4) (g' / (nu alpha))^(-1/4)``: the exact
  Rayleigh-Benard balance ``epsilon = (nu^3 / D^4) (Nu - 1) Ra Pr^-2`` (Shraiman and Siggia, Phys.
  Rev. A 42, 1990) with ``Nu - 1 >= 1``, which holds far below any Rayleigh number this check
  acts on.
- The cell size is the mean cell volume's edge (in 2-D, the mean cell area's, the empty
  direction's thickness divided out).

Anything it cannot read, or any number that is not finite and positive, gives no estimate.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass

#: J/(kmol K), as OpenFOAM's `specie` uses it with `molWeight` in kg/kmol.
R_UNIVERSAL = 8314.462618

#: The onset of turbulent natural convection: Ra about 1e9 along a vertical heated surface
#: (Incropera and DeWitt, *Fundamentals of Heat and Mass Transfer*, 6th ed., section 9.6). It is
#: the highest of the textbook values (a heated plate facing up turns turbulent near 1e7, Lloyd
#: and Moran 1974; Rayleigh-Benard convection near 4e7, Heslot, Castaing and Libchaber 1987), so
#: the check uses the one least likely to call a laminar flow turbulent. A literature value, not
#: a fitted one.
RA_TRANSITION = 1.0e9

#: Groetzbach's resolution criterion for a direct simulation of turbulent convection: a mean
#: grid spacing of at most pi times the Kolmogorov scale (Groetzbach, J. Comput. Phys. 49, 1983).
DNS_SPACING = math.pi

#: Equations of state whose density is the perfect gas law at the pressure `CaseConfig.pressure`
#: reads. `incompressiblePerfectGas` evaluates it at its own `pRef`, which is not read, so it is
#: not here: such a case gets no estimate.
PERFECT_GAS = frozenset({"perfectGas"})

#: What OpenFOAM's `readGravitationalAcceleration.H` prints when a solver reads `constant/g`
#: (`Info<< "\nReading g" << endl;`). A `constant/g` the solver never read moves nothing, so the
#: check asks the solver's own output, not a list of solver names.
_READ_G = re.compile(r"^Reading g[ \t]*$", re.MULTILINE)


def solver_read_gravity(log_text: str | None) -> bool:
    """Whether the solver's log says it read `constant/g`."""
    return bool(log_text) and _READ_G.search(log_text[:200_000]) is not None


@dataclass(frozen=True)
class BuoyantRegime:
    rayleigh: float              # a lower bound, over `height`
    kolmogorov: float            # m, an upper bound, over `span`
    cell: float                  # m, the mean cell size
    height: float                # m, the least separation of the hot and cold walls along g
    span: float                  # m, the greatest
    reduced_gravity: float       # m/s^2
    nu: float                    # m^2/s
    alpha: float                 # m^2/s
    temperatures: tuple[float, float]

    @property
    def turbulent(self) -> bool:
        return self.rayleigh > RA_TRANSITION

    @property
    def unresolved(self) -> bool:
        return self.cell > DNS_SPACING * self.kolmogorov


def _ok(*xs) -> bool:
    """Every value a finite positive number."""
    try:
        return all(math.isfinite(x) and x > 0 for x in xs)
    except TypeError:
        return False


def _fluid(config, t_cold: float, t_hot: float, g: float) -> tuple[float, float, float] | None:
    """(g', nu, alpha) from what the solver read, or None."""
    tp = config.transport or {}
    pr = tp.get("Pr")
    if not _ok(pr):
        return None
    if config.equation_of_state is None:                  # no thermophysicalProperties read
        nu, beta = tp.get("nu"), tp.get("beta")
        if not _ok(nu, beta):
            return None
        return g * beta * (t_hot - t_cold), nu, nu / pr
    if config.equation_of_state not in PERFECT_GAS:
        return None
    mu, mw, p = tp.get("mu"), tp.get("molWeight"), config.pressure
    if not _ok(mu, mw, p):
        return None
    rho_hot = p * mw / (R_UNIVERSAL * t_hot)
    if not _ok(rho_hot):
        return None
    nu = mu / rho_hot
    return g * (1.0 - t_cold / t_hot), nu, nu / pr


def _heights(box, down) -> tuple[float, float] | None:
    """(lowest, highest) height of a box's corners, height measured against `down` (unit g)."""
    if box is None:
        return None
    hs = [-sum(c[k] * down[k] for k in range(3))
          for c in itertools.product(*zip(box[0], box[1]))]
    return (min(hs), max(hs)) if all(math.isfinite(h) for h in hs) else None


def _cell(mesh, solved) -> float | None:
    box, volume, n = mesh.bounding_box, mesh.total_volume, mesh.n_cells
    if box is None or not _ok(volume, n):
        return None
    span = [hi - lo for lo, hi in zip(box[0], box[1])]
    dims = sum(solved)
    if dims == 3:
        return (volume / n) ** (1.0 / 3.0)
    if dims == 2:
        thickness = next(s for s, d in zip(span, solved) if not d)
        return math.sqrt(volume / thickness / n) if _ok(thickness) else None
    return None


def buoyant_regime(config, mesh) -> BuoyantRegime | None:
    """The conservative estimate above for this case, or None when it cannot be made."""
    if config is None or mesh is None or config.gravity is None:
        return None
    solved = mesh.solution_directions
    if solved is None or len(solved) != 3:
        return None
    g_vec = [c if d else 0.0 for c, d in zip(config.gravity, solved)]
    if not all(math.isfinite(c) for c in g_vec):
        return None
    g = math.sqrt(sum(c * c for c in g_vec))
    temps = config.fixed_temperature_range
    if not _ok(g) or temps is None:
        return None
    t_cold, t_hot = temps
    if not _ok(t_cold, t_hot) or not t_cold < t_hot:
        return None
    down = [c / g for c in g_vec]
    hot, cold = _heights(config.hot_faces_box, down), _heights(config.cold_faces_box, down)
    if hot is None or cold is None:
        return None
    height, span = cold[0] - hot[1], cold[1] - hot[0]    # heated from below: cold above hot
    fluid = _fluid(config, t_cold, t_hot, g)
    cell = _cell(mesh, solved)
    if fluid is None or cell is None or not _ok(height, span, cell):
        return None
    reduced, nu, alpha = fluid
    if not _ok(reduced, nu, alpha, nu * alpha):
        return None
    pr = nu / alpha
    per_m3 = reduced / (nu * alpha)
    try:                                         # a float power overflows by raising
        ra = per_m3 * height ** 3
        eta = math.sqrt(pr) * span ** 0.25 * per_m3 ** -0.25 if _ok(per_m3) else math.nan
    except (OverflowError, ZeroDivisionError, ValueError):
        return None
    if not _ok(ra, eta):
        return None
    return BuoyantRegime(rayleigh=ra, kolmogorov=eta, cell=cell, height=height, span=span,
                         reduced_gravity=reduced, nu=nu, alpha=alpha,
                         temperatures=(t_cold, t_hot))


__all__ = ["DNS_SPACING", "PERFECT_GAS", "RA_TRANSITION", "R_UNIVERSAL", "BuoyantRegime",
           "buoyant_regime", "solver_read_gravity"]
