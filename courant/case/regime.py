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

#: The onset of convection in a fluid layer between rigid plates heated from below, Ra = 1707.762
#: (Pellew and Southwell, Proc. R. Soc. A 176, 1940; Chandrasekhar, *Hydrodynamic and
#: Hydromagnetic Stability*, 1961, chapter II): below it buoyancy does not move the fluid, which
#: conducts. A side wall only raises the onset (a box is more stable than an unbounded layer). It
#: says nothing of a motion something else drives, so it is the laminar bound of `regime_numbers`
#: only where buoyancy alone moves the fluid (`courant.case.reynolds.buoyancy_alone`), against the
#: upper bound of the Rayleigh number (`BuoyantRegime.rayleigh_upper`).
RA_ONSET = 1707.762

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
    #: TRD VER-5b: an upper bound of the Rayleigh number, from the same reading
    #: (`_upper_rayleigh`): over the domain's whole extent along g, with the density contrast
    #: and the viscosity taken where they overstate it, between the held walls' temperatures.
    #: inf where it cannot be bounded. A bound only where nothing else moves or heats the fluid,
    #: which `regime_numbers` asks (`still`).
    rayleigh_upper: float = math.inf

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
                         temperatures=(t_cold, t_hot),
                         rayleigh_upper=_upper_rayleigh(config, mesh, down, g, t_cold, t_hot))


def _upper_rayleigh(config, mesh, down, g: float, t_cold: float, t_hot: float) -> float:
    """TRD VER-5b: an upper bound of the Rayleigh number of any motion in the domain, from the
    reading `buoyant_regime` made, each input taken where it overstates: the domain's whole extent
    along g (its bounding box's corners), which no distance between two of its points exceeds;
    ``g' = |g| (T_hot / T_cold - 1)`` for a perfect gas (``|g| beta dT`` with ``beta = 1 /
    T_cold``, above the density contrast at uniform pressure, `_fluid`'s ``|g| (1 - T_cold /
    T_hot)``), the case's ``|g| beta dT`` for a Boussinesq one; ``nu = mu / rho`` at ``T_cold``
    (the densest gas, so the least ``nu``), ``alpha = nu / Pr``. inf when any of it cannot be
    read."""
    extent = _heights(mesh.bounding_box, down)
    tp = config.transport or {}
    pr = tp.get("Pr")
    if extent is None or not _ok(pr):
        return math.inf
    length = extent[1] - extent[0]
    if config.equation_of_state is None:
        nu, beta = tp.get("nu"), tp.get("beta")
        if not _ok(nu, beta):
            return math.inf
        contrast = g * beta * (t_hot - t_cold)
    else:
        mu, mw, p = tp.get("mu"), tp.get("molWeight"), config.pressure
        if config.equation_of_state not in PERFECT_GAS or not _ok(mu, mw, p):
            return math.inf
        nu = mu / (p * mw / (R_UNIVERSAL * t_cold))
        contrast = g * (t_hot / t_cold - 1.0)
    try:
        ra = contrast * length ** 3 / (nu * nu / pr)
    except (OverflowError, ZeroDivisionError, ValueError):
        return math.inf
    return ra if _ok(ra) else math.inf


# ------------------------------------------------------------------------------ VER-5b
#: The circular cylinder's wake, Re on its diameter and the free stream's speed (Roshko, NACA
#: Report 1191, 1954): a stable laminar vortex street from 40 to 150, a transition range from 150
#: to 300, an irregular (turbulent) wake above 300. The laminar bound is the street's upper
#: limit, below the 3-D transition later work puts at 188.5 (Barkley and Henderson, J. Fluid
#: Mech. 322, 1996) and below a square cylinder's, about 160 (Robichaux, Balachandar and Vanka,
#: Phys. Fluids 11, 1999). A literature value, not a fitted one. Given for flow past one bluff
#: body only (`courant.case.reynolds`); no duct flow gives a Reynolds number (the final review,
#: of 8249ee4: no laminar bound is shown for every 2-D duct geometry).
RE_EXTERNAL_LAMINAR = 150.0
RE_EXTERNAL_TURBULENT = 300.0


@dataclass(frozen=True)
class RegimeNumber:
    """One regime number of a case (TRD VER-5b), as `physics.laminar_unstable` weighs it: an
    upper bound, which may set a transient run's residual history aside when it is below the
    flow's laminar bound, and a lower bound (None: not bounded), which the finding cites when it
    is above the flow's turbulent bound."""

    name: str                    # "Reynolds" or "Rayleigh"
    flow: str                    # the flow it is for, as a verdict says it
    upper: float
    lower: float | None
    laminar: float
    turbulent: float
    #: Where each bound comes from, as a verdict says it.
    laminar_words: str
    turbulent_words: str
    #: What each bound is taken over, as a verdict says it after `flow`.
    upper_on: str = ""
    lower_on: str = ""

    @property
    def below(self) -> bool:
        return self.upper < self.laminar

    @property
    def above(self) -> bool:
        return self.lower is not None and self.lower > self.turbulent


def regime_numbers(buoyant: BuoyantRegime | None, reynolds, *,
                   still: bool = False) -> tuple[RegimeNumber, ...]:
    """The case's regime numbers (TRD VER-5b): the Reynolds number the case gives unambiguously
    (`courant.case.reynolds.reynolds_number`, flow past one bluff body), and
    physics.laminar_regime's Rayleigh number for a case heated from below (`buoyant_regime`,
    read once per run). Either may be absent.

    The Rayleigh number's upper bound is `BuoyantRegime.rayleigh_upper` only where `still`, the
    case's files showing that nothing but buoyancy moves the fluid and nothing but the held walls
    heats it (`courant.case.reynolds.buoyancy_alone`); otherwise inf. Below the onset of
    convection buoyancy drives no motion, which says nothing of a flow a lid, an inlet or a
    source drives, nor bounds a heater the estimate does not count (the review of 26 Oct
    2026)."""
    out: list[RegimeNumber] = []
    if reynolds is not None:
        out.append(RegimeNumber(
            name="Reynolds", flow="flow past one bluff body",
            upper_on="on its largest extent", lower_on="on its extent across the flow",
            upper=reynolds.upper, lower=reynolds.lower,
            laminar=RE_EXTERNAL_LAMINAR, turbulent=RE_EXTERNAL_TURBULENT,
            laminar_words=("where a circular cylinder's wake is still a laminar vortex street "
                           "(Roshko 1954)"),
            turbulent_words="where a circular cylinder's wake is irregular (Roshko 1954)"))
    if buoyant is not None:
        out.append(RegimeNumber(
            name="Rayleigh", flow="natural convection heated from below",
            upper_on="over the domain's height, nothing but buoyancy moving the fluid",
            lower_on="between its hot and cold walls",
            upper=buoyant.rayleigh_upper if still else math.inf, lower=buoyant.rayleigh,
            laminar=RA_ONSET, turbulent=RA_TRANSITION,
            laminar_words=("where a layer heated from below does not yet convect "
                           "(Chandrasekhar 1961)"),
            turbulent_words=("where natural convection turns turbulent "
                             "(Incropera and DeWitt, section 9.6)")))
    return tuple(out)


__all__ = ["DNS_SPACING", "PERFECT_GAS", "RA_ONSET", "RA_TRANSITION", "RE_EXTERNAL_LAMINAR",
           "RE_EXTERNAL_TURBULENT", "R_UNIVERSAL", "BuoyantRegime", "RegimeNumber",
           "buoyant_regime", "regime_numbers", "solver_read_gravity"]
