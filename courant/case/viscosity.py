"""The kinematic viscosities a case sets, as its solver reads them (TRD PRE-6). One definition,
for `physics.nu` and the knowledge base's `nu_out_of_range`.

An incompressible case writes `nu` in `constant/transportProperties`: that is the number. An
interFoam case writes one per phase, in the phase's own subdictionary of that file (`phases
(water air); water { transportModel Newtonian; nu 1e-06; }`). The compressible and buoyant
families write a dynamic viscosity, `mu`, in `constant/thermophysicalProperties`
(`mixture/transport/mu`, `const` transport), and the solver divides it by a density its equation
of state gives. v0.5.0 checked the top-level `nu` only, so a units error in a phase's `nu` or in
`mu` was never checked (the hard-proofing audit, 21 Oct 2026). Under `physics.nu.solved` (behind
`--checks next`, GATE-13) each Newtonian phase's `nu` is checked as written, and `mu` through
that density:

- `perfectGas`: ``rho = p M / (R T)`` at the case's absolute pressure (`0/p`'s uniform internal
  field, else `pRef`) and `molWeight`, over the temperatures its walls are held at (`0/T`), or
  over `T_RANGE` when it holds none;
- `rhoConst`: its `rho`;
- anything else, or a density the reader could not read: every density a fluid in a continuum
  solve has, `RHO_RANGE`. Only a `mu` no fluid could pair with a plausible `nu` is then flagged.

The result is a range of `nu`, and it is out of range only when all of it is: each bound is
taken where it understates the fault.

Not read, in either mode: a phase's `mu` or `nu` in its own `thermophysicalProperties.<phase>`
or `physicalProperties.<phase>` (compressibleInterFoam, OpenFOAM.org's layout), and a
non-Newtonian model's coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from courant.case.regime import R_UNIVERSAL

#: The plausible kinematic viscosity of a fluid, m^2/s (v0.5.0's `physics.nu` range).
NU_MIN, NU_MAX = 1e-8, 1e-2
#: K. Wider than any temperature a case without held walls is likely to run at.
T_RANGE = (50.0, 5000.0)
#: kg/m^3. A gas at a few hundred pascals to a liquid metal.
RHO_RANGE = (1e-3, 2.5e4)

@dataclass(frozen=True)
class Viscosity:
    """The kinematic viscosity a case implies, m^2/s, as a range."""

    lo: float
    hi: float
    #: "nu" (written as such) or "mu" (divided by a density).
    source: str
    mu: float | None = None
    #: How the density was obtained, for a finding's sentence.
    basis: str = ""
    #: The phase whose `nu` this is (`CaseConfig.phase_nu`), or None for the case's own.
    phase: str | None = None
    #: How a finding names that phase (`config.phase_pseudonym`): the name is the case
    #: author's and stays local.
    alias: str | None = None

    @property
    def out_of_range(self) -> bool:
        return self.lo <= 0 or self.hi < NU_MIN or self.lo > NU_MAX


def _positive(*xs) -> bool:
    try:
        return all(math.isfinite(x) and x > 0 for x in xs)
    except TypeError:
        return False


def _density(config) -> tuple[float, float, str]:
    """(least, greatest, how) density of the fluid, kg/m^3."""
    tp = config.transport or {}
    eos = config.equation_of_state
    if eos == "perfectGas" and _positive(tp.get("molWeight"), config.pressure):
        temps = config.fixed_temperature_range
        held = temps is not None and _positive(*temps)
        t_lo, t_hi = temps if held else T_RANGE
        p, mw = config.pressure, tp["molWeight"]
        how = (f"density from perfectGas at p = {p:g} Pa, T {t_lo:g} to {t_hi:g} K"
               + ("" if held else " (no wall temperature read)"))
        return p * mw / (R_UNIVERSAL * t_hi), p * mw / (R_UNIVERSAL * t_lo), how
    if eos == "rhoConst" and _positive(tp.get("rho")):
        return tp["rho"], tp["rho"], f"density {tp['rho']:g} kg/m^3 (rhoConst)"
    lo, hi = RHO_RANGE
    return lo, hi, (f"density not read ({eos or 'no equation of state'}): any fluid's, "
                    f"{lo:g} to {hi:g} kg/m^3")


def viscosities(config, checks: str = "default") -> tuple[Viscosity, ...]:
    """Every kinematic viscosity `config` sets that the gate reads, the case's own first.

    The top-level `nu` when the case writes it, in every mode. Under `physics.nu.solved` (behind
    `--checks next`) also each Newtonian phase's `nu`, in the order the case lists its phases,
    and, when the case writes no top-level `nu`, its `mu` through the density. Under the default
    the answer is v0.5.0's."""
    tp = config.transport or {}
    nu = tp.get("nu")
    out: list[Viscosity] = [Viscosity(nu, nu, "nu")] if nu is not None else []
    from courant.verify.checks import enabled

    if not enabled("physics.nu.solved", checks):
        return tuple(out)
    mu = tp.get("mu")
    if nu is None and mu is not None:
        rho_lo, rho_hi, how = _density(config)
        out.append(Viscosity(mu / rho_hi, mu / rho_lo, "mu", mu=mu, basis=how))
    from courant.case.config import phase_pseudonym

    phase_nu = getattr(config, "phase_nu", None) or {}
    listed = getattr(config, "phases", None) or list(phase_nu)  # a config written before `phases`
    out.extend(Viscosity(v, v, "nu", phase=phase, alias=phase_pseudonym(listed, phase))
               for phase, v in phase_nu.items())
    return tuple(out)


def kinematic(config, checks: str = "default") -> Viscosity | None:
    """The case's own kinematic viscosity (not a phase's), or None when it sets none the gate
    reads: the first of `viscosities` that is not a phase's."""
    return next((v for v in viscosities(config, checks) if v.phase is None), None)


__all__ = ["NU_MAX", "NU_MIN", "RHO_RANGE", "T_RANGE", "Viscosity", "kinematic", "viscosities"]
