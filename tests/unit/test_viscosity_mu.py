"""`physics.nu` on every viscosity the solver reads (TRD PRE-6).

The compressible and buoyant families set `mu` in `constant/thermophysicalProperties`, not `nu`.
The case reader has read it since T-CASE-THERMO, and its comment said that made a units error in
`mu` detectable, but the check read `nu` only: the hard-proofing audit (21 Oct 2026) set
hotRoom's `mu` to 1.8e-10, and to 1.8e-01 (nu about 1.5e-10 and 0.15 m^2/s, outside the check's
1e-8 to 1e-2), the run finished, and preflight said NO_SETUP_FAULT_FOUND and verify accepted it
at grade A, where the same value written as `nu` on an incompressible case is a setup fault.
interFoam's phases write `nu` in their own subdictionaries of `constant/transportProperties`
(damBreak's `water` and `air`), which the reader did not read at all: water at `nu 1e-01` was
NO_SETUP_FAULT_FOUND too.

Under `physics.nu.solved` (behind `--checks next`, GATE-13: a WARN finding turns an accept into a
reject) each Newtonian phase's `nu` is held to the range, and a `mu` is held to it as `mu / rho`,
the density from the case's equation of state (`courant.case.viscosity`). The cases are the
committed `tests/fixtures/buoyant/hotRoom_laminar_044` (perfectGas, p 1e5 Pa, walls at 300 to
600 K) with its `mu` edited, and a synthetic damBreak (`tests/unit/test_solved_pressure.py`).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.case.viscosity import NU_MAX, NU_MIN, kinematic
from courant.diagnosis.checkers import CheckContext, PhysicsChecker
from courant.verify.preflight import preflight_path

ROOT = Path(__file__).resolve().parents[2]
HOTROOM = ROOT / "tests" / "fixtures" / "buoyant" / "hotRoom_laminar_044"


def hotroom(tmp_path: Path, mu: str) -> Path:
    case = tmp_path / "hotRoom"
    shutil.copytree(HOTROOM, case)
    thermo = case / "constant" / "thermophysicalProperties"
    text = thermo.read_text()
    assert "mu              1.8e-05;" in text
    thermo.write_text(text.replace("mu              1.8e-05;", f"mu              {mu};"))
    return case


def _nu_findings(config: CaseConfig, checks: str):
    return [f for f in PhysicsChecker().run(CheckContext(run_id="r", events=[], config=config,
                                                         checks=checks)) if f.id == "physics.nu"]


def _air(mu: float, **kw) -> CaseConfig:
    return CaseConfig(transport={"mu": mu, "molWeight": 28.9, "Pr": 0.7}, **kw)


# ------------------------------------------------------------------ the kinematic viscosity
def test_a_perfect_gas_gives_nu_from_its_pressure_and_wall_temperatures():
    v = kinematic(_air(1.8e-5, equation_of_state="perfectGas", pressure=1e5,
                       fixed_temperature_range=(300.0, 600.0)), "next")
    # rho = p M / (R T): 1.159 kg/m^3 at 300 K, 0.579 at 600 K.
    assert v.lo == pytest.approx(1.8e-5 / 1.15869, rel=1e-4)
    assert v.hi == pytest.approx(1.8e-5 / 0.579345, rel=1e-4)
    assert "perfectGas" in v.basis and "300 to 600 K" in v.basis


def test_rhoconst_gives_nu_from_its_density():
    v = kinematic(CaseConfig(transport={"mu": 1e-3, "rho": 1000.0},
                             equation_of_state="rhoConst"), "next")
    assert v.lo == v.hi == pytest.approx(1e-6)


def test_an_unknown_density_bounds_nu_by_every_fluids_density():
    v = kinematic(_air(1.8e-5), "next")                  # no equation of state read
    assert v.lo < 1.8e-5 / 1.2 < v.hi and "density not read" in v.basis


def test_nu_written_as_nu_is_read_as_before_in_both_modes():
    cfg = CaseConfig(transport={"nu": 1e-5})
    assert kinematic(cfg).lo == kinematic(cfg, "next").hi == 1e-5


def test_the_default_reads_no_mu():
    assert kinematic(_air(1.8e-10, equation_of_state="perfectGas", pressure=1e5)) is None


# ------------------------------------------------------------------ the finding
@pytest.mark.parametrize("mu", [1.8e-10, 1.8e-1, 0.0])
def test_a_mu_whose_nu_lies_outside_the_range_is_flagged_under_next(mu):
    cfg = _air(mu, equation_of_state="perfectGas", pressure=1e5,
               fixed_temperature_range=(300.0, 600.0))
    (f,) = _nu_findings(cfg, "next")
    assert f.severity.value == "warn" and f.suggests == ["PHYS.TRANSPORT_UNITS"]
    assert f"mu={mu:g}" in f.detail and f"[{NU_MIN:g}, {NU_MAX:g}]" in f.detail
    assert _nu_findings(cfg, "default") == []            # v0.5.0 read nu only


def test_airs_own_mu_is_not_flagged():
    cfg = _air(1.8e-5, equation_of_state="perfectGas", pressure=1e5,
               fixed_temperature_range=(300.0, 600.0))
    assert _nu_findings(cfg, "next") == []


def test_with_no_density_only_a_value_no_fluid_could_have_is_flagged():
    assert _nu_findings(_air(1.8e-5), "next") == []
    assert _nu_findings(_air(1.8e-10), "next") == []     # air's units error, density unknown
    assert len(_nu_findings(_air(1e-14), "next")) == 1


def test_the_nu_finding_is_v050s_word_for_word():
    (f,) = _nu_findings(CaseConfig(transport={"nu": 1e-9}), "next")
    assert f.detail == "nu=1e-09 m^2/s is outside typical fluid range [1e-8, 1e-2]"
    assert f.value == 1e-9


def test_the_knowledge_base_predicate_reads_the_same_definition():
    from courant.diagnosis.kb import CONFIG_PREDICATES

    assert CONFIG_PREDICATES["nu_out_of_range"](CaseConfig(transport={"nu": 1e-9})) is True
    assert CONFIG_PREDICATES["nu_out_of_range"](CaseConfig(transport={"nu": 1e-5})) is False


# ------------------------------------------------------------------ preflight, end to end
@pytest.mark.parametrize("mu", ["1.8e-10", "1.8e-01"])
def test_the_audits_hotroom_is_a_setup_fault_under_next(tmp_path, mu):
    case = hotroom(tmp_path, mu)
    cfg = CaseConfig.from_foam(str(case))
    assert cfg.transport["mu"] == float(mu) and cfg.equation_of_state == "perfectGas"
    nxt = preflight_path(case, checks="next")
    assert nxt.verdict == "SETUP_FAULT"
    nu = next(r for r in nxt.reasons if r.code == "physics.nu")
    assert nu.input == "constant/transportProperties or constant/thermophysicalProperties"
    assert preflight_path(case).verdict == "NO_SETUP_FAULT_FOUND"     # v0.5.0


def test_the_shipped_hotroom_mu_is_not_flagged(tmp_path):
    case = hotroom(tmp_path, "1.8e-05")
    assert "physics.nu" not in {r.code for r in preflight_path(case, checks="next").reasons}


# ------------------------------------------------------------------ each phase's nu
def _dambreak(tmp_path: Path, water_nu: str = "1e-06", air: str | None = None) -> Path:
    from tests.unit.test_solved_pressure import OPEN_U, TOTAL_P_RGH, dambreak

    case = dambreak(tmp_path, atmosphere_p_rgh=TOTAL_P_RGH, atmosphere_u=OPEN_U)
    tp = case / "constant" / "transportProperties"
    text = tp.read_text()
    assert "water { transportModel Newtonian; nu 1e-06;" in text
    text = text.replace("water { transportModel Newtonian; nu 1e-06;",
                        f"water {{ transportModel Newtonian; nu {water_nu};")
    if air is not None:
        text = text.replace("air { transportModel Newtonian; nu 1.48e-05; rho 1; }", air)
    tp.write_text(text)
    return case


def test_the_reader_reads_each_newtonian_phases_nu(tmp_path):
    cfg = CaseConfig.from_foam(str(_dambreak(tmp_path)))
    assert cfg.phase_nu == {"water": 1e-06, "air": 1.48e-05}
    assert "nu" not in (cfg.transport or {})                   # no top-level nu in damBreak


def test_a_phase_whose_viscosity_model_is_not_newtonian_is_not_read(tmp_path):
    air = "air { transportModel CrossPowerLaw; CrossPowerLawCoeffs { nu0 1e-06; } rho 1; }"
    cfg = CaseConfig.from_foam(str(_dambreak(tmp_path, air=air)))
    assert cfg.phase_nu == {"water": 1e-06}


def test_a_phase_nu_outside_the_range_is_flagged_under_next():
    cfg = CaseConfig(phase_nu={"water": 0.1, "air": 1.48e-05})
    (f,) = _nu_findings(cfg, "next")
    assert f.value == 0.1 and f.suggests == ["PHYS.TRANSPORT_UNITS"]
    # A phase by its place, never its name: the name is the case author's, and a finding
    # reaches the Evidence Bundle (`config.phase_pseudonym`; the final review, 21 Oct 2026).
    assert f.detail == ("phase1 nu=0.1 m^2/s (constant/transportProperties, phase1 by its place "
                        "in `phases`) is outside typical fluid range [1e-8, 1e-2]")
    assert _nu_findings(cfg, "default") == []                  # v0.5.0 read top-level nu only


def test_every_phase_out_of_range_is_named():
    (f,) = _nu_findings(CaseConfig(phase_nu={"water": 0.1, "air": 1e-12}), "next")
    assert "phase1 nu=0.1" in f.detail and "phase2 nu=1e-12" in f.detail
    assert "water" not in f.detail and "air" not in f.detail


def test_the_audits_dambreak_is_a_setup_fault_under_next(tmp_path):
    case = _dambreak(tmp_path, water_nu="1e-01")
    nxt = preflight_path(case, checks="next")
    assert nxt.verdict == "SETUP_FAULT" and "physics.nu" in {r.code for r in nxt.reasons}
    assert preflight_path(case).verdict == "NO_SETUP_FAULT_FOUND"          # v0.5.0


def test_the_shipped_dambreak_phases_are_not_flagged(tmp_path):
    assert "physics.nu" not in {r.code for r in
                                preflight_path(_dambreak(tmp_path), checks="next").reasons}


def test_the_phases_do_not_cross_the_trust_boundary():
    """Phase names are the case author's words, like patch names: `phase_nu` is not sent."""
    from courant.diagnosis.redact import sanitize_config_view

    view = CaseConfig(phase_nu={"acme_coolant_v3": 1e-6}).model_dump(mode="json")
    assert "phase_nu" not in sanitize_config_view(view)
