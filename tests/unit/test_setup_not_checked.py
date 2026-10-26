"""What the setup rules read none of is said, not left silent (TRD PRE-7; hard-proofing audit,
21 Oct 2026).

`physics.nu` (the units check) read a top-level `nu` only, and `bc.overconstrained` (the boundary
check) a field named `p` only. On the families that write neither, the two never ran and nothing
said so: a hotRoom with `mu` 1.8e-05 -> 1.8e-01, a damBreak with water at `nu 1` and one whose
`atmosphere` fixes both `p_rgh` and `U` were each NO_SETUP_FAULT_FOUND, exit 0, with no line under
"not checked", while preflight's help promises "boundaries, units".

TRD PRE-5 and PRE-6 (`bc.solved_pressure`, `physics.nu.solved`) read them under `--checks next`.
Under the default (GATE-13: no verdict moves) preflight's `not_checked` and verify's
`not_assessed` now say which viscosity and which pressure the default checks did not read, and,
in both modes, when no viscosity the check reads was found at all. The cases are the committed
hotRoom 044 fixture with its `mu`, the synthetic damBreak of `tests/unit/test_solved_pressure.py`,
and the committed corpus sidecars of damBreak and hotRoom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.verify import sentences
from courant.verify.preflight import preflight_path, setup_not_checked

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"

MU = ("Viscosity was not checked against a fluid's range for mu in "
      "constant/thermophysicalProperties, which only --checks next reads.")
PHASES = ("Viscosity was not checked against a fluid's range for the nu of phases water, air in "
          "constant/transportProperties, which only --checks next reads.")
NONE = ("Viscosity was not checked against a fluid's range: no viscosity the check reads was found "
        "in constant/transportProperties or constant/thermophysicalProperties.")
P_RGH = ("Over-constrained pressure-velocity boundaries were checked on p and U only: this case's "
         "p_rgh, the pressure its solver solves, is read there only under --checks next.")
OURS = (MU, PHASES, NONE, P_RGH)


def _ours(lines) -> list[str]:
    """Ours among them, sorted: verify sorts its not-assessed sentences (`pipeline._dedupe`)."""
    return sorted(s for s in lines if s in OURS)


def _hotroom(tmp_path: Path, mu: str = "1.8e-05") -> Path:
    from tests.unit.test_viscosity_mu import hotroom

    return hotroom(tmp_path, mu)


def _dambreak(tmp_path: Path, *, water_nu: str = "1e-06", fixed: bool = False) -> Path:
    from tests.unit.test_solved_pressure import FIXED_P_RGH, FIXED_U, OPEN_U, TOTAL_P_RGH
    from tests.unit.test_viscosity_mu import _dambreak

    case = _dambreak(tmp_path, water_nu)
    if fixed:                                  # the audit's atmosphere: p_rgh and U both fixed
        for name, was, now in (("p_rgh", TOTAL_P_RGH, FIXED_P_RGH), ("U", OPEN_U, FIXED_U)):
            f = case / "0" / name
            f.write_text(f.read_text().replace(was, now))
    return case


# ---------------------------------------------------------------- the one function
def test_a_case_with_a_top_level_nu_and_p_says_nothing_new_in_either_mode():
    cfg = CaseConfig(transport={"nu": 1e-05}, boundary={"inlet": {"p": "zeroGradient",
                                                                  "U": "fixedValue"}})
    assert setup_not_checked(cfg, "default") == setup_not_checked(cfg, "next") == []


def test_mu_alone_is_named_by_default_and_read_under_next():
    cfg = CaseConfig(transport={"mu": 1.8e-05, "molWeight": 28.9}, equation_of_state="perfectGas")
    assert setup_not_checked(cfg, "default") == [MU]
    assert setup_not_checked(cfg, "next") == []


def test_each_phase_is_named_by_default_in_the_order_the_case_lists_them():
    cfg = CaseConfig(phase_nu={"water": 1e-06, "air": 1.48e-05})
    assert setup_not_checked(cfg, "default") == [PHASES]
    assert setup_not_checked(cfg, "next") == []


def test_no_viscosity_at_all_is_said_in_both_modes():
    cfg = CaseConfig(transport={"sigma": 0.07})
    assert setup_not_checked(cfg, "default") == setup_not_checked(cfg, "next") == [NONE]


def test_a_p_rgh_case_is_named_by_default_and_read_under_next():
    cfg = CaseConfig(transport={"nu": 1e-06},
                     boundary={"atmosphere": {"p_rgh": "totalPressure", "U": "fixedValue"}})
    assert setup_not_checked(cfg, "default") == [P_RGH]
    assert setup_not_checked(cfg, "next") == []


def test_no_configuration_says_nothing_here():
    """NO_CONFIG already says the setup rules were not run."""
    assert setup_not_checked(None, "default") == []


@pytest.mark.parametrize("solver", ["pimpleFoam", "simpleFoam", "icoFoam"])
def test_a_stray_p_rgh_on_a_solver_that_solves_p_is_not_called_its_pressure(solver):
    """The review of 21 Oct 2026: any `0/p_rgh` was called "the pressure its solver solves",
    whatever the solver, and `bc.overconstrained` under next asked it. Both now ask one helper,
    `CaseConfig.solved_pressures`, which reads `p_rgh` only for a solver that solves it."""
    cfg = CaseConfig(solver=solver, transport={"nu": 1e-06},
                     boundary={"atmosphere": {"p_rgh": "fixedValue", "U": "fixedValue"}})
    assert setup_not_checked(cfg, "default") == setup_not_checked(cfg, "next") == []
    assert cfg.solved_pressures("default") == cfg.solved_pressures("next") == ("p",)
    assert cfg.overconstrained_patches("next") == []


@pytest.mark.parametrize("solver", ["interFoam", "compressibleInterFoam", "buoyantPimpleFoam",
                                    "buoyantSimpleFoam", "driftFluxFoam", None, "foamRun"])
def test_p_rgh_is_the_solved_pressure_on_the_solvers_that_solve_it(solver):
    """And on a case whose solver is not known (no `application`, or org's `foamRun`, whose
    solver module the reader does not read): the case's own `0/p_rgh` is the evidence."""
    cfg = CaseConfig(solver=solver, transport={"nu": 1e-06},
                     boundary={"atmosphere": {"p_rgh": "fixedValue", "U": "fixedValue"}})
    assert cfg.solved_pressures("default") == ("p",)
    assert cfg.solved_pressures("next") == ("p", "p_rgh")
    assert cfg.overconstrained_patches("next") == ["atmosphere"]
    assert setup_not_checked(cfg, "default") == [P_RGH] and setup_not_checked(cfg, "next") == []


def test_the_fixed_sentences_carry_a_cause_tag():
    """`courant`: what this release of Courant does not read, which a newer release or
    `--checks next` may; not the solver fork (RRD RM-22 counts `fork`), and not the run (the
    review of 21 Oct 2026)."""
    assert sentences.NOT_CHECKED_CAUSE[sentences.VISCOSITY_NONE] == "courant"
    assert sentences.NOT_CHECKED_CAUSE[sentences.PRESSURE_P_RGH] == "courant"
    assert "courant" in sentences.CAUSES
    assert {e.cause for e in sentences.INSUFFICIENT.values()} <= {"run", "spec", "fork"}
    assert (sentences.VISCOSITY_NONE, sentences.PRESSURE_P_RGH) == (NONE, P_RGH)


# ---------------------------------------------------------------- preflight, end to end
@pytest.mark.parametrize("mu", ["1.8e-05", "1.8e-01"])
def test_the_audits_hotroom_says_its_mu_was_not_read_by_default(tmp_path, mu):
    """hotRoom 044 has no 0/U or 0/p_rgh, so only its viscosity is named."""
    case = _hotroom(tmp_path, mu)
    pre = preflight_path(case)
    assert pre.verdict == "NO_SETUP_FAULT_FOUND", "GATE-13: no default verdict moves"
    assert _ours(pre.not_checked) == [MU]
    nxt = preflight_path(case, checks="next")
    assert _ours(nxt.not_checked) == []
    assert nxt.verdict == ("SETUP_FAULT" if mu == "1.8e-01" else "NO_SETUP_FAULT_FOUND")


@pytest.mark.parametrize("water_nu,fixed", [("1e-06", False), ("1", False), ("1e-06", True)])
def test_the_audits_dambreaks_say_what_the_default_did_not_read(tmp_path, water_nu, fixed):
    case = _dambreak(tmp_path, water_nu=water_nu, fixed=fixed)
    pre = preflight_path(case)
    assert pre.verdict == "NO_SETUP_FAULT_FOUND", "GATE-13: no default verdict moves"
    assert _ours(pre.not_checked) == sorted([PHASES, P_RGH])
    nxt = preflight_path(case, checks="next")
    assert _ours(nxt.not_checked) == []
    faulty = water_nu != "1e-06" or fixed
    assert nxt.verdict == ("SETUP_FAULT" if faulty else "NO_SETUP_FAULT_FOUND")


def test_a_pitzdaily_shaped_case_is_unchanged(tmp_path):
    """The control: a top-level nu and a field named p."""
    case = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__shipped"
    assert _ours(preflight_path(case).not_checked) == []


# ---------------------------------------------------------------- verify, on the committed sidecars
@pytest.mark.parametrize("sid,default,nxt", [
    ("interFoam_damBreak__healthy", [NONE, P_RGH], [NONE]),
    ("buoyantPimpleFoam_hotRoom__healthy", [MU, P_RGH], []),
    ("rhoPimpleFoam_RAS_cavity__healthy", [MU], []),
    ("icoFoam_cavity__healthy", [], []),
])
def test_verify_says_it_under_not_assessed_and_moves_no_verdict(sid, default, nxt):
    """The damBreak sidecar was captured before the reader read phases, so it carries none: no
    viscosity is read from it in either mode."""
    from courant.verify import verify_path

    log, side = CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json"
    d, n = verify_path(log, side), verify_path(log, side, checks="next")
    assert _ours(d.not_assessed) == sorted(default)
    assert _ours(n.not_assessed) == sorted(nxt)
    pre_d, pre_n = preflight_path(log, side), preflight_path(log, side, checks="next")
    assert _ours(pre_d.not_checked) == sorted(default) and _ours(pre_n.not_checked) == sorted(nxt)
