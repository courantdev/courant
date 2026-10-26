"""TRD VER-19: a local-time-stepping run (`ddtSchemes default localEuler`) is judged on where it
ended up, behind `--checks next` (`case.local_euler`, GATE-13).

LTS gives each cell its own time step, set from `maxCo` and the PIMPLE dictionary's limits, so
the run's time is a pseudo-time: only its end state is a result, as on a steady run. v0.5.0
classed it transient and judged it on stability, which for LTS means almost nothing: an LTS
solver prints no Courant number (`rhoPimpleFoam.C` and `interFoam.C`, v2406: `if (LTS)
setRDeltaT.H; else CourantNo.H, setDeltaT.H`). So how far its residuals had fallen never
entered the verdict (an internal evaluation, 20 Oct 2026).

What does not change: LTS is not steady for relaxation (it has a pseudo time derivative, and
v2406's LTS tutorials ship `".*" 1`), nor for the stall detector.

The logs here are synthetic, in the shape v2406's rhoPimpleFoam prints with LTS on.
"""

from __future__ import annotations

import json

import pytest

from courant.case.config import CaseConfig
from courant.converge.series import Clock, regime_of
from courant.diagnosis.checkers import CheckContext, run_all_checkers
from courant.verification.checks import ConvergenceCheck, VerifyContext
from courant.verify import RunInputs, verify_inputs
from courant.verify.schema import InputFile
from courant.watcher.parser import LogParser


def _lts_log(final: dict[str, float], n: int = 80, solver: str = "rhoPimpleFoam") -> str:
    """An LTS run: integer pseudo-times, the flow time scale lines setRDeltaT.H prints, no
    Courant number; each field's initial residual falls geometrically from 1 to `final`."""
    out = [f"Exec   : {solver}", "", "PIMPLE: Operating solver in transient mode with 1 outer "
           "corrector", ""]
    for i in range(1, n + 1):
        out += ["Flow time scale min/max = 1e-07, 0.0001",
                "Smoothed flow time scale min/max = 1e-07, 0.0001",
                f"Time = {i}", ""]
        for name, last in final.items():
            r = last ** (i / n)
            out.append(f"DILUPBiCG:  Solving for {name}, Initial residual = {r:.6g}, "
                       f"Final residual = {r / 50:.6g}, No Iterations 3")
        out += [f"ExecutionTime = {0.02 * i:.2f} s  ClockTime = {i} s", ""]
    return "\n".join(out + ["End", ""])


def _lts(**kw) -> CaseConfig:
    return CaseConfig(**{"solver": "rhoPimpleFoam", "ddt_scheme": "localEuler", "delta_t": 1.0,
                         "end_time": 80.0, "turbulence_type": "RAS",
                         "relaxation_factors": {'".*"': 1.0},
                         "relaxation_scopes": {'".*"': "equations"}, **kw})


def _events(text: str):
    return list(LogParser(run_id="lts").parse_text(text))


def _verify(text: str, cfg: CaseConfig, checks: str):
    return verify_inputs(RunInputs(
        run_id="lts", log_text=text, log_name="log.rhoPimpleFoam",
        config_view=cfg.model_dump(mode="json"),
        inputs=(InputFile.of_text(text, "log.rhoPimpleFoam"),),
        source="corpus", solver_hint=cfg.solver, checks=checks))


# ---------------------------------------------------------------- the classification
@pytest.mark.parametrize("ddt, lts", [
    ("localEuler", True), ("localEuler rDeltaT", True),   # OpenFOAM's test: the first word
    ("localeuler", False),                                # a word OpenFOAM would not select
    ("Euler", False), ("steadyState", False), ("CoEuler phi rho 1", False), (None, False),
])
def test_lts_is_openfoams_own_test(ddt, lts):
    assert CaseConfig(ddt_scheme=ddt).is_local_euler() is lts


def test_lts_is_not_steady_and_marches_to_a_steady_state_only_under_next():
    cfg = _lts()
    assert not cfg.is_steady() and not cfg.is_steady("next")
    assert not cfg.marches_to_steady_state()                     # v0.5.0: transient
    assert cfg.marches_to_steady_state("next")
    assert not CaseConfig(solver="pimpleFoam", ddt_scheme="Euler").marches_to_steady_state("next")
    assert CaseConfig(solver="simpleFoam",
                      ddt_scheme="steadyState").marches_to_steady_state("next")


# ---------------------------------------------------------------- the residual bar
def test_an_lts_run_that_stopped_far_from_steady_is_judged_on_its_residuals_under_next():
    """p ends at 3e-2, three times the gate's MAJOR line. Under the default it is a transient
    run with no Courant number to fail, and passes."""
    events = _events(_lts_log({"Ux": 1e-4, "p": 3e-2, "e": 1e-4}))
    default = ConvergenceCheck().run(VerifyContext(run_id="r", events=events, config=_lts()))
    nxt = ConvergenceCheck().run(VerifyContext(run_id="r", events=events, config=_lts(),
                                               checks="next"))
    assert default.severity.value == "ok" and "transient" in default.detail
    assert nxt.severity.value == "major" and nxt.value == pytest.approx(3e-2)


def test_the_verdict_follows_under_next_and_the_default_keeps_v050s():
    text = _lts_log({"Ux": 1e-4, "p": 3e-2, "e": 1e-4})
    default, nxt = _verify(text, _lts(), "default"), _verify(text, _lts(), "next")
    assert (default.verdict, default.rule) == ("accept", "VER-5.6")
    assert (nxt.verdict, nxt.rule) == ("reject", "VER-5.3")
    assert "convergence.residual" in {r.code for r in nxt.reasons}
    assert json.loads(nxt.to_json())["tool"]["checks"] == "next"


def test_a_residual_just_above_the_target_raises_the_not_converged_finding_under_next():
    events = _events(_lts_log({"Ux": 1e-4, "p": 4e-3}))
    ids = lambda checks: {f.id for f in run_all_checkers(CheckContext(  # noqa: E731
        run_id="r", events=events, config=_lts(), checks=checks))}
    assert "residual.not_converged" not in ids("default")
    assert "residual.not_converged" in ids("next")


def test_a_converged_lts_run_is_accepted_in_both_modes():
    text = _lts_log({"Ux": 1e-5, "p": 2e-5, "e": 1e-5})
    for checks in ("default", "next"):
        v = _verify(text, _lts(), checks)
        assert (v.verdict, v.rule) == ("accept", "VER-5.6"), checks


# ---------------------------------------------------------------- what LTS is not
@pytest.mark.parametrize("checks", ["default", "next"])
def test_lts_relaxation_of_one_is_not_a_finding(checks):
    """v2406's LTS tutorials (angledDuctLTS, DLR_A_LTS, DTCHull, ...) ship `".*" 1`: the pseudo
    time derivative is the damping."""
    events = _events(_lts_log({"Ux": 1e-5, "p": 2e-5}))
    found = {f.id for f in run_all_checkers(CheckContext(run_id="r", events=events,
                                                         config=_lts(), checks=checks))}
    assert "scheme.relaxation" not in found


def test_an_lts_case_is_not_a_fixed_timestep_under_next():
    """`adjustTimeStep no` on an LTS case says nothing: each cell's step is set from maxCo."""
    events = _events(_lts_log({"Ux": 1e-5, "p": 2e-5}))
    cfg = _lts(adjust_timestep=False)
    found = lambda checks: {f.id for f in run_all_checkers(CheckContext(  # noqa: E731
        run_id="r", events=events, config=cfg, checks=checks))}
    assert "scheme.fixed_dt" in found("default")                 # v0.5.0's INFO note
    assert "scheme.fixed_dt" not in found("next")


def test_the_convergence_regime_is_steady_under_next_with_no_conflict():
    clock = Clock([float(i) for i in range(1, 11)], [None] * 10, [None] * 10)
    assert regime_of(_lts(), "rhoPimpleFoam", clock) == ("transient", "config", None)
    assert regime_of(_lts(), "rhoPimpleFoam", clock, "next") == ("steady", "config", None)
    # A real disagreement is still one.
    steady_pimple = CaseConfig(solver="pimpleFoam", ddt_scheme="steadyState")
    assert regime_of(steady_pimple, "pimpleFoam", clock, "next")[2] == {
        "config": "steady", "solver": "transient"}


# ---------------------------------------------------------------- the cause ranking asks the same
def test_the_fixed_step_predicate_asks_the_same_question_as_the_fixed_step_note():
    """The knowledge base's `fixed_dt_no_adaptivity` ranks `NUM.MISSING_ADJUSTABLE_DT` from the
    configuration. On an LTS case under `next` `scheme.fixed_dt` is silent (each cell's step is
    set from maxCo, whatever `adjustTimeStep` says), so the predicate must be too: otherwise a
    reject on an LTS run's residuals ranks adjustTimeStep as a cause. Under the default an LTS
    case is transient, as in v0.5.0, and both say it is a fixed step."""
    from courant.diagnosis.kb import CONFIG_PREDICATES
    from courant.diagnosis.kb_data import build_default_kb

    cfg = _lts(adjust_timestep=False)
    fixed = CONFIG_PREDICATES["fixed_dt_no_adaptivity"]
    assert fixed(cfg) and fixed(cfg, "default")
    assert not fixed(cfg, "next")
    assert fixed(CaseConfig(solver="pimpleFoam", ddt_scheme="Euler", adjust_timestep=False),
                 "next")

    def why(checks):
        return {m.code: m.why for m in build_default_kb().query(config=cfg, checks=checks)}

    assert "config:fixed_dt_no_adaptivity" in why("default").get("NUM.MISSING_ADJUSTABLE_DT", [])
    assert "config:fixed_dt_no_adaptivity" not in why("next").get(
        "NUM.MISSING_ADJUSTABLE_DT", [])


def test_a_reject_on_an_lts_runs_residuals_ranks_no_adjust_time_step_cause_under_next(
        monkeypatch):
    """The review's path: p ends at 3e-2, and VER-19 rejects it on its residuals under `next`.
    The ranking the verdict lists from holds no `NUM.MISSING_ADJUSTABLE_DT`."""
    from courant.diagnosis import deterministic

    seen = []
    real = deterministic.DeterministicDiagnoser.diagnose
    monkeypatch.setattr(deterministic.DeterministicDiagnoser, "diagnose",
                        lambda self, bundle: seen.append(real(self, bundle)) or seen[-1])
    v = _verify(_lts_log({"Ux": 1e-4, "p": 3e-2, "e": 1e-4}), _lts(adjust_timestep=False),
                "next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.3")
    assert "NUM.MISSING_ADJUSTABLE_DT" not in seen[-1].ranked_codes()
    assert "NUM.MISSING_ADJUSTABLE_DT" not in {c.code for c in v.probable_causes}
