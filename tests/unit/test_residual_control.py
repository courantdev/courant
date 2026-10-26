"""The residual bar per field from `residualControl`, and the stalled-residual detector.

S's decision, recorded in RRD RM-35 (13 Oct 2026): the flat 1e-3 bar stays until a detector of
its own catches a stalled residual history; then the bar follows each field's declared
`residualControl` tolerance, both shipped together as one gate version.

The fixtures are excerpts of real RM-35 interim logs (`tests/fixtures/residual_control/`: the
`Exec`, `Time =`, `Solving for`, `bounding`, convergence and `End` lines, nothing else):

- `c10k.simpleFoam_pitzDaily.058.healthy`: a healthy reference. Every field ends under its own
  `residualControl` (p 1.02e-3 against 1e-2); the flat bar rejected it.
- `c10k.simpleFoam_pitzDaily.021.catalogue` and `.045.catalogue`: the catalogue's
  `residual_plateau` fault (equation relaxation 0.05). OpenFOAM prints "SIMPLE solution
  converged" after 594 and 523 iterations; the history barely moves after step 50. Only the flat
  bar rejected them.

**Behind `--checks next` until 0.6.0** (TRD GATE-13, S's decision of 15 Oct 2026): the bar per
field is `residual.control` and the stall `residual.stalled` in `courant.verify.checks.NEXT`.
Every judgement, context and verdict here runs under `next`; the default mode's v0.5.0
behaviour on the same runs is `tests/unit/test_checks_next.py`'s.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.diagnosis.checkers import CheckContext as _CheckContext
from courant.diagnosis.checkers import ResidualAnalyzer, run_all_checkers
from courant.events import EventKind
from courant.verification.checks import ConvergenceCheck
from courant.verification.checks import VerifyContext as _VerifyContext
from courant.verification.convergence import (
    RESIDUAL_LOOSE,
    RESIDUAL_TARGET,
    declared_tolerance,
    equation_relaxation,
    residual_bar,
)
from courant.verification.convergence import judge_residuals as _judge_residuals
from courant.watcher.parser import LogParser

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "residual_control"
CORPUS = ROOT / "certification" / "corpus"
MONITORED = ROOT / "certification" / "monitored"

#: VER-16 ships behind `--checks next` (GATE-13): every judgement in this module is under it.
CHECKS = "next"


def judge_residuals(events, config, checks: str = CHECKS):
    return _judge_residuals(events, config, checks)


def CheckContext(**kw):
    return _CheckContext(**{"checks": CHECKS, **kw})


def VerifyContext(**kw):
    return _VerifyContext(**{"checks": CHECKS, **kw})


#: pitzDaily's own `SIMPLE/residualControl`, OpenFOAM v2406 as shipped.
PITZ_RC = {"p": 0.01, "U": 0.001, '"(k|epsilon|omega|f|v2)"': 0.001}


def _text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def _events(path: Path):
    return list(LogParser(run_id=path.name).parse_text(_text(path)))


def _pitz_config(*, relaxation: float = 0.9, residual_control: dict | None = PITZ_RC) -> dict:
    """The certification corpus's pitzDaily sidecar (read from a real staged v2406 case), with
    the relaxation and residualControl a test names."""
    cfg = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())["config"]
    cfg["relaxation_factors"] = {"U": relaxation, '".*"': relaxation}
    cfg["relaxation_scopes"] = {"U": "equations", '".*"': "equations"}
    if residual_control is not None:
        cfg["residual_control"] = {"SIMPLE": dict(residual_control)}
    else:
        cfg.pop("residual_control", None)
    return cfg


def _steady(control: dict) -> CaseConfig:
    """A steady simpleFoam case whose SIMPLE/residualControl is `control`."""
    return CaseConfig(residual_control={"SIMPLE": control} if control else {},
                      solver="simpleFoam", ddt_scheme="steadyState")


def _relaxed(tmp_path: Path, relaxation: str) -> CaseConfig:
    """The configuration the case reader gives for `system/fvSolution` holding
    `relaxationFactors { <relaxation> }`."""
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
        f"relaxationFactors\n{{\n    {relaxation}\n}}\n")
    return CaseConfig.from_foam(str(tmp_path))


def _verify(log: Path, config: dict, tmp_path: Path, checks: str = CHECKS):
    """`courant verify <log> --case-json <sidecar> --checks next`, the certification corpus's
    path."""
    from courant.verify.ingest import from_log
    from courant.verify.pipeline import verify_inputs_v1

    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"] = config
    side = tmp_path / (log.name.split(".log")[0] + ".case.json")
    side.write_text(json.dumps(meta))
    return verify_inputs_v1(from_log(log, side).model_copy(update={"checks": checks}))


def _codes(verdict) -> set[str]:
    return {r.code for r in verdict.reasons}


HEALTHY_058 = FIX / "c10k.simpleFoam_pitzDaily.058.healthy.log.gz"
PLATEAU_021 = FIX / "c10k.simpleFoam_pitzDaily.021.catalogue.log.gz"
PLATEAU_045 = FIX / "c10k.simpleFoam_pitzDaily.045.catalogue.log.gz"


# ------------------------------------------------------------------ reading residualControl
class TestTheReaderReadsResidualControl:
    def test_simple_scalars_and_a_regex_key_in_declaration_order(self):
        cfg = CaseConfig.from_foam(str(FIX / "simple_regex"))
        assert cfg.residual_control == {"SIMPLE": PITZ_RC}
        assert list(cfg.residual_control["SIMPLE"]) == ["p", "U", '"(k|epsilon|omega|f|v2)"']

    def test_esi_pimple_dictionaries_are_not_recorded(self):
        # ESI's PIMPLE `residualControl { p { tolerance; relTol; } }` ends the outer loop inside
        # a time step; it is no run criterion, so it is not read as one.
        assert CaseConfig.from_foam(str(FIX / "pimple_dict")).residual_control == {}

    def test_foundation_pimple_scalars_are_recorded_under_pimple(self):
        cfg = CaseConfig.from_foam(str(FIX / "foamrun_pimple"))
        assert cfg.residual_control["PIMPLE"] == {"p": 1e-3, "U": 1e-4}
        assert cfg.residual_control["SIMPLE"] == {"p": 0.5}           # stale, never read

    def test_a_case_without_residual_control_declares_nothing(self):
        assert CaseConfig.from_foam(str(FIX / "no_control")).residual_control == {}

    def test_the_stock_tutorial_as_committed(self):
        cfg = CaseConfig.from_foam(str(MONITORED / "simpleFoam_pitzDaily__shipped"))
        assert cfg.residual_control == {"SIMPLE": PITZ_RC}


# ------------------------------------------------------------------ one bar per field
class TestTheBarFollowsOpenFOAMsMatching:
    steady = CaseConfig.model_validate(_pitz_config())

    def test_vector_components_take_the_vector_fields_tolerance(self):
        # OpenFOAM checks `U` against the largest of its solved components; the log prints Ux, Uy.
        assert declared_tolerance(self.steady, "Ux") == 1e-3
        assert declared_tolerance(self.steady, "Uy") == 1e-3

    def test_a_component_key_never_applies_and_other_names_are_not_components(self):
        cfg = _steady({"Ux": 1e-6, "Tx": 1e-4})
        assert declared_tolerance(cfg, "Ux") is None          # OpenFOAM looks up `U`, not `Ux`
        assert declared_tolerance(cfg, "Tx") == 1e-4          # a scalar named Tx is Tx

    def test_a_declared_tolerance_is_met_strictly_as_in_openfoam(self):
        events = LogParser(run_id="r").parse_text(_multi_solve_log())
        assert judge_residuals(events, _steady({"p": 3e-3, "U": 1e-3})).over == ("p",)

    def test_a_quoted_key_is_a_regular_expression_matched_whole(self):
        assert declared_tolerance(self.steady, "k") == 1e-3
        assert declared_tolerance(self.steady, "epsilon") == 1e-3
        assert declared_tolerance(self.steady, "kl") is None        # whole-word match only

    def test_the_first_matching_entry_wins(self):
        # solutionControl::applyToField returns the first entry, in declaration order, that
        # matches the field name.
        cfg = _steady({'".*"': 1e-4, "p": 1e-2})
        assert declared_tolerance(cfg, "p") == 1e-4

    def test_the_bar_is_the_declared_tolerance(self):
        assert residual_bar(self.steady, "p") == 1e-2
        assert residual_bar(self.steady, "Uy") == 1e-3

    def test_an_undeclared_field_keeps_the_flat_bar(self):
        assert residual_bar(self.steady, "nuTilda") == RESIDUAL_TARGET

    def test_no_residual_control_keeps_the_flat_bar(self):
        cfg = CaseConfig.model_validate(_pitz_config(residual_control=None))
        assert residual_bar(cfg, "p") == RESIDUAL_TARGET

    def test_no_config_keeps_the_flat_bar(self):
        assert residual_bar(None, "p") == RESIDUAL_TARGET

    def test_a_transient_case_keeps_the_flat_bar(self):
        # PIMPLE's residualControl is a tolerance on the outer loop inside one time step, not a
        # convergence criterion for the run; a transient run is judged on stability anyway.
        cfg = CaseConfig(residual_control={"SIMPLE": {"p": 1e-2}}, solver="pimpleFoam",
                         ddt_scheme="Euler")
        assert residual_bar(cfg, "p") == RESIDUAL_TARGET

    def test_a_declared_tolerance_cannot_loosen_the_bar_past_the_loose_band(self):
        # Above RESIDUAL_LOOSE the gate has always graded a residual MAJOR. A residualControl of
        # 1 would otherwise let a run "converge" on its second iteration.
        cfg = _steady({"p": 1.0})
        assert residual_bar(cfg, "p") == RESIDUAL_LOOSE

    def test_a_tighter_declared_tolerance_tightens_the_bar(self):
        cfg = _steady({"p": 1e-5})
        assert residual_bar(cfg, "p") == 1e-5


class TestTheDictionaryIsTheSolversAlgorithms:
    """Which residualControl applies is the solver's algorithm, not the file's layout
    (`courant.case.solvers.residual_control_loop`)."""

    def test_a_leftover_pimple_block_on_simplefoam_is_never_read(self):
        cfg = CaseConfig(residual_control={"PIMPLE": {"p": 1e-2}}, solver="simpleFoam",
                         ddt_scheme="steadyState")
        assert residual_bar(cfg, "p") == RESIDUAL_TARGET

    def test_esi_steady_pimple_has_no_run_criterion(self):
        cfg = CaseConfig.from_foam(str(FIX / "pimple_dict"))
        cfg = cfg.model_copy(update={"solver": "pimpleFoam", "ddt_scheme": "steadyState"})
        assert residual_bar(cfg, "p") == RESIDUAL_TARGET

    def test_foamrun_reads_pimple_and_ignores_a_stale_simple_block(self):
        cfg = CaseConfig.from_foam(str(FIX / "foamrun_pimple"))
        assert cfg.solver == "foamRun" and cfg.is_steady()
        assert residual_bar(cfg, "p") == 1e-3
        assert residual_bar(cfg, "Ux") == 1e-4

    def test_the_loop_is_named_in_one_place(self):
        from courant.case.solvers import residual_control_loop

        assert residual_control_loop("simpleFoam") == "SIMPLE"
        assert residual_control_loop("mpirun -np 4 rhoSimpleFoam -parallel") == "SIMPLE"
        assert residual_control_loop("foamRun") == "PIMPLE"
        assert residual_control_loop("pimpleFoam") is None
        assert residual_control_loop(None) is None


class TestAMalformedToleranceGivesAVerdict:
    """A tolerance of 0 divided by zero; NaN from a JSON sidecar was never exceeded."""

    def test_zero_is_declared_and_never_met(self):
        cfg = _steady({"p": 0.0})
        assert declared_tolerance(cfg, "p") == 0.0
        judged = judge_residuals(_events(HEALTHY_058), cfg)
        assert judged.over == ("p",) and judged.field == "p"

    def test_a_negative_tolerance_is_never_met(self):
        judged = judge_residuals(_events(HEALTHY_058), _steady({"p": -1e-3, "U": 1e-3}))
        assert "p" in judged.over and judged.field == "p"

    def test_nan_is_no_tolerance(self):
        cfg = _steady({"p": float("nan")})
        assert declared_tolerance(cfg, "p") is None
        assert residual_bar(cfg, "p") == RESIDUAL_TARGET

    @pytest.mark.parametrize("tol", [0.0, -1.0, float("nan"), float("inf")])
    def test_the_gate_gives_a_verdict(self, tol, tmp_path):
        cfg = _pitz_config(residual_control={"p": tol, "U": 1e-3})
        v = _verify(HEALTHY_058, cfg, tmp_path)
        assert v.verdict in ("TRUSTWORTHY", "NOT_TRUSTWORTHY")
        if tol <= 0:
            assert v.verdict == "NOT_TRUSTWORTHY" and "residual.not_converged" in _codes(v)


def _multi_solve_log() -> str:
    """Two iterations, each solving p twice (as under non-orthogonal correctors)."""
    lines = ["Exec   : simpleFoam"]
    for i, first_p in ((1, "0.5"), (2, "3e-3")):
        lines += [f"Time = {i}",
                  "smoothSolver:  Solving for Ux, Initial residual = 1e-4, "
                  "Final residual = 1e-6, No Iterations 2",
                  f"GAMG:  Solving for p, Initial residual = {first_p}, "
                  "Final residual = 1e-7, No Iterations 5",
                  "GAMG:  Solving for p, Initial residual = 2e-4, "
                  "Final residual = 1e-7, No Iterations 5"]
    return "\n".join(lines + ["End"]) + "\n"


class TestTheDeclaredBarReadsTheFirstSolve:
    """residualControl compares each field's FIRST initial residual in an iteration
    (`simpleControl::criteriaSatisfied`, `sp.first()`); the parser kept only the last."""

    def test_the_parser_keeps_both(self):
        steps = [e.payload for e in LogParser(run_id="r").parse_text(_multi_solve_log())
                 if e.kind is EventKind.RESIDUAL]
        assert steps[-1].initial_residuals["p"] == 2e-4               # unchanged: the last
        assert steps[-1].first_initial_residuals["p"] == 3e-3         # new: the first

    def test_a_declared_bar_is_judged_on_the_first_solve(self):
        events = LogParser(run_id="r").parse_text(_multi_solve_log())
        judged = judge_residuals(events, _steady({"p": 1e-3, "U": 1e-3}))
        assert judged.final["p"] == 3e-3 and judged.over == ("p",)

    def test_the_flat_bar_keeps_the_value_it_was_always_judged_on(self):
        """Under the default checks, v0.5.0's judgement. Under `next` VER-16a reads the first
        solve for the flat bar too (the test below; `tests/unit/test_first_solve.py`)."""
        events = LogParser(run_id="r").parse_text(_multi_solve_log())
        judged = judge_residuals(events, _steady({}), "default")
        assert judged.final["p"] == 2e-4 and judged.over == ()

    def test_under_next_the_flat_bar_reads_the_first_solve_too(self):
        """TRD VER-16a (`residual.first_solve`): residualControl's convention for every field."""
        events = LogParser(run_id="r").parse_text(_multi_solve_log())
        judged = judge_residuals(events, _steady({}))
        assert judged.final["p"] == 3e-3 and judged.over == ("p",)


class TestEquationRelaxation:
    def test_equations_scope_with_a_regex_default(self):
        cfg = CaseConfig.model_validate(_pitz_config(relaxation=0.05))
        assert equation_relaxation(cfg, "Ux") == 0.05
        assert equation_relaxation(cfg, "k") == 0.05

    def test_an_exact_key_beats_a_pattern(self):
        cfg = CaseConfig(relaxation_factors={'".*"': 0.5, "k": 0.7},
                         relaxation_scopes={'".*"': "equations", "k": "equations"})
        assert equation_relaxation(cfg, "k") == 0.7
        assert equation_relaxation(cfg, "epsilon") == 0.5

    def test_field_relaxation_is_not_equation_relaxation(self):
        # `fields { p 0.3; }` (airFoil2D) relaxes the p *field* after its solve; the p matrix,
        # and so the residual OpenFOAM prints for it, is not changed.
        cfg = CaseConfig(relaxation_factors={"p": 0.3, "U": 0.7},
                         relaxation_scopes={"p": "fields", "U": "equations"})
        assert equation_relaxation(cfg, "p") == 1.0
        assert equation_relaxation(cfg, "Ux") == 0.7

    def test_the_pressure_equation_is_not_relaxed_by_a_catch_all(self):
        # simpleFoam relaxes U, k and epsilon through `UEqn.relax()` and friends, and p through
        # `p.relax()` (a field relaxation): `".*"` under `equations` never reaches p's matrix.
        cfg = CaseConfig.model_validate(_pitz_config(relaxation=0.05))
        assert equation_relaxation(cfg, "p") == 1.0
        assert equation_relaxation(cfg, "p_rgh") == 1.0

    def test_nothing_declared_is_unrelaxed(self):
        assert equation_relaxation(CaseConfig(), "Ux") == 1.0
        assert equation_relaxation(None, "Ux") == 1.0

    @pytest.mark.parametrize(("relaxation", "u", "k"), [
        # `solution::relaxEquation` falls back to the `default` entry of `equations`
        ("equations { default 0.05; }", 0.05, 0.05),
        ('equations { "U.*" 0.5; default 0.05; }', 0.5, 0.05),
        # `fields` and `equations` are two dictionaries: one never overwrites the other
        ("equations { U 0.05; } fields { U 0.9; }", 0.05, 1.0),
        ("fields { U 0.9; } equations { U 0.05; }", 0.05, 1.0),
        # with either sub-dictionary, a flat entry beside it is no equation's factor
        ("fields { p 0.3; } U 0.05;", 1.0, 1.0),
        # with neither, the flat dictionary is the equations' (`needsCompat`), default included
        ("p 0.3; U 0.05; default 0.5;", 0.05, 0.5),
        # a keyword not evaluated may be the one found, or nothing is and the default applies
        ('equations { "(?=U)U" 0.5; default 0.05; }', 0.05, 0.05),
        ('equations { "(?=k)k" 0.02; default 0.5; }', 0.02, 0.02),
        # a factor the reader cannot read is an entry found, not one to look past
        ("equations { U table ((0 0.5) (10 0.9)); default 0.05; }", 1.0, 0.05),
    ])
    def test_read_as_v2406_solution_reads_it(self, tmp_path, relaxation, u, k):
        """v2406's `solution::read` takes `relaxationFactors/equations` whole (or, with neither
        `fields` nor `equations`, the flat dictionary), and `relaxEquation` looks the name up in
        it and then its `default` (`src/OpenFOAM/matrices/solution/solution.C`). The reader
        flattened `fields` and `equations` into one map, so a later `fields { U .. }` overwrote
        `equations { U .. }`, and the `default` was never read: both hid an under-relaxation
        below 0.1, the only one the stall detector can fire on (the review of the 0.6.0
        follow-ups, 22 Oct 2026)."""
        config = _relaxed(tmp_path, relaxation)
        assert (equation_relaxation(config, "Ux"), equation_relaxation(config, "k")) == (u, k)
        assert equation_relaxation(config, "p") == 1.0          # pressure: a field relaxation

    def test_the_flattened_view_the_checkers_read_is_unchanged(self, tmp_path):
        """GATE-13: `relaxation_factors` and `relaxation_scopes` are what v0.5.0 read; the
        equations' dictionary is a field of its own."""
        config = _relaxed(tmp_path, "equations { U 0.05; } fields { U 0.9; }")
        assert config.relaxation_factors == {"U": 0.9}
        assert config.relaxation_scopes == {"U": "fields"}
        assert config.relaxation_equations == {"U": 0.05}

    def test_a_sidecar_from_before_reads_the_flattened_view(self):
        """A configuration captured before the equations' dictionary was read (a corpus sidecar)
        is read from `relaxation_factors` and `relaxation_scopes`, `default` included."""
        cfg = CaseConfig(relaxation_factors={"default": 0.05},
                         relaxation_scopes={"default": "equations"})
        assert cfg.relaxation_equations is None
        assert equation_relaxation(cfg, "Ux") == 0.05
        mixed = CaseConfig(relaxation_factors={"p": 0.3, "U": 0.05},
                           relaxation_scopes={"p": "fields", "U": ""})
        assert equation_relaxation(mixed, "Ux") == 1.0


# ------------------------------------------------------------------ the stall detector
class TestTheStalledResidualDetector:
    @pytest.mark.parametrize("log", [PLATEAU_021, PLATEAU_045], ids=["021", "045"])
    def test_the_residual_plateau_twins_are_stalled(self, log):
        cfg = CaseConfig.model_validate(_pitz_config(relaxation=0.05))
        judged = judge_residuals(_events(log), cfg)
        assert not judged.over                       # every field under its residualControl
        assert judged.stall is not None
        s = judged.stall
        assert s.window == 200 and s.steps in (594, 523)
        assert s.relaxation == 0.05
        # The deflation undone puts the deciding field a decade or more above its bar.
        assert s.residual / s.relaxation >= 10 * s.bar

    @pytest.mark.parametrize("relaxation", ["equations { default 0.05; }",
                                            "equations { U 0.05; } fields { U 0.9; }"],
                             ids=["default", "fields-after-equations"])
    @pytest.mark.parametrize("log", [PLATEAU_021, PLATEAU_045], ids=["021", "045"])
    def test_an_under_relaxation_the_solver_reads_is_a_stall(self, log, relaxation, tmp_path):
        """The review's two cases: OpenFOAM relaxes U's matrix by 0.05 in both, and the reader
        took 1.0 (the `default` not read; `fields { U 0.9; }` overwriting `equations { U 0.05;
        }`), so the plateau twins' spurious "converged" was accepted. Read as the case writes it,
        through the verdict's own path (`--case-json`)."""
        read = _relaxed(tmp_path, relaxation)
        cfg = {**_pitz_config(), **read.model_dump(
            include={"relaxation_factors", "relaxation_scopes", "relaxation_equations"})}
        judged = judge_residuals(_events(log), CaseConfig.model_validate(cfg))
        assert not judged.over and judged.stall is not None
        assert judged.stall.relaxation == 0.05
        v = _verify(log, cfg, tmp_path)
        assert (v.verdict, v.rule) == ("NOT_TRUSTWORTHY", "VER-5.4")
        assert "residual.stalled" in _codes(v)

    def test_a_healthy_reference_is_not_stalled(self):
        judged = judge_residuals(_events(HEALTHY_058), CaseConfig.model_validate(_pitz_config()))
        assert not judged.over and judged.stall is None

    def test_the_same_history_at_the_tutorials_relaxation_is_not_a_stall_finding(self):
        # A flat history alone is not the finding: at the tutorial's 0.9 the residual OpenFOAM
        # prints is within a few percent of the unrelaxed one, so a field just under its
        # tolerance is just under it.
        judged = judge_residuals(_events(PLATEAU_021), CaseConfig.model_validate(_pitz_config()))
        assert judged.stall is None

    def test_a_transient_run_is_never_judged_for_a_stall(self):
        cfg = CaseConfig.model_validate({**_pitz_config(relaxation=0.05), "ddt_scheme": "Euler",
                                    "solver": "pimpleFoam"})
        assert judge_residuals(_events(PLATEAU_021), cfg).stall is None

    @pytest.mark.parametrize("steps, stalls", [(150, False), (250, True)])
    def test_a_run_shorter_than_the_window_is_not_judged_for_a_stall(self, steps, stalls):
        # A flat history just under its bar, under relaxation 0.05: M1-P-2 has no window below
        # 200 iterations (M1-F-1), so the plateau is not decided, and neither is the stall.
        lines = ["Exec   : simpleFoam"]
        for i in range(1, steps + 1):
            lines += [f"Time = {i}",
                      "smoothSolver:  Solving for k, Initial residual = 9e-4, "
                      "Final residual = 1e-6, No Iterations 2"]
        events = LogParser(run_id="r").parse_text("\n".join(lines + ["End"]) + "\n")
        cfg = CaseConfig.model_validate(_pitz_config(relaxation=0.05))
        assert (judge_residuals(events, cfg).stall is not None) is stalls

    @pytest.mark.parametrize("log", [PLATEAU_021, PLATEAU_045], ids=["021", "045"])
    def test_the_stalled_field_is_one_the_convergence_block_calls_plateaued(self, log):
        # One definition of a plateau (M1-P-2): residual.stalled cannot contradict the
        # convergence block's CONV.RESIDUAL_PLATEAU / CONV.RESIDUAL_FALLING for its field.
        from courant.converge.series import from_events
        from courant.converge.steady import residual_summary

        events = _events(log)
        stall = judge_residuals(events, CaseConfig.model_validate(
            _pitz_config(relaxation=0.05))).stall
        series, _, _ = from_events(events)
        assert residual_summary(stall.field, series[stall.field]).plateaued is True

    def test_no_config_no_stall(self):
        assert judge_residuals(_events(PLATEAU_021), None).stall is None

    @pytest.mark.parametrize("case", ["simpleFoam_pitzDaily__fixed_budget",
                                      "simpleFoam_T3A__fixed_budget",
                                      "simpleFoam_pitzDailyExptInlet__fixed_budget"])
    def test_a_stock_tutorial_resting_on_its_floor_is_not_a_stall_finding(self, case):
        # Fixed iteration budgets with residualControl removed: pitzDaily sits flat at 1e-5 to
        # 1.6e-4 for its last 1,500 iterations, T3A at 1e-6 to 1e-9. Flat, and converged.
        d = MONITORED / case
        judged = judge_residuals(_events(d / "log.simpleFoam"), CaseConfig.from_foam(str(d)))
        assert judged.stall is None and not judged.over


# ------------------------------------------------------------------ one decision, two reporters
AGREEMENT_CASES = {
    "058 with residualControl": (HEALTHY_058, _pitz_config()),
    "058 without residualControl": (HEALTHY_058, _pitz_config(residual_control=None)),
    "021 at 0.05": (PLATEAU_021, _pitz_config(relaxation=0.05)),
    "045 at 0.05": (PLATEAU_045, _pitz_config(relaxation=0.05)),
    "021 at 0.05, no residualControl": (PLATEAU_021,
                                        _pitz_config(relaxation=0.05, residual_control=None)),
    "021 at the tutorial's 0.9": (PLATEAU_021, _pitz_config()),
}


class TestResidualAnalyzerAndConvergenceCheckAgree:
    @pytest.mark.parametrize("mode", ["next", "default"])
    @pytest.mark.parametrize("name", sorted(AGREEMENT_CASES))
    def test_a_finding_exactly_when_the_check_is_not_ok(self, name, mode):
        log, cfg = AGREEMENT_CASES[name]
        events = _events(log)
        config = CaseConfig.model_validate(cfg)
        findings = {f.id for f in ResidualAnalyzer().run(CheckContext(
            run_id="r", events=events, config=config, checks=mode))}
        check = ConvergenceCheck().run(VerifyContext(run_id="r", events=events, config=config,
                                                     checks=mode))
        residual_findings = findings & {"residual.not_converged", "residual.stalled"}
        assert bool(residual_findings) == (check.severity.value != "ok"), (findings, check)


# ------------------------------------------------------------------ the gate's verdicts
class TestTheVerdicts:
    def test_058_is_accepted_under_its_own_residual_control(self, tmp_path):
        v = _verify(HEALTHY_058, _pitz_config(), tmp_path)
        assert v.verdict == "TRUSTWORTHY", [(r.code, r.summary) for r in v.reasons]
        assert not _codes(v) & {"residual.not_converged", "residual.stalled",
                                "convergence.residual"}

    def test_058_without_residual_control_is_judged_exactly_as_before(self, tmp_path):
        v = _verify(HEALTHY_058, _pitz_config(residual_control=None), tmp_path)
        assert (v.verdict, v.rule) == ("NOT_TRUSTWORTHY", "VER-5.4")
        by_code = {r.code: r for r in v.reasons}
        assert by_code["residual.not_converged"].summary == (
            "Final residual above convergence criterion: final worst residual 0.00102 > 0.001")
        assert by_code["convergence.residual"].summary == (
            "final worst residual 1.02e-03 (target 1e-03)")
        assert "residual.stalled" not in by_code

    @pytest.mark.parametrize("log", [PLATEAU_021, PLATEAU_045], ids=["021", "045"])
    def test_the_plateau_twins_are_rejected_by_the_stall_detector(self, log, tmp_path):
        v = _verify(log, _pitz_config(relaxation=0.05), tmp_path)
        assert (v.verdict, v.rule) == ("NOT_TRUSTWORTHY", "VER-5.4")
        codes = _codes(v)
        assert "residual.stalled" in codes
        assert "residual.not_converged" not in codes      # the bar alone would have accepted
        stalled = next(r for r in v.reasons if r.code == "residual.stalled")
        assert "0.05" in stalled.summary and "200" in stalled.summary

    def test_the_stock_pitzdaily_is_accepted(self, tmp_path):
        log = CORPUS / "simpleFoam_pitzDaily__healthy.log.gz"
        v = _verify(log, _pitz_config(), tmp_path)
        assert v.verdict == "TRUSTWORTHY", [(r.code, r.summary) for r in v.reasons]

    def test_an_airfoil2d_like_run_that_meets_its_tolerances_is_accepted(self, tmp_path):
        # airFoil2D's shape: p relaxed as a *field* (0.3), U and nuTilda as equations (0.7),
        # residualControl 1e-5 on every field, a history falling a decade every 80 iterations.
        log = tmp_path / "log.simpleFoam"
        log.write_text(_synthetic_log(("Ux", "Uy", "p", "nuTilda"), decades_per_step=1 / 80,
                                      tol=1e-5))
        cfg = _pitz_config()
        cfg.update(relaxation_factors={"p": 0.3, "U": 0.7, "nuTilda": 0.7},
                   relaxation_scopes={"p": "fields", "U": "equations", "nuTilda": "equations"},
                   residual_control={"p": 1e-5, "U": 1e-5, "nuTilda": 1e-5},
                   turbulence_model="SpalartAllmaras")
        v = _verify(log, cfg, tmp_path)
        assert v.verdict == "TRUSTWORTHY", [(r.code, r.summary) for r in v.reasons]

    def test_a_tighter_tolerance_that_was_not_met_is_not_converged(self, tmp_path):
        log = tmp_path / "log.simpleFoam"
        log.write_text(_synthetic_log(("Ux", "Uy", "p"), decades_per_step=1 / 80, tol=1e-4,
                                      converged_line=False))
        cfg = _pitz_config(residual_control={"p": 1e-6, "U": 1e-6})
        v = _verify(log, cfg, tmp_path)
        assert v.verdict == "NOT_TRUSTWORTHY"
        nc = next(r for r in v.reasons if r.code == "residual.not_converged")
        assert "residualControl" in nc.summary


def _synthetic_log(fields, *, decades_per_step: float, tol: float,
                   converged_line: bool = True) -> str:
    """A simpleFoam-shaped log whose initial residuals fall geometrically from 1 until every
    field is under `tol`, then `SIMPLE solution converged` and `End`."""
    lines = ["Exec   : simpleFoam"]
    step = 0
    while True:
        step += 1
        r = 10 ** (-decades_per_step * step)
        lines.append(f"Time = {step}")
        for i, f in enumerate(fields):
            ri = r * (1 - 0.1 * i)
            lines.append(f"smoothSolver:  Solving for {f}, Initial residual = {ri:.6g}, "
                         f"Final residual = {ri * 0.01:.6g}, No Iterations 3")
        if r < tol:
            break
    if converged_line:
        lines.append(f"SIMPLE solution converged in {step} iterations")
    lines.append("End")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ diagnosis and the boundary
def test_the_stall_is_evidence_for_the_residual_plateau_code():
    from courant.diagnosis.kb_data import build_default_kb

    entry = build_default_kb().entries["CONV.RESIDUAL_PLATEAU"]
    assert "residual.stalled" in entry.signature.finding_ids


def test_the_stall_finding_suggests_the_plateau_and_names_its_numbers():
    events = _events(PLATEAU_021)
    cfg = CaseConfig.model_validate(_pitz_config(relaxation=0.05))
    found = [f for f in run_all_checkers(CheckContext(run_id="r", events=events, config=cfg))
             if f.id == "residual.stalled"]
    assert len(found) == 1
    f = found[0]
    assert f.suggests == ["CONV.RESIDUAL_PLATEAU"]
    assert f.severity.value == "warn"
    assert math.isclose(f.threshold, 1e-3)


def test_residual_control_crosses_the_trust_boundary_like_relaxation_factors():
    from courant.diagnosis.redact import CONFIG_FIELDS_VERBATIM, sanitize_config_view

    assert "residual_control" in CONFIG_FIELDS_VERBATIM
    view = CaseConfig.model_validate(_pitz_config()).model_dump(mode="json")
    assert sanitize_config_view(view)["residual_control"] == {"SIMPLE": PITZ_RC}


@pytest.mark.parametrize("sid, top", [
    ("simpleFoam_pitzDaily__unbounded_scheme", "NUM.UNBOUNDED_SCHEME"),
    ("simpleFoam_pitzDaily__residual_plateau", "CONV.RESIDUAL_PLATEAU"),
])
def test_a_run_already_above_its_bar_gets_no_stall_finding(sid, top):
    """The stall is the evidence the bar cannot see. Reported on a run the bar already rejects,
    it re-ranked the certification corpus's unbounded_scheme diagnosis above the code it is
    certified on."""
    from courant.verify.ingest import from_log
    from courant.verify.pipeline import verify_inputs_v1

    v = verify_inputs_v1(from_log(CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json")
                         .model_copy(update={"checks": CHECKS}))
    assert v.verdict == "NOT_TRUSTWORTHY"
    assert "residual.not_converged" in _codes(v)
    assert "residual.stalled" not in _codes(v)
    assert v.probable_causes[0].code == top
