"""TRD VER-16c: a `residualControl` tolerance may loosen the residual bar and never tightens it.

S's decision on H-VER16-ENDTIME (26 Oct 2026). Under `--checks next` VER-16 held each field to its
declared tolerance, tighter ones too, so a steady run whose `residualControl` is set far below what
it reaches was rejected by its own stopping rule: an internal evaluation's rebuilt natural-
convection cavity (1e-8 and 1e-9; Ux ended at 5.2e-8, its Nusselt number within 0.24% of the
benchmark) was a false alarm. v2406's `simpleControl::loop` ends a run early only when
`criteriaSatisfied()` holds and otherwise runs to endTime (`src/finiteVolume/cfdTools/general/
solutionControl/simpleControl/simpleControl.C`, lines 136 to 161, read in the pinned image), so a
tolerance below what the run can reach decides only how long it runs.

Now (`residual.control.floor`, behind `--checks next` for 0.7.0) a field's bar is its tolerance
clamped to [RESIDUAL_TARGET, RESIDUAL_LOOSE]. A field whose tolerance is below the target is held
to the target, met as the flat bar is met (`<=`, not "declared"), at both solves: the one the
tolerance is read at, the field's first of the step, and the one the flat bar reads. On a steady
run under `next` they are the same solve (VER-16a), so the field is judged exactly as on the flat
bar. On a local-time-stepping run the flat bar reads the last solve, so the field is judged at
the first and the last, and is never judged more leniently than with no tolerance at all (the
reviews of T-VER16C). The tolerance it did not reach is a note under "not assessed", never a
reason. Under the default nothing changes: no `residualControl` is read for the bar
(`tests/unit/test_checks_next.py`).

The verdicts are on cases built here or from scratch: synthetic logs in the shapes of steady
v2406 runs, pitzDaily's certification sidecar, the RM-35 fixtures that must stay rejected, and a
differentially heated cavity solved in the pinned image (`tests/fixtures/real_openfoam/
heated_cavity`, `make_case.py`).
"""

from __future__ import annotations

import gzip
import json
import shutil
from itertools import pairwise
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.verification.convergence import (
    RESIDUAL_LOOSE,
    RESIDUAL_TARGET,
    UnmetTolerance,
    judge_residuals,
    residual_bar,
    sets_bar,
)
from courant.verify import sentences
from courant.watcher.parser import LogParser

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
RC = ROOT / "tests" / "fixtures" / "residual_control"
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "heated_cavity"
PITZ_RC = {"p": 0.01, "U": 0.001, '"(k|epsilon|omega|f|v2)"': 0.001}


def _steady(control: dict, solver: str = "simpleFoam") -> CaseConfig:
    return CaseConfig(residual_control={"SIMPLE": control} if control else {}, solver=solver,
                      ddt_scheme="steadyState")


def _log(last: dict[str, float], steps: int = 300, *, solver: str = "simpleFoam",
         converged: bool = False) -> str:
    """A steady log whose initial residuals fall geometrically from 1 to `last` at its last step,
    one solve per field a step, then `End`: no "SIMPLE solution converged" unless `converged`."""
    lines = [f"Exec   : {solver}", "", "Starting time loop", ""]
    for i in range(1, steps + 1):
        lines += [f"Time = {i}", ""]
        for name, end in last.items():
            r = end ** (i / steps)
            lines.append(f"smoothSolver:  Solving for {name}, Initial residual = {r:.6g}, "
                         f"Final residual = {r * 0.01:.6g}, No Iterations 3")
        lines += [f"ExecutionTime = {0.01 * i:.2f} s  ClockTime = {i} s", ""]
    if converged:
        lines.append(f"SIMPLE solution converged in {steps} iterations")
    return "\n".join(lines + ["End", ""])


def _events(text: str):
    return list(LogParser(run_id="floor").parse_text(text))


def _gz_events(path: Path):
    with gzip.open(path, "rt") as fh:
        return list(LogParser(run_id=path.name).parse_text(fh.read()))


#: The natural-convection cavity's shape: residualControl 1e-8 (p_rgh) and 1e-9 (U, T), the
#: last step's residuals all far under the flat bar and above those tolerances.
CAVITY_LAST = {"Ux": 5.2e-8, "Uy": 4.1e-8, "T": 2.0e-9, "p_rgh": 3.1e-7}
CAVITY_RC = {"p_rgh": 1e-8, "U": 1e-9, "T": 1e-9}


# ------------------------------------------------------------------ the rule
def test_the_behaviour_is_named_for_0_7_0_behind_the_flag():
    from courant.verify.checks import NEXT, enabled

    assert NEXT["residual.control.floor"] == "0.7.0"
    assert not enabled("residual.control.floor") and enabled("residual.control.floor", "next")


@pytest.mark.parametrize(("tolerance", "sets"), [
    (1e-9, False), (9.99e-4, False), (0.0, False), (-1e-3, False),
    (1e-3, True), (5e-3, True), (1e-2, True), (1.0, True),
])
def test_only_a_tolerance_at_or_above_the_target_sets_the_bar(tolerance, sets):
    assert sets_bar(tolerance, "next") is sets
    assert sets_bar(tolerance, "default") is True        # VER-16 as 0.6.0 ships it


@pytest.mark.parametrize(("tolerance", "bar"), [
    (1e-9, RESIDUAL_TARGET), (1e-5, RESIDUAL_TARGET), (0.0, RESIDUAL_TARGET),
    (-1.0, RESIDUAL_TARGET), (1e-3, 1e-3), (5e-3, 5e-3), (1e-2, 1e-2), (1.0, RESIDUAL_LOOSE),
])
def test_the_bar_is_the_tolerance_clamped_to_the_target_and_the_loose_band(tolerance, bar):
    assert residual_bar(_steady({"p": tolerance}), "p", "next") == bar


def test_without_the_flag_a_tighter_tolerance_still_tightens_the_bar():
    """`residual_bar`'s default is VER-16 as 0.6.0 ships it (the bar per field itself is behind
    `--checks next`, so no default verdict reads it)."""
    assert residual_bar(_steady({"p": 1e-5}), "p") == 1e-5


# ------------------------------------------------------------------ judged as on the flat bar
def test_a_tighter_tolerance_is_judged_exactly_as_the_flat_bar():
    """Every part of the judgement is the flat bar's, the run with no residualControl at all;
    only the note differs."""
    events = _events(_log(CAVITY_LAST, solver="buoyantBoussinesqSimpleFoam"))
    tight = judge_residuals(events, _steady(CAVITY_RC, "buoyantBoussinesqSimpleFoam"), "next")
    flat = judge_residuals(events, _steady({}, "buoyantBoussinesqSimpleFoam"), "next")
    assert tight.over == () and tight.converged
    for part in ("final", "bars", "over", "field", "residual", "bar", "worst", "declared",
                 "field_declared", "stall", "unsure", "field_unsure"):
        assert getattr(tight, part) == getattr(flat, part), part
    assert not tight.per_field and tight.bar_words() == "the target 1e-03"
    assert set(tight.bars.values()) == {RESIDUAL_TARGET}
    assert flat.unmet == ()
    assert tight.unmet == (UnmetTolerance("Ux", 5.2e-8, 1e-9, True),
                           UnmetTolerance("Uy", 4.1e-8, 1e-9, True),
                           UnmetTolerance("T", 2.0e-9, 1e-9, True),
                           UnmetTolerance("p_rgh", 3.1e-7, 1e-8, True))


def _floor_off(monkeypatch):
    """VER-16c put back under `--checks next`: VER-16 as 0.6.0 ships it, every tolerance sets the
    bar, tighter ones too."""
    from courant.verify import checks

    real = checks.enabled
    monkeypatch.setattr(checks, "enabled", lambda name, mode="default": (
        False if name == "residual.control.floor" else real(name, mode)))


def test_before_the_floor_the_same_run_was_rejected_by_its_own_tolerance(monkeypatch):
    """The false alarm, with VER-16c put back: every field held to its tolerance."""
    events = _events(_log(CAVITY_LAST, solver="buoyantBoussinesqSimpleFoam"))
    config = _steady(CAVITY_RC, "buoyantBoussinesqSimpleFoam")
    assert not judge_residuals(events, config, "next").over
    _floor_off(monkeypatch)
    judged = judge_residuals(events, config, "next")
    assert judged.over == ("T", "Ux", "Uy", "p_rgh") and judged.field_declared
    assert judged.field == "Ux" and judged.bar_words() == "its residualControl 1e-09"
    assert judged.unmet == ()


def test_at_the_flat_bar_itself_a_tighter_tolerance_is_met_as_the_flat_bar_is():
    """The flat bar keeps the gate's `<=`; a tolerance that sets the bar is met strictly, as
    OpenFOAM meets it (`residual < tolerance`, simpleControl.C lines 70 to 71)."""
    events = _events(_log({"p": 1e-3, "Ux": 1e-5}))
    assert judge_residuals(events, _steady({"p": 1e-8}), "next").over == ()
    assert judge_residuals(events, _steady({}), "next").over == ()
    assert judge_residuals(events, _steady({"p": 1e-3}), "next").over == ("p",)


def _lts(control: dict) -> CaseConfig:
    """An OpenFOAM.org foamRun case marching in local time steps (VER-19, judged on its final
    residuals under `next`), its run criterion in PIMPLE (`solvers.residual_control_loop`)."""
    return CaseConfig(solver="foamRun", ddt_scheme="localEuler",
                      residual_control={"PIMPLE": control})


def _steady_nonorth(control: dict) -> CaseConfig:
    return CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                      n_non_orthogonal_correctors=1, residual_control={"SIMPLE": control})


REGIMES = {"steady": _steady_nonorth, "lts": _lts}


def _two_solves_log(first: float, last: float, solver: str = "simpleFoam") -> str:
    """p solved twice a step (one non-orthogonal corrector), its first solve ending at `first`
    and its second at `last`; Ux at 1e-5 (`test_first_solve.nonorth_log`)."""
    from tests.unit.test_first_solve import nonorth_log

    text = nonorth_log(last=(("Ux", 1e-5, None), ("p", first, last)))
    return text.replace("Exec   : simpleFoam", f"Exec   : {solver}", 1)


def _two_solves(first: float, last: float):
    return _events(_two_solves_log(first, last))


def _grade(events, config) -> int:
    from courant.verification.checks import ConvergenceCheck, VerifyContext

    return ConvergenceCheck().run(VerifyContext(run_id="r", events=events, config=config,
                                                checks="next")).severity.rank


def _verify_lts(first: float, last: float, control: dict, checks: str = "next"):
    """`courant verify` on a foamRun local-time-stepping run's log and its configuration."""
    from courant.verify import RunInputs, verify_inputs
    from courant.verify.schema import InputFile

    text = _two_solves_log(first, last, solver="foamRun")
    return verify_inputs(RunInputs(
        run_id="lts", log_text=text, log_name="log.foamRun",
        config_view=_lts(control).model_dump(mode="json"),
        inputs=(InputFile.of_text(text, "log.foamRun"),), source="corpus",
        solver_hint="foamRun", checks=checks))


def test_a_tighter_tolerance_is_read_at_both_solves():
    """The bar is the tolerance clamped. A tolerance below the target is read at both solves:
    where the tolerance is read, and where the flat bar reads. On a steady run under `next`,
    where VER-16a holds, the two are the same solve, the first. On a local-time-stepping run
    (VER-19) the flat bar reads the last solve of a step (VER-16a is steady only), and
    residualControl reads the first (`residuals.first() = cmptMax(sp.first().initialResidual())`,
    solutionControl.C line 232, compared in simpleControl.C lines 70 to 71). A field whose
    tolerance is below the target is held to the target at both, met as the flat bar is met
    (`<=`), and said in the flat bar's words. Here the first solve, the nearer of the two to
    the bar, decides."""
    events = _two_solves(5e-4, 2e-4)
    tight = judge_residuals(events, _lts({"p": 1e-8}), "next")
    assert tight.final["p"] == pytest.approx(5e-4) and tight.bars["p"] == RESIDUAL_TARGET
    assert tight.over == () and not tight.declared and not tight.per_field
    assert tight.field == "p" and tight.bar_words() == "the target 1e-03"
    assert tight.worst == pytest.approx(5e-4)
    assert tight.unmet == (UnmetTolerance("p", pytest.approx(5e-4), 1e-8, True),)
    loose = judge_residuals(events, _lts({"p": 1e-2}), "next")
    assert loose.final["p"] == pytest.approx(5e-4) and loose.field_declared
    none = judge_residuals(events, _lts({}), "next")       # no tolerance: the flat bar's solve
    assert none.final["p"] == pytest.approx(2e-4) and none.unmet == ()


@pytest.mark.parametrize("tolerance", [1e-8, 9.9e-4, 0.0])
@pytest.mark.parametrize(("first", "last", "rule", "grade"), [
    (5e-4, 2e-2, "VER-5.3", "C"), (2e-4, 2e-3, "VER-5.4", "B")])
def test_on_lts_a_tolerance_below_the_target_is_never_more_lenient_than_none(
        tolerance, first, last, rule, grade, monkeypatch):
    """The final review of T-VER16C. On a local-time-stepping foamRun run p's first solve at the
    last step is 5e-4 and its last corrector 2e-2. The flat bar reads the last corrector on such
    a run, so with no residualControl p is over 1e-3, past RESIDUAL_LOOSE, MAJOR: rejected
    (VER-5.3, C). Read only at its first solve under `p 1e-8`, p was within 1e-3 and the run was
    accepted (VER-5.6, A). A tolerance written below what the run reaches decided the verdict,
    which S's decision says it should not do. Read at both solves, the run is rejected as with no
    tolerance, in the same words. 0.6.0 reads the tolerance alone, at the first solve, strictly:
    it rejects the run under 1e-8 and 0 (VER-5.4, p's first solve against its own tolerance)
    and accepts it under 9.9e-4, which p's first solve meets. The run whose solves are (2e-4,
    2e-3) is the same shape inside the loose band: over, MINOR."""
    events = _two_solves(first, last)
    tight = judge_residuals(events, _lts({"p": tolerance}), "next")
    none = judge_residuals(events, _lts({}), "next")
    assert tight.over == none.over == ("p",)
    assert tight.final["p"] == pytest.approx(last) and tight.bars["p"] == RESIDUAL_TARGET
    assert tight.worst == none.worst == pytest.approx(last)
    assert not tight.declared and tight.bar_words() == none.bar_words() == "the target 1e-03"
    assert _grade(events, _lts({"p": tolerance})) >= _grade(events, _lts({}))
    reached = first < tolerance                     # as OpenFOAM tests it, at the first solve
    assert tight.unmet == (() if reached else (
        UnmetTolerance("p", pytest.approx(first), tolerance, True),))
    v, without = _verify_lts(first, last, {"p": tolerance}), _verify_lts(first, last, {})
    assert (v.verdict, v.rule, v.grade) == (without.verdict, without.rule, without.grade) == (
        "reject", rule, grade), [(r.code, r.summary) for r in v.reasons]
    assert [(r.code, r.summary) for r in v.reasons] == \
        [(r.code, r.summary) for r in without.reasons]
    _floor_off(monkeypatch)                                 # VER-16 as 0.6.0 ships it
    before = _verify_lts(first, last, {"p": tolerance})
    assert (before.verdict, before.rule) == (
        ("accept", "VER-5.6") if reached else ("reject", "VER-5.4"))


@pytest.mark.parametrize("tolerance", [1e-3, 9.9e-4, 1e-8, 0.0])
def test_on_lts_a_stricter_tolerance_never_accepts_what_1e_3_rejects(tolerance, monkeypatch):
    """The review of T-VER16C: p's first solve at the last step is 2e-2 (the pseudo-time step
    still moves p by 2e-2) and its last corrector 5e-4. Held to 1e-3 at its first solve it is
    over, and past RESIDUAL_LOOSE: MAJOR, rejected (VER-5.3, C). A stricter tolerance is clamped
    to the same 1e-3 at the same solve, so it is rejected the same way, as 0.6.0 rejected it.
    Judged on the flat bar's solve, the last, a stricter tolerance was accepted (VER-5.6, A)."""
    events = _two_solves(2e-2, 5e-4)
    judged = judge_residuals(events, _lts({"p": tolerance}), "next")
    assert judged.over == ("p",) and judged.final["p"] == pytest.approx(2e-2)
    assert judged.bars["p"] == RESIDUAL_TARGET and judged.worst == pytest.approx(2e-2)
    assert judged.field_declared is sets_bar(tolerance, "next")
    v = _verify_lts(2e-2, 5e-4, {"p": tolerance})
    assert (v.verdict, v.rule, v.grade) == ("reject", "VER-5.3", "C"), \
        [(r.code, r.summary) for r in v.reasons]
    _floor_off(monkeypatch)                                 # VER-16 as 0.6.0 ships it
    before = _verify_lts(2e-2, 5e-4, {"p": tolerance})
    assert (before.verdict, before.rule, before.grade) == ("reject", "VER-5.3", "C")


#: Loosest first. A residual of exactly 1e-3 is the one place a stricter tolerance is met more
#: leniently: the flat bar's `<=` meets it, a tolerance of 1e-3 (met strictly, as OpenFOAM meets
#: it) does not (`test_at_the_flat_bar_itself_a_tighter_tolerance_is_met_as_the_flat_bar_is`).
#: No pair below sits on it.
TOLERANCES = (1.0, 1e-2, 5e-3, 1e-3, 9.9e-4, 1e-4, 1e-8, 0.0, -1.0)
PAIRS = [(2e-2, 5e-4), (5e-4, 2e-2), (2e-3, 2e-4), (2e-4, 2e-3), (5e-3, 5e-3), (8e-4, 3e-4),
         (1.5e-3, 9e-4), (3e-2, 3e-2), (4e-3, 1e-4)]


@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("keyword", ["named", "before a match", "alone"])
@pytest.mark.parametrize(("first", "last"), PAIRS)
def test_a_stricter_tolerance_never_judges_more_leniently(regime, keyword, first, last):
    """Tightening p's tolerance, loosest to strictest, never takes p out of `over`, never lowers
    the worst residual and never lowers ConvergenceCheck's grade, on a steady run and a
    local-time-stepping one: named, after a keyword the matcher does not evaluate, and given by
    such a keyword alone. The floor only loosens a bar a tolerance would have tightened.

    The no-tolerance column (the final review of T-VER16C): a tolerance below the target is
    never more lenient than no residualControl at all, the flat bar at the solve it reads. Read
    only at its first solve on a local-time-stepping run, `p 1e-8` accepted p at 5e-4 then 2e-2,
    which the flat bar rejects."""
    events = _two_solves(first, last)
    control = {"named": lambda t: {"p": t},
               "before a match": lambda t: {'"(?=p)p"': t, "p": 1e-2},
               "alone": lambda t: {'"(?=p)p"': t}}[keyword]

    def judged_on(config) -> tuple:
        judged = judge_residuals(events, config, "next")
        return "p" in judged.over, judged.worst, _grade(events, config)

    seen = [(tolerance, *judged_on(REGIMES[regime](control(tolerance))))
            for tolerance in TOLERANCES]
    for (_, *looser), (tol, *stricter) in pairwise(seen):
        assert all(s >= o for s, o in zip(stricter, looser)), (tol, seen)
    none = judged_on(REGIMES[regime]({}))
    below = [(tol, *at) for tol, *at in seen if tol < RESIDUAL_TARGET]
    assert len(below) == 5
    for tol, *stricter in below:
        assert all(s >= o for s, o in zip(stricter, none)), (tol, none, seen)


@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("control", [
    {"p": 1.0}, {"p": 1e-2}, {"p": 5e-3}, {"p": 1e-3}, {"p": 9.9e-4}, {"p": 1e-8}, {"p": 0.0},
    {'"(?=p)p"': 5e-3, "p": 1e-8}, {'"(?=p)p"': 1e-8, "p": 5e-3}, {'"(?=p)p"': 2.0, "p": 1e-3},
    {'"(?=p)p"': 1e-3, "p": 1e-8},
])
@pytest.mark.parametrize(("first", "last"), [(4e-3, 1e-4), (1e-4, 4e-3), (5e-4, 5e-4),
                                             (1e-3, 1e-3), (0.5, 0.5)])
def test_residual_bar_is_the_bar_the_judgement_holds_a_certain_tolerance_to(regime, control,
                                                                            first, last):
    """One clamp for both (`convergence._clamp`): for a field whose tolerance certainly applies,
    the bar `judge_residuals` decides is `residual_bar`'s, on a steady run and a local-time-
    stepping one, whatever its residuals. The review of T-VER16C found them apart: 5e-3 decided
    where `residual_bar` said 1e-3."""
    config = REGIMES[regime](control)
    judged = judge_residuals(_two_solves(first, last), config, "next")
    assert judged.bars["p"] == residual_bar(config, "p", "next")


def test_a_tolerance_below_the_target_after_a_keyword_not_evaluated():
    """The review's case: `"(?=p)p" 5e-3; p 1e-8;` on a local-time-stepping run, p's first solve
    at 4e-3 and its last at 1e-4. If the lookahead does not match, OpenFOAM holds p to 1e-8,
    which the floor puts at the target, and p's first solve is four times over it; if it does,
    to 5e-3, which p meets. Over, against the target, in the flat bar's words. The 1e-8 the note
    names is the plain `p` entry's, which certainly matches p: certain, though the keyword may
    pre-empt it (the final review of T-VER16C)."""
    config = _lts({'"(?=p)p"': 5e-3, "p": 1e-8})
    judged = judge_residuals(_two_solves(4e-3, 1e-4), config, "next")
    assert judged.over == ("p",) and judged.bars["p"] == RESIDUAL_TARGET
    assert residual_bar(config, "p", "next") == RESIDUAL_TARGET
    assert not judged.field_declared and judged.bar_words() == "the target 1e-03"
    assert judged.unmet == (UnmetTolerance("p", pytest.approx(4e-3), 1e-8, True),)


_MAY = ", which a residualControl keyword the reader does not evaluate may set"


@pytest.mark.parametrize(("control", "certain", "values"), [
    # the tightest is the matching entry's; the keyword's 1e-2 may name p, and sets its bar
    ({'"(?=U)U"': 1e-2, "U": 1e-9}, True,
     "(Ux 5e-07 against 1e-09; Uy 1e-07 against 1e-09)"),
    # the tightest is the keyword's, which may also name p
    ({'"(?=U)U"': 1e-9, "U": 1e-2}, False,
     f"(p 0.0002 against 1e-09{_MAY}; Ux 5e-07 against 1e-09{_MAY}; "
     f"Uy 1e-07 against 1e-09{_MAY})"),
    # the same value: the matching entry's as well
    ({'"(?=U)U"': 1e-9, "U": 1e-9}, True,
     f"(p 0.0002 against 1e-09{_MAY}; Ux 5e-07 against 1e-09; Uy 1e-07 against 1e-09)"),
    # no entry certainly matches
    ({'"(?=U)U"': 1e-9}, False,
     f"(p 0.0002 against 1e-09{_MAY}; Ux 5e-07 against 1e-09{_MAY}; "
     f"Uy 1e-07 against 1e-09{_MAY})"),
])
def test_the_note_says_a_keyword_may_set_the_tolerance_only_when_one_may(control, certain,
                                                                         values):
    """The final review of T-VER16C: under `"(?=U)U" 1e-2; U 1e-9;` the note said U's 1e-9 was a
    value "which a residualControl keyword the reader does not evaluate may set". The 1e-9 is
    the plain `U` entry's, which certainly matches U; the keyword can only pre-empt it, with
    1e-2. An unmet tolerance is certain when it is the value of the entry that certainly matches
    the field, whatever keyword comes before it. The matcher does not evaluate a lookahead at
    all, so the keyword may name p too."""
    events = _events(_log({"p": 2e-4, "Ux": 5e-7, "Uy": 1e-7}))
    judged = judge_residuals(events, _steady(control), "next")
    unmet = {u.field: u for u in judged.unmet}
    assert unmet["Ux"] == UnmetTolerance("Ux", 5e-7, 1e-9, certain)
    assert unmet["Uy"] == UnmetTolerance("Uy", 1e-7, 1e-9, certain)
    assert values in sentences.tolerances_unmet(judged.unmet)


@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("control", [{"p": 1.0}, {'"(?=p)p"': 2.0, "p": 1.0}])
def test_no_tolerance_loosens_the_judgement_past_the_loose_band(regime, control):
    """`residualControl { p 1; }` would make any run converge on its second iteration: the
    judgement itself holds p to RESIDUAL_LOOSE, so p at 0.5 is over, and MAJOR."""
    from courant.verification.grade import VSeverity

    config = REGIMES[regime](control)
    events = _two_solves(0.5, 0.5)
    judged = judge_residuals(events, config, "next")
    assert judged.over == ("p",) and judged.bars["p"] == RESIDUAL_LOOSE
    assert _grade(events, config) == VSeverity.MAJOR.rank


def test_a_tighter_tolerance_beside_a_loosening_one():
    """pitzDaily's own `p 1e-2` still loosens p's bar while U's 1e-6 is judged on the flat bar:
    the reason names the field that decides and the bar it is held to."""
    from courant.diagnosis.checkers import CheckContext, ResidualAnalyzer

    config = _steady({"p": 1e-2, "U": 1e-6})
    within = judge_residuals(_events(_log({"p": 5e-3, "Ux": 5e-4, "Uy": 2e-4})), config, "next")
    assert within.over == () and within.declared and within.bars == {
        "p": 1e-2, "Ux": RESIDUAL_TARGET, "Uy": RESIDUAL_TARGET}
    assert [u.field for u in within.unmet] == ["Ux", "Uy"]
    events = _events(_log({"p": 5e-3, "Ux": 2e-3, "Uy": 2e-4}))
    over = judge_residuals(events, config, "next")
    assert over.over == ("Ux",) and over.field == "Ux" and not over.field_declared
    finding = next(f for f in ResidualAnalyzer().run(CheckContext(
        run_id="r", events=events, config=config, checks="next"))
        if f.id == "residual.not_converged")
    assert finding.detail == "final residual 0.002 on Ux > the target 1e-03"
    assert finding.threshold == RESIDUAL_TARGET


def test_a_tighter_tolerance_that_was_met_is_no_note():
    events = _events(_log({"p": 5e-9, "Ux": 5e-10}, converged=True))
    judged = judge_residuals(events, _steady({"p": 1e-8, "U": 1e-9}), "next")
    assert judged.over == () and judged.unmet == ()


@pytest.mark.parametrize("tolerance", [0.0, -1e-3])
def test_a_tolerance_no_residual_can_meet_is_judged_on_the_flat_bar(tolerance):
    """Zero or a negative tolerance: OpenFOAM's `residual < tolerance` never holds, so the run
    always runs to its endTime. Below the target, it is judged on the flat bar and noted."""
    events = _events(_log({"p": 2e-4, "Ux": 1e-5}))
    judged = judge_residuals(events, _steady({"p": tolerance}), "next")
    assert judged.over == () and [u.field for u in judged.unmet] == ["p"]
    assert judge_residuals(_events(_log({"p": 2e-3, "Ux": 1e-5})),
                           _steady({"p": tolerance}), "next").over == ("p",)


# ------------------------------------------------------------------ uncertain tolerances
def test_a_tighter_tolerance_only_a_keyword_not_evaluated_may_give():
    """With no entry certain to match, the field may have that keyword's tolerance or none. Below
    the target both readings are the flat bar's: never declared, never unsure, and the note says
    a keyword may set it. The matcher does not evaluate a lookahead at all, so the keyword may
    name any field, p too."""
    events = _events(_log({"p": 2e-4, "Ux": 5e-7, "Uy": 1e-7}))
    judged = judge_residuals(events, _steady({'"(?=U)U"': 1e-9}), "next")
    flat = judge_residuals(events, _steady({}), "next")
    assert (judged.over, judged.final, judged.bars) == (flat.over, flat.final, flat.bars)
    assert not judged.declared and not judged.unsure and not judged.per_field
    assert judged.unmet == (UnmetTolerance("p", 2e-4, 1e-9, False),
                            UnmetTolerance("Ux", 5e-7, 1e-9, False),
                            UnmetTolerance("Uy", 1e-7, 1e-9, False))
    said = sentences.tolerances_unmet(judged.unmet)
    assert "Ux 5e-07 against 1e-09, which a residualControl keyword the reader does not " \
           "evaluate may set" in said


@pytest.mark.parametrize(("first", "last"), [(2e-2, 5e-4), (5e-4, 2e-3), (5e-4, 5e-4),
                                             (5e-3, 5e-3), (2e-3, 2e-4)])
def test_every_tolerance_a_field_may_have_is_judged(first, last):
    """A keyword not evaluated before the entry that matches may be the one OpenFOAM applies: p
    may have 1e-9 (clamped to the target, met as the flat bar is met) or the match's 1e-2 (met
    strictly), both read at the first solve, where residualControl reads them. The field is over
    when either reading is."""
    events = _two_solves(first, last)
    both = judge_residuals(events, _lts({'"(?=p)p"': 1e-9, "p": 1e-2}), "next")
    if_loose = judge_residuals(events, _lts({"p": 1e-2}), "next")
    if_tight = judge_residuals(events, _lts({"p": 1e-9}), "next")
    assert ("p" in both.over) is ("p" in if_loose.over or "p" in if_tight.over)
    assert both.worst == max(if_loose.worst, if_tight.worst)
    assert ("p", False) in [(u.field, u.certain) for u in both.unmet]


# ------------------------------------------------------------------ the stall, on its bars
def _twin_config(control: dict) -> CaseConfig:
    """pitzDaily's certification sidecar under the catalogue's equation relaxation of 0.05 (the
    residual_plateau twins'), with this residualControl."""
    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    cfg = meta["config"]
    cfg["relaxation_factors"] = {"U": 0.05, '".*"': 0.05}
    cfg["relaxation_scopes"] = {"U": "equations", '".*"': "equations"}
    cfg["residual_control"] = {"SIMPLE": control}
    return CaseConfig.model_validate(cfg)


@pytest.mark.parametrize("twin", ["021", "045"])
def test_the_stall_is_asked_on_the_bars_the_floor_gives(twin, monkeypatch):
    """The catalogue's residual_plateau twins with p's own 1e-2 and U, k and epsilon set below
    the target. Held to 1e-6, every one of those was over its bar, and the run was rejected for
    that. On the floor they are on the flat bar, as pitzDaily's own 1e-3 puts them, and the stall
    detector finds the plateau there: it compares the deflated residual with the bar the field
    then has, the target."""
    config = _twin_config({"p": 1e-2, "U": 1e-6, '"(k|epsilon)"': 1e-6})
    events = _gz_events(RC / f"c10k.simpleFoam_pitzDaily.{twin}.catalogue.log.gz")
    judged = judge_residuals(events, config, "next")
    assert judged.over == () and judged.stall is not None and not judged.converged
    assert judged.stall.bar == RESIDUAL_TARGET
    as_shipped = judge_residuals(events, _twin_config(PITZ_RC), "next")
    assert judged.stall == as_shipped.stall
    _floor_off(monkeypatch)
    before = judge_residuals(events, config, "next")
    assert before.over and before.stall is None


@pytest.mark.parametrize("twin", ["021", "045"])
def test_a_twin_whose_every_tolerance_is_below_the_target_is_over_the_flat_bar(twin):
    """With p's tolerance below the target too, p is judged on the flat bar, which always
    rejected these twins (TRD VER-16, the default's judgement of them)."""
    config = _twin_config({"p": 1e-6, "U": 1e-6, '"(k|epsilon)"': 1e-6})
    events = _gz_events(RC / f"c10k.simpleFoam_pitzDaily.{twin}.catalogue.log.gz")
    judged = judge_residuals(events, config, "next")
    assert "p" in judged.over and not judged.declared
    assert judged.over == judge_residuals(events, _twin_config({}), "next").over


# ------------------------------------------------------------------ the verdicts
def _verify_sidecar(text: str, control: dict, tmp_path: Path, checks: str):
    """`courant verify log.simpleFoam --case-json <pitzDaily's sidecar with this residualControl>`
    (the certification corpus's path)."""
    from courant.verify import verify_inputs
    from courant.verify.ingest import from_log

    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"]["residual_control"] = {"SIMPLE": control}
    log = tmp_path / "log.simpleFoam"
    log.write_text(text)
    side = tmp_path / "log.case.json"
    side.write_text(json.dumps(meta))
    return verify_inputs(from_log(log, side).model_copy(update={"checks": checks}))


def test_a_run_to_endtime_tolerance_is_accepted_with_a_note(tmp_path):
    text = _log({"Ux": 4e-6, "Uy": 3e-6, "p": 2e-5, "k": 5e-6, "epsilon": 6e-6})
    control = {"p": 1e-8, "U": 1e-9, '"(k|epsilon|omega|f|v2)"': 1e-9}
    v = _verify_sidecar(text, control, tmp_path, "next")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6"), [(r.code, r.summary) for r in v.reasons]
    notes = [s for s in v.not_assessed if "did not reach the residualControl" in s]
    assert notes == [
        "Ux, Uy, p, k and epsilon did not reach the residualControl tolerance set below the "
        "residual target 1e-03 (Ux 4e-06 against 1e-09; Uy 3e-06 against 1e-09; p 2e-05 against "
        "1e-08; k 5e-06 against 1e-09; epsilon 6e-06 against 1e-09): a tolerance below the "
        "target is not held against the run, so they were judged against the target."]
    assert sentences.cause_of(notes[0]) == "run"
    assert not any(r.severity in ("warn", "minor") and "residual" in r.code for r in v.reasons)


@pytest.mark.usefixtures("checks_as_060")
def test_the_default_says_nothing_of_it_and_judges_as_before(tmp_path):
    text = _log({"Ux": 4e-6, "Uy": 3e-6, "p": 2e-5, "k": 5e-6, "epsilon": 6e-6})
    control = {"p": 1e-8, "U": 1e-9, '"(k|epsilon|omega|f|v2)"': 1e-9}
    v = _verify_sidecar(text, control, tmp_path, "default")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")
    assert not [s for s in v.not_assessed if "residualControl" in s]


def test_a_tighter_tolerance_above_the_flat_bar_is_rejected_on_the_flat_bar(tmp_path):
    """The floor never loosens past the target: a field above 1e-3 is rejected, in the flat bar's
    words, and the tolerance it did not reach is noted as well."""
    text = _log({"Ux": 4e-6, "Uy": 3e-6, "p": 2e-3, "k": 5e-6, "epsilon": 6e-6})
    v = _verify_sidecar(text, {"p": 1e-8, "U": 1e-9}, tmp_path, "next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.4")
    by_code = {r.code: r for r in v.reasons}
    assert by_code["residual.not_converged"].summary == (
        "Final residual above convergence criterion: final worst residual 0.002 > 0.001")
    assert any(s.startswith("Ux, Uy and p did not reach") for s in v.not_assessed)


def test_pitzdailys_own_loose_tolerance_still_loosens(tmp_path):
    """The RM-35 interim's healthy draw 058: p ends at 1.02e-3, within pitzDaily's own 1e-2, and is
    accepted under `next`; with its residualControl removed the flat bar rejects it."""
    with gzip.open(RC / "c10k.simpleFoam_pitzDaily.058.healthy.log.gz", "rt") as fh:
        text = fh.read()
    v = _verify_sidecar(text, PITZ_RC, tmp_path, "next")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")
    assert not [s for s in v.not_assessed if "residualControl" in s]
    assert residual_bar(_steady(PITZ_RC), "p", "next") == 1e-2


@pytest.mark.parametrize("twin", ["021", "045"])
def test_the_stalled_twins_stay_rejected(twin, tmp_path):
    from courant.verify import verify_inputs
    from courant.verify.ingest import from_log

    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"]["relaxation_factors"] = {"U": 0.05, '".*"': 0.05}
    meta["config"]["relaxation_scopes"] = {"U": "equations", '".*"': "equations"}
    meta["config"]["residual_control"] = {"SIMPLE": PITZ_RC}
    side = tmp_path / f"c10k.simpleFoam_pitzDaily.{twin}.catalogue.case.json"
    side.write_text(json.dumps(meta))
    log = RC / f"c10k.simpleFoam_pitzDaily.{twin}.catalogue.log.gz"
    v = verify_inputs(from_log(log, side).model_copy(update={"checks": "next"}))
    assert (v.verdict, v.rule) == ("reject", "VER-5.4")
    assert "residual.stalled" in [r.code for r in v.reasons]


# ------------------------------------------------------------------ the heated cavity (v2406)
def _unpack(src: Path, dst: Path) -> None:
    for f in src.rglob("*"):
        if not f.is_file() or f.name in ("make_case.py", "run.sh"):
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".gz":
            with gzip.open(f, "rb") as s, open(out.with_suffix(""), "wb") as d:
                shutil.copyfileobj(s, d)
        else:
            shutil.copyfile(f, out)


@pytest.fixture
def cavity(tmp_path):
    _unpack(CAVITY / "run", tmp_path / "run")
    return tmp_path / "run"


#: The cavity's sentence under `next`: its last step's first solves against its tolerances.
CAVITY_NOTE = (
    "Ux, Uy, T and p_rgh did not reach the residualControl tolerance set below the residual "
    "target 1e-03 (Ux 1.37e-08 against 1e-10; Uy 1.5e-08 against 1e-10; T 1.54e-08 against "
    "1e-10; p_rgh 6.87e-08 against 1e-09): a tolerance below the target is not held against the "
    "run, so they were judged against the target.")


def test_the_cavity_runs_to_its_endtime_on_a_tolerance_it_never_meets(cavity):
    """v2406 printed its criteria and never "SIMPLE solution converged": the run stopped at its
    endTime, 200 iterations, every residual between its tolerance and the flat bar."""
    text = (cavity / "log.buoyantBoussinesqSimpleFoam").read_text()
    assert "SIMPLE: convergence criteria" in text and "field p_rgh\t tolerance 1e-09" in text
    assert "solution converged" not in text and text.rstrip().endswith("End")
    config = CaseConfig.from_foam(str(cavity))
    assert config.solver == "buoyantBoussinesqSimpleFoam" and config.is_steady("next")
    assert config.residual_control == {"SIMPLE": {"p_rgh": 1e-9, "U": 1e-10, "T": 1e-10}}
    judged = judge_residuals(_events(text), config, "next")
    assert judged.over == () and judged.converged and not judged.declared
    assert all(1e-10 < v < RESIDUAL_TARGET for v in judged.final.values())
    assert [u.field for u in judged.unmet] == ["Ux", "Uy", "T", "p_rgh"]


def test_the_cavity_is_accepted_under_next_with_the_note(cavity):
    from courant.verify import verify_path

    v = verify_path(cavity, checks="next")
    assert (v.verdict, v.rule, v.grade) == ("accept", "VER-5.6", "A"), \
        [(r.code, r.summary) for r in v.reasons]
    assert CAVITY_NOTE in v.not_assessed
    assert not {"residual.not_converged", "physics.laminar_unstable", "residual.stalled",
                "convergence.residual"} & {r.code for r in v.reasons}


def test_with_the_floor_put_back_the_cavity_is_the_false_alarm(cavity, monkeypatch):
    """VER-16 as 0.6.0 ships it: held to its own 1e-10, the laminar run "never reached
    tolerance" too, and is rejected (VER-5.4)."""
    from courant.verify import verify_path

    _floor_off(monkeypatch)
    v = verify_path(cavity, checks="next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.4")
    by_code = {r.code: r for r in v.reasons}
    assert by_code["residual.not_converged"].summary == (
        "Final residual above convergence criterion: final residual 1.54e-08 on T > its "
        "residualControl 1e-10")
    assert "physics.laminar_unstable" in by_code
    assert CAVITY_NOTE not in v.not_assessed


@pytest.mark.usefixtures("checks_as_060")
def test_the_default_accepts_the_cavity_and_its_bytes_do_not_depend_on_the_floor(cavity,
                                                                                 monkeypatch):
    from courant.verify import verify_path

    default = verify_path(cavity)
    assert (default.verdict, default.rule) == ("accept", "VER-5.6")
    assert not [s for s in default.not_assessed if "residualControl" in s]
    _floor_off(monkeypatch)
    assert verify_path(cavity).to_json() == default.to_json()
