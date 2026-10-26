"""A field's residual at a step is its FIRST solve's, as residualControl reads it (TRD VER-16a).

The hard-proofing audit (21 Oct 2026) ran v2406's `pitzDailyExptInlet` with residualControl
removed and `nNonOrthogonalCorrectors 1`, so `p` is solved twice per iteration. At the last step
the first solve's initial residual was 1.73e-3, above the flat 1e-3 bar, and the second 2.21e-4:
the second corrector starts from an almost solved pressure, so its initial residual is small by
construction (about `relTol` times the first). The gate read the last solve and accepted the run
at grade A; with the second `p` line of each step deleted it rejected it (VER-5.4). OpenFOAM's
own criterion (`simpleControl::criteriaSatisfied`, `residuals.first()`) reads the first solve.

**Behind `--checks next` until 0.6.0** (GATE-13: it turns an accept into a reject). Under the
default every field undeclared by residualControl keeps the last solve, v0.5.0's judgement.

The log here is synthetic, in the shape of that run's: a steady simpleFoam history of `Ux`,
`Uy`, `k`, `epsilon` and two `p` solves per iteration, falling geometrically to the last step's
numbers above.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.converge.series import assess_case, from_events
from courant.events import EventKind
from courant.verification.convergence import (
    RESIDUAL_TARGET,
    judge_residuals,
    reads_first_solve,
)
from courant.watcher.parser import LogParser

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"

#: The last step of the audit's run: (field, first solve, second solve or None).
LAST = (("Ux", 6.0e-4, None), ("Uy", 5.1e-4, None), ("p", 1.72629e-3, 2.20514e-4),
        ("epsilon", 3.2e-4, None), ("k", 4.4e-4, None))


#: The audit's second pair (B1, B2: the same budget with `nNonOrthogonalCorrectors` 0 and 2): B2's
#: last iteration solved `p` three times, its initial residuals 1.25e-3, 1.65e-4 and 2.88e-5.
LAST_B2 = (("Ux", 6.0e-4, None), ("Uy", 5.1e-4, None), ("p", 1.24822e-3, 1.65374e-4, 2.87712e-5),
           ("epsilon", 3.2e-4, None), ("k", 4.4e-4, None))


def nonorth_log(steps: int = 60, last=LAST) -> str:
    """A steady simpleFoam log with non-orthogonal correctors: `p` solved once per value after
    the field name in `last` (twice a step for `LAST`, one corrector)."""
    lines = ["Exec   : simpleFoam", "", "Starting time loop", ""]
    for i in range(1, steps + 1):
        f = 10 ** (2.0 * (steps - i) / steps)          # 100x at the start, 1x at the end
        lines.append(f"Time = {i}")
        lines.append("")
        for name, first, *more in last:
            solver = "GAMG" if name == "p" else "smoothSolver"
            lines.append(f"{solver}:  Solving for {name}, Initial residual = {first * f:.6g}, "
                         f"Final residual = {first * f * 0.01:.6g}, No Iterations 3")
            for again in (r for r in more if r is not None):
                lines.append(f"{solver}:  Solving for {name}, Initial residual = "
                             f"{again * f:.6g}, Final residual = {again * f * 0.01:.6g}, "
                             "No Iterations 2")
        lines.append(f"ExecutionTime = {0.1 * i:.1f} s  ClockTime = {i} s")
        lines.append("")
    return "\n".join(lines + ["End", ""])


def _events(text: str):
    return list(LogParser(run_id="nonorth").parse_text(text))


def _steady() -> CaseConfig:
    return CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                      n_non_orthogonal_correctors=1)


def _verify(text: str, tmp_path: Path, checks: str):
    """`courant verify log.simpleFoam --case-json <pitzDaily's sidecar, residualControl removed>`."""
    from courant.verify.ingest import from_log
    from courant.verify.pipeline import verify_inputs_v1

    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"].pop("residual_control", None)
    meta["config"]["n_non_orthogonal_correctors"] = 1
    log = tmp_path / "log.simpleFoam"
    log.write_text(text)
    side = tmp_path / "log.case.json"
    side.write_text(json.dumps(meta))
    return verify_inputs_v1(from_log(log, side).model_copy(update={"checks": checks}))


# ------------------------------------------------------------------ the judgement
def test_the_parser_keeps_both_solves_of_the_last_step():
    step = [e.payload for e in _events(nonorth_log()) if e.kind is EventKind.RESIDUAL][-1]
    assert step.first_initial_residuals["p"] == pytest.approx(1.72629e-3)
    assert step.initial_residuals["p"] == pytest.approx(2.20514e-4)


def test_next_judges_every_field_on_its_first_solve():
    judged = judge_residuals(_events(nonorth_log()), _steady(), "next")
    assert judged.final["p"] == pytest.approx(1.72629e-3)
    assert judged.over == ("p",) and judged.field == "p"
    assert judged.bars["p"] == RESIDUAL_TARGET and not judged.declared
    assert judged.worst == pytest.approx(1.72629e-3)


def test_the_default_keeps_the_last_solve_as_v050_did():
    judged = judge_residuals(_events(nonorth_log()), _steady(), "default")
    assert judged.final["p"] == pytest.approx(2.20514e-4)
    assert judged.over == ()


def test_a_field_solved_once_reads_the_same_either_way():
    events = _events(nonorth_log())
    a = judge_residuals(events, _steady(), "default").final
    b = judge_residuals(events, _steady(), "next").final
    assert {k: v for k, v in a.items() if k != "p"} == {k: v for k, v in b.items() if k != "p"}


def test_the_series_the_stall_detector_and_the_block_read_is_the_first_solve_under_next():
    events = _events(nonorth_log())
    first, _, n = from_events(events, first_solve=True)
    last, _, _ = from_events(events)
    assert n == 60
    assert first["p"][-1] == pytest.approx(1.72629e-3)
    assert last["p"][-1] == pytest.approx(2.20514e-4)
    assert first["Ux"] == last["Ux"]


def test_assess_case_takes_the_mode():
    text = nonorth_log()
    events = _events(text)
    a = assess_case(None, log_text=text, events=events, config=_steady())
    b = assess_case(None, log_text=text, events=events, config=_steady(), checks="next")
    assert a.data["residuals"]["p"]["r_final"] == pytest.approx(2.20514e-4)
    assert b.data["residuals"]["p"]["r_final"] == pytest.approx(1.72629e-3)


def test_a_transient_run_keeps_the_last_solve_in_both_modes():
    """PISO and PIMPLE correctors start each step from the last step's solution, so a first
    solve measures how far the step moved the field: residualControl's convention is a steady
    one. Read on the committed corpus's healthy damBreak it rejected that run."""
    transient = CaseConfig(solver="pisoFoam", ddt_scheme="Euler")
    events = _events(nonorth_log())
    for mode in ("default", "next"):
        assert judge_residuals(events, transient, mode).final["p"] == pytest.approx(2.20514e-4)
        assert judge_residuals(events, None, mode).final["p"] == pytest.approx(2.20514e-4)
        assert not reads_first_solve(transient, mode) and not reads_first_solve(None, mode)
    assert reads_first_solve(_steady(), "next") and not reads_first_solve(_steady())


def test_the_healthy_dambreak_is_accepted_under_next():
    from courant.verify import verify_path

    sid = "interFoam_damBreak__healthy"
    v = verify_path(CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json", checks="next")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")


# ------------------------------------------------------------------ the verdicts
def test_the_audits_run_is_rejected_under_next(tmp_path):
    v = _verify(nonorth_log(), tmp_path, "next")
    assert (v.verdict, v.rule) == ("NOT_TRUSTWORTHY", "VER-5.4")
    by_code = {r.code: r for r in v.reasons}
    assert by_code["residual.not_converged"].summary == (
        "Final residual above convergence criterion: final worst residual 0.00173 > 0.001")
    assert by_code["convergence.residual"].summary == (
        "final worst residual 1.73e-03 (target 1e-03)")


def test_the_default_accepts_it_as_v050_did(tmp_path):
    v = _verify(nonorth_log(), tmp_path, "default")
    assert (v.verdict, v.rule, v.grade) == ("TRUSTWORTHY", "VER-5.6", "A")


def test_with_the_second_solve_deleted_both_modes_reject(tmp_path):
    """The audit's control: the same log with the second `p` line of each step removed."""
    lines, out, seen = nonorth_log().splitlines(), [], 0
    for line in lines:
        if line.startswith("Time = "):
            seen = 0
        if "Solving for p," in line:
            seen += 1
            if seen > 1:
                continue
        out.append(line)
    text = "\n".join(out) + "\n"
    for mode in ("default", "next"):
        v = _verify(text, tmp_path, mode)
        assert (v.verdict, v.rule) == ("NOT_TRUSTWORTHY", "VER-5.4"), mode


def test_two_correctors_hide_it_by_default_and_not_under_next(tmp_path):
    """B2's shape: three `p` solves an iteration, the first above the bar and the last 43 times
    below it. Its corrector-free twin B1 was rejected in both modes; B2 is under `next` too."""
    text = nonorth_log(last=LAST_B2)
    step = [e.payload for e in _events(text) if e.kind is EventKind.RESIDUAL][-1]
    assert step.first_initial_residuals["p"] == pytest.approx(1.24822e-3)
    assert step.initial_residuals["p"] == pytest.approx(2.87712e-5)
    assert (_verify(text, tmp_path, "default").verdict) == "TRUSTWORTHY"       # v0.5.0
    v = _verify(text, tmp_path, "next")
    assert (v.verdict, v.rule) == ("NOT_TRUSTWORTHY", "VER-5.4")
    assert "0.00125 > 0.001" in {r.code: r for r in v.reasons}["residual.not_converged"].summary
