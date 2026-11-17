"""The verdict-path audit's findings in the residual judgement and the pipeline's next action.

Fixed:

- VPA-CHK-1 (behind `residual.no_time_markers`, NEXT 0.8.0): a steady log with no `Time =` line
  (VER-5.15's input) was rejected VER-5.3 grade C when the case declares residualControl: the
  declared tolerance was read at the first solve of the parser's single step (iteration 1, a
  residual near 1). `reads_first_solve` guarded only the flat bar.
- VPA-ORDER-3 (a next-action-only change, GATE-13): the specification's advice was given
  whenever the specification had something to say, so a crashed run (VER-5.2) was told to read
  the specification and a truncated run (VER-5.16) lost "verify the finished run". It is given
  only when a specification rule decided.

Open, as strict xfails: VPA-PARSER-1, VPA-PARSER-2, VPA-PARSER-5, VPA-PARSER-6, VPA-CHK-4.
"""

from __future__ import annotations

import gzip
import inspect
import json
import math
import re
import shutil
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.verify import checks
from courant.watcher.parser import LogParser, _to_float

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "tests" / "fixtures" / "real_openfoam"
CORPUS = ROOT / "certification" / "corpus"
DUCT = REAL / "duct" / "converged"


def _events(text: str):
    return list(LogParser(run_id="vpa").parse_text(text))


def _strip_time_lines(text: str) -> str:
    return "".join(line for line in text.splitlines(keepends=True)
                   if not re.match(r"^Time = ", line))


def _duct_copy(tmp_path: Path, *, strip: bool) -> Path:
    case = tmp_path / "duct"
    shutil.copytree(DUCT, case)
    gz = case / "log.rhoSimpleFoam.gz"
    with gzip.open(gz, "rt") as fh:
        text = fh.read()
    gz.unlink()
    (case / "log.rhoSimpleFoam").write_text(_strip_time_lines(text) if strip else text)
    return case


# ---------------------------------------------------------------- VPA-CHK-1
def test_the_name_is_in_next_for_0_8_0():
    assert checks.NEXT[checks.RESIDUAL_NO_TIME_MARKERS] == "0.8.0"
    assert not checks.enabled(checks.RESIDUAL_NO_TIME_MARKERS, "default")
    assert checks.enabled(checks.RESIDUAL_NO_TIME_MARKERS, "next")


def test_the_stripped_duct_is_rejected_by_default_as_before(tmp_path):
    """The default is byte for byte unchanged: the declared tolerance read at iteration 1."""
    from courant.verify import verify_path

    v = verify_path(_duct_copy(tmp_path, strip=True))
    assert (v.verdict, v.rule, v.grade) == ("reject", "VER-5.3", "C")
    assert any("above its residualControl 1e-05" in r.summary for r in v.reasons), \
        [r.summary for r in v.reasons]


def test_the_stripped_duct_is_insufficient_under_next(tmp_path):
    """VER-5.15: no step carries a step number, so no step's first solve is read; the tolerance
    reads the last solve, as the flat bar does, and the log is insufficient, not rejected."""
    from courant.verify import verify_path

    v = verify_path(_duct_copy(tmp_path, strip=True), checks="next")
    assert (v.verdict, v.rule, v.grade) == ("insufficient", "VER-5.15", None), \
        [(r.code, r.summary) for r in v.reasons]


@pytest.mark.parametrize("mode", ["default", "next"])
def test_the_unstripped_duct_is_still_accepted(mode, tmp_path):
    from courant.verify import verify_path

    v = verify_path(_duct_copy(tmp_path, strip=False), checks=mode)
    assert v.verdict == "accept", [(r.code, r.summary) for r in v.reasons]


def _stripped_pitzdaily_with_control(tmp_path: Path) -> tuple[Path, Path]:
    sid = "simpleFoam_pitzDaily__healthy"
    with gzip.open(CORPUS / f"{sid}.log.gz", "rt") as fh:
        text = fh.read()
    log = tmp_path / f"{sid}.log"
    log.write_text(_strip_time_lines(text))
    meta = json.loads((CORPUS / f"{sid}.case.json").read_text())
    # the tutorial's own residualControl, nested under its loop dictionary as CaseConfig has it
    meta["config"]["residual_control"] = {
        "SIMPLE": {"p": 1e-2, "U": 1e-3, "(k|epsilon|omega|f|v2)": 1e-3}}
    side = tmp_path / f"{sid}.case.json"
    side.write_text(json.dumps(meta))
    return log, side


def test_the_stripped_pitzdaily_with_residual_control_is_rejected_by_default(tmp_path):
    from courant.verify import verify_path

    log, side = _stripped_pitzdaily_with_control(tmp_path)
    v = verify_path(log, side)
    assert (v.verdict, v.rule) == ("reject", "VER-5.3"), [(r.code, r.summary) for r in v.reasons]


def test_the_stripped_pitzdaily_with_residual_control_is_insufficient_under_next(tmp_path):
    from courant.verify import verify_path

    log, side = _stripped_pitzdaily_with_control(tmp_path)
    v = verify_path(log, side, checks="next")
    assert (v.verdict, v.rule, v.grade) == ("insufficient", "VER-5.15", None), \
        [(r.code, r.summary) for r in v.reasons]


# ---------------------------------------------------------------- VPA-ORDER-3
@pytest.fixture
def demo_case(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path / "demo")


def _unreadable_spec(tmp_path: Path) -> Path:
    """A specification whose one item the demo case cannot give (it has no `0/`)."""
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema": "courant.spec/1", "items": [
        {"name": "inlet velocity", "file": "0/U", "patch": "inlet", "value": [10, 0, 0]}]}))
    return path


def test_a_crashed_run_with_an_unreadable_item_is_told_to_read_the_crash(demo_case, tmp_path):
    from courant.verify import verify_path

    log = demo_case / "log.simpleFoam"
    log.write_text((REAL / "cavity_cfl_diverged.log").read_text())
    v = verify_path(demo_case, spec=_unreadable_spec(tmp_path))
    assert (v.verdict, v.rule) == ("reject", "VER-5.2")
    assert v.conformance is not None and v.conformance.unreadable == 1
    assert "could not be read from the case" not in v.next_action
    assert "did not complete" in v.next_action or "Suggested, not applied" in v.next_action, \
        v.next_action


def test_a_truncated_run_with_an_unreadable_item_is_told_to_verify_it_finished(demo_case,
                                                                                tmp_path):
    from courant.verify import verify_path

    log = demo_case / "log.simpleFoam"
    lines = log.read_text().splitlines(keepends=True)
    end = max(i for i, line in enumerate(lines) if line.strip() == "End")
    log.write_text("".join(lines[:end - 40]))
    run = demo_case / "run.json"
    if run.exists():                                  # an exit code would end the run
        data = json.loads(run.read_text())
        for key in [k for k in data if "exit" in k.lower()]:
            data.pop(key)
        run.write_text(json.dumps(data))
    v = verify_path(demo_case, spec=_unreadable_spec(tmp_path))
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.16"), (v.verdict, v.rule)
    assert "once it has finished" in v.next_action, v.next_action


def test_an_unreadable_item_on_a_clean_run_still_gets_the_spec_advice(demo_case, tmp_path):
    from courant.verify import verify_path

    v = verify_path(demo_case, spec=_unreadable_spec(tmp_path))
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.9")
    assert "could not be read from the case" in v.next_action


# ---------------------------------------------------------------- VPA-PARSER-1
@pytest.mark.parametrize("token", ["nan", "-nan", "-nan(ind)", "1.#QNAN", "-1.#INF"])
def test_these_spellings_are_non_finite(token):
    x = _to_float(token)
    assert x is not None and not math.isfinite(x)


def _ind_log(steps: int = 3) -> str:
    lines = (REAL / "cavity_converged.log").read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("Time = ")]
    first = starts[-steps]
    tail = [re.sub(r"(Initial residual = )[^,]+", r"\g<1>-1.#IND", line) for line in lines[first:]]
    return "".join(lines[:first] + tail)


@pytest.mark.xfail(strict=True, reason="VPA-PARSER-1: _to_float reads msvcrt's -1.#IND as None "
                   "and the residual line is dropped, so the run is accepted (VER-5.6)")
def test_msvcrt_nan_is_rejected(tmp_path):
    from courant.verify import verify_path

    log = tmp_path / "log.icoFoam"
    log.write_text(_ind_log())
    v = verify_path(log)
    assert (v.verdict, v.rule) == ("reject", "VER-5.2")


# ---------------------------------------------------------------- VPA-PARSER-2
@pytest.mark.xfail(strict=True, reason="VPA-PARSER-2: an unrecognised log holding only a fatal "
                   "signature is insufficient VER-5.1, which the TRD limits to no fatal signature")
def test_a_bare_stack_trace_is_rejected(tmp_path):
    from courant.verify import verify_path

    text = (REAL / "cavity_cfl_diverged.log").read_text()
    assert "Foam::sigFpe::sigHandler" in text
    log = tmp_path / "log.icoFoam"
    log.write_text(text[text.index("[stack trace]"):])
    v = verify_path(log)
    assert (v.verdict, v.rule) == ("reject", "VER-5.2")


# ---------------------------------------------------------------- VPA-PARSER-6
@pytest.mark.xfail(strict=True, reason="VPA-PARSER-6: policy._GROUP has no "
                   "('conformance', 'major'), so a VER-5.12 reason sorts after every check")
def test_a_major_conformance_reason_is_ordered_first():
    from courant.verify.policy import order_reasons
    from courant.verify.schema import Reason

    check = Reason(code="residual.converged", kind="check", severity="major", summary="s",
                   input="log", method="m")
    conf = Reason(code="conformance.nu", kind="conformance", severity="major", summary="s",
                  input="spec", method="m")
    assert order_reasons([check, conf])[0] is conf


# ---------------------------------------------------------------- VPA-PARSER-5
@pytest.mark.xfail(strict=True, reason="VPA-PARSER-5: policy.decide re-implements is_suspect "
                   "inline (VER-5.3/5.4) where the TRD says it must call it")
def test_decide_calls_is_suspect():
    from courant.verify import policy

    assert "is_suspect(" in inspect.getsource(policy.decide)


# ---------------------------------------------------------------- VPA-CHK-4
@pytest.mark.xfail(strict=True, reason="VPA-CHK-4: a time-marching run whose Courant number was "
                   "never read passes convergence.residual, so nothing says stability was unjudged")
def test_an_unread_courant_number_leaves_stability_unjudged():
    from courant.verification.checks import ConvergenceCheck, VerifyContext

    text = "".join(line for line in (REAL / "cavity_converged.log").read_text()
                   .splitlines(keepends=True) if not line.startswith("Courant Number"))
    config = CaseConfig(solver="icoFoam", ddt_scheme="Euler")
    result = ConvergenceCheck().run(VerifyContext(run_id="r", events=_events(text),
                                                  config=config))
    assert result.skipped and result.detail
