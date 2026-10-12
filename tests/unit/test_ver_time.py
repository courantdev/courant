"""VER-5.15 (T-VER-TIME): residuals with every `Time =` marker gone are not a residual history.

A solver log whose time markers were destroyed cannot show which step the run reached, or that
its time loop ran to the end, so it cannot be accepted. Before T-VER55 these 11 real logs, each
stripped of every `Time =` line with its residual lines intact, were TRUSTWORTHY (found by the
TST-9 fuzzer, `tests/robust/test_log_fuzz.py`). T-VER55 made them `insufficient` under VER-5.5,
whose sentence says the log has no residual history, which is not what is wrong with them. VER-5.15
names it.
"""

from __future__ import annotations

import gzip
import re
import shutil
from pathlib import Path

import pytest

from courant.events import Event, EventKind, ResidualPayload
from courant.verify import policy, sentences, verify_path

ROOT = Path(__file__).resolve().parents[2]

#: The 11 logs the fuzzer found accepted with every time marker removed (10 Oct 2026).
STRIPPED_WERE_ACCEPTED = (
    "certification/corpus/buoyantPimpleFoam_hotRoom__healthy.log.gz",
    "certification/corpus/icoFoam_cavity__healthy.log.gz",
    "certification/corpus/pimpleFoam_planarContraction__healthy.log.gz",
    "certification/corpus/simpleFoam_pitzDaily__healthy.log.gz",
    "certification/corpus/simpleFoam_pitzDaily__overconstrained_pu.log.gz",
    "certification/monitored/simpleFoam_T3A__fixed_budget/log.simpleFoam",
    "certification/monitored/simpleFoam_pitzDailyExptInlet__fixed_budget/log.simpleFoam",
    "certification/monitored/simpleFoam_pitzDaily__fixed_budget/log.simpleFoam",
    "certification/monitored/simpleFoam_pitzDaily__shipped/log.simpleFoam",
    "tests/fixtures/real_openfoam/buoyantPimpleFoam_hotRoom_healthy.log",
    "tests/fixtures/real_openfoam/cavity_converged.log",
)


def _text(rel: str) -> str:
    path = ROOT / rel
    data = path.read_bytes()
    return (gzip.decompress(data) if path.suffix == ".gz" else data).decode("utf-8", "replace")


def _without_time(text: str) -> str:
    return "".join(ln for ln in text.splitlines(keepends=True) if not re.match(r"^Time = ", ln))


@pytest.mark.parametrize("rel", STRIPPED_WERE_ACCEPTED)
def test_a_log_with_every_time_marker_removed_is_insufficient_by_ver_5_15(tmp_path, rel):
    whole, stripped = tmp_path / "whole.log", tmp_path / "stripped.log"
    text = _text(rel)
    whole.write_text(text)
    stripped.write_text(_without_time(text))
    assert verify_path(whole).verdict == "accept"
    v = verify_path(stripped)
    assert (v.verdict, v.rule, v.grade) == ("insufficient", "VER-5.15", None)
    assert v.not_assessed[0] == sentences.NO_TIME_MARKERS
    assert sentences.NO_RESIDUALS not in v.not_assessed


def test_a_log_with_no_field_residual_stays_ver_5_5(tmp_path):
    log = tmp_path / "log.checkMesh"
    shutil.copy(ROOT / "tests/fixtures/real_openfoam/cavity_checkmesh.log", log)
    assert verify_path(log).rule == "VER-5.5"
    both = tmp_path / "both.log"
    text = _text("tests/fixtures/real_openfoam/cavity_converged.log")
    both.write_text("".join(ln for ln in _without_time(text).splitlines(keepends=True)
                            if "Solving for" not in ln))
    assert verify_path(both).rule == "VER-5.5"


def test_a_crash_still_rejects_without_time_markers(tmp_path):
    log = tmp_path / "diverged.log"
    log.write_text(_without_time(_text("tests/fixtures/real_openfoam/cavity_cfl_diverged.log")))
    assert verify_path(log).verdict == "reject"


def _ev(**kw):
    return Event(run_id="r", seq=0, kind=EventKind.RESIDUAL, payload=ResidualPayload(**kw))


def test_no_time_markers_needs_residuals_and_no_step_number_anywhere():
    assert policy.no_time_markers([_ev(initial_residuals={"p": 0.1})])
    assert not policy.no_time_markers([_ev(initial_residuals={"p": 0.1}),
                                       _ev(iteration=2, courant_max=0.3)])
    assert not policy.no_time_markers([_ev(courant_max=0.3)])
    assert not policy.no_time_markers([])
