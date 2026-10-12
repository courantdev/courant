"""SB-11: one outcome from one function (`courant.verify.schema.outcome_of`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from courant.verify import verify_path
from courant.verify.schema import OUTCOMES, RESOLVE_OUTCOMES, executed_in_log, outcome_of


def outcome(verdict, **kw) -> dict:
    """The advisory tool's view of `outcome_of`: its fields as a dict."""
    return outcome_of(verdict, **kw).model_dump()

CORPUS = Path(__file__).resolve().parents[2] / "certification" / "corpus"


@pytest.fixture
def demo(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path / "demo")


def _locked(case: Path, tmp: Path, edit=None):
    from courant.spec import draft_spec, spec_sha256

    spec = tmp / "spec.json"
    spec.write_text(draft_spec(case).to_json())
    lock = spec_sha256(spec)
    if edit:
        edit()
    return verify_path(case, spec=spec, spec_locked_sha256=lock)


def test_the_list_is_closed():
    assert OUTCOMES == ("valid_conformant", "invalid", "nonconformant", "insufficient",
                        "did_not_execute", *RESOLVE_OUTCOMES)
    # One list: the module once bound OUTCOMES twice and the second dropped the mesh floor.
    assert "resolve_mesh_below_floor" in OUTCOMES and len(set(OUTCOMES)) == len(OUTCOMES)


def test_an_accept_on_a_locked_specification_is_valid_conformant(demo, tmp_path):
    v = _locked(demo, tmp_path)
    got = outcome(v, executed=True, executed_source="resolve", reported_success=True)
    assert got["outcome"] == "valid_conformant"
    assert got["silent_error"] is False and got["unverified_success"] is False


def test_execution_alone_scores_nothing(demo):
    v = verify_path(demo)                              # accepted, but against no specification
    assert v.verdict == "accept"
    got = outcome(v, executed=True, executed_source="resolve", reported_success=True)
    assert got["outcome"] == "insufficient" and got["silent_error"] is False
    # The outcome is insufficient and success was claimed: an unverified success (SB-11).
    assert got["unverified_success"] is True and got["passed"] is False


def test_a_wrong_problem_is_nonconformant_and_a_silent_error_when_success_was_reported(demo, tmp_path):
    def edit():
        tp = demo / "constant" / "transportProperties"
        tp.write_text(tp.read_text().replace("1e-05", "3e-05"))

    v = _locked(demo, tmp_path, edit)
    claimed = outcome(v, executed=True, executed_source="agent_log", reported_success=True)
    assert claimed["outcome"] == "nonconformant" and claimed["silent_error"] is True
    unknown = outcome(v, executed=True, executed_source="agent_log")
    assert unknown["silent_error"] is False and unknown["reported_success"] is None
    honest = outcome(v, executed=True, executed_source="agent_log", reported_success=False)
    assert honest["silent_error"] is False


def test_an_invalid_run_that_executed_is_invalid_and_both_majors_agree():
    v = verify_path(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz")
    a = outcome(v, executed=True, executed_source="agent_log", reported_success=True)
    b = outcome(v.to_v1(), executed=True, executed_source="agent_log", reported_success=True)
    assert a == b and a["outcome"] == "invalid" and a["silent_error"] is True


def test_insufficient_with_a_success_claim_is_an_unverified_success_not_a_silent_error(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("nothing\n")
    v = verify_path(log)
    got = outcome(v, executed=True, executed_source="agent_log", reported_success=True)
    assert got["outcome"] == "insufficient"
    assert got["unverified_success"] is True and got["silent_error"] is False


def test_a_run_that_did_not_execute_is_never_a_silent_error():
    v = verify_path(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz")
    got = outcome(v, executed=False, executed_source="resolve", reported_success=True)
    assert got["outcome"] == "did_not_execute" and got["silent_error"] is False


def test_the_re_solves_classes_pass_through_and_are_never_silent_errors():
    v = verify_path(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz")
    for cls in RESOLVE_OUTCOMES:
        got = outcome(v, executed=None, executed_source="resolve", reported_success=True,
                      resolve_class=cls)
        assert got["outcome"] == cls and got["silent_error"] is False


def test_bad_arguments_are_refused():
    v = verify_path(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz")
    with pytest.raises(ValueError):
        outcome(v, executed=True, executed_source="guess")
    with pytest.raises(ValueError):
        outcome(v, executed=True, executed_source="resolve", resolve_class="made_up")


def test_executed_in_log():
    assert executed_in_log("Time = 1\n\nEnd\n")
    assert not executed_in_log("Time = 1\n")
    assert not executed_in_log("--> FOAM FATAL ERROR\nEnd\n")


def test_no_verdict_is_did_not_execute_only_when_the_run_is_known_not_to_have_executed():
    assert outcome(None, executed=False, executed_source="agent_log")["outcome"] == "did_not_execute"
    assert outcome(None, executed=None, executed_source="resolve")["outcome"] == "resolve_failed"
    assert outcome(None, executed=True, executed_source="resolve")["outcome"] == "resolve_failed"
