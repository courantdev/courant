"""Mutation-070: tests that kill the surviving mutants of `courant/verify/policy.py`,
`courant/verify/checks.py` and `courant/watcher/exitcode.py`.

Each test pins the documented behaviour: SPEC-4 (VER-5.7, VER-5.8: a reject on the run's own
evidence keeps its rule), VER-6 (reason ordering), GATE-13 (RM-37's promotion of VER-18's
`phi` balance) and VER-3 / VER-3a (whose `End` is the run's end). Pure functions, called with
small stand-ins, as `tests/unit/test_verify.py` and `tests/unit/test_spec3.py` do.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from courant.verify import checks, policy
from courant.verify.schema import Reason
from courant.watcher.exitcode import ends_in_end, infer_exit_code, run_exit_code


def _block(*, nonconformant=0, unreadable=0, locked=None, invalid=()):
    """A conformance block as `apply_conformance` reads it."""
    return SimpleNamespace(spec_sha256="a", locked_sha256=locked, nonconformant=nonconformant,
                           unreadable=unreadable, invalid=invalid)


# ----------------------------------------------------------------- apply_conformance (SPEC-4)
# kills 84a8b43e26 courant/verify/policy.py:133
def test_a_void_spec_leaves_a_reject_on_the_runs_own_evidence_as_it_is():
    void = _block(nonconformant=1, locked="b")
    assert policy.apply_conformance("NOT_TRUSTWORTHY", "VER-5.2", void) == \
        ("NOT_TRUSTWORTHY", "VER-5.2")
    assert policy.apply_conformance("NOT_TRUSTWORTHY", "VER-5.4", void, extracted=True) == \
        ("NOT_TRUSTWORTHY", "VER-5.4")
    assert policy.apply_conformance("TRUSTWORTHY", "VER-5.6", void) == \
        ("NOT_ASSESSABLE", "VER-5.8")


# kills 32b88134d5 courant/verify/policy.py:137
def test_a_mismatch_keeps_the_rule_of_a_run_already_rejected():
    """VER-5.7: NOT_TRUSTWORTHY "unless already so by another rule, whose id is kept"; and an
    extracted spec never lowers a reject to insufficient (VER-5.12)."""
    mismatch = _block(nonconformant=1)
    assert policy.apply_conformance("NOT_TRUSTWORTHY", "VER-5.2", mismatch) == \
        ("NOT_TRUSTWORTHY", "VER-5.2")
    assert policy.apply_conformance("NOT_TRUSTWORTHY", "VER-5.3", mismatch, extracted=True) == \
        ("NOT_TRUSTWORTHY", "VER-5.3")


# ----------------------------------------------------------------- order_reasons (VER-6)
def _r(code, kind, severity, log_offset=None):
    return Reason(code=code, kind=kind, severity=severity, summary="s", input="i", method="m",
                  log_offset=log_offset)


def _codes(reasons):
    return [r.code for r in policy.order_reasons(reasons)]


# kills e07559ebf5 policy.py:169, 3286f581ea policy.py:170, 2a20267f30 policy.py:171 (1 -> 0)
def test_conformance_and_events_sort_before_critical_findings_whatever_their_codes():
    assert _codes([_r("a.finding", "finding", "critical"),
                   _r("conformance.nu", "conformance", "critical")]) == \
        ["conformance.nu", "a.finding"]
    assert _codes([_r("a.finding", "finding", "critical"),
                   _r("run.crash", "event", "critical")]) == ["run.crash", "a.finding"]


# kills 67d5c8f5b1 policy.py:171 (1 -> 2), 3d9d32c7d9 policy.py:172 (2 -> 1),
# 3fd69763b4 policy.py:173 (2 -> 1)
def test_critical_findings_sort_before_major_and_critical_checks():
    assert _codes([_r("a.check", "check", "major"), _r("z.finding", "finding", "critical")]) == \
        ["z.finding", "a.check"]
    assert _codes([_r("a.check", "check", "critical"),
                   _r("z.finding", "finding", "critical")]) == ["z.finding", "a.check"]


# kills 1b9a93856b policy.py:172 (2 -> 3)
def test_critical_checks_sort_before_warn_findings():
    assert _codes([_r("a.finding", "finding", "warn"), _r("z.check", "check", "critical")]) == \
        ["z.check", "a.finding"]


# kills df8254cbca policy.py:174 (3 -> 4), 9d67845c95 policy.py:175 (4 -> 3)
def test_warn_findings_sort_before_minor_checks():
    assert _codes([_r("a.check", "check", "minor"), _r("z.finding", "finding", "warn")]) == \
        ["z.finding", "a.check"]


# kills 804bf6822b policy.py:175 (4 -> 5), ca5c2bd671 policy.py:182 (5 -> 4)
def test_minor_checks_sort_before_an_insufficient_reason():
    """The `insufficient` reason (kind insufficient, severity major, `verdict2`) is in no VER-6
    group: it comes after every grouped reason."""
    assert _codes([_r("a.insufficient", "insufficient", "major"),
                   _r("z.check", "check", "minor")]) == ["z.check", "a.insufficient"]


# kills 554194454c policy.py:183 (cond -> False), 3b0458c90d policy.py:183 (is not -> is)
def test_reasons_with_one_code_are_ordered_by_log_offset():
    ordered = policy.order_reasons([_r("run.crash", "event", "critical", 20),
                                    _r("run.crash", "event", "critical", 5)])
    assert [r.log_offset for r in ordered] == [5, 20]


# kills 1cbeb65eb9 policy.py:183 (cond -> True), ac76dc010a policy.py:183 (-1 -> 0)
def test_a_reason_with_no_log_offset_sorts_before_one_with_an_offset():
    for offset in (0, 7):
        ordered = policy.order_reasons([_r("run.crash", "event", "critical", offset),
                                        _r("run.crash", "event", "critical", None)])
        assert [r.log_offset for r in ordered] == [None, offset]


# ----------------------------------------------------------------- checks (GATE-13)
# kills 4be69b7f92 courant/verify/checks.py:36
def test_ver_18s_phi_balance_is_promoted_on_rm_37s_result():
    assert checks.PROMOTE_MASS_FLUX_PHI is True
    assert checks.MASS_FLUX_PHI in checks.PROMOTED
    assert checks.MASS_FLUX_PHI not in checks.NEXT
    assert checks.enabled(checks.MASS_FLUX_PHI, "default")


# ----------------------------------------------------------------- exitcode (VER-3, VER-3a)
CUT = "Time = 1\nsmoothSolver:  Solving for Ux\n\nTime = 2\nsmoothSolver:  Solving for Ux\n"
UTILITY = "Exec   : blockMesh\n\nEnd\n\n"


# kills 07ca285fed courant/watcher/exitcode.py:83 (group(1) -> group(0)),
# fa02d51f69 exitcode.py:83 (cond -> True)
def test_an_unprefixed_log_reads_its_end_after_the_last_step_as_the_default_does():
    """VER-3a reads an End behind blanks only where the last step sits behind them
    (foamMultiRun); a log with no prefix keeps the default's line rule (`\\nEnd\\n`)."""
    text = CUT + "End \nFinalising parallel run\n"
    assert not ends_in_end(text)
    assert not ends_in_end(text, after_last_step=True)


# kills 0c070c7991 exitcode.py:89 (cond -> False), a3f879160d exitcode.py:89 (or -> and)
@pytest.mark.parametrize("text", ["End", "End\n\n", CUT + "End", CUT + "End  \n"])
def test_a_watched_log_whose_last_line_is_end_has_ended(text):
    """`courant watch` asks with `last=True` once the stream is over: its last line counts."""
    assert ends_in_end(text, last=True)


# kills 94af2342b6 exitcode.py:92 (0 -> 1), 835ae3dc73 exitcode.py:92 (>= -> >)
def test_a_watched_tail_that_begins_at_the_end_marker_has_ended():
    """The watcher's tail is a window: it can start exactly at "\\nEnd\\n"."""
    assert ends_in_end("\nEnd\nFinalising parallel run\n", last=True)


# kills 08c5f212d0 exitcode.py:95
def test_infer_exit_code_reads_any_end_by_default():
    """VER-3a is opt-in on `infer_exit_code` (the verdict asks it through `verdict_exit_code`)."""
    assert infer_exit_code(UTILITY + CUT) == 0
    assert infer_exit_code(UTILITY + CUT, after_last_step=True) is None


# kills f29eb31438 exitcode.py:102
def test_run_exit_code_reads_any_end_by_default():
    assert run_exit_code(UTILITY + CUT, None) == 0
    assert run_exit_code(UTILITY + CUT, None, after_last_step=True) is None
