"""Unit tests for the Trust Grade computation (deterministic)."""

from courant.verification.grade import (
    TrustGrade,
    TrustReport,
    VerificationResult,
    VSeverity,
    compute_grade,
)


def _r(code, severity, passed=True, skipped=False):
    return VerificationResult(check_code=code, check_name=code, severity=severity,
                              passed=passed, skipped=skipped)


def test_all_pass_is_A():
    results = [_r("a", VSeverity.OK), _r("b", VSeverity.OK)]
    assert compute_grade(results) is TrustGrade.A


def test_minor_note_is_B():
    results = [_r("a", VSeverity.OK), _r("b", VSeverity.MINOR)]
    assert compute_grade(results) is TrustGrade.B


def test_major_concern_is_C():
    results = [_r("a", VSeverity.MINOR), _r("b", VSeverity.MAJOR)]
    assert compute_grade(results) is TrustGrade.C


def test_critical_fails_physics_is_D():
    results = [_r("a", VSeverity.OK), _r("b", VSeverity.CRITICAL), _r("c", VSeverity.MINOR)]
    assert compute_grade(results) is TrustGrade.D


def test_worst_check_dominates():
    results = [_r("a", VSeverity.MAJOR), _r("b", VSeverity.CRITICAL), _r("c", VSeverity.OK)]
    assert compute_grade(results) is TrustGrade.D


def test_skipped_checks_do_not_affect_grade():
    results = [_r("a", VSeverity.OK), _r("b", VSeverity.CRITICAL, skipped=True)]
    assert compute_grade(results) is TrustGrade.A


def test_trust_report_partitions():
    results = [_r("a", VSeverity.OK), _r("b", VSeverity.MAJOR),
               _r("c", VSeverity.CRITICAL, skipped=True)]
    report = TrustReport.from_results("run-1", results)
    assert report.grade is TrustGrade.C
    assert [r.check_code for r in report.deductions] == ["b"]
    assert [r.check_code for r in report.skipped] == ["c"]
