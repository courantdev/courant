"""Physics Verification Suite — deterministic A-D Trust Grade over the solution."""

from courant.verification.checks import (
    DEFAULT_CHECKS,
    VerifyContext,
    run_checks,
)
from courant.verification.fields import FieldData
from courant.verification.grade import (
    TrustGrade,
    TrustReport,
    VerificationResult,
    VSeverity,
    compute_grade,
)
from courant.verification.suite import verify, verify_run

__all__ = [
    "VSeverity",
    "TrustGrade",
    "VerificationResult",
    "TrustReport",
    "compute_grade",
    "VerifyContext",
    "FieldData",
    "DEFAULT_CHECKS",
    "run_checks",
    "verify",
    "verify_run",
]
