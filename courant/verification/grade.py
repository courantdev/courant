"""Verification results + Trust Grade (Tech Arch §7, §7.1).

The verification suite is entirely deterministic code over the run's solution +
history. It produces an A-D Trust Grade where every deduction links to evidence.
The grade is computed by code alone — the LLM may explain it, never change it.

Grade rule:  start at A; each check can only lower the grade.
  CRITICAL -> D   (fails physics)
  MAJOR    -> C   (major concern)
  MINOR    -> B   (minor notes)
  OK       -> no change
The final grade is the worst any single check forced.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class VSeverity(str, Enum):
    OK = "ok"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"ok": 0, "minor": 1, "major": 2, "critical": 3}[self.value]


class TrustGrade(str, Enum):
    A = "A"        # all pass
    B = "B"        # minor notes
    C = "C"        # major concern
    D = "D"        # fails physics

    @property
    def quality(self) -> int:
        return {"A": 3, "B": 2, "C": 1, "D": 0}[self.value]


# worst grade a check of a given severity permits
_SEVERITY_CAP = {
    VSeverity.OK: TrustGrade.A,
    VSeverity.MINOR: TrustGrade.B,
    VSeverity.MAJOR: TrustGrade.C,
    VSeverity.CRITICAL: TrustGrade.D,
}


class VerificationResult(BaseModel):
    """One deterministic physics check over the run."""

    model_config = ConfigDict(frozen=True)

    check_code: str                      # e.g. "conservation.mass"
    check_name: str
    severity: VSeverity
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str = ""                     # terse, evidence-bearing
    evidence: dict = Field(default_factory=dict)
    skipped: bool = False                # e.g. needs solution fields not available


def compute_grade(results: list[VerificationResult]) -> TrustGrade:
    grade = TrustGrade.A
    for r in results:
        if r.skipped:
            continue
        cap = _SEVERITY_CAP[r.severity]
        if cap.quality < grade.quality:
            grade = cap
    return grade


class TrustReport(BaseModel):
    """The Trust Grade plus every check that produced it."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    grade: TrustGrade
    results: list[VerificationResult]

    @property
    def deductions(self) -> list[VerificationResult]:
        return [r for r in self.results
                if not r.skipped and r.severity is not VSeverity.OK]

    @property
    def skipped(self) -> list[VerificationResult]:
        return [r for r in self.results if r.skipped]

    @classmethod
    def from_results(cls, run_id: str,
                     results: list[VerificationResult]) -> "TrustReport":
        return cls(run_id=run_id, grade=compute_grade(results), results=results)
