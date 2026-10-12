"""Diagnosis output contract (Tech Arch §5.2).

The Diagnosis Engine — whether the Phase-0 heuristic stub or the Phase-1 LLM —
produces a ``DiagnosisResponse``: a ranked list of root-cause hypotheses that
each cite real deterministic Findings, plus an optional proposed minimal fix.

Schema-validated before use. The model defers to deterministic Findings and
expresses calibrated confidence; it never declares a result physically valid.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Hypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_cause_code: str                 # must map to the failure taxonomy
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)  # Finding ids cited
    explanation: str = ""                # terse, human-readable

    @field_validator("root_cause_code")
    @classmethod
    def _code_in_taxonomy(cls, v: str) -> str:
        from courant.taxonomy import TAXONOMY

        if v not in TAXONOMY:
            raise ValueError(f"root_cause_code {v!r} is not in the taxonomy")
        return v


class ProposedFix(BaseModel):
    model_config = ConfigDict(frozen=True)

    diff: str                            # unified diff over whitelisted paths
    rationale: str
    escalation_tier: Literal["numerics", "schemes", "mesh"]


class DiagnosisResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypotheses: list[Hypothesis]
    proposed_fix: ProposedFix | None = None

    @field_validator("hypotheses")
    @classmethod
    def _sorted(cls, v: list[Hypothesis]) -> list[Hypothesis]:
        """An **empty** diagnosis is valid, and it is the honest answer to a run
        that produced no signal.

        This used to raise, and the cost was paid one layer up: the deterministic
        engine kept a "default guess" — `CONV.RESIDUAL_PLATEAU` at 0.1
        confidence — purely to satisfy this rule. A named taxonomy code does not
        read as a guess to anything downstream. The certification harness
        recorded it as *the* diagnosis, the report renders it as the leading
        hypothesis, and the fix loop proposes repairs for it. A low confidence
        number is not a disclaimer.

        It was reached on real runs: `icoFoam_cavity::missing_dict` landed here
        for a whole certification round because a checkMesh failure aborted the
        solve before it could print the error identifying the fault.

        The API already treats an empty result as a first-class outcome — it
        drops uncited hypotheses and *refunds the quota* rather than charging a
        customer for "I don't know". The schema was the last thing insisting on
        an answer.
        """
        return v

    def ranked_codes(self) -> list[str]:
        """Root-cause codes, most-confident first."""
        return [
            h.root_cause_code
            for h in sorted(self.hypotheses, key=lambda h: h.confidence, reverse=True)
        ]

    def top_k(self, k: int) -> list[str]:
        return self.ranked_codes()[:k]
