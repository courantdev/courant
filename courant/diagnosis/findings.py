"""The Finding contract — typed output of the deterministic checkers (Tech Arch §5.1).

Deterministic checkers are pure functions over {config, mesh, events}. Each returns
Findings: hard, evidence-backed facts with stable ids. A Finding is NOT a diagnosis —
it is a fact that a diagnosis may cite. The LLM reasons over Findings and must cite
their ids (``evidence_refs``); it never invents facts not present here.

``suggests`` links a Finding to the taxonomy code(s) it is evidence for — a
deterministic hint the diagnosis layer may use and the LLM may override.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FindingSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warn": 1, "critical": 2}[self.value]


class Finding(BaseModel):
    """A single hard fact established by a deterministic checker."""

    model_config = ConfigDict(frozen=True)

    id: str                                  # stable within a run; cited by hypotheses
    checker: str                             # producing checker, e.g. "cfl"
    title: str                               # short label
    severity: FindingSeverity
    detail: str = ""                         # terse fact (numbers ok; not re-parsed)
    value: float | None = None               # headline number, e.g. max Courant
    threshold: float | None = None           # the limit it violated, if any
    field: str | None = None                 # solved field involved, if any
    location: str | None = None              # cell id / patch / time / file
    log_offset: int | None = None            # provenance into the raw log
    suggests: list[str] = Field(default_factory=list)  # taxonomy codes

    @field_validator("suggests")
    @classmethod
    def _codes_valid(cls, v: list[str]) -> list[str]:
        from courant.taxonomy import TAXONOMY

        for code in v:
            if code not in TAXONOMY:
                raise ValueError(f"suggested code {code!r} is not in the taxonomy")
        return v


class FindingSet:
    """An ordered, id-unique collection of Findings for one run."""

    def __init__(self, findings: list[Finding] | None = None):
        self._by_id: dict[str, Finding] = {}
        for f in findings or []:
            self.add(f)

    def add(self, finding: Finding) -> Finding:
        if finding.id in self._by_id:
            raise ValueError(f"duplicate finding id: {finding.id!r}")
        self._by_id[finding.id] = finding
        return finding

    def extend(self, findings: list[Finding]) -> None:
        for f in findings:
            self.add(f)

    def get(self, finding_id: str) -> Finding | None:
        return self._by_id.get(finding_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def ids(self) -> set[str]:
        return set(self._by_id.keys())

    def by_checker(self, checker: str) -> list[Finding]:
        return [f for f in self if f.checker == checker]

    def by_severity(self, minimum: FindingSeverity) -> list[Finding]:
        return [f for f in self if f.severity.rank >= minimum.rank]

    def suggested_codes(self) -> list[str]:
        """Union of taxonomy codes suggested by findings, ranked by cumulative
        evidence weight (critical=3, warn=2, info=1)."""
        weight: dict[str, int] = {}
        w = {FindingSeverity.CRITICAL: 3, FindingSeverity.WARN: 2, FindingSeverity.INFO: 1}
        for f in self:
            for code in f.suggests:
                weight[code] = weight.get(code, 0) + w[f.severity]
        return [c for c, _ in sorted(weight.items(), key=lambda kv: kv[1], reverse=True)]

    def max_severity(self) -> FindingSeverity:
        if not self._by_id:
            return FindingSeverity.INFO
        return max((f.severity for f in self), key=lambda s: s.rank)
