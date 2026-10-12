"""Knowledge Base — curated failure taxonomy with signatures + canonical fixes.

Tech Arch §6: "The KB is organized around a curated taxonomy. Each entry carries a
signature (log/mesh/config patterns), a deterministic test, canonical fixes ordered
by cost, and reference cases. Hand-authored first for quality, then grown from real
runs."

MVP retrieval is **deterministic signature matching**, not embeddings — auditable
and dependency-free, matching the control-plane ethos. Three signals combine:
  * finding_ids   — which checker Findings indicate this entry,
  * log_patterns  — regexes over the raw log,
  * config_predicates — named predicates over CaseConfig.

The ``KnowledgeBase`` is the ``query_kb`` tool the LLM calls, and its matches feed
the Evidence Bundle. An embedding/pgvector retriever can be added behind the same
interface once a real corpus exists (§16), without changing callers.
"""

from __future__ import annotations

import re
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from courant.case.config import CaseConfig
from courant.case.schemes import unbounded_convection_schemes
from courant.diagnosis.findings import FindingSet
from courant.taxonomy import EscalationTier, get as taxo_get


class CanonicalFix(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str
    tier: EscalationTier
    diff_hint: str = ""          # human-readable template of the change


class KBSignature(BaseModel):
    model_config = ConfigDict(frozen=True)

    finding_ids: list[str] = Field(default_factory=list)   # checker finding ids
    log_patterns: list[str] = Field(default_factory=list)  # regexes over raw log
    config_predicates: list[str] = Field(default_factory=list)  # named predicates


class KBEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str                    # taxonomy code
    signature: KBSignature
    canonical_fixes: list[CanonicalFix]
    reference_cases: list[str] = Field(default_factory=list)

    @property
    def title(self) -> str:
        return taxo_get(self.code).title

    @property
    def summary(self) -> str:
        return taxo_get(self.code).summary


class KBMatch(BaseModel):
    code: str
    score: float
    why: list[str]               # human-readable reasons the signature matched
    canonical_fixes: list[CanonicalFix]


# Named config predicates the signatures can reference.
CONFIG_PREDICATES: dict[str, Callable[[CaseConfig], bool]] = {
    # 1.0 and steady-only, matching `SchemeToleranceChecker`. At >= 0.9 with no
    # solver gate this scored `NUM.OVER_RELAXATION` +2.0 on `simpleFoam/pitzDaily`
    # and `interFoam/damBreak` — two unmodified tutorials that ship those values
    # and converge fine. See the checker for the argument.
    "relaxation_high": lambda c: c.is_steady() and any(
        v >= 1.0 for v in c.relaxation_factors.values()),
    "fixed_dt_no_adaptivity": lambda c: c.adjust_timestep is False and not c.is_steady(),
    "laminar": lambda c: c.is_laminar(),
    "nu_out_of_range": lambda c: (
        (nu := c.transport.get("nu")) is not None and (nu <= 0 or nu > 1e-2 or nu < 1e-8)
    ),
    "overconstrained_pu": lambda c: any(
        f.get("p") == "fixedValue" and f.get("U") == "fixedValue"
        for f in c.boundary.values()
    ),
    # Shared with the checker rather than reimplemented. This predicate carried
    # the *original* rule — `"bounded" not in s and "upwind" not in s` — for a
    # full phase after the checker's copy was corrected, so it went on scoring
    # `NUM.UNBOUNDED_SCHEME` +2.0 on every RAS case for `limitedLinear`, a
    # definitionally limited scheme. Nothing noticed, because nothing tested the
    # two implementations against each other. See `courant/case/schemes.py`.
    "unbounded_div": lambda c: bool(unbounded_convection_schemes(c.div_schemes)),
}


class KnowledgeBase:
    """Deterministic signature-match retrieval over hand-authored entries."""

    # scoring weights per matching signal
    W_FINDING = 3.0
    W_LOG = 2.0
    W_CONFIG = 2.0
    W_SUGGEST = 1.0

    #: How much a matching finding counts, by its severity.
    #:
    #: Every finding used to count 3.0 whatever its severity, which meant *two
    #: INFO findings outranked one CRITICAL*. On a real diverging pisoFoam case
    #: that is exactly what happened: `bounding.k` and `bounding.epsilon` are
    #: INFO and together scored BC.TURB_INLET at 7.0, above NUM.CFL_VIOLATION at
    #: 6.0 — so a Courant blow-up was blamed on the turbulence inlet condition.
    #: Bounding of k and epsilon is a *symptom* of divergence at least as often
    #: as a cause of it, which is exactly what INFO severity records. The scoring
    #: should not discard the judgement the checker already made.
    #:
    #: **Only INFO is demoted, on purpose.** The first attempt also promoted
    #: CRITICAL to 2.0, which is more principled-looking and measurably worse:
    #: `cfl.max` is CRITICAL and a Courant excursion is a common *secondary*
    #: symptom, so doubling it made NUM.CFL_VIOLATION win four cases it should
    #: have lost and cost 13 points of top-1. Demoting INFO alone fixes the real
    #: case and leaves everything that already worked untouched.
    #:
    #: A steeper spread scored better still on The Bench at `info=0.34` — and
    #: that was rejected. Tuning a constant to a knife edge on 15 fixtures
    #: authored against the *old* scoring is fitting the fixtures, not the world,
    #: which is the circularity the real-solver certification exists to escape.
    SEVERITY_WEIGHT = {"critical": 1.0, "warn": 1.0, "info": 0.5}

    def __init__(self, entries: list[KBEntry]):
        self.entries = {e.code: e for e in entries}
        self._compiled = {
            e.code: [re.compile(p, re.I) for p in e.signature.log_patterns]
            for e in entries
        }

    def query(self, findings: FindingSet | None = None, log_text: str = "",
              config: CaseConfig | None = None, top_k: int = 5) -> list[KBMatch]:
        finding_ids = findings.ids if findings else set()
        suggested = set((findings.suggested_codes() if findings else []))

        matches: list[KBMatch] = []
        for code, entry in self.entries.items():
            score = 0.0
            why: list[str] = []
            sig = entry.signature

            hit_findings = [fid for fid in sig.finding_ids if fid in finding_ids]
            if hit_findings:
                detail = []
                for fid in hit_findings:
                    hit = findings.get(fid) if findings else None
                    severity = hit.severity.value if hit is not None else "warn"
                    weight = self.SEVERITY_WEIGHT.get(severity, 1.0)
                    score += self.W_FINDING * weight
                    detail.append(f"{fid}({severity[:4]}×{weight:g})")
                why.append("findings " + ", ".join(detail))

            for i, rx in enumerate(self._compiled[code]):
                if log_text and rx.search(log_text):
                    score += self.W_LOG
                    why.append(f"log~/{sig.log_patterns[i]}/")

            if config is not None:
                for pred in sig.config_predicates:
                    fn = CONFIG_PREDICATES.get(pred)
                    if fn and fn(config):
                        score += self.W_CONFIG
                        why.append(f"config:{pred}")

            if code in suggested:
                score += self.W_SUGGEST
                why.append("finding-suggested")

            if score > 0:
                matches.append(KBMatch(code=code, score=score, why=why,
                                       canonical_fixes=entry.canonical_fixes))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def get(self, code: str) -> KBEntry | None:
        return self.entries.get(code)

    def __len__(self) -> int:
        return len(self.entries)
