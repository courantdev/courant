"""The deterministic diagnoser: rank taxonomy codes from the checkers' Findings and KB signature
matches, cite the Findings behind each, no language model.

It is part of the verdict's pipeline (the top causes `courant verify` reports), so it lives with
the checks. Every diagnoser maps an ``EvidenceBundle`` to a ``DiagnosisResponse`` (ranked
hypotheses citing real Findings); the model door's ``LLMDiagnoser`` implements the same protocol
and falls back to this one.
"""

from __future__ import annotations

from typing import Protocol

from courant.diagnosis.bundle import EvidenceBundle
from courant.diagnosis.schema import DiagnosisResponse, Hypothesis


class Diagnoser(Protocol):
    name: str

    def diagnose(self, bundle: EvidenceBundle) -> DiagnosisResponse:
        ...


#: An uncited hypothesis is scaled to sit below every cited one.
#:
#: Phase 1's definition of done is *"every diagnosis cites at least one real
#: Finding"*, and nothing enforced it — a code could reach the top of the
#: ranking on a config predicate alone. `PHYS.LAMINAR_TURBULENT` did exactly
#: that on a crashed case: `simulationType laminar` scored 2.0 from
#: configuration, every finding-backed code scored less, and the report led with
#: a claim it could not point at a single line of the log to support.
#:
#: That is a worse failure than being wrong. An engineer can argue with a wrong
#: diagnosis that shows its evidence; a right one that shows none is
#: indistinguishable from a guess, and the whole product rests on not being one.
#: Configuration remains ranking signal — it just cannot be the entire case.
#:
#: **Expressed as confidence, not as list order.** The first attempt at this
#: sorted the hypothesis list, which changed nothing anywhere: `ranked_codes()`
#: re-sorts by confidence, and the fix loop, the certification harness and The
#: Bench all read *that*. Only `hypotheses[0]` saw the reordering. A ranking
#: rule has to be written in whatever the consumers actually rank by.
#:
#: The smallest score a match can carry is 1.0 (a single `W_SUGGEST`), so the
#: lowest cited confidence is 1/(1+3) = 0.25. Scaling by 0.24 puts every uncited
#: hypothesis below that while preserving their order among themselves.
UNCITED_SCALE = 0.24


def _confidence(score: float, *, cited: bool) -> float:
    raw = score / (score + 3.0)                     # bounded 0..1
    return round(raw if cited else raw * UNCITED_SCALE, 3)


class DeterministicDiagnoser:
    """Rank taxonomy codes from KB signature-match scores; cite the Findings that
    drove each. This is Layer 1 acting alone — the honest deterministic baseline."""

    name = "deterministic"

    def __init__(self, kb=None):
        from courant.diagnosis.kb_data import build_default_kb

        self._kb = kb or build_default_kb()

    def diagnose(self, bundle: EvidenceBundle) -> DiagnosisResponse:
        finding_ids = {f.id for f in bundle.det_findings}

        def evidence_for(code: str, kb_finding_ids: list[str]) -> list[str]:
            refs = [f.id for f in bundle.det_findings if code in f.suggests]
            refs += [fid for fid in kb_finding_ids
                     if fid in finding_ids and fid not in refs]
            return refs

        from courant.taxonomy import TAXONOMY

        hypotheses: list[Hypothesis] = []
        for m in bundle.kb_matches[:3]:
            entry = self._kb.get(m.code)
            kb_fids = entry.signature.finding_ids if entry else []
            taxo = TAXONOMY.get(m.code)
            refs = evidence_for(m.code, kb_fids)
            hypotheses.append(Hypothesis(
                root_cause_code=m.code,
                confidence=_confidence(m.score, cited=bool(refs)),
                evidence_refs=refs,
                explanation=taxo.summary if taxo else "signature match",
            ))

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        # No signal at all: say nothing.
        #
        # This used to emit `CONV.RESIDUAL_PLATEAU` at 0.1 confidence as a
        # "default guess", and a named taxonomy code does not read as a guess to
        # anyone downstream. The certification harness recorded it as *the*
        # diagnosis; the report renders it as the leading hypothesis; the fix
        # loop proposes repairs for it. A confidence number is not a disclaimer.
        #
        # It was reached more often than it looks. `icoFoam_cavity::missing_dict`
        # landed here for a whole run because a checkMesh failure aborted the
        # solve before it could print the error that identifies the fault — so
        # the answer to "what went wrong" was a plausible-sounding code with no
        # connection to the case at all.
        #
        # An empty diagnosis is already representable and already handled
        # everywhere it matters. "We could not tell" is a worse-sounding answer
        # and a much better one.
        return DiagnosisResponse(hypotheses=hypotheses)
