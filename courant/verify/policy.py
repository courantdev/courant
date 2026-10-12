"""The verdict policy: the only place a verdict is decided (TRD VER-5, VER-6, ARC-2).

A pure function of what the deterministic nodes established. No model output
reaches it (INV-1). Diagnosis is attached as explanation and never consulted
here (ARC-3).
"""

from __future__ import annotations

import re

from courant.events import EventKind
from courant.events.diverged import diverged

#: Markers any OpenFOAM solver log carries. Without one of them the text is not
#: a solver log, and inferring "it crashed" from a missing `End` would judge a
#: file we cannot read.
_RECOGNISED = re.compile(
    r"^(?:Exec\s*:|Time = |\S+:\s+Solving for |--> FOAM FATAL|\[\d+\] --> FOAM FATAL)",
    re.MULTILINE)


#: Findings that report Courant's own inability to read an input. They are a
#: gap in what was checked, not evidence about the run (INV-3), so they go under
#: "not assessed" and never decide a verdict. RRD RM-01 v0 found the first one
#: deciding a healthy tutorial's verdict.
READER_LIMITATIONS = frozenset({"case.unreadable_dict"})


def evidential(findings):
    """Findings that are evidence about the run, as opposed to about our reader."""
    return [f for f in findings if f.id not in READER_LIMITATIONS]


def recognised(log_text: str) -> bool:
    return bool(_RECOGNISED.search(log_text))


def is_suspect(findings, trust) -> bool:
    """Whether a run that finished still needs explaining.

    There are two independent signals, because they catch different mistakes:

    * A checker raised something above INFO: a viscosity outside any
      plausible fluid range, a residual plateau, a bounding storm. The run
      finished, and the setup is still wrong.
    * The Trust Grade is C or D: the deterministic physics suite found a
      conservation or convergence failure.

    It stays silent otherwise. A tool that cries wolf on healthy runs gets
    uninstalled, which is why certification asserts that an unmodified tutorial
    produces no findings. This is the one definition of "suspect". The
    orchestrator and the verifier both call it.
    """
    from courant.diagnosis.findings import FindingSeverity
    from courant.verification.grade import TrustGrade

    if any(f.severity is not FindingSeverity.INFO for f in evidential(findings)):
        return True
    return trust is not None and trust.grade in (TrustGrade.C, TrustGrade.D)


def is_residual_step(event) -> bool:
    """VER-5.5: whether an event is residual evidence. A RESIDUAL event counts only when it is
    a numbered solver step (`iteration` set, from a `Time =` line) that carries at least one
    field's initial residual (a `Solving for` line).

    The parser emits a RESIDUAL event for any step with data, so a step holding only a Courant
    number, continuity errors or a timing line is one too, and so is the single step of a log
    with no time loop (a checkMesh log). None of those is a residual history: counting them
    accepted `tests/fixtures/real_openfoam/cavity_checkmesh.log` as TRUSTWORTHY grade A, and a
    healthy corpus log with every `Solving for` line, or every `Time =` line, removed.
    """
    if event.kind is not EventKind.RESIDUAL:
        return False
    p = getattr(event, "payload", None)
    return p is not None and p.iteration is not None and bool(p.initial_residuals)


def no_time_markers(events) -> bool:
    """VER-5.15: field residuals were read, but no step carries a step number, because no
    `Time =` line was read. The log cannot show which step the run reached, or that its time
    loop ran to the end, so its residuals are not a history (T-VER-TIME)."""
    steps = [e.payload for e in events
             if e.kind is EventKind.RESIDUAL and getattr(e, "payload", None) is not None]
    return (any(p.initial_residuals for p in steps)
            and all(p.iteration is None for p in steps))


def decide(is_recognised: bool, events, findings, trust) -> tuple[str, str]:
    """(verdict, rule). Rules are evaluated in order; the first that matches wins."""
    if not is_recognised or not events:
        return "NOT_ASSESSABLE", "VER-5.1"
    kinds = {e.kind for e in events}
    if EventKind.CRASH in kinds or diverged(events):            # ADP-5: one definition
        return "NOT_TRUSTWORTHY", "VER-5.2"
    from courant.verification.grade import TrustGrade

    if trust is not None and trust.grade in (TrustGrade.C, TrustGrade.D):
        return "NOT_TRUSTWORTHY", "VER-5.3"
    from courant.diagnosis.findings import FindingSeverity

    if any(f.severity is not FindingSeverity.INFO for f in evidential(findings)):
        return "NOT_TRUSTWORTHY", "VER-5.4"
    if not any(is_residual_step(e) for e in events):
        if no_time_markers(events):
            return "NOT_ASSESSABLE", "VER-5.15"
        return "NOT_ASSESSABLE", "VER-5.5"
    return "TRUSTWORTHY", "VER-5.6"


def apply_conformance(verdict: str, rule: str, block, *,
                      extracted: bool = False) -> tuple[str, str]:
    """Fold deck-versus-spec conformance into the verdict (PRD 2.0 P2-SPEC-2, P2-SPEC-3).

    - VER-5.8: the specification is not the one locked before the run. Its conformance is
      void; a verdict already rejected on the run's own evidence stays rejected.
    - VER-5.7: a value the specification names does not match the case. The run solved a
      different problem, however well it converged (RRD RM-18).
    - VER-5.11: the specification is impossible in itself (SPEC-9): no case could conform, so
      it says nothing about this one. Never `reject` on its account.
    - VER-5.12: `extracted` (SPEC-7): the specification was drafted by a model and nobody
      confirmed it, so it is not binding. A mismatch gives `insufficient`, never `reject`; with
      no mismatch the verdict is the run's own (an accept is decided by validity, not by the
      specification). It is an input here, before anything is decided, so no later label can
      be the only thing standing between an unconfirmed extraction and a `reject`.
    - VER-5.9: a named value could not be read from the case, so an otherwise clean run
      cannot be accepted as the problem that was asked.
    """
    if block.locked_sha256 and block.locked_sha256 != block.spec_sha256:
        return (verdict, rule) if verdict == "NOT_TRUSTWORTHY" else ("NOT_ASSESSABLE", "VER-5.8")
    if spec_invalid(block):
        return (verdict, rule) if verdict == "NOT_TRUSTWORTHY" else ("NOT_ASSESSABLE", "VER-5.11")
    if block.nonconformant:
        if verdict == "NOT_TRUSTWORTHY":
            return verdict, rule
        return ("NOT_ASSESSABLE", "VER-5.12") if extracted else ("NOT_TRUSTWORTHY", "VER-5.7")
    if block.unreadable and verdict == "TRUSTWORTHY":
        return "NOT_ASSESSABLE", "VER-5.9"
    return verdict, rule


def spec_void(block) -> bool:
    return bool(block.locked_sha256 and block.locked_sha256 != block.spec_sha256)


def spec_invalid(block) -> bool:
    """SPEC-9: the specification is impossible in itself (and not void, which comes first)."""
    return bool(getattr(block, "invalid", ())) and not spec_void(block)


_GROUP = {
    ("conformance", "critical"): 0,
    ("event", "critical"): 0,
    ("finding", "critical"): 1,
    ("check", "critical"): 2,
    ("check", "major"): 2,
    ("finding", "warn"): 3,
    ("check", "minor"): 4,
}


def order_reasons(reasons):
    """VER-6: by group, then code, then log offset."""
    return tuple(sorted(reasons, key=lambda r: (
        _GROUP.get((r.kind, r.severity), 5), r.code,
        r.log_offset if r.log_offset is not None else -1)))
