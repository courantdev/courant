"""The verification suite — run every check, compute the Trust Grade (Tech Arch §7)."""

from __future__ import annotations

from typing import Protocol

from courant.case.config import CaseConfig
from courant.verification.checks import VerifyContext, run_checks
from courant.verification.fields import FieldData
from courant.verification.grade import TrustReport


class CompletedRun(Protocol):
    """What `verify_run` reads of a completed run: its id and its events (a sandbox
    `RunResult` is one; the gate does not depend on the sandbox that produces it)."""

    run_id: str
    events: list


def verify(ctx: VerifyContext) -> TrustReport:
    return TrustReport.from_results(ctx.run_id, run_checks(ctx))


def verify_run(result: CompletedRun, config: CaseConfig | None = None,
               fields: FieldData | None = None) -> TrustReport:
    """Grade a completed run from its events (+ optional config/solution fields)."""
    ctx = VerifyContext(run_id=result.run_id, events=result.events,
                        config=config, fields=fields)
    return verify(ctx)
