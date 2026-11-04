"""One definition of "no time markers" (TRD VER-5.15).

Field residuals were read, but no step carries a step number, because no `Time =` line was
read. The log cannot show which step the run reached, or that its time loop ran to the end, so
its residuals are not a history (T-VER-TIME). The verifier's rule VER-5.15
(`courant.verify.policy`) and the first-solve reading (`courant.verification.convergence.
reads_first_solve`, VER-16a) both ask it, so the two never disagree on which log has no steps.
"""

from __future__ import annotations

from courant.events.core import EventKind

__all__ = ["no_time_markers"]


def no_time_markers(events) -> bool:
    """True when residuals were read and none of them carries a step number."""
    steps = [e.payload for e in events
             if e.kind is EventKind.RESIDUAL and getattr(e, "payload", None) is not None]
    return (any(p.initial_residuals for p in steps)
            and all(p.iteration is None for p in steps))
