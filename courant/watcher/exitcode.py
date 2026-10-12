"""Whether a run finished, read from its log (one definition, TRD VER-3).

OpenFOAM prints a bare `End` as the last line of a solver that ran to
completion. Everything else exited abnormally.

This was a private helper in the replay tier. The verifier needs the same
answer, and two copies of one fact is the bug class this codebase has fixed
seven times, so it lives here and both import it.

A converged steady run, a plateau that ran to its iteration limit, and a
transient run that reached its end time all print `End`. Calling a plateau a
crash made `crash.*` signatures reachable and pushed `BC.REVERSED_OUTFLOW`
into two diagnoses the real pipeline gets right.
"""

from __future__ import annotations

_END_MARKER = "\nEnd\n"


def infer_exit_code(log_text: str) -> int:
    """0 if the log shows a completed solver run, else 1."""
    return 0 if log_text.rstrip().endswith("End") or _END_MARKER in log_text else 1
