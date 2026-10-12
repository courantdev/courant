"""Checks behind `--checks next` (TRD GATE-13).

A new check ships here first: its findings are left out of every verdict unless the run is
verified with `--checks next`, and a verdict that ran them says so (`tool.checks: "next"`). After
one minor release it is promoted: its id leaves `NEXT` and it is on by default, and the release's
changelog counts the verdicts it turned on the corpus (`scripts/release.py`, GATE-13).

`NEXT` maps a finding id (`courant.diagnosis.checkers`) to the version it shipped in. Today no
check is behind the flag.
"""

from __future__ import annotations

MODES = ("default", "next")

#: {finding id: the version it shipped in behind `--checks next`}.
NEXT: dict[str, str] = {}


def enabled(finding_id: str, mode: str = "default") -> bool:
    """Whether a finding enters the verdict under this mode."""
    if mode not in MODES:
        raise ValueError(f"--checks is one of {', '.join(MODES)}, not {mode!r}")
    return mode == "next" or finding_id not in NEXT


def only_enabled(finding_set, mode: str = "default"):
    """The findings that enter the verdict under this mode, in their order."""
    from courant.diagnosis.findings import FindingSet

    if mode == "next" or not NEXT:
        enabled("", mode)                     # still refuse an unknown mode
        return finding_set
    return FindingSet([f for f in finding_set if enabled(f.id, mode)])


__all__ = ["MODES", "NEXT", "enabled", "only_enabled"]
