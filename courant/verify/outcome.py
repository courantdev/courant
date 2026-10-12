"""What a verdict means for a check that must pass or fail (TRD GATE-7, ARC-13).

The gate decides a verdict; a CI check, the Action, needs one more answer: does this run fail
the check. That answer is a reading of the verdict, so it lives beside the verdict's own code
(ARC-13: other modules read a verdict through helpers here, and decide nothing).

It is a policy, not an outcome. An agent run's outcome (SB-11: `valid_conformant`,
`nonconformant`, ..., with `silent_error` and `unverified_success`) has one definition,
`courant.verify.schema.outcome_of`; nothing here names one. A CI run has no producer's claim and
no re-solve, and the check passes an `accept` judged without a specification, which SB-11 scores
`insufficient`; so the check reads the verdict, as GATE-7 says, and not the outcome.

`fail_on`:

- `not-accept` (the default): anything but `accept` fails the check;
- `reject`: `insufficient` passes, and the summary says so on every run it applies to.

Whatever `fail_on` says, a run whose verdict is void because of the specification fails the
check: an `insufficient` that means "the specification was changed after it was locked" is
not the same as one that means "the log is too short to judge", and a check that let the first
through would let a specification be edited until the run fits it.
"""

from __future__ import annotations

from collections.abc import Iterable

from courant.verify import policy

FAIL_ON = ("not-accept", "reject")

#: The specification problems GATE-4 names that are not a hash mismatch: VER-5.9 (a named value
#: could not be read), VER-5.11 (invalid), VER-5.12 (an extracted specification does not match
#: the confirmed one) and VER-5.13 (a role the case does not resolve). A hash mismatch (VER-5.8)
#: is `policy.spec_void`, the one definition of it.
SPEC_INTEGRITY_RULES = frozenset({"VER-5.9", "VER-5.11", "VER-5.12", "VER-5.13"})
#: The closed vocabulary's names (GATE-4) for the same four.
SPEC_INTEGRITY_CODES = frozenset({
    "spec_value_unreadable", "spec_invalid", "spec_extracted_mismatch", "spec_role_unresolved"})


def spec_integrity(verdict) -> bool:
    """Whether the verdict turned on the specification's integrity, not on the run."""
    if verdict.conformance is not None and policy.spec_void(verdict.conformance):
        return True
    if verdict.rule in SPEC_INTEGRITY_RULES:
        return True
    return any(r.code.removeprefix("insufficient.") in SPEC_INTEGRITY_CODES
               for r in verdict.reasons)


def check_fails(verdict, fail_on: str = "not-accept") -> bool:
    """Whether `verdict` fails a check set to `fail_on`."""
    if fail_on not in FAIL_ON:
        raise ValueError(f"fail-on must be one of {', '.join(FAIL_ON)}, not {fail_on!r}")
    if spec_integrity(verdict):
        return True
    if verdict.verdict == "accept":
        return False
    if verdict.verdict == "reject":
        return True
    return fail_on == "not-accept"                       # insufficient


def first_reason(verdict) -> tuple[str, str] | None:
    """(code, sentence) of the verdict's first reason, or None when it gives none."""
    return (verdict.reasons[0].code, verdict.reasons[0].summary) if verdict.reasons else None


def tally(verdicts: Iterable) -> dict[str, int]:
    out = {"accept": 0, "reject": 0, "insufficient": 0}
    for v in verdicts:
        out[v.verdict] += 1
    return out
