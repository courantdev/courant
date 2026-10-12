"""The Courant Test scorer (TRD CT-7). Hash-locked (DEV-9).

Every rate is returned as its numerator and denominator with a Wilson 95%
interval. "0 false alarms" over 7 healthy runs and over 700 are different
claims, and the scorecard never lets them look the same.

Definitions (RRD RM-01):
- far      false accepts: faulted tasks judged TRUSTWORTHY, over faulted tasks
           that got a definite verdict
- frr      false alarms: healthy tasks judged NOT_TRUSTWORTHY, over healthy
           tasks that got a definite verdict
- abstain  NOT_ASSESSABLE, over all tasks
- accuracy correct verdicts, over all tasks (an abstention is not correct)
- top2     faulted tasks with expects_code among the first two codes, over
           faulted tasks that have an expects_code
"""

from __future__ import annotations

import math

from courant.ctest.schema import JudgeAnswer, Label, Task
from courant.ctest.splits import split_of

Z = 1.959963984540054


def wilson(n: int, of: int) -> tuple[float | None, float | None]:
    if of == 0:
        return None, None
    p = n / of
    denom = 1 + Z * Z / of
    centre = (p + Z * Z / (2 * of)) / denom
    half = Z * math.sqrt(p * (1 - p) / of + Z * Z / (4 * of * of)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def _rate(n: int, of: int) -> dict:
    lo, hi = wilson(n, of)
    return {"n": n, "of": of, "lo": lo, "hi": hi}


def _card(pairs: list[tuple[Task, Label, JudgeAnswer]]) -> dict:
    faulted = [(t, l, a) for t, l, a in pairs if l.truth == "NOT_TRUSTWORTHY"]
    healthy = [(t, l, a) for t, l, a in pairs if l.truth == "TRUSTWORTHY"]
    f_def = [x for x in faulted if x[2].verdict != "NOT_ASSESSABLE"]
    h_def = [x for x in healthy if x[2].verdict != "NOT_ASSESSABLE"]
    diag = [x for x in faulted if x[1].expects_code]
    return {
        "tasks": len(pairs),
        "accuracy": _rate(sum(1 for _, l, a in pairs if a.verdict == l.truth), len(pairs)),
        "far": _rate(sum(1 for _, _, a in f_def if a.verdict == "TRUSTWORTHY"), len(f_def)),
        "frr": _rate(sum(1 for _, _, a in h_def if a.verdict == "NOT_TRUSTWORTHY"), len(h_def)),
        "abstain": _rate(sum(1 for _, _, a in pairs if a.verdict == "NOT_ASSESSABLE"), len(pairs)),
        "top2": _rate(sum(1 for _, l, a in diag if l.expects_code in a.codes[:2]), len(diag)),
        "errors": sum(1 for _, _, a in pairs if a.error),
    }


def score(tasks: list[Task], labels: dict[str, Label], answers: list[JudgeAnswer]) -> dict:
    by_id = {a.task_id: a for a in answers}
    missing = [t.task_id for t in tasks if t.task_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} tasks have no answer, e.g. {missing[0]}")
    pairs = [(t, labels[t.task_id], by_id[t.task_id]) for t in tasks]
    card = {"all": _card(pairs)}
    for split in ("held_in", "held_out"):
        card[split] = _card([p for p in pairs if split_of(p[0].family) == split])
    families = sorted({t.family for t in tasks})
    card["by_family"] = {f: _card([p for p in pairs if p[0].family == f]) for f in families}
    return card


def misses(tasks: list[Task], labels: dict[str, Label], answers: list[JudgeAnswer]) -> list[dict]:
    """What the judge got wrong, for weakness mining (DEV-10). Labels included: loop-side only."""
    by_id = {a.task_id: a for a in answers}
    out = []
    for t in tasks:
        l, a = labels[t.task_id], by_id[t.task_id]
        kind = None
        if a.verdict == "NOT_ASSESSABLE":
            kind = "abstain"
        elif a.verdict != l.truth:
            kind = "false_accept" if l.truth == "NOT_TRUSTWORTHY" else "false_alarm"
        elif l.expects_code and l.expects_code not in a.codes[:2]:
            kind = "top2_miss"
        if kind:
            out.append({"task_id": t.task_id, "family": t.family, "split": split_of(t.family),
                        "kind": kind, "judge_rationale": a.rationale,
                        "hint": f"expected {l.truth}" + (f", cause {l.expects_code}" if l.expects_code else "")})
    return out
