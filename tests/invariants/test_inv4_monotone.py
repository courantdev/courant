"""Invariant 4: the verdict is monotone in every check's severity (TRD VER-5.x, VER-6).

Raising the severity of any one finding (info < warn < critical) or any one physics check
(ok < minor < major < critical), or un-skipping a check, never raises the verdict
(accept > insufficient > reject) and never improves the grade (A > B > C > D). Three levels:

- `grade.compute_grade` on generated result lists;
- `policy.decide` on the events of every corpus run (and mutants of them), with generated
  findings and physics checks, every single-step raise of every one of them;
- the whole pipeline (specification and version rules included), with a finding and a check
  injected into what the checkers and the suite returned, at each severity in turn.
"""

from __future__ import annotations

import itertools
import random
from functools import cache

import pytest

from courant.diagnosis.findings import Finding, FindingSeverity
from courant.verification.grade import (
    TrustGrade,
    TrustReport,
    VerificationResult,
    VSeverity,
    compute_grade,
)
from courant.verify import from_log, policy, verify_inputs
from tests.invariants import gen

FSEV = list(FindingSeverity)                  # info, warn, critical
VSEV = list(VSeverity)                        # ok, minor, major, critical
#: Finding ids the generator draws: two real evidential ones and a reader limitation, which the
#: policy sets aside whatever its severity (policy.READER_LIMITATIONS).
FIDS = ("solver.loose_tolerance", "mesh.aspect_ratio", "case.unreadable_dict")


def _result(rng: random.Random, i: int) -> VerificationResult:
    return VerificationResult(check_code=f"check.{i}", check_name=f"check {i}",
                              severity=rng.choice(VSEV), passed=rng.random() < 0.5,
                              skipped=rng.random() < 0.2)


def _finding(rng: random.Random, i: int) -> Finding:
    fid = rng.choice(FIDS)
    return Finding(id=fid, checker=fid.split(".")[0], title=f"finding {i}",
                   severity=rng.choice(FSEV))


def _raises(results):
    """Every single-step raise of one result: its severity one rank up, or un-skipped."""
    for i, r in enumerate(results):
        if r.severity.rank < 3:
            up = VSEV[r.severity.rank + 1]
            yield f"result {i} {r.severity.value}->{up.value}", \
                [*results[:i], r.model_copy(update={"severity": up}), *results[i + 1:]]
        if r.skipped:
            yield f"result {i} unskipped", \
                [*results[:i], r.model_copy(update={"skipped": False}), *results[i + 1:]]


def _finding_raises(findings):
    for i, f in enumerate(findings):
        if f.severity.rank < 2:
            up = FSEV[f.severity.rank + 1]
            yield f"finding {i} {f.severity.value}->{up.value}", \
                [*findings[:i], f.model_copy(update={"severity": up}), *findings[i + 1:]]


N_GRADE_FAST, N_GRADE_SLOW = 5000, 100_000


def _grades(n: int) -> int:
    pairs = 0
    for k in range(n):
        rng = random.Random(f"inv4:grade:{k}")
        results = [_result(rng, i) for i in range(rng.randint(0, 8))]
        g = compute_grade(results)
        for what, raised in _raises(results):
            pairs += 1
            assert compute_grade(raised).quality <= g.quality, f"[{k}] {what}"
    return pairs


def test_the_grade_never_improves_when_a_check_gets_worse():
    assert _grades(N_GRADE_FAST) > 10_000


@pytest.mark.slow
def test_the_grade_never_improves_when_a_check_gets_worse_at_scale():
    _grades(N_GRADE_SLOW)


@cache
def _events(k: int):
    """The parsed events of log case k, as the pipeline parses them, and whether its end is
    unknown (VER-5.16)."""
    from courant.watcher.exitcode import verdict_exit_code
    from courant.watcher.parser import LogParser

    _, log, side = gen.log_case(k)
    inputs = from_log(log, side)
    exit_code = verdict_exit_code(inputs.log_text, None, "default")
    events = list(LogParser(run_id="x").parse_text(inputs.log_text, return_code=exit_code))
    return policy.recognised(inputs.log_text), tuple(events), exit_code is None


N_POLICY_FAST, N_POLICY_SLOW = 1500, 30_000
N_LOGS = 60                                    # the log cases whose events are drawn from


def _decide_rank(recognised, events, findings, results, truncated):
    trust = None if results is None else TrustReport.from_results("x", results)
    v, _ = policy.decide(recognised, list(events), findings, trust, truncated=truncated)
    return gen.RANK_V1[v]


def _policy(n: int) -> int:
    pairs, bad = 0, []
    for k in range(n):
        rng = random.Random(f"inv4:policy:{k}")
        recognised, events, truncated = _events(rng.randrange(N_LOGS))
        findings = [_finding(rng, i) for i in range(rng.randint(0, 5))]
        results = None if rng.random() < 0.1 else \
            [_result(rng, i) for i in range(rng.randint(0, 6))]
        base = _decide_rank(recognised, events, findings, results, truncated)
        for what, raised in _finding_raises(findings):
            pairs += 1
            if _decide_rank(recognised, events, raised, results, truncated) > base:
                bad.append(f"[{k}] {what}")
        for what, raised in _raises(results or []):
            pairs += 1
            if _decide_rank(recognised, events, findings, raised, truncated) > base:
                bad.append(f"[{k}] {what}")
    assert not bad, f"{len(bad)} of {pairs} raises raised the verdict: {bad[:5]}"
    return pairs


def test_the_policy_is_monotone_in_every_severity():
    assert _policy(N_POLICY_FAST) > 3000


@pytest.mark.slow
def test_the_policy_is_monotone_in_every_severity_at_scale():
    _policy(N_POLICY_SLOW)


# --- the whole pipeline, with one finding and one check injected ------------------------------

class _Inject:
    """Monkeypatches the checkers and the suite to add one finding and one physics check."""

    def __init__(self, monkeypatch):
        from courant.diagnosis import checkers
        from courant.verification import suite

        self.finding: FindingSeverity | None = None
        self.check: VSeverity | None = None
        real_run, real_verify = checkers.run_all_checkers, suite.verify

        def run_all(ctx):
            fs = real_run(ctx)
            if self.finding is None:
                return fs
            from courant.diagnosis.findings import FindingSet
            return FindingSet([*fs, Finding(id="solver.injected", checker="solver",
                                            title="injected", severity=self.finding)])

        def verify(ctx):
            report = real_verify(ctx)
            if self.check is None:
                return report
            extra = VerificationResult(check_code="injected", check_name="injected",
                                       severity=self.check, passed=self.check is VSeverity.OK)
            return TrustReport.from_results(report.run_id, [*report.results, extra])

        monkeypatch.setattr(checkers, "run_all_checkers", run_all)
        monkeypatch.setattr(suite, "verify", verify)


def _chain(inject: _Inject, inputs):
    """The verdict at each step of a chain of raises, from nothing injected to both critical."""
    out = []
    inject.finding = inject.check = None
    out.append(("none", verify_inputs(inputs)))
    for s in VSEV:
        inject.check = s
        out.append((f"check {s.value}", verify_inputs(inputs)))
    for s in FSEV:
        inject.finding = s
        out.append((f"check critical + finding {s.value}", verify_inputs(inputs)))
    return out


N_PIPE_FAST, N_PIPE_SLOW = 8, 500


def _pipeline(n: int, monkeypatch):
    inject = _Inject(monkeypatch)
    bad, steps = [], 0
    for k in range(n):
        cid, log, side = gen.log_case(k)
        inputs = from_log(log, side)
        chain = _chain(inject, inputs)
        for (w0, a), (w1, b) in itertools.pairwise(chain):
            steps += 1
            ga = TrustGrade(a.grade).quality if a.grade else None
            gb = TrustGrade(b.grade).quality if b.grade else None
            if gen.RANK[b.verdict] > gen.RANK[a.verdict] or \
                    (ga is not None and gb is not None and gb > ga):
                bad.append(f"{cid}: {w0} {a.verdict}/{a.grade} -> {w1} {b.verdict}/{b.grade}")
    assert not bad, f"{len(bad)} of {steps} steps raised a verdict or grade: {bad[:5]}"


def test_the_pipeline_is_monotone_in_an_injected_severity(monkeypatch):
    _pipeline(N_PIPE_FAST, monkeypatch)


@pytest.mark.slow
def test_the_pipeline_is_monotone_in_an_injected_severity_at_scale(monkeypatch):
    _pipeline(N_PIPE_SLOW, monkeypatch)
