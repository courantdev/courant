"""Invariant 2: a specification can only lower a verdict or leave it, never raise it (PRD 2.0).

Order: accept > insufficient > reject (`gen.RANK`). Two levels:

- the policy, exhaustively: every (verdict, rule) `policy.decide` can return, through
  `apply_conformance` with every combination of what a conformance block can say (lock absent,
  matching or not; impossible; a mismatch; an unreadable item; extracted or not), then
  `apply_version` with each answer. Thousands of combinations, all of them;
- the pipeline: a fixture case verified with no specification and with a generated one
  (`gen.spec_for`: the case's own draft, kept, perturbed, unreadable or impossible, with a lock
  that matches, does not, or is absent), under both `--checks` modes.
"""

from __future__ import annotations

import itertools
from functools import cache

import pytest

from courant.verify import from_case_dir, policy, verify_inputs, with_spec
from courant.verify.schema import ConformanceBlock
from tests.invariants import gen

#: Every (verdict, rule) `policy.decide` returns (policy.py).
DECIDED = (("TRUSTWORTHY", "VER-5.6"),
           ("NOT_TRUSTWORTHY", "VER-5.2"), ("NOT_TRUSTWORTHY", "VER-5.3"),
           ("NOT_TRUSTWORTHY", "VER-5.4"),
           ("NOT_ASSESSABLE", "VER-5.1"), ("NOT_ASSESSABLE", "VER-5.5"),
           ("NOT_ASSESSABLE", "VER-5.15"), ("NOT_ASSESSABLE", "VER-5.16"))

SHA, OTHER = "a" * 64, "b" * 64


def blocks():
    """Every combination of what a conformance block says that the policy reads."""
    for locked, invalid, nonconf, unread in itertools.product(
            (None, SHA, OTHER), ((), ("min above max",)), (0, 1, 3), (0, 1, 2)):
        yield ConformanceBlock(spec_schema="courant.spec/1", spec_sha256=SHA,
                               locked_sha256=locked, items=(), checked=nonconf + 1,
                               nonconformant=nonconf, unreadable=unread, invalid=invalid)


def policy_cases():
    for (verdict, rule), block, extracted, measured in itertools.product(
            DECIDED, list(blocks()), (False, True), (None, True, False)):
        yield verdict, rule, block, extracted, measured


N_POLICY = sum(1 for _ in policy_cases())


def test_the_policy_never_raises_a_verdict_on_a_specification():
    bad = []
    n = 0
    for verdict, rule, block, extracted, measured in policy_cases():
        n += 1
        v1, r1 = policy.apply_conformance(verdict, rule, block, extracted=extracted)
        v2, r2 = policy.apply_version(v1, r1, measured)
        v0, _ = policy.apply_version(verdict, rule, measured)
        if gen.RANK_V1[v1] > gen.RANK_V1[verdict] or gen.RANK_V1[v2] > gen.RANK_V1[v0]:
            bad.append((verdict, rule, block.locked_sha256, block.invalid, block.nonconformant,
                        block.unreadable, extracted, measured, v2, r2))
    assert n == N_POLICY >= 2000
    assert not bad, f"{len(bad)} of {n} raised: {bad[:5]}"


@cache
def _bare(rel: str, checks: str) -> str:
    return verify_inputs(from_case_dir(gen.pristine(rel), checks=checks)).verdict


def _with_spec(k: int, checks: str, provenance=type):
    rel, case = gen.dir_case(k)
    what, data = gen.spec_for(k, rel, checks, provenance=provenance)
    path, sha = gen.write_spec(data, k)
    v = verify_inputs(with_spec(from_case_dir(case, checks=checks), path, gen.lock_for(k, sha)))
    return rel, what, v


N_FAST, N_SLOW = 80, 3000


def _run(n: int):
    bad, moved = [], 0
    for k in range(n):
        checks = "next" if k % 2 else "default"
        rel, what, v = _with_spec(k, checks)
        before = _bare(rel, checks)
        moved += v.verdict != before
        if gen.RANK[v.verdict] > gen.RANK[before]:
            bad.append(f"[{k}] {rel} {checks} {what}: {before} -> {v.verdict} ({v.rule})")
    assert not bad, f"{len(bad)} of {n} specifications raised a verdict: {bad[:5]}"
    assert moved, "no generated specification moved a verdict: the generator is too gentle"


def test_a_specification_never_raises_a_verdict():
    _run(N_FAST)


@pytest.mark.slow
def test_a_specification_never_raises_a_verdict_at_scale():
    _run(N_SLOW)
