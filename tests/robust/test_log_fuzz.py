"""TST-9 at scale: at least 20,000 deterministic mutants of every committed real solver log.

Two properties, on every mutant:
1. the verifier never raises anything but `InputError`, and its verdict serialises and renders;
2. it never accepts (`accept`, courant.verdict/2) a log whose residuals or time markers were
   destroyed (`logfuzz.destroyed`).

The default run takes a fixed sample of about 500 mutants (every 40th of the full plan, so the
sample is a subset of the full run); `-m slow` runs all of them. The deterministic destroyers
(`logfuzz.DESTROYERS`) run on every log in both modes.

Known failures would be strict xfails listed here, so each flips to a failure the day it is fixed.
There are none: the 56 VER-5.5 cases found on 10 Oct 2026 were fixed by T-VER55, and the 11
logs accepted with every time marker removed are `insufficient` under VER-5.15 (T-VER-TIME,
regression tests in `tests/unit/test_ver_time.py`).
CPU class: full. The default run is about a minute on one core; the full run about half an hour.
"""

from __future__ import annotations

import pytest

from courant.verify import InputError, from_path, render_text, verify_inputs
from tests.robust import logfuzz

LOGS = logfuzz.corpus()


def _check(path):
    """The verdict, or 'input_error'. Anything else raised is the failure TST-9 forbids."""
    try:
        inputs = from_path(path)
    except InputError:
        return "input_error"
    v = verify_inputs(inputs)
    v.to_json()
    render_text(v)
    return v.verdict


def _run(tmp_path, cases):
    """Every (log, k) mutant through the verifier; the failures, as one list of lines."""
    failures = []
    p = tmp_path / "mutant.log"
    for rel, k in cases:
        mid, data = logfuzz.mutant(rel, k)
        p.write_bytes(data)
        try:
            verdict = _check(p)
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f"{mid}: raised {type(exc).__name__}: {str(exc)[:120]}")
            continue
        why = logfuzz.destroyed(data)
        if why and verdict == "accept":
            failures.append(f"{mid}: accepted although {why}")
    return failures


def _by_log(cases):
    out: dict[str, list] = {rel: [] for rel in LOGS}
    for rel, k in cases:
        out[rel].append((rel, k))
    return out


SAMPLE = _by_log(logfuzz.sample())


def test_the_plan_is_at_least_20000_mutants_over_every_solver_log():
    plan = logfuzz.plan()
    assert len(plan) >= 20_000
    assert {rel for rel, _ in plan} == set(LOGS)
    assert len(LOGS) >= 40, LOGS
    assert not any(logfuzz.NOT_SOLVER.search(rel) for rel in LOGS)
    sample = logfuzz.sample()
    assert 450 <= len(sample) <= 600
    assert set(sample) <= set(plan)


def test_mutants_are_deterministic():
    rel = LOGS[0]
    for k in (0, 7, 199):
        assert logfuzz.mutant(rel, k) == logfuzz.mutant(rel, k)
    assert logfuzz.mutant(rel, 1)[1] != logfuzz.mutant(rel, 2)[1] or \
        logfuzz.mutant(rel, 1)[0] != logfuzz.mutant(rel, 2)[0]


def test_every_operator_is_used_by_the_sample_and_on_every_log_by_the_full_plan():
    assert {n for rel, k in logfuzz.sample() for n in logfuzz.operators(rel, k)[1]} \
        == set(logfuzz.RANDOM_OPS)
    for rel in LOGS:
        used = {n for k in range(logfuzz.counts()[rel]) for n in logfuzz.operators(rel, k)[1]}
        assert used == set(logfuzz.RANDOM_OPS), rel


@pytest.mark.parametrize("rel", LOGS)
def test_an_untouched_trustworthy_log_is_not_called_destroyed(tmp_path, rel):
    """The predicate's sanity: every accepted original keeps its residuals and time markers."""
    data = logfuzz.source(rel)
    p = tmp_path / "orig.log"
    p.write_bytes(data)
    if _check(p) == "accept":
        assert logfuzz.destroyed(data) is None


@pytest.mark.parametrize("rel", LOGS)
def test_sample_of_mutants_never_raises_or_accepts_a_destroyed_log(tmp_path, rel):
    failures = _run(tmp_path, SAMPLE[rel])
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.parametrize("rel", LOGS)
def test_all_mutants_never_raise_or_accept_a_destroyed_log(tmp_path, rel):
    failures = _run(tmp_path, [(rel, k) for k in range(logfuzz.counts()[rel])])
    assert not failures, "\n".join(failures)


def _destroyer_cases():
    for rel in LOGS:
        for name in sorted(logfuzz.DESTROYERS):
            yield pytest.param(rel, name, id=f"{rel}::{name}")


@pytest.mark.parametrize("rel,name", list(_destroyer_cases()))
def test_a_destroyed_log_is_never_accepted(tmp_path, rel, name):
    data = logfuzz.DESTROYERS[name](logfuzz.source(rel))
    p = tmp_path / "destroyed.log"
    p.write_bytes(data)
    verdict = _check(p)
    assert verdict in ("accept", "reject", "insufficient", "input_error")
    if logfuzz.destroyed(data):
        assert verdict != "accept", f"{rel} after {name}: {logfuzz.destroyed(data)}"
