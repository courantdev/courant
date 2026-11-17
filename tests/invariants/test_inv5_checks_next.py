"""Invariant 5: `--checks next` never changes a verdict 0.7.0's default set gave unless one of
the NEXT checks fired (`courant.verify.checks`, GATE-13), and a verdict says which set ran.

"Fired" is made exact by ablation: run `next` with every name in `checks.NEXT` switched off
(`enabled` answers as under the default for exactly those names, and `only_enabled` filters as
the default does). If no code path decides on the mode string itself, that is the default's
document byte for byte, but for `tool.checks`; so every difference `next` makes is a NEXT name
whose behaviour ran. Where `next` moves a verdict, each name is then switched on alone, and the
names that move it on their own are reported (`test_attribution_*`, for the findings note).

The stream: log cases (`gen.log_case`, mutated corpus runs) and the fixture case directories,
which `from_case_dir` reads in the mode too.
"""

from __future__ import annotations

import json

import pytest

from courant.verify import checks, from_case_dir, from_log, verify_inputs
from tests.invariants import gen


def _ablate(m, keep: frozenset = frozenset()):
    """Under `next`, every NEXT name off but `keep`."""
    real = checks.enabled

    def enabled(name, mode="default"):
        real(name, mode)                                  # refuse an unknown mode, as before
        if name in checks.NEXT:
            return mode == "next" and name in keep
        return True

    def only_enabled(finding_set, mode="default"):
        from courant.diagnosis.findings import FindingSet

        enabled("", mode)
        return FindingSet([f for f in finding_set if enabled(f.id, mode)])

    m.setattr(checks, "enabled", enabled)
    m.setattr(checks, "only_enabled", only_enabled)


def _doc(v) -> dict:
    d = json.loads(v.to_json())
    d["tool"].pop("checks", None)
    return d


def _inputs(source, mode: str):
    kind, a, b = source
    if kind == "log":
        inp = from_log(a, b)
        return inp if mode == "default" else inp.model_copy(update={"checks": mode})
    return from_case_dir(a, checks=mode)


def _sources(n_logs: int, n_dirs: int):
    for k in range(n_logs):
        cid, log, side = gen.log_case(k)
        yield cid, ("log", log, side)
    for rel in gen.CASE_DIRS[:n_dirs]:
        yield rel, ("dir", gen.pristine(rel), None)


def _run(n_logs: int, n_dirs: int, monkeypatch):
    bad, moved, n = [], [], 0
    for cid, src in _sources(n_logs, n_dirs):
        n += 1
        d = verify_inputs(_inputs(src, "default"))
        x = verify_inputs(_inputs(src, "next"))
        if d.tool.checks is not None or x.tool.checks != "next":
            bad.append(f"{cid}: tool.checks {d.tool.checks!r} / {x.tool.checks!r}")
        with monkeypatch.context() as m:
            _ablate(m)
            a = verify_inputs(_inputs(src, "next"))
        if _doc(a) != _doc(d):
            diff = sorted(k for k in _doc(d) if _doc(d).get(k) != _doc(a).get(k))
            bad.append(f"{cid}: next with every NEXT name off differs from the default in "
                       f"{diff} ({d.verdict} / {a.verdict})")
        if x.verdict != d.verdict:
            moved.append((cid, src, d.verdict, x.verdict))
    assert not bad, f"{len(bad)} of {n}: {bad[:5]}"
    return n, moved


N_LOG_FAST, N_LOG_SLOW = 40, 2000


def test_next_differs_from_the_default_only_through_next_names(monkeypatch):
    _run(N_LOG_FAST, len(gen.CASE_DIRS), monkeypatch)


@pytest.mark.slow
def test_next_differs_from_the_default_only_through_next_names_at_scale(monkeypatch):
    _run(N_LOG_SLOW, len(gen.CASE_DIRS), monkeypatch)


def test_attribution_of_every_verdict_next_moves(monkeypatch, capsys):
    """Each fixture case whose verdict `next` moves, and the NEXT names that move it alone."""
    _, moved = _run(0, len(gen.CASE_DIRS), monkeypatch)
    lines = []
    for cid, src, before, after in moved:
        alone = []
        for name in sorted(checks.NEXT):
            with monkeypatch.context() as m:
                _ablate(m, frozenset({name}))
                if verify_inputs(_inputs(src, "next")).verdict != before:
                    alone.append(name)
        lines.append(f"{cid}: {before} -> {after}; alone: {alone or 'none (only together)'}")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    assert moved, "no fixture case moves under next: the attribution has nothing to say"
