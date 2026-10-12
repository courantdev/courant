"""What broke when Courant met three solver families it had never seen.

The certification matrix covered four incompressible tutorial cases. Every bug
that pipeline has ever found was of the form *our model of what OpenFOAM prints
was wrong* — so covering a quarter of OpenFOAM meant a quarter of the model had
been checked.

Compressible (`rhoPimpleFoam`), multiphase VOF (`interFoam`) and buoyant
(`buoyantPimpleFoam`) were run for the first time. Six bugs, all of them
invisible to the incompressible corpus, two of them false positives on cases the
OpenFOAM project ships as correct. Each test below is pinned to the real log
that demonstrated it.
"""

from __future__ import annotations

import pathlib

import pytest

from courant.diagnosis.checkers import CheckContext, run_all_checkers
from courant.diagnosis.findings import FindingSeverity
from courant.watcher.parser import LogParser

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "real_openfoam"


def _analyse(name: str, **kw):
    text = (FIXTURES / name).read_text()
    events = list(LogParser(run_id="r").parse_text(text, return_code=0))
    findings = run_all_checkers(
        CheckContext(run_id="r", events=events, log_text=text, **kw))
    return text, events, {f.id: f for f in findings}


class TestMultiphaseFieldNames:
    """`alpha.water` is the field that *is* the free surface, and the parser
    could not see it.

    `Solving for (\\w+),` — and `\\w` excludes `.`. Every multiphase and
    multi-region solver OpenFOAM ships names fields with dots.
    """

    def test_the_phase_fraction_is_parsed(self):
        _text, events, _ = _analyse("interFoam_damBreak_healthy.log")
        fields = {k for e in events if e.kind.value == "RESIDUAL"
                  for k in e.payload.initial_residuals}

        assert "alpha.water" in fields, (
            "the phase fraction was invisible, so the run was judged converged "
            "on the pressure residual alone")

    def test_the_ordinary_fields_still_parse(self):
        _text, events, _ = _analyse("interFoam_damBreak_healthy.log")
        fields = {k for e in events if e.kind.value == "RESIDUAL"
                  for k in e.payload.initial_residuals}

        assert {"p_rgh", "pcorr"} <= fields


class TestTheInterfaceCourantNumberIsNotTheCourantNumber:
    """Multiphase solvers print two Courant numbers per step and the pattern
    matched both by substring, so the second silently overwrote the first.

    It fails in the dangerous direction: the interface number stays low while
    the bulk number is the one that breaks a solve.
    """

    def test_the_bulk_number_is_the_one_reported(self):
        import re

        text, events, _ = _analyse("interFoam_damBreak_healthy.log")
        parsed = max(e.payload.courant_max for e in events
                     if e.kind.value == "RESIDUAL"
                     and e.payload.courant_max is not None)
        bulk = max(float(m.group(1)) for m in
                   re.finditer(r"^Courant Number mean: \S+ max: (\S+)", text, re.M))
        interface = max(float(m.group(1)) for m in re.finditer(
            r"^Interface Courant Number mean: \S+ max: (\S+)", text, re.M))

        assert parsed == pytest.approx(bulk)
        assert parsed != pytest.approx(interface)


class TestAStartupTransientIsNotAStabilityFailure:
    """The most expensive class of bug this product has: a CRITICAL finding on a
    case the OpenFOAM project ships as correct.

    `buoyantPimpleFoam/hotRoom` relaxes a 600 K floor patch into a 300 K room.
    The Courant number spikes to 25.7 on the second timestep, decays
    monotonically, and sits at 0.63 for the remaining 990 steps while the
    residuals converge. Scored on the bare maximum that was CRITICAL.
    """

    def test_a_transient_that_recovered_is_not_critical(self):
        _text, _events, found = _analyse(
            "buoyantPimpleFoam_hotRoom_healthy.log")

        assert found["cfl.max"].severity is FindingSeverity.INFO
        assert "settled back" in found["cfl.max"].detail

    def test_it_stops_pointing_at_a_cause(self):
        """A transient that recovered explains nothing about a run that then
        went wrong for another reason."""
        _text, _events, found = _analyse(
            "buoyantPimpleFoam_hotRoom_healthy.log")

        assert found["cfl.max"].suggests == []

    def test_a_real_blow_up_is_still_critical(self):
        """The guard rail. This checker is what the whole product rests on and
        the fix must not soften it."""
        _text, _events, found = _analyse("cavity_cfl_diverged.log")

        assert found["cfl.max"].severity is FindingSeverity.CRITICAL
        assert found["cfl.max"].suggests == ["NUM.CFL_VIOLATION"]

    def test_a_healthy_incompressible_case_still_says_nothing(self):
        _text, _events, found = _analyse("cavity_converged.log")

        assert "cfl.max" not in found


class TestCompressibleStartupIsNotACHTFault:
    """`thermophysical` matched `Reading thermophysical properties` — the
    startup banner of every compressible and buoyant solver — so PHYS.CHT_THERMO
    scored on every healthy run of two solver families, on cases with no
    conjugate heat transfer anywhere in them.

    Third instance of the same shape, after the `trapFpe` banner and the
    `reverse flow` function-object announcement.
    """

    @pytest.mark.parametrize("line", [
        "Reading thermophysical properties",
        "Selecting thermodynamics package {type hePsiThermo; mixture pureMixture;}",
    ])
    def test_routine_startup_lines_do_not_match(self, line):
        from courant.diagnosis.kb_data import build_default_kb

        kb = build_default_kb()
        matches = {m.code for m in kb.query(log_text=line)}

        assert "PHYS.CHT_THERMO" not in matches

    def test_a_genuine_inconsistency_still_matches(self):
        from courant.diagnosis.kb_data import build_default_kb

        matches = {m.code for m in build_default_kb().query(
            log_text="--> FOAM FATAL ERROR: inconsistent thermo between regions")}

        assert "PHYS.CHT_THERMO" in matches


class TestAPhaseFractionCannotDiverge:
    """`alpha.water` is bounded in [0, 1] by construction — MULES enforces it
    every step — so its residual does not measure convergence. It measures how
    fast the interface is *moving*.

    On `interFoam/damBreak` the interface accelerates down a collapsing column
    of water, which is the entire point of the case. Fitting a divergence slope
    to that reported DIVERGENCE on an unmodified tutorial, with `is_nan=False`
    and a residual of 1.3e-03, which `classify_status` turns into
    `RunStatus.DIVERGED` — a healthy multiphase run declared a failure.

    Only reachable once the parser learned to read dotted field names. Two bugs
    deep in the same log.
    """

    def test_the_healthy_dam_break_does_not_diverge(self):
        from courant.watcher import Watcher

        text = (FIXTURES / "interFoam_damBreak_healthy.log").read_text()
        events = list(Watcher("r").feed(text))

        assert [e for e in events if e.kind.value == "DIVERGENCE"] == []

    def test_the_phase_fraction_is_still_recorded(self):
        """Excluded from the *divergence fit*, not from the run. It is still
        parsed, still in the residual series, and still available to anything
        that wants to reason about it."""
        _text, events, _ = _analyse("interFoam_damBreak_healthy.log")

        assert any("alpha.water" in e.payload.initial_residuals
                   for e in events if e.kind.value == "RESIDUAL")

    def test_an_ordinary_field_still_diverges(self):
        """The guard rail: this must not blunt divergence detection generally.

        Asserted on a monotonic blow-up of `Ux` rather than on the real diverged
        cavity, because that log **crashes** (`sigFpe`) before the soft-slope
        detector ever fires — 17 residual steps and one CRASH, no DIVERGENCE
        event at all. Writing the guard against the fixture would have asserted
        something that was never true of it.
        """
        from courant.events import Event, EventKind, ResidualPayload
        from courant.watcher.anomaly import AnomalyDetector

        detector = AnomalyDetector("r")
        emitted = []
        for i in range(8):
            emitted += detector.observe(Event(
                run_id="r", seq=i, kind=EventKind.RESIDUAL,
                payload=ResidualPayload(time=float(i),
                                        initial_residuals={"Ux": 1e-3 * 10 ** i})))

        assert [e for e in emitted if e.kind is EventKind.DIVERGENCE]


class TestAVOFRunIsNotJudgedOnItsInterface:
    """`alpha.water`'s initial residual measures how far the interface *moved*,
    not how far the solution is from satisfying its equations — and on a
    transient free-surface run the interface never stops moving.

    Measured on an unmodified `interFoam/damBreak`: `p_rgh` settles to 5.29e-04
    while `alpha.water` sits at 1.08e-02, so the convergence verdict was
    negative on the one field that was never going to settle. Every customer
    running VOF would have been told their correct run had failed.
    """

    def test_the_healthy_dam_break_is_judged_converged(self):
        from courant.verification.convergence import solve_converged

        _text, events, _ = _analyse("interFoam_damBreak_healthy.log")
        converged, why = solve_converged(events)

        assert converged, why
        assert "5.29e-04" in why, "the verdict should come from the pressure"

    def test_an_incompressible_verdict_is_unchanged(self):
        from courant.verification.convergence import solve_converged

        _text, events, _ = _analyse("cavity_converged.log")

        assert solve_converged(events)[0]

    def test_a_real_blow_up_is_still_not_converged(self):
        from courant.verification.convergence import solve_converged

        _text, events, _ = _analyse("cavity_cfl_diverged.log")

        assert not solve_converged(events)[0]
