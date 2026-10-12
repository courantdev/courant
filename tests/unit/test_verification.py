"""Unit tests for the verification suite and its checks."""

import numpy as np

from courant.case.config import CaseConfig
from courant.events import (
    ContinuityErrors,
    CrashPayload,
    DivergencePayload,
    Event,
    EventKind,
    ResidualPayload,
    Severity,
    WarnPayload,
)
from courant.verification import FieldData, TrustGrade, VerifyContext, verify
from courant.verification.checks import (
    MassFluxCheck,
    PlausibilityCheck,
    RegimeCheck,
    YPlusCheck,
)
from courant.verification.grade import VSeverity


def residual(seq, res, cont=None, courant=None):
    return Event(run_id="r", seq=seq, kind=EventKind.RESIDUAL,
                 payload=ResidualPayload(iteration=seq + 1, initial_residuals=res,
                                         continuity=cont, courant_max=courant))


def warn(seq, field):
    return Event(run_id="r", seq=seq, kind=EventKind.WARN, severity=Severity.WARN,
                 payload=WarnPayload(category="bounding", field=field))


def _ctx(events, **kw):
    return VerifyContext(run_id="r", events=events, **kw)


class TestEventGrading:
    def test_healthy_run_is_grade_A(self):
        events = [
            residual(0, {"p": 0.1, "Ux": 0.05}, ContinuityErrors(cumulative=1e-9)),
            residual(1, {"p": 1e-4, "Ux": 5e-5}, ContinuityErrors(cumulative=1e-9)),
        ]
        report = verify(_ctx(events))
        assert report.grade is TrustGrade.A
        assert not report.deductions

    def test_nan_divergence_is_grade_D(self):
        events = [
            residual(0, {"p": 0.1}),
            Event(run_id="r", seq=1, kind=EventKind.DIVERGENCE, severity=Severity.ERROR,
                  payload=DivergencePayload(field="p", is_nan=True)),
        ]
        report = verify(_ctx(events))
        assert report.grade is TrustGrade.D
        assert any(d.check_code == "stability.divergence" for d in report.deductions)

    def test_crash_is_grade_D(self):
        events = [Event(run_id="r", seq=0, kind=EventKind.CRASH, severity=Severity.ERROR,
                        payload=CrashPayload(tag="foam_fatal", foam_fatal=True))]
        assert verify(_ctx(events)).grade is TrustGrade.D

    def test_turbulence_bounding_is_grade_B(self):
        events = [residual(0, {"p": 1e-4}, ContinuityErrors(cumulative=1e-9)),
                  warn(1, "k"), warn(2, "epsilon")]
        report = verify(_ctx(events))
        assert report.grade is TrustGrade.B
        assert any(d.check_code == "bounds.turbulence" for d in report.deductions)

    def test_poor_convergence_is_grade_C(self):
        # final worst residual 0.05 >> 1e-2 tolerance -> MAJOR -> C
        events = [residual(0, {"p": 0.05}, ContinuityErrors(cumulative=1e-9))]
        report = verify(_ctx(events))
        assert report.grade is TrustGrade.C

    def test_field_checks_skipped_offline(self):
        report = verify(_ctx([residual(0, {"p": 1e-4})]))
        skipped_codes = {r.check_code for r in report.skipped}
        assert "conservation.mass_flux" in skipped_codes
        assert "nearwall.yplus" in skipped_codes


class TestFieldChecks:
    def test_mass_flux_balanced_ok(self):
        f = FieldData(time=100, patch_fluxes={"inlet": -1.0, "outlet": 1.0})
        assert MassFluxCheck().run(_ctx([], fields=f)).severity is VSeverity.OK

    def test_mass_flux_imbalanced_major(self):
        f = FieldData(time=100, patch_fluxes={"inlet": -1.0, "outlet": 0.5})
        r = MassFluxCheck().run(_ctx([], fields=f))
        assert r.severity is VSeverity.MAJOR and not r.passed

    def test_yplus_in_log_layer_ok(self):
        f = FieldData(time=100, yplus=np.full(100, 120.0))
        assert YPlusCheck().run(_ctx([], fields=f)).severity is VSeverity.OK

    def test_yplus_out_of_range_major(self):
        f = FieldData(time=100, yplus=np.full(100, 2.0))  # far below log-layer
        assert YPlusCheck().run(_ctx([], fields=f)).severity is VSeverity.MAJOR

    def test_laminar_at_turbulent_reynolds_major(self):
        f = FieldData(time=100, inlet_speed=10.0, length_scale=1.0, nu=1e-4)  # Re=1e5
        r = RegimeCheck().run(_ctx([], config=CaseConfig(turbulence_type="laminar"), fields=f))
        assert r.severity is VSeverity.MAJOR and r.value == 1e5

    def test_negative_turbulence_field_major(self):
        f = FieldData(time=100, internal={"k": np.array([0.1, -0.01, 0.2])})
        assert PlausibilityCheck().run(_ctx([], fields=f)).severity is VSeverity.MAJOR

    def test_full_field_run_grade_A(self):
        f = FieldData(time=100, patch_fluxes={"inlet": -1.0, "outlet": 1.0},
                      yplus=np.full(50, 80.0),
                      internal={"k": np.array([0.1, 0.2]), "U": np.array([[1, 0, 0], [1, 0, 0]])},
                      inlet_speed=1.0, length_scale=0.1, nu=1e-5)
        events = [residual(0, {"p": 1e-4}, ContinuityErrors(cumulative=1e-9))]
        report = verify(_ctx(events, fields=f))
        assert report.grade is TrustGrade.A
        assert not report.skipped  # all field checks ran
