"""Unit tests for the Event contract (the system's spine)."""

import time

import pytest
from pydantic import ValidationError

from courant.events import (
    ContinuityErrors,
    CrashPayload,
    DivergencePayload,
    Event,
    EventKind,
    ResidualPayload,
    Severity,
)


def _residual_event(**kw) -> Event:
    defaults = dict(
        run_id="run-1",
        seq=0,
        kind=EventKind.RESIDUAL,
        payload=ResidualPayload(time=0.1, initial_residuals={"p": 1e-3, "Ux": 2e-4}),
    )
    defaults.update(kw)
    return Event(**defaults)


class TestConstruction:
    def test_residual_event_builds(self):
        ev = _residual_event()
        assert ev.kind is EventKind.RESIDUAL
        assert ev.payload.initial_residuals["p"] == 1e-3
        assert ev.severity is Severity.INFO  # default
        assert ev.ts_wall <= time.time() + 1

    def test_discriminated_union_selects_right_payload(self):
        ev = Event(
            run_id="r",
            seq=1,
            kind=EventKind.CRASH,
            severity=Severity.ERROR,
            payload=CrashPayload(return_code=139, signal="SIGSEGV", foam_fatal=True),
        )
        assert isinstance(ev.payload, CrashPayload)
        assert ev.payload.return_code == 139

    def test_divergence_payload_nan_flag(self):
        ev = Event(
            run_id="r",
            seq=2,
            kind=EventKind.DIVERGENCE,
            severity=Severity.ERROR,
            payload=DivergencePayload(field="p", is_nan=True, slope=1.5),
        )
        assert ev.payload.is_nan is True
        assert ev.payload.field == "p"


class TestInvariants:
    def test_kind_must_match_payload(self):
        # kind says CRASH but payload is a ResidualPayload -> rejected
        with pytest.raises(ValidationError):
            Event(
                run_id="r",
                seq=0,
                kind=EventKind.CRASH,
                payload=ResidualPayload(time=0.1),
            )

    def test_event_is_immutable(self):
        ev = _residual_event()
        with pytest.raises(ValidationError):
            ev.seq = 5  # frozen model

    def test_payload_is_immutable(self):
        ev = _residual_event()
        with pytest.raises(ValidationError):
            ev.payload.time = 9.9


class TestSerialization:
    def test_round_trip_preserves_payload_type(self):
        ev = _residual_event(
            payload=ResidualPayload(
                time=0.5,
                iteration=5,
                initial_residuals={"p": 0.01},
                courant_max=0.85,
                continuity=ContinuityErrors(sum_local=1e-8, cumulative=1e-9),
            )
        )
        dumped = ev.model_dump_json()
        restored = Event.model_validate_json(dumped)
        assert restored == ev
        assert isinstance(restored.payload, ResidualPayload)
        assert restored.payload.courant_max == 0.85
        assert restored.payload.continuity.sum_local == 1e-8

    def test_continuity_global_alias(self):
        # 'global' is a python keyword; ensure the alias works both ways
        c = ContinuityErrors.model_validate({"global": 1e-10, "sum_local": 1e-8})
        assert c.global_ == 1e-10
        assert c.model_dump(by_alias=True)["global"] == 1e-10
