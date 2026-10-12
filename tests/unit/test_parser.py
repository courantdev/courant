"""Unit tests for the log/residual parser against realistic OpenFOAM output."""

import math

from courant.events import EventKind
from courant.watcher.parser import LogParser, _to_float

# --- Realistic OpenFOAM log fixtures ---------------------------------------

HEALTHY_SIMPLEFOAM = """\
Time = 1

smoothSolver:  Solving for Ux, Initial residual = 0.234, Final residual = 8.9e-06, No Iterations 4
smoothSolver:  Solving for Uy, Initial residual = 0.189, Final residual = 7.2e-06, No Iterations 4
GAMG:  Solving for p, Initial residual = 0.156, Final residual = 9.1e-07, No Iterations 12
time step continuity errors : sum local = 1.2e-08, global = -3.4e-19, cumulative = -3.4e-19
ExecutionTime = 0.34 s  ClockTime = 1 s

Time = 2

smoothSolver:  Solving for Ux, Initial residual = 0.0234, Final residual = 8.9e-07, No Iterations 3
smoothSolver:  Solving for Uy, Initial residual = 0.0189, Final residual = 7.2e-07, No Iterations 3
GAMG:  Solving for p, Initial residual = 0.0156, Final residual = 9.1e-08, No Iterations 8
time step continuity errors : sum local = 1.2e-09, global = -3.4e-20, cumulative = -3.4e-20
ExecutionTime = 0.68 s  ClockTime = 1 s
"""

TRANSIENT_PIMPLE_DIVERGES = """\
Time = 0.001

Courant Number mean: 0.023 max: 0.451
smoothSolver:  Solving for Ux, Initial residual = 0.0234, Final residual = 8.9e-06, No Iterations 4
GAMG:  Solving for p, Initial residual = 0.156, Final residual = 9.1e-07, No Iterations 12
time step continuity errors : sum local = 1.2e-08, global = -3.4e-19, cumulative = -3.4e-19
ExecutionTime = 0.34 s  ClockTime = 1 s

Time = 0.002

Courant Number mean: 4.2 max: 87.3
smoothSolver:  Solving for Ux, Initial residual = 12.4, Final residual = 3.2, No Iterations 1000
GAMG:  Solving for p, Initial residual = nan, Final residual = nan, No Iterations 1000
ExecutionTime = 0.71 s  ClockTime = 2 s
"""

CRASHED_FOAM_FATAL = """\
Time = 5

smoothSolver:  Solving for Ux, Initial residual = 0.5, Final residual = 0.02, No Iterations 20
[0]
[0]
[0] --> FOAM FATAL ERROR:
[0] Maximum number of iterations exceeded
[0]
[0]     From function ...
"""

BOUNDING_WARNINGS = """\
Time = 10

smoothSolver:  Solving for k, Initial residual = 0.3, Final residual = 1e-5, No Iterations 5
bounding k, min: -0.001 max: 0.5 average: 0.02
smoothSolver:  Solving for epsilon, Initial residual = 0.4, Final residual = 2e-5, No Iterations 6
bounding epsilon, min: -0.02 max: 10 average: 1.2
GAMG:  Solving for p, Initial residual = 0.1, Final residual = 1e-6, No Iterations 10
ExecutionTime = 2.0 s  ClockTime = 3 s
"""


def _by_kind(events, kind):
    return [e for e in events if e.kind == kind]


class TestNumericParsing:
    def test_scientific_and_plain(self):
        assert _to_float("9.1e-07") == 9.1e-07
        assert _to_float("0.234") == 0.234
        assert _to_float("(1.2e-08)") == 1.2e-08  # stripped parens

    def test_nan_inf(self):
        assert math.isnan(_to_float("nan"))
        assert math.isinf(_to_float("inf"))
        assert math.isnan(_to_float("-nan"))
        assert _to_float("garbage") is None


class TestHealthyRun:
    def test_two_residual_events_emitted(self):
        events = LogParser("r").parse_text(HEALTHY_SIMPLEFOAM, return_code=0)
        residuals = _by_kind(events, EventKind.RESIDUAL)
        assert len(residuals) == 2
        assert _by_kind(events, EventKind.CRASH) == []
        assert _by_kind(events, EventKind.DIVERGENCE) == []

    def test_fields_and_continuity_captured(self):
        events = LogParser("r").parse_text(HEALTHY_SIMPLEFOAM, return_code=0)
        first = _by_kind(events, EventKind.RESIDUAL)[0].payload
        assert first.iteration == 1  # steady: integer Time -> iteration
        assert first.initial_residuals["p"] == 0.156
        assert first.final_residuals["Ux"] == 8.9e-06
        assert first.solver_iterations["p"] == 12
        assert first.continuity.sum_local == 1.2e-08
        assert first.continuity.global_ == -3.4e-19
        assert first.exec_time == 0.34


class TestDivergence:
    def test_nan_residual_raises_divergence(self):
        events = LogParser("r").parse_text(TRANSIENT_PIMPLE_DIVERGES, return_code=1)
        div = _by_kind(events, EventKind.DIVERGENCE)
        assert len(div) == 1
        assert div[0].payload.field == "p"
        assert div[0].payload.is_nan is True
        assert div[0].severity.value == "error"

    def test_courant_number_captured(self):
        events = LogParser("r").parse_text(TRANSIENT_PIMPLE_DIVERGES, return_code=1)
        residuals = _by_kind(events, EventKind.RESIDUAL)
        assert residuals[0].payload.courant_max == 0.451
        assert residuals[1].payload.courant_max == 87.3  # blowing up

    def test_nonzero_exit_emits_single_crash(self):
        events = LogParser("r").parse_text(TRANSIENT_PIMPLE_DIVERGES, return_code=1)
        crashes = _by_kind(events, EventKind.CRASH)
        assert len(crashes) == 1
        assert crashes[0].payload.tag == "nonzero_exit"


class TestCrash:
    def test_foam_fatal_detected_once(self):
        events = LogParser("r").parse_text(CRASHED_FOAM_FATAL, return_code=1)
        crashes = _by_kind(events, EventKind.CRASH)
        # inline FOAM FATAL crash, not double-counted by close()
        assert len(crashes) == 1
        assert crashes[0].payload.foam_fatal is True
        assert crashes[0].log_offset is not None


class TestBounding:
    def test_bounding_warnings_emitted(self):
        events = LogParser("r").parse_text(BOUNDING_WARNINGS, return_code=0)
        warns = _by_kind(events, EventKind.WARN)
        assert len(warns) == 2
        assert {w.payload.field for w in warns} == {"k", "epsilon"}
        assert all(w.payload.category == "bounding" for w in warns)


class TestStreaming:
    def test_chunked_feed_equals_whole(self):
        whole = LogParser("r").parse_text(HEALTHY_SIMPLEFOAM, return_code=0)
        # feed one byte at a time
        p = LogParser("r")
        streamed = []
        for ch in HEALTHY_SIMPLEFOAM:
            streamed.extend(p.feed(ch))
        streamed.extend(p.close(return_code=0))
        assert [e.kind for e in streamed] == [e.kind for e in whole]
        assert len(_by_kind(streamed, EventKind.RESIDUAL)) == 2

    def test_seq_is_monotonic(self):
        events = LogParser("r").parse_text(TRANSIENT_PIMPLE_DIVERGES, return_code=1)
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)  # unique

    def test_offsets_point_into_log(self):
        events = LogParser("r").parse_text(CRASHED_FOAM_FATAL, return_code=1)
        crash = _by_kind(events, EventKind.CRASH)[0]
        # the byte at that offset should begin the FOAM FATAL line region
        assert "FOAM FATAL ERROR" in CRASHED_FOAM_FATAL[crash.log_offset:crash.log_offset + 40]
