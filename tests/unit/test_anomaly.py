"""Unit tests for anomaly detection over residual history."""

from courant.events import Event, EventKind, ResidualPayload
from courant.watcher.anomaly import AnomalyConfig, AnomalyDetector


def _residual_event(run_id: str, seq: int, iteration: int, rep: float) -> Event:
    """A residual event whose worst field 'p' has initial residual == rep."""
    return Event(
        run_id=run_id,
        seq=seq,
        kind=EventKind.RESIDUAL,
        payload=ResidualPayload(
            iteration=iteration,
            initial_residuals={"p": rep, "Ux": rep * 0.1},
        ),
        ts_wall=1000.0 + iteration,
    )


def _run(detector: AnomalyDetector, reps: list[float]) -> list[Event]:
    out = []
    for i, r in enumerate(reps):
        out.extend(detector.observe(_residual_event(detector.run_id, i, i + 1, r)))
    return out


class TestDivergence:
    def test_climbing_residual_raises_soft_divergence(self):
        det = AnomalyDetector("r", AnomalyConfig(slope_window=5, divergence_slope=0.15))
        # residual growing ~3x/step -> log10 slope ~0.48 >> 0.15
        reps = [1e-3 * (3 ** i) for i in range(6)]
        events = _run(det, reps)
        div = [e for e in events if e.kind == EventKind.DIVERGENCE]
        assert len(div) == 1
        assert div[0].payload.field == "p"
        assert div[0].payload.slope > 0.15

    def test_debounced_single_event_per_episode(self):
        det = AnomalyDetector("r", AnomalyConfig(slope_window=5, divergence_slope=0.15))
        reps = [1e-3 * (3 ** i) for i in range(12)]  # sustained growth
        events = _run(det, reps)
        div = [e for e in events if e.kind == EventKind.DIVERGENCE]
        assert len(div) == 1  # not one per step

    def test_converging_run_raises_nothing(self):
        det = AnomalyDetector("r", AnomalyConfig(slope_window=5, divergence_slope=0.15))
        reps = [1e-1 * (0.5 ** i) for i in range(20)]  # steadily converging
        events = _run(det, reps)
        assert [e for e in events if e.kind == EventKind.DIVERGENCE] == []


class TestPlateau:
    def test_flat_residual_above_target_raises_stall(self):
        det = AnomalyDetector(
            "r",
            AnomalyConfig(plateau_window=10, plateau_rel_improve=0.10,
                          convergence_target=1e-3),
        )
        reps = [0.05] * 15  # dead flat, well above 1e-3 target
        events = _run(det, reps)
        stalls = [e for e in events if e.kind == EventKind.STALL]
        assert len(stalls) == 1
        assert stalls[0].payload.reason == "residual_plateau"

    def test_flat_but_converged_is_not_a_stall(self):
        det = AnomalyDetector(
            "r",
            AnomalyConfig(plateau_window=10, convergence_target=1e-3),
        )
        reps = [1e-6] * 15  # flat but below target -> converged, not stalled
        events = _run(det, reps)
        assert [e for e in events if e.kind == EventKind.STALL] == []


class TestWallStall:
    def test_wall_clock_stall(self):
        det = AnomalyDetector("r", AnomalyConfig(wall_stall_seconds=60))
        det.observe(_residual_event("r", 0, 1, 0.1))  # last progress at ts_wall=1001
        assert det.check_wall_stall(now=1030) == []      # 29s idle, ok
        events = det.check_wall_stall(now=1100)          # 99s idle -> stall
        assert len(events) == 1
        assert events[0].payload.reason == "wall_clock"
        # debounced: second poll during same episode is silent
        assert det.check_wall_stall(now=1200) == []
