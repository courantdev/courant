"""The Watcher façade — attaches to a run and turns a raw line stream into the
complete Event stream (parser facts + history-based anomalies).

This is the single object the Run Sandbox feeds. It owns a LogParser (line -> facts)
and an AnomalyDetector (history -> anomalies), and merges their output in order.
"""

from __future__ import annotations

from courant.events import Event, EventKind
from courant.watcher.anomaly import AnomalyConfig, AnomalyDetector
from courant.watcher.parser import LogParser


class Watcher:
    """Owns parsing + anomaly detection for one run.

    Feed it raw log chunks via :meth:`feed`; poll :meth:`tick` from a wall timer;
    call :meth:`finalize` when the solver exits. Each returns Events in order.
    """

    def __init__(self, run_id: str, anomaly_config: AnomalyConfig | None = None):
        self.run_id = run_id
        self._parser = LogParser(run_id)
        self._anomaly = AnomalyDetector(run_id, anomaly_config)
        self.events: list[Event] = []

    def feed(self, text: str) -> list[Event]:
        produced: list[Event] = []
        for ev in self._parser.feed(text):
            produced.append(ev)
            if ev.kind is EventKind.RESIDUAL:
                produced.extend(self._anomaly.observe(ev))
        self.events.extend(produced)
        return produced

    def tick(self, now: float) -> list[Event]:
        produced = self._anomaly.check_wall_stall(now)
        self.events.extend(produced)
        return produced

    def finalize(self, return_code: int | None) -> list[Event]:
        produced: list[Event] = []
        for ev in self._parser.close(return_code=return_code):
            produced.append(ev)
            if ev.kind is EventKind.RESIDUAL:
                produced.extend(self._anomaly.observe(ev))
        self.events.extend(produced)
        return produced
