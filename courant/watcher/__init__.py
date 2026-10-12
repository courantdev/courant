"""The Watcher — raw solver output -> structured Events + anomaly detection."""

from courant.watcher.anomaly import AnomalyConfig, AnomalyDetector
from courant.watcher.parser import LogParser, parse_log_file
from courant.watcher.watcher import Watcher

__all__ = [
    "LogParser",
    "parse_log_file",
    "AnomalyDetector",
    "AnomalyConfig",
    "Watcher",
]
