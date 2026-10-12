"""One definition of "diverged" (TRD ADP-5).

A run diverged when a residual went non-finite: a solver line printed an initial or final
residual of NaN or Inf, which the log parser turns into a DIVERGENCE event with `is_nan` or
`is_inf` set (`courant.watcher.parser`). That is the whole definition, and this module is the only
place it is written. The certification sandbox's run status (`classify_status`, which the
certification harness reports), the verifier's rule VER-5.2 and
its `run.divergence` reasons and counts, the replay judge, the stability check and the
diagnosis all ask it.

The anomaly detector's soft DIVERGENCE (`courant.watcher.anomaly`: the representative residual
climbing over a five-step window, `slope` set, every value finite) is a live early warning,
not a divergence. Here it is called a residual climb. RRD RM-18 v1 found what two definitions
cost: on `pisoFoam/RAS/cavity` with the lid speed doubled and with nu tripled, the detector
fired on the startup transient (100 steps, no NaN, final residual 8e-3, Co 0.5), the sandbox
reported the run `diverged`, and `courant verify` on the same log, whose parser emits no such
event, returned TRUSTWORTHY. A climbing residual that matters reaches a verdict through the
residual and convergence checks, which judge the whole run and not five steps of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from courant.events.core import Event, EventKind

__all__ = ["diverged", "divergences", "is_divergence", "is_residual_climb", "non_finite"]


def non_finite(payload) -> bool:
    """Whether a DIVERGENCE payload records a NaN or infinite residual. Takes the payload
    model, or its serialised form (`Event.model_dump`, as the live stream carries it)."""
    if isinstance(payload, Mapping):
        return bool(payload.get("is_nan") or payload.get("is_inf"))
    return bool(getattr(payload, "is_nan", False) or getattr(payload, "is_inf", False))


def is_divergence(event: Event) -> bool:
    """A residual went non-finite: the one event that makes a run diverged."""
    return event.kind is EventKind.DIVERGENCE and non_finite(event.payload)


def is_residual_climb(event: Event) -> bool:
    """The anomaly detector's soft DIVERGENCE: a finite residual climbing. Not a divergence."""
    return event.kind is EventKind.DIVERGENCE and not non_finite(event.payload)


def divergences(events: Iterable[Event]) -> list[Event]:
    """The events that make a run diverged, in order."""
    return [e for e in events if is_divergence(e)]


def diverged(events: Iterable[Event]) -> bool:
    """Whether the run diverged: some residual went NaN or infinite."""
    return any(is_divergence(e) for e in events)
