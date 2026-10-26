"""Anomaly detection over residual history — rules, not ML (Tech Architecture §8.1).

The parser turns lines into facts. This layer watches the *history* of those facts
and raises typed anomaly Events the orchestrator transitions on:

  * soft DIVERGENCE — the representative residual is climbing over a window of
    iterations (caught *before* it reaches NaN/Inf, which the parser catches).
  * residual-plateau STALL — the residual has stopped improving while still above
    a convergence target (a silent non-convergence).
  * wall-clock STALL — no solver progress for too long (driven by the sandbox's
    wall timer via ``check_wall_stall``).

Every threshold is explicit and configurable. Episodes are debounced so a single
sustained anomaly raises a single Event, not one per iteration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from courant.events import (
    DivergencePayload,
    Event,
    EventKind,
    ResidualPayload,
    Severity,
    StallPayload,
)


#: Volume-fraction fields, whose residual tracks interface motion rather than
#: convergence. `alpha.water`, `alpha.air`, `alpha.mercury`, and the `alphas`
#: sum multiphaseInterFoam prints.
def _is_phase_fraction(field: str) -> bool:
    lowered = field.lower()
    return lowered.startswith("alpha.") or lowered == "alphas"


@dataclass
class AnomalyConfig:
    # --- soft divergence ---
    slope_window: int = 5          # steps used to fit the residual trend
    divergence_slope: float = 0.15  # log10(residual) growth/step to flag (≈ +40%/step)
    #: Residuals already below this are not diverging, whatever their slope.
    #:
    #: Without this floor a *converged* solve is reported as diverged. A real
    #: cavity case finished with residuals fluctuating between 7.9e-07 and
    #: 1.03e-06 — a 30% wiggle in a noise floor six orders of magnitude below
    #: any tolerance — which fits a slope of 0.20/step and fired DIVERGENCE with
    #: `is_nan=False, is_inf=False`. The run had printed "End", exited 0, and
    #: reached a continuity error of 3e-17.
    #:
    #: The consequence was not cosmetic: `classify_status` turns a DIVERGENCE
    #: event into `RunStatus.DIVERGED`, so the fix loop saw every successful
    #: re-run as a failure and could never report a fix as working. The
    #: plateau check below has always had this guard; divergence did not.
    #:
    #: Hard divergence (NaN/Inf) is detected in the parser and is unaffected —
    #: that must fire at any magnitude.
    divergence_floor: float = 1e-3
    # --- residual plateau ---
    plateau_window: int = 50       # steps of no improvement to flag a stall
    plateau_rel_improve: float = 0.10  # <10% improvement over the window == flat
    #: Only a stall if still above this residual. **Not `judge_residuals`'s per-field bar (TRD
    #: VER-16), and why:** this detector runs on the live stream, one step at a time, with no
    #: case configuration (it never sees `residualControl`, the solver or the relaxation), and
    #: what it emits is a STALL event for the live orchestrator, not a verdict. `courant
    #: verify` never runs it (the gate parses with `LogParser` alone), so this number decides
    #: no verdict; the gate's plateau is M1-P-2 (`converge.steady.residual_summary`), asked by
    #: `convergence.stalled`. It stays the flat `RESIDUAL_TARGET` value.
    convergence_target: float = 1e-3
    # --- wall-clock stall ---
    wall_stall_seconds: float = 120.0


@dataclass
class _History:
    reps: list[float] = field(default_factory=list)      # representative residual/step
    fields: list[str] = field(default_factory=list)      # worst field/step
    iters: list[int] = field(default_factory=list)       # x-axis for slope fit


class AnomalyDetector:
    """Stateful consumer of RESIDUAL events. One instance per run.

    Feed it events with :meth:`observe`; poll :meth:`check_wall_stall` from the
    sandbox timer. Both return lists of anomaly Events ready to publish.
    """

    def __init__(self, run_id: str, config: AnomalyConfig | None = None,
                 start_seq: int = 10_000):
        self.run_id = run_id
        self.cfg = config or AnomalyConfig()
        self._seq = start_seq          # separate seq band from the parser
        self._h = _History()
        self._last_progress_wall: float | None = None
        self._diverging = False        # debounce flags
        self._plateaued = False
        self._wall_stalled = False

    # -- public API ---------------------------------------------------------

    def observe(self, event: Event) -> list[Event]:
        if event.kind is not EventKind.RESIDUAL:
            return []
        payload: ResidualPayload = event.payload
        rep, worst = self._representative(payload)
        if rep is None:
            return []
        self._last_progress_wall = event.ts_wall
        idx = payload.iteration if payload.iteration is not None else len(self._h.reps)
        self._h.reps.append(rep)
        self._h.fields.append(worst)
        self._h.iters.append(idx)

        out: list[Event] = []
        out.extend(self._check_divergence(event))
        out.extend(self._check_plateau(event))
        return out

    def check_wall_stall(self, now: float) -> list[Event]:
        """Poll from the sandbox wall timer. Emits at most one wall STALL/episode."""
        if self._last_progress_wall is None:
            return []
        idle = now - self._last_progress_wall
        if idle >= self.cfg.wall_stall_seconds and not self._wall_stalled:
            self._wall_stalled = True
            return [
                self._emit(
                    EventKind.STALL,
                    StallPayload(reason="wall_clock", idle_seconds=idle),
                    Severity.WARN,
                )
            ]
        if idle < self.cfg.wall_stall_seconds:
            self._wall_stalled = False
        return []

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _representative(p: ResidualPayload) -> tuple[float | None, str | None]:
        """Worst (largest, positive, finite) initial residual and its field.

        **Phase fractions are excluded**, and the reason is physical rather than
        a tuning convenience. `alpha.water` is bounded in [0, 1] by construction
        — MULES enforces it every step — so it cannot run away, and its residual
        does not measure convergence at all: it measures how fast the interface
        is *moving*. On `interFoam/damBreak` the interface accelerates down a
        collapsing column of water, which is the entire point of the case, and
        the residual rises accordingly.

        Fitting a divergence slope to that reported DIVERGENCE on an unmodified
        tutorial — with `is_nan=False`, `is_inf=False`, and a residual of
        1.3e-03 — which `classify_status` then turns into `RunStatus.DIVERGED`.
        A healthy multiphase case would have been declared a failed run.

        It only became reachable once the parser learned to read dotted field
        names at all; before that `alpha.water` was invisible and the question
        never arose. Two bugs deep in the same log.
        """
        best_val: float | None = None
        best_field: str | None = None
        for name, val in p.initial_residuals.items():
            if val is None or val <= 0 or math.isnan(val) or math.isinf(val):
                continue
            if _is_phase_fraction(name):
                continue
            if best_val is None or val > best_val:
                best_val, best_field = val, name
        return best_val, best_field

    def _check_divergence(self, event: Event) -> list[Event]:
        w = self.cfg.slope_window
        if len(self._h.reps) < w:
            return []
        ys = np.log10(np.asarray(self._h.reps[-w:]))
        xs = np.asarray(self._h.iters[-w:], dtype=float)
        # guard against a degenerate x-axis
        if np.ptp(xs) == 0:
            xs = np.arange(w, dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0])

        # A residual already inside the convergence tolerance is not diverging,
        # however steeply it wiggles. See `divergence_floor`.
        if self._h.reps[-1] <= self.cfg.divergence_floor:
            self._diverging = False
            return []

        if slope >= self.cfg.divergence_slope:
            if not self._diverging:
                self._diverging = True
                return [
                    self._emit(
                        EventKind.DIVERGENCE,
                        DivergencePayload(
                            field=self._h.fields[-1],
                            residual=self._h.reps[-1],
                            slope=slope,
                        ),
                        Severity.ERROR,
                    )
                ]
        else:
            self._diverging = False
        return []

    def _check_plateau(self, event: Event) -> list[Event]:
        w = self.cfg.plateau_window
        if len(self._h.reps) < w:
            return []
        window = self._h.reps[-w:]
        rep_now = window[-1]
        # still converged? no stall.
        if rep_now <= self.cfg.convergence_target:
            self._plateaued = False
            return []
        best, worst = min(window), max(window)
        # relative improvement across the window
        improve = (worst - best) / worst if worst > 0 else 0.0
        if improve < self.cfg.plateau_rel_improve:
            if not self._plateaued:
                self._plateaued = True
                return [
                    self._emit(
                        EventKind.STALL,
                        StallPayload(
                            reason="residual_plateau",
                            plateau_iterations=w,
                            field=self._h.fields[-1],
                        ),
                        Severity.WARN,
                    )
                ]
        else:
            self._plateaued = False
        return []

    def _emit(self, kind: EventKind, payload, severity: Severity) -> Event:
        ev = Event(
            run_id=self.run_id,
            seq=self._seq,
            kind=kind,
            severity=severity,
            payload=payload,
        )
        self._seq += 1
        return ev
