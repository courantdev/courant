"""The live state of a run that is still writing its log (TRD LIVE-2 to LIVE-4).

One engine per run. Feed it log text as it arrives; ask it for the state whenever you like.

Three properties hold by construction, and the tests check each one:
- **Chunking-invariant.** Text fed in any number of pieces gives the same events as the same
  text fed at once, because the parser buffers partial lines.
- **Causal.** The state at a byte offset depends only on the text before it. The exit code
  is unknown until the run finishes, so an unfinished log never reads as a crash.
- **FAILING is monotone.** It is decided only from append-only signals (a CRASH event, a
  non-finite residual, sustained residual growth) and is never revoked. So the earliest
  alarm can be found by bisection, and a healthy run that is not FAILING at its end was
  never FAILING on any prefix.

The engine never acts on the run (INV-5). It reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from courant.events.core import EventKind
from courant.events.diverged import is_divergence
from courant.watcher.parser import LogParser

WAITING, RUNNING, FAILING, CONVERGED, FINISHED = "WAITING", "RUNNING", "FAILING", "CONVERGED", "FINISHED"

#: Sustained growth (LIVE-2): the worst initial residual, excluding phase fractions, has risen
#: this many orders of magnitude above its running minimum and sits above the absolute floor.
#: A startup transient in a healthy run does not move a residual three orders upward from its
#: best. A blow-up does, on its way to NaN.
GROWTH_ORDERS = 3
GROWTH_FLOOR = 1.0e-1


@dataclass(frozen=True)
class Alarm:
    code: str                  # live.crash | live.non_finite | live.residual_growth
    summary: str
    step: int                  # residual steps completed when it fired
    time: float | None         # simulated time
    exec_s: float | None       # solver ExecutionTime at the alarm
    log_offset: int | None

    def as_dict(self) -> dict:
        return {"code": self.code, "summary": self.summary, "step": self.step, "time": self.time,
                "exec_s": self.exec_s, "log_offset": self.log_offset}


@dataclass(frozen=True)
class LiveState:
    state: str
    offset: int                # bytes consumed
    steps: int                 # residual steps seen
    time: float | None
    exec_s: float | None
    clock_s: float | None
    alarm: Alarm | None = None
    convergence: dict | None = None
    verdict: object | None = None    # the full Verdict, once FINISHED

    def as_dict(self) -> dict:
        v = self.verdict
        return {"state": self.state, "offset": self.offset, "steps": self.steps,
                "time": self.time, "exec_s": self.exec_s, "clock_s": self.clock_s,
                "alarm": self.alarm.as_dict() if self.alarm else None,
                "convergence": self.convergence,
                "verdict": None if v is None else {"verdict": v.verdict, "rule": v.rule,
                                                   "grade": v.grade}}


def _phase(name: str) -> bool:
    return name.startswith("alpha.") or name == "alphas"


@dataclass
class LiveEngine:
    run_id: str = "live"
    _parser: LogParser = field(init=False)
    events: list = field(default_factory=list)
    offset: int = 0
    steps: int = 0
    _time: float | None = None
    _exec: float | None = None
    _clock: float | None = None
    _best: float | None = None
    alarm: Alarm | None = None

    def __post_init__(self) -> None:
        self._parser = LogParser(run_id=self.run_id)

    # -- input ----------------------------------------------------------------------------
    def feed(self, text: str) -> None:
        self.offset = self.offset + len(text.encode("utf-8"))
        for e in self._parser.feed(text):
            self._observe(e)

    def _observe(self, e) -> None:
        self.events.append(e)
        if e.kind is EventKind.RESIDUAL:
            p = e.payload
            self.steps = self.steps + 1
            self._time = p.time if p.time is not None else self._time
            self._exec = p.exec_time if p.exec_time is not None else self._exec
            self._clock = p.clock_time if p.clock_time is not None else self._clock
            self._growth(p, e.log_offset)
        elif e.kind is EventKind.CRASH and self.alarm is None:
            self._raise("live.crash", "the solver crashed", e.log_offset)
        elif is_divergence(e) and self.alarm is None:
            self._raise("live.non_finite", "a residual became non-finite", e.log_offset)

    def _growth(self, p, log_offset) -> None:
        worst = None
        for name, v in p.initial_residuals.items():
            if _phase(name) or v is None:
                continue
            if not math.isfinite(v):
                return                                  # the parser's DIVERGENCE covers it
            if worst is None or v > worst:
                worst = v
        if worst is None or worst <= 0.0:
            return
        if self._best is None or worst < self._best:
            self._best = worst
        if self.alarm is None and worst > GROWTH_FLOOR and \
                worst >= self._best * 10.0 ** GROWTH_ORDERS:
            self._raise("live.residual_growth",
                        f"the worst residual rose {GROWTH_ORDERS} orders of magnitude above its "
                        f"best ({self._best:.3g} to {worst:.3g})", log_offset)

    def setup_fault(self, summary: str) -> None:
        """PRE-3: a fault preflight found in the case. FAILING before the first timestep."""
        if self.alarm is None:
            self.alarm = Alarm("live.setup", summary, 0, None, 0.0, 0)

    def _raise(self, code: str, summary: str, log_offset) -> None:
        self.alarm = Alarm(code, summary, self.steps, self._time, self._exec, log_offset)

    # -- output ---------------------------------------------------------------------------
    def state(self) -> LiveState:
        if self.alarm is not None:
            s = FAILING
        elif self.steps == 0:
            s = WAITING
        else:
            s = RUNNING
        return LiveState(s, self.offset, self.steps, self._time, self._exec, self._clock, self.alarm)


def state_of_prefix(text: str, run_id: str = "live") -> LiveState:
    """The live state of a log truncated here: what an engine fed this text would say."""
    eng = LiveEngine(run_id)
    eng.feed(text)
    return eng.state()
