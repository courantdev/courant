"""The Event contract — the system's spine (Master Tech Architecture §3.2).

Every component speaks Events. An Event is immutable, typed, and carries enough
structured context to be replayed. Design rule: an Event NEVER contains free-form
prose the system later parses — structured fields only. The raw log stays in
object storage, referenced by ``log_offset`` so any diagnosis can be traced to
exact log bytes.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventKind(str, Enum):
    """The finite set of event kinds flowing through the system."""

    RESIDUAL = "RESIDUAL"        # per-timestep residuals / Courant / continuity
    CHECKPOINT = "CHECKPOINT"    # solver wrote a restartable time directory
    WARN = "WARN"               # bounding / clipping / non-fatal solver warning
    STALL = "STALL"             # no progress: wall-clock stall or frozen residuals
    DIVERGENCE = "DIVERGENCE"    # residuals exploding toward NaN/Inf
    CRASH = "CRASH"             # solver exited non-zero
    INTERVENE = "INTERVENE"      # orchestrator intervened (raised diagnose)
    FIX = "FIX"                 # a fix diff was applied
    VERIFY = "VERIFY"           # a verification/trust-grade result


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Kind-specific payloads (structured fields only)
# ---------------------------------------------------------------------------


class ContinuityErrors(BaseModel):
    """OpenFOAM 'time step continuity errors' line."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    sum_local: float | None = None
    global_: float | None = Field(default=None, alias="global")
    cumulative: float | None = None


class ResidualPayload(BaseModel):
    """One solver time-step / outer-iteration: residuals for every solved field,
    plus Courant number, continuity errors, and timing."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.RESIDUAL] = EventKind.RESIDUAL

    time: float | None = None                       # simulation time
    iteration: int | None = None                    # global step index
    initial_residuals: dict[str, float] = Field(default_factory=dict)
    final_residuals: dict[str, float] = Field(default_factory=dict)
    solver_iterations: dict[str, int] = Field(default_factory=dict)
    courant_mean: float | None = None
    courant_max: float | None = None
    continuity: ContinuityErrors | None = None
    delta_t: float | None = None
    exec_time: float | None = None                  # cumulative CPU seconds
    clock_time: float | None = None                 # cumulative wall seconds


class CheckpointPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.CHECKPOINT] = EventKind.CHECKPOINT

    time_dir: str                                   # e.g. "0.5" — restartable dir
    time: float | None = None


class WarnPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.WARN] = EventKind.WARN

    category: str                                   # e.g. "bounding", "clipping"
    field: str | None = None                        # field being bounded, if any
    count: int = 1                                  # occurrences aggregated
    detail: str | None = None                       # short structured tag, not prose
    #: For a bounding event: what OpenFOAM printed. A clip whose minimum is 0, or a few
    #: parts in a million below it, is a floor touched rather than a field going negative.
    minimum: float | None = None
    maximum: float | None = None
    average: float | None = None


class StallPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.STALL] = EventKind.STALL

    reason: Literal["wall_clock", "residual_plateau", "no_output"]
    idle_seconds: float | None = None               # wall stall duration
    plateau_iterations: int | None = None           # residual flat for N iters
    field: str | None = None


class DivergencePayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.DIVERGENCE] = EventKind.DIVERGENCE

    field: str | None = None                        # field that diverged
    residual: float | None = None                   # value at detection
    is_nan: bool = False
    is_inf: bool = False
    slope: float | None = None                      # residual growth slope (log10/iter)


class CrashPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.CRASH] = EventKind.CRASH

    return_code: int | None = None
    signal: str | None = None                       # e.g. "SIGFPE"
    foam_fatal: bool = False                         # OpenFOAM 'FOAM FATAL ERROR'
    tag: str | None = None                          # short error class, not prose


class IntervenePayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.INTERVENE] = EventKind.INTERVENE

    trigger: str                                    # which anomaly triggered diagnose


class FixPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.FIX] = EventKind.FIX

    diagnosis_id: str | None = None
    escalation_tier: Literal["numerics", "schemes", "mesh"] | None = None
    applied: bool = False


class VerifyPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[EventKind.VERIFY] = EventKind.VERIFY

    grade: Literal["A", "B", "C", "D"] | None = None
    check_code: str | None = None
    passed: bool | None = None


Payload = Annotated[
    Union[
        ResidualPayload,
        CheckpointPayload,
        WarnPayload,
        StallPayload,
        DivergencePayload,
        CrashPayload,
        IntervenePayload,
        FixPayload,
        VerifyPayload,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# The Event
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """An immutable, typed, replayable unit of what happened in a run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    seq: int                                        # monotonic order within a run
    kind: EventKind
    severity: Severity = Severity.INFO
    payload: Payload
    ts_sim: float | None = None                     # simulation time, if applicable
    ts_wall: float = Field(default_factory=time.time)  # unix wall clock
    log_offset: int | None = None                   # byte offset into raw log

    @model_validator(mode="after")
    def _kind_matches_payload(self) -> "Event":
        if self.payload.kind != self.kind:
            raise ValueError(
                f"Event.kind={self.kind} does not match payload.kind={self.payload.kind}"
            )
        return self
