"""Log/residual parser — raw OpenFOAM solver output -> structured Events.

This is the Watcher's parsing layer (Master Tech Architecture §8.1): a regex +
state-machine parser that turns solver stdout into typed Events. It is streaming:
feed it chunks as they arrive from the sandbox, and it yields Events as complete
information becomes available, buffering partial trailing lines.

It NEVER emits free-form prose — only structured fields, with byte offsets back
into the raw log for provenance. Anomaly detection over residual *history*
(slope, plateau, wall stall) lives in ``anomaly.py``; this module handles the
per-line facts and per-line NaN/Inf divergence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

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

# --- Regexes (compiled once) ------------------------------------------------

_RE_TIME = re.compile(r"^Time = (\S+)")
#: The **bulk** Courant number, and only that one.
#:
#: `(?<![A-Za-z] )` is the whole fix. Multiphase and compressible solvers print
#: more than one Courant number per step, and this pattern matched every one of
#: them by substring — so on `interFoam` the line
#:
#:     Interface Courant Number mean: 0.0421794 max: 0.762621
#:
#: matched, landed in the same step as the bulk reading that preceded it, and
#: **overwrote it**. Measured on a real `damBreak` run: the parser reported a
#: peak Courant of 1.11447, which is the interface peak exactly; the true bulk
#: peak was 1.24333. Every CFL check, the stability verdict, and the timestep
#: the fix proposer computes were all reading the wrong number on every VOF
#: case — and reading one that stays *low* while the bulk number is the one that
#: breaks a solve, so it fails silently and in the dangerous direction.
#:
#: The interface number is a real control (`maxAlphaCo`) and deserves its own
#: reading eventually. Conflating the two is what could not stand.
_RE_COURANT = re.compile(
    r"(?<![A-Za-z] )Courant Number mean:\s*(\S+)\s+max:\s*(\S+)")
_RE_DELTAT = re.compile(r"^deltaT = (\S+)")
#: `[\w.]+`, not `\w+`. OpenFOAM field names contain dots — `alpha.water`,
#: `alpha.air`, `U.air`, `T.solid` — and `\w` excludes `.`, so
#:
#:     smoothSolver:  Solving for alpha.water, Initial residual = 0.0095, ...
#:
#: never matched. On a real `interFoam/damBreak` run the parser produced 121
#: residual steps carrying `p_rgh` and `pcorr` and nothing else: the phase
#: fraction — the field that *is* the free surface, and the whole point of a VOF
#: solve — was invisible. `worst_final_residual` then reported 5.3e-04 from the
#: pressure alone and called the run converged while `alpha.water` sat above
#: 1e-02 throughout.
#:
#: Every multiphase and multi-region solver OpenFOAM ships names fields this way.
_RE_RESIDUAL = re.compile(
    r"Solving for ([\w.]+),\s*Initial residual = (\S+),\s*Final residual = (\S+),"
    r"\s*No Iterations (\d+)"
)
#: The region headers v2406's multi-region solvers print before solving a region (read from
#: `chtMultiRegionFoam.C`, `chtMultiRegionSimpleFoam.C`, `chtMultiRegionTwoPhaseEulerFoam.C` and
#: their `solveFluid.H`, `solveSolid.H`): `Solving for fluid region bottomAir`, `Solving for
#: solid region heater`. Every `Solving for <field>` line after one, up to the next header or the
#: next `Time =`, is that region's.
_REGION_HEADER = r"^[ \t]*Solving for (?:fluid|solid) region (\S+)[ \t]*$"
_RE_REGION = re.compile(_REGION_HEADER)
_RE_REGION_IN_TEXT = re.compile(_REGION_HEADER, re.MULTILINE)
#: The coupled energy solve, `Solving energy coupled regions`: one matrix across the coupled
#: regions, and in chtMultiRegionSimpleFoam each fluid region's pressure solve follows it with no
#: region header of its own. A residual printed there belongs to no region the log names.
_COUPLED_HEADER = r"^[ \t]*Solving energy coupled regions"
_RE_COUPLED = re.compile(_COUPLED_HEADER)
_RE_COUPLED_IN_TEXT = re.compile(_COUPLED_HEADER, re.MULTILINE)
#: `compressibleCourantNo.C` (chtMultiRegionFoam/fluid): `Region: bottomAir Courant Number mean:
#: 0.01 max: 0.3`, one line per fluid region. `_RE_COURANT` never matched it (the region name
#: precedes "Courant"), so v0.5.0 read no Courant number from a CHT log.
_RE_REGION_COURANT = re.compile(
    r"^Region: (\S+) Courant Number mean:\s*(\S+)\s+max:\s*(\S+)")
#: OpenFOAM.org's `foamMultiRun` (v11 on) prints no region header. Its `regionSolvers` sets
#: `Sout.prefix()`, which `prefixOSstream` writes at the start of every line (read on 21 Oct 2026
#: from OpenFOAM-11's and OpenFOAM-12's `regionSolvers.C`, `foamMultiRun.C` and
#: `prefixOSstream.C`): while a region's solver runs, the region's name padded with spaces to one
#: more character than the longest region name; elsewhere in the time loop
#: (`setGlobalPrefix`), that many spaces, the `Time =`, `deltaT =` and `ExecutionTime =` lines
#: among them. The banner and the setup, written before the regions' solvers exist, carry none.
#: The first `Time =` behind blanks gives the width.
_RE_PREFIXED_TIME = re.compile(r"^( +)Time = \S")
#: A prefix, once its width is known: a region's name and its padding, or blanks.
_RE_PREFIX = re.compile(r"^(\S*) +$")
#: The separator between a field and its region in a residual key under `regions=True`: `h@heater`.
#: `@` is outside the field-name pattern `_RE_RESIDUAL` reads (`[\w.]+`), so no field read from a
#: log carries one, and a suffix keeps the field's own name first: `alpha.water@fluid` is still a
#: phase fraction to every reader that asks `startswith("alpha.")`.
REGION_SEPARATOR = "@"


def regions_in(text: str) -> tuple[str, ...]:
    """The regions a multi-region log names in its headers, in the order they first appear."""
    return tuple(dict.fromkeys(m.group(1) for m in _RE_REGION_IN_TEXT.finditer(text)))


def has_coupled_solve(text: str) -> bool:
    """Whether the log has a coupled energy solve, whose residuals no region header names."""
    return _RE_COUPLED_IN_TEXT.search(text) is not None


#: `_RE_REGION_COURANT` and `_RE_PREFIXED_TIME` anywhere in a log, for `names_regions`.
_RE_REGION_COURANT_IN_TEXT = re.compile(_RE_REGION_COURANT.pattern, re.MULTILINE)
_RE_PREFIXED_TIME_IN_TEXT = re.compile(_RE_PREFIXED_TIME.pattern, re.MULTILINE)


def names_regions(text: str | None) -> bool:
    """Whether a log shows a multi-region run in any way this parser reads one: ESI's region
    headers (`regions_in`), its coupled energy solve, its `Region: <name> Courant Number` lines
    (compressibleCourantNo.C; a window of the log may hold one with no header), or a `Time =`
    behind foamMultiRun's prefix (`_RE_PREFIXED_TIME`). Every Courant number a multi-region log
    gives is read from such a line, so a log this says no of gave none of a region's (TRD
    VER-13a, `checkers.steady_courant`)."""
    t = text or ""
    return bool(regions_in(t) or has_coupled_solve(t)
                or _RE_REGION_COURANT_IN_TEXT.search(t) or _RE_PREFIXED_TIME_IN_TEXT.search(t))


#: The command line as v2406 prints it in its banner (argList.C: `"Exec   : " << commandLine_`,
#: the arguments as given, joined by spaces).
_RE_EXEC = re.compile(r"^Exec[ \t]*:", re.MULTILINE)
#: A region named on that command line: `-region <name>` (addRegionOption.H, read by
#: getRegionOption.H), `-regions <names>` or `-allRegions` (addAllRegionOptions.H).
_RE_EXEC_REGION = re.compile(
    r"^Exec[ \t]*:[^\n]*?[ \t]-(?:region|regions|allRegions)(?![^ \t\r\n])", re.MULTILINE)
#: The line every v2406 solver prints before it reads its case (createTime.H: `Info<< "Create
#: time\n"`), after its banner.
_RE_CREATE_TIME = re.compile(r"^Create time[ \t\r]*$", re.MULTILINE)
#: A mesh created for a region other than the default (createMesh.H, createNamedMesh.H: `Create
#: mesh`, then ` <region>` when `polyMesh::regionName(regionName)` is not empty, then ` for time
#: = `). createDynamicFvMesh.H, pimpleFoam's, prints `Create mesh for time = ` whatever the
#: region, so this line alone does not show one.
_RE_NAMED_MESH = re.compile(r"^Create mesh \S+ for time = ", re.MULTILINE)


def default_region_run(text: str | None) -> bool:
    """Whether a log shows a run of its case's default region from the run's start: it holds the
    solver's command line (`_RE_EXEC`) and its `Create time` line (`_RE_CREATE_TIME`), no command
    line in it names a region (`_RE_EXEC_REGION`), and no mesh is created for a named region
    (`_RE_NAMED_MESH`). Such a run read the case's top-level `system/fvSchemes`, not a region's
    (whether it read the content the file holds now, this does not say: `reread_fvschemes`): a
    solver run on region `<name>` reads `system/<name>/fvSchemes` (schemesLookup.C:
    `IOobject(dictName, obr.time().system(), obr, ...)`; IOobject.C, `path()`:
    `instance()/db_.dbDir()`; polyMesh.C, `dbDir()`: the region's name, empty for the default
    region). A window of a log, or one
    without its banner (argList.C prints it only when `bannerEnabled()`, an `infoDetailLevel`
    above 0), does not show it (TRD VER-13a, `checkers.steady_courant`; the review of
    hp6/final)."""
    t = text or ""
    return bool(_RE_EXEC.search(t) and _RE_CREATE_TIME.search(t)
                and not _RE_EXEC_REGION.search(t) and not _RE_NAMED_MESH.search(t))


#: What v2406 prints when the solver reads a watched dictionary again during its run
#: (src/OpenFOAM/db/regIOobject/regIOobjectRead.C, `regIOobject::readIfModified`:
#: `"    Re-reading object " << name() << " from file " << fName`, then, when a file it includes
#: changed, `" because of modified file "`). With controlDict's `runTimeModifiable` on,
#: `Time::readModifiedObjects` (TimeIO.C) asks every registered object
#: (`objectRegistry::readModifiedObjects`, objectRegistry.C), and `fvSchemes` is watched
#: (schemesLookup.C: MUST_READ becomes READ_MODIFIED, then `addWatch()`). It is printed through
#: `Info`, as the banner is (messageStream.C, `stream()`; argList.C), so a log that holds the
#: banner holds this line when the file was read again.
_REREAD = "Re-reading object "


def reread_fvschemes(text: str | None) -> bool:
    """Whether a log says the solver read a `fvSchemes` again during the run: a line on which
    `_REREAD` is followed by `fvSchemes` (the dictionary's own name, fvSchemes.H, or a file it
    names). The run's steps before that line ran on the file as it was then, so the file as it
    is now is not shown to be what they read (TRD VER-13a, `checkers.steady_courant`; the review
    of 5e2a8d5). Linear in the log's length: each line is looked at once."""
    t = text or ""
    at = t.find(_REREAD)
    while at != -1:
        end = t.find("\n", at)
        end = len(t) if end == -1 else end
        if "fvSchemes" in t[at:end]:
            return True
        at = t.find(_REREAD, end)
    return False


_RE_CONTINUITY = re.compile(
    r"time step continuity errors :\s*sum local = (\S+),\s*global = (\S+),"
    r"\s*cumulative = (\S+)"
)
_RE_BOUNDING = re.compile(r"^bounding (\w+),\s*min:\s*(\S+)\s*max:\s*(\S+)(?:\s*average:\s*(\S+))?")
_RE_EXECTIME = re.compile(r"ExecutionTime = (\S+) s\s+ClockTime = (\S+) s")
_RE_FOAM_FATAL = re.compile(r"FOAM FATAL (?:IO )?ERROR")

#: OpenFOAM's own banner, printed by **every** run whose build traps FPEs — the
#: default:
#:
#:     trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).
#:
#: It announces that trapping is *on*, not that anything went wrong. Matching it
#: as a crash gave a 100% false-positive rate on real solves: a healthy cavity
#: case emitted CRASH at line 18, before the time loop started. Checked first
#: and always, because the words it contains are the words a crash uses.
_RE_FPE_BANNER = re.compile(r"trapFpe:|trapping enabled|FOAM_SIGFPE")

#: What an actual fatal signal looks like in the log. Measured against a real
#: SIGFPE from a diverged icoFoam run (exit 136), not inferred:
#:
#:     [stack trace]
#:     #1  Foam::sigFpe::sigHandler(int) in …/libOpenFOAM.so
#:
#: `Foam::sig*::sigHandler` is the handler actually firing. `[stack trace]` is
#: the block header OpenFOAM prints for any fatal signal, so it also covers
#: SIGSEGV and SIGABRT. The bare phrase "Floating point exception" is the
#: *shell's* message and lands in the log only when stderr is merged into it —
#: kept for that case, and only when the banner has been ruled out.
_RE_FATAL_SIGNAL = re.compile(
    r"Foam::sig(?:Fpe|Segv|Int|Quit)::sigHandler"
    r"|Floating point exception"
    r"|Segmentation fault"
)

#: The block header OpenFOAM prints before any fatal-signal backtrace. Handled
#: separately from the patterns above because it says *that* a signal fired
#: without saying which — the naming frame follows it.
_RE_STACK_TRACE_HEADER = re.compile(r"\[stack trace\]")

#: Which signal, when the log says. Ordered: the handler name is definitive, the
#: shell message is a fallback.
_RE_SIGNAL_NAME = [
    (re.compile(r"sigFpe::sigHandler|Floating point exception"), "SIGFPE"),
    (re.compile(r"sigSegv::sigHandler|Segmentation fault"), "SIGSEGV"),
]


def _signal_name(line: str) -> str:
    """Name the signal from the log line, defaulting to a truthful unknown."""
    for pattern, name in _RE_SIGNAL_NAME:
        if pattern.search(line):
            return name
    # `[stack trace]` alone says a fatal signal fired without saying which. The
    # honest answer is that we do not know, not a guess at the most common one:
    # the diagnosis reads this field.
    return "FATAL"


def _to_float(token: str) -> float | None:
    """Parse an OpenFOAM numeric token, tolerating nan/inf/(nan)/-nan etc."""
    t = token.strip().strip("()")
    try:
        return float(t)
    except ValueError:
        low = t.lower()
        if "nan" in low:
            return math.nan
        if "inf" in low:
            return math.inf if not low.startswith("-") else -math.inf
        return None


def _is_bad(x: float | None) -> bool:
    return x is not None and (math.isnan(x) or math.isinf(x))


@dataclass
class _StepBuilder:
    """Accumulates one solver time-step until its boundary is reached."""

    time: float | None = None
    iteration: int | None = None
    initial_residuals: dict[str, float] = field(default_factory=dict)
    first_initial_residuals: dict[str, float] = field(default_factory=dict)
    final_residuals: dict[str, float] = field(default_factory=dict)
    solver_iterations: dict[str, int] = field(default_factory=dict)
    courant_mean: float | None = None
    courant_max: float | None = None
    continuity: ContinuityErrors | None = None
    delta_t: float | None = None
    exec_time: float | None = None
    clock_time: float | None = None
    _has_data: bool = False

    def to_payload(self) -> ResidualPayload:
        return ResidualPayload(
            time=self.time,
            iteration=self.iteration,
            initial_residuals=dict(self.initial_residuals),
            first_initial_residuals=dict(self.first_initial_residuals),
            final_residuals=dict(self.final_residuals),
            solver_iterations=dict(self.solver_iterations),
            courant_mean=self.courant_mean,
            courant_max=self.courant_max,
            continuity=self.continuity,
            delta_t=self.delta_t,
            exec_time=self.exec_time,
            clock_time=self.clock_time,
        )


class LogParser:
    """Streaming parser. One instance per run.

    Usage::

        parser = LogParser(run_id="r1")
        for event in parser.feed(chunk):
            ...
        for event in parser.close(return_code=rc):
            ...
    """

    def __init__(self, run_id: str, start_seq: int = 0, *, regions: bool = False,
                 multirun: bool = False):
        """`regions` (TRD VER-20; `log.regions`, behind `--checks next` in the verdict): key each
        residual by the region the log names for it, `<field>@<region>`, so a multi-region
        log's regions are never merged; take a step's Courant number as the largest region's,
        as `compressibleMultiRegionCourantNo.H` takes it for the time step; and store no
        residual from a coupled energy solve, which no region header names (a NaN there is
        still a divergence). Off, a multi-region log parses as v0.5.0 parsed it: one set of
        fields, a field's last solve in a step standing for every region that solves it.

        `multirun` (with `regions`): the run is OpenFOAM.org's `foamMultiRun`, whose lines name
        their region in a prefix rather than a header (`_RE_PREFIXED_TIME`). Each line is read
        behind its prefix, under the region it names. Before the first `Time =` the width is not
        known, and no Courant number is read there: the setup's and the first step's pre-solve,
        which belong to no numbered step."""
        self.run_id = run_id
        self._regions = regions
        self._multirun = regions and multirun
        self._region: str | None = None      # the region the last header or prefix named
        self._coupled = False                # inside a coupled energy solve, this step
        #: Residual lines inside a coupled energy solve that were not stored (`regions=True`).
        self.unattributed = 0
        #: The width of foamMultiRun's prefix, once read (`multirun`); None until then.
        self.prefix_width: int | None = None
        #: Whether a region's own Courant number was read (`regions=True`): the verdict says the
        #: step's is the largest region's only when one was.
        self.region_courant = False
        self._seq = start_seq
        self._offset = 0                 # byte offset of next unconsumed char
        self._buf = ""                   # partial trailing line
        self._step = _StepBuilder()
        self._iter_index = 0             # global step counter
        self._foam_fatal_seen = False
        self._crash_emitted = False      # dedupe inline crash vs close() crash
        self._stack_trace_at: int | None = None   # a deferred [stack trace]

    # -- public API ---------------------------------------------------------

    def feed(self, text: str) -> list[Event]:
        """Feed a chunk of raw log text; returns Events ready to emit."""
        buf = self._buf + text
        events: list[Event] = []
        # One moving index, and the remainder is kept once at the end.
        # Re-slicing the buffer after every line copied the whole rest of it
        # 110,061 times on a 10 MB log: 40 of the 44 seconds `courant verify`
        # took on the largest certified run (TRD PERF-1).
        pos = 0
        while True:
            nl = buf.find("\n", pos)
            if nl == -1:
                break
            line = buf[pos:nl]
            line_offset = self._offset
            # advance offset by the consumed bytes (line + newline)
            self._offset += len(line.encode("utf-8")) + 1
            pos = nl + 1
            events.extend(self._consume_line(line, line_offset))
        self._buf = buf[pos:]
        return events

    def close(self, return_code: int | None = None) -> list[Event]:
        """Flush the final in-progress step and, if the run failed, emit CRASH."""
        events: list[Event] = []
        # flush trailing partial line as a full line (best effort)
        if self._buf:
            events.extend(self._consume_line(self._buf, self._offset))
            self._offset += len(self._buf.encode("utf-8"))
            self._buf = ""
        if self._step._has_data:
            events.append(self._emit(EventKind.RESIDUAL, self._step.to_payload()))
            self._step = _StepBuilder()
        # A stack trace whose frames never named a signal is still a crash.
        # Reported as FATAL rather than guessed at: the diagnosis reads this.
        if self._stack_trace_at is not None and not self._crash_emitted:
            self._crash_emitted = True
            events.append(
                self._emit(
                    EventKind.CRASH,
                    CrashPayload(signal="FATAL", tag="fatal_signal"),
                    severity=Severity.ERROR,
                    log_offset=self._stack_trace_at,
                )
            )
            self._stack_trace_at = None

        if return_code not in (None, 0) and not self._crash_emitted:
            events.append(
                self._emit(
                    EventKind.CRASH,
                    CrashPayload(return_code=return_code, tag="nonzero_exit"),
                    severity=Severity.ERROR,
                )
            )
        return events

    def parse_text(self, text: str, return_code: int | None = 0) -> list[Event]:
        """Convenience: parse a complete log string in one call."""
        events = self.feed(text)
        events.extend(self.close(return_code=return_code))
        return events

    # -- internals ----------------------------------------------------------

    def _emit(self, kind: EventKind, payload, severity: Severity = Severity.INFO,
              log_offset: int | None = None) -> Event:
        ev = Event(
            run_id=self.run_id,
            seq=self._seq,
            kind=kind,
            severity=severity,
            payload=payload,
            ts_sim=self._step.time,
            log_offset=log_offset,
        )
        self._seq += 1
        return ev

    def _behind_prefix(self, line: str) -> str | None:
        """A foamMultiRun line read behind its prefix, which names the line's region (`None`
        for blanks); None for a line not read at all."""
        if self.prefix_width is None:
            m = _RE_PREFIXED_TIME.match(line)
            if m is None:
                return None if _RE_COURANT.search(line) else line
            self.prefix_width = len(m.group(1))
        head = _RE_PREFIX.match(line[:self.prefix_width])
        if head is None:
            return line                  # written with no prefix (an error message): as it is
        self._region = head.group(1) or None
        return line[self.prefix_width:]

    def _take_region_courant(self, co: float | None) -> None:
        """One region's Courant number: the solver's own CoNum for the step is the largest over
        its fluid regions. No mean is merged: the regions' means are of different meshes."""
        if co is None:
            return
        prev = self._step.courant_max
        self._step.courant_max = (co if prev is None else math.nan
                                  if math.isnan(prev) or math.isnan(co) else max(prev, co))
        self._step._has_data = True
        self.region_courant = True

    def _consume_line(self, line: str, offset: int) -> list[Event]:
        events: list[Event] = []

        if self._multirun:
            line = self._behind_prefix(line)
            if line is None:
                return events

        # New time-step boundary: flush the previous step first.
        m = _RE_TIME.match(line)
        if m:
            if self._step._has_data:
                events.append(self._emit(EventKind.RESIDUAL, self._step.to_payload()))
            self._step = _StepBuilder()
            self._iter_index += 1
            self._region, self._coupled = None, False
            val = _to_float(m.group(1))
            self._step.time = val
            # steady solvers print integer 'Time' = iteration index
            if val is not None and float(val).is_integer():
                self._step.iteration = int(val)
            else:
                self._step.iteration = self._iter_index
            self._step._has_data = True
            return events

        m = _RE_DELTAT.match(line)
        if m:
            self._step.delta_t = _to_float(m.group(1))
            self._step._has_data = True
            return events

        m = _RE_COURANT.search(line)
        if m:
            if self._multirun and self._region is not None:
                self._take_region_courant(_to_float(m.group(2)))     # behind a region's prefix
            else:
                self._step.courant_mean = _to_float(m.group(1))
                self._step.courant_max = _to_float(m.group(2))
                self._step._has_data = True
            return events

        if self._regions:
            m = _RE_REGION.match(line)
            if m:
                self._region, self._coupled = m.group(1), False
                return events
            if _RE_COUPLED.match(line):
                self._region, self._coupled = None, True
                return events
            m = _RE_REGION_COURANT.match(line)
            if m:
                self._take_region_courant(_to_float(m.group(3)))
                return events

        m = _RE_RESIDUAL.search(line)
        if m:
            fieldname, init_s, final_s, iters_s = m.groups()
            init = _to_float(init_s)
            fin = _to_float(final_s)
            if self._regions and self._region is not None:
                fieldname = f"{fieldname}{REGION_SEPARATOR}{self._region}"
            if self._regions and self._coupled:
                # No region the log names: stored under none, never merged into one.
                self.unattributed += 1
            else:
                if init is not None:
                    self._step.initial_residuals[fieldname] = init
                    self._step.first_initial_residuals.setdefault(fieldname, init)
                if fin is not None:
                    self._step.final_residuals[fieldname] = fin
                self._step.solver_iterations[fieldname] = int(iters_s)
                self._step._has_data = True
            # per-line divergence: a NaN/Inf residual is an immediate red flag
            if _is_bad(init) or _is_bad(fin):
                events.append(
                    self._emit(
                        EventKind.DIVERGENCE,
                        DivergencePayload(
                            field=fieldname,
                            residual=init if _is_bad(init) else fin,
                            is_nan=(_is_bad(init) and math.isnan(init))
                            or (_is_bad(fin) and math.isnan(fin)),
                            is_inf=(_is_bad(init) and math.isinf(init))
                            or (_is_bad(fin) and math.isinf(fin)),
                        ),
                        severity=Severity.ERROR,
                        log_offset=offset,
                    )
                )
            return events

        m = _RE_CONTINUITY.search(line)
        if m:
            self._step.continuity = ContinuityErrors(
                sum_local=_to_float(m.group(1)),
                **{"global": _to_float(m.group(2))},
                cumulative=_to_float(m.group(3)),
            )
            self._step._has_data = True
            return events

        m = _RE_BOUNDING.match(line)
        if m:
            events.append(
                self._emit(
                    EventKind.WARN,
                    WarnPayload(category="bounding", field=m.group(1), count=1,
                                minimum=_to_float(m.group(2)), maximum=_to_float(m.group(3)),
                                average=_to_float(m.group(4)) if m.group(4) else None),
                    severity=Severity.WARN,
                    log_offset=offset,
                )
            )
            return events

        m = _RE_EXECTIME.search(line)
        if m:
            self._step.exec_time = _to_float(m.group(1))
            self._step.clock_time = _to_float(m.group(2))
            self._step._has_data = True
            return events

        if _RE_FOAM_FATAL.search(line):
            self._foam_fatal_seen = True
            self._crash_emitted = True
            events.append(
                self._emit(
                    EventKind.CRASH,
                    CrashPayload(foam_fatal=True, tag="foam_fatal"),
                    severity=Severity.ERROR,
                    log_offset=offset,
                )
            )
            return events

        # The banner wins. It contains the same words a real crash does, and it
        # is present on every healthy run, so it has to be excluded before the
        # crash patterns are consulted rather than after.
        if _RE_FPE_BANNER.search(line):
            return events

        if _RE_STACK_TRACE_HEADER.search(line):
            # The header says a fatal signal fired; the frame naming *which* one
            # is a line or two below it. Remember the offset and wait — emitting
            # here would permanently record "FATAL" for a crash the log
            # identifies precisely, and events are immutable once emitted.
            if not self._crash_emitted and self._stack_trace_at is None:
                self._stack_trace_at = offset
            return events

        if _RE_FATAL_SIGNAL.search(line):
            # A stack trace is many lines; one CRASH per run is the fact.
            if self._crash_emitted:
                return events
            self._crash_emitted = True
            events.append(
                self._emit(
                    EventKind.CRASH,
                    CrashPayload(signal=_signal_name(line), tag="fatal_signal"),
                    severity=Severity.ERROR,
                    # The header, not this frame: it is where the failure starts
                    # in the log, and it is the line a reader wants shown.
                    log_offset=self._stack_trace_at
                    if self._stack_trace_at is not None else offset,
                )
            )
            self._stack_trace_at = None
            return events

        return events


def parse_log_file(path: str, run_id: str, return_code: int | None = 0) -> list[Event]:
    """Parse a complete log file from disk into Events."""
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    return LogParser(run_id=run_id).parse_text(text, return_code=return_code)
