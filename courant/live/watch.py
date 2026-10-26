"""Following a running solve, read-only (TRD LIVE-1, LIVE-5, LIVE-6).

`run_watch` consumes log chunks (from `courant.cli.tail.LogTail`, or from a list in a test)
and prints a status line whenever the state changes. When the solver prints `End`, it hands
the case to `courant verify`, prints that verdict and returns its exit code. It never sends a
signal, never writes into the case, and never edits a dictionary. With `exit_on_fail` it
returns 2 at the first FAILING state, so the user's own job script can act.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from courant.live.engine import CONVERGED, FAILING, FINISHED, RUNNING, LiveEngine, LiveState
from courant.watcher.exitcode import ends_in_end

#: Either verdict major's answer to its exit code (GATE-1: accept 0, reject 2, insufficient 3).
EXIT = {"TRUSTWORTHY": 0, "NOT_TRUSTWORTHY": 2, "NOT_ASSESSABLE": 3,
        "accept": 0, "reject": 2, "insufficient": 3}
EXIT_IDLE = 3
EXIT_FAILING = 2
HEAD_BYTES = 20_000
#: How much of the log's end `End` is looked for in: enough for a parallel run's finalize lines.
TAIL_BYTES = 4_096
#: LIVE-6: never spend more than about a fifth of a core on evaluation.
OVERHEAD_FACTOR = 5.0
CONVERGENCE_MIN_STEPS = 200


@dataclass
class WatchOutcome:
    exit_code: int
    state: LiveState
    verdict: object | None
    evaluations: int
    eval_seconds: float


def _status(st: LiveState) -> str:
    t = f" · time {st.time:g}" if st.time is not None else ""
    return f"  step {st.steps:,}{t} · {st.state}"


def run_watch(chunks: Iterable, *, case_dir: Path | None, log_path: Path,
              emit: Callable[[str], None], as_json: bool = False, exit_on_fail: bool = False,
              interval: float = 2.0, clock=time.monotonic,
              verify_fn: Callable | None = None, convergence: bool = True,
              once: bool = False) -> WatchOutcome:
    from courant.verify import render_json, render_text, verify_path

    verify_fn = verify_fn or verify_path
    engine = LiveEngine(run_id=(case_dir or log_path).name)
    setup = _setup_summary(case_dir) if case_dir is not None else None
    if setup:
        engine.setup_fault(setup)
    head, tail = "", ""
    ended = False
    last_eval, eval_cost, evaluations = -1e18, 0.0, 0
    shown: tuple | None = None
    advisory: dict | None = None

    def show(st: LiveState, extra: dict | None = None) -> None:
        nonlocal shown
        key = (st.state, st.alarm.code if st.alarm else None,
               (extra or {}).get("converged_at_step"))
        if key == shown:
            return
        shown = key
        if as_json:
            doc = st.as_dict()
            if extra:
                doc["convergence"] = extra
            emit(json.dumps(doc, sort_keys=True))
            return
        emit(_status(st))
        if st.alarm:
            a = st.alarm
            emit(f"  !! FAILING at step {a.step:,}: {a.summary}. This run will not produce a "
                 f"trustworthy result. Stopping it is your decision; Courant does not act.")
        if extra and st.state == CONVERGED:
            emit(f"  ok The monitored answer stopped changing at step {extra['converged_at_step']:,} "
                 f"({extra['binding_monitor']}). Further iterations have not moved it beyond its "
                 f"tolerance. Stopping the run is your decision; Courant does not act.")

    for chunk in chunks:
        if getattr(chunk, "restarted", False):
            engine = LiveEngine(run_id=engine.run_id)
            if setup:
                engine.setup_fault(setup)
            head, tail, ended, advisory = "", "", False, None
        text = chunk.text
        if len(head) < HEAD_BYTES:
            head = (head + text)[:HEAD_BYTES]
        engine.feed(text)
        tail = (tail + text)[-TAIL_BYTES:]
        # VER-3's one reading of End, as it applies to a log still being written: the last End,
        # followed by nothing but what a finished parallel run prints after it ("Finalising
        # parallel run", Open MPI's help lines), which `courant verify` reads as ended too. A
        # utility's End with the solver's banner arriving after it is not the run's end.
        if ends_in_end(tail, last=True):
            ended = True
            break
        now = clock()
        if now - last_eval < max(interval, OVERHEAD_FACTOR * eval_cost):
            continue
        t0 = time.perf_counter()
        st = engine.state()
        if convergence and case_dir is not None and st.state == RUNNING and \
                st.steps >= CONVERGENCE_MIN_STEPS:
            advisory = _advisory(case_dir, head, engine) or advisory
            if advisory and advisory.get("status") == "PASS":
                st = LiveState(CONVERGED, st.offset, st.steps, st.time, st.exec_s, st.clock_s,
                               None, advisory)
        eval_cost = time.perf_counter() - t0
        evaluations = evaluations + 1
        last_eval = now
        show(st, advisory)
        if st.state == FAILING and exit_on_fail:
            return WatchOutcome(EXIT_FAILING, st, None, evaluations, eval_cost)

    ended = ended or ends_in_end(tail, last=True)       # the stream is over: its last line too
    st = engine.state()
    show(st, advisory)
    if st.state == FAILING and exit_on_fail:
        return WatchOutcome(EXIT_FAILING, st, None, evaluations, eval_cost)
    if not ended and st.state != FAILING:
        if once:
            return WatchOutcome(0, st, None, evaluations, eval_cost)
        emit(f"  the log stopped growing at step {st.steps:,} without the solver's End; "
             f"verify it once it has finished")
        return WatchOutcome(EXIT_IDLE, st, None, evaluations, eval_cost)
    verdict = verify_fn(case_dir if case_dir is not None else log_path)
    final = LiveState(FINISHED, st.offset, st.steps, st.time, st.exec_s, st.clock_s, st.alarm,
                      advisory, verdict)
    if as_json:
        emit(json.dumps({"state": FINISHED, "verdict": json.loads(render_json(verdict))},
                        sort_keys=True))
    else:
        emit("  the solver finished; verifying\n")
        emit(render_text(verdict).rstrip("\n"))
    return WatchOutcome(EXIT[verdict.verdict], final, verdict, evaluations, eval_cost)


def _setup_summary(case_dir: Path) -> str | None:
    """PRE-3: the case's setup faults, as one line, or None."""
    from courant.verify.preflight import preflight_path

    try:
        p = preflight_path(case_dir)
    except Exception:                                       # noqa: BLE001 - advisory only
        return None
    if p.verdict != "SETUP_FAULT":
        return None
    return "a setup fault found before the first timestep: " + "; ".join(r.code for r in p.reasons)


def _advisory(case_dir: Path, head: str, engine: LiveEngine) -> dict | None:
    """The convergence assessment on the run so far, steady runs with monitors only."""
    from courant.converge import summary
    from courant.converge.series import assess_case

    try:
        conv = assess_case(case_dir, log_text=head, events=list(engine.events))
    except Exception:                                       # noqa: BLE001 - advisory only
        return None
    if conv.regime != "steady" or not conv.data.get("monitors"):
        return None
    return summary(conv)
