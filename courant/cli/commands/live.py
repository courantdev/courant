"""`courant watch`: follow a running solve and say, while it runs, whether it has failed or settled.

Read-only (TRD LIVE-1). It never sends a signal, never edits the case and never writes into
it. To act on its advice, let your own job script use the exit code:

    courant watch ./case --exit-on-fail || scancel "$SLURM_JOB_ID"

Exit codes:
- 0: the run finished and the verdict is accept (or `--once` and it is still running);
- 2: reject, or FAILING with `--exit-on-fail`;
- 3: insufficient, or the log went idle without the solver's End;
- 4: no log could be found.

The previous product's copilot, which applied fixes, is still available as `courant copilot`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer


def watch(
    case: str = typer.Argument(..., help="The case directory to follow, or a solver log file."),
    log: str = typer.Option(None, "--log", "-l", help="The log to follow. Default: the solver log in the case (it may not exist yet)."),
    interval: float = typer.Option(2.0, "--interval", help="Seconds between evaluations (raised automatically to keep the watcher light)."),
    as_json: bool = typer.Option(False, "--json", help="One JSON object per state change, then the verdict."),
    exit_on_fail: bool = typer.Option(False, "--exit-on-fail", help="Exit 2 at the first FAILING state, so a job script can act."),
    once: bool = typer.Option(False, "--once", help="Evaluate what is written now and exit."),
    idle_timeout: float = typer.Option(600.0, "--idle-timeout", help="Give up after the log has not grown for this long (s)."),
    start_timeout: float = typer.Option(600.0, "--start-timeout", help="How long to wait for the log to appear (s)."),
) -> None:
    """Follow a running solve, read-only: FAILING or converged as it happens, the verdict when it ends."""
    from courant.cli.tail import Chunk, LogTail, find_log, wait_for_log
    from courant.live.watch import run_watch

    path = Path(case)
    case_dir: Path | None
    if path.is_file():
        log_path = path
        case_dir = path.parent if (path.parent / "system").is_dir() else None
    elif path.is_dir():
        case_dir = path
        found = Path(log) if log else (find_log(path) if once else
                                       wait_for_log(path, timeout=start_timeout))
        if found is None:
            print(f"courant watch: no solver log (log.<solver>) in {path}", file=sys.stderr)
            raise typer.Exit(4)
        log_path = Path(found)
    else:
        print(f"courant watch: no such case or log: {path}", file=sys.stderr)
        raise typer.Exit(4)

    def emit(line: str) -> None:
        print(line, flush=True)

    if not as_json:
        emit(f"Courant is watching {(case_dir or log_path).name} · {log_path.name} · read-only: "
             f"it reports, it never stops or edits the run")
    if once:
        chunks = [Chunk(text=log_path.read_text(errors="replace"), offset=0)]
    else:
        chunks = LogTail(log_path, idle_timeout=idle_timeout, start_timeout=start_timeout).follow()
    try:
        outcome = run_watch(chunks, case_dir=case_dir, log_path=log_path, emit=emit,
                            as_json=as_json, exit_on_fail=exit_on_fail, interval=interval,
                            once=once)
    except KeyboardInterrupt:
        print("\n  stopped watching; the run itself is untouched", file=sys.stderr)
        raise typer.Exit(130) from None
    raise typer.Exit(outcome.exit_code)
