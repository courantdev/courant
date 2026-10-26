"""How a run ended, as far as its log or its caller says (one definition, TRD VER-3, VER-5.16).

OpenFOAM prints a bare `End` as the last line of a solver that ran to
completion. A converged steady run, a plateau that ran to its iteration limit,
and a transient run that reached its end time all print it. Calling a plateau
a crash made `crash.*` signatures reachable and pushed `BC.REVERSED_OUTFLOW`
into two diagnoses the real pipeline gets right.

A log that does not end in `End` is not evidence of a crash. It may be a run
still writing, a window someone copied out of a longer log, or a run stopped
from outside without a word from the solver. Until 0.6.0 this module returned
1 for every such log; the parser turned the 1 into a CRASH, and VER-5.2
rejected the run as crashed, citing "return code 1", a value no process had
returned: published windows of logs from runs that exited 0 were rejected as
crashes. So the code inferred here is 0 or unknown, never a nonzero value. A
crash is read: from a fatal signature in the log, which the parser recognises,
or from an exit code the caller gives. A log that shows neither is
`insufficient.truncated_log` (VER-5.16).

This was a private helper in the replay tier. The verifier needs the same
answer, and two copies of one fact is the bug class this codebase has fixed
seven times, so it lives here. A verdict's reading is `verdict_exit_code`, under
the checks it runs (VER-3a under `--checks next`): the pipeline decides VER-5.16
on it, and the same code is what the endTime item and the ended-short line read
(`courant.spec.as_run.Run.exit_code`), and what the HTML report and `courant
convergence` parse the log with. The replay tier and `courant watch` ask
`infer_exit_code` and `ends_in_end` here too. One reader does not yet:
`courant.verify.schema.executed_in_log`, SB-11's advisory `executed` (the MCP
`score_agent_output` and SilentBench's `grade`), still reads a line `End`
anywhere and no "FOAM FATAL" its own way (TRD VER-3; graph node T-EXECUTED-ONE,
since it feeds SilentBench's scoring).
"""

from __future__ import annotations

import re

_END_MARKER = "\nEnd\n"

#: Where a time step begins, and the blanks before it. A utility written to the same file ahead
#: of the solver (`(blockMesh && icoFoam) > log`) prints an `End` of its own before the first
#: one. OpenFOAM.org's foamMultiRun writes its `Time =` lines, and its `End`, behind a prefix of
#: blanks (`setGlobalPrefix`; the parser's `_RE_PREFIXED_TIME`, VER-20), so the blanks are read.
_TIME_STEP = re.compile(r"^( *)Time = \S", re.MULTILINE)

#: What a finished parallel run can print after its `End`: OpenFOAM's "Finalising parallel run",
#: and Open MPI's help-aggregation lines ("[host:pid] 3 more processes have sent help message
#: ...", "[host:pid] Set MCA parameter ... to see all help / error messages").
_AFTER_END = re.compile(
    r"Finalising parallel run"
    r"|\[[^\]\s]+:\d+\] (?:\d+ more process(?:es)? (?:has|have) sent help message"
    r"|Set MCA parameter)")


def _only_finalising(rest: str) -> bool:
    return all(not ln.strip() or _AFTER_END.match(ln.strip()) for ln in rest.split("\n"))


def ends_in_end(log_text: str, *, after_last_step: bool = False, last: bool = False) -> bool:
    """Whether the log shows a solver run that completed: OpenFOAM's `End`, last or on a line of
    its own (a parallel run prints "Finalising parallel run" after it).

    `after_last_step` is VER-3a, behind `--checks next` (GATE-13: it turns an accept into
    insufficient): only an `End` after the log's last time step counts, so a utility's `End`
    ahead of a cut solver's output is not taken for the solver's. The last step may sit behind
    blanks, and its `End` behind the same blanks, as OpenFOAM.org's foamMultiRun writes both
    (VER-20; a parallel one prints "Finalising parallel run" after it). A log with no time step
    is read as before.

    `last` is `courant watch`'s reading of a log still being written (LIVE-3): the last `End`
    counts only when nothing follows it but blank lines and what a finished parallel run prints
    after it, so a utility's `End` is not taken for the solver's while the solver's banner is
    still arriving. A line not yet finished counts as something that follows."""
    behind = None
    if after_last_step:
        step = None
        for step in _TIME_STEP.finditer(log_text):
            pass
        if step is not None:
            log_text = log_text[step.start():]
            if step.group(1):                   # foamMultiRun's End sits behind the same blanks
                behind = re.compile(rf"^{step.group(1)}End[ \t]*$", re.MULTILINE)
    stripped = log_text.rstrip()
    if not last:
        return stripped.endswith("End") or _END_MARKER in log_text or \
            (behind is not None and behind.search(log_text) is not None)
    if stripped == "End" or stripped.endswith("\nEnd"):
        return True
    at = log_text.rfind(_END_MARKER)
    return at >= 0 and _only_finalising(log_text[at + len(_END_MARKER):])


def infer_exit_code(log_text: str, *, after_last_step: bool = False) -> int | None:
    """0 if the log shows a completed solver run; otherwise None, unknown. Never a nonzero
    code: a log that stops holds no value a process returned."""
    return 0 if ends_in_end(log_text, after_last_step=after_last_step) else None


def run_exit_code(log_text: str, given: int | None, *,
                  after_last_step: bool = False) -> int | None:
    """The exit code the log parser is given (VER-1): the caller's, as read, when it gave one;
    otherwise the one inferred from the log. None means the run's end is unknown, and the
    verdict on it is insufficient unless the log shows a crash (VER-5.16)."""
    return given if given is not None else infer_exit_code(log_text,
                                                           after_last_step=after_last_step)


def verdict_exit_code(log_text: str, given: int | None, checks: str = "default") -> int | None:
    """The exit code a verdict reads under `checks` (`--checks`, GATE-13): `run_exit_code`, with
    VER-3a's reading of whose `End` it is when `log.end_after_last_step` is on (`--checks next`).
    One definition: the pipeline decides VER-5.16 on it and hands it to `as_run.Run`, so the
    endTime item (SPEC-13) and `sentences.ENDED_SHORT` read the end the verdict read; the HTML
    report and `courant convergence` parse the log with it."""
    from courant.verify.checks import enabled

    return run_exit_code(log_text, given,
                         after_last_step=enabled("log.end_after_last_step", checks))
