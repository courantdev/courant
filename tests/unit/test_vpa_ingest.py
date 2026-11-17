"""The verdict-path audit's log-choice findings (research/notes/verdict-path-audit/FINDINGS.md).

VPA-INGEST-1: a solver run with `-postProcess` writes a newer log that prints `Time =` for each
time it processes; `find_log` ranked it level with the solve and, newer, took it for the run's
log. A crashed run then read as insufficient (VER-5.5), or, with a function that solves an
equation, accepted. Under `--checks next` (`log.post_process`) it is a utility's log. The
default keeps 0.7.0's choice (GATE-13).

VPA-INGEST-1b: `_log_rank` read only a log's first `Exec` line, so `(blockMesh; simpleFoam) >
log.simpleFoam` was a utility's log and the case had "no solver log" (exit 4). Under next every
Exec line is read.
"""

from __future__ import annotations

import os

from courant.cli.tail import find_log

_SOLVE = """\
Exec   : simpleFoam
nProcs : 1
Starting time loop

Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.01, No Iterations 2
End
"""

_POST = """\
Exec   : simpleFoam -postProcess -func scalarTransport
nProcs : 1

Time = 1

smoothSolver:  Solving for s, Initial residual = 1e-09, Final residual = 1e-10, No Iterations 1
End
"""

_BLOCKMESH_THEN_SOLVE = """\
Exec   : blockMesh
nProcs : 1
End

Exec   : simpleFoam
nProcs : 1
Starting time loop

Time = 1

smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 0.01, No Iterations 2
End
"""


def _case(tmp_path, files):
    for i, (name, text) in enumerate(files):
        p = tmp_path / name
        p.write_text(text)
        os.utime(p, (1_000_000 + i, 1_000_000 + i))      # later in the list is newer
    return tmp_path


def test_vpa_ingest_1_a_post_process_run_of_the_solver_is_not_the_runs_log_under_next(tmp_path):
    case = _case(tmp_path, [("log.simpleFoam", _SOLVE), ("log.scalarTransport", _POST)])
    assert find_log(case, checks="next").name == "log.simpleFoam"
    # The default is 0.7.0's: the newer post-processing log wins (the finding, kept by GATE-13).
    assert find_log(case).name == "log.scalarTransport"


def test_vpa_ingest_1_a_post_process_log_alone_is_no_solver_log_under_next(tmp_path):
    case = _case(tmp_path, [("log.wallHeatFlux", _POST)])
    assert find_log(case, checks="next") is None


def test_vpa_ingest_1b_a_log_that_runs_blockmesh_then_the_solver_is_the_solves(tmp_path):
    case = _case(tmp_path, [("log.run", _BLOCKMESH_THEN_SOLVE)])
    assert find_log(case, checks="next").name == "log.run"
    assert find_log(case) is None                       # 0.7.0's default: the first Exec line
