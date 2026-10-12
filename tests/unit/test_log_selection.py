"""Pick the solver's log, not the utility log that ran after it.

`find_log` took the newest `log.*` by modification time. That is wrong for the
workflow this product is aimed at: a parallel run is `blockMesh · decomposePar ·
mpirun solver · reconstructPar`, so the newest log in a *finished* parallel case
is `log.reconstructPar` — a utility log with no residuals in it.

Measured on a decomposed cavity case that had diverged to a Courant number of
1580: Courant followed `log.decomposePar` and reported **Trust Grade A**. A false
pass, in the terminal, in front of whoever is watching the demo.

The obvious content rule does not work either, which is why these tests exist:
`decomposePar` and `reconstructPar` both print `Time = <t>` per time directory.
Real `log.decomposePar` output from OpenFOAM v2406 contains one. So the banner
decides for known utilities and content is only the fallback.
"""

from __future__ import annotations

import os

from courant.cli.tail import _log_rank, find_log

BANNER = (
    "/*--------------------------------*- C++ -*----------------------------*\\\n"
    "| =========                 |                                          |\n"
    "\\*---------------------------------------------------------------------*/\n"
    "Build  : _be01ca78-20240625 OPENFOAM=2406\n"
    "Exec   : {app}\n"
    "Date   : Sep 01 2026\n"
)

SOLVER_LOG = BANNER.format(app="icoFoam -parallel") + """
Starting time loop

Time = 0.005
Courant Number mean: 0.0083 max: 0.2
smoothSolver:  Solving for Ux, Initial residual = 1, Final residual = 8e-06
End
"""

# Both of these genuinely print a Time line — that is the trap.
RECONSTRUCT_LOG = BANNER.format(app="reconstructPar") + """
Reconstructing fields for mesh region0

Time = 7
End
"""

DECOMPOSE_LOG = BANNER.format(app="decomposePar") + """
Time = 0

Processor 0: field transfer
End
"""


def _write(path: str, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text)


def test_the_solver_log_wins_even_when_a_utility_ran_after_it(tmp_path) -> None:
    case = str(tmp_path)
    _write(os.path.join(case, "log.icoFoam"), SOLVER_LOG)
    _write(os.path.join(case, "log.decomposePar"), DECOMPOSE_LOG)
    _write(os.path.join(case, "log.reconstructPar"), RECONSTRUCT_LOG)

    # reconstructPar is the newest, as it always is in a real parallel workflow.
    now = os.path.getmtime(os.path.join(case, "log.icoFoam"))
    os.utime(os.path.join(case, "log.icoFoam"), (now - 100, now - 100))
    os.utime(os.path.join(case, "log.decomposePar"), (now - 50, now - 50))
    os.utime(os.path.join(case, "log.reconstructPar"), (now, now))

    chosen = find_log(case)
    assert chosen is not None and chosen.name == "log.icoFoam", (
        f"followed {chosen.name if chosen else None} — a utility log has no "
        "residuals in it, so the run gets reported as clean"
    )


def test_utilities_that_print_a_time_line_are_still_utilities(tmp_path) -> None:
    """The reason content alone is not enough."""
    case = str(tmp_path)
    _write(os.path.join(case, "log.reconstructPar"), RECONSTRUCT_LOG)
    _write(os.path.join(case, "log.decomposePar"), DECOMPOSE_LOG)
    assert "Time = " in RECONSTRUCT_LOG and "Time = " in DECOMPOSE_LOG
    assert _log_rank(tmp_path / "log.reconstructPar") == 0
    assert _log_rank(tmp_path / "log.decomposePar") == 0


def test_an_unknown_application_outranks_a_utility(tmp_path) -> None:
    """A solver we have never heard of is still a better guess than blockMesh."""
    case = str(tmp_path)
    _write(os.path.join(case, "log.myCustomFoam"),
           BANNER.format(app="myCustomFoam") + "\nTime = 1\nEnd\n")
    _write(os.path.join(case, "log.blockMesh"), BANNER.format(app="blockMesh") + "End\n")
    chosen = find_log(case)
    assert chosen is not None and chosen.name == "log.myCustomFoam"


def test_newest_still_wins_between_two_solves(tmp_path) -> None:
    """The original behaviour, which is right: a re-run supersedes yesterday's."""
    case = str(tmp_path)
    _write(os.path.join(case, "log.icoFoam"), SOLVER_LOG)
    _write(os.path.join(case, "log.icoFoam.1"), SOLVER_LOG)
    base = os.path.getmtime(os.path.join(case, "log.icoFoam"))
    os.utime(os.path.join(case, "log.icoFoam.1"), (base - 500, base - 500))
    chosen = find_log(case)
    assert chosen is not None and chosen.name == "log.icoFoam"


# --- Only utility logs: there is no solver log, and the gate says so --------------------------
#
# Found on the kept certification case for `interFoam_damBreak::max_co_too_high`: the harness
# keeps the solver's log outside the case directory, so the case held log.blockMesh,
# log.setFields and log.checkMesh. `find_log` returned log.checkMesh, and `courant verify`
# judged a mesh report as the run: insufficient, VER-5.5, with no word that the solver's log
# was missing.

CHECKMESH_LOG = BANNER.format(app="checkMesh") + "\nTime = 0\n\nMesh stats\nEnd\n"


def _only_utilities(case: str) -> None:
    _write(os.path.join(case, "log.blockMesh"), BANNER.format(app="blockMesh") + "End\n")
    _write(os.path.join(case, "log.setFields"), BANNER.format(app="setFields") + "End\n")
    _write(os.path.join(case, "log.checkMesh"), CHECKMESH_LOG)


def test_only_utility_logs_is_no_solver_log(tmp_path) -> None:
    from courant.cli.tail import utility_logs

    _only_utilities(str(tmp_path))
    assert find_log(tmp_path) is None
    assert utility_logs(tmp_path) == ["log.blockMesh", "log.checkMesh", "log.setFields"]


def test_the_solver_log_wins_over_a_newer_checkmesh(tmp_path) -> None:
    case = str(tmp_path)
    _write(os.path.join(case, "log.interFoam"), SOLVER_LOG)
    _only_utilities(case)
    now = os.path.getmtime(os.path.join(case, "log.checkMesh"))
    os.utime(os.path.join(case, "log.interFoam"), (now - 100, now - 100))
    chosen = find_log(case)
    assert chosen is not None and chosen.name == "log.interFoam"


def test_verify_on_a_case_with_only_utility_logs_says_no_solver_log(tmp_path) -> None:
    from typer.testing import CliRunner

    from courant.cli.main import app

    _only_utilities(str(tmp_path))
    out = CliRunner().invoke(app, ["verify", str(tmp_path)])
    assert out.exit_code == 4, out.output
    assert "no solver log" in out.output
    assert "log.checkMesh" in out.output and "not the run's output" in out.output
