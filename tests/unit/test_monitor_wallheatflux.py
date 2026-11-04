"""T-MONITOR-WALLHEATFLUX: monitor discovery tells a per-patch table from a convergence monitor.

v0.6.0's monitor discovery (`courant.converge.series.discover`) opens every
`postProcessing/<name>/<t>/*.dat` and takes each column after time for a monitor. v2406's
`wallHeatFlux` function object writes one row per wall patch at each time under the header
`Time patch min max integral` (`writeFileHeader`, the columns
`courant.verification._heat_reader.COLUMNS` names), so a committed buoyantSimpleFoam solve is
read as four monitors none of whose rows parse (found by T-ENERGY-BALANCE's test-author, 2 Nov
2026). Under the default that reading is v0.6.0's and stays (GATE-13). Behind `--checks next`
(`monitors.per_patch`, its own name, 0.7.0: it moves runs with no energy equation too, so it is
not VER-22's) such a table, and any table whose rows carry a word second (a patch name, or
solverInfo's solver name), is listed as ignored and never taken for a monitor.

Every case is a committed solve copied into `tmp_path` first (TST-4).
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from courant.converge.series import discover
from tests.unit.test_mass_flux import _unpack_fixture

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "tests" / "fixtures" / "real_openfoam"
HOT_BOX = REAL / "hot_box" / "converged"
CAVITY = REAL / "heated_cavity" / "run"
TABLE = "postProcessing/wallHeatFlux/0/wallHeatFlux.dat"
NAME = "monitors.per_patch"


def _hot_box(tmp: Path) -> Path:
    return _unpack_fixture(HOT_BOX, tmp / HOT_BOX.name)


def _hot_box_header() -> list[str]:
    """The two comment lines v2406's wallHeatFlux wrote for the committed hot_box solve."""
    with gzip.open(HOT_BOX / "postProcessing/wallHeatFlux/0/wallHeatFlux.dat.gz", "rt") as fh:
        return [fh.readline().rstrip("\n"), fh.readline().rstrip("\n")]


def _cavity(tmp: Path, *, table: bool) -> Path:
    """The heated cavity copied, with (`table`) a v2406-shaped wallHeatFlux table for its three
    walls: hot_box's own header, one row per patch at each of the run's 200 iterations."""
    case = _unpack_fixture(CAVITY, tmp / CAVITY.name)
    if table:
        lines = _hot_box_header()
        for t in range(1, 201):
            for patch, q in (("hot", 0.15), ("cold", -0.15), ("adiabatic", 0.0)):
                lines.append(f"{t}                 \t{patch}\t{q:.10e}\t{q:.10e}\t{q / 10:.10e}")
        out = case / TABLE
        out.parent.mkdir(parents=True)
        out.write_text("\n".join(lines) + "\n")
    return case


def _doc(case: Path, checks: str) -> dict:
    from courant.verify import from_case_dir, verify_inputs

    return json.loads(verify_inputs(from_case_dir(case, checks=checks)).to_json())


def test_the_name_is_its_own_and_ships_in_070():
    from courant.verify.checks import NEXT

    assert NEXT[NAME] == "0.7.0"


def test_the_heat_balance_flag_does_not_decide_discovery(tmp_path, monkeypatch):
    """The behaviour asks its own name: with only VER-22's name behind the flag, discovery under
    the default sets the table aside (this name is on); with this one behind it, it does not."""
    from courant.verify import checks

    case = _hot_box(tmp_path)
    monkeypatch.setattr(checks, "NEXT", {"conservation.energy.wallheatflux": "0.7.0"})
    assert discover(case).monitors == []
    monkeypatch.setattr(checks, "NEXT", {NAME: "0.7.0"})
    assert len(discover(case).monitors) == 4


def test_the_table_header_is_the_heat_readers_one_fact():
    """The header v2406 wrote in the committed solve is the one `_heat_reader.COLUMNS` names."""
    from courant.verification._heat_reader import COLUMNS

    assert tuple(_hot_box_header()[1].lstrip("#").split()) == COLUMNS


def test_the_default_reads_the_table_as_four_monitors_as_v060_did(tmp_path):
    """GATE-13: v0.6.0's reading, kept under the default (pinned here so a change is seen)."""
    case = _hot_box(tmp_path)
    found = discover(case)
    assert [m.id for m in found.monitors] == [
        f"wallHeatFlux/wallHeatFlux:{c}" for c in ("patch", "min", "max", "integral")]
    assert all(m.not_assessable == "UNREADABLE_ROWS" for m in found.monitors)
    assert [p.relative_to(case).as_posix() for p in found.files] == [TABLE]
    assert found.ignored == []


def test_next_lists_the_heat_table_as_ignored_and_finds_no_monitor(tmp_path):
    case = _hot_box(tmp_path)
    found = discover(case, checks="next")
    assert found.monitors == [] and found.files == []
    assert found.ignored == [TABLE]


@pytest.mark.parametrize("checks", ["default", "next"])
def test_discovery_refuses_an_unknown_mode_only_as_enabled_does(tmp_path, checks):
    discover(_hot_box(tmp_path), checks=checks)
    with pytest.raises(ValueError):
        discover(_hot_box(tmp_path / "x"), checks="nope")


def test_the_cavity_default_counts_four_monitors_and_next_none(tmp_path):
    """The finding as recorded (T-ENERGY-BALANCE log, test-author note 3): on a heated_cavity copy
    with a v2406-shaped table the default's `monitors_found` goes 0 to 4. Under next the
    convergence block is the one of the same copy with no table."""
    bare = _doc(_cavity(tmp_path / "bare", table=False), "default")
    with_table = _doc(_cavity(tmp_path / "with", table=True), "default")
    assert bare["convergence"]["monitors_found"] == 0
    assert with_table["convergence"]["monitors_found"] == 4

    bare_next = _doc(_cavity(tmp_path / "bare_next", table=False), "next")
    next_doc = _doc(_cavity(tmp_path / "with_next", table=True), "next")
    assert next_doc["convergence"] == bare_next["convergence"]
    assert next_doc["convergence"]["monitors_found"] == 0


def test_hot_box_under_next_has_the_convergence_block_of_no_table(tmp_path):
    case = _hot_box(tmp_path / "with")
    bare = _hot_box(tmp_path / "bare")
    shutil.rmtree(bare / "postProcessing")
    with_doc, bare_doc = _doc(case, "next"), _doc(bare, "next")
    assert with_doc["convergence"] == bare_doc["convergence"]
    assert with_doc["convergence"]["monitors_found"] == 0
    # read (its header decided it, and VER-22's reader reads it): so listed (VER-14)
    assert TABLE in {i["path"] for i in with_doc["run"]["inputs"]}


def test_the_default_document_does_not_move(tmp_path):
    """The default document of hot_box keeps v0.6.0's four monitors and lists the table."""
    doc = _doc(_hot_box(tmp_path), "default")
    assert doc["convergence"]["monitors_found"] == 4
    assert doc["convergence"]["status"] == "NOT_ASSESSABLE"
    assert TABLE in {i["path"] for i in doc["run"]["inputs"]}


def _write(case: Path, rel: str, lines: list[str]) -> None:
    out = case / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def test_a_patch_header_whose_names_parse_as_numbers_is_not_read_as_restarts(tmp_path):
    """A `Time patch ...` table whose patch names happen to parse as numbers reads, by default,
    as one row per patch, each time going back: restarts (CONV.RESTART_OVERLAP). Under next it is
    a per-patch table by its header, and ignored."""
    case = tmp_path / "c"
    rows = [f"{t}\t{p}\t1.0\t2.0\t{0.1 * p}" for t in range(1, 101) for p in (1, 2, 3)]
    _write(case, "postProcessing/q/0/q.dat", ["# Time patch min max integral", *rows])
    default = discover(case)
    assert default.monitors and any(
        f.code == "CONV.RESTART_OVERLAP" for m in default.monitors for f in m.reader_findings)
    nxt = discover(case, checks="next")
    assert nxt.monitors == [] and nxt.ignored == ["postProcessing/q/0/q.dat"]


def test_any_table_whose_second_column_is_a_patch_name_is_ignored_under_next(tmp_path):
    """v2406's yPlus (`Time patch min max average`) and a table with no header whose every row
    names a patch second: both per-patch, neither a monitor under next."""
    case = tmp_path / "c"
    _write(case, "postProcessing/yPlus/0/yPlus.dat",
           ["# y+ ()", "# Time\tpatch\tmin\tmax\taverage",
            *(f"{t}\twalls\t0.5\t3.0\t1.2" for t in range(1, 50))])
    _write(case, "postProcessing/bare/0/bare.dat",
           [*(f"{t}\tinlet\t0.5\t3.0" for t in range(1, 50)), "50"])
    assert {m.id.split(":")[0] for m in discover(case).monitors} == {"bare/bare", "yPlus/yPlus"}
    nxt = discover(case, checks="next")
    assert nxt.monitors == []
    assert nxt.ignored == ["postProcessing/bare/0/bare.dat", "postProcessing/yPlus/0/yPlus.dat"]


def test_a_monitor_with_one_stray_word_stays_a_monitor_under_next(tmp_path):
    """No over-reach: a forces-style table whose first row carries a stray word is a monitor in
    both modes, its readable rows the same."""
    case = tmp_path / "c"
    _write(case, "postProcessing/forces/0/force.dat",
           ["# Time total_x total_y total_z", "1 oops 0 0",
            *(f"{t} {1.0 / t} 0 0" for t in range(2, 400))])
    default, nxt = discover(case), discover(case, checks="next")
    assert [m.id for m in nxt.monitors] == [m.id for m in default.monitors]
    assert [m.values for m in nxt.monitors] == [m.values for m in default.monitors]
    assert nxt.ignored == default.ignored == []


@pytest.mark.parametrize("checks", ["default", "next"])
def test_the_html_page_charts_what_the_verdict_found(tmp_path, monkeypatch, checks):
    """The HTML page's convergence chart asks the discovery the verdict ran (its `--checks`),
    never a second reading of its own."""
    from courant.converge import series
    from courant.verify import from_case_dir
    from courant.verify.html import _series_from_inputs

    asked = []
    real = series.discover

    def watched(case_dir, checks="default"):
        asked.append(checks)
        return real(case_dir, checks=checks)

    monkeypatch.setattr(series, "discover", watched)
    _series_from_inputs(from_case_dir(_hot_box(tmp_path), checks=checks))
    assert asked and set(asked) == {checks}


def test_a_solverinfo_table_is_set_aside_under_next(tmp_path):
    """v2406's solverInfo writes the solver's name second (`# Time U_solver Ux_initial ...`,
    rows `1 smoothSolver 0.01 ...`): by default every row is unreadable, so its columns are
    monitors none of which can be assessed; under next it is a table with a word second, and
    ignored. (A Foundation table's `N/A` residuals are words too.)"""
    case = tmp_path / "c"
    rows = [f"{t}\tsmoothSolver\t{0.1 / t:.6e}\t{0.1 / t:.6e}\t2\tfalse" for t in range(1, 80)]
    _write(case, "postProcessing/solverInfo1/0/solverInfo.dat",
           ["# Solver information",
            "# Time\tU_solver\tUx_initial\tUx_final\tUx_iters\tU_converged", *rows])
    default = discover(case)
    assert default.monitors and all(m.not_assessable == "UNREADABLE_ROWS"
                                    for m in default.monitors)
    nxt = discover(case, checks="next")
    assert nxt.monitors == []
    assert nxt.ignored == ["postProcessing/solverInfo1/0/solverInfo.dat"]
