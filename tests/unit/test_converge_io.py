"""Monitor readers, restarts, the log's clock, and assess_case end to end (TRD CONV-3 to 5, 13)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from courant.converge.series import assess_case, discover, parse_table, ranks_from_log
from courant.watcher.parser import LogParser

COEFF_HEADER = """# Force and moment coefficients
# dragDir     : (1 0 0)
# sideDir     : (0 0 1)
# liftDir     : (0 1 0)
# rollAxis    : (1 0 0)
# pitchAxis   : (0 0 1)
# yawAxis     : (0 1 0)
# magUInf     : 26
# lRef        : 1
# Aref        : 1
# CofR        : (0.25 0 0)
#
# Time          Cd              Cd(f)           Cd(r)           Cl              Cl(f)           Cl(r)           CmPitch         CmRoll          CmYaw           Cs              Cs(f)           Cs(r)
"""


def _cd(i: int) -> float:
    return 0.0128 + 0.01 * math.exp(-i / 120.0)


def _cl(i: int) -> float:
    return 0.85 - 0.2 * math.exp(-i / 90.0)


def write_coefficients(path: Path, start: int, end: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [COEFF_HEADER.rstrip("\n")]
    for i in range(start, end + 1):
        cd, cl = _cd(i), _cl(i)
        vals = [cd, cd / 2, cd / 2, cl, cl / 2, cl / 2, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        lines.append(f"{i}\t" + "\t".join(f"{v:.8e}" for v in vals))
    path.write_text("\n".join(lines) + "\n")


def write_log(path: Path, n: int, nprocs: int | None = 4, solver: str = "simpleFoam") -> None:
    out = ["/*---------------------------------------------------------------------------*\\",
           "Build  : v2406", f"Exec   : {solver}" + (" -parallel" if nprocs else ""),
           "Date   : Sep 30 2026"]
    if nprocs:
        out.append(f"nProcs : {nprocs}")
    out += ["", "SIMPLE: convergence criteria", "", "Starting time loop", ""]
    for i in range(1, n + 1):
        r = 10.0 ** (-6.0 * i / n)
        out += [f"Time = {i}", "",
                f"smoothSolver:  Solving for Ux, Initial residual = {r:.6g}, Final residual = {r / 50:.6g}, No Iterations 2",
                f"smoothSolver:  Solving for Uy, Initial residual = {r:.6g}, Final residual = {r / 50:.6g}, No Iterations 2",
                f"GAMG:  Solving for p, Initial residual = {r * 3:.6g}, Final residual = {r / 30:.6g}, No Iterations 7",
                "time step continuity errors : sum local = 1e-06, global = 1e-09, cumulative = 1e-08",
                f"ExecutionTime = {0.05 * i:.2f} s  ClockTime = {int(0.06 * i)} s", ""]
    out += ["End", ""]
    path.write_text("\n".join(out))


def make_case(tmp_path: Path, n: int = 2000) -> Path:
    case = tmp_path / "airfoil"
    write_coefficients(case / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat", 1, n)
    write_log(case / "log.simpleFoam", n)
    return case


def _assess(case: Path, **kw):
    text = (case / "log.simpleFoam").read_text()
    events = LogParser(run_id="t").parse_text(text, return_code=0)
    return assess_case(case, log_text=text, events=events, **kw)


# ------------------------------------------------------------------ tables
def test_force_coefficients_table():
    case_text = COEFF_HEADER + "1\t" + "\t".join(["0.1"] * 12) + "\n2\t" + "\t".join(["0.2"] * 12) + "\n"
    t = parse_table(case_text)
    assert t.columns[:4] == ["Cd", "Cd(f)", "Cd(r)", "Cl"] and len(t.columns) == 12
    assert t.rows[1] == (2.0, [0.2] * 12)


def test_forces_with_vectors_and_the_legacy_header():
    modern = ("# Force\n# CofR : (0 0 0)\n#\n# Time total_x total_y total_z pressure_x pressure_y "
              "pressure_z viscous_x viscous_y viscous_z\n1 1 2 3 4 5 6 7 8 9\n")
    t_modern = parse_table(modern)
    assert t_modern.columns[0] == "total_x"
    assert t_modern.vectors["viscous_z"] == "viscous"      # separate columns are still a vector
    legacy = ("# Time forces(pressure viscous porous) moment(pressure viscous porous)\n"
              "1 ((1 2 3) (4 5 6) (0 0 0)) ((7 8 9) (1 1 1) (0 0 0))\n")
    t = parse_table(legacy)
    assert t.columns[:3] == ["forces_pressure_x", "forces_pressure_y", "forces_pressure_z"]
    assert t.vectors["moment_viscous_y"] == "moment_viscous"
    assert t.rows[0][1][:3] == [1.0, 2.0, 3.0]


def test_surface_field_value_and_probes():
    sfv = "# Region type : patch outlet\n# Faces : 30\n# Time        \tareaAverage(p)\n1\t-5.0\n2\t-5.1\n"
    assert parse_table(sfv).columns == ["areaAverage(p)"]
    probes = ("# Probe 0 (0.01 0.02 0)\n# Probe 1 (0.03 0.02 0)\n#       Probe             0             1\n"
              "#        Time\n1  (1 0 0)  (2 0 0)\n2  (1.1 0 0)  (2.1 0 0)\n")
    t = parse_table(probes)
    assert t.columns == ["probe0_x", "probe0_y", "probe0_z", "probe1_x", "probe1_y", "probe1_z"]


def test_bad_rows_are_counted_not_guessed():
    text = "# Time a b\n1 1 2\n2 1 x\n3 1\n4 1 2\n"
    t = parse_table(text)
    assert t.bad_rows == 2 and t.total_rows == 4 and len(t.rows) == 2


# ------------------------------------------------------------------ discovery
def test_discovery_reads_every_object_and_ignores_min_max(tmp_path):
    case = make_case(tmp_path, n=300)
    mm = case / "postProcessing" / "fieldMinMax1" / "0" / "fieldMinMax.dat"
    mm.parent.mkdir(parents=True)
    mm.write_text("# Time field min max\n1 p -1 1\n1 U 0 2\n")
    d = discover(case)
    assert {m.id for m in d.monitors} >= {"forceCoeffs1/coefficient:Cd", "forceCoeffs1/coefficient:CmPitch"}
    assert "postProcessing/fieldMinMax1" in d.ignored


def test_a_restart_keeps_the_later_trajectory(tmp_path):
    case = tmp_path / "c"
    write_coefficients(case / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat", 1, 500)
    write_coefficients(case / "postProcessing" / "forceCoeffs1" / "400" / "coefficient.dat", 401, 1000)
    cd = next(m for m in discover(case).monitors if m.id.endswith(":Cd"))
    assert cd.times[0] == 1.0 and cd.times[-1] == 1000.0 and len(cd.times) == 1000
    f = next(f for f in cd.reader_findings if f.code == "CONV.RESTART_OVERLAP")
    assert f.value["discarded_samples"] == 100 and cd.not_assessable is None


def test_mostly_overlapping_restarts_are_not_assessable(tmp_path):
    case = tmp_path / "c"
    write_coefficients(case / "postProcessing" / "fc" / "0" / "coefficient.dat", 1, 900)
    write_coefficients(case / "postProcessing" / "fc" / "10" / "coefficient.dat", 11, 300)
    cd = next(m for m in discover(case).monitors if m.id.endswith(":Cd"))
    assert cd.not_assessable == "RESTART_OVERLAP_MAJORITY"


def test_the_log_names_its_rank_count():
    assert ranks_from_log("Exec   : simpleFoam -parallel\nnProcs : 16\n") == (16, "log_nprocs")
    assert ranks_from_log("Exec   : simpleFoam\n") == (1, "serial")


# ------------------------------------------------------------------ end to end
def test_assess_case_finds_the_stopping_point_and_the_waste(tmp_path):
    case = make_case(tmp_path)
    conv = _assess(case)
    assert conv.status == "PASS", [(f.code, f.severity) for f in conv.findings]
    d = conv.data
    assert d["regime"] == "steady" and d["regime_basis"] == "solver"
    assert d["binding_monitor"].startswith("forceCoeffs1/coefficient:")
    w = d["waste"]
    assert 0.4 < w["wasted_fraction"] < 0.9
    assert w["ranks"] == 4 and w["wasted_wall_clock_s"] is not None
    assert w["wasted_core_hours"] == pytest.approx(4 * w["wasted_wall_clock_s"] / 3600)
    cd = d["monitors"]["forceCoeffs1/coefficient:Cd"]
    assert cd["tolerance"] == {"branch": "absolute", "value": 1e-4, "source": "default"}
    assert cd["delta_counts"] is not None and cd["delta_counts"] < 1.0
    assert [i.path for i in conv.inputs] == ["postProcessing/forceCoeffs1/0/coefficient.dat"]


def test_celik_orders_come_from_the_log(tmp_path):
    conv = _assess(make_case(tmp_path))
    res = conv.data["residuals"]
    assert set(res) == {"Ux", "Uy", "p"}
    assert res["Ux"]["orders_dropped_floor"] == 5      # 1e-0.003 -> 1e-6, the ref is sample 2
    f = next(f for f in conv.findings if f.code == "CONV.CELIK_RESIDUAL_ORDERS")
    assert f.severity == "INFO"


def test_a_log_without_a_case_has_no_monitors(tmp_path):
    case = make_case(tmp_path)
    text = (case / "log.simpleFoam").read_text()
    events = LogParser(run_id="t").parse_text(text, return_code=0)
    conv = assess_case(None, log_text=text, events=events)
    assert conv.status == "NOT_ASSESSABLE"
    assert conv.data["residuals"]


def test_a_user_tolerance_is_recorded(tmp_path):
    from courant.converge.assess import Options

    conv = _assess(make_case(tmp_path), options=Options(tol_abs={"Cd": 1e-6}))
    cd = conv.data["monitors"]["forceCoeffs1/coefficient:Cd"]
    assert cd["tolerance"] == {"branch": "absolute", "value": 1e-6, "source": "user"}
    assert conv.data["options_used"]["tol_abs"] == {"Cd": 1e-6}


def test_assessment_is_read_only(tmp_path):
    case = make_case(tmp_path, n=400)
    before = {p: p.read_bytes() for p in case.rglob("*") if p.is_file()}
    _assess(case)
    after = {p: p.read_bytes() for p in case.rglob("*") if p.is_file()}
    assert before == after


# ------------------------------------------------------------------ the CLI and MCP (CONV-13)
def _cli(args):
    from typer.testing import CliRunner

    from courant.cli.main import app

    return CliRunner().invoke(app, ["convergence", *args])


def test_cli_prints_the_five_lines_in_order(tmp_path):
    res = _cli([str(make_case(tmp_path))])
    assert res.exit_code == 0, res.output
    keys = ["1  Answer", "2  Change", "3  Iterations", "4  Wall clock", "5  Core-hours"]
    pos = [res.output.index(k) for k in keys]
    assert pos == sorted(pos)
    assert "of 1,999 past convergence" in res.output        # every share has its denominator


def test_cli_exit_codes(tmp_path):
    case = make_case(tmp_path)
    assert _cli([str(case / "log.simpleFoam")]).exit_code == 3       # a log alone: no monitors
    assert _cli([str(case), "--tol", "Cd"]).exit_code == 4           # malformed NAME=VALUE
    assert _cli([str(case), "--tol", "Cd=1e-4", "--tol-rel", "Cd=0.01"]).exit_code == 4
    drift = tmp_path / "drift"
    p = drift / "postProcessing" / "sfv" / "0" / "surfaceFieldValue.dat"
    p.parent.mkdir(parents=True)
    p.write_text("# Time areaAverage(p)\n" + "".join(f"{i} {1 + 1e-3 * i}\n" for i in range(1, 2001)))
    write_log(drift / "log.simpleFoam", 2000)
    res = _cli([str(drift)])
    assert res.exit_code == 2 and "did not reach a fixed point" in res.output


def test_cli_json_is_canonical_and_deterministic(tmp_path):
    import json

    case = make_case(tmp_path)
    a, b = _cli([str(case), "--json"]), _cli([str(case), "--json"])
    assert a.exit_code == 0 and a.output == b.output
    doc = json.loads(a.output)
    assert doc["schema"] == "courant.convergence/1"
    assert doc["convergence"]["status"] == "PASS"
    assert a.output == json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def test_mcp_check_convergence_is_confined(tmp_path):
    from courant.mcp import tools

    case = make_case(tmp_path)
    out = tools.check_convergence(str(case), confine=tools.Confinement([tmp_path]))
    assert out["status"] == "PASS" and out["wasted_fraction"] > 0.4
    with pytest.raises(tools.OutsideRoot):
        tools.check_convergence("/etc", confine=tools.Confinement([tmp_path]))


def test_the_default_monitors_are_totals_not_splits(tmp_path):
    from courant.converge.assess import Options

    conv = _assess(make_case(tmp_path, n=400))
    mons = set(conv.data["monitors"])
    assert "forceCoeffs1/coefficient:Cd" in mons and "forceCoeffs1/coefficient:Cd(f)" not in mons
    assert "forceCoeffs1/coefficient:Cd(f)" in conv.data["monitors_unselected"]
    assert conv.data["monitor_selection"] == "default_totals"
    only = _assess(make_case(tmp_path / "b", n=400), options=Options(monitors=("*:Cd",)))
    assert set(only.data["monitors"]) == {"forceCoeffs1/coefficient:Cd"}
    assert only.data["monitor_selection"] == "user_patterns"


def test_a_period_two_alternation_is_the_iteration_not_the_flow():
    import random

    from courant.converge.assess import Clock, MonitorSeries, RunMeta, assess

    rng = random.Random(1)
    vals = [1.6 + 0.2 * math.exp(-i / 60) + (1.0 if i % 2 else -1.0) * a + 1e-7 * rng.gauss(0, 1)
            for i, a in enumerate([4e-5] * 1500)]
    mon = MonitorSeries(id="fc/coefficient:Cl", column="Cl", source="s",
                        times=[float(i + 1) for i in range(1500)], values=vals)
    conv = assess([mon], {}, Clock(), RunMeta(regime="steady"))
    assert conv.status == "PASS"           # 8e-5 peak to peak sits inside one count: not judged
    big = [v + (1.0 if i % 2 else -1.0) * 2e-4 for i, v in enumerate(vals)]
    conv = assess([MonitorSeries(id=mon.id, column="Cl", source="s", times=mon.times, values=big)],
                  {}, Clock(), RunMeta(regime="steady"))
    f = next(f for f in conv.findings if f.code == "CONV.ITERATION_OSCILLATION")
    assert f.severity == "FAIL" and "numerical oscillation of the steady iteration" in f.message
    assert not any(f.code == "CONV.UNSTEADY_STEADY" for f in conv.findings)


# ------------------------------------------------------------------ the verdict (CONV-12)
def test_a_case_verdict_carries_the_convergence_block(tmp_path):
    import json

    from courant.verify import verify_path

    v = verify_path(make_case(tmp_path))
    assert v.convergence is not None and v.convergence.status == "PASS"
    doc = json.loads(v.to_json())
    assert doc["convergence"]["converged_at_step"] == v.convergence.converged_at_step
    assert any(i["path"].startswith("postProcessing/") for i in doc["run"]["inputs"])


def test_a_log_only_verdict_keeps_its_v0_shape(tmp_path):
    import json

    from courant.verify import verify_path

    v = verify_path(make_case(tmp_path) / "log.simpleFoam")
    assert v.convergence is None
    assert "convergence" not in json.loads(v.to_json())


def test_convergence_does_not_change_the_verdict_while_ver_5_4b_is_off(tmp_path):
    from courant.verify import verify_path

    drift = tmp_path / "drift"
    p = drift / "postProcessing" / "sfv" / "0" / "surfaceFieldValue.dat"
    p.parent.mkdir(parents=True)
    p.write_text("# Time areaAverage(p)\n" + "".join(f"{i} {1 + 1e-3 * i}\n" for i in range(1, 2001)))
    write_log(drift / "log.simpleFoam", 2000)
    with_monitor = verify_path(drift)
    p.unlink()
    without = verify_path(drift)
    assert with_monitor.convergence.status == "FAIL"
    assert (with_monitor.verdict, with_monitor.rule) == (without.verdict, without.rule)


def test_a_pack_rehashes_the_monitors(tmp_path):
    from courant.verify import verify_path
    from courant.verify.pack import check_pack, write_pack

    case = make_case(tmp_path)
    out = write_pack(verify_path(case), tmp_path / "pack", case.name, None)
    ok, msg = check_pack(out, case)
    assert ok, msg
    f = case / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"
    f.write_text(f.read_text().replace("1.28", "1.29", 1))
    ok, msg = check_pack(out, case)
    assert not ok and "coefficient.dat" in msg


# ------------------------------------------------------- one manifest for one run (VER-14)
def _convergence_inputs(case: Path) -> list[dict]:
    import json

    from typer.testing import CliRunner

    from courant.cli.main import app

    result = CliRunner().invoke(app, ["convergence", str(case), "--json"])
    assert result.exit_code in (0, 2, 3), (result.exit_code, result.output)
    return json.loads(result.output)["convergence"]["inputs"]


def test_convergence_lists_the_same_files_as_the_verdict(tmp_path):
    """The convergence document's inputs come from the verdict's manifest function. On the demo
    case (a real simpleFoam/pitzDaily run, where the verdict reads nothing convergence does not)
    the two lists are equal. On the test case the verdict lists, beyond them, only the field
    file the field reader read."""
    from courant.data import extract_demo_case
    from courant.verify import verify_path
    from tests.unit.test_verify_pack import _case, _manifest

    demo = extract_demo_case(tmp_path / "demo")
    assert _convergence_inputs(demo) == _manifest(verify_path(demo))

    case = _case(tmp_path)
    conv = {e["path"]: e for e in _convergence_inputs(case)}
    verdict = {e["path"]: e for e in _manifest(verify_path(case))}
    assert all(verdict.get(p) == e for p, e in conv.items()), sorted(set(conv) - set(verdict))
    assert sorted(set(verdict) - set(conv)) == ["0.5/U"]


def test_convergence_inputs_move_with_the_deck_and_list_an_unparsed_monitor(tmp_path):
    """The regime comes from `system/fvSchemes`, so editing it changes the convergence inputs.
    A monitor that was read and does not parse is in both documents."""
    from courant.data import extract_demo_case
    from courant.verify import verify_path

    demo = extract_demo_case(tmp_path / "demo")
    junk = demo / "postProcessing" / "courantJunk" / "0" / "junk.dat"
    junk.parent.mkdir(parents=True)
    junk.write_text("# a monitor header\n# and nothing else\n")
    before = {e["path"]: e for e in _convergence_inputs(demo)}
    rel = "postProcessing/courantJunk/0/junk.dat"
    assert rel in before and rel in {f.path for f in verify_path(demo).run.inputs}

    schemes = demo / "system" / "fvSchemes"
    schemes.write_text(schemes.read_text().replace("steadyState", "Euler"))
    after = {e["path"]: e for e in _convergence_inputs(demo)}
    assert after["system/fvSchemes"]["sha256"] != before["system/fvSchemes"]["sha256"]
