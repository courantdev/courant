"""Preflight: setup-only findings, verdicts, exit codes, the live step-0 alarm (TRD PRE-1 to PRE-3)."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.cli.main import app
from courant.verify import from_path, verify_inputs
from courant.verify.preflight import SETUP_FINDINGS, preflight_path, preflight_views

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
MINI = ROOT / "tests" / "fixtures" / "mini_case"


def _side(name):
    return CORPUS / f"{name}.case.json"


@pytest.mark.parametrize("name,code", [("icoFoam_cavity__transport_units", "physics.nu"),
                                       ("pisoFoam_RAS_cavity__loose_solver_tolerance", "solver.loose_tolerance")])
def test_a_setup_fault_is_named_before_the_run(name, code):
    p = preflight_path("unused", _side(name))
    assert p.verdict == "SETUP_FAULT"
    assert code in {r.code for r in p.reasons}
    assert all(r.code in SETUP_FINDINGS for r in p.reasons)
    assert p.next_action and p.next_action.startswith("Suggested, not applied")


@pytest.mark.parametrize("name", ["icoFoam_cavity__healthy", "pisoFoam_RAS_cavity__healthy",
                                  "simpleFoam_pitzDaily__healthy"])
def test_a_healthy_setup_is_not_flagged(name):
    assert preflight_path("unused", _side(name)).verdict == "NO_SETUP_FAULT_FOUND"


def test_no_configuration_is_not_assessable():
    p = preflight_views("x", None, None)
    assert p.verdict == "NOT_ASSESSABLE" and p.reasons == ()


@pytest.mark.parametrize("name", ["icoFoam_cavity__transport_units", "pisoFoam_RAS_cavity__loose_solver_tolerance",
                                  "simpleFoam_pitzDaily__overconstrained_pu"])
def test_every_preflight_finding_is_in_the_full_verdict(name):
    pre = {r.code for r in preflight_path("unused", _side(name)).reasons}
    full = verify_inputs(from_path(CORPUS / f"{name}.log.gz", _side(name)))
    assert pre <= {r.code for r in full.reasons}


def test_json_is_canonical_and_deterministic():
    a = preflight_path("unused", _side("icoFoam_cavity__transport_units")).to_json()
    b = preflight_path("unused", _side("icoFoam_cavity__transport_units")).to_json()
    assert a == b and json.loads(a)["schema"] == "courant.preflight/1"


def test_a_case_directory_is_read_read_only(tmp_path):
    case = tmp_path / "mini"
    shutil.copytree(MINI, case)
    before = {p: p.read_bytes() for p in case.rglob("*") if p.is_file()}
    p = preflight_path(case)
    assert p.verdict == "SETUP_FAULT" and "physics.nu" in {r.code for r in p.reasons}
    assert {p: p.read_bytes() for p in case.rglob("*") if p.is_file()} == before


def test_cli_exit_codes(tmp_path):
    r = CliRunner()
    assert r.invoke(app, ["preflight", "x", "--case-json", str(_side("icoFoam_cavity__transport_units"))]).exit_code == 2
    assert r.invoke(app, ["preflight", "x", "--case-json", str(_side("icoFoam_cavity__healthy"))]).exit_code == 0
    assert r.invoke(app, ["preflight", str(tmp_path / "nope")]).exit_code == 4


def test_watch_fails_at_step_zero_on_a_setup_fault(tmp_path):
    from courant.cli.tail import Chunk
    from courant.live.watch import run_watch

    case = tmp_path / "mini"
    shutil.copytree(MINI, case)
    text = gzip.open(CORPUS / "icoFoam_cavity__healthy.log.gz", "rt").read()
    lines: list[str] = []
    out = run_watch([Chunk(text=text, offset=0)], case_dir=case, log_path=case / "log.icoFoam",
                    emit=lines.append, exit_on_fail=True, clock=iter(range(10 ** 6)).__next__)
    assert out.exit_code == 2 and out.state.alarm.code == "live.setup" and out.state.alarm.step == 0
    assert any("setup fault found before the first timestep" in ln for ln in lines)


@pytest.mark.parametrize("mode", ["default", "next"])
def test_preflight_lists_what_it_read(tmp_path, mode):
    """C5 and C7 (VER-14). Preflight builds its manifest with the verifier's recorder: the 0/
    files the case-config reader parsed and the checkMesh log are listed, and a file nothing read
    is not. The test case and the observer are `test_verify_pack`'s."""
    from tests.unit.test_verify_pack import _case, _observe

    case = _case(tmp_path)
    with _observe(case) as opened:
        p = preflight_path(case, checks=mode)

    listed = [f.path for f in p.run.inputs]
    read = {"0/U", "log.checkMesh"}
    assert read <= set(listed), f"read but not listed: {sorted(read - set(listed))}; listed: {listed}"
    # constant/g: the regime's (VER-17), read only under --checks next (GATE-13).
    assert ("constant/g" in listed) == (mode == "next"), listed
    unread = {"system/decomposeParDict", "Allrun"}
    assert not unread & set(listed), f"listed, and nothing read them: {sorted(unread & set(listed))}"
    assert listed == sorted(opened), (f"read but not listed: {sorted(set(opened) - set(listed))}; "
                                      f"listed but not read: {sorted(set(listed) - set(opened))}")
