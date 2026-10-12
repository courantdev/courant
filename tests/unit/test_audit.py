"""The compute audit: accounting, denominators, money only at the user's rate (TRD AUDIT-1 to 4)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.cli.main import app
from courant.verify.audit import audit, core_hours, render_html, render_text

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
MON = ROOT / "certification" / "monitored"


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    for name in ("icoFoam_cavity__healthy", "icoFoam_cavity__transport_units", "icoFoam_cavity__missing_dict"):
        for ext in (".log.gz", ".case.json"):
            shutil.copy(CORPUS / f"{name}{ext}", d / f"{name}{ext}")
    shutil.copytree(MON / "simpleFoam_pitzDaily__fixed_budget", d / "pitz_fixed")
    return d


def test_core_hours_come_from_ranks_and_the_solver_clock():
    log = "nProcs : 8\n" + "".join(f"ExecutionTime = {i}.0 s  ClockTime = {i * 2} s\n" for i in range(1, 6))
    assert core_hours(log) == (8 * 10 / 3600, 8)
    assert core_hours("Exec : simpleFoam\n") == (None, 1)


def test_the_totals_add_up_with_their_denominators(folder):
    doc = audit(folder)
    t = doc["totals"]
    assert t["runs"] == 4 and t["readable"] == 4
    assert t["trustworthy"] + t["not_trustworthy"] + t["not_assessable"] == t["readable"]
    clocked = [r for r in doc["runs"] if r.get("core_hours") is not None]
    assert t["runs_with_clock"] == len(clocked)
    assert t["core_hours"] == pytest.approx(math.fsum(r["core_hours"] for r in clocked))
    untrusted = [r for r in clocked if r["verdict"] == "NOT_TRUSTWORTHY"]
    assert t["core_hours_not_trustworthy"] == pytest.approx(math.fsum(r["core_hours"] for r in untrusted))
    setup = [r for r in doc["runs"] if r.get("preflight") == "SETUP_FAULT"]
    assert t["runs_setup_fault"] == len(setup) >= 1          # transport_units: viscosity, before the run
    assert t["runs_settled"] == 1 and t["core_hours_after_convergence"] > 0   # the fixed-budget pitzDaily
    assert doc["money"] is None                                # no rate, no money


def test_money_appears_only_at_the_users_rate(folder):
    doc = audit(folder, cost_per_core_hour=0.05)
    m, t = doc["money"], doc["totals"]
    assert m["basis"] == "the user's own rate" and m["cost"] == pytest.approx(t["core_hours"] * 0.05)


def test_deterministic_read_only_and_self_contained(folder):
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    a, b = audit(folder), audit(folder)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before
    page = render_html(a)
    assert "<script" not in page.lower() and "http" not in page.split("<style>")[0].lower()
    assert "Compute audit" in page
    assert "Untrusted" in render_text(a) and "core-hours" in render_text(a)


def test_cli_exit_code_and_html(folder, tmp_path):
    r = CliRunner().invoke(app, ["audit", str(folder), "--html", str(tmp_path / "a.html")])
    assert r.exit_code == 2 and "Untrusted" in r.output and (tmp_path / "a.html").exists()
    assert CliRunner().invoke(app, ["audit", str(tmp_path / "nope")]).exit_code == 4
