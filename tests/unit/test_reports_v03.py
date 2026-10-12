"""HTML and JUnit reports are deterministic, self-contained and complete (TRD REP-6, REP-7)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from courant.verify import from_path, verify_inputs
from courant.verify.batch import verify_all
from courant.verify.html import render_html
from courant.verify.junit import render_junit

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__fixed_budget"
CORPUS = ROOT / "certification" / "corpus"


def _html(path=CASE):
    inputs = from_path(path)
    return render_html(verify_inputs(inputs), inputs), verify_inputs(inputs)


def test_html_is_deterministic_and_self_contained():
    a, v = _html()
    b, _ = _html()
    assert a == b
    assert "<script" not in a.lower()
    assert not re.search(r"(src|href)\s*=\s*['\"]?https?://", a)
    assert "@media (prefers-color-scheme:dark)" in a
    assert str(ROOT) not in a                                   # no absolute paths


def test_html_carries_the_verdict_the_convergence_and_every_input():
    page, v = _html()
    assert f'class="badge {v.verdict}"' in page
    assert f"step {v.convergence.converged_at_step:,}" in page
    for f in v.run.inputs:
        assert f.sha256 in page
    assert page.count("<svg") == 2                              # the monitor and the residuals
    assert "stopped changing" in page


def test_html_numbers_in_the_text_are_in_the_verdict():
    page, v = _html()
    text = re.sub(r"<svg.*?</svg>", "", page, flags=re.S)       # charts are drawn from inputs
    j = v.to_json()
    assert v.convergence.wasted_steps is not None
    for token in (str(v.convergence.wasted_steps), v.convergence.wasted_core_hours):
        assert token in j
        assert f"{int(token):,}" in text if token.isdigit() else token in text


def test_html_for_a_log_only_run_has_no_convergence_section():
    page, v = _html(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz")
    assert v.convergence is None and "<h2>Convergence</h2>" not in page
    assert "reject" in page and "<h2>Residuals</h2>" in page


SUBSET = ("icoFoam_cavity__healthy", "icoFoam_cavity__cfl_blowup", "pisoFoam_RAS_cavity__healthy",
          "icoFoam_cavity__missing_dict")


def _subset(tmp_path):
    import shutil

    for name in SUBSET:
        for ext in (".log.gz", ".case.json"):
            shutil.copy(CORPUS / f"{name}{ext}", tmp_path / f"{name}{ext}")
    return tmp_path


def test_junit_is_valid_deterministic_and_counts_add_up(tmp_path):
    d = _subset(tmp_path)
    results = verify_all(d)
    a, b = render_junit(results, d), render_junit(verify_all(d), d)
    assert a == b
    root = ET.fromstring(a)
    suite = root.find("testsuite")
    assert int(suite.get("tests")) == len(suite.findall("testcase")) == len(SUBSET)
    fails = [v for _, v, e in results if not e and v.verdict == "reject"]
    assert int(suite.get("failures")) == len(suite.findall("testcase/failure")) == len(fails) == 2
    assert "timestamp" not in a and "time=" not in a
    names = [c.get("name") for c in suite.findall("testcase")]
    assert names == sorted(names)


def test_junit_reports_input_errors_as_errors(tmp_path):
    bad = tmp_path / "broken.log"
    bad.write_bytes(b"")
    (tmp_path / "ok").mkdir()
    out = render_junit([(bad, None, "the log is empty")], tmp_path)
    case = ET.fromstring(out).find("testsuite/testcase")
    assert case.find("error").get("message") == "the log is empty"


def test_the_cli_writes_html_and_junit(tmp_path):
    from typer.testing import CliRunner

    from courant.cli.main import app

    r = CliRunner().invoke(app, ["verify", str(CASE), "--html", str(tmp_path / "r.html")])
    assert r.exit_code == 0 and (tmp_path / "r.html").read_text().startswith("<!doctype html>")
    d = _subset(tmp_path / "runs") if (tmp_path / "runs").mkdir() is None else None
    r = CliRunner().invoke(app, ["verify", "--batch", str(d), "--junit", str(tmp_path / "j.xml")])
    assert r.exit_code == 2 and ET.parse(tmp_path / "j.xml").getroot().tag == "testsuites"
