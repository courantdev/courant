"""The support bundle (TRD OPS-6): `courant doctor --bundle FILE` writes `courant.support/1`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from courant import __version__
from courant.cli.main import app
from courant.verify import support, verify_path

CORPUS = Path(__file__).resolve().parents[2] / "certification" / "corpus"


def test_the_bundle_has_versions_platform_command_and_verdict_and_nothing_else(tmp_path):
    log = CORPUS / "icoFoam_cavity__cfl_blowup.log.gz"
    verdict_file = tmp_path / "v.json"
    verdict_file.write_text(verify_path(log).to_json())
    out = tmp_path / "bundle.json"
    result = CliRunner().invoke(app, ["doctor", "--bundle", str(out), "--verdict",
                                      str(verdict_file)])
    assert result.exit_code == 0, result.output
    doc = json.loads(out.read_text())
    assert doc["schema"] == "courant.support/1"
    assert doc["courant"]["version"] == __version__
    assert {"system", "machine", "python"} <= set(doc["platform"])
    assert isinstance(doc["command_line"], list) and doc["command_line"]
    assert doc["verdict"]["schema"] == "courant.verdict/2" and doc["verdict"]["verdict"] == "reject"
    assert set(doc) == {"schema", "courant", "platform", "command_line", "verdict",
                         "pseudonymised"}


def test_the_bundle_refuses_a_file_that_is_not_a_verdict(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("secret notes")
    out = tmp_path / "bundle.json"
    result = CliRunner().invoke(app, ["doctor", "--bundle", str(out), "--verdict", str(other)])
    assert result.exit_code == 4 and not out.exists()
    assert "secret notes" not in result.output


def test_the_bundle_without_a_verdict_is_still_valid():
    doc = support.build(["courant", "doctor", "--bundle", "x"])
    assert doc["verdict"] is None and doc["schema"] == "courant.support/1"


def test_a_version_1_verdict_is_read_and_carried_as_version_2(tmp_path):
    log = CORPUS / "icoFoam_cavity__healthy.log.gz"
    f = tmp_path / "v1.json"
    f.write_text(verify_path(log).to_v1().to_json())
    doc = support.build(["courant"], f)
    assert doc["verdict"]["schema"] == "courant.verdict/2"


def test_the_cli_pseudonymise_flag_writes_a_pseudonymised_bundle(tmp_path):
    log = CORPUS / "icoFoam_cavity__cfl_blowup.log.gz"
    verdict_file = tmp_path / "v.json"
    verdict_file.write_text(verify_path(log).to_json())
    out = tmp_path / "b.json"
    result = CliRunner().invoke(app, ["doctor", "--bundle", str(out), "--verdict",
                                      str(verdict_file), "--pseudonymise"])
    assert result.exit_code == 0, result.output
    doc = json.loads(out.read_text())
    assert doc["pseudonymised"] is True and doc["verdict"]["run"]["id"] == "run1"
    assert "cfl_blowup" not in out.read_text()
    assert str(tmp_path) not in out.read_text()
