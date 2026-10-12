"""courant-mcp tools (TRD §8)."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import socket
from pathlib import Path

import pytest

from courant.mcp import tools

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"


@pytest.fixture
def confine(tmp_path):
    return tools.Confinement([tmp_path, CORPUS])


def test_verify_run_returns_the_same_verdict_json_as_the_cli(confine):
    from courant.verify import verify_path
    log = CORPUS / "icoFoam_cavity__cfl_blowup.log.gz"
    cj = CORPUS / "icoFoam_cavity__cfl_blowup.case.json"
    got = tools.verify_run(str(log), str(cj), confine=confine)
    import json
    assert got == json.loads(verify_path(log, cj).to_json())
    assert got["verdict"] == "reject" and got["schema"] == "courant.verdict/2"


def test_paths_outside_the_allowed_roots_are_refused(confine):
    with pytest.raises(tools.OutsideRoot):
        tools.verify_run("/etc/hosts", confine=confine)
    with pytest.raises(tools.OutsideRoot):
        tools.verify_run(str(CORPUS / ".." / ".." / "pyproject.toml"), confine=confine)


def test_a_missing_log_is_an_error_payload_not_a_crash(confine, tmp_path):
    assert "error" in tools.verify_run(str(tmp_path / "nope.log"), confine=confine)


def test_the_tools_are_read_only_and_offline(confine, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    shutil.copy(CORPUS / "icoFoam_cavity__healthy.log.gz", tmp_path / "x.log.gz")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    tools.verify_run(str(tmp_path / "x.log.gz"), confine=confine)
    tools.explain_code("NUM.CFL_VIOLATION")
    tools.list_failure_modes()
    tools.verifier_certification()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    assert before == after


def test_explain_code_says_whether_the_mode_is_certified():
    got = tools.explain_code("NUM.CFL_VIOLATION")
    assert got["certified_on_real_output"] is True and got["typical_corrections"]
    assert tools.explain_code("PHYS.TURB_REGIME")["certified_on_real_output"] is False
    assert "error" in tools.explain_code("NOT.A_CODE")


def test_the_certified_list_is_the_published_thirteen():
    assert len(tools.certified_codes()) == 13
    modes = tools.list_failure_modes()
    assert len(modes) == 25 and sum(m["certified_on_real_output"] for m in modes) == 13


def test_verifier_certification_reads_only_the_committed_results():
    got = tools.verifier_certification()
    assert got["certified"] is True
    assert set(got["false_accepts"]) == {"n", "of", "lo", "hi"}


def test_the_server_registers_six_read_only_tools():
    pytest.importorskip("mcp")
    from courant.mcp.server import build

    server = build([str(CORPUS)])
    listed = asyncio.run(server.list_tools())
    names = sorted(t.name for t in listed)
    assert names == ["check_convergence", "explain_code", "list_failure_modes",
                     "score_agent_output", "verifier_certification", "verify_run"]
    assert all(t.annotations.read_only_hint for t in listed)
