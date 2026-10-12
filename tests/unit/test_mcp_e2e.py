"""MCP end to end, with no model (TRD GATE-5, GATE-6; T-GATE-MCP).

A scripted MCP client starts `courant-mcp` as a subprocess, talks to it over stdio, and calls
every tool on the demo case. `score_agent_output` is advisory: it returns verdict/2, the task
class and the silent-error flags, and a schema option (1|2, default 2) picks the verdict major.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from courant.mcp import tools

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def demo(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path / "demo")


def _spec(case: Path, out: Path) -> tuple[Path, str]:
    from courant.spec import draft_spec, spec_sha256

    path = out / "spec.json"
    path.write_text(draft_spec(case).to_json())
    return path, spec_sha256(path)


def _run(root: Path, calls: list[tuple[str, dict]]):
    """Start the server, call each tool in order, return [(is_error, structured or text)]."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "courant.mcp.server", "--root", str(root)],
                                   env=env, cwd=str(ROOT))

    async def go():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            out = []
            for name, args in calls:
                res = await session.call_tool(name, args)
                if res.structured_content is not None:
                    body = res.structured_content
                else:
                    body = "".join(getattr(c, "text", "") for c in res.content)
                out.append((res.is_error, body))
            return listed, out

    return asyncio.run(asyncio.wait_for(go(), timeout=120))


def _doc(body):
    return body.get("result", body) if isinstance(body, dict) else json.loads(body)


def test_a_scripted_client_calls_every_tool_on_the_demo_case(demo, tmp_path):
    spec, lock = _spec(demo, tmp_path)
    calls = [
        ("verify_run", {"path": str(demo), "spec": str(spec), "spec_sha256": lock}),
        ("verify_run", {"path": str(demo), "spec": str(spec), "spec_sha256": lock, "schema": 1}),
        ("verify_run", {"path": str(demo), "schema": 3}),
        ("score_agent_output", {"case": str(demo), "spec": str(spec), "spec_sha256": lock,
                                "reported_success": True}),
        ("check_convergence", {"path": str(demo)}),
        ("explain_code", {"code": "NUM.CFL_VIOLATION"}),
        ("list_failure_modes", {}),
        ("verifier_certification", {}),
        ("verify_run", {"path": "/etc/hosts"}),
    ]
    listed, results = _run(tmp_path, calls)

    names = sorted(t.name for t in listed.tools)
    assert names == ["check_convergence", "explain_code", "list_failure_modes",
                     "score_agent_output", "verifier_certification", "verify_run"]
    assert all(t.annotations.read_only_hint for t in listed.tools)

    (e1, v2), (e2, v1), (e3, bad), (e4, scored), (e5, conv), (e6, code), (e7, modes), \
        (e8, cert), (e9, refused) = results
    assert not any((e1, e2, e3, e4, e5, e6, e7, e8))
    d2, d1 = _doc(v2), _doc(v1)
    assert d2["schema"] == "courant.verdict/2" and d2["verdict"] == "accept"
    assert d2["basis"] == "validity+conformance"
    assert d1["schema"] == "courant.verdict/1" and d1["verdict"] == "TRUSTWORTHY"
    assert "error" in _doc(bad) and "1 or 2" in _doc(bad)["error"]
    s = _doc(scored)
    assert s["advisory"] is True and s["executed_source"] == "agent_log"
    assert s["verdict"]["schema"] == "courant.verdict/2" and s["verdict"]["verdict"] == "accept"
    assert s["task_class"] == s["outcome"] and s["silent_error"] is False
    assert _doc(conv)["status"]
    assert _doc(code)["code"] == "NUM.CFL_VIOLATION"
    assert len(_doc(modes)) == 25
    assert _doc(cert)["certified"] is True
    assert e9 and "outside the allowed roots" in str(refused)


def test_a_wrong_problem_is_a_silent_error_when_the_agent_reports_success(demo, tmp_path):
    spec, lock = _spec(demo, tmp_path)
    tp = demo / "constant" / "transportProperties"
    tp.write_text(tp.read_text().replace("1e-05", "3e-05"))
    args = {"case": str(demo), "spec": str(spec), "spec_sha256": lock}
    _listed, results = _run(tmp_path, [
        ("score_agent_output", {**args, "reported_success": True}),
        ("score_agent_output", {**args, "reported_success": False}),
        ("score_agent_output", args),
        ("score_agent_output", {**args, "spec_sha256": "0" * 64, "reported_success": True}),
    ])
    claimed, honest, silent, unlocked = (_doc(b) for _, b in results)
    assert claimed["verdict"]["verdict"] == "reject" and claimed["task_class"] == "nonconformant"
    assert claimed["silent_error"] is True and claimed["unverified_success"] is False
    assert honest["silent_error"] is False and honest["reported_success"] is False
    assert silent["silent_error"] is False and silent["reported_success"] is None
    assert unlocked["verdict"]["verdict"] == "insufficient"
    assert unlocked["unverified_success"] is True and unlocked["silent_error"] is False
    assert unlocked["task_class"] == "insufficient"


def test_score_agent_output_in_schema_1_returns_the_old_major(demo, tmp_path):
    spec, lock = _spec(demo, tmp_path)
    got = tools.score_agent_output(str(demo), str(spec), lock, True, 1,
                                   confine=tools.Confinement([tmp_path]))
    assert got["verdict"]["schema"] == "courant.verdict/1"
    assert got["verdict"]["verdict"] == "TRUSTWORTHY" and got["task_class"] == "valid_conformant"


def test_the_tool_stays_confined_and_read_only(demo, tmp_path):
    spec, lock = _spec(demo, tmp_path)
    confine = tools.Confinement([demo])
    with pytest.raises(tools.OutsideRoot):
        tools.score_agent_output(str(demo), str(spec), lock, True, confine=confine)


def test_an_unexpected_exception_is_one_line_never_a_traceback(demo, tmp_path, monkeypatch):
    from courant import __version__
    from courant.mcp.server import build

    def boom(*a, **k):
        raise RuntimeError("injected detail")

    monkeypatch.setattr(tools, "score_agent_output", boom)
    server = build([str(tmp_path)])
    spec, lock = _spec(demo, tmp_path)

    async def go():
        return await server.call_tool("score_agent_output",
                                      {"case": str(demo), "spec": str(spec), "spec_sha256": lock})

    with pytest.raises(Exception) as err:
        asyncio.run(go())
    msg = str(err.value)
    assert "internal error (RuntimeError)" in msg and __version__ in msg
    assert "Traceback" not in msg and "injected detail" not in msg


def test_print_config_gives_snippets_that_parse(tmp_path):
    from typer.testing import CliRunner

    from courant.cli.main import app

    res = CliRunner().invoke(app, ["mcp", "--print-config", "--root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    out = res.output
    claude_json = out[out.index("{"):out.rindex("}") + 1]
    cfg = json.loads(claude_json)["mcpServers"]["courant"]
    assert cfg["command"] == "courant" and cfg["args"][:1] == ["mcp"] and str(tmp_path) in cfg["args"]
    toml_block = out[out.index("[mcp_servers.courant]"):]
    parsed = tomllib.loads(toml_block)["mcp_servers"]["courant"]
    assert parsed["command"] == "courant" and parsed["args"] == cfg["args"]
    assert "claude mcp add courant" in out
