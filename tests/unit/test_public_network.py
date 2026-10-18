"""Network honesty, held by tests that run in this repository (TRD SEC-1, INV-5; docs/NETWORK.md).

1. `courant verify` on the demo runs that ship inside the package opens no socket of any kind and
   starts no process: each of its outputs (JSON, HTML, Markdown, an evidence pack and its
   re-check, a record, a batch, the published error rates) and the Python call behind it.
2. No other visible command opens a network connection unless that is its documented purpose,
   the user asked for it, and it names what it contacts. Every visible command, every `spec`
   subcommand and every MCP tool runs here on the demo data with network connections refused,
   and none tries to connect.
3. The commands whose purpose is a network call (`courant update`, `courant doctor --models`, a
   language-model judge, `courant spec extract`) run blocked too: each names its host and then
   stops or reports cleanly, never a traceback, or contacts nothing when the call would be
   refused anyway.
4. docs/NETWORK.md lists every command that contacts anything.

The model-free half of the same promise is tests/unit/test_public_verdict_path.py: no module the
verdict is made of imports a language-model client or a network library.
"""

from __future__ import annotations

import concurrent.futures
import errno
import multiprocessing
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.cli.main import app
from courant.data import DEMO

ROOT = Path(__file__).resolve().parents[2]
HEALTHY = DEMO / "icoFoam_cavity__healthy.log.gz"
HEALTHY_CASE = DEMO / "icoFoam_cavity__healthy.case.json"
BLOWUP = DEMO / "icoFoam_cavity__cfl_blowup.log.gz"
BLOWUP_CASE = DEMO / "icoFoam_cavity__cfl_blowup.case.json"
UNITS = DEMO / "icoFoam_cavity__transport_units.log.gz"
UNITS_CASE = DEMO / "icoFoam_cavity__transport_units.case.json"
FAKE_KEY = "test-value-not-a-key"          # a test value: no request ever leaves this file
INTERNET = (socket.AF_INET, socket.AF_INET6)
LOOPBACK = ("localhost", "127.0.0.1", "::1")
REFUSED = "network blocked by tests/unit/test_public_network.py"


# ------------------------------------------------------------------ the guard
class Watch:
    """What the guard saw. `connections`: every attempt to reach an internet address (a connect,
    a name lookup), refused as an offline machine would. `sockets`: the family of every socket
    made, of any kind. Local Unix-domain sockets (the local Docker daemon, asyncio's self-pipe)
    are not the network and stay usable, but are listed here, so a test can require none at all.
    (An internet socket cannot be made at all here, so nothing can be sent on one; but a library
    may probe for IPv6 when it is imported, so only `courant verify` is held to making none.)"""

    def __init__(self) -> None:
        self.connections: list[tuple[str, object]] = []
        self.sockets: list[int] = []

    @property
    def internet_sockets(self) -> list[int]:
        return [f for f in self.sockets if f in INTERNET]


@pytest.fixture
def watch(monkeypatch, tmp_path):
    """Network connections refused and every socket recorded."""
    seen = Watch()
    real_init, real_connect = socket.socket.__init__, socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def init(self, family=-1, type=-1, proto=-1, fileno=None):
        seen.sockets.append(socket.AF_INET if family == -1 and fileno is None else family)
        if fileno is None and family in (-1, *INTERNET):
            raise OSError(errno.ENETUNREACH, REFUSED)
        real_init(self, family, type, proto, fileno)

    def connect(self, address):
        if self.family in INTERNET:
            seen.connections.append(("connect", address))
            raise OSError(errno.ENETUNREACH, REFUSED)
        return real_connect(self, address)

    def connect_ex(self, address):
        if self.family in INTERNET:
            seen.connections.append(("connect_ex", address))
            return errno.ENETUNREACH
        return real_connect_ex(self, address)

    def create_connection(address, *a, **k):
        seen.connections.append(("create_connection", address))
        raise OSError(errno.ENETUNREACH, REFUSED)

    def lookup(kind):
        real = getattr(socket, kind)

        def refuse(host, *a, **k):
            if host in LOOPBACK:                  # this machine's own name is not the network
                return real(host, *a, **k)
            seen.connections.append((kind, host))
            raise socket.gaierror(socket.EAI_NONAME, REFUSED)
        return refuse

    monkeypatch.setattr(socket.socket, "__init__", init)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
        monkeypatch.setattr(socket, name, lookup(name))
    # No developer state: no keys, no model ledger, no settings from the home directory.
    for var in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "COURANT_TOKEN",
                "COURANT_OPENROUTER_MODEL", "COURANT_ALLOW_PAID_LLM", "DOCKER_HOST"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("COURANT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COURANT_LLM_STATE", str(tmp_path / "llm"))
    return seen


@pytest.fixture
def no_processes(monkeypatch):
    """Every way to start a process, recorded and refused."""
    started: list[str] = []

    def refuse(name):
        def refused(*a, **k):
            started.append(name)
            raise OSError(errno.EPERM, f"{name} refused by tests/unit/test_public_network.py")
        return refused

    monkeypatch.setattr(subprocess.Popen, "__init__", refuse("subprocess.Popen"))
    monkeypatch.setattr(multiprocessing.Process, "start", refuse("multiprocessing.Process"))
    monkeypatch.setattr(concurrent.futures.ProcessPoolExecutor, "__init__",
                        refuse("ProcessPoolExecutor"))
    for name in ("system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp"):
        monkeypatch.setattr(os, name, refuse(f"os.{name}"), raising=False)
    return started


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def _no_traceback(res) -> None:
    assert res.exception is None or isinstance(res.exception, SystemExit), repr(res.exception)
    assert "Traceback" not in res.output


def test_the_guard_can_fail(watch, no_processes):
    """A guard that cannot fail is decoration: each thing it watches for, done once."""
    with pytest.raises(OSError):
        socket.create_connection(("example.invalid", 80))
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.invalid", 80)
    with pytest.raises(OSError):
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    local = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)      # local: allowed, but seen
    local.close()
    assert [k for k, _ in watch.connections] == ["create_connection", "getaddrinfo"]
    assert watch.sockets == [socket.AF_INET, socket.AF_UNIX]
    assert watch.internet_sockets == [socket.AF_INET]
    for start in (lambda: subprocess.run(["true"], check=False), lambda: os.system("true"),
                  lambda: os.popen("true"), lambda: multiprocessing.Process(target=int).start(),
                  lambda: concurrent.futures.ProcessPoolExecutor(1)):
        with pytest.raises(OSError):
            start()
    assert len(no_processes) == 5, no_processes


# ------------------------------------------------------------------ the inputs
@pytest.fixture
def work(tmp_path):
    from courant.data import extract_demo_case
    from courant.spec import draft_spec

    case = extract_demo_case(tmp_path / "demo")
    spec = tmp_path / "spec.json"
    spec.write_text(draft_spec(case).to_json())
    runs = tmp_path / "runs"
    runs.mkdir()
    for log, sidecar in ((HEALTHY, HEALTHY_CASE), (BLOWUP, BLOWUP_CASE), (UNITS, UNITS_CASE)):
        shutil.copy(log, runs / log.name)
        shutil.copy(sidecar, runs / sidecar.name)
    task = tmp_path / "task.txt"
    task.write_text("Run the pitzDaily tutorial with an inlet velocity of 10 m/s.")
    return {"case": case, "spec": spec, "runs": runs, "task": task, "tmp": tmp_path}


# ------------------------------------------------------------------ 1. courant verify
VERDICT_EXITS = {0, 2, 3}      # accept, reject, insufficient


def _verify_invocations(w: dict) -> list[tuple[list[str], set[int]]]:
    """`courant verify` on the demo, in each of its outputs. (args, the exit codes it may have)."""
    from courant.spec import load_spec

    case, spec, tmp = str(w["case"]), str(w["spec"]), w["tmp"]
    _, spec_sha = load_spec(w["spec"])
    pack, record = str(tmp / "pack"), str(tmp / "record.json")
    return [
        ([str(HEALTHY), "--case-json", str(HEALTHY_CASE), "--json"], {0}),
        ([str(BLOWUP), "--case-json", str(BLOWUP_CASE)], {2}),
        ([str(UNITS), "--case-json", str(UNITS_CASE), "--report", "md"], VERDICT_EXITS),
        ([case], VERDICT_EXITS),
        ([case, "--json"], VERDICT_EXITS),
        ([case, "--report", "md"], VERDICT_EXITS),
        ([case, "--html", str(tmp / "r.html")], VERDICT_EXITS),
        ([case, "--spec", spec, "--spec-sha256", spec_sha], VERDICT_EXITS),
        ([case, "--spec", spec, "--spec-sha256", "0" * 64], VERDICT_EXITS),   # a changed spec
        ([case, "--pack", pack], VERDICT_EXITS),
        (["--check-pack", pack, "--root", case], {0}),
        ([case, "--record", record], VERDICT_EXITS),
        (["--check-record", record, "--root", case], {0}),
        (["--batch", str(w["runs"]), "--junit", str(tmp / "j.xml")], VERDICT_EXITS),
        (["--about"], {0}),
        (["--about", "--json"], {0}),
    ]


def test_courant_verify_on_the_demo_opens_no_socket_and_starts_no_process(watch, no_processes,
                                                                          work):
    """Not even a local socket: verification reads files and prints, and that is all it does."""
    for args, exits in _verify_invocations(work):
        res = _run("verify", *args)
        _no_traceback(res)
        assert res.exit_code in exits, (args, res.exit_code, res.output[-800:])
    assert watch.connections == [], watch.connections
    assert watch.sockets == [], f"courant verify made a socket: {watch.sockets}"
    assert no_processes == [], f"courant verify started a process: {no_processes}"


def test_the_python_call_behind_verify_opens_no_socket_and_starts_no_process(watch, no_processes,
                                                                             work):
    from courant.verify import verify_path

    v = verify_path(HEALTHY, HEALTHY_CASE)
    assert v.verdict == "accept"
    assert verify_path(BLOWUP, BLOWUP_CASE).verdict == "reject"
    assert verify_path(work["case"]).verdict in {"accept", "reject", "insufficient"}
    assert verify_path(work["case"], spec=work["spec"]).exit_code in VERDICT_EXITS
    assert watch.connections == [] and watch.sockets == [] and no_processes == []


def test_the_demo_command_opens_no_socket_and_starts_no_process(watch, no_processes, work):
    res = _run("demo", "--keep", str(work["tmp"] / "demo-out"))
    _no_traceback(res)
    assert res.exit_code in VERDICT_EXITS, res.output[-800:]
    assert watch.connections == [] and watch.sockets == [] and no_processes == []


# ------------------------------------------------------------------ 2. nothing else connects
def _offline_invocations(w: dict) -> dict[str, list[list[str]]]:
    """Every visible command, each subcommand and the main options, on the demo data."""
    case, spec, runs, tmp = str(w["case"]), str(w["spec"]), str(w["runs"]), w["tmp"]
    return {
        # (courant verify in all its outputs is held to more, by the first test above)
        "verify": [[case], [str(HEALTHY), "--case-json", str(HEALTHY_CASE), "--json"],
                   [case, "--spec", spec], ["--about"]],
        "gate": [["--root", str(tmp), "--paths", "runs/*.log.gz", "--packs-dir",
                  str(tmp / "packs")]],
        "spec": [["init", case, "-o", str(tmp / "drafted.json")], ["hash", spec],
                 ["check", case, "--spec", spec], ["lock", spec],
                 ["confirm", spec, "--by", "test", "-o", str(tmp / "confirmed.json")]],
        "convergence": [[case, "--json"]],
        "preflight": [[case]],
        "audit": [[runs]],
        "watch": [[str(BLOWUP), "--once"]],
        "test": [["build"], ["run", "--judge", "courant"],
                 ["run", "--judge", "constant:NOT_ASSESSABLE", "--split", "held-out"]],
        "demo": [["--keep", str(tmp / "demo-out")]],
        "mcp": [["--print-config", "--root", runs]],
        "doctor": [[], ["--bundle", str(tmp / "bundle.json")]],
        "version": [[]],
        "update": [[], ["--check"]],
    }


def _visible() -> set[str]:
    import typer.main

    cmd = typer.main.get_command(app)
    return {name for name, c in cmd.commands.items() if not getattr(c, "hidden", False)}


def _subcommands(group: str) -> set[str]:
    import typer.main

    return set(typer.main.get_command(app).commands[group].commands)


def test_every_visible_command_is_covered(work):
    covered = _offline_invocations(work)
    assert set(covered) == _visible(), "a visible command has no no-network case here"
    # Every subcommand of a group is run, offline or (spec extract, an llm judge) below.
    assert {a[0] for a in covered["spec"]} | {"extract"} == _subcommands("spec")
    assert {"run", "build"} == _subcommands("test")


def test_no_visible_command_opens_a_network_connection(watch, work, monkeypatch):
    import courant.ctest.build as ctest_build

    # `courant test build` rewrites the packaged tasks; here it builds them and writes nothing.
    monkeypatch.setattr(ctest_build, "write", lambda tasks, labels: None)
    for command, cases in _offline_invocations(work).items():
        for args in cases:
            res = _run(command, *args)
            _no_traceback(res)
            assert watch.connections == [], (
                f"courant {command} {' '.join(args)} tried {watch.connections}")


def test_mcp_tools_open_no_network_connection(watch, work):
    from courant.mcp import tools

    confine = tools.Confinement([work["tmp"], DEMO])
    tools.verify_run(str(HEALTHY), str(HEALTHY_CASE), confine=confine)
    tools.verify_run(str(work["case"]), spec=str(work["spec"]), confine=confine)
    tools.score_agent_output(str(work["case"]), str(work["spec"]), reported_success=True,
                             confine=confine)
    tools.check_convergence(str(work["case"]), confine=confine)
    tools.explain_code("NUM.CFL_VIOLATION")
    tools.list_failure_modes()
    tools.verifier_certification()
    assert watch.connections == []


# ------------------------------------------------------------------ 3. the network commands
def test_update_contacts_nothing_unless_it_downloads(watch):
    res = _run("update")                      # a pip installation: prints the pip command
    assert res.exit_code == 0 and watch.connections == []
    assert "pip install --upgrade courant-dev" in res.output
    assert "Nothing was contacted" in res.output


def test_update_names_github_and_reports_cleanly_when_blocked(watch, monkeypatch, tmp_path):
    from courant.cli.commands import update

    monkeypatch.setattr(update, "is_packaged", lambda: True)
    monkeypatch.setattr(update, "target_triple", lambda: "darwin-arm64")
    res = _run("update", "--check")
    assert res.exit_code == 0 and watch.connections == []
    assert "github.com" in res.output and "Nothing was contacted" in res.output

    monkeypatch.setattr(update.sys, "executable", str(tmp_path / "courant"))
    res = _run("update")
    _no_traceback(res)
    assert res.exit_code == 1
    assert "contacting github.com" in res.output
    assert "Could not download the update from github.com" in res.output
    assert watch.connections, "update downloads; the guard should have seen it try"
    assert not (tmp_path / "courant").exists()


def test_doctor_models_names_openrouter_and_falls_back_when_blocked(watch):
    res = _run("doctor", "--models")
    _no_traceback(res)
    assert res.exit_code == 0
    assert "contacting openrouter.ai" in res.output
    assert "Could not read OpenRouter's live listing" in res.output
    hosts = {str(a[1][0] if isinstance(a[1], tuple) else a[1]) for a in watch.connections}
    assert hosts and all("openrouter.ai" in h for h in hosts), hosts


def test_a_model_judge_no_model_may_take_contacts_nothing(watch, monkeypatch):
    """No allowlisted model is measured for judging, so the door refuses: before its allowance
    read or anything else reaches the network."""
    from courant.llm.allowlist import ALLOWLIST

    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    model = ALLOWLIST["openrouter"][0]["id"]
    res = _run("test", "run", "--judge", f"llm:openrouter:{model}")
    _no_traceback(res)
    assert res.exit_code == 3, res.output         # a stop: nothing is scored or written
    assert watch.connections == [], watch.connections
    assert "not measured for judge (ARC-12)" in res.output
    assert "Nothing was contacted" in res.output
    assert "contacting" not in res.output and FAKE_KEY not in res.output


def test_a_model_judge_the_door_lets_through_names_its_endpoint_first(watch, monkeypatch):
    """With a model measured for judging, the judge says what it contacts before it does."""
    from courant.llm import allowlist

    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    first = dict(allowlist.ALLOWLIST["openrouter"][0], measured={"judge": "RM-99 2026-10-04"})
    monkeypatch.setitem(allowlist.ALLOWLIST, "openrouter",
                        [first, *allowlist.ALLOWLIST["openrouter"][1:]])
    import courant.ctest.run as ctest_run

    def stop(*a, **k):                                 # the batch itself is the door's test
        from courant.llm import Refused

        raise Refused("stopped by the test before the batch")

    monkeypatch.setattr(ctest_run, "run", stop)
    res = _run("test", "run", "--judge", f"llm:openrouter:{first['id']}")
    _no_traceback(res)
    assert "contacting openrouter.ai (openrouter) with your OPENROUTER_API_KEY" in res.output
    assert FAKE_KEY not in res.output and watch.connections == []


def test_spec_extract_no_model_may_take_contacts_nothing(watch, monkeypatch, work):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    res = _run("spec", "extract", str(work["task"]))
    _no_traceback(res)
    assert res.exit_code == 4
    assert watch.connections == []
    assert "Nothing was contacted" in res.output
    assert "contacting" not in res.output and FAKE_KEY not in res.output

    monkeypatch.delenv("OPENROUTER_API_KEY")
    res = _run("spec", "extract", str(work["task"]))
    assert res.exit_code == 4 and watch.connections == []


# ------------------------------------------------------------------ 4. the document
def test_network_md_lists_every_command_that_contacts_anything():
    text = (ROOT / "docs" / "NETWORK.md").read_text()
    for needle in ("courant update", "github.com", "courant doctor --models", "openrouter.ai",
                   "courant test run --judge llm:", "courant spec extract",
                   "api.anthropic.com", "github.com/courantdev/courant/releases"):
        assert needle in text, needle


def test_network_md_names_the_tests_that_hold_it_and_they_exist():
    """The document says which tests hold it true, and each is a file in this tree: a test that
    is named and not shipped would be a promise nothing keeps."""
    text = (ROOT / "docs" / "NETWORK.md").read_text()
    named = set(re.findall(r"`(tests/[\w/]+\.py)`", text))
    assert {"tests/unit/test_public_network.py",
            "tests/unit/test_public_verdict_path.py"} <= named, named
    for rel in named:
        assert (ROOT / rel).is_file(), rel
