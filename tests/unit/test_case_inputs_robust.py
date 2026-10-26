"""A case file that is missing, cannot be reached, or is not a regular file (TRD VER-14, PRE-1,
MCP-5; the hard-proofing audit of 21 Oct 2026, robustness group).

Absent and unavailable are two facts, and the readers used to report both as absent:

- a dictionary every OpenFOAM solver reads (`system/controlDict`, `system/fvSchemes`,
  `system/fvSolution`) that the case directory does not hold is named under "not checked";
  before, removing `system/fvSolution` from a case its tolerance rejected gave accept, grade A,
  with nothing said;
- a dictionary behind a directory that cannot be searched (`chmod 000 system`, or `600`) is
  named as unreadable, as a dictionary that cannot be opened or parsed always was; before, it
  was absent;
- a case directory from which no case dictionary was read says "the case configuration could
  not be read", as a bare log does, and preflight on it is NOT_ASSESSABLE, exit 3 (PRE-1);
  before, an empty or mistyped directory was NO_SETUP_FAULT_FOUND, exit 0;
- a FIFO, socket or device in place of a case file is never opened: opening a FIFO with no
  writer blocks forever, and it hung `courant verify`, `courant preflight` and the MCP
  `verify_run` with no output. It is unreadable, like a file that cannot be opened;
- an MCP tool call that has not answered within the server's limit is answered with an error
  (MCP-5), so an agent's call always returns.

A `courant verify` verdict does not move on any of these (GATE-13): accept stays accept, with the
sentence. Preflight's NOT_ASSESSABLE on a directory with nothing read is PRE-1's own verdict. The
fixture is the committed `cavity_extreme_aspect_ratio` case.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.verify import verify_path
from courant.verify.preflight import preflight_path

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"

_root_reads_everything = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 directory")
_fifos = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this platform")

NOT_FOUND_ONE = "A case dictionary every OpenFOAM solver reads was not found: "
NOT_FOUND_MANY = "Case dictionaries every OpenFOAM solver reads were not found: "
NO_CONFIG = "The case configuration could not be read; setup rules that need it were not run."
UNREADABLE = "Case dictionary could not be parsed: "


def _case(tmp_path: Path, name: str = "cavity", *, tolerance: str | None = None) -> Path:
    """The committed cavity, its log unzipped; `tolerance` replaces U's solver tolerance."""
    case = tmp_path / name
    shutil.copytree(CAVITY, case)
    for gz in case.glob("log.*.gz"):
        with gzip.open(gz, "rb") as src, open(case / gz.name.removesuffix(".gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    if tolerance is not None:
        fv = case / "system" / "fvSolution"
        text = fv.read_text()
        assert "tolerance       1e-05;" in text
        fv.write_text(text.replace("tolerance       1e-05;", f"tolerance       {tolerance};"))
    return case


def _app():
    from courant.cli.main import app

    return app


def _listed(verdict) -> set[str]:
    return {f.path for f in verdict.run.inputs}


class _Mode:
    """A directory's mode set for the block and restored after it, whatever happens."""

    def __init__(self, path: Path, mode: int):
        self.path, self.mode = path, mode

    def __enter__(self):
        self.before = self.path.stat().st_mode
        self.path.chmod(self.mode)

    def __exit__(self, *exc):
        self.path.chmod(self.before)


def _within(seconds: float, call, *fifos: Path):
    """`call()`, failing the test (not hanging the suite) when it has not returned in `seconds`.
    A reader blocked opening one of `fifos` is released by opening it for writing."""
    box: dict = {}

    def run():
        try:
            box["value"] = call()
        except BaseException as exc:            # noqa: BLE001 - re-raised below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        deadline = time.monotonic() + 10
        while worker.is_alive() and time.monotonic() < deadline:
            for fifo in fifos:
                try:
                    os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
                except OSError:
                    pass
            worker.join(0.1)
        pytest.fail(f"did not return within {seconds} s: a case file blocked its reader")
    if "error" in box:
        raise box["error"]
    return box["value"]


# ---------------------------------------------------------------- absent: named, never silent
def test_the_control_a_loose_tolerance_rejects(tmp_path):
    verdict = verify_path(_case(tmp_path, tolerance="0.5"))
    assert (verdict.verdict, verdict.rule) == ("reject", "VER-5.4")
    assert "solver.loose_tolerance" in {r.code for r in verdict.reasons}
    assert not any(s.startswith((NOT_FOUND_ONE, NOT_FOUND_MANY, UNREADABLE, NO_CONFIG))
                   for s in verdict.not_assessed)


def test_a_missing_fvsolution_is_named_not_silent(tmp_path):
    """The audit's case: the tolerance that rejected the run is in `system/fvSolution`; remove the
    file and the run was accepted at A with nothing said. It is still accepted (GATE-13: the
    sentence is reporting), and the sentence says what was not there."""
    case = _case(tmp_path, tolerance="0.5")
    (case / "system" / "fvSolution").unlink()
    verdict = verify_path(case)
    assert verdict.verdict == "accept"
    assert (NOT_FOUND_ONE + "system/fvSolution. The setup rules that need it were not run."
            in verdict.not_assessed)
    assert NO_CONFIG not in verdict.not_assessed


def test_each_dictionary_a_solver_reads_is_named_when_missing(tmp_path):
    case = _case(tmp_path)
    for rel in ("system/fvSchemes", "system/fvSolution"):
        (case / rel).unlink()
    verdict = verify_path(case)
    assert (NOT_FOUND_MANY + "system/fvSchemes, system/fvSolution. The setup rules that need "
            "them were not run.") in verdict.not_assessed


def test_a_case_directory_with_no_dictionary_says_so_as_a_bare_log_does(tmp_path):
    """A log alone in a directory: nothing in `system/` or `constant/` was read, so the verdict
    says the configuration could not be read, as `courant verify log.icoFoam` says."""
    case = _case(tmp_path)
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(case / "log.icoFoam", alone / "log.icoFoam")
    in_dir = verify_path(alone)
    bare = verify_path(alone / "log.icoFoam")
    assert NO_CONFIG in in_dir.not_assessed and NO_CONFIG in bare.not_assessed
    assert (in_dir.verdict, in_dir.rule, in_dir.grade) == (bare.verdict, bare.rule, bare.grade)
    assert not any(s.startswith((NOT_FOUND_ONE, NOT_FOUND_MANY)) for s in in_dir.not_assessed)


# ---------------------------------------------------------------- unreachable: unreadable
@_root_reads_everything
@pytest.mark.parametrize("mode", [0o000, 0o600], ids=["000", "600"])
def test_a_system_directory_that_cannot_be_searched_is_unreadable_not_absent(tmp_path, mode):
    """`chmod 000 system` (or `600`: listable, not searchable) made every dictionary in it read
    as absent, so the rejected case was accepted with a not-checked list identical to a healthy
    case's. Now each is named as unreadable, as a mode-000 `fvSolution` already was."""
    case = _case(tmp_path, tolerance="0.5")
    healthy = verify_path(_case(tmp_path, "healthy"))
    with _Mode(case / "system", mode):
        verdict = verify_path(case)
        preflight = preflight_path(case)
    note = next(s for s in verdict.not_assessed if s.startswith(UNREADABLE))
    for rel in ("system/controlDict", "system/fvSchemes", "system/fvSolution"):
        assert f"{rel}: PermissionError: Permission denied" in note
    assert "not absent" in note and str(tmp_path) not in note
    assert verdict.not_assessed != healthy.not_assessed
    assert not any(s.startswith((NOT_FOUND_ONE, NOT_FOUND_MANY)) for s in verdict.not_assessed)
    assert not any(p.startswith("system/") for p in _listed(verdict))
    assert note in preflight.not_checked


@_root_reads_everything
def test_an_unreadable_dictionary_is_named_by_preflight_as_verify_names_it(tmp_path):
    case = _case(tmp_path)
    with _Mode(case / "system" / "fvSolution", 0o000):
        verdict = verify_path(case)
        preflight = preflight_path(case)
    note = next(s for s in verdict.not_assessed if s.startswith(UNREADABLE))
    assert "system/fvSolution: PermissionError: Permission denied" in note
    assert note in preflight.not_checked
    assert preflight.verdict == "NO_SETUP_FAULT_FOUND"


# ---------------------------------------------------------------- preflight: PRE-1's third verdict
def test_preflight_on_an_empty_directory_is_not_assessable_exit_3(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    p = preflight_path(empty)
    assert p.verdict == "NOT_ASSESSABLE" and p.reasons == () and p.run.inputs == ()
    res = CliRunner().invoke(_app(), ["preflight", str(empty), "--json"])
    assert res.exit_code == 3, res.output
    assert json.loads(res.stdout)["verdict"] == "NOT_ASSESSABLE"


_TRANSPORT_HEADER = ("FoamFile { version 2.0; format ascii; class dictionary; "
                     "object transportProperties; }\n")


def test_preflight_on_a_directory_whose_only_reading_is_a_phase_list_is_not_assessable(tmp_path):
    """`phases (a b);` lists the phases and neither phase's setting is read: neither is
    Newtonian, so neither `nu` is a viscosity. Nothing a check reads was read, as in an empty
    directory, but the list counted as a reading, so preflight said NO_SETUP_FAULT_FOUND, exit 0,
    where 0.5.0 said NOT_ASSESSABLE, exit 3 (the review of the 0.6.0 hard-proofing)."""
    from courant.case.config import CaseConfig

    d = tmp_path / "phases_only"
    (d / "constant").mkdir(parents=True)
    tp = d / "constant" / "transportProperties"
    tp.write_text(_TRANSPORT_HEADER + "phases (a b);\n"
                  "a { transportModel CrossPowerLaw; nu 1e-3; }\n"
                  "b { transportModel BirdCarreau; nu 1e-5; }\n")
    assert CaseConfig.from_foam(str(d)).nothing_read()
    assert preflight_path(d).verdict == "NOT_ASSESSABLE"
    res = CliRunner().invoke(_app(), ["preflight", str(d), "--json"])
    assert res.exit_code == 3, res.output
    assert json.loads(res.stdout)["verdict"] == "NOT_ASSESSABLE"
    # a Newtonian phase's nu is a setting read: the directory is assessed on it
    tp.write_text(_TRANSPORT_HEADER + "phases (a b);\n"
                  "a { transportModel Newtonian; nu 1e-6; }\n"
                  "b { transportModel BirdCarreau; nu 1e-5; }\n")
    assert not CaseConfig.from_foam(str(d)).nothing_read()
    assert CliRunner().invoke(_app(), ["preflight", str(d), "--json"]).exit_code != 3


def test_preflight_on_the_parent_of_a_faulted_case_is_not_a_green_light(tmp_path):
    """A mistyped path: the case itself has a setup fault (exit 2); its parent was
    NO_SETUP_FAULT_FOUND, exit 0."""
    case = _case(tmp_path, tolerance="0.5")
    assert preflight_path(case).verdict == "SETUP_FAULT"
    run = CliRunner()
    assert run.invoke(_app(), ["preflight", str(case)]).exit_code == 2
    assert run.invoke(_app(), ["preflight", str(tmp_path)]).exit_code == 3


def test_preflight_names_a_missing_dictionary_and_still_checks_the_rest(tmp_path):
    case = _case(tmp_path)
    (case / "system" / "fvSolution").unlink()
    p = preflight_path(case)
    assert p.verdict == "NO_SETUP_FAULT_FOUND"
    assert (NOT_FOUND_ONE + "system/fvSolution. The setup rules that need it were not run."
            in p.not_checked)
    transport = case / "constant" / "transportProperties"
    transport.write_text(transport.read_text().replace("0.00848", "1"))
    assert preflight_path(case).verdict == "SETUP_FAULT"            # physics.nu is still read


def test_the_healthy_case_gains_no_sentence(tmp_path):
    case = _case(tmp_path)
    verdict, p = verify_path(case), preflight_path(case)
    for said in (verdict.not_assessed, p.not_checked):
        assert not any(s.startswith((NOT_FOUND_ONE, NOT_FOUND_MANY, UNREADABLE, NO_CONFIG))
                       for s in said)


# ---------------------------------------------------------------- a FIFO is never opened
@_fifos
@pytest.mark.parametrize("rel", ["system/controlDict", "system/fvSolution",
                                 "constant/transportProperties"])
def test_a_fifo_dictionary_is_unreadable_and_verify_and_preflight_return(tmp_path, rel):
    case = _case(tmp_path)
    (case / rel).unlink()
    os.mkfifo(case / rel)
    verdict = _within(30, lambda: verify_path(case), case / rel)
    preflight = _within(30, lambda: preflight_path(case), case / rel)
    note = next(s for s in verdict.not_assessed if s.startswith(UNREADABLE))
    assert f"{rel}: NotRegularFile: a FIFO, not a regular file" in note
    assert rel not in _listed(verdict) and note in preflight.not_checked


@_fifos
def test_a_fifo_field_file_is_a_missing_field_and_verify_returns(tmp_path):
    """A field file that cannot be read is a missing field (T-MANIFEST-2): the verdict is the one
    the case gives without it, but for the sentence VER-7b adds by default for a field file
    that exists and could not be read (`sentences.fields_unread`)."""
    from courant.verify import sentences

    case = _case(tmp_path)
    (case / "0.5").mkdir()
    (case / "0.5" / "p").write_text("FoamFile{version 2.0;format ascii;class volScalarField;"
                                    "object p;}\ninternalField uniform 0.5;\n")
    gone = verify_path(case)
    os.mkfifo(case / "0.5" / "U")
    verdict = _within(30, lambda: verify_path(case), case / "0.5" / "U")
    assert "0.5/U" not in _listed(verdict)
    got, want = json.loads(verdict.to_json()), json.loads(gone.to_json())
    said = sentences.fields_unread("0.5", ("U",))
    assert [s for s in got["not_assessed"] if s not in want["not_assessed"]] == [said]
    got["not_assessed"] = [s for s in got["not_assessed"] if s != said]
    assert got == want


@_fifos
def test_a_fifo_is_never_opened_by_a_reader(tmp_path):
    from courant.case.reads import NotRegularFile, reading, recording

    fifo = tmp_path / "f"
    os.mkfifo(fifo)
    ran = []
    with recording(tmp_path) as reads:
        with pytest.raises(NotRegularFile, match="a FIFO, not a regular file"):
            with reading(fifo):
                ran.append(True)
        with pytest.raises(NotRegularFile, match="a directory"), reading(tmp_path):
            ran.append(True)
    assert ran == [] and reads.paths() == []
    assert issubclass(NotRegularFile, OSError)          # every reader's fallback already takes it


@_fifos
def test_the_mcp_tool_returns_on_a_case_with_a_fifo(tmp_path):
    from courant.mcp import tools

    case = _case(tmp_path)
    (case / "system" / "controlDict").unlink()
    os.mkfifo(case / "system" / "controlDict")
    got = _within(30, lambda: tools.verify_run(str(case), confine=tools.Confinement([tmp_path])),
                  case / "system" / "controlDict")
    assert got["schema"] == "courant.verdict/2" and got["verdict"] in ("accept", "reject",
                                                                        "insufficient")
    assert any("system/controlDict: NotRegularFile" in s for s in got["not_assessed"])


@_fifos
def test_a_fifo_sidecar_is_an_input_error_for_preflight_as_for_verify(tmp_path):
    from courant.verify import InputError

    case = _case(tmp_path)
    side = tmp_path / "x.case.json"
    os.mkfifo(side)
    with pytest.raises(InputError, match="no such case sidecar"):
        _within(30, lambda: preflight_path(case / "log.icoFoam", side), side)
    with pytest.raises(InputError, match="no such case sidecar"):
        _within(30, lambda: verify_path(case / "log.icoFoam", side), side)


# ---------------------------------------------------------------- MCP-5: every call answers
def test_an_mcp_call_that_does_not_answer_in_time_is_an_error_not_a_hang(tmp_path, monkeypatch):
    import asyncio

    pytest.importorskip("mcp")
    from courant.mcp import tools
    from courant.mcp.server import build

    release = threading.Event()

    def blocks(*a, **k):
        release.wait(30)
        return {"late": True}

    monkeypatch.setattr(tools, "verify_run", blocks)
    server = build([str(tmp_path)], timeout=0.5)

    async def go():
        return await server.call_tool("verify_run", {"path": str(tmp_path)})

    started = time.monotonic()
    try:
        with pytest.raises(Exception) as err:
            asyncio.run(go())
    finally:
        release.set()
    assert time.monotonic() - started < 10
    msg = str(err.value)
    assert "verify_run" in msg and "no answer within 0.5 s" in msg and "Traceback" not in msg


def test_an_mcp_call_that_answers_in_time_is_unchanged(tmp_path, monkeypatch):
    import asyncio

    pytest.importorskip("mcp")
    from courant.mcp import tools
    from courant.mcp.server import TIMEOUT, build, within

    assert TIMEOUT == 600
    assert within(5, lambda: 42) == 42 and within(None, lambda: 7) == 7
    with pytest.raises(ValueError, match="inner"):
        within(5, lambda: (_ for _ in ()).throw(ValueError("inner")))
    monkeypatch.setattr(tools, "verify_run", lambda *a, **k: {"ok": 1})
    server = build([str(tmp_path)], timeout=5)

    async def go():
        return await server.call_tool("verify_run", {"path": str(tmp_path)})

    assert "ok" in json.dumps(asyncio.run(go()), default=str)


def test_both_mcp_commands_take_the_limit_and_say_its_default(monkeypatch):
    """`courant mcp` and `courant-mcp` are one server (PKG-8): both take `--timeout`."""
    import sys

    from courant.mcp import server

    seen = []

    class _Server:
        def run(self, transport):
            seen.append(transport)

    monkeypatch.setattr(server, "build", lambda roots, timeout: seen.append(timeout) or _Server())
    monkeypatch.setattr(sys, "argv", ["courant-mcp", "--root", ".", "--timeout", "7"])
    server.main()
    pytest.importorskip("mcp")
    res = CliRunner().invoke(_app(), ["mcp", "--root", ".", "--timeout", "0"])
    assert res.exit_code == 0, res.output
    res = CliRunner().invoke(_app(), ["mcp", "--root", "."])
    assert seen == [7.0, "stdio", 0.0, "stdio", server.TIMEOUT, "stdio"]
    import inspect

    from courant.cli.commands import mcp as command

    option = inspect.signature(command.mcp).parameters["timeout"].default
    assert f"default {server.TIMEOUT:g}" in option.help      # the help says the server's own
    # and says what it cannot answer (MCP-5; the final review of 0.6.0)
    assert "holds the interpreter's lock" in option.help


# ---------------------------------------------------------------- the same words on any machine
@_root_reads_everything
def test_a_spec_readers_note_on_an_unreadable_file_names_no_absolute_path(tmp_path):
    """An `OSError`'s message names the absolute path (`[Errno 13] Permission denied:
    '/Users/x/case/0/U'`), and the deck reader's and `as_run`'s notes embedded it, so the same
    case gave different conformance items and not-assessed lines on two machines (VER-9; the final
    review of 0.6.0, 21 Oct 2026). The note gives the error's type and reason, and the file by
    its case-relative name (`courant.case.reads.reason`)."""
    from courant.case.reads import reason
    from courant.spec import deck
    from courant.spec.as_run import AsRun
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    u = case / "0" / "U"
    item = SpecItem(name="lid", file="0/U", patch="movingWall", value=[1.0, 0.0, 0.0])
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    (poly / "owner").write_text("FoamFile { class labelList; object owner; }\n0\n(\n)\n")
    with _Mode(u, 0o000), _Mode(poly / "owner", 0o000):
        found = deck.read(case, item)
        mesh = AsRun(case)._read_mesh(poly)
    assert found.value is None
    assert found.note == "0/U could not be read: PermissionError: Permission denied"
    assert mesh.note.startswith("constant/polyMesh could not be read: PermissionError")
    for note in (found.note, mesh.note):
        assert str(tmp_path) not in note and "Errno" not in note
    # the reason reads the same wherever the file is
    assert reason(PermissionError(13, "Permission denied", "/a/b")) == \
        reason(PermissionError(13, "Permission denied", "/c/d")) == \
        "PermissionError: Permission denied"
    assert reason(ValueError("no list")) == "ValueError: no list"

