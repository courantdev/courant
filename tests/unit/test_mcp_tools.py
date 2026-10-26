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


#: Written outside every root; no tool may return it, its hash, or anything judged on it.
OUTSIDE_VALUE = "outside-the-roots-4be1"


@pytest.fixture
def linked(tmp_path):
    """The demo case inside one root, and a directory beside the root that no tool may read."""
    from courant.data import extract_demo_case

    root, outside = tmp_path / "root", tmp_path / "outside"
    outside.mkdir()
    (outside / "notes").write_text(f"password  {OUTSIDE_VALUE};\n")
    return tools.Confinement([root]), extract_demo_case(root), outside


def _spec(path: Path, *items: dict) -> str:
    import json

    path.write_text(json.dumps({"schema": "courant.spec/1", "task": "t", "items": list(items)}))
    return str(path)


def _move_out(case: Path, rel: str, outside: Path) -> None:
    """Move `case/rel` outside the roots and leave a link to it in its place."""
    (case / rel).rename(outside / rel)
    (case / rel).symlink_to(outside / rel, target_is_directory=(outside / rel).is_dir())


def test_a_link_in_a_root_to_a_file_outside_every_root_is_refused_not_read(linked):
    # MCP-3: the path a reader opens is confined, not only the argument. Before, the case was
    # judged on the outside file and its value came back in the reason.
    confine, case, outside = linked
    (case / "constant" / "notes").symlink_to(outside / "notes")
    spec = _spec(case.parent / "spec.json",
                 {"name": "pw", "file": "constant/notes", "key": "password", "value": "x"})
    for call in (lambda: tools.verify_run(str(case), spec=spec, confine=confine),
                 lambda: tools.score_agent_output(str(case), spec, confine=confine)):
        with pytest.raises(tools.OutsideRoot) as refused:
            call()
        said = str(refused.value)
        assert "constant/notes" in said and "outside the allowed roots" in said
        assert OUTSIDE_VALUE not in said


def test_a_linked_directory_that_leaves_the_roots_is_refused_and_nothing_is_hashed(linked):
    confine, case, outside = linked
    (case / "linked").symlink_to(outside, target_is_directory=True)
    spec = _spec(case.parent / "spec.json", {"name": "creds", "file": "linked/notes",
                                             "sha256": "0" * 64})
    with pytest.raises(tools.OutsideRoot, match="linked is a link") as refused:
        tools.verify_run(str(case), spec=spec, confine=confine)
    assert hashlib.sha256((outside / "notes").read_bytes()).hexdigest() not in str(refused.value)
    # The link is what is refused. The file the reader reached under it is not named: that it
    # was reached would say that `notes` exists outside the roots.
    assert "linked/notes" not in str(refused.value)


@pytest.mark.parametrize("rel", ["log.simpleFoam", "constant", "system", "postProcessing"])
def test_a_case_entry_linked_outside_the_roots_refuses_the_call(linked, rel):
    confine, case, outside = linked
    _move_out(case, rel, outside)
    with pytest.raises(tools.OutsideRoot, match=rel):
        tools.verify_run(str(case), confine=confine)
    if rel in ("log.simpleFoam", "postProcessing"):    # what check_convergence reads
        with pytest.raises(tools.OutsideRoot, match=rel):
            tools.check_convergence(str(case), confine=confine)


@pytest.mark.parametrize("rel, item", [
    ("constant/probe", {"key": "x", "value": 1}),       # a specification item names it
    ("constant/probe", {"sha256": "0" * 64}),
    ("system/fvSchemes", None),                         # the case reader reads it
    ("log.simpleFoam", None),                           # the solver's log
    ("postProcessing/probe", None),                     # the series discovery lists it
    ("constant/triSurface/probe.obj", None),            # no reader opens it
])
def test_a_link_out_of_the_roots_is_refused_whether_or_not_its_target_exists(linked, rel, item):
    # MCP-3: refused only once a reader opened it, and a reader looks first whether its file is
    # there, a link to a path outside that does not exist was answered as a missing file where
    # one to a path that does was refused: one call told whether any path on the machine exists.
    confine, case, outside = linked
    spec = _spec(case.parent / "spec.json", {"name": "p", "file": rel, **item}) if item else None
    link = case / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    calls = [lambda: tools.verify_run(str(case), spec=spec, confine=confine),
             lambda: tools.check_convergence(str(case), confine=confine)]
    if spec:
        calls.append(lambda: tools.score_agent_output(str(case), spec, confine=confine))
    for target in (outside / "notes", outside / "absent"):
        link.unlink(missing_ok=True)
        link.symlink_to(target)
        for call in calls:
            with pytest.raises(tools.OutsideRoot, match=rel):
                call()
    link.unlink()
    link.symlink_to(case.parent / "absent")             # a dangling link inside the root
    if rel != "log.simpleFoam":                         # (with no log there is nothing to judge)
        assert "verdict" in tools.verify_run(str(case), spec=spec, confine=confine)


def _refusals(calls: dict) -> dict[str, str]:
    said = {}
    for name, call in calls.items():
        with pytest.raises(tools.OutsideRoot) as refused:
            call()
        said[name] = str(refused.value)
    return said


@pytest.mark.parametrize("via", ["sha256 item", "key item", "constant", "processor0"])
def test_a_refusal_under_a_linked_directory_is_the_same_whether_or_not_the_file_exists(linked,
                                                                                      via):
    # MCP-3: a reader asks before it opens, and most look first whether their file is there, so
    # a reader's own refusal (with the count of the rest) said that the file under a linked
    # directory exists, where with no file only the link was named: one call told whether any
    # path on the machine exists. A linked `processor0/` put the names of outside entries
    # (`processor0/<time>/U`) into the message too. The link is refused, and only it is named.
    confine, case, outside = linked
    far, spec = outside / "far", None
    if via.endswith("item"):              # a specification item names a path under the link
        link, rel = case / "l", "l/deep/nu"
        item = {"sha256": "0" * 64} if via == "sha256 item" else {"key": "nu", "value": 1}
        spec = _spec(case.parent / "spec.json", {"name": "p", "file": rel, **item})
        present = {"deep/nu": "nu 1;\n"}
    elif via == "constant":               # the case reader's own files
        link = case / "constant"
        present = {p.name: p.read_text() for p in link.iterdir()}
        shutil.rmtree(link)
    else:                                 # a decomposed run's results: listed, then read
        link = case / "processor0"
        present = {"1000/U": "internalField uniform (1 0 0);\n", "1000/p": "internalField uniform 0;\n"}
    link.symlink_to(far, target_is_directory=True)
    calls = {"verify_run": lambda: tools.verify_run(str(case), spec=spec, confine=confine),
             "check_convergence": lambda: tools.check_convergence(str(case), confine=confine)}
    if spec:
        calls["score_agent_output"] = lambda: tools.score_agent_output(str(case), spec,
                                                                       confine=confine)
    said = {}
    for exists in (True, False):
        shutil.rmtree(far, ignore_errors=True)
        far.mkdir()
        for rel, text in present.items() if exists else ():
            (far / rel).parent.mkdir(parents=True, exist_ok=True)
            (far / rel).write_text(text)
        said[exists] = _refusals(calls)
    assert said[True] == said[False]
    for text in said[True].values():
        assert str(case.resolve() / link.name) in text and "outside the allowed roots" in text
        assert "more" not in text                       # one link, and nothing counted under it
        rest = text.replace(str(case.resolve().parent.parent), "<tmp>")
        for rel in present:                             # nothing of what is under the link
            assert rel not in rest and rel.split("/")[0] not in rest


def test_a_refusal_names_the_link_in_the_root_never_where_it_leads(linked):
    # MCP-3: the error named the link and `resolve()` of it, every link along the outside path
    # followed, so it read back any link's target on the machine (`/etc` as `/private/etc`). It
    # names what is inside the roots, and the roots, only.
    confine, case, outside = linked
    (outside / "far-real").mkdir()
    (outside / "far-real" / "nu").write_text("nu 1;\n")
    (outside / "hop").symlink_to(outside / "far-real", target_is_directory=True)
    (case / "constant" / "x").symlink_to(outside / "hop", target_is_directory=True)
    spec = _spec(case.parent / "spec.json", {"name": "nu", "file": "constant/x/nu", "key": "nu",
                                             "value": 1})
    (case.parent / "elsewhere").symlink_to(outside / "hop", target_is_directory=True)
    said = _refusals({
        "verify_run": lambda: tools.verify_run(str(case), spec=spec, confine=confine),
        "check_convergence": lambda: tools.check_convergence(str(case), confine=confine),
        "score_agent_output": lambda: tools.score_agent_output(str(case), spec, confine=confine),
        # the path a tool is given: a link in a root, and a path outside every root
        "argument": lambda: tools.verify_run(str(case.parent / "elsewhere"), confine=confine),
        "outside": lambda: tools.verify_run(str(outside / "hop" / "nu"), confine=confine),
    })
    for name, text in said.items():
        assert "outside the allowed roots" in text, name
        assert "far-real" not in text, name
    for name in ("verify_run", "check_convergence", "score_agent_output"):
        assert str(case.resolve() / "constant" / "x") in said[name]
        assert "hop" not in said[name] and str(outside) not in said[name]
    assert "elsewhere" in said["argument"] and "hop" not in said["argument"]
    assert str(outside / "hop" / "nu") in said["outside"]        # as the agent named it


def test_with_several_links_out_the_refusal_does_not_depend_on_the_listing_order(linked,
                                                                               monkeypatch):
    # Which link the error named followed os.scandir's order, which is the filesystem's: the same
    # case gave one message on APFS and another on ext4. The links are named in sorted order.
    import os

    confine, case, outside = linked
    for rel in ("system/zeta", "constant/alpha"):
        (case / rel).symlink_to(outside / "notes")

    def said() -> str:
        return _refusals({"c": lambda: tools.check_convergence(str(case), confine=confine)})["c"]

    first = said()
    listed = os.scandir

    class Reversed:
        def __init__(self, path="."):
            with listed(path) as it:
                self.entries = list(it)[::-1]

        def __iter__(self):
            return iter(self.entries)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

    monkeypatch.setattr(os, "scandir", Reversed)
    assert said() == first
    assert first.startswith(str(case.resolve() / "constant" / "alpha"))
    assert "1 more" in first


def test_the_scan_does_not_list_the_root_through_a_link_that_holds_the_case(linked):
    # A link in the case to `..` or to the root made every call list the whole root, and a link
    # out of the roots in another case then refused this one, naming the other case. A link to a
    # directory that holds the case is refused itself (through it the readers could reach every
    # file under it), and nothing under it is listed or named.
    confine, case, outside = linked
    other = case.parent / "other"
    other.mkdir()
    (other / "secret").symlink_to(outside / "notes")
    before = tools.verify_run(str(case), confine=confine)
    (case / "self").symlink_to(".", target_is_directory=True)  # the case itself: nothing new
    assert tools.verify_run(str(case), confine=confine) == before
    for name, to in (("up", ".."), ("top", case.parent)):
        (case / name).symlink_to(to, target_is_directory=True)
        for call in (lambda: tools.verify_run(str(case), confine=confine),
                     lambda: tools.check_convergence(str(case), confine=confine)):
            with pytest.raises(tools.OutsideRoot) as refused:
                call()
            said = str(refused.value)
            assert said.startswith(str(case.resolve() / name)) and "holds the case" in said
            rest = said.replace(str(case.resolve().parent.parent), "<tmp>")
            assert "other" not in rest and "secret" not in rest and "more" not in rest
        (case / name).unlink()


@pytest.mark.parametrize("via", ["specification items", "case file"])
def test_in_a_directory_that_cannot_be_listed_a_linked_file_is_refused_whether_or_not_it_exists(
        linked, via):
    # The scan cannot see a link in a directory it cannot list, and a reader still opens a file
    # there by name. The readers that looked first whether the file was there (a specification
    # item, a sha256 item, the case reader's dictionaries) now ask before they look, so the answer
    # is the same whether the file exists.
    import os

    confine, case, outside = linked
    far = outside / "far"
    if via == "specification items":
        hidden, rel, names = case / "hidden", "hidden/l", ("nu", "h")
        hidden.mkdir()
        (hidden / "l").symlink_to(far, target_is_directory=True)
        spec = _spec(case.parent / "spec.json",
                     {"name": "nu", "file": "hidden/l/nu", "key": "nu", "value": 1},
                     {"name": "h", "file": "hidden/l/h", "sha256": "0" * 64})
        calls = {"verify_run": lambda: tools.verify_run(str(case), spec=spec, confine=confine),
                 "score_agent_output": lambda: tools.score_agent_output(str(case), spec,
                                                                        confine=confine)}
    else:
        hidden, rel, names = case / "constant", "constant/transportProperties", ()
        (hidden / "transportProperties").unlink()
        (hidden / "transportProperties").symlink_to(far / "transportProperties")
        calls = {"verify_run": lambda: tools.verify_run(str(case), confine=confine)}
    said = {}
    try:
        for exists in (True, False):
            shutil.rmtree(far, ignore_errors=True)
            far.mkdir()
            for name in (*names, "transportProperties") if exists else ():
                (far / name).write_text(f"{name} 1;\nnu [0 2 -1 0 0 0 0] 1e-05;\n")
            os.chmod(hidden, 0o311)                      # passable, not listable
            said[exists] = _refusals(calls)
            os.chmod(hidden, 0o755)
    finally:
        os.chmod(hidden, 0o755)
    assert said[True] == said[False]
    assert all(str(case.resolve() / rel) in t for t in said[True].values())


def test_a_link_loop_in_a_case_is_not_a_traceback(linked):
    confine, case, _ = linked
    before = tools.verify_run(str(case), confine=confine)
    (case / "constant" / "loop").symlink_to("loop")
    assert tools.verify_run(str(case), confine=confine)["verdict"] == before["verdict"]


def test_a_link_that_stays_inside_a_root_is_followed_as_before(linked):
    # A shared mesh or dictionary linked from elsewhere in a root is ordinary OpenFOAM practice.
    confine, case, _ = linked
    before = tools.verify_run(str(case), confine=confine)
    shared = case.parent / "shared"
    shared.mkdir()
    _move_out(case, "constant", shared)
    assert tools.verify_run(str(case), confine=confine) == before


def test_outside_a_confinement_a_link_is_followed_as_before(linked):
    # The CLI reads what its user can read: only the MCP tools are confined (GATE-13: no
    # verdict of `courant verify` changes).
    from courant.verify import verify_path

    _, case, outside = linked
    before = verify_path(case).to_json()
    _move_out(case, "constant", outside)
    assert verify_path(case).to_json() == before


#: Run in a child process, because an audit hook cannot be removed once added: it records every
#: file the tools open, and every directory they list, whose resolution is under the outside
#: directory.
_OPENS_OUTSIDE = r"""
import json, os, sys
root, case, spec, outside = sys.argv[1:5]
outside = os.path.realpath(outside)
opened = set()

def hook(event, args):
    if (event in ("open", "os.listdir", "os.scandir", "glob.glob") and args
            and isinstance(args[0], (str, bytes, os.PathLike))):
        p = os.path.realpath(os.fsdecode(args[0]) or ".")
        if p == outside or p.startswith(outside + os.sep):
            opened.add(f"{event} {p}")

sys.addaudithook(hook)
from courant.mcp import tools
confine = tools.Confinement([root])
said = {}
for name, call in (
        ("verify_run", lambda: tools.verify_run(case, spec=spec, confine=confine)),
        ("score_agent_output", lambda: tools.score_agent_output(case, spec, confine=confine)),
        ("check_convergence", lambda: tools.check_convergence(case, confine=confine))):
    try:
        call()
        said[name] = "answered"
    except tools.OutsideRoot:
        said[name] = "refused"
print(json.dumps({"said": said, "opened": sorted(opened)}))
"""


@pytest.mark.parametrize("fixture", ["demo", "cavity"])
@pytest.mark.parametrize("linked_out", ["entries", "files", "processor0"])
def test_no_tool_opens_a_file_outside_the_roots(tmp_path, fixture, linked_out):
    # `entries`: every top-level entry of the case but its logs is a link out of the roots;
    # `files`: every file but the logs is, each by its own link, its directories real. The logs
    # stay, so the readers after them run too. `processor0`: a decomposed run's results linked
    # out, which the field reader listed for its times. Each tool is refused, and opens and
    # lists nothing outside.
    import gzip
    import json
    import subprocess
    import sys

    from courant.data import extract_demo_case
    from courant.spec import draft_spec

    root, outside = tmp_path / "root", tmp_path / "outside"
    outside.mkdir()
    if fixture == "demo":
        case = extract_demo_case(root)
    else:                                # 0/ fields and their boundary conditions are read too
        case = root / "cavity"
        shutil.copytree(ROOT / "tests/fixtures/real_openfoam/cavity_extreme_aspect_ratio", case)
        (case / "log.icoFoam").write_bytes(gzip.decompress((case / "log.icoFoam.gz").read_bytes()))
        (case / "log.icoFoam.gz").unlink()
    spec = root / "spec.json"
    spec.write_text(draft_spec(case).to_json())
    if linked_out == "entries":
        rels = [p.name for p in case.iterdir()]
    elif linked_out == "files":
        rels = [str(p.relative_to(case)) for p in case.rglob("*") if p.is_file()]
    else:
        rels = ["processor0"]
        (case / "processor0" / "1000").mkdir(parents=True)
        (case / "processor0" / "1000" / "U").write_text("internalField uniform (1 0 0);\n")
    for rel in sorted(r for r in rels if not r.startswith("log.")):
        (outside / rel).parent.mkdir(parents=True, exist_ok=True)
        _move_out(case, rel, outside)
    got = subprocess.run([sys.executable, "-c", _OPENS_OUTSIDE, str(root), str(case), str(spec),
                          str(outside)], capture_output=True, text=True, cwd=str(ROOT), timeout=300,
                         check=False)
    assert got.returncode == 0, got.stderr
    out = json.loads(got.stdout.strip().splitlines()[-1])
    assert out["opened"] == []
    assert out["said"] == {"verify_run": "refused", "score_agent_output": "refused",
                           "check_convergence": "refused"}


def test_the_server_will_not_default_its_root_to_the_disk_or_the_home_directory(tmp_path,
                                                                                monkeypatch):
    # MCP hosts often start a server in / or in the home directory; with no --root that root
    # would allow every file the user can read.
    from courant.mcp import server

    monkeypatch.setenv("HOME", str(tmp_path))
    for where in ("/", tmp_path):
        monkeypatch.chdir(where)
        assert "--root" in server.unconfined_default([])
        assert server.unconfined_default([str(tmp_path / "runs")]) is None
        monkeypatch.setattr("sys.argv", ["courant-mcp"])
        with pytest.raises(SystemExit) as stopped:
            server.main()
        assert stopped.value.code == 2
    (tmp_path / "runs").mkdir()
    monkeypatch.chdir(tmp_path / "runs")
    assert server.unconfined_default([]) is None


def test_the_server_will_not_default_its_root_to_a_directory_that_holds_the_home_directory(
        tmp_path, monkeypatch):
    # Started in /Users or /home with no --root, the root would hold every user's home directory.
    from typer.testing import CliRunner

    from courant.cli.main import app
    from courant.mcp import server

    home = tmp_path / "Users" / "me"
    (home / "runs").mkdir(parents=True)
    (tmp_path / "Users" / "other").mkdir()
    monkeypatch.setenv("HOME", str(home))
    for where in (tmp_path / "Users", tmp_path):
        monkeypatch.chdir(where)
        assert "--root" in server.unconfined_default([])
        assert server.unconfined_default([str(home / "runs")]) is None
        res = CliRunner().invoke(app, ["mcp", "--print-config"])
        assert res.exit_code == 2 and "--root" in res.output
    for where in (home / "runs", tmp_path / "Users" / "other"):  # below it, or beside it
        monkeypatch.chdir(where)
        assert server.unconfined_default([]) is None


def test_courant_mcp_will_not_default_its_root_to_the_home_directory(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from courant.cli.main import app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for args in (["mcp"], ["mcp", "--print-config"]):
        res = CliRunner().invoke(app, args)
        assert res.exit_code == 2 and "--root" in res.output
    res = CliRunner().invoke(app, ["mcp", "--print-config", "--root", str(tmp_path / "runs")])
    assert res.exit_code == 0


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
