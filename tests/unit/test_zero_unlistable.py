"""A `0/` (or any directory a verdict reads) that cannot be listed or searched (T-ZERO-UNLISTABLE,
TRD VER-3; found while fixing T-HP-ROBUST-1, reproduced 3 Nov 2026 on 0.6.0 as released).

On a copy of the committed `hot_box/converged` solve, its momentum equation's relaxation set to 1
so that its own `fvSolution` rejects it (`scheme.relaxation`, VER-5.4):

- `0/` at mode 000 made `read_case_config` raise on listing it, so the whole configuration was
  lost: under the default the run is accepted (VER-5.6) with "the case configuration could not be
  read". The default keeps that document (GATE-13). Under `--checks next` (`case.zero_unlisted`)
  `0/` is recorded as unreadable and the rest is read, so the run is rejected as the readable copy
  is, and "not assessed" says `0/` was not read.
- Under `--checks next` the same case ended `courant verify` in an internal error: Python 3.11's
  `Path.is_dir` raises on `EACCES` (`courant.case.mesh._instance` on `0/polyMesh`). A traceback is
  never a verdict, in either mode: every directory a verdict reads, at mode 000 or 644 (listable,
  not searchable), gives a document.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from courant.verify import verify_path
from courant.verify.preflight import preflight_path
from tests.unit.test_case_inputs_robust import _Mode
from tests.unit.test_mass_flux import _unpack_fixture

ROOT = Path(__file__).resolve().parents[2]
HOT_BOX = ROOT / "tests" / "fixtures" / "real_openfoam" / "hot_box" / "converged"
NO_CONFIG = "The case configuration could not be read; setup rules that need it were not run."
ZERO_UNREAD = "0/: PermissionError: Permission denied"

pytestmark = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                                reason="root reads a mode-000 directory")


@pytest.fixture(scope="module")
def hot_box(tmp_path_factory) -> Path:
    """The committed solve, unpacked once; each test copies it (cases are read-only). Its
    `fields { rho 1; }` is no longer a reason (T-RELAX-RHO, 0.7.0), so the momentum equation's
    factor is set to 1, an over-relaxation `scheme.relaxation` still rejects, to keep a case its
    own fvSolution rejects."""
    case = _unpack_fixture(HOT_BOX, tmp_path_factory.mktemp("hot_box") / "converged")
    solution = case / "system" / "fvSolution"
    text = solution.read_text()
    assert text.count("U               0.5;") == 1, "the fixture's U factor moved"
    solution.write_text(text.replace("U               0.5;", "U               1;"))
    return case


def _copy(src: Path, tmp_path: Path) -> Path:
    case = tmp_path / "converged"               # the fixture's own name: the same document
    shutil.copytree(src, case)
    return case


def _doc(verdict) -> dict:
    return json.loads(verdict.to_json())


def _codes(verdict) -> list[str]:
    return [r.code for r in verdict.reasons]


# ---------------------------------------------------------------- the reproduction
def test_the_readable_copy_is_rejected_on_its_relaxation_in_both_modes(hot_box, tmp_path):
    case = _copy(hot_box, tmp_path)
    for checks in ("default", "next"):
        v = verify_path(case, checks=checks)
        assert (v.verdict, v.rule) == ("reject", "VER-5.4")
        assert "scheme.relaxation" in _codes(v)
        assert not any("0/" in s and "PermissionError" in s for s in v.not_assessed)


#: The default document 0.6.0 gives on this case (its U factor at 1) with `0/` at mode 000,
#: `tool.version` set to "*" and the keys sorted, so a release's version string does not move it:
#: computed with 0.6.0 as released (9999475) and with this tree, the same, 3 Nov 2026.
V060_ZERO_000 = "592bd9600ebfedb12416d9d9df69d4490f24ae4bc09d91deb9c5807adbd9e149"


def _normalised_sha(verdict) -> str:
    import hashlib

    d = _doc(verdict)
    d["tool"]["version"] = "*"
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


@pytest.mark.usefixtures("checks_as_060")
def test_the_default_keeps_060s_document_on_a_zero_that_cannot_be_listed(hot_box, tmp_path):
    """GATE-13: under the default the listing still raises, the configuration is lost, and the
    document is the one 0.6.0 gave (accept, VER-5.6, the configuration not read), byte for byte
    but the version."""
    case = _copy(hot_box, tmp_path)
    with _Mode(case / "0", 0o000):
        got = verify_path(case)
    assert (got.verdict, got.rule, got.grade) == ("accept", "VER-5.6", "A")
    assert NO_CONFIG in got.not_assessed
    assert not any(ZERO_UNREAD in s for s in got.not_assessed)
    assert _normalised_sha(got) == V060_ZERO_000


def test_next_records_zero_as_unreadable_and_rejects_as_the_readable_copy(hot_box, tmp_path):
    case = _copy(hot_box, tmp_path)
    readable = verify_path(case, checks="next")
    with _Mode(case / "0", 0o000):
        got = verify_path(case, checks="next")
    assert (got.verdict, got.rule) == (readable.verdict, readable.rule) == ("reject", "VER-5.4")
    assert _codes(got) == _codes(readable)
    assert NO_CONFIG not in got.not_assessed
    said = [s for s in got.not_assessed if ZERO_UNREAD in s]
    assert len(said) == 1 and str(tmp_path) not in said[0]
    # What was not read is not listed (VER-14): no file under 0/ is an input.
    assert not any(f.path.startswith("0/") for f in got.run.inputs)


def test_next_says_zero_was_not_read_when_it_lists_and_cannot_be_searched(hot_box, tmp_path):
    """Mode 644: `os.listdir` names the files and none can be opened. 0.6.0 read the rest (no
    exception), so the default already rejected; under next `0/` is said."""
    case = _copy(hot_box, tmp_path)
    with _Mode(case / "0", 0o644):
        default = verify_path(case)
        nxt = verify_path(case, checks="next")
    assert (default.verdict, default.rule) == ("reject", "VER-5.4")
    assert not any(ZERO_UNREAD in s for s in default.not_assessed)
    assert (nxt.verdict, nxt.rule) == ("reject", "VER-5.4")
    assert any(ZERO_UNREAD in s for s in nxt.not_assessed)


def test_preflight_under_next_reads_the_rest_and_names_zero(hot_box, tmp_path):
    case = _copy(hot_box, tmp_path)
    with _Mode(case / "0", 0o000):
        default = preflight_path(case)
        nxt = preflight_path(case, checks="next")
    assert not any(ZERO_UNREAD in s for s in default.not_checked)
    assert any(ZERO_UNREAD in s for s in nxt.not_checked)
    assert "scheme.relaxation" in {r.code for r in nxt.reasons}


def test_the_reader_records_zero_only_when_asked(hot_box, tmp_path):
    from courant.case._foam_reader import ZERO_DIR, read_case_config

    case = _copy(hot_box, tmp_path)
    with _Mode(case / "0", 0o000):
        with pytest.raises(PermissionError):
            read_case_config(str(case), buoyancy=False)
        config = read_case_config(str(case), buoyancy=True, directives=True, steady_ddt=True,
                                  zero_unlisted=True)
    assert config.unreadable == {ZERO_DIR: "PermissionError: Permission denied"}
    assert config.relaxation_factors and config.solver == "buoyantSimpleFoam"
    assert config.boundary == {}


# ---------------------------------------------------------------- never a traceback
@pytest.mark.parametrize("checks", ["default", "next"])
@pytest.mark.parametrize("rel,mode", [
    ("0", 0o000), ("0", 0o644), ("2243", 0o000), ("2243", 0o644),
    ("constant", 0o000), ("constant", 0o644), ("constant/polyMesh", 0o000),
    ("constant/polyMesh", 0o644), ("system", 0o000), ("postProcessing", 0o000),
    ("postProcessing", 0o644)],
    ids=lambda x: oct(x)[2:] if isinstance(x, int) else str(x).replace("/", "-"))
def test_no_directory_a_verdict_reads_ends_it_in_a_traceback(hot_box, tmp_path, rel, mode,
                                                           checks):
    """0.6.0 under next ended in an internal error on every one of these but `system` (and on
    `postProcessing` under the default too): `Path.is_dir`/`is_file` on a path under a directory
    that cannot be searched, or an unguarded listing (`converge.series.discover`)."""
    case = _copy(hot_box, tmp_path)
    with _Mode(case / rel, mode):
        v = verify_path(case, checks=checks)
    assert v.verdict in ("accept", "reject", "insufficient")
    assert v.to_json()


def test_the_cli_exits_with_a_verdict_not_an_internal_error(hot_box, tmp_path):
    from typer.testing import CliRunner

    from courant.cli.main import app

    case = _copy(hot_box, tmp_path)
    with _Mode(case / "0", 0o000):
        result = CliRunner().invoke(app, ["verify", str(case), "--json", "--checks", "next"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["rule"] == "VER-5.4"


# ---------------------------------------------------------------- the one test of a path
def test_kind_tells_absent_from_cannot_be_told(tmp_path):
    from courant.case.reads import is_dir, is_file, kind, reason, unknown

    d = tmp_path / "d"
    (d / "polyMesh").mkdir(parents=True)
    (d / "owner").write_text("x")
    assert kind(d / "polyMesh") is True and kind(d / "owner", stat.S_ISREG) is True
    assert kind(d / "nothing") is False and unknown(d / "nothing") is None
    assert kind(d / "owner") is False                       # a file is not a directory
    with _Mode(d, 0o000):
        assert kind(d / "polyMesh") is None
        assert kind(d / "owner", stat.S_ISREG) is None
        assert reason(unknown(d / "polyMesh")) == "PermissionError: Permission denied"
        assert is_dir(d / "polyMesh") is False and is_file(d / "owner") is False
    assert kind("a\0b") is False


def test_a_start_time_that_cannot_be_searched_leaves_the_mesh_instance_unknown(hot_box, tmp_path):
    """`findInstance` would look in `0/` for a `polyMesh/`: one that cannot be searched may hold
    the mesh the run read, so the reader does not name `constant/polyMesh` in its place."""
    from courant.case.mesh import _instance, solver_mesh

    case = _copy(hot_box, tmp_path)
    assert solver_mesh(case).dirs == (case / "constant" / "polyMesh",)
    with _Mode(case / "0", 0o000):
        d, why = _instance(case, 0.0)
        found = solver_mesh(case)
    assert d is None and why == "time 0 cannot be searched for its own polyMesh"
    assert found.dirs == () and found.why == why
    with _Mode(case / "constant" / "polyMesh", 0o000):
        assert _instance(case, 0.0) == (None, "constant/polyMesh cannot be searched")
    # A later time that cannot be searched is not the run's start: the instance stands.
    with _Mode(case / "2243", 0o000):
        assert _instance(case, 0.0) == (case / "constant" / "polyMesh", None)
