"""Verdict-path audit, the input and output edges (VPA-CLI-1..5, VPA-INGEST-3/4, VPA-ORDER-1/2).

Under the default none of these fixes changes a verdict or its exit code: they make an unreadable
input an input error (exit 4) instead of an internal error, keep a record's verdict in version 2,
and make every surface name the reason a run is insufficient. Two refuse what 0.7.0 judged, a
`--spec-sha256` that is not 64 hexadecimal digits (`spec.lock_strict`) and an option that does
nothing (`cli.options_strict`), and each is behind `--checks next` (GATE-13): each test of a refusal
holds the default's verdict too.
"""

from __future__ import annotations

import gzip
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.cli.main import app
from courant.verify import InputError

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "tests" / "fixtures" / "gate-repo" / "runs"
DUCT = ROOT / "tests" / "fixtures" / "real_openfoam" / "duct" / "converged"
HOT_BOX = ROOT / "tests" / "fixtures" / "real_openfoam" / "hot_box" / "converged"


def _cli(*args):
    return CliRunner().invoke(app, [str(a) for a in args])


def _cut_gz(dst: Path) -> Path:
    data = (RUNS / "cavity_healthy.log.gz").read_bytes()
    dst.write_bytes(data[: len(data) // 2])
    return dst


@pytest.fixture
def duct(tmp_path) -> Path:
    """The converged duct with its log as plain text (a case directory's log is never a .gz)."""
    case = tmp_path / "duct"
    shutil.copytree(DUCT, case)
    gz = case / "log.rhoSimpleFoam.gz"
    (case / "log.rhoSimpleFoam").write_bytes(gzip.decompress(gz.read_bytes()))
    gz.unlink()
    return case


@pytest.fixture
def spec(tmp_path, duct) -> Path:
    from courant.spec import draft_spec

    path = tmp_path / "spec.json"
    path.write_text(draft_spec(str(duct)).to_json())
    return path


def _sha(path: Path) -> str:
    from courant.spec import load_spec

    return load_spec(path)[1]


# ------------------------------------------------------------------ VPA-CLI-3, VPA-INGEST-4
def test_vpa_cli_3_a_cut_gzip_log_is_an_input_error(tmp_path):
    """VPA-CLI-3: a .log.gz cut to half its bytes is exit 4 with one line saying it could not be
    decompressed, not 'internal error (EOFError)'."""
    r = _cli("verify", _cut_gz(tmp_path / "run.log.gz"))
    assert r.exit_code == 4
    assert "internal error" not in r.output
    assert "could not be" in r.output
    assert len(r.output.strip().splitlines()) == 1


def test_vpa_cli_3_a_plain_file_named_gz_is_an_input_error(tmp_path):
    """VPA-CLI-3: a plain text file named .log.gz (BadGzipFile) is exit 4, not an internal error."""
    log = tmp_path / "run.log.gz"
    log.write_text("Time = 1\nEnd\n")
    r = _cli("verify", log)
    assert r.exit_code == 4
    assert "internal error" not in r.output and "could not be" in r.output


def test_vpa_cli_3_a_cut_gzip_log_does_not_abort_a_batch(tmp_path):
    """VPA-CLI-3: one cut .log.gz in a batch is that run's ERROR line; the healthy run beside it
    is still judged and every run gets its line."""
    shutil.copy(RUNS / "cavity_healthy.log.gz", tmp_path / "cavity_healthy.log.gz")
    shutil.copy(RUNS / "cavity_healthy.case.json", tmp_path / "cavity_healthy.case.json")
    _cut_gz(tmp_path / "cut.log.gz")
    r = _cli("verify", "--batch", tmp_path, "--no-cache")
    assert "internal error" not in r.output
    lines = r.output.splitlines()
    [healthy] = [line for line in lines if line.rstrip().endswith("cavity_healthy.log.gz")]
    [cut] = [line for line in lines if "cut.log.gz" in line]
    assert healthy.split()[0] in ("accept", "reject", "insufficient")
    assert cut.startswith("ERROR") and "could not be" in cut
    assert "1 of 2 unreadable" in r.output
    assert r.exit_code in (0, 2, 3, 4)


def test_vpa_ingest_4_a_log_that_cannot_be_opened_is_an_input_error(tmp_path, monkeypatch):
    """VPA-INGEST-4: a log that cannot be opened (PermissionError) is an InputError naming the
    file, not a traceback."""
    from courant.verify.ingest import from_log

    log = tmp_path / "run.log"
    log.write_text("Time = 1\nEnd\n")
    real = Path.read_text

    def denied(self, *a, **k):
        if self == log:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(InputError, match="run.log"):
        from_log(log)


def test_vpa_ingest_4_a_sidecar_that_is_not_an_object_is_an_input_error(tmp_path):
    """VPA-INGEST-4: a sidecar holding JSON that is not an object (`[]`) is an InputError."""
    from courant.verify.ingest import read_sidecar

    cj = tmp_path / "run.case.json"
    cj.write_text("[]")
    with pytest.raises(InputError, match="run.case.json"):
        read_sidecar(cj)


# ------------------------------------------------------------------------------ VPA-ORDER-1
@pytest.mark.parametrize("given", ["", "   ", "abc", "g" * 64])
def test_vpa_order_1_lock_hash_refuses_anything_but_64_hex(given):
    """VPA-ORDER-1: the one validator of a lock hash refuses empty, blank and malformed values."""
    from courant.verify.ingest import lock_hash

    with pytest.raises(InputError, match="64-hex SHA-256"):
        lock_hash(given)


def test_vpa_order_1_lock_hash_normalises_a_valid_hash():
    """VPA-ORDER-1: a valid hash is stripped and lower-cased; None stays None."""
    from courant.verify.ingest import lock_hash

    assert lock_hash(None) is None
    assert lock_hash(f"  {'AB' * 32} ") == "ab" * 32


@pytest.mark.parametrize("given", ["", "   ", "abc"])
def test_vpa_order_1_next_refuses_a_lock_that_is_not_64_hex(duct, spec, given):
    """VPA-ORDER-1, SEC070-18, `spec.lock_strict`: under `--checks next`, `--spec-sha256 "$LOCK"`
    with LOCK unset, blank or malformed is exit 4, not a spec judged as if no lock was asked for."""
    r = _cli("verify", duct, "--spec", spec, "--spec-sha256", given, "--checks", "next")
    assert r.exit_code == 4
    assert "64-hex SHA-256" in r.output


@pytest.mark.parametrize("given", ["", "   "])
def test_vpa_order_1_the_default_reads_an_empty_or_blank_lock_as_no_lock(duct, spec, given):
    """GATE-13: under the default an empty or blank lock keeps 0.7.0's verdict and exit code. An
    empty one is byte for byte the verdict with the option left out; a blank one differs there
    only in the conformance block's `locked_sha256`, "" where that has null, as in 0.7.0."""
    plain = _cli("verify", duct, "--spec", spec, "--json")
    assert plain.exit_code == 0, plain.output
    r = _cli("verify", duct, "--spec", spec, "--spec-sha256", given, "--json")
    assert r.exit_code == 0
    if not given:
        assert r.stdout == plain.stdout
    got, want = json.loads(r.stdout), json.loads(plain.stdout)
    assert got["conformance"].pop("locked_sha256") == (None if not given else "")
    want["conformance"].pop("locked_sha256")
    assert got == want


def test_vpa_order_1_the_default_keeps_a_malformed_lock_as_ver_5_8(duct, spec):
    """GATE-13: under the default a malformed lock is what it was in 0.7.0: it does not match, so
    the verdict is insufficient VER-5.8, exit 3, with its document."""
    r = _cli("verify", duct, "--spec", spec, "--spec-sha256", "abc", "--json")
    assert r.exit_code == 3, r.output
    got = json.loads(r.stdout)
    assert (got["verdict"], got["rule"]) == ("insufficient", "VER-5.8")


def test_vpa_order_1_an_upper_case_lock_is_still_accepted(duct, spec):
    """VPA-ORDER-1: a valid hash in upper case is the lock (basis validity+conformance)."""
    for mode in ("default", "next"):
        r = _cli("verify", duct, "--spec", spec, "--spec-sha256", _sha(spec).upper(), "--json",
                 "--checks", mode)
        assert r.exit_code != 4, r.output
        assert json.loads(r.stdout)["basis"] == "validity+conformance"


def test_vpa_order_1_an_empty_lock_without_a_spec(duct):
    """`spec.lock_strict`: `--spec-sha256 ""` without `--spec` is exit 4 under next. Under the
    default it is 0.7.0's: nothing to lock, so the run's own verdict; a blank one needs --spec."""
    assert _cli("verify", duct, "--spec-sha256", "", "--checks", "next").exit_code == 4
    assert _cli("verify", duct, "--spec-sha256", "").exit_code == _cli("verify", duct).exit_code
    r = _cli("verify", duct, "--spec-sha256", "   ")
    assert r.exit_code == 4 and "--spec-sha256 needs --spec" in r.output


def test_vpa_order_1_a_batch_keeps_the_default(duct, spec):
    """GATE-13: a batch runs the default checks alone, so an empty lock is no lock there, and a
    malformed one VER-5.8."""
    empty = _cli("verify", "--batch", duct.parent, "--spec", spec, "--spec-sha256", "",
                 "--no-cache")
    plain = _cli("verify", "--batch", duct.parent, "--spec", spec, "--no-cache")
    assert (empty.exit_code, empty.stdout) == (plain.exit_code, plain.stdout)
    bad = _cli("verify", "--batch", duct.parent, "--spec", spec, "--spec-sha256", "abc",
               "--no-cache")
    assert bad.exit_code == 3 and "VER-5.8" in bad.stdout


def test_vpa_order_1_check_pack(tmp_path, duct, spec):
    """`spec.lock_strict`: `--check-pack --spec-sha256 ""` is refused under next. Under the default
    it is 0.7.0's, no lock asked: the check passes, and a malformed lock is a mismatch (exit 5),
    which a pack whose own manifest holds one gets too (`verify_path` under the default)."""
    from courant.verify.pack import check_pack

    pack = tmp_path / "pack"
    r = _cli("verify", duct, "--spec", spec, "--spec-sha256", _sha(spec), "--pack", pack)
    assert r.exit_code != 4, r.output
    assert _cli("verify", "--check-pack", pack, "--root", duct).exit_code == 0
    r = _cli("verify", "--check-pack", pack, "--root", duct, "--spec-sha256", "", "--checks",
             "next")
    assert r.exit_code == 4 and "64-hex SHA-256" in r.output
    assert _cli("verify", "--check-pack", pack, "--root", duct, "--spec-sha256", "").exit_code == 0
    assert check_pack(pack, duct, spec_sha256="  ")[0]
    assert _cli("verify", "--check-pack", pack, "--root", duct, "--spec-sha256",
                "abc").exit_code == 5


def test_vpa_order_1_spec_check(duct, spec):
    """`spec.lock_strict`: `courant spec check --spec-sha256 ""` is exit 4 under next; under the
    default it is 0.7.0's, no lock, and a malformed lock is SPECIFICATION CHANGED, exit 3."""
    r = _cli("spec", "check", duct, "--spec", spec, "--spec-sha256", "", "--checks", "next")
    assert r.exit_code == 4 and "64-hex SHA-256" in r.output
    r = _cli("spec", "check", duct, "--spec", spec, "--spec-sha256", "")
    assert r.exit_code == _cli("spec", "check", duct, "--spec", spec).exit_code != 4
    r = _cli("spec", "check", duct, "--spec", spec, "--spec-sha256", "abc")
    assert r.exit_code == 3 and r.output.startswith("SPECIFICATION CHANGED")


def test_vpa_order_1_with_spec_and_the_mcp_tools(tmp_path, duct, spec):
    """`spec.lock_strict` is asked of the inputs' own checks (`ingest.lock_of`): `with_spec` refuses
    under next. The MCP tools and `verify_path` run the default, so they keep 0.7.0's verdicts: an
    empty lock is no lock, and a malformed one VER-5.8."""
    from courant.mcp import tools
    from courant.verify import from_path, with_spec

    with pytest.raises(InputError, match="64-hex SHA-256"):
        with_spec(from_path(duct, checks="next"), spec, "")
    confine = tools.Confinement([tmp_path])
    got = tools.verify_run(str(duct), spec=str(spec), spec_sha256="", confine=confine)
    assert got == tools.verify_run(str(duct), spec=str(spec), confine=confine)
    got = tools.verify_run(str(duct), spec=str(spec), spec_sha256="abc", confine=confine)
    assert got["rule"] == "VER-5.8" and "error" not in got


# ------------------------------------------------------------------------------ VPA-ORDER-2
def _extracted_mismatch(tmp_path: Path, spec: Path) -> Path:
    data = json.loads(spec.read_text())
    data["provenance"] = {"extracted_by": "courant spec extract"}
    item = next(i for i in data["items"] if i["name"] == "T at inlet")
    item["value"] = 350.0
    out = tmp_path / "extracted.json"
    out.write_text(json.dumps(data))
    return out


def test_vpa_order_2_spec_check_does_not_reject_on_an_unconfirmed_extraction(tmp_path, duct,
                                                                            spec):
    """VPA-ORDER-2: an extracted, unconfirmed specification that does not conform is not
    binding: `courant spec check` exits 3 (as `courant verify --spec` does, VER-5.12) and says
    so in its head line, never 2."""
    extracted = _extracted_mismatch(tmp_path, spec)
    r = _cli("spec", "check", duct, "--spec", extracted)
    assert r.exit_code == 3, r.output
    assert "UNCONFIRMED EXTRACTION: not binding" in r.output.splitlines()[0]
    assert _cli("spec", "check", duct, "--spec", extracted, "--json").exit_code == 3
    v = _cli("verify", duct, "--spec", extracted)
    assert v.exit_code == 3


def test_vpa_order_2_a_confirmed_mismatch_still_does_not_conform(tmp_path, duct, spec):
    """VPA-ORDER-2: once confirmed, the same mismatch is binding: exit 2."""
    data = json.loads(_extracted_mismatch(tmp_path, spec).read_text())
    data["provenance"]["confirmed_by"] = "S"
    confirmed = tmp_path / "confirmed.json"
    confirmed.write_text(json.dumps(data))
    r = _cli("spec", "check", duct, "--spec", confirmed)
    assert r.exit_code == 2 and r.output.startswith("DOES NOT CONFORM")


# -------------------------------------------------------------------------------- VPA-CLI-1
def test_vpa_cli_1_a_schema_1_record_checks(tmp_path, duct, spec):
    """VPA-CLI-1: `--schema 1 --record` writes the version 2 verdict, so the record re-derives."""
    rec = tmp_path / "r.json"
    r = _cli("verify", duct, "--schema", "1", "--record", rec)
    assert r.exit_code != 4, r.output
    assert json.loads(rec.read_text())["verdict"]["schema"] == "courant.verdict/2"
    c = _cli("verify", "--check-record", rec, "--root", duct)
    assert c.exit_code == 0 and c.output.startswith("OK"), c.output


def test_vpa_cli_1_an_extracted_spec_keeps_its_basis_in_a_schema_1_record(tmp_path, duct, spec):
    """VPA-CLI-1: an unconfirmed extracted spec is recorded on basis validity+extracted-spec
    under --schema 1 too, and the record checks."""
    extracted = _extracted_mismatch(tmp_path, spec)
    rec = tmp_path / "r.json"
    _cli("verify", duct, "--schema", "1", "--spec", extracted, "--record", rec)
    assert json.loads(rec.read_text())["verdict"]["basis"] == "validity+extracted-spec"
    assert _cli("verify", "--check-record", rec, "--root", duct).exit_code == 0


def test_vpa_cli_1_write_record_refuses_a_version_1_verdict(tmp_path, duct):
    """VPA-CLI-1: a record is written from the version 2 verdict; a version 1 verdict is refused
    instead of being converted back with what version 1 dropped. (`build_record` still converts
    one: tests/unit/test_record.py::test_a_version_1_verdict_is_recorded_as_version_2 pins it.)"""
    from courant.record.record import write_record
    from courant.verify import verify_path

    v = verify_path(duct)
    write_record(v, tmp_path / "ok.json", target_name="duct")
    with pytest.raises(TypeError):
        write_record(v.to_v1(), tmp_path / "v1.json", target_name="duct")
    assert not (tmp_path / "v1.json").exists()


# -------------------------------------------------------------------------------- VPA-CLI-2
def test_vpa_cli_2_every_renderer_names_the_insufficient_reason(tmp_path, duct, spec):
    """VPA-CLI-2: for a VER-5.8 run in a batch, text, Markdown, HTML, JSON and JUnit all name
    `insufficient.spec_hash_mismatch`; JUnit keeps `skipped` (REP-7) with that reason as its
    message, not the first not-assessed sentence."""
    from courant.verify import (
        from_path,
        render_json,
        render_markdown,
        render_text,
        verify_inputs,
        with_spec,
    )
    from courant.verify.html import render_html

    j = tmp_path / "j.xml"
    r = _cli("verify", "--batch", duct.parent, "--spec", spec, "--spec-sha256", "0" * 64,
             "--no-cache", "--junit", j)
    assert r.exit_code == 3, r.output
    skipped = ET.parse(j).getroot().find("testsuite/testcase/skipped")
    assert skipped is not None
    message = skipped.get("message")
    assert "insufficient.spec_hash_mismatch" in message
    inputs = with_spec(from_path(duct), spec, "0" * 64)
    v = verify_inputs(inputs)
    assert v.not_assessed[0] not in message
    [why] = [r for r in v.reasons if r.code == "insufficient.spec_hash_mismatch"]
    assert why.summary in message
    for page in (render_text(v), render_markdown(v), render_html(v, inputs), render_json(v)):
        assert "spec_hash_mismatch" in page or why.summary in page


def test_vpa_cli_2_a_version_1_insufficient_falls_back_to_not_assessed(tmp_path, duct, spec):
    """VPA-CLI-2: a version 1 verdict has no `insufficient.*` reason; JUnit names its first
    not-assessed sentence as before."""
    from courant.verify import verify_path
    from courant.verify.junit import render_junit

    v1 = verify_path(duct, spec=spec, spec_locked_sha256="0" * 64).to_v1()
    out = render_junit([(duct, v1, None)], duct.parent)
    message = ET.fromstring(out).find("testsuite/testcase/skipped").get("message")
    assert message == f"NOT_ASSESSABLE: {v1.not_assessed[0]}"


# -------------------------------------------------------------------------------- VPA-CLI-5
def test_vpa_cli_5_an_unwritable_html_is_one_line_exit_4(tmp_path, duct):
    """VPA-CLI-5: `--html nodir/x.html` is exit 4 with one 'cannot write' line, no traceback."""
    r = _cli("verify", duct, "--html", tmp_path / "nodir" / "x.html")
    assert r.exit_code == 4
    assert "internal error" not in r.output
    assert "courant verify: cannot write" in r.output


def test_vpa_cli_5_next_refuses_an_unknown_report_format(duct):
    """VPA-CLI-5, `cli.options_strict`: under next `--report X` with X not md or html is exit 4."""
    r = _cli("verify", duct, "--report", "pdf", "--checks", "next")
    assert r.exit_code == 4 and "--report" in r.output


def test_vpa_cli_5_next_refuses_junit_without_batch(tmp_path, duct):
    """VPA-CLI-5, `cli.options_strict`: under next `--junit` without `--batch` is exit 4."""
    r = _cli("verify", duct, "--junit", tmp_path / "j.xml", "--checks", "next")
    assert r.exit_code == 4 and "--junit needs --batch" in r.output


@pytest.mark.parametrize("option", [("--report", "pdf"), ("--junit", "j.xml")])
def test_vpa_cli_5_the_default_warns_and_keeps_the_verdict(tmp_path, duct, spec, option,
                                                           monkeypatch):
    """GATE-13: under the default each is one warning on stderr, and the verdict, its exit code
    and the report on stdout are 0.7.0's; no JUnit file is written."""
    monkeypatch.chdir(tmp_path)
    plain = _cli("verify", duct, "--spec", spec)
    r = _cli("verify", duct, "--spec", spec, *option)
    assert plain.exit_code == 0, plain.output
    assert (r.exit_code, r.stdout) == (0, plain.stdout)
    assert f"warning: {option[0]}" in r.stderr
    assert not (tmp_path / "j.xml").exists()


# ------------------------------------------------------------------------------ open findings
@pytest.mark.xfail(strict=True, reason="VPA-CLI-4: open, S to decide (VER-12 is silent): "
                   "summarise ranks insufficient (3) above unreadable (4); courant gate gives 4")
def test_vpa_cli_4_an_unreadable_run_outranks_insufficient(duct, spec):
    """VPA-CLI-4: a batch with an insufficient run and an unreadable run exits 4, as the gate."""
    from courant.verify import verify_path
    from courant.verify.batch import summarise

    v = verify_path(duct, spec=spec, spec_locked_sha256="0" * 64)
    assert v.verdict == "insufficient"
    _, code = summarise([(duct, v, None), (duct, None, "unreadable")])
    assert code == 4


def test_vpa_ingest_3_a_case_with_only_a_gz_log_is_read(tmp_path):
    """VPA-INGEST-3: from_case_dir reads a case whose solver log is gzipped."""
    from courant.verify.ingest import from_case_dir

    case = tmp_path / "hot_box"
    shutil.copytree(HOT_BOX, case)
    inputs = from_case_dir(case)
    assert inputs.log_text
