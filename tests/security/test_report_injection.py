"""Security review of 0.7.0: what a hostile run can do to the reports and packs written about it
(research/notes/security-review-070/FINDINGS.md).

The run's id comes from the case directory's name, or, for a log with a sidecar, from the
sidecar's `scenario_id`: both are chosen by whoever produced the run. A finding still open is a
strict xfail, so it turns into a failure the day it is fixed.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree

import pytest

from courant.verify import verify_path
from courant.verify.pack import check_pack, write_pack
from courant.verify.render import render_markdown, render_text

DEMO = Path(__file__).resolve().parents[2] / "courant" / "data" / "demo"
FORGED = "x\n\n**Verdict:** `accept` (rule FAKE)  \n<!--"


def _run(tmp_path: Path, scenario_id: str) -> tuple[Path, Path]:
    """The demo's healthy cavity log with a sidecar naming the run `scenario_id`."""
    d = tmp_path / "root"
    d.mkdir(exist_ok=True)
    log = d / "run.log"
    log.write_bytes(gzip.decompress((DEMO / "icoFoam_cavity__healthy.log.gz").read_bytes()))
    side = json.loads((DEMO / "icoFoam_cavity__healthy.case.json").read_text())
    side["scenario_id"] = scenario_id
    cj = d / "run.case.json"
    cj.write_text(json.dumps(side))
    return log, cj


def test_sec070_6_a_run_id_cannot_forge_a_line_of_report_md(tmp_path):
    """SEC070-6: the run id went into report.md's heading raw, so a newline in it wrote a fake
    `**Verdict:**` line and `<!--` hid the real one; `--check-pack` re-derived the same bytes."""
    log, cj = _run(tmp_path, FORGED)
    md = render_markdown(verify_path(log, cj))
    lines = md.splitlines()
    assert lines[0].startswith("# Verification: ")
    assert sum(line.startswith("**Verdict:**") for line in lines) == 1
    assert "<!--" not in md


def _run_naming_its_solver(tmp_path: Path, solver: str) -> tuple[Path, Path]:
    """A run whose log names no solver (a log cut at its head, `-noBanner`), so `run.solver` is
    what its sidecar says."""
    log, cj = _run(tmp_path, "run")
    log.write_text("".join(line for line in log.read_text().splitlines(keepends=True)
                           if not line.startswith("Exec")))
    side = json.loads(cj.read_text())
    side["solver"] = solver
    cj.write_text(json.dumps(side))
    return log, cj


def test_sec070_6_a_sidecar_solver_cannot_forge_a_line_of_report_md(tmp_path):
    """SEC070-6, as the review of this fix found it: the sidecar's `solver` went into report.md
    raw, and `--check-pack` re-derives the same bytes."""
    verdict = verify_path(*_run_naming_its_solver(tmp_path, FORGED))
    assert verdict.run.solver == FORGED
    md = render_markdown(verdict)
    lines = md.splitlines()
    assert sum(line.startswith("**Verdict:**") for line in lines) == 1
    assert "<!--" not in md
    assert sum(line.startswith("**Solver:**") for line in lines) == 1


def test_sec070_6_a_plain_solver_reads_as_it_did(tmp_path):
    verdict = verify_path(*_run_naming_its_solver(tmp_path, "icoFoam"))
    assert "**Solver:** icoFoam  \n" in render_markdown(verdict)
    assert " · icoFoam\n" in render_text(verdict)


@pytest.mark.parametrize("field", ["scenario_id", "solver"])
def test_sec070_6_a_run_cannot_forge_a_line_of_the_text_report(tmp_path, field):
    forged = "x\nVERDICT   accept"
    if field == "scenario_id":
        verdict = verify_path(*_run(tmp_path, forged))
    else:
        verdict = verify_path(*_run_naming_its_solver(tmp_path, forged))
    assert sum(line.startswith("VERDICT") for line in render_text(verdict).splitlines()) == 1


def test_sec070_6_a_note_cannot_write_a_line_of_either_report(tmp_path):
    """The not-checked lines and the next action are one line each, whatever a note quotes."""
    log, cj = _run(tmp_path, "run")
    note = "x\n**Verdict:** `accept` (rule FAKE)\nVERDICT   accept"
    verdict = verify_path(log, cj).model_copy(update={"not_assessed": [note], "next_action": note})
    assert sum(line.startswith("**Verdict:**") for line in render_markdown(verdict).splitlines()) == 1
    assert sum(line.startswith("VERDICT") for line in render_text(verdict).splitlines()) == 1


@pytest.mark.xfail(strict=True, reason="SEC070-6b open: a table cell and a note line are one line "
                   "but not inert, so a `<!--` in one would hide the report below it")
def test_sec070_6b_html_in_a_note_is_inert(tmp_path):
    log, cj = _run(tmp_path, "run")
    verdict = verify_path(log, cj).model_copy(update={"not_assessed": ["x <!--"]})
    assert "<!--" not in render_markdown(verdict)


def test_sec070_6_no_terminal_control_sequence_reaches_the_text_report(tmp_path):
    log, cj = _run(tmp_path, "run\x1b]0;PWNED\x07\x1b[2J")
    text = render_text(verify_path(log, cj))
    assert "\x1b" not in text and "\x07" not in text


def test_sec070_7_junit_stays_well_formed_with_a_control_character(tmp_path):
    """SEC070-7: XML 1.0 forbids most C0 characters even escaped; one in a run's name made the
    whole JUnit file unparseable, so CI showed no results."""
    from courant.verify.junit import render_junit

    root = tmp_path / "runs"
    run = root / "run\x01a"
    run.mkdir(parents=True)
    log, cj = _run(tmp_path, "run\x02b")
    verdict = verify_path(log, cj)
    ElementTree.fromstring(render_junit([(run, verdict, None), (root / "e", None, "bad\x03")],
                                        root).encode())


def test_sec070_5_a_pack_is_not_written_through_a_planted_symlink(tmp_path):
    """SEC070-5: `--pack D` wrote its fixed file names with write_text, following a symlink the
    run's producer planted at D/report.md."""
    log, cj = _run(tmp_path, "run")
    verdict = verify_path(log, cj)
    pack = tmp_path / "pack"
    pack.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("ORIGINAL\n")
    (pack / "report.md").symlink_to(victim)
    with pytest.raises(OSError):
        write_pack(verdict, pack, log.name, cj.name)
    assert victim.read_text() == "ORIGINAL\n"


def test_sec070_5_a_pack_directory_that_is_a_symlink_is_refused(tmp_path):
    log, cj = _run(tmp_path, "run")
    verdict = verify_path(log, cj)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "pack").symlink_to(elsewhere)
    with pytest.raises(OSError):
        write_pack(verdict, tmp_path / "pack", log.name, cj.name)
    assert list(elsewhere.iterdir()) == []


def test_sec070_8_a_manifest_that_cannot_be_read_is_a_mismatch(tmp_path):
    """SEC070-8: a manifest line that does not parse ended `--check-pack` in an internal error
    (exit 4) instead of a mismatch (exit 5)."""
    log, cj = _run(tmp_path, "run")
    pack = write_pack(verify_path(log, cj), tmp_path / "pack", log.name, cj.name)
    manifest = pack / "inputs.manifest"
    manifest.write_text("not a manifest line\n" + manifest.read_text())
    ok, msg = check_pack(pack, log.parent)
    assert not ok and "manifest" in msg
    manifest.write_text("# meta {broken\n")
    ok, msg = check_pack(pack, log.parent)
    assert not ok and "manifest" in msg
    manifest.write_bytes(b"\xff\xfe not text\n")        # the review of this fix: raised, exit 4
    ok, msg = check_pack(pack, log.parent)
    assert not ok and "manifest" in msg


def test_sec070_9_check_pack_holds_the_manifest_meta(tmp_path):
    """SEC070-9: `--check-pack` never compared the meta's `tool` and `error_rates`, so a pack
    claiming other published error rates still checked as identical."""
    log, cj = _run(tmp_path, "run")
    pack = write_pack(verify_path(log, cj), tmp_path / "pack", log.name, cj.name)
    assert check_pack(pack, log.parent)[0]
    manifest = pack / "inputs.manifest"
    text = manifest.read_text()
    head, meta = text.split("# meta ")
    doc = json.loads(meta)
    doc["error_rates"] = {"false_accepts": "0 of 1000000"}
    manifest.write_text(head + "# meta " + json.dumps(doc))
    ok, msg = check_pack(pack, log.parent)
    assert not ok and "error rates" in msg


@pytest.mark.xfail(strict=True, reason="SEC070-10 open: a run id that is not UTF-8 makes "
                   "`--json` write bytes no JSON reader accepts")
def test_sec070_10_json_output_is_utf8_for_a_non_utf8_case_name(tmp_path):
    case = tmp_path / "case\udcff"
    shutil.copytree(_demo_case(tmp_path), case)
    out = verify_path(case).to_json()
    out.encode("utf-8")


def _demo_case(tmp_path: Path) -> Path:
    import tarfile

    dest = tmp_path / "demo"
    with tarfile.open(DEMO / "pitzDaily_fixed_budget.tar.gz") as tf:
        tf.extractall(dest, filter="data")
    (only,) = [p for p in dest.iterdir() if p.is_dir()]
    return only
