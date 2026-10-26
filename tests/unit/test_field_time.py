"""The field checks judge the fields of the run the log records, or say they did not (TRD VER-7b;
hard-proofing audit, 21 Oct 2026).

`read_fields` took the highest time directory at the case root and never compared it with the
log. With `0.5/U` holding a NaN and the log ending at t = 0.5005, the run was rejected (VER-5.3);
add a clean `0.6/U` left by an earlier, longer run and it was accepted at grade A, the NaN never
read. A root reconstructed at 0.1 hid ranks holding a NaN at 0.5 the same way. A `0.5/U` cut
mid-write gave no fields at all, and the verdict said "no time directory with fields was found",
which was false: one was found and could not be read. The module's own rule says reading a time
that is not the result is worse than failing.

Now (GATE-13):
- by default the same directory is read as before, and the verdict says when it is later than the
  log's last step, when `processor*/` hold a later time than the root it read, when the run's
  final fields were not found, and when the fields at that time exist and could not be read;
- under `--checks next` (`fields.log_time`) the fields are read at the latest time the log
  reached, a later directory is named and not read, and the ranks are read when they went further
  than the reconstructed root.

The case is the committed 2000 x 1 cavity fixture with its own log (last step t = 0.5005; written
every 20 steps of 0.0035, so one write interval is 0.07).
"""

from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"

HDR = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volVectorField;\n"
       "    object U;\n}\ndimensions [0 1 -1 0 0 0 0];\n")
BF = ("boundaryField\n{\n    movingWall { type fixedValue; value uniform (1 0 0); }\n"
      "    fixedWalls { type noSlip; }\n    frontAndBack { type empty; }\n}\n")
GOOD = [(0.1 * i, 0.01, 0) for i in range(8)]
NAN = GOOD[:4] + [("nan", "nan", 0)] + GOOD[5:]


def _u(values) -> str:
    rows = "\n".join(f"({a} {b} {c})" for a, b, c in values)
    return HDR + f"internalField nonuniform List<vector> {len(values)}\n(\n{rows}\n);\n" + BF


CUT = _u(NAN)[:_u(NAN).index("(nan") + 12]           # written up to just past the NaN row


def _case(tmp_path: Path, files: dict[str, str]) -> Path:
    case = tmp_path / "case"
    shutil.copytree(CAVITY, case, ignore=shutil.ignore_patterns("*.gz"))
    with gzip.open(CAVITY / "log.icoFoam.gz", "rt") as fh:
        (case / "log.icoFoam").write_text(fh.read())
    for rel, text in files.items():
        (case / rel).parent.mkdir(parents=True, exist_ok=True)
        (case / rel).write_text(text)
    return case


def _verify(case: Path, checks: str):
    from courant.verify import verify_path

    return verify_path(case, checks=checks)


def _fields_lines(v) -> list[str]:
    return [s for s in v.not_assessed if s.startswith(("Solution fields", "Time directories"))]


def _unread_reasons(v) -> list[str]:
    return [r.summary for r in v.reasons if r.code == "fields.unread"]


NO_FIELDS = "Solution fields were not read: no time directory with fields was found."

STALE_LATER = {"0.5/U": _u(NAN), "0.6/U": _u(GOOD)}
PARTIAL = {"0.1/U": _u(GOOD), "processor0/0.1/U": _u(GOOD), "processor1/0.1/U": _u(GOOD),
           "processor0/0.5/U": _u(NAN), "processor1/0.5/U": _u(NAN)}


# ---------------------------------------------------------------- as before
@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_nan_at_the_logs_last_time_is_rejected_in_both_modes(tmp_path, checks):
    v = _verify(_case(tmp_path, {"0.5/U": _u(NAN)}), checks)
    assert (v.verdict, v.rule) == ("reject", "VER-5.3"), (v.verdict, v.rule)
    assert _fields_lines(v) == []


@pytest.mark.parametrize("checks", ["default", "next"])
def test_clean_fields_at_the_logs_last_time_say_nothing_new(tmp_path, checks):
    v = _verify(_case(tmp_path, {"0.5/U": _u(GOOD)}), checks)
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")
    assert _fields_lines(v) == []


@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_case_with_no_time_directory_keeps_the_fixed_sentence(tmp_path, checks):
    v = _verify(_case(tmp_path, {}), checks)
    assert _fields_lines(v) == [NO_FIELDS]


# ---------------------------------------------------------------- a later directory the log never reached
def test_by_default_a_later_directory_is_read_and_the_verdict_says_it_is_later(tmp_path):
    v = _verify(_case(tmp_path, STALE_LATER), "default")
    assert v.verdict == "accept", "GATE-13: no default verdict moves"
    assert _fields_lines(v) == [
        "Solution fields were read at t = 0.6, later than the log's last time step "
        "(t = 0.5005): they may be another run's, and this run's final fields were not checked."]


def test_under_next_the_fields_are_read_at_the_logs_last_time(tmp_path):
    v = _verify(_case(tmp_path, STALE_LATER), "next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.3"), (v.verdict, v.rule)
    assert _fields_lines(v) == [
        "Time directories later than the log's last time step (t = 0.5005) were not read "
        "(0.6): they are not part of the run the log records."]
    listed = {f.path for f in v.run.inputs}
    assert "0.5/U" in listed and "0.6/U" not in listed


def test_a_stale_diverged_later_directory_no_longer_rejects_a_clean_run_under_next(tmp_path):
    """The reverse: the later directory is the bad one. By default it is still read (and the
    verdict says it is later than the log); under next the run's own fields are."""
    files = {"0.5/U": _u(GOOD), "0.6/U": _u(NAN)}
    assert _verify(_case(tmp_path, files), "default").verdict == "reject"
    assert _verify(_case(tmp_path / "n", files), "next").verdict == "accept"


# ---------------------------------------------------------------- a partial reconstruct
def test_by_default_a_partial_reconstruct_reads_the_root_and_says_the_ranks_went_further(
        tmp_path):
    v = _verify(_case(tmp_path, PARTIAL), "default")
    assert v.verdict == "accept"
    assert _fields_lines(v) == [
        "Solution fields were read from the case root at t = 0.1; processor*/ hold a later time "
        "(t = 0.5), whose fields were not checked."]


def test_under_next_a_partial_reconstruct_reads_the_ranks(tmp_path):
    v = _verify(_case(tmp_path, PARTIAL), "next")
    assert (v.verdict, v.rule) == ("reject", "VER-5.3"), (v.verdict, v.rule)
    assert _fields_lines(v) == []
    assert {"processor0/0.5/U", "processor1/0.5/U"} <= {f.path for f in v.run.inputs}


# ---------------------------------------------------------------- fields that exist and do not parse
@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_field_cut_mid_write_is_named_not_called_missing(tmp_path, checks):
    """By default the grade is v0.5.0's (A, the field checks skipped); under next a field the run
    wrote at the time judged and that could not be read caps it at B (`fields.unread`, VER-7b:
    the review of 21 Oct 2026 found the cut field still accepted at grade A in both modes)."""
    v = _verify(_case(tmp_path, {"0.4/U": _u(GOOD), "0.5/U": CUT}), checks)
    assert v.verdict == "accept"
    lines = _fields_lines(v)
    assert NO_FIELDS not in lines, lines
    assert lines == ["Solution fields at t = 0.5 were found and could not be read (U): the "
                     "checks that need them were not run."], lines
    assert (v.grade, _unread_reasons(v)) == (("A", []) if checks == "default" else (
        "B", ["Solution fields at t = 0.5 could not be read (U): the run's state at the time "
              "judged was not checked"])), (v.grade, v.reasons)


@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_collated_field_is_named_as_not_read(tmp_path, checks):
    """`processors2/0.5/U` as the collated file handler writes it (`decomposedBlockData`)."""
    inner = _u(NAN)
    block = f"{len(inner)}\n({inner})\n"
    collated = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class decomposedBlockData;\n"
                "    location \"0.5\";\n    object U;\n}\n" + f"2\n(\n{block}{block})\n")
    v = _verify(_case(tmp_path, {"processors2/0.5/U": collated}), checks)
    assert v.verdict == "accept"
    assert _fields_lines(v) == ["Solution fields at t = 0.5 were found and could not be read "
                                "(U): the checks that need them were not run."]
    assert v.grade == ("A" if checks == "default" else "B")


def test_one_unreadable_field_beside_readable_ones_is_named(tmp_path):
    files = {"0.5/U": _u(GOOD), "0.5/p": "FoamFile { class volScalarField; }\ninternalField ("}
    v = _verify(_case(tmp_path, files), "default")
    assert "Solution fields at t = 0.5 were found and could not be read (p): the checks that "\
           "need them were not run." in _fields_lines(v)


# ---------------------------------------------------------------- the run's final fields missing
@pytest.mark.parametrize("checks", ["default", "next"])
def test_fields_more_than_a_write_interval_before_the_logs_end_are_said_to_be_early(tmp_path,
                                                                                   checks):
    v = _verify(_case(tmp_path, {"0.2/U": _u(GOOD)}), checks)
    assert _fields_lines(v) == [
        "Solution fields were read at t = 0.2, the latest time written, more than one write "
        "interval before the log's last time step (t = 0.5005): the run's final fields were "
        "not found."]


@pytest.mark.parametrize("checks", ["default", "next"])
def test_the_last_write_within_one_interval_of_the_end_is_the_runs_final_state(tmp_path, checks):
    """Written every 20 steps, the run's last write (0.49) is 3 steps before its last step."""
    v = _verify(_case(tmp_path, {"0.49/U": _u(GOOD)}), checks)
    assert _fields_lines(v) == []


# ---------------------------------------------------------------- the reader
def test_read_fields_stops_at_until_and_names_what_it_left(tmp_path):
    from courant.verification._fluid_reader import read_fields

    case = str(_case(tmp_path, STALE_LATER))
    assert read_fields(case).time == 0.6, "the default reader is v0.5.0's"
    f = read_fields(case, until=0.5005)
    assert (f.time, f.source_time, f.after_log) == (0.5, "0.5", ("0.6",))


def test_read_fields_names_the_unreadable_files_at_the_time_it_read(tmp_path):
    from courant.verification._fluid_reader import read_fields

    f = read_fields(str(_case(tmp_path, {"0.4/U": _u(GOOD), "0.5/U": CUT})))
    assert (f.source_time, f.internal, f.unread) == ("0.5", {}, ("U",))


def test_a_field_file_holding_no_field_of_its_shape_is_named_as_not_read(tmp_path, monkeypatch):
    """`_internal` gives None for a file that parsed and holds no field of the shape asked (a
    collated block in some layouts): read, so not missing."""
    from courant.verification import _fluid_reader

    case = str(_case(tmp_path, {"0.5/U": _u(GOOD)}))

    def no_field(path, vector):
        open(path).close()                      # an absent file is missing, as foamlib finds it
        return None

    monkeypatch.setattr(_fluid_reader, "_internal", no_field)
    f = _fluid_reader.read_fields(case)
    assert (f.source_time, f.internal, f.unread) == ("0.5", {}, ("U",))


# ---------------------------------------------------------------- the review of 21 Oct 2026
# One last time step. `courant.spec.as_run.Run.last_time` (the time of the log's last step) is
# what VER-3's end time (ENDED_SHORT, SPEC-13) reads; the field read had a second definition
# (the `ts_sim` of whatever event came last) and a second equal-time tolerance. Pinned by moving
# `Run.last_time`: the field read and the end-time sentence must both follow it.
@pytest.mark.parametrize("checks", ["default", "next"])
def test_the_field_read_and_the_end_time_ask_one_last_time_step(tmp_path, monkeypatch, checks):
    from courant.spec import as_run

    case = _case(tmp_path, {"0.4/U": _u(GOOD), "0.49/U": _u(GOOD)})
    control = case / "system" / "controlDict"
    control.write_text(control.read_text().replace("endTime         0.5;", "endTime         1;"))
    monkeypatch.setattr(as_run.Run, "last_time", property(lambda self: 0.42))
    v = _verify(case, checks)
    ended = [s for s in v.not_assessed if s.startswith("The run ended at")]
    assert len(ended) == 1 and ended[0].startswith("The run ended at t = 0.42 "), v.not_assessed
    assert _fields_lines(v) == {
        "default": ["Solution fields were read at t = 0.49, later than the log's last time step "
                    "(t = 0.42): they may be another run's, and this run's final fields were not "
                    "checked."],
        "next": ["Time directories later than the log's last time step (t = 0.42) were not read "
                 "(0.49): they are not part of the run the log records."]}[checks]


def test_later_than_the_log_is_the_equal_time_rule_conformance_uses():
    from courant.spec.as_run import _same_time
    from courant.verification._fluid_reader import after

    for t, last in ((0.5, 0.5), (0.5 + 1e-12, 0.5), (0.5005, 0.5), (0.4995, 0.5),
                    (1000.0000005, 1000.0), (1000.01, 1000.0), (2e-7, 1e-7)):
        assert after(t, last) == (t > last and not _same_time(t, last)), (t, last)
        assert after(str(t), last) == after(t, last)


# A decomposed case whose ranks did not all read a field: the checks ran on part of the domain,
# so "could not be read: the checks that need them were not run" was false.
RANK_CUT = {"processor0/0.5/U": _u(GOOD), "processor1/0.5/U": CUT}
RANKS = ("Solution fields at t = 0.5 were read on some ranks only (U on 1 of 2): the checks that "
         "need them judged only those ranks' cells.")


@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_field_one_rank_could_not_read_is_said_to_be_read_on_part_of_the_domain(tmp_path,
                                                                                  checks):
    v = _verify(_case(tmp_path, RANK_CUT), checks)
    assert v.verdict == "accept"
    assert _fields_lines(v) == [RANKS]
    assert (v.grade, _unread_reasons(v)) == (("A", []) if checks == "default" else (
        "B", ["Solution fields at t = 0.5 were read on some ranks only (U on 1 of 2): the run's "
              "state at the time judged was checked on part of the domain"])), v.reasons


@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_nan_on_the_ranks_read_is_still_rejected(tmp_path, checks):
    v = _verify(_case(tmp_path, {"processor0/0.5/U": _u(NAN), "processor1/0.5/U": CUT}), checks)
    assert (v.verdict, v.rule) == ("reject", "VER-5.3"), (v.verdict, v.rule)
    assert _fields_lines(v) == [RANKS]


def test_a_field_missing_from_one_rank_is_read_on_part_of_the_domain(tmp_path):
    """Absent from a rank, or present and unreadable there: either way the checks judged the
    other ranks' cells only."""
    v = _verify(_case(tmp_path, {"processor0/0.5/U": _u(GOOD), "processor1/0.5/U.orig": "",
                                 "processor1/0.5/phi": ""}), "default")
    assert _fields_lines(v) == [RANKS], _fields_lines(v)


def test_the_grade_reason_names_a_field_unread_and_one_read_in_part_together():
    from courant.verification.checks import fields_unread
    from courant.verification.fields import FieldData

    r = fields_unread(FieldData(time=0.5, source_time="0.5", unread=("p",),
                                partial=(("U", 1, 2),)))
    assert (r.check_code, r.severity.value, r.detail) == (
        "fields.unread", "minor", "Solution fields at t = 0.5 could not be read (p) or were read "
        "on some ranks only (U on 1 of 2): the run's state at the time judged was not checked "
        "whole")
    assert fields_unread(FieldData(time=0.5, source_time="0.5")) is None


def test_read_fields_counts_the_ranks_each_field_was_read_on(tmp_path):
    from courant.verification._fluid_reader import read_fields

    f = read_fields(str(_case(tmp_path, RANK_CUT)))
    assert (f.source_time, f.unread, f.partial) == ("0.5", (), (("U", 1, 2),))
    assert f.internal["U"].shape == (len(GOOD), 3)


# No directory at or before the log's end, under next: NO_FIELDS said "no time directory with
# fields was found" beside the sentence naming the later one it did not read.
def test_under_next_a_case_with_only_later_directories_is_not_said_to_have_none(tmp_path):
    v = _verify(_case(tmp_path, {"0.6/U": _u(GOOD)}), "next")
    assert _fields_lines(v) == [
        "Solution fields were not read: no time directory with fields was found at or before "
        "the log's last time step (t = 0.5005).",
        "Time directories later than the log's last time step (t = 0.5005) were not read (0.6): "
        "they are not part of the run the log records."]


def test_under_next_an_early_read_is_not_called_the_latest_time_written(tmp_path):
    v = _verify(_case(tmp_path, {"0.2/U": _u(GOOD), "0.6/U": _u(GOOD)}), "next")
    assert _fields_lines(v) == [
        "Solution fields were read at t = 0.2, the latest time written at or before the log's "
        "last time step (t = 0.5005) and more than one write interval before it: the run's final "
        "fields were not found.",
        "Time directories later than the log's last time step (t = 0.5005) were not read (0.6): "
        "they are not part of the run the log records."]


# A field file that exists and cannot be opened is not a missing field.
@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="root opens a file of mode 000")
@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_field_file_that_cannot_be_opened_is_named_not_called_missing(tmp_path, checks):
    case = _case(tmp_path, {"0.5/U": _u(GOOD)})
    (case / "0.5" / "U").chmod(0)
    try:
        v = _verify(case, checks)
    finally:
        (case / "0.5" / "U").chmod(0o644)
    assert _fields_lines(v) == ["Solution fields at t = 0.5 were found and could not be read "
                                "(U): the checks that need them were not run."], _fields_lines(v)
    assert "0.5/U" not in {f.path for f in v.run.inputs}, "VER-14: not read, so not listed"
