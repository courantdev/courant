"""Verified records (TRD REC-1, REC-2; T-REC-1).

`courant verify --record` writes `courant.record/1`; it re-derives byte for byte; the reference
dataset filter admits only a matching `accept` on `validity+conformance`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.record import (
    SCHEMA,
    RecordError,
    admit,
    build_record,
    check_record,
    filter_records,
    read_record,
    write_record,
)
from courant.spec import draft_spec, spec_sha256
from courant.verify import verify_path
from courant.verify.schema import canonical_json
from tests.unit.test_verify_pack import _case, _copy


def _app():
    from courant.cli.main import app

    return app


@pytest.fixture
def demo(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path / "demo")


@pytest.fixture
def locked(demo, tmp_path):
    """The demo case, its specification (outside the case) and the hash it was locked at."""
    spec = tmp_path / "spec.json"
    spec.write_text(draft_spec(demo).to_json())
    return demo, spec, spec_sha256(spec)


def _record(tmp_path, locked, name="rec.json", **kw):
    demo, spec, lock = locked
    verdict = verify_path(demo, spec=spec, spec_locked_sha256=lock)
    return write_record(verdict, tmp_path / name, target_name=demo.name, spec_path=spec,
                        spec_locked_sha256=lock, **kw)


# ---------------------------------------------------------------------------- REC-1: the record


def test_a_record_holds_the_spec_the_verdict_the_provenance_and_the_manifest(tmp_path, locked):
    demo, spec, lock = locked
    path = _record(tmp_path, locked, producer="agent-x/1.2", requester="s@example.org")
    text = path.read_text()
    rec = json.loads(text)
    verdict = verify_path(demo, spec=spec, spec_locked_sha256=lock)
    assert rec["schema"] == SCHEMA == "courant.record/1"
    assert text == canonical_json(rec)                                    # canonical JSON (VER-9)
    assert rec["verdict"] == json.loads(verdict.to_json())
    assert rec["manifest"] == rec["verdict"]["run"]["inputs"] and len(rec["manifest"]) == len(verdict.run.inputs) >= 11
    assert rec["provenance"] == {"producer": "agent-x/1.2", "requester": "s@example.org",
                                 "courant_version": verdict.tool.version}
    assert rec["spec"] == {"text": spec.read_text(), "sha256": lock, "locked_sha256": lock}
    assert rec["verdict"]["basis"] == "validity+conformance"
    assert rec["target"] == {"name": demo.name, "source": "case_dir", "case_json": None}


def test_a_record_names_no_absolute_path(tmp_path, locked):
    text = _record(tmp_path, locked).read_text()
    assert str(tmp_path) not in text and "/private/" not in text and "/Users/" not in text


def test_a_record_is_the_same_bytes_when_written_twice(tmp_path, locked):
    a = _record(tmp_path, locked, "a.json", producer="p").read_bytes()
    b = _record(tmp_path, locked, "b.json", producer="p").read_bytes()
    assert a == b


def test_a_record_of_a_run_with_no_specification(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    v = verify_path(log, cj)
    rec = build_record(v, target_name=log.name, case_json_name=cj.name)
    assert rec["spec"] is None and rec["target"]["source"] == "corpus"
    assert [m["path"] for m in rec["manifest"]] == sorted([log.name, cj.name])


def test_a_version_1_verdict_is_recorded_as_version_2(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    v1 = verify_path(log, cj).to_v1()
    rec = build_record(v1, target_name=log.name, case_json_name=cj.name)
    assert rec["verdict"]["schema"] == "courant.verdict/2"


# ------------------------------------------------------------ re-derive byte for byte


def test_an_untouched_record_re_derives_byte_for_byte(tmp_path, locked):
    path = _record(tmp_path, locked, producer="agent-x/1.2")
    ok, message = check_record(path, locked[0])
    assert ok, message
    n = len(read_record(path)["manifest"])
    assert message == f"identical: {n} inputs re-hashed, record re-derived byte for byte"


def test_a_log_record_re_derives(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__cfl_blowup")
    path = write_record(verify_path(log, cj), tmp_path / "r.json", target_name=log.name,
                        case_json_name=cj.name, requester="ci")
    assert check_record(path, tmp_path) == (
        True, "identical: 2 inputs re-hashed, record re-derived byte for byte")


def test_a_case_directory_record_with_extra_files_re_derives(tmp_path):
    case = _case(tmp_path)
    path = write_record(verify_path(case), tmp_path / "r.json", target_name=case.name)
    assert check_record(path, case)[0]


def test_a_changed_input_is_named(tmp_path, locked):
    path = _record(tmp_path, locked)
    demo = locked[0]
    u = demo / "system" / "fvSchemes"
    u.write_text(u.read_text() + "\n// edit\n")
    ok, message = check_record(path, demo)
    assert not ok and message.startswith("input changed: system/fvSchemes")


def test_a_missing_input_is_named(tmp_path, locked):
    path = _record(tmp_path, locked)
    (locked[0] / "log.simpleFoam").unlink()
    ok, message = check_record(path, locked[0])
    assert not ok and message == "input missing: log.simpleFoam"


def test_a_tampered_verdict_is_caught(tmp_path, locked):
    path = _record(tmp_path, locked)
    rec = json.loads(path.read_text())
    rec["verdict"]["grade"] = "D"
    path.write_text(canonical_json(rec))
    ok, message = check_record(path, locked[0])
    assert not ok and message.startswith("record differs at byte")


def test_a_swapped_specification_is_caught(tmp_path, locked):
    path = _record(tmp_path, locked)
    rec = json.loads(path.read_text())
    rec["spec"]["text"] += "\n"
    path.write_text(canonical_json(rec))
    ok, message = check_record(path, locked[0])
    assert not ok and message.startswith("record differs")


def test_a_record_from_another_courant_version_says_so(tmp_path, locked):
    path = _record(tmp_path, locked)
    rec = json.loads(path.read_text())
    rec["tool"]["version"] = "0.0.1"
    path.write_text(canonical_json(rec))
    ok, message = check_record(path, locked[0])
    assert not ok and "written by Courant 0.0.1" in message


def test_a_file_that_is_not_a_record_is_refused_not_a_traceback(tmp_path, locked):
    junk = tmp_path / "junk.json"
    junk.write_text('{"schema":"courant.verdict/2"}')
    ok, message = check_record(junk, locked[0])
    assert not ok and "not a courant.record/1" in message
    with pytest.raises(RecordError):
        read_record(tmp_path / "missing.json")


def test_the_cli_writes_and_checks_a_record(tmp_path, locked):
    demo, spec, lock = locked
    out = tmp_path / "out" / "rec.json"
    runner = CliRunner()
    r = runner.invoke(_app(), ["verify", str(demo), "--spec", str(spec), "--spec-sha256", lock,
                               "--record", str(out), "--producer", "agent-x/1", "--requester",
                               "ci", "--json"])
    assert r.exit_code == 0, r.output
    rec = json.loads(out.read_text())
    assert rec["provenance"]["producer"] == "agent-x/1" and rec["provenance"]["requester"] == "ci"
    assert json.loads(r.stdout)["verdict"] == "accept"
    ok = runner.invoke(_app(), ["verify", "--check-record", str(out), "--root", str(demo)])
    assert ok.exit_code == 0 and ok.stdout.startswith("OK ")
    (demo / "system" / "fvSchemes").write_text("changed\n")
    bad = runner.invoke(_app(), ["verify", "--check-record", str(out), "--root", str(demo)])
    assert bad.exit_code == 5 and bad.stdout.startswith("MISMATCH input changed")


def test_the_cli_refuses_a_record_option_without_its_run(tmp_path):
    runner = CliRunner()
    assert runner.invoke(_app(), ["verify", "--producer", "x"]).exit_code == 4
    assert runner.invoke(_app(), ["verify", "--batch", str(tmp_path), "--record",
                                  str(tmp_path / "r.json")]).exit_code == 4


# ---------------------------------------------------------------------------- REC-2: the filter


def test_the_filter_admits_a_matching_accept_on_validity_plus_conformance(tmp_path, locked):
    path = _record(tmp_path, locked)
    a = admit(path, locked[0], expected_spec_sha256=locked[2])
    assert a.admitted and a.reason.startswith("accept on validity+conformance; identical:")
    assert admit(path, locked[0], expected_spec_sha256=locked[2], rederive=False).reason.endswith("not re-derived")


def test_the_filter_refuses_an_accept_on_validity_alone(tmp_path):
    log, cj = _copy(tmp_path, "icoFoam_cavity__healthy")
    v = verify_path(log, cj)
    assert v.verdict == "accept" and v.basis == "validity"
    path = write_record(v, tmp_path / "r.json", target_name=log.name, case_json_name=cj.name)
    a = admit(path, tmp_path, expected_spec_sha256="0" * 64)
    assert not a.admitted and a.reason == "the basis is validity, not validity+conformance"


def test_the_filter_refuses_an_unlocked_specification(tmp_path, locked):
    demo, spec, _ = locked
    v = verify_path(demo, spec=spec)
    path = write_record(v, tmp_path / "r.json", target_name=demo.name, spec_path=spec)
    a = admit(path, demo, expected_spec_sha256=locked[2])
    assert not a.admitted and "validity+unlocked-spec" in a.reason


def test_the_filter_refuses_a_reject_and_an_insufficient(tmp_path, locked):
    demo, spec, _ = locked
    reject = _case(tmp_path)
    v = verify_path(reject)
    assert v.verdict == "reject"
    path = write_record(v, tmp_path / "rej.json", target_name=reject.name)
    assert admit(path, reject, expected_spec_sha256="0" * 64).reason == "the verdict is reject, not accept"
    wrong = verify_path(demo, spec=spec, spec_locked_sha256="0" * 64)
    path = write_record(wrong, tmp_path / "ins.json", target_name=demo.name, spec_path=spec,
                        spec_locked_sha256="0" * 64)
    assert admit(path, demo, expected_spec_sha256=locked[2]).reason == "the verdict is insufficient, not accept"


def test_the_filter_refuses_when_a_file_on_disk_no_longer_matches(tmp_path, locked):
    path = _record(tmp_path, locked)
    demo = locked[0]
    (demo / "system" / "fvSolution").write_text("changed\n")
    a = admit(path, demo, expected_spec_sha256=locked[2], rederive=False)
    assert not a.admitted and a.reason.startswith("input changed: system/fvSolution")
    (demo / "system" / "fvSolution").unlink()
    assert admit(path, demo, expected_spec_sha256=locked[2], rederive=False).reason == "input missing: system/fvSolution"


def test_the_filter_does_not_take_a_forged_accept_at_its_word(tmp_path):
    """A reject record edited to say accept on validity+conformance, its manifest and files
    untouched, passes every cheap check. Only the re-derivation refuses it."""
    case = _case(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(draft_spec(case).to_json())
    lock = spec_sha256(spec)
    v = verify_path(case, spec=spec, spec_locked_sha256=lock)
    assert v.verdict == "reject" and v.basis == "validity+conformance"
    path = write_record(v, tmp_path / "r.json", target_name=case.name, spec_path=spec,
                        spec_locked_sha256=lock)
    rec = json.loads(path.read_text())
    rec["verdict"]["verdict"] = "accept"
    path.write_text(canonical_json(rec))
    assert admit(path, case, expected_spec_sha256=lock, rederive=False).admitted             # the cheap checks are fooled
    forged = admit(path, case, expected_spec_sha256=lock)                                     # the default is not
    assert not forged.admitted and forged.reason.startswith("record differs at byte")


def test_the_filter_refuses_a_manifest_that_is_not_the_verdicts(tmp_path, locked):
    path = _record(tmp_path, locked)
    rec = json.loads(path.read_text())
    rec["manifest"] = rec["manifest"][:-1]
    path.write_text(canonical_json(rec))
    assert admit(path, locked[0], expected_spec_sha256=locked[2]).reason == "the record's manifest is not its verdict's inputs"


def test_the_filter_refuses_a_specification_that_is_not_the_one_judged(tmp_path, locked):
    path = _record(tmp_path, locked)
    rec = json.loads(path.read_text())
    rec["spec"]["locked_sha256"] = "1" * 64
    path.write_text(canonical_json(rec))
    assert "not the locked one" in admit(path, locked[0], expected_spec_sha256=locked[2]).reason


def test_filter_records_splits_a_batch_into_admitted_and_refused(tmp_path, locked):
    good = _record(tmp_path, locked, "good.json")
    other = tmp_path / "other"
    other.mkdir()
    reject = _case(other)
    bad = write_record(verify_path(reject), tmp_path / "bad.json", target_name=reject.name)
    junk = tmp_path / "junk.json"
    junk.write_text("not json")
    admitted, refused = filter_records([good, bad, junk], locked[0], expected_spec_sha256=locked[2])
    assert admitted == [good]
    assert [Path(r.path).name for r in refused] == ["bad.json", "junk.json"]



def test_a_self_made_specification_is_not_the_one_the_dataset_owner_expects(tmp_path, locked):
    """`courant spec init` on the agent's own run drafts a specification that run satisfies. A
    record that carries it, locked at its own hash, re-derives and says accept on
    validity+conformance: the problem the agent set itself. Only the owner's hash refuses it."""
    demo, _, lock = locked
    path = _record(tmp_path, locked)
    owner = "7" * 64                                            # the hash of the spec the owner set
    refused = admit(path, demo, expected_spec_sha256=owner)
    assert not refused.admitted
    assert "not one the dataset owner expects" in refused.reason
    assert admit(path, demo, expected_spec_sha256=[owner, lock]).admitted
    assert admit(path, demo, expected_spec_sha256=lock.upper()).admitted
    assert not admit(path, demo, expected_spec_sha256=[]).admitted
    assert not admit(path, demo, expected_spec_sha256="", rederive=False).admitted


def test_admit_cannot_be_called_without_the_owners_hash(tmp_path, locked):
    path = _record(tmp_path, locked)
    with pytest.raises(TypeError):
        admit(path, locked[0])


def test_filter_records_refuses_every_record_for_an_unexpected_specification(tmp_path, locked):
    good = _record(tmp_path, locked, "good.json")
    admitted, refused = filter_records([good], locked[0], expected_spec_sha256="1" * 64)
    assert admitted == [] and len(refused) == 1
