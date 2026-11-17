"""Mutation-070: tests that kill the surviving mutants of `courant/verify/schema.py`.

Each test pins behaviour the module's docstrings state: the verifier's contracts are frozen
models (module docstring), `RunInputs`' spec and case-directory fields are never serialised, a
manifest names why a noted file could not be hashed, an empty `invalid` is dropped but a
non-empty one kept, an outcome carries the verdict and rule it came from, and `num` writes a
number as the shortest string that round-trips.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from courant.verify.schema import (
    ConformanceBlock,
    ConformanceItem,
    ConvergenceBlock,
    InputError,
    InputFile,
    Outcome,
    ProbableCause,
    Reason,
    RunInfo,
    RunInputs,
    Stats,
    ToolInfo,
    Verdict,
    manifest,
    num,
    outcome_of,
)

CORPUS = Path(__file__).resolve().parents[2] / "certification" / "corpus"


def _item(**kw) -> ConformanceItem:
    data = {"name": "nu", "file": "constant/transportProperties", "location": "nu",
            "specified": "1e-05", "found": "1e-05", "conforms": True, "tolerance": "1%"}
    data.update(kw)
    return ConformanceItem(**data)


def _block(**kw) -> ConformanceBlock:
    data = {"spec_schema": "courant.spec/1", "spec_sha256": "s" * 64, "items": (_item(),),
            "checked": 1, "nonconformant": 0, "unreadable": 0}
    data.update(kw)
    return ConformanceBlock(**data)


def _stats() -> Stats:
    return Stats(events=0, residual_steps=0, crashes=0, divergences=0, findings_by_severity={})


def _run() -> RunInfo:
    return RunInfo(id="r", source="log", solver=None, inputs=(InputFile.of_text("x", "log"),))


def _reason() -> Reason:
    return Reason(code="C", kind="event", severity="major", summary="s", input="log", method="m")


def _verdict() -> Verdict:
    return Verdict(tool=ToolInfo(version="0"), run=_run(), verdict="TRUSTWORTHY", rule="VER-5.6",
                   grade=None, reasons=(_reason(),), probable_causes=(), not_assessed=(),
                   next_action=None, stats=_stats())


def _convergence() -> ConvergenceBlock:
    return ConvergenceBlock(
        status="PASS", regime=None, enters_verdict=False, converged_at_step=None,
        binding_monitor=None, change_after_stop=None, change_counts=None, change_fraction=None,
        wasted_steps=None, steps_total=None, wasted_fraction=None, wasted_wall_clock_s=None,
        wasted_core_hours=None, monitors_assessed=0, monitors_found=0)


def _outcome() -> Outcome:
    return outcome_of(None, executed=False, executed_source="resolve")


# Kills cc92091ced schema.py:31, dfe7f5d769 schema.py:82, 300c7eafa9 schema.py:107,
# 0378e95dda schema.py:130, c4430cec60 schema.py:138, 8f8197f1fb schema.py:148,
# 307698dc2f schema.py:157, ae19d82f69 schema.py:177, de7ae6be5b schema.py:200,
# d2fd38c21c schema.py:219, 9b32d99374 schema.py:243 (frozen), 6483763c5f schema.py:298
# (each `frozen=True` -> `frozen=False`).
@pytest.mark.parametrize("build, field, value", [
    (lambda: InputFile.of_text("x", "log"), "sha256", "0" * 64),
    (lambda: RunInputs(run_id="r", log_text="", log_name="log"), "log_text", "End\n"),
    (_reason, "severity", "minor"),
    (lambda: ProbableCause(rank=1, code="C", name="n"), "rank", 2),
    (_stats, "crashes", 1),
    (_run, "solver", "icoFoam"),
    (lambda: ToolInfo(version="0"), "version", "1"),
    (_convergence, "status", "FAIL"),
    (_item, "conforms", False),
    (_block, "nonconformant", 1),
    (_verdict, "verdict", "NOT_TRUSTWORTHY"),
    (_outcome, "passed", True),
], ids=["InputFile", "RunInputs", "Reason", "ProbableCause", "Stats", "RunInfo", "ToolInfo",
        "ConvergenceBlock", "ConformanceItem", "ConformanceBlock", "Verdict", "Outcome"])
def test_every_contract_is_frozen(build, field, value):
    """The module's contract: every edge carries a frozen model, so a verdict, its reasons and
    its manifest cannot be changed after they were decided."""
    obj = build()
    with pytest.raises(ValidationError):
        setattr(obj, field, value)
    assert getattr(obj, field) != value


# Kills 21ccacc79f schema.py:94, cd7c7cfea0 schema.py:97, 1ebc183c78 schema.py:98,
# 5f1f893839 schema.py:99, ab54a896f4 schema.py:101 (`exclude=True` -> `exclude=False`).
def test_run_inputs_never_serialise_the_case_path_the_spec_or_the_checks():
    """`case_dir` is an absolute local path ("Never serialised"); the specification, its hashes
    and the checks switch are inputs to the policy, not part of the inputs' serial form."""
    inputs = RunInputs(run_id="r", log_text="End\n", log_name="log", case_dir="/home/someone/case",
                       spec={"schema": "courant.spec/1", "items": []}, spec_sha256="a" * 64,
                       spec_locked_sha256="b" * 64, checks="next")
    for dumped in (inputs.model_dump(), inputs.model_dump(mode="json")):
        for key in ("case_dir", "spec", "spec_sha256", "spec_locked_sha256", "checks"):
            assert key not in dumped
    assert "/home/someone/case" not in inputs.model_dump_json()
    assert inputs.case_dir == "/home/someone/case" and inputs.checks == "next"


# Kills da4a1a3a11 schema.py:65 (`exc.strerror or type(exc).__name__` -> `and`).
def test_a_manifest_names_why_a_noted_file_could_not_be_hashed(tmp_path):
    """The error names the operating system's reason, not the exception's class name."""
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text("x")
    with pytest.raises(InputError) as err:
        manifest(tmp_path, ["system/controlDict", "system/fvSchemes"])
    msg = str(err.value)
    assert msg.startswith("system/fvSchemes was read for the verdict")
    assert msg.endswith(": No such file or directory")
    assert "FileNotFoundError" not in msg


# Kills 1cfd0af35f schema.py:237 (`not data.get("invalid")` -> `True`).
def test_an_invalid_specification_keeps_its_reasons_in_the_block():
    """SPEC-9 / VER-5.11: the reasons a specification is impossible are written; only an empty
    `invalid` is left out, so every other block keeps its bytes."""
    kept = _block(invalid=("endTime is before startTime",)).model_dump(mode="json")
    assert kept["invalid"] == ["endTime is before startTime"]
    assert "invalid" not in _block().model_dump(mode="json")


# Kills cad21348e9 schema.py:209 (`modelling_choice: bool = False` -> `True`).
def test_a_conformance_item_read_without_the_key_is_not_a_modelling_choice():
    """A stored verdict whose item does not carry `modelling_choice` (read back through
    `load_verdict`, which validates it) is not marked a modelling choice."""
    item = ConformanceItem.model_validate(
        {"name": "nu", "file": "constant/transportProperties", "location": "nu",
         "specified": "1e-05", "found": "1e-05", "conforms": True, "tolerance": "1%"})
    assert item.modelling_choice is False
    assert item.model_dump(mode="json")["modelling_choice"] is False


# Kills fa7bb5c8f4 schema.py:277 (`OUTCOMES[5:]` -> `OUTCOMES[4:]`).
def test_did_not_execute_is_not_a_re_solve_class():
    """The re-solve's own classes are the last five outcomes (SB-18, SB-19); `did_not_execute`
    is a judgement of the run, never a class the re-solve may pass through."""
    with pytest.raises(ValueError, match="is not a re-solve class"):
        outcome_of(None, executed=False, executed_source="resolve",
                   resolve_class="did_not_execute")


# Kills 48fb09c6ad schema.py:337 (`executed_source not in (...)` -> `False`).
def test_an_unknown_executed_source_is_refused_by_name_before_the_verdict_is_read():
    """The refusal is outcome_of's own, naming the argument, and comes before the verdict is
    read: without it a bad source surfaces as whatever reading the verdict raises first."""
    with pytest.raises(ValueError, match="executed_source is 'resolve' or 'agent_log', not 'guess'"):
        outcome_of(None, executed=True, executed_source="guess")
    with pytest.raises(ValueError, match="executed_source is"):
        outcome_of(object(), executed=True, executed_source="guess")


# Kills 26d057e552 schema.py:346 and dadd12ffc1 schema.py:347 (`v2 is not None` -> `False`).
def test_an_outcome_carries_the_verdict_and_rule_it_came_from():
    from courant.verify import verify_path

    v = verify_path(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz")
    got = outcome_of(v, executed=True, executed_source="agent_log", reported_success=True)
    assert got.outcome == "invalid"
    assert (got.verdict, got.rule) == (v.verdict, v.rule) == ("reject", "VER-5.3")


# Kills 5ae0eadd38 schema.py:428 (`hasattr(value, "__index__")` -> `True`).
def test_num_does_not_truncate_a_non_float_fraction():
    """`num` writes the shortest string that round-trips; a number that is neither a Python
    float nor an integer (a Decimal, numpy's float32) keeps its fraction."""
    import numpy as np

    assert num(Decimal("2.5")) == "2.5"
    assert num(np.float32(0.5)) == "0.5"
    assert num(np.int64(3)) == "3" and num(7) == "7" and num(0.1) == "0.1"


# Kills 3be6debc61 schema.py:428 (`hasattr(value, "__index__")` -> `False`).
def test_num_writes_an_integer_like_value_as_a_number():
    """An integer-like value (here a bool) is written as the integer it is, never as a word."""
    assert num(True) == "1" and num(False) == "0"
