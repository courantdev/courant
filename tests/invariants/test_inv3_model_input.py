"""Invariant 3: a model-derived input can only lower a verdict to `insufficient`.

The one model-derived input the gate reads is a specification `courant spec extract` drafted
(SPEC-7: `provenance.extracted_by`, no `confirmed_by`). Its verdict is the run's own or
`insufficient`, never a `reject` the run's own evidence did not give (VER-5.12), and never better
than the same specification confirmed by a person. Its basis says so, in every document that
carries one (the verdict and the record, GATE-2).

Nothing else model-derived reaches a verdict: the pipeline runs with `courant.llm` unimportable.
"""

from __future__ import annotations

import importlib.abc
import json
import sys
from functools import cache

import pytest

from courant.verify import from_case_dir, from_log, verify_inputs, with_spec
from tests.invariants import gen

N_FAST, N_SLOW = 60, 3000


@cache
def _bare(rel: str, checks: str) -> str:
    return verify_inputs(from_case_dir(gen.pristine(rel), checks=checks)).verdict


def _verdict(k: int, checks: str, provenance):
    rel, case = gen.dir_case(k)
    what, data = gen.spec_for(k, rel, checks, provenance=provenance)
    path, sha = gen.write_spec(data, k)
    v = verify_inputs(with_spec(from_case_dir(case, checks=checks), path, gen.lock_for(k, sha)))
    return rel, what, v


def _run(n: int):
    bad = []
    for k in range(n):
        checks = "next" if k % 2 else "default"
        rel, what, ext = _verdict(k, checks, "extracted")
        _, _, conf = _verdict(k, checks, "confirmed")
        before = _bare(rel, checks)
        tag = f"[{k}] {rel} {checks} {what}"
        if ext.verdict not in (before, "insufficient"):
            bad.append(f"{tag}: {before} -> {ext.verdict} ({ext.rule})")
        if gen.RANK[ext.verdict] < gen.RANK[conf.verdict] and ext.verdict == "reject":
            bad.append(f"{tag}: extracted {ext.verdict} below confirmed {conf.verdict}")
        if ext.basis != "validity+extracted-spec":
            bad.append(f"{tag}: basis {ext.basis}")
    assert not bad, f"{len(bad)} of {n}: {bad[:5]}"


def test_an_extracted_specification_only_lowers_to_insufficient():
    _run(N_FAST)


@pytest.mark.slow
def test_an_extracted_specification_only_lowers_to_insufficient_at_scale():
    _run(N_SLOW)


def _nonconformant_spec(confirmed_by):
    """A model's specification of the duct whose first value is ten times the case's."""
    rel = "tests/fixtures/real_openfoam/duct/converged"
    items = [dict(i) for i in gen.drafted(rel, "default")]
    item = next(i for i in items if isinstance(i.get("value"), (int, float)) and i["value"])
    item["value"] = item["value"] * 10
    prov = {"extracted_by": "courant spec extract 0.7.0", "model": "example/model:free",
            "date": "2026-11-04"}
    if confirmed_by is not None:
        prov["confirmed_by"] = confirmed_by
    data = {"schema": "courant.spec/1", "items": items, "provenance": prov}
    path, sha = gen.write_spec(data, 9000 + len(str(confirmed_by)))
    return rel, path, sha


def test_an_unconfirmed_extraction_that_mismatches_is_insufficient_not_reject():
    rel, path, sha = _nonconformant_spec(None)
    v = verify_inputs(with_spec(from_case_dir(gen.pristine(rel)), path, sha))
    assert (v.verdict, v.rule, v.basis) == ("insufficient", "VER-5.12",
                                            "validity+extracted-spec")


@pytest.mark.xfail(strict=True, reason="INV3-F1: a blank confirmed_by binds a model's "
                   "specification (research/notes/invariants-070/FINDINGS.md)")
@pytest.mark.parametrize("blank", [" ", "\t", "\n "])
def test_a_blank_confirmed_by_does_not_confirm_a_model_specification(blank):
    """`courant spec confirm --by` refuses a blank name (`extract.confirm`); a file that says
    `"confirmed_by": " "` was confirmed by nobody, and binds as little as one that says nothing."""
    rel, path, sha = _nonconformant_spec(blank)
    v = verify_inputs(with_spec(from_case_dir(gen.pristine(rel)), path, sha))
    assert v.verdict == "insufficient" and v.basis == "validity+extracted-spec"


def test_a_record_under_schema_1_keeps_the_extracted_basis(tmp_path):
    """GATE-2: the record holds version 2, and its basis is what the policy was told. Under
    `--schema 1` the CLI handed the record a version 1 verdict, which carries no provenance, so
    the record said `validity+conformance` for a model's unconfirmed specification (INV3-F2)."""
    from typer.testing import CliRunner

    from courant.cli.main import app

    rel = "tests/fixtures/real_openfoam/duct/converged"
    items = [dict(i) for i in gen.drafted(rel, "default")][:3]
    data = {"schema": "courant.spec/1", "items": items,
            "provenance": {"extracted_by": "courant spec extract 0.7.0", "model": "m",
                           "date": "2026-11-04"}}
    path, sha = gen.write_spec(data, 9100)
    for schema in ("1", "2"):
        rec = tmp_path / f"record-{schema}.json"
        res = CliRunner().invoke(app, ["verify", str(gen.pristine(rel)), "--spec", str(path),
                                       "--spec-sha256", sha, "--schema", schema,
                                       "--record", str(rec)])
        assert res.exit_code == 0, res.output
        assert json.loads(rec.read_text())["verdict"]["basis"] == "validity+extracted-spec", \
            f"--schema {schema}"


class _NoModel(importlib.abc.MetaPathFinder):
    """Refuses `courant.llm`, and records the attempt: an import inside a broad `try` would
    swallow the `ImportError` and pass unseen."""

    def __init__(self):
        self.attempts: list[str] = []

    def find_spec(self, name, path=None, target=None):
        if name == "courant.llm" or name.startswith("courant.llm."):
            self.attempts.append(name)
            raise ImportError(f"invariant 3: the verdict path imported {name}")


def test_no_verdict_imports_the_model_door(monkeypatch):
    """Over the generated stream, with `courant.llm` unimportable and unloaded."""
    for name in [m for m in sys.modules if m == "courant.llm" or m.startswith("courant.llm.")]:
        monkeypatch.delitem(sys.modules, name)
    finder = _NoModel()
    sys.meta_path.insert(0, finder)
    try:
        for k in range(30):
            _, log, side = gen.log_case(k)
            verify_inputs(from_log(log, side))
        for k in range(10):
            _verdict(k, "next" if k % 2 else "default", type)
    finally:
        sys.meta_path.remove(finder)
    assert finder.attempts == []
