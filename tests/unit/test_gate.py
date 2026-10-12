"""courant.verdict/2 (TRD GATE-1, GATE-2, GATE-3, ARC-16; PRD 2.0 P2-GATE-1, T-VERDICT-2).

Version 2 is version 1's decision in three words, with its basis stated: the mapping is exact over
every policy rule, `to_v1()` gives version 1 back byte for byte, and every surface writes
version 2 by default and version 1 behind `--schema 1`.
"""

from __future__ import annotations

import gzip
import json
import shutil
import typing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.cli.main import app
from courant.verify import (
    from_path,
    render_json,
    render_markdown,
    render_text,
    sentences,
    verify_inputs,
    verify_path,
    with_spec,
)
from courant.verify.pipeline import verify_inputs_v1
from courant.verify.schema import (
    ConformanceBlock,
    RunInfo,
    Stats,
    ToolInfo,
    Verdict,
    VerdictValue,
)
from courant.verify.verdict2 import (
    EXIT,
    FROM_V1,
    SCHEMA_V1,
    SCHEMA_V2,
    TO_V1,
    VerdictV2,
    basis_of,
    extracted_unconfirmed,
    in_schema,
    load_verdict,
    to_v2,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
RUNNER = CliRunner()


def corpus(sid: str) -> tuple[Path, Path]:
    return CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json"


def _cli(*args: str):
    return RUNNER.invoke(app, ["verify", *args])


@pytest.fixture
def demo(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path / "demo")


def _spec(case: Path, out: Path) -> Path:
    from courant.spec import draft_spec

    path = out / "spec.json"
    path.write_text(draft_spec(case).to_json())
    return path


# ------------------------------------------------------------------ GATE-1: the mapping
def test_the_mapping_is_exact_and_one_to_one():
    v1_values = set(typing.get_args(VerdictValue))
    assert set(FROM_V1) == v1_values
    assert FROM_V1 == {"TRUSTWORTHY": "accept", "NOT_TRUSTWORTHY": "reject",
                       "NOT_ASSESSABLE": "insufficient"}
    assert {TO_V1[v2]: v2 for v2 in TO_V1} == FROM_V1
    assert EXIT == {"accept": 0, "reject": 2, "insufficient": 3}
    from courant.cli.commands.verify import EXIT as CLI_EXIT

    assert all(CLI_EXIT[v1] == EXIT[v2] == CLI_EXIT[v2] for v1, v2 in FROM_V1.items())


def _v1(verdict: str, rule: str, conformance=None) -> Verdict:
    return Verdict(tool=ToolInfo(version="0"), run=RunInfo(id="r", source="log", solver=None,
                                                           inputs=()),
                   verdict=verdict, rule=rule, grade=None, reasons=(), probable_causes=(),
                   not_assessed=("a sentence",), next_action=None,
                   stats=Stats(events=0, residual_steps=0, crashes=0, divergences=0,
                               findings_by_severity={}),
                   conformance=conformance)


#: Every rule of VER-5 and the verdict each gives (TRD VER-5, VER-5.7..5.9).
RULES = [("NOT_ASSESSABLE", "VER-5.1"), ("NOT_TRUSTWORTHY", "VER-5.2"),
         ("NOT_TRUSTWORTHY", "VER-5.3"), ("NOT_TRUSTWORTHY", "VER-5.4"),
         ("NOT_ASSESSABLE", "VER-5.5"), ("TRUSTWORTHY", "VER-5.6"),
         ("NOT_TRUSTWORTHY", "VER-5.7"), ("NOT_ASSESSABLE", "VER-5.8"),
         ("NOT_ASSESSABLE", "VER-5.9"), ("NOT_ASSESSABLE", "VER-5.11"),
         ("NOT_ASSESSABLE", "VER-5.12"), ("NOT_ASSESSABLE", "VER-5.15")]


@pytest.mark.parametrize("verdict, rule", RULES)
def test_every_policy_rule_maps_exactly(verdict, rule):
    """GATE-1: the version 2 answer is FROM_V1 of version 1's, the rule is kept, the exit code is
    the one version 1 had, and `to_v1()` is the original to the byte."""
    from courant.cli.commands.verify import EXIT as CLI_EXIT

    v1 = _v1(verdict, rule)
    v2 = to_v2(v1)
    assert (v2.verdict, v2.rule, v2.schema_) == (FROM_V1[verdict], rule, SCHEMA_V2)
    assert v2.exit_code == CLI_EXIT[verdict]
    assert v2.to_v1().to_json() == v1.to_json()


def test_the_policy_produces_only_mapped_rules():
    """The rules above are every rule the policy can return (a guard on RULES)."""
    import inspect

    from courant.verify import policy

    src = inspect.getsource(policy.decide) + inspect.getsource(policy.apply_conformance)
    import re

    assert set(re.findall(r'"(VER-5\.\d+)"', src)) == {r for _, r in RULES}


def test_every_real_run_maps_exactly_and_round_trips():
    """Over the 31 corpus logs: version 2 is the same decision, and `to_v1()` is byte for byte
    what the version 1 pipeline writes."""
    logs = sorted(CORPUS.glob("*.log.gz"))
    assert len(logs) == 31
    seen = set()
    for log in logs:
        side = log.with_name(log.name.removesuffix(".log.gz") + ".case.json")
        inputs = from_path(log, side if side.exists() else None)
        v1, v2 = verify_inputs_v1(inputs), verify_inputs(inputs)
        assert v2.verdict == FROM_V1[v1.verdict] and (v2.rule, v2.grade) == (v1.rule, v1.grade)
        assert v2.reasons == v1.reasons and v2.stats == v1.stats
        assert v2.to_v1().to_json() == v1.to_json(), log.name
        seen.add(v2.verdict)
    assert seen == {"accept", "reject"}


# ------------------------------------------------------------------ GATE-2: the basis
def test_without_a_specification_the_basis_is_validity_and_says_so():
    v = verify_path(*corpus("icoFoam_cavity__healthy"))
    assert (v.verdict, v.basis) == ("accept", "validity")
    assert v.not_assessed[-1] == sentences.NO_SPEC
    assert sentences.NO_SPEC == "No specification given: conformance to the task was not assessed."
    assert sentences.NO_SPEC not in v.to_v1().not_assessed


def test_the_basis_with_a_specification(demo, tmp_path):
    """Unlocked: `validity+unlocked-spec`. Locked at its own hash: `validity+conformance`. Not the
    one locked: `validity+unlocked-spec`, and VER-5.8 makes it `insufficient`."""
    from courant.spec import spec_sha256

    spec = _spec(demo, tmp_path)
    unlocked = verify_path(demo, spec=spec)
    locked = verify_path(demo, spec=spec, spec_locked_sha256=spec_sha256(spec))
    wrong = verify_path(demo, spec=spec, spec_locked_sha256="0" * 64)
    assert unlocked.basis == "validity+unlocked-spec"
    assert locked.basis == "validity+conformance"
    assert (wrong.basis, wrong.verdict, wrong.rule) == ("validity+unlocked-spec", "insufficient",
                                                        "VER-5.8")
    for v in (unlocked, locked, wrong):
        assert sentences.NO_SPEC not in v.not_assessed
        assert v.to_v1().not_assessed == v.not_assessed


def test_an_extracted_unconfirmed_specification_is_its_own_basis():
    """SPEC-7: an extracted specification nobody confirmed. Extraction does not write
    `provenance` yet, so this is the rule, ready for it."""
    block = ConformanceBlock(spec_schema="courant.spec/1", spec_sha256="a" * 64,
                             locked_sha256="a" * 64, items=(), checked=0, nonconformant=0,
                             unreadable=0)
    assert basis_of(block) == "validity+conformance"
    assert basis_of(block, extracted=True) == "validity+extracted-spec"
    assert basis_of(None, extracted=True) == "validity"
    assert extracted_unconfirmed({"provenance": {"extracted_by": "courant spec extract"}})
    assert not extracted_unconfirmed({"provenance": {"extracted_by": "x", "confirmed_by": "S"}})
    assert not extracted_unconfirmed(None) and not extracted_unconfirmed({})
    v2 = to_v2(_v1("TRUSTWORTHY", "VER-5.6", block), extracted=True)
    assert v2.basis == "validity+extracted-spec"


def test_the_run_states_its_fork_and_version_fields():
    """§32: `run.fork` and `run.solver_version` are always present (null until ADP-1)."""
    doc = json.loads(verify_path(*corpus("icoFoam_cavity__healthy")).to_json())
    assert {"fork", "solver_version"} <= set(doc["run"]) and doc["schema"] == SCHEMA_V2
    assert set(doc) == {"schema", "tool", "run", "verdict", "basis", "rule", "grade", "reasons",
                        "probable_causes", "not_assessed", "next_action", "stats"}


# ------------------------------------------------------------------ ARC-16: both majors
def test_load_verdict_reads_both_majors():
    v2 = verify_path(*corpus("icoFoam_cavity__cfl_blowup"))
    v1_json = v2.to_v1().to_json()
    assert load_verdict(v2.to_json()) == v2
    back = load_verdict(v1_json)
    assert isinstance(back, VerdictV2) and back.to_v1().to_json() == v1_json
    assert back.to_json() == v2.to_json()
    with pytest.raises(ValueError):
        load_verdict('{"schema": "courant.preflight/1"}')
    with pytest.raises(ValueError):
        in_schema(v2, 3)


@pytest.mark.parametrize("fmt", [[], ["--json"], ["--report", "md"], ["--report", "html"]],
                         ids=["text", "json", "md", "html"])
def test_schema_1_is_version_1_byte_for_byte(fmt, demo):
    """`--schema 1` writes exactly what version 1's renderers write from the version 1 verdict;
    the default writes the same renderers' version 2. Same exit code either way. On a log with
    its sidecar and on a case directory."""
    from courant.verify.html import render_html

    for target in (list(corpus("icoFoam_cavity__cfl_blowup")), [demo]):
        args = [str(target[0])] + (["--case-json", str(target[1])] if len(target) > 1 else [])
        old, new = _cli(*args, *fmt, "--schema", "1"), _cli(*args, *fmt)
        assert old.exit_code == new.exit_code and old.exit_code in (0, 2, 3), old.output
        inputs = from_path(*target)
        v1, v2 = verify_inputs_v1(inputs), verify_inputs(inputs)
        if fmt == ["--report", "html"]:
            want_old, want_new = render_html(v1, inputs), render_html(v2, inputs)
        else:
            render = {"--json": render_json, "--report": render_markdown}.get(
                fmt[0] if fmt else "", render_text)
            want_old, want_new = render(v1), render(v2)
        assert old.output == want_old
        assert new.output == want_new and new.output != old.output
    assert _cli(str(demo), "--schema", "3").exit_code == 4


# ------------------------------------------------------------------ GATE-3: every surface
def test_every_surface_writes_version_2(tmp_path, demo):
    log, cj = corpus("icoFoam_cavity__cfl_blowup")
    js = json.loads(_cli(str(log), "--case-json", str(cj), "--json").output)
    assert (js["schema"], js["verdict"], js["basis"]) == (SCHEMA_V2, "reject", "validity")

    text = _cli(str(log), "--case-json", str(cj)).output
    assert text.startswith("VERDICT   reject   (rule VER-5.3)\nBASIS     validity\n")
    md = _cli(str(log), "--case-json", str(cj), "--report", "md").output
    assert "**Verdict:** `reject`" in md and "**Basis:** `validity`" in md
    assert f"schema `{SCHEMA_V2}`" in md

    # MCP
    from courant.mcp import tools

    got = tools.verify_run(str(log), str(cj), confine=tools.Confinement([CORPUS]))
    assert got == js

    # batch and JUnit
    runs = tmp_path / "runs"
    runs.mkdir()
    for sid in ("icoFoam_cavity__healthy", "icoFoam_cavity__cfl_blowup"):
        for p in corpus(sid):
            shutil.copy(p, runs / p.name)
    b2 = _cli("--batch", str(runs), "--junit", str(tmp_path / "j2.xml"))
    b1 = _cli("--batch", str(runs), "--junit", str(tmp_path / "j1.xml"), "--schema", "1")
    assert b2.exit_code == b1.exit_code == 2
    assert "2 runs: 1 of 2 accept, 1 of 2 reject, 0 of 2 insufficient" in b2.output
    assert "2 runs: 1 of 2 trustworthy, 1 of 2 not trustworthy, 0 of 2 not assessable" in b1.output
    assert 'failure type="reject"' in (tmp_path / "j2.xml").read_text()
    assert 'failure type="NOT_TRUSTWORTHY"' in (tmp_path / "j1.xml").read_text()

    # packs: each major re-checks in the schema it was written in
    for schema in ("2", "1"):
        out = tmp_path / f"pack{schema}"
        r = _cli(str(log), "--case-json", str(cj), "--pack", str(out), "--schema", schema)
        assert r.exit_code == 2
        stored = json.loads((out / "verdict.json").read_text())
        assert stored["schema"] == (SCHEMA_V2 if schema == "2" else SCHEMA_V1)
        ok = _cli("--check-pack", str(out), "--root", str(CORPUS))
        assert ok.exit_code == 0 and ok.output.startswith("OK"), ok.output

    # the live watcher's exit code
    from courant.live.watch import EXIT as LIVE_EXIT

    assert all(LIVE_EXIT[v1] == LIVE_EXIT[v2] == EXIT[v2] for v1, v2 in FROM_V1.items())


def test_the_courant_test_keeps_its_version_1_labels():
    """GATE-3: the judge maps version 2 back, so the locked scorer reads the labels it always
    read and nothing needs relocking (DEV-9)."""
    from courant.ctest.judges.courant import CourantJudge
    from courant.ctest.schema import Task

    log, cj = corpus("icoFoam_cavity__cfl_blowup")
    meta = json.loads(cj.read_text())
    with gzip.open(log, "rt") as fh:
        text = fh.read()
    task = Task(task_id="t", family="icoFoam_cavity", solver="icoFoam", log_text=text,
                log_name=log.name, config_view=meta.get("config"), mesh_view=meta.get("mesh"))
    assert CourantJudge().judge(task).verdict == "NOT_TRUSTWORTHY"


def test_with_spec_keeps_the_v1_api_reachable(demo, tmp_path):
    """`with_spec` + `verify_inputs` is the API the docs name; it answers in version 2 and
    `to_v1()` reaches version 1."""
    v = verify_inputs(with_spec(from_path(demo), _spec(demo, tmp_path)))
    assert isinstance(v, VerdictV2) and v.to_v1().schema_ == SCHEMA_V1
