"""The closed insufficient vocabulary (TRD GATE-4, VER-15; T-GATE-UX).

Every `insufficient` verdict carries exactly one `insufficient.<name>` reason from one closed
table, with one fixed sentence and one VER-5.x rule; no other code reaches one. A named location
the reader parsed completely and did not find is a nonconformance, not a gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courant.verify import sentences, verify_path
from courant.verify.schema import Verdict
from courant.verify.verdict2 import insufficient_reason, load_verdict, to_v2

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
FIXTURES = ROOT / "tests" / "fixtures" / "real_openfoam"

NAMES = ("no_output", "unrecognised_log", "no_residuals", "no_time_markers", "truncated_log",
         "spec_hash_mismatch", "spec_value_unreadable", "unsupported_version", "spec_invalid",
         "spec_extracted_mismatch", "spec_role_unresolved", "time_scheme_unreadable")


@pytest.fixture
def demo(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path / "demo")


def _spec_file(tmp_path: Path, *items: dict) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema": "courant.spec/1", "items": list(items)}))
    return path


# ---------------------------------------------------------------- the table
def test_the_vocabulary_is_closed_and_is_exactly_gate_4s_list():
    assert tuple(sentences.INSUFFICIENT) == NAMES
    assert {e.code for e in sentences.INSUFFICIENT.values()} == {f"insufficient.{n}" for n in NAMES}


def test_each_code_has_one_sentence_and_the_rules_run_5_10_to_5_14_in_order():
    table = sentences.INSUFFICIENT
    assert len({e.sentence for e in table.values()}) == len(NAMES)
    assert all(e.sentence.endswith(".") and "\n" not in e.sentence for e in table.values())
    later = [table[n].rule for n in ("unsupported_version", "spec_invalid",
                                     "spec_extracted_mismatch", "spec_role_unresolved",
                                     "time_scheme_unreadable")]
    assert later == ["VER-5.10", "VER-5.11", "VER-5.12", "VER-5.13", "VER-5.14"]
    assert [table[n].rule for n in ("no_output", "no_residuals", "no_time_markers",
                                    "truncated_log", "spec_hash_mismatch",
                                    "spec_value_unreadable")] == [
        "VER-5.1", "VER-5.5", "VER-5.15", "VER-5.16", "VER-5.8", "VER-5.9"]


def test_a_rule_has_one_code_and_every_code_is_reached_through_its_rule():
    for rule, entry in sentences.BY_RULE.items():
        assert entry.rule == rule
    assert set(sentences.BY_RULE) | {"VER-5.1"} == {e.rule for e in sentences.INSUFFICIENT.values()}


def test_existing_sentences_are_reused_where_one_exists():
    t = sentences.INSUFFICIENT
    assert t["no_output"].sentence == sentences.NO_OUTPUT
    assert t["no_residuals"].sentence == sentences.NO_RESIDUALS
    assert t["spec_hash_mismatch"].sentence == sentences.SPEC_VOID


def test_cause_tags_follow_gate_4():
    fork = {n for n, e in sentences.INSUFFICIENT.items() if e.cause == "fork"}
    assert fork == {"unrecognised_log", "unsupported_version", "spec_role_unresolved",
                    "time_scheme_unreadable"}
    assert {e.cause for e in sentences.INSUFFICIENT.values()} <= set(sentences.CAUSES)
    assert set(sentences.NOT_CHECKED_CAUSE.values()) <= set(sentences.CAUSES)
    # every fixed sentence in the module has a cause tag
    fixed = {v for k, v in vars(sentences).items()
             if k.isupper() and isinstance(v, str) and k not in ("CAUSES",)}
    assert fixed <= set(sentences.NOT_CHECKED_CAUSE)
    for sentence, cause in sentences.NOT_CHECKED_CAUSE.items():
        assert sentences.cause_of(sentence) == cause
    assert sentences.cause_of("simpleFoam: a check's own detail") is None


def test_adp_1s_not_checked_lines_are_built_here_and_carry_the_fork_cause():
    """GATE-4 (RRD RM-22 counts not-checked lines by cause): every line ADP-1 can add to a
    verdict, a version outside the measured set, one not read, one read from a file the run
    wrote, one only `system/controlDict`'s header names, is filled from a template in this module
    and counts as `fork`. The templates name no fork and no version (ARC-14's lint reads them)."""
    from courant import forks

    banner, header = forks.LOG_BANNER, "the header of 100/U"
    given = forks.SolverVersion("foundation", "12", "the header of system/controlDict")
    seen = [
        forks.SolverVersion("esi", "v2512", banner), forks.SolverVersion("foundation", "12", banner),
        forks.SolverVersion("esi", "v2512", header, headers_searched=True),
        forks.SolverVersion("esi", "v2406", header, headers_searched=True),
        forks.SolverVersion("esi", None, banner), forks.SolverVersion(None, "v2406", banner),
        forks.SolverVersion(None, "v2412", banner), forks.SolverVersion("foam-extend", "4.1", banner),
        forks.SolverVersion(), forks.SolverVersion(headers_searched=True),
        forks.SolverVersion(headers_searched=True, given=given),
    ]
    lines = [forks.not_checked(s) for s in seen]
    assert all(lines) and len(set(lines)) == len(lines)
    assert [sentences.cause_of(line) for line in lines] == ["fork"] * len(lines)
    assert forks.not_checked(forks.SolverVersion("esi", "v2406", banner)) is None


def test_a_rule_outside_the_table_raises_rather_than_minting_a_code():
    for rule in ("VER-5.2", "VER-5.3", "VER-5.4", "VER-5.6", "VER-5.7", "VER-9.9"):
        with pytest.raises(ValueError, match="closed list"):
            sentences.insufficient_for(rule)


def test_every_code_builds_a_valid_reason():
    for entry in sentences.INSUFFICIENT.values():
        if entry.name == "unrecognised_log":
            continue                                     # reached through ADP-3, not a rule of its own
        r = insufficient_reason(entry.rule)
        assert (r.kind, r.code, r.summary) == ("insufficient", entry.code, entry.sentence)
        assert r.input and r.method


# ---------------------------------------------------------------- on real verdicts
def _corpus_logs():
    return sorted(CORPUS.glob("*.log.gz"))


def test_over_the_corpus_only_insufficient_verdicts_carry_one_closed_code():
    for log in _corpus_logs():
        cj = log.with_name(log.name.replace(".log.gz", ".case.json"))
        v = verify_path(log, cj if cj.exists() else None)
        codes = [r.code for r in v.reasons if r.code.startswith("insufficient.")]
        kinds = [r for r in v.reasons if r.kind == "insufficient"]
        if v.verdict == "insufficient":
            assert len(codes) == 1 and len(kinds) == 1, log.name
            assert codes[0] in {e.code for e in sentences.INSUFFICIENT.values()}
        else:
            assert codes == [] and kinds == [], log.name


def test_a_log_with_no_solver_output_is_no_output(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("this is not a solver log\n")
    v = verify_path(log)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.1")
    [r] = [r for r in v.reasons if r.kind == "insufficient"]
    assert r.code == "insufficient.no_output" and r.summary == sentences.NO_OUTPUT


def test_a_checkmesh_log_has_no_residuals():
    v = verify_path(FIXTURES / "cavity_checkmesh.log")
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.5")
    [r] = [r for r in v.reasons if r.kind == "insufficient"]
    assert r.code == "insufficient.no_residuals" and r.summary == sentences.NO_RESIDUALS


def test_a_changed_specification_is_spec_hash_mismatch_and_cites_both_hashes(demo, tmp_path):
    from courant.spec import draft_spec

    spec = tmp_path / "s.json"
    spec.write_text(draft_spec(demo).to_json())
    v = verify_path(demo, spec=spec, spec_locked_sha256="0" * 64)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.8")
    codes = [r.code for r in v.reasons]
    assert codes.count("insufficient.spec_hash_mismatch") == 1
    assert codes[0] == "conformance.spec_changed"        # VER-6: conformance reasons first
    r = next(r for r in v.reasons if r.kind == "insufficient")
    assert (r.threshold, r.value) == ("0" * 64, v.conformance.spec_sha256)


def test_a_named_value_in_a_missing_file_is_spec_value_unreadable(demo, tmp_path):
    spec = _spec_file(tmp_path, {"name": "inlet velocity", "file": "0/U", "patch": "inlet",
                                 "value": [10, 0, 0]})
    v = verify_path(demo, spec=spec)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.9")
    r = next(r for r in v.reasons if r.kind == "insufficient")
    assert r.code == "insufficient.spec_value_unreadable" and r.value == "inlet velocity"


def test_the_reason_is_in_every_renderer(demo, tmp_path):
    from courant.verify import render_json, render_markdown, render_text
    from courant.verify.html import render_html

    log = tmp_path / "log.txt"
    log.write_text("nothing here\n")
    v = verify_path(log)
    for out in (render_text(v), render_markdown(v), render_json(v)):
        assert sentences.NO_OUTPUT in out
    assert "insufficient.no_output" in render_json(v)
    from courant.verify import from_path

    assert "insufficient.no_output" in render_html(v, from_path(log))


def test_version_1_never_carries_an_insufficient_reason_and_the_round_trip_is_exact(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("nothing here\n")
    v2 = verify_path(log)
    v1 = v2.to_v1()
    assert isinstance(v1, Verdict) and all(r.kind != "insufficient" for r in v1.reasons)
    assert to_v2(v1).to_json() == v2.to_json()
    assert load_verdict(v2.to_json()).to_json() == v2.to_json()
    assert load_verdict(v1.to_json()).to_json() == v2.to_json()


# ---------------------------------------------------------------- VER-15: a missing location (see also test_deck_missing.py)
def test_a_key_absent_from_a_file_is_a_gap_not_a_reject(demo, tmp_path):
    """A key can have a synonym or an OpenFOAM default, so its absence is not a nonconformance
    (review of T-GATE-UX); only a patch certain not to exist is (tests/unit/test_deck_missing.py)."""
    spec = _spec_file(tmp_path, {"name": "diffusivity", "file": "constant/transportProperties",
                                 "key": "noSuchKeyAnywhere", "value": 5})
    v = verify_path(demo, spec=spec)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.9")
    assert v.conformance.nonconformant == 0 and v.conformance.unreadable == 1


def test_a_location_a_directive_could_hide_stays_a_gap_not_a_nonconformance(tmp_path):
    from courant.spec import check_conformance

    case = tmp_path / "case"
    (case / "system").mkdir(parents=True)
    (case / "system" / "controlDict").write_text('#include "other"\napplication x;\n')
    spec = {"schema": "courant.spec/1", "items": [
        {"name": "dt", "file": "system/controlDict", "key": "deltaT", "value": 1}]}
    block = check_conformance(spec, case, spec_sha256="h")
    assert block.items[0].conforms is None and block.unreadable == 1


def test_a_missing_file_is_still_a_gap(demo, tmp_path):
    spec = _spec_file(tmp_path, {"name": "p", "file": "0/nothere", "key": "internalField",
                                 "value": 0})
    v = verify_path(demo, spec=spec)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.9")


# ---------------------------------------------------------------- GATE-10: no traceback
STAGES = [("courant.verify.policy.decide", "the policy"),
          ("courant.verify.pipeline._reasons", "the reasons"),
          ("courant.verify.pipeline._stats", "the statistics"),
          ("courant.verify.verdict2.to_v2", "the version 2 mapping"),
          ("courant.verify.render.render_text", "the renderer")]


@pytest.mark.parametrize("target", [t for t, _ in STAGES], ids=[n for _, n in STAGES])
def test_an_exception_at_any_stage_is_exit_4_and_one_line(target, monkeypatch):
    from typer.testing import CliRunner

    from courant import __version__
    from courant.cli.main import app

    def boom(*a, **k):
        raise RuntimeError("injected")

    module, name = target.rsplit(".", 1)
    monkeypatch.setattr(f"{module}.{name}", boom)
    monkeypatch.delenv("COURANT_DEBUG", raising=False)
    # `render_text` is imported into the command's namespace at call time from courant.verify
    monkeypatch.setattr("courant.verify.render_text", boom, raising=False)
    result = CliRunner().invoke(app, ["verify", str(CORPUS / "icoFoam_cavity__healthy.log.gz")])
    assert result.exit_code == 4, result.output
    out = result.output
    assert "Traceback" not in out and "injected" not in out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("courant verify:") and __version__ in lines[0] and "--debug" in lines[0]


def test_debug_prints_the_traceback(monkeypatch):
    from typer.testing import CliRunner

    from courant.cli.main import app

    def boom(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr("courant.verify.policy.decide", boom)
    result = CliRunner().invoke(app, ["verify", str(CORPUS / "icoFoam_cavity__healthy.log.gz"),
                                      "--debug"])
    assert result.exit_code == 4
    assert "Traceback" in result.output and "RuntimeError: injected" in result.output


def test_other_commands_are_guarded_too(monkeypatch, capsys):
    import sys

    from courant.cli.main import app

    def boom(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr("courant.converge.from_run_inputs", boom)
    monkeypatch.setattr(sys, "argv", ["courant", "convergence", str(CORPUS)])
    monkeypatch.delenv("COURANT_DEBUG", raising=False)
    with pytest.raises(SystemExit) as stop:
        app(["convergence", str(CORPUS / "icoFoam_cavity__healthy.log.gz")])
    assert stop.value.code == 4
    err = capsys.readouterr().err
    assert "Traceback" not in err and "courant convergence: internal error" in err


def test_a_usage_error_a_command_raises_passes_through_the_guard():
    """typer 0.27 vendors click, so `typer.BadParameter` is not `click.ClickException`; the guard
    named the latter and turned a usage error (exit 2) into an internal error (exit 4)."""
    import typer

    from courant.cli.guard import guarded

    @guarded("x")
    def cmd():
        raise typer.BadParameter("not a number")

    with pytest.raises(typer.BadParameter):
        cmd()


def test_the_cli_starts_without_click_installed():
    """`click` is not a declared dependency (typer 0.27 vendors its own), so a plain
    `pip install` has none; the CLI must not import it. GitHub's gate-action job found this."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = ("import sys; sys.modules['click'] = None; sys.path.insert(0, sys.argv[1]); "
            "from courant.cli.main import app; app(['version'])")
    r = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 0, r.stderr[-800:]


def test_a_user_error_is_not_an_internal_error():
    from typer.testing import CliRunner

    from courant.cli.main import app

    result = CliRunner().invoke(app, ["verify", "/no/such/place.log"])
    assert result.exit_code == 4 and "internal error" not in result.output


# ---------------------------------------------------------------- every rule the policy can return
def test_every_rule_the_policy_can_make_insufficient_has_a_code():
    """Enumerated from the policy's own source, so a new NOT_ASSESSABLE rule cannot ship without
    a code in the closed list (it raised in to_v2 and ended in exit 4 once, for VER-5.15)."""
    import re

    from courant.verify import policy

    src = Path(policy.__file__).read_text()
    rules = set(re.findall(r'"NOT_ASSESSABLE",\s*"(VER-5\.\d+)"', src))
    assert {"VER-5.1", "VER-5.5", "VER-5.15", "VER-5.16", "VER-5.8", "VER-5.9"} <= rules
    assert rules <= set(sentences.BY_RULE), rules - set(sentences.BY_RULE)


def test_logs_with_no_time_markers_are_insufficient_exit_3_with_their_code(tmp_path):
    import re

    from typer.testing import CliRunner

    from courant.cli.main import app

    text = (FIXTURES / "cavity_converged.log").read_text()
    stripped = "".join(ln for ln in text.splitlines(keepends=True) if not re.match(r"Time = ", ln))
    log = tmp_path / "log.icoFoam"
    log.write_text(stripped)
    result = CliRunner().invoke(app, ["verify", str(log)])
    assert result.exit_code == 3, result.output
    assert "internal error" not in result.output
    v = verify_path(log)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.15")
    [r] = [r for r in v.reasons if r.kind == "insufficient"]
    assert r.code == "insufficient.no_time_markers" and r.summary == sentences.NO_TIME_MARKERS
