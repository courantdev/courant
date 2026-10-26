"""TRD VER-5.16: a log that stops before End is not a crash.

Courant 0.5.0 inferred exit code 1 for every log that does not end in `End`. The parser turned it
into a CRASH ("return code 1"), and VER-5.2 rejected the run as crashed, citing a return code no
process had returned. A crash is read from the log (a fatal signature the parser recognises) or
from an exit code the caller gives. A log that shows neither is `insufficient`, with the
closed-list code `insufficient.truncated_log`, no grade, and a convergence assessment that reports
no waste figure for a run whose end is unknown.

Fixtures are this repository's own committed runs (the certification corpus), cut in a temporary
directory, and synthetic lines.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from courant.events import EventKind
from courant.verify import (
    from_log, render_markdown, render_text, sentences, verify_inputs, verify_path,
)
from courant.watcher.exitcode import ends_in_end, infer_exit_code, run_exit_code
from courant.watcher.parser import LogParser

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
HEALTHY = sorted(p.name.removesuffix(".log.gz") for p in CORPUS.glob("*__healthy.log.gz"))
INFERRED = "exit code unknown, inferred: the log does not end in End"


def _text(sid: str) -> str:
    with gzip.open(CORPUS / f"{sid}.log.gz", "rt") as fh:
        return fh.read()


def _cut(text: str, frac: float) -> str:
    """The log as a reader finds it mid-run, or as a window someone copied: whole lines up to
    `frac` of its bytes, no End."""
    head = text[: int(len(text) * frac)]
    head = head[: head.rfind("\n") + 1]
    assert not ends_in_end(head)
    return head


def _write(tmp_path: Path, sid: str, text: str) -> tuple[Path, Path]:
    log = tmp_path / f"{sid}.log"
    log.write_text(text)
    return log, CORPUS / f"{sid}.case.json"


def _insufficient(v):
    return [r for r in v.reasons if r.kind == "insufficient"]


# ---------------------------------------------------------------- the exit code (VER-3)
def test_an_inferred_exit_code_is_0_or_unknown_never_a_nonzero_value():
    assert infer_exit_code("Time = 1\nEnd\n") == 0
    assert infer_exit_code("Time = 1\nEnd") == 0
    assert infer_exit_code("Time = 1\n") is None
    assert infer_exit_code("") is None
    # The crash is the parser's to read from the signature, not a number made up for it.
    assert infer_exit_code("--> FOAM FATAL ERROR:\n") is None


def test_an_exit_code_the_caller_gives_is_used_as_read():
    assert run_exit_code("Time = 1\n", 137) == 137
    assert run_exit_code("Time = 1\nEnd\n", 2) == 2
    assert run_exit_code("Time = 1\n", 0) == 0
    assert run_exit_code("Time = 1\n", None) is None
    assert run_exit_code("Time = 1\nEnd\n", None) == 0


# ---------------------------------------------------------------- the committed corpus
def test_every_corpus_log_without_end_carries_a_fatal_signature_and_still_rejects():
    """The 31-run corpus does not move: each of its logs that stops before End has a signature
    the parser reads (FOAM FATAL ERROR, a signal handler, the shell's signal message), so VER-5.2
    still rejects it, citing the signature."""
    cut = 0
    for log in sorted(CORPUS.glob("*.log.gz")):
        sid = log.name.removesuffix(".log.gz")
        text = _text(sid)
        if ends_in_end(text):
            continue
        cut += 1
        events = LogParser(sid).parse_text(text, return_code=None)
        tags = {e.payload.tag for e in events if e.kind is EventKind.CRASH}
        assert tags and tags <= {"foam_fatal", "fatal_signal"}, sid
        v = verify_path(log, CORPUS / f"{sid}.case.json")
        assert (v.verdict, v.rule) == ("reject", "VER-5.2"), sid
        [crash] = [r for r in v.reasons if r.code == "run.crash"]
        assert crash.value in ("SIGFPE", "SIGSEGV", "FATAL", "FOAM FATAL ERROR"), (sid, crash.value)
        assert "return code" not in v.to_json()
    assert cut == 11


# ---------------------------------------------------------------- a cut healthy log
@pytest.mark.parametrize("frac", [0.3, 0.6, 0.95])
@pytest.mark.parametrize("sid", HEALTHY)
def test_a_healthy_log_cut_mid_run_is_insufficient_not_a_crash(tmp_path, sid, frac):
    log, sidecar = _write(tmp_path, sid, _cut(_text(sid), frac))
    v = verify_path(log, sidecar)
    assert (v.verdict, v.rule, v.grade) == ("insufficient", "VER-5.16", None)
    [r] = _insufficient(v)
    assert (r.code, r.summary, r.value) == ("insufficient.truncated_log", sentences.TRUNCATED_LOG,
                                            INFERRED)
    assert v.not_assessed[0] == sentences.TRUNCATED_LOG
    assert v.stats.crashes == 0
    assert not {r.code for r in v.reasons} & {"run.crash", "completion.crash", "crash.fatal"}
    for shown in (v.to_json(), render_text(v), render_markdown(v)):
        assert "return code" not in shown


def test_the_healthy_corpus_is_seven_runs():
    assert len(HEALTHY) == 7, HEALTHY


def test_a_cut_log_is_exit_3_from_the_cli_and_names_its_code(tmp_path):
    from typer.testing import CliRunner

    from courant.cli.main import app

    log, sidecar = _write(tmp_path, HEALTHY[0], _cut(_text(HEALTHY[0]), 0.5))
    result = CliRunner().invoke(app, ["verify", str(log), "--case-json", str(sidecar)])
    assert result.exit_code == 3, result.output
    assert "insufficient.truncated_log" in result.output
    assert "inferred: the log does not end in End" in result.output
    assert "return code" not in result.output


def test_a_run_that_has_only_printed_its_banner_has_no_output_not_a_crash(tmp_path):
    """A solver just started: VER-5.1 comes first (no solver output; the exit code is unknown
    and the log has no fatal signature), where 0.5.0's inferred crash made it a reject."""
    head = _text("simpleFoam_pitzDaily__healthy")
    banner = head[: head.index("Create time")]
    assert "Exec" in banner and "Time =" not in banner
    log, sidecar = _write(tmp_path, "simpleFoam_pitzDaily__healthy", banner)
    v = verify_path(log, sidecar)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.1")
    assert v.stats.crashes == 0


def test_the_next_action_says_to_verify_the_finished_run(tmp_path):
    log, sidecar = _write(tmp_path, HEALTHY[0], _cut(_text(HEALTHY[0]), 0.5))
    v = verify_path(log, sidecar)
    assert "End" in v.next_action and "finished" in v.next_action


def test_a_run_stopped_from_outside_is_not_a_solver_failure(tmp_path):
    """The shell's 'Killed' or 'Terminated' says something stopped the run, not that the solver
    failed: a converged run killed at its wall-clock limit is not wrong. Without a fatal error
    from the solver or an exit code given, the run's end is unknown."""
    sid = "simpleFoam_pitzDaily__healthy"
    for stop in ("bash: line 5:   741 Killed                  simpleFoam\n",
                 "bash: line 5:   741 Terminated              simpleFoam\n", "Killed\n"):
        log, sidecar = _write(tmp_path, sid, _cut(_text(sid), 0.6) + stop)
        v = verify_path(log, sidecar)
        assert (v.verdict, v.rule) == ("insufficient", "VER-5.16"), stop


# ---------------------------------------------------------------- what still rejects a cut log
@pytest.mark.parametrize("tail,value", [
    ("--> FOAM FATAL ERROR: (openfoam-2406)\nMaximum number of iterations exceeded\n\n"
     "    From Foam::solve\n\nFOAM exiting\n", "FOAM FATAL ERROR"),
    ("[stack trace]\n=============\n#1  Foam::sigFpe::sigHandler(int) at ??:?\n", "SIGFPE"),
    ("bash: line 5:   741 Floating point exceptionicoFoam\n", "SIGFPE"),
    ("bash: line 5:   741 Segmentation fault      icoFoam\n", "SIGSEGV"),
])
def test_a_fatal_signature_in_a_cut_log_still_rejects_and_is_cited(tmp_path, tail, value):
    sid = "icoFoam_cavity__healthy"
    log, sidecar = _write(tmp_path, sid, _cut(_text(sid), 0.6) + tail)
    v = verify_path(log, sidecar)
    assert (v.verdict, v.rule) == ("reject", "VER-5.2")
    [crash] = [r for r in v.reasons if r.code == "run.crash"]
    assert crash.value == value
    assert sentences.TRUNCATED_LOG not in v.not_assessed


def test_a_non_finite_residual_in_a_cut_log_still_rejects(tmp_path):
    sid = "icoFoam_cavity__healthy"
    text = _cut(_text(sid), 0.6)
    at = text.rfind("Initial residual = ")
    end = text.index(",", at)
    text = text[:at] + "Initial residual = nan" + text[end:]
    log, sidecar = _write(tmp_path, sid, text)
    v = verify_path(log, sidecar)
    assert (v.verdict, v.rule) == ("reject", "VER-5.2")
    assert any(r.code == "run.divergence" for r in v.reasons)


# ---------------------------------------------------------------- whose End (VER-3a, next)
#: A utility's output ahead of the solver's in one file, as `(blockMesh && icoFoam) > log` writes
#: it (synthetic lines in blockMesh's format).
UTILITY_FIRST = "Exec   : blockMesh\n\nCreating block mesh topology\n\nWriting polyMesh\n\nEnd\n\n"

#: What Open MPI can print at finalize, after a parallel solver's End (synthetic lines in its
#: format).
MPI_FINALIZE = (
    "\nFinalising parallel run\n"
    "[node01:04711] 3 more processes have sent help message help-mpi-btl-base.txt / btl:no-nics\n"
    "[node01:04711] Set MCA parameter \"orte_base_help_aggregate\" to 0 to see all help / error "
    "messages\n")


def test_only_an_end_after_the_last_time_step_is_the_solvers_end_under_next():
    cut = "Time = 1\nsmoothSolver:  Solving for Ux\n\nTime = 2\nsmoothSolver:  Solving for Ux\n"
    utility = "Exec   : blockMesh\n\nEnd\n\n"
    for after in (False, True):
        assert ends_in_end(cut + "\nEnd\n", after_last_step=after)
        assert ends_in_end(cut + "\nEnd\n" + MPI_FINALIZE, after_last_step=after)
        assert ends_in_end(utility + cut + "\nEnd\n", after_last_step=after)
        assert not ends_in_end(cut, after_last_step=after)
        assert ends_in_end("Exec   : checkMesh\n\nEnd\n", after_last_step=after)  # no time step
    assert ends_in_end(utility + cut)                              # 0.5.0's reading, the default
    assert not ends_in_end(utility + cut, after_last_step=True)
    assert infer_exit_code(utility + cut, after_last_step=True) is None
    assert run_exit_code(utility + cut, None, after_last_step=True) is None
    assert run_exit_code(utility + cut, 0, after_last_step=True) == 0     # given, used as read
    # OpenFOAM.org's foamMultiRun writes every `Time =` line behind a prefix of blanks
    # (`setGlobalPrefix`, the parser's VER-20 reading), and its End behind it too: its last step
    # is found there, not taken for absent (synthetic lines in its format).
    pad = " " * 10
    multi = (f"{pad}Time = 1\nbottomAir DILUPBiCGStab:  Solving for h\n{pad}\n"
             f"{pad}Time = 2\nbottomAir DILUPBiCGStab:  Solving for h\n")
    assert ends_in_end(utility + multi)                            # 0.5.0's reading, the default
    assert not ends_in_end(utility + multi, after_last_step=True)
    assert run_exit_code(utility + multi, None, after_last_step=True) is None
    assert ends_in_end(utility + multi + f"{pad}End\n{pad}\n", after_last_step=True)
    assert ends_in_end(utility + multi + "\nEnd\n", after_last_step=True)
    # In parallel its End, behind the blanks, is followed by "Finalising parallel run": the End
    # is read behind the same blanks as the last step under next; 0.5.0's reading finds no End
    # there (insufficient, kept by default).
    parallel = multi + f"{pad}End\n{pad}\n" + MPI_FINALIZE
    assert ends_in_end(parallel, after_last_step=True)
    assert ends_in_end(utility + parallel, after_last_step=True)
    assert not ends_in_end(parallel)
    # Behind other blanks than the last step's, or after an unprefixed step, it is not the End.
    assert not ends_in_end(utility + multi + "    End\n" + MPI_FINALIZE, after_last_step=True)
    assert not ends_in_end(utility + cut + f"{pad}End\n" + MPI_FINALIZE, after_last_step=True)


def test_one_reading_of_the_end_serves_the_verdict_the_report_and_the_convergence(tmp_path,
                                                                                 monkeypatch):
    """`verdict_exit_code` is the end a verdict reads under its checks (VER-3, VER-3a): the
    pipeline, the HTML report's series and `courant convergence` all parse the log with it, so
    none of them takes a utility's End for a cut run's under next while the verdict does not
    (the final review of 0.6.0, 21 Oct 2026, found `as_run.Run.ended` reading its own)."""
    import courant.watcher.exitcode as exitcode
    from courant.converge import from_run_inputs
    from courant.verify.html import _series_from_inputs

    utility = "Exec   : blockMesh\n\nEnd\n\n"
    cut = "Time = 1\nsmoothSolver:  Solving for Ux\n\nTime = 2\nsmoothSolver:  Solving for Ux\n"
    assert exitcode.verdict_exit_code(utility + cut, None, "default") == 0     # 0.5.0's reading
    assert exitcode.verdict_exit_code(utility + cut, None, "next") is None      # VER-3a
    assert exitcode.verdict_exit_code(utility + cut, 0, "next") == 0            # given, as read
    assert exitcode.verdict_exit_code(utility + cut, 3, "default") == 3

    sid = "icoFoam_cavity__healthy"
    log, sidecar = _write(tmp_path, sid, UTILITY_FIRST + _cut(_text(sid), 0.6))
    asked: list[tuple[str, int | None]] = []
    real = exitcode.verdict_exit_code

    def spy(text, given, checks="default"):
        code = real(text, given, checks)
        asked.append((checks, code))
        return code

    monkeypatch.setattr(exitcode, "verdict_exit_code", spy)
    inputs = from_log(log, sidecar).model_copy(update={"checks": "next"})
    verify_inputs(inputs)
    _series_from_inputs(inputs)
    from_run_inputs(inputs)
    assert asked == [("next", None)] * 3


def test_a_utilitys_end_ahead_of_a_cut_solver_log_is_not_the_runs_end_under_next(tmp_path):
    """VER-3 as released reads `End` anywhere on a line of its own, so a utility's `End` ahead of
    the solver's output reads as the run's end, and a run cut mid-way is judged as finished: here
    accepted, grade A. VER-3a, behind `--checks next` (GATE-13: an accept that becomes
    insufficient), counts only an `End` after the last time step."""
    sid = "icoFoam_cavity__healthy"
    log, sidecar = _write(tmp_path, sid, UTILITY_FIRST + _cut(_text(sid), 0.6))
    nxt = verify_path(log, sidecar, checks="next")
    assert (nxt.verdict, nxt.rule, nxt.grade) == ("insufficient", "VER-5.16", None)
    [r] = _insufficient(nxt)
    assert (r.code, r.value) == ("insufficient.truncated_log", INFERRED)
    default = verify_path(log, sidecar)
    assert (default.verdict, default.rule) == ("accept", "VER-5.6")   # 0.5.0's reading, kept


@pytest.mark.parametrize("before,after", [(UTILITY_FIRST, ""), ("", MPI_FINALIZE),
                                          (UTILITY_FIRST, MPI_FINALIZE)])
def test_a_finished_run_reads_as_finished_in_both_modes(tmp_path, before, after):
    sid = "icoFoam_cavity__healthy"
    log, sidecar = _write(tmp_path, sid, before + _text(sid).rstrip("\n") + "\n" + after)
    for checks in ("default", "next"):
        v = verify_path(log, sidecar, checks=checks)
        assert (v.verdict, v.rule) == ("accept", "VER-5.6"), checks


# ---------------------------------------------------------------- an exit code given (VER-1)
def test_an_exit_code_given_with_the_run_decides_as_read(tmp_path):
    sid = "icoFoam_cavity__healthy"
    log, sidecar = _write(tmp_path, sid, _cut(_text(sid), 0.6))
    inputs = from_log(log, sidecar)

    clean = verify_inputs(inputs.model_copy(update={"exit_code": 0}))
    assert clean.rule != "VER-5.16" and sentences.TRUNCATED_LOG not in clean.not_assessed

    killed = verify_inputs(inputs.model_copy(update={"exit_code": 137}))
    assert (killed.verdict, killed.rule) == ("reject", "VER-5.2")
    [crash] = [r for r in killed.reasons if r.code == "run.crash"]
    assert crash.value == "return code 137"
    assert "given" in crash.input and crash.input != log.name
    assert "log parser" not in crash.method

    whole, _ = _write(tmp_path, sid, _text(sid))
    failed = verify_inputs(from_log(whole, sidecar).model_copy(update={"exit_code": 1}))
    assert (failed.verdict, failed.rule) == ("reject", "VER-5.2")


# ---------------------------------------------------------------- with a specification
@pytest.fixture
def cut_demo(tmp_path):
    from courant.data import extract_demo_case

    case = extract_demo_case(tmp_path / "demo")
    log = case / "log.simpleFoam"
    log.write_text(_cut(log.read_text(), 0.5))
    return case


def _spec(tmp_path: Path, value: float) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"schema": "courant.spec/1", "items": [
        {"name": "viscosity", "file": "constant/transportProperties", "key": "nu",
         "value": value}]}))
    return path


def test_a_specification_cannot_raise_a_cut_log_but_a_wrong_problem_still_rejects(tmp_path,
                                                                                 cut_demo):
    same = verify_path(cut_demo, spec=_spec(tmp_path, 1e-05))
    assert (same.verdict, same.rule, same.grade) == ("insufficient", "VER-5.16", None)
    other = verify_path(cut_demo, spec=_spec(tmp_path, 3e-05))
    assert (other.verdict, other.rule) == ("reject", "VER-5.7")
    assert sentences.TRUNCATED_LOG in other.not_assessed
    # The specification decides the rule, not whether a run that may not be over has a grade.
    assert other.grade is None


@pytest.mark.parametrize("rule", ["VER-5.8", "VER-5.11", "VER-5.12"])
def test_no_specification_rule_gives_a_cut_log_a_grade(tmp_path, cut_demo, rule):
    """A specification rule that replaces a cut log's VER-5.16 with another `insufficient`
    leaves it ungraded: the run may not be over, whatever the specification says."""
    path = _spec(tmp_path, {"VER-5.11": -1e-05, "VER-5.12": 3e-05}.get(rule, 1e-05))
    if rule == "VER-5.12":                                     # extracted, nobody confirmed it
        data = json.loads(path.read_text())
        data["provenance"] = {"extracted_by": "courant spec extract"}
        path.write_text(json.dumps(data))
    locked = "0" * 64 if rule == "VER-5.8" else None           # not the hash it was locked at
    v = verify_path(cut_demo, spec=path, spec_locked_sha256=locked)
    assert (v.verdict, v.rule, v.grade) == ("insufficient", rule, None)


# ---------------------------------------------------------------- convergence of a cut run
def _unknown_end(conv):
    return [f for f in conv.findings if f.code == "CONV.RUN_END_UNKNOWN"]


def test_the_convergence_of_a_cut_monitored_run_reports_no_waste(cut_demo):
    """The monitors of a steady case settle; its log is cut mid-run. 0.5.0's inferred crash
    withheld the waste figure for the wrong reason; without it the run read as finished and the
    iterations, wall clock and core-hours 'past convergence' of a run whose end is unknown were
    reported. The end is unknown: the status is WARN, a fixed sentence says why, and no waste
    figure is given."""
    from courant.converge import convergence_path
    from courant.converge import sentences as conv_sentences

    conv = convergence_path(cut_demo)
    assert conv.data["converged_at_step"] is not None         # the answer did settle
    assert conv.status == "WARN"
    assert conv.data["waste"] is None
    [f] = _unknown_end(conv)
    assert f.value["ended_by"] == "unknown"
    assert f.message == conv_sentences.RUN_END_UNKNOWN.format(
        last_complete_step=f.value["last_complete_step"])
    assert not any(f.code == "CONV.RUN_KILLED" for f in conv.findings)
    assert not any(re.search(r"crash|stopped by a signal", f.message or "")
                   for f in conv.findings)

    v = verify_path(cut_demo)
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.16")
    c = v.convergence
    assert (c.status, c.wasted_steps, c.wasted_wall_clock_s, c.wasted_core_hours) == (
        "WARN", None, None, None)
    assert f.message in c.messages


def test_a_finished_or_given_end_still_reports_the_waste(cut_demo, tmp_path):
    """The same case with its whole log, and the cut log with an exit code of 0 given with the
    run (used as read, VER-1): the end is known and the waste figure is reported."""
    from courant.converge import from_run_inputs
    from courant.data import extract_demo_case
    from courant.verify import from_path

    whole = from_run_inputs(from_path(extract_demo_case(tmp_path / "whole")))
    assert whole.status == "PASS" and whole.data["waste"] is not None and not _unknown_end(whole)
    given = from_run_inputs(from_path(cut_demo).model_copy(update={"exit_code": 0}))
    assert given.data["waste"] is not None and not _unknown_end(given)


def test_mcp_and_the_cli_report_no_waste_for_a_cut_monitored_run(cut_demo):
    from typer.testing import CliRunner

    from courant.cli.main import app
    from courant.converge import sentences as conv_sentences
    from courant.mcp import tools

    got = tools.check_convergence(str(cut_demo), confine=tools.Confinement([cut_demo]))
    assert got["status"] == "WARN"
    assert (got["wasted_steps"], got["wasted_wall_clock_s"], got["wasted_core_hours"]) == (
        None, None, None)
    assert any(m.startswith(conv_sentences.RUN_END_UNKNOWN.split("{")[0]) for m in got["messages"])
    result = CliRunner().invoke(app, ["convergence", str(cut_demo)])
    assert result.exit_code == 0, result.output
    assert "past convergence" not in result.output
    assert "is not known" in result.output


# ---------------------------------------------------------------- mid-run: MCP and watch
def test_mcp_verify_run_on_a_log_still_being_written_is_insufficient_not_a_crash(tmp_path):
    from courant.mcp import tools

    sid = "pisoFoam_RAS_cavity__healthy"
    log, _ = _write(tmp_path, sid, _cut(_text(sid), 0.4))
    got = tools.verify_run(str(log), confine=tools.Confinement([tmp_path]))
    assert (got["verdict"], got["rule"]) == ("insufficient", "VER-5.16")
    assert got["reasons"][0]["code"] == "insufficient.truncated_log"
    assert "return code" not in json.dumps(got)


def test_mcp_check_convergence_on_a_log_still_being_written_does_not_say_it_crashed(tmp_path):
    from courant.mcp import tools

    sid = "simpleFoam_pitzDaily__healthy"
    log, _ = _write(tmp_path, sid, _cut(_text(sid), 0.4))
    got = tools.check_convergence(str(log), confine=tools.Confinement([tmp_path]))
    assert not any(re.search(r"crash|stopped by a signal", m) for m in got["messages"]), \
        got["messages"]


def test_watch_and_verify_agree_on_when_a_run_ended(tmp_path):
    """One reading of `End` (VER-3). A parallel OpenFOAM run prints "Finalising parallel run"
    after `End`; `courant verify` called it finished, while `courant watch` waited for an `End`
    on the last line, went idle and told the user to verify it once it had finished."""
    from courant.cli.tail import Chunk
    from courant.live.watch import run_watch

    sid = "icoFoam_cavity__healthy"
    text = _text(sid).rstrip("\n") + "\n\nFinalising parallel run\n"
    log, _ = _write(tmp_path, sid, text)
    assert verify_path(log).verdict == "accept"
    lines: list[str] = []
    out = run_watch([Chunk(text=text[i:i + 3000], offset=i) for i in range(0, len(text), 3000)],
                    case_dir=None, log_path=log, emit=lines.append,
                    clock=iter(range(10 ** 6)).__next__)
    assert out.verdict is not None and out.verdict.verdict == "accept", lines
    assert out.exit_code == 0


def test_watch_reads_open_mpis_finalize_lines_after_end_as_finished(tmp_path):
    from courant.cli.tail import Chunk
    from courant.live.watch import run_watch

    sid = "icoFoam_cavity__healthy"
    text = _text(sid).rstrip("\n") + "\n" + MPI_FINALIZE
    log, _ = _write(tmp_path, sid, text)
    lines: list[str] = []
    out = run_watch([Chunk(text=text[i:i + 3000], offset=i) for i in range(0, len(text), 3000)],
                    case_dir=None, log_path=log, emit=lines.append,
                    clock=iter(range(10 ** 6)).__next__)
    assert out.verdict is not None and out.verdict.verdict == "accept", lines


def test_watch_does_not_take_a_utilitys_end_for_the_solvers_as_the_solver_starts(tmp_path):
    """`(blockMesh && icoFoam) > log`, followed live: a chunk that ends partway into the
    solver's first banner line, just after the utility's End, is not the end of the run."""
    from courant.cli.tail import Chunk
    from courant.live.watch import EXIT_IDLE, run_watch

    sid = "icoFoam_cavity__healthy"
    solver = _cut(_text(sid), 0.5)
    assert "\n" not in solver[:10]
    text = UTILITY_FIRST + solver
    log, _ = _write(tmp_path, sid, text)
    at = len(UTILITY_FIRST) + 10
    verified: list = []
    lines: list[str] = []
    out = run_watch([Chunk(text=text[:at], offset=0), Chunk(text=text[at:], offset=at)],
                    case_dir=None, log_path=log, emit=lines.append,
                    clock=iter(range(10 ** 6)).__next__, verify_fn=verified.append)
    assert (verified, out.verdict, out.exit_code) == ([], None, EXIT_IDLE), lines
    assert not any("the solver finished" in ln for ln in lines), lines


def test_watch_on_a_log_that_stops_before_end_neither_alarms_nor_verifies(tmp_path):
    from courant.cli.tail import Chunk
    from courant.live import RUNNING
    from courant.live.watch import EXIT_IDLE, run_watch

    sid = "icoFoam_cavity__healthy"
    text = _cut(_text(sid), 0.5)
    log, _ = _write(tmp_path, sid, text)
    lines: list[str] = []
    out = run_watch([Chunk(text=text, offset=0)], case_dir=None, log_path=log,
                    emit=lines.append, clock=iter(range(10 ** 6)).__next__)
    assert (out.exit_code, out.state.state, out.state.alarm, out.verdict) == (
        EXIT_IDLE, RUNNING, None, None)
    assert not any("FAILING" in ln or "crash" in ln for ln in lines), lines
