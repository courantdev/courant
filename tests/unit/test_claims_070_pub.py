"""Claims audit 0.7.0 (research/notes/claims-audit-070): public/README.md, public/SECURITY.md,
public/CONTRIBUTING.md (the public repository's README.md, SECURITY.md and CONTRIBUTING.md).

Each test pins one claim those documents make about what Courant checks, reads, refuses or
does, where no other test pinned it. Line numbers are public/README.md's (README.md in the
public repository) and public/SECURITY.md's after the 0.7.0 audit's edits. Only public code
and public data (courant/, certification/corpus, packaging/install.sh) are read.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.cli.main import app
from courant.data import DEMO
from courant.verify import verify_path

ROOT = Path(__file__).resolve().parents[2]
#: Here public/README.md is the public README; in the public repository it is README.md.
README = next(p for p in (ROOT / "public" / "README.md", ROOT / "README.md") if p.exists())
CORPUS = ROOT / "certification" / "corpus"
INSTALLER = ROOT / "packaging" / "install.sh"
BLOWUP = "icoFoam_cavity__cfl_blowup"


def _demo(sid: str) -> tuple[Path, Path]:
    return DEMO / f"{sid}.log.gz", DEMO / f"{sid}.case.json"


def _corpus(sid: str) -> tuple[Path, Path]:
    return CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json"


# ------------------------------------------------------------------------- the verdict


def test_the_readme_sample_verdict_is_what_courant_verify_prints(tmp_path, monkeypatch):
    """README:101-137: the `text` block is the output of `courant verify` on the demo's icoFoam
    cavity with a timestep ten times too large, `# exit 2` (README:98)."""
    text = README.read_text()
    block = re.search(r"```text\n\$ courant verify (?P<cmd>[^\n]+)\n(?P<out>.*?)```", text,
                      re.DOTALL)
    assert block, "the README's sample verdict block is gone"
    d = tmp_path / "demo" / BLOWUP
    d.mkdir(parents=True)
    for src in _demo(BLOWUP):
        shutil.copy(src, d / src.name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    res = CliRunner().invoke(app, ["verify", *block.group("cmd").split()])
    assert res.exit_code == 2, res.output
    assert res.output == block.group("out"), "the README's sample verdict is not today's output"


def test_the_readme_python_example_gives_what_it_says():
    """README:279-284: `verify_path(log, case_json)` on the demo's blow-up gives
    ('reject', 'VER-5.3', 'cfl.max'), exit code 2, and `to_json()` is courant.verdict/2."""
    v = verify_path(*_demo(BLOWUP))
    assert (v.verdict, v.rule, v.reasons[0].code) == ("reject", "VER-5.3", "cfl.max")
    assert v.exit_code == 2
    assert json.loads(v.to_json())["schema"] == "courant.verdict/2"


def test_the_exit_code_is_the_verdicts(tmp_path, monkeypatch):
    """README:140-141: "The exit code is the verdict's: `accept` 0, `reject` 2, `insufficient`
    3, and 4 for an input that cannot be read." """
    monkeypatch.chdir(tmp_path)
    r = CliRunner()
    log, cj = _demo("icoFoam_cavity__healthy")
    assert r.invoke(app, ["verify", str(log), "--case-json", str(cj)]).exit_code == 0
    log, cj = _demo(BLOWUP)
    assert r.invoke(app, ["verify", str(log), "--case-json", str(cj)]).exit_code == 2
    (tmp_path / "log.simpleFoam").write_text("hello\nworld\n")
    assert r.invoke(app, ["verify", "log.simpleFoam"]).exit_code == 3
    assert r.invoke(app, ["verify", "nope.log"]).exit_code == 4


def test_every_reason_cites_its_input_and_method_but_not_always_a_threshold():
    """README:10-11 (as corrected): "every reason cites the file it read and the method, with
    the threshold and the value wherever its check has one". Before the audit it said every
    reason cites "the threshold and the value": on the committed corpus most reasons have no
    threshold, so the sentence was false. The input and the method are always there."""
    reasons = []
    for log in sorted(CORPUS.glob("*.log.gz"))[:10]:            # the first ten: fast enough
        cj = log.with_name(log.name.replace(".log.gz", ".case.json"))
        reasons += verify_path(log, cj).reasons
    assert len(reasons) > 20
    assert all(r.input.strip() and r.method.strip() for r in reasons)
    assert any(r.threshold is None for r in reasons), (
        "every reason now carries a threshold: README:10-11 may say so again")


def test_the_two_pitzdaily_runs_of_the_readme_figure():
    """README:23-28: run A is pitzDaily as shipped, run B the same case with its pressure
    solver's tolerance loosened from 1e-06 to 0.01 (and, as corrected, its relative tolerance
    from 0.1 to 0.3); both logs end with `End`; A is accepted (VER-5.6), B rejected (VER-5.4),
    first reason `residual.not_converged`."""
    a, b = _corpus("simpleFoam_pitzDaily__healthy"), _corpus("simpleFoam_pitzDaily__mass_imbalance")
    ca, cb = (json.loads(p[1].read_text())["config"]["linear_solvers"]["p"] for p in (a, b))
    assert (ca["tolerance"], cb["tolerance"]) == (1e-06, 0.01)
    assert (ca["rel_tol"], cb["rel_tol"]) == (0.1, 0.3)
    others = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    assert others == {"tolerance", "rel_tol"}, "B now differs from A in more than the README says"
    for log, _ in (a, b):
        with gzip.open(log, "rt") as fh:
            assert fh.read().rstrip().splitlines()[-1].strip() == "End"
    va, vb = verify_path(*a), verify_path(*b)
    assert (va.verdict, va.rule) == ("accept", "VER-5.6")
    assert (vb.verdict, vb.rule, vb.reasons[0].code) == ("reject", "VER-5.4",
                                                          "residual.not_converged")


# ------------------------------------------------------------------------- install


def test_the_distribution_its_python_and_its_commands():
    """README:60-64: Python 3.11 or later; the distribution is `courant-dev`, with an `mcp`
    extra; the commands are `courant` and `courant-mcp`."""
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert meta["name"] == "courant-dev"
    assert meta["requires-python"] == ">=3.11"
    assert "mcp" in meta["optional-dependencies"]
    from importlib.metadata import distribution

    commands = {e.name for e in distribution("courant-dev").entry_points
                if e.group == "console_scripts"}
    assert commands == {"courant", "courant-mcp"}


def _sh(script: str, path: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/sh", "-c", f'COURANT_INSTALL_LIB=1 . "{INSTALLER}"; {script}'],
                          env={"PATH": path, "HOME": os.environ.get("HOME", "/tmp"), **env},
                          capture_output=True, text=True, timeout=30, check=False)


def _tools(tmp_path: Path, names: tuple[str, ...]) -> str:
    d = tmp_path / "bin"
    d.mkdir()
    for name in names:
        found = shutil.which(name, path="/usr/bin:/bin")
        if found:
            (d / name).symlink_to(found)
    return str(d)


@pytest.mark.skipif(not (shutil.which("sha256sum") or shutil.which("shasum")),
                    reason="no sha256sum or shasum on this machine")
def test_the_installer_refuses_a_binary_whose_checksum_does_not_match(tmp_path):
    """README:75-76 and SECURITY.md:20-22: the installer checks the binary's SHA-256 against the
    release's SHA256SUMS, and "it refuses on a mismatch"."""
    binary = tmp_path / "courant"
    binary.write_bytes(b"binary")
    import hashlib

    good = hashlib.sha256(b"binary").hexdigest()
    path = _tools(tmp_path, ("sha256sum", "shasum", "cut", "printf"))
    assert _sh(f'verify_checksum "{binary}" {good}', path).returncode == 0
    bad = _sh(f'verify_checksum "{binary}" {"0" * 64}', path)
    assert bad.returncode == 1 and "Checksum mismatch" in bad.stderr


def test_without_a_hash_tool_the_installer_refuses_unless_told_to_skip(tmp_path):
    """SECURITY.md:25-27 (added by the audit): with neither `sha256sum` nor `shasum` the
    installer cannot compare the binary's SHA-256 and refuses, unless COURANT_SKIP_VERIFY=1,
    which installs it without that comparison (README's settings table names it too)."""
    binary = tmp_path / "courant"
    binary.write_bytes(b"binary")
    path = _tools(tmp_path, ("cut", "printf"))
    run = f'verify_checksum "{binary}" {"0" * 64} && echo PASSED'
    refused = _sh(run, path)
    assert refused.returncode == 1 and "PASSED" not in refused.stdout
    assert "COURANT_SKIP_VERIFY=1" in refused.stderr
    skipped = _sh(run, path, COURANT_SKIP_VERIFY="1")
    assert skipped.returncode == 0 and "PASSED" in skipped.stdout


def test_the_installer_refuses_a_signed_sums_that_does_not_list_the_binary():
    """SECURITY.md:25 (added by the audit): the installer refuses when the signed SHA256SUMS
    does not list the binary it downloaded."""
    text = INSTALLER.read_text()
    assert re.search(r'\[ -n "\$expected" \] \|\| die "The signed SHA256SUMS does not list', text)


# ------------------------------------------------------------------------- settings


def _gate_options() -> dict[str, list[str] | None]:
    import typer

    cmd = typer.main.get_command(app).commands["gate"]
    out = {}
    for p in cmd.params:
        if p.opts and p.opts[0].startswith("--") and p.opts[0] != "--help":
            env = p.envvar
            out[p.opts[0]] = [env] if isinstance(env, str) else env
    return out


def test_courant_gate_options_and_their_environment_variables():
    """README:252 (as corrected): "Every `courant gate` option but `--jobs`, `--cache` and
    `--debug` as an environment variable" (`COURANT_GATE_<OPTION>`). Before the audit it said
    every option; those three have none."""
    opts = _gate_options()
    without = {o for o, env in opts.items() if not env}
    assert without == {"--jobs", "--cache", "--debug"}
    for opt, env in opts.items():
        if env:
            assert env == ["COURANT_GATE_" + opt[2:].upper().replace("-", "_")], (opt, env)


def test_the_gate_reads_its_options_from_the_environment(tmp_path, monkeypatch):
    """README:252: the Action sets the gate's options as COURANT_GATE_* variables: the paths
    and the summary file named only in the environment are used."""
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    for src in _demo(BLOWUP):
        shutil.copy(src, repo / "runs" / src.name)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("COURANT_GATE_ROOT", str(repo))
    monkeypatch.setenv("COURANT_GATE_PATHS", "runs/*.log.gz")
    monkeypatch.setenv("COURANT_GATE_SUMMARY", str(summary))
    res = CliRunner().invoke(app, ["gate"])
    assert res.exit_code != 0, res.output                  # the blow-up is rejected
    assert summary.exists() and "cfl.max" in summary.read_text()


def test_doctor_says_whether_a_model_key_is_set(monkeypatch):
    """README:257: "`courant doctor` says whether one [a model key] is set". A placeholder
    value, not a key; doctor without --models contacts nothing."""
    for name in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    unset = CliRunner().invoke(app, ["doctor"])
    assert "no model provider configured" in unset.output, unset.output
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key-for-a-test")
    with_key = CliRunner().invoke(app, ["doctor"])
    assert "no model provider configured" not in with_key.output
    assert "openrouter" in with_key.output.lower(), with_key.output
