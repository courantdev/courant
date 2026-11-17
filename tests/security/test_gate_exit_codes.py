"""Security review of 0.7.0: paths where CI read exit 0, "the check passes", for a run that was
not judged as asked (research/notes/security-review-070/FINDINGS.md)."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from courant.gate.config import ConfigError, GateConfig, RunGroup
from courant.verify.batch import discover

DEMO = Path(__file__).resolve().parents[2] / "courant" / "data" / "demo"
COURANT = str(Path(sys.executable).parent / "courant")


def _log(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.decompress((DEMO / "icoFoam_cavity__healthy.log.gz").read_bytes()))
    return path


def _courant(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([COURANT, *args], cwd=cwd, capture_output=True, text=True, timeout=120, check=False,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(cwd)})


def test_sec070_11_a_run_named_by_two_groups_with_other_specs_is_refused(tmp_path):
    """SEC070-11: `judge_groups` judged a run only with the first group naming it, and the
    Action's paths and spec come after courant.toml's groups, so a spec-less `cases/*` group in
    courant.toml dropped the specification the Action passed: the gate accepted, exit 0."""
    from courant.gate.run import judge_groups

    _log(tmp_path / "cases" / "c1" / "log.icoFoam")
    (tmp_path / "spec.json").write_text("{}")
    config = GateConfig(runs=(RunGroup(paths=("cases/*",)),
                              RunGroup(paths=("cases/c1",), spec="spec.json",
                                       spec_sha256="0" * 64)))
    with pytest.raises(ConfigError, match="c1"):
        judge_groups(tmp_path, config)


@pytest.mark.parametrize("make", ["missing", "empty", "file"])
def test_sec070_12_a_batch_with_no_runs_is_not_a_pass(tmp_path, make):
    """SEC070-12: `courant verify --batch` exited 0 ("0 runs") for a root that does not exist, is
    empty, or is a file: a typo in CI read as every run accepted."""
    root = tmp_path / "runs"
    if make == "empty":
        root.mkdir()
    elif make == "file":
        root.write_text("not a directory")
    r = _courant("verify", "--batch", str(root), cwd=tmp_path)
    assert r.returncode == 4, (r.returncode, r.stdout, r.stderr)


def test_sec070_12_a_path_with_batch_is_refused(tmp_path):
    """The positional path was silently ignored next to --batch."""
    _log(tmp_path / "runs" / "a.log")
    bad = _log(tmp_path / "other.log")
    r = _courant("verify", str(bad), "--batch", str(tmp_path / "runs"), cwd=tmp_path)
    assert r.returncode == 4, (r.returncode, r.stdout, r.stderr)


def test_sec070_12_audit_of_no_runs_is_not_a_pass(tmp_path):
    (tmp_path / "runs").mkdir()
    r = _courant("audit", str(tmp_path / "runs"), cwd=tmp_path)
    assert r.returncode == 4, (r.returncode, r.stdout, r.stderr)


def test_sec070_13_only_processor_n_directories_are_skipped(tmp_path):
    """SEC070-13: any directory starting with `processor` was skipped, so `processor_study/`
    hid a run from the batch; decomposed runs use `processor<N>` only."""
    _log(tmp_path / "processor_study" / "bad.log")
    _log(tmp_path / "processor0" / "x.log")
    found = [t.relative_to(tmp_path).as_posix() for t, _ in discover(tmp_path)]
    assert found == ["processor_study/bad.log"]


@pytest.mark.parametrize("blank", ["", "   "])
def test_sec070_18_a_blank_spec_lock_is_refused(tmp_path, blank):
    """SEC070-18: `--spec-sha256 "$LOCK"` with the variable unset judged the specification
    unlocked, and an unlocked accept exits 0 like a locked one; the gate already refuses this.
    Under `--checks next` (`spec.lock_strict`): the default keeps 0.7.0's verdict (GATE-13)."""
    log = _log(tmp_path / "run.log")
    (tmp_path / "spec.json").write_text("{}")
    r = _courant("verify", str(log), "--spec", str(tmp_path / "spec.json"), "--spec-sha256",
                 blank, "--checks", "next", cwd=tmp_path)
    assert r.returncode == 4 and "spec-sha256" in r.stderr, (r.returncode, r.stderr)


@pytest.mark.parametrize("blank", ["", "   "])
def test_sec070_18_a_blank_spec_lock_keeps_the_default_verdict(tmp_path, blank):
    """SEC070-18 under the default: a blank lock is no lock, and the run gets the verdict and the
    exit code 0.7.0 gave it, the same as with the option left out (GATE-13). In this process, as
    `tests/unit/test_vpa_cli.py`, which holds the document byte for byte: the installed `courant`
    may be another checkout's."""
    from typer.testing import CliRunner

    from courant.cli.main import app
    from courant.spec.model import SPEC_SCHEMA, Spec, SpecItem

    log = _log(tmp_path / "run.log")
    spec = tmp_path / "spec.json"
    spec.write_text(Spec(schema=SPEC_SCHEMA, task=None, items=(
        SpecItem(name="nu", file="constant/transportProperties", key="nu", value=1e-5),
    )).to_json())
    args = ["verify", str(log), "--spec", str(spec), "--json"]
    plain = CliRunner().invoke(app, args)
    r = CliRunner().invoke(app, [*args, "--spec-sha256", blank])
    assert plain.exit_code in (0, 2, 3), plain.output
    got, want = json.loads(r.stdout), json.loads(plain.stdout)
    assert r.exit_code == plain.exit_code
    assert (got["verdict"], got["rule"], got["basis"]) == (want["verdict"], want["rule"],
                                                           want["basis"])


def test_sec070_19_an_empty_mcp_root_is_not_the_working_directory(tmp_path, monkeypatch):
    """SEC070-19: `--root ""` passed the default-root guard and resolved to the working
    directory, so a client config with an unset variable, started in /, confined to the disk."""
    from courant.mcp.server import unconfined_default

    monkeypatch.chdir("/")
    assert unconfined_default([""]) is not None
    assert unconfined_default(["  "]) is not None
    assert unconfined_default([str(tmp_path)]) is None
