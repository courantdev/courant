"""The courant-gate Action and courant.toml (TRD GATE-7, GATE-8; T-GATE-ACTION).

The Action has no logic of its own: `courant gate` decides, and this file tests it without
GitHub, on `tests/fixtures/gate-repo/`, a repository whose check fails with the cited reason.
The Action's YAML is tested as data: its inputs, the wiring from each input to the command's
environment, which steps may use the network. Jobs are capped at 2.
"""

from __future__ import annotations

import re
from courant import __version__
import shutil
import socket
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from courant.gate.config import ConfigError, load_config, make_group, merged
from courant.verify.pack import check_pack

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "gate-repo"
ACTION = ROOT / ".github" / "actions" / "courant-gate" / "action.yml"
WORKFLOW = ROOT / ".github" / "workflows" / "gate-action.yml"
CORPUS = ROOT / "certification" / "corpus"


def _app():
    from courant.cli.main import app

    return app


@pytest.fixture
def repo(tmp_path):
    """A writable copy of the fixture repository."""
    dest = tmp_path / "repo"
    shutil.copytree(FIXTURE, dest)
    return dest


def _gate(repo: Path, *args: str, env: dict | None = None):
    base = {"COURANT_CACHE_DIR": str(repo.parent / "cache")}
    return CliRunner().invoke(_app(), ["gate", "--root", str(repo), *args], env={**base, **(env or {})})


def _add_corpus(repo: Path, sid: str, dest: str = "runs", name: str | None = None) -> None:
    name = name or sid
    (repo / dest).mkdir(parents=True, exist_ok=True)
    shutil.copy(CORPUS / f"{sid}.log.gz", repo / dest / f"{name}.log.gz")
    shutil.copy(CORPUS / f"{sid}.case.json", repo / dest / f"{name}.case.json")


def _toml(repo: Path, text: str) -> None:
    (repo / "courant.toml").write_text(text)


# ----------------------------------------------------------------------- GATE-7: the fixture


def test_the_fixture_repository_fails_with_the_cited_reason(repo):
    summary = repo.parent / "summary.md"
    r = _gate(repo, "--packs-dir", str(repo.parent / "packs"), "--summary", str(summary))
    assert r.exit_code == 1, r.output
    text = summary.read_text()
    assert text == r.stdout
    assert "## Courant gate: 1 of 2 runs accepted" in text
    row = next(ln for ln in text.splitlines() if "cavity_blowup" in ln and ln.startswith("|"))
    assert "**reject (VER-5.3), grade C**" in row and "`cfl.max`" in row
    assert "Courant number exceeds stability limit" in row and "| validity |" in row
    ok = next(ln for ln in text.splitlines() if "cavity_healthy" in ln and ln.startswith("|"))
    assert "accept (VER-5.6)" in ok and "**" not in ok
    assert "**The check fails:** 1 of 2 runs fail it." in text
    assert "- `runs/cavity_blowup.log.gz`: reject (VER-5.3), first reason `cfl.max`" in text


def test_one_summary_row_per_run_with_run_verdict_basis_and_first_reason(repo):
    r = _gate(repo)
    header = next(ln for ln in r.stdout.splitlines() if ln.startswith("| Run"))
    assert header == "| Run | Verdict | Basis | First reason |"
    rows = [ln for ln in r.stdout.splitlines() if ln.startswith("| `")]
    assert len(rows) == 2 and all(len(ln.split(" | ")) == 4 for ln in rows)


def test_the_summary_is_appended_to_the_file_a_workflow_names(repo):
    summary = repo.parent / "step-summary.md"
    summary.write_text("an earlier step wrote this\n")
    _gate(repo, "--summary", str(summary))
    text = summary.read_text()
    assert text.startswith("an earlier step wrote this\n## Courant gate")


def test_each_pack_is_written_kept_in_the_pack_store_and_re_derives(repo):
    packs = repo.parent / "packs"
    r = _gate(repo, "--packs-dir", str(packs))
    assert r.exit_code == 1
    for name in ("runs__cavity_blowup", "runs__cavity_healthy"):
        for where in (packs / name, repo / "evidence" / name):
            assert {p.name for p in where.iterdir()} == {"verdict.json", "report.md",
                                                         "inputs.manifest"}
    blow = (packs / "runs__cavity_blowup" / "verdict.json").read_text()
    assert '"verdict":"reject"' in blow and "cfl.max" in blow
    ok, message = check_pack(packs / "runs__cavity_blowup", repo / "runs")
    assert ok, message
    assert (packs / "runs__cavity_blowup" / "verdict.json").read_bytes() == \
        (repo / "evidence" / "runs__cavity_blowup" / "verdict.json").read_bytes()
    assert "kept in `evidence/`" in r.stdout and "Warning" not in r.stdout


def test_without_a_pack_store_the_summary_warns_that_packs_expire(repo):
    _toml(repo, '[[run]]\npaths = ["runs/*.log.gz"]\n')
    r = _gate(repo, "--packs-dir", str(repo.parent / "packs"))
    assert "the evidence packs exist only as workflow artefacts, which expire" in r.stdout
    assert not (repo / "evidence").exists()
    quiet = _gate(repo)                                          # no packs asked for: no warning
    assert "Warning" not in quiet.stdout


def test_the_check_passes_when_every_run_is_accepted(repo):
    (repo / "runs" / "cavity_blowup.log.gz").unlink()
    (repo / "runs" / "cavity_blowup.case.json").unlink()
    r = _gate(repo)
    assert r.exit_code == 0
    assert "**The check passes:** 1 of 1 runs accepted." in r.stdout


def test_a_cache_hit_gives_the_same_summary(repo):
    assert _gate(repo).exit_code == 1
    assert not (repo.parent / "cache").exists()                  # off unless asked for
    first = _gate(repo, "--cache")
    second = _gate(repo, "--cache")
    assert first.stdout == second.stdout and first.exit_code == second.exit_code == 1
    assert len(list((repo.parent / "cache").glob("*.json"))) == 2


def test_jobs_give_the_same_check(repo):
    one = _gate(repo)
    two = _gate(repo, "--jobs", "2")
    assert one.stdout == two.stdout and one.exit_code == two.exit_code == 1


# ----------------------------------------------------------------------- GATE-7: fail-on


def _insufficient_log(repo: Path, name: str = "noise") -> None:
    """A log the gate cannot recognise as solver output: insufficient."""
    (repo / "runs" / f"{name}.log").write_text("this is not a solver log\nnothing here\n")


def test_insufficient_fails_by_default_and_passes_under_fail_on_reject_saying_so(repo):
    (repo / "runs" / "cavity_blowup.log.gz").unlink()
    (repo / "runs" / "cavity_blowup.case.json").unlink()
    _insufficient_log(repo)
    _toml(repo, '[[run]]\npaths = ["runs/*.log*"]\n')
    strict = _gate(repo)
    assert strict.exit_code == 1 and "insufficient (VER-5.1)" in strict.stdout
    lenient = _gate(repo, "--fail-on", "reject")
    assert lenient.exit_code == 0, lenient.output
    row = next(ln for ln in lenient.stdout.splitlines() if "noise" in ln and ln.startswith("|"))
    assert "insufficient (VER-5.1) – passes: fail-on is reject" in row
    assert "`fail-on: reject`" in lenient.stdout
    assert "let through by `fail-on: reject`" in lenient.stdout


def test_a_reject_fails_under_either_fail_on(repo):
    assert _gate(repo, "--fail-on", "reject").exit_code == 1


def test_the_toml_sets_fail_on_and_an_input_overrides_it(repo):
    (repo / "runs" / "cavity_blowup.log.gz").unlink()
    (repo / "runs" / "cavity_blowup.case.json").unlink()
    _insufficient_log(repo)
    _toml(repo, 'fail-on = "reject"\n[[run]]\npaths = ["runs/*.log*"]\n')
    assert _gate(repo).exit_code == 0
    assert _gate(repo, "--fail-on", "not-accept").exit_code == 1


def _demo_repo(repo: Path, lock: str | None) -> Path:
    """The bundled demo case, a specification drafted from it, and `lock` as the hash it was
    locked at (None: the right one)."""
    from courant.data import extract_demo_case
    from courant.spec import draft_spec, spec_sha256

    shutil.rmtree(repo / "runs")
    (repo / "cases").mkdir()
    case = extract_demo_case(repo / "cases" / "demo")
    spec = repo / "specs" / "task.json"
    spec.parent.mkdir()
    spec.write_text(draft_spec(case).to_json())
    real = spec_sha256(spec)
    _toml(repo, f'fail-on = "reject"\n[[run]]\npaths = ["cases/demo/*"]\nspec = "specs/task.json"\n'
                f'spec-sha256 = "{lock or real}"\n')
    return spec


def test_a_locked_specification_that_matches_gives_an_accept_on_conformance(repo):
    _demo_repo(repo, None)
    r = _gate(repo)
    assert r.exit_code == 0, r.output
    assert "accept (VER-5.6), grade" in r.stdout and "| validity+conformance |" in r.stdout


def test_a_specification_edited_after_its_lock_fails_whatever_fail_on_says(repo):
    _demo_repo(repo, "0" * 64)
    r = _gate(repo)                                    # the file says fail-on = "reject"
    assert r.exit_code == 1, r.output
    assert "insufficient (VER-5.8)" in r.stdout
    assert "the specification's integrity: VER-5.8" in r.stdout
    assert "passes: fail-on is reject" not in r.stdout


def test_a_specification_needs_its_lock_hash(repo):
    _toml(repo, '[[run]]\npaths = ["runs/*.log.gz"]\nspec = "specs/task.json"\n')
    r = _gate(repo)
    assert r.exit_code == 4 and "'spec' needs 'spec-sha256'" in r.stderr
    flat = _gate(repo, "--paths", "runs/*.log.gz", "--spec", "specs/task.json")
    assert flat.exit_code == 4 and "needs 'spec-sha256'" in flat.stderr


def test_a_missing_specification_file_is_unjudged_not_a_pass(repo):
    _toml(repo, '[[run]]\npaths = ["runs/*.log.gz"]\nspec = "specs/nope.json"\n'
                f'spec-sha256 = "{"a" * 64}"\n')
    r = _gate(repo)
    assert r.exit_code == 4 and "no such specification" in r.stdout


# ----------------------------------------------------------------------- GATE-7: outputs and exit 4


def test_outputs_name_where_each_runs_log_comes_from(repo):
    shutil.rmtree(repo / "runs")
    (repo / "cases" / "cavity_healthy").mkdir(parents=True)
    (repo / "cases" / "cavity_blowup").mkdir(parents=True)
    _add_corpus(repo, "icoFoam_cavity__healthy", "solver-output", "cavity_healthy")
    _add_corpus(repo, "icoFoam_cavity__cfl_blowup", "solver-output", "cavity_blowup")
    _toml(repo, '[[run]]\npaths = ["cases/*"]\noutputs = "solver-output"\n')
    r = _gate(repo)
    assert r.exit_code == 1, r.output
    assert "`cfl.max`" in r.stdout and "1 of 2 runs accepted" in r.stdout


def test_a_run_with_no_output_fails_the_check_whatever_fail_on_says(repo):
    shutil.rmtree(repo / "runs")
    (repo / "cases" / "cavity_healthy").mkdir(parents=True)
    (repo / "cases" / "cavity_lost").mkdir(parents=True)
    _add_corpus(repo, "icoFoam_cavity__healthy", "solver-output", "cavity_healthy")
    _toml(repo, 'fail-on = "reject"\n[[run]]\npaths = ["cases/*"]\noutputs = "solver-output"\n')
    r = _gate(repo)
    assert r.exit_code == 4
    row = next(ln for ln in r.stdout.splitlines() if "cavity_lost" in ln and ln.startswith("|"))
    assert "**no output**" in row and "no output for 'cavity_lost' under solver-output" in row


def test_a_pattern_that_matches_nothing_is_unjudged(repo):
    _toml(repo, '[[run]]\npaths = ["nowhere/*"]\n')
    r = _gate(repo)
    assert r.exit_code == 4 and "matches no case directory or log" in r.stdout


def test_an_unreadable_run_is_exit_4_and_the_others_are_still_reported(repo):
    (repo / "runs" / "empty_case").mkdir()
    _toml(repo, '[[run]]\npaths = ["runs/*"]\n')
    r = _gate(repo)
    assert r.exit_code == 4, r.output
    assert "**unreadable**" in r.stdout and "no solver log" in r.stdout
    assert "cavity_blowup" in r.stdout and "cavity_healthy" in r.stdout


def test_no_paths_anywhere_is_an_error(tmp_path):
    r = CliRunner().invoke(_app(), ["gate", "--root", str(tmp_path)],
                           env={"COURANT_CACHE_DIR": str(tmp_path / "c")})
    assert r.exit_code == 4 and "no runs named" in r.stderr


def test_a_pack_store_outside_the_repository_or_a_named_store_is_refused(repo):
    out = _gate(repo, "--pack-store", "../elsewhere")
    assert out.exit_code == 4 and "outside the repository" in out.stderr
    named = _gate(repo, "--pack-store", "store:team")
    assert named.exit_code == 4 and "named pack stores are not available" in named.stderr


def test_a_pattern_cannot_leave_the_repository(repo):
    r = _gate(repo, "--paths", "../*")
    assert r.exit_code == 4 and "leaves the repository" in r.stderr


def test_an_unexpected_error_is_one_line_and_exit_4_with_debug_for_the_traceback(repo, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("courant.gate.run.run_gate", boom)
    r = _gate(repo)
    assert r.exit_code == 4
    assert r.stderr.strip().count("\n") == 0
    assert f"internal error (courant {__version__}): RuntimeError: disk on fire" in r.stderr
    assert "--debug" in r.stderr and "Traceback" not in r.stderr
    again = _gate(repo, "--debug")
    assert isinstance(again.exception, RuntimeError)


def test_the_gate_uses_no_network(repo, monkeypatch):
    def no_socket(*a, **k):
        raise AssertionError("the gate opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_socket)
    monkeypatch.setattr(socket, "create_connection", no_socket)
    monkeypatch.setattr(socket, "getaddrinfo", no_socket)
    assert _gate(repo, "--packs-dir", str(repo.parent / "packs")).exit_code == 1


def test_the_gate_reads_the_repository_and_writes_only_where_it_is_told(repo):
    before = {p.relative_to(repo).as_posix(): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    _toml(repo, '[[run]]\npaths = ["runs/*.log.gz"]\n')          # no pack-store
    _gate(repo)
    after = {p.relative_to(repo).as_posix(): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    after.pop("courant.toml", None)
    before.pop("courant.toml", None)
    assert after == before


# ----------------------------------------------------------------------- GATE-8: courant.toml


def test_the_fixtures_courant_toml_reads():
    cfg = load_config(FIXTURE / "courant.toml")
    assert cfg.fail_on == "not-accept" and cfg.pack_store == "evidence"
    assert [g.paths for g in cfg.runs] == [("runs/*.log.gz",)]


def test_every_key_courant_toml_allows_is_read(tmp_path):
    p = tmp_path / "courant.toml"
    p.write_text(f'''fail-on = "reject"
pack-store = "evidence"
courant-version = "latest"
rules = ["rules/org.json"]

[[run]]
paths = ["a/*", "b/*"]
outputs = ["out1", "out2"]
spec = "specs/t.json"
spec-sha256 = "{"AB" * 32}"

[[run]]
paths = "c/*"
''')
    cfg = load_config(p)
    assert (cfg.fail_on, cfg.pack_store, cfg.courant_version, cfg.rules) == (
        "reject", "evidence", "latest", ("rules/org.json",))
    a, c = cfg.runs
    assert a.paths == ("a/*", "b/*") and a.outputs == ("out1", "out2")
    assert a.spec == "specs/t.json" and a.spec_sha256 == "ab" * 32
    assert c.paths == ("c/*",) and c.outputs == () and c.spec is None


@pytest.mark.parametrize("text, message", [
    ('fial-on = "reject"\n', "unknown key 'fial-on' in the top level (allowed: fail-on, "),
    ('fail_on = "reject"\n', "unknown key 'fail_on'"),
    ('[[run]]\npaths = ["a"]\nspecc = "x"\n', "unknown key 'specc' in [[run]] #1 (allowed: paths, "),
    ('[[run]]\nouputs = "o"\npaths = ["a"]\n', "unknown key 'ouputs'"),
    ('[tool]\nx = 1\n', "unknown key 'tool'"),
    ('fail-on = "sometimes"\n', "fail-on must be one of not-accept, reject"),
    ('fail-on = 3\n', "fail-on must be a string"),
    ('run = "x"\n', "'run' must be an array of tables"),
    ('[[run]]\noutputs = "o"\n', "has no 'paths'"),
    ('[[run]]\npaths = [1]\n', "must be a string or a list of strings"),
    (f'[[run]]\npaths = ["a"]\nspec = "s.json"\nspec-sha256 = "{"z" * 64}"\n', "64 hexadecimal digits"),
    ('[[run]]\npaths = ["a"]\nspec-sha256 = "' + "a" * 64 + '"\n', "'spec-sha256' without 'spec'"),
    ('pack-store = 4\n', "pack-store must be a path"),
    ('this is = not toml\n[\n', "is not valid TOML"),
])
def test_an_unknown_or_malformed_key_is_an_error_never_a_silent_default(tmp_path, text, message):
    p = tmp_path / "courant.toml"
    p.write_text(text)
    with pytest.raises(ConfigError) as exc:
        load_config(p)
    assert message in str(exc.value)
    assert str(exc.value).startswith("courant.toml") or "courant.toml" in str(exc.value) \
        or "inputs" in str(exc.value)


def test_the_cli_turns_a_bad_courant_toml_into_exit_4_naming_the_key(repo):
    _toml(repo, 'fail-ni = "reject"\n[[run]]\npaths = ["runs/*.log.gz"]\n')
    r = _gate(repo)
    assert r.exit_code == 4
    assert r.stderr.startswith("courant gate: courant.toml: unknown key 'fail-ni'")
    assert "Traceback" not in r.stderr and not r.stdout


def test_a_named_settings_file_that_is_missing_is_an_error(repo):
    r = _gate(repo, "--config", "nope.toml")
    assert r.exit_code == 4 and "no such settings file" in r.stderr


def test_inputs_add_a_group_and_override_the_top_level(repo):
    base = load_config(FIXTURE / "courant.toml")
    group = make_group(["extra/*"], ["out"], None, None)
    cfg = merged(base, group=group, fail_on="reject", pack_store="elsewhere",
                 courant_version="0.4.0")
    assert len(cfg.runs) == 2 and cfg.runs[1] is group
    assert (cfg.fail_on, cfg.pack_store, cfg.courant_version) == ("reject", "elsewhere", "0.4.0")
    kept = merged(base)
    assert (kept.fail_on, kept.pack_store) == ("not-accept", "evidence")


def test_outputs_spec_without_paths_is_an_error(repo):
    r = _gate(repo, "--outputs", "out")
    assert r.exit_code == 4 and "apply to `paths`" in r.stderr


def test_a_pinned_courant_version_that_is_not_this_one_refuses(repo):
    r = _gate(repo, "--courant-version", "0.0.1")
    assert r.exit_code == 4 and f"pins Courant 0.0.1; this is Courant {__version__}" in r.stderr
    assert _gate(repo, "--courant-version", "latest").exit_code == 1
    assert _gate(repo, "--courant-version", __version__).exit_code == 1


def test_rules_files_are_read_and_must_exist(repo):
    _toml(repo, 'rules = ["rules/org.json"]\n[[run]]\npaths = ["runs/*.log.gz"]\n')
    assert "rules file not found: rules/org.json" in _gate(repo).stderr
    (repo / "rules").mkdir()
    (repo / "rules" / "org.json").write_text("{}")
    assert _gate(repo).exit_code == 1


# ----------------------------------------------------------------------- the Action's YAML


@pytest.fixture(scope="module")
def action() -> dict:
    return yaml.safe_load(ACTION.read_text())


def _steps(action: dict) -> dict:
    return {s["name"]: s for s in action["runs"]["steps"]}


def test_the_action_is_a_composite_with_the_inputs_the_trd_names(action):
    assert action["name"] == "courant-gate" and action["runs"]["using"] == "composite"
    # GATE-7's inputs: paths, outputs, spec (with spec-sha256), fail-on, pack-store,
    # courant-version; plus where the settings file is, the working directory and install.
    assert set(action["inputs"]) == {
        "paths", "outputs", "spec", "spec-sha256", "fail-on", "pack-store", "courant-version",
        "config", "working-directory", "install"}
    for name, spec in action["inputs"].items():
        assert spec.get("description"), name
        assert spec["required"] is False and "default" in spec
    assert action["inputs"]["courant-version"]["default"] == "latest"
    assert action["inputs"]["install"]["default"] == "true"
    assert action["inputs"]["fail-on"]["default"] == ""        # courant.toml, then not-accept


def test_every_input_reaches_the_command_and_every_variable_is_one_the_command_reads(action):
    import typer.main

    verify = _steps(action)["Verify"]
    env = verify["env"]
    command = typer.main.get_command(_app()).commands["gate"]
    read = {p.envvar for p in command.params if p.envvar}
    assert set(env) <= read, f"set by the Action, read by nothing: {sorted(set(env) - read)}"
    assert read - set(env) == {"COURANT_GATE_ROOT", "COURANT_GATE_SUMMARY"}
    body = yaml.dump(action["runs"])
    for name in action["inputs"]:
        assert f"inputs.{name}" in body, f"input {name} is never used"


def test_the_actions_environment_run_on_the_fixture_gives_the_check_that_fails(
        action, repo, monkeypatch):
    """The Verify step's environment, filled in the way GitHub fills it, drives `courant gate`
    to the fixture's failure: the wiring from input to command is what ships."""
    inputs = {"paths": "runs/*.log.gz", "outputs": "", "spec": "", "spec-sha256": "",
              "fail-on": "", "pack-store": "", "courant-version": "latest", "config": ""}
    env = {}
    for key, value in _steps(action)["Verify"]["env"].items():
        env[key] = re.sub(r"\$\{\{\s*inputs\.([\w-]+)\s*\}\}", lambda m: inputs[m.group(1)], value)
        env[key] = env[key].replace("${{ runner.temp }}", str(repo.parent / "tmp"))
    env["COURANT_GATE_SUMMARY"] = str(repo.parent / "summary.md")
    env["COURANT_CACHE_DIR"] = str(repo.parent / "cache")
    monkeypatch.chdir(repo)
    r = CliRunner().invoke(_app(), ["gate"], env=env)
    assert r.exit_code == 1, r.output
    assert "`cfl.max`" in (repo.parent / "summary.md").read_text()
    assert (repo.parent / "tmp" / "courant-packs" / "runs__cavity_blowup" / "verdict.json").is_file()
    # courant.toml in the repository supplied `fail-on` and `pack-store`; the inputs added a group
    assert (repo / "evidence" / "runs__cavity_blowup").is_dir()


def test_the_action_has_no_logic_of_its_own(action):
    """The command decides. The YAML installs, runs `courant gate`, uploads and passes the code on."""
    for name, step in _steps(action).items():
        script = step.get("run", "")
        assert not re.search(r"\b(accept|reject|insufficient|verdict|jq|awk|sed|grep)\b", script), name
    verify = _steps(action)["Verify"]["run"]
    assert verify.count("courant gate") == 1


def test_only_the_install_and_the_artefact_upload_touch_the_network(action):
    steps = _steps(action)
    net = {name for name, s in steps.items()
           if re.search(r"\b(curl|wget|pip|pipx|npm|git clone|install\.sh)\b", s.get("run", ""))
           or "upload-artifact" in s.get("uses", "") or "download-artifact" in s.get("uses", "")}
    assert net == {"Install Courant", "Keep the evidence packs"}
    assert steps["Install Courant"]["if"] == "inputs.install == 'true'"
    # the installer from the action's own repository and ref, which names the release repository
    assert 'sh "$GITHUB_ACTION_PATH/../../../packaging/install.sh"' in steps["Install Courant"]["run"]
    assert (ACTION.parents[3] / "packaging" / "install.sh").is_file()
    assert "courant" not in steps["Keep the evidence packs"].get("run", "")


def test_the_packs_are_uploaded_even_when_the_check_fails_and_the_failure_is_passed_on(action):
    steps = list(action["runs"]["steps"])
    names = [s["name"] for s in steps]
    assert names == ["Install Courant", "Verify", "Keep the evidence packs", "Fail the check"]
    upload = steps[2]
    assert upload["if"] == "always()" and upload["uses"].startswith("actions/upload-artifact@")
    assert upload["with"]["path"] == "${{ runner.temp }}/courant-packs"
    verify, fail = steps[1], steps[3]
    # the command's code is captured, not allowed to stop the job before the upload
    assert "set +e" in verify["run"] and "exit-code=$code" in verify["run"]
    assert fail["if"] == "steps.gate.outputs.exit-code != '0'" and 'exit "${CODE:-4}"' in fail["run"]
    assert verify["id"] == "gate"
    assert action["outputs"]["exit-code"]["value"] == "${{ steps.gate.outputs.exit-code }}"


def test_a_courant_that_cannot_start_is_not_a_failing_run(action, tmp_path):
    """Python exits 1 when `courant` cannot even import (gate-action, 11 Oct 2026: no `click`),
    and 1 means "a run fails the check". The Verify step checks that `courant` starts, and
    reports 4 (nothing judged) when it does not. The step's own script, run with a `courant`
    that crashes and one that rejects."""
    import os
    import stat
    import subprocess

    script = _steps(action)["Verify"]["run"]
    for crash, want in ((True, "4"), (False, "1")):
        bindir = tmp_path / ("crash" if crash else "rejects")
        bindir.mkdir()
        fake = bindir / "courant"
        fake.write_text("#!/bin/sh\n" + ("exit 1\n" if crash else
                                          '[ "$1" = version ] && exit 0\nexit 1\n'))
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        out = tmp_path / f"out-{crash}"
        env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
               "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
               "COURANT_GATE_PACKS_DIR": str(tmp_path / "packs")}
        r = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True,
                           check=False)
        assert r.returncode == 0, r.stderr
        assert f"exit-code={want}" in out.read_text().splitlines()


def test_the_actions_own_workflow_runs_it_on_the_fixture_and_requires_the_failure():
    wf = yaml.safe_load(WORKFLOW.read_text())
    assert "contents: read" in WORKFLOW.read_text()
    job = wf["jobs"]["fixture-fails-with-its-reason"]
    uses = next(s for s in job["steps"] if s.get("uses") == "./.github/actions/courant-gate")
    assert uses["continue-on-error"] is True
    assert uses["with"] == {"install": "false", "working-directory": "tests/fixtures/gate-repo"}
    check = next(s for s in job["steps"] if "OUTCOME" in (s.get("env") or {}))
    assert 'test "$OUTCOME" = "failure"' in check["run"] and 'test "$CODE" = "1"' in check["run"]
    assert "cfl.max" in check["run"]
    assert "secrets." not in WORKFLOW.read_text()


# ----------------------------------------------------------------------- verify --batch reads it too


def test_verify_batch_reads_courant_toml_in_the_directory(repo):
    """GATE-8: `courant verify --batch` and the Action read the same file."""
    _add_corpus(repo, "icoFoam_cavity__missing_dict", "elsewhere")        # not named by `paths`
    r = CliRunner().invoke(_app(), ["verify", "--batch", str(repo), "--no-cache"])
    assert r.exit_code == 2, r.output
    assert "2 runs: 1 of 2 accept, 1 of 2 reject, 0 of 2 insufficient" in r.stdout
    assert "elsewhere" not in r.stdout
    assert "runs/cavity_blowup.log.gz" in r.stdout


def test_verify_batch_refuses_an_unknown_key_in_courant_toml(repo):
    _toml(repo, 'fial-on = "reject"\n[[run]]\npaths = ["runs/*.log.gz"]\n')
    r = CliRunner().invoke(_app(), ["verify", "--batch", str(repo)])
    assert r.exit_code == 4 and "unknown key 'fial-on'" in r.stderr and not r.stdout


def test_verify_batch_in_a_directory_without_courant_toml_is_what_it_was(tmp_path):
    _add_corpus(tmp_path, "icoFoam_cavity__healthy", ".")
    r = CliRunner().invoke(_app(), ["verify", "--batch", str(tmp_path), "--no-cache"])
    assert r.exit_code == 0 and "1 runs: 1 of 1 accept" in r.stdout


def test_verify_batch_reports_a_run_with_no_output_as_unreadable(repo):
    _toml(repo, '[[run]]\npaths = ["runs/*.log.gz"]\noutputs = "solver-output"\n')
    r = CliRunner().invoke(_app(), ["verify", "--batch", str(repo), "--no-cache"])
    assert r.exit_code == 4 and "no output for 'cavity_blowup'" in r.stdout


def test_spec_integrity_uses_the_policys_own_spec_void():
    """One implementation: the hash-mismatch case is `policy.spec_void`, not a copy of its rule ids."""
    import inspect

    from courant.verify import outcome, policy

    src = inspect.getsource(outcome.spec_integrity)
    assert "policy.spec_void" in src
    assert '"VER-5.8"' not in inspect.getsource(outcome)
    from types import SimpleNamespace as NS

    block = NS(locked_sha256="a" * 64, spec_sha256="b" * 64)
    void = NS(rule="VER-5.3", reasons=(), conformance=block)
    assert policy.spec_void(block) and outcome.spec_integrity(void)
    ok = NS(rule="VER-5.3", reasons=(), conformance=NS(locked_sha256="a" * 64, spec_sha256="a" * 64))
    assert not outcome.spec_integrity(ok)
