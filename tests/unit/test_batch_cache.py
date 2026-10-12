"""Batch at scale (TRD GATE-11, T-GATE-BATCH).

`courant verify --batch --jobs N` gives the results a serial run gives, and a content-addressed
cache re-verifies only the runs that changed. Jobs are capped at 2 here (the laptop).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from courant.verify import cache as vcache
from courant.verify.batch import verify_all, verify_batch
from courant.verify.cache import SCHEMA, VerdictCache
from tests.unit.test_verify_pack import CORPUS, _case, _copy, _spec

JOBS = 2


def _tree(tmp: Path) -> Path:
    """Three corpus logs with sidecars, and one case directory (an unread `Allrun` among its
    files)."""
    root = tmp / "runs"
    root.mkdir()
    for sid in ("icoFoam_cavity__healthy", "icoFoam_cavity__cfl_blowup",
                "buoyantPimpleFoam_hotRoom__healthy"):
        _copy(root, sid)
    work = tmp / "work"
    work.mkdir()
    shutil.copytree(_case(work), root / "case")
    return root


def _bytes(results):
    return [(t.name, None if v is None else v.to_json(), e) for t, v, e in results]


def _app():
    from courant.cli.main import app

    return app


# ------------------------------------------------------------------------------ jobs


def test_jobs_give_the_results_a_serial_run_gives(tmp_path):
    root = _tree(tmp_path)
    serial = verify_all(root, jobs=1)
    parallel = verify_all(root, jobs=JOBS)
    assert len(serial) == 4
    assert _bytes(parallel) == _bytes(serial)
    assert [t for t, _, _ in parallel] == [t for t, _, _ in serial]      # the same order


def test_jobs_and_serial_agree_with_a_specification(tmp_path):
    root = _tree(tmp_path)
    spec = _spec(root / "case", tmp_path)
    serial = verify_all(root, spec=spec)
    parallel = verify_all(root, jobs=JOBS, spec=spec)
    assert _bytes(parallel) == _bytes(serial)
    case = next(v for t, v, _ in serial if t.name == "case")
    assert case.basis == "validity+unlocked-spec" and case.conformance is not None


def test_an_unreadable_run_is_reported_the_same_way_in_parallel(tmp_path):
    root = _tree(tmp_path)
    (root / "empty").mkdir()
    (root / "empty" / "log.icoFoam").write_text("")
    spec = tmp_path / "missing-spec.json"
    serial = verify_all(root, spec=spec)
    parallel = verify_all(root, jobs=JOBS, spec=spec)
    assert all(v is None and "no such specification" in e for _, v, e in serial)
    assert _bytes(parallel) == _bytes(serial)


def test_the_cli_prints_the_same_lines_for_any_jobs(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setenv("COURANT_CACHE_DIR", str(tmp_path / "cache"))
    runner = CliRunner()
    one = runner.invoke(_app(), ["verify", "--batch", str(root), "--no-cache"])
    two = runner.invoke(_app(), ["verify", "--batch", str(root), "--jobs", str(JOBS), "--no-cache"])
    assert one.exit_code == two.exit_code == 2
    assert one.stdout == two.stdout
    assert one.stdout.splitlines()[-1].startswith("4 runs: ")


# ------------------------------------------------------------------------------ the cache


def _cache(tmp: Path) -> VerdictCache:
    return VerdictCache(tmp / "cache")


def _no_verify(monkeypatch):
    """Make a re-verification impossible: anything the cache serves was not verified now."""
    def boom(*a, **k):
        raise AssertionError("verified, not served from the cache")

    monkeypatch.setattr("courant.verify.verify_path", boom)


def test_a_second_batch_is_served_from_the_cache_with_the_same_bytes(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    first = verify_batch(root, cache=cache)
    assert (first.hits, first.misses) == (0, 4)
    _no_verify(monkeypatch)
    second = verify_batch(root, cache=cache)
    assert (second.hits, second.misses) == (4, 0)
    assert _bytes(second.results) == _bytes(first.results)


def test_a_cached_batch_equals_an_uncached_one(tmp_path):
    root = _tree(tmp_path)
    uncached = verify_all(root)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)                                       # fills
    assert _bytes(verify_batch(root, cache=cache).results) == _bytes(uncached)


def test_an_entry_is_courant_cache_1_and_carries_its_own_hash(tmp_path):
    import hashlib

    from courant.verify.schema import canonical_json

    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    entries = sorted((tmp_path / "cache").glob("*.json"))
    assert len(entries) == 4
    for p in entries:
        e = json.loads(p.read_text())
        assert e["schema"] == SCHEMA == "courant.cache/1"
        assert p.name == e["key"] + ".json"
        body = {k: v for k, v in e.items() if k not in ("sha256", "hmac")}
        assert e["sha256"] == hashlib.sha256(canonical_json(body).encode()).hexdigest()
        assert e["verdict"]["schema"] == "courant.verdict/2"
        assert e["fingerprint"]["inputs"] == e["verdict"]["run"]["inputs"]


def test_only_the_run_that_changed_is_verified_again(tmp_path):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    log = root / "icoFoam_cavity__healthy.log.gz"
    # a different (valid) log under the same name
    shutil.copy(CORPUS / "icoFoam_cavity__cfl_blowup.log.gz", log)
    again = verify_batch(root, cache=cache)
    assert (again.hits, again.misses) == (3, 1)
    fresh = {t.name: v.to_json() for t, v, _ in verify_all(root)}
    assert {t.name: v.to_json() for t, v, _ in again.results} == fresh
    changed = next(v for t, v, _ in again.results if t.name == log.name)
    assert changed.verdict == "reject"                                    # not the stale accept


def test_an_edit_to_a_file_a_reader_read_is_a_miss(tmp_path):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    u = root / "case" / "0.5" / "U"
    u.write_text(u.read_text() + "\n// edited\n")
    again = verify_batch(root, cache=cache)
    assert (again.hits, again.misses) == (3, 1)


def test_an_edit_to_a_file_nothing_read_is_a_hit(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    first = verify_batch(root, cache=cache)
    read = {f.path for f in next(v for t, v, _ in first.results if t.name == "case").run.inputs}
    assert "Allrun" not in read
    allrun = root / "case" / "Allrun"
    allrun.write_text("#" * allrun.stat().st_size)             # same size: nothing a reader sees
    _no_verify(monkeypatch)
    assert verify_batch(root, cache=cache).misses == 0


def test_a_file_that_appears_or_goes_is_a_miss(tmp_path):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    (root / "case" / "log.simpleFoam").write_text("a later log that find_log would rank\n")
    assert verify_batch(root, cache=cache).misses == 1
    (root / "case" / "log.simpleFoam").unlink()
    assert verify_batch(root, cache=cache).misses == 1                    # back to the first set


def test_a_log_gaining_a_sidecar_is_a_miss(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    log, sidecar = _copy(tmp_path, "icoFoam_cavity__healthy")
    shutil.move(str(log), root / log.name)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    shutil.move(str(sidecar), root / sidecar.name)
    assert verify_batch(root, cache=cache).misses == 1


def test_the_version_the_policy_the_specification_and_its_lock_are_in_the_key(tmp_path,
                                                                                monkeypatch):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    spec = _spec(root / "case", tmp_path)
    verify_batch(root, cache=cache, spec=spec)
    assert verify_batch(root, cache=cache, spec=spec).misses == 0
    assert verify_batch(root, cache=cache).misses == 4                    # no spec: another key
    assert verify_batch(root, cache=cache, spec=spec, spec_locked_sha256="0" * 64).misses == 4
    spec.write_text(spec.read_text() + "\n")
    assert verify_batch(root, cache=cache, spec=spec).misses == 4         # other spec bytes
    import courant

    monkeypatch.setattr(courant, "__version__", "9.9.9")
    assert verify_batch(root, cache=cache, spec=spec).misses == 4
    monkeypatch.undo()
    monkeypatch.setattr(vcache, "policy_digest", lambda: "another policy")
    assert verify_batch(root, cache=cache, spec=spec).misses == 4


def test_the_context_names_the_platform_and_the_schema_major():
    ctx = vcache.context()
    assert ctx["verdict_schema_major"] == 2
    assert ctx["platform"] and ctx["courant_version"] and len(ctx["policy"]) == 64


def test_a_damaged_entry_is_discarded_never_trusted(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    first = verify_batch(root, cache=cache)
    for p in sorted((tmp_path / "cache").glob("*.json")):
        text = p.read_text()
        if '"verdict":"reject"' in text:
            # flip the answer and leave the entry's own hash alone
            p.write_text(text.replace('"verdict":"reject"', '"verdict":"accept"', 1))
        else:
            p.write_text(text[: len(text) // 2])                           # torn
    again = verify_batch(root, cache=cache)
    assert again.hits == 0 and again.misses == 4
    assert _bytes(again.results) == _bytes(first.results)
    assert verify_batch(root, cache=cache).misses == 0                     # rewritten whole


def test_an_entry_whose_manifest_disagrees_with_its_verdict_is_discarded(tmp_path):
    from courant.verify.cache import _entry_digest

    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    for p in (tmp_path / "cache").glob("*.json"):
        e = json.loads(p.read_text())
        e["fingerprint"]["inputs"] = e["fingerprint"]["inputs"][:-1] or []
        e["sha256"] = _entry_digest(e)                                     # a consistent forgery
        p.write_text(json.dumps(e))
    assert verify_batch(root, cache=cache).hits == 0


def test_unreadable_runs_are_not_cached(tmp_path):
    root = _tree(tmp_path)
    spec = tmp_path / "nope.json"
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache, spec=spec)
    assert not list((tmp_path / "cache").glob("*.json"))


def test_jobs_fill_the_cache_without_tearing_an_entry(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    parallel = verify_batch(root, jobs=JOBS, cache=cache)
    assert parallel.misses == 4
    assert not list((tmp_path / "cache").glob(".entry-*"))
    _no_verify(monkeypatch)
    assert _bytes(verify_batch(root, jobs=JOBS, cache=cache).results) == _bytes(parallel.results)


def test_a_cache_that_cannot_be_written_does_not_fail_the_batch(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    out = verify_batch(_tree(tmp_path), cache=VerdictCache(blocker / "sub"))
    assert out.misses == 4 and all(e is None for _, _, e in out.results)


def test_the_cli_uses_the_cache_and_no_cache_turns_it_off(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cdir = tmp_path / "cache"
    monkeypatch.setenv("COURANT_CACHE_DIR", str(cdir))
    runner = CliRunner()
    monkeypatch.delenv("CI", raising=False)
    r1 = runner.invoke(_app(), ["verify", "--batch", str(root), "--jobs", str(JOBS)])
    assert r1.exit_code == 2 and len(list(cdir.glob("*.json"))) == 4
    assert "0 of 4 runs served from the cache" in r1.output
    r2 = runner.invoke(_app(), ["verify", "--batch", str(root)])
    assert "4 of 4 runs served from the cache" in r2.output
    assert r2.stdout == r1.stdout
    shutil.rmtree(cdir)
    r3 = runner.invoke(_app(), ["verify", "--batch", str(root), "--no-cache"])
    assert r3.exit_code == 2 and not cdir.exists() and "cache" not in r3.output


def test_a_version_1_batch_is_served_from_the_same_entries(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setenv("COURANT_CACHE_DIR", str(tmp_path / "cache"))
    runner = CliRunner()
    runner.invoke(_app(), ["verify", "--batch", str(root), "--cache"])
    r = runner.invoke(_app(), ["verify", "--batch", str(root), "--schema", "1", "--cache"])
    assert "4 of 4 runs served" in r.output and "NOT_TRUSTWORTHY" in r.stdout


def test_the_cache_is_a_batch_feature_only():
    """MCP-4: no MCP tool imports the cache. Its users are the batch driver, the two CLI commands
    that batch (`verify --batch`, `gate`) and the golden-file generator, which writes one entry."""
    root = Path(__file__).resolve().parents[2] / "courant"
    users = {str(p.relative_to(root)) for p in root.rglob("*.py")
             if "verify.cache" in p.read_text() or "from courant.verify import cache" in p.read_text()}
    assert users <= {"verify/batch.py", "cli/commands/verify.py", "cli/commands/gate.py",
                     "schemas/golden.py"}, users
    assert not any(u.startswith("mcp/") for u in users)


def test_the_default_directory_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("COURANT_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert vcache.default_dir() == tmp_path / "courant" / "verdicts"
    monkeypatch.setenv("COURANT_CACHE_DIR", str(tmp_path / "x"))
    assert vcache.default_dir() == tmp_path / "x"
    monkeypatch.delenv("COURANT_CACHE_DIR")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert vcache.default_dir() == Path.home() / ".cache" / "courant" / "verdicts"


@pytest.mark.parametrize("jobs", [0, -3])
def test_nonsense_jobs_mean_one(tmp_path, jobs):
    root = tmp_path / "runs"
    root.mkdir()
    _copy(root, "icoFoam_cavity__healthy")
    assert len(verify_all(root, jobs=jobs)) == 1


# ------------------------------------------------------------------ review: a wrong verdict is worse than none


def _hit_count(root, cache):
    return verify_batch(root, cache=cache)


def test_a_directory_that_appears_or_goes_is_a_miss_even_when_empty(tmp_path):
    """A crashed run leaves an empty `1000/`; the readers take the latest time directory."""
    root = _tree(tmp_path)
    case = root / "case"
    (case / "1000").mkdir()
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    (case / "1000").rmdir()
    assert verify_batch(root, cache=cache).misses == 1
    assert verify_batch(root, cache=cache).misses == 0
    (case / "2000").mkdir()
    assert verify_batch(root, cache=cache).misses == 1


def test_a_symlinked_directory_is_followed_and_its_target_is_in_the_fingerprint(tmp_path):
    root = _tree(tmp_path)
    shared = tmp_path / "shared0"
    shared.mkdir()
    (shared / "a").write_text("1")
    (root / "case" / "linked").symlink_to(shared, target_is_directory=True)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    assert verify_batch(root, cache=cache).misses == 0
    (shared / "k").write_text("new file behind the link")
    assert verify_batch(root, cache=cache).misses == 1
    (root / "case" / "linked").unlink()
    other = tmp_path / "other0"
    other.mkdir()
    (other / "a").write_text("1")
    (root / "case" / "linked").symlink_to(other, target_is_directory=True)     # same shape, new target
    assert verify_batch(root, cache=cache).misses == 1


def test_a_symlink_cycle_terminates_and_is_marked(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "log.icoFoam").write_text("x")
    (case / "loop").symlink_to(case, target_is_directory=True)
    (case / "sub").mkdir()
    (case / "sub" / "up").symlink_to(case / "sub", target_is_directory=True)
    snap = vcache.snapshot(case, None)
    assert any(s.startswith("c:") for s in snap), snap
    assert any(s.startswith("l:loop->") for s in snap)


def test_the_snapshot_lists_directories_sizes_and_modes(tmp_path):
    case = tmp_path / "case"
    (case / "0").mkdir(parents=True)
    (case / "0" / "U").write_text("12345")
    (case / "0" / "U").chmod(0o640)
    snap = vcache.snapshot(case, None)
    assert any(s.startswith("d:0/:") for s in snap), snap
    assert "f:0/U:5:640" in snap, snap


def test_a_permission_change_alone_is_a_miss(tmp_path):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    u = root / "case" / "0.5" / "U"
    u.chmod(0o000)
    try:
        assert verify_batch(root, cache=cache).misses == 1
    finally:
        u.chmod(0o644)
    assert verify_batch(root, cache=cache).misses == 1          # and back again is a miss too


def test_a_size_change_in_a_file_nothing_read_is_a_miss_a_same_size_edit_is_not(tmp_path,
                                                                                monkeypatch):
    """A reader skips a file over a size limit (postProcessing groups): size is read-relevant
    even when the bytes were never hashed."""
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    allrun = root / "case" / "Allrun"
    n = allrun.stat().st_size
    allrun.write_text("#" * n)
    _no_verify(monkeypatch)
    assert verify_batch(root, cache=cache).misses == 0
    monkeypatch.undo()
    allrun.write_text("#" * (n + 1))
    assert verify_batch(root, cache=cache).misses == 1


def test_an_entry_signed_with_another_key_or_not_at_all_is_discarded(tmp_path):
    from courant.verify.cache import _entry_digest

    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    first = verify_batch(root, cache=cache)
    for p in (tmp_path / "cache").glob("*.json"):
        e = json.loads(p.read_text())
        assert len(e["hmac"]) == 64
        e["verdict"]["verdict"] = "reject" if e["verdict"]["verdict"] == "accept" else "accept"
        e["sha256"] = _entry_digest(e)                             # self-consistent
        p.write_text(json.dumps(e))
    again = verify_batch(root, cache=cache)
    assert again.hits == 0 and _bytes(again.results) == _bytes(first.results)


def test_an_entry_from_another_users_key_is_a_miss(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    verify_batch(root, cache=_cache(tmp_path))
    monkeypatch.setenv("COURANT_CACHE_KEY_FILE", str(tmp_path / "other.key"))
    assert verify_batch(root, cache=_cache(tmp_path)).hits == 0


def test_the_secret_lives_outside_the_cache_directory_with_mode_0600(tmp_path):
    verify_batch(_tree(tmp_path), cache=_cache(tmp_path))
    key = Path(os.environ["COURANT_CACHE_KEY_FILE"])
    assert key.is_file() and not key.is_relative_to(tmp_path / "cache")
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and key.stat().st_size == 32


def test_a_key_file_others_can_read_disables_the_cache(tmp_path):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    Path(os.environ["COURANT_CACHE_KEY_FILE"]).chmod(0o644)
    out = verify_batch(root, cache=_cache(tmp_path))
    assert out.hits == 0 and all(e is None for _, _, e in out.results)


def test_a_cache_directory_that_is_not_private_is_refused(tmp_path):
    root = _tree(tmp_path)
    d = tmp_path / "cache"
    d.mkdir(mode=0o755)
    d.chmod(0o755)
    cache = VerdictCache(d)
    assert verify_batch(root, cache=cache).misses == 4
    assert not list(d.glob("*.json"))                              # nothing written
    d.chmod(0o700)
    verify_batch(root, cache=cache)
    assert verify_batch(root, cache=cache).hits == 4
    d.chmod(0o770)
    assert verify_batch(root, cache=cache).hits == 0               # and nothing read


def test_the_cache_directory_is_created_private(tmp_path):
    verify_batch(_tree(tmp_path), cache=_cache(tmp_path))
    assert stat.S_IMODE((tmp_path / "cache").stat().st_mode) == 0o700


def test_the_context_names_the_python_and_the_libraries_the_readers_parse_with(monkeypatch):
    ctx = vcache.context()
    assert set(ctx["dependencies"]) >= {"foamlib", "numpy", "pydantic", "pydantic_core"}
    assert ctx["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}.")
    before = vcache.run_key(ctx, "x")
    monkeypatch.setattr(vcache, "_dist_version", lambda name: "9.9.9")
    assert vcache.run_key(vcache.context(), "x") != before


def test_a_changed_library_version_is_a_miss(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cache = _cache(tmp_path)
    verify_batch(root, cache=cache)
    monkeypatch.setattr(vcache, "_dist_version", lambda name: "0.0.0-changed")
    assert verify_batch(root, cache=cache).misses == 4


def test_the_gate_does_not_cache_unless_asked_and_neither_does_ci(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    cdir = tmp_path / "cdir"
    monkeypatch.setenv("COURANT_CACHE_DIR", str(cdir))
    monkeypatch.delenv("CI", raising=False)
    runner = CliRunner()
    runner.invoke(_app(), ["gate", "--root", str(root), "--paths", "*.log.gz"])
    assert not cdir.exists() or not list(cdir.glob("*.json"))
    runner.invoke(_app(), ["gate", "--root", str(root), "--paths", "*.log.gz", "--cache"])
    assert len(list(cdir.glob("*.json"))) == 3
    shutil.rmtree(cdir)
    monkeypatch.setenv("CI", "true")
    r = runner.invoke(_app(), ["verify", "--batch", str(root)])
    assert "cache:" not in r.output and not cdir.exists()
    r = runner.invoke(_app(), ["verify", "--batch", str(root), "--cache"])
    assert "cache: 0 of 4" in r.output and len(list(cdir.glob("*.json"))) == 4
    monkeypatch.delenv("CI")
    shutil.rmtree(cdir)
    r = runner.invoke(_app(), ["verify", "--batch", str(root)])
    assert "cache: 0 of 4" in r.output
    both = runner.invoke(_app(), ["verify", "--batch", str(root), "--cache", "--no-cache"])
    assert both.exit_code == 4
