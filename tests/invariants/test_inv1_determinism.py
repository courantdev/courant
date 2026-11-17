"""Invariant 1: the same case and specification give the same verdict bytes.

Three readings of "the same":

- twice in one process, on inputs rebuilt from the same bytes (`courant.verdict/2` and `/1`
  JSON, byte for byte);
- in two processes with different `PYTHONHASHSEED`s (set and dict iteration order is the
  classic source of a nondeterministic document);
- one case directory at two different absolute paths under the same name, with its files'
  modification times changed (a verdict must not depend on where or when it was read).

The fast run is a prefix of the slow one (`gen.log_case`, `gen.dir_case`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

from courant.verify import InputError, from_case_dir, from_log, verify_inputs, with_spec
from tests.invariants import gen

N_LOG_FAST, N_LOG_SLOW = 40, 3000
N_DIR_FAST, N_DIR_SLOW = 30, 1000
N_SEED_FAST, N_SEED_SLOW = 20, 1000


def _bytes(inputs) -> tuple[bytes, bytes]:
    v = verify_inputs(inputs)
    return v.to_json().encode(), v.to_v1().to_json().encode()


def _log_verdicts(k: int):
    cid, log, side = gen.log_case(k)
    try:
        first = _bytes(from_log(log, side))
    except InputError:
        return cid, None, None
    second = _bytes(from_log(log, side))
    return cid, first, second


def _run_logs(n: int):
    bad = []
    for k in range(n):
        cid, a, b = _log_verdicts(k)
        if a != b:
            bad.append(cid)
    assert not bad, f"{len(bad)} of {n} generated runs gave different bytes: {bad[:10]}"


def test_log_runs_give_the_same_bytes_twice():
    _run_logs(N_LOG_FAST)


@pytest.mark.slow
def test_log_runs_give_the_same_bytes_twice_at_scale():
    _run_logs(N_LOG_SLOW)


def _dir_bytes(case, spec_k: int | None, rel: str):
    inputs = from_case_dir(case)
    if spec_k is not None:
        _, data = gen.spec_for(spec_k, rel)
        path, sha = gen.write_spec(data, spec_k)
        inputs = with_spec(inputs, path, gen.lock_for(spec_k, sha))
    return _bytes(inputs)


def _run_dirs(n: int, tmp_path):
    bad = []
    for k in range(n):
        rel, case = gen.dir_case(k)
        spec_k = k if k % 3 else None
        # The same case at another absolute path, same name, other modification times.
        other = tmp_path / f"elsewhere-{k}" / case.name
        shutil.copytree(case, other, symlinks=True)
        for p in other.rglob("*"):
            if p.is_file():
                os.utime(p, (1_000_000 + k, 1_000_000 + k))
        a = _dir_bytes(case, spec_k, rel)
        b = _dir_bytes(case, spec_k, rel)
        c = _dir_bytes(other, spec_k, rel)
        if not (a == b == c):
            where = "same path" if a != b else "another path"
            bad.append(f"dir[{k}]:{rel} spec={spec_k} ({where})")
        shutil.rmtree(other.parent)
    assert not bad, f"{len(bad)} of {n} case directories gave different bytes: {bad[:10]}"


def test_case_directories_give_the_same_bytes_anywhere(tmp_path):
    _run_dirs(N_DIR_FAST, tmp_path)


@pytest.mark.slow
def test_case_directories_give_the_same_bytes_anywhere_at_scale(tmp_path):
    _run_dirs(N_DIR_SLOW, tmp_path)


_CHILD = r"""
import hashlib, json, sys
from courant.verify import InputError, from_log, verify_inputs
from tests.invariants import gen
out = {}
for k in range(int(sys.argv[1])):
    cid, log, side = gen.log_case(k)
    try:
        v = verify_inputs(from_log(log, side))
    except InputError:
        out[cid] = "input_error"
        continue
    out[cid] = hashlib.sha256(v.to_json().encode()).hexdigest()
print(json.dumps(out, sort_keys=True))
"""


def _digests(n: int, seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    res = subprocess.run([sys.executable, "-c", _CHILD, str(n)], cwd=gen.ROOT, env=env,
                         capture_output=True, text=True, timeout=1800, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def _run_seeds(n: int):
    a, b = _digests(n, "1"), _digests(n, "4242")
    assert len(a) == n
    differ = sorted(k for k in a if a[k] != b.get(k))
    assert not differ, f"{len(differ)} of {n} runs differ across hash seeds: {differ[:10]}"


def test_verdict_bytes_do_not_depend_on_the_hash_seed():
    _run_seeds(N_SEED_FAST)


@pytest.mark.slow
def test_verdict_bytes_do_not_depend_on_the_hash_seed_at_scale():
    _run_seeds(N_SEED_SLOW)


def test_the_generator_is_a_pure_function_of_the_index():
    """A failure reproduces from its id: case k is the same bytes however it is reached."""
    a = [gen.log_case(k) for k in (3, 7)]
    da = [hashlib.sha256(p.read_bytes()).hexdigest() for _, p, _ in a]
    b = [gen.log_case(k) for k in (7, 3)][::-1]
    db = [hashlib.sha256(p.read_bytes()).hexdigest() for _, p, _ in b]
    assert [i for i, _, _ in a] == [i for i, _, _ in b] and da == db
    assert gen.spec_for(5, gen.CASE_DIRS[0]) == gen.spec_for(5, gen.CASE_DIRS[0])
