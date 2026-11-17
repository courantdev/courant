"""Invariant 6, the inputs that are not case files: a solver log (`.log.gz` too), its sidecar
(`.case.json`) and a specification (`courant.spec/1`). Each, mutated by `readerfuzz.OPS` (byte
flips, truncation, deep nesting, huge numbers, unicode, blocks, emptiness) and, for the JSON
ones, by structural edits, is read or refused with `InputError` (TST-9, GATE-10), never anything
else. Reading them is cheap, so the readers run on thousands; the verdict on a sample.
"""

from __future__ import annotations

import gzip
import json
import random

import pytest

from courant.verify import InputError, from_log, verify_inputs, with_spec
from tests.invariants import gen, readerfuzz

N_FAST, N_SLOW = 600, 12_000
N_VERDICT_FAST, N_VERDICT_SLOW = 30, 1500


def _json_edit(data: bytes, rng: random.Random) -> bytes:
    """A structural edit of a JSON document: a value replaced by another JSON type, a huge or
    non-finite number, a deep nest, or a key dropped."""
    try:
        doc = json.loads(data)
    except ValueError:
        return data
    paths = []

    def walk(node, path):
        paths.append(path)
        if isinstance(node, dict):
            for k in node:
                walk(node[k], (*path, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, (*path, i))

    walk(doc, ())
    path = rng.choice(paths)
    new = rng.choice([None, [], {}, "", 0, -1, 1e308, 10 ** 400, "x" * 10_000, True,
                      [[[[[[[[[[1]]]]]]]]]], {"a": {"b": {"c": {}}}}])
    if not path:
        doc = new
    else:
        parent = doc
        for p in path[:-1]:
            parent = parent[p]
        parent[path[-1]] = new
    text = json.dumps(doc)
    if rng.random() < 0.2:
        text = "[" * 100_000 + "]" * 100_000            # deeper than json's recursion limit
    return text.encode()


def _mutate(data: bytes, rng: random.Random, as_json: bool) -> tuple[bytes, list[str]]:
    names = rng.sample(sorted(readerfuzz.OPS), rng.choice([1, 1, 2]))
    if as_json and rng.random() < 0.5:
        return _json_edit(data, rng), ["json"]
    for n in names:
        data = readerfuzz.OPS[n](data, rng)
    return data, names


def _case(k: int, tmp):
    """(id, log path, sidecar path, spec path) of input mutant k: one of the three mutated."""
    rng = random.Random(f"invariants:inputs:{k}")
    log_rel, side_rel = rng.choice(gen.corpus())
    log = gzip.decompress((gen.ROOT / log_rel).read_bytes())
    side = (gen.ROOT / side_rel).read_bytes()
    rel = rng.choice(gen.CASE_DIRS)
    spec = json.dumps({"schema": "courant.spec/1", "items": list(gen.drafted(rel, "default"))
                       or [{"name": "nu", "file": "constant/transportProperties", "key": "nu",
                            "value": 1e-5}]}).encode()
    which = rng.choice(["log", "log.gz", "sidecar", "spec"])
    if which == "log":
        log, ops = _mutate(log, rng, False)
    elif which == "log.gz":
        z = gzip.compress(log, mtime=0)
        z, ops = _mutate(z, rng, False)
    elif which == "sidecar":
        side, ops = _mutate(side, rng, True)
    else:
        spec, ops = _mutate(spec, rng, True)
    stem = log_rel.split("/")[-1][: -len(".log.gz")]
    lp = tmp / (f"{stem}.log.gz" if which == "log.gz" else f"{stem}.log")
    lp.write_bytes(z if which == "log.gz" else log)
    sp, cp = tmp / f"{stem}.case.json", tmp / "spec.json"
    sp.write_bytes(side)
    cp.write_bytes(spec)
    return f"inputs[{k}]:{stem}:{which}+{','.join(ops)}", lp, sp, cp


def _read(k: int, tmp, verdict: bool):
    """None if the input was read or refused with InputError; else what escaped."""
    mid, lp, sp, cp = _case(k, tmp)
    try:
        inputs = from_log(lp, sp)
        inputs = with_spec(inputs, cp)
        if verdict:
            v = verify_inputs(inputs)
            v.to_json()
    except InputError:
        return None
    except Exception as exc:                                      # noqa: BLE001
        return f"{mid}: {type(exc).__name__}: {str(exc)[:120]}"
    return None


def _run(n: int, tmp_path, verdict: bool):
    bad = []
    for k in range(n):
        d = tmp_path / str(k % 4)
        d.mkdir(exist_ok=True)
        for p in d.iterdir():
            p.unlink()
        why = _read(k, d, verdict)
        if why:
            bad.append(why)
    assert not bad, f"{len(bad)} of {n} inputs escaped as other than InputError:\n" + \
        "\n".join(bad[:20])


def test_inputs_are_read_or_refused(tmp_path):
    _run(N_FAST, tmp_path, verdict=False)


@pytest.mark.slow
def test_inputs_are_read_or_refused_at_scale(tmp_path):
    _run(N_SLOW, tmp_path, verdict=False)


def test_mutated_inputs_get_a_verdict_or_an_input_error(tmp_path):
    _run(N_VERDICT_FAST, tmp_path, verdict=True)


@pytest.mark.slow
def test_mutated_inputs_get_a_verdict_or_an_input_error_at_scale(tmp_path):
    _run(N_VERDICT_SLOW, tmp_path, verdict=True)
