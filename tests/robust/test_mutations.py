"""No input makes the verifier raise anything but InputError (TRD TST-9).

A deterministic mutation corpus: every mutant is a function of the seed and of a corpus log,
so a failure here reproduces exactly. Each one must produce a Verdict that serialises, and that
every renderer can show. The same holds for the convergence module, on corrupted monitor files.
CPU class: full (about a hundred small verifications, one process).
"""

from __future__ import annotations

import gzip
import random
import shutil
from pathlib import Path

import pytest

from courant.converge import convergence_path, render_text as conv_text
from courant.verify import InputError, from_path, render_markdown, render_text, verify_inputs
from courant.verify.html import render_html

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"
MONITORED = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__fixed_budget"
LOGS = ("icoFoam_cavity__healthy", "icoFoam_cavity__cfl_blowup", "icoFoam_cavity__missing_dict",
        "pisoFoam_RAS_cavity__healthy", "simpleFoam_pitzDaily__healthy",
        "rhoPimpleFoam_RAS_cavity__cfl_blowup")


def _mutants(text: str, seed: int):
    rng = random.Random(seed)
    b = text.encode()
    n = len(b)
    for k in (1, 3, 5, 7, 9):
        yield f"truncate_{k}0pct", b[: n * k // 10]
    yield "crlf", text.replace("\n", "\r\n").encode()
    noisy = bytearray(b)
    for _ in range(20):
        noisy.insert(rng.randrange(len(noisy)), rng.choice([0xff, 0xfe, 0xc0, 0x80]))
    yield "invalid_utf8", bytes(noisy)
    yield "megabyte_line", b + b"x" * (1 << 20)
    mid = text[n // 3: 2 * n // 3]
    yield "duplicated_block", (text[: 2 * n // 3] + mid + text[2 * n // 3:]).encode()
    yield "reversed_lines", "\n".join(reversed(text.splitlines())).encode()
    yield "nul_bytes", b.replace(b"\n", b"\x00\n", 50)
    yield "nan_residuals", text.replace("Initial residual = ", "Initial residual = nan", 3).encode()
    yield "huge_numbers", text.replace("Initial residual = ", "Initial residual = 1e+309", 3).encode()
    yield "binary_noise", bytes(rng.randrange(256) for _ in range(4096))
    yield "empty", b""


def _check(path: Path, case_json: Path | None = None) -> str:
    try:
        inputs = from_path(path, case_json)
    except InputError:
        return "input_error"
    v = verify_inputs(inputs)
    v.to_json()
    render_text(v)
    render_markdown(v)
    render_html(v, inputs)
    return v.verdict


@pytest.mark.parametrize("name", LOGS)
def test_no_mutated_log_makes_the_verifier_raise(tmp_path, name):
    text = gzip.open(CORPUS / f"{name}.log.gz", "rt").read()
    seen = {}
    for label, data in _mutants(text, seed=LOGS.index(name)):
        p = tmp_path / f"{label}.log"
        p.write_bytes(data)
        seen[label] = _check(p)
    assert set(seen.values()) <= {"accept", "reject", "insufficient", "input_error"}
    assert seen["empty"] in ("insufficient", "input_error")
    assert seen["binary_noise"] in ("insufficient", "input_error")


MONITOR_MUTANTS = {
    "header_only": lambda t: t.split("\n", 6)[0] + "\n",
    "truncated_row": lambda t: t[: len(t) // 2],
    "garbage_rows": lambda t: t.replace("\n2", "\nnot a number 2", 50),
    "nan_values": lambda t: t.replace("e-", "nan e-", 30),
    "empty": lambda t: "",
    "binary": lambda t: "\x00\xff" * 512,
    "no_header": lambda t: "\n".join(ln for ln in t.splitlines() if not ln.startswith("#")),
}


@pytest.mark.parametrize("label", sorted(MONITOR_MUTANTS))
def test_no_corrupted_monitor_makes_convergence_raise(tmp_path, label):
    case = tmp_path / "case"
    shutil.copytree(MONITORED, case)
    f = case / "postProcessing" / "courantWallForces" / "0" / "force.dat"
    f.write_text(MONITOR_MUTANTS[label](f.read_text()), errors="replace")
    conv = convergence_path(case)
    assert conv.status in ("PASS", "WARN", "FAIL", "NOT_ASSESSABLE")
    conv_text(conv)
    assert _check(case) in ("accept", "reject", "insufficient")
