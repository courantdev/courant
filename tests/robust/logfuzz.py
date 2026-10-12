"""A deterministic mutation fuzzer over every committed real solver log (TRD TST-9 at scale).

Every mutant is a pure function of (log path, index): `mutant(log, k)` rebuilds mutant k of a
log without generating the others, so a failure reproduces from its id alone, and the fast
sample is a strict subset of the full set.

The corpus is every git-tracked OpenFOAM *solver* log outside the synthetic `benchmark/`: the
certification corpus, the monitored runs, the packaged demo logs and the real fixtures, read
from disk by a fixed list of globs and de-duplicated by content. Mesh-utility logs (`blockMesh`,
`checkMesh`) are not solver logs and are left out; the operators below make logs like them.

Two families of mutation:
- `RANDOM_OPS`, chained one to three at a time from a seeded generator: truncation, byte flips,
  duplicated and missing lines, NaN and overflow injection into residual lines, interleaving
  with another log, encoding damage, shuffled windows, a lost head.
- `DESTROYERS`, deterministic, one per log: each removes or poisons every residual line or
  every time marker. The verifier must never accept their output.

`destroyed(data)` decides, from the mutated bytes alone, whether a mutant's residuals or time
markers were destroyed: no residual line with finite values is left, no line starts with a
`Time = ` marker (lines split as `str.splitlines` splits them, so a log whose line endings
changed keeps its markers), or a residual value is NaN or infinite. It is deliberately
lenient: it claims destruction only where the evidence is plainly gone.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import random
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Where the committed real solver logs are. Globs, not a list of names, so a log added to the
#: corpus is fuzzed without editing this file.
GLOBS = ("certification/corpus/*.log.gz", "certification/monitored/*/log.*",
         "courant/data/demo/*.log.gz", "tests/fixtures/real_openfoam/*.log")
#: Mesh utilities write logs that are not solver logs.
NOT_SOLVER = re.compile(r"blockMesh|checkMesh|snappyHexMesh", re.IGNORECASE)

#: At least this many mutants over the corpus; the fast sample takes every SAMPLE_STRIDE-th.
TOTAL = 20_000
SAMPLE_STRIDE = 40
#: A log above this size costs about a second a verification; it gets LARGE_COUNT mutants and
#: the small logs share the rest, so the full run stays under half an hour on one core.
LARGE_BYTES = 256 * 1024
LARGE_COUNT = 40

_NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|-?nan|inf|-?inf|NaN|Inf)"
RESIDUAL_LINE = re.compile(
    rf"Solving for [^,\n]+, Initial residual = ({_NUM}), Final residual = ({_NUM}), "
    rf"No Iterations \d+")
#: A time marker at the start of a line, however lines end (`str.splitlines` boundaries).
TIME_MARKER = re.compile(r"Time = [-+]?\.?\d")


# ---------------------------------------------------------------- the corpus
def _read(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


@lru_cache(maxsize=1)
def corpus() -> tuple[str, ...]:
    """Relative paths of the real solver logs, sorted, one per distinct content."""
    seen: set[str] = set()
    out = []
    for pattern in GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            rel = path.relative_to(ROOT).as_posix()
            if NOT_SOLVER.search(path.name) or not path.is_file():
                continue
            digest = hashlib.sha256(_read(path)).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            out.append(rel)
    return tuple(sorted(out))


@lru_cache(maxsize=8)
def source(rel: str) -> bytes:
    return _read(ROOT / rel)


@lru_cache(maxsize=1)
def counts() -> dict[str, int]:
    """Mutants per log: LARGE_COUNT for a large log, the rest shared by the small ones."""
    logs = corpus()
    large = [r for r in logs if (ROOT / r).stat().st_size > LARGE_BYTES
             or len(source(r)) > LARGE_BYTES]
    small = [r for r in logs if r not in large]
    per_small = -(-(TOTAL - LARGE_COUNT * len(large)) // max(1, len(small)))
    return {r: (LARGE_COUNT if r in large else per_small) for r in logs}


def plan() -> list[tuple[str, int]]:
    """Every (log, index) of the full run, in a fixed order."""
    return [(rel, k) for rel in corpus() for k in range(counts()[rel])]


def sample() -> list[tuple[str, int]]:
    """The fast sample: every SAMPLE_STRIDE-th mutant of the full plan."""
    return plan()[::SAMPLE_STRIDE]


# ---------------------------------------------------------------- random operators
def _lines(b: bytes) -> list[bytes]:
    return b.splitlines(keepends=True)


def op_truncate(b: bytes, rng: random.Random) -> bytes:
    return b[: rng.randrange(len(b) + 1)]


def op_truncate_line(b: bytes, rng: random.Random) -> bytes:
    lines = _lines(b)
    return b"".join(lines[: rng.randrange(len(lines) + 1)])


def op_lose_head(b: bytes, rng: random.Random) -> bytes:
    return b[rng.randrange(len(b) + 1):]


def op_byte_flips(b: bytes, rng: random.Random) -> bytes:
    if not b:
        return b
    out = bytearray(b)
    for _ in range(rng.choice((1, 2, 4, 8, 16, 64, 256))):
        out[rng.randrange(len(out))] ^= rng.randrange(1, 256)
    return bytes(out)


def op_duplicate_lines(b: bytes, rng: random.Random) -> bytes:
    lines = _lines(b)
    if not lines:
        return b
    i = rng.randrange(len(lines))
    block = lines[i: i + rng.randint(1, 200)] * rng.choice((1, 2, 10))
    j = rng.randrange(len(lines) + 1)
    return b"".join(lines[:j] + block + lines[j:])


def op_drop_lines(b: bytes, rng: random.Random) -> bytes:
    p = rng.choice((0.001, 0.01, 0.05, 0.2, 0.5))
    return b"".join(ln for ln in _lines(b) if rng.random() >= p)


def op_drop_block(b: bytes, rng: random.Random) -> bytes:
    lines = _lines(b)
    if not lines:
        return b
    i = rng.randrange(len(lines))
    return b"".join(lines[:i] + lines[i + rng.randint(1, max(1, len(lines) // 4)):])


_POISON = (b"nan", b"-nan", b"NaN", b"inf", b"-inf", b"1e+309", b"1e+300")


def op_nan_inject(b: bytes, rng: random.Random) -> bytes:
    lines = _lines(b)
    hits = [i for i, ln in enumerate(lines) if b"Initial residual = " in ln]
    if not hits:
        return b
    for i in rng.sample(hits, min(len(hits), rng.choice((1, 3, 10, 100)))):
        key = rng.choice((b"Initial residual = ", b"Final residual = "))
        ln = lines[i]
        at = ln.find(key)
        if at < 0:
            continue
        start = at + len(key)
        end = start
        while end < len(ln) and ln[end:end + 1] not in (b",", b"\n", b" "):
            end += 1
        lines[i] = ln[:start] + rng.choice(_POISON) + ln[end:]
    return b"".join(lines)


def op_interleave(b: bytes, rng: random.Random) -> bytes:
    donors = [r for r in corpus() if counts()[r] != LARGE_COUNT]
    donor = _lines(source(rng.choice(donors)))
    lines = _lines(b)
    if not donor or not lines:
        return b + b"".join(donor)
    d0 = rng.randrange(len(donor))
    window = donor[d0: d0 + rng.randint(1, 300)]
    at = rng.randrange(len(lines) + 1)
    mixed = []
    for i, ln in enumerate(window):
        mixed.extend(lines[at + i: at + i + 1])
        mixed.append(ln)
    return b"".join(lines[:at] + mixed + lines[at + len(window):])


def op_encoding(b: bytes, rng: random.Random) -> bytes:
    kind = rng.randrange(8)
    if kind == 0:                                   # invalid UTF-8 bytes
        out = bytearray(b)
        for _ in range(rng.choice((1, 20, 200))):
            out.insert(rng.randrange(len(out) + 1), rng.choice((0xFF, 0xFE, 0xC0, 0x80, 0xED)))
        return bytes(out)
    if kind == 1:
        return b"\xef\xbb\xbf" + b                  # a byte-order mark
    if kind == 2:
        return b.decode("utf-8", "replace").encode("utf-16")
    if kind == 3:
        return b.replace(b"\n", b"\r\n")
    if kind == 4:
        return b.replace(b"\n", b"\r")              # old Mac line endings
    if kind == 5:
        return b.replace(b"\n", b"\x00\n", rng.choice((1, 50, 10_000)))
    if kind == 6:
        return b.decode("utf-8", "replace").encode("latin-1", "replace") + "é".encode("latin-1")
    return b + b"x" * (1 << 20)                     # a one-megabyte line


def op_shuffle_window(b: bytes, rng: random.Random) -> bytes:
    lines = _lines(b)
    if len(lines) < 2:
        return b
    i = rng.randrange(len(lines))
    window = lines[i: i + rng.randint(2, 200)]
    rng.shuffle(window)
    return b"".join(lines[:i] + window + lines[i + len(window):])


RANDOM_OPS = {
    "truncate": op_truncate, "truncate_line": op_truncate_line, "lose_head": op_lose_head,
    "byte_flips": op_byte_flips, "duplicate_lines": op_duplicate_lines,
    "drop_lines": op_drop_lines, "drop_block": op_drop_block, "nan_inject": op_nan_inject,
    "interleave": op_interleave, "encoding": op_encoding, "shuffle_window": op_shuffle_window,
}


def operators(rel: str, k: int) -> tuple[random.Random, list[str]]:
    """The seeded generator of mutant k of a log, and the one to three operators it applies."""
    rng = random.Random(f"tst9-fuzz:{rel}:{k}")
    return rng, [rng.choice(sorted(RANDOM_OPS)) for _ in range(rng.choice((1, 1, 2, 3)))]


def mutant(rel: str, k: int) -> tuple[str, bytes]:
    """(id, bytes) of mutant k of a log."""
    rng, names = operators(rel, k)
    data = source(rel)
    for name in names:
        data = RANDOM_OPS[name](data, rng)
    return f"{rel}#{k}:{'+'.join(names)}", data


# ---------------------------------------------------------------- deterministic destroyers
def _keep(b: bytes, drop) -> bytes:
    return b"".join(ln for ln in _lines(b) if not drop(ln))


def _is_residual(ln: bytes) -> bool:
    return b"Initial residual = " in ln


def _is_time(ln: bytes) -> bool:
    return ln.startswith(b"Time = ")


def _cut_before_first_residual(b: bytes) -> bytes:
    lines = _lines(b)
    for i, ln in enumerate(lines):
        if _is_residual(ln):
            return b"".join(lines[:i])
    return b


def _poison_all(b: bytes) -> bytes:
    return re.sub(rb"(Initial residual = |Final residual = )[^,\s]+", rb"\1nan", b)


DESTROYERS = {
    "strip_residual_lines": lambda b: _keep(b, _is_residual),
    "strip_time_markers": lambda b: _keep(b, _is_time),
    "strip_residuals_and_time": lambda b: _keep(b, lambda ln: _is_residual(ln) or _is_time(ln)),
    "cut_before_first_residual": _cut_before_first_residual,
    "nan_every_residual": _poison_all,
    "garble_residual_keyword": lambda b: b.replace(b"Initial residual", b"Initial resid\xffual"),
}


# ---------------------------------------------------------------- what was destroyed
def destroyed(data: bytes) -> str | None:
    """Why a mutant's residuals or time markers count as destroyed, or None when they do not."""
    text = data.decode("utf-8", "replace")
    finite = False
    for m in RESIDUAL_LINE.finditer(text):
        values = [float(x) for x in m.groups()]
        if any(not math.isfinite(v) for v in values):
            return "a residual is NaN or infinite"
        finite = True
    if not finite:
        return "no residual line is left"
    if not any(TIME_MARKER.match(ln) for ln in text.splitlines()):
        return "no time marker is left"
    return None
