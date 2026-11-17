"""Invariant 6's mutants and their runner: every reader refuses or reads, never crashes.

A mutant is a pure function of its index: `mutant(k)` picks a fixture case
(`gen.CASE_DIRS`), one of its real dictionaries or field files (`files(rel)`, `.gz` read as the
solver reads it), and applies one or two operators: byte flips, truncation, deep nesting,
huge numbers, unicode, a corrupted gzip stream, an empty file, a duplicated or dropped block.

Run as a module, it verifies mutants `start..stop` in one process and prints one JSON line per
mutant, `{"k": k, "start": id}` before and `{"k": k, "ok": true}` or `{"k": k, "crash": ...}`
after, flushed, so the test (`test_inv6_readers.py`) can tell a hang (a start with no end within
the budget) from a crash. Two lanes:

- `readers`: the readers alone, each against its own contract: `CaseConfig.from_foam` records
  what it cannot read; `read_mesh_quality`, `forks.read`, `deck.read`, `check_conformance` and
  `draft_spec` never raise; `read_fields` may raise only what `pipeline._fields` sets aside, and
  says a field it could not read (`FieldData.unread`) rather than reading nonsense silently;
- `verify`: `from_case_dir` then `verify_inputs`, both `--checks` modes, with the case's own
  draft specification: a verdict, or `InputError` (TST-9, GATE-10), and nothing else.
"""

from __future__ import annotations

import gzip
import json
import random
import shutil
import sys
import traceback
from functools import cache
from pathlib import Path

from tests.invariants import gen

#: Files under these are fuzzed; `processor*` and `postProcessing` are left out.
_TOP = ("system", "constant", "0")


@cache
def files(rel: str) -> tuple[str, ...]:
    """The case's dictionaries and field files, relative, sorted: system/, constant/ (with
    polyMesh), 0/ and every time directory."""
    root = gen.pristine(rel)
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("log."):
            continue
        parts = p.relative_to(root).parts
        top = parts[0]
        if top in _TOP or _is_time(top):
            out.append(str(p.relative_to(root)))
    return tuple(out)


def _is_time(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


#: RTL override, combining acute, emoji, e acute, a CJK ideograph, BOM, line separator, NUL,
#: the last code point before the surrogates, an Arabic-Indic digit.
_UNICODE = tuple(chr(c) for c in (0x202E, 0x0301, 0x1F600, 0x00E9, 0x4E2D, 0xFEFF, 0x2028,
                                  0x0000, 0xD7FF, 0x0663))


def op_flip(b: bytes, rng):
    if not b:
        return b"\xff"
    a = bytearray(b)
    for _ in range(rng.randint(1, 8)):
        i = rng.randrange(len(a))
        a[i] = rng.randrange(256)
    return bytes(a)


def op_truncate(b: bytes, rng):
    return b[: rng.randrange(len(b) + 1)] if b else b


def op_nest(b: bytes, rng):
    depth = rng.choice([50, 500, 5000, 50_000])
    opener = rng.choice([b"{ a ", b"(", b"( (", b"sub { "])
    closer = {b"{ a ": b"}", b"(": b")", b"( (": b"))", b"sub { ": b"}"}[opener]
    i = rng.randrange(len(b) + 1)
    tail = closer * depth if rng.random() < 0.5 else b""
    return b[:i] + b"\nx " + opener * depth + tail + b";\n" + b[i:]


_NUM = __import__("re").compile(rb"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def op_huge(b: bytes, rng):
    nums = list(_NUM.finditer(b))
    if not nums:
        return b + b"\nx 1e999999;\n"
    m = rng.choice(nums)
    new = rng.choice([b"1e999999", b"-1e999999", b"9" * 5000, b"1" + b"0" * 400 + b".5",
                      b"nan", b"-nan", b"inf", b"1e-999999", b"0x1F", b"1_000",
                      b"4.9e-324", b"1.7976931348623157e309"])
    return b[: m.start()] + new + b[m.end():]


def op_unicode(b: bytes, rng):
    s = b.decode("utf-8", errors="surrogateescape")
    for _ in range(rng.randint(1, 5)):
        i = rng.randrange(len(s) + 1)
        s = s[:i] + rng.choice(_UNICODE) * rng.choice([1, 3, 100]) + s[i:]
    out = s.encode("utf-8", errors="surrogateescape")
    if rng.random() < 0.3:                                   # invalid UTF-8 as well
        i = rng.randrange(len(out) + 1)
        out = out[:i] + rng.choice([b"\xc3", b"\xff\xfe", b"\xed\xa0\x80", b"\x80"]) + out[i:]
    return out


def op_block(b: bytes, rng):
    lines = b.splitlines(keepends=True)
    if len(lines) < 2:
        return b + b
    i, j = sorted(rng.sample(range(len(lines) + 1), 2))
    if rng.random() < 0.5:
        return b"".join(lines[:i] + lines[j:])               # drop a block
    return b"".join(lines[:j] + lines[i:j] + lines[j:])      # duplicate it


def op_empty(b: bytes, rng):
    return rng.choice([b"", b"\n", b"FoamFile\n{\n}\n", b"\x00" * 64])


OPS = {"flip": op_flip, "truncate": op_truncate, "nest": op_nest, "huge": op_huge,
       "unicode": op_unicode, "block": op_block, "empty": op_empty}


def mutant(k: int) -> tuple[str, str, str, bytes]:
    """(id, case rel, file rel as written, bytes) of mutant k."""
    rng = random.Random(f"invariants:reader:{k}")
    rel = rng.choice(gen.CASE_DIRS)
    target = rng.choice(files(rel))
    raw = (gen.pristine(rel) / target).read_bytes()
    gz = target.endswith(".gz")
    data = gzip.decompress(raw) if gz else raw
    names = rng.sample(sorted(OPS), rng.choice([1, 1, 2]))
    for name in names:
        data = OPS[name](data, rng)
    written = target
    if gz:
        r = rng.random()
        if r < 0.4:
            data = gzip.compress(data, mtime=0)
        elif r < 0.55:                                   # a corrupted gzip stream
            data = op_flip(gzip.compress(data, mtime=0)[:max(1, len(data) // 2)], rng)
            names.append("gzip-corrupt")
        else:                                            # written plain, as foamlib also reads
            written = target[: -len(".gz")]
    return f"reader[{k}]:{rel}:{written}+{','.join(names)}", rel, written, data


# --- the runner -------------------------------------------------------------------------------

def _work(rel: str) -> Path:
    """A working copy of the case, made once per process."""
    return _copies(rel)


@cache
def _copies(rel: str) -> Path:
    dst = gen.scratch() / "fuzz" / rel.replace("/", "__") / Path(rel).name
    shutil.copytree(gen.pristine(rel), dst, symlinks=True)
    return dst


def _apply(case: Path, rel: str, written: str, data: bytes):
    """Write the mutant into the working copy; returns how to restore it."""
    target = case / written
    original = gen.pristine(rel) / written
    gz_twin = None
    if not written.endswith(".gz") and (case / (written + ".gz")).exists():
        gz_twin = case / (written + ".gz")                 # the plain file replaces the .gz
        gz_twin.rename(case / (written + ".gz.held"))
    target.write_bytes(data)

    def restore():
        if original.exists():
            shutil.copyfile(original, target)
        else:
            target.unlink(missing_ok=True)
        if gz_twin is not None:
            (case / (written + ".gz.held")).rename(gz_twin)
    return restore


def _readers(case: Path, rel: str) -> None:
    from courant import forks
    from courant.case.config import CaseConfig
    from courant.case.mesh import read_mesh_quality
    from courant.spec import deck
    from courant.spec.check import check_conformance, draft_spec
    from courant.spec.model import SpecItem
    from courant.verification._fluid_reader import read_fields

    for flag in (False, True):
        CaseConfig.from_foam(str(case), buoyancy=flag, directives=flag, steady_ddt=flag,
                             zero_unlisted=flag, regions=flag, solver=None)
        read_mesh_quality(str(case), log_text="", polymesh=flag)
    forks.read("", str(case))
    for mode in ("default", "next"):
        items = gen.drafted(rel, mode)
        for item in items:
            deck.read(case, SpecItem.model_validate(item), macros=mode == "next")
        spec = {"schema": "courant.spec/1", "items": list(items)}
        check_conformance(spec, case, spec_sha256="0" * 64, checks=mode)
        draft_spec(case, checks=mode)
    try:
        read_fields(str(case), fluxes=True)
    except Exception:                                      # noqa: BLE001, S110
        pass                     # `pipeline._fields` sets it aside, and says no fields were read


def _verify(case: Path, rel: str) -> None:
    from courant.verify import InputError, from_case_dir, verify_inputs, with_spec
    from courant.verify.render import render_markdown, render_text

    for mode in ("default", "next"):
        items = gen.drafted(rel, mode)
        spec_path = gen.scratch() / "fuzz" / f"spec-{rel.replace('/', '__')}-{mode}.json"
        if not spec_path.exists():
            spec_path.write_text(json.dumps({"schema": "courant.spec/1", "items": list(items)}))
        try:
            inputs = from_case_dir(case, checks=mode)
        except InputError:
            continue
        for with_ in (False, True):
            try:
                v = verify_inputs(with_spec(inputs, spec_path) if with_ else inputs)
            except InputError:
                continue
            v.to_json()
            render_text(v)
            render_markdown(v)


def _with_the_guard() -> None:
    """Every reader in this process asks the guard, in both modes: a mutant foamlib hangs on would
    otherwise hang this lane under the default, where A6-1 is open until T-FOAMGUARD-2. So the lane
    holds each mode's readers on what the guard leaves them. Its budget of seconds per mutant is a
    test's: the guard's bounds of a hang, about ten minutes, are lowered to fit it."""
    from courant.case import foamguard
    from courant.verify import checks

    checks.NEXT.pop(checks.FOAM_GUARD)
    foamguard._ASKED.set(True)
    foamguard.MAX_MISMATCHED = 512
    foamguard.MAX_EXPONENT_READ = 24


def main(argv: list[str]) -> int:
    _with_the_guard()
    lane, start, stop = argv[0], int(argv[1]), int(argv[2])
    run = _readers if lane == "readers" else _verify
    for k in range(start, stop):
        mid, rel, written, data = mutant(k)
        print(json.dumps({"k": k, "start": mid}), flush=True)
        case = _work(rel)
        restore = _apply(case, rel, written, data)
        try:
            run(case, rel)
            print(json.dumps({"k": k, "ok": True}), flush=True)
        except RecursionError as exc:
            print(json.dumps({"k": k, "crash": f"RecursionError: {str(exc)[:100]}",
                              "where": _where(exc)}), flush=True)
        except Exception as exc:                            # noqa: BLE001
            print(json.dumps({"k": k, "crash": f"{type(exc).__name__}: {str(exc)[:160]}",
                              "where": _where(exc)}), flush=True)
        finally:
            restore()
    return 0


def _where(exc: BaseException) -> str:
    """The innermost frame in courant/ that the exception passed through."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "/courant/" in f.filename
              and "/tests/" not in f.filename]
    f = frames[-1] if frames else traceback.extract_tb(exc.__traceback__)[-1]
    return f"{Path(f.filename).name}:{f.lineno} {f.name}"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
