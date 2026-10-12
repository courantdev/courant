"""TST-14: the deck reader under property-based fuzzing.

Generated OpenFOAM dictionaries (macros, `#include`s, regex-grouped patches, per-face lists,
comments, several entries on one line, decoy entries in nested blocks) with a known value in a
random shape, and random corruptions of them. Three properties:

1. Oracle: on a generated deck, the reader reads exactly the value the generator wrote.
2. It never raises: not `read`, which guards itself, and not the reader underneath it either.
3. It reads the same thing twice.

`hypothesis` is not installed in this environment and the build is offline, so the generator is a
seeded `random.Random`: every case is reproducible from its seed, printed on failure. The
`foamDictionary -expand` oracle in the pinned solver image is a `full` check (TRD TST-14), run at
milestones, not here.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from courant.spec import SpecItem, deck

HEAD = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n}\n"
N_ORACLE = 400
N_CORRUPT = 400


def _num(rng: random.Random) -> float:
    return float(f"{rng.uniform(-50, 50) * 10 ** rng.randint(-6, 3):.4g}")


def _fmt(x: float) -> str:
    return f"{x:g}"


def _comment(rng: random.Random) -> str:
    return rng.choice(["", "", "// a comment; with { braces }\n", "/* nu 99; value 7; */ ",
                       "/* multi\n   line { */\n"])


#: Shapes whose value is a directive: the reader must report it unreadable, never a number.
DIRECTIVES = ("calc", "calc_braced", "calc_macro", "eval")


def _scalar_entry(rng: random.Random, key: str, x: float, macros: dict, inc: dict) -> str:
    """`key <x>;` in one of OpenFOAM's shapes; macro definitions go to `macros` or `inc`. A
    directive shape (`DIRECTIVES`) is marked by `macros["__directive__"]`."""
    shape = rng.choice(["plain", "dims", "old", "macro", "macro_braced", "include",
                        *DIRECTIVES])
    if shape in DIRECTIVES:
        macros["__directive__"] = True
        name = f"{key}Half{rng.randint(0, 99)}"
        if shape == "eval":
            return f'{key} #eval "{_fmt(x)}*2";'
        if shape == "calc_macro":
            macros[name] = f'#calc "{_fmt(x)}*2"'
            return f"{key} ${name};"
        macros[name] = _fmt(x)
        ref = f"${{{name}}}" if shape == "calc_braced" else f"${name}"
        return f'{key} #calc "{ref}*2";'
    if shape == "plain":
        return f"{key} {_fmt(x)};"
    if shape == "dims":
        return f"{key} [0 2 -1 0 0 0 0] {_fmt(x)};"
    if shape == "old":
        return f"{key} {key} [0 2 -1 0 0 0 0] {_fmt(x)};"
    name = f"{key}Value{rng.randint(0, 99)}"
    (inc if shape == "include" else macros)[name] = _fmt(x)
    return f"{key} ${{{name}}};" if shape == "macro_braced" else f"{key} ${name};"


def _decoys(rng: random.Random, key: str) -> str:
    """Entries with the same key that the reader must not take: nested, or commented out."""
    return rng.choice(["", f"other\n{{\n    {key} 12345;\n}}\n", f"// {key} 54321;\n",
                       f"/* {key} 999; */\n"])


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _key_case(rng: random.Random, root: Path):
    """A dictionary holding `x` at a random block path; returns (item, x)."""
    x = _num(rng)
    path = rng.choice([(), (), ("mixture", "transport"), ("water",)])
    macros, inc = {}, {}
    entry = _scalar_entry(rng, "nu", x, macros, inc)
    if macros.pop("__directive__", False):
        x = None                                         # unreadable, by design
    fillers = [f"a{i} {rng.randint(0, 9)};" for i in range(rng.randint(0, 3))]
    line = " ".join(rng.sample(fillers + [entry], len(fillers) + 1)) if rng.random() < 0.5 \
        else "\n".join(fillers + [entry])
    body = line
    for name in reversed(path):
        body = f"{name}\n{{\n{_decoys(rng, 'nu')}{body}\n}}"
    top = "".join(f"{k} {v};\n" for k, v in macros.items())
    if inc:
        _write(root, "constant/include/values", "".join(f"{k} {v};\n" for k, v in inc.items()))
        top += '#include "include/values"\n'
    text = HEAD + _comment(rng) + _decoys(rng, "nu") + top + _comment(rng) + body + "\n"
    _write(root, "constant/props", text)
    return SpecItem(name="nu", file="constant/props", path=path, key="nu", value=0.0), x


def _patch_case(rng: random.Random, root: Path):
    """A field file whose patch holds `want` in a random shape; returns (item, want)."""
    vector = rng.random() < 0.5
    vals = [_num(rng) for _ in range(3)] if vector else [_num(rng)]
    want = vals if vector else vals[0]
    lit = "(" + " ".join(_fmt(v) for v in vals) + ")" if vector else _fmt(vals[0])
    shape = rng.choice(["uniform", "macro", "nonuniform"])
    top = ""
    if shape == "uniform":
        value = f"uniform {lit}"
    elif shape == "macro":
        top = f"inletValue {lit};\n"
        value = "uniform $inletValue"
    else:
        other = "(" + " ".join(_fmt(v + 1) for v in vals) + ")" if vector else _fmt(vals[0] + 1)
        n_mode, n_other = rng.randint(3, 8), rng.randint(0, 2)
        faces = [lit] * n_mode + [other] * n_other
        rng.shuffle(faces)
        kind = "vector" if vector else "scalar"
        value = f"nonuniform List<{kind}> {len(faces)}({' '.join(faces)})"
    group = rng.choice(["inlet", '"(inlet|top)"', '"in.*"'])
    one_line = rng.random() < 0.5
    bc = (f"{group} {{ type fixedValue; value {value}; }}" if one_line
          else f"{group}\n    {{\n        type fixedValue;\n        value {value};\n    }}")
    others = rng.sample(["outlet { type zeroGradient; }",
                         'walls { type fixedValue; value uniform 0; }',
                         '#includeEtc "caseDicts/setConstraintTypes"'], rng.randint(0, 3))
    patches = others + [bc]
    rng.shuffle(patches)
    text = (HEAD + top + _comment(rng) + "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform 0;\n"
            "boundaryField\n{\n" + "\n".join(patches) + "\n}\n")
    _write(root, "0/X", text)
    return SpecItem(name="x", file="0/X", patch="inlet",
                    value=[0.0, 0.0, 0.0] if vector else 0.0), want


def _same(found, want) -> bool:
    if want is None:
        return found is None
    if isinstance(want, list):
        return isinstance(found, list) and len(found) == 3 and all(
            abs(a - b) <= 1e-12 * max(1.0, abs(b)) for a, b in zip(found, want))
    return isinstance(found, float) and abs(found - want) <= 1e-12 * max(1.0, abs(want))


@pytest.mark.parametrize("seed", range(0, N_ORACLE, 50))
def test_the_reader_reads_what_the_generator_wrote(tmp_path, seed):
    for s in range(seed, seed + 50):
        rng = random.Random(f"tst14:{s}")
        root = tmp_path / f"c{s}"
        item, want = (_key_case if rng.random() < 0.5 else _patch_case)(rng, root)
        found = deck.read(root, item)
        text = (root / item.file).read_text()
        assert _same(found.value, want), f"seed {s}: read {found!r}, wrote {want!r}\n{text}"


_JUNK = ["{", "}", ";", '"', "$", "$x", "#include \"nope\"", '#include "../../../etc/passwd"',
         "(", ")", "nonuniform List<scalar> 3(", "[0 2", "/*", "*/", "//", "\n", "\x00", "é",
         "uniform", "value", "inlet", "\"(inlet|\"", "\"[\"", "$nu", "${", "*", "\\",
         "#calc \"$a*2\" ", "#eval ", "#codeStream", " #calc"]


def _corrupt(rng: random.Random, text: str) -> str:
    chars = list(text)
    for _ in range(rng.randint(1, 8)):
        op = rng.random()
        i = rng.randrange(len(chars) + 1)
        if op < 0.4:
            chars[i:i] = list(rng.choice(_JUNK))
        elif op < 0.7 and chars:
            del chars[i:i + rng.randint(1, 12)]
        elif chars:
            j = rng.randrange(len(chars))
            chars[i:i] = chars[j:j + rng.randint(1, 20)]
    return "".join(chars)


@pytest.mark.parametrize("seed", range(0, N_CORRUPT, 50))
def test_the_reader_never_raises_and_reads_the_same_thing_twice(tmp_path, seed):
    for s in range(seed, seed + 50):
        rng = random.Random(f"tst14-corrupt:{s}")
        root = tmp_path / f"c{s}"
        item, _ = (_key_case if rng.random() < 0.5 else _patch_case)(rng, root)
        p = root / item.file
        p.write_text(_corrupt(rng, p.read_text()))
        try:
            inner = deck._read_one(root, item.file, item)     # beneath read()'s own guard
        except Exception as exc:                               # noqa: BLE001
            pytest.fail(f"seed {s}: the reader raised {type(exc).__name__}: {exc}\n"
                        f"{p.read_text()}")
        first, second = deck.read(root, item), deck.read(root, item)
        assert first == second == inner, f"seed {s}"
        assert first.value is None or isinstance(first.value, (float, list, str))
        if first.value is None:
            assert first.note, f"seed {s}: no value and no reason"


def test_an_include_path_that_is_no_path_does_not_raise(tmp_path):
    """Seed 5907 of a 6,000-case run: a NUL byte in an `#include` path raised from pathlib."""
    root = tmp_path / "case"
    _write(root, "constant/props", HEAD + '#include "incl\x00ude/values"\nnu $nuValue;\n')
    item = SpecItem(name="nu", file="constant/props", key="nu", value=0.0)
    found = deck._read_one(root, item.file, item)
    assert found.value is None and found.note


@pytest.mark.parametrize("line", ['nu #calc "$a*2";', 'b #calc "1*2";\nnu $b;',
                                  'nu #eval "0.1*2";', "nu uniform #calc \"0.1\";"])
def test_a_directive_is_never_read_as_a_number(tmp_path, line):
    root = tmp_path / "case"
    _write(root, "constant/props", HEAD + "a 0.1;\n" + line + "\n")
    found = deck.read(root, SpecItem(name="nu", file="constant/props", key="nu", value=0.0))
    assert found.value is None and "directive" in found.note


def test_an_include_outside_the_case_is_never_read(tmp_path):
    outside = tmp_path / "secret"
    outside.write_text("nuValue 42;\n")
    root = tmp_path / "case"
    _write(root, "constant/props", HEAD + '#include "../../secret"\nnu $nuValue;\n')
    found = deck.read(root, SpecItem(name="nu", file="constant/props", key="nu", value=0.0))
    assert found.value is None and "could not be found" in found.note
