"""A list foamlib's number patterns backtrack on is refused in linear time, never parsed for ever
(research/notes/invariants-070/FINDINGS.md, INV6-F1).

foamlib 1.7.10 matches an ASCII numeric list with `(?:\\s*[0-9.+-einfaty...]+)*\\s*\\)`: a number
may split into any number of pieces with an empty `\\s*` between, so a list that does not close as
a list of numbers backtracks over every split. Measured on this foamlib: six numbers of a scalar
list cut mid-write take 15 s, each further number about 16 times longer; 24 vectors 18 s. A
field file cut mid-write (VER-7b's own case), or a typo inside any list, hung `courant verify`.

`courant.case.foamguard` scans what foamlib would read, in linear time, and refuses a list whose
failing prefix is long enough to take foamlib more than seconds; every reader that hands a file
to foamlib asks it first, inside the try that already records an unreadable file. These tests run
each reading in a child process with a deadline, so a regression fails instead of hanging.

Shrunk from the fuzz mutants `reader[35]` (heated_cavity 200/U cut in half), `reader[24]`
(wedge_pipe 1214/T truncated) and the hotRoom `0/T` cut in half.

The guard is behind `--checks next` (`case.foam_guard`, TRD VER-3h): each reader here runs under
`next`, or inside `foamguard.guarded(True)`. `check_bytes` is the rule itself, asked directly.
Under the default no reader asks (`test_under_the_default_no_reader_asks_the_guard`).
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOTROOM = ROOT / "tests/fixtures/buoyant/hotRoom_laminar_044"
DUCT = ROOT / "tests/fixtures/real_openfoam/duct/converged"
DEADLINE = 30

HEAD = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class volScalarField;\n"
        "    object T;\n}\ndimensions [0 0 0 1 0 0 0];\n")


def _within(code: str, deadline: float = DEADLINE) -> str:
    """Run `code` in a child; its stdout, or a failure naming the hang."""
    t = time.monotonic()
    try:
        res = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT,
                             capture_output=True, text=True, timeout=deadline,
                             check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"still parsing after {deadline} s: foamlib's backtracking is not guarded")
    assert res.returncode == 0, res.stderr[-2000:]
    assert time.monotonic() - t < deadline
    return res.stdout


def _unpack(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    for lg in dst.glob("log.*.gz"):
        (dst / lg.name[:-3]).write_bytes(gzip.decompress(lg.read_bytes()))
        lg.unlink()
    return dst


def test_the_guard_refuses_a_scalar_list_cut_mid_write():
    from courant.case import foamguard

    cut = (HEAD + "internalField nonuniform List<scalar> 8(" + "299.3 " * 8).encode()
    with pytest.raises(foamguard.Backtracking, match="list"):
        foamguard.check_bytes(cut)


def test_the_guard_refuses_a_typo_inside_a_list_of_numbers():
    from courant.case import foamguard

    typo = (HEAD + "x (" + "299.3 " * 8 + "@);\n").encode()
    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes(typo)
    vectors = (HEAD + "x (" + "(1.5 2.5 3.5) " * 30 + "(1 2 @));\n").encode()
    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes(vectors)
    # Groups of mixed size close as a list, but not as a list of vectors: foamlib's vector
    # pattern backtracks over every vector before the odd group (24 vectors: 19 s).
    mixed = (HEAD + "internalField nonuniform List<vector> 31(" + "(1.5 2.5 3.5) " * 30
             + "(1 2 3 4));\n").encode()
    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes(mixed)


def test_the_guard_refuses_a_long_list_whose_count_does_not_match():
    """A list of numbers whose count is not its length fails foamlib's fast path after it has
    matched, and its general parser then takes time quadratic in the length (800 numbers 4 s).
    Refused where that wait is a hang: `MAX_MISMATCHED` numbers, a row of a vector counting three.
    Shrunk from `reader[1]`: wedge_pipe's 1214/U with a block of rows duplicated."""
    from courant.case import foamguard

    numbers = foamguard.MAX_MISMATCHED
    rows = numbers // 3 + 1
    for count, body in ((numbers - 1, "1.5 " * numbers),
                        (rows - 1, "(1.5 2.5 3.5)\n" * rows)):
        with pytest.raises(foamguard.Backtracking, match="count"):
            foamguard.check_bytes((HEAD + f"x nonuniform List<vector> {count}(" + body + ");\n")
                                  .encode())
    # One row short of it, and a whole list at any length, are read.
    foamguard.check_bytes((HEAD + f"x nonuniform List<vector> {rows - 2}("
                           + "(1.5 2.5 3.5)\n" * (rows - 1) + ");\n").encode())
    foamguard.check_bytes((HEAD + f"x nonuniform List<vector> {numbers}("
                           + "(1.5 2.5 3.5)\n" * numbers + ");\n").encode())
    foamguard.check_bytes((HEAD + "x 3(1 2 3);\ny 2((1 2 3) (4 5 6));\n").encode())


def test_the_guard_refuses_deep_nesting():
    """foamlib's general parser tries a dictionary, a keyword entry and a data entry at every
    level: balanced parentheses 14 deep take 0.3 s, each level twice as long. Shrunk from
    `reader[174]` (transportProperties with 51 nested pairs)."""
    from courant.case import foamguard

    with pytest.raises(foamguard.Backtracking, match="deep"):
        foamguard.check_bytes((HEAD + "x " + "(" * 51 + ")" * 51 + ";\n").encode())
    foamguard.check_bytes((HEAD + "x ((((1 2) (3 4))));\n").encode())


def test_the_guard_counts_whitespace_as_foamlib_backtracks_on_it():
    """foamlib's patterns also split every gap of whitespace: an indented scalar list cut after
    five lines takes 7 s, an indented vector list with a typo after eight rows 2 s, each line
    more many times longer (review of the INV6-F1 fix, B2)."""
    from courant.case import foamguard

    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes((HEAD + "internalField nonuniform List<scalar> 8\n(\n"
                               + "    0\n" * 7).encode())
    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes((HEAD + "internalField nonuniform List<vector> 12\n(\n"
                               + "    (1 0 0)\n" * 11 + "    (1 0 @)\n);\n").encode())


def test_the_guard_passes_a_binary_field():
    """A binary payload is bytes, not text: its `(` and `)` bytes are not parentheses (B1)."""
    import numpy as np

    from courant.case import foamguard

    vals = np.random.default_rng(0).random(12_000 * 3)
    vals[:3] = np.frombuffer(b"(" * 24, dtype="<f8")         # 24 `(` bytes in a row
    head = HEAD.replace("format ascii", "format binary").replace("volScalarField",
                                                                 "volVectorField")
    data = (head.encode() + b"internalField nonuniform List<vector> 12000(" + vals.tobytes()
            + b");\nboundaryField\n{\n    wall\n    {\n        type zeroGradient;\n    }\n}\n")
    foamguard.check_bytes(data)


def test_a_comment_inside_a_dictionary_list_of_points_is_passed():
    """foamlib reads a list of points with a comment in it, in a dictionary, in 0.07 s (M3)."""
    from courant.case import foamguard

    foamguard.check_bytes((HEAD + "probeLocations (\n" + "(0.00 0.05 0)\n" * 24
                           + "// second rake\n" + "(0.00 0.10 0)\n" * 2 + ");\n").encode())


def test_the_guard_is_linear_on_unterminated_comments_and_strings():
    """foamlib stops at an unterminated `/*` at once; the guard must too (M1)."""
    from courant.case import foamguard

    for junk in ("/*x", '"', "#{"):
        data = (HEAD + "internalField uniform 300;\n" + junk * 60_000).encode()
        t = time.monotonic()
        try:
            foamguard.check_bytes(data)
        except foamguard.Backtracking:
            pass
        assert time.monotonic() - t < 5, junk


@pytest.mark.parametrize("text", [
    "internalField nonuniform List<scalar> 3(1 2 3);\n",
    "internalField nonuniform List<vector> 2((1 2 3) (4 5 6));\n",
    "x (1e-05 0 0);\nfields (U p T);\npatches (fan1 fan2 fan3 inlet outlet);\n",
    "x (" + "299.3 " * 4 + "@);\n",          # short: foamlib fails fast, and says why
    "/* (1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 */ y 1;\n",
    'libs ("lib1.so" "lib2.so");\n',
])
def test_the_guard_passes_what_foamlib_reads_in_time(text):
    from courant.case import foamguard

    foamguard.check_bytes((HEAD + text).encode())


def test_the_guard_is_linear_on_a_large_valid_field():
    from courant.case import foamguard

    big = (HEAD + "internalField nonuniform List<vector> 200000(\n"
           + "(1.25 -2.5e-05 3)\n" * 200_000 + ");\n").encode()
    t = time.monotonic()
    foamguard.check_bytes(big)
    cut = big[: len(big) // 2]
    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes(cut)
    assert time.monotonic() - t < 10


def test_the_guard_passes_every_committed_case_file():
    """No over-refusal: every dictionary and field file of the committed cases foamlib reads
    today is passed."""
    from courant.case import foamguard

    roots = [ROOT / "tests/fixtures", ROOT / "certification/monitored"]
    n = 0
    for root in roots:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix in (".log", ".json", ".py", ".sh", ".md", ".dat",
                                               ".csv", ".xy", ".txt") \
                    or p.name.startswith("log.") or p.name.endswith(".log.gz"):
                continue
            data = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
            if b"FoamFile" not in data[:4096]:
                continue
            foamguard.check_bytes(data, p.name)
            n += 1
    assert n > 300


def test_a_case_whose_zero_field_was_cut_mid_write_is_read(tmp_path):
    case = _unpack(HOTROOM, tmp_path / "hotRoom")
    t = case / "0" / "T"
    t.write_bytes(t.read_bytes()[: len(t.read_bytes()) // 2])
    out = _within(f"""
        from courant.verify import verify_path
        v = verify_path({str(case)!r}, checks="next")
        print(v.verdict)
    """)
    assert out.split()[-1] in ("accept", "reject", "insufficient")


def test_a_latest_field_cut_mid_write_is_unread_not_a_hang(tmp_path):
    case = _unpack(DUCT, tmp_path / "duct")
    gz = case / "120" / "T.gz"
    data = gzip.decompress(gz.read_bytes())
    gz.unlink()
    (case / "120" / "T").write_bytes(data[: len(data) // 2])
    out = _within(f"""
        from courant.verification._fluid_reader import read_fields
        from courant.verify import verify_path
        from courant.case.foamguard import guarded
        with guarded(True):
            r = read_fields({str(case)!r})
        print(sorted(r.unread))
        v = verify_path({str(case)!r}, checks="next")
        print(v.verdict)
        print(json.dumps(list(v.not_assessed)))
    """.replace("from courant.verify import", "import json\n        from courant.verify import"))
    unread, verdict, said = out.strip().splitlines()[-3:]
    assert "'T'" in unread
    assert verdict in ("accept", "reject", "insufficient")
    assert any("could not be read (T)" in s and "t = 120" in s for s in json.loads(said))


def test_a_dictionary_with_a_broken_list_is_unreadable_not_a_hang(tmp_path):
    case = _unpack(HOTROOM, tmp_path / "hotRoom")
    cd = case / "system" / "controlDict"
    cd.write_text(cd.read_text() + "\nbroken (" + "299.3 " * 9 + "@);\n")
    out = _within(f"""
        from courant.case.config import CaseConfig
        from courant.case.foamguard import guarded
        with guarded(True):
            c = CaseConfig.from_foam({str(case)!r})
        print(sorted(c.unreadable))
    """)
    assert "system/controlDict" in out


# ------------------------------------------------- what is refused is what foamlib hangs on (VER-3h)
# Timed on foamlib 1.7.10 itself, each input in a child with a 12 s deadline, in the merge of
# 6 Nov 2026 (research/notes/invariants-070/FINDINGS.md). These run no such hang: the guard answers
# in linear time, and foamlib is run only on what it reads at once.
N9 = "299.3 " * 9
D9 = "1.2345 " * 9
FIELDS_12 = ("functions\n{\n    probes\n    {\n        type probes;\n        fields\n        (\n"
             "            T\n            p\n        );\n    }\n}\n")
COMMENTS_4 = "fields\n(\n    // one\n    // two\n    // three\n    U\n);\n"
MACRO_VECTOR = "internalField uniform (1.2345678901 2.3456789012 $z);\n"
T_MEAN = "fields\n(\n            T\n            T:mean\n);\n"

#: Valid: foamlib reads each in under 0.01 s.
READ_AT_ONCE = {
    "fields, 12 spaces": FIELDS_12,
    "the same list at the top": "fields\n(\n            T\n            p\n);\n",
    "a list opened by three comments": COMMENTS_4,
    "a macro in a vector": MACRO_VECTOR,
    "a word whose stem is in the number class": T_MEAN,
    "then a word": f"x ({D9}word);\n",
    "then a macro": f"x ({D9}$z);\n",
    "then a directive word": f"x ({D9}#z);\n",
    "then a string": f'x ({D9}"s");\n',
    "then a list": f"x ({D9}(1 2 3));\n",
    "then a dictionary": f"x ({D9}{{a 1;}});\n",
    "then a keyword entry's semicolon": f"x ({D9};);\n",
    "closed after a comment": f"x ({D9}// c\n);\n",
    "a boundary's list of dictionaries": ("boundary\n(\n    inlet\n    {\n        type patch;\n"
                                          "        faces ((0 4 7 3));\n    }\n);\n"),
}
#: Hung past 12 s. Those after the first four hold a byte that continues a list right after the
#: numeric prefix, so the byte alone cannot tell them: the statement does not read.
HANGS = {
    "@": f"x ({N9}@);\n",
    "; then cut": f"x ({N9};\n",
    "}": f"x ({N9}}}\n",
    "cut": f"x ({N9}",
    "a word then @": f"x ({N9}word @);\n",
    "a word, closed, then @": f"x ({N9}word) @;\n",
    "a word, cut": f"x ({N9}word\n",
    "a list, cut": f"x ({N9}(1 2\n",
    "a string, cut": f'x ({N9}"str\n',
    "closed, no semicolon": f"x ({N9}word)\n}}\n",
    "closed with a comment, then @": f"x ({N9}// c\n) @;\n",
    "no keyword": f"({N9}word)\n",
    "a count, no keyword": f"8 ({N9}word)\n",
    "after a block": f"x {{ a 1; }} ({N9}word);\n",
    "a number glued to a word": f"x ({N9}1.5abc);\n",
    "two semicolons": f"x ({N9};;);\n",
    "after a directive": f"#inputMode merge\n({N9}word);\n",
    "a scalar field, a word": f"internalField nonuniform List<scalar> 9({N9}word);\n",
    "a scalar field, valid, a comment": f"internalField nonuniform List<scalar> 9({N9}// c\n);\n",
    "a scalar field, cut": f"internalField nonuniform List<scalar> 9({N9}",
    "a scalar field in a patch, cut": ("boundaryField\n{\n    a\n    {\n        value nonuniform "
                                       f"List<scalar> 9({N9}"),
    "a vector field in a patch, a typo": ("boundaryField\n{\n    a\n    {\n        value nonuniform"
                                          " List<vector> 30(" + "(1.5 2.5 3.5) " * 29
                                          + "(1 2 @));\n    }\n}\n"),
}
#: foamlib stops on each in under 0.02 s, with its own error: the patterns never run.
FAILS_AT_ONCE = {
    "in a brace, @": f"functions\n{{\n    y ({N9}@);\n}}\n",
    "in a brace, cut": f"functions\n{{\n    y ({N9}",
    "in a brace, a word then @": f"functions\n{{\n    y ({N9}word @);\n}}\n",
    "in two braces": ("a\n{\n    b\n    {\n        c\n        (\n            T\n            p\n"
                      "            @\n        );\n    }\n}\n"),
    "in a list": f"x (a ({N9}@));\n",
}


@pytest.mark.parametrize("name", READ_AT_ONCE)
def test_what_foamlib_reads_at_once_is_passed_and_read(name):
    from foamlib import FoamFile

    from courant.case import foamguard

    text = HEAD + READ_AT_ONCE[name]
    foamguard.check_bytes(text.encode())
    FoamFile.loads(text)


@pytest.mark.parametrize("name", HANGS)
def test_what_foamlib_hangs_on_is_refused(name):
    from courant.case import foamguard

    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes((HEAD + HANGS[name]).encode())


@pytest.mark.parametrize("name", FAILS_AT_ONCE)
def test_a_list_the_patterns_never_reach_is_left_to_foamlib(name):
    """It fails in foamlib in under 0.02 s, with foamlib's own error, as it did in 0.7.0."""
    from foamlib import FoamFile

    from courant.case import foamguard

    text = HEAD + FAILS_AT_ONCE[name]
    foamguard.check_bytes(text.encode())
    t = time.monotonic()
    with pytest.raises(ValueError):
        FoamFile.loads(text)
    assert time.monotonic() - t < 5


@pytest.mark.parametrize("text", [
    *(t.replace(N9, "1 2 3 ") for n, t in HANGS.items() if "field" not in n),
    *(t.replace(D9, "1 2 3 ") for n, t in READ_AT_ONCE.items() if n != "fields, 12 spaces"),
])
def test_the_guard_says_of_a_statement_what_foamlib_does(text):
    """Each statement with three short numbers, on which the patterns fail at once: where the
    guard says the patterns are reached, foamlib fell back to standalone data, or stopped."""
    from foamlib import FoamFile

    from courant.case import foamguard

    data = foamguard._visible((HEAD + text).encode())
    tops = foamguard._top_lists(data)
    p = max(tops)
    try:
        standalone = None in FoamFile.loads(data)
    except ValueError:
        standalone = True
    assert foamguard._reached(data, p, *tops[p]) == standalone


@pytest.mark.parametrize("comment", ['// 6" duct, run 3\n', "// a #{ in a comment\n",
                                     '/* a " and a #{ */\n'], ids=["quote", "verbatim", "block"])
def test_a_quote_in_a_comment_hides_no_cut_field(comment):
    """One pass, in order of first occurrence (B1): a `"` or `#{` in a comment opened a string to
    the end of the file, so the guard saw no list at all, and foamlib hung on a normal crash."""
    from courant.case import foamguard

    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes((HEAD + comment + "internalField nonuniform List<scalar> 8("
                               + "299.3 " * 8).encode())


def test_a_comment_mark_in_a_string_is_no_comment():
    """The other order of the same pass: a `//` in a string does not comment out the list."""
    from courant.case import foamguard

    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes((HEAD + 'url "http://x";\ninternalField nonuniform List<scalar> 8('
                               + "299.3 " * 8).encode())


def test_comment_text_is_not_charged_as_digits():
    """B2b: 70 dashes in a comment are in the number class, and are no number. foamlib fails its
    patterns at once on this list and reads it with its general parser."""
    from foamlib import FoamFile

    from courant.case import foamguard

    text = (HEAD + "boundaryField\n{\n    a\n    {\n        type fixedValue;\n"
            "        value nonuniform List<vector> 2\n(\n// " + "-" * 70
            + "\n(1 2 3)\n(4 5 6)\n);\n    }\n}\n")
    foamguard.check_bytes(text.encode())
    t = time.monotonic()
    assert "boundaryField" in FoamFile.loads(text)
    assert time.monotonic() - t < 5


def test_a_field_list_that_closes_with_a_macro_is_held_to_the_bound_of_a_hang():
    """B2c: foamlib pays its failing patterns (2^22 paths), then reads the list with its general
    parser. `9(N word)` is 2^36 and stays refused (`HANGS`), and so does a typo that does not
    balance or whose stop byte continues nothing."""
    from foamlib import FoamFile

    from courant.case import foamguard

    text = (HEAD + "boundaryField\n{\n    a\n    {\n        type fixedValue;\n"
            "        value nonuniform List<scalar> 3(1.2345678901 2.3456789012 $z);\n    }\n}\n")
    foamguard.check_bytes(text.encode())
    FoamFile.loads(text)
    for broken in ("3(1.2345678901 2.3456789012 @);", "3(1.2345678901 2.3456789012 $z;"):
        with pytest.raises(foamguard.Backtracking):
            foamguard.check_bytes((HEAD + "x nonuniform List<scalar> " + broken + "\n").encode())


def test_under_the_default_no_reader_asks_the_guard(tmp_path, monkeypatch):
    """GATE-13: under the default every reader behaves as main does. The rule is replaced by one
    that refuses everything: nothing under the default reaches it, whichever reader, on a case
    with a field, a `phi`, a dictionary and a mesh-motion file; under `next` it refuses."""
    from courant.case import foamguard
    from courant.verify import verify_path
    from courant.verify.preflight import preflight_path

    asked = []

    def refuse(data, shown="the file"):
        asked.append(shown)
        raise foamguard.Backtracking(shown)

    for rel, src in (("duct", DUCT), ("hotRoom", HOTROOM),
                     ("mixer", ROOT / "tests/fixtures/moving_mesh/mixer_moving")):
        case = _unpack(src, tmp_path / rel)
        plain = verify_path(case).to_json()
        plain_pre = preflight_path(case).model_dump_json()
        with monkeypatch.context() as m:
            m.setattr(foamguard, "check_bytes", refuse)
            assert verify_path(case).to_json() == plain
            assert preflight_path(case).model_dump_json() == plain_pre
            assert asked == []
            verify_path(case, checks="next")
            assert asked, rel                       # the test can fail: under next they ask
            asked.clear()
            preflight_path(case, checks="next")
            assert asked, rel
            asked.clear()


def test_outside_guarded_the_sites_do_nothing(tmp_path):
    """Not even the read: a file foamlib hangs on, and a path that is not there."""
    from courant.case import foamguard

    cut = tmp_path / "T"
    cut.write_text(HEAD + "internalField nonuniform List<scalar> 9(" + N9)
    foamguard.check_path(cut)
    foamguard.check_path(tmp_path / "absent")
    foamguard.check_text(HEAD + "internalField nonuniform List<scalar> 9(" + N9)
    with foamguard.guarded(True):
        with pytest.raises(foamguard.Backtracking):
            foamguard.check_path(cut)
        with foamguard.guarded(False):
            foamguard.check_path(cut)
        with pytest.raises(foamguard.Backtracking):
            foamguard.check_text(cut.read_text())
    foamguard.check_path(cut)


def test_every_reader_asks_through_the_switch():
    """`check_bytes` judges whatever it is given, so no reader may call it itself."""
    found = [p for p in (ROOT / "courant").rglob("*.py")
             if "foamguard.check_bytes" in p.read_text(errors="replace")
             or "import check_bytes" in p.read_text(errors="replace")]
    assert found == []


def test_a_valid_scalar_field_with_a_comment_is_refused_only_where_it_hangs():
    """Valid: foamlib reads it after its uncommented pattern fails. Comment after 6 numbers 1.8 s,
    after 7 38 s (read, in 0.7.0 too), after 9 past any deadline."""
    from courant.case import foamguard

    def text(n):
        return (HEAD + f"internalField nonuniform List<scalar> {n}(" + "299.3 " * n
                + "// c\n);\n").encode()

    foamguard.check_bytes(text(6))
    foamguard.check_bytes(text(7))
    with pytest.raises(foamguard.Backtracking):
        foamguard.check_bytes(text(9))


@pytest.mark.parametrize("text", [
    "coeffs_2 (" + "1.5 " * 600 + ");\n",
    "a.3 (" + "1.5 " * 600 + ");\n",
    "x 2 (" + "1.5 " * 600 + ");\n",        # a number and a list: foamlib reads it, 600 in 2.2 s
    "x 599(" + "1.5 " * 600 + ");\n",       # 4.5 s, where 600 for 600 takes 2.3 s
])
def test_a_dictionary_list_is_not_refused_for_a_count_it_does_not_have(text):
    from courant.case import foamguard

    foamguard.check_bytes((HEAD + text).encode())


def test_the_word_before_a_list_is_no_count():
    from courant.case import foamguard

    def count(text):
        return foamguard._count_before(text, text.index(b"("))

    assert count(b"coeffs_2 (") is None
    assert count(b"a.3 (") is None
    assert count(b"x-3 (") is None
    assert count(b"x2(") is None
    assert count(b"x 3(") == 3
    assert count(b"3 (") == 3
    assert count(b"\n12\n(") == 12


def test_the_guards_tables_are_foamlibs():
    """A foamlib that is not 1.7.10's: this names what to measure again."""
    from foamlib._files._parsing import _parser as fp

    from courant.case import foamguard

    assert foamguard._NUM == fp._POSSIBLE_FLOAT.pattern[:-1]
    assert foamguard._WORD_START == {c for c in range(256) if fp._IS_TOKEN_START[c]}
    assert foamguard._WORD_BYTE == {c for c in range(256) if fp._IS_TOKEN_CONTINUATION[c]}


def _with_dict(case: Path, *entries: str) -> None:
    cd = case / "system" / "controlDict"
    cd.write_text(cd.read_text() + "\n" + "".join(entries))


@pytest.mark.parametrize("entry", [FIELDS_12, COMMENTS_4, MACRO_VECTOR, T_MEAN],
                         ids=["fields", "comments", "macro", "T:mean"])
def test_a_valid_controldict_is_read_whole(tmp_path, entry):
    """The review's three inputs: refused by the guard as it first stood, each one unread
    `system/controlDict` and every value in it lost."""
    from courant.case import foamguard
    from courant.case._foam_reader import read_case_config

    case = _unpack(HOTROOM, tmp_path / "hotRoom")
    before = read_case_config(str(case))
    _with_dict(case, entry)
    with foamguard.guarded(True):
        foamguard.check_path(case / "system" / "controlDict")
        after = read_case_config(str(case))
    assert "system/controlDict" not in after.unreadable
    assert after.unreadable == before.unreadable
    assert after.end_time == before.end_time and after.end_time is not None


def _settling_at_courant_3(tmp_path: Path) -> Path:
    """The hotRoom run with its Courant numbers raised to a maximum of 3, adaptive under its own
    `maxCo 5` with outer correctors, so its Courant number is within the limit it declared."""
    case = _unpack(HOTROOM, tmp_path / "hotRoom")
    cd = case / "system" / "controlDict"
    text = re.sub(r"adjustTimeStep\s+\w+;", "adjustTimeStep yes;", cd.read_text())
    text, written = re.subn(r"maxCo\s+[^;]+;", "maxCo 5;", text)
    assert "adjustTimeStep yes;" in text and written == 1
    cd.write_text(text)
    (case / "system" / "fvSolution").write_text(
        HEAD.replace("volScalarField", "dictionary").replace("object T", "object fvSolution")
        .split("dimensions")[0] + "PIMPLE\n{\n    nOuterCorrectors 3;\n}\n")
    log = case / "log.buoyantPimpleFoam"
    scale = 3.0 / 0.513986
    log.write_text(re.sub(r"(Courant Number mean: \S+ max: )(\S+)",
                          lambda m: f"{m.group(1)}{float(m.group(2)) * scale:.6f}",
                          log.read_text(errors="replace")))
    return case


def test_a_valid_controldict_keeps_an_adaptive_runs_accept(tmp_path):
    """The review's verdict: a good adaptive run under `maxCo 5` settling at Courant 3, with a
    multi-line `fields` list in its controlDict. Were the file refused, `maxCo` would be lost and
    the limit fall to 1: reject (VER-5.4), on a valid file."""
    from courant.verify import verify_path

    case = _settling_at_courant_3(tmp_path)
    plain = verify_path(case, checks="next")
    assert plain.verdict == "accept"
    _with_dict(case, FIELDS_12)
    listed = verify_path(case, checks="next")
    assert listed.verdict == "accept"
    assert listed.rule == plain.rule
    assert not any("controlDict" in s for s in listed.not_assessed)


def test_the_flow_readers_ask_the_guard(tmp_path):
    """`conservation.energy.flow` (`--checks next`) reads `T`, `phi` and `constant/g` itself: each
    cut mid-write is said, not parsed for ever."""
    (tmp_path / "T").write_text(HEAD + "internalField nonuniform List<scalar> 9(" + N9)
    (tmp_path / "constant").mkdir()
    (tmp_path / "constant" / "g").write_text(HEAD + f"value ({N9}")
    out = _within(f"""
        from pathlib import Path
        from courant.case.foamguard import Backtracking, guarded
        from courant.verification._flow_heat import _field, _gravity
        with guarded(True):
            try:
                _field(Path({str(tmp_path / "T")!r}), "volScalarField")
            except Backtracking:
                print("field refused")
            print(_gravity(Path({str(tmp_path)!r}), "", str))
    """)
    assert "field refused" in out
    assert "constant/g could not be read" in out


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="INV6-F4: foamlib parses a long dictionary list in time "
                   "quadratic in its length; not refused (research/notes/invariants-070)")
def test_a_controldict_with_a_thousand_probe_points_is_read_in_time(tmp_path):
    """foamlib 1.7.10 reads `probeLocations` of 300 points in 8 s, 1000 in over a minute: a
    valid controlDict. Refusing it would change real verdicts, so it is left open."""
    case = _unpack(HOTROOM, tmp_path / "hotRoom")
    cd = case / "system" / "controlDict"
    cd.write_text(cd.read_text() + "\nfunctions\n{\n    probes\n    {\n        type probes;\n"
                  "        probeLocations\n        (\n"
                  + "            (0.1 0.2 0.3)\n" * 1000 + "        );\n    }\n}\n")
    _within(f"""
        from courant.case.config import CaseConfig
        CaseConfig.from_foam({str(case)!r})
    """, deadline=20)
