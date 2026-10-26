"""`courant.case.dictscan` and the deck reader's two linear scans (TRD VER-13a; the review of
hp6/final, MAJOR 5).

The scan reads a dictionary file's statements as v2406's tokenizer splits them (ISstream.C,
entryIO.C, primitiveEntryIO.C, wordI.H), for one question: is foamlib's reading of the file the
solver's, keyword for keyword. Anything it does not follow is an answer of no, so it must never
read a file differently from v2406 and answer yes; and it must take time linear in the file's
length, as must the deck reader's comment stripping and entry lookup, which took time in the
square of it on hostile input.
"""

from __future__ import annotations

import random
import re
import subprocess
import time
import warnings
from pathlib import Path

import pytest

from courant.case import dictscan
from courant.spec import deck

ROOT = Path(__file__).resolve().parents[2]


def _keys(text: str, *scope: str) -> list[str] | None:
    found = dictscan.scan(text)
    return None if found is None else found.keys(*scope)


# ---------------------------------------------------------------- the committed dictionaries
def _committed_fvschemes() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*fvSchemes"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    return [ROOT / f for f in out if f.endswith("fvSchemes")]


def test_every_committed_fvschemes_is_read_as_foamlib_reads_it_or_holds_a_macro():
    """25 committed `fvSchemes`: the scan gives foamlib's top-level keywords, in order, on every
    one that writes no `$` or `#` (20), and refuses the rest (5, the certification corpus's
    simpleFoam cases), each of which writes a `$macro` (pitzDaily's `$turbulence`, and the like):
    nothing else is refused."""
    from foamlib import FoamFile

    from courant.case._foam_reader import _entries_of

    files = _committed_fvschemes()
    assert len(files) == 25
    followed = 0
    for f in files:
        text = f.read_text(errors="replace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            top = [k for k, _ in _entries_of(FoamFile(f))]
        keys = _keys(text)
        if keys is None:
            assert re.search(r"[$#]", deck.strip_comments(text)), f
            continue
        followed += 1
        assert [k for k in keys if k != "FoamFile"] == top, f
    assert followed == 20


# ---------------------------------------------------------------- as v2406 reads it
def test_comment_markers_inside_a_string_are_string():
    """`ISstream::read(string&)` reads a string to its closing quote, comment markers and all. The
    deck reader's `strip_comments` strips `/* ... */` across the strings, which hid the first
    `ddtSchemes` from its count; the scan sees both."""
    text = 'x "/*";\nddtSchemes { ddt(T) Euler; }\ny "*/";\nddtSchemes { default steadyState; }\n'
    assert _keys(text) == ["x", "ddtSchemes", "y", "ddtSchemes"]
    assert _keys(text, "ddtSchemes") == ["ddt(T)", "default"]
    assert deck.strip_comments(text).count("ddtSchemes") == 1


def test_a_comment_ends_a_word_as_word_valid_says():
    """`word::valid` refuses `/` (wordI.H), so `abc//x` is the word `abc` then a comment."""
    assert _keys("a abc//; b c;\n;\nd e;") == ["a", "d"]
    assert _keys("a b; /* c d; */ e f; // g h;\n") == ["a", "e"]


def test_statements_are_entry_new_s():
    text = ('FoamFile { version 2.0; }\nddtSchemes { default steadyState; "ddt\\(U\\)" Euler; }\n'
            "fluxRequired { default no; p; }\n;\ndiv { div(phi,U) Gauss limited corrected 0.33;"
            " div((nuEff*dev2(T(grad(U))))) Gauss linear; x ( 1 ( 2 3 ) ); }\n")
    assert _keys(text) == ["FoamFile", "ddtSchemes", "fluxRequired", "div"]
    assert _keys(text, "ddtSchemes") == ["default", '"ddt\\(U\\)"']
    assert _keys(text, "fluxRequired") == ["default", "p"]
    assert _keys(text, "div") == ["div(phi,U)", "div((nuEff*dev2(T(grad(U)))))", "x"]
    found = dictscan.scan(text)
    assert [e.dictionary for e in found.at()] == [True, True, True, True]
    assert dictscan.unquoted('"ddtSchemes"') == "ddtSchemes" == dictscan.unquoted("ddtSchemes")


@pytest.mark.parametrize("text", [
    '#include "x"\n', "a $b;\n", "$b;\n", "a { #inputMode protect\n }", "a #calc \"1\";",
    "#{ x #}", '"ddtSchemes/ddt(T)" Euler;', "a b/c;", "a 'b';", "1 2;", "(a b);",
    'a "b\nc";', "/* unclosed", '"unclosed', "a { b c; ", "a b }", "}", "a (b; c);",
    "a { b { c d; } }", "a b {c d;};", "a (b;", "a b)c;", "a b", "a", "a  b;",
    "a\x00 b;", '"a\x01" b;', "a (b\n", "{ a b; }", "a b(c d;",
], ids=lambda t: repr(t)[:30])
def test_what_the_scan_does_not_follow_is_an_answer_of_no(text):
    """A directive, a macro, a slash-scoped keyword or a lone `/`, a single quote, a keyword that
    is a number or a list, a string over its line, an unclosed comment, string, brace or
    parenthesis, a brace or `;` where a value runs, a value with no `;`, a character that is not
    printable ASCII. `a { b { c d; } }` is followed: dictionaries nest."""
    if text == "a { b { c d; } }":
        assert _keys(text, "a", "b") == ["c"]
    else:
        assert dictscan.scan(text) is None


def test_a_string_value_may_hold_what_a_keyword_may_not():
    assert _keys('location "system/x#$";\na b;') == ["location", "a"]
    assert dictscan.scan('"a/b" c;') is None and dictscan.scan('"a$" c;') is None


# ---------------------------------------------------------------- linear time
_HOSTILE = {
    "a keyword on every line of a quoted keyword": lambda n: '"' + "ddtSchemes\n" * n + '" { }',
    "a keyword on every line of a value": lambda n: "a 1;\n" + "ddtSchemes\n" * n + "}",
    "unclosed comment openers in a string": lambda n: 'dummy "' + "/*x" * n + '";',
    "unclosed comment openers": lambda n: "/*x" * n,
    "deep parentheses": lambda n: "a " + "(" * n + ";",
    "a long word": lambda n: "a " + "b" * (10 * n),
}


@pytest.mark.parametrize("shape", sorted(_HOSTILE))
def test_the_scans_are_linear_on_hostile_input(shape):
    """300,000 repetitions (some 3 MB): each scan returns in well under a second here; the deck
    reader's two regular expressions took minutes to hours on the first and third shapes."""
    text = _HOSTILE[shape](300_000)
    t0 = time.perf_counter()
    dictscan.scan(text)
    deck.strip_comments(text)
    deck._entries(text, "ddtSchemes")
    assert time.perf_counter() - t0 < 30


# ---------------------------------------------------------------- the deck reader, unchanged
def _entries_by_pattern(text: str, key: str) -> list[str]:
    """`deck._entries` before the review of hp6/final: the pattern it now computes linearly."""
    return [re.sub(r"\s+", " ", m.group(1)).strip() for m in re.finditer(
        rf"(?m)(?:^|(?<=[;{{}}]))[ \t]*{re.escape(key)}\s+([^;{{}}]*);", deck._top_level(text))]


def _strip_by_pattern(text: str) -> str:
    """`deck.strip_comments` before the review of hp6/final."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return deck._BRACED_MACRO.sub(r"$\1", text)


_PIECES = ["a", "ab", "b", " ", "\t", "\n", ";", "{", "}", "/*", "*/", "/", "*", "//", "\r",
           "\x1c", " ", " ", "\x0b", '"', "$", "${a}", "x", "(", ")", "  ", "ab ", "a\t"]
_KEYS = ["a", "ab", "b", "a b", "ab ", "a\t", "x", "", " a", "\ta", "(", "a(", "*"]


def test_the_deck_readers_linear_scans_give_the_patterns_answers():
    """30,000 seeded texts of up to 40 pieces (whitespace of every kind `\\s` matches, delimiters,
    comment markers, macros), three keys each (90,000 pairs): `deck._entries` and
    `deck.strip_comments` give what their patterns gave, on every one."""
    rng = random.Random(20261026)
    for _ in range(30_000):
        text = "".join(rng.choice(_PIECES) for _ in range(rng.randint(0, 40)))
        for key in rng.sample(_KEYS, 3):
            assert deck._entries(text, key) == _entries_by_pattern(text, key), (text, key)
        assert deck.strip_comments(text) == _strip_by_pattern(text), text
