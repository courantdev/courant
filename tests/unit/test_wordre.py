"""The keyword matcher, `courant.case.wordre`, and the lookups built on it (TRD SPEC-2, VER-16; the
hard-proofing audit of 21 Oct 2026 and the review of its robustness part). Ported to main with
the matcher from hp/audit-robustness (25f2de8) at the final review of the 0.6.0 integration;
the readers it bounds on main are in `test_keyword_patterns_bounded.py`.

A quoted keyword in an OpenFOAM dictionary is a regular expression the case writes, which the
readers match against a patch or a field name. Three properties, each with its evidence here:

1. **It means what `re.fullmatch` means** for every pattern it evaluates. A seeded fuzz over every
   construct it evaluates (literals, `.`, classes with ranges and categories, alternation, bounded
   and unbounded repeats, greedy and lazy, anchors, inline and scoped flags, Unicode case
   folding), and every pattern with a nested or an empty loop against every short name. The
   counts are asserted: they are the denominator the TRD cites.
2. **Its work is bounded** per match (`MAX_STATES`), per dictionary lookup (`LOOKUP_STEPS`) and per
   verify (`RUN_STEPS`), so a case cannot stall the gate however many keywords it writes.
3. **A keyword it does not evaluate is never skipped where it may decide a value**: in
   `residualControl` before the first entry that matches a field, or among the relaxation
   factors after the last one, the field is held to the strictest tolerance or relaxation it may
   have. Skipped, `"(?=U)U" 1e-6; ".*" 1e-2;` held U to 1e-2, where OpenFOAM (ESI's `wordRe` is
   ECMAScript, which has lookahead) holds it to 1e-6. (Naming such a keyword under "not
   assessed", the branch's `case.keyword_not_evaluated`, is not on main: T-VER3B-ONE.)
"""

from __future__ import annotations

import gzip
import itertools
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from courant.case import wordre
from courant.case.config import CaseConfig

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "residual_control"
CORPUS = ROOT / "certification" / "corpus"
PITZ = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__shipped"
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"
HEALTHY_058 = FIX / "c10k.simpleFoam_pitzDaily.058.healthy.log.gz"


# ---------------------------------------------------------------- 1. re's meaning
_ANCHORS = ("^", "$", r"\A", r"\Z", r"\b", r"\B")
_CHARS = ("a", "A", "b", "B", "_", "1", r"\.", r"\-", " ", r"\n", "ſ", "K", "İ", "ı", "ß", "é")
_CATEGORIES = (r"\d", r"\w", r"\s", r"\D", r"\W", r"\S")
_CLASS_PARTS = ("a-c", "A-Z", "0-9", "_", r"\d", r"\w", r"\s", r"\D", ".", "b", "B", r"\n", "ſ",
                "K", "é-ı", "İ")
_QUANTIFIERS = ("*", "+", "?", "{2}", "{0,2}", "{1,3}", "{2,}", "{,2}")
_SCOPED_FLAGS = ("i", "s", "m", "a", "-i", "i-s", "im")
_GLOBAL_FLAGS = ("i", "s", "m", "a", "im", "is")
_NAME_CHARS = "aAbB_1 \n.-"
_UNICODE = "ſKİıßé"


def _atom(r: random.Random, depth: int) -> tuple[str, bool]:
    """(an atom, whether it holds a repeat)."""
    k = r.randrange(11 if depth < 3 else 6)
    if k == 0:
        return r.choice(_CHARS), False
    if k == 1:
        return ".", False
    if k == 2:
        return r.choice(_CATEGORIES), False
    if k == 3:
        parts = "".join(r.choice(_CLASS_PARTS) for _ in range(r.randint(1, 3)))
        return "[" + ("^" if r.random() < 0.3 else "") + parts + "]", False
    if k == 4:
        return r.choice(_ANCHORS), False
    if k == 5:
        return r.choice(_UNICODE), False
    body, repeats = _expr(r, depth + 1)
    if k in (6, 7):
        return f"({body})", repeats
    if k == 8:
        return f"(?:{body})", repeats
    if k == 9:
        return f"(?{r.choice(_SCOPED_FLAGS)}:{body})", repeats
    other, more = _expr(r, depth + 1)
    return f"({body}|{other})", repeats or more


def _piece(r: random.Random, depth: int) -> tuple[str, bool]:
    text, repeats = _atom(r, depth)
    # One level of repetition here: a repeat inside a repeat is the shape on which `re` itself
    # backtracks exponentially (it ran for seconds on a five-character name), so those are
    # tested apart, against every short name (`NESTED`).
    if repeats or text in _ANCHORS or r.random() < 0.5:
        return text, repeats
    return text + r.choice(_QUANTIFIERS) + ("?" if r.random() < 0.3 else ""), True


def _expr(r: random.Random, depth: int = 0) -> tuple[str, bool]:
    pieces = [_piece(r, depth) for _ in range(r.randint(0, 3))]
    if depth < 2 and r.random() < 0.15:
        pieces += [("|", False), *(_piece(r, depth) for _ in range(r.randint(0, 2)))]
    return "".join(t for t, _ in pieces), any(q for _, q in pieces)


def _pattern(r: random.Random) -> str:
    body, _ = _expr(r)
    return (f"(?{r.choice(_GLOBAL_FLAGS)})" if r.random() < 0.15 else "") + body


def _name(r: random.Random) -> str:
    alphabet = _NAME_CHARS + (_UNICODE if r.random() < 0.2 else "")
    return "".join(r.choice(alphabet) for _ in range(r.randint(0, 6)))


SEED, PATTERNS, NAMES = 2610, 2000, 40
#: The fuzz's denominator: pattern-name pairs evaluated, and of them those that match.
FUZZ_EVALUATED, FUZZ_MATCHED = 80_000, 9_929


def _fuzz(seed: int, patterns: int, names: int) -> tuple[int, int, list]:
    r = random.Random(seed)
    evaluated = matched = 0
    differ = []
    for _ in range(patterns):
        p = _pattern(r)
        ns = [_name(r) for _ in range(names)]
        compiled = re.compile(p)
        for n in ns:
            got = wordre.fullmatch(p, n)
            if got is None:
                continue
            evaluated += 1
            want = compiled.fullmatch(n) is not None
            matched += want
            if got is not want:
                differ.append((p, n, got))
    return evaluated, matched, differ


def test_the_matcher_says_what_re_says_on_a_seeded_fuzz():
    evaluated, matched, differ = _fuzz(SEED, PATTERNS, NAMES)
    assert differ == []
    assert (evaluated, matched) == (FUZZ_EVALUATED, FUZZ_MATCHED)


def test_every_name_a_pattern_matches_begins_with_its_literal_prefix_on_a_seeded_fuzz():
    """`literal_prefix` never says more than a match must begin with (TRD VER-5a: a solver entry
    is left unjudged only when every name its key matches begins with `pcorr`). The seeded
    patterns above, each against 40 names and the same names behind its prefix: of 160,000
    pairs 20,478 match, and every one begins with the prefix (142 of the 2,000 patterns have
    one)."""
    r = random.Random(SEED)
    pairs = matched = with_prefix = 0
    wrong = []
    for _ in range(PATTERNS):
        p = _pattern(r)
        head = wordre.literal_prefix(p)
        with_prefix += bool(head)
        compiled = re.compile(p)
        for _ in range(NAMES):
            n = _name(r)
            for name in (n, head + n):
                pairs += 1
                if compiled.fullmatch(name):
                    matched += 1
                    if not name.startswith(head):
                        wrong.append((p, name, head))
    assert wrong == []
    assert (pairs, matched, with_prefix) == (160_000, 20_478, 142)
    for pattern, head in (("pcorr.*", "pcorr"), ("(pcorr|pcorrFinal)", "pcorr"),
                          ("[p]corr", "pcorr"), ("(p|pcorr)", "p"), ("rho.*|pcorr.*", ""),
                          ("(?i)pcorr", ""), ("pcor*", "pco"), ("(pcorr", "")):
        assert wordre.literal_prefix(pattern) == head, pattern


def test_literal_prefix_parses_no_pattern_a_lookup_could_not_pay_to_compile(monkeypatch):
    """`re`'s parser moves an alternation's common start out of it one character a round, deleting
    the first item of every alternative each time (re/_parser.py, `_parse_sub`), so a key of two
    alternatives sharing n characters costs it some n squared moves: 0.29 s at 80,000 characters
    on this machine, roughly 100 s at 1 MB. `wordre`'s matcher charges a parse
    `COMPILE_STEPS_PER_CHAR` a character against a lookup's `LOOKUP_STEPS`, so it never parses
    such a key; `literal_prefix` parsed any (the review of hp6/integrate: a 1 MB fvSolution key
    and no pcorr line in the log hung `courant verify` by default). Past `MAX_PARSED_CHARS`, what
    a lookup could pay to compile, it parses nothing and says "", so the entry is judged."""
    assert wordre.MAX_PARSED_CHARS == (wordre.LOOKUP_STEPS - wordre.COMPILE_STEPS_PER_STATE
                                       * wordre.MAX_STATES) // wordre.COMPILE_STEPS_PER_CHAR
    assert wordre.MAX_PARSED_CHARS == 19_900

    def shared(n: int) -> str:              # two alternatives, `pcorr` and n `a`s each
        side = "pcorr" + "a" * n
        return f"({side}|{side})"

    at = shared((wordre.MAX_PARSED_CHARS - 13) // 2)
    past = shared(500_000)
    assert len(at) == wordre.MAX_PARSED_CHARS - 1 and len(past) > 1_000_000
    wordre.literal_prefix.cache_clear()
    assert wordre.literal_prefix(at).startswith("pcorr")

    class NoParse:
        @staticmethod
        def parse(*args, **kwargs):
            raise AssertionError("parsed past the bound")

    monkeypatch.setattr(wordre, "_sre_parse", NoParse)
    wordre.literal_prefix.cache_clear()
    assert wordre.literal_prefix(past) == ""
    assert wordre.literal_prefix("pcorr" + "a" * wordre.MAX_PARSED_CHARS) == ""


def test_a_pcorr_key_too_long_to_parse_is_judged():
    """`checkers._unsolved_pcorr` leaves an entry out only when every name its key matches begins
    with `pcorr`; a key too long to parse says nothing of that, so its loose tolerance is judged."""
    from courant.case.config import LinearSolverSpec
    from courant.diagnosis.checkers import loose_solvers

    side = "pcorr" + "a" * 500_000
    key = f'"({side}|{side})"'
    log = ("Exec   : pimpleFoam\n\nCreate time\n\nTime = 0.1\n\n"
           "GAMG:  Solving for p, Initial residual = 1, Final residual = 0.001, No Iterations 3\n")
    config = CaseConfig(linear_solvers={
        key: LinearSolverSpec(field=key, tolerance=0.02, rel_tol=0.0),
        '"pcorr.*"': LinearSolverSpec(field='"pcorr.*"', tolerance=0.02, rel_tol=0.0)})
    assert set(loose_solvers(config, log)) == {key}


#: Nested and empty loops, anchors inside loops: the shapes where a matcher's own loop handling
#: (an empty iteration, a repeat of a repeat) could part from `re`.
NESTED = ("(a*)*", "(a*)*b", "((a|)*b)*", "(|a)*", "(a?){2,}", "(a|b?)+", "(?:a*|b)*a", r"(\b)*a",
          "(^|a)*$", "(a|$)*", r"((?:a|\n)*\n?)*", "(?:(?:a?)*)*b", "(a{0,2}){2,3}",
          r"(?m)(^a$\n?)*", r"(?:\B|a)*", "((a*)*)*b?", "(a+?)+?b", "(?:a|ab|b)*", "(?s)(.?)*",
          "(?i)(A|b*)*", r"(?:a*\Z)*", "(?:$\n?)*a?", "(?:(?:^)*a)+", "(a??b??)*?")
#: Every name over "ab\n" of 0 to 5 characters.
SHORT = ["".join(t) for n in range(6) for t in itertools.product("ab\n", repeat=n)]


def test_nested_and_empty_loops_say_what_re_says_on_every_short_name():
    pairs = 0
    for p in NESTED:
        compiled = re.compile(p)
        for n in SHORT:
            assert wordre.fullmatch(p, n) is (compiled.fullmatch(n) is not None), (p, n)
            pairs += 1
    assert pairs == 24 * 364


@pytest.mark.parametrize(("pattern", "name"), [
    ("(?i)i", "İ"), ("(?i)İ", "i"), ("(?i)[é-ı]", "I"), ("(?i)s", "ſ"), ("(?i)k", "K"),
    ("(?ia)k", "K"), ("(?i)ǅ", "Ǆ"), ("(?i)σ", "ς"), ("(?i)[^k]", "K"), ("(?i)ß", "ẞ")])
def test_case_folding_is_res_own(pattern, name):
    """`re` folds case by its own tables (`İ` lowers to `i`, Kelvin's K is a k): the matcher's
    one-character tests are compiled by `re`, so they fold as it does."""
    assert wordre.fullmatch(pattern, name) is (re.fullmatch(pattern, name) is not None)


# ---------------------------------------------------------------- 1b. a set of names
# A boundaryField that groups its patches by name, `"(wing_upper|wing_lower|...)"`, is an
# alternation of plain names. Past some 42 names of ten characters its automaton has more than
# `MAX_STATES` states, so it was not evaluated, and a specification item for one of its patches
# was not checked by default, where 0.5.0's `re` judged it (the review of the 0.6.0 hard-proofing,
# 22 Oct 2026). Such a pattern is a set, and is matched as one.

#: The pieces a name is written with: plain characters, escaped metacharacters, code-point
#: escapes and characters `re` folds under IGNORECASE, each with the text it stands for.
_NAME_PIECES = {"a": "a", "b": "b", "A": "A", "_": "_", "1": "1", "W": "W", r"\.": ".",
                r"\|": "|", r"\(": "(", r"\-": "-", " ": " ", r"\n": "\n", r"\x41": "A",
                r"\u00e9": "é", "é": "é", "ſ": "ſ", "K": "K", "İ": "İ"}
#: What may surround the alternation: nothing, a group of each kind, flags that fold no case (a
#: set) and flags that fold it (the automaton, whose answer is compared as well).
_AROUND = ("{}", "({})", "(?:{})", "(?P<g>{})", "((?:{}))", "(?s:{})", "(?x){}", "(?a)({})",
           "(?i)(?-i:{})", "(?i){}", "(?i:{})", "({})?", "(?>{})", "({})x")
_SET_NAME_CHARS = "aAbB_1W .|(-\néſKİ"


def _alternation(r: random.Random) -> tuple[str, list[str]]:
    """(a pattern, the names its alternatives stand for), the alternatives sharing a prefix as
    patch groups do; one in seven has 40 to 70 names of 8 to 10 pieces, past `MAX_STATES`."""
    pieces = list(_NAME_PIECES)
    prefix = [r.choice(pieces) for _ in range(r.choice((0, 0, 1, 2, 3)))]
    many = r.random() < 1 / 7
    alternatives = [prefix + [r.choice(pieces) for _ in range(r.randint(8, 10) if many
                                                              else r.randint(0, 4))]
                    for _ in range(r.randint(40, 70) if many else r.randint(1, 6))]
    written = "|".join("".join(a) for a in alternatives)
    return (r.choice(_AROUND).format(written),
            ["".join(_NAME_PIECES[p] for p in a) for a in alternatives])


def _candidates(r: random.Random, meant: list[str]) -> list[str]:
    """Twenty names to ask: four of the pattern's own, each cut short, lengthened and with its
    case swapped, two of them joined, and three at random."""
    own = [r.choice(meant) for _ in range(4)]
    out = list(own)
    for m in own:
        out += [m[:-1], m + r.choice(_SET_NAME_CHARS), m.swapcase()]
    out.append(r.choice(meant) + r.choice(meant))
    out += ["".join(r.choice(_SET_NAME_CHARS) for _ in range(r.randint(0, 8))) for _ in range(3)]
    return out


SET_SEED, SET_PATTERNS = 2210, 1000
#: The set fuzz's denominators, of its 20,000 pattern-name pairs: those evaluated, by either path,
#: and of them those that match; the patterns matched as a set, their pairs and of those the ones
#: that match; and the set patterns whose automaton is past `MAX_STATES`, not evaluated before.
#: The 111 patterns not evaluated fold case, or hold an atomic group, past `MAX_STATES`.
SET_EVALUATED, SET_MATCHED = 17_780, 4_882
AS_SETS, AS_SET_PAIRS, AS_SET_MATCHED, PAST_STATES = 637, 12_740, 3_421, 59


def test_an_alternation_of_names_says_what_re_says_on_a_seeded_fuzz(monkeypatch):
    """Every answer, the set's or the automaton's, is `re.fullmatch`'s; on the patterns matched
    as sets the automaton, where it answers, agrees as well. A pattern that folds case, holds a
    group other than a plain one, or has anything after its alternation, keeps the automaton."""
    r = random.Random(SET_SEED)
    cases = []
    for _ in range(SET_PATTERNS):
        p, meant = _alternation(r)
        cases.append((p, _candidates(r, meant)))
    wordre._compile.cache_clear()
    evaluated = matched = set_pairs = set_matched = 0
    differ, sets = [], []
    for p, names in cases:
        compiled = re.compile(p)
        as_set = isinstance(wordre._compile(p), wordre._NameSet)
        if as_set:
            sets.append((p, names))
        for n in names:
            got = wordre.fullmatch(p, n)
            if got is None:
                continue
            evaluated += 1
            want = compiled.fullmatch(n) is not None
            matched += want
            set_pairs += as_set
            set_matched += as_set and want
            if got is not want:
                differ.append((p, n, got))
    # the same patterns through the automaton alone, as before the set
    monkeypatch.setattr(wordre, "_name_set", lambda parsed, flags: None)
    wordre._compile.cache_clear()
    past = 0
    try:
        for p, names in sets:
            if wordre._compile(p) is None:
                past += 1
                continue
            for n in names:
                if wordre.fullmatch(p, n) is not (re.fullmatch(p, n) is not None):
                    differ.append(("automaton", p, n))
    finally:
        monkeypatch.undo()
        wordre._compile.cache_clear()
    assert differ == []
    assert (evaluated, matched) == (SET_EVALUATED, SET_MATCHED)
    assert (len(sets), set_pairs, set_matched, past) == (AS_SETS, AS_SET_PAIRS, AS_SET_MATCHED,
                                                         PAST_STATES)


#: 46 names of ten characters, one of them the cavity's lid: an automaton of 551 states.
WINGS = ("movingWall", *(f"wing_{i:05d}" for i in range(45)))


def test_a_long_alternation_of_names_is_matched_and_names_its_block():
    from courant.spec.deck import block

    key = "(" + "|".join(WINGS) + ")"
    for name, want in (("movingWall", True), ("wing_00044", True), ("wing_0004", False),
                       ("wing_000440", False), ("fixedWalls", False), ("", False)):
        assert wordre.fullmatch(key, name) is want is (re.fullmatch(key, name) is not None)
    bf = f'"{key}" {{ a 1; }} fixedWalls {{ a 2; }} "(?:{"|".join(WINGS[1:])})" {{ a 3; }}'
    assert block(bf, "movingWall") == " a 1; "
    assert block(bf, "wing_00007") == " a 3; "           # the last declared that matches
    assert block(bf, "fixedWalls") == " a 2; "
    assert block(bf, "frontAndBack") is None


def test_a_set_of_names_is_charged_its_name_and_hostile_sets_stay_bounded():
    """A set costs the name's length, against the same budgets; a set too long to parse within
    them is not evaluated, and no shape of one raises or stalls."""
    key = "(" + "|".join(WINGS) + ")"
    with wordre.budget(1_000_000):
        assert wordre.fullmatch(key, "movingWall") is True
        with wordre.budget(11):                          # the pattern is paid for outside
            assert wordre.fullmatch(key, "wing_00001") is True             # 11 steps
            assert wordre.fullmatch(key, "wing_00002") is None             # none left
    out = _bounded(
        "import time\n"
        "from courant.case import wordre\n"
        "t = time.perf_counter()\n"
        "huge = '(' + '|'.join('n%07d' % i for i in range(200_000)) + ')'\n"
        "prefixed = '(' + 'a' * 15_000 + 'b|' + 'a' * 15_000 + 'c)'\n"
        "wide = '(' + '|'.join('a' * 30 + str(i) for i in range(1_000)) + ')'\n"
        "print(wordre.fullmatch(huge, 'n0000001'), wordre.fullmatch(prefixed, 'a' * 15_000 + 'c'),"
        " wordre.fullmatch(wide, 'a' * 30 + '7'))\n"
        "keys = ['(' + '|'.join('k%d_%08d' % (j, i) for i in range(50)) + ')' for j in range(3000)]\n"
        "hit, unsure = wordre.first_match(keys, 'k2999_00000049')\n"
        "print(hit, 0 < len(unsure) < 3000, time.perf_counter() - t < 20)", seconds=60)
    first, second = out.splitlines()
    assert first == "None True True"
    assert second == "None True True"


# ---------------------------------------------------------------- 2. bounded work
def _bounded(code: str, seconds: float = 60) -> str:
    """Run `code` in a fresh interpreter and fail, not hang, past `seconds`."""
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                           text=True, timeout=seconds)
    except subprocess.TimeoutExpired:
        pytest.fail(f"did not return within {seconds} s: case-written keywords stalled the gate")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_an_automaton_past_a_few_hundred_states_is_not_evaluated():
    """OpenFOAM's own patterns need tens of states (32 is the largest in the committed cases).
    `(.*){1600}x` asks for 4,803, which was evaluated, at about 5 ms a name."""
    assert 100 <= wordre.MAX_STATES <= 500
    assert wordre.fullmatch("(.*){1600}x", "U") is None
    assert wordre.fullmatch("(?:.*){100}x", "U") is False            # 301 states: evaluated
    # a class `re`'s compiler would walk 65,536 code points of, whatever the pattern's length
    assert wordre.fullmatch("[\x00-￿]", "a") is None
    assert wordre.fullmatch("(?i)[Ѐ-ӿ]+", "Жж") is True


def test_one_lookup_is_bounded_however_many_keywords_it_tries():
    """A dictionary lookup tries every keyword against one name, so it is bounded as a whole: a
    keyword it cannot pay for is not evaluated, and is returned as such, never a stall."""
    out = _bounded(
        "from courant.case import wordre\n"
        "keys = ['(?:.*){160}x%d' % i for i in range(3000)]\n"
        "hit, unsure = wordre.first_match(keys, 'epsilon')\n"
        "print(hit, len(unsure), unsure[0] > 0, list(unsure) == list(range(unsure[0], 3000)))",
        seconds=30)
    hit, unevaluated, some_evaluated, rest_unevaluated = out.split()
    assert (hit, some_evaluated, rest_unevaluated) == ("None", "True", "True")
    assert 0 < int(unevaluated) < 3000


def test_a_verify_is_bounded_however_many_lookups_it_makes():
    """Inside one `budget` (a verify, a preflight) every lookup draws on the same steps: once they
    are spent nothing more is evaluated, and the answer says so, never a guess."""
    out = _bounded(
        "from courant.case import wordre\n"
        "keys = ['(?:.*){160}x%d' % i for i in range(300)]\n"
        "with wordre.budget():\n"
        "    runs = [wordre.first_match(keys, 'U%d' % i) for i in range(400)]\n"
        "print(runs[0][1] != tuple(range(300)), all(r == (None, tuple(range(300)))"
        " for r in runs[-10:]), wordre.first_match(['(U|k)'], 'k'))", seconds=30)
    assert out == "True True (0, ())"


def test_the_first_match_names_every_keyword_before_it_that_was_not_evaluated():
    assert wordre.first_match(["(?=q)q", "(inlet|outlet)", "(?=x)x"], "inlet") == (1, (0,))
    assert wordre.first_match(["(inlet|outlet)", "(?=q)q"], "inlet") == (0, ())
    assert wordre.first_match(["(?=q)q", "wall"], "inlet") == (None, (0,))
    # a keyword that is a name, not a pattern, is compared as one, at no cost
    assert wordre.first_match(["div(phi,U)", "(U|k)"], "div(phi,U)",
                              pattern=lambda k: None if k.startswith("div") else k) == (0, ())


def test_the_spec_reader_looks_a_block_up_as_openfoam_does():
    """`dictionary::csearch`: the keyword whose text is the name, quoted or not (the hash table
    holds a keyword by its text), then the quoted patterns, the last declared first. Only a
    quoted key is a pattern: a word is a name (`keyType`), so `in.et` written unquoted does not
    name `inlet`, which 0.5.0's `re.fullmatch` on every key said it did (the final review of
    0.6.0, 21 Oct 2026)."""
    from courant.spec.deck import KeywordNotEvaluated, block

    assert block("in.et { a 1; }", "inlet") is None                    # a word is a name
    assert block('"in.et" { a 1; }', "inlet") == " a 1; "              # a quoted key, a pattern
    assert block("in.et { a 1; }", "in.et") == " a 1; "
    # the text first, quoted or not, then the patterns from the last
    assert block('"inlet" { a 1; } "in.*" { a 2; }', "inlet") == " a 1; "
    assert block('"in.*" { a 1; } "(inlet|outlet)" { a 2; } wall { a 3; }', "inlet") == " a 2; "
    assert block('"(inlet|outlet)" { a 2; } "in.*" { a 1; }', "inlet") == " a 1; "
    # a pattern not evaluated after the last match may be the one: the block cannot be told
    with pytest.raises(KeywordNotEvaluated):
        block('"in.*" { a 1; } "(?=i)inlet" { a 2; }', "inlet")
    assert block('"(?=i)inlet" { a 2; } "in.*" { a 1; }', "inlet") == " a 1; "
    # an unquoted word is never a pattern, so it is never left unevaluated either
    assert block("(?=i)inlet { a 2; } \"in.*\" { a 1; }", "inlet") == " a 1; "


def test_one_csearch_serves_the_spec_reader_and_the_relaxation_lookup():
    """There were two `dictionary::csearch`es and they disagreed: the relaxation lookup took a
    quoted `"U"` for a pattern, so `equations { "U" 0.7; ".*" 0.3; }` gave U 0.3, where v2406's
    hash table finds `"U"` by its text and the solver relaxes U by 0.7 (the review of the 0.6.0
    hard-proofing, 22 Oct 2026). One helper now, `wordre.csearch`, v2406's order: the key whose
    text is the name, quoted or not, then the patterns from the last declared."""
    from courant.verification import convergence
    from courant.verification.convergence import equation_relaxation

    assert wordre.csearch([("U", True), (".*", True)], "U") == ((0,), ())
    assert wordre.csearch([("U", False), ("U", True), (".*", True)], "U") == ((0, 1), ())
    assert wordre.csearch([("U.*", True), ("(U|k)", True), ("p", False)], "U") == ((1,), ())
    assert wordre.csearch([("U.*", False), ("p", True)], "U") == ((), ())       # a word
    assert wordre.csearch([("U.*", True), ("(?=U)U", True)], "U") == ((0,), (1,))
    assert wordre.csearch([("(?=U)U", True), ("U.*", True)], "U") == ((1,), ())
    assert not hasattr(convergence, "_dictionary_lookup")                     # no second one

    scopes = {'"U"': "equations", '".*"': "equations"}
    config = _steady({}, relaxation_factors={'"U"': 0.7, '".*"': 0.3}, relaxation_scopes=scopes)
    assert equation_relaxation(config, "U") == equation_relaxation(config, "Ux") == 0.7
    assert equation_relaxation(config, "k") == 0.3
    # an unquoted U and a quoted one are one entry in the hash table: the last written stands
    both = _steady({}, relaxation_factors={"U": 0.5, '"U"': 0.7, '".*"': 0.3},
                   relaxation_scopes={"U": "equations", **scopes})
    assert equation_relaxation(both, "U") == 0.7


def test_a_block_written_again_under_a_pattern_keeps_its_first_place(tmp_path):
    """v2406's `dictionary::add` merges a dictionary written again under a keyword the dictionary
    already holds into the first entry (`#inputMode merge`, `entryIO.C`): the entry keeps the
    first's place among the patterns (`patterns_`) and the first's keyword. `csearch` put the
    later copy first and returned its body alone, so `"inlet.*" { fixedValue (5 0 0) } ".*" {
    zeroGradient } "inlet.*" { value (10 0 0) }` gave inlet1 the value (10 0 0), where the
    solver's `findInPatterns` meets `".*"` first and runs zeroGradient: a spec item
    `inlet1 velocity = (10 0 0)` conformed on a run whose inlet was zeroGradient (the review of
    the 0.6.0 follow-ups, 22 Oct 2026). A value written again (`dictionaries=False`, the
    relaxation lookup) replaces the entry and is pushed to the front of the patterns instead,
    under the first's keyword."""
    from courant.spec.deck import block, read
    from courant.spec.model import SpecItem

    keys = [("inlet.*", True), (".*", True), ("inlet.*", True)]
    assert wordre.csearch(keys, "inlet1") == ((1,), ())
    assert wordre.csearch([(".*", True), ("inlet.*", True), (".*", True)], "inlet1") == ((1,), ())
    assert wordre.csearch([("in.*", True), ("wall", False), ("in.*", True)], "inlet1") == \
        ((0, 2), ())
    # the merged entry keeps the first's keyword: written first as a word, it is no pattern
    assert wordre.csearch([("in.*", False), ("in.*", True)], "inlet1") == ((), ())
    # a pattern not evaluated, merged into its first place, is no longer tried before the match
    assert wordre.csearch([("(?=i)inlet", True), ("in.*", True), ("(?=i)inlet", True)],
                          "inlet") == ((1,), ())
    assert wordre.csearch([("in.*", True), ("(?=i)inlet", True), ("in.*", True)],
                          "inlet") == ((0, 2), (1,))
    # values: each writing replaces the entry and goes to the front, the first's keyword kept
    assert wordre.csearch(keys, "inlet1", dictionaries=False) == ((0, 2), ())
    assert wordre.csearch([("U.*", True), (".*", True), ("U.*", False)], "U",
                          dictionaries=False) == ((0, 2), ())
    assert wordre.csearch([("U.*", False), (".*", True), ("U.*", True)], "U",
                          dictionaries=False) == ((1,), ())

    text = ('"inlet.*" { type fixedValue; value uniform (5 0 0); } ".*" { type zeroGradient; } '
            '"inlet.*" { value uniform (10 0 0); }')
    assert block(text, "inlet1") == " type zeroGradient; "
    assert block('".*" { a 1; } "in.*" { b 2; } ".*" { c 3; }', "inlet1") == " b 2; "
    assert block('"in.*" { a 1; } ".*" { b 2; } "in.*" { c 3; }', "outlet") == " b 2; "
    assert block('"in.*" { a 1; } wall { } "in.*" { c 3; }', "inlet1") == " a 1; \n c 3; "

    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U").write_text(
        "FoamFile { class volVectorField; object U; }\ninternalField uniform (0 0 0);\n"
        f"boundaryField\n{{\n    {text}\n}}\n")
    found = read(tmp_path, SpecItem(name="inlet1 velocity", file="0/U", patch="inlet1",
                                    value=[10, 0, 0]))
    assert found.value is None and found.note == "no entry 'value'"


def test_the_relaxation_a_case_writes_under_a_quoted_name_is_the_one_read(tmp_path):
    """The same dictionary written in `system/fvSolution` and read by the case reader, whose
    keys keep their quotes."""
    from courant.verification.convergence import equation_relaxation

    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
        "relaxationFactors\n{\n    equations\n    {\n        \"U\" 0.7;\n        \".*\" 0.3;\n"
        "    }\n}\n")
    config = CaseConfig.from_foam(str(tmp_path))
    assert config.relaxation_factors == {'"U"': 0.7, '".*"': 0.3}
    assert equation_relaxation(config, "Ux") == 0.7


def test_a_block_a_pattern_may_name_is_noted_without_the_pattern(tmp_path):
    """The note on an item whose block a pattern not evaluated may name quoted up to 40
    characters of the keyword, which is the case's text, into the verdict and the support
    bundle (the review of the 0.6.0 hard-proofing). It names the file that holds a quoted
    keyword, and nothing the case wrote."""
    from courant.spec.deck import block, read
    from courant.spec.model import SpecItem

    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U").write_text(
        "FoamFile { class volVectorField; object U; }\ninternalField uniform (0 0 0);\n"
        "boundaryField\n{\n    \"(?!acmeKeyQ)lid\" { type fixedValue; value uniform (1 0 0); }\n}\n")
    found = read(tmp_path, SpecItem(name="U at lid", file="0/U", patch="lid", value=[1, 0, 0]))
    assert found.value is None
    assert found.note == ("a quoted keyword in 0/U is a pattern the reader does not evaluate, "
                          "and it may be the one that names this block")
    with pytest.raises(Exception) as raised:
        block('"(?!acmeKeyQ)lid" { a 1; }', "lid")
    assert "acmeKeyQ" not in str(raised.value)


def _case(tmp_path: Path, name: str, source: Path = CAVITY) -> Path:
    case = tmp_path / name
    shutil.copytree(source, case)
    for gz in case.glob("log.*.gz"):
        with gzip.open(gz, "rb") as src, open(case / gz.name.removesuffix(".gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    return case


def test_many_keywords_in_a_boundary_field_do_not_stall_verify_with_a_spec(tmp_path):
    """Two thousand keywords of 4,803 states each, after the one that groups movingWall: each
    spec item for movingWall tried every one (about 15 ms each). It returns at once; the keywords
    it cannot evaluate come after the match, so the item is not checked (VER-5.9)."""
    from typer.testing import CliRunner

    from courant.cli.main import app

    ref = _case(tmp_path, "reference")
    spec = tmp_path / "spec.json"
    assert CliRunner().invoke(app, ["spec", "init", str(ref), "-o", str(spec)]).exit_code == 0
    agent = _case(tmp_path, "agent")
    u = agent / "0" / "U"
    keys = "".join(f'    "(.*){{1600}}k{i}"\n    {{\n        type zeroGradient;\n    }}\n'
                   for i in range(2000))
    text = u.read_text().replace("    movingWall\n", '    "(movingWall|lid)"\n', 1)
    end = text.rindex("}")                               # boundaryField's own, the file's last
    u.write_text(text[:end] + keys + text[end:])
    out = _bounded(
        "import json\n"
        "from courant.verify import from_path, verify_inputs\n"
        "from courant.verify.ingest import with_spec\n"
        f"v = json.loads(verify_inputs(with_spec(from_path({str(agent)!r}), {str(spec)!r}))"
        ".to_json())\n"
        "items = {i['name']: i for i in v['conformance']['items']}\n"
        "print(v['verdict'], items['U at movingWall'].get('conforms'),"
        " items['U at movingWall'].get('note'))", seconds=60)
    verdict, conforms, note = out.split(" ", 2)
    assert (verdict, conforms) == ("insufficient", "None")
    assert "pattern the reader does not evaluate" in note


def test_a_patch_grouped_with_forty_five_others_is_checked_by_default(tmp_path):
    """The cavity's lid written in a group of 46 names of ten characters: the item for it is
    found through the group and conforms, and the verdict is the reference's own, as in 0.5.0.
    Past `MAX_STATES`, the group was not evaluated and the accept was insufficient (VER-5.9)."""
    from typer.testing import CliRunner

    from courant.cli.main import app
    from courant.verify import from_path, verify_inputs
    from courant.verify.ingest import with_spec

    ref = _case(tmp_path, "reference")
    spec = tmp_path / "spec.json"
    assert CliRunner().invoke(app, ["spec", "init", str(ref), "-o", str(spec)]).exit_code == 0
    agent = _case(tmp_path, "agent")
    u = agent / "0" / "U"
    u.write_text(u.read_text().replace("    movingWall\n", f'    "({"|".join(WINGS)})"\n', 1))

    def judged(case: Path) -> dict:
        return json.loads(verify_inputs(with_spec(from_path(str(case)), str(spec))).to_json())

    want, got = judged(ref), judged(agent)
    items = {i["name"]: i for i in got["conformance"]["items"]}
    assert items["U at movingWall"]["conforms"] is True
    assert (got["verdict"], got["rule"]) == (want["verdict"], want["rule"]) == ("accept", "VER-5.6")


def test_many_residual_control_keywords_do_not_stall_verify_under_next(tmp_path):
    """Two thousand five hundred keywords of 483 states each in pitzDaily's residualControl,
    before its own entries: every field's tolerance tried every one, several times a verify
    (`judge_residuals` is asked by three checks), each a recompile once they outnumbered the
    matcher's cache. Bounded per verify, it returns; what it did not evaluate holds a field to
    1e-2 at most, no looser than its own entry, so the verdict is the shipped case's."""
    case = _case(tmp_path, "pitz", PITZ)
    fv = case / "system" / "fvSolution"
    keys = "".join(f'        "(?:.*){{160}}x{i}" 1e-2;\n' for i in range(2500))
    text = fv.read_text()
    anchor = "    residualControl\n    {\n"
    assert anchor in text
    fv.write_text(text.replace(anchor, anchor + keys, 1))
    out = _bounded(
        "import json\n"
        "from courant.verify import verify_path\n"
        f"v = verify_path({str(case)!r}, checks='next')\n"
        "print(v.verdict, v.rule)", seconds=90)
    assert out == "accept VER-5.6"


# ---------------------------------------------------------------- 3. never skipped
def _steady(control: dict, **kw) -> CaseConfig:
    return CaseConfig(residual_control={"SIMPLE": control}, solver="simpleFoam",
                      ddt_scheme="steadyState", **kw)


def test_a_residual_control_keyword_not_evaluated_before_the_match_holds_the_strictest():
    """`solutionControl` applies the first entry that matches. A lookahead before `".*"` may be
    that entry, so U is held to the stricter of the two, as `re` (and ESI) reads it, never to
    the `".*"` behind it."""
    from courant.verification.convergence import (
        RESIDUAL_TARGET,
        declared_tolerance,
        residual_bar,
    )

    config = _steady({'"(?=U)U"': 1e-6, '".*"': 1e-2})
    assert declared_tolerance(config, "Ux") == 1e-6 and residual_bar(config, "Ux") == 1e-6
    # after the first match it cannot matter
    after = _steady({'".*"': 1e-2, '"(?=U)U"': 1e-6})
    assert declared_tolerance(after, "U") == 1e-2
    # nothing matches for certain: the flat bar is one of the outcomes, so the stricter wins
    alone = _steady({'"(?=U)U"': 5e-3})
    assert declared_tolerance(alone, "U") == RESIDUAL_TARGET


def test_a_relaxation_keyword_not_evaluated_after_the_last_match_gives_the_strictest():
    """`dictionary::csearch` tries the last pattern declared first: a lookahead after `"U.*"` may
    be the one it finds, so U's equation relaxation is the smaller of the two."""
    from courant.verification.convergence import equation_relaxation

    config = _steady({}, relaxation_factors={'"U.*"': 0.5, '"(?=U)U"': 0.05},
                     relaxation_scopes={'"U.*"': "equations", '"(?=U)U"': "equations"})
    assert equation_relaxation(config, "Ux") == 0.05
    before = _steady({}, relaxation_factors={'"(?=U)U"': 0.05, '"U.*"': 0.5},
                     relaxation_scopes={'"(?=U)U"': "equations", '"U.*"': "equations"})
    assert equation_relaxation(before, "U") == 0.5


def _events_058():
    from courant.watcher.parser import LogParser

    with gzip.open(HEALTHY_058, "rt") as fh:
        return list(LogParser(run_id="058").parse_text(fh.read()))


def _config_058(control: dict) -> CaseConfig:
    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"]["residual_control"] = {"SIMPLE": control}
    return CaseConfig.model_validate(meta["config"])


def test_a_tolerance_only_a_keyword_not_evaluated_may_give_is_never_said_to_be_declared():
    """With only a keyword the matcher does not evaluate before it, and none certain to match, a
    field may have that keyword's tolerance or none: the case is not said to declare one (the
    review of the 0.6.0 hard-proofing found `field_declared` and "its residualControl" said of
    the flat bar). The healthy 058 run: p ends at 1.02e-3 on its first solve."""
    from courant.diagnosis.checkers import CheckContext, ResidualAnalyzer
    from courant.verification.checks import ConvergenceCheck, VerifyContext
    from courant.verification.convergence import judge_residuals

    events = _events_058()
    flat = _config_058({'"(?=p)p"': 5e-3})
    judged = judge_residuals(events, flat, "next")
    # p is above the flat bar, which applies if the keyword does not name it: not converged
    assert judged.over == ("p",) and judged.field == "p" and judged.bar == 1e-3
    assert not judged.declared and not judged.field_declared and not judged.field_unsure
    assert judged.bar_words() == "the target 1e-03"
    check = ConvergenceCheck().run(VerifyContext(run_id="r", events=events, config=flat,
                                                 checks="next"))
    assert "residualControl" not in check.detail and "(target 1e-03)" in check.detail
    # a tolerance stricter than the flat bar, which the keyword may give: judged on it, and
    # said to be the keyword's, not the case's
    stricter = _config_058({"p": 1e-2, '"(k|epsilon|omega|f|v2)"': 1e-3, '"(?=U)U"': 1e-4})
    judged = judge_residuals(events, stricter, "next")
    assert judged.declared and set(judged.over) == {"Ux", "Uy"} and judged.field == "Uy"
    assert not judged.field_declared and judged.field_unsure and judged.bar == 1e-4
    assert judged.bar_words() == ("0.0001, which a residualControl keyword the reader does not "
                                  "evaluate may set")
    finding = next(f for f in ResidualAnalyzer().run(CheckContext(
        run_id="r", events=events, config=stricter, checks="next"))
        if f.id == "residual.not_converged")
    assert finding.field == "Uy" and finding.threshold == 1e-4
    assert "its residualControl" not in finding.detail
    assert "which a residualControl keyword the reader does not evaluate may set" in finding.detail


def _lts(control: dict) -> CaseConfig:
    """An OpenFOAM.org foamRun case marching in local time steps: under `next` it is judged on its
    final residuals (VER-19) and its fields on their last solve, but for one a tolerance applies
    to, read at its first as residualControl reads it."""
    return CaseConfig(solver="foamRun", ddt_scheme="localEuler",
                      residual_control={"PIMPLE": control})


@pytest.mark.parametrize(("first", "last"), [(5e-4, 2e-3), (2e-3, 5e-4), (5e-4, 5e-4),
                                             (1e-3, 1e-3), (8e-3, 4e-4), (2e-2, 2e-2),
                                             (2e-2, 5e-3), (5e-3, 2e-2)])
@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 6e-4])
def test_a_tolerance_a_keyword_may_give_never_accepts_what_a_certain_reading_rejects(
        first, last, tolerance):
    """The keyword may give p its tolerance (read at the first solve, met strictly) or none (the
    flat bar, read at the last solve on a local-time-stepping run): the judgement rejects when
    either reading would. Taken as declared, p solved at 5e-4 then 2e-3 was accepted against the
    flat 1e-3 its last solve exceeds.

    The grade too: past `RESIDUAL_LOOSE` a residual is MAJOR whatever the case says, on either
    reading. The worst residual was the deciding reading's alone, so p at 2e-2 then 5e-3 under
    a tolerance of 1e-2 was decided on the flat reading (5e-3 is further over 1e-3 than 2e-2 is
    over 1e-2), its worst 5e-3 was inside LOOSE, and `ConvergenceCheck` graded MINOR (B, VER-5.4)
    a run the named reading grades MAJOR (C, VER-5.3) (the review of the 0.6.0 follow-ups)."""
    from courant.verification.checks import ConvergenceCheck, VerifyContext
    from courant.verification.convergence import RESIDUAL_TARGET, judge_residuals
    from courant.verification.grade import VSeverity
    from tests.unit.test_first_solve import _events, nonorth_log

    events = _events(nonorth_log(last=(("Ux", 1e-5, None), ("p", first, last))))
    configs = {"unsure": _lts({'"(?=p)p"': tolerance}), "named": _lts({"p": tolerance}),
               "not": _lts({})}
    unsure, if_named, if_not = (judge_residuals(events, c, "next") for c in configs.values())
    assert if_named.field_declared and not if_not.declared               # the two readings
    assert if_named.final["p"] == pytest.approx(first) and if_not.final["p"] == pytest.approx(last)
    assert ("p" in unsure.over) is ("p" in if_named.over or "p" in if_not.over)
    assert not unsure.declared and not unsure.field_declared
    assert unsure.bars["p"] in (min(tolerance, 1e-2), RESIDUAL_TARGET)
    assert unsure.worst == max(if_named.worst, if_not.worst)
    grade = {name: ConvergenceCheck().run(VerifyContext(run_id="r", events=events, config=c,
                                                        checks="next")).severity
             for name, c in configs.items()}
    assert grade["unsure"].rank == max(grade["named"].rank, grade["not"].rank)
    if max(first, last) > 1e-2:
        assert grade["unsure"] is VSeverity.MAJOR


def _verify_058(control: dict, tmp_path: Path, checks: str):
    from courant.verify.ingest import from_log
    from courant.verify.pipeline import verify_inputs_v1

    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"]["residual_control"] = {"SIMPLE": control}
    side = tmp_path / "c10k.simpleFoam_pitzDaily.058.healthy.case.json"
    side.write_text(json.dumps(meta))
    return verify_inputs_v1(from_log(HEALTHY_058, side).model_copy(update={"checks": checks}))


def test_the_review_case_a_lookahead_before_a_looser_entry_is_not_converged(tmp_path):
    """The healthy 058 run (U ends at 1.0e-3 on Uy) meets `U 1e-3`. With `"(?=U)U" 1e-6;` first,
    OpenFOAM holds U to 1e-6, which it did not meet: not converged, as when `re` matched it."""
    control = {'"(?=U)U"': 1e-6, "p": 1e-2, "U": 1e-3, '"(k|epsilon|omega|f|v2)"': 1e-3}
    baseline = {"p": 1e-2, "U": 1e-3, '"(k|epsilon|omega|f|v2)"': 1e-3}
    assert _verify_058(baseline, tmp_path, "next").verdict == "TRUSTWORTHY"
    v = _verify_058(control, tmp_path, "next")
    assert v.verdict == "NOT_TRUSTWORTHY"
    assert "residual.not_converged" in {r.code for r in v.reasons}
    # the default mode holds every field to the flat bar and reads no keyword: v0.5.0's verdict
    default = _verify_058(control, tmp_path, "default")
    assert default.verdict == _verify_058(baseline, tmp_path, "default").verdict
