"""Read one named value out of an OpenFOAM case deck, for conformance (PRD 2.0 P2-SPEC-2).

A small reader on purpose, separate from `CaseConfig`:

- **It must read every file a specification can name**, including the shapes foamlib
  refuses: `#include`d values, `$macro` substitutions, patches grouped under a regex key,
  dimensioned entries. `CaseConfig` is a fixed view of the settings the checkers need; a
  specification names arbitrary locations.
- **It must never raise.** A value it cannot read comes back as `None` with a note saying
  why, and the conformance check reports the item as not checked. A reader that crashes on
  an unexpected file would turn a reader limitation into a verdict.
- **It reads the deck as set up, not as rewritten by the run.** `setFields` and friends
  rewrite `0/` during a solve; OpenFOAM's convention keeps the pristine initial conditions
  in `0.orig/`. A boundary value that has become a per-face list in `0/` is read from
  `0.orig/` when it exists (RM-18 found this on `buoyantPimpleFoam/hotRoom`).

Deterministic, read-only, no subprocess, no network (INV-5).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from courant.case import reads

Value = float | list[float] | str

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_DIMS = re.compile(r"^\s*\[([^\]]*)\]")
_MACRO = re.compile(r"\$\{?([A-Za-z_][\w.]*)\}?")
#: The `#include` alone, as 0.6.0's macro lookup follows it (`_resolve_macros`, the spec reader's
#: lookup under the default `--checks`). `INCLUDE_DIRECTIVES` is what the solver follows.
_INCLUDE = re.compile(r'(?m)^\s*#include\s+"([^"]+)"')
_HEX = re.compile(r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)")


@dataclass(frozen=True)
class Found:
    """What the reader found for one item."""

    value: Value | None
    file: str                                  # the file actually read, relative to the case
    note: str | None = None                    # why `value` is None
    dimensions: tuple[float, ...] | None = None
    #: How `value` was obtained when it is not a single written value, e.g. "most common of
    #: 400 face values (380)". None for an ordinary entry.
    basis: str | None = None
    #: Every face value, when `value` was read from a per-face list (floats, or lists of three).
    #: A bound is checked on all of them, never on the mode (SPEC-12).
    faces: tuple | None = None
    #: The reader parsed the file completely and the named location is not in it (a renamed
    #: patch, a missing key). That is a nonconformance of the case (VER-15, VER-5.7), where a
    #: value the reader could not parse is a gap in what was checked (VER-5.9).
    missing: bool = False


# ---------------------------------------------------------------- text helpers
def strip_comments(text: str) -> str:
    text = _without_block_comments(text)
    text = re.sub(r"//[^\n]*", "", text)
    # `${name}` is `$name`: its braces are not a block's, and counted as one they hid every
    # entry after them (TST-14's fuzzer found it). Followed by a word character the braces
    # delimit the name, so it is left alone and stays unresolved rather than misread.
    return _BRACED_MACRO.sub(r"$\1", text)


def _without_block_comments(text: str) -> str:
    """`re.sub(r"/\\*.*?\\*/", " ", text, flags=re.DOTALL)`, the same string, in time linear in
    the text's length: each `/*` with a `*/` after it, through the first such `*/`, as one space.
    The pattern tried every `/*` against the rest of the text, so a file of 500,000 unclosed
    `/*x` took time in the square of its length and held `courant verify <case>` (the review of
    hp6/final). With no `*/` after a `/*` there is none after any later one either, so the scan
    stops there, as the pattern then matches nothing more."""
    out: list[str] = []
    i = 0
    while True:
        start = text.find("/*", i)
        end = text.find("*/", start + 2) if start >= 0 else -1
        if end < 0:
            break
        out += [text[i:start], " "]
        i = end + 2
    out.append(text[i:])
    return "".join(out)


_BRACED_MACRO = re.compile(r"\$\{([A-Za-z_][\w.]*)\}(?![\w.])")
#: A directive in a value (`#calc`, `#eval`, `#codeStream`, `#include...`): its result is the
#: solver's to compute, so the reader reports the value unreadable rather than guess it.
_DIRECTIVE = re.compile(r"(?:^|[\s(\"])#[A-Za-z]")


def _depths(text: str) -> list[int]:
    """Brace depth at every character (the depth *before* the character)."""
    out, d = [], 0
    for ch in text:
        out.append(d)
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
    return out


def _body_after(text: str, open_index: int) -> str | None:
    """The text between the brace at `open_index` and its match."""
    close = _block_end(text, open_index)
    return None if close is None else text[open_index + 1:close]


def _block_end(text: str, open_index: int) -> int | None:
    """The index of the brace that closes the one at `open_index`."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


#: A block starts a line or follows a `;`, `{` or `}`: `inlet { ... } outlet { ... }` is two blocks.
_BLOCK_HEAD = re.compile(r'(?m)(?:^|(?<=[;{}]))[ \t]*(?:"([^"\n]*)"|([^\s{};"]+))\s*\{')


def _quoted_blocks(text: str) -> list[tuple[str, bool, str]]:
    """Every top-level `key { ... }` in `text`, as (key without its quotes, whether it was
    quoted, body), in order."""
    depth = _depths(text)
    out = []
    for m in _BLOCK_HEAD.finditer(text):
        if depth[m.start()] != 0:
            continue
        body = _body_after(text, m.end() - 1)
        if body is not None:
            quoted = m.group(1) is not None
            out.append((m.group(1) if quoted else m.group(2), quoted, body))
    return out


def blocks(text: str) -> list[tuple[str, str]]:
    """Every top-level `key { ... }` in `text`, as (key, body), in order. Quotes are kept off."""
    return [(key, body) for key, _, body in _quoted_blocks(text)]


class KeywordNotEvaluated(ValueError):
    """A keyword pattern decides which block a name finds and is not one `wordre` evaluates (a
    backreference, a lookaround, or larger than its bound): the block cannot be told (VER-5.9).
    Its message quotes nothing the case wrote: the keyword is the case's text, and the note made
    from it reaches the verdict and the support bundle (`not_evaluated`)."""

    keywords: tuple[str, ...]           # set where raised (`block`); an annotation only


def not_evaluated(rel: str, included_by: str | None = None) -> str:
    """The note on a value whose block a keyword pattern not evaluated may name, in file `rel`
    (which `included_by` `#include`s, when the keyword came in through one): the file, never the
    keyword (the review of the 0.6.0 hard-proofing, 22 Oct 2026, found 40 characters of it
    quoted)."""
    where = rel if included_by is None else f"{rel}, which {included_by} includes,"
    return (f"a quoted keyword in {where} is a pattern the reader does not evaluate, and it may "
            "be the one that names this block")


def block(text: str, name: str, found: list[tuple[str, bool, str]] | None = None) -> str | None:
    """The body of the top-level block `name`, matching OpenFOAM's lookup
    (`dictionary::csearch`, `courant.case.wordre.csearch`). `found`: the text's own
    `_quoted_blocks`, where the caller has them, so a text looked up by many names is scanned
    once (`courant.case.reynolds._boundary`, one lookup a patch).

    A key whose text is `name` wins, quoted or not (OpenFOAM's hash table holds a keyword by its
    text); otherwise the last declared quoted key that matches `name` as a regular expression,
    which is how a boundary condition grouped as `"(inlet|outlet)"` is found. A pattern written
    again is merged into its first writing and keeps that place (`dictionary::add`): in
    `"in.*" {..} ".*" {..} "in.*" {..}`, `".*"` is the later one. Only a quoted key is a
    pattern (OpenFOAM's `keyType`: a word is a name): 0.5.0 matched every key as one, so an
    unquoted `in.et` named `inlet` (the final review of 0.6.0, 21 Oct 2026). The keys are the
    case's, so they are matched with bounded work (`courant.case.wordre`, one lookup's budget):
    on `"(.*)*!"` Python's `re` doubled its time with each character of the patch name, and held
    the interpreter's lock while it did (hard-proofing audit and the final review, 21 Oct 2026).
    A pattern not evaluated that is tried before the one that matches raises
    `KeywordNotEvaluated`, because it might be the one OpenFOAM uses.

    A block written more than once is merged, as OpenFOAM's default `#inputMode merge` does:
    the bodies are read as one, in order, so each repeated entry takes its last definition
    (`entry`) and each repeated inner block merges in turn.
    """
    from courant.case import wordre

    found = _quoted_blocks(text) if found is None else found
    hits, unsure = wordre.csearch([(key, quoted) for key, quoted, _ in found], name,
                                  dictionaries=True)
    if unsure:
        exc = KeywordNotEvaluated("a quoted keyword is a pattern the reader does not evaluate, "
                                  "and it may be the one that names this block")
        exc.keywords = tuple(found[i][0] for i in unsure)   # never in the message: case text
        raise exc
    return "\n".join(found[i][2] for i in hits) if hits else None


def _top_level(text: str) -> str:
    """`text` with the inside of every nested block blanked out, braces kept.

    Keeping the braces is what stops a block header from being read as an entry: blanked to
    `water { }`, the header cannot match `key value;`.
    """
    depth = _depths(text)
    return "".join(ch if (depth[i] == 0 or (ch == "}" and depth[i] == 1))
                   else ("\n" if ch == "\n" else " ")
                   for i, ch in enumerate(text))


def entry(text: str, key: str) -> str | None:
    """The raw value of the top-level entry `key` (without the trailing `;`).

    An entry starts a line or follows a `;`, `{` or `}`: OpenFOAM allows several entries on
    one line, as in `inlet { type fixedValue; value uniform 1; }`.

    A key written more than once takes its LAST definition: in OpenFOAM's default
    `#inputMode merge`, a repeated primitive entry overwrites the earlier one. (The first was
    read before 11 Oct 2026: a case that set `nu` twice was compared on the value the solver
    discarded.) `_read_one` refuses the cases where a directive changes that rule.
    """
    found = _entries(text, key)
    return found[-1] if found else None


#: What ends an entry's value, and what an entry may start after (with a line's start).
_DELIMITER = re.compile(r"[;{}]")
_ENTRY_START = re.compile(r"[\n;{}]")


def _entries(text: str, key: str) -> list[str]:
    """Every top-level definition of `key`, in order: each match of `(?m)(?:^|(?<=[;{}]))[ \\t]*
    <key>\\s+([^;{}]*);` in `_top_level(text)`, its value with its whitespace runs made one space
    and stripped, the same list the pattern gives, in time linear in the text's length.

    The pattern scanned from every line's start to the next delimiter, so `key` written on
    300,000 lines with no `;` after them (inside a string the case writes, which `_top_level`
    keeps) took time in the square of the text's length and held `courant verify <case>` (the
    review of hp6/final). Here the next delimiter is found once for every start before it: the
    starts are taken in order, so the one found for an earlier start is the next for a later one
    until the scan passes it. A key that is empty or begins with a blank, which no dictionary
    writes, is matched by the pattern itself, whose `[ \\t]*` may then give back blanks."""
    t = _top_level(text)
    if not key or key[0] in " \t":
        return [re.sub(r"\s+", " ", m.group(1)).strip() for m in re.finditer(
            rf"(?m)(?:^|(?<=[;{{}}]))[ \t]*{re.escape(key)}\s+([^;{{}}]*);", t)]
    n, k = len(t), len(key)
    out: list[str] = []
    resume = 0                           # a match ends at its `;`; the next is looked for after
    found = -1                           # the next delimiter at or after the last asked position

    def delimiter(at: int) -> int:
        nonlocal found
        if found < at:
            m = _DELIMITER.search(t, at)
            found = m.start() if m else n
        return found

    starts = (m.end() for m in _ENTRY_START.finditer(t))
    for start in (0, *starts) if n else ():
        if start < resume:
            continue
        q = start
        while q < n and t[q] in " \t":
            q += 1
        r = q + k
        if not t.startswith(key, q) or r >= n or not t[r].isspace():
            continue
        end = delimiter(r)
        if end < n and t[end] == ";":
            out.append(re.sub(r"\s+", " ", t[r:end]).strip())
            resume = end + 1
    return out


#: `#inputMode` other than the default `merge`: a repeated key no longer means "the last one".
_INPUT_MODE = re.compile(r"#inputMode\s+(?!merge\b|default\b)(\w+)")


def _entry_directive(text: str, name: str) -> str | None:
    """A per-entry directive (`#default`, `#overwrite`, `#merge`, `#warn`, `#error`, `#remove`)
    naming `name` anywhere in the file, which this reader does not follow."""
    m = re.search(rf"#(default|overwrite|merge|warn|error|remove)\b[ \t(\"]*"
                  rf"(?:[^\n;]*\s)?\"?{re.escape(name)}\b", text)
    return m.group(1) if m else None


def _repeated(text: str, name: str) -> bool:
    """Whether `name` is written more than once at the top level of `text`, as an entry or as
    a literal block."""
    return len(_entries(text, name)) + sum(1 for k, _ in blocks(text) if k == name) > 1


def repeated_under_input_mode(text: str, names: tuple[str, ...]) -> str | None:
    """The `#inputMode` other than merge under which a name on the way to `names` is written
    more than once in `text`, or None: the rule by which `_read_one` refuses such a value, for the
    case reader (TRD VER-3b)."""
    mode = _INPUT_MODE.search(text)
    if mode is None:
        return None
    scope: str | None = text
    for name in names:
        if scope is None:
            break
        if _repeated(scope, name):
            return mode.group(1)
        try:
            scope = block(scope, name)
        except KeywordNotEvaluated:
            break                       # the block cannot be told, as when it is not found
    return None


def numbers(raw: str) -> list[float]:
    return [float(x) for x in _NUM.findall(raw)]


# ---------------------------------------------------------------- value parsing
#: The name lookups one value's `$name`s stand for. A bound on a hang, not on a value: 0.7.0 made
#: each of them, at 33 microseconds on 1 KB of dictionary (a 1,048,576-lookup fan-out read in
#: 34.6 s), so this many held it for 37 minutes. No value of any length reaches it by being long:
#: one `$list` is one lookup (security review 0.7.0, SEC070-17). Past it the value is unread.
MAX_MACRO_LOOKUPS = 2**26


def _resolve_macros(raw: str, text: str, path: Path, case_root: Path, depth: int = 0) -> str | None:
    """Substitute `$name` from the same file, or from a file it `#include`s. None if not found,
    or past `MAX_MACRO_LOOKUPS` or `MAX_REPEATED_CHARS`."""
    return _resolved(raw, text, path, case_root, depth, {}, [0])[0]


def _resolved(raw: str, text: str, path: Path, case_root: Path, depth: int, memo: dict,
              repeated: list[int]) -> tuple[str | None, int]:
    """(`_resolve_macros`' value, the lookups it stands for). Each name is looked up once per
    level (`memo`; pure in the name and the level, so the value is the one looking it up each
    time gives), and the lookups are still counted, so the bound does not move with the memo.
    `repeated`: the characters substituted for a name already substituted at its level."""
    if "$" not in raw:
        return raw, 0
    if depth > 4:
        return None, 0

    def lookup(name: str) -> str | None:
        defined = _entries(text, name)
        if len(defined) > 1:
            # `$name` expands to the definition before its use, and this reader does not track
            # positions: a macro defined twice stays unresolved rather than read wrong.
            return None
        v = defined[0] if defined else None
        if v is not None:
            return v
        for inc in _INCLUDE.findall(text):
            inc_text = _include_text(path, inc, case_root)
            if inc_text is not None:                     # not read: the macro stays unresolved
                v = entry(inc_text, name)
                if v is not None:
                    return v
        return None

    out, lookups = raw, 0
    names = _MACRO.findall(raw)
    if not names:
        return None, 0                                  # a scoped `$:a.b` or `$..x` reference
    for name in names:
        first = (name, depth) not in memo
        if first:
            v = lookup(name)
            found, below = (None, 0) if v is None else _resolved(v, text, path, case_root,
                                                                 depth + 1, memo, repeated)
            memo[name, depth] = (found, below + 1)
        v, cost = memo[name, depth]
        lookups += cost
        if v is None:
            return None, lookups
        if not first:
            repeated[0] += len(v)
        if lookups > MAX_MACRO_LOOKUPS or repeated[0] > MAX_REPEATED_CHARS:
            return None, lookups
        out = re.sub(rf"\$\{{?{re.escape(name)}\}}?", lambda _m, v=v: v, out, count=1)
    return (None if "$" in out else out), lookups


# ---------------------------------------------------------------- the dictionary as the solver reads it
#: OpenFOAM's tags for an include's directory (`<case>/...`), and `$FOAM_CASE`, which every
#: install sets to the case.
_INCLUDE_TAGS = {"<case>": "", "$FOAM_CASE": "", "${FOAM_CASE}": "", "<system>": "system",
                 "<constant>": "constant"}
#: The includes of a file the solver reads from the case: `#include`, and `#includeIfPresent` and
#: `#sinclude`, which it skips when the file is absent. `#includeEtc`, `#includeFunc` and
#: `#includeModel` name files of the OpenFOAM installation, not of the case: not followed. The one
#: list, for both readers (`expand`; the case reader's `_foam_reader._FOLLOWED_INCLUDES` is it).
INCLUDE_DIRECTIVES = ("#include", "#includeIfPresent", "#sinclude")
_INCLUDES = re.compile(r'#(' + "|".join(d[1:] for d in INCLUDE_DIRECTIVES)
                       + r')[ \t]+"([^"\n]*)"')
_HEADER = re.compile(r"(?m)^[ \t]*FoamFile\s*\{")


def include_target(path: Path, inc: str, case_root: Path) -> tuple[Path | None, str | None]:
    """(the path an `#include "inc"` in the file at `path` names, None) when it is a file inside
    the case; (None, why) otherwise. Relative to the including file's directory, as OpenFOAM
    reads it, or to the case for `<case>`, `<system>`, `<constant>` and `$FOAM_CASE`. Nothing
    is opened. The path is lexical (VER-14 notes the path used, not its resolution)."""
    for tag, sub in _INCLUDE_TAGS.items():
        if inc.startswith(tag + "/"):
            target = case_root / sub / inc[len(tag) + 1:]
            break
    else:
        target = path.parent / inc
    try:
        target.resolve().relative_to(case_root.resolve())
        is_file = target.resolve().is_file()
    except RuntimeError:                                # a link loop (`Path.resolve`, 3.11)
        return None, _LINK_LOOP
    except (ValueError, OSError):                       # outside the case, or no path at all
        return None, "the file is outside the case"     # (a NUL byte: TST-14's fuzzer)
    return (target, None) if is_file else (None, _NOT_IN_CASE)


#: `include_target`'s reason for a target inside the case that is not a file there.
_NOT_IN_CASE = "no such file in the case"
#: Its reason for a path with a link that resolves to itself: never a traceback, and not "outside".
_LINK_LOOP = "a link on its path loops"


def _include_text(path: Path, inc: str, case_root: Path) -> str | None:
    """The comment-stripped text of the file `#include "inc"` names, or None when it is not a
    file inside the case or cannot be read. Noted once read (VER-14); confined (MCP-3)."""
    target, _ = include_target(path, inc, case_root)
    if target is None:
        return None
    try:
        with reads.reading(target):
            return strip_comments(target.read_text(errors="replace"))
    except OSError:
        return None


def expand(text: str, path: Path, case_root: Path) -> str:
    """`text`, the comment-stripped dictionary at `path`, as the solver reads it (TRD VER-3b):

    - each `#include`, `#includeIfPresent` and `#sinclude` of a file inside the case is replaced
      by the file's entries (its own includes followed in turn, four deep); an optional one whose
      file is absent is dropped, as the solver drops it;
    - then each `$name` is replaced by the last definition of `name` before it in its own
      dictionary, else in the dictionaries around it, innermost first, as OpenFOAM expands a
      macro while it reads; a `$name;` standing as an entry is replaced by the entries of the
      dictionary it names (`pFinal { $p; relTol 0; }`), a quoted key matching as a regex.

    What cannot be followed stays as written, for the reader to report: an include of a file that
    is missing or outside the case, `#includeEtc`, a macro with no definition before it, a scoped
    one (`$a.b`, `$:a`, `$..a`), and every evaluated directive (`#calc`, `#eval`, `#codeStream`).
    Under an `#inputMode` other than merge (`protect`, `warn`, `error`) a name written more than
    once is not expanded: which writing stands is the mode's. Bounded: past `_Includes`' bounds,
    which only files included again and again reach (a cycle, a fan-out), the includes stay as
    written, and past `MAX_EXPANDED` characters expanded the macros do. Linear in what the bounds
    let through, the first reading of each file and the repeats up to the bounds, with two
    exceptions not bounded here and open under T-FOAMGUARD-2 (TRD VER-23): foamlib's parse of a
    long plain list, quadratic in its items (review 0.7.0, R2-1), and `$name;` merges, bounded
    only by `MAX_EXPANDED` (R2-2).

    Deterministic, read-only, no subprocess (INV-5)."""
    try:
        inlined = _inline(text, path, case_root, (_where(path),), _Includes(_where(path)))
    except _TooLarge:                                   # SEC070-16: the includes stay as written
        return text
    try:
        return _macros(inlined, strict=_INPUT_STRICT.search(inlined) is not None)
    except _TooLarge:
        return inlined


def _where(path: Path) -> str:
    """A file's identity among the includes: its resolved path, or the path when it has none."""
    try:
        return str(path.resolve())
    except (OSError, ValueError):
        return str(path)


#: Five bounds on one expansion's includes, each on a hang and none on what a case holds: the
#: first reading of a file is never charged, so distinct files of any size are read. They bound
#: the includes, not the parse: foamlib parses a long plain list in time quadratic in its items
#: (1.7.10, review 0.7.0, R2-1), and `_macros`' `$name;` merges are bounded only by `MAX_EXPANDED`
#: (R2-2). Both are on 0.7.0 too, and open under T-FOAMGUARD-2 (TRD VER-23). Measured on 0.7.0's
#: code under `courant verify` on this repository's laptop under load (security review 0.7.0,
#: SEC070-16; a cost bound, no physics):
#:
#: - the inclusions repeated once the includes form a cycle. A cycle the solver itself cannot end,
#:   and what it leaves at the depth limit makes the case reader quadratic: a file including
#:   itself four times took 7 s, five times 100 s (3,905 repeats), six times (9,330) did not
#:   finish in 120 s, and twelve never returned. Up to this bound a cycle reads as 0.7.0 read it;
MAX_CYCLIC_REPEATS = 2**12
#: - the inclusions repeated, cycle or not: 180 microseconds each to inline (111,110 in 19.9 s), so
#:   over three minutes before the first is parsed, and reading 20,000 took 22.7 s and is
#:   quadratic in them;
MAX_REPEATED_INCLUDES = 2**20
#: - the characters of those repeats, set at 2 GiB by SEC070-16 from a run of 64 MiB (8 MiB, eight
#:   times) that did not finish in 240 s. The review of 0.7.0 (A13-R1) measured the same input at
#:   11 s, 0.17 microseconds a character, so 2 GiB costs some six minutes, not hours. Kept for
#:   `_resolve_macros`, the spec reader's default lookup, byte for byte as 0.7.0 (GATE-13);
MAX_REPEATED_CHARS = 2**31
#: - and, for the includes alone, a lower bound on the same characters. A string of 8 MiB
#:   included eight times (56 MiB repeated) reads in 11 s on 0.7.0's code (f5aaebb), sixteen times
#:   (120 MiB) in 21.5 s, and the same on this one (the rate above). At 64 MiB the gate reads to the
#:   bound in some 12 s; a file of 17 MiB read twice is charged 17 MiB and reads. Only `_Includes`
#:   uses it: `_resolve_macros`, the spec reader's default lookup, keeps 0.7.0's bound (GATE-13);
MAX_REPEATED_INCLUDE_CHARS = 2**26
#: - the words and the punctuation (`;`, `{`, `}`, `(`, `)`) of those repeats, each something the
#:   parser reads (`_words`). A fan-out of files under 1 KB names its last file thousands of
#:   times, and what that file holds is parsed each time: entries, or the includes it leaves as
#:   written (review 0.7.0, A1-3 and A13-R1). Measured under `courant verify`, `nu` before the
#:   includes, on 0.7.0's code (f5aaebb) and on this one:
#:   - seven files each including the next k times leave k^6 includes at the depth limit. 0.7.0
#:     took 79 s at k=5 and did not finish k=6 in 30 s (its reader was quadratic in them). k=7
#:     (274,428 words repeated) now reads in 2.1 s; k=8 (599,088) is past the bound, in 0.4 s;
#:   - six files each including the next k times, the last of 120 entries: k=4 (370,968 repeated)
#:     reads in 3.1 s, as on 0.7.0 (3.3 s); k=5 (1,132,400) took 9.4 s on 0.7.0, k=6 21.3 s, and
#:     k=8 would take minutes; past the bound they read in 0.3 s;
#:   - one file of 175,000 entries included twice (525,000 repeated) did not finish in 120 s on
#:     0.7.0. Inside a block (`junk { ... }`), which 0.7.0 read in linear time, it took 8.5 s;
#:     174,000 of them (522,005, below the bound) read in 8.1 s now, as before.
#:   Below the bound an expansion reads as 0.7.0 read it, in linear time but for R2-1 and R2-2
#:   above: at the bound the gate has parsed some 175,000 repeated entries, eight seconds. The
#:   cost of repeated entries is linear on 0.7.0 too, so past the bound some inputs 0.7.0 read in
#:   eight seconds or more, a block or a 120-entry file repeated as above, now stay as written,
#:   noted as not followed: no bound below a hang spares them all.
MAX_REPEATED_TOKENS = 2**19


def _words(text: str) -> int:
    """The words and punctuation of `text` the parser reads (`MAX_REPEATED_TOKENS`)."""
    return len(text.split()) + sum(text.count(c) for c in ";{}()")


class _Includes:
    """What one expansion has inlined, against the bounds above, and each include it has met,
    resolved and read once (`found`): a fan-out names the same few files thousands of times, and
    resolving and reading each again was most of `_inline`'s cost (review 0.7.0, A1-3)."""

    __slots__ = ("chars", "cyclic", "found", "read", "repeats", "tokens", "words")

    def __init__(self, root: str) -> None:
        self.read = {root}
        self.repeats = self.chars = self.tokens = 0
        self.cyclic = False
        self.found: dict[tuple[str, str], tuple[Path | None, str | None, str | None, str]] = {}
        self.words: dict[str, int] = {}                 # `_words` of each file repeated, once

    def include(self, path: Path, inc: str, case_root: Path
                ) -> tuple[Path | None, str | None, str | None, str]:
        """(target, why, its text, its identity) of `#include "inc"` in the file at `path`, as
        `include_target`, `_include_text` and `_where` give them, each asked once per expansion."""
        key = (str(path.parent), inc)
        if key not in self.found:
            target, why = include_target(path, inc, case_root)
            text = None if target is None else _include_text(path, inc, case_root)
            self.found[key] = (target, why, text, "" if target is None else _where(target))
        return self.found[key]

    def charge(self, where: str, stack: tuple[str, ...], text: str) -> None:
        """Count one inclusion of the file `where`, whose text is `text`; `_TooLarge` past a
        bound."""
        if where not in self.read:
            self.read.add(where)
            return
        self.repeats += 1
        self.chars += len(text)
        self.cyclic = self.cyclic or where in stack
        if (self.repeats > (MAX_CYCLIC_REPEATS if self.cyclic else MAX_REPEATED_INCLUDES)
                or self.chars > min(MAX_REPEATED_CHARS, MAX_REPEATED_INCLUDE_CHARS)):
            raise _TooLarge
        if where not in self.words:
            self.words[where] = _words(text)
        self.tokens += self.words[where]
        if self.tokens > MAX_REPEATED_TOKENS:
            raise _TooLarge


def _inline(text: str, path: Path, case_root: Path, stack: tuple[str, ...], spent: _Includes,
            depth: int = 0) -> str:
    """`text` with its includes inlined, four deep, `stack` the files being inlined. `_TooLarge`
    past `spent`'s bounds."""
    if depth > 4 or "#" not in text:
        return text

    def one(m: re.Match) -> str:
        kind, inc = m.group(1), m.group(2)
        target, why, included, where = spent.include(path, inc, case_root)
        if target is None:
            # An optional include is dropped only when its file is absent from the case, as the
            # solver drops it. One outside the case stays, to be reported: the solver read it if
            # it is there, and whether it is is not looked at (MCP-3). `include_target` is the one
            # resolution, tags included (`$FOAM_CASE/../x` is outside the case).
            optional_absent = kind != "include" and why == _NOT_IN_CASE
            return "" if optional_absent else m.group(0)
        if included is None:
            return m.group(0)
        spent.charge(where, stack, included)
        return "\n" + _inline(_without_header(included), target, case_root, (*stack, where),
                              spent, depth + 1) + "\n"

    return _INCLUDES.sub(one, text)


def _without_header(text: str) -> str:
    m = _HEADER.search(text)
    if m is None:
        return text
    close = _block_end(text, m.end() - 1)
    return text if close is None else text[:m.start()] + text[close + 1:]


def _elements(text: str):
    """(start, end, key, open brace or None) of each top-level element of a dictionary body: an
    entry `key value;`, a block `key { ... }`, or a directive line (`#...`, key None). A value is
    read to its `;` outside quotes, brackets and braces (`#codeStream { ... };`, a list of
    dictionaries)."""
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            return
        start = i
        if text[i] == "#":
            end = text.find("\n", i)
            end = n if end < 0 else end
            yield start, end, None, None
            i = end
            continue
        if text[i] == '"':
            close = text.find('"', i + 1)
            k = n if close < 0 else close + 1
        else:
            k = i
            while k < n and not text[k].isspace() and text[k] not in "{};":
                k += 1
        if k == i:                                      # a stray `;` or `}`
            yield start, i + 1, None, None
            i += 1
            continue
        key, p = text[i:k], k
        while p < n and text[p].isspace():
            p += 1
        if p < n and text[p] == "{":
            close = _block_end(text, p)
            end = n if close is None else close + 1
            yield start, end, key, p
            i = end
            continue
        q, brackets, braces = p, 0, 0
        while q < n:
            c = text[q]
            if c == '"':
                close = text.find('"', q + 1)
                q = n if close < 0 else close + 1
                continue
            if c in "([":
                brackets += 1
            elif c in ")]":
                brackets -= 1
            elif c == "{":
                braces += 1
            elif c == "}":
                if braces == 0:
                    break
                braces -= 1
            elif c == ";" and brackets <= 0 and braces == 0:
                q += 1
                break
            q += 1
        yield start, q, key, None
        i = q


#: The most characters the expansion of one dictionary builds (`expand`), counted at every level
#: it builds them. `$name;` merges that double at each level grow a text as 2^levels; past this
#: the dictionary is left with its macros as written, so what they would set is read as unread,
#: never guessed. A cost bound, not a physics threshold: the largest committed dictionary is some
#: 1,000 times smaller (the review of T-VER3B-ONE, 3 Nov 2026).
MAX_EXPANDED = 16 * 2**20
#: An `#inputMode` under which a name written twice does not stand as its last writing: v2406's
#: `protect` keeps the first, `warn` keeps the first and warns, `error` stops.
_INPUT_STRICT = re.compile(r"#inputMode\s+(?:protect|warn|error)\b")


class _TooLarge(Exception):
    """The expansion passed `MAX_EXPANDED`, or `_Includes`' bounds."""


class _Scope:
    """One dictionary's names as its expansion has met them so far: each entry's value (as
    `entry` reads it, the last writing) and how often it was written, and each block as
    `_quoted_blocks` lists it, for `block`. Kept as the scope grows, so a lookup costs the name,
    not the text before it (the review of T-VER3B-ONE: `entry` on the growing text rescanned it
    for every `$name`, quadratic in the dictionary)."""

    __slots__ = ("blocks", "counts", "entries")

    def __init__(self) -> None:
        self.entries: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        self.blocks: list[tuple[str, bool, str]] = []

    def add(self, piece: str) -> None:
        """Register the top-level elements of `piece`, expanded text of this scope."""
        for start, end, key, open_ in _elements(piece):
            if key is None:
                continue
            if open_ is not None:
                close = end - 1 if piece[end - 1:end] == "}" else end
                quoted = len(key) > 1 and key[0] == key[-1] == '"'
                self.blocks.append((key[1:-1] if quoted else key, quoted, piece[open_ + 1:close]))
                continue
            found = _entries(piece[start:end], key)
            if found:
                self.entries[key] = found[-1]
                self.counts[key] = self.counts.get(key, 0) + 1


def _macros(text: str, outer: tuple[_Scope, ...] = (), depth: int = 0, *,
            strict: bool = False, spent: list[int] | None = None) -> str:
    """`text` (one dictionary's body) with its macros expanded; `outer` holds, outermost first,
    the names of each enclosing dictionary up to the block that holds this one. `strict`: an
    `#inputMode` other than merge is in force (`_INPUT_STRICT`), so a name written more than once
    is not expanded. `spent`: the characters built so far, against `MAX_EXPANDED`."""
    if spent is None:
        spent = [0]
    if "$" not in text or depth > 16:
        return text
    here = _Scope()
    parts: list[str] = []
    pos = 0
    for start, end, key, open_ in _elements(text):
        parts.append(text[pos:start])
        piece = text[start:end]
        if open_ is not None and "$" in piece:
            close = end - 1 if text[end - 1:end] == "}" else end
            piece = (text[start:open_ + 1]
                     + _macros(text[open_ + 1:close], (*outer, here), depth + 1, strict=strict,
                               spent=spent)
                     + text[close:end])
        elif key is not None and "$" in piece:
            scopes = (here, *reversed(outer))
            if key.startswith("$") and text[start + len(key):end].strip() in ("", ";"):
                merged = _named_block(key[1:], scopes, strict)
                if merged is not None:
                    piece = merged
            else:
                piece = key + _substitute(text[start + len(key):end], scopes, strict)
        spent[0] += len(piece)
        if spent[0] > MAX_EXPANDED:
            raise _TooLarge
        here.add(piece)
        parts.append(piece)
        pos = end
    parts.append(text[pos:])
    return "".join(parts)


def _named_block(name: str, scopes: tuple[_Scope, ...], strict: bool = False) -> str | None:
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return None                                     # scoped: not followed
    for scope in scopes:
        try:
            found = block("", name, scope.blocks)
        except KeywordNotEvaluated:
            return None                 # stays as written, for the reader to report
        if found is not None:
            if strict and sum(1 for k, _, _ in scope.blocks if k == name) > 1:
                return None             # which writing stands is the mode's: not guessed
            return found
    return None


def _substitute(raw: str, scopes: tuple[_Scope, ...], strict: bool = False) -> str:
    def one(m: re.Match) -> str:
        name = m.group(1)
        if "." in name or raw[m.end():m.end() + 1] in (".", ":"):
            return m.group(0)                           # scoped: not followed
        for scope in scopes:
            found = scope.entries.get(name)
            if found is not None:
                # Under `#inputMode protect` (or warn, error) v2406 keeps the first writing, or
                # stops: a name written twice stays as written, read as unread (the review of
                # T-VER3B-ONE found the last one taken, a false accept under next).
                return m.group(0) if strict and scope.counts[name] > 1 else found
        return m.group(0)

    return _MACRO.sub(one, raw)


_LIST = re.compile(r"^nonuniform\s+List<(\w+)>\s*(\d+)?\s*\(")


def nonuniform_mode(raw: str) -> tuple[Value, int, int] | None:
    """(most common value, its count, the number of values) of an ASCII per-face list."""
    from collections import Counter

    keys = nonuniform_values(raw)
    if keys is None:
        return None
    value, count = Counter(keys).most_common(1)[0]
    return (list(value) if isinstance(value, tuple) else value), count, len(keys)


def nonuniform_values(raw: str) -> list | None:
    """Every value of an ASCII per-face list (floats, or 3-tuples for vectors), in face order.
    None when it is not one the reader can read completely."""
    m = _LIST.match(raw)
    if not m:
        return None
    body = raw[m.end():]
    body = body[:body.rfind(")")] if ")" in body else body
    if m.group(1) == "scalar":
        keys: list = [float(x) for x in _NUM.findall(body)]
    elif m.group(1) == "vector":
        tuples = [t.split() for t in re.findall(r"\(([^()]*)\)", body)]
        if any(not _NUM.fullmatch(x) for t in tuples for x in t):
            return None                                 # not a list of numbers (TST-14)
        keys = [tuple(float(x) for x in t) for t in tuples]
        keys = [k for k in keys if len(k) == 3]
    else:
        return None
    if not keys or (m.group(2) and int(m.group(2)) != len(keys)):
        return None                                     # binary, or not the list it says it is
    return keys


#: The SI base units in OpenFOAM's order of a dimension set (kg m s K mol A cd), and the derived
#: units of v2406's default `DimensionSets` (etc/controlDict, factor 1), as exponents of those.
_UNITS: dict[str, tuple[int, ...]] = {
    "kg": (1, 0, 0, 0, 0, 0, 0), "m": (0, 1, 0, 0, 0, 0, 0), "s": (0, 0, 1, 0, 0, 0, 0),
    "K": (0, 0, 0, 1, 0, 0, 0), "mol": (0, 0, 0, 0, 1, 0, 0), "A": (0, 0, 0, 0, 0, 1, 0),
    "cd": (0, 0, 0, 0, 0, 0, 1), "Cd": (0, 0, 0, 0, 0, 0, 1),
    "N": (1, 1, -2, 0, 0, 0, 0), "Pa": (1, -1, -2, 0, 0, 0, 0), "J": (1, 2, -2, 0, 0, 0, 0),
    "W": (1, 2, -3, 0, 0, 0, 0),
}
#: One factor of a dimension set in unit names: an operator, or a unit with its power.
_UNIT_TOKEN = re.compile(r"\s*(?:([*/])|([A-Za-z]+)(?:\^([-+]?\d+(?:\.\d+)?))?)")


def _dimension_set(inner: str) -> tuple[float, ...] | None:
    """The exponents a dimension set's contents give, as v2406's `dimensionSet` reads them
    (`spec.read_as_solver`, VPA-SPEC-4): 5 or 7 numbers are the exponents; otherwise unit names
    (`m^2/s`, `kg m^-1 s^-2`, `kg*m/s^2`), multiplied by a blank or `*`, the factor after a `/`
    divided, are the 7 SI base exponents. None when it is neither (an unknown unit, another count
    of numbers, an expression the reader does not parse): the set is then not stated."""
    tokens = inner.split()
    if not tokens:
        return ()
    if all(_NUM.fullmatch(t) for t in tokens):
        return tuple(float(t) for t in tokens) if len(tokens) in (5, 7) else None
    total = [0.0] * 7
    pos, sign, factors = 0, 1.0, 0
    text = inner.strip()
    while pos < len(text):
        m = _UNIT_TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            return None
        pos = m.end()
        if m.group(1):
            if sign < 0 or (m.group(1) == "/" and not factors):
                return None                     # `//`, `/*`, or a leading `/`
            if m.group(1) == "/":
                sign = -1.0
            continue
        unit = _UNITS.get(m.group(2))
        if unit is None:
            return None
        power = float(m.group(3)) if m.group(3) else 1.0
        total = [t + sign * power * u for t, u in zip(total, unit)]
        sign, factors = 1.0, factors + 1
    return tuple(total) if factors and sign > 0 else None


#: v2406's Function1 types (`Function1::New`'s run-time table), the words a note may name.
_FUNCTION1 = frozenset({
    "uniform", "zero", "one", "polynomial", "sine", "cosine", "square", "squarePulse", "table",
    "tableFile", "csvFile", "scale", "coded", "linearRamp", "quadraticRamp", "halfCosineRamp",
    "quarterSineRamp", "quarterCosineRamp", "reverseRamp", "stepRamp", "expression",
    "functionObjectValue", "functionObjectTrigger", "inputValueMapper", "sample", "Function1",
    "limitRange", "lookup"})
#: The characters OpenFOAM's `word::valid` refuses.
_NOT_WORD = re.compile(r"[\s\"'/;{}]")


def word(raw: str) -> str:
    """`raw` with the quotes off when it is one quoted token whose content is a valid word: as
    OpenFOAM's word reader converts a string token (`spec.read_as_solver`, VPA-SPEC-6). Anything
    else is returned as it is."""
    if len(raw) > 2 and raw[0] == raw[-1] == '"' and not _NOT_WORD.search(raw[1:-1]):
        return raw[1:-1]
    return raw


def parse_value(raw: str, *, text: bool = False, solver: bool = False,
                whole: bool = False) -> tuple[Value | None, tuple[float, ...] | None, str | None]:
    """(value, dimensions, note) from a raw entry value. `solver` (`spec.read_as_solver`, `--checks
    next`): read as the solver reads it, a dimension set in unit names in SI exponents
    (VPA-SPEC-4), an inline Function1 left unread (VPA-SPEC-3), a quoted word the word
    (VPA-SPEC-6); without it, 0.7.0's reading. A `text` value is its last word, as 0.7.0 reads it;
    with `whole` (`spec.read_whole`, `--checks next`) it is every word, as the solver reads it:
    `default CrankNicolson 0.9;` is "CrankNicolson 0.9", not "0.9", and `div(phi,U) Gauss linear;`
    is "Gauss linear" (deck-oracle DO-3)."""
    dims = None
    m = _DIMS.match(raw)
    if m:
        dims = _dimension_set(m.group(1)) if solver else tuple(numbers(m.group(1)))
        raw = raw[m.end():].strip()
    else:
        # The pre-v1.7 form repeats the name before the dimension set: `nu nu [..] 1e-05`.
        m2 = re.match(r"^[A-Za-z_]\w*\s+(\[[^\]]*\])", raw)
        if m2:
            inner = m2.group(1)[1:-1]
            dims = _dimension_set(inner) if solver else tuple(numbers(inner))
            raw = raw[m2.end():].strip()
    if raw.startswith("nonuniform"):
        return None, dims, "a per-face list (nonuniform), not a single value"
    if raw.startswith("uniform"):
        raw = raw[len("uniform"):].strip()
    if text:
        token = (" ".join(raw.split()) if whole else raw.split()[-1]) if raw.split() else ""
        if solver:
            token = word(token)
        return (token or None), dims, (None if token else "empty entry")
    if solver:
        # A Function1 (v2406's `Function1::New`): a word first is its type. `constant` is a
        # value; any other (`table`, `polynomial`, `sine`, `tableFile`, ...) is a function of time,
        # whose last number is not the value the solver applied.
        first = re.match(r"[A-Za-z_]\w*", raw)
        if first and first.group(0) != "constant":
            # Named only when it is one of OpenFOAM's own types: any other word is the case's
            # text, never quoted in a note (`_read_one`).
            kind = f" ({first.group(0)})" if first.group(0) in _FUNCTION1 else ""
            return None, dims, f"a function of time{kind} the reader does not evaluate"
        if first:
            raw = raw[first.end():].strip()
    nums = numbers(raw)
    if not nums:
        return None, dims, "no number in the entry"           # never quoted (`_read_one`)
    if raw.lstrip().startswith("("):
        return nums, dims, None
    return nums[-1] if len(nums) > 1 else nums[0], dims, None


# ---------------------------------------------------------------- the reader
def _read_text(case: Path, rel: str, gunzip: bool = False) -> tuple[str | None, str | None]:
    """`rel`'s text, its comments stripped, or (None, why not). With `gunzip`
    (`case.zero_compressed`, `--checks next`, VER-3f) a `.gz` is decompressed as OpenFOAM reads it
    (`_gunzip`); without it the bytes are read as text, as 0.7.0 read them."""
    p = case / rel
    if not reads.may_read(p):            # MCP-3: asked before it is looked for, so a path outside
        return None, f"{rel} could not be read: outside the allowed roots"  # is refused alike
    if not p.is_file():
        return None, f"{rel} does not exist"
    try:
        with reads.reading(p):                           # VER-14; MCP-3
            text = _gunzip(p) if gunzip and rel.endswith(".gz") else p.read_text(errors="replace")
    except OSError as exc:
        return None, f"{rel} could not be read: {reads.reason(exc)}"
    except ValueError as exc:                            # cut short, corrupt, past the bound
        return None, f"{rel} could not be read: {exc}"
    return strip_comments(text), None


def _gunzip(p: Path) -> str:
    """A compressed file's text, as OpenFOAM reads a `.gz` (IFstream), through the mesh reader's
    bounded decompression (`polymesh._raw`, the one implementation: a small file can decompress to
    gigabytes). Raises `OSError` on a file that is not gzip, `ValueError` on one cut short or
    corrupt or past the bound; never `EOFError` or `zlib.error`."""
    from courant.case import polymesh

    text = polymesh._raw(p).decode(errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")       # as `read_text` reads a newline


def opened(case: Path, rel: str) -> str:
    """`rel`, or its `.gz` where that is the file OpenFOAM opens (`reads.plain_or_gz`, the one
    rule), asked only inside the allowed roots (MCP-3)."""
    p = case / rel
    return rel + ".gz" if reads.may_read(p) and reads.plain_or_gz(str(p)) != str(p) else rel


def _complete(text: str) -> bool:
    """Whether the reader saw all of the file's entries: no directive (`#include`, `#calc`, ...)
    or `$` macro can hide the location from it."""
    return "#" not in text and "$" not in text


def _patch_absent(case: Path, rel: str, text: str, patch: str) -> bool:
    """Whether `patch` certainly does not exist: the field file was read in full, the mesh's own
    `constant/polyMesh/boundary` was read, and the name (as a patch or as a group in `inGroups`)
    appears nowhere in it. A mesh not yet built, or any doubt, is not absence (VER-5.9)."""
    if not _complete(text):
        return False
    boundary, _ = _read_text(case, "constant/polyMesh/boundary")
    if boundary is None or not blocks(boundary):
        return False
    return re.search(rf"(?<![\w.]){re.escape(patch)}(?![\w.])", boundary) is None


#: The mesh's patch list, whose `inGroups` and types say which group entries name a patch.
_BOUNDARY = "constant/polyMesh/boundary"
#: Patch types whose patch v2406 puts in the group of the type's name whatever its `inGroups`:
#: a wall (wallPolyPatch.C) and every constraint type (polyPatch.C, `constraintType`).
_TYPE_GROUPS = frozenset({"wall", "cyclic", "cyclicAMI", "cyclicACMI", "cyclicSlip",
                          "nonuniformTransformCyclic", "processor", "processorCyclic"})


def _groups_of(case: Path, patch: str, *, regions: bool = False
               ) -> tuple[frozenset[str] | None, str | None]:
    """(the groups `patch` is in, None), from the mesh's `constant/polyMesh/boundary`, in either
    format (its entries are text in both, FLD-7): its `inGroups` words
    (`courant.case.reynolds._patch_groups`, the one reader of them), and its type where the type
    is a group (`_TYPE_GROUPS`, `reynolds._CONSTRAINT`). (None, None) when there is no such file
    (no mesh yet: the solver has read no groups either); (None, why) when it is there and its
    groups cannot be read. `regions`: the case may have been solved as regions
    (`_foam_reader.regions_not_ruled_out`, decided by the caller)."""
    from courant.case import polymesh, reynolds

    path = Path(reads.plain_or_gz(str(case / _BOUNDARY)))
    why = f"the groups of patch '{patch}' in {_BOUNDARY} could not be read, and a group's entry " \
          "is applied before a pattern's"
    # The verdict-path audit's review: a decomposed run read each rank's own
    # `processor*/constant/polyMesh/boundary` (a mesh snappyHexMesh wrote in parallel, which the
    # case's own `constant/polyMesh` does not hold), and a region's field the region's
    # `constant/<region>/polyMesh/boundary`. This reader reads neither, so it does not tell.
    if regions or any(
            (proc / _BOUNDARY).is_file() or (proc / (_BOUNDARY + ".gz")).is_file()
            for proc in case.glob("processor*")):
        return None, (f"the groups of patch '{patch}' are those of a decomposed or multi-region "
                      "mesh, which this reader does not read, and a group's entry is applied "
                      "before a pattern's")
    if not reads.may_read(path):
        return None, why
    if not path.is_file():
        return None, None
    try:
        groups = polymesh._read(path, lambda p: reynolds._patch_groups(p, binary=True))
        types = {p.name: p.type for p in polymesh.read_boundary(path, binary=True)}
    except Exception:                                   # noqa: BLE001 - a reader limit
        return None, why
    if groups is None:
        return None, why
    for name, words in groups:
        if name == patch:
            # `_patch_groups` keeps every word of the entry; a list's size and `List<word>` are
            # not group names.
            named = {w for w in words if not w.isdigit() and w != "List<word>"}
            kind = types.get(name, "patch")
            if kind in _TYPE_GROUPS or kind in reynolds._CONSTRAINT:
                named.add(kind)
            return frozenset(named), None
    return frozenset(), None                            # not a patch of the mesh: in no group


def patch_entry(case: Path, bf: str, patch: str, *, regions: bool = False
                ) -> tuple[str | None, str, str | None]:
    """(the body of the boundaryField entry the solver applies to `patch`, the key it is written
    under, None), or (None, patch, why it cannot be told), as v2406's
    `GeometricBoundaryField::readField` finds it (`spec.read_as_solver`, VPA-SPEC-1):

    1. an entry whose unquoted key is the patch's name;
    2. else an entry whose unquoted key is one of the patch's groups (`_groups_of`), the last
       written of them (readField walks the dictionary in reverse; a key written again is merged
       into its first writing and keeps its place, `dictionary::add`);
    3. else `block`'s lookup: a quoted key whose text is the name, then the patterns.

    A boundary file that is there and whose groups cannot be read leaves a patch not found under
    its own name not told, when an unquoted key that might be a group is written: a group's entry
    would come before any pattern. `block`'s `KeywordNotEvaluated` propagates, as there."""
    found = _quoted_blocks(bf)
    literal = [body for key, quoted, body in found if not quoted and key == patch]
    if literal:
        return "\n".join(literal), patch, None
    keys = list(dict.fromkeys(key for key, quoted, _ in found if not quoted))
    groups, why = _groups_of(case, patch, regions=regions)
    if groups is None:
        if why is not None and keys:
            return None, patch, why
        groups = frozenset()
    hit = next((key for key in reversed(keys) if key in groups), None)
    if hit is not None:
        return "\n".join(body for key, quoted, body in found if not quoted and key == hit), hit, None
    return block(bf, patch, found), patch, None


#: Keys OpenFOAM reads as the same setting, with the value it uses when neither is written.
_ALIASES = {"scale": ("convertToMeters", "1"), "convertToMeters": ("scale", "1")}


def _read_one(case: Path, rel: str, item, macros: bool = False, *, solver: bool = False,
              regions: bool = False, whole: bool = False, gunzip: bool = False) -> Found:
    """The value `item` names in the case file `rel`. `gunzip` (`case.zero_compressed`): a `.gz`
    decompressed (`_read_text`). `macros` (`spec.directives`, `--checks
    next`): the file is read as the solver reads it, through `expand`, the case reader's own
    expansion under `case.directives`, so a `$name` is the definition nearest the entry, an
    `#includeIfPresent` is followed and a merged `$name;` is read; without it, 0.6.0's lookup
    (`_resolve_macros`: the file's top level, then its `#include`s). `solver`
    (`spec.read_as_solver`, `--checks next`): a patch is found as the solver finds it
    (`patch_entry`, told `regions`) and its value parsed as the solver parses it
    (`parse_value`). `whole` (`spec.read_whole`, `--checks next`): a word value is every word of
    the entry (`parse_value`, deck-oracle DO-3)."""
    if item.sha256 is not None:
        p = case / rel
        if not reads.may_read(p):        # MCP-3: asked before it is looked for, as `_read_text`
            return Found(None, rel, f"{rel} could not be read: outside the allowed roots")
        if not p.is_file():
            return Found(None, rel, f"{rel} does not exist")
        try:
            with reads.reading(p):                       # VER-14; MCP-3
                data = p.read_bytes()
        except OSError as exc:
            return Found(None, rel, f"{rel} could not be read: {reads.reason(exc)}")
        return Found(hashlib.sha256(data).hexdigest(), rel)

    text, why = _read_text(case, rel, gunzip=gunzip)
    if text is None:
        return Found(None, rel, why)
    as_written = text
    if macros:
        text = expand(text, case / rel, case)

    if item.blocks == "cells":
        matched = list(_HEX.finditer(text))
        n_hex = len(re.findall(r"\bhex\s*\(", text))
        if not matched:
            return Found(None, rel, "no hex block with cell counts")
        if len(matched) != n_hex:
            # A partial read must not pass as a complete one.
            return Found(None, rel, f"{len(matched)} of {n_hex} hex blocks give literal cell "
                                    "counts; the rest use a macro or an expression")
        return Found([float(x) for m in matched for x in m.groups()], rel)

    try:
        located = _locate(case, rel, text, item, solver=solver, regions=regions)
    except KeywordNotEvaluated as exc:
        holder = _holder(case, rel, as_written, exc) if text is not as_written else rel
        if holder == rel:
            raise
        return Found(None, rel, not_evaluated(holder, rel) if holder else not_evaluated(
            f"a file {rel} includes"))
    if isinstance(located, Found):
        return located
    raw, written = located
    # A value the reader cannot read is named by its entry and never quoted: what the case wrote
    # (`$acmeInletSpeed`) is the customer's text, and the note reaches the verdict and the
    # support bundle, which pseudonymises names it knows and not this (the review of the 0.6.0
    # follow-ups, 22 Oct 2026; `not_evaluated` for a keyword).
    if _DIRECTIVE.search(raw):
        return Found(None, rel, f"'{written}' is written as a directive the reader does not "
                                "evaluate")
    if macros:
        resolved = None if "$" in raw else raw          # `expand` left it: not defined before it
    else:
        resolved = _resolve_macros(raw, text, case / rel, case)
    if resolved is None:
        return Found(None, rel, f"'{written}' refers to a value that could not be found")
    if _DIRECTIVE.search(resolved):
        # A macro that expands to `#calc ...` is no more a number than `#calc` written out:
        # reading its last number misread `nu #calc "$a*2"` as 2 (review, 10 Oct 2026).
        return Found(None, rel, f"'{written}' expands to a directive the reader does not "
                                "evaluate")
    if resolved.startswith("nonuniform") and not isinstance(item.value, str):
        faces = nonuniform_values(resolved)
        if faces is None:
            return Found(None, rel, "a per-face list (nonuniform) that could not be read")
        # not None: nonuniform_values read the same text above
        value, count, total = cast("tuple[Value, int, int]", nonuniform_mode(resolved))
        return Found(value, rel, basis=f"most common of {total} face values ({count})",
                     faces=tuple(list(f) if isinstance(f, tuple) else f for f in faces))
    value, dims, note = parse_value(resolved, text=isinstance(item.value, str), solver=solver,
                                    whole=whole)
    return Found(value, rel, note, dims)


def _locate(case: Path, rel: str, text: str, item, *,
            solver: bool = False, regions: bool = False) -> tuple[str, str] | Found:
    """(the raw value of the entry `item` names in `text`, the key it is written under), or the
    `Found` that says why there is none. `solver` (`spec.read_as_solver`): a patch's entry is the
    one the solver applies to it (`patch_entry`), its groups' before any pattern."""
    scope = text
    #: (scope, name) of every lookup on the way to the value, for the repeated-key rules.
    steps: list[tuple[str, str]] = []
    if item.patch is not None:
        steps.append((scope, "boundaryField"))
        bf = block(text, "boundaryField")
        if bf is None:
            return Found(None, rel, "no boundaryField")
        if solver:
            scope, named, why = patch_entry(case, bf, item.patch, regions=regions)
            if why is not None:
                return Found(None, rel, why)
        else:
            scope, named = block(bf, item.patch), item.patch
        steps.append((bf, named))
        if scope is None:
            return Found(None, rel, f"no patch '{item.patch}' in boundaryField",
                         missing=_patch_absent(case, rel, text, item.patch))
        key = item.key or "value"
    else:
        for name in item.path:
            steps.append((scope, name))
            scope = block(scope, name)
            if scope is None:
                return Found(None, rel, f"no '{name}' block")
        key = item.key
    steps.append((scope, key))
    for name in (n for _, n in steps):
        directive = _entry_directive(text, name)
        if directive:
            return Found(None, rel, f"'{name}' is named by a #{directive} directive, which the "
                                    "reader does not follow")
    mode = _INPUT_MODE.search(text)
    if mode and any(_repeated(s, n) for s, n in steps):
        return Found(None, rel, f"'{key}' is written more than once under #inputMode "
                                f"{mode.group(1)}, whose rule the reader does not follow")
    raw, written = entry(scope, key), key
    if raw is None and rel.endswith("blockMeshDict") and key in _ALIASES and not item.path:
        other, fallback = _ALIASES[key]                 # one setting under two names
        raw, written = entry(scope, other), other
        if raw is None and _complete(text):
            return Found(float(fallback), rel, basis=f"not written; OpenFOAM's default {fallback}")
    if raw is None:
        return Found(None, rel, f"no entry '{key}'")
    return raw, written


def _holder(case: Path, rel: str, text: str, exc: KeywordNotEvaluated) -> str | None:
    """Which file holds the keywords `exc` names, read through `expand` (TRD VER-3b): `rel` when
    its own text `text` writes each of them, else the file one of its `#include`s names that
    does, relative to the case; None when neither (a file included in turn). A note on such a
    keyword names it (T-FOLLOWUP-MINORS (7): the including file was named)."""
    keywords = getattr(exc, "keywords", ())

    def holds(body: str) -> bool:
        return bool(keywords) and all(f'"{k}"' in body for k in keywords)

    if holds(text):
        return rel
    for m in _INCLUDES.finditer(text):
        target, _ = include_target(case / rel, m.group(2), case)
        if target is None:
            continue
        included = _include_text(case / rel, m.group(2), case)      # read by `expand` already
        if included is not None and holds(included):
            # Resolved both, as `include_target` confined it: a case given by a relative path,
            # or an absolute include through a link (/tmp, /private/tmp), is still the case's
            # file; never the absolute path (the review of T-FOLLOWUP-MINORS, M3).
            try:
                return target.resolve().relative_to(case.resolve()).as_posix()
            except (ValueError, OSError):
                return None
    return None


def read(case_dir: str | Path, item, *, macros: bool = False, solver: bool = False,
         regions: bool = False, whole: bool = False, compressed: bool = False) -> Found:
    """The value `item` names, read from the case the way the solver read it. Never raises.

    `macros` (`spec.directives` in `courant.verify.checks.NEXT`, `--checks next`, GATE-13): the
    file is read through `expand`, as the case reader reads it under `case.directives`: a
    `$name` is the definition nearest the entry, looked up outwards, and every include the
    solver follows is followed. Without it, 0.6.0's lookup: the file's top level, then its
    `#include`s.

    `solver` (`spec.read_as_solver` in `NEXT`, `--checks next`, GATE-13): a patch reached by one
    of its groups is read from the group's entry before any pattern (`patch_entry`), an inline
    Function1 is not read, a dimension set in unit names is read in SI exponents and a quoted word
    is the word (`parse_value`). Without it, 0.7.0's reading. `regions`, with `solver`: the case
    may have been solved as regions (`AsRun.regions` decides, once per case), so the groups of its
    `constant/polyMesh/boundary` do not tell a patch's.

    `whole` (`spec.read_whole` in `NEXT`, `--checks next`, GATE-13): a word value is every word of
    the entry, not its last (`parse_value`, deck-oracle DO-3). Without it, 0.7.0's reading.

    For a file under `0/`:
    - the solver reads `0/`, so `0/` decides. A value the run rewrote into a per-face list
      (`setFields`) is compared on its most common face value, and the basis is recorded;
    - only when the file is missing from `0/` is it read from `0.orig/`.

    Two wrong versions came before this one, both found by measurement (RRD RM-18 v2): reading
    `0.orig/` whenever `0/` held a per-face list masked an edit the run had used; requiring
    `0.orig/` to agree as well rejected a correct answer that left `0.orig/` stale.

    `compressed` (`case.zero_compressed`, `--checks next`, TRD VER-3f): a file that is not there
    is read from its `.gz` when that is (`opened`), as OpenFOAM opens it, before any `0.orig/`: a
    run written with `writeCompression on` leaves `setFields`' `0/T.gz` and no `0/T`, and
    `0.orig/T` is the field before `setFields` (the field-oracle sweep, FO-1); and a `.gz` the
    item names is decompressed. Not for a `sha256` item, whose hash is of the bytes named. Without
    it, 0.7.0's reading: `0/T` falls back to `0.orig/T`, and a `0/T.gz` the item names is not read.
    """
    case = Path(case_dir)
    rel = item.file                                     # the file being read
    try:
        if compressed and item.sha256 is None:
            rel = opened(case, rel)                     # what OpenFOAM opened
        found = _read_one(case, rel, item, macros, solver=solver, regions=regions, whole=whole,
                          gunzip=compressed)
        if found.value is None and rel == item.file and item.file.startswith("0/") \
                and not (case / item.file).is_file():
            orig = "0.orig/" + item.file[2:]
            if (case / orig).is_file():
                rel = orig
                alt = _read_one(case, orig, item, macros, solver=solver, regions=regions,
                                whole=whole, gunzip=compressed)
                if alt.value is not None:
                    return alt
        return found
    except KeywordNotEvaluated:                         # a pattern that may name the block
        return Found(None, item.file, not_evaluated(rel))
    except Exception as exc:                            # noqa: BLE001 - a reader limit, not a verdict
        return Found(None, item.file, f"unreadable: {reads.reason(exc)}")
