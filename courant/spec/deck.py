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

from courant.case import reads

Value = float | list[float] | str

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_DIMS = re.compile(r"^\s*\[([^\]]*)\]")
_MACRO = re.compile(r"\$\{?([A-Za-z_][\w.]*)\}?")
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


def not_evaluated(rel: str) -> str:
    """The note on a value whose block a keyword pattern not evaluated may name, in file `rel`:
    the file, never the keyword (the review of the 0.6.0 hard-proofing, 22 Oct 2026, found 40
    characters of it quoted)."""
    return (f"a quoted keyword in {rel} is a pattern the reader does not evaluate, and it may be "
            "the one that names this block")


def block(text: str, name: str) -> str | None:
    """The body of the top-level block `name`, matching OpenFOAM's lookup
    (`dictionary::csearch`, `courant.case.wordre.csearch`).

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

    found = _quoted_blocks(text)
    hits, unsure = wordre.csearch([(key, quoted) for key, quoted, _ in found], name,
                                  dictionaries=True)
    if unsure:
        raise KeywordNotEvaluated("a quoted keyword is a pattern the reader does not evaluate, "
                                  "and it may be the one that names this block")
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
def _resolve_macros(raw: str, text: str, path: Path, case_root: Path, depth: int = 0) -> str | None:
    """Substitute `$name` from the same file, or from a file it `#include`s. None if not found."""
    if "$" not in raw:
        return raw
    if depth > 4:
        return None

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

    out = raw
    names = _MACRO.findall(raw)
    if not names:
        return None                                     # a scoped `$:a.b` or `$..x` reference
    for name in names:
        v = lookup(name)
        if v is None:
            return None
        v = _resolve_macros(v, text, path, case_root, depth + 1)
        if v is None:
            return None
        out = re.sub(rf"\$\{{?{re.escape(name)}\}}?", lambda _m, v=v: v, out, count=1)
    return None if "$" in out else out


# ---------------------------------------------------------------- the dictionary as the solver reads it
#: OpenFOAM's tags for an include's directory (`<case>/...`), and `$FOAM_CASE`, which every
#: install sets to the case.
_INCLUDE_TAGS = {"<case>": "", "$FOAM_CASE": "", "${FOAM_CASE}": "", "<system>": "system",
                 "<constant>": "constant"}
#: The includes of a file the solver reads from the case: `#include`, and `#includeIfPresent` and
#: `#sinclude`, which it skips when the file is absent. `#includeEtc`, `#includeFunc` and
#: `#includeModel` name files of the OpenFOAM installation, not of the case: not followed.
_INCLUDES = re.compile(r'#(include|includeIfPresent|sinclude)[ \t]+"([^"\n]*)"')
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
    except (ValueError, OSError):                       # outside the case, or no path at all
        return None, "the file is outside the case"     # (a NUL byte: TST-14's fuzzer)
    return (target, None) if is_file else (None, _NOT_IN_CASE)


#: `include_target`'s reason for a target inside the case that is not a file there.
_NOT_IN_CASE = "no such file in the case"


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
    Deterministic, read-only, no subprocess (INV-5)."""
    return _macros(_inline(text, path, case_root))


def _inline(text: str, path: Path, case_root: Path, depth: int = 0) -> str:
    if depth > 4 or "#" not in text:
        return text

    def one(m: re.Match) -> str:
        kind, inc = m.group(1), m.group(2)
        target, why = include_target(path, inc, case_root)
        if target is None:
            # An optional include is dropped only when its file is absent from the case, as the
            # solver drops it. One outside the case stays, to be reported: the solver read it if
            # it is there, and whether it is is not looked at (MCP-3). `include_target` is the one
            # resolution, tags included (`$FOAM_CASE/../x` is outside the case).
            optional_absent = kind != "include" and why == _NOT_IN_CASE
            return "" if optional_absent else m.group(0)
        included = _include_text(path, inc, case_root)
        if included is None:
            return m.group(0)
        return "\n" + _inline(_without_header(included), target, case_root, depth + 1) + "\n"

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


def _macros(text: str, outer: tuple[str, ...] = (), depth: int = 0) -> str:
    """`text` (one dictionary's body) with its macros expanded; `outer` holds, outermost first,
    the expanded text of each enclosing dictionary up to the block that holds this one."""
    if "$" not in text or depth > 16:
        return text
    done, pos = "", 0
    for start, end, key, open_ in _elements(text):
        done += text[pos:start]
        piece = text[start:end]
        if open_ is not None and "$" in piece:
            close = end - 1 if text[end - 1:end] == "}" else end
            piece = (text[start:open_ + 1] + _macros(text[open_ + 1:close], (*outer, done),
                                                       depth + 1) + text[close:end])
        elif key is not None and "$" in piece:
            scopes = (done, *reversed(outer))
            if key.startswith("$") and text[start + len(key):end].strip() in ("", ";"):
                merged = _named_block(key[1:], scopes)
                if merged is not None:
                    piece = merged
            else:
                piece = key + _substitute(text[start + len(key):end], scopes)
        done += piece
        pos = end
    return done + text[pos:]


def _named_block(name: str, scopes: tuple[str, ...]) -> str | None:
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return None                                     # scoped: not followed
    for scope in scopes:
        try:
            found = block(scope, name)
        except KeywordNotEvaluated:
            return None                 # stays as written, for the reader to report
        if found is not None:
            return found
    return None


def _substitute(raw: str, scopes: tuple[str, ...]) -> str:
    def one(m: re.Match) -> str:
        name = m.group(1)
        if "." in name or raw[m.end():m.end() + 1] in (".", ":"):
            return m.group(0)                           # scoped: not followed
        for scope in scopes:
            found = entry(scope, name)
            if found is not None:
                return found
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


def parse_value(raw: str, *, text: bool = False) -> tuple[Value | None, tuple[float, ...] | None, str | None]:
    """(value, dimensions, note) from a raw entry value."""
    dims = None
    m = _DIMS.match(raw)
    if m:
        dims = tuple(numbers(m.group(1)))
        raw = raw[m.end():].strip()
    else:
        # The pre-v1.7 form repeats the name before the dimension set: `nu nu [..] 1e-05`.
        m2 = re.match(r"^[A-Za-z_]\w*\s+(\[[^\]]*\])", raw)
        if m2:
            dims = tuple(numbers(m2.group(1)[1:-1]))
            raw = raw[m2.end():].strip()
    if raw.startswith("nonuniform"):
        return None, dims, "a per-face list (nonuniform), not a single value"
    if raw.startswith("uniform"):
        raw = raw[len("uniform"):].strip()
    if text:
        token = raw.split()[-1] if raw.split() else ""
        return (token or None), dims, (None if token else "empty entry")
    nums = numbers(raw)
    if not nums:
        return None, dims, "no number in the entry"           # never quoted (`_read_one`)
    if raw.lstrip().startswith("("):
        return nums, dims, None
    return nums[-1] if len(nums) > 1 else nums[0], dims, None


# ---------------------------------------------------------------- the reader
def _read_text(case: Path, rel: str) -> tuple[str | None, str | None]:
    p = case / rel
    if not reads.may_read(p):            # MCP-3: asked before it is looked for, so a path outside
        return None, f"{rel} could not be read: outside the allowed roots"  # is refused alike
    if not p.is_file():
        return None, f"{rel} does not exist"
    try:
        with reads.reading(p):                           # VER-14; MCP-3
            text = p.read_text(errors="replace")
    except OSError as exc:
        return None, f"{rel} could not be read: {reads.reason(exc)}"
    return strip_comments(text), None


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


#: Keys OpenFOAM reads as the same setting, with the value it uses when neither is written.
_ALIASES = {"scale": ("convertToMeters", "1"), "convertToMeters": ("scale", "1")}


def _read_one(case: Path, rel: str, item) -> Found:
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

    text, why = _read_text(case, rel)
    if text is None:
        return Found(None, rel, why)

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

    scope = text
    #: (scope, name) of every lookup on the way to the value, for the repeated-key rules.
    steps: list[tuple[str, str]] = []
    if item.patch is not None:
        steps.append((scope, "boundaryField"))
        bf = block(text, "boundaryField")
        if bf is None:
            return Found(None, rel, "no boundaryField")
        steps.append((bf, item.patch))
        scope = block(bf, item.patch)
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
    # A value the reader cannot read is named by its entry and never quoted: what the case wrote
    # (`$acmeInletSpeed`) is the customer's text, and the note reaches the verdict and the
    # support bundle, which pseudonymises names it knows and not this (the review of the 0.6.0
    # follow-ups, 22 Oct 2026; `not_evaluated` for a keyword).
    if _DIRECTIVE.search(raw):
        return Found(None, rel, f"'{written}' is written as a directive the reader does not "
                                "evaluate")
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
        value, count, total = nonuniform_mode(resolved)
        return Found(value, rel, basis=f"most common of {total} face values ({count})",
                     faces=tuple(list(f) if isinstance(f, tuple) else f for f in faces))
    value, dims, note = parse_value(resolved, text=isinstance(item.value, str))
    return Found(value, rel, note, dims)


def read(case_dir: str | Path, item) -> Found:
    """The value `item` names, read from the case the way the solver read it. Never raises.

    For a file under `0/`:
    - the solver reads `0/`, so `0/` decides. A value the run rewrote into a per-face list
      (`setFields`) is compared on its most common face value, and the basis is recorded;
    - only when the file is missing from `0/` is it read from `0.orig/`.

    Two wrong versions came before this one, both found by measurement (RRD RM-18 v2): reading
    `0.orig/` whenever `0/` held a per-face list masked an edit the run had used; requiring
    `0.orig/` to agree as well rejected a correct answer that left `0.orig/` stale.
    """
    case = Path(case_dir)
    rel = item.file                                     # the file being read
    try:
        found = _read_one(case, rel, item)
        if found.value is None and item.file.startswith("0/") and not (case / item.file).is_file():
            orig = "0.orig/" + item.file[2:]
            if (case / orig).is_file():
                rel = orig
                alt = _read_one(case, orig, item)
                if alt.value is not None:
                    return alt
        return found
    except KeywordNotEvaluated:                         # a pattern that may name the block
        return Found(None, item.file, not_evaluated(rel))
    except Exception as exc:                            # noqa: BLE001 - a reader limit, not a verdict
        return Found(None, item.file, f"unreadable: {reads.reason(exc)}")
