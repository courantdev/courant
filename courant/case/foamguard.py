"""What foamlib must not be handed: a list its parser would take minutes, or for ever, to read.

**Behind `--checks next` (`case.foam_guard`, 0.8.0, GATE-13, TRD VER-3h).** Under the default no
reader asks: `check_path` and `check_text` return at once, every file reads as 0.7.0 read it, and a
list foamlib hangs on still hangs (review of 0.7.0, A6-1, open until TRD VER-23). `guarded` says
which: the ingest, the pipeline and preflight each enter it where the mode is known.

foamlib 1.7.10 (the version `packaging/requirements-release.lock` pins) reads an ASCII list of
numbers with regular expressions (`foamlib/_files/_parsing/_parser.py`,
`_ASCIINumericListParser`). When the list does not close as a list of numbers they backtrack:

- a scalar list over every split of every number (`(?:\\s*[0-9.+-einfaty...]+)*`: "299.3" splits
  16 ways) and, in the commented variant, every split of every gap of whitespace
  (`(?:\\s+|comment)+`: a gap of w characters 2^(w-1) ways). Measured: six numbers of a list
  cut mid-write 15 s; an indented list of five lines cut mid-write 7 s; each further line about
  16 times longer. In a dictionary too: `x (299.3 299.3 299.3 299.3 299.3 @);` 4 s;
- a list of vectors or tensors, after `nonuniform List<...>`, over every split of every gap
  between groups between the trailing and the leading `\\s*` (w + 1 ways): 24 vectors and a typo
  19 s; eight indented rows 2 s.

Its recursive-descent parser tries three readings at every level of parentheses, so nesting is
exponential (14 deep 0.3 s, doubling), and reads a list whose count is not its length in time
quadratic in the length (800 numbers 4 s). A field file cut mid-write (TRD VER-7b's own case), a
duplicated block, or a typo inside a list made `courant verify` hang
(research/notes/invariants-070/FINDINGS.md, INV6-F1).

`check_bytes` scans what foamlib would read, in linear time (possessive patterns, which never
backtrack), estimates the exponent of each failing match from the same quantities foamlib's
patterns branch on, and raises `Backtracking` when it reaches `MAX_EXPONENT` (2^18 paths: about
0.3 s to 2 s of foamlib's time, doubling with each unit), when a field's list whose count is not
its length holds `MAX_MISMATCHED` numbers, or when parentheses nest `MAX_DEPTH` deep. A list that
closes as numbers matches at once and is passed whole; a failing prefix below the bound fails as
fast as before, with foamlib's own error. A binary payload (`format binary`) is cut out first: its
bytes are not text. Every reader that hands a file to foamlib asks `check_path` first, inside the
`try` that already records a file it could not read. A refused dictionary is said under "not
assessed" and a refused field of the latest time under "could not be read". **A refused `0/` file
is not said:** `_boundary_types` and `_field_file` drop it with nothing recorded, as they drop any
`0/` file that does not parse (M6 of the guard's review, for TRD VER-23).

The cost model has open gaps both ways (TRD VER-23), which is why it is not on the
default path.

**A cost is charged only where foamlib runs the patterns** (TRD VER-3h, measured on 1.7.10:
research/notes/invariants-070/FINDINGS.md). They run on a field's `nonuniform List<...>`, and
outside a field only through `_parse_standalone_data`, which `_parse_file` falls back to when a
statement at the top of the file does not read as a dictionary entry, and which walks that
statement's words and lists until it meets one it cannot read. A list inside `{ }` or inside
another list is never reached: the walk stops at the `{`, and the general parser that reads what
is inside runs no number pattern. Neither is any list of a statement that reads: `fields (T p);`
over many lines, or a list opened by comments, holds a long failing prefix and costs nothing,
since foamlib reads it with its general parser. So a costly top-level list is refused only when
its statement does not read: at once when the list never closes or the byte after its numeric
prefix can continue no list, and otherwise by asking foamlib's own parser about that one
statement (`_statement_reads`).

Not bounded here, and said in the findings (INV6-F4): foamlib's general parser is quadratic in
the length of any long dictionary list (300 probe locations 8 s), counted or not. Refusing a valid
file would change real verdicts.
"""

from __future__ import annotations

import gzip
import math
import os
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

#: Whether a reader's question reaches the guard: `case.foam_guard` (`guarded`). Off, as under the
#: default checks.
_ASKED: ContextVar[bool] = ContextVar("courant_foam_guard", default=False)

#: foamlib's character class for a number (`_POSSIBLE_FLOAT`).
_NUM = rb"[0-9.\-+einfatyEINFATY]"
_TOKEN = re.compile(_NUM + rb"++")
_WS = re.compile(rb"\s++")
#: A comment, a string and verbatim code, each consumed to the end of the file when it is not
#: closed (foamlib stops there at once), so one pass of `finditer` is linear.
_COMMENT = rb"/\*(?:[^*]|\*(?!/))*+(?:\*/|\Z)|//(?:\\\n|[^\n])*+"
_COMMENT_RE = re.compile(_COMMENT)
#: One pass, in order of first occurrence: a `"` in a comment opens no string, and a `//` in a string
#: opens no comment.
_OPAQUE = re.compile(_COMMENT + rb'|"(?:[^"\\]|\\.)*+(?:"|\Z)|#\{(?:[^#]|#(?!\}))*+(?:#\}|\Z)',
                     re.DOTALL)
#: The prefix after `(` that the scalar patterns consume: whitespace and number characters
#: (`_uncommented_pattern`), and comments too (`_pattern`).
_RUN = re.compile(rb"(?:\s|" + _NUM + rb")*+")
_RUN_C = re.compile(rb"(?:\s|" + _NUM + rb"|" + _COMMENT + rb")*+")
#: Runs of groups of exactly k numbers: what the vector, symmetric-tensor and tensor patterns
#: consume. A run of mixed sizes is none of them.
_SHAPED = tuple(re.compile(rb"(?:\s*+\(\s*+(?:" + _NUM + rb"++\s*+){%d}\))*+" % k)
                for k in (3, 6, 9))
_SHAPED_C = tuple(re.compile(rb"(?:(?:\s|" + _COMMENT + rb")*+\((?:\s|" + _COMMENT + rb")*+(?:"
                             + _NUM + rb"++(?:\s|" + _COMMENT + rb")*+){%d}\))*+" % k)
                  for k in (3, 6, 9))
_CLOSE = re.compile(rb"\s*+\)")
_CLOSE_C = re.compile(rb"(?:\s|" + _COMMENT + rb")*+\)")
#: Where foamlib runs its number patterns whatever the rest of the file is: a field's
#: `nonuniform List<T>` (`_parse_field`).
_FIELD_LIST = re.compile(rb"nonuniform\s++List<\s*+(?:scalar|vector|symmTensor|tensor)\s*+>\s*+"
                         rb"(?:[0-9]++\s*+)?\(")
_LEAD_COUNT = re.compile(rb"([0-9]+)\s*+$")
#: foamlib's tokens: what starts one (`_IS_TOKEN_START`) and what continues one
#: (`_IS_TOKEN_CONTINUATION`). `coeffs_2 (` is the word `coeffs_2`, not a count of 2.
_WORD_START = frozenset(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_#$./")
_WORD_BYTE = _WORD_START | frozenset(b"0123456789._<>#$:+-*/|^%&=!")
#: What may follow a numeric prefix in a list foamlib's general parser reads: a word, a further
#: list, a dimension set, a dictionary, the list's own end, or the `;` of a keyword entry
#: (`(1 2 3;)`). A prefix can stop inside a word (`T:mean`, whose `T` is in the number class), so
#: any byte of a word. Anything else (`@`, `}`, the end of the file) ends the reading there.
_CONTINUES_A_LIST = _WORD_BYTE | frozenset(b'"()[{;')
#: A comment, or a byte that opens or closes a list or a block, or ends a statement.
_STRUCTURE = re.compile(_COMMENT + rb"|[(){};]")
_BINARY = re.compile(rb"\bformat\s++binary\s*+;")
_BINARY_LIST = re.compile(rb"List<\s*+(\w+)\s*+>\s*+([0-9]++)\s*+\(")
#: Bytes per item of a binary list foamlib reads (64-bit scalars; Courant reads no other,
#: `_fluid_reader._binary_as_read`); a 32-bit label list is the smallest reading.
_ITEM_BYTES = {b"scalar": 8, b"vector": 24, b"symmTensor": 48, b"tensor": 72, b"label": 4}
#: A binary polyMesh file: its lists have no `List<T>` before them, and its header's `class` and
#: `arch` give their items (`courant.case.polymesh` reads these, and `parse_arch` the line).
_BARE_LIST = re.compile(rb"\b([0-9]++)\s*+\(")
_CLASS = re.compile(rb"\bclass\s++(\w+)\s*+;")
_ARCH = re.compile(rb'\barch\s++"([^"]*+)"')
_BARE_ITEMS = {b"labelList": ("label", 1), b"faceCompactList": ("label", 1),
               b"vectorField": ("scalar", 3)}

#: 2^18 failing paths: 0.3 s to 2 s of foamlib 1.7.10's time on the inputs measured, doubling
#: with each unit further.
MAX_EXPONENT = 18
#: A list that closes once comments are allowed is read in the end, after the uncommented pattern
#: has failed: valid, and refused only where that wait is a hang. Measured on a scalar field's
#: list with a comment in it: 2^24 paths 1.8 s, 2^28 38 s, so 2^32 about ten minutes.
MAX_EXPONENT_READ = 32
#: A field's list whose count is not its length goes to foamlib's general parser, whose time is
#: quadratic in the length, where a whole one matches at once. Counted in numbers, a vector's row as
#: three: at the bound of a hang, about ten minutes. Measured at nice 15 beside a solver sweep:
#: 1,000 numbers 11 s, 3,000 155 s; 500 rows 59 s, 1,000 rows 207 s; extrapolated, 5,000 numbers
#: about 8 minutes and 1,667 rows about 10. A file the solver refuses; below this it reads as 0.7.0
#: read it. Only a field: a dictionary's list takes that time whatever its count (600 numbers 2.2 s,
#: counted or not), and `x 2 (...)` is a number and a list, which foamlib reads.
MAX_MISMATCHED = 5000
#: The general parser tries three readings at every level of parentheses: 14 deep take 0.3 s,
#: each further level twice as long. No dictionary OpenFOAM ships nests half as deep.
MAX_DEPTH = 20


class Backtracking(ValueError):
    """The file holds a list foamlib's parser would take longer on than a verdict may wait.
    Said, like any file that could not be parsed."""


def _scalar_exponent(prefix: bytes, gaps: bool) -> int:
    """log2 of the paths a failing scalar pattern tries over `prefix`: each number of n
    characters splits 2^(n-1) ways; with `gaps` (the commented pattern) each run of w whitespace
    characters too, a comment counting as one."""
    if gaps:                                    # a comment's own text is no number
        prefix = _COMMENT_RE.sub(b" ", prefix)
    e = sum(len(t) - 1 for t in _TOKEN.findall(prefix))
    if gaps:
        e += sum(len(w) - 1 for w in _WS.findall(prefix))
    return e


def _groups_exponent(prefix: bytes) -> float:
    """log2 of the paths a failing fixed-shape pattern tries over a run of groups: each gap of w
    whitespace characters between two groups splits w + 1 ways."""
    gaps = re.findall(rb"\)(\s*+)\(", prefix)
    return sum(math.log2(len(g) + 1) for g in gaps)


def _count_before(data: bytes, p: int) -> int | None:
    """The list count written right before the `(` at `p` (`3(1 2 3)`), or None."""
    m = _LEAD_COUNT.search(data, max(0, p - 24), p)
    if m is None or (m.start() > 0 and data[m.start() - 1] in _WORD_BYTE):
        return None
    return int(m.group(1))


def _judge(data: bytes, p: int, field: bool) -> tuple[int | None, str | None, str | None]:
    """(where the list opening at `p` ends when it closes as numbers, or None; why foamlib would
    take too long on it wherever it stands, or None; what its number patterns would cost if they
    run on it, or None when that is below the bound).

    The cost is of the failing patterns, the uncommented first: it holds only where they run,
    which `check_bytes` decides (`_top_lists`, `_reached`). A list that closes once comments are
    allowed is read in the end, so it is held to the bound of a hang (`MAX_EXPONENT_READ`)."""
    start = p + 1
    # Closes as numbers: foamlib's pattern matches at once. Its count must be its length.
    for run, per_row in ((_RUN, 0), *zip(_SHAPED, (3, 6, 9), strict=True)):
        groups = per_row > 0
        m = run.match(data, start)
        if groups and m.end() == start:
            continue
        close = _CLOSE.match(data, m.end())
        if close is None:
            continue
        body = data[start:m.end()]
        n = body.count(b"(") if groups else len(body.split())
        count = _count_before(data, p)
        if field and count is not None and count != n and n * max(per_row, 1) >= MAX_MISMATCHED:
            return None, f"a list whose count, {count}, is not its length, {n}", None
        return close.end(), None, None
    # Fails uncommented; with comments allowed, a pattern may still close it.
    end_c = _RUN_C.match(data, start).end()
    scalar_closes = bool(_CLOSE_C.match(data, end_c))
    shaped_c = [g.match(data, start).end() for g in _SHAPED_C]
    closes = scalar_closes or any(e > start and _CLOSE_C.match(data, e) for e in shaped_c)
    bound = MAX_EXPONENT_READ if closes else MAX_EXPONENT
    end = _RUN.match(data, start).end()
    e = _scalar_exponent(data[start:end], gaps=False)
    if not scalar_closes:
        e = max(e, _scalar_exponent(data[start:end_c], gaps=True))
    if e >= bound and not (field and e < MAX_EXPONENT_READ and _reads_after(data, p, end_c)):
        return None, None, f"a list foamlib tries as numbers and cannot match (2^{e} readings)"
    for g in _SHAPED:
        m = g.match(data, start)
        if m.end() > start:
            ge = _groups_exponent(data[start:m.end()])
            if ge >= bound:
                return None, None, (f"a list of {data.count(b'(', start, m.end())} groups of "
                                    f"numbers foamlib cannot match (2^{ge:.0f} readings)")
    return None, None, None


def _reads_after(data: bytes, p: int, stop: int) -> bool:
    """Whether a field's list that fails as numbers may yet be read by foamlib's general parser,
    after its patterns have failed: it balances, and the byte at `stop`, after its numeric prefix,
    continues a list (`3(1.2345678901 2.3456789012 $z)`). So it is held to the bound of a hang."""
    if stop >= len(data) or data[stop] not in _CONTINUES_A_LIST:
        return False
    depth = 0
    for m in _STRUCTURE.finditer(data, p):
        c = m.group()
        if c == b"(":
            depth += 1
        elif c == b")":
            depth -= 1
            if not depth:
                return True
    return False


def _top_lists(data: bytes) -> dict[int, tuple[int, bool]]:
    """Each list that stands at the top of the file, outside any `{ }` and any other list:
    {its `(`: (where its statement starts, whether the list closes)}. A `}` with no `{` open
    ends foamlib's reading (`dictionary::read`), so nothing after it is read."""
    tops: dict[int, tuple[int, bool]] = {}
    braces = parens = statement = 0
    top = -1
    for m in _STRUCTURE.finditer(data):
        c = m.group()
        if c == b"(":
            if not parens and not braces:
                top = m.start()
                tops[top] = (statement, False)
            parens += 1
        elif c == b")":
            parens = max(parens - 1, 0)
            if not parens and not braces and top in tops:
                tops[top] = (tops[top][0], True)
        elif c == b"{":
            braces += 1
        elif c == b"}":
            if not braces and not parens:
                break
            braces = max(braces - 1, 0)
            if not braces and not parens:
                statement = m.end()
        elif c == b";" and not braces and not parens:
            statement = m.end()
    return tops


def _statement_reads(data: bytes, statement: int, p: int) -> bool:
    """Whether foamlib reads the file from `statement` past the `(` at `p` as dictionary entries,
    never falling back to standalone data: the step of `_parse_file`, with foamlib's own parser.
    Its time is what foamlib spends on that statement in any case. A failure foamlib does not
    fall back from says yes (it stops there, before any pattern), and so does a foamlib without
    these functions: nothing is known of it, and a valid file is never refused on a guess."""
    try:
        from foamlib._files._parsing import _parser as fp

        not_read = fp.ParseError
    except (ImportError, AttributeError):
        return True
    pos = statement
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")             # foamlib on a repeated key
            while True:
                pos = fp._skip(data, pos)
                if pos > p:
                    return True
                if data[pos:pos + 1] == b";":
                    pos += 1
                    continue
                keyword, pos = fp._parse_token(data, pos)
                pos = fp._skip(data, pos)
                if keyword.startswith("#"):
                    value, pos = fp._parse_data_entry(data, pos)
                    _, pos = fp._parse_directive_value(data, keyword, value, pos)
                    continue
                try:
                    _, pos = fp._parse_subdictionary(data, pos)
                    continue
                except not_read:
                    pass
                try:
                    value, pos = fp._parse_data(data, pos)
                except not_read:
                    value = None
                if keyword.startswith("$") and value is None:
                    if data[pos:pos + 1] == b";":
                        pos += 1
                else:
                    pos = fp._expect(data, pos, b";")
    except not_read:
        return False
    except Exception:                                   # noqa: BLE001 - foamlib's own stop
        return True


def _reached(data: bytes, p: int, statement: int, closes: bool) -> bool:
    """Whether foamlib runs its number patterns on the top-level list at `p`. In linear time when
    the list cannot be read: it never closes, or the byte after its numeric prefix can continue
    no list. Otherwise only foamlib's parser knows, and is asked."""
    if not closes:
        return True
    stop = _RUN_C.match(data, p + 1).end()
    if stop >= len(data) or data[stop] not in _CONTINUES_A_LIST:
        return True
    return not _statement_reads(data, statement, p)


def _visible(data: bytes) -> bytes:
    """`data` as foamlib's list patterns see it: binary payloads cut out, strings and verbatim
    code each reduced to a placeholder (their parentheses are not parentheses)."""
    if _BINARY.search(data[:4096]):
        bare = _bare_item(data[:4096])
        parts, pos = [], 0
        for m in (_BARE_LIST if bare else _BINARY_LIST).finditer(data):
            if m.start() < pos:
                continue
            size = bare or _ITEM_BYTES.get(m.group(1))
            if size is None:
                continue
            parts.append(data[pos:m.end()])
            pos = min(len(data), m.end() + int(m.group(m.lastindex)) * size)
        parts.append(data[pos:])
        data = b"".join(parts)
    return _OPAQUE.sub(lambda m: m.group() if m.group()[:1] == b"/" else b"_", data)


def _bare_item(head: bytes) -> int | None:
    """Bytes per item of the lists of a binary polyMesh file, from its header, or None for any
    other file or an `arch` line that does not say."""
    from courant.case.polymesh import parse_arch

    klass, arch = _CLASS.search(head), _ARCH.search(head)
    item = _BARE_ITEMS.get(klass.group(1)) if klass else None
    if item is None or arch is None:
        return None
    bits = parse_arch(arch.group(1).decode("ascii", "replace"))[1].get(item[0], "")
    return int(bits) // 8 * item[1] if bits in ("32", "64") else None


def _opens(data: bytes) -> list[int]:
    """The offsets of every `(` outside a comment."""
    out, pos = [], 0
    for m in _COMMENT_RE.finditer(data):
        out.extend(_offsets(data, pos, m.start()))
        pos = m.end()
    out.extend(_offsets(data, pos, len(data)))
    return out


def _offsets(data: bytes, a: int, b: int):
    i = data.find(b"(", a, b)
    while i != -1:
        yield i
        i = data.find(b"(", i + 1, b)


def check_bytes(data: bytes, shown: str = "the file") -> None:
    """Raise `Backtracking` if foamlib's parser would take seconds or more on `data`."""
    data = _visible(bytes(data))
    _check_depth(data, shown)
    fields = {m.end() - 1 for m in _FIELD_LIST.finditer(data)}

    def refused(p: int, why: str) -> Backtracking:
        return Backtracking(f"{shown} holds {why} at byte {p}; it is not parsed, since the "
                            f"parser's time on it grows faster than its length")

    # Linear overall: a list that closes as numbers is judged once, from its own `(`, and the
    # `(`s inside it are passed over. Every field is judged before foamlib's parser is asked
    # about any list, since that parser runs the patterns on a field.
    skip_to = -1
    costly = []
    for p in _opens(data):
        if p < skip_to:
            continue
        field = p in fields
        end, why, cost = _judge(data, p, field)
        if why is not None:
            raise refused(p, why)
        if cost is not None:
            if field:
                raise refused(p, cost)
            costly.append((p, cost))
        if end is not None:
            skip_to = end
    if costly:
        tops = _top_lists(data)
        for p, cost in costly:
            if p in tops and _reached(data, p, *tops[p]):
                raise refused(p, cost)


def _check_depth(data: bytes, shown: str) -> None:
    """Refuse parentheses nested `MAX_DEPTH` deep outside comments. A `)` with no `(` open
    counts for nothing: a depth never goes below zero."""
    import numpy as np

    a = np.frombuffer(_COMMENT_RE.sub(b" ", data), dtype=np.uint8)
    parens = a[(a == ord("(")) | (a == ord(")"))]
    if parens.size < MAX_DEPTH:
        return
    total = np.cumsum(np.where(parens == ord("("), 1, -1))
    depth = total - np.minimum(np.minimum.accumulate(total), 0)
    if int(depth.max()) >= MAX_DEPTH:
        raise Backtracking(f"{shown} nests parentheses {MAX_DEPTH} deep; it is not parsed, "
                           f"since the parser's time doubles with each level")


@contextmanager
def guarded(on: bool) -> Iterator[None]:
    """Readers within ask the guard when `on`: `enabled(FOAM_GUARD, checks)`, where the mode is
    known. Outside any, they do not."""
    token = _ASKED.set(on)
    try:
        yield
    finally:
        _ASKED.reset(token)


def check_path(path: str | os.PathLike) -> None:
    """`check_bytes` on what foamlib reads at `path` (decompressed when it ends in `.gz`), when
    `guarded`; else nothing, not even the read. An `OSError` propagates, as foamlib's own would."""
    if not _ASKED.get():
        return
    path = os.fspath(path)
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".gz"):
        data = gzip.decompress(data)
    check_bytes(data, os.path.basename(path))


def check_text(text: str, shown: str = "the dictionary") -> None:
    """`check_bytes` on expanded text, when `guarded`; else nothing."""
    if not _ASKED.get():
        return
    check_bytes(text.encode("utf-8", errors="replace"), shown)


__all__ = ["MAX_DEPTH", "MAX_EXPONENT", "MAX_EXPONENT_READ", "MAX_MISMATCHED", "Backtracking",
           "check_bytes", "check_path", "check_text", "guarded"]
