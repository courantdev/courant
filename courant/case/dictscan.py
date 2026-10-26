"""The entries of a dictionary file as v2406 reads its statements, in one pass (TRD VER-13a).

`scan` reads a dictionary file's text the way v2406's tokenizer splits it into statements, and
gives every keyword it holds, with the dictionaries it lies in, or None for a file it cannot
follow all of. It is a reading of structure only: no value is parsed, no directive is followed.
A caller asks it whether foamlib's reading of the file is the solver's, entry for entry
(`courant.case._foam_reader._ddt_read_whole`), and anything this scan does not follow is an
answer of no:

- **Comments** as `ISstream::nextValid` reads them (src/OpenFOAM/db/IOstreams/Sstreams/
  ISstream.C): `//` through the end of its line, `/*` through the first `*/` after it, each only
  between tokens. A word ends at `/` (`word::valid`, src/OpenFOAM/primitives/strings/word/
  wordI.H, refuses whitespace, `"`, `'`, `/`, `;`, `{` and `}`), so `abc//x` is the word `abc` and
  a comment, as the solver reads it. A `/*` with no `*/` after it is a file this scan does not
  follow. A comment marker inside a string is string, as `ISstream::read(string&)` reads it.
- **Strings** run to the next `"` not escaped by a backslash; one that reaches the end of its
  line is not followed (v2406: "Unescaped '\\n' while reading string"). A string value is read
  as a string, whatever it holds; a quoted keyword is a keyword (the next item).
- **Statements** as `entry::New` reads them (src/OpenFOAM/db/dictionary/entry/entryIO.C): a
  keyword (a word, or a quoted keyword), then a dictionary in braces, or a value to the `;` that
  ends it at depth zero (`primitiveEntry::read`, primitiveEntryIO.C); a spurious `;` between
  statements is skipped (`entry::getKeyword`).
- **Not followed**, so no: a `#` or a `$` outside a comment and a string value (a directive,
  which may set entries anywhere: `#include`, `#inputMode`, `#remove`, a conditional, a verbatim
  `#{`; a macro, or a `$name;` dictionary merge, `dictionary::substituteScopedKeyword`); a `/`
  that does not start a comment, and a quoted keyword holding `/`, `#` or `$` (a keyword holding
  `/` is a slash-scoped entry, which sets an entry of another dictionary: entryIO.C,
  `keyword.contains('/')`, `makeScopedDict`; ISstream.C reads an unquoted `/` as the divide
  token, so only a quoted keyword can hold one); a single quote; a keyword that does not begin
  with a letter, `_` or `"` (a number or a `(` there begins a list of entries,
  `dictionaryListEntry`); a brace in a value, or a `;` inside its parentheses; unbalanced braces
  or parentheses; a value with no `;`; and any character outside a comment that is neither
  printable ASCII nor ASCII whitespace (v2406 reads bytes, and only the ASCII whitespace is
  `isspace`).

Linear in the file's length: every character is looked at a bounded number of times, and each
comment's end is found with `str.find` (the review of hp6/final: the deck reader's
`/\\*.*?\\*/` and its entry pattern took time in the square of the length on unclosed comment
openers in a string and on a keyword written on every line of a quoted keyword, and held
`courant verify <case>`).
"""

from __future__ import annotations

from dataclasses import dataclass

#: `isspace` on a byte, as ISstream.C asks it.
_SPACE = frozenset(" \t\n\r\f\v")
#: What ends a word (`word::valid`), beside whitespace, and the list punctuation this scan
#: counts.
_ENDS_WORD = frozenset("\"'/;{}()") | _SPACE
#: What this scan does not follow outside a comment (the module docstring).
_REFUSED = frozenset("#$'")
#: The characters a keyword may begin with.
_KEY_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_\"")


@dataclass(frozen=True)
class Entry:
    #: The keywords of the dictionaries it lies in, outermost first, as written; () at the top.
    scope: tuple[str, ...]
    #: Its keyword as written, a quoted keyword with its quotes.
    key: str
    #: Whether it is a dictionary (a keyword followed by braces).
    dictionary: bool


@dataclass(frozen=True)
class Scan:
    entries: tuple[Entry, ...]

    def keys(self, *scope: str) -> list[str]:
        """The keywords written directly in the dictionary at `scope`, in order, as written."""
        return [e.key for e in self.entries if e.scope == scope]

    def at(self, *scope: str) -> list[Entry]:
        return [e for e in self.entries if e.scope == scope]


def unquoted(key: str) -> str:
    """A keyword's text: a quoted keyword without its quotes (v2406's dictionary holds a keyword
    by its text, quoted or not, `dictionary::csearch`'s hash lookup)."""
    return key[1:-1] if len(key) >= 2 and key[0] == key[-1] == '"' else key


def _plain(c: str) -> bool:
    return c in _SPACE or " " <= c <= "~"


def _skip(text: str, i: int) -> int:
    """The index of the next token at or after `i`, past whitespace and comments; len(text) at
    the end; -1 at a `/` that starts no comment, an unclosed `/*` or a character not followed."""
    n = len(text)
    while i < n:
        c = text[i]
        if c in _SPACE:
            i += 1
        elif c == "/":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == "/":
                j = text.find("\n", i + 2)
                i = n if j < 0 else j + 1
            elif nxt == "*":
                j = text.find("*/", i + 2)
                if j < 0:
                    return -1
                i = j + 2
            else:
                return -1
        elif not _plain(c) or c in _REFUSED:
            return -1
        else:
            return i
    return n


def _string_end(text: str, i: int, *, key: bool = False) -> int:
    """The index after the string that opens at `i`, or -1 when it is not followed: no closing
    quote on its line, a control character or one that is not ASCII in it, or, in a quoted
    keyword, a `#`, a `$` or a `/` (a quoted keyword holding `/` is slash-scoped too, entryIO.C)."""
    refused = "#$/" if key else ""
    escaped = False
    j = i + 1
    n = len(text)
    while j < n:
        c = text[j]
        if c == "\\":
            escaped = not escaped
        elif c == '"' and not escaped:
            return j + 1
        elif c == "\n" or not (" " <= c <= "~" or c == "\t") or c in refused:
            return -1
        else:
            escaped = False
        j += 1
    return -1


def _word_end(text: str, i: int) -> int:
    """The index after the word that starts at `i`, as `ISstream::read(word&)` reads it: up to a
    character `word::valid` refuses, with `(` and `)` part of it while balanced, and a `)` with
    no `(` before it in the word not part of it. -1 when its parentheses do not close."""
    depth = 0
    j = i
    n = len(text)
    while j < n:
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                break
            depth -= 1
        elif c in _ENDS_WORD or c in _REFUSED or not _plain(c):
            break
        j += 1
    return -1 if depth else j


def scan(text: str) -> Scan | None:
    """Every keyword `text` writes, in order, with the dictionaries it lies in; None when the
    text is not one this scan follows all of (the module docstring)."""
    n = len(text)
    entries: list[Entry] = []
    scope: list[str] = []
    i = 0
    while True:
        i = _skip(text, i)
        if i < 0:
            return None
        if i >= n:
            break
        c = text[i]
        if c == ";":                                  # a spurious terminator
            i += 1
            continue
        if c == "}":
            if not scope:
                return None
            scope.pop()
            i += 1
            continue
        if c not in _KEY_START:
            return None
        j = _string_end(text, i, key=True) if c == '"' else _word_end(text, i)
        if j < 0:
            return None
        key = text[i:j]
        i = _skip(text, j)
        if i < 0 or i >= n:
            return None
        if text[i] == "{":
            entries.append(Entry(tuple(scope), key, True))
            scope.append(key)
            i += 1
            continue
        entries.append(Entry(tuple(scope), key, False))
        i = _value_end(text, i)
        if i < 0:
            return None
    if scope:
        return None
    return Scan(tuple(entries))


def _value_end(text: str, i: int) -> int:
    """The index after the `;` that ends the value starting at `i`, or -1."""
    n = len(text)
    depth = 0
    while True:
        i = _skip(text, i)
        if i < 0 or i >= n:
            return -1
        c = text[i]
        if c == ";" and depth == 0:
            return i + 1
        if c in ";{}":
            return -1
        if c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return -1
            i += 1
        elif c == '"':
            i = _string_end(text, i)
            if i < 0:
                return -1
        else:
            j = _word_end(text, i)
            if j <= i:
                return -1
            i = j


__all__ = ["Entry", "Scan", "scan", "unquoted"]
