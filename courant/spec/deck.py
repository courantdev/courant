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
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    # `${name}` is `$name`: its braces are not a block's, and counted as one they hid every
    # entry after them (TST-14's fuzzer found it). Followed by a word character the braces
    # delimit the name, so it is left alone and stays unresolved rather than misread.
    return _BRACED_MACRO.sub(r"$\1", text)


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
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
    return None


#: A block starts a line or follows a `;`, `{` or `}`: `inlet { ... } outlet { ... }` is two blocks.
_BLOCK_HEAD = re.compile(r'(?m)(?:^|(?<=[;{}]))[ \t]*(?:"([^"\n]*)"|([^\s{};"]+))\s*\{')


def blocks(text: str) -> list[tuple[str, str]]:
    """Every top-level `key { ... }` in `text`, as (key, body), in order. Quotes are kept off."""
    depth = _depths(text)
    out = []
    for m in _BLOCK_HEAD.finditer(text):
        if depth[m.start()] != 0:
            continue
        body = _body_after(text, m.end() - 1)
        if body is not None:
            out.append((m.group(1) if m.group(1) is not None else m.group(2), body))
    return out


def block(text: str, name: str) -> str | None:
    """The body of the top-level block `name`, matching OpenFOAM's lookup.

    A literal key wins; otherwise the last quoted key that matches `name` as a regex, which is
    how a boundary condition grouped as `"(inlet|outlet)"` is found.

    A block written more than once is merged, as OpenFOAM's default `#inputMode merge` does:
    the bodies are read as one, in order, so each repeated entry takes its last definition
    (`entry`) and each repeated inner block merges in turn.
    """
    found = blocks(text)
    literal = [body for key, body in found if key == name]
    if literal:
        return "\n".join(literal)
    match = None
    for key, body in found:
        try:
            if re.fullmatch(key, name):
                match = body
        except re.error:
            continue
    return match


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


def _entries(text: str, key: str) -> list[str]:
    """Every top-level definition of `key`, in order."""
    return [re.sub(r"\s+", " ", m.group(1)).strip() for m in re.finditer(
        rf"(?m)(?:^|(?<=[;{{}}]))[ \t]*{re.escape(key)}\s+([^;{{}}]*);", _top_level(text))]


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
            try:
                inc_path = (path.parent / inc).resolve()
                inc_path.relative_to(case_root.resolve())
                is_file = inc_path.is_file()
            except (ValueError, OSError):                # outside the case, or no path at all
                continue                                 # (a NUL byte: TST-14's fuzzer)
            if is_file:
                try:
                    inc_text = strip_comments(inc_path.read_text(errors="replace"))
                except OSError:
                    continue                             # not read: the macro stays unresolved
                reads.note(path.parent / inc)    # VER-14: by the path used, not its resolution
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
        return None, dims, f"no number in '{raw[:40]}'"
    if raw.lstrip().startswith("("):
        return nums, dims, None
    return nums[-1] if len(nums) > 1 else nums[0], dims, None


# ---------------------------------------------------------------- the reader
def _read_text(case: Path, rel: str) -> tuple[str | None, str | None]:
    p = case / rel
    if not p.is_file():
        return None, f"{rel} does not exist"
    try:
        text = p.read_text(errors="replace")
    except OSError as exc:
        return None, f"{rel} could not be read: {exc}"
    reads.note(p)                                        # VER-14
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
        if not p.is_file():
            return Found(None, rel, f"{rel} does not exist")
        try:
            data = p.read_bytes()
        except OSError as exc:
            return Found(None, rel, f"{rel} could not be read: {exc}")
        reads.note(p)                                    # VER-14
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
    raw = entry(scope, key)
    if raw is None and rel.endswith("blockMeshDict") and key in _ALIASES and not item.path:
        other, fallback = _ALIASES[key]                 # one setting under two names
        raw = entry(scope, other)
        if raw is None and _complete(text):
            return Found(float(fallback), rel, basis=f"not written; OpenFOAM's default {fallback}")
    if raw is None:
        return Found(None, rel, f"no entry '{key}'")
    if _DIRECTIVE.search(raw):
        return Found(None, rel, f"'{raw[:40]}' is a directive the reader does not evaluate")
    resolved = _resolve_macros(raw, text, case / rel, case)
    if resolved is None:
        return Found(None, rel, f"'{raw[:40]}' refers to a value that could not be found")
    if _DIRECTIVE.search(resolved):
        # A macro that expands to `#calc ...` is no more a number than `#calc` written out:
        # reading its last number misread `nu #calc "$a*2"` as 2 (review, 10 Oct 2026).
        return Found(None, rel, f"'{raw[:40]}' expands to a directive the reader does not "
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
    try:
        found = _read_one(case, item.file, item)
        if found.value is None and item.file.startswith("0/") and not (case / item.file).is_file():
            orig = "0.orig/" + item.file[2:]
            if (case / orig).is_file():
                alt = _read_one(case, orig, item)
                if alt.value is not None:
                    return alt
        return found
    except Exception as exc:                            # noqa: BLE001 - a reader limit, not a verdict
        return Found(None, item.file, f"unreadable: {type(exc).__name__}: {exc}")
