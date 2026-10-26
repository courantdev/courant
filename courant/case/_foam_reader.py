"""Read a real OpenFOAM case directory into a CaseConfig (live path).

Uses foamlib to parse the dictionaries that matter for diagnosis. Best-effort and
defensive: a missing file or key is skipped, never fatal — real cases vary. This is
what makes the live DockerRunner path work end to end (config_view over real files).
"""

from __future__ import annotations

import os
import re

from courant.case import wordre
from courant.case.config import CaseConfig, LinearSolverSpec, phase_pseudonym
from courant.case.reads import may_read, reading


def _scalar(v):
    """Extract a float from foamlib's varied return types (number, dimensioned, tuple)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "value"):
        try:
            return float(v.value)  # foamlib Dimensioned
        except Exception:  # noqa: BLE001
            return None
    if isinstance(v, (tuple, list)) and v:
        last = v[-1]
        return float(last) if isinstance(last, (int, float)) else None
    return None


def _tokens(v) -> str | None:
    """Render a scheme value (tuple of tokens or string) as a string."""
    if isinstance(v, (tuple, list)):
        return " ".join(str(t) for t in v)
    if v is None:
        return None
    return str(v)


def _get(f, *key):
    if isinstance(f, _Dictionary):
        return f.need(*(key[0] if len(key) == 1 and isinstance(key[0], tuple) else key))
    try:
        return f[key if len(key) > 1 else key[0]]
    except Exception:  # noqa: BLE001 — missing key, or an unreadable file
        return None


#: The words OpenFOAM's `Switch` reads (`Switch.C`), each to its value.
_SWITCH = {"true": True, "on": True, "yes": True, "y": True, "t": True,
           "false": False, "off": False, "no": False, "n": False, "f": False, "none": False}


def _solver_log(v) -> float | None:
    """A solver entry's `log`, as a number (a word foamlib reads as a boolean counts as one), or
    None when it is not written or is not a number."""
    if isinstance(v, bool):
        return float(v)
    return _scalar(v)


def _switch(v) -> bool | None:
    """A `Switch` entry as OpenFOAM reads it, or None when absent or not a switch word."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return _SWITCH.get(str(v).strip().lower()) if isinstance(v, str) else None


# ---------------------------------------------------------------- directives (TRD VER-3b)
#: Directive keys that set nothing the checks read: `#inputMode` changes how a repeated key is
#: read (its own rule, `deck.repeated_under_input_mode`); `#includeFunc` and `#includeModel` add
#: a function object or a model.
_INERT = frozenset({"#inputMode", "#includeFunc", "#includeModel"})
#: The include directives `deck.expand` follows inside the case.
_FOLLOWED_INCLUDES = ("#include", "#includeIfPresent", "#sinclude")


def _unexpanded(v) -> str | None:
    """The macro or directive a value is written as, when foamlib returns it unexpanded
    (`$nuVal`, `#calc "0.25*2"`, `(DimensionSet, '$nuVal')`), else None."""
    if isinstance(v, str):
        return v if v.startswith(("$", "#")) else None
    if isinstance(v, (tuple, list)):
        words = [x for x in v if isinstance(x, str)]
        for i, w in enumerate(words):
            if w.startswith(("$", "#")):
                return " ".join(words[i:])
    return None


def _entries_of(scope) -> list[tuple[str, object]]:
    """(key, value) of each entry of one dictionary level, in order, a repeated directive key
    (two `#include`s) once per occurrence. By `keys()`: a `FoamFile`'s own `items()` walks every
    level at once."""
    if not hasattr(scope, "keys"):
        return []
    try:
        keys = [k for k in scope.keys() if k is not None]
    except Exception:  # noqa: BLE001
        return []
    out, seen = [], {}
    for k in keys:
        i = seen[k] = seen.get(k, -1) + 1
        try:
            v = list(scope.getall(k))[i] if keys.count(k) > 1 and hasattr(scope, "getall") \
                else scope[k]
        except Exception:  # noqa: BLE001
            v = None
        out.append((str(k), v))
    return out


def _hiding(name: str) -> bool:
    """Whether a key foamlib returns is a directive that may set the entries around it: an
    include, a `$name;` dictionary merge, `#remove`, `#default`, ..."""
    return name.startswith("$") or (name.startswith("#") and name not in _INERT)


def _shown(name: str, value) -> str:
    if name.startswith("$") or value is None:
        return name
    words = value if isinstance(value, (tuple, list)) else (value,)
    return " ".join([name, *(str(w) for w in words)])


def _names(keys, name: str) -> bool:
    """Whether one of a dictionary's `keys` names `name` for certain: the name itself, or a
    quoted keyword, a regular expression the case wrote, that matches it in full. Matched with
    bounded work (`courant.case.wordre`): Python's `re` backtracks exponentially on `"(a+)+b"`
    against a long name, holding the interpreter's lock, so `courant verify <case>` and
    `courant preflight` hung on a `$name;` in a solver entry (final review, 21 Oct 2026). A
    keyword the matcher does not evaluate is not taken to name it, so the `$name;` is noted as
    not read."""
    def pattern(key):
        quoted = isinstance(key, str) and len(key) > 1 and key[0] == key[-1] == '"'
        return key[1:-1] if quoted else None

    hit, _ = wordre.first_match([str(k) for k in keys], name, pattern=pattern)
    return hit is not None


class _Dictionary:
    """One case dictionary, for the values the checks read (TRD VER-3b).

    foamlib expands no macro and follows no include: it returns `$nuVal` as written and an
    `#include` as a key, and `_scalar` made the first None, so a viscosity of 0.5 behind a macro
    was a viscosity not checked, in silence. Every value is read through `need`, which notes it as
    not read when foamlib returns it unexpanded or a directive in its scope may set it (any
    directive when it is absent; one after it when it is written: OpenFOAM reads in order), and
    when its file writes it twice under an `#inputMode` other than merge. The value itself is
    returned as foamlib read it, so a verdict under the default checks moves on nothing (GATE-13);
    what was not read reaches `CaseConfig.unread_values` and "not assessed".

    With `follow` (`--checks next`, `case.directives`), a dictionary with such a value is read
    again as the solver reads it (`courant.spec.deck.expand`: its includes inside the case
    followed, its macros expanded), once, and only what still cannot be read is noted. A solver
    entry that merges another (`pFinal { $p; relTol 0; }`, as every tutorial writes it) inherits
    values the checks read on that entry, so the merge is not noted.
    """

    def __init__(self, raw, path: str, rel: str, case_dir: str, follow: bool):
        self.raw, self.path, self.rel, self.case_dir, self.follow = raw, path, rel, case_dir, follow
        self.expanded = None
        self._tried = False
        self._text: str | None = None
        self.hidden: list[tuple[str, str, str]] = []        # (where, kind, as written)

    @property
    def source(self):
        return self.expanded if self.expanded is not None else self.raw

    def text(self) -> str | None:
        if self._text is None:
            from pathlib import Path

            from courant.spec import deck

            self._text, _ = deck._read_text(Path(self.case_dir), self.rel)
        return self._text

    def as_written(self) -> str | None:
        """The file's text as written, its comments kept (`courant.case.dictscan` reads them as
        the solver does, where `text` strips them without regard to strings), or None when it
        cannot be read. Read when asked, for TRD VER-13a under `--checks next` alone, and noted
        as `text` notes it (VER-14; MCP-3 asked first)."""
        from courant.case.reads import reading

        try:
            with reading(self.path):
                with open(self.path, "rb") as f:
                    return f.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — outside the roots, gone, not a file: not read
            return None

    def expand(self) -> bool:
        """Read the dictionary as the solver reads it, once; whether that gave a dictionary."""
        if not self._tried:
            self._tried = True
            import warnings
            from pathlib import Path

            from foamlib import FoamFile

            from courant.spec import deck

            text = self.text()
            if text is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")     # foamlib on a merged repeated key
                        self.expanded = FoamFile.loads(
                            deck.expand(text, Path(self.path), Path(self.case_dir)))
                except Exception:  # noqa: BLE001 — what stays unread is noted as such
                    self.expanded = None
        return self.expanded is not None

    def need(self, *keys):
        """The value at `keys` as foamlib read it (None when absent), noting it when it was not
        read as the solver read it."""
        value, why = self._look(keys)
        if why is not None and self.follow and self.expand():
            value, why = self._look(keys)
        if why is not None:
            self.hidden.append(why)
        return value

    def scope(self, *keys):
        """The dictionary at `keys`, for a reader that reads each of its entries: under `follow`
        read as the solver reads it first, when anything in it is not as written; a directive
        among its entries is noted (an entry it may set is not read)."""
        if self.follow and _tainted(_get_quiet(self.source, *keys)):
            self.expand()
        found = self.need(*keys)
        for name, v in _entries_of(found):
            if _hiding(name) and not self._inherited(name, keys):
                self.hidden.append(("/".join(keys), "directive", _shown(name, v)))
        return found

    def _look(self, keys) -> tuple[object, tuple[str, str, str] | None]:
        scope, value = self.source, None
        for depth, key in enumerate(keys):
            entries = _entries_of(scope)
            names = [k for k, _ in entries]
            at = names.index(key) if key in names else None
            for j, (name, v) in enumerate(entries):
                if (at is None or j > at) and _hiding(name) and \
                        not self._inherited(name, keys[:depth]):
                    try:
                        found = _walk(self.source, keys)
                    except Exception:  # noqa: BLE001
                        found = None
                    return found, ("/".join(keys[:depth + 1]), "directive", _shown(name, v))
            if at is None:
                return None, None
            try:
                scope = scope[key]
            except Exception:  # noqa: BLE001
                return None, None
            value = scope
        token = _unexpanded(value)
        if token is not None:
            return value, ("/".join(keys), "token", token)
        if "#inputMode" in _keys(self.source):
            from courant.spec import deck

            text = self.text()
            mode = deck.repeated_under_input_mode(text, tuple(keys)) if text else None
            if mode is not None:
                return value, ("/".join(keys), "inputMode", mode)
        return value, None

    def _inherited(self, name: str, scope_keys) -> bool:
        """A `$name;` in a solver entry naming another solver entry of the same `solvers`."""
        if not name.startswith("$") or len(scope_keys) != 2 or scope_keys[0] != "solvers":
            return False
        return _names([k for k, _ in _entries_of(_get_quiet(self.source, "solvers"))], name[1:])

    def report(self) -> str | None:
        """What was not read, in a sentence per cause, for `CaseConfig.unread_values`: local, it
        quotes what the case wrote (bounded, `_quoted`)."""
        groups: dict[tuple[str, str], list[str]] = {}
        for where, kind, shown in self.hidden:
            wheres = groups.setdefault((kind, shown), [])
            if where not in wheres:
                wheres.append(where)
        return "; ".join(self._sentence(kind, shown, wheres)
                         for (kind, shown), wheres in groups.items()) or None

    def keys_report(self, phases=()) -> str | None:
        """The same for `CaseConfig.unread_keys` and so the finding `case.unread_value`, which
        reaches the Evidence Bundle: the key paths the reader read and how each is written
        (`_kind`), never what was written (a macro's name, an include's path, an expression: the
        customer's text, which pattern scrubbing cannot recognise; the review of 21 Oct 2026). A
        key path is OpenFOAM's keywords and the entry names the case writes, as written (a
        solver entry's field or pattern, `solvers/U`), with every phase name in it, one of
        `phases`, written as its place in that list (`crudeOilX/nu` is `phase1/nu`,
        `solvers/alpha.crudeOilX/tolerance` is `solvers/alpha.phase1/tolerance`;
        `config.phase_pseudonym`, the final review of 0.6.0)."""
        groups: dict[str, list[str]] = {}
        for where, kind, shown in self.hidden:
            wheres = groups.setdefault(_kind(kind, shown), [])
            where = _pseudonymous(where, phases)
            if where not in wheres:
                wheres.append(where)
        return "; ".join(f"{', '.join(wheres)} ({kind})" for kind, wheres in groups.items()) \
            or None

    def _sentence(self, kind: str, shown: str, wheres: list[str]) -> str:
        what = ", ".join(wheres)
        verb = "is" if len(wheres) == 1 else "are"
        written = _quoted(shown)
        if kind == "inputMode":
            return (f"{what} {verb} written more than once under '#inputMode {written}', whose "
                    "rule Courant does not follow")
        if kind == "token":
            if not shown.startswith("$"):
                return f"{what} {verb} written as '{written}', which Courant does not evaluate"
            if self.follow:
                return f"{what} {verb} written as '{written}', a macro that could not be expanded"
            return (f"{what} {verb} written as '{written}', a macro the default checks do not "
                    "expand (--checks next does)")
        head = f"{what} may be set by '{written}', "
        directive = shown.split()[0]
        if directive in _FOLLOWED_INCLUDES:
            from pathlib import Path

            from courant.spec import deck

            inc = shown[len(directive):].strip().strip('"')
            _, why = deck.include_target(Path(self.path), inc, Path(self.case_dir))
            if why is not None:
                return head + f"which could not be followed: {why}"
        elif not directive.startswith("$"):
            return head + "which Courant does not follow"
        if self.follow:
            return head + "which could not be followed"
        return head + "which the default checks do not follow (--checks next does)"


#: The most of what a case wrote the local sentence quotes.
_QUOTED = 60


def _pseudonymous(where: str, phases) -> str:
    """`where`, a key path, with each of `phases` it names as a whole word (between characters
    that are not letters, digits or `_`: `alpha.water`, `water/nu`) written as its place in
    `phases` (`config.phase_pseudonym`)."""
    listed = [str(p) for p in phases or () if str(p)]
    if not listed:
        return where
    words = "|".join(re.escape(p) for p in sorted(listed, key=len, reverse=True))
    return re.sub(rf"(?<![A-Za-z0-9_])(?:{words})(?![A-Za-z0-9_])",
                  lambda m: phase_pseudonym(listed, m.group(0)), where)


def _kind(kind: str, shown: str) -> str:
    """How a value not read is written, in the finding's fixed vocabulary (`keys_report`):
    `macro` for a value written as `$name`; `directive` for one written as `#calc ...`, or set
    by a directive in its scope (an include, a `$name;` merge, `#remove`); `#inputMode` for a key
    written twice under it."""
    if kind == "token":
        return "macro" if shown.startswith("$") else "directive"
    return "#inputMode" if kind == "inputMode" else "directive"


def _quoted(shown: str) -> str:
    return shown if len(shown) <= _QUOTED else shown[:_QUOTED].rstrip() + "..."


def _keys(scope) -> list:
    try:
        return list(scope.keys()) if hasattr(scope, "keys") else []
    except Exception:  # noqa: BLE001
        return []


def _walk(src, keys):
    v = src
    for k in keys:
        v = v[k]
    return v


def _tainted(v) -> bool:
    """Whether anything in `v` is written through a directive or a macro."""
    if _unexpanded(v) is not None:
        return True
    return any(_hiding(k) or _tainted(x) for k, x in _entries_of(v))


def _get_quiet(src, *keys):
    try:
        return _walk(src, keys)
    except Exception:  # noqa: BLE001
        return None


#: The file whose presence makes a case multi-region: v2406's chtMultiRegionFoam reads its
#: regions from it (`regionProperties rp(runTime)`, createMeshes.H; src/meshTools/regionModel/
#: regionProperties/regionProperties.C reads `constant/regionProperties`), and each region reads
#: its own `system/<region>/fvSchemes`.
REGION_PROPERTIES = os.path.join("constant", "regionProperties")


def _ddt_read_whole(fs: _Dictionary, case_dir: str) -> bool:
    """TRD VER-13a, under `--checks next` only (`cfl.steady`): whether foamlib's reading of
    `ddtSchemes` in `system/fvSchemes` is the solver's, every entry, so that its `default` and
    its other entries (`CaseConfig.ddt_terms`) are the case's declaration
    (`CaseConfig.ddt_read_whole`). Asked right after `ddtSchemes/default` is read. Yes only when
    every one of these holds, else no, and the run is judged as v0.5.0 judged it:

    - foamlib parsed the file as written (`fs.raw`), the reader did not read it again expanded
      (`fs.expanded`), and no value under `ddtSchemes` was noted as unread;
    - the case holds no `constant/regionProperties` (`REGION_PROPERTIES`): a multi-region case,
      whose regions read `system/<region>/fvSchemes`;
    - `courant.case.dictscan` follows all of the file, its comments and strings read as v2406's
      tokenizer reads them, and finds no `#` or `$` outside a comment or a string value: so no
      `#include`, followed or not (the review of hp6/final: under `--checks next` an include
      after `ddtSchemes { ddt(T) Euler; }` that writes `ddtSchemes { default steadyState; }` was
      expanded, foamlib kept the later block where v2406 merges the two, and a transient run's
      breach was spared), no `$name;` merge, no `#remove`, and no `#inputMode` in any
      dictionary of the file (the review of hp6/final: under `#inputMode protect` v2406 keeps the
      first of two `default`s, inputMode.C sets `entry::globalInputMode` for every entry read
      after it, and `_hiding` takes `#inputMode` for inert); nor a slash-scoped keyword
      (`ddtSchemes/ddt(T) Euler;` at the top level sets an entry of `ddtSchemes`, entryIO.C);
    - at the file's top level `ddtSchemes` is written once, as a dictionary, under that word,
      and no `select` is written: v2406's `schemesLookup::selectedDict` then reads the
      dictionary `select` names in place of the file's top level (schemesLookup.C,
      `readIfPresent("select", select, keyType::LITERAL)`, `subDict(select)`; the review of
      hp6/final);
    - inside `ddtSchemes` no keyword is written twice, quoted or not, and no entry is a
      dictionary. v2406 holds a keyword by its text, and which of two writings it keeps depends
      on the `#inputMode` in force (entryIO.C: merge and overwrite the later, protect the earlier,
      error neither); this replaces a test of foamlib's keys, which hold one copy and so could
      never see a second (the review of hp6/final);
    - foamlib's keywords, at the top level and in `ddtSchemes`, are the scan's, in order, as
      written: where the two readings of the file differ, foamlib's is not taken for the
      solver's.

    Linear in the file's length (`dictscan`): the review of hp6/final found the comment pattern
    `deck._read_text` strips with and `deck._repeated`'s entry pattern quadratic on hostile input
    (an unclosed comment, a keyword written many times inside a string), asked on every default
    `courant verify <case>`. Not asked under the default checks: the reader reads nothing for
    VER-13a there (`read_case_config`'s `steady_ddt`)."""
    from courant.case import dictscan

    if fs.raw is None or fs.expanded is not None:
        return False
    if any(where == "ddtSchemes" or where.startswith("ddtSchemes/") for where, _, _ in fs.hidden):
        return False
    if os.path.lexists(os.path.join(case_dir, REGION_PROPERTIES)):
        return False
    text = fs.as_written()
    found = dictscan.scan(text) if text is not None else None
    if found is None:
        return False
    top = found.at()
    if [e.key for e in top].count("FoamFile") > 1:
        return False
    top = [e for e in top if e.key != "FoamFile"]       # the header: foamlib lists no key for it
    words = [dictscan.unquoted(e.key) for e in top]
    if words.count("ddtSchemes") != 1 or "select" in words or \
            not any(e.key == "ddtSchemes" and e.dictionary for e in top):
        return False
    inner = found.at("ddtSchemes")
    names = [dictscan.unquoted(e.key) for e in inner]
    if len(names) != len(set(names)) or any(e.dictionary for e in inner):
        return False
    return ([k for k, _ in _entries_of(fs.raw)] == [e.key for e in top]
            and [k for k, _ in _entries_of(_get_quiet(fs.raw, "ddtSchemes"))]
            == [e.key for e in inner])


#: The dictionaries every OpenFOAM solver reads before its first step. A case directory without
#: one is not the case as it ran (or has lost the file since), so the setup rules that read it
#: had nothing to read: `CaseConfig.missing` names it, and the verdict says so (VER-7).
REQUIRED = ("system/controlDict", "system/fvSchemes", "system/fvSolution")


def _readable(f, path: str, rel: str, unreadable: dict[str, str], missing: list[str]) -> bool:
    """Whether this file parses at all, recording the reason if not: absent (`missing`, for a
    dictionary in `REQUIRED`) or unreadable, two facts.

    A file that cannot be reached (a directory on the way that cannot be searched), opened, or
    that is not a regular file (a FIFO would block the read forever; `reads.regular_file`) is
    unreadable, as one that does not parse is. Only a path that does not exist is absent. The
    reason for an `OSError` is its type and its message, never the path: the verdict names the
    file by its case-relative name, wherever the case lives.

    Every `_get` below swallows its exception, which is right for a missing key
    and wrong for a file that cannot be parsed: the two are indistinguishable
    afterwards, so an unreadable `controlDict` produced a `CaseConfig` with no
    solver, no timestep and no write control — *silently*, exactly as though the
    case had chosen to omit them.

    That is reachable on real cases. `foamDictionary -set` rewrites a dictionary
    with its `#include` directives expanded, and OpenFOAM's own
    `pisoFoam/RAS/cavity` expands into `fields ( U { mean on; ... } );` — a list
    containing an inline dictionary, which foamlib cannot parse. Any case an
    engineer has edited with OpenFOAM's own tool can arrive in this state, and
    every controlDict-derived checker and KB predicate then reads nothing and
    says nothing about why.
    """
    try:
        _parse(f, path)
    except (FileNotFoundError, NotADirectoryError):
        if rel in REQUIRED:
            missing.append(rel)
        return False
    except OSError as exc:
        unreadable[rel] = f"{type(exc).__name__}: {exc.strerror or exc}"
        return False
    except Exception as exc:            # noqa: BLE001
        unreadable[rel] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        return False
    return True


def _parse(f, path: str) -> None:
    """Parse a `FoamFile` once, noting `path` (the path the reader used, not foamlib's
    resolution of it) when the bytes were read (VER-14).

    foamlib reads lazily, and every `_get` swallows its exception, so this is the one place a
    read can be seen to succeed. A file that cannot be opened (`OSError`) is not noted; a file
    that was read and does not parse is noted, then the error propagates.
    """
    with reading(path):
        try:
            list(f.items())
        except KeyError:
            pass                        # parsed; simply has no entries we asked for


def _subdict_items(f, key):
    """Yield (name, value) for a subdict; empty if absent."""
    d = f.scope(*(key if isinstance(key, tuple) else (key,))) if isinstance(f, _Dictionary) \
        else _get(f, key)
    if d is None:
        return []
    try:
        return list(d.items())
    except Exception:  # noqa: BLE001
        return []


def _residual_control(fv) -> dict[str, dict[str, float]]:
    """`residualControl` per loop dictionary, as written: `{"SIMPLE": {key: tol}, ...}`.

    SIMPLE's entries are scalars (`p 1e-2;`) or a dictionary's `tolerance`, which
    `solutionControl::read` reads alike. PIMPLE's scalar entries are OpenFOAM.org's run
    criterion; ESI's per-field PIMPLE dictionaries (`p { tolerance 1e-4; relTol 0; }`) end the
    outer loop inside a time step and are not recorded. Which loop applies is the solver's
    (`solvers.residual_control_loop`), not this function's. Keys keep their quotes, so a
    regular expression stays one, and their order, because OpenFOAM applies the first entry
    that matches a field. A value that is not a number is left out.
    """
    out: dict[str, dict[str, float]] = {}
    for loop in ("SIMPLE", "PIMPLE"):
        entries: dict[str, float] = {}
        for key, value in _subdict_items(fv, (loop, "residualControl")):
            if hasattr(value, "get"):
                tol = _scalar(value.get("tolerance")) if loop == "SIMPLE" else None
            else:
                tol = _scalar(value)
            if tol is not None:
                entries[str(key)] = tol
        if entries:
            out[loop] = entries
    return out


@wordre.bounded                                     # the case's keyword patterns (SPEC-2)
def read_case_config(case_dir: str, *, buoyancy: bool = True, directives: bool = False,
                     steady_ddt: bool = False) -> CaseConfig:
    """The case's settings as the checks need them. A value its dictionary writes through a
    directive the reader did not follow is reported in `unread_values` (TRD VER-3b); with
    `directives` (`--checks next`) the dictionary is read as the solver reads it, its includes
    inside the case followed and its macros expanded, and only what still cannot be read is.
    `steady_ddt` (`--checks next`, `cfl.steady`, TRD VER-13a): `ddtSchemes`' other entries are
    read beside its `default` (`CaseConfig.ddt_terms`), and whether foamlib read it as the
    solver does (`CaseConfig.ddt_read_whole`); without it neither is read, so the default reads
    of `fvSchemes` what v0.5.0 read."""
    from foamlib import FoamFile

    data: dict = {}
    unreadable: dict[str, str] = {}
    missing: list[str] = []
    opened: list[_Dictionary] = []
    ddt_read_whole = False                       # TRD VER-13a, `_ddt_read_whole`

    def open_file(rel):
        p = os.path.join(case_dir, rel)
        if not may_read(p):             # MCP-3: asked before it is looked for
            return None
        f = FoamFile(p)                 # lazy: `_readable` reads it, and notes it (VER-14)
        failed: dict[str, str] = {}
        parsed = _readable(f, p, rel, failed, missing)
        d = _Dictionary(f if parsed else None, p, rel, case_dir, directives)
        if not parsed and not (directives and d.expand()):     # `${name}` stops foamlib
            unreadable.update(failed)
            return None
        opened.append(d)
        return d

    # --- controlDict ---
    cd = open_file("system/controlDict")
    if cd is not None:
        data["solver"] = _get(cd, "application")
        data["delta_t"] = _scalar(_get(cd, "deltaT"))
        data["end_time"] = _scalar(_get(cd, "endTime"))
        adj = _get(cd, "adjustTimeStep")
        data["adjust_timestep"] = bool(adj) if adj is not None else None
        data["max_co"] = _scalar(_get(cd, "maxCo"))
        wc = _get(cd, "writeControl")
        data["write_control"] = str(wc) if wc is not None else None
        data["write_interval"] = _scalar(_get(cd, "writeInterval"))

    # --- fvSchemes ---
    fs = open_file("system/fvSchemes")
    if fs is not None:
        ddt = _get(fs, "ddtSchemes", "default")
        data["ddt_scheme"] = _tokens(ddt)
        # VER-13a, under `--checks next` only (`steady_ddt`): the other entries, which a field's
        # lookup takes before `default` (`schemesLookup::lookupDetail::lookup`, src/OpenFOAM/
        # matrices/schemes/schemesLookupDetail.C), from the file as foamlib parsed it. Read
        # quietly, beside the default only: they can only keep a Courant number judged
        # (`checkers.steady_courant`), so nothing about them is noted and nothing else reads
        # them. A directive among them is kept as an entry, so it is not taken for steadyState.
        if steady_ddt and data["ddt_scheme"] is not None:
            data["ddt_terms"] = {str(k): _tokens(v) for k, v in
                                 _entries_of(_get_quiet(fs.raw, "ddtSchemes"))
                                 if k != "default"}
            # Whether that is every entry as the solver reads the file
            # (`CaseConfig.ddt_read_whole`), asked before anything else of it is read or noted.
            ddt_read_whole = _ddt_read_whole(fs, case_dir)
        div = {k: _tokens(v) for k, v in _subdict_items(fs, "divSchemes")}
        if div:
            data["div_schemes"] = {k: v for k, v in div.items() if v}

    # --- fvSolution ---
    fv = open_file("system/fvSolution")
    if fv is not None:
        relax: dict[str, float] = {}
        scopes: dict[str, str] = {}
        # What `solution::read` relaxes equations from (`CaseConfig.relaxation_equations`):
        # `equations`, or with neither sub-dictionary the flat entries. Read from the same values
        # as the flattened map, so nothing more is read and nothing more is noted.
        equations: dict[str, float | None] = {}
        rf = fv.scope("relaxationFactors")
        if rf is not None:
            # flatten nested fields/equations, or a flat mapping
            try:
                items = list(rf.items())
            except Exception:  # noqa: BLE001
                items = []
            nested = any(k in ("fields", "equations") and hasattr(v, "items") for k, v in items)
            for k, v in items:
                if k in ("fields", "equations"):
                    try:
                        for kk, _vv in fv.scope("relaxationFactors", k).items():
                            s = _scalar(_get(fv, "relaxationFactors", k, kk))
                            if k == "equations":
                                equations[str(kk)] = s
                            if s is not None:
                                relax[kk] = s
                                scopes[kk] = k
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    s = _scalar(_get(fv, "relaxationFactors", k))
                    if not nested:
                        equations[str(k)] = s
                    if s is not None:
                        relax[k] = s
                        scopes[k] = ""
        if relax:
            data["relaxation_factors"] = relax
            data["relaxation_scopes"] = scopes
            data["relaxation_equations"] = equations
        for loop in ("SIMPLE", "PIMPLE"):
            nno = _scalar(_get(fv, loop, "nNonOrthogonalCorrectors"))
            if nno is not None:
                data["n_non_orthogonal_correctors"] = int(nno)
            ncorr = _scalar(_get(fv, loop, "nCorrectors"))
            if ncorr is not None:
                data["n_correctors"] = int(ncorr)
        nouter = _scalar(_get(fv, "PIMPLE", "nOuterCorrectors"))
        if nouter is not None:
            data["n_outer_correctors"] = int(nouter)
        control = _residual_control(fv)
        if control:
            data["residual_control"] = control
        solvers: dict[str, LinearSolverSpec] = {}
        for field, spec in _subdict_items(fv, "solvers"):
            if not hasattr(spec, "get"):
                continue
            try:
                solvers[field] = LinearSolverSpec(
                    field=field, solver=_tokens(_get(fv, "solvers", field, "solver")),
                    tolerance=_scalar(_get(fv, "solvers", field, "tolerance")),
                    rel_tol=_scalar(_get(fv, "solvers", field, "relTol")),
                    log=_solver_log(_get(fv, "solvers", field, "log")))
            except Exception:  # noqa: BLE001
                pass
        if solvers:
            data["linear_solvers"] = solvers

    # --- turbulence ---
    for rel in ("constant/momentumTransport", "constant/turbulenceProperties"):
        tf = open_file(rel)
        if tf is not None:
            sim = _get(tf, "simulationType")
            if sim is not None:
                data["turbulence_type"] = str(sim)
            # `RASModel` / `LESModel`, which is what OpenFOAM actually calls
            # them. The key here was `model`, so `turbulence_model` has been
            # `None` on every RAS case in the corpus — `pisoFoam/RAS/cavity` and
            # `simpleFoam/pitzDaily` included — while `RASModel kEpsilon;` sat in
            # the dictionary being read. Found independently by two solver
            # families at once, and the asymmetry is telling: the fault injector
            # already writes `RAS/RASModel` correctly, so the writer and the
            # reader disagreed about the same key.
            model = (_get(tf, "RAS", "RASModel") or _get(tf, "LES", "LESModel")
                     or _get(tf, "RAS", "model") or _get(tf, "LES", "model")
                     or _get(tf, "model"))
            if model is not None:
                data["turbulence_model"] = str(model)
            # The model's own switch (`getOrDefault<Switch>("turbulence", true)`, RASModel.C and
            # LESModel.C): off, the model solves nothing (TRD VER-17a).
            block = {"RAS": "RAS", "LES": "LES"}.get(str(sim).strip().upper()) if sim else None
            switch = _switch(_get(tf, block, "turbulence")) if block else None
            if switch is not None:
                data["turbulence_switch"] = switch
            break

    # --- transport ---
    tp = open_file("constant/transportProperties")
    if tp is not None:
        transport = {}
        for name in ("nu", "rho", "Pr", "Prt", "beta", "TRef"):
            val = _scalar(_get(tp, name))
            if val is not None:
                transport[name] = val
        if transport:
            data["transport"] = transport
        # Each phase's own `nu` (interFoam, `phases (water air);`): the solver reads it from the
        # phase's subdictionary, which the loop above does not look in (TRD PRE-6). Only a
        # Newtonian phase's (`transportModel Newtonian`, or none written): another model reads
        # its own coefficients, and a `nu` beside them is not its viscosity.
        phases = _get(tp, "phases")
        listed = [str(p) for p in phases] if isinstance(phases, (list, tuple)) else []
        if listed:                      # local; a finding names a phase by its place in it
            data["phases"] = listed
        per_phase: dict[str, float] = {}
        for phase in listed:
            model = _get(tp, phase, "transportModel")
            if model is not None and str(model) != "Newtonian":
                continue
            val = _scalar(_get(tp, phase, "nu"))
            if val is not None:
                per_phase[phase] = val
        if per_phase:
            data["phase_nu"] = per_phase

    # --- boundary conditions (0/ field files) ---
    boundary: dict[str, dict[str, str]] = {}
    zero = os.path.join(case_dir, "0")
    if os.path.isdir(zero) and may_read(zero):      # MCP-3: listed, so confined like a read
        for fname in os.listdir(zero):
            fpath = os.path.join(zero, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                ff = FoamFile(fpath)
                _parse(ff, fpath)       # VER-14: noted once read, parsed or not
                for patch, spec in _subdict_items(ff, "boundaryField"):
                    t = _tokens(spec.get("type")) if hasattr(spec, "get") else None
                    if t:
                        boundary.setdefault(patch, {})[fname] = t
            except Exception:  # noqa: BLE001
                continue
    if boundary:
        data["boundary"] = boundary

    # --- thermophysicalProperties: the compressible and buoyant families ---
    #
    # `transport` was read only from `constant/transportProperties`, which
    # compressible and buoyant cases **do not have**. Their viscosity is `mu`,
    # not `nu`, and it lives at `mixture/transport/mu` in a differently shaped
    # file, with `Cp` under `mixture/thermodynamics`. Measured on real v2406
    # tutorials: `CaseConfig.from_foam(...).transport == {}` for both
    # `rhoPimpleFoam/RAS/cavity` and `buoyantPimpleFoam/hotRoom`.
    #
    # The consequence is a permanent blind spot rather than a false positive:
    # `nu_out_of_range` and the `physics.nu` finding cannot fire on any solver
    # in either family, so a units error in `mu` — the single most common
    # real mistake there is, and the exact fault `PHYS.TRANSPORT_UNITS` names —
    # was undetectable. Reported independently by both families.
    #
    # Reading `mu` did not close it on its own: the check still read `nu` only,
    # and the hard-proofing audit (21 Oct 2026) found a hotRoom with `mu`
    # 1.8e-10 accepted at grade A. `physics.nu` reads `mu` through the
    # density of the case's equation of state since TRD PRE-6, behind
    # `--checks next` until 0.6.0 (`courant.case.viscosity`); `rho` is read for
    # `rhoConst` for that.
    thermo = open_file("constant/thermophysicalProperties")
    if thermo is not None:
        properties: dict[str, float] = dict(data.get("transport") or {})
        for group, keys in (("transport", ("mu", "Pr")),
                            ("thermodynamics", ("Cp", "Hf")),
                            ("specie", ("molWeight",)),
                            ("equationOfState", ("rho",))):
            for key in keys:
                value = _scalar(_get(thermo, "mixture", group, key))
                if value is not None:
                    properties.setdefault(key, value)
        if properties:
            data["transport"] = properties
        eos = _get(thermo, "thermoType", "equationOfState")
        if eos is not None:
            data["equation_of_state"] = str(eos)
        p_ref = _scalar(_get(thermo, "pRef"))
        if p_ref is not None:
            data["pressure"] = p_ref

    # --- buoyancy (TRD VER-17): g, the fixed temperatures and the pressure the solver read ---
    # Without `buoyancy` (the gate under the default --checks, GATE-13) `constant/g` is not
    # read, so neither is the mesh: no gravity, no wall faces.
    gf = open_file("constant/g") if buoyancy else None
    if gf is not None:
        g = _vector(_get(gf, "value"))
        if g is not None:
            data["gravity"] = g
    # `0/` files as the boundary loop above reads them: one that cannot be read is absent.
    tf = _field_file(case_dir, "0/T")
    if tf is not None:
        data.update(_held_walls(case_dir, _fixed_values(tf), data.get("gravity")))
    pf = _field_file(case_dir, "0/p")
    if pf is not None:
        p = _uniform(_get(pf, "internalField"))
        if p is not None:
            data["pressure"] = p                     # the field the solver read wins over pRef

    # --- mesh motion (TRD VER-5a): whether constant/dynamicMeshDict moves the mesh. Read in both
    # modes for a line of reporting, never a verdict; a file that cannot be read is said so in
    # that line, not taken as a case dictionary that failed (`unreadable`).
    from courant.case.motion import read_mesh_motion

    motion = read_mesh_motion(case_dir)
    if motion is not None:
        data["mesh_motion"] = motion.by

    unread = {d.rel: why for d in opened if (why := d.report()) is not None}
    if unread:                                       # TRD VER-3b: what a directive hid
        data["unread_values"] = unread
        data["unread_keys"] = {d.rel: d.keys_report(data.get("phases") or ())
                               for d in opened if d.rel in unread}
    if unreadable:
        data["unreadable"] = unreadable
    if missing:
        data["missing"] = missing
    config = CaseConfig.from_dict({k: v for k, v in data.items() if v is not None})
    # Set here and nowhere else: `from_dict` never takes it from a dictionary (TRD VER-13a).
    return config.model_copy(update={"ddt_read_whole": True}) if ddt_read_whole else config


def _field_file(case_dir: str, rel: str):
    """A `0/` field file, parsed and noted (VER-14), or None when it is missing or unreadable."""
    from foamlib import FoamFile

    path = os.path.join(case_dir, rel)
    if not os.path.isfile(path):
        return None
    try:
        f = FoamFile(path)
        _parse(f, path)
    except Exception:  # noqa: BLE001
        return None
    return f


def _numbers(v) -> list[float] | None:
    """Every number in a field value as foamlib returns it (a scalar, a list or an array)."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return [float(v)]
    try:
        values = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return values or None


def _vector(v) -> tuple[float, float, float] | None:
    values = _numbers(getattr(v, "value", v))
    return tuple(values) if values is not None and len(values) == 3 else None


def _uniform(v) -> float | None:
    values = _numbers(v)
    return values[0] if values is not None and len(set(values)) == 1 else None


def _fixed_values(tf) -> dict[str, list[float]]:
    """{patch: its values} for every `fixedValue` patch of a scalar field file: one value when
    uniform, one per face when a list (as `setFields` writes it)."""
    out: dict[str, list[float]] = {}
    for patch, spec in _subdict_items(tf, "boundaryField"):
        if not hasattr(spec, "get") or _tokens(spec.get("type")) != "fixedValue":
            continue
        values = _numbers(spec.get("value"))
        if values is not None:
            out[str(patch)] = values
    return out


def _held_walls(case_dir: str, fixed: dict[str, list[float]], gravity) -> dict:
    """`fixed_temperature_range`, and the boxes of the hottest and coldest held wall faces (TRD
    VER-17). The mesh files are read only for a case with a gravity vector and more than one held
    temperature, through `solver_polymesh` (the mesh the solver read, within the cost bounds).
    With the mesh, only `wall` patches count, and every value must match the patch's faces."""
    import math

    values = [v for vs in fixed.values() for v in vs]
    if not values:
        return {}
    out: dict = {"fixed_temperature_range": (min(values), max(values))}
    if not gravity or not any(gravity) or len(set(values)) < 2:
        return out
    from courant.case.mesh import solver_polymesh

    mesh = solver_polymesh(case_dir)
    if mesh is None:
        return out
    walls = {p.name: p for p in mesh.boundary if p.type == "wall"}
    held = {name: vs for name, vs in fixed.items() if name in walls}
    faces: list[tuple[float, tuple[float, float, float]]] = []
    for name, vs in held.items():
        p = walls[name]
        if len(vs) not in (1, p.n):
            return out                       # values that do not match the faces
        centres = mesh.face_centres[p.start:p.start + p.n]
        faces += [(vs[0] if len(vs) == 1 else vs[i], tuple(float(x) for x in c))
                  for i, c in enumerate(centres)]
    temps = [t for t, _ in faces]
    if not temps:
        return {}                                   # no wall is held at a temperature
    if not all(math.isfinite(t) for t in temps):
        return out
    lo, hi = min(temps), max(temps)

    def box(t):
        pts = [c for v, c in faces if v == t]
        return (tuple(min(c[k] for c in pts) for k in range(3)),
                tuple(max(c[k] for c in pts) for k in range(3)))

    return {"fixed_temperature_range": (lo, hi), "hot_faces_box": box(hi),
            "cold_faces_box": box(lo)}
