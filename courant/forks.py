"""Which OpenFOAM fork and version produced a run, and the set the error rates were measured on
(TRD ADP-1, VER-5.10).

ARC-14 keeps every fork and version out of the decision code: they live in the adapters, and
this module is the one that reads the version. The decision code asks it one question,
`SolverVersion.measured`, and names neither a fork nor a version.

**Where the version is read**, in order, and the first that names one wins:

1. The solver log's banner, before its first `Exec :` line. The committed ESI v2406 runs print
   `| Version:  2406 |`, `| Website:  www.openfoam.com |` and
   `Build  : _be01ca78-20240625 OPENFOAM=2406 version=2406`; older ESI releases print
   `Version:  v2012`. The Foundation fork prints `| Website:  https://openfoam.org`,
   `| Version:  12` and `Build  : 12-<commit>` (`dev` for its development line). The `Build`
   line gives the version before the `Version` line does. The website names the project: where
   it and the version's numbering disagree, the website is believed (foam-extend, OpenCFD's own
   2.x releases on www.OpenFOAM.com) or, failing a project that explains them (another fork's
   website, or one of neither fork), the fork is left unread. A fork's website alone names it
   and no version.
2. When the log has no banner (a window cut from a longer log, a stripped one), the header of a
   file the run wrote: the files of the latest time directory after 0, by name, dotfiles left
   out, until one names a fork. Under `--checks next` (`fields.log_time`, VER-7b) the latest
   no later than the log's last time step, as the field checks read it: a later one belongs to
   another run (the final review of 0.6.0, 21 Oct 2026).

What the run did not write is never taken for the version it was solved on: `system/controlDict`
is an input, which may have been copied from another version's tutorial. When nothing the run
wrote names a fork, what its header names is kept apart (`SolverVersion.given`) and said in the
not-checked line only; the fork and version stay unread, as with no banner. Every file read
joins the manifest (VER-14). Nothing is guessed from a path, a host name or a solver's name: a
`Case :` line can lie, and a solver exists in both forks.

**The measured set** is `MEASURED`, the one definition: the gate's error rates (the
certification matrix, `courant/data/certification.json`) were measured on the pinned ESI v2406
image only. A version
outside it is reported by default (a not-checked line); VER-5.10, behind `--checks next` in
0.6.0 (GATE-13), makes a provisional accept on it `insufficient`. A version that could not be
read is said to be unread, and judged as before.

The not-checked lines are templates in `courant.verify.sentences` (GATE-4: each carries the
`fork` cause there); this module fills them in.
"""

from __future__ import annotations

import gzip
import os
import re
import zlib
from dataclasses import dataclass

#: ADP-1, VER-5.10: the (fork, version) pairs the gate's error rates were measured on: the
#: certification matrix ran on the pinned ESI v2406 image (`packaging/images.lock`,
#: `courant.config.OPENFOAM_IMAGE`). Every sentence that names the measured set reads it here.
MEASURED: frozenset[tuple[str, str]] = frozenset({("esi", "v2406")})

#: The forks and projects, as `run.fork` writes them, to the name a reader sees. `opencfd` is
#: OpenCFD's own releases (2.x and earlier, www.OpenFOAM.com), before the Foundation's.
FORKS: dict[str, str] = {"esi": "ESI", "foundation": "Foundation", "opencfd": "OpenCFD",
                         "foam-extend": "foam-extend"}

#: A log's banner is at its top; this much of the log is searched for it.
LOG_HEAD = 64 * 1024
#: A dictionary's header comment is its first lines; this much of a file is read for it.
FILE_HEAD = 2048

LOG_BANNER = "the solver log's banner"
CONTROL_DICT = os.path.join("system", "controlDict")

# `[ \t]*`, never `\s*`, around a key's colon: an empty `Build  :` must not take the next line
# for its value.
_EXEC = re.compile(r"^Exec[ \t]*:", re.MULTILINE)
_BUILD = re.compile(r"^Build[ \t]*:[ \t]*(?P<value>[^\n]*)$", re.MULTILINE)
_VERSION = re.compile(r"\|[ \t]*Version:[ \t]*(?P<value>[^\s|]+)")
_WEBSITE = re.compile(r"\|[ \t]*Web(?:site)?:[ \t]*(?P<value>[^\s|]+)", re.IGNORECASE)

_ESI_BUILD = re.compile(r"\b(?:OPENFOAM|version)=(?P<n>\d{4})\b")
_TOKEN = re.compile(r"\bOpenFOAM-(?P<value>[\w.+]+)", re.IGNORECASE)
_FOUNDATION_BUILD = re.compile(r"^(?P<value>\d{1,2}(?:\.(?:\d+|x))*|dev)-[0-9a-f]{6,}\b")
_ESI_VERSION = re.compile(r"^v?(?P<n>\d{4})\+?$")             # 2406, v2406, v1712+
_ESI_PLUS = re.compile(r"^v\d+(?:\.\d+)+\+?$")                # v3.0+ (OpenFOAM+)
_FOUNDATION_VERSION = re.compile(r"^(?:\d{1,2}(?:\.(?:\d+|x))*|dev)$")   # 12, 2.3.0, 5.x, dev


@dataclass(frozen=True)
class SolverVersion:
    """What ADP-1 read: the fork (`esi`, `foundation`) and the version (`v2406`, `12`), each
    `None` when not read, where they were read (the log's banner, or the header of a file the
    run wrote), and whether the case's file headers were searched (a case directory whose log
    has no banner). `given` is what `system/controlDict`'s header names when nothing the run
    wrote names a fork: an input, said in the not-checked line and never in `fork`, `version`
    or `measured`."""

    fork: str | None = None
    version: str | None = None
    source: str | None = None
    headers_searched: bool = False
    given: SolverVersion | None = None

    @property
    def measured(self) -> bool | None:
        """VER-5.10's one question. True inside the measured set, False outside it, None when
        that cannot be told: nothing read, a fork the set holds with no version, a version it
        holds with no fork, or only an input's header (`given`), which never counts."""
        if (self.fork, self.version) in MEASURED:
            return True
        if self.fork is not None and self.fork not in {fork for fork, _ in MEASURED}:
            return False
        if self.version is not None and self.version not in {v for _, v in MEASURED}:
            return False
        return None

    @property
    def label(self) -> str | None:
        return label(self.fork, self.version)


def label(fork: str | None, version: str | None) -> str | None:
    """`OpenFOAM v2512 (ESI)`, `OpenFOAM 12 (Foundation)`, `OpenFOAM v2406 (fork not told)`;
    None when nothing was read."""
    if fork is None:
        return f"OpenFOAM {version} (fork not told)" if version else None
    name = FORKS.get(fork, fork)
    return f"OpenFOAM {version} ({name})" if version else f"OpenFOAM ({name}, version not read)"


def measured_label() -> str:
    """The measured set, as a reader sees it: `OpenFOAM v2406 (ESI)`."""
    return " and ".join(label(f, v) or "" for f, v in sorted(MEASURED))


# ---------------------------------------------------------------- reading a banner
def _classify(token: str) -> tuple[str, str] | None:
    token = token.strip()
    m = _TOKEN.search(token)
    if m:
        token = m.group("value")
    m = _ESI_VERSION.match(token)
    if m:
        return "esi", f"v{m.group('n')}"
    if _ESI_PLUS.match(token):
        return "esi", token
    if _FOUNDATION_VERSION.match(token):
        return "foundation", token
    return None


def _from_build(value: str) -> tuple[str, str] | None:
    m = _ESI_BUILD.search(value)
    if m:
        return "esi", f"v{m.group('n')}"
    m = _TOKEN.search(value)
    if m:
        return _classify(m.group("value"))
    m = _FOUNDATION_BUILD.match(value.strip())
    if m:
        return "foundation", m.group("value")
    m = re.match(r"v(\d{4})\b", value.strip())
    return ("esi", f"v{m.group(1)}") if m else None


#: What `_website` gives for the website of a project that is neither fork nor foam-extend.
_OTHER_PROJECT = "another project"


def _website(value: str) -> str:
    """The project a banner's website names: a fork, foam-extend (and its extend-project
    forerunner), or `_OTHER_PROJECT` for any other website, whose numbering names no fork."""
    value = value.lower()
    if "foam-extend" in value or "extend-project" in value:
        return "foam-extend"
    if "openfoam.com" in value:
        return "esi"
    if "openfoam.org" in value:
        return "foundation"
    return _OTHER_PROJECT


def from_banner(text: str) -> tuple[str | None, str | None]:
    """(fork, version) from an OpenFOAM banner: a log's head, or a dictionary's header comment.
    (None, None) when the text names neither."""
    m = _BUILD.search(text)
    found = _from_build(m.group("value")) if m else None
    if found is None:
        m = _VERSION.search(text)
        found = _classify(m.group("value")) if m else None
    m = _WEBSITE.search(text)
    site = _website(m.group("value")) if m else None
    if found is None:
        return (None if site == _OTHER_PROJECT else site), None
    fork, version = found
    if site is None or site == fork:
        return found
    # The website and the version's numbering name different projects: the numbering alone
    # never names the fork then.
    if site == "foam-extend":
        return site, version                 # foam-extend numbers its releases as 4.1, 5.0
    if site == "esi" and fork == "foundation":
        return "opencfd", version            # 2.0.1 on www.OpenFOAM.com: OpenCFD's own release
    return None, version                     # another fork's website, or neither fork's


def _log_banner(log_text: str) -> str:
    head = log_text[:LOG_HEAD]
    m = _EXEC.search(head)
    return head[:m.start()] if m else head


# ---------------------------------------------------------------- the case's own headers
def _written_files(case_dir: str, until: float | None = None) -> list[str]:
    """The files of the latest time directory after 0, by name: what the run wrote. Dotfiles
    are left out (a `.DS_Store` macOS Finder writes sorts before every field). With `until`, the
    log's last time step, a directory later than it is not this run's (`_fluid_reader.after`,
    the field reader's one rule)."""
    from courant.verification._fluid_reader import _numeric_time_dirs, after

    times = [t for t in _numeric_time_dirs(case_dir)
             if float(t) > 0.0 and (until is None or not after(t, until))]
    if not times:
        return []
    where = os.path.join(case_dir, times[-1])
    try:
        names = sorted(n for n in os.listdir(where)
                       if not n.startswith(".") and os.path.isfile(os.path.join(where, n)))
    except OSError:
        return []
    return [os.path.join(times[-1], name) for name in names]


def _head_of(path: str) -> str:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        return fh.read(FILE_HEAD).decode("utf-8", "replace")


def _header(case_dir: str, rel: str) -> SolverVersion | None:
    """What the header of `case_dir/rel` names, or None: no such file, unreadable, or naming no
    fork and no version. The read is noted once its bytes were read (VER-14)."""
    from courant.case.reads import reading

    path = os.path.join(case_dir, rel)
    if not os.path.isfile(path):
        return None
    try:
        with reading(path):
            text = _head_of(path)
    except (OSError, EOFError, zlib.error):         # unreadable, or does not inflate: next
        return None
    fork, version = from_banner(text)
    if fork is None and version is None:
        return None
    return SolverVersion(fork, version, f"the header of {rel.replace(os.sep, '/')}",
                         headers_searched=True)


def _from_headers(case_dir: str, until: float | None = None) -> SolverVersion:
    for rel in _written_files(case_dir, until):
        found = _header(case_dir, rel)
        if found is not None:
            return found
    return SolverVersion(headers_searched=True, given=_header(case_dir, CONTROL_DICT))


def read(log_text: str, case_dir: str | None = None, *,
         until: float | None = None) -> SolverVersion:
    """ADP-1: the fork and version of the run behind `log_text`, from its banner; failing
    that, from the headers of the files the run wrote in `case_dir` (and, kept apart as
    `given`, `system/controlDict`'s), in no time directory later than `until` when it is given
    (the log's last time step, under `--checks next`). Never raises: what cannot be read is
    None."""
    fork, version = from_banner(_log_banner(log_text or ""))
    if fork is not None or version is not None:
        return SolverVersion(fork, version, LOG_BANNER)
    if not case_dir:
        return SolverVersion()
    try:
        return _from_headers(case_dir, until)
    except Exception:                                   # noqa: BLE001  never a traceback (ADP-1)
        return SolverVersion(headers_searched=True)


# ---------------------------------------------------------------- what the verdict says
def not_checked(seen: SolverVersion) -> str | None:
    """The not-checked line ADP-1 adds to every verdict, or None for a version the error rates
    cover, read from the log's banner. The templates, each of cause `fork`, are
    `courant.verify.sentences`'."""
    from courant.verify import sentences

    measured = measured_label()
    if seen.measured is False:
        where = seen.source if seen.source == LOG_BANNER else \
            f"{seen.source} (a file the run wrote; the solver log has no banner)"
        return sentences.VERSION_UNMEASURED.format(measured=measured, where=where,
                                                   label=seen.label)
    if seen.measured is True:
        if seen.source == LOG_BANNER:
            return None
        return sentences.VERSION_FROM_RUN_FILE.format(label=seen.label, source=seen.source)
    if seen.given is not None:
        return sentences.VERSION_GIVEN_ONLY.format(source=seen.given.source,
                                                   label=seen.given.label, measured=measured)
    if seen.fork is not None:
        what = "version was"
        why = f"{seen.source} names the {FORKS.get(seen.fork, seen.fork)} fork and no version"
    elif seen.version is not None:
        what = "fork was"
        why = (f"{seen.source} names version {seen.version} under the website of another fork "
               "or project")
    else:
        what = "fork and version were"
        why = ("the solver log has no banner, and no header of the case's files names one"
               if seen.headers_searched else "the solver log has no banner")
    return sentences.VERSION_UNREAD.format(what=what, why=why, measured=measured)


__all__ = ["CONTROL_DICT", "FORKS", "LOG_BANNER", "MEASURED", "SolverVersion", "from_banner",
           "label", "measured_label", "not_checked", "read"]
