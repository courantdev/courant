"""Which case files the readers read (TRD VER-14).

A verdict's manifest is the set of files that decided it, and only a reader knows what it read.
So each reader notes its file at the line that reads it, and the verifier collects the notes
inside a `recording`. Nothing here guesses which files matter.

Outside a recording `note` does nothing, so a reader called from anywhere else (the watcher, the
fix loop) is unchanged. A note costs one `os.path.isfile`: no hashing and no other I/O. The
manifest hashes afterwards (`courant.verify.schema.manifest`).

A file is noted only once its read has succeeded (T-MANIFEST-2). A read that fails with an
`OSError` (the file cannot be opened, or vanished) read nothing, so the file is left out and the
reader's own fallback stands: an unreadable `0/U` is what it was before the manifest existed, a
missing field, and never a traceback. A read that got the bytes and could not parse them did read
the file, so it is noted. `reading(path)` is that rule, for readers whose failure is an exception.

Where a reader may read is decided here too (TRD MCP-3). Inside `confined(confinement)` (the MCP
tools), every reader asks `refuse_outside(path)` before it opens a case file, `reading` asks it for
them, and a path that resolves outside every allowed root is refused before it is opened: a link
inside a root to a file outside them is refused like that file named directly. On the way out
every link in the case is asked too, and a link out of the roots refuses the call by its own path
alone, whether or not anything exists where it leads (`links_out`). A refusal names what is
inside the roots, never where a link leads. Outside a confinement the check does nothing, so
`courant verify` reads what its user can read, as before.

A reader opens regular files only (`regular_file`). A FIFO, a socket or a device in place of a case
file is not opened: opening a FIFO with no writer blocks forever, and one in `system/` hung `courant
verify`, `courant preflight` and the MCP tool with no output (hard-proofing audit, 21 Oct 2026). It
raises `NotRegularFile`, an `OSError`, so each reader's fallback takes it as a file it could not open.
"""

from __future__ import annotations

import os
import stat
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


class OutsideRoot(PermissionError):
    """A path that resolves outside every allowed root (TRD MCP-3). An `OSError`, so a reader's
    own fallback takes it as a file that could not be read; `confined` keeps it, and refuses the
    whole call however the reader fell back."""


def _resolved(path: Path) -> Path:
    """`path` with every link followed; a link loop, which Python 3.11 raises on, resolved as far
    as it goes. One resolution for a root and for a path in it."""
    try:
        return path.resolve()
    except RuntimeError:
        return Path(os.path.realpath(path))


class Confinement:
    """Only paths inside one of these roots may be read (TRD MCP-3). No root: the working
    directory."""

    def __init__(self, roots: list[str | os.PathLike[str]]):
        self.roots = [_resolved(Path(r)) for r in (roots or [Path.cwd()])]

    def inside(self, p: Path) -> bool:
        return any(p == r or p.is_relative_to(r) for r in self.roots)

    def check(self, path: str | os.PathLike[str]) -> Path:
        """`path` resolved, every link in it followed, or `OutsideRoot` when that is outside
        every root."""
        given = Path(path).expanduser()
        p = _resolved(given)
        if self.inside(p):
            return p
        raise self.refusal(given)

    def refusal(self, path: str | os.PathLike[str]) -> OutsideRoot:
        """The refusal of `path`, said in what is inside the roots: the path as it was named,
        the link in a root it leaves them through, and the roots. Never where a link leads,
        which is outside them: that would read back any link's target on the machine (MCP-3)."""
        named = Path(os.path.abspath(path))
        roots = ", ".join(str(r) for r in self.roots)
        link = self._exit(path)
        if link is None:
            return OutsideRoot(f"{named} is outside the allowed roots: {roots}; it was not read, "
                               "and a --root that holds it would allow it")
        said = (f"{named} is a link" if link == named
                else f"{named} is reached through {link}, a link")
        return OutsideRoot(f"{said} that leads outside the allowed roots: {roots}; it was not "
                           "read, and a --root that holds where it leads would allow it")

    def _exit(self, path: str | os.PathLike[str]) -> Path | None:
        """The link in a root through which `path` last leaves the roots, by its path there, or
        None when no link takes it out (it was never inside a root, or `..` took it out). Walked
        a component at a time, as the system resolves it; a link outside the roots is followed
        to know where the walk goes, and never named."""
        parts = Path(os.path.join(os.getcwd(), os.fspath(path))).parts
        here, exit_ = Path(parts[0]), None
        for part in parts[1:]:
            was_inside, step, link = self.inside(here), here / part, None
            if part == "..":
                step = here.parent
            elif os.path.islink(step):
                step, link = Path(os.path.realpath(step)), step
            if was_inside != self.inside(step):
                exit_ = link if was_inside else None
            here = step
        return exit_


_confined: ContextVar[tuple[Confinement, dict[str, OutsideRoot]] | None] = ContextVar(
    "courant_case_confined", default=None)


@contextmanager
def confined(confinement: Confinement, *trees: str | os.PathLike[str]) -> Iterator[None]:
    """Every case file read in the block is inside `confinement`'s roots (MCP-3).

    A reader that meets a path outside them is refused before it opens it (`refuse_outside`).
    The reader may fall back as for any unreadable file, so on the way out the block itself is
    refused with the first refusal in sorted order (and how many more there were), whatever it
    returned or raised: its answer was judged without the file, and is not given.

    A reader asks only for what it opens, and most look first whether the file is there, so
    whether a reader was refused at all says whether the file it looked for exists. So on the way
    out every link under `trees` (the case a tool was given) is asked too (`links_out`), and when
    one is refused the block is refused with the links alone, sorted: whatever the readers met
    under them is dropped, count and all, so the answer is the same whether or not anything
    exists where a link leads, and whether or not a reader opened it. The readers' own refusals
    are said only when no link was found (one in a directory that cannot be listed).
    """
    refused: dict[str, OutsideRoot] = {}
    links: dict[str, OutsideRoot] = {}
    token = _confined.set((confinement, refused))
    try:
        try:
            yield
        finally:
            links = links_out(confinement, trees)
    except Exception as exc:
        if not (links or refused):
            raise
        refusal = _first_refusal(links or refused)
        if refusal is exc:                              # a reader let the refusal through
            raise
        if links:                                       # nothing of what a reader met is said
            raise refusal from None
        raise refusal from exc
    finally:
        _confined.reset(token)
    if links or refused:
        raise _first_refusal(links or refused)


def _within(p: str, d: str) -> bool:
    return p == d or p.startswith(os.path.join(d, ""))


def links_out(confinement: Confinement, trees) -> dict[str, OutsideRoot]:
    """Every link under `trees` (the case a tool was given) that is refused, by the path it is
    found at (MCP-3): one that leads outside `confinement`'s roots, and one to a directory that
    holds the case, through which a reader could reach every file under it (the whole root, and
    every other case in it). Nothing is opened: directories are listed, never files, each once
    (a loop ends there), in sorted order, so the result is the same on every filesystem. A link
    into a root, to a directory outside the case, is followed (a shared mesh); one to a
    directory inside the case is not, the case being listed anyway. A directory that cannot be
    listed is passed over: the readers that open a file in one by name ask before they look
    (`refuse_outside`)."""
    cases = [os.path.realpath(t) for t in trees if os.path.isdir(t)]
    found: dict[str, OutsideRoot] = {}
    todo, listed = deque(sorted(cases)), set()
    while todo:
        d = todo.popleft()
        if (real := os.path.realpath(d)) in listed:
            continue
        listed.add(real)
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            try:
                if not e.is_symlink():
                    if e.is_dir(follow_symlinks=False):
                        todo.append(e.path)
                    continue
                confinement.check(e.path)
                to = os.path.realpath(e.path)
                if any(_within(c, to) and c != to for c in cases):
                    found[os.path.abspath(e.path)] = OutsideRoot(
                        f"{os.path.abspath(e.path)} is a link to a directory that holds the case, "
                        "through which a reader could reach every file under it; it was not "
                        "followed: link to the directory the case needs instead")
                elif e.is_dir() and not any(_within(to, c) for c in cases):
                    todo.append(e.path)
            except OutsideRoot as exc:
                found[os.path.abspath(e.path)] = exc
            except OSError:             # a loop, or an entry gone since it was listed
                continue
    return dict(sorted(found.items()))


def _first_refusal(refused: dict[str, OutsideRoot]) -> OutsideRoot:
    """One refusal for the block: the first path in sorted order, and how many more."""
    first = refused[min(refused)]
    more = len(refused) - 1
    if not more:
        return first
    return OutsideRoot(f"{first} ({more} more {'path' if more == 1 else 'paths'} in the case "
                       f"{'is' if more == 1 else 'are'} refused too)")


def refuse_outside(path: str | os.PathLike[str]) -> None:
    """Before a reader opens `path` or lists it: `OutsideRoot` when a confinement is active and
    `path` resolves outside every root (MCP-3). Outside a confinement nothing happens."""
    active = _confined.get()
    if active is None:
        return
    confinement, refused = active
    try:
        confinement.check(path)
    except OutsideRoot as exc:
        refused.setdefault(os.path.abspath(os.fspath(path)), exc)
        raise


def may_read(path: str | os.PathLike[str]) -> bool:
    """`refuse_outside` for a reader whose fallback is a branch: False when `path` is refused,
    the refusal kept for `confined` all the same."""
    try:
        refuse_outside(path)
    except OutsideRoot:
        return False
    return True


class NotRegularFile(OSError):
    """A path that exists and is not a regular file (or a link to one). Never opened."""


def reason(exc: BaseException) -> str:
    """Why a read failed, in words that are the same wherever the case lives: an `OSError` by
    its type and `strerror` (its message names the absolute path, `[Errno 13] Permission denied:
    '/Users/x/case/0/U'`, so one case gave different verdict bytes on two machines; the final
    review of 0.6.0, 21 Oct 2026), any other exception by its type and message. The reader names
    the file by its case-relative name beside it."""
    if isinstance(exc, OSError):
        named = exc.filename is not None or exc.filename2 is not None
        detail = exc.strerror or ("" if named else str(exc))
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return f"{type(exc).__name__}: {exc}"


#: What a path that is not a regular file is, in the reason a reader records.
_KINDS = ((stat.S_ISFIFO, "a FIFO"), (stat.S_ISSOCK, "a socket"), (stat.S_ISCHR, "a device"),
          (stat.S_ISBLK, "a device"), (stat.S_ISDIR, "a directory"))


def regular_file(path: str | os.PathLike[str]) -> None:
    """Return when `path` is a regular file a reader may open; raise `OSError` otherwise, before
    anything opens it: `FileNotFoundError` when it is absent, `PermissionError` when a directory on
    the way cannot be searched, `NotRegularFile` when it is a FIFO, a socket, a device or a
    directory. One `os.stat`, which follows links and opens nothing."""
    mode = os.stat(path).st_mode
    if not stat.S_ISREG(mode):
        kind = next((name for test, name in _KINDS if test(mode)), "a special file")
        raise NotRegularFile(f"{kind}, not a regular file")


class Reads:
    """The regular files read under one root, as root-relative POSIX paths.

    Paths are compared lexically (`os.path.abspath`), never resolved, so a symlinked `0/` still
    records as `0/U`: the path the reader used, which is the path a re-check will use.
    """

    def __init__(self, root: str | os.PathLike[str], parent: Reads | None = None) -> None:
        self.root = os.path.abspath(os.fspath(root))
        self._prefix = os.path.join(self.root, "")
        self._paths: set[str] = set()
        self._parent = parent

    def add(self, path: str | os.PathLike[str]) -> None:
        p = os.path.abspath(os.fspath(path))
        if p.startswith(self._prefix) and os.path.isfile(p):
            self._paths.add(p[len(self._prefix):].replace(os.sep, "/"))
        if self._parent is not None:
            self._parent.add(p)

    def paths(self) -> list[str]:
        return sorted(self._paths)


_current: ContextVar[Reads | None] = ContextVar("courant_case_reads", default=None)


@contextmanager
def recording(root: str | os.PathLike[str]) -> Iterator[Reads]:
    """Collect every `note` made in the block for files under `root`. The recording that was
    active before sees the same notes (a file read for `courant convergence` inside a verdict was
    read for the verdict too), and it is restored on exit."""
    reads = Reads(root, parent=_current.get())
    token = _current.set(reads)
    try:
        yield reads
    finally:
        _current.reset(token)


def note(path: str | os.PathLike[str]) -> None:
    """A reader read `path`. Recorded when a recording is active and `path` is a regular file
    under its root; otherwise nothing happens."""
    reads = _current.get()
    if reads is not None:
        reads.add(path)


@contextmanager
def reading(path: str | os.PathLike[str]) -> Iterator[None]:
    """Note `path` once the block has read it (VER-14: listed only after a successful read).

    The block ends normally: read, so noted. It raises `OSError`: nothing was read (the file
    cannot be opened, or vanished), so it is not noted. It raises anything else: the bytes were
    read and did not parse, so it is noted. The exception propagates either way, to the reader's
    own fallback. Inside a confinement, a path outside it never reaches the block: `OutsideRoot`,
    an `OSError`, so not noted (MCP-3). A path that is not a regular file never reaches it either
    (`regular_file`).
    """
    refuse_outside(path)
    regular_file(path)
    try:
        yield
    except OSError:
        raise
    except Exception:
        note(path)
        raise
    note(path)
