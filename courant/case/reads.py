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
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


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
    own fallback.
    """
    try:
        yield
    except OSError:
        raise
    except Exception:
        note(path)
        raise
    note(path)
