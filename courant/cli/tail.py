"""Following a solver log that is still being written (Tech Bible §06).

``courant watch`` attaches to a solve that is already running, or about to be.
That means reading a file whose author is a separate process which may not have
started yet, may be writing faster than we read, and may — when the engineer
re-runs after a fix — replace the file underneath us mid-read.

Handling those three cases is the whole job, and each one is a real thing that
happens on a cluster rather than a hypothetical:

  * **The file does not exist yet.** ``courant watch`` is typed *before*
    ``simpleFoam > log.simpleFoam``, because that is the natural order and
    because on a scheduler the solve starts whenever the queue says so. Waiting
    is the correct behaviour, not an error.
  * **The file is truncated.** A re-run with the same redirect resets the file
    to zero bytes. For this product that is not corruption — it is the *most
    important* event in the loop: the run being verified after a fix. It is
    reported as a restart, not swallowed.
  * **The file is replaced.** Log rotation, or a re-run with ``>`` creating a
    new inode. Same signal, detected by inode rather than size.

We always read from byte 0, never from the tail. Attaching to a solve that is
already 4,000 iterations in and diagnosing only what happens next would mean
diagnosing without the residual history, which is most of the evidence. It also
makes every offset an absolute file offset, which is what the server dedupes
uploaded events on.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from courant.case.reads import reading

#: How often to look for new bytes. Solver logs are written in bursts of a few
#: hundred bytes per iteration; polling is both simpler and more portable than
#: inotify/FSEvents, and 250ms is far below human perception on a live display.
POLL_INTERVAL = 0.25

#: Give up waiting for a log that never appears. Long enough to survive a
#: scheduler queue of a few minutes; short enough that a typo in the path does
#: not hang a terminal all night.
DEFAULT_START_TIMEOUT = 600.0

#: Silence long enough to mean "the solver is done, or wedged". OpenFOAM writes
#: at least one line per iteration, and an iteration that takes longer than this
#: on any real case is itself the anomaly the watcher exists to report.
DEFAULT_IDLE_TIMEOUT = 120.0

READ_SIZE = 1 << 20        # 1 MiB per poll — bounds memory on a fast solve


@dataclass(frozen=True, slots=True)
class Chunk:
    """New log text, and where it came from."""

    text: str
    offset: int                 # absolute byte offset this chunk starts at
    restarted: bool = False     # the file was truncated or replaced before this


class LogTail:
    """Follow a growing file, yielding chunks until it stops growing.

    Not a general-purpose ``tail -f``: it is specialised to the one thing the
    agent needs, which is a byte-accurate, restart-aware stream with a bounded
    definition of "finished".
    """

    def __init__(
        self,
        path: str | Path,
        *,
        poll_interval: float = POLL_INTERVAL,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        idle_timeout: float | None = DEFAULT_IDLE_TIMEOUT,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self.start_timeout = start_timeout
        self.idle_timeout = idle_timeout
        self._clock = clock
        self._sleep = sleep
        self._stopped = False
        self.position = 0
        self._inode: int | None = None
        # The first bytes this file had when we read them. Linux reuses a deleted file's inode
        # number at once, so a replacement can keep the inode; a different head shows it.
        self._head: bytes = b""

    def stop(self) -> None:
        """Ask the follow loop to finish. Safe from a signal handler."""
        self._stopped = True

    # -- waiting ------------------------------------------------------------

    def wait_for_file(self) -> bool:
        """Block until the log appears. False if it never did."""
        deadline = self._clock() + self.start_timeout
        while not self._stopped:
            if self.path.exists():
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self.poll_interval)
        return False

    # -- following ----------------------------------------------------------

    def follow(self) -> Iterator[Chunk]:
        """Yield chunks until the file goes quiet, or ``stop()`` is called.

        Ends by returning, never by raising, so a caller's ``for`` loop is also
        its shutdown path: whatever stops the stream, the run still gets
        verified and reported.
        """
        if not self.wait_for_file():
            return

        last_data_at = self._clock()

        while not self._stopped:
            restarted = self._detect_restart()
            text = self._read_new()

            if text:
                yield Chunk(text=text, offset=self.position - len(text.encode()),
                            restarted=restarted)
                last_data_at = self._clock()
            elif restarted:
                # The file was replaced but nothing is in it yet. Tell the
                # caller now rather than sitting on it: this is the re-run
                # starting, and the display should say so.
                yield Chunk(text="", offset=0, restarted=True)
                last_data_at = self._clock()
            elif self.idle_timeout is not None and (
                self._clock() - last_data_at >= self.idle_timeout
            ):
                return

            self._sleep(self.poll_interval)

    def read_all(self) -> str:
        """Everything written so far, without following. For a finished run."""
        if not self.path.exists():
            return ""
        self._detect_restart()
        return self._read_new()

    # -- internals ----------------------------------------------------------

    def _detect_restart(self) -> bool:
        """True if the file was truncated or replaced since the last read."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            # Deleted mid-run. Treat the next appearance as a restart rather
            # than resuming at a stale offset into a different file.
            self._inode, self.position, self._head = None, 0, b""
            return False

        replaced = self._inode is not None and stat.st_ino != self._inode
        truncated = stat.st_size < self.position
        if not replaced and not truncated and self._head:
            replaced = self._read_head(len(self._head)) != self._head
        self._inode = stat.st_ino

        if replaced or truncated:
            self.position, self._head = 0, b""
            return True
        return False

    def _read_head(self, n: int) -> bytes:
        try:
            with self.path.open("rb") as handle:
                return handle.read(n)
        except FileNotFoundError:
            return b""

    def _read_new(self) -> str:
        """Bytes since ``position``, decoded leniently."""
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.position)
                raw = handle.read(READ_SIZE)
        except FileNotFoundError:
            return ""

        if not raw:
            return ""

        # A chunk boundary can split a multi-byte character. Dropping the
        # partial tail and rewinding is what keeps offsets byte-exact — decoding
        # with `errors="replace"` would silently change the length.
        text, consumed = _decode_whole_characters(raw)
        if self.position < HEAD_BYTES:
            self._head = (self._head + raw[:consumed])[:HEAD_BYTES]
        self.position += consumed
        return text


#: How many of a log's first bytes identify it (a solver banner, then the case header).
HEAD_BYTES = 256


def _decode_whole_characters(raw: bytes) -> tuple[str, int]:
    """Decode as much valid UTF-8 as possible; return the text and bytes used."""
    for cut in range(len(raw), max(len(raw) - 4, -1), -1):
        try:
            return raw[:cut].decode("utf-8"), cut
        except UnicodeDecodeError:
            continue
    # Not a truncation problem — genuinely invalid bytes. Solver logs are ASCII
    # in practice; a binary blob in one is not worth stalling a run over.
    return raw.decode("utf-8", errors="replace"), len(raw)


def wait_for_log(
    case_dir: str | Path,
    *,
    timeout: float = DEFAULT_START_TIMEOUT,
    poll_interval: float = 1.0,
    on_wait: Callable[[], None] | None = None,
    clock=time.monotonic,
    sleep=time.sleep,
) -> Path | None:
    """Wait for a solver log to appear in a case directory.

    ``LogTail`` can already wait for a *named* file that does not exist yet. This
    is the case where we do not know the name either, which is the normal one:
    an engineer opens two panes and types ``courant watch .`` in the first
    before ``simpleFoam > log.simpleFoam`` in the second. Exiting with "no log
    found" would make the tool demand an ordering the engineer did not choose.
    """
    deadline = clock() + timeout
    announced = False
    while True:
        found = find_log(case_dir)
        if found is not None:
            return found
        if clock() >= deadline:
            return None
        if on_wait is not None and not announced:
            on_wait()
            announced = True
        sleep(poll_interval)


#: OpenFOAM utilities that write a `log.<name>` beside the solver's. Named
#: rather than inferred because the decisive one runs *after* the solver:
#: `reconstructPar` is the last step of every parallel workflow, so its log is
#: always newer than the solve it belongs to.
_UTILITIES = frozenset({
    "blockMesh", "snappyHexMesh", "decomposePar", "reconstructPar",
    "reconstructParMesh", "checkMesh", "setFields", "mapFields", "topoSet",
    "createPatch", "extrudeMesh", "surfaceFeatureExtract", "renumberMesh",
    "transformPoints", "refineMesh", "splitMeshRegions", "foamDictionary",
    "changeDictionary", "redistributePar", "collapseEdges", "zipUpMesh",
})

#: A solver iterates; a utility does not. Either marker is decisive.
_TIME_LOOP = re.compile(r"^(?:Starting time loop|Time = )", re.M)

#: Every OpenFOAM log opens with a banner naming what produced it.
_EXEC_LINE = re.compile(r"^Exec\s*:\s*(\S+)", re.M)


def _log_rank(path: Path) -> int:
    """2 = a solve · 1 = unknown · 0 = a known utility.

    THE BANNER WINS, and this order is the whole point. The obvious rule —
    "a solver iterates, a utility does not, so look for `Time =`" — is wrong,
    and measurably so: `decomposePar` and `reconstructPar` both print a
    `Time = <t>` line per time directory they touch. Ranking those two by
    content puts them level with the solve they bracket.

    So a log whose `Exec :` line names a known utility is a utility, whatever it
    printed. Content is the fallback for an application we do not have in the
    list, where evidence of a time loop is the best signal available.
    """
    try:                                # VER-14: this head decided which log is the solve's
        with reading(path), open(path, "rb") as fh:
            head = fh.read(64 * 1024).decode("utf-8", "replace")
    except OSError:
        return 1

    exec_match = _EXEC_LINE.search(head)
    if exec_match:
        app = exec_match.group(1).rsplit("/", 1)[-1]
        if app in _UTILITIES:
            return 0
        # A named application that is not a known utility: a solver, or a solver
        # we have never heard of. Either way it outranks a utility.
        return 2 if _TIME_LOOP.search(head) else 1

    if _TIME_LOOP.search(head):
        return 2

    # No banner: not an OpenFOAM log, or truncated before the header was written.
    stem = path.name[4:] if path.name.startswith("log.") else path.stem
    return 0 if stem in _UTILITIES else 1


def find_log(case_dir: str | Path) -> Path | None:
    """Guess the log an OpenFOAM run is writing to.

    Convention is ``log.<solver>`` in the case directory — near-universal, and
    what every OpenFOAM tutorial and Allrun script produces.

    NEWEST IS NOT ENOUGH, and the case where it fails is the one we most want to
    work. A parallel workflow is `blockMesh · decomposePar · mpirun solver ·
    reconstructPar`, so the newest `log.*` in a finished parallel case is
    `log.reconstructPar` — a few lines of utility output with no residuals in it.
    Watching that file reports a clean run. Measured on a decomposed cavity case
    that had diverged to a Courant number of 1580: Courant followed
    `log.decomposePar` and returned **Trust Grade A**.

    So: rank by what the file contains, and only then by age. Newest still wins
    among logs of equal rank, which is what makes re-running a case pick up the
    new solve rather than yesterday's.

    A known utility's log is never the run's log. When only those exist (the
    solver's output went elsewhere, or the solver never ran) there is no solver
    log, and None says so; `utility_logs` names what was there. Returning
    `log.checkMesh` made `courant verify` judge a mesh report as the run
    (insufficient, VER-5.5) on a case whose solver had run, without saying why.
    """
    case = Path(case_dir)
    candidates = [
        p for p in case.glob("log.*")
        if p.is_file() and not p.name.endswith((".gz", ".bak"))
    ]
    candidates += [p for p in case.glob("*.log") if p.is_file()]
    if not candidates:
        return None
    rank, _, best = max((_log_rank(p), p.stat().st_mtime, p) for p in candidates)
    return best if rank > 0 else None


def utility_logs(case_dir: str | Path) -> list[str]:
    """The known utilities' logs in a case directory, by name: what `find_log` passed over."""
    case = Path(case_dir)
    return sorted(p.name for p in case.glob("log.*")
                  if p.is_file() and p.name[4:] in _UTILITIES)
