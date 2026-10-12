"""The batch verdict cache (TRD GATE-11): re-verify only the runs that changed.

A wrong cached verdict is worse than no cache, so every choice here is the conservative one.

An entry (`courant.cache/1`) holds one verdict, the fingerprint of the run it was judged on, its
own SHA-256 and an HMAC. It is read and written by `courant verify --batch` and `courant gate`
only: no MCP tool uses it (MCP-4), and a single `courant verify` never does. It is off by default
for `courant gate` and wherever the `CI` variable is set (`--cache` turns it on): a CI runner is
a shared, throw-away machine, and the cache is a convenience for a person's own.

**The key.** Everything the verdict could depend on goes into it:

- the context: the Courant version, the policy (a digest of the engine's own source, so a
  development tree that changes a rule without changing the version cannot serve a stale
  answer), the platform, the Python version, the versions of the libraries the readers parse with
  (a parser upgrade can change what a file means), the verdict schema major, the specification's
  bytes and its lock hash, and the rules files (OVR-3, none yet);
- the run's fingerprint, taken before the run is verified: the VER-14 manifest (the hashes of the
  files the readers read) and a snapshot of the run's tree: every directory (an empty `1000/`
  that appears or goes changes which time directory a reader takes), every file with its size and
  permission bits (a reader skips a file over a size limit, and cannot read one it may not),
  every symbolic link with its target, symlinked directories followed (cycles marked and not
  followed). A file nothing opened cannot change the verdict by its bytes (the manifest test
  proves it), so its bytes are not hashed; its size and mode are.

A lookup re-hashes the stored manifest's files and compares the snapshot. It never parses a log.

**Trust.** The cache directory is trusted no further than its owner. An entry carries an HMAC-SHA256
over everything in it, keyed with a per-user secret kept outside the cache directory (a 32-byte
file, mode 0600, created on first use: `$COURANT_CACHE_KEY_FILE`, else `$XDG_CONFIG_HOME/courant/
cache.key`, else `~/.config/courant/cache.key`), so a file dropped into the directory by someone
else is a miss. The directory itself must be owned by the user and mode 0700, and the key file
must be owned by the user with no group or other bits; otherwise the cache is not read or written.
An entry is also discarded when its SHA-256 does not match, when it names another key, or when
its manifest and its verdict's `run.inputs` disagree. A discarded entry is a miss, never an
error: the run is verified again and the entry rewritten.

The cache lives outside every case, in `$COURANT_CACHE_DIR`, else `$XDG_CACHE_HOME/courant/
verdicts`, else `~/.cache/courant/verdicts`. Entries are written to a temporary file and moved
into place, so two jobs writing at once cannot tear one.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import stat
import sys
import tempfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from courant.verify.schema import canonical_json
from courant.verify.verdict2 import VerdictV2, load_verdict

SCHEMA = "courant.cache/1"
#: The verdict schema major the entries hold. Version 1 is derived from it on the way out.
STORED_MAJOR = 2


def default_dir() -> Path:
    env = os.environ.get("COURANT_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "courant" / "verdicts"


@lru_cache(maxsize=1)
def policy_digest() -> str:
    """SHA-256 over every source file of the `courant` package, by relative path.

    The version alone cannot say which rules decided: between releases the tree changes under
    one version string. Keying on the source makes the cache wrong-answer-proof by construction.
    """
    import courant

    root = Path(courant.__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _sha_of(path: Path | None) -> str | None:
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


#: The libraries the readers parse with. A new version can read the same bytes differently.
READER_LIBRARIES = ("foamlib", "numpy", "pydantic", "pydantic_core")


def _dist_version(name: str) -> str:
    """The installed version of a distribution, or its module's `__version__` (a packaged
    binary has no distribution metadata), or `absent`."""
    import importlib
    from importlib import metadata

    try:
        return metadata.version(name.replace("_", "-"))
    except metadata.PackageNotFoundError:
        pass
    try:
        return str(importlib.import_module(name).__version__)
    except Exception:                                   # noqa: BLE001 - absent or unreadable
        return "absent"


def context(*, spec: str | Path | None = None, spec_locked_sha256: str | None = None,
            rules: Iterable[str | Path] = ()) -> dict:
    """The run-independent part of the key."""
    from courant import __version__

    try:
        spec_sha = _sha_of(Path(spec)) if spec is not None else None
    except OSError:
        spec_sha = "unreadable"        # the run will report the unreadable file; never cache it
    return {
        "courant_version": __version__,
        "policy": policy_digest(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "python": sys.version.split()[0],
        "dependencies": {name: _dist_version(name) for name in READER_LIBRARIES},
        "verdict_schema_major": STORED_MAJOR,
        "spec_sha256": spec_sha,
        "spec_locked_sha256": spec_locked_sha256.strip().lower() if spec_locked_sha256 else None,
        "rules": sorted(_sha_of(Path(r)) or "" for r in rules),
    }


def run_key(ctx: dict, run_name: str) -> str:
    """The cache key of one run: the context and where the run sits under the batch root."""
    return hashlib.sha256(canonical_json({"context": ctx, "run": run_name}).encode()).hexdigest()


def _file_entry(rel: str, st: os.stat_result) -> str:
    return f"f:{rel}:{st.st_size}:{stat.S_IMODE(st.st_mode):o}"


def snapshot(target: Path, sidecar: Path | None) -> list[str]:
    """What a run's tree looks like to a reader, as sorted lines (the cache's fingerprint).

    `d:<dir>/:<mode>` for every directory, `f:<path>:<size>:<mode>` for every file,
    `l:<path>-><target>` for every symbolic link, `c:<dir>/` where a symlinked directory leads
    back into the directories being walked (not followed), `e:<path>` for what could not be
    listed. Symlinked directories are followed, so a file that appears behind a link is seen.
    For a log, the log and its sidecar."""
    target = Path(target)
    if not target.is_dir():
        out = []
        for p in [target, *([sidecar] if sidecar is not None else [])]:
            try:
                out.append(_file_entry(p.name, p.stat()))
            except OSError:
                out.append(f"e:{p.name}")
        return sorted(out)

    out: list[str] = []

    def walk(directory: Path, rel: str, ancestors: tuple[str, ...]) -> None:
        real = os.path.realpath(directory)
        try:
            out.append(f"d:{rel}/:{stat.S_IMODE(directory.stat().st_mode):o}")
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            out.append(f"e:{rel}/")
            return
        for e in entries:
            path = f"{rel}/{e.name}" if rel else e.name
            try:
                if e.is_symlink():
                    out.append(f"l:{path}->{os.readlink(e.path)}")
                if e.is_dir():                              # follows a symlink
                    child = os.path.realpath(e.path)
                    if child == real or child in ancestors:
                        out.append(f"c:{path}/")
                    else:
                        walk(Path(e.path), path, (*ancestors, real))
                else:
                    out.append(_file_entry(path, e.stat()))
            except OSError:
                out.append(f"e:{path}")

    walk(target, "", ())
    return sorted(out)


def run_base(target: Path) -> Path:
    """Where a run's manifest paths are relative to."""
    return Path(target) if Path(target).is_dir() else Path(target).parent


def _entry_digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k not in ("sha256", "hmac")}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()


def key_file() -> Path:
    env = os.environ.get("COURANT_CACHE_KEY_FILE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "courant" / "cache.key"


def _private(path: Path, *, directory: bool) -> bool:
    """Owned by this user and closed to everyone else (a directory: exactly 0700)."""
    try:
        st = path.lstat()
    except OSError:
        return False
    if directory and not stat.S_ISDIR(st.st_mode):
        return False
    if not directory and not stat.S_ISREG(st.st_mode):
        return False
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return False
    mode = stat.S_IMODE(st.st_mode)
    return mode == 0o700 if directory else (mode & 0o077) == 0


def _secret() -> bytes | None:
    """The per-user HMAC secret, created on first use (32 random bytes, mode 0600). None when
    there is none that can be trusted."""
    path = key_file()
    try:
        if not path.exists():
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(os.urandom(32))
        if not _private(path, directory=False):
            return None
        data = path.read_bytes()
        return data if len(data) == 32 else None
    except OSError:
        return None


class VerdictCache:
    """A directory of `courant.cache/1` entries. `key` is the HMAC secret (tests give one; by
    default it is the user's key file)."""

    def __init__(self, directory: str | Path | None = None, key: bytes | None = None):
        self.dir = Path(directory) if directory is not None else default_dir()
        self._key = key

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _secret(self) -> bytes | None:
        if self._key is None:
            self._key = _secret()
        return self._key

    def _mac(self, entry: dict, secret: bytes) -> str:
        body = {k: v for k, v in entry.items() if k != "hmac"}
        return hmac.new(secret, canonical_json(body).encode(), hashlib.sha256).hexdigest()

    def get(self, key: str, target: Path, sidecar: Path | None,
            snap: list[str] | None = None) -> VerdictV2 | None:
        """The verdict cached for this run, if the run is byte-for-byte what it was judged on."""
        secret = self._secret()
        if secret is None or not _private(self.dir, directory=True):
            return None
        try:
            entry = json.loads(self._path(key).read_text(encoding="utf-8"))
            if (entry.get("schema") != SCHEMA or entry.get("key") != key
                    or not isinstance(entry.get("hmac"), str)
                    or not hmac.compare_digest(entry["hmac"], self._mac(entry, secret))
                    or entry.get("sha256") != _entry_digest(entry)):
                return None
            fp = entry["fingerprint"]
            verdict = load_verdict(entry["verdict"])
            stored = [dict(f) for f in fp["inputs"]]
            if stored != [f.model_dump(mode="json") for f in verdict.run.inputs]:
                return None
            if fp["listing"] != (snapshot(target, sidecar) if snap is None else snap):
                return None
            base = run_base(target)
            for f in stored:
                data = (base / f["path"]).read_bytes()
                if len(data) != f["bytes"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
                    return None
            return verdict
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def put(self, key: str, target: Path, sidecar: Path | None, verdict: VerdictV2,
            snap: list[str] | None = None) -> None:
        """Store a verdict. `snap` is the run's snapshot taken before it was verified (a file
        that changed while the run was read must not be recorded as the one that was judged).
        A cache that cannot be written is a cache that stays empty, not a failed batch."""
        secret = self._secret()
        if secret is None:
            return
        entry = {
            "schema": SCHEMA, "key": key,
            "fingerprint": {
                "inputs": [f.model_dump(mode="json") for f in verdict.run.inputs],
                "listing": snapshot(target, sidecar) if snap is None else snap},
            "verdict": json.loads(verdict.to_json()),
        }
        entry["sha256"] = _entry_digest(entry)
        entry["hmac"] = self._mac(entry, secret)
        try:
            if not self.dir.exists():
                self.dir.parent.mkdir(parents=True, exist_ok=True)
                self.dir.mkdir(mode=0o700)
                self.dir.chmod(0o700)
            if not _private(self.dir, directory=True):
                return
            fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".entry-", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(canonical_json(entry))
            os.replace(tmp, self._path(key))
        except OSError:
            return
