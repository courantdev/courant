"""Gather what the verifier reads, before anything is parsed (TRD VER-3).

Three sources: a case directory, a log file, or a log with a corpus sidecar
(`*.case.json`, the format of `certification/corpus/`). The ingest reads and
hashes files. It never writes, runs a process, or opens a socket (INV-5).

For a case directory the manifest is what the readers read (VER-14): each reader
notes its files at the line that reads them (`courant.case.reads`), and the
manifest hashes exactly those. Nothing here guesses which files matter.
"""

from __future__ import annotations

import gzip
import json
import stat
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

from courant.verify.schema import InputError, InputFile, RunInputs, manifest

__all__ = ["InputError", "from_case_dir", "from_log", "from_path", "lock_hash", "lock_of",
           "read_sidecar", "sidecar_of", "with_spec"]

#: What a lock is (VPA-ORDER-1): 64 hexadecimal digits, after strip and lower-case.
LOCK_HASH_REQUIRED = "the 64-hex SHA-256 the specification was locked at"


def lock_hash(value: str | None, flag: str = "--spec-sha256") -> str | None:
    """The hash a specification was locked at, normalised (stripped, lower-case), or None when
    none was given. Anything given that is not 64 hexadecimal digits, the empty string and
    whitespace included, is an InputError: an unset `$LOCK` must not drop the lock (VPA-ORDER-1,
    SEC070-18). The one validator of a lock, asked under `spec.lock_strict` (`lock_of`); the
    pattern is `courant.toml`'s own."""
    from courant.gate.config import _HEX64

    if value is None:
        return None
    locked = value.strip().lower()
    if not _HEX64.match(locked):
        shown = repr(value) if len(value) <= 80 else repr(value[:77] + "...")
        raise InputError(f"{flag} must be {LOCK_HASH_REQUIRED}, not {shown}")
    return locked


def lock_of(value: str | None, checks: str = "default") -> str | None:
    """The lock a `--spec-sha256` value asks for, under `checks`: the one reading of the option
    (`with_spec`, `courant verify`, `courant spec check`). With `spec.lock_strict` (`--checks
    next`), `lock_hash`. Without it, 0.7.0's: an empty value is no lock, and a blank or malformed
    one is kept, so a blank one is no lock (`policy.spec_void`) and a malformed one is VER-5.8."""
    from courant.verify.checks import SPEC_LOCK_STRICT, enabled

    if enabled(SPEC_LOCK_STRICT, checks):
        return lock_hash(value)
    return value.strip().lower() if value else None


def sidecar_of(log: str | Path) -> Path:
    """Where a log's corpus sidecar is, whether or not it is there: `<run>.case.json` beside it,
    `<run>` being the log's name without `.gz` and `.log`. The one pairing `--batch`, `courant
    gate` and `--check-pack` hold a log to."""
    log = Path(log)
    return log.with_name(log.name.removesuffix(".gz").removesuffix(".log") + ".case.json")


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="replace")


@contextmanager
def _as_input_error(path: Path) -> Iterator[None]:
    """A run log or sidecar that is there and cannot be read (it cannot be opened, it vanished
    or was swapped for a FIFO since it was found, a directory on the way cannot be searched, or a
    `.gz` is corrupt or truncated) is an `InputError` that names it by its file name (VER-14,
    T-MANIFEST-3): an input, not an internal error, so a batch reports it beside the other runs.
    The reason never names the absolute path (`reason`). An MCP-3 refusal (`OutsideRoot`) is
    never reworded: it passes through as it is. Put this outside `reading(path)`, so a file
    that could not be opened is not noted as read."""
    from courant.case.reads import OutsideRoot, reason

    try:
        yield
    except OutsideRoot:
        raise
    except zlib.error as exc:                   # `type(exc).__name__` is just "error"
        raise InputError(f"{path.name} could not be read: corrupt compressed data: {exc}") from exc
    except (OSError, EOFError) as exc:
        raise InputError(f"{path.name} could not be read: {reason(exc)}") from exc


def _is_file(path: Path) -> bool:
    """`Path.is_file`, with a path that cannot be told (a directory on the way cannot be
    searched) an `InputError` naming it, never a `PermissionError` (`reads.kind`)."""
    from courant.case.reads import kind, reason, unknown

    found = kind(path, stat.S_ISREG)
    if found is None:
        raise InputError(f"{path.name} could not be read: {reason(unknown(path))}")
    return found


def _double_int(text: str) -> int:
    """A JSON integer that a double can hold, else OverflowError (`read_sidecar`)."""
    v = int(text)
    float(v)
    return v


def read_sidecar(case_json: str | Path) -> dict:
    """A corpus sidecar (`*.case.json`), for `courant verify` and `courant preflight` alike. Only a
    regular file is opened: a FIFO would block the read forever."""
    case_json = Path(case_json)
    if not _is_file(case_json):
        raise InputError(f"no such case sidecar: {case_json}")
    with _as_input_error(case_json):
        data = case_json.read_bytes()
    try:
        # JSON is UTF-8 (RFC 8259), whatever the locale. An integer past a double's range is
        # refused here: the checks compare numbers as floats, and `float()` of it raises.
        meta = json.loads(data, parse_int=_double_int)
    except OverflowError as exc:
        raise InputError(f"{case_json.name} holds an integer too large to read as a number"
                         ) from exc
    except (ValueError, RecursionError) as exc:     # not JSON, not UTF-8, or nested too deep
        raise InputError(f"{case_json.name} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise InputError(f"{case_json.name} is not a JSON object")
    return meta


def from_log(log_path: str | Path, case_json: str | Path | None = None) -> RunInputs:
    log_path = Path(log_path)
    if not _is_file(log_path):
        raise InputError(f"no such log file: {log_path}")
    with _as_input_error(log_path):
        text = _read_text(log_path)
    inputs = [InputFile.of(log_path, log_path.name)]
    meta: dict = {}
    if case_json is not None:
        case_json = Path(case_json)
        meta = read_sidecar(case_json)
        inputs.append(InputFile.of(case_json, case_json.name))
    stem = log_path.name.removesuffix(".gz").removesuffix(".log")
    from pydantic import ValidationError

    try:
        return _log_inputs(meta, stem, text, log_path, inputs, case_json)
    except ValidationError as exc:          # INV6-F2: a sidecar field of the wrong JSON type
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise InputError(f"{case_json.name if case_json else log_path.name}: {where}: "
                         f"{first.get('msg')}") from exc


def _log_inputs(meta, stem, text, log_path, inputs, case_json) -> RunInputs:
    return RunInputs(
        run_id=meta.get("scenario_id") or stem,
        log_text=text, log_name=log_path.name,
        config_view=meta.get("config"), mesh_view=meta.get("mesh"),
        inputs=tuple(sorted(inputs, key=lambda f: f.path)),
        source="corpus" if case_json else "log",
        solver_hint=meta.get("solver"),
    )


def from_case_dir(case_dir: str | Path, *, checks: str = "default") -> RunInputs:
    """A case directory's inputs. `checks` is `--checks` (GATE-13), decided here because it
    decides what is read: with "next", a case with no checkMesh log has its aspect ratio
    computed from `constant/polyMesh` (VER-7a), and `constant/g` and, for a case with gravity,
    the wall faces are read for the regime check (VER-17). Under the default neither is read,
    so the manifest is v0.5.0's."""
    from courant.case.foamguard import guarded
    from courant.case.mesh import read_mesh_quality
    from courant.case.reads import reading, recording, unreadable_dir
    from courant.case.solvers import solver_exec
    from courant.cli.tail import find_log, utility_logs
    from courant.verify.checks import CASE_MERGE_WITH_VALUE, FOAM_GUARD, enabled

    enabled("", checks)                                 # refuse an unknown mode first

    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise InputError(f"no such case directory: {case_dir}")
    # A directory that cannot be listed or searched says so, not that it holds no solver log
    # (GATE-14, T-PRECHECK-2 C1, C2), and `find_log` never meets it.
    why = unreadable_dir(case_dir)
    if why is not None:
        raise InputError(f"{case_dir.name} {why}")
    with recording(case_dir) as reads, guarded(enabled(FOAM_GUARD, checks)):
        # notes every log it ranked; a `log.*.gz` only when no plain log is the solve's (T-GZ-CASE)
        log = find_log(case_dir, checks=checks, finished=True)
        if log is None:
            utilities = utility_logs(case_dir, checks=checks, finished=True)
            only = (f"; only utility logs ({', '.join(utilities)}), which are not the run's "
                    f"output. Pass the solver's log file instead" if utilities else "")
            raise InputError(f"no solver log (log.<solver>) found in {case_dir}{only}")
        log = Path(log)
        with _as_input_error(log), reading(log):        # hashed once, with the rest
            text = _read_text(log)

        config_view = None
        try:
            from courant.case.config import CaseConfig

            config_view = CaseConfig.from_foam(
                str(case_dir), buoyancy=enabled("physics.laminar_regime", checks),
                directives=enabled("case.directives", checks),
                steady_ddt=enabled("cfl.steady", checks),           # VER-13a, behind next
                zero_unlisted=enabled("case.zero_unlisted", checks),
                regions=enabled("case.regions", checks),            # VER-3d, behind next
                regions_whole=enabled("case.regions.whole", checks),  # VER-3e, behind next
                merge_values=enabled(CASE_MERGE_WITH_VALUE, checks),  # DO-2, behind next
                compressed=enabled("case.zero_compressed", checks),  # VER-3f, behind next
                patches=enabled("case.boundary_patches", checks),   # VER-3g, behind next
                solver=ran.program if (ran := solver_exec(text)) is not None else None,
            ).model_dump(mode="json")
        except Exception:                               # noqa: BLE001
            config_view = None                          # recorded as not assessed
        # Notes the checkMesh log it parsed, or the polyMesh files it computed the aspect from.
        mesh = read_mesh_quality(str(case_dir), log_text=text,
                                 polymesh=enabled("mesh.aspect_ratio.polymesh", checks),
                                 regions=(config_view or {}).get("regions") or None,
                                 by_solver=enabled("case.regions.whole", checks))  # VER-3e

    return RunInputs(
        run_id=case_dir.name, log_text=text, log_name=log.name,
        config_view=config_view,
        mesh_view=mesh.model_dump(mode="json") if mesh is not None else None,
        inputs=manifest(case_dir, reads.paths()),
        source="case_dir", case_dir=str(case_dir.resolve()),
        checks=cast(Literal["default", "next"], checks),  # enabled() refused any other mode
    )


def from_path(path: str | Path, case_json: str | Path | None = None, *,
              checks: str = "default") -> RunInputs:
    """The inputs at `path`, to be verified under `checks` (GATE-13; `RunInputs.checks`)."""
    from courant.case.reads import kind, reason, unknown

    path = Path(path)
    if (is_dir := kind(path)) is None:              # a directory on the way cannot be searched
        raise InputError(f"{path.name} could not be read: {reason(unknown(path))}")
    if is_dir:
        return from_case_dir(path, checks=checks)
    inputs = from_log(path, case_json)
    return inputs if checks == "default" else inputs.model_copy(update={"checks": checks})


def with_spec(inputs: RunInputs, spec_path: str | Path,
              locked_sha256: str | None = None) -> RunInputs:
    """The same inputs, plus the task specification the run was asked to satisfy.

    `locked_sha256` is the hash the specification was locked at before the run; a file that
    no longer matches it makes the verdict NOT_ASSESSABLE (PRD 2.0 P2-SPEC-3).
    """
    from courant.spec import SpecError, load_spec

    try:
        spec, sha = load_spec(spec_path)
    except SpecError as exc:
        raise InputError(str(exc)) from exc
    locked = lock_of(locked_sha256, inputs.checks)
    return inputs.model_copy(update={
        "spec": spec.model_dump(mode="json", by_alias=True),
        "spec_sha256": sha, "spec_locked_sha256": locked})
