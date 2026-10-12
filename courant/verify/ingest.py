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
from pathlib import Path

from courant.verify.schema import InputError, InputFile, RunInputs, manifest

__all__ = ["InputError", "from_case_dir", "from_log", "from_path", "with_spec"]


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="replace")


def from_log(log_path: str | Path, case_json: str | Path | None = None) -> RunInputs:
    log_path = Path(log_path)
    if not log_path.is_file():
        raise InputError(f"no such log file: {log_path}")
    text = _read_text(log_path)
    inputs = [InputFile.of(log_path, log_path.name)]
    meta: dict = {}
    if case_json is not None:
        case_json = Path(case_json)
        if not case_json.is_file():
            raise InputError(f"no such case sidecar: {case_json}")
        try:
            meta = json.loads(case_json.read_text())
        except json.JSONDecodeError as exc:
            raise InputError(f"{case_json.name} is not valid JSON: {exc}") from exc
        inputs.append(InputFile.of(case_json, case_json.name))
    stem = log_path.name.removesuffix(".gz").removesuffix(".log")
    return RunInputs(
        run_id=meta.get("scenario_id") or stem,
        log_text=text, log_name=log_path.name,
        config_view=meta.get("config"), mesh_view=meta.get("mesh"),
        inputs=tuple(sorted(inputs, key=lambda f: f.path)),
        source="corpus" if case_json else "log",
        solver_hint=meta.get("solver"),
    )


def from_case_dir(case_dir: str | Path) -> RunInputs:
    from courant.case.mesh import read_mesh_quality
    from courant.case.reads import note, recording
    from courant.cli.tail import find_log, utility_logs

    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise InputError(f"no such case directory: {case_dir}")
    with recording(case_dir) as reads:
        log = find_log(case_dir)                        # notes every log it ranked
        if log is None:
            utilities = utility_logs(case_dir)
            only = (f"; only utility logs ({', '.join(utilities)}), which are not the run's "
                    f"output. Pass the solver's log file instead" if utilities else "")
            raise InputError(f"no solver log (log.<solver>) found in {case_dir}{only}")
        log = Path(log)
        text = _read_text(log)
        note(log)                                       # hashed once, with the rest

        config_view = None
        try:
            from courant.case.config import CaseConfig

            config_view = CaseConfig.from_foam(str(case_dir)).model_dump(mode="json")
        except Exception:                               # noqa: BLE001
            config_view = None                          # recorded as not assessed
        mesh = read_mesh_quality(str(case_dir))         # notes the checkMesh log it parsed

    return RunInputs(
        run_id=case_dir.name, log_text=text, log_name=log.name,
        config_view=config_view,
        mesh_view=mesh.model_dump(mode="json") if mesh is not None else None,
        inputs=manifest(case_dir, reads.paths()),
        source="case_dir", case_dir=str(case_dir.resolve()),
    )


def from_path(path: str | Path, case_json: str | Path | None = None) -> RunInputs:
    path = Path(path)
    return from_case_dir(path) if path.is_dir() else from_log(path, case_json)


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
    locked = locked_sha256.strip().lower() if locked_sha256 else None
    return inputs.model_copy(update={
        "spec": spec.model_dump(mode="json", by_alias=True),
        "spec_sha256": sha, "spec_locked_sha256": locked})
