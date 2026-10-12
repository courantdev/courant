"""`courant.record/1`: write one, and re-derive it (TRD REC-1).

    {"schema": "courant.record/1",
     "tool": {"name", "version"},
     "provenance": {"producer", "requester", "courant_version"},
     "target": {"name", "source", "case_json"},          # where the run was, relative to the root
     "spec": {"text", "sha256", "locked_sha256"} | null, # the file's own bytes, as text
     "manifest": [{"path", "bytes", "sha256"}, ...],     # VER-14, equal to verdict.run.inputs
     "verdict": <courant.verdict/2>}

The record names no absolute path and nothing machine-specific, so it is the same bytes on any
machine that has the same inputs and the same Courant (VER-9, INV-6). The specification travels
inside it as its exact text, so the record re-derives with nothing but the run's files.

The verdict inside is decided by `courant.verify`, never here (ARC-13): `build_record` takes a
verdict, and `check_record` asks `verify_path` for a fresh one.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from courant.verify.schema import canonical_json
from courant.verify.verdict2 import VerdictV2, as_v2

SCHEMA = "courant.record/1"


class RecordError(ValueError):
    """The record cannot be read as a `courant.record/1`."""


def build_record(verdict, *, target_name: str, case_json_name: str | None = None,
                 spec_text: str | None = None, spec_locked_sha256: str | None = None,
                 producer: str | None = None, requester: str | None = None) -> dict:
    """The record as data. `verdict` is either major; the record holds version 2."""
    v: VerdictV2 = as_v2(verdict)
    spec = None
    if spec_text is not None:
        spec = {"text": spec_text,
                "sha256": hashlib.sha256(spec_text.encode("utf-8")).hexdigest(),
                "locked_sha256": spec_locked_sha256.strip().lower() if spec_locked_sha256 else None}
    return {
        "schema": SCHEMA,
        "tool": {"name": v.tool.name, "version": v.tool.version},
        "provenance": {"producer": producer, "requester": requester,
                       "courant_version": v.tool.version},
        "target": {"name": target_name, "source": v.run.source, "case_json": case_json_name},
        "spec": spec,
        "manifest": [f.model_dump(mode="json") for f in v.run.inputs],
        "verdict": json.loads(v.to_json()),
    }


def write_record(verdict, out: str | Path, *, target_name: str, case_json_name: str | None = None,
                 spec_path: str | Path | None = None, spec_locked_sha256: str | None = None,
                 producer: str | None = None, requester: str | None = None) -> Path:
    """Write `out` (canonical JSON, VER-9)."""
    text = Path(spec_path).read_bytes().decode("utf-8") if spec_path is not None else None
    record = build_record(verdict, target_name=target_name, case_json_name=case_json_name,
                          spec_text=text, spec_locked_sha256=spec_locked_sha256,
                          producer=producer, requester=requester)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(canonical_json(record), encoding="utf-8")
    return out


def read_record(path: str | Path) -> dict:
    """The record at `path`, checked to be a `courant.record/1` of the shape REC-1 names."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecordError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        found = data.get("schema") if isinstance(data, dict) else None
        raise RecordError(f"not a {SCHEMA}: schema {found!r}")
    for key in ("tool", "provenance", "target", "manifest", "verdict"):
        if key not in data:
            raise RecordError(f"{SCHEMA} without {key!r}")
    return data


def manifest_problem(record: dict, root: str | Path) -> str | None:
    """The first manifest entry whose file under `root` is missing or changed, as a sentence."""
    for entry in record["manifest"]:
        p = Path(root) / entry["path"]
        if not p.is_file():
            return f"input missing: {entry['path']}"
        data = p.read_bytes()
        have = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or have != entry["sha256"]:
            return f"input changed: {entry['path']} (sha256 {have[:12]} != {entry['sha256'][:12]})"
    return None


def check_record(record_path: str | Path, root: str | Path) -> tuple[bool, str]:
    """Re-derive a record from its inputs under `root`. (identical, message).

    Every manifest file is re-hashed; then the run is verified again with the specification the
    record carries, the record is rebuilt from that verdict and the record's own provenance, and
    its bytes are compared with the file's. The first difference is named."""
    from courant.verify import InputError, verify_path

    try:
        record = read_record(record_path)
    except RecordError as exc:
        return False, str(exc)
    problem = manifest_problem(record, root)
    if problem:
        return False, problem
    root = Path(root)
    target = record["target"]
    spec = record.get("spec")
    locked = (spec or {}).get("locked_sha256")
    with tempfile.TemporaryDirectory(prefix="courant-record-") as tmp:
        spec_path = None
        if spec is not None:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_bytes(spec["text"].encode("utf-8"))
        try:
            if target["source"] == "case_dir":
                verdict = verify_path(root, spec=spec_path, spec_locked_sha256=locked)
            else:
                cj = root / target["case_json"] if target.get("case_json") else None
                verdict = verify_path(root / target["name"], cj, spec=spec_path,
                                      spec_locked_sha256=locked)
        except InputError as exc:
            return False, f"the run no longer reads: {exc}"
    prov = record["provenance"]
    fresh = canonical_json(build_record(
        verdict, target_name=target["name"], case_json_name=target.get("case_json"),
        spec_text=spec["text"] if spec else None, spec_locked_sha256=locked,
        producer=prov.get("producer"), requester=prov.get("requester")))
    stored = Path(record_path).read_text(encoding="utf-8")
    if fresh == stored:
        return True, (f"identical: {len(record['manifest'])} inputs re-hashed, record re-derived "
                      f"byte for byte")
    for i, (a, b) in enumerate(zip(fresh, stored, strict=False)):
        if a != b:
            note = ""
            if record["tool"]["version"] != verdict.tool.version:
                note = (f" (written by Courant {record['tool']['version']}, re-derived by "
                        f"{verdict.tool.version})")
            return False, f"record differs at byte {i}: ...{stored[max(0, i - 40):i + 40]!r}{note}"
    return False, "record differs in length"
