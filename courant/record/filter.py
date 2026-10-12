"""The reference dataset filter (TRD REC-2): the interface a surrogate-training pipeline needs.

A record is admitted only when ALL of these hold, and `admit` checks each:

- it is a `courant.record/1` whose verdict is `courant.verdict/2`, and its manifest equals its
  verdict's `run.inputs`;
- its verdict is `accept` on the basis `validity+conformance`;
- the specification it carries is the one the DATASET OWNER expects: its SHA-256, the hash it
  was locked at and the hash the verdict was judged against are all in `expected_spec_sha256`,
  an argument with no default. A record is evidence about a run against whatever specification it
  carries, and the agent that produced the run can write that specification itself
  (`courant spec init` drafts one that its own run satisfies); only a hash the owner fixed
  before the run says the run was asked the owner's question. A record whose specification is
  any other is refused, however well it re-derives;
- every file in its manifest is on disk under the root and hashes to what the manifest says;
- by default, the record re-derives byte for byte (`check_record`, with the specification the
  record carries, which the previous check has tied to the owner's hash). `rederive=False`
  skips this last step, and the admission then says so; the hashes are then checked but the
  verdict is the record's own claim.

What is not enforced: that the owner's specification describes the problem they meant (that is
their review), and the provenance fields (`producer`, `requester`), which are claims.

The filter decides nothing about physics: it reads what the gate decided (ARC-13).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from courant.record.record import RecordError, check_record, manifest_problem, read_record


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str
    path: str = ""


def _expected(value: str | Iterable[str]) -> frozenset[str]:
    values = [value] if isinstance(value, str) else list(value)
    return frozenset(v.strip().lower() for v in values if v and v.strip())


def admit(record_path: str | Path, root: str | Path, *, expected_spec_sha256: str | Iterable[str],
          rederive: bool = True) -> Admission:
    """Whether the record at `record_path` may enter a dataset whose files are under `root` and
    whose owner fixed the specification hash(es) `expected_spec_sha256` before the runs."""
    expected = _expected(expected_spec_sha256)
    name = str(record_path)
    try:
        record = read_record(record_path)
    except RecordError as exc:
        return Admission(False, str(exc), name)
    verdict = record["verdict"]
    if verdict.get("schema") != "courant.verdict/2":
        return Admission(False, f"the verdict is {verdict.get('schema')!r}, not courant.verdict/2",
                         name)
    if verdict.get("verdict") != "accept":
        return Admission(False, f"the verdict is {verdict.get('verdict')}, not accept", name)
    if verdict.get("basis") != "validity+conformance":
        return Admission(False, f"the basis is {verdict.get('basis')}, not validity+conformance",
                         name)
    if record["manifest"] != verdict.get("run", {}).get("inputs"):
        return Admission(False, "the record's manifest is not its verdict's inputs", name)
    spec = record.get("spec")
    conformance = verdict.get("conformance") or {}
    if (spec is None or not spec.get("locked_sha256")
            or spec["locked_sha256"] != conformance.get("spec_sha256")
            or spec["sha256"] != conformance.get("spec_sha256")):
        return Admission(False, "the specification is not the locked one the verdict was judged on",
                         name)
    if {spec["sha256"], spec["locked_sha256"], conformance.get("spec_sha256")} - expected:
        return Admission(False, f"the specification (sha256 {str(spec['sha256'])[:12]}) is not one "
                                "the dataset owner expects", name)
    problem = manifest_problem(record, root)
    if problem:
        return Admission(False, problem, name)
    if not rederive:
        return Admission(True, "accept on validity+conformance; manifest hashes match; "
                               "not re-derived", name)
    ok, message = check_record(record_path, root)
    return Admission(ok, ("accept on validity+conformance; " + message) if ok else message, name)


def filter_records(records: Iterable[str | Path], root: str | Path, *,
                   expected_spec_sha256: str | Iterable[str], rederive: bool = True) -> tuple[list[Path], list[Admission]]:
    """(the admitted record paths, the refusals with their reasons)."""
    admitted, refused = [], []
    for r in records:
        a = admit(r, root, expected_spec_sha256=expected_spec_sha256, rederive=rederive)
        if a.admitted:
            admitted.append(Path(r))
        else:
            refused.append(a)
    return admitted, refused
