"""`courant.verdict/2`: accept, reject or insufficient, with its basis stated (TRD GATE-1..GATE-3,
ARC-16, §32; PRD 2.0 P2-GATE-1).

Version 2 is a re-statement of version 1, not a second judgement. The pipeline decides once
(`courant.verify.policy`, in version 1's words), and `to_v2` maps that decision through one
table, `FROM_V1`: TRUSTWORTHY to `accept`, NOT_TRUSTWORTHY to `reject`, NOT_ASSESSABLE to
`insufficient`. The rule that decided (`VER-5.x`), the grade, the reasons, the causes, the stats and
the convergence and conformance blocks are carried unchanged. `VerdictV2.to_v1()` rebuilds the
version 1 verdict byte for byte, so `--schema 1` (ARC-16) and the Courant Test's locked labels
(GATE-3) are the same decision in the old words.

What version 2 adds:

- `basis` (GATE-2): what the verdict was judged against. `validity` with no specification;
  `validity+conformance` with a specification whose SHA-256 was locked before the run and
  matches; `validity+unlocked-spec` with a specification given without a lock, or one that does
  not match its lock (VER-5.8 then makes the verdict `insufficient`); `validity+extracted-spec`
  with an extracted specification nobody confirmed (SPEC-7, `courant spec extract`). The basis
  states what the policy was already told: the pipeline passes the same `extracted` flag to
  `policy.apply_conformance`, where VER-5.12 turns a mismatch into `insufficient`, so the label
  never stands in for the rule.
- `run.fork` and `run.solver_version` (ADP-1). Always present: what `courant.forks` read from
  the log's banner or the header of a file the run wrote, `null` where it read nothing (and
  where only `system/controlDict`'s header, an input, names one).
- With no specification, the not-checked list ends with `sentences.NO_SPEC` (GATE-2).

Readers accept both majors (`load_verdict`). Every surface writes version 2 by default and
version 1 behind `--schema 1` for one minor release.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from courant.verify import sentences
from courant.verify.schema import (
    SCHEMA as SCHEMA_V1,
)
from courant.verify.schema import (
    ConformanceBlock,
    ConvergenceBlock,
    InputFile,
    ProbableCause,
    Reason,
    RunInfo,
    Source,
    Stats,
    ToolInfo,
    Verdict,
    VerdictValue,
    canonical_json,
)

SCHEMA_V2 = "courant.verdict/2"
#: The schema majors a writer can emit (ARC-16): 2 by default, 1 for one minor release.
SCHEMAS = {1: SCHEMA_V1, 2: SCHEMA_V2}

Decision = Literal["accept", "reject", "insufficient"]
Basis = Literal["validity", "validity+conformance", "validity+unlocked-spec",
                "validity+extracted-spec"]

#: GATE-1: the version 1 answer to the version 2 answer. A bijection; `TO_V1` is its inverse.
FROM_V1: dict[str, str] = {"TRUSTWORTHY": "accept", "NOT_TRUSTWORTHY": "reject",
                           "NOT_ASSESSABLE": "insufficient"}
TO_V1: dict[str, str] = {v2: v1 for v1, v2 in FROM_V1.items()}

#: The exit code of each answer (GATE-1): 0, 2 and 3; 4 is an unreadable input (GATE-10).
EXIT: dict[str, int] = {"accept": 0, "reject": 2, "insufficient": 3}


class RunInfoV2(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: Source
    solver: str | None
    #: ADP-1: the solver fork (`esi` or `foundation`) and version read from the log banner.
    fork: str | None = None
    solver_version: str | None = None
    inputs: tuple[InputFile, ...]


class VerdictV2(BaseModel):
    """The verdict, version 2 (§32). Field for field version 1, plus `basis`, `run.fork` and
    `run.solver_version`, with `verdict` in version 2's words."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_: Literal["courant.verdict/2"] = Field(default=SCHEMA_V2, alias="schema")
    tool: ToolInfo
    run: RunInfoV2
    verdict: Decision
    basis: Basis
    rule: str
    grade: Literal["A", "B", "C", "D"] | None
    reasons: tuple[Reason, ...]
    probable_causes: tuple[ProbableCause, ...]
    not_assessed: tuple[str, ...]
    next_action: str | None
    stats: Stats
    convergence: ConvergenceBlock | None = None
    conformance: ConformanceBlock | None = None

    @property
    def accepted(self) -> bool:
        return self.verdict == "accept"

    @property
    def exit_code(self) -> int:
        return EXIT[self.verdict]

    def to_json(self) -> str:
        """Canonical JSON (VER-9). An absent convergence or conformance block is left out, as in
        version 1."""
        data = self.model_dump(mode="json", by_alias=True)
        for key in ("convergence", "conformance"):
            if data.get(key) is None:
                data.pop(key, None)
        return canonical_json(data)

    def to_v1(self) -> Verdict:
        """The same decision as `courant.verdict/1`, byte for byte what version 1 wrote."""
        not_assessed = self.not_assessed
        if self.basis == "validity" and not_assessed and not_assessed[-1] == sentences.NO_SPEC:
            not_assessed = not_assessed[:-1]
        reasons = tuple(r for r in self.reasons if r.kind != "insufficient")
        return Verdict(
            tool=self.tool,
            run=RunInfo(id=self.run.id, source=self.run.source, solver=self.run.solver,
                        inputs=self.run.inputs),
            verdict=TO_V1[self.verdict], rule=self.rule, grade=self.grade,
            reasons=reasons, probable_causes=self.probable_causes,
            not_assessed=not_assessed, next_action=self.next_action, stats=self.stats,
            convergence=self.convergence, conformance=self.conformance)


AnyVerdict = Verdict | VerdictV2


def basis_of(conformance: ConformanceBlock | None, *, extracted: bool = False) -> str:
    """GATE-2. What the verdict was judged against, from its conformance block alone (so a
    version 1 verdict maps without its inputs). `extracted`: the specification records an
    extraction and no confirmation (SPEC-7). It labels; it does not decide: the policy was given
    the same flag (VER-5.12, `policy.apply_conformance`)."""
    if conformance is None:
        return "validity"
    if extracted:
        return "validity+extracted-spec"
    if conformance.locked_sha256 and conformance.locked_sha256 == conformance.spec_sha256:
        return "validity+conformance"
    return "validity+unlocked-spec"


def extracted_unconfirmed(spec: Mapping | None) -> bool:
    """Whether a specification (as JSON data) says it was extracted and nobody confirmed it
    (SPEC-7: `provenance.extracted_by` without `provenance.confirmed_by`)."""
    prov = (spec or {}).get("provenance") or {}
    return bool(prov.get("extracted_by")) and not prov.get("confirmed_by")


#: VER-5.16: the value `insufficient.truncated_log` carries, what was inferred said as inferred.
#: Courant 0.5.0 printed "return code 1" for the same log, a value no process had returned.
TRUNCATED_LOG_VALUE = "exit code unknown, inferred: the log does not end in End"


def insufficient_reason(rule: str, conformance: ConformanceBlock | None = None, *,
                        fork: str | None = None, solver_version: str | None = None) -> Reason:
    """GATE-4. The one reason an `insufficient` verdict carries: a code from the closed table
    in `sentences.INSUFFICIENT`, its fixed sentence, and (where the evidence is in the verdict)
    the hashes, the items or the version it rests on. A rule outside the table raises."""
    from courant import forks

    entry = sentences.insufficient_for(rule)
    threshold = value = None
    if entry.name == "truncated_log":
        value = TRUNCATED_LOG_VALUE
    elif conformance is not None and entry.name == "spec_hash_mismatch":
        threshold, value = conformance.locked_sha256, conformance.spec_sha256
    elif conformance is not None and entry.name == "spec_value_unreadable":
        value = ", ".join(i.name for i in conformance.items if i.conforms is None) or None
    elif entry.name == "unsupported_version":          # VER-5.10: what was read, against the set
        threshold, value = forks.measured_label(), forks.label(fork, solver_version)
    return Reason(code=entry.code, kind="insufficient", severity="major", summary=entry.sentence,
                  input=entry.input, method=entry.method, threshold=threshold, value=value)


def to_v2(v1: Verdict, *, extracted: bool = False, fork: str | None = None,
          solver_version: str | None = None) -> VerdictV2:
    """GATE-1. The version 1 verdict in version 2: the one mapping. An `insufficient` verdict
    gains the closed-vocabulary reason of GATE-4 first; `to_v1()` takes it off again."""
    basis = basis_of(v1.conformance, extracted=extracted)
    not_assessed = v1.not_assessed
    if basis == "validity":
        not_assessed = (*not_assessed, sentences.NO_SPEC)
    reasons = v1.reasons
    if v1.verdict == "NOT_ASSESSABLE":
        # After the conformance reasons, which VER-6 sorts first, and before everything else.
        lead = sum(1 for _ in itertools.takewhile(lambda r: r.kind == "conformance", reasons))
        reasons = (*reasons[:lead], insufficient_reason(v1.rule, v1.conformance, fork=fork,
                                                        solver_version=solver_version),
                   *reasons[lead:])
    return VerdictV2(
        tool=v1.tool,
        run=RunInfoV2(id=v1.run.id, source=v1.run.source, solver=v1.run.solver, fork=fork,
                      solver_version=solver_version, inputs=v1.run.inputs),
        verdict=FROM_V1[v1.verdict], basis=basis, rule=v1.rule, grade=v1.grade,
        reasons=reasons, probable_causes=v1.probable_causes, not_assessed=not_assessed,
        next_action=v1.next_action, stats=v1.stats, convergence=v1.convergence,
        conformance=v1.conformance)


def as_v1(v: AnyVerdict) -> Verdict:
    return v if isinstance(v, Verdict) else v.to_v1()


def as_v2(v: AnyVerdict) -> VerdictV2:
    return v if isinstance(v, VerdictV2) else to_v2(v)


def in_schema(v: AnyVerdict, schema: int) -> AnyVerdict:
    """The verdict as the writer of `schema` (1 or 2) emits it (ARC-16)."""
    if schema not in SCHEMAS:
        raise ValueError(f"unknown verdict schema {schema}: use 1 or 2")
    return as_v1(v) if schema == 1 else as_v2(v)


def load_verdict(data: str | bytes | Mapping) -> VerdictV2:
    """Read a verdict of either major (ARC-16). A version 1 verdict comes back as version 2
    through `to_v2`; its `to_v1()` gives the original back."""
    if isinstance(data, (str, bytes)):
        data = json.loads(data)
    schema = data.get("schema")
    if schema == SCHEMA_V2:
        return VerdictV2.model_validate(data)
    if schema == SCHEMA_V1:
        return to_v2(Verdict.model_validate(data))
    raise ValueError(f"not a Courant verdict: schema {schema!r}")


#: Version 1's model, under the name readers of both majors use.
VerdictV1 = Verdict

__all__ = ["EXIT", "FROM_V1", "SCHEMAS", "SCHEMA_V1", "SCHEMA_V2", "TO_V1", "AnyVerdict", "Basis",
           "Decision", "RunInfoV2", "VerdictV1", "VerdictV2", "VerdictValue", "as_v1", "as_v2",
           "basis_of", "extracted_unconfirmed", "in_schema", "load_verdict", "to_v2"]
