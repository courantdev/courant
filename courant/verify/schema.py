"""The verifier's contracts (TRD VER-1, VER-2, VER-4, VER-9).

Every edge of the verification graph carries one of these frozen models. The
verdict is serialised as canonical JSON, so identical inputs give identical
bytes on the same platform (INV-6).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

from courant.case.reads import refuse_outside

SCHEMA = "courant.verdict/1"
VerdictValue = Literal["TRUSTWORTHY", "NOT_TRUSTWORTHY", "NOT_ASSESSABLE"]
Source = Literal["case_dir", "log", "corpus"]


class InputError(ValueError):
    """The inputs could not be read. Exit code 4. (Defined beside the manifest, which raises it;
    `courant.verify.ingest` re-exports it, where it has always been imported from.)"""


class InputFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str          # relative to the case, or the log's file name
    bytes: int
    sha256: str

    @classmethod
    def of(cls, path: Path, name: str) -> InputFile:
        refuse_outside(path)            # MCP-3: hashed only inside the roots
        data = path.read_bytes()
        return cls(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    @classmethod
    def of_text(cls, text: str, name: str) -> InputFile:
        data = text.encode("utf-8", errors="replace")
        return cls(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def manifest(root: str | Path, paths: Iterable[str]) -> tuple[InputFile, ...]:
    """The manifest entries for `paths`, relative to `root`: hashed, and sorted by path (VER-14).

    `paths` are the files the readers noted (`courant.case.reads`), each after a read that
    succeeded. Every case-directory manifest (the verdict's, preflight's and the convergence
    document's) is assembled here, so every entry is hashed the same way, after the reading is
    done. A noted file that cannot be hashed now (it vanished, or lost its permissions, after it
    was read) is an `InputError` that names it, never a traceback: what decided the verdict can
    no longer be re-checked, and a pack keyed on a guess would be worse than none.
    """
    base = Path(root)
    entries = []
    for rel in sorted(set(paths)):
        try:
            entries.append(InputFile.of(base / rel, rel))
        except OSError as exc:
            why = exc.strerror or type(exc).__name__
            raise InputError(f"{rel} was read for the verdict and could not be hashed for its "
                             f"manifest: {why}") from exc
    return tuple(entries)


def merge(a: Iterable[InputFile], b: Iterable[InputFile]) -> tuple[InputFile, ...]:
    """The union of two manifests by path, sorted by path. Where both list a path, `a`'s entry
    is kept: it was hashed first."""
    by_path = {f.path: f for f in b}
    by_path.update((f.path, f) for f in a)
    return tuple(sorted(by_path.values(), key=lambda f: f.path))


class RunInputs(BaseModel):
    """Everything the verifier reads, gathered before anything is parsed."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    log_text: str
    log_name: str
    config_view: dict | None = None
    mesh_view: dict | None = None
    exit_code: int | None = None
    inputs: tuple[InputFile, ...] = ()
    source: Source = "log"
    solver_hint: str | None = None
    #: Absolute path, used only to read solution fields. Never serialised.
    case_dir: str | None = Field(default=None, exclude=True)
    #: The task specification, when one was given (`courant.spec/1`, as JSON data), its file
    #: hash, and the hash it was locked at before the run. See `courant.spec`.
    spec: dict | None = Field(default=None, exclude=True)
    spec_sha256: str | None = Field(default=None, exclude=True)
    spec_locked_sha256: str | None = Field(default=None, exclude=True)
    #: "next" also runs the checks behind `--checks next` (GATE-13); the verdict says so.
    checks: Literal["default", "next"] = Field(default="default", exclude=True)


class Reason(BaseModel):
    """One cited observation behind a verdict (INV-2)."""

    model_config = ConfigDict(frozen=True)

    code: str
    #: `insufficient` is version 2's alone (GATE-4): the one closed-vocabulary reason of an
    #: `insufficient` verdict. Version 1 never carries one.
    kind: Literal["event", "finding", "check", "conformance", "insufficient"]
    severity: Literal["critical", "major", "warn", "minor"]
    summary: str
    input: str
    method: str
    threshold: str | None = None
    value: str | None = None
    log_offset: int | None = None

    @field_validator("input", "method", "summary", "code")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("a reason must say what it read, how, and what it found")
        return v


class ProbableCause(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    code: str
    name: str


class Stats(BaseModel):
    model_config = ConfigDict(frozen=True)

    events: int
    residual_steps: int
    crashes: int
    divergences: int
    findings_by_severity: dict[str, int]


class RunInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source: Source
    solver: str | None
    inputs: tuple[InputFile, ...]


class ToolInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "courant"
    version: str
    #: "next" when the checks behind `--checks next` ran (GATE-13, `courant.verify.checks`).
    #: Absent otherwise, so a verdict on the default checks keeps its bytes.
    checks: Literal["next"] | None = None

    @model_serializer(mode="wrap")
    def _drop_default_checks(self, handler):
        data = handler(self)
        if data.get("checks") is None:
            data.pop("checks", None)
        return data


class ConvergenceBlock(BaseModel):
    """Where the monitored answer stopped changing (TRD CONV-12). Beside the verdict, not in it:
    VER-5.4b is off until the founder decides (H-CONV-POLICY)."""

    model_config = ConfigDict(frozen=True)

    status: Literal["PASS", "WARN", "FAIL", "NOT_ASSESSABLE"]
    regime: str | None
    enters_verdict: bool
    converged_at_step: int | None
    binding_monitor: str | None
    change_after_stop: str | None
    change_counts: str | None
    change_fraction: str | None
    wasted_steps: int | None
    steps_total: int | None
    wasted_fraction: str | None
    wasted_wall_clock_s: str | None
    wasted_core_hours: str | None
    monitors_assessed: int
    monitors_found: int
    messages: tuple[str, ...] = ()


class ConformanceItem(BaseModel):
    """One value the task specification names, against the case (PRD 2.0 P2-SPEC-2)."""

    model_config = ConfigDict(frozen=True)

    name: str
    file: str                      # the file actually read (may be under 0.orig/)
    location: str
    specified: str | None
    found: str | None
    conforms: bool | None          # None: the value could not be read from the case
    tolerance: str
    modelling_choice: bool = False
    note: str | None = None
    #: How the value was read when it was not a single written value, e.g. the most common
    #: face value of a per-face list the run rewrote.
    basis: str | None = None


class ConformanceBlock(BaseModel):
    """Whether the case is the problem that was asked. Present only when a spec was given."""

    model_config = ConfigDict(frozen=True)

    spec_schema: str
    spec_sha256: str
    #: The hash the spec was locked at before the run, when one was given (P2-SPEC-3).
    locked_sha256: str | None = None
    task: str | None = None
    items: tuple[ConformanceItem, ...]
    checked: int
    nonconformant: int
    unreadable: int
    #: SPEC-9: why the specification is impossible in itself; its items were then not compared
    #: (VER-5.11). Written only when there is a reason, so every other block keeps its bytes.
    invalid: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _drop_empty_invalid(self, handler):
        data = handler(self)
        if not data.get("invalid"):
            data.pop("invalid", None)
        return data


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_: str = Field(default=SCHEMA, alias="schema")
    tool: ToolInfo
    run: RunInfo
    verdict: VerdictValue
    rule: str
    grade: Literal["A", "B", "C", "D"] | None
    reasons: tuple[Reason, ...]
    probable_causes: tuple[ProbableCause, ...]
    not_assessed: tuple[str, ...]
    next_action: str | None
    stats: Stats
    #: Present only when a case directory was verified, so a log-only verdict keeps its v0
    #: bytes (PRD P-NFR-6).
    convergence: ConvergenceBlock | None = None
    #: Present only when a task specification was given (PRD 2.0 P2-SPEC-2), so a verdict
    #: without one keeps its v0 bytes.
    conformance: ConformanceBlock | None = None

    def to_json(self) -> str:
        data = self.model_dump(mode="json", by_alias=True)
        for key in ("convergence", "conformance"):
            if data.get(key) is None:
                data.pop(key, None)
        return canonical_json(data)


#: SB-11: the closed list of outcomes. The last five are the re-solve's own classes (SB-18,
#: SB-19), named here so the gate never imports the re-solver, which is not part of the gate
#: (INV-10); a test holds the two lists equal.
OUTCOMES = ("valid_conformant", "invalid", "nonconformant", "insufficient", "did_not_execute",
            "forbidden_code", "resolve_refused_file", "resolve_include_outside",
            "resolve_budget_exceeded", "resolve_failed", "resolve_mesh_below_floor")
RESOLVE_OUTCOMES = OUTCOMES[5:]
ExecutedSource = Literal["resolve", "agent_log"]


#: The outcomes that are a judgement of a run the re-solve executed.
JUDGED = ("valid_conformant", "invalid", "nonconformant", "insufficient")


def outcome_flags(outcome: str, *, executed: bool | None,
                  reported_success: bool | None) -> dict[str, bool]:
    """`passed`, `silent_error` and `unverified_success` from an outcome (SB-4, SB-11): the one
    rule, for `outcome_of` and for every reader that re-derives them from a stored record."""
    claimed = reported_success is True and executed is True
    return {"passed": outcome == "valid_conformant",
            "silent_error": claimed and outcome in ("invalid", "nonconformant"),
            "unverified_success": claimed and outcome == "insufficient"}


class Outcome(BaseModel):
    """One run's outcome (SB-11), from `outcome_of` and nowhere else."""

    model_config = ConfigDict(frozen=True)

    outcome: str
    #: True only on `valid_conformant` (with a specification locked before the run).
    passed: bool
    #: Executed, reported a success by its producer, and rejected by the gate (SB-4).
    silent_error: bool
    #: The same, with `insufficient` (SB-11). Reported beside silent errors, never merged.
    unverified_success: bool
    executed: bool | None
    executed_source: ExecutedSource
    reported_success: bool | None
    verdict: str | None
    rule: str | None
    why: str


def outcome_of(verdict, *, executed: bool | None, executed_source: ExecutedSource = "resolve",
               reported_success: bool | None = None, resolve_class: str | None = None,
               verify_error: str | None = None) -> Outcome:
    """SB-11. The one function that turns a verdict into an outcome; the MCP tool, the scorer,
    the environment, evaluations and the SilentBench agent runner all call it (ARC-13), and no
    other module names an outcome or computes its flags (`tests/unit/test_outcome_lint.py`).
    The CI gate's pass or fail is a policy on the verdict (`courant.verify.outcome.check_fails`),
    not a second outcome.

    In order: a re-solve class is the outcome (never a silent error); a gate that raised on the
    re-solved copy is `resolve_failed`, so a run the grader could not judge never passes; with
    no verdict, a run known not to have executed is `did_not_execute` and any other
    `resolve_failed`; a run that did not execute is `did_not_execute`, whatever the verdict; `reject` is
    `nonconformant` when VER-5.7 decided it and `invalid` otherwise; `insufficient` is itself;
    `accept` is `valid_conformant` only when the verdict was judged against a specification
    whose SHA-256 was locked before the run and matches (basis `validity+conformance`), and
    `insufficient` otherwise. Execution alone scores nothing.

    `executed` comes from the re-solve's process record when scoring (`executed_source:
    resolve`, SB-18); only the advisory MCP tool passes `agent_log` (GATE-5). `verdict` is a
    `courant.verdict/1` or `/2` object, or None when there is none.
    """
    if executed_source not in ("resolve", "agent_log"):
        raise ValueError(f"executed_source is 'resolve' or 'agent_log', not {executed_source!r}")

    def make(outcome: str, why: str, *, v2=None) -> Outcome:
        return Outcome(
            outcome=outcome,
            **outcome_flags(outcome, executed=executed, reported_success=reported_success),
            executed=executed, executed_source=executed_source,
            reported_success=reported_success,
            verdict=v2.verdict if v2 is not None else None,
            rule=v2.rule if v2 is not None else None, why=why)

    if resolve_class is not None:
        if resolve_class not in RESOLVE_OUTCOMES:
            raise ValueError(f"{resolve_class!r} is not a re-solve class {RESOLVE_OUTCOMES}")
        return make(resolve_class, "the re-solve ended in its own class (SB-18, SB-19)")
    if verify_error is not None:
        return make("resolve_failed", f"the gate could not judge the re-solved case: "
                                      f"{verify_error}")
    if verdict is None:
        if executed is False:
            return make("did_not_execute", "the solver left no run to judge")
        return make("resolve_failed", "no verdict was taken")
    # Imported here: verdict2 imports this module.
    from courant.verify.verdict2 import as_v2

    v2 = as_v2(verdict)
    if executed is not True:
        return make("did_not_execute", "the solver did not complete", v2=v2)
    if v2.verdict == "reject":
        if v2.rule == "VER-5.7":
            return make("nonconformant", "the run is not the problem that was asked", v2=v2)
        return make("invalid", f"the gate rejected the run ({v2.rule})", v2=v2)
    if v2.verdict == "insufficient":
        return make("insufficient", f"the gate could not decide ({v2.rule})", v2=v2)
    if v2.basis != "validity+conformance":
        return make("insufficient", f"accepted on basis {v2.basis}, not against a specification "
                                    f"locked before the run", v2=v2)
    return make("valid_conformant", "accepted against the locked specification", v2=v2)


def executed_in_log(log_text: str) -> bool:
    """Whether a solver log shows a run that executed to its end: it prints `End` on a line of
    its own and carries no fatal error. The one reading of an agent's own log; for scoring, the
    re-solve's process record replaces it (SB-18). Only the advisory paths pass its answer to
    `outcome_of` (`executed_source: agent_log`).

    This is not `courant.watcher.exitcode`'s reading of End, which the verdict uses: a line
    `End` anywhere counts here, so a utility's End ahead of a cut solver is `executed` while
    `--checks next` (VER-3a) calls the same log `truncated_log`. Routing it through
    `exitcode.verdict_exit_code` is graph node T-EXECUTED-ONE (TRD VER-3), measured first,
    since it feeds SilentBench's scoring."""
    import re

    return bool(re.search(r"^End\s*$", log_text, re.MULTILINE)) and "FOAM FATAL" not in log_text


def canonical_json(obj) -> str:
    """Sorted keys, compact separators, UTF-8, no NaN, one trailing newline (VER-9)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False) + "\n"


def num(value: float | None) -> str | None:
    """A number as the shortest string that round-trips. Never fixed precision."""
    if value is None:
        return None
    if isinstance(value, float):
        # float() first: numpy 2 prints np.float64(0.1) from repr().
        return repr(float(value))
    return str(int(value)) if hasattr(value, "__index__") else str(value)
