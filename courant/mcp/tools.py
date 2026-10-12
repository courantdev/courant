"""The MCP tools as plain functions (TRD MCP-2), testable without a transport.

Read-only by construction (MCP-4): no tool writes a file, runs a process or
opens a socket. Paths are confined to the roots the user allows (MCP-3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class OutsideRoot(PermissionError):
    pass


class Confinement:
    """Only paths inside one of these roots may be read."""

    def __init__(self, roots: list[str | Path]):
        self.roots = [Path(r).resolve() for r in (roots or [Path.cwd()])]

    def check(self, path: str | Path) -> Path:
        p = Path(path).expanduser().resolve()
        if not any(p == r or p.is_relative_to(r) for r in self.roots):
            raise OutsideRoot(f"{p} is outside the allowed roots: "
                              + ", ".join(str(r) for r in self.roots))
        return p


def _schema_error(schema: int) -> dict | None:
    from courant.verify.verdict2 import SCHEMAS

    if schema not in SCHEMAS:
        return {"error": f"schema must be 1 or 2, not {schema!r}: 2 is courant.verdict/2 (the "
                         "default), 1 is courant.verdict/1, kept for one minor release"}
    return None


def verify_run(path: str, case_json: str | None = None, spec: str | None = None,
               spec_sha256: str | None = None, schema: int = 2, *, confine: Confinement) -> dict:
    """Verify a completed simulation (a case directory or a solver log).

    Returns the verdict JSON (schema courant.verdict/2 by default, courant.verdict/1 with
    `schema=1`): accept, reject or insufficient, the basis it was judged on, the cited reasons,
    what was not checked, and a suggested next action that is never applied. With `spec` (a
    courant.spec/1 file), every value the task specification names is also checked against the
    case, and a case that set up a different problem is rejected however well it ran.
    """
    from courant.verify import InputError, verify_path
    from courant.verify.verdict2 import in_schema

    if (bad := _schema_error(schema)) is not None:
        return bad
    target = confine.check(path)
    sidecar = confine.check(case_json) if case_json else None
    spec_path = confine.check(spec) if spec else None
    try:
        verdict = verify_path(target, sidecar, spec=spec_path, spec_locked_sha256=spec_sha256)
    except InputError as exc:
        return {"error": str(exc)}
    return json.loads(in_schema(verdict, schema).to_json())


#: GATE-5 / SB-18: said with every score, because the score is advice and not a certification.
ADVISORY = ("Advisory: this judges the run as you left it, from your own log. It is not a "
            "certification; a score for the record is taken on a re-solve of your final case.")


#: The fields of SB-11's outcome the advisory tool returns (GATE-5).
_SCORED = {"outcome", "executed", "executed_source", "reported_success", "silent_error",
           "unverified_success"}


def score_agent_output(case: str, spec: str, spec_sha256: str | None = None,
                       reported_success: bool | None = None, schema: int = 2, *,
                       confine: Confinement) -> dict:
    """Judge an agent's finished case against the task it was given (GATE-5). Advisory.

    `case` is the case directory the agent left, its own log included; `spec` is the task
    specification (courant.spec/1) and `spec_sha256` the hash it was locked at; `reported_success`
    is whether the agent is about to tell its user the run worked. Returns the verdict document
    (courant.verdict/2 by default), SB-11's `outcome` (also as `task_class`), and the flags
    `silent_error` (the run executed, you reported success, and the gate rejects it) and
    `unverified_success` (the same, with insufficient). Read-only.
    """
    from courant.verify import InputError, from_path, verify_inputs, with_spec
    from courant.verify.schema import executed_in_log, outcome_of
    from courant.verify.verdict2 import in_schema

    if (bad := _schema_error(schema)) is not None:
        return bad
    target = confine.check(case)
    spec_path = confine.check(spec)
    try:
        inputs = with_spec(from_path(target), spec_path, spec_sha256)
        verdict = verify_inputs(inputs)
    except InputError as exc:
        return {"error": str(exc)}
    scored = outcome_of(verdict, executed=executed_in_log(inputs.log_text),
                        executed_source="agent_log", reported_success=reported_success
                        ).model_dump(include=_SCORED)
    return {"advisory": True, "note": ADVISORY,
            "verdict": json.loads(in_schema(verdict, schema).to_json()),
            "task_class": scored["outcome"], **scored}


def check_convergence(path: str, transient_average: bool = False, *, confine: Confinement) -> dict:
    """Where a run's monitored answer stopped changing, and what the rest of the run cost.

    `path` is a case directory with function-object output under postProcessing/, or a log.
    Read-only. Returns the status (PASS, WARN, FAIL or NOT_ASSESSABLE), the stopping step,
    the change after it, and the iterations, wall clock and core-hours spent after it.
    """
    from courant.converge import Options, convergence_path, summary

    target = confine.check(path)
    conv = convergence_path(target, None, Options(transient_average=transient_average))
    out = summary(conv)
    out["messages"] = [f.message for f in conv.findings if f.severity in ("WARN", "FAIL") and f.message]
    return out


def explain_code(code: str) -> dict:
    """What a failure-mode code means, and the usual correction (a suggestion only)."""
    from courant.diagnosis.kb_data import build_default_kb
    from courant.taxonomy import TAXONOMY

    entry = TAXONOMY.get(code)
    if entry is None:
        return {"error": f"unknown code {code!r}", "known": sorted(TAXONOMY)}
    kb = build_default_kb().get(code)
    fixes = [f.description for f in kb.canonical_fixes] if kb else []
    return {"code": code, "name": entry.title, "description": entry.summary,
            "typical_corrections": fixes,
            "certified_on_real_output": code in certified_codes()}


def _parse_certified_codes(text: str) -> set[str]:
    """Tier 1 of docs/COVERAGE.md: named on output a real solver produced."""
    m = re.search(r"## 1\..*?(?=\n## 2\.)", text, re.S)
    return set(re.findall(r"`([A-Z]+\.[A-Z_]+)`", m.group(0))) if m else set()


def certified_codes() -> set[str]:
    """The packaged copy (PKG-5), so an installed server answers outside the repository."""
    from courant.data import certified_codes as packaged

    return set(packaged())


def list_failure_modes() -> list[dict]:
    """Every failure mode Courant knows, and whether it is certified on real output."""
    from courant.taxonomy import TAXONOMY

    certified = certified_codes()
    return [{"code": code, "name": entry.title, "certified_on_real_output": code in certified}
            for code, entry in sorted(TAXONOMY.items())]


def verifier_certification() -> dict:
    """The verifier's own measured error rates, from the packaged results only (PKG-5)."""
    from courant.data import certification

    data = certification()
    if data is None:
        return {"certified": False, "note": "no certification results in this build"}
    card = data["scorecard"]["all"]
    return {"certified": True, "tag": data.get("tag"), "date": data.get("date"),
            "policy": data.get("policy"), "corpus": data.get("corpus"),
            "false_accepts": {k: card["far"][k] for k in ("n", "of", "lo", "hi")},
            "false_alarms": {k: card["frr"][k] for k in ("n", "of", "lo", "hi")},
            "abstentions": {k: card["abstain"][k] for k in ("n", "of")},
            "hypotheses": data.get("hypotheses", {})}
