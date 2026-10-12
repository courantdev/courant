"""The verification graph, end to end (TRD §2).

ingest → parse → facts → checks → diagnose → policy → verdict.

Every node here is deterministic, and nothing in this module calls a model
(INV-1). The Courant Test and the verifier's certification call
`verify_inputs`, the same function users call, so there is no second path
that could drift from the product (ARC-4).
"""

from __future__ import annotations

import re
from functools import lru_cache

from courant.case.reads import Reads, recording
from courant.events import EventKind
from courant.events.diverged import divergences, is_divergence
from courant.verify import checks, policy, sentences
from courant.verify.schema import (
    InputFile,
    ProbableCause,
    Reason,
    RunInfo,
    RunInputs,
    Stats,
    ToolInfo,
    Verdict,
    manifest,
    merge,
    num,
)
from courant.verify.verdict2 import VerdictV2, extracted_unconfirmed, to_v2

_EXEC = re.compile(r"^Exec\s*:\s*(\S+)", re.MULTILINE)

#: What each checker reads when it is not the log (INV-2: a reason names its input).
_CHECKER_INPUT = {
    "mesh": "log.checkMesh (mesh quality metrics)",
    "scheme": "system/fvSchemes",
    "bc": "0/ boundary conditions",
    "physics": "constant/transportProperties or constant/thermophysicalProperties",
    "case": "case dictionaries",
}
_ID_INPUT = {"solver.": "system/fvSolution"}


def _finding_input(f, log_name: str) -> str:
    base = next((v for k, v in _ID_INPUT.items() if f.id.startswith(k)), None)
    base = base or _CHECKER_INPUT.get(f.checker) or log_name
    if f.location and f.location not in base:
        where = f"t = {f.location}" if f.location.replace(".", "", 1).isdigit() else f.location
        return f"{base} ({where})"
    return base


@lru_cache(maxsize=1)
def _kb():
    from courant.diagnosis.kb_data import build_default_kb

    return build_default_kb()


def _validated(model, view):
    if not view:
        return None
    try:
        return model.model_validate(view)
    except Exception:                                   # noqa: BLE001
        return None


def _fields(case_dir: str | None):
    if not case_dir:
        return None
    try:
        from courant.verification._fluid_reader import read_fields

        data = read_fields(case_dir)
    except Exception:                                   # noqa: BLE001
        return None
    return data if getattr(data, "internal", None) else None


def _solver(inputs: RunInputs, config) -> str | None:
    m = _EXEC.search(inputs.log_text)
    if m:
        return m.group(1)
    if inputs.solver_hint:
        return inputs.solver_hint
    return getattr(config, "solver", None) if config is not None else None


def verify_inputs(inputs: RunInputs) -> VerdictV2:
    """The verdict on `inputs`, as `courant.verdict/2` (GATE-1..GATE-3). `.to_v1()` gives the
    same decision as `courant.verdict/1`, byte for byte (ARC-16).

    The policy decides once, in version 1's words (`verify_inputs_v1`), and `to_v2` maps it.
    For a case directory, the graph runs inside one recording, so every case file it reads
    (field files, monitors, every location conformance compares) is noted by the reader that
    read it and joins the ingest's manifest (VER-14). For a log, the manifest is the ingest's.
    """
    return to_v2(verify_inputs_v1(inputs), extracted=extracted_unconfirmed(inputs.spec))


def verify_inputs_v1(inputs: RunInputs) -> Verdict:
    """The decision as `courant.verdict/1`. Callers want `verify_inputs`; this is its first half."""
    if not inputs.case_dir:
        return _verify(inputs, None)
    with recording(inputs.case_dir) as reads:
        return _verify(inputs, reads)


def _manifest(inputs: RunInputs, reads: Reads | None) -> tuple[InputFile, ...]:
    if reads is None:
        return inputs.inputs
    return merge(inputs.inputs, manifest(inputs.case_dir, reads.paths()))


def _verify(inputs: RunInputs, reads: Reads | None) -> Verdict:
    from courant import __version__
    from courant.watcher.exitcode import infer_exit_code

    not_assessed: list[str] = [sentences.DISCRETISATION, sentences.VALIDATION]
    is_recognised = policy.recognised(inputs.log_text)
    exit_code = inputs.exit_code if inputs.exit_code is not None else infer_exit_code(inputs.log_text)

    from courant.case.config import CaseConfig
    from courant.case.mesh import MeshQuality

    config = _validated(CaseConfig, inputs.config_view)
    mesh = _validated(MeshQuality, inputs.mesh_view)
    fields = _fields(inputs.case_dir)
    if config is None:
        not_assessed.append(sentences.NO_CONFIG)
    if mesh is None:
        not_assessed.append(sentences.NO_MESH)
    if fields is None:
        not_assessed.append(sentences.NO_FIELDS)

    events, findings, trust, ranked = [], [], None, []
    if is_recognised:
        from courant.watcher.parser import LogParser

        events = list(LogParser(run_id=inputs.run_id).parse_text(
            inputs.log_text, return_code=exit_code))
    if is_recognised and events:
        from courant.diagnosis.deterministic import DeterministicDiagnoser
        from courant.diagnosis.bundle import assemble_bundle
        from courant.diagnosis.checkers import CheckContext, run_all_checkers
        from courant.verification.checks import VerifyContext
        from courant.verification.suite import verify

        finding_set = run_all_checkers(CheckContext(
            run_id=inputs.run_id, events=events, config=config, mesh=mesh,
            log_text=inputs.log_text))
        finding_set = checks.only_enabled(finding_set, inputs.checks)
        findings = list(finding_set)
        trust = verify(VerifyContext(run_id=inputs.run_id, events=events,
                                     config=config, fields=fields))
        bundle = assemble_bundle(run_id=inputs.run_id, events=events, findings=finding_set,
                                 kb=_kb(), config=config, mesh=mesh, log_text=inputs.log_text)
        ranked = DeterministicDiagnoser(_kb()).diagnose(bundle).ranked_codes()[:3]

    convergence = None
    if inputs.case_dir and is_recognised and events:
        convergence = _convergence(inputs, events, config)
        if convergence.status == "NOT_ASSESSABLE" and not convergence.monitors_found:
            not_assessed.append(sentences.NO_MONITORS)

    verdict, rule = policy.decide(is_recognised, events, findings, trust)
    if rule == "VER-5.1":
        not_assessed.insert(0, sentences.NO_OUTPUT)
    elif rule == "VER-5.5":
        not_assessed.insert(0, sentences.NO_RESIDUALS)
    elif rule == "VER-5.15":
        not_assessed.insert(0, sentences.NO_TIME_MARKERS)

    conformance = None
    if inputs.spec is not None:
        from courant.spec import check_conformance

        conformance = check_conformance(inputs.spec, inputs.case_dir,
                                        spec_sha256=inputs.spec_sha256 or "",
                                        locked_sha256=inputs.spec_locked_sha256)
        # SPEC-7: whether the specification binds is an input to the policy (VER-5.12), read
        # from the specification's own hashed provenance, not a label added afterwards.
        extracted = extracted_unconfirmed(inputs.spec)
        verdict, rule = policy.apply_conformance(verdict, rule, conformance,
                                                 extracted=extracted)
        if extracted:
            not_assessed.append(sentences.SPEC_EXTRACTED)
        if policy.spec_void(conformance):
            not_assessed.append(sentences.SPEC_VOID)
        elif policy.spec_invalid(conformance):
            not_assessed.append(sentences.SPEC_INVALID)
        else:
            not_assessed.append(sentences.SPEC_SCOPE)
            not_assessed.extend(f"Specification item '{i.name}' was not checked: {i.note}"
                                for i in conformance.items if i.conforms is None)
    if trust is not None:
        not_assessed.extend(f"{r.check_name}: {r.detail}" for r in trust.results
                            if r.skipped and r.detail)
    not_assessed.extend(f"{f.title}: {f.detail}" if f.detail else f.title
                        for f in findings if f.id in policy.READER_LIMITATIONS)

    extracted = conformance is not None and extracted_unconfirmed(inputs.spec)
    reasons = policy.order_reasons(_reasons(inputs, events, findings, trust)
                                   + _conformance_reasons(conformance, extracted=extracted))
    causes = tuple(_cause(i + 1, code) for i, code in enumerate(ranked))
    grade = trust.grade.value if (trust is not None and rule not in ("VER-5.1", "VER-5.5", "VER-5.15")) else None

    return Verdict(
        tool=ToolInfo(version=__version__, checks="next" if inputs.checks == "next" else None),
        run=RunInfo(id=inputs.run_id, source=inputs.source,
                    solver=_solver(inputs, config), inputs=_manifest(inputs, reads)),
        verdict=verdict, rule=rule, grade=grade,
        reasons=reasons, probable_causes=causes,
        not_assessed=_dedupe(not_assessed),
        next_action=(_conformance_action(conformance, extracted=extracted)
                     or _next_action(verdict, causes, events, inputs.log_name)),
        stats=_stats(events, findings),
        convergence=convergence,
        conformance=conformance,
    )


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "item"


def _conformance_reasons(block, *, extracted: bool = False) -> list[Reason]:
    if block is None:
        return []
    if policy.spec_void(block):
        return [Reason(
            code="conformance.spec_changed", kind="conformance", severity="critical",
            summary="The task specification is not the one locked before the run",
            input="task specification", method="SHA-256 of the specification file",
            threshold=block.locked_sha256, value=block.spec_sha256)]
    if policy.spec_invalid(block):
        return [Reason(
            code="conformance.spec_invalid", kind="conformance", severity="critical",
            summary=f"The task specification is impossible in itself: {why}",
            input="task specification", method="SPEC-9 (courant.spec.impossible)")
            for why in block.invalid]
    out = []
    for item in block.items:
        if item.conforms is not False:
            continue
        what = item.note or f"{item.found} in the case, {item.specified} specified"
        whose = "the unconfirmed extracted specification" if extracted else \
            "the task specification"
        out.append(Reason(
            code=f"conformance.{_slug(item.name)}", kind="conformance",
            severity="major" if extracted else "critical",
            summary=f"{item.name} does not match {whose}: {what}",
            input=f"{item.file} ({item.location})",
            method="deck-versus-spec conformance (courant.spec/1)",
            threshold=f"{item.specified} ({item.tolerance})", value=item.found))
    return out


def _conformance_action(block, *, extracted: bool = False) -> str | None:
    if block is None:
        return None
    if policy.spec_void(block):
        return ("Verify against the specification that was locked before the run, or lock this "
                "one and re-run.")
    if policy.spec_invalid(block):
        return ("Correct the task specification (no case can meet it as written), lock it again "
                "and re-verify.")
    wrong = [i for i in block.items if i.conforms is False]
    if wrong and extracted:
        return ("The case differs from a specification a model extracted and nobody confirmed: "
                "review it, confirm it (courant spec confirm --by NAME) and lock its new hash, "
                "then re-verify.")
    if wrong:
        first = wrong[0]
        more = f" (and {len(wrong) - 1} more)" if len(wrong) > 1 else ""
        return (f"The case does not set up the problem that was asked: set {first.name} to "
                f"{first.specified} in {first.file}{more}, then re-run.")
    if block.unreadable:
        return ("Some values the specification names could not be read from the case; see "
                "'Not checked'.")
    return None


def _convergence(inputs: RunInputs, events, config):
    """CONV-12: the convergence block. It sits beside the verdict; VER-5.4b is off."""
    from courant.converge import summary
    from courant.converge.series import assess_case
    from courant.verify.schema import ConvergenceBlock, num

    conv = assess_case(inputs.case_dir, log_text=inputs.log_text, events=events, config=config,
                       solver=inputs.solver_hint)
    s = summary(conv)
    return ConvergenceBlock(
        status=s["status"], regime=s["regime"], enters_verdict=s["enters_verdict"],
        converged_at_step=s["converged_at_step"], binding_monitor=s["binding_monitor"],
        change_after_stop=num(s["change_after_stop"]), change_counts=num(s["change_counts"]),
        change_fraction=num(s["change_fraction"]), wasted_steps=s["wasted_steps"],
        steps_total=s["steps_total"], wasted_fraction=num(s["wasted_fraction"]),
        wasted_wall_clock_s=num(s["wasted_wall_clock_s"]),
        wasted_core_hours=num(s["wasted_core_hours"]),
        monitors_assessed=s["monitors_assessed"], monitors_found=s["monitors_found"],
        messages=tuple(f.message for f in conv.findings if f.severity in ("WARN", "FAIL") and f.message))


def _dedupe(items: list[str]) -> tuple[str, ...]:
    fixed = [s for s in sentences.ORDER if s in items]
    rest = sorted({s for s in items if s not in sentences.ORDER})
    return tuple(fixed + rest)


def _reasons(inputs: RunInputs, events, findings, trust) -> list[Reason]:
    out: list[Reason] = []
    log = inputs.log_name
    seen_div: set = set()
    for e in events:
        if e.kind is EventKind.CRASH:
            p = e.payload
            what = p.signal or (f"return code {p.return_code}" if p.return_code is not None else None)
            out.append(Reason(
                code="run.crash", kind="event", severity="critical",
                summary="The solver did not complete" + (f" ({p.tag})" if p.tag else ""),
                input=log, method="log parser: fatal-error and exit signatures",
                value=what or ("FOAM FATAL ERROR" if p.foam_fatal else None),
                log_offset=e.log_offset))
        elif is_divergence(e):
            p = e.payload
            if p.field in seen_div:
                continue
            seen_div.add(p.field)
            state = "NaN" if p.is_nan else "infinity" if p.is_inf else num(p.residual)
            out.append(Reason(
                code="run.divergence", kind="event", severity="critical",
                summary=f"Residual of {p.field or 'a field'} diverged",
                input=log, method="log parser: divergence detector",
                value=state, log_offset=e.log_offset))
    for f in policy.evidential(findings):
        if f.severity.value == "info":
            continue
        out.append(Reason(
            code=f.id, kind="finding",
            severity="critical" if f.severity.value == "critical" else "warn",
            summary=f.title + (f": {f.detail}" if f.detail else ""),
            input=_finding_input(f, log), method=f"checker {f.checker}",
            threshold=num(f.threshold), value=num(f.value), log_offset=f.log_offset))
    if trust is not None:
        for r in trust.results:
            if r.skipped or r.severity.value == "ok":
                continue
            out.append(Reason(
                code=r.check_code, kind="check", severity=r.severity.value,
                summary=r.detail or r.check_name,
                input=str(r.evidence.get("input") or log),
                method=f"check {r.check_code} ({r.check_name})",
                threshold=num(r.threshold), value=num(r.value)))
    return out


def _cause(rank: int, code: str) -> ProbableCause:
    from courant.taxonomy import TAXONOMY

    entry = TAXONOMY.get(code)
    return ProbableCause(rank=rank, code=code, name=entry.title if entry else code)


def _next_action(verdict: str, causes, events, log_name: str) -> str | None:
    if verdict == "TRUSTWORTHY":
        return None
    if verdict == "NOT_ASSESSABLE":
        return ("Point Courant at the solver log (log.<solver>) of a completed or "
                "running case.")
    if causes:
        entry = _kb().get(causes[0].code)
        if entry is not None and entry.canonical_fixes:
            return f"Suggested, not applied: {entry.canonical_fixes[0].description}"
    if any(e.kind is EventKind.CRASH for e in events):
        return (f"The run did not complete. Read the fatal error in {log_name} and correct "
                "the setup before re-running.")
    return "Review the reasons above before using this result."


def _stats(events, findings) -> Stats:
    by_sev: dict[str, int] = {"critical": 0, "info": 0, "warn": 0}
    for f in findings:
        by_sev[f.severity.value] = by_sev.get(f.severity.value, 0) + 1
    return Stats(
        events=len(events),
        residual_steps=sum(1 for e in events if e.kind is EventKind.RESIDUAL),
        crashes=sum(1 for e in events if e.kind is EventKind.CRASH),
        divergences=len(divergences(events)),
        findings_by_severity=by_sev,
    )
