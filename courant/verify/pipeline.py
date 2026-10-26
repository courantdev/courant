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

from courant import forks
from courant.case import wordre
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

from courant.case.solvers import solver_exec  # noqa: E402 - one Exec reading, conformance's too

#: What each checker reads when it is not the log (INV-2: a reason names its input).
_CHECKER_INPUT = {
    "mesh": "log.checkMesh (mesh quality metrics)",
    "scheme": "system/fvSchemes",
    "bc": "0/ boundary conditions",
    "physics": "constant/transportProperties or constant/thermophysicalProperties",
    "case": "case dictionaries",
}
_ID_INPUT = {"solver.": "system/fvSolution",
             "physics.laminar_regime": ("the solver log (Reading g), constant/g, 0/T, 0/p, "
                                        "constant/thermophysicalProperties or "
                                        "constant/transportProperties, constant/polyMesh "
                                        "(wall faces), and the mesh extent (log.checkMesh or "
                                        "constant/polyMesh)")}


#: A mesh finding's input when the metric was computed from the mesh files (VER-7a).
_POLYMESH_INPUT = "constant/polyMesh (cell aspect ratio)"


def _finding_input(f, log_name: str, mesh_view: dict | None = None) -> str:
    from courant.case.mesh import POLYMESH_SOURCE

    base = next((v for k, v in _ID_INPUT.items() if f.id.startswith(k)), None)
    if f.checker == "mesh" and (mesh_view or {}).get("source") == POLYMESH_SOURCE:
        base = _POLYMESH_INPUT
    base = base or _CHECKER_INPUT.get(f.checker) or log_name
    if f.location and f.location not in base:
        where = f"t = {f.location}" if f.location.replace(".", "", 1).isdigit() else f.location
        return f"{base} ({where})"
    return base


def limitations(findings, config=None) -> list[str]:
    """The not-checked sentences of the findings that report Courant's own inability to read an
    input (`policy.READER_LIMITATIONS`: a dictionary that could not be reached, opened or
    parsed, a value not read as the solver read it), each in `limitation`'s words. `courant
    verify` and `courant preflight` both say them."""
    return [limitation(f, config) for f in findings if f.id in policy.READER_LIMITATIONS]


@lru_cache(maxsize=1)
def _kb():
    from courant.diagnosis.kb_data import build_default_kb

    return build_default_kb()


def _fields(case_dir: str | None, last: float | None = None, mode: str = "default"):
    """What the field reader read (TRD VER-7b), or None for no case directory. Under `--checks
    next` (`fields.log_time`) it stops at `last`, the log's last time step, and
    (`conservation.mass_flux.phi`, VER-18) it reads the boundary flux from that time's `phi`."""
    if not case_dir:
        return None
    try:
        from courant.verification._fluid_reader import read_fields

        # VER-18: the boundary flux from the `phi` of the time read, behind --checks next.
        fluxes = checks.enabled("conservation.mass_flux.phi", mode)
        if last is not None and checks.enabled("fields.log_time", mode):
            return read_fields(case_dir, fluxes=fluxes, until=last)
        return read_fields(case_dir, fluxes=fluxes)
    except Exception:                                   # noqa: BLE001
        return None


def _shown_time(t: float) -> str:
    return format(t, ".10g")


def _field_sentences(read, last: float | None, config) -> list[str]:
    """What "not assessed" says about the fields (TRD VER-7b): which time was read against the
    log's last step (`Run.last_time`, the one definition), what was left unread or read on some
    ranks only, and NO_FIELDS when there were none to read."""
    from courant.verification._fluid_reader import after

    out: list[str] = []
    used = bool(getattr(read, "internal", None))
    if read is not None and read.unread:
        out.append(sentences.fields_unread(read.source_time, read.unread))
    elif not used:
        out.append(sentences.fields_none_by_log(_shown_time(last))
                   if read is not None and read.after_log and last is not None
                   else sentences.NO_FIELDS)
    if read is None:
        return out
    if read.partial and read.source_time is not None:
        out.append(sentences.fields_partly_read(read.source_time, read.partial))
    if read.after_log and last is not None:
        out.append(sentences.fields_left_after_log(read.after_log, _shown_time(last)))
    if read.later_ranks is not None and read.source_time is not None:
        out.append(sentences.fields_ranks_later(read.source_time, read.later_ranks))
    if used and last is not None and read.source_time is not None:
        period = config.write_period() if config is not None else None
        if after(read.source_time, last):
            out.append(sentences.fields_after_log(read.source_time, _shown_time(last)))
        elif period is not None and read.later_ranks is None and \
                last - float(read.source_time) > period * (1 + 1e-6):   # the ranks hold them
            out.append(sentences.fields_early(read.source_time, _shown_time(last),
                                              later_left=bool(read.after_log)))
    return out


def _zero_components(fields, events, config, mode: str, last: float | None):
    """VER-16b: the velocity components the latest U shows to be zero to machine precision
    (`convergence.zero_components`) that the residual bar would judge: solved in the log's last
    step with residuals, on a run held to the bar (`convergence.judged_on_the_bar`), and only in
    a U read at that step, `last` (`Run.last_time`), by the one rule conformance compares times
    with (`as_run._same_time`). () else, so a transient run, a log with no U solve and a case
    whose U was not read change nothing, and neither does a U from another time: under the
    default the field reader reads the latest time directory whatever the log's last step
    (`fields.log_time` is behind `--checks next`), and a U another run left after it, or the
    run's own from before its end, says nothing of the components this run's last step solved
    (the review of hp6/integrate: a no-swirl run's later U set aside the stalled Uz of a rerun
    with swirl). Such a U can only add a reject, as in v0.5.0, never remove one."""
    from courant.spec.as_run import _same_time
    from courant.verification.convergence import (
        final_residuals, judged_on_the_bar, zero_components)

    if fields is None or last is None or not judged_on_the_bar(config, mode):
        return ()
    at = getattr(fields, "source_time", None)
    try:
        if not _same_time(float(at if at is not None else fields.time), last):
            return ()
    except (TypeError, ValueError):
        return ()
    solved = final_residuals(events)
    return tuple(c for c in zero_components(fields) if c.name in solved)


def _zero_sentence(zero, read) -> list[str]:
    if not zero:
        return []
    time = getattr(read, "source_time", None) or format(getattr(read, "time", 0.0), "g")
    return [sentences.zero_components(zero, time)]


def _steady_courant_sentence(events, config, log_text: str | None,
                             mode: str = "default") -> list[str]:
    """VER-13a's sentence, when the log printed a Courant number on a declared steady case:
    under `--checks next` only (`cfl.steady`); under the default, none."""
    from courant.diagnosis.checkers import steady_courant
    from courant.verification.convergence import peak_courant

    why = steady_courant(config, log_text, mode)
    peak = peak_courant(events) if why else None
    return [] if peak is None else [sentences.COURANT_STEADY.format(co=f"{peak:.3g}", why=why)]


def _solver(inputs: RunInputs, config) -> str | None:
    """`run.solver`: the log's solver (`solvers.solver_exec`, conformance's one reading: the last
    Exec line that names a solver, not a log's first, which may be a mesher's), else the
    caller's hint, else the deck's `application`."""
    ex = solver_exec(inputs.log_text)
    if ex is not None:
        return ex.program
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

    `run.fork` and `run.solver_version` are what ADP-1 read (`courant.forks`); version 1 has
    no field for them.
    """
    v1, seen = _decided(inputs)
    return to_v2(v1, extracted=extracted_unconfirmed(inputs.spec), fork=seen.fork,
                 solver_version=seen.version)


def verify_inputs_v1(inputs: RunInputs) -> Verdict:
    """The decision as `courant.verdict/1`. Callers want `verify_inputs`; this is its first half."""
    return _decided(inputs)[0]


@wordre.bounded                                     # the case's keyword patterns (SPEC-2)
def _decided(inputs: RunInputs) -> tuple[Verdict, forks.SolverVersion]:
    if not inputs.case_dir:
        return _verify(inputs, None)
    with recording(inputs.case_dir) as reads:
        return _verify(inputs, reads)


def _manifest(inputs: RunInputs, reads: Reads | None) -> tuple[InputFile, ...]:
    if reads is None:
        return inputs.inputs
    return merge(inputs.inputs, manifest(inputs.case_dir, reads.paths()))


def _verify(inputs: RunInputs, reads: Reads | None) -> tuple[Verdict, forks.SolverVersion]:
    from courant import __version__
    from courant.watcher.exitcode import verdict_exit_code

    not_assessed: list[str] = [sentences.DISCRETISATION, sentences.VALIDATION]
    is_recognised = policy.recognised(inputs.log_text)
    # The caller's exit code, as read; else 0 for a log that ends in End; else None, unknown,
    # never a nonzero code nobody returned (VER-3, VER-5.16). Whose End: VER-3a, behind
    # --checks next, counts only an End after the last time step. The parse below is the same
    # either way (0 and None emit no crash); VER-5.16 reads the difference, and so do the
    # endTime item and the ended-short line, through `Run.exit_code` (one reading).
    exit_code = verdict_exit_code(inputs.log_text, inputs.exit_code, inputs.checks)

    from courant.case.config import CaseConfig
    from courant.case.mesh import MeshQuality

    # VER-13a's `ddt_read_whole` is the case reader's own reading of `system/fvSchemes`, so it is
    # kept only for `ingest.from_case_dir`'s view (`CaseConfig.from_view`), as VER-7d's below.
    config = CaseConfig.from_view(inputs.config_view, read_here=inputs.source == "case_dir")
    # VER-7d's reading is computed from the mesh files by `ingest.from_case_dir` alone; a view
    # from anywhere else (a corpus sidecar, an MCP `case_json`) states it, which is no evidence.
    mesh = MeshQuality.from_view(inputs.mesh_view, computed_here=inputs.source == "case_dir")
    if mesh is not None and mesh.source is not None and \
            not checks.enabled("mesh.aspect_ratio.polymesh", inputs.checks):
        mesh = None             # computed from the mesh files: behind --checks next (VER-7a)
    if config is None or config.nothing_read():         # a directory with no dictionary in it
        not_assessed.append(sentences.NO_CONFIG)
    elif config.missing:
        not_assessed.append(sentences.not_found(config.missing))
    if mesh is None:
        not_assessed.append(sentences.NO_MESH)
    elif mesh.source is not None:                       # the aspect ratio only (RM-35 dev)
        not_assessed.append(sentences.MESH_PARTIAL)
    skew_set_aside = sentences.skewness_zero_area(mesh)  # VER-7d, as preflight says it
    if skew_set_aside:
        not_assessed.append(skew_set_aside)
    from courant.verify.preflight import setup_not_checked

    not_assessed.extend(setup_not_checked(config, inputs.checks))     # PRE-7, as preflight says
    # VER-5a: a moving or sliding mesh, whose balance across the interface is not judged.
    # Reporting, on by default.
    moving = sentences.moving_mesh(config)
    if moving is not None:
        not_assessed.append(moving)
    #: Where ADP-1's line goes, once the run's last step is known (below).
    version_at = len(not_assessed)

    events, findings, trust, diagnosis = [], [], None, None
    if is_recognised:
        from courant.case.solvers import is_multi_run
        from courant.watcher.parser import LogParser

        # VER-20: a multi-region log's residuals keyed by region under `--checks next`; under
        # the default read as v0.5.0 read them, and the verdict says which.
        keyed = checks.enabled("log.regions", inputs.checks)
        multirun = is_multi_run(_solver(inputs, config))
        parser = LogParser(run_id=inputs.run_id, regions=keyed, multirun=multirun)
        events = list(parser.parse_text(inputs.log_text, return_code=exit_code))
        not_assessed.extend(_region_sentences(inputs.log_text, parser, keyed=keyed,
                                              multirun=multirun))
    from courant.spec.as_run import Run

    #: The solver's own record of the run, for what conformance, the end time and the field read
    #: read of it: one last time step (`Run.last_time`) for all three, and the exit code VER-5.16
    #: is decided on, so the endTime item and `ENDED_SHORT` read the End the verdict read.
    run = Run(inputs.log_text, inputs.log_name, tuple(events), exit_code=exit_code) \
        if inputs.log_text.strip() else None
    # VER-7b: the fields are judged against the log's last step, so they are read after it.
    last = run.last_time if run is not None else None
    # ADP-1: the fork and version, from the log's banner or the header of a file the run wrote
    # (system/controlDict's, an input, is reported only), under next (`fields.log_time`) in no
    # time directory later than the log's last step, as the fields are read. Reporting, on by
    # default; the policy reads only whether they are in the measured set (VER-5.10).
    seen = forks.read(inputs.log_text, inputs.case_dir,
                      until=last if checks.enabled("fields.log_time", inputs.checks) else None)
    version_line = forks.not_checked(seen)
    if version_line:
        not_assessed.insert(version_at, version_line)
    read = _fields(inputs.case_dir, last, inputs.checks)
    fields = read if getattr(read, "internal", None) else None
    not_assessed.extend(_field_sentences(read, last, config))
    if is_recognised and events:
        from courant.diagnosis.deterministic import DeterministicDiagnoser
        from courant.diagnosis.bundle import assemble_bundle
        from courant.diagnosis.checkers import CheckContext, run_all_checkers
        from courant.verification.checks import VerifyContext
        from courant.verification.suite import verify

        # VER-16b: the velocity components the latest U shows to be zero to machine precision,
        # left out of the residual bar by both its askers, and named below.
        zero = _zero_components(fields, events, config, inputs.checks, last)
        not_assessed.extend(_zero_sentence(zero, read))
        names = tuple(c.name for c in zero)
        finding_set = run_all_checkers(CheckContext(
            run_id=inputs.run_id, events=events, config=config, mesh=mesh,
            log_text=inputs.log_text, checks=inputs.checks, zero_components=names))
        finding_set = checks.only_enabled(finding_set, inputs.checks)
        findings = list(finding_set)
        # VER-13a, under `--checks next`: a steady run's Courant number is judged by no check,
        # and the verdict says so.
        not_assessed.extend(_steady_courant_sentence(events, config, inputs.log_text,
                                                     inputs.checks))
        # The solver the verdict names (the log's Exec line), not only controlDict's
        # `application`: the flux balance a run owes is the solver's that ran (VER-18).
        trust = verify(VerifyContext(run_id=inputs.run_id, events=events,
                                     config=config, fields=fields, checks=inputs.checks,
                                     solver=_solver(inputs, config), zero_components=names))
        if checks.enabled("fields.unread", inputs.checks):          # VER-7b, behind next
            from courant.verification.checks import fields_unread
            from courant.verification.grade import TrustReport

            unread = fields_unread(read)
            if unread is not None:
                trust = TrustReport.from_results(trust.run_id, [*trust.results, unread])
        bundle = assemble_bundle(run_id=inputs.run_id, events=events, findings=finding_set,
                                 kb=_kb(), config=config, mesh=mesh, log_text=inputs.log_text,
                                 checks=inputs.checks)
        diagnosis = DeterministicDiagnoser(_kb()).diagnose(bundle)

    convergence = None
    if inputs.case_dir and is_recognised and events:
        convergence = _convergence(inputs, events, config, exit_code)
        if convergence.status == "NOT_ASSESSABLE" and not convergence.monitors_found:
            not_assessed.append(sentences.NO_MONITORS)

    verdict, rule = policy.decide(is_recognised, events, findings, trust,
                                  truncated=exit_code is None)
    # Whether there is a grade is the run's own rule's to say, before any specification rule
    # replaces it: a specification can say the case is the wrong problem, not that a run whose
    # end is unknown is over (VER-5.16).
    graded = trust is not None and rule not in _UNGRADED
    if rule == "VER-5.1":
        not_assessed.insert(0, sentences.NO_OUTPUT)
    elif rule == "VER-5.16":
        not_assessed.insert(0, sentences.TRUNCATED_LOG)
    elif rule == "VER-5.5":
        not_assessed.insert(0, sentences.NO_RESIDUALS)
    elif rule == "VER-5.15":
        not_assessed.insert(0, sentences.NO_TIME_MARKERS)

    conformance = None
    if inputs.spec is not None:
        from courant.spec import check_conformance

        # SPEC-13 (`conformance.as_run`, behind --checks next): the values are also checked
        # against what ran, the solver's log and the mesh it read.
        conformance = check_conformance(inputs.spec, inputs.case_dir,
                                        spec_sha256=inputs.spec_sha256 or "",
                                        locked_sha256=inputs.spec_locked_sha256,
                                        checks=inputs.checks, run=run)
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
    ended = _ended_elsewhere(inputs, config, run, conformance)
    if ended is not None:
        not_assessed.append(ended)
    if checks.enabled("version.unsupported", inputs.checks):     # VER-5.10, behind --checks next
        verdict, rule = policy.apply_version(verdict, rule, seen.measured)
    if trust is not None:
        not_assessed.extend(f"{r.check_name}: {r.detail}" for r in trust.results
                            if r.skipped and r.detail)
    not_assessed.extend(limitations(findings, config))

    extracted = conformance is not None and extracted_unconfirmed(inputs.spec)
    reasons = policy.order_reasons(_reasons(inputs, events, findings, trust)
                                   + _conformance_reasons(conformance, extracted=extracted))
    causes = tuple(_cause(i + 1, code)
                   for i, code in enumerate(_listed_causes(diagnosis, reasons)))
    grade = trust.grade.value if graded else None

    return (Verdict(
        tool=ToolInfo(version=__version__, checks="next" if inputs.checks == "next" else None),
        run=RunInfo(id=inputs.run_id, source=inputs.source,
                    solver=_solver(inputs, config), inputs=_manifest(inputs, reads)),
        verdict=verdict, rule=rule, grade=grade,
        reasons=reasons, probable_causes=causes,
        not_assessed=_dedupe(not_assessed),
        next_action=(_conformance_action(conformance, extracted=extracted)
                     or _next_action(verdict, rule, causes, events, inputs.log_name)),
        stats=_stats(events, findings),
        convergence=convergence,
        conformance=conformance,
    ), seen)


#: Rules whose verdict has nothing to grade: no output, no residual history, no time markers, and
#: a log whose end is unknown (VER-5.16), where a grade would judge a run that may not be over.
_UNGRADED = frozenset({"VER-5.1", "VER-5.5", "VER-5.15", "VER-5.16"})


def limitation(finding, config=None) -> str:
    """The not-assessed sentence of a finding that reports Courant's own inability to read an
    input (`policy.READER_LIMITATIONS`), in `courant verify` and `courant preflight` alike. For
    `case.unread_value`, the local sentence that quotes what the case wrote (TRD VER-3b); its
    finding says it in fixed vocabulary."""
    if finding.id == "case.unread_value":
        from courant.diagnosis.checkers import unread_values_sentence

        local = unread_values_sentence(config)
        if local is not None:
            return local
    return f"{finding.title}: {finding.detail}" if finding.detail else finding.title


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


def _convergence(inputs: RunInputs, events, config, exit_code: int | None):
    """CONV-12: the convergence block. It sits beside the verdict; VER-5.4b is off. The exit
    code is the one the log was parsed with: None, an unknown end, reports no waste (VER-5.16)."""
    from courant.converge import summary
    from courant.converge.series import assess_case
    from courant.verify.schema import ConvergenceBlock, num

    conv = assess_case(inputs.case_dir, log_text=inputs.log_text, events=events, config=config,
                       solver=inputs.solver_hint, exit_code=exit_code, checks=inputs.checks)
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


def _ended_elsewhere(inputs: RunInputs, config, run, conformance=None) -> str | None:
    """`sentences.ENDED_SHORT` for a run whose log shows it finished away from its endTime, not
    on a criterion of its own (`as_run.end_against`, the endTime item's one definition). Reporting,
    on by default: whether the task asked for that stop is the task's to say. Not said when a
    specification's endTime item was compared with the log (SPEC-13, under --checks next): that
    item says it, and decides. A specification that does not bind (void, SPEC-8; impossible,
    SPEC-9) compares no item, so the line is said then too."""
    end = getattr(config, "end_time", None)
    if run is None or end is None:
        return None
    if (conformance is not None and inputs.spec is not None
            and not policy.spec_void(conformance) and not policy.spec_invalid(conformance)
            and checks.enabled("conformance.as_run", inputs.checks) and any(
                isinstance(i, dict) and i.get("file") == "system/controlDict"
                and i.get("key") == "endTime" and not i.get("path")
                for i in (inputs.spec.get("items") or ()))):
        return None
    from pathlib import Path

    from courant.spec import as_run, deck
    from courant.spec.model import DEFAULT_REL_TOL

    how, reached, _ = as_run.end_against(run, end, rel=DEFAULT_REL_TOL)
    if how in ("before", "past") and inputs.case_dir:     # a runTimeControl may have ended it
        text, _ = deck._read_text(Path(inputs.case_dir), "system/controlDict")
        how, reached, _ = as_run.end_against(run, end, rel=DEFAULT_REL_TOL,
                                             control_text=text or "")
    if how not in ("before", "past"):
        return None
    return sentences.ENDED_SHORT.format(reached=as_run._fmt(reached), log=run.log_name,
                                        where=how, end=as_run._fmt(end))


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
            # A nonzero exit is only ever an exit code the caller gave (VER-1): the log never
            # yields one (VER-3), so its reason names that input, not the log.
            given = p.tag == "nonzero_exit"
            what = p.signal or (f"return code {p.return_code}" if p.return_code is not None else None)
            out.append(Reason(
                code="run.crash", kind="event", severity="critical",
                summary="The solver did not complete" + (f" ({p.tag})" if p.tag else ""),
                input=_GIVEN_EXIT if given else log,
                method=(_GIVEN_EXIT_METHOD if given
                        else "log parser: fatal-error and exit signatures"),
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
            input=_finding_input(f, log, inputs.mesh_view), method=f"checker {f.checker}",
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


def _region_sentences(text: str, parser, *, keyed: bool, multirun: bool) -> list[str]:
    """What a verdict says of a multi-region log (TRD VER-20): how its residuals were read. ESI's
    CHT solvers name a region in a header (`regions_in`), OpenFOAM.org's foamMultiRun in a prefix
    on each line, read under `next` once its width is (`LogParser.prefix_width`)."""
    from courant.watcher.parser import has_coupled_solve, regions_in

    if not (regions_in(text) or parser.prefix_width is not None):
        return [sentences.MULTI_REGION_PREFIX_UNREAD] if multirun and not keyed else []
    if not keyed:
        return [sentences.MULTI_REGION_MERGED]
    out = [sentences.MULTI_REGION_KEYED_COURANT if parser.region_courant
           else sentences.MULTI_REGION_KEYED]
    if has_coupled_solve(text):
        out.append(sentences.MULTI_REGION_COUPLED)
    return out


def _listed_causes(diagnosis, reasons) -> list[str]:
    """The probable causes a verdict lists (TRD VER-8a): the deterministic ranking's top three
    among the hypotheses that cite a finding the verdict reports as a reason.

    A probable cause explains what the reasons report, so it must cite one of them. The ranking
    also scores configuration predicates and INFO findings, which no verdict reports, and keeps a
    hypothesis with nothing else behind it (scaled down, `deterministic.UNCITED_SCALE`). v0.5.0
    listed the top three whatever stood behind them: every laminar run carried
    `PHYS.LAMINAR_TURBULENT` (`simulationType laminar` alone), on its accept and on every verdict
    that was not one (a conformance reject, VER-5.7, and the `insufficient` verdicts a
    specification gives included: VER-5.8, VER-5.9, VER-5.11, VER-5.12), and a missing
    dictionary carried `NUM.MISSING_ADJUSTABLE_DT` from the INFO `scheme.fixed_dt` (an internal
    evaluation and its review, 20 and 21 Oct 2026). An accept (VER-5.6) reports no finding
    above INFO by its own rule, so it lists none.

    The ranking is the diagnosis's own, so a listed cause keeps its place; only the uncited are
    left out. The text and HTML reports show no cause on an accept either way.
    """
    if diagnosis is None:
        return []
    reported = {r.code for r in reasons if r.kind == "finding"}
    return [h.root_cause_code for h in diagnosis.ranked()
            if reported.intersection(h.evidence_refs)][:3]


#: A crash reason's input and method when the exit code came with the run (VER-1).
_GIVEN_EXIT = "exit code given with the run"
_GIVEN_EXIT_METHOD = "the run's exit code as the caller gave it, not inferred from the log"


def _cause(rank: int, code: str) -> ProbableCause:
    from courant.taxonomy import TAXONOMY

    entry = TAXONOMY.get(code)
    return ProbableCause(rank=rank, code=code, name=entry.title if entry else code)


#: VER-5.10's next action. It names no version (ARC-14): the not-checked line does.
_UNMEASURED_ACTION = ("The gate's error rates were not measured on this run's OpenFOAM fork or "
                      "version (see 'Not checked'): review the result by hand, or re-run the case "
                      "on a measured version.")


def _next_action(verdict: str, rule: str, causes, events, log_name: str) -> str | None:
    if verdict == "TRUSTWORTHY":
        return None
    if rule == "VER-5.10":
        return _UNMEASURED_ACTION
    if rule == "VER-5.16":
        return (f"{log_name} stops before the solver's End. If the run is still going, verify it "
                "once it has finished; if it has finished, verify its whole log.")
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
