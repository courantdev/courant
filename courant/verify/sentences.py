"""Fixed sentences for what could not be checked (TRD VER-7).

These are product surface. They are quoted exactly in reports and in the
MCP output, so changing one is a visible change, not a refactor.
"""

import re
from dataclasses import dataclass

DISCRETISATION = "Discretisation error was not quantified: a single mesh was supplied."
VALIDATION = "Accuracy against a reference was not assessed: no reference data was supplied."
NO_FIELDS = "Solution fields were not read: no time directory with fields was found."
NO_CONFIG = "The case configuration could not be read; setup rules that need it were not run."
NO_MESH = "Mesh quality metrics were not available: no log.checkMesh was found."
MESH_PARTIAL = ("Mesh quality was read from constant/polyMesh for the cell aspect ratio only: "
                "no log.checkMesh was found, so skewness, non-orthogonality and negative "
                "volumes were not checked.")
NO_OUTPUT = "No solver output was recognised in the log."
NO_RESIDUALS = "The log contains no residual history."
NO_TIME_MARKERS = ("The log's residuals carry no time step: no 'Time =' line was found, so which "
                   "step the run reached, and that it ended, cannot be shown.")
#: VER-5.16: no exit code was given, the log does not end in the solver's `End`, and it shows no
#: fatal signature (`courant.watcher.exitcode`). A run still writing, a window of a longer log,
#: or a run stopped from outside: none of them is evidence that the solver failed.
TRUNCATED_LOG = ("The log stops before the solver's End and shows no fatal error or crash "
                 "signal, so whether the run finished, failed or is still running cannot be told.")

#: TRD VER-20: a multi-region log, ESI's CHT solvers (`Solving for fluid region <name>` headers)
#: or OpenFOAM.org's foamMultiRun (each time-loop line behind a prefix naming its region).
_region_dicts = ("The regions' own dictionaries (system/<region>/, constant/<region>/, "
                 "0/<region>/) were not read.")
#: Under the default an ESI multi-region log's residuals are read as v0.5.0 read them, and the
#: verdict says so. A field one region alone solves was judged on that region's own residual.
MULTI_REGION_MERGED = ("This is a multi-region log, and its residuals were read as one set: a "
                       "field solved in more than one region was judged on the last region's "
                       "residual only (--checks next keys them by region). " + _region_dicts)
#: Under `--checks next` (`log.regions`): each residual under the region the log names for it.
#: The Courant clause only when a region's Courant number was read: a steady CHT solver
#: (chtMultiRegionSimpleFoam) prints none.
MULTI_REGION_KEYED = ("This is a multi-region log: each residual was read under the region the "
                      "log names for it (field@region). " + _region_dicts)
MULTI_REGION_KEYED_COURANT = ("This is a multi-region log: each residual was read under the "
                              "region the log names for it (field@region), and the Courant "
                              "number is the largest region's. " + _region_dicts)
MULTI_REGION_COUPLED = ("Residuals solved after 'Solving energy coupled regions' were not judged: "
                        "the log does not say which region each belongs to.")
#: Under the default an OpenFOAM.org foamMultiRun log is read as v0.5.0 read it, which reads no
#: line behind its prefix: no `Time =` line, so no numbered step (VER-5.15), and no region. The
#: sentence names neither the fork nor the solver (ARC-14: the decision code names none).
MULTI_REGION_PREFIX_UNREAD = ("This is a multi-region log that writes each line of its time loop "
                              "behind a prefix, the region's name or blanks of the same width. "
                              "The prefix was not read (--checks next reads it), so neither its "
                              "'Time =' lines nor its regions were read.")

#: A run whose log shows the solver finished away from its endTime, not on a criterion of its
#: own (`courant.spec.as_run.end_against`; the hard-proofing audit, 21 Oct 2026: a cavity with
#: `stopAt nextWrite` ended at t = 0.1 of 0.5 and was accepted with no word of it). Reporting:
#: whether the task asked for that stop is the task's to say (TRD SPEC-13).
ENDED_SHORT = ("The run ended at t = {reached} ({log}), {where} its endTime {end}: whether it was "
               "meant to stop there was not checked against what ran.")

#: Order in which fixed sentences appear, before per-check sentences.
ORDER = (NO_OUTPUT, TRUNCATED_LOG, NO_RESIDUALS, NO_TIME_MARKERS, NO_CONFIG, NO_MESH, MESH_PARTIAL,
         NO_FIELDS, DISCRETISATION, VALIDATION)

# ---------------------------------------------------------------- VER-7b: which fields were judged
def fields_unread(time: str, names) -> str:
    """The fields at the time read exist and could not be read, on any rank (in place of
    NO_FIELDS when none could)."""
    return (f"Solution fields at t = {time} were found and could not be read "
            f"({', '.join(names)}): the checks that need them were not run.")


def _on_ranks(partial) -> str:
    return ", ".join(f"{name} on {read} of {ranks}" for name, read, ranks in partial)


def fields_partly_read(time: str, partial) -> str:
    """A decomposed case's field that some ranks read and others did not (absent there, or there
    and not read): the checks ran on the cells of the ranks read. `partial` is
    `FieldData.partial`, (name, ranks read, ranks)."""
    return (f"Solution fields at t = {time} were read on some ranks only ({_on_ranks(partial)}): "
            "the checks that need them judged only those ranks' cells.")


def fields_none_by_log(last: str) -> str:
    """Under `--checks next`, in NO_FIELDS' place when every solved time directory is later than
    the log's last step (`fields_left_after_log` names them): NO_FIELDS would say none was
    found."""
    return ("Solution fields were not read: no time directory with fields was found at or before "
            f"the log's last time step (t = {last}).")


def fields_unread_reason(time: str, unread, partial) -> str:
    """Under `--checks next` (`fields.unread`): the reason a field the run wrote at the time
    judged and that was not read, or was read on some ranks only, gives (a minor check, grade B
    at most). Fixed vocabulary: a time and field names."""
    if unread and partial:
        return (f"Solution fields at t = {time} could not be read ({', '.join(unread)}) or were "
                f"read on some ranks only ({_on_ranks(partial)}): the run's state at the time "
                "judged was not checked whole")
    if unread:
        return (f"Solution fields at t = {time} could not be read ({', '.join(unread)}): the "
                "run's state at the time judged was not checked")
    return (f"Solution fields at t = {time} were read on some ranks only ({_on_ranks(partial)}): "
            "the run's state at the time judged was checked on part of the domain")


def fields_after_log(read: str, last: str) -> str:
    """By default: the fields read are later than the log's last step."""
    return (f"Solution fields were read at t = {read}, later than the log's last time step "
            f"(t = {last}): they may be another run's, and this run's final fields were not "
            "checked.")


def fields_left_after_log(times, last: str) -> str:
    """Under `--checks next`: the time directories later than the log's last step, not read."""
    return (f"Time directories later than the log's last time step (t = {last}) were not read "
            f"({', '.join(times)}): they are not part of the run the log records.")


def fields_ranks_later(read: str, ranks: str) -> str:
    """By default: the root was read, and the ranks hold a later time."""
    return (f"Solution fields were read from the case root at t = {read}; processor*/ hold a "
            f"later time (t = {ranks}), whose fields were not checked.")


def fields_early(read: str, last: str, *, later_left: bool = False) -> str:
    """The latest fields written are more than one write interval before the log's last step.
    With `later_left` (under `--checks next`, later directories not read), the latest at or
    before that step."""
    if later_left:
        return (f"Solution fields were read at t = {read}, the latest time written at or before "
                f"the log's last time step (t = {last}) and more than one write interval before "
                "it: the run's final fields were not found.")
    return (f"Solution fields were read at t = {read}, the latest time written, more than one "
            f"write interval before the log's last time step (t = {last}): the run's final "
            "fields were not found.")


# ---------------------------------------------------------------- PRE-7: what the setup rules did not read
#: `physics.nu` found no viscosity it reads, in either mode (`courant.case.viscosity`): a case with
#: no top-level `nu` and no Newtonian phase's `nu` or `mu` the reader read (a non-Newtonian model,
#: OpenFOAM.org's `physicalProperties`, a sidecar captured before phases were read).
VISCOSITY_NONE = ("Viscosity was not checked against a fluid's range: no viscosity the check "
                  "reads was found in constant/transportProperties or "
                  "constant/thermophysicalProperties.")
#: By default: `bc.overconstrained` asks `p` and `U` (v0.5.0's rule), and the case has `p_rgh`, which
#: interFoam and the buoyant solvers solve; `bc.solved_pressure` (PRE-5) reads it under next.
PRESSURE_P_RGH = ("Over-constrained pressure-velocity boundaries were checked on p and U only: "
                  "this case's p_rgh, the pressure its solver solves, is read there only under "
                  "--checks next.")


def viscosity_next_only(parts) -> str:
    """By default: the viscosities the case sets that only `physics.nu.solved` (PRE-6) reads,
    each part naming one and its file."""
    return (f"Viscosity was not checked against a fluid's range for {' and '.join(parts)}, which "
            "only --checks next reads.")


NO_MONITORS = ("Convergence of the answer was not assessed: no engineering monitor "
               "(function-object output under postProcessing/) was found.")


# ---------------------------------------------------------------- VER-7d, VER-13a, VER-16b
#: VER-7d: checkMesh's maximum skewness was a zero-area face's, on an `empty` patch, and the
#: faces with an area were judged instead (`MeshQuality.skewness_with_area`). The sentence names
#: no fork or version (ARC-14, `tests/unit/test_architecture_lint.py`): checkMesh is the tool
#: whose arithmetic it describes (v2406's `primitiveMeshTools::faceSkewness`). Nor does it name
#: the patch: a patch name is the case author's, and the pseudonymised support bundle maps only
#: the ones conformance and the reasons cite (`support._names`), so `{where}` counts the `empty`
#: patches, as the moving-mesh line names none (the review of hp6/integrate).
SKEW_ZERO_AREA = ("Mesh skewness was not assessed on {faces} of zero area on {where}: checkMesh "
                  "divides a distance by its floor ROOTVSMALL (1e-150) for a face with no area, "
                  "so checkMesh's maximum, {printed}, measures no face, and the solver does not "
                  "discretise an empty patch. The largest skewness of the other faces, computed "
                  "from constant/polyMesh as checkMesh computes it, is {computed}.")
#: VER-13a: a steady case's log printed a Courant number, which no check judges there.
COURANT_STEADY = ("The Courant number the log prints (largest {co}) was not judged: the case is "
                  "steady ({why}), so its time step is a pseudo-step that enters no equation.")
#: VER-16b: a velocity component zero to machine precision was left out of the residual bar.
ZERO_COMPONENT = ("{names} not held to the residual bar: in U at t = {time} {which} zero to "
                  "machine precision everywhere ({ratios} of the largest |U|) and the other "
                  "components are not, so {its} normalised residual divides round-off by "
                  "round-off.")


def skewness_zero_area(mesh) -> str | None:
    """VER-7d's sentence for a mesh whose zero-area faces' skewness was set aside, or None: None
    too for a reading that is not whole (`MeshQuality.zero_area_set_aside`), whose checkMesh
    maximum is then what `mesh.skewness` judges."""
    if mesh is None or not mesh.zero_area_set_aside():
        return None
    n, patches = mesh.n_zero_area_faces, len(mesh.zero_area_patches)
    where = "an empty patch" if patches == 1 else f"{patches} empty patches"
    return SKEW_ZERO_AREA.format(
        faces=f"{n} face" if n == 1 else f"{n} faces", where=where,
        printed=f"{mesh.max_skewness:.3g}", computed=f"{mesh.skewness_with_area:.3g}")


def zero_components(components, time: str) -> str:
    """VER-16b's sentence: `components` are `convergence.ZeroComponent`s, in component order."""
    names = [c.name for c in components]
    one = len(names) == 1
    return ZERO_COMPONENT.format(
        names=(f"{names[0]} was" if one else f"{', '.join(names[:-1])} and {names[-1]} were"),
        time=time, which="it is" if one else "they are",
        ratios=", ".join(f"{c.name} at most {c.ratio:.2g}" for c in components),
        its="its" if one else "their")


def not_found(names) -> str:
    """The dictionaries every OpenFOAM solver reads that the case directory does not hold
    (`CaseConfig.missing`). A solver cannot start without them, so the directory is not the case
    as it ran, or has lost them since, and the setup rules that read them had nothing to read."""
    names = list(names)
    if len(names) == 1:
        return (f"A case dictionary every OpenFOAM solver reads was not found: {names[0]}. "
                "The setup rules that need it were not run.")
    return (f"Case dictionaries every OpenFOAM solver reads were not found: {', '.join(names)}. "
            "The setup rules that need them were not run.")

#: Present whenever a task specification was given (PRD 2.0 P2-SPEC-2).
SPEC_SCOPE = ("Conformance covers only the values the task specification names; anything it "
              "leaves open, such as a modelling choice it does not state, was not checked.")
#: GATE-2: the last not-checked sentence of a `courant.verdict/2` given no specification.
NO_SPEC = "No specification given: conformance to the task was not assessed."

SPEC_VOID = ("Conformance to the task specification was not assessed: the specification does "
             "not match the hash it was locked at before the run.")


# ---------------------------------------------------------------- GATE-4: insufficient, closed
#: The cause tags (GATE-4): what a reader must change for the verdict to move. `run` the run's own
#: output, `spec` the task specification, `fork` the solver fork or version (RRD RM-22 counts them).
#: `courant`, on a not-checked sentence only: what this release of Courant does not read, which
#: `--checks next` or a later release may (`VISCOSITY_NONE`, `PRESSURE_P_RGH`); no insufficient
#: code carries it.
CAUSES = ("run", "spec", "fork", "courant")

UNRECOGNISED_LOG = "Too little of the log was recognised as solver output to judge the run."
SPEC_UNREADABLE = ("A value the task specification names could not be read from the case, so the "
                   "case cannot be confirmed to be the problem that was asked.")
UNSUPPORTED_VERSION = ("This OpenFOAM fork or version is outside the set the gate's error rates "
                       "were measured on.")
SPEC_INVALID = "The task specification is impossible in itself, so nothing can conform to it."
SPEC_EXTRACTED_MISMATCH = ("A value in an extracted specification that nobody confirmed does not "
                           "match the case; an unconfirmed specification can only make the "
                           "verdict insufficient.")
SPEC_ROLE_UNRESOLVED = ("A value the task specification names by role resolves to no location, or "
                        "to more than one, in this case.")
TIME_SCHEME_UNREADABLE = ("The time scheme could not be read, so the run cannot be judged steady "
                          "or transient.")


#: SPEC-7, VER-5.12, VER-15 step 6: the specification was extracted by a model and nobody
#: confirmed it. A not-checked sentence on every verdict judged against one, accept included.
SPEC_EXTRACTED = ("Conformance was checked against an unconfirmed extracted specification and is "
                  "not binding.")


# ---------------------------------------------------------------- ADP-1: the fork and version
#: ADP-1's not-checked lines, templates of cause `fork` (GATE-4). `courant.forks.not_checked`
#: fills them with the OpenFOAM it read and the measured set (`courant.forks.MEASURED`), so
#: none names a fork or a version here (ARC-14). Default-on: reporting, no verdict change.
#: A version outside the measured set, read from the log's banner or a file the run wrote.
VERSION_UNMEASURED = ("The gate's error rates were measured on {measured} only; {where} names "
                      "{label}, which they do not cover.")
#: Nothing the run wrote names the fork, the version, or either.
VERSION_UNREAD = ("The OpenFOAM {what} not read: {why}; the gate's error rates were measured on "
                  "{measured} only.")
#: Only an input names one: `system/controlDict`'s header, never taken for the run's version.
VERSION_GIVEN_ONLY = ("The OpenFOAM fork and version the run was solved on were not read: the "
                      "solver log has no banner, and no file the run wrote names one; {source}, "
                      "an input that may have been copied from another version's tutorial, "
                      "names {label}. The gate's error rates were measured on {measured} only.")
#: A measured version, read from the header of a file the run wrote: the log has no banner.
VERSION_FROM_RUN_FILE = ("The OpenFOAM fork and version ({label}) were read from {source}, a file "
                         "the run wrote: the solver log has no banner.")


#: TRD VER-5a (an internal evaluation on rebuilt runs, 22 Oct 2026): a run on a moving mesh, or one joined
#: by a non-conformal interface (`courant.case.motion`). The flux and the torque that cross a
#: sliding interface are not summed, so whether its interpolation conserved them is not judged;
#: neither is the boundary flux of a mesh whose motion changes the domain (VER-18 skips it under
#: `--checks next`). Reporting, on by default: it moves no verdict. Fixed words: no patch name,
#: no type the case wrote; `{by}` is the keyword `constant/dynamicMeshDict` moves the mesh with.
SLIDING_INTERFACE = ("The balance across a sliding interface (flux and torque) is not assessed: "
                     "the mesh moves (constant/dynamicMeshDict, {by}) and a non-conformal (AMI) "
                     "interface joins it.")
NONCONFORMAL_INTERFACE = ("The balance across a non-conformal (AMI) interface (flux and torque) "
                          "is not assessed: a boundary condition of the case is that "
                          "interface's.")
MOVING_MESH = ("The flux balance of a moving mesh is not assessed: constant/dynamicMeshDict "
               "moves or changes the mesh ({by}).")
MESH_MOTION_UNREAD = ("The balance across a moving or sliding mesh (flux and torque) is not "
                      "assessed: constant/dynamicMeshDict could not be read, so whether the mesh "
                      "moves is unknown.")


#: The keywords `courant.case.motion.read_mesh_motion` says a mesh moves by: the only words `{by}`
#: is filled with.
_MOTION_KEYWORDS = frozenset({"dynamicFvMesh", "mover", "topoChanger", "refiner"})


def moving_mesh(config) -> str | None:
    """The not-assessed line for a run on a moving mesh, or one a non-conformal interface joins
    (TRD VER-5a); None for neither. `config` is the case's `CaseConfig`: `mesh_motion` and
    `interface_patches()`."""
    if config is None:
        return None
    motion = getattr(config, "mesh_motion", None)
    joined = bool(config.interface_patches())
    if motion is not None and motion not in _MOTION_KEYWORDS:    # "unreadable", or a word no
        return MESH_MOTION_UNREAD                                # reader writes: never quoted
    if motion is not None:
        return (SLIDING_INTERFACE if joined else MOVING_MESH).format(by=motion)
    return NONCONFORMAL_INTERFACE if joined else None


@dataclass(frozen=True)
class Insufficient:
    """One entry of the closed vocabulary: a code, its one VER-5.x rule, one sentence, its cause,
    and what a reason with that code says it read and how."""

    name: str
    rule: str
    sentence: str
    cause: str
    input: str
    method: str

    @property
    def code(self) -> str:
        return f"insufficient.{self.name}"


#: GATE-4 / VER-15: every `insufficient` verdict carries exactly one of these, and no other code
#: reaches one. Adding a code is a visible change to the product's surface (a test pins the set).
INSUFFICIENT: dict[str, Insufficient] = {e.name: e for e in (
    Insufficient("no_output", "VER-5.1", NO_OUTPUT, "run",
                 "solver log", "log parser: solver output markers"),
    Insufficient("unrecognised_log", "VER-5.1", UNRECOGNISED_LOG, "fork",
                 "solver log", "log parser: share of the log recognised"),
    Insufficient("no_residuals", "VER-5.5", NO_RESIDUALS, "run",
                 "solver log", "log parser: numbered steps with initial residuals"),
    Insufficient("no_time_markers", "VER-5.15", NO_TIME_MARKERS, "run",
                 "solver log", "log parser: a 'Time =' line for each residual step"),
    Insufficient("truncated_log", "VER-5.16", TRUNCATED_LOG, "run",
                 "solver log", "log parser: the solver's End line, fatal-error and signal "
                 "signatures; no exit code was given"),
    Insufficient("spec_hash_mismatch", "VER-5.8", SPEC_VOID, "spec",
                 "task specification", "SHA-256 of the specification file against the locked hash"),
    Insufficient("spec_value_unreadable", "VER-5.9", SPEC_UNREADABLE, "spec",
                 "case dictionaries the task specification names", "deck reader (courant.spec)"),
    Insufficient("unsupported_version", "VER-5.10", UNSUPPORTED_VERSION, "fork",
                 "solver log banner, or the header of a file the run wrote",
                 "banner reader against the measured set (ADP-1)"),
    Insufficient("spec_invalid", "VER-5.11", SPEC_INVALID, "spec",
                 "task specification", "specification self-consistency (SPEC-9)"),
    Insufficient("spec_extracted_mismatch", "VER-5.12", SPEC_EXTRACTED_MISMATCH, "spec",
                 "task specification (extracted, unconfirmed)",
                 "deck-versus-spec conformance (SPEC-7)"),
    Insufficient("spec_role_unresolved", "VER-5.13", SPEC_ROLE_UNRESOLVED, "fork",
                 "task specification (role)", "role resolver for the case's fork (SPEC-11)"),
    Insufficient("time_scheme_unreadable", "VER-5.14", TIME_SCHEME_UNREADABLE, "fork",
                 "system/fvSchemes", "ddtSchemes reader (ADP-4)"),
)}

#: The rule that decided, to the code its reason carries. VER-5.1 gives `no_output` until ADP-3's
#: recognised-fraction bar exists; `unrecognised_log` is then the same rule's second code.
BY_RULE: dict[str, Insufficient] = {
    "VER-5.1": INSUFFICIENT["no_output"], "VER-5.5": INSUFFICIENT["no_residuals"],
    "VER-5.15": INSUFFICIENT["no_time_markers"], "VER-5.16": INSUFFICIENT["truncated_log"],
    "VER-5.8": INSUFFICIENT["spec_hash_mismatch"],
    "VER-5.9": INSUFFICIENT["spec_value_unreadable"],
    "VER-5.10": INSUFFICIENT["unsupported_version"], "VER-5.11": INSUFFICIENT["spec_invalid"],
    "VER-5.12": INSUFFICIENT["spec_extracted_mismatch"],
    "VER-5.13": INSUFFICIENT["spec_role_unresolved"],
    "VER-5.14": INSUFFICIENT["time_scheme_unreadable"],
}

#: Every not-checked sentence of a fixed text, and every template one is filled from, with its
#: cause tag (GATE-4). A template (`ENDED_SHORT`) is keyed as written, `{placeholders}` and all,
#: which no sentence a verdict carries equals: `cause_of` reads a sentence's tag, templates too.
NOT_CHECKED_CAUSE: dict[str, str] = {
    NO_OUTPUT: "run", NO_RESIDUALS: "run", NO_TIME_MARKERS: "run", TRUNCATED_LOG: "run",
    NO_CONFIG: "run", NO_MESH: "run", NO_FIELDS: "run",
    ENDED_SHORT: "spec",
    VISCOSITY_NONE: "courant", PRESSURE_P_RGH: "courant",
    MESH_PARTIAL: "run",
    DISCRETISATION: "run", NO_MONITORS: "run",
    VALIDATION: "spec", NO_SPEC: "spec", SPEC_SCOPE: "spec", SPEC_VOID: "spec",
    UNRECOGNISED_LOG: "fork", UNSUPPORTED_VERSION: "fork", TIME_SCHEME_UNREADABLE: "fork",
    SPEC_UNREADABLE: "spec", SPEC_INVALID: "spec", SPEC_EXTRACTED_MISMATCH: "spec",
    SPEC_ROLE_UNRESOLVED: "fork", SPEC_EXTRACTED: "spec",
    VERSION_UNMEASURED: "fork", VERSION_UNREAD: "fork", VERSION_GIVEN_ONLY: "fork",
    VERSION_FROM_RUN_FILE: "fork",
    # A multi-region log is the run's own output; only the prefix the default does not read is a
    # limitation of the fork's reading (RRD RM-22 counts `fork`).
    MULTI_REGION_MERGED: "run", MULTI_REGION_KEYED: "run", MULTI_REGION_KEYED_COURANT: "run",
    MULTI_REGION_COUPLED: "run", MULTI_REGION_PREFIX_UNREAD: "fork",
    # What the gate does not sum, on a mesh that moves or a non-conformal interface: Courant's.
    SLIDING_INTERFACE: "courant", NONCONFORMAL_INTERFACE: "courant", MOVING_MESH: "courant",
    MESH_MOTION_UNREAD: "courant",
    # What the run's own mesh, regime and fields show, each set aside on that evidence.
    SKEW_ZERO_AREA: "run", COURANT_STEADY: "run", ZERO_COMPONENT: "run",
}

_FIELD = re.compile(r"\{\w+\}")
#: Each template, and the pattern a sentence filled from it matches.
_TEMPLATES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (t, re.compile(".+?".join(re.escape(part) for part in _FIELD.split(t)), re.DOTALL))
    for t in NOT_CHECKED_CAUSE if _FIELD.search(t))


def cause_of(sentence: str) -> str | None:
    """GATE-4: the cause tag of a not-checked sentence, a fixed one or one filled from a template
    here; None for a sentence this module did not write (a check's own detail)."""
    if sentence in NOT_CHECKED_CAUSE:
        return NOT_CHECKED_CAUSE[sentence]
    for template, pattern in _TEMPLATES:
        if pattern.fullmatch(sentence):
            return NOT_CHECKED_CAUSE[template]
    return None


#: The same lookup under the name the conformance tests use (one implementation: `cause_of`).
not_checked_cause = cause_of


def insufficient_for(rule: str) -> Insufficient:
    """The closed vocabulary's entry for the rule that made a verdict insufficient. A rule
    outside the table is a bug, never a new code."""
    try:
        return BY_RULE[rule]
    except KeyError:
        raise ValueError(f"rule {rule!r} is not in GATE-4's closed list of insufficient rules") from None
