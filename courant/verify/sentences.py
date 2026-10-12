"""Fixed sentences for what could not be checked (TRD VER-7).

These are product surface. They are quoted exactly in reports and in the
MCP output, so changing one is a visible change, not a refactor.
"""

from dataclasses import dataclass

DISCRETISATION = "Discretisation error was not quantified: a single mesh was supplied."
VALIDATION = "Accuracy against a reference was not assessed: no reference data was supplied."
NO_FIELDS = "Solution fields were not read: no time directory with fields was found."
NO_CONFIG = "The case configuration could not be read; setup rules that need it were not run."
NO_MESH = "Mesh quality metrics were not available: no log.checkMesh was found."
NO_OUTPUT = "No solver output was recognised in the log."
NO_RESIDUALS = "The log contains no residual history."
NO_TIME_MARKERS = ("The log's residuals carry no time step: no 'Time =' line was found, so which "
                   "step the run reached, and that it ended, cannot be shown.")

#: Order in which fixed sentences appear, before per-check sentences.
ORDER = (NO_OUTPUT, NO_RESIDUALS, NO_TIME_MARKERS, NO_CONFIG, NO_MESH, NO_FIELDS, DISCRETISATION,
         VALIDATION)

NO_MONITORS = ("Convergence of the answer was not assessed: no engineering monitor "
               "(function-object output under postProcessing/) was found.")

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
CAUSES = ("run", "spec", "fork")

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
    Insufficient("spec_hash_mismatch", "VER-5.8", SPEC_VOID, "spec",
                 "task specification", "SHA-256 of the specification file against the locked hash"),
    Insufficient("spec_value_unreadable", "VER-5.9", SPEC_UNREADABLE, "spec",
                 "case dictionaries the task specification names", "deck reader (courant.spec)"),
    Insufficient("unsupported_version", "VER-5.10", UNSUPPORTED_VERSION, "fork",
                 "solver log banner", "banner reader against the measured set"),
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
    "VER-5.15": INSUFFICIENT["no_time_markers"],
    "VER-5.8": INSUFFICIENT["spec_hash_mismatch"],
    "VER-5.9": INSUFFICIENT["spec_value_unreadable"],
    "VER-5.10": INSUFFICIENT["unsupported_version"], "VER-5.11": INSUFFICIENT["spec_invalid"],
    "VER-5.12": INSUFFICIENT["spec_extracted_mismatch"],
    "VER-5.13": INSUFFICIENT["spec_role_unresolved"],
    "VER-5.14": INSUFFICIENT["time_scheme_unreadable"],
}

#: Every not-checked sentence of a fixed text, with its cause tag (GATE-4).
NOT_CHECKED_CAUSE: dict[str, str] = {
    NO_OUTPUT: "run", NO_RESIDUALS: "run", NO_TIME_MARKERS: "run", NO_CONFIG: "run", NO_MESH: "run", NO_FIELDS: "run",
    DISCRETISATION: "run", NO_MONITORS: "run",
    VALIDATION: "spec", NO_SPEC: "spec", SPEC_SCOPE: "spec", SPEC_VOID: "spec",
    UNRECOGNISED_LOG: "fork", UNSUPPORTED_VERSION: "fork", TIME_SCHEME_UNREADABLE: "fork",
    SPEC_UNREADABLE: "spec", SPEC_INVALID: "spec", SPEC_EXTRACTED_MISMATCH: "spec",
    SPEC_ROLE_UNRESOLVED: "fork", SPEC_EXTRACTED: "spec",
}


def insufficient_for(rule: str) -> Insufficient:
    """The closed vocabulary's entry for the rule that made a verdict insufficient. A rule
    outside the table is a bug, never a new code."""
    try:
        return BY_RULE[rule]
    except KeyError:
        raise ValueError(f"rule {rule!r} is not in GATE-4's closed list of insufficient rules") from None
