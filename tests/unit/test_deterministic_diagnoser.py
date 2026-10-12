"""The deterministic diagnoser's ranking rules.

This is the engine every customer falls back to when they are offline or out of
quota, and the one CI gates on — so its ordering rules deserve their own tests
rather than being inferred from The Bench's aggregate score.
"""

from courant.diagnosis.deterministic import DeterministicDiagnoser
from courant.diagnosis.bundle import CaseSummary, EvidenceBundle
from courant.diagnosis.findings import Finding, FindingSeverity
from courant.diagnosis.kb import KBMatch


def _finding(fid: str, suggests: list[str]):
    return Finding(id=fid, checker="c", title=fid,
                   severity=FindingSeverity.WARN, detail="d", suggests=suggests)


def _bundle(findings, matches):
    return EvidenceBundle(
        run_id="r", case_summary=CaseSummary(solver="icoFoam"),
        det_findings=findings,
        kb_matches=[KBMatch(code=c, score=s, why=[], canonical_fixes=[])
                    for c, s in matches])


class TestEveryDiagnosisCitesAFinding:
    """Phase 1's definition of done — *"every diagnosis cites at least one real
    Finding"* — had nothing enforcing it, and a code could reach the top of the
    list on a config predicate alone.

    `PHYS.LAMINAR_TURBULENT` did exactly that on a crashed case:
    `simulationType laminar` scored 2.0 from configuration, every finding-backed
    code scored less, and the report led with a claim it could not point at a
    single line of the log to support.

    That is worse than being wrong. An engineer can argue with a wrong diagnosis
    that shows its evidence; a right one that shows none is indistinguishable
    from a guess.
    """

    def test_an_uncited_code_is_ranked_below_a_cited_one(self):
        bundle = _bundle(
            findings=[_finding("cfl.max", ["NUM.CFL_VIOLATION"])],
            # The uncited code scores higher, and still loses.
            matches=[("PHYS.LAMINAR_TURBULENT", 9.0), ("NUM.CFL_VIOLATION", 2.0)])

        ranked = DeterministicDiagnoser().diagnose(bundle).ranked_codes()

        assert ranked[0] == "NUM.CFL_VIOLATION"

    def test_two_cited_codes_are_still_decided_on_score(self):
        """The demotion must not become a second ranking rule. Where both sides
        have evidence, the score is the argument."""
        bundle = _bundle(
            findings=[_finding("cfl.max", ["NUM.CFL_VIOLATION"]),
                      _finding("mesh.skewness", ["MESH.SKEWNESS"])],
            matches=[("MESH.SKEWNESS", 4.0), ("NUM.CFL_VIOLATION", 8.0)])

        ranked = DeterministicDiagnoser().diagnose(bundle).ranked_codes()

        assert ranked[0] == "NUM.CFL_VIOLATION"

    def test_an_all_uncited_diagnosis_keeps_its_score_order(self):
        """Nothing to promote. The configuration is still ranking signal — it
        just cannot outrank evidence."""
        bundle = _bundle(findings=[],
                         matches=[("BC.TURB_INLET", 2.0),
                                  ("PHYS.LAMINAR_TURBULENT", 6.0)])

        ranked = DeterministicDiagnoser().diagnose(bundle).ranked_codes()

        assert ranked[0] == "PHYS.LAMINAR_TURBULENT"

    def test_the_kb_signature_counts_as_a_citation(self):
        """A finding named in the entry's signature is evidence for it even when
        the finding does not name the code back."""
        bundle = _bundle(
            findings=[_finding("residual.plateau", [])],
            matches=[("CONV.RESIDUAL_PLATEAU", 3.0), ("BC.TURB_INLET", 5.0)])

        top = DeterministicDiagnoser().diagnose(bundle).hypotheses[0]

        assert top.root_cause_code == "CONV.RESIDUAL_PLATEAU"
        assert top.evidence_refs == ["residual.plateau"]


class TestNoSignalIsNotAGuess:
    """"We could not tell" is a worse-sounding answer and a much better one.

    The engine used to emit `CONV.RESIDUAL_PLATEAU` at 0.1 confidence whenever
    no KB signature matched, purely because the schema demanded a hypothesis. A
    named taxonomy code does not read as a guess to anything downstream: the
    certification harness recorded it as *the* diagnosis, the report renders it
    as the leading hypothesis, and the fix loop proposes repairs for it.

    It was reached on real runs. `icoFoam_cavity::missing_dict` landed here for a
    whole certification round because a `checkMesh` failure aborted the solve
    before it could print the very error that identifies the fault — so the
    answer to "what went wrong" was a plausible code with no connection to the
    case at all.
    """

    def test_an_empty_bundle_produces_no_hypotheses(self):
        bundle = _bundle(findings=[], matches=[])

        assert DeterministicDiagnoser().diagnose(bundle).hypotheses == []

    def test_the_schema_accepts_an_empty_diagnosis(self):
        """The schema was the last thing insisting on an answer, and the guess
        existed only to satisfy it."""
        from courant.diagnosis.schema import DiagnosisResponse

        assert DiagnosisResponse(hypotheses=[]).ranked_codes() == []
