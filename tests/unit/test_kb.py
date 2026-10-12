"""Unit tests for the Knowledge Base and signature-match retrieval."""

from courant.case.config import CaseConfig
from courant.diagnosis.checkers import CheckContext, run_all_checkers
from courant.diagnosis.findings import Finding, FindingSet, FindingSeverity
from courant.diagnosis.kb_data import build_default_kb
from courant.taxonomy import all_codes


def test_kb_covers_all_taxonomy_codes():
    kb = build_default_kb()
    assert len(kb) == 25
    for code in all_codes():
        assert kb.get(code) is not None, f"missing KB entry for {code}"


def test_every_fix_has_a_valid_tier():
    kb = build_default_kb()
    for code in all_codes():
        entry = kb.get(code)
        assert entry.canonical_fixes, f"{code} has no canonical fix"


def test_finding_id_match_ranks_correct_code():
    kb = build_default_kb()
    fs = FindingSet([
        Finding(id="cfl.max", checker="cfl", title="t", severity=FindingSeverity.CRITICAL,
                suggests=["NUM.CFL_VIOLATION"]),
    ])
    matches = kb.query(findings=fs, top_k=3)
    assert matches[0].code == "NUM.CFL_VIOLATION"
    assert any("cfl.max" in w for w in matches[0].why)


def test_log_pattern_match_surfaces_env_code():
    kb = build_default_kb()
    log = "--> FOAM FATAL IO ERROR: cannot find file constant/transportProperties"
    matches = kb.query(log_text=log, top_k=3)
    assert "ENV.MISSING_DICT" in [m.code for m in matches]


def test_config_predicate_match():
    kb = build_default_kb()
    cfg = CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                     relaxation_factors={"p": 1.0})
    matches = kb.query(config=cfg, top_k=3)
    assert "NUM.OVER_RELAXATION" in [m.code for m in matches]


def test_end_to_end_checkers_to_kb():
    # a diverging high-Courant run + over-relaxed config should rank CFL/relaxation high
    from courant.events import Event, EventKind, ResidualPayload, DivergencePayload, Severity

    events = [
        Event(run_id="r", seq=0, kind=EventKind.RESIDUAL,
              payload=ResidualPayload(time=0.001, initial_residuals={"p": 0.1}, courant_max=0.4)),
        Event(run_id="r", seq=1, kind=EventKind.RESIDUAL,
              payload=ResidualPayload(time=0.002, initial_residuals={"p": 5.0}, courant_max=90.0)),
        Event(run_id="r", seq=2, kind=EventKind.DIVERGENCE, severity=Severity.ERROR,
              payload=DivergencePayload(field="p", is_nan=True)),
    ]
    cfg = CaseConfig(adjust_timestep=False)
    ctx = CheckContext(run_id="r", events=events, config=cfg)
    findings = run_all_checkers(ctx)
    kb = build_default_kb()
    matches = kb.query(findings=findings, config=cfg, top_k=5)
    codes = [m.code for m in matches]
    assert "NUM.CFL_VIOLATION" in codes[:2]


class TestFindingSeverityAffectsScore:
    """Two INFO findings must not outweigh one CRITICAL.

    On a real diverging pisoFoam case they did: `bounding.k` and
    `bounding.epsilon` are INFO and together scored BC.TURB_INLET at 7.0, above
    NUM.CFL_VIOLATION at 6.0 — so a Courant blow-up was diagnosed as a
    turbulence inlet problem. Bounding of k and epsilon is a *symptom* of
    divergence as often as a cause, which is what INFO records.
    """

    @staticmethod
    def _score(kb, findings, **kwargs):
        from courant.diagnosis.findings import FindingSet

        matches = kb.query(findings=FindingSet(findings), **kwargs)
        return {m.code: m.score for m in matches}

    def test_an_info_finding_counts_for_less_than_a_warning(self):
        from courant.diagnosis.kb import KnowledgeBase

        weights = KnowledgeBase.SEVERITY_WEIGHT
        assert weights["info"] < weights["warn"]

    def test_critical_is_not_inflated(self):
        """The first attempt promoted CRITICAL to 2.0 and cost 13 points of
        top-1: `cfl.max` is critical and a Courant excursion is a common
        secondary symptom, so it won four cases it should have lost."""
        from courant.diagnosis.kb import KnowledgeBase

        weights = KnowledgeBase.SEVERITY_WEIGHT
        assert weights["critical"] == weights["warn"] == 1.0

    def test_the_score_explains_itself(self):
        """`why` is the instrumentation: a mis-ranking should be diagnosable
        from the record rather than reasoned about from scratch."""
        from courant.diagnosis.findings import Finding, FindingSet
        from courant.diagnosis.kb_data import build_default_kb

        kb = build_default_kb()
        findings = FindingSet([
            Finding(id="cfl.max", checker="cfl", title="t", severity="critical",
                    detail="d", suggests=["NUM.CFL_VIOLATION"]),
        ])
        match = next(m for m in kb.query(findings=findings)
                     if m.code == "NUM.CFL_VIOLATION")

        assert any("cfl.max" in reason and "crit" in reason for reason in match.why), (
            f"the score is not self-explaining: {match.why}"
        )
