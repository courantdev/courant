"""Unit tests for the Evidence Bundle assembler."""

from courant.case.config import CaseConfig
from courant.diagnosis.bundle import (
    MAX_EVENT_WINDOW,
    MAX_RESIDUAL_POINTS,
    assemble_bundle,
)
from courant.diagnosis.checkers import CheckContext, run_all_checkers
from courant.diagnosis.kb_data import build_default_kb
from courant.events import Event, EventKind, ResidualPayload


def _long_run(n=200):
    events = []
    for i in range(n):
        r = max(0.02, 0.5 * (0.95 ** i))
        events.append(Event(run_id="r", seq=i, kind=EventKind.RESIDUAL,
                            payload=ResidualPayload(iteration=i + 1,
                                                    initial_residuals={"p": r, "Ux": r * 0.1},
                                                    courant_max=0.4)))
    return events


KB = build_default_kb()


def _bundle(events, config=None):
    ctx = CheckContext(run_id="r", events=events, config=config)
    findings = run_all_checkers(ctx)
    return assemble_bundle("r", events, findings, KB, config=config, cell_count=1000)


class TestAssembly:
    def test_residual_series_downsampled(self):
        b = _bundle(_long_run(200))
        assert len(b.residual_iterations) <= MAX_RESIDUAL_POINTS + 2
        for series in b.residual_series.values():
            assert len(series) == len(b.residual_iterations)
        # first and last iterations preserved
        assert b.residual_iterations[0] == 1.0
        assert b.residual_iterations[-1] == 200.0

    def test_event_window_capped(self):
        b = _bundle(_long_run(200))
        assert len(b.event_window) <= MAX_EVENT_WINDOW

    def test_config_view_and_digest(self):
        cfg = CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                         relaxation_factors={"p": 1.0})
        b = _bundle(_long_run(10), config=cfg)
        assert b.case_summary.solver == "simpleFoam"
        assert b.config_view["relaxation_factors"] == {"p": 1.0}
        assert len(b.case_summary.key_dicts_digest) == 12

    def test_kb_matches_present(self):
        cfg = CaseConfig(solver="simpleFoam", ddt_scheme="steadyState",
                         relaxation_factors={"p": 1.0})
        b = _bundle(_long_run(10), config=cfg)
        assert any(m.code == "NUM.OVER_RELAXATION" for m in b.kb_matches)


class TestDeterminism:
    def test_identical_inputs_identical_bundle(self):
        events = _long_run(120)
        cfg = CaseConfig(solver="simpleFoam")
        b1 = _bundle(events, cfg)
        b2 = _bundle(events, cfg)
        assert b1.to_llm_json() == b2.to_llm_json()
        assert b1.case_summary.key_dicts_digest == b2.case_summary.key_dicts_digest


class TestBudget:
    def test_token_budget_under_8k(self):
        b = _bundle(_long_run(5000))  # a very long run
        assert b.approx_tokens() < 8000
