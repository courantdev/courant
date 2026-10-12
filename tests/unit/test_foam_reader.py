"""Unit tests for reading a real OpenFOAM case into CaseConfig (foamlib)."""

import os

from courant.case.config import CaseConfig

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "mini_case")


def test_from_foam_parses_mini_case():
    cfg = CaseConfig.from_foam(FIXTURE)
    assert cfg.solver == "icoFoam"
    assert cfg.delta_t == 0.05
    assert cfg.adjust_timestep is False
    assert cfg.end_time == 10
    assert cfg.is_steady()  # ddtSchemes.default == steadyState


def test_from_foam_reads_schemes_and_relaxation():
    cfg = CaseConfig.from_foam(FIXTURE)
    assert cfg.div_schemes.get("div(phi,U)") == "Gauss linear"
    # nested relaxationFactors (fields/equations) flattened
    assert cfg.relaxation_factors.get("p") == 0.95
    assert cfg.relaxation_factors.get("U") == 0.95
    assert cfg.n_non_orthogonal_correctors == 0


def test_from_foam_reads_transport_and_solvers():
    cfg = CaseConfig.from_foam(FIXTURE)
    assert cfg.transport.get("nu") == 5.0
    assert "p" in cfg.linear_solvers
    assert cfg.linear_solvers["p"].solver == "GAMG"
    assert cfg.linear_solvers["p"].tolerance == 1e-6


def test_from_foam_missing_dir_is_graceful():
    cfg = CaseConfig.from_foam("/nonexistent/case")
    assert isinstance(cfg, CaseConfig)  # empty config, no crash
