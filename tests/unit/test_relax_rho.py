"""T-RELAX-RHO: `rho 1` in fvSolution's relaxationFactors is what v2406's own steady tutorials
ship (tests/fixtures/v2406_rho_relaxation.txt, a grep of the pinned image), so it is no
over-relaxation: a measured false alarm, fixed on by default in 0.7.0 (GATE-13's exception,
named in the changelog). Every other factor at 1 still is."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.diagnosis.checkers import SchemeToleranceChecker, undamped_relaxation
from courant.diagnosis.kb import CONFIG_PREDICATES
from tests.unit.test_checkers import ctx
from tests.unit.test_mass_flux import _unpack_fixture

ROOT = Path(__file__).resolve().parents[2]
LISTED = ROOT / "tests" / "fixtures" / "v2406_rho_relaxation.txt"


def _shipped() -> list[tuple[str, str, float]]:
    rows = []
    for line in LISTED.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        app, path, rho = line.split("|")
        rows.append((app, path, float(rho.split()[1].rstrip(";"))))
    return rows


def test_the_shipped_list_holds_the_tutorials_named():
    rows = _shipped()
    at_one = [r for r in rows if r[2] >= 1.0]
    assert len(at_one) == 16, at_one
    assert sum(1 for r in at_one if r[0] == "buoyantSimpleFoam") == 5


@pytest.mark.parametrize("app,path,rho", [r for r in _shipped() if r[2] >= 1.0])
def test_a_shipped_rho_factor_is_not_over_relaxation(app, path, rho):
    c = CaseConfig(solver=app or "chtMultiRegionSimpleFoam", ddt_scheme="steadyState",
                   relaxation_factors={"rho": rho, "p_rgh": 0.7, "U": 0.3, "h": 0.3},
                   relaxation_scopes={"rho": "fields", "p_rgh": "fields", "U": "equations",
                                      "h": "equations"})
    assert not [f for f in SchemeToleranceChecker().run(ctx(config=c))
                if f.id == "scheme.relaxation"]
    assert not CONFIG_PREDICATES["relaxation_high"](c)


@pytest.mark.parametrize("field", ["p_rgh", "U", "h", "p", "k"])
def test_every_other_factor_at_one_still_is(field):
    factors = {"rho": 1.0, "p_rgh": 0.7, "U": 0.3, "h": 0.3, field: 1.0}
    c = CaseConfig(solver="buoyantSimpleFoam", ddt_scheme="steadyState",
                   relaxation_factors=factors)
    found = [f for f in SchemeToleranceChecker().run(ctx(config=c)) if f.id == "scheme.relaxation"]
    assert found and found[0].field == field, found
    assert CONFIG_PREDICATES["relaxation_high"](c)
    assert set(undamped_relaxation(factors)) == {field}


def test_the_hot_box_solve_is_no_longer_rejected_on_rho_alone(tmp_path):
    """The committed buoyantSimpleFoam solve (T-ENERGY-BALANCE) ships `fields { rho 1; }`: 0.6.0
    rejected it on scheme.relaxation alone (VER-5.4); no such reason now, in either mode."""
    from courant.verify import from_case_dir, verify_inputs

    case = _unpack_fixture(ROOT / "tests/fixtures/real_openfoam/hot_box/converged", tmp_path / "c")
    for checks in ("default", "next"):
        doc = json.loads(verify_inputs(from_case_dir(case, checks=checks)).to_json())
        assert not [r for r in doc["reasons"] if r["code"] == "scheme.relaxation"], doc["reasons"]


def test_the_over_relaxation_cause_does_not_blame_rho():
    """The cause `residual.growth` suggests asks the same test: `rho 1` alone suggests nothing."""
    from courant.diagnosis.checkers import _growth_suggests

    c = CaseConfig(solver="buoyantSimpleFoam", ddt_scheme="steadyState",
                   relaxation_factors={"rho": 1.0, "p_rgh": 0.7, "U": 0.3, "h": 0.3})
    assert _growth_suggests(c) == []
    c = CaseConfig(solver="buoyantSimpleFoam", ddt_scheme="steadyState",
                   relaxation_factors={"rho": 1.0, "p_rgh": 0.7, "U": 0.9, "h": 0.3})
    assert _growth_suggests(c) == ["NUM.OVER_RELAXATION"]


def test_a_region_s_rho_is_not_over_relaxation_either(tmp_path):
    """A multi-region case keys its factors by region (`rho@region1`, VER-3d, under next): the
    exemption is the field's (found by RM-38's harness, 3 Nov 2026: the committed cht_plate solve,
    whose air region ships `rho 1`, was rejected under next on it alone)."""
    import json

    from courant.verify import from_case_dir, verify_inputs

    assert set(undamped_relaxation({"rho@region1": 1.0, "h@region1": 1.0})) == {"h@region1"}
    case = _unpack_fixture(ROOT / "tests/fixtures/real_openfoam/cht_plate/converged", tmp_path / "c")
    for checks in ("default", "next"):
        doc = json.loads(verify_inputs(from_case_dir(case, checks=checks)).to_json())
        assert not [r for r in doc["reasons"] if r["code"] == "scheme.relaxation"], doc["reasons"]
