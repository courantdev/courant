"""`bc.overconstrained` asks the pressure the solver solves (TRD PRE-5).

On interFoam and the buoyant solvers the solved pressure is `p_rgh`; their `0/p` is `calculated`
or absent. The check read a field named `p` only, so a patch fixing both `p_rgh` and `U` was
never flagged: the hard-proofing audit (21 Oct 2026) set damBreak's `atmosphere` to fixedValue in
both `0/p_rgh` and `0/U` and preflight said NO_SETUP_FAULT_FOUND, where the same mistake on
simpleFoam is SETUP_FAULT. It also counted only `fixedValue` as fixing the pressure, where
`totalPressure`, `prghPressure` and `prghTotalPressure` fix it as surely.

**Behind `--checks next` until 0.6.0** (GATE-13: a WARN finding turns an accept into a reject).
The cases are synthetic, in the shape of v2406's damBreak.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.diagnosis.checkers import BoundaryConfigChecker, CheckContext
from courant.verify.preflight import preflight_path

_HEAD = "FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n"


def _foam(cls: str, obj: str, body: str) -> str:
    return _HEAD.format(cls=cls, obj=obj) + body


def dambreak(tmp_path: Path, *, atmosphere_p_rgh: str, atmosphere_u: str) -> Path:
    """A damBreak-shaped interFoam case: walls, and an open `atmosphere` patch."""
    case = tmp_path / "damBreak"
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True, exist_ok=True)
    (case / "system/controlDict").write_text(_foam("dictionary", "controlDict", (
        "application interFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n"
        "endTime 1;\ndeltaT 0.001;\nwriteControl adjustable;\nwriteInterval 0.05;\n"
        "adjustTimeStep yes;\nmaxCo 1;\nmaxAlphaCo 1;\nmaxDeltaT 1;\n")))
    (case / "system/fvSchemes").write_text(_foam("dictionary", "fvSchemes", (
        "ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\n"
        "divSchemes { div(rhoPhi,U) Gauss linearUpwind grad(U); }\n"
        "laplacianSchemes { default Gauss linear corrected; }\n")))
    (case / "system/fvSolution").write_text(_foam("dictionary", "fvSolution", (
        "solvers\n{\n    p_rgh { solver PCG; preconditioner DIC; tolerance 1e-07; "
        "relTol 0.05; }\n}\nPIMPLE { nOuterCorrectors 1; nCorrectors 3; "
        "nNonOrthogonalCorrectors 0; }\n")))
    (case / "constant/transportProperties").write_text(_foam("dictionary", "transportProperties", (
        "phases (water air);\nwater { transportModel Newtonian; nu 1e-06; rho 1000; }\n"
        "air { transportModel Newtonian; nu 1.48e-05; rho 1; }\nsigma 0.07;\n")))
    (case / "constant/turbulenceProperties").write_text(
        _foam("dictionary", "turbulenceProperties", "simulationType laminar;\n"))
    walls = "    leftWall { type %s; }\n    rightWall { type %s; }\n    lowerWall { type %s; }\n"
    (case / "0/U").write_text(_foam("volVectorField", "U", (
        "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n"
        + walls % ("noSlip", "noSlip", "noSlip")
        + f"    atmosphere {{ {atmosphere_u} }}\n    defaultFaces {{ type empty; }}\n}}\n")))
    (case / "0/p_rgh").write_text(_foam("volScalarField", "p_rgh", (
        "dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
        + walls % ("fixedFluxPressure", "fixedFluxPressure", "fixedFluxPressure")
        + f"    atmosphere {{ {atmosphere_p_rgh} }}\n    defaultFaces {{ type empty; }}\n}}\n")))
    (case / "0/alpha.water").write_text(_foam("volScalarField", "alpha.water", (
        "dimensions [0 0 0 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n"
        + walls % ("zeroGradient", "zeroGradient", "zeroGradient")
        + "    atmosphere { type inletOutlet; inletValue uniform 0; value uniform 0; }\n"
        "    defaultFaces { type empty; }\n}\n")))
    return case


FIXED_P_RGH = "type fixedValue; value uniform 0;"
FIXED_U = "type fixedValue; value uniform (0 0 0);"
#: v2406's damBreak as shipped.
TOTAL_P_RGH = "type totalPressure; p0 uniform 0;"
OPEN_U = "type pressureInletOutletVelocity; value uniform (0 0 0);"


def _found(config: CaseConfig, checks: str):
    return [f for f in BoundaryConfigChecker().run(CheckContext(
        run_id="r", events=[], config=config, checks=checks)) if f.id == "bc.overconstrained"]


# ------------------------------------------------------------------ the checker
@pytest.mark.parametrize("pressure,fields", [
    ("p_rgh", {"U": "fixedValue", "p_rgh": "fixedValue", "alpha.water": "inletOutlet"}),
    ("p_rgh", {"U": "fixedValue", "p_rgh": "prghPressure"}),
    ("p_rgh", {"U": "fixedValue", "p_rgh": "prghTotalPressure"}),
    ("p", {"U": "fixedValue", "p": "totalPressure"}),
])
def test_next_flags_a_patch_fixing_the_solved_pressure_and_the_velocity(pressure, fields):
    cfg = CaseConfig(boundary={"atmosphere": fields, "wall": {"U": "noSlip"}})
    (f,) = _found(cfg, "next")
    assert f.location == "atmosphere" and pressure in f.detail
    assert _found(cfg, "default") == []                 # v0.5.0 read `p` fixedValue only


def test_the_p_check_v050_had_is_kept_in_both_modes():
    cfg = CaseConfig(boundary={"outlet": {"U": "fixedValue", "p": "fixedValue"}})
    assert [f.location for f in _found(cfg, "default")] == ["outlet"]
    assert [f.location for f in _found(cfg, "next")] == ["outlet"]


@pytest.mark.parametrize("fields", [
    {"U": "pressureInletOutletVelocity", "p_rgh": "totalPressure"},    # damBreak's atmosphere
    {"U": "noSlip", "p_rgh": "fixedFluxPressure"},                     # a wall
    {"U": "fixedValue", "p_rgh": "fixedFluxPressure"},                 # an inlet
    {"U": "fixedValue", "p": "calculated", "p_rgh": "fixedFluxPressure"},
])
def test_a_well_posed_patch_is_not_flagged(fields):
    assert _found(CaseConfig(boundary={"x": fields}), "next") == []


# ------------------------------------------------------------------ preflight, end to end
def test_the_audits_dambreak_is_a_setup_fault_under_next(tmp_path):
    case = dambreak(tmp_path, atmosphere_p_rgh=FIXED_P_RGH, atmosphere_u=FIXED_U)
    assert CaseConfig.from_foam(str(case)).boundary["atmosphere"] == {
        "U": "fixedValue", "p_rgh": "fixedValue", "alpha.water": "inletOutlet"}
    nxt = preflight_path(case, checks="next")
    assert nxt.verdict == "SETUP_FAULT"
    assert {r.code: r.input for r in nxt.reasons} == {
        "bc.overconstrained": "0/ boundary conditions (atmosphere)"}
    assert preflight_path(case).verdict == "NO_SETUP_FAULT_FOUND"     # v0.5.0


def test_the_shipped_dambreak_is_not_flagged(tmp_path):
    case = dambreak(tmp_path, atmosphere_p_rgh=TOTAL_P_RGH, atmosphere_u=OPEN_U)
    for mode in ("default", "next"):
        assert preflight_path(case, checks=mode).verdict == "NO_SETUP_FAULT_FOUND", mode


def test_the_knowledge_base_predicate_and_the_checker_read_one_definition():
    from courant.diagnosis.kb import CONFIG_PREDICATES

    over = CaseConfig(boundary={"outlet": {"U": "fixedValue", "p": "fixedValue"}})
    prgh = CaseConfig(boundary={"atmosphere": {"U": "fixedValue", "p_rgh": "fixedValue"}})
    assert CONFIG_PREDICATES["overconstrained_pu"](over) is True
    # The predicate ranks causes in every mode with v0.5.0's reading; under `next` the
    # finding itself is the evidence that scores BC.OVERCONSTRAINED_PU.
    assert CONFIG_PREDICATES["overconstrained_pu"](prgh) is False
    assert over.overconstrained_patches() == ["outlet"]
    assert prgh.overconstrained_patches("next") == ["atmosphere"]
