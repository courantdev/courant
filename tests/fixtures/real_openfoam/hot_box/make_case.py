"""The case directories of this fixture, written from scratch (TRD VER-22, T-ENERGY-BALANCE).

    python make_case.py <dir>

writes <dir>/converged and <dir>/early, the same closed square box of air for buoyantSimpleFoam,
each with its own endTime; `run.sh`, in the pinned image (opencfd/openfoam-default:2406, two
cores), meshes and solves both. What is committed beside this file is their output: the
dictionaries, `0/`, the mesh, the solver's log, the last time directory's fields and the
`wallHeatFlux` table (the last four gzipped).

The box: 0.05 m by 0.05 m, one cell deep (0.005 m; front and back empty), 40 x 40 x 1 cells.
`hot` (x = 0, wall): T fixedValue 310 K. `cold` (x = 0.05 m, wall): T fixedValue 290 K. `top`
and `bottom` (walls): T zeroGradient. Every wall noSlip, p_rgh fixedFluxPressure. Started from
rest at 290 K, the cold wall's temperature, and 1e5 Pa, so a run stopped early has taken in heat
that has not yet left. Laminar air: heRhoThermo, pureMixture, const transport (mu 1.8e-5 Pa s,
Pr 0.7), hConst (Cp 1005 J/kg/K), perfectGas, molWeight 28.96, sensibleEnthalpy; g (0 -9.81 0).
The Rayleigh number on the box's side is about 2.5e5. buoyantSimpleFoam, `writeFormat ascii;
writeCompression off;`, no fvOptions, no fvModels and no radiation; `SIMPLE/residualControl`
p_rgh, U and h at 1e-6. One `wallHeatFlux` function object with no `patches` entry, so v2406
reports every wall patch; its field is written at write times only (`writeControl writeTime`),
its table and log lines at every step.

- `converged`: endTime 5000, so the solver stops on its residualControl and its log ends in
  "SIMPLE solution converged in N iterations".
- `early`: endTime 20.

These settings were fixed before any solve, and none was changed after. The record of each
solve is `SOLVED` below.
"""
import sys
from pathlib import Path

#: As solved (2 Nov 2026, pinned image, two cores, one attempt each; no setting changed).
SOLVED = """
converged: "SIMPLE solution converged in 2243 iterations"; the last step's initial residuals Ux
  7.90e-7, Uy 4.94e-7, h 9.97e-7, p_rgh 8.43e-7 (GAMG reached its 1000-iteration cap each step,
  its 1e-12 tolerance being out of reach), in 48.9 s. wallHeatFlux at 2243 (the integral
  column): hot 1.5914856500e-02, cold -1.5905522012e-02, top 0, bottom 0 W; |net| over the heat
  in 5.9e-4.
early: 20 iterations to its endTime; the last step's initial residuals Ux 0.0505, Uy 0.0280,
  h 0.00833, p_rgh 0.0309, in 0.06 s. wallHeatFlux at 20: hot 3.3116898887e-02, cold
  -4.1469029257e-08, top 0, bottom 0 W; |net| over the heat in 0.999.
Each wrote its last time only (2243 and 20); the field `wallHeatFlux` the function object wrote
there, and `uniform/`, are not committed.
"""

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  Courant test fixture: a closed heated box of air, buoyantSimpleFoam (T-ENERGY-BALANCE)
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    {loc}object      {obj};
}}
"""


def write(path: Path, obj: str, body: str, cls: str = "dictionary", loc: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=obj, loc=loc) + body)


L, DZ = 0.05, 0.005
N = 40
#: Each run's endTime: the converged run stops on its residualControl before 5000.
END = {"converged": 5000, "early": 20}

BLOCKMESH = f"""
scale 1;

vertices
(
    (0 0 0) ({L} 0 0) ({L} {L} 0) (0 {L} 0)
    (0 0 {DZ}) ({L} 0 {DZ}) ({L} {L} {DZ}) (0 {L} {DZ})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({N} {N} 1) simpleGrading (1 1 1)
);

boundary
(
    hot
    {{
        type wall;
        faces ((0 4 7 3));
    }}
    cold
    {{
        type wall;
        faces ((1 2 6 5));
    }}
    top
    {{
        type wall;
        faces ((3 7 6 2));
    }}
    bottom
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);
"""


def control(end: int) -> str:
    return f"""
application     buoyantSimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          1;
writeControl    timeStep;
writeInterval   {end};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;

functions
{{
    wallHeatFlux
    {{
        type            wallHeatFlux;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
    }}
}}
"""


SCHEMES = """
ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,h)      bounded Gauss linearUpwind grad(h);
    div(phi,K)      bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""

SOLUTION = """
solvers
{
    p_rgh
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-12;
        relTol          0.01;
    }
    "(U|h)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-12;
        relTol          0.1;
    }
}

SIMPLE
{
    momentumPredictor yes;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       100000;

    residualControl
    {
        p_rgh           1e-6;
        U               1e-6;
        h               1e-6;
    }
}

relaxationFactors
{
    fields
    {
        rho             1;
        p_rgh           0.7;
    }
    equations
    {
        U               0.5;
        h               0.5;
    }
}
"""

THERMO = """
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie
    {
        molWeight   28.96;
    }
    thermodynamics
    {
        Cp          1005;
        Hf          0;
    }
    transport
    {
        mu          1.8e-05;
        Pr          0.7;
    }
}
"""

TURBULENCE = """
simulationType  laminar;
"""

GRAVITY = """
dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);
"""

WALLS = ("hot", "cold", "top", "bottom")


def _field(dims: str, internal: str, patches: dict[str, str]) -> str:
    body = f"\ndimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for name, bc in patches.items():
        body += f"    {name}\n    {{\n        {bc}\n    }}\n"
    return body + "}\n"


def write_case(case: Path, end: int) -> None:
    write(case / "system" / "blockMeshDict", "blockMeshDict", BLOCKMESH)
    write(case / "system" / "controlDict", "controlDict", control(end))
    write(case / "system" / "fvSchemes", "fvSchemes", SCHEMES)
    write(case / "system" / "fvSolution", "fvSolution", SOLUTION)
    write(case / "constant" / "thermophysicalProperties", "thermophysicalProperties", THERMO)
    write(case / "constant" / "turbulenceProperties", "turbulenceProperties", TURBULENCE)
    write(case / "constant" / "g", "g", GRAVITY, cls="uniformDimensionedVectorField")
    empty = {"frontAndBack": "type empty;"}
    zero = case / "0"
    loc = 'location    "0";\n    '
    write(zero / "U", "U", _field("[0 1 -1 0 0 0 0]", "uniform (0 0 0)", {
        **{w: "type noSlip;" for w in WALLS}, **empty}), cls="volVectorField", loc=loc)
    write(zero / "T", "T", _field("[0 0 0 1 0 0 0]", "uniform 290", {
        "hot": "type fixedValue;\n        value uniform 310;",
        "cold": "type fixedValue;\n        value uniform 290;",
        "top": "type zeroGradient;",
        "bottom": "type zeroGradient;", **empty}), cls="volScalarField", loc=loc)
    write(zero / "p", "p", _field("[1 -1 -2 0 0 0 0]", "uniform 100000", {
        **{w: "type calculated;\n        value uniform 100000;" for w in WALLS}, **empty}),
        cls="volScalarField", loc=loc)
    write(zero / "p_rgh", "p_rgh", _field("[1 -1 -2 0 0 0 0]", "uniform 100000", {
        **{w: "type fixedFluxPressure;\n        value uniform 100000;" for w in WALLS},
        **empty}), cls="volScalarField", loc=loc)


def main(root: Path) -> None:
    for name, end in END.items():
        write_case(root / name, end)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
