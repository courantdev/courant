"""The case directory of this fixture, written from scratch (TRD VER-16c).

    python make_case.py <dir>

writes <dir>/run, a differentially heated square cavity for buoyantBoussinesqSimpleFoam;
`run.sh`, in the pinned image (opencfd/openfoam-default:2406, two cores), meshes and solves it.
What is committed beside this file, in `run/`, is its output: the dictionaries, the mesh, the
solver log and the last time directory's fields (each of those three gzipped).

The cavity: side 0.05 m, 24 x 24 cells, one cell deep (front and back empty). The left wall at
305 K, the right at 295 K, top and bottom adiabatic; gravity along -y. Laminar, Boussinesq: nu
5.4e-5 m2/s, Pr 0.71, beta 1/300 per K, so a Rayleigh number of g beta dT L^3 Pr / nu^2, about
1.0e4, and a steady flow. Its `SIMPLE/residualControl` (1e-9 for p_rgh, 1e-10 for U and T) is
set far below what the run reaches in its 200 iterations: a run-to-endTime device, so the solver
prints no "SIMPLE solution converged" and stops at its endTime, with every residual some orders
under the gate's flat bar of 1e-3 (Ux 1.37e-8, Uy 1.50e-8, T 1.54e-8, p_rgh 6.87e-8 at the last
step). Solved once in the image on 26 Oct 2026, in 0.2 s; the same case run on to its tolerances
converges in 299 iterations.
"""
import sys
from pathlib import Path

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  Courant test fixture: a differentially heated cavity, run to its endTime (T-VER16C)
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


L, N, DEPTH = 0.05, 24, 0.005
END = 200

BLOCKMESH = f"""
scale 1;

vertices
(
    (0 0 0) ({L} 0 0) ({L} {L} 0) (0 {L} 0)
    (0 0 {DEPTH}) ({L} 0 {DEPTH}) ({L} {L} {DEPTH}) (0 {L} {DEPTH})
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
    adiabatic
    {{
        type wall;
        faces ((0 1 5 4) (3 7 6 2));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);
"""

CONTROL = f"""
application     buoyantBoussinesqSimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {END};
deltaT          1;
writeControl    timeStep;
writeInterval   {END};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
"""

SCHEMES = """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,T)      bounded Gauss linearUpwind grad(T);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

SOLUTION = """
solvers
{
    p_rgh
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-14;
        relTol          0.01;
    }
    "(U|T)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-14;
        relTol          0.1;
    }
}

SIMPLE
{
    momentumPredictor yes;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;

    // Far below what 200 iterations reach: the run stops at its endTime.
    residualControl
    {
        p_rgh           1e-9;
        U               1e-10;
        T               1e-10;
    }
}

relaxationFactors
{
    fields
    {
        p_rgh           0.3;
    }
    equations
    {
        U               0.7;
        T               0.9;
    }
}
"""

TRANSPORT = """
transportModel  Newtonian;
nu              5.4e-05;
beta            0.0033333333;
TRef            300;
Pr              0.71;
Prt             0.85;
"""

TURBULENCE = """
simulationType  laminar;
"""

G = """
dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);
"""


def _field(cls: str, obj: str, dims: str, internal: str, patches: dict[str, str]) -> str:
    body = f"\ndimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for name, bc in patches.items():
        body += f"    {name}\n    {{\n        {bc}\n    }}\n"
    return body + "}\n"


def main(root: Path) -> None:
    case = root / "run"
    write(case / "system" / "blockMeshDict", "blockMeshDict", BLOCKMESH)
    write(case / "system" / "controlDict", "controlDict", CONTROL)
    write(case / "system" / "fvSchemes", "fvSchemes", SCHEMES)
    write(case / "system" / "fvSolution", "fvSolution", SOLUTION)
    write(case / "constant" / "transportProperties", "transportProperties", TRANSPORT)
    write(case / "constant" / "turbulenceProperties", "turbulenceProperties", TURBULENCE)
    write(case / "constant" / "g", "g", G, cls="uniformDimensionedVectorField",
          loc='location    "constant";\n    ')
    walls = ("hot", "cold", "adiabatic")
    empty = {"frontAndBack": "type empty;"}
    zero = case / "0"
    write(zero / "U", "U", _field("volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)",
                                  {**{w: "type noSlip;" for w in walls}, **empty}),
          cls="volVectorField", loc='location    "0";\n    ')
    write(zero / "T", "T", _field("volScalarField", "T", "[0 0 0 1 0 0 0]", "uniform 300",
                                  {"hot": "type fixedValue;\n        value uniform 305;",
                                   "cold": "type fixedValue;\n        value uniform 295;",
                                   "adiabatic": "type zeroGradient;", **empty}),
          cls="volScalarField", loc='location    "0";\n    ')
    write(zero / "p_rgh", "p_rgh",
          _field("volScalarField", "p_rgh", "[0 2 -2 0 0 0 0]", "uniform 0",
                 {**{w: "type fixedFluxPressure;\n        value uniform 0;" for w in walls},
                  **empty}),
          cls="volScalarField", loc='location    "0";\n    ')
    write(zero / "p", "p",
          _field("volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0",
                 {**{w: "type calculated;\n        value uniform 0;" for w in walls}, **empty}),
          cls="volScalarField", loc='location    "0";\n    ')
    write(zero / "alphat", "alphat",
          _field("volScalarField", "alphat", "[0 2 -1 0 0 0 0]", "uniform 0",
                 {**{w: "type calculated;\n        value uniform 0;" for w in walls}, **empty}),
          cls="volScalarField", loc='location    "0";\n    ')


if __name__ == "__main__":
    main(Path(sys.argv[1]))
