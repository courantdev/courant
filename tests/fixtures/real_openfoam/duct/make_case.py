"""The case directories of this fixture, written from scratch (TRD VER-18, T-MASSFLUX-2).

    python make_case.py <dir>

writes <dir>/converged and <dir>/early, the same straight 2-D channel for rhoSimpleFoam, each
with its own endTime; `run.sh`, in the pinned image (opencfd/openfoam-default:2406, two cores),
meshes and solves both, then writes the converged run's `phi` once more in binary
(`foamFormatConvert`). What is committed beside this file is their output: in `converged/` and
`early/` the dictionaries, `0/`, the mesh, the solver's log and the last time directory's fields
(the last three gzipped); in `binary/<t>/phi.gz` the converged run's `phi` as OpenFOAM wrote it
binary, and nothing else of that copy.

The channel: 0.2 m long, 0.02 m high, one cell deep (0.002 m; front and back empty), 50 x 10 x 1
cells. `inlet` (x = 0): U fixedValue (1 0 0), T fixedValue 300 K, p zeroGradient. `outlet`
(x = 0.2 m): p fixedValue 1e5 Pa, U zeroGradient, T inletOutlet (300 K on any face taking flow
in). `walls` (y = 0 and y = 0.02 m): noSlip, T and p zeroGradient. Started from rest (U 0), at
300 K and 1e5 Pa. Laminar. Air as a perfect gas: hePsiThermo, pureMixture, const transport (mu
1.8e-5 Pa s, Pr 0.7), hConst (Cp 1005 J/kg/K), molWeight 28.96, sensibleEnthalpy; so a density
of about 1.16 kg/m3 and a Reynolds number on the height of about 1,290. rhoSimpleFoam, `writeFormat
ascii; writeCompression off;`, no fvOptions and no fvModels; `SIMPLE/residualControl` p, U and h
at 1e-5 in both runs.

- `converged`: endTime 3000, so the solver stops on its residualControl and its log ends in
  "SIMPLE solution converged in N iterations".
- `early`: endTime 20, where residualControl is out of reach: a steady compressible run stopped
  before it converged, the shape of the run that prompted VER-18.

These settings were fixed before any solve, and none was changed after: the first attempt
reached its residualControl.

As solved (2 Nov 2026, pinned image, two cores, one attempt each): `converged` stopped on its
residualControl, "SIMPLE solution converged in 120 iterations", its last step's initial residuals
Ux 5.07e-7, Uy 5.98e-6, h 3.85e-7, p 9.10e-6, in 0.11 s; `early` ran its 20 iterations to its
endTime, the last step's initial residuals Ux 0.0583, Uy 0.143, h 0.0577, p 0.151, in 0.03 s.
Each wrote its last time only (120 and 20). OpenFOAM's own `flowRatePatch` sums of these `phi`
files are `../flow_rates/duct_converged.log` and `../flow_rates/duct_early.log`: inlet
-4.644127065e-05 and outlet 4.64412723e-05 kg/s at 120, inlet -4.644137675e-05 and outlet
4.642753676e-05 kg/s at 20, the walls 0 in both. `binary/120/phi.gz` is `foamFormatConvert
-latestTime` of a copy of `converged` with `writeFormat binary` (run.sh).
"""
import sys
from pathlib import Path

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  Courant test fixture: a compressible laminar channel, rhoSimpleFoam (T-MASSFLUX-2)
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


LX, LY, DZ = 0.2, 0.02, 0.002
NX, NY = 50, 10
#: Each run's endTime: the converged run stops on its residualControl well before 3000.
END = {"converged": 3000, "early": 20}

BLOCKMESH = f"""
scale 1;

vertices
(
    (0 0 0) ({LX} 0 0) ({LX} {LY} 0) (0 {LY} 0)
    (0 0 {DZ}) ({LX} 0 {DZ}) ({LX} {LY} {DZ}) (0 {LY} {DZ})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NY} 1) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    walls
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


def control(end: int) -> str:
    return f"""
application     rhoSimpleFoam;
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
"""


SCHEMES = """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,h)      bounded Gauss upwind;
    div(phi,K)      bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

SOLUTION = """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-12;
        relTol          0.05;
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
    nNonOrthogonalCorrectors 0;
    pMinFactor      0.5;
    pMaxFactor      1.5;

    residualControl
    {
        p               1e-5;
        U               1e-5;
        h               1e-5;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        h               0.7;
    }
}
"""

THERMO = """
thermoType
{
    type            hePsiThermo;
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
    empty = {"frontAndBack": "type empty;"}
    zero = case / "0"
    loc = 'location    "0";\n    '
    write(zero / "U", "U", _field("[0 1 -1 0 0 0 0]", "uniform (0 0 0)", {
        "inlet": "type fixedValue;\n        value uniform (1 0 0);",
        "outlet": "type zeroGradient;",
        "walls": "type noSlip;", **empty}), cls="volVectorField", loc=loc)
    write(zero / "T", "T", _field("[0 0 0 1 0 0 0]", "uniform 300", {
        "inlet": "type fixedValue;\n        value uniform 300;",
        "outlet": ("type inletOutlet;\n        inletValue uniform 300;\n"
                   "        value uniform 300;"),
        "walls": "type zeroGradient;", **empty}), cls="volScalarField", loc=loc)
    write(zero / "p", "p", _field("[1 -1 -2 0 0 0 0]", "uniform 100000", {
        "inlet": "type zeroGradient;",
        "outlet": "type fixedValue;\n        value uniform 100000;",
        "walls": "type zeroGradient;", **empty}), cls="volScalarField", loc=loc)


def main(root: Path) -> None:
    for name, end in END.items():
        write_case(root / name, end)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
