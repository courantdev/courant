"""The case directories of this fixture, written from scratch (TRD VER-22, T-ENERGY-BALANCE).

    python make_case.py <dir>

writes <dir>/converged and <dir>/early, the same two-region case for chtMultiRegionSimpleFoam,
each with its own endTime; `run.sh`, in the pinned image (opencfd/openfoam-default:2406, two
cores), meshes both (blockMesh with two cell zones, then `splitMeshRegions -cellZones
-overwrite`) and solves them. What is committed beside this file is their output: the
dictionaries, `0/<region>/`, each region's mesh, the solver's log, the last time directory's
fields and each region's `wallHeatFlux` table (the last four gzipped).

The case: a solid `plate` 0.05 m by 0.005 m under a fluid `air` channel 0.05 m by 0.02 m, one cell
deep (0.005 m; front and back empty), 50 cells along x, 5 across the plate and 20 across the
channel. `constant/regionProperties`: fluid (air), solid (plate). splitMeshRegions names the
interface `plate_to_air` (in `plate`) and `air_to_plate` (in `air`), each a `mappedWall` naming the
other; it deletes the patches a region has no face of.

- `plate`: `heater` (y = 0, wall) T fixedValue 350 K; `plate_left` and `plate_right` (walls) T
  zeroGradient. heSolidThermo, pureMixture, constIso (kappa 200 W/m/K), hConst (Cp 900 J/kg/K),
  rhoConst (rho 2700 kg/m3), sensibleEnthalpy.
- `air`: `inlet` (x = 0, patch) U fixedValue (0.5 0 0), T fixedValue 300 K, p_rgh
  fixedFluxPressure; `outlet` (x = 0.05 m, patch) p_rgh fixedValue 1e5 Pa, U and T inletOutlet;
  `top` (y = 0.025 m, wall) noSlip, T zeroGradient. Laminar air as in `../hot_box`: heRhoThermo,
  pureMixture, const transport (mu 1.8e-5 Pa s, Pr 0.7), hConst (Cp 1005 J/kg/K), perfectGas,
  molWeight 28.96, sensibleEnthalpy; g (0 -9.81 0).
- The interface: T `compressible::turbulentTemperatureCoupledBaffleMixed` on both sides
  (kappaMethod solidThermo in the plate, fluidThermo in the air); U noSlip on the air side.
- Initial T 300 K in both regions, U 0, p 1e5 Pa.
- One `wallHeatFlux` function object per region (`region plate;`, `region air;`), with no
  `patches` entry, so v2406 reports every wall patch of the region; its field is written at write
  times only, its table and log lines at every step.
- chtMultiRegionSimpleFoam reads no residualControl (TRD VER-21), so it runs to endTime:
  `converged` 3000, `early` 10.

`writeFormat ascii; writeCompression off;`, no fvOptions, no fvModels and no radiation. These
settings were fixed before any solve, and none was changed after. The record of each solve
is `SOLVED` below.
"""
import sys
from pathlib import Path

#: As solved (2 Nov 2026, pinned image, two cores, one attempt each; no setting changed).
SOLVED = """
splitMeshRegions made `plate` (250 cells: heater, plate_left, plate_right, frontAndBack,
  plate_to_air) and `air` (1000 cells: inlet, outlet, top, frontAndBack, air_to_plate), each
  interface patch a mappedWall naming the other, and deleted the patches a region has none of.
converged: 3000 iterations to its endTime; the last step's initial residuals, air Ux 1.84e-10,
  Uy 1.24e-9, h 1.15e-11, p_rgh 1.19e-8 (GAMG at its 1000-iteration cap), plate h 3.59e-12, in
  186 s. wallHeatFlux at 3000: plate heater 1.6264992763e-01, plate_left 0, plate_right 0,
  plate_to_air -1.6264992763e-01 W; air top 0, air_to_plate 1.6264992763e-01 W; the plate's
  balance and the interface's 0 at the digits written.
early: 10 iterations to its endTime; the last step's initial residuals, air Ux 0.0508, Uy 0.0894,
  h 0.0879, p_rgh 0.0573, plate h 0.0468, in 0.05 s. wallHeatFlux at 10: plate heater
  6.1363290243e+02, plate_left 0, plate_right 0, plate_to_air -1.2847946579e-01 W; air top 0,
  air_to_plate 1.0121296230e-01 W; the plate's |net| over the heat in 0.9998, the interface's
  |q_a + q_b| over the larger side 0.212.
Each wrote its last time only (3000 and 10); the field `wallHeatFlux` the function objects wrote
there, `uniform/`, the whole mesh blockMesh made (`constant/polyMesh`), `constant/cellToRegion`
and the region meshes' zone and addressing files are not committed.
"""

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  Courant test fixture: a heated plate under an air channel, chtMultiRegionSimpleFoam
  (T-ENERGY-BALANCE)
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


LX, HP, HA, DZ = 0.05, 0.005, 0.02, 0.005
NX, NP, NA = 50, 5, 20
#: Each run's endTime (chtMultiRegionSimpleFoam reads no residualControl).
END = {"converged": 3000, "early": 10}
Y1, Y2 = HP, HP + HA

BLOCKMESH = f"""
scale 1;

vertices
(
    (0 0 0) ({LX} 0 0) ({LX} {Y1} 0) (0 {Y1} 0)
    (0 0 {DZ}) ({LX} 0 {DZ}) ({LX} {Y1} {DZ}) (0 {Y1} {DZ})
    ({LX} {Y2} 0) (0 {Y2} 0) ({LX} {Y2} {DZ}) (0 {Y2} {DZ})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) plate ({NX} {NP} 1) simpleGrading (1 1 1)
    hex (3 2 8 9 7 6 10 11) air ({NX} {NA} 1) simpleGrading (1 1 1)
);

boundary
(
    heater
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    plate_left
    {{
        type wall;
        faces ((0 4 7 3));
    }}
    plate_right
    {{
        type wall;
        faces ((1 2 6 5));
    }}
    inlet
    {{
        type patch;
        faces ((3 7 11 9));
    }}
    outlet
    {{
        type patch;
        faces ((2 8 10 6));
    }}
    top
    {{
        type wall;
        faces ((9 11 10 8));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7) (3 9 8 2) (7 6 10 11));
    }}
);
"""


def control(end: int) -> str:
    return f"""
application     chtMultiRegionSimpleFoam;
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
    wallHeatFlux_plate
    {{
        type            wallHeatFlux;
        libs            (fieldFunctionObjects);
        region          plate;
        writeControl    writeTime;
    }}
    wallHeatFlux_air
    {{
        type            wallHeatFlux;
        libs            (fieldFunctionObjects);
        region          air;
        writeControl    writeTime;
    }}
}}
"""


#: The top level's schemes and solution, read by blockMesh and splitMeshRegions only.
EMPTY_SCHEMES = """
ddtSchemes {}
gradSchemes {}
divSchemes {}
laplacianSchemes {}
interpolationSchemes {}
snGradSchemes {}
"""

EMPTY_SOLUTION = """
"""

AIR_SCHEMES = """
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

PLATE_SCHEMES = """
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

AIR_SOLUTION = """
solvers
{
    rho
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0;
    }
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
        h               0.7;
    }
}
"""

PLATE_SOLUTION = """
solvers
{
    h
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-12;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    equations
    {
        h               0.7;
    }
}
"""

REGIONS = """
regions
(
    fluid       (air)
    solid       (plate)
);
"""

AIR_THERMO = """
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

PLATE_THERMO = """
thermoType
{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIso;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie
    {
        molWeight   26.98;
    }
    thermodynamics
    {
        Cp          900;
        Hf          0;
    }
    transport
    {
        kappa       200;
    }
    equationOfState
    {
        rho         2700;
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


def _coupled(method: str) -> str:
    return ("type            compressible::turbulentTemperatureCoupledBaffleMixed;\n"
            "        Tnbr            T;\n"
            f"        kappaMethod     {method};\n"
            "        value           uniform 300;")


def _field(dims: str, internal: str, patches: dict[str, str]) -> str:
    body = f"\ndimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for name, bc in patches.items():
        body += f"    {name}\n    {{\n        {bc}\n    }}\n"
    return body + "}\n"


def write_case(case: Path, end: int) -> None:
    write(case / "system" / "blockMeshDict", "blockMeshDict", BLOCKMESH)
    write(case / "system" / "controlDict", "controlDict", control(end))
    write(case / "system" / "fvSchemes", "fvSchemes", EMPTY_SCHEMES)
    write(case / "system" / "fvSolution", "fvSolution", EMPTY_SOLUTION)
    write(case / "system" / "air" / "fvSchemes", "fvSchemes", AIR_SCHEMES)
    write(case / "system" / "air" / "fvSolution", "fvSolution", AIR_SOLUTION)
    write(case / "system" / "plate" / "fvSchemes", "fvSchemes", PLATE_SCHEMES)
    write(case / "system" / "plate" / "fvSolution", "fvSolution", PLATE_SOLUTION)
    write(case / "constant" / "regionProperties", "regionProperties", REGIONS)
    write(case / "constant" / "g", "g", GRAVITY, cls="uniformDimensionedVectorField")
    write(case / "constant" / "air" / "thermophysicalProperties", "thermophysicalProperties",
          AIR_THERMO)
    write(case / "constant" / "air" / "turbulenceProperties", "turbulenceProperties",
          TURBULENCE)
    write(case / "constant" / "plate" / "thermophysicalProperties", "thermophysicalProperties",
          PLATE_THERMO)

    empty = {"frontAndBack": "type empty;"}
    loc = 'location    "0";\n    '
    air = case / "0" / "air"
    walls = ("top", "air_to_plate")
    write(air / "U", "U", _field("[0 1 -1 0 0 0 0]", "uniform (0 0 0)", {
        "inlet": "type fixedValue;\n        value uniform (0.5 0 0);",
        "outlet": ("type inletOutlet;\n        inletValue uniform (0 0 0);\n"
                   "        value uniform (0 0 0);"),
        **{w: "type noSlip;" for w in walls}, **empty}), cls="volVectorField", loc=loc)
    write(air / "T", "T", _field("[0 0 0 1 0 0 0]", "uniform 300", {
        "inlet": "type fixedValue;\n        value uniform 300;",
        "outlet": ("type inletOutlet;\n        inletValue uniform 300;\n"
                   "        value uniform 300;"),
        "top": "type zeroGradient;",
        "air_to_plate": _coupled("fluidThermo"), **empty}), cls="volScalarField", loc=loc)
    write(air / "p", "p", _field("[1 -1 -2 0 0 0 0]", "uniform 100000", {
        **{p: "type calculated;\n        value uniform 100000;"
           for p in ("inlet", "outlet", *walls)}, **empty}), cls="volScalarField", loc=loc)
    write(air / "p_rgh", "p_rgh", _field("[1 -1 -2 0 0 0 0]", "uniform 100000", {
        "inlet": "type fixedFluxPressure;\n        value uniform 100000;",
        "outlet": "type fixedValue;\n        value uniform 100000;",
        **{w: "type fixedFluxPressure;\n        value uniform 100000;" for w in walls},
        **empty}), cls="volScalarField", loc=loc)

    plate = case / "0" / "plate"
    write(plate / "T", "T", _field("[0 0 0 1 0 0 0]", "uniform 300", {
        "heater": "type fixedValue;\n        value uniform 350;",
        "plate_left": "type zeroGradient;",
        "plate_right": "type zeroGradient;",
        "plate_to_air": _coupled("solidThermo"), **empty}), cls="volScalarField", loc=loc)
    write(plate / "p", "p", _field("[1 -1 -2 0 0 0 0]", "uniform 100000", {
        **{p: "type calculated;\n        value uniform 100000;"
           for p in ("heater", "plate_left", "plate_right", "plate_to_air")}, **empty}),
        cls="volScalarField", loc=loc)


def main(root: Path) -> None:
    for name, end in END.items():
        write_case(root / name, end)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
