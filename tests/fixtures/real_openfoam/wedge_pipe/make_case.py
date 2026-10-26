"""The case directories of this fixture, written from scratch (TRD VER-7d, VER-13a, VER-16b).

    python make_case.py <dir>

writes <dir>/flow (simpleFoam) and <dir>/energy (scalarTransportFoam on the flow's velocity,
`ddtSchemes default steadyState`); `run.sh`, in the pinned image (opencfd/openfoam-default:2406,
two cores), meshes, checks and solves them. What is committed beside this file is their output:
the dictionaries, the mesh (gzipped), checkMesh's log, the solver logs (gzipped) and the last
time directory of each run. `inverted/log.checkMesh.gz` is `checkMesh -allGeometry -allTopology`
on the same mesh with one corner (both its wedge-plane copies, points 1364 and 3889) moved half a
diagonal past the opposite corner of its cell, which turns that cell inside out.
`skewed/` is the same mesh with one vertex (both its wedge-plane copies, points 1262 and 3787, at
mid-length on radial level 12) moved 0.9 of a cell length (1.8 mm) along the axis, which skews one
face with an area past 4 and inverts nothing: its points as checkMesh read them
(`constant/polyMesh/points.gz`, written by `courant.case.polymesh.write_polymesh`), and
`checkMesh -allGeometry -allTopology`'s log and `skewFaces` set on it, from the pinned image.

The pipe: radius 5 mm, length 0.2 m, a 5 degree wedge (2.5 either side of the x-y plane), axis
along x, 100 x 24 cells graded towards the wall. Its axis is two vertex pairs that coincide
without being merged, so blockMesh writes the axis as 100 faces of zero area, typed `empty`.
Laminar, nu 1e-5 m2/s, a uniform 0.05 m/s inlet (Re 50 on the diameter); for the energy run an
inlet at 300 K and a wall at 320 K.
"""
import math
import sys
from pathlib import Path

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  Courant test fixture: a laminar axisymmetric pipe on a wedge (T-WEDGE-MESH)
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


R, L, HALF = 0.005, 0.2, math.radians(2.5)
NX, NR, GRADE = 100, 24, "0.25"
Y, Z = R * math.cos(HALF), R * math.sin(HALF)

BLOCKMESH = f"""
scale 1;

// The axis is vertices 0 and 4 (inlet) and 1 and 5 (outlet): each pair at one point, kept as
// two vertices, so the axis faces have zero area. They are typed empty.
vertices
(
    (0 0 0)
    ({L} 0 0)
    ({L} {Y:.12g} {-Z:.12g})
    (0 {Y:.12g} {-Z:.12g})
    (0 0 0)
    ({L} 0 0)
    ({L} {Y:.12g} {Z:.12g})
    (0 {Y:.12g} {Z:.12g})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {NR} 1) simpleGrading (1 {GRADE} 1)
);

boundary
(
    inlet  {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    wall   {{ type wall;  faces ((3 7 6 2)); }}
    front  {{ type wedge; faces ((4 5 6 7)); }}
    back   {{ type wedge; faces ((0 3 2 1)); }}
    axis   {{ type empty; faces ((0 1 5 4)); }}
);
"""

CONTROL = """
application     {app};
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end};
deltaT          1;
writeControl    timeStep;
writeInterval   {every};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
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
    div(phi,T)      bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

SOLUTION = """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0.05; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-09; relTol 0.1; }
    T { solver PBiCGStab; preconditioner DILU; tolerance 1e-09; relTol 0; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl { p 1e-6; U 1e-7; T 1e-7; }
}
relaxationFactors
{
    equations { U 0.9; T 0.9; }
}
"""

TRANSPORT = """
transportModel  Newtonian;
nu              1e-05;
DT              1.4e-05;
"""

TURBULENCE = """
simulationType  laminar;
"""


def field(name: str, cls: str, dims: str, internal: str, patches: dict[str, str]) -> str:
    body = f"\ndimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for patch, bc in patches.items():
        body += f"    {patch}\n    {{\n        {bc}\n    }}\n"
    return body + "}\n"


CONSTRAINED = {"front": "type wedge;", "back": "type wedge;", "axis": "type empty;"}


def case(d: Path, app: str, end: int, every: int) -> None:
    write(d / "system/blockMeshDict", "blockMeshDict", BLOCKMESH)
    write(d / "system/controlDict", "controlDict", CONTROL.format(app=app, end=end, every=every))
    write(d / "system/fvSchemes", "fvSchemes", SCHEMES)
    write(d / "system/fvSolution", "fvSolution", SOLUTION)
    write(d / "constant/transportProperties", "transportProperties", TRANSPORT)
    write(d / "constant/turbulenceProperties", "turbulenceProperties", TURBULENCE)
    write(d / "0/U", "U", field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)", {
        "inlet": "type fixedValue; value uniform (0.05 0 0);", "outlet": "type zeroGradient;",
        "wall": "type noSlip;", **CONSTRAINED}), cls="volVectorField")
    write(d / "0/p", "p", field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "uniform 0", {
        "inlet": "type zeroGradient;", "outlet": "type fixedValue; value uniform 0;",
        "wall": "type zeroGradient;", **CONSTRAINED}), cls="volScalarField")
    write(d / "0/T", "T", field("T", "volScalarField", "[0 0 0 1 0 0 0]", "uniform 300", {
        "inlet": "type fixedValue; value uniform 300;", "outlet": "type zeroGradient;",
        "wall": "type fixedValue; value uniform 320;", **CONSTRAINED}), cls="volScalarField")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    case(root / "flow", "simpleFoam", 1000, 1000)
    case(root / "energy", "scalarTransportFoam", 1500, 500)
