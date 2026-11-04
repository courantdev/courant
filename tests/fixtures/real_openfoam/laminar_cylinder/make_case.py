"""The case directory of this fixture, written from scratch (TRD VER-5b, T-LAMINAR-REGIME).

    python make_case.py <dir>

writes <dir>/system, constant and 0; `run.sh`, in the pinned image (opencfd/openfoam-default:2406,
two cores), meshes it, decomposes it over two ranks, solves it with pimpleFoam in parallel and
reconstructs the last time. What is committed beside this file is their output: the
dictionaries, the case's mesh and each rank's (gzipped), each rank's `0/`, the reconstructed last
time (gzipped), and the solver's log (gzipped); the ranks' later time directories are not kept.

The flow is v2406's own laminar cylinder at Re 100 (`incompressible/pimpleFoam/laminar/
cylinder2D`): its diameter 0.12 m, its inflow 0.012984 m/s, its nu 1.558e-5 m2/s, its
`fvSchemes` (Euler, Gauss linear) and its `fvSolution` (GAMG p, smoothSolver U, PIMPLE with two
correctors), written out here. The rest is this fixture's: a coarse O-grid of the plane around
the cylinder (4 blocks of 16 x 30 cells, 1,920 cells, the first cell D/20 thick, the far field
15 D away), its upstream half the inlet and its downstream half the outlet; the inflow turned 5
degrees for its first 50 s (back to straight by 75 s) so the wake leaves symmetry and sheds; and a
time step adjusted to a maximum Courant number of 0.9 (the tutorial fixes it, at about 0.25),
which ends each step's residuals above the flat 1e-3 bar while the wake sheds at Re 100.

As solved (26 Oct 2026, pinned image, two ranks): 1,487 steps to 600 s, the largest Courant
number 0.895; the last step's initial residuals Ux 9.4e-4, Uy 9.0e-3, p 2.1e-2 (first corrector);
the lift coefficient's swing grows from 0.08 (100 to 200 s) to 0.52 (500 to 600 s), a laminar
vortex street still growing toward saturation, its period about 67 s (St about 0.14 on this coarse
grid; forceCoeffs1's output and checkMesh's log, 1,920 cells, mesh OK, are not committed).
checkMesh: largest aspect ratio 1.20, non-orthogonality 1.5e-6, skewness 0.035.
"""
import math
import sys
from pathlib import Path

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  Courant test fixture: a laminar cylinder at Re 100 (T-LAMINAR-REGIME)
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


D, U0, NU = 0.12, 0.012984, 1.558e-5          # cylinder2D's: Re = U0 D / NU = 100.0
R, RO, Z = D / 2, 15 * D, 0.0075
NTHETA, NR, FIRST = 16, 30, D / 20
TILT = math.radians(5.0)


def expansion() -> float:
    """blockMesh's simpleGrading ratio (last cell over first) for NR cells from FIRST to RO - R."""
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if FIRST * (r ** NR - 1) / (r - 1) < RO - R:
            lo = r
        else:
            hi = r
    return r ** (NR - 1)


def blockmesh() -> str:
    pts = []
    for z in (-Z, Z):
        for k in range(4):
            a = k * math.pi / 2
            pts.append((R * math.cos(a), R * math.sin(a), z))      # inner k
        for k in range(4):
            a = k * math.pi / 2
            pts.append((RO * math.cos(a), RO * math.sin(a), z))    # outer k
    verts = "\n".join(f"    ({x:.12g} {y:.12g} {z:.12g})" for x, y, z in pts)

    def i(k, top=0):
        return k % 4 + 8 * top

    def o(k, top=0):
        return 4 + k % 4 + 8 * top

    g = expansion()
    blocks = "\n".join(
        f"    hex ({i(k)} {o(k)} {o(k + 1)} {i(k + 1)} {i(k, 1)} {o(k, 1)} {o(k + 1, 1)} "
        f"{i(k + 1, 1)}) ({NR} {NTHETA} 1) simpleGrading ({g:.10g} 1 1)" for k in range(4))
    edges = []
    for top, z in ((0, -Z), (1, Z)):
        for k in range(4):
            a = (k + 0.5) * math.pi / 2
            edges.append(f"    arc {i(k, top)} {i(k + 1, top)} "
                         f"({R * math.cos(a):.12g} {R * math.sin(a):.12g} {z:.12g})")
            edges.append(f"    arc {o(k, top)} {o(k + 1, top)} "
                         f"({RO * math.cos(a):.12g} {RO * math.sin(a):.12g} {z:.12g})")

    def outer(k):
        return f"({o(k)} {o(k, 1)} {o(k + 1, 1)} {o(k + 1)})"

    cylinder = " ".join(f"({i(k)} {i(k + 1)} {i(k + 1, 1)} {i(k, 1)})" for k in range(4))
    sides = " ".join(f"({i(k)} {i(k + 1)} {o(k + 1)} {o(k)}) "
                     f"({i(k, 1)} {o(k, 1)} {o(k + 1, 1)} {i(k + 1, 1)})" for k in range(4))
    return f"""
scale 1;

// An O-grid about the cylinder: four quarter-annulus blocks, radial then angular. The upstream
// half of the far field (90 to 270 degrees) is the inlet, the downstream half the outlet.
vertices
(
{verts}
);

blocks
(
{blocks}
);

edges
(
{chr(10).join(edges)}
);

boundary
(
    inlet        {{ type patch; faces ({outer(1)} {outer(2)}); }}
    outlet       {{ type patch; faces ({outer(3)} {outer(0)}); }}
    cylinder     {{ type wall;  faces ({cylinder}); }}
    frontAndBack {{ type empty; faces ({sides}); }}
);
"""


CONTROL = """
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         600;
deltaT          0.05;
writeControl    adjustableRunTime;
writeInterval   600;
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable false;
adjustTimeStep  yes;
maxCo           0.9;
maxDeltaT       1;

functions
{
    forceCoeffs1
    {
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        log             false;
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         0.012984;
        lRef            0.12;
        Aref            0.0018;
    }
}
"""

# v2406's tutorials/incompressible/pimpleFoam/laminar/cylinder2D/system/fvSchemes, as shipped.
SCHEMES = """
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         Gauss linear;
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
fluxRequired    { default no; p ; }
"""

# v2406's tutorials/incompressible/pimpleFoam/laminar/cylinder2D/system/fvSolution, as shipped.
SOLUTION = """
solvers
{
    p
    {
        solver           GAMG;
        smoother         DICGaussSeidel;
        tolerance        1e-6;
        relTol           0.01;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    UFinal
    {
        $U;
        relTol          0;
    }
}
PIMPLE
{
    nNonOrthogonalCorrectors 0;
    nCorrectors         2;
}
"""

TRANSPORT = f"""
transportModel  Newtonian;
nu              {NU};
"""

TURBULENCE = """
simulationType  laminar;
"""

DECOMPOSE = """
numberOfSubdomains  2;
method              hierarchical;
coeffs
{
    n               (2 1 1);
}
"""


def field(dims: str, internal: str, patches: dict[str, str]) -> str:
    body = f"\ndimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for patch, bc in patches.items():
        body += f"    {patch}\n    {{\n{bc}\n    }}\n"
    return body + "}\n"


def inlet() -> str:
    """The inflow: turned 5 degrees until 50 s, straight from 75 s (a uniformFixedValue table)."""
    tilted = f"({U0 * math.cos(TILT):.8g} {U0 * math.sin(TILT):.8g} 0)"
    straight = f"({U0:g} 0 0)"
    rows = "\n".join(f"            ({t} {v})" for t, v in
                     ((0, tilted), (50, tilted), (75, straight), (10000, straight)))
    return ("        type            uniformFixedValue;\n"
            f"        uniformValue    table\n        (\n{rows}\n        );")


def case(d: Path) -> None:
    write(d / "system/blockMeshDict", "blockMeshDict", blockmesh())
    write(d / "system/controlDict", "controlDict", CONTROL)
    write(d / "system/fvSchemes", "fvSchemes", SCHEMES)
    write(d / "system/fvSolution", "fvSolution", SOLUTION)
    write(d / "system/decomposeParDict", "decomposeParDict", DECOMPOSE)
    write(d / "constant/transportProperties", "transportProperties", TRANSPORT)
    write(d / "constant/turbulenceProperties", "turbulenceProperties", TURBULENCE)
    empty = "        type            empty;"
    write(d / "0/U", "U", field("[0 1 -1 0 0 0 0]", f"uniform ({U0:g} 0 0)", {
        "inlet": inlet(),
        "outlet": ("        type            inletOutlet;\n        inletValue      uniform (0 0 0);"
                   f"\n        value           uniform ({U0:g} 0 0);"),
        "cylinder": "        type            noSlip;",
        "frontAndBack": empty}), cls="volVectorField", loc='location    "0";\n    ')
    write(d / "0/p", "p", field("[0 2 -2 0 0 0 0]", "uniform 0", {
        "inlet": "        type            zeroGradient;",
        "outlet": "        type            fixedValue;\n        value           uniform 0;",
        "cylinder": "        type            zeroGradient;",
        "frontAndBack": empty}), cls="volScalarField", loc='location    "0";\n    ')


if __name__ == "__main__":
    case(Path(sys.argv[1]))
