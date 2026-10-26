"""TRD VER-18: the boundary flux balance, fed from the `phi` the solver wrote.

Courant 0.5.0 accepted, grade A, a rhoSimpleFoam nozzle run whose boundary mass flux was out of
balance (one rebuilt run, an internal evaluation of 20 Oct 2026, not a measurement):
`MassFluxCheck` existed and graded on net over inflow, but nothing filled
`FieldData.patch_fluxes`, so it skipped on every case. These tests hold the reader that fills
it (`read_fields(fluxes=True)`), the check's scope (which runs must balance), and GATE-13: only
`courant verify --checks next` reads `phi`; under the default nothing new is opened and the
verdict is v0.5.0's.

Every case here is synthetic, or this repository's own committed pitzDaily run with a synthetic
solved time directory added.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from courant.case.config import CaseConfig
from courant.case.reads import recording
from courant.verification._fluid_reader import read_fields
from courant.verification.checks import MassFluxCheck, VerifyContext
from courant.verification.fields import FieldData, FluxSource
from courant.verification.grade import VSeverity

ROOT = Path(__file__).resolve().parents[2]
PITZ = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__shipped"

VOLUME, MASS = "[0 3 -1 0 0 0 0]", "[1 0 -1 0 0 0 0]"
U_FIELD = ("FoamFile{version 2.0;format ascii;class volVectorField;object U;}\n"
           "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (1 0 0);\nboundaryField{}\n")
P_FIELD = ("FoamFile{version 2.0;format ascii;class volScalarField;object p;}\n"
           "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField{}\n")


def _list(values) -> str:
    return f"nonuniform List<scalar> {len(values)}({' '.join(repr(float(v)) for v in values)})"


def _phi(patches: dict[str, tuple[str, str | None]], dims: str = VOLUME,
         internal: str = "nonuniform List<scalar> 3(0.5 -0.25 0.75)", fmt: str = "ascii",
         cls: str = "surfaceScalarField") -> str:
    """A `phi` file: {patch: (type, the value's text or None for no value)}."""
    body = "".join(
        f"    {name}\n    {{\n        type {ptype};\n"
        + (f"        value {value};\n" if value is not None else "") + "    }\n"
        for name, (ptype, value) in patches.items())
    return (f"FoamFile\n{{\n    version 2.0;\n    format {fmt};\n    class {cls};\n"
            f"    object phi;\n}}\ndimensions {dims};\noriented oriented;\n"
            f"internalField {internal};\nboundaryField\n{{\n{body}}}\n")


def _boundary(patches: dict[str, tuple[str, int]]) -> str:
    """A `polyMesh/boundary` file: {patch: (type, nFaces)}."""
    start, body = 100, ""
    for name, (ptype, n) in patches.items():
        body += (f"    {name}\n    {{\n        type {ptype};\n        nFaces {n};\n"
                 f"        startFace {start};\n    }}\n")
        start += n
    return ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class polyBoundaryMesh;\n"
            "    location \"constant/polyMesh\";\n    object boundary;\n}\n"
            f"{len(patches)}\n(\n{body})\n")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _solved(case: Path, phi: str, time: str = "1") -> Path:
    _write(case / time / "U", U_FIELD)
    _write(case / time / "p", P_FIELD)
    _write(case / time / "phi", phi)
    return case


#: A synthetic imbalance: 1.0 in through the inlet, 1.219 out, so 0.219 of the inflow.
IMBALANCED = {"inlet": ("calculated", _list([-0.25] * 4)),
              "outlet": ("calculated", _list([0.3, 0.3, 0.3, 0.319])),
              "wall": ("calculated", "uniform 0"),
              "symmetry": ("symmetryPlane", "uniform 0"),
              "front": ("empty", None), "back": ("empty", None)}
BALANCED = {**IMBALANCED, "outlet": ("calculated", _list([0.25, 0.25, 0.25, 0.2500001]))}


def _ctx(fields: FieldData, *, solver: str = "simpleFoam", ddt: str | None = "steadyState",
         config: bool = True) -> VerifyContext:
    cfg = CaseConfig(solver=solver, ddt_scheme=ddt) if config else None
    return VerifyContext(run_id="r", events=[], config=cfg, fields=fields)


# ------------------------------------------------------------------------------- the reader
def test_the_reader_sums_each_patch_that_bounds_the_domain_as_the_solver_signed_it(tmp_path):
    """Empty and coupled patches are not the boundary; symmetry planes and walls are, and a
    `uniform` value counts once per face, from the mesh's boundary file."""
    patches = {**IMBALANCED,
               "inlet": ("calculated", "uniform -0.25"),
               "periodic_half0": ("cyclic", _list([-5.0, -5.0])),
               "periodic_half1": ("cyclic", _list([5.0, 5.0])),
               "rotor_side": ("cyclicAMI", _list([-3.0])),
               "procBoundary0to1": ("processor", _list([-7.0]))}
    case = _solved(tmp_path, _phi(patches))
    _write(case / "constant" / "polyMesh" / "boundary", _boundary(
        {"inlet": ("patch", 4), "outlet": ("patch", 4), "wall": ("wall", 2),
         "symmetry": ("symmetryPlane", 2), "front": ("empty", 9), "back": ("empty", 9),
         "periodic_half0": ("cyclic", 2), "periodic_half1": ("cyclic", 2),
         "rotor_side": ("cyclicAMI", 1), "procBoundary0to1": ("processor", 1)}))
    with recording(case) as reads:
        f = read_fields(str(case), fluxes=True)
    assert f.flux_note is None
    assert f.patch_fluxes == pytest.approx({"inlet": -1.0, "outlet": 1.219, "wall": 0.0,
                                            "symmetry": 0.0})
    assert f.flux_source == FluxSource(file="1/phi", kind="volume", inflow=1.0,
                                       outflow=pytest.approx(1.219), largest=0.75)
    assert {"1/phi", "constant/polyMesh/boundary"} <= set(reads.paths())


@pytest.mark.parametrize("n, note", [
    (10**12, None),                                     # counted, never allocated
    (None, "patch inlet of 1/phi has no face count in the mesh's boundary file"),
])
def test_a_uniform_value_is_counted_from_the_boundary_file_without_allocating_it(
        tmp_path, n, note):
    case = _solved(tmp_path, _phi({**BALANCED, "inlet": ("calculated", "uniform -0.25")}))
    mesh = {"outlet": ("patch", 4), "wall": ("wall", 2), "symmetry": ("symmetryPlane", 2)}
    if n is not None:
        mesh["inlet"] = ("patch", n)
    _write(case / "constant" / "polyMesh" / "boundary", _boundary(mesh))
    f = read_fields(str(case), fluxes=True)
    assert f.flux_note == note
    if n is not None:
        assert f.patch_fluxes["inlet"] == -0.25 * n and f.flux_source.inflow == 0.25 * n


def test_a_compressible_solvers_phi_is_a_mass_flux_and_a_compressed_one_is_read(tmp_path):
    os.makedirs(tmp_path / "1")
    with gzip.open(tmp_path / "1" / "phi.gz", "wt") as fh:
        fh.write(_phi(IMBALANCED, dims=MASS))
    f = read_fields(str(tmp_path), fluxes=True)
    assert f.flux_source.kind == "mass" and f.flux_source.file == "1/phi.gz"
    assert sum(f.patch_fluxes.values()) == pytest.approx(0.219)


def test_under_the_default_phi_is_never_opened(tmp_path):
    """GATE-13: the default reads what v0.5.0 read. No flux, no note, nothing noted."""
    case = _solved(tmp_path, _phi(IMBALANCED))
    with recording(case) as reads:
        f = read_fields(str(case))
    assert f.patch_fluxes == {} and f.flux_source is None and f.flux_note is None
    assert "1/phi" not in reads.paths()
    assert set(f.internal) == {"U", "p"}


@pytest.mark.parametrize("shape, phrase, noted", [
    ("absent", "no 1/phi (the face flux at the latest time) was found", False),
    ("binary", "1/phi is written binary; only an ASCII phi is read", True),
    ("garbage", "1/phi could not be read", True),
    ("truncated", "1/phi could not be read", True),
    ("volume_field", "1/phi is a volScalarField, not a face flux", True),
    ("velocity_dims", "1/phi has dimensions [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]", True),
    ("all_coupled", "every patch of 1/phi is empty or coupled", True),
    ("no_value", "patch inlet of 1/phi holds no flux values", True),
    ("uniform_no_mesh", "1/phi holds a uniform boundary value and constant/polyMesh/boundary",
     True),
    ("overset", "patch hole of 1/phi is overset", True),
])
def test_a_phi_that_cannot_be_summed_as_written_is_not_checked_and_says_why(
        tmp_path, shape, phrase, noted):
    """Never a traceback, never a guess: the check skips with a sentence for "not checked"."""
    case = _solved(tmp_path, _phi(BALANCED))
    phi = case / "1" / "phi"
    if shape == "absent":
        phi.unlink()
    elif shape == "binary":
        phi.write_text(_phi(BALANCED, fmt="binary"))
    elif shape == "garbage":
        phi.write_bytes(b"\x00\xff not a dictionary {{{")
    elif shape == "truncated":
        phi.write_text(_phi(BALANCED)[:-60])
    elif shape == "volume_field":
        phi.write_text(_phi(BALANCED, cls="volScalarField"))
    elif shape == "velocity_dims":
        phi.write_text(_phi(BALANCED, dims="[0 1 -1 0 0 0 0]"))
    elif shape == "all_coupled":
        phi.write_text(_phi({"a": ("cyclic", _list([1.0])), "b": ("cyclic", _list([-1.0])),
                             "c": ("empty", None)}))
    elif shape == "no_value":
        phi.write_text(_phi({**BALANCED, "inlet": ("calculated", None)}))
    elif shape == "uniform_no_mesh":
        phi.write_text(_phi({**BALANCED, "inlet": ("calculated", "uniform -0.25")}))
    elif shape == "overset":
        phi.write_text(_phi({**BALANCED, "hole": ("overset", _list([0.0]))}))
    with recording(case) as reads:
        f = read_fields(str(case), fluxes=True)
    assert f.patch_fluxes == {} and f.flux_source is None
    assert f.flux_note.startswith(phrase), f.flux_note
    assert ("1/phi" in reads.paths()) is noted
    r = MassFluxCheck().run(_ctx(f))
    assert r.skipped and r.severity is VSeverity.OK and r.detail == f.flux_note


def test_a_decomposed_run_is_not_summed_across_its_ranks(tmp_path):
    for rank in range(2):
        _solved(tmp_path / f"processor{rank}", _phi(IMBALANCED))
    f = read_fields(str(tmp_path), fluxes=True)
    assert f.patch_fluxes == {}
    assert f.flux_note.startswith("the run is decomposed and was not reconstructed")


@pytest.mark.parametrize("text, moving", [
    ("dynamicFvMesh staticFvMesh;\n", False),
    ("dynamicFvMesh dynamicMotionSolverFvMesh;\nmotionSolver solidBody;\n", True),
    ("mover\n{\n    type motionSolver;\n    motionSolver solidBody;\n}\n", True),
])
def test_a_moving_mesh_is_not_judged(tmp_path, text, moving):
    """A moving mesh's flux is relative to the mesh's motion; its boundary balances the change of
    the domain's volume, which is not read."""
    case = _solved(tmp_path, _phi(IMBALANCED))
    _write(case / "constant" / "dynamicMeshDict",
           "FoamFile{version 2.0;format ascii;class dictionary;object dynamicMeshDict;}\n" + text)
    with recording(case) as reads:
        f = read_fields(str(case), fluxes=True)
    assert "constant/dynamicMeshDict" in reads.paths()
    if moving:
        assert f.patch_fluxes == {} and f.flux_note.startswith("the mesh is dynamic")
    else:
        assert f.flux_note is None and f.patch_fluxes


def test_a_case_that_never_wrote_a_result_reads_no_flux(tmp_path):
    """The reader's rule 2 holds for `phi` too: `0/phi` is never the result."""
    _write(tmp_path / "0" / "phi", _phi(IMBALANCED))
    with recording(tmp_path) as reads:
        f = read_fields(str(tmp_path), fluxes=True)
    assert f.patch_fluxes == {} and reads.paths() == []


# ------------------------------------------------------------------------------- the check
def test_the_nozzles_imbalance_is_major_and_cites_its_value_bar_and_file(tmp_path):
    f = read_fields(str(_solved(tmp_path, _phi(IMBALANCED, dims=MASS), time="2111")),
                    fluxes=True)
    r = MassFluxCheck().run(_ctx(f, solver="rhoSimpleFoam"))
    assert r.severity is VSeverity.MAJOR and not r.passed and not r.skipped
    assert r.value == pytest.approx(0.219) and r.threshold == MassFluxCheck.LOOSE == 1e-2
    assert r.evidence == {"input": "2111/phi (boundary face fluxes)"}
    assert r.detail == ("net boundary flux 0.219 kg/s against an inflow of 1 kg/s: out of "
                        "balance by 2.19e-01 of the inflow (above 1e-02)")


def test_a_balanced_boundary_passes(tmp_path):
    f = read_fields(str(_solved(tmp_path, _phi(BALANCED))), fluxes=True)
    r = MassFluxCheck().run(_ctx(f))
    assert r.severity is VSeverity.OK and r.passed and not r.skipped
    assert r.value == pytest.approx(1e-7) and r.threshold == MassFluxCheck.TARGET


def test_between_the_two_bars_it_is_minor(tmp_path):
    f = read_fields(str(_solved(tmp_path, _phi(
        {**BALANCED, "outlet": ("calculated", _list([0.25, 0.25, 0.25, 0.255]))}))), fluxes=True)
    r = MassFluxCheck().run(_ctx(f))
    assert r.severity is VSeverity.MINOR and r.passed
    assert r.value == pytest.approx(5e-3) and r.threshold == MassFluxCheck.TARGET


def test_a_patch_carrying_flow_both_ways_is_judged_face_by_face(tmp_path):
    """An external-flow case's one far-field patch takes flow in on one side and out on the
    other. Its net is about zero; the balance's denominator is the flux in through its faces,
    not the sign of its net, or a balanced run would be judged against round-off."""
    f = read_fields(str(_solved(tmp_path, _phi({
        "freestream": ("calculated", _list([-2.0, -2.0, 2.0, 2.0 - 1e-9])),
        "body": ("calculated", "uniform 0")}))), fluxes=True)
    assert f.flux_source.inflow == 4.0
    r = MassFluxCheck().run(_ctx(f))
    assert r.severity is VSeverity.OK and r.value == pytest.approx(2.5e-10)


def test_a_closed_domain_whose_walls_carry_round_off_is_not_judged(tmp_path):
    """A lid-driven cavity: the moving lid's flux is round-off (1e-20 of a face flux of 0.75).
    Net over that "inflow" is order one; judging it would reject a healthy run."""
    f = read_fields(str(_solved(tmp_path, _phi({
        "movingWall": ("calculated", _list([-1e-20, 3e-20, -2e-20])),
        "fixedWalls": ("calculated", "uniform 0"),
        "frontAndBack": ("empty", None)}))), fluxes=True)
    r = MassFluxCheck().run(_ctx(f, solver="icoFoam", ddt="Euler"))
    assert r.skipped, r
    assert r.detail == ("no flow crosses the boundary in 1/phi (a closed domain): there is no "
                        "balance to judge")


def test_a_round_off_wall_does_not_close_a_domain_whose_outflow_is_real(tmp_path):
    """Closed means inflow and outflow both round-off next to the largest face flux. A wall whose
    faces carry round-off of either sign makes the inflow tiny and positive; the outflow of 1.0
    must still be judged, and with nothing real coming in, all of it is imbalance (a draining or
    unconverged steady run)."""
    f = read_fields(str(_solved(tmp_path, _phi({
        "outlet": ("calculated", _list([0.5, 0.5])),
        "wall": ("calculated", _list([-1e-20, 0.0]))}))), fluxes=True)
    assert f.flux_source.inflow == pytest.approx(1e-20) and f.flux_source.largest == 0.75
    r = MassFluxCheck().run(_ctx(f))
    assert r.severity is VSeverity.MAJOR and not r.passed and not r.skipped
    assert r.value == pytest.approx(1.0) and r.threshold == MassFluxCheck.LOOSE
    assert "against an outflow of 1 m^3/s" in r.detail


@pytest.mark.parametrize("bad", ["inf", "nan", "-inf"])
def test_a_non_finite_interior_flux_is_major_and_never_reads_as_a_closed_domain(tmp_path, bad):
    """An inf among the internal faces made the largest face flux inf, so every inflow was
    round-off next to it and the run skipped as "a closed domain"; a nan slipped past both
    comparisons. A non-finite flux anywhere in phi is MAJOR."""
    f = read_fields(str(_solved(tmp_path, _phi(
        BALANCED, internal=f"nonuniform List<scalar> 3(0.5 {bad} 0.75)"))), fluxes=True)
    r = MassFluxCheck().run(_ctx(f))
    assert r.severity is VSeverity.MAJOR and not r.passed and not r.skipped
    assert r.detail == "an interior face flux in 1/phi is not finite (nan or inf)"


def test_with_no_inflow_all_the_outflow_is_imbalance(tmp_path):
    f = read_fields(str(_solved(tmp_path, _phi({
        "outlet": ("calculated", _list([0.5, 0.5])), "wall": ("calculated", "uniform 0")}))),
        fluxes=True)
    r = MassFluxCheck().run(_ctx(f))
    assert r.severity is VSeverity.MAJOR and r.value == pytest.approx(1.0)
    assert "against an outflow of 1 m^3/s" in r.detail


def test_a_non_finite_flux_is_major(tmp_path):
    f = read_fields(str(_solved(tmp_path, _phi({
        **BALANCED, "outlet": ("calculated", "nonuniform List<scalar> 2(nan 0.5)")}))),
        fluxes=True)
    r = MassFluxCheck().run(_ctx(f))
    assert r.severity is VSeverity.MAJOR and not r.passed
    assert r.detail == "the boundary flux is not finite (nan or inf in phi)"


@pytest.mark.parametrize("dims, solver, ddt, judged", [
    (MASS, "rhoSimpleFoam", "steadyState", True),        # a steady state: in equals out
    (MASS, "rhoPimpleFoam", "Euler", False),             # mass inside may change
    (MASS, "rhoPimpleFoam", "localEuler", False),        # LTS is read as transient, as elsewhere
    (VOLUME, "simpleFoam", "steadyState", True),
    (VOLUME, "pimpleFoam", "Euler", True),               # incompressible: every step balances
    (VOLUME, "interFoam", "Euler", True),                # two incompressible phases
    (VOLUME, "compressibleInterFoam", "Euler", False),   # its volume flux carries expansion
    (VOLUME, "mpirun -np 4 pisoFoam -parallel", "Euler", True),
    (VOLUME, None, "Euler", False),                      # an unnamed solver
])
def test_only_a_run_that_must_balance_is_judged(tmp_path, dims, solver, ddt, judged):
    f = read_fields(str(_solved(tmp_path, _phi(IMBALANCED, dims=dims))), fluxes=True)
    r = MassFluxCheck().run(_ctx(f, solver=solver, ddt=ddt))
    if judged:
        assert r.severity is VSeverity.MAJOR and not r.skipped
    else:
        assert r.skipped and r.detail.startswith("1/phi holds ")


def test_whether_the_run_is_steady_is_asked_in_the_verdicts_mode():
    """A solver only v2406's own table knows to be steady (`V2406_STEADY`), its fvSchemes not
    read: under next it is steady for the residual bar (`is_steady("next")`), and so for the
    balance too, where the check asked the default and said the run was transient (the final
    review of 0.6.0, 21 Oct 2026). Under the default it is transient for both, as in 0.5.0."""
    from courant.verification.checks import _balance_not_owed

    f = FieldData(time=1.0, internal={}, nu=None, inlet_speed=None,
                  flux_source=FluxSource(file="1/phi", kind="mass", inflow=1.0, outflow=1.0,
                                         largest=1.0))
    config = CaseConfig(solver="overRhoSimpleFoam")             # no ddtSchemes read
    assert config.is_steady("next") and not config.is_steady("default")
    assert _balance_not_owed(f, config, None, "next") is None    # a steady run: it must balance
    assert _balance_not_owed(f, config, None, "default").startswith(
        "1/phi holds a mass flux and the run is transient")


@pytest.mark.parametrize("named, ran, judged", [
    ("interFoam", "compressibleInterFoam", False),   # a copied damBreak, run with another solver
    ("interFoam", "interFoam", True),
    (None, "pimpleFoam", True),                      # no application read: the run decides
    ("pimpleFoam", None, True),                      # no Exec line: controlDict decides
])
def test_the_solver_that_ran_decides_whether_the_run_must_balance(tmp_path, named, ran, judged):
    """The verdict names the solver the log's Exec line names (`pipeline._solver`), and the
    balance is owed by that solver's run, not by controlDict's `application`. When the two
    disagree the case was set up for another solver, and the balance is not judged."""
    f = read_fields(str(_solved(tmp_path, _phi(IMBALANCED))), fluxes=True)
    r = MassFluxCheck().run(VerifyContext(
        run_id="r", events=[], config=CaseConfig(solver=named, ddt_scheme="Euler"), fields=f,
        solver=ran))
    if judged:
        assert r.severity is VSeverity.MAJOR and not r.skipped
    else:
        assert r.skipped and r.detail == (
            "the solver that ran is compressibleInterFoam and system/controlDict names "
            "interFoam: on a case set up for another solver, whether 1/phi must balance is not "
            "judged")


_FV = "FoamFile{version 2.0;format ascii;class dictionary;object %s;}\n"


@pytest.mark.parametrize("rel, entry, solver, ddt, judged", [
    # OpenFOAM.org 9 on: interPhaseChangeFoam's cavitation is interFoam with this fvModel
    ("constant/fvModels", "cavitation { type VoFCavitation; model SchnerrSauer; }",
     "interFoam", "Euler", False),
    # a source of mass or volume breaks the balance at a steady state too
    ("constant/fvModels", "inject { type massSource; massFlowRate 0.1; }",
     "simpleFoam", "steadyState", False),
    ("constant/fvModels", "inject { type volumeSource; volumetricFlowRate 0.1; }",
     "pimpleFoam", "Euler", False),
    # a semi-implicit or coded source may act on any field, continuity's included
    ("system/fvOptions", "heat { type scalarSemiImplicitSource; }",
     "simpleFoam", "steadyState", False),
    ("constant/fvOptions", "porosity1 { type explicitPorositySource; }",   # momentum only
     "simpleFoam", "steadyState", True),
    ("constant/fvModels", "disk { type actuationDiskSource; }\nlibs (\"x\");",
     "pimpleFoam", "Euler", True),
])
def test_a_case_whose_own_sources_may_add_mass_or_volume_is_not_judged(
        tmp_path, rel, entry, solver, ddt, judged):
    """The solver reads fvModels and fvOptions; a mass, volume or phase-change source among
    them puts flux into the domain that its boundary does not carry. Read only under `next`."""
    case = _solved(tmp_path, _phi(IMBALANCED))
    _write(case / rel, _FV % rel.rsplit("/", 1)[1] + entry + "\n")
    with recording(case) as reads:
        f = read_fields(str(case), fluxes=True)
    assert rel in reads.paths()
    r = MassFluxCheck().run(_ctx(f, solver=solver, ddt=ddt))
    if judged:
        assert r.severity is VSeverity.MAJOR and not r.skipped
    else:
        assert f.patch_fluxes == {} and "1/phi" not in reads.paths()
        assert f.flux_note.startswith(f"{rel} declares "), f.flux_note
        assert r.skipped and r.detail == f.flux_note
    with recording(case) as reads:
        read_fields(str(case))
    assert rel not in reads.paths()


def test_the_sentence_names_the_source_and_why_it_is_not_judged(tmp_path):
    case = _solved(tmp_path, _phi(IMBALANCED))
    _write(case / "constant" / "fvModels",
           _FV % "fvModels" + "cavitation { type VoFCavitation; model SchnerrSauer; }\n")
    f = read_fields(str(case), fluxes=True)
    assert f.flux_note == (
        "constant/fvModels declares cavitation, of type VoFCavitation, which is not known to act "
        "on momentum, energy or turbulence alone: a mass, volume or phase-change source puts "
        "flux into the domain that its boundary does not carry, so the balance is not judged")


def test_an_unreadable_source_file_is_not_judged(tmp_path):
    case = _solved(tmp_path, _phi(IMBALANCED))
    _write(case / "constant" / "fvModels", "cavitation { type VoFCavitation; ")
    f = read_fields(str(case), fluxes=True)
    assert f.patch_fluxes == {}
    assert f.flux_note == ("constant/fvModels could not be read, so whether it declares a mass, "
                           "volume or phase-change source is unknown")


def test_without_the_case_configuration_whether_it_must_balance_is_unknown(tmp_path):
    f = read_fields(str(_solved(tmp_path, _phi(IMBALANCED))), fluxes=True)
    r = MassFluxCheck().run(_ctx(f, config=False))
    assert r.skipped and "whether the run must balance" in r.detail


# ------------------------------------------------------------------------------- the gate
def _pitz(tmp: Path, phi: str | None) -> Path:
    """This repository's committed pitzDaily run (simpleFoam, steady, accepted), with a
    synthetic solution at its last step, 282."""
    case = tmp / "pitzDaily"
    shutil.copytree(PITZ, case)
    _write(case / "282" / "U", U_FIELD)
    _write(case / "282" / "p", P_FIELD)
    if phi is not None:
        _write(case / "282" / "phi", phi)
    return case


def _pitz_phi(outlet: list[float]) -> str:
    return _phi({"inlet": ("calculated", _list([-0.25] * 4)),
                 "outlet": ("calculated", _list(outlet)),
                 "upperWall": ("calculated", "uniform 0"),
                 "lowerWall": ("calculated", "uniform 0"),
                 "frontAndBack": ("empty", None)})


def _verdict(case: Path, checks: str) -> dict:
    from courant.verify import from_case_dir, verify_inputs

    return json.loads(verify_inputs(from_case_dir(case, checks=checks)).to_json())


SKIPPED_050 = "Net boundary mass flux ~ 0: no field flux data"


def test_next_rejects_an_imbalanced_run_and_cites_the_file(tmp_path):
    """The run is accepted, grade B (a minor bounding note), by the default and by `next`
    without a `phi`; an imbalanced `phi` makes it grade C, and VER-5.3 rejects it."""
    v = _verdict(_pitz(tmp_path, _pitz_phi([0.3, 0.3, 0.3, 0.319])), "next")
    assert (v["verdict"], v["rule"], v["grade"]) == ("reject", "VER-5.3", "C")
    reason = next(r for r in v["reasons"] if r["code"] == "conservation.mass_flux")
    assert reason["severity"] == "major" and reason["kind"] == "check"
    assert float(reason["value"]) == pytest.approx(0.219) and reason["threshold"] == "0.01"
    assert reason["input"] == "282/phi (boundary face fluxes)"
    assert reason["summary"].startswith("net boundary flux 0.219 m^3/s against an inflow of 1")
    assert "282/phi" in {f["path"] for f in v["run"]["inputs"]}
    assert SKIPPED_050 not in v["not_assessed"]


def test_next_accepts_a_balanced_run_and_no_longer_lists_the_check_as_not_done(tmp_path):
    v = _verdict(_pitz(tmp_path / "a", _pitz_phi([0.25, 0.25, 0.25, 0.2500001])), "next")
    before = _verdict(_pitz(tmp_path / "b", None), "next")
    assert (v["verdict"], v["rule"], v["grade"]) == ("accept", "VER-5.6", "B")
    assert (v["reasons"], v["grade"]) == (before["reasons"], before["grade"])
    assert not [s for s in v["not_assessed"] if s.startswith("Net boundary mass flux")]
    assert "Net boundary mass flux ~ 0: no 282/phi (the face flux at the latest time) was " \
        "found" in before["not_assessed"]
    assert "282/phi" in {f["path"] for f in v["run"]["inputs"]}


def test_next_names_why_the_balance_was_not_checked(tmp_path):
    case = _pitz(tmp_path, _pitz_phi([0.25] * 4).replace("format ascii", "format binary"))
    v = _verdict(case, "next")
    assert (v["verdict"], v["rule"]) == ("accept", "VER-5.6")
    assert ("Net boundary mass flux ~ 0: 282/phi is written binary; only an ASCII phi is read"
            in v["not_assessed"])


def test_next_judges_the_balance_with_the_solver_the_log_names(tmp_path):
    """A pitzDaily copy whose log ran rhoSimpleFoam: the verdict names rhoSimpleFoam, so the
    case's simpleFoam controlDict does not decide which balance is owed. Not judged, and why."""
    case = _pitz(tmp_path, _pitz_phi([0.3, 0.3, 0.3, 0.319]))
    log = case / "log.simpleFoam"
    log.write_text(log.read_text().replace("Exec   : simpleFoam", "Exec   : rhoSimpleFoam"))
    v = _verdict(case, "next")
    assert v["run"]["solver"] == "rhoSimpleFoam"
    assert not [r for r in v["reasons"] if r["code"] == "conservation.mass_flux"]
    assert ("Net boundary mass flux ~ 0: the solver that ran is rhoSimpleFoam and "
            "system/controlDict names simpleFoam: on a case set up for another solver, whether "
            "282/phi must balance is not judged") in v["not_assessed"]


def test_next_sums_the_phi_of_the_time_the_field_checks_judge_not_a_later_one(tmp_path):
    """VER-18 with VER-7b: under `next` the field checks read the log's last time step, and the
    flux balance sums that time's `phi`, not one a later, longer run left beside it. The log
    reached 282, out of balance; a directory at 500 the log never reached holds a balanced
    `phi`, and is named and not read."""
    case = _pitz(tmp_path, _pitz_phi([0.3, 0.3, 0.3, 0.319]))
    _solved(case, _pitz_phi([0.25, 0.25, 0.25, 0.2500001]), time="500")
    v = _verdict(case, "next")
    assert (v["verdict"], v["rule"], v["grade"]) == ("reject", "VER-5.3", "C")
    reason = next(r for r in v["reasons"] if r["code"] == "conservation.mass_flux")
    assert reason["input"] == "282/phi (boundary face fluxes)"
    paths = {f["path"] for f in v["run"]["inputs"]}
    assert "282/phi" in paths and "500/phi" not in paths


def test_the_default_verdict_does_not_read_phi_and_is_the_one_without_it(tmp_path):
    """GATE-13, the check's own side: under the default an imbalanced run is judged exactly as
    the same run with no `phi` at all, byte for byte, and as v0.5.0 judged it (accepted)."""
    with_phi = _verdict(_pitz(tmp_path / "a", _pitz_phi([0.3, 0.3, 0.3, 0.319])), "default")
    without = _verdict(_pitz(tmp_path / "b", None), "default")
    assert with_phi == without
    assert (with_phi["verdict"], with_phi["rule"]) == ("accept", "VER-5.6")
    assert SKIPPED_050 in with_phi["not_assessed"]
    assert "282/phi" not in {f["path"] for f in with_phi["run"]["inputs"]}


@pytest.mark.parametrize("mode", ["default", "next"])
def test_the_manifest_is_exactly_what_the_readers_opened_with_a_phi(tmp_path, mode):
    """VER-14 against the interpreter's own record of what was opened (the audit hook of
    `test_verify_pack`): `phi` is opened, and listed, under `next` only."""
    from tests.unit import test_verify_pack as tvp

    case = tvp._case(tmp_path)
    _write(case / "0.5" / "phi", _phi({"inlet": ("calculated", _list([-1.0])),
                                        "outlet": ("calculated", _list([1.0])),
                                        "walls": ("calculated", "uniform 0")}))
    verdict, opened = tvp._verify_observed(case, None, mode)
    listed = {f.path for f in verdict.run.inputs}
    assert listed == opened
    assert ("0.5/phi" in opened) is (mode == "next")


def test_a_preflight_says_what_checks_conservation_in_either_mode():
    from courant.data import DEMO
    from courant.verify.preflight import NEEDS_THE_RUN, NEEDS_THE_RUN_NEXT, preflight_path

    log, side = DEMO / "icoFoam_cavity__healthy.log.gz", DEMO / "icoFoam_cavity__healthy.case.json"
    default = preflight_path(log, side).not_checked
    nxt = preflight_path(log, side, checks="next").not_checked
    assert NEEDS_THE_RUN in default and NEEDS_THE_RUN_NEXT not in default
    assert NEEDS_THE_RUN_NEXT in nxt and NEEDS_THE_RUN not in nxt
    # v0.5.0's default said conservation is checked; corrected (H-PREFLIGHT-CONSERVATION)
    assert "conservation, bounding) is checked" not in NEEDS_THE_RUN
    assert "Conservation is not judged by default" in NEEDS_THE_RUN
    assert "read only by courant verify --checks next." in NEEDS_THE_RUN
    assert "is reported, and graded only when it is not finite" in NEEDS_THE_RUN
    assert "Conservation is judged only by courant verify --checks next" in NEEDS_THE_RUN_NEXT
    # The log's continuity error is graded when it is not finite (ContinuityCheck: MAJOR).
    assert "is reported, and graded only when it is not finite." in NEEDS_THE_RUN_NEXT
    assert "reported, not graded" not in NEEDS_THE_RUN_NEXT


def test_the_hand_built_balance_still_judges_net_over_inflow():
    """Data built without a reader (`flux_source` None: the watcher, tests) is judged on net over
    inflow with v0.5.0's bars and words, and three changes from v0.5.0, each pinned here: a
    MAJOR cites the bar it crossed, 1e-2 (v0.5.0 cited 1e-3 at every grade); fluxes that are all
    zero skip as a closed domain (v0.5.0 graded them OK, 0 over a stand-in inflow of 1.0); with
    no inflow the outflow is the denominator, so the value is 1.0 (v0.5.0 divided |net| by that
    stand-in 1.0, a number in the flux's own units)."""
    ok = FieldData(time=1, patch_fluxes={"inlet": -1.0, "outlet": 1.0 + 1e-4})
    bad = FieldData(time=1, patch_fluxes={"inlet": -1.0, "outlet": 0.5})
    ctx = VerifyContext(run_id="r", events=[])
    assert MassFluxCheck().run(VerifyContext(run_id="r", events=[], fields=ok)).severity \
        is VSeverity.OK
    r = MassFluxCheck().run(VerifyContext(run_id="r", events=[], fields=bad))
    assert r.severity is VSeverity.MAJOR and r.detail == "net/inflow flux imbalance 5.00e-01"
    assert MassFluxCheck().run(ctx).detail == "no field flux data"
    assert np.isclose(r.value, 0.5)
    assert r.threshold == MassFluxCheck.LOOSE == 1e-2                    # v0.5.0: 1e-3
    zero = MassFluxCheck().run(VerifyContext(
        run_id="r", events=[], fields=FieldData(time=1, patch_fluxes={"wall": 0.0})))
    assert zero.skipped and zero.detail.startswith("no flow crosses the boundary")  # v0.5.0: OK
    out = MassFluxCheck().run(VerifyContext(
        run_id="r", events=[], fields=FieldData(time=1, patch_fluxes={"outlet": 3.0})))
    assert out.severity is VSeverity.MAJOR and out.value == 1.0          # v0.5.0: 3.0
