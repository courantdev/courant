"""The review of the verdict-path audit's fixes (research/notes/verdict-path-audit/FINDINGS.md,
"The review"). Each is a reading under `--checks next` that the fix itself got wrong.

VPA-R1: the group lookup (`spec.read_as_solver`) read `constant/polyMesh/boundary` alone; a
decomposed run read each rank's mesh and a region its own, so it does not tell. Whether a case was
solved as regions is decided once (`_foam_reader.regions_not_ruled_out`, `AsRun.regions`) on the
log's solver, else `application`: a stray `regionProperties` beside a single-region solver changes
nothing, and a binary `boundary` tells a patch's groups as an ascii one does (FLD-7).
VPA-R2: round-off of a field's max |value| (`bounds.negative_roundoff`) passed a temperature in
kelvin, 300 to 350, with one cell at -0.2: the offset set the scale. Round-off is now of the field's
range.
VPA-R3: the regime the log selected (`spec.turbulence_block`) was the first `Selecting RAS|LES`
of the log, which in a multi-region log is one region's.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from courant.spec import deck
from courant.verification.checks import _negative_beyond_roundoff

_BOUNDARY = """FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
2
(
    inlet { type patch; inGroups List<word> 1(inflow); nFaces 1; startFace 0; }
    outlet { type patch; nFaces 1; startFace 1; }
)
"""

_BF = """
    inflow { type fixedValue; value uniform (1 0 0); }
    "in.*" { type fixedValue; value uniform (10 0 0); }
    outlet { type zeroGradient; }
"""


def _mesh(root, where="constant/polyMesh"):
    d = root / where
    d.mkdir(parents=True, exist_ok=True)
    (d / "boundary").write_text(_BOUNDARY)


def test_vpa_r1_the_groups_are_read_from_the_reconstructed_mesh(tmp_path):
    _mesh(tmp_path)
    body, key, why = deck.patch_entry(tmp_path, _BF, "inlet")
    assert key == "inflow" and "(1 0 0)" in body and why is None


def test_vpa_r1_a_decomposed_mesh_does_not_tell_the_groups(tmp_path):
    _mesh(tmp_path)
    _mesh(tmp_path, "processor0/constant/polyMesh")
    body, _, why = deck.patch_entry(tmp_path, _BF, "inlet")
    assert body is None and "decomposed or multi-region" in why


def test_vpa_r1_a_multi_region_case_does_not_tell_the_groups(tmp_path):
    _mesh(tmp_path)
    (tmp_path / "constant" / "regionProperties").write_text("regions (fluid (air));\n")
    body, _, why = deck.patch_entry(tmp_path, _BF, "inlet", regions=True)
    assert body is None and why is not None


def _stray_case(tmp_path, application="simpleFoam"):
    """A case with a pattern, a group and an `outlet` entry in `0/U`, and a `regionProperties`."""
    from courant.spec.model import SPEC_SCHEMA, Spec, SpecItem

    _mesh(tmp_path)
    (tmp_path / "constant" / "regionProperties").write_text("regions (fluid (air));\n")
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U").write_text(
        "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
        "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
        'boundaryField {\n "in.*" { type fixedValue; value uniform (1 0 0); }\n'
        " outlet { type zeroGradient; }\n}\n")
    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        f"application {application};\n")
    item = SpecItem(name="U at inlet", file="0/U", patch="inlet", value=(1.0, 0.0, 0.0))
    return Spec(schema=SPEC_SCHEMA, items=(item,))


def _run(solver):
    from courant.spec.as_run import Run

    return Run(f"Exec : {solver}\nCreate mesh for time = 0\n", f"log.{solver}", (), exit_code=None)


@pytest.mark.parametrize("log_solver, application, read", [
    ("simpleFoam", "simpleFoam", True),                 # a stray file beside the log's solver
    (None, "simpleFoam", True),                         # no log: `application` says
    ("simpleFoam", "chtMultiRegionFoam", True),         # the log outranks `application`
    ("chtMultiRegionFoam", "chtMultiRegionFoam", False),
    (None, "chtMultiRegionFoam", False),
    ("myInHouseFoam", "myInHouseFoam", False),          # unknown: not taken to be single-region
])
def test_vpa_r1_a_stray_region_properties_beside_a_single_region_solver(tmp_path, log_solver,
                                                                        application, read):
    """A stray `constant/regionProperties` beside a known single-region solver changes nothing:
    `inlet` is reached by its pattern and checked. Beside a multi-region or unknown solver the
    groups are not told, so the item is not checked (VER-5.9). The one predicate decides, on the
    log's solver, else `application`."""
    from courant.spec import check_conformance

    spec = _stray_case(tmp_path, application)
    run = _run(log_solver) if log_solver else None
    block = check_conformance(spec, tmp_path, spec_sha256="0" * 64, checks="next", run=run)
    [got] = block.items
    assert (got.conforms is True) == read, got.note
    assert got.conforms is (True if read else None)


def test_vpa_r1_the_default_never_asks_about_regions(tmp_path):
    """GATE-13: under the default the pattern is read as 0.7.0 read it, whatever the solver."""
    from courant.spec import check_conformance
    from courant.spec.as_run import AsRun

    spec = _stray_case(tmp_path, "chtMultiRegionFoam")
    assert AsRun(tmp_path, _run("chtMultiRegionFoam")).regions is False
    [got] = check_conformance(spec, tmp_path, spec_sha256="0" * 64).items
    assert got.conforms is True


def test_vpa_r1_the_spec_reader_holds_no_test_of_its_own_for_the_file():
    """One fact, one implementation: `deck.py` and `as_run.py` hold no code that names
    `regionProperties`; only comments and docstrings may."""
    from courant.spec import as_run

    for module in (deck, as_run):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
                and n.body and isinstance(n.body[0], ast.Expr)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                    id(node) not in docs):
                assert "regionProperties" not in node.value, module.__name__


def test_vpa_r1_a_binary_boundary_tells_the_groups(tmp_path):
    """FLD-7: a `format binary` boundary's entries are text, and the group is read from them, so a
    binary polyMesh case's pattern-reached patch is not a wrongful VER-5.9."""
    d = tmp_path / "constant" / "polyMesh"
    d.mkdir(parents=True)
    (d / "boundary").write_text(_BOUNDARY.replace("format ascii;", "format binary;"))
    body, key, why = deck.patch_entry(tmp_path, _BF, "inlet")
    assert key == "inflow" and "(1 0 0)" in body and why is None


def test_vpa_r2_a_kelvin_field_one_cell_below_zero_is_negative():
    t = np.full(100, 300.0)
    t[:10] = 350.0
    t[0] = -0.2                       # 0.2 < 1e-3 * 350: within the offset's "round-off"
    assert _negative_beyond_roundoff(t)


def test_vpa_r2_a_passive_scalar_at_round_off_below_zero_is_not():
    s = np.linspace(0.0, 1.0, 101)
    s[0] = -1e-6
    assert not _negative_beyond_roundoff(s)


def test_vpa_r3_a_multi_region_log_does_not_name_a_regions_regime(tmp_path):
    from courant.spec.as_run import AsRun, Run

    (tmp_path / "constant" / "air").mkdir(parents=True)
    (tmp_path / "constant" / "regionProperties").write_text("regions (fluid (fluid air));\n")
    (tmp_path / "constant" / "air" / "momentumTransport").write_text(
        "simulationType LES;\nLES { model Smagorinsky; }\nRAS { model kEpsilon; }\n")
    log = ("Exec : chtMultiRegionFoam\nSelecting RAS turbulence model kEpsilon\n"
           "Selecting LES turbulence model Smagorinsky\n")
    run = Run(log, "log.chtMultiRegionFoam", (), exit_code=None)
    probe = AsRun(tmp_path, run, solver=True, turbulence_block=True)
    assert probe._regime("constant/air/momentumTransport") == "LES"


def test_vpa_r3_a_stray_region_properties_leaves_the_log_deciding_the_regime(tmp_path):
    """A single-region solver's log prints one `Selecting`, which is the case's: a stray
    `regionProperties` beside it does not send the regime back to the file."""
    from courant.spec.as_run import AsRun, Run

    (tmp_path / "constant").mkdir()
    (tmp_path / "constant" / "regionProperties").write_text("regions (fluid (air));\n")
    (tmp_path / "constant" / "momentumTransport").write_text("simulationType LES;\n")
    log = "Exec : simpleFoam\nSelecting RAS turbulence model kEpsilon\n"
    run = Run(log, "log.simpleFoam", (), exit_code=None)
    probe = AsRun(tmp_path, run, solver=True, turbulence_block=True)
    assert probe.regions is False and probe._regime("constant/momentumTransport") == "RAS"
