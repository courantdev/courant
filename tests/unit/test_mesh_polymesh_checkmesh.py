"""TRD VER-7a, pinned to OpenFOAM: the aspect ratio computed from a polyMesh equals checkMesh's.

The meshes are the ones blockMesh wrote in the pinned ESI v2406 image (13 Oct 2026, one
container, two cores): the four dev tutorials whose meshes are not unit squares, and the
cavity as `faults.extreme_aspect_ratio` meshes it. Each is committed gzipped with the
checkMesh log of the same container. The expected numbers are the certification corpus's,
recorded by checkMesh in an earlier run (`certification/corpus/*.case.json`), so the formula is
checked against two independent checkMesh runs, never against itself.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest

from courant.case.mesh import POLYMESH_SOURCE, MeshQuality, polymesh_quality
from courant.case.polymesh import FILES

ROOT = Path(__file__).resolve().parents[2]
MESHES = ROOT / "tests" / "fixtures" / "real_openfoam" / "polymesh"
CORPUS = ROOT / "certification" / "corpus"

CASES = {                              # mesh fixture -> certification corpus sidecar
    "interFoam_damBreak": "interFoam_damBreak__healthy",
    "simpleFoam_pitzDaily": "simpleFoam_pitzDaily__healthy",
    "pimpleFoam_planarContraction": "pimpleFoam_planarContraction__healthy",
    "buoyantPimpleFoam_hotRoom": "buoyantPimpleFoam_hotRoom__healthy",
    "icoFoam_cavity__extreme_aspect_ratio": "icoFoam_cavity__extreme_aspect_ratio",
}


def _case(tmp: Path, name: str) -> Path:
    poly = tmp / name / "constant" / "polyMesh"
    poly.mkdir(parents=True)
    for n in FILES:
        with gzip.open(MESHES / name / "constant" / "polyMesh" / f"{n}.gz", "rb") as src, \
                open(poly / n, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return tmp / name


def _recorded(sidecar: str) -> float:
    return json.loads((CORPUS / f"{sidecar}.case.json").read_text())["mesh"]["max_aspect_ratio"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_aspect_ratio_is_checkmeshs_on_meshes_openfoam_wrote(tmp_path, name):
    recorded = _recorded(CASES[name])                     # 2.35093, 8.1407, 76.857148, 1, 2000
    same_container = MeshQuality.from_checkmesh(
        (MESHES / name / "log.checkMesh").read_text()).max_aspect_ratio
    got = polymesh_quality(str(_case(tmp_path, name)))
    assert got is not None and got.source == POLYMESH_SOURCE
    # Equal to checkMesh's last printed digit: rounded to as many significant figures as
    # checkMesh printed, the two are the same number.
    assert _as_printed(got.max_aspect_ratio, recorded) == recorded
    assert _as_printed(got.max_aspect_ratio, same_container) == same_container


def _as_printed(value: float, printed: float) -> float:
    digits = len(f"{printed:.12g}".replace(".", "").replace("-", "").lstrip("0").rstrip("0")) or 1
    return float(f"{value:.{max(digits, 6)}g}")      # OpenFOAM writes 6 by default, zeros dropped


def test_the_recorded_values_are_the_ones_the_review_named():
    assert {n: _recorded(s) for n, s in CASES.items()} == {
        "interFoam_damBreak": 2.35093, "simpleFoam_pitzDaily": 8.1407,
        "pimpleFoam_planarContraction": 76.857148, "buoyantPimpleFoam_hotRoom": 1.0,
        "icoFoam_cavity__extreme_aspect_ratio": 2000.0}


def test_the_fixtures_are_ascii_and_their_cell_counts_are_checkmeshs(tmp_path):
    for name in CASES:
        case = _case(tmp_path, name)
        head = (case / "constant/polyMesh/faces").read_text()[:2000]
        assert "format      ascii" in head or "format ascii" in " ".join(head.split())
        log = MeshQuality.from_checkmesh((MESHES / name / "log.checkMesh").read_text())
        assert polymesh_quality(str(case)).n_cells == log.n_cells
