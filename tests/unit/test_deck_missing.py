"""A named location that is not there (TRD VER-15, VER-5.7 and VER-5.9; review of T-GATE-UX).

`reject` on a missing location is claimed only when it is certain: the patch name is absent from
`constant/polyMesh/boundary` (the mesh's own list, groups included) and from every key of the
field's `boundaryField`. Everything the reader could be wrong about stays `insufficient`:
a key with a synonym or an OpenFOAM default, a patch reached through a group, a regex key,
two blocks on one line. Each correct case below must not be rejected.
"""

from __future__ import annotations

from pathlib import Path

from courant.spec import check_conformance

BOUNDARY = """FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }
3
(
    inlet { type patch; nFaces 10; startFace 100; }
    outlet { type patch; nFaces 10; startFace 110; }
    lowerWall { type wall; inGroups List<word> 1(wall); nFaces 10; startFace 120; }
)
"""


def _case(tmp: Path, u_body: str, *, boundary: str | None = BOUNDARY,
          block_mesh: str = "convertToMeters 0.001;\n") -> Path:
    (tmp / "0").mkdir(parents=True)
    (tmp / "system").mkdir()
    (tmp / "constant" / "polyMesh").mkdir(parents=True)
    (tmp / "0" / "U").write_text("FoamFile { class volVectorField; object U; }\n"
                                 "internalField uniform (0 0 0);\n"
                                 f"boundaryField\n{{\n{u_body}\n}}\n")
    (tmp / "system" / "blockMeshDict").write_text("FoamFile { object blockMeshDict; }\n" + block_mesh)
    if boundary is not None:
        (tmp / "constant" / "polyMesh" / "boundary").write_text(boundary)
    return tmp


def _spec(*items):
    return {"schema": "courant.spec/1", "items": list(items)}


def _first(case: Path, item: dict):
    block = check_conformance(_spec(item), case, spec_sha256="h")
    return block.items[0].conforms, block


def _patch_item(patch: str, key: str, value):
    return {"name": f"{patch} {key}", "file": "0/U", "patch": patch, "key": key, "value": value}


STD = """    inlet { type fixedValue; value uniform (10 0 0); }
    outlet { type zeroGradient; }
    lowerWall { type noSlip; }
"""


def test_a_patch_absent_from_the_mesh_and_the_field_is_a_nonconformance(tmp_path):
    case = _case(tmp_path, STD)
    conforms, block = _first(case, _patch_item("upperWall", "type", "noSlip"))
    assert conforms is False and block.nonconformant == 1
    assert "no patch 'upperWall'" in block.items[0].note


def test_without_the_mesh_boundary_file_a_missing_patch_is_a_gap_not_a_reject(tmp_path):
    case = _case(tmp_path, STD, boundary=None)
    conforms, block = _first(case, _patch_item("upperWall", "type", "noSlip"))
    assert conforms is None and block.nonconformant == 0


def test_a_directive_in_the_field_file_could_hide_the_patch(tmp_path):
    case = _case(tmp_path, '    #includeEtc "caseDicts/setConstraintTypes"\n' + STD)
    conforms, block = _first(case, _patch_item("upperWall", "type", "noSlip"))
    assert conforms is None and block.nonconformant == 0


# ---- the five forms a correct case takes -------------------------------------------------
def test_form_1_a_synonym_is_read_as_the_key_the_spec_names(tmp_path):
    case = _case(tmp_path, STD, block_mesh="convertToMeters 0.001;\n")
    item = {"name": "mesh scale", "file": "system/blockMeshDict", "key": "scale", "value": 0.001}
    assert _first(case, item)[0] is True
    case2 = _case(tmp_path / "b", STD, block_mesh="scale 0.001;\n")
    item2 = {"name": "mesh scale", "file": "system/blockMeshDict", "key": "convertToMeters",
             "value": 0.001}
    assert _first(case2, item2)[0] is True


def test_form_2_an_omitted_scale_is_openfoams_default_of_one(tmp_path):
    case = _case(tmp_path, STD, block_mesh="")
    one = {"name": "mesh scale", "file": "system/blockMeshDict", "key": "scale", "value": 1}
    assert _first(case, one)[0] is True
    other = {"name": "mesh scale", "file": "system/blockMeshDict", "key": "scale", "value": 0.001}
    assert _first(case, other)[0] is False            # the default is 1: a real difference


def test_form_3_a_regex_key_and_separate_blocks_are_the_same_patches(tmp_path):
    regex = _case(tmp_path / "a", '    "(inlet|outlet)" { type fixedValue; value uniform (1 0 0); }\n'
                  "    lowerWall { type noSlip; }\n")
    assert _first(regex, _patch_item("outlet", "type", "fixedValue"))[0] is True
    split = _case(tmp_path / "b", STD)
    assert _first(split, _patch_item("inlet", "type", "fixedValue"))[0] is True
    # the spec names a patch the agent's file covers only through a regex that also covers others
    assert _first(regex, _patch_item("inlet", "type", "fixedValue"))[0] is True


def test_form_4_a_patch_reached_through_a_group_is_never_a_reject(tmp_path):
    case = _case(tmp_path, "    inlet { type fixedValue; value uniform (1 0 0); }\n"
                 "    outlet { type zeroGradient; }\n    wall { type noSlip; }\n")
    conforms, block = _first(case, _patch_item("lowerWall", "type", "noSlip"))
    assert conforms is not False and block.nonconformant == 0


def test_form_5_two_blocks_on_one_line_are_both_read(tmp_path):
    case = _case(tmp_path, "    inlet { type fixedValue; value uniform (10 0 0); } "
                 "outlet { type zeroGradient; }\n    lowerWall { type noSlip; }\n")
    assert _first(case, _patch_item("outlet", "type", "zeroGradient"))[0] is True
    assert _first(case, _patch_item("inlet", "type", "fixedValue"))[0] is True


def test_a_missing_key_stays_a_gap_not_a_reject(tmp_path):
    case = _case(tmp_path, STD)
    conforms, block = _first(case, _patch_item("outlet", "value", 0))
    assert conforms is None and block.nonconformant == 0
