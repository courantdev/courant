"""Mutation-070, courant/spec/check.py lines > 202: what the conformance block says on a bound
(SPEC-12, VER-5.7) and what `draft_spec` drafts (P2-SPEC-2). Each test names the mutants it
kills."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from courant.spec import Spec, check_conformance, draft_spec

HEAD = "FoamFile\n{\n    version 2.0;\n    format ascii;\n}\n"


def _spec(*items: dict) -> Spec:
    return Spec.model_validate({"schema": "courant.spec/1", "items": list(items)})


def _put(case: Path, rel: str, body: str) -> None:
    p = case / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD + body)


def _field(case: Path, name: str, patches: str, where: str = "0") -> None:
    _put(case, f"{where}/{name}", "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
                                  "boundaryField\n{\n" + patches + "}\n")


def _by_name(spec: Spec) -> dict:
    return {i.name: i for i in spec.items}


# ---------------------------------------------------------------- bounds in the conformance block
FACES = "nonuniform List<scalar> 5(300 300 300 300 900)"


def _faces_case(tmp_path: Path) -> Path:
    _put(tmp_path, "0/T", "boundaryField\n{\n    floor { type fixedValue; value " + FACES + "; }\n}\n")
    return tmp_path


# f9d8b3bd54, ab7320282d: check.py:204 (`not ok` -> False / `ok`): the note on a per-face list
# outside its bound names the face values and the bound.
def test_a_per_face_list_outside_its_bound_says_so_in_the_note(tmp_path):
    case = _faces_case(tmp_path)
    item = check_conformance(_spec({"name": "T", "file": "0/T", "patch": "floor", "max": 400.0}),
                             case, spec_sha256="x").items[0]
    assert item.conforms is False
    assert item.note == "face values 300 to 900 in the case, ≤ 400 specified"


# 9ba8219f71, ab7320282d: check.py:204 (`not ok` -> True / `ok`): a per-face list inside its
# bound carries no nonconformance note.
def test_a_per_face_list_inside_its_bound_has_no_nonconformance_note(tmp_path):
    case = _faces_case(tmp_path)
    item = check_conformance(_spec({"name": "T", "file": "0/T", "patch": "floor", "min": 250.0,
                                    "max": 1000.0}), case, spec_sha256="x").items[0]
    assert item.conforms is True
    assert item.note is None


def _vector_case(tmp_path: Path) -> Path:
    _field(tmp_path, "U", "    inlet { type fixedValue; value uniform (3 4 0); }\n")
    return tmp_path


# 80b44e48ca, f46132e220: check.py:208 (`q is not None` -> False / `is`): a bounded vector is
# shown as the magnitude the bound was compared with, not as the vector.
def test_a_bounded_vector_shows_its_magnitude(tmp_path):
    case = _vector_case(tmp_path)
    item = check_conformance(_spec({"name": "U", "file": "0/U", "patch": "inlet", "max": 10.0}),
                             case, spec_sha256="x").items[0]
    assert item.conforms is True
    assert item.found == "5"


# 3e06a8d869: check.py:211 (`ok is False` -> True): a bounded scalar inside its bound carries no
# nonconformance note; one outside it does.
def test_a_bounded_scalar_note_only_when_outside(tmp_path):
    _put(tmp_path, "constant/transportProperties", "nu 1e-05;\n")
    nu = {"name": "nu", "file": "constant/transportProperties", "key": "nu"}
    inside = check_conformance(_spec({**nu, "max": 1.0}), tmp_path, spec_sha256="x").items[0]
    outside = check_conformance(_spec({**nu, "max": 1e-06}), tmp_path, spec_sha256="x").items[0]
    assert inside.conforms is True and inside.note is None
    assert outside.conforms is False
    assert outside.note == "1e-05 in the case, ≤ 1e-06 specified"


# b822f26847, f46132e220: check.py:208 (`q is not None` -> True / `is`): a bound on endTime when
# the run stops otherwise (`stopAt writeNow`) shows what the case says, never an empty value.
def test_a_bound_on_a_value_read_as_text_shows_the_text(tmp_path):
    _put(tmp_path, "system/controlDict", "application simpleFoam;\nendTime 100;\nstopAt writeNow;\n")
    item = check_conformance(_spec({"name": "endTime", "file": "system/controlDict",
                                    "key": "endTime", "max": 200.0}),
                             tmp_path, spec_sha256="x").items[0]
    assert item.conforms is None
    assert item.found == "stopAt writeNow"


# ---------------------------------------------------------------- draft_spec: properties
def _case_a(case: Path) -> Path:
    """transportProperties, thermophysicalProperties, g, controlDict, blockMeshDict `scale` (no
    blocks), turbulenceProperties RAS."""
    _put(case, "constant/transportProperties", "nu [0 2 -1 0 0 0 0] 1e-05;\nsigma 0.07;\n")
    _put(case, "constant/thermophysicalProperties",
         "mixture\n{\n    specie { molWeight 28.9; }\n    thermodynamics { Cp 1005; }\n"
         "    transport { mu 1.8e-05; }\n}\n")
    _put(case, "constant/g", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n")
    _put(case, "system/controlDict", "application simpleFoam;\nendTime 100;\n")
    _put(case, "system/blockMeshDict", "scale 0.1;\n")
    _put(case, "constant/turbulenceProperties",
         "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n}\n")
    return case


def _case_b(case: Path) -> Path:
    """physicalProperties nu, blockMeshDict `convertToMeters`, turbulenceProperties LES."""
    _put(case, "constant/physicalProperties", "nu 1e-06;\n")
    _put(case, "system/blockMeshDict", "convertToMeters 0.01;\n")
    _put(case, "constant/turbulenceProperties",
         "simulationType LES;\nLES\n{\n    LESModel Smagorinsky;\n}\n")
    return case


def _case_c(case: Path) -> Path:
    """momentumTransport RAS."""
    _put(case, "constant/momentumTransport", "simulationType RAS;\nRAS\n{\n    model kOmegaSST;\n}\n")
    return case


def _case_d(case: Path) -> Path:
    """momentumTransport LES."""
    _put(case, "constant/momentumTransport", "simulationType LES;\nLES\n{\n    model WALE;\n}\n")
    return case


# 2708ac8039 (check.py:254), 8cf3db9906 (:256), 19173a34b2 (:257), d69df3c639 (:258),
# 2e5b443951 (:259), 38d955fd23 (:260), c6733e356f (:261), 6238a6dbf7 (:262), f06c4d0572 (:263),
# 929a089d29 (:265), 3b60b4e524 (:266), f0f5577c50 (:255), 2fd4b1cc5f (:267),
# 78584c9716 (:268), f1740d3f1b (:269), a656d5d369 (:270): `_PROPERTIES`' modelling-choice flags
# (True <-> False). A property is part of the problem; the solver and turbulence model are
# modelling choices.
@pytest.mark.parametrize("build, file, key, name, choice", [
    (_case_a, "constant/transportProperties", "nu", "nu", False),
    (_case_b, "constant/physicalProperties", "nu", "nu", False),
    (_case_a, "constant/thermophysicalProperties", "mu", "mu", False),
    (_case_a, "constant/thermophysicalProperties", "Cp", "Cp", False),
    (_case_a, "constant/thermophysicalProperties", "molWeight", "molWeight", False),
    (_case_a, "constant/transportProperties", "sigma", "surface tension", False),
    (_case_a, "constant/g", "value", "gravity", False),
    (_case_a, "system/controlDict", "application", "solver", True),
    (_case_a, "system/controlDict", "endTime", "endTime", False),
    (_case_a, "system/blockMeshDict", "scale", "mesh scale", False),
    (_case_a, "constant/turbulenceProperties", "simulationType", "turbulence regime", True),
    (_case_a, "constant/turbulenceProperties", "RASModel", "turbulence model", True),
    (_case_b, "constant/turbulenceProperties", "LESModel", "turbulence model", True),
    (_case_c, "constant/momentumTransport", "simulationType", "turbulence regime", True),
    (_case_c, "constant/momentumTransport", "model", "turbulence model", True),
    (_case_d, "constant/momentumTransport", "model", "turbulence model", True),
])
def test_each_drafted_property_carries_its_modelling_choice(tmp_path, build, file, key, name,
                                                            choice):
    items = _by_name(draft_spec(build(tmp_path)))
    assert (items[name].file, items[name].key) == (file, key)
    assert items[name].modelling_choice is choice


# 11fe0155f3: check.py:379 (`found.dimensions` -> False): a property's dimension set is drafted.
# 1406db4e9c: check.py:379 (`found.dimensions` -> True): an empty one is drafted as none.
def test_a_drafted_property_carries_its_dimensions_and_an_empty_set_is_none(tmp_path):
    items = _by_name(draft_spec(_case_a(tmp_path / "a")))
    assert items["nu"].dimensions == (0, 2, -1, 0, 0, 0, 0)
    _put(tmp_path / "e", "constant/transportProperties", "nu [] 1e-05;\n")
    assert _by_name(draft_spec(tmp_path / "e"))["nu"].dimensions is None


# 2d443296ae: check.py:276 (`_MODULE`'s modelling choice True -> False): under --checks next, a
# foamRun case's solver module is drafted as a modelling choice.
def test_the_foamrun_module_is_drafted_as_a_modelling_choice(tmp_path):
    _put(tmp_path, "system/controlDict",
         "application foamRun;\nsolver incompressibleFluid;\nendTime 1;\n")
    module = _by_name(draft_spec(tmp_path, checks="next"))["solver module"]
    assert module.value == "incompressibleFluid"
    assert module.modelling_choice is True


# ---------------------------------------------------------------- draft_spec: boundary conditions
# bdc8a5962b, 61f97c30fe: check.py:341 (`and` -> `or`, either one): a hidden file or a `.orig`
# file in 0/ is not a field the solver reads, and nothing is drafted from it.
def test_hidden_and_orig_files_in_zero_are_not_drafted(tmp_path):
    patch = "    inlet { type fixedValue; value uniform (1 0 0); }\n"
    _field(tmp_path, "U", patch)
    _field(tmp_path, ".U", patch)
    _field(tmp_path, "U.orig", patch)
    names = set(_by_name(draft_spec(tmp_path)))
    assert names == {"U at inlet"}


# 20e54e2ce5: check.py:349 (`bf is None` -> False): a 0/ file without a boundaryField is skipped,
# not a crash.
def test_a_zero_file_without_a_boundary_field_is_skipped(tmp_path):
    _field(tmp_path, "U", "    inlet { type fixedValue; value uniform (1 0 0); }\n")
    _put(tmp_path, "0/notes", "something 1;\n")
    assert set(_by_name(draft_spec(tmp_path))) == {"U at inlet"}


# d83b37090b: check.py:356 (the label's quoting -> never): a patch named by a pattern is quoted
# in the item's name.
def test_a_pattern_patch_is_quoted_in_the_drafted_name(tmp_path):
    _field(tmp_path, "U", '    "(left|right)" { type fixedValue; value uniform (0 0 0); }\n')
    items = _by_name(draft_spec(tmp_path))
    assert set(items) == {'U at "(left|right)"'}
    assert items['U at "(left|right)"'].patch == "(left|right)"


# 0c767a8fdc: check.py:357 (`key == "value"` -> True), 34275390d4 / 5719735d4c / 1d61c3dc26:
# check.py:359 (`key == "value"` -> True / `!=` / False): a `value` item is named by its patch and
# has no key; any other entry is named by it and keyed by it.
def test_a_drafted_boundary_item_is_named_and_keyed_by_its_entry(tmp_path):
    _field(tmp_path, "U", "    inlet { type fixedValue; value uniform (1 0 0); }\n"
                          "    outlet { type inletOutlet; inletValue uniform (0 0 0); "
                          "value uniform (0 0 0); }\n")
    items = _by_name(draft_spec(tmp_path))
    assert set(items) == {"U at inlet", "U at outlet (inletValue)"}
    assert items["U at inlet"].key is None
    assert items["U at outlet (inletValue)"].key == "inletValue"
    assert items["U at outlet (inletValue)"].value == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- draft_spec: phases, mesh, geometry
# 8d1f7ff966 (check.py:384, `[0]` -> `[1]`), f262be9b93 (:385, `phases_raw` -> False),
# e7b8619996 / 700c38863a (:391, `is not None` -> False / `is`), b5b665bf70 (:393, `or` -> `and`):
# each named phase's nu and rho are drafted, with their dimensions.
def test_each_phase_property_is_drafted_with_its_dimensions(tmp_path):
    _put(tmp_path, "constant/transportProperties",
         "phases (water air);\nwater\n{\n    nu [0 2 -1 0 0 0 0] 1e-06;\n    rho 1000;\n}\n"
         "air\n{\n    nu 1.48e-05;\n}\n")
    items = _by_name(draft_spec(tmp_path))
    assert {"water nu", "water rho", "air nu"} <= set(items)
    assert "air rho" not in items
    assert items["water nu"].path == ("water",)
    assert items["water nu"].dimensions == (0, 2, -1, 0, 0, 0, 0)
    assert items["water rho"].value == 1000.0


BLOCKS = ("vertices ();\nblocks\n(\n    hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1 1 1)\n);\n")


# 2d7381d06e, a2139dd80e: check.py:396 (`include_mesh and ...` -> True / `or`): without
# include_mesh no cell counts are drafted; with it they are.
def test_include_mesh_false_drafts_no_cell_counts(tmp_path):
    _put(tmp_path, "system/blockMeshDict", "convertToMeters 1;\n" + BLOCKS)
    assert "mesh cells" in _by_name(draft_spec(tmp_path))
    assert "mesh cells" not in _by_name(draft_spec(tmp_path, include_mesh=False))


# 7e3f1fe720: check.py:399 (`cells` -> True): a blockMeshDict without literal cell counts drafts
# its scale and no cell item, and does not crash.
def test_a_blockmeshdict_without_cell_counts_drafts_no_cells(tmp_path):
    _put(tmp_path, "system/blockMeshDict", "convertToMeters 0.1;\n")
    assert set(_by_name(draft_spec(tmp_path))) == {"mesh scale"}


# 70329782de (check.py:403, `tri.is_dir()` -> False), d9da878c80 (:405, `in` -> `not in`): every
# surface file in constant/triSurface is drafted by its SHA-256, and only surface files.
def test_surface_geometry_is_drafted_by_hash_and_only_surfaces(tmp_path):
    tri = tmp_path / "constant" / "triSurface"
    tri.mkdir(parents=True)
    (tri / "body.STL").write_bytes(b"solid body\nendsolid body\n")
    (tri / "README.txt").write_text("not a surface\n")
    items = _by_name(draft_spec(tmp_path))
    assert set(items) == {"geometry body.STL"}
    assert items["geometry body.STL"].sha256 == hashlib.sha256(
        b"solid body\nendsolid body\n").hexdigest()


# f70bde1405: check.py:410 (`not items` -> False): a case with nothing readable is refused, not
# drafted as an empty specification.
def test_nothing_readable_is_refused(tmp_path):
    (tmp_path / "system").mkdir()
    with pytest.raises(ValueError, match="could be read into a specification"):
        draft_spec(tmp_path)
