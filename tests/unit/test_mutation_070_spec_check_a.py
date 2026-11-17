"""Mutation-070: deck-versus-spec conformance (`courant/spec/check.py`, lines up to 202).

Each test pins the original behaviour a surviving mutant changed: the comparison functions
(`conforms`, `bounded_quantity`, the tolerance and dimension checks) and what the conformance
block shows (`specified`, `tolerance`, `found`, `checked`). No solver, no network.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from courant.spec import Spec, SpecItem, check_conformance, conforms
from courant.spec.check import bounded_quantity

HEAD = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n}\n"


def _spec(*items: dict, **top) -> Spec:
    return Spec.model_validate({"schema": "courant.spec/1", "items": list(items), **top})


def _item(**kw) -> SpecItem:
    return SpecItem(**{"name": "x", "file": "constant/x", "key": "x", **kw})


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD + body)


# ---------------------------------------------------------------- how a number is shown
# kills f159a8bf67 (check.py:33 `<` -> `<=`) and fb35b0ea36 (check.py:33 `1e15` -> `2e15`)
def test_a_bound_of_1e15_or_more_is_shown_in_float_form():
    block = check_conformance(_spec({"name": "a", "file": "system/blockMeshDict",
                                     "blocks": "cells", "max": 1e15},
                                    {"name": "b", "file": "system/blockMeshDict",
                                     "blocks": "cells", "max": 1.5e15},
                                    {"name": "c", "file": "system/blockMeshDict",
                                     "blocks": "cells", "max": 999999999999999.0}),
                              None, spec_sha256="x")
    assert [i.specified for i in block.items] == [
        "≤ 1000000000000000.0", "≤ 1500000000000000.0", "≤ 999999999999999"]


# ---------------------------------------------------------------- _close, through conforms
# kills ad553d45d2 (check.py:38 `tol == 0.0` -> `False`): inf against inf with no tolerance
def test_an_infinite_value_conforms_to_the_same_infinite_value_with_no_tolerance():
    item = _item(value=math.inf)
    assert conforms(item, math.inf, rel=0.0, angle_deg=1.0) is True


# kills 395f072a59 (check.py:38 `0.0` -> `1.0`): a tolerance of exactly 1 is still a tolerance
def test_a_tolerance_of_exactly_one_accepts_a_value_inside_it():
    item = _item(value=10.0)
    assert conforms(item, 10.5, rel=0.1, angle_deg=1.0) is True      # tol = 0.1 * 10 = 1.0


# kills b6cdfbd98a (check.py:40 `<=` -> `<`): a value exactly at the tolerance conforms
def test_a_value_exactly_at_the_tolerance_conforms():
    item = _item(value=4.0)
    assert conforms(item, 6.0, rel=0.5, angle_deg=1.0) is True        # |6 - 4| = 2 = 0.5 * 4
    assert conforms(_item(value=4.0, abs_tol=0.5), 4.5, rel=0.0, angle_deg=1.0) is True


# ---------------------------------------------------------------- bounded_quantity (SPEC-12)
# kills 2160935735 (check.py:52 `len(counts) % 3` -> `False`)
def test_block_counts_not_in_threes_have_no_cell_total():
    item = SpecItem(name="mesh", file="system/blockMeshDict", blocks="cells", max=1e6)
    assert bounded_quantity(item, [20.0, 20.0]) is None


# kills 04a6e2f984 (check.py:54 `*` -> `/`)
def test_the_cell_total_is_the_sum_of_each_blocks_product():
    item = SpecItem(name="mesh", file="system/blockMeshDict", blocks="cells", max=1e6)
    assert bounded_quantity(item, [2.0, 3.0, 4.0, 10.0, 10.0, 1.0]) == 124.0


# ---------------------------------------------------------------- exact items
# kills 8689cd6b24 (check.py:79 COND False), 1affb2f436 (check.py:82 COND False),
# b5fb9c1868 (check.py:83 COND False), 23a3c7d980 (check.py:84 `==` -> `!=`),
# 0da0c4a670 (check.py:84 `return None`)
def test_a_files_sha256_conforms_whatever_the_case_of_its_hex_digits(tmp_path):
    _write(tmp_path, "system/fvSchemes", "ddtSchemes { default Euler; }\n")
    digest = hashlib.sha256((tmp_path / "system/fvSchemes").read_bytes()).hexdigest()
    block = check_conformance(_spec({"name": "schemes", "file": "system/fvSchemes",
                                     "sha256": digest.upper()}), tmp_path, spec_sha256="x")
    assert block.items[0].conforms is True
    other = check_conformance(_spec({"name": "schemes", "file": "system/fvSchemes",
                                     "sha256": "0" * 64}), tmp_path, spec_sha256="x")
    assert other.items[0].conforms is False


# kills 1affb2f436 (check.py:82 COND False) on block counts: they are compared exactly
def test_block_counts_are_compared_exactly_not_within_a_tolerance():
    item = SpecItem(name="mesh", file="system/blockMeshDict", blocks="cells",
                    value=(40.0, 40.0, 1.0))
    assert conforms(item, [40.0, 40.0, 1.0], rel=0.01, angle_deg=1.0) is True
    assert conforms(item, [40.0, 40.2, 1.0], rel=0.01, angle_deg=1.0) is False


# kills a0694afb53 (check.py:83 `or` -> `and`): a text value against block counts is a verdict,
# not a crash ("never raises on a case it cannot read")
def test_block_counts_written_as_text_give_a_verdict_not_a_crash(tmp_path):
    _write(tmp_path, "system/blockMeshDict",
           "vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1));\n"
           "blocks ( hex (0 1 2 3 4 5 6 7) (40 40 1) simpleGrading (1 1 1) );\n")
    block = check_conformance(_spec({"name": "mesh", "file": "system/blockMeshDict",
                                     "blocks": "cells", "value": "40 40 1"}), tmp_path,
                              spec_sha256="x")
    assert block.items[0].found == "(40 40 1)"
    assert block.items[0].conforms is False


# ---------------------------------------------------------------- scalars, lists, vectors
# kills 29529f7b5c (check.py:90 `and` -> `or`), 844ad6896e and 580eaecbb3 (check.py:92
# `or` -> `and`), 9335d659f9 (check.py:93 `False` -> `True`), ebe73cbf58 (check.py:93
# `return None`)
@pytest.mark.parametrize("want, found", [
    (1.0, [1.0, 0.0, 0.0]),
    ((1.0, 0.0, 0.0), 1.0),
])
def test_a_scalar_against_a_list_does_not_conform(want, found):
    assert conforms(_item(value=want), found, rel=0.01, angle_deg=1.0) is False


# kills 17cf067773 (check.py:92 COND False), 9335d659f9 and ebe73cbf58 (check.py:93)
def test_lists_of_different_lengths_do_not_conform():
    assert conforms(_item(value=(1.0, 2.0)), [1.0, 2.0, 3.0], rel=0.01, angle_deg=1.0) is False
    assert conforms(_item(value=(1.0, 2.0, 3.0, 4.0)), [1.0, 2.0, 3.0], rel=0.01,
                    angle_deg=1.0) is False


# kills 8532643982 (check.py:94 `len(want_l) == 3` -> `True`): a 2-list is compared component
# by component, not as a vector by magnitude and direction
def test_a_two_component_list_is_compared_component_by_component():
    assert conforms(_item(value=(3.0, 4.0)), [4.0, 3.0], rel=0.01, angle_deg=20.0) is False


# kills ba8533ce8d (check.py:103 `return None`)
def test_equal_lists_that_are_not_vectors_conform():
    assert conforms(_item(value=(1.0, 2.0)), [1.0, 2.0], rel=0.01, angle_deg=1.0) is True
    assert conforms(_item(value=(1.0, 2.0, 3.0, 4.0)), [1.0, 2.0, 3.0, 4.5], rel=0.01,
                    angle_deg=1.0) is False


# kills d58451044c (check.py:99 `or` -> `and`), 5d9d18aa98 and cc4aabd3d9 (check.py:100 `or`
# -> `and`): a zero vector specified, a found vector inside the absolute tolerance
def test_a_zero_vector_conforms_to_a_found_vector_inside_the_absolute_tolerance():
    item = _item(value=(0.0, 0.0, 0.0), abs_tol=1e-9)
    assert conforms(item, [0.0, 0.0, 1e-12], rel=0.01, angle_deg=1.0) is True


# kills fa8fef017a (check.py:100 `<=` -> `<`): exactly at the absolute tolerance conforms
def test_a_zero_vector_conforms_to_one_exactly_at_the_absolute_tolerance():
    item = _item(value=(0.0, 0.0, 0.0), abs_tol=0.5)
    assert conforms(item, [0.0, 0.0, 0.5], rel=0.01, angle_deg=1.0) is True


# kills e2abcf1e0d (check.py:102 `<=` -> `<`): a direction exactly at the angle tolerance
def test_a_direction_exactly_at_the_angle_tolerance_conforms():
    assert conforms(_item(value=(1.0, 0.0, 0.0)), [1.0, 0.0, 0.0], rel=0.01,
                    angle_deg=0.0) is True
    assert conforms(_item(value=(1.0, 0.0, 0.0)), [0.0, 1.0, 0.0], rel=0.01,
                    angle_deg=90.0) is True


# kills df31bd63fe (check.py:102 `1.0` -> `2.0`) and ce0d0d9393 (check.py:102 `-1.0` ->
# `-2.0`): a cosine rounded just past +-1 is clamped, never a math domain error
def test_a_cosine_rounded_past_one_is_clamped():
    v = (0.1, 0.1, 0.3)                       # |v|^2 rounds so that v.v / (|v||v|) > 1
    assert sum(x * x for x in v) / math.sqrt(sum(x * x for x in v)) ** 2 > 1.0
    assert conforms(_item(value=v), list(v), rel=0.01, angle_deg=1.0) is True
    assert conforms(_item(value=v), [-x for x in v], rel=0.01, angle_deg=1.0) is False


# ---------------------------------------------------------------- dimensions and tolerances
# kills 131554d171 (check.py:112 `0.0` -> `1.0`), 1a703773e2 (`7` -> `6`), 29be068076
# (`-` -> `+`): a 5-exponent dimension set equals the same set written with 7
def test_five_dimension_exponents_match_the_same_seven(tmp_path):
    _write(tmp_path, "constant/transportProperties", "nu [0 2 -1 0 0 0 0] 1e-05;\n")
    _write(tmp_path, "constant/x", "a [1 0 0 0 0 0 0] 2;\n")
    item = {"name": "nu", "file": "constant/transportProperties", "key": "nu", "value": 1e-05,
            "dimensions": [0, 2, -1, 0, 0]}
    block = check_conformance(_spec(item), tmp_path, spec_sha256="x")
    assert block.items[0].conforms is True
    wrong = check_conformance(_spec({"name": "a", "file": "constant/x", "key": "a",
                                     "value": 2.0, "dimensions": [0, 1, 0, 0, 0]}),
                              tmp_path, spec_sha256="x")
    assert wrong.items[0].conforms is False
    assert wrong.items[0].note == "dimensions (1 0 0 0 0 0 0) in the case, (0 1 0 0 0) specified"


# kills b7b4bb6761 (check.py:180 COND False) and d0a76e3d65 (check.py:181 COND False): an
# item's own tolerances override the specification's
def test_an_items_own_tolerances_override_the_specifications(tmp_path):
    _write(tmp_path, "constant/transportProperties", "nu 1.05e-05;\n")
    _write(tmp_path, "constant/g", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0.5 -9.81);\n")
    nu = {"name": "nu", "file": "constant/transportProperties", "key": "nu", "value": 1e-05}
    g = {"name": "g", "file": "constant/g", "key": "value", "value": [0, 0, -9.81]}
    tight = check_conformance(_spec(nu, g, rel_tol=0.01, angle_tol_deg=1.0), tmp_path,
                              spec_sha256="x")
    assert [i.conforms for i in tight.items] == [False, False]
    loose = check_conformance(_spec({**nu, "rel_tol": 0.1}, {**g, "angle_tol_deg": 5.0},
                                    rel_tol=0.01, angle_tol_deg=1.0),
                              tmp_path, spec_sha256="x")
    assert [i.conforms for i in loose.items] == [True, True]
    assert [i.tolerance for i in loose.items] == ["10%", "1%, 5°"]


# kills e8ac73ae14, 4a6bc67b29, 11bbaf0dc5, b821e85f3e (check.py:120, what `specified` shows),
# 786e00f983 (124), 6872bf4c21 and 1dfcc4ab91 (126), 8bfbe9229f and 9c116ebb8a (128),
# 8af5567165, 3d241c4cd7, 36ee8e81bd (130), f0877af2ab (131), 54e6b5bfbf, ffccdd8d89,
# 6d08ad37d8, 5c8ffc0776, d707f39783 (133): what each item says was specified, and its tolerance
def test_each_kind_of_item_states_what_was_specified_and_its_tolerance():
    sha = "ab" * 32
    block = check_conformance(_spec(
        {"name": "nu", "file": "constant/transportProperties", "key": "nu", "value": 1e-05,
         "abs_tol": 0.5},
        {"name": "dt", "file": "system/controlDict", "key": "deltaT", "value": 0.001},
        {"name": "g", "file": "constant/g", "key": "value", "value": [0, 0, -9.81]},
        {"name": "pair", "file": "constant/x", "key": "pair", "value": [1, 2]},
        {"name": "quad", "file": "constant/x", "key": "quad", "value": [1, 2, 3, 4]},
        {"name": "solver", "file": "system/controlDict", "key": "application",
         "value": "simpleFoam"},
        {"name": "mesh", "file": "system/blockMeshDict", "blocks": "cells",
         "value": [20, 20, 1]},
        {"name": "schemes", "file": "system/fvSchemes", "sha256": sha},
        {"name": "cells", "file": "system/blockMeshDict", "blocks": "cells", "max": 1000},
    ), None, spec_sha256="x")
    assert [i.specified for i in block.items] == [
        "1e-05", "0.001", "(0 0 -9.81)", "(1 2)", "(1 2 3 4)",
        "simpleFoam", "(20 20 1)", sha, "≤ 1000"]
    assert [i.tolerance for i in block.items] == [
        "1%, or 0.5", "1%", "1%, 1°", "1%", "1%", "exact", "exact", "exact", "bound"]


# ---------------------------------------------------------------- the block
# kills f6c76f6bf9 (check.py:176 `0` -> `1`): an impossible specification checks nothing
def test_an_impossible_specification_checks_no_item():
    block = check_conformance(_spec({"name": "cells", "file": "system/blockMeshDict",
                                     "blocks": "cells", "min": 100, "max": 10}),
                              None, spec_sha256="x")
    assert block.invalid
    assert (block.checked, block.unreadable) == (0, 1)


# kills 5b0c3a8bb8 (check.py:202 `lo == hi` -> `True`), c0de124c7a (`==` -> `!=`),
# bcf143e791 (`lo == hi` -> `False`): one face value is shown as a number, a spread as a range
def test_a_per_face_list_shows_its_value_or_its_range(tmp_path):
    def field(faces: str) -> Path:
        _write(tmp_path, "0/U", "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
                                "boundaryField\n{\n    inlet\n    {\n        type fixedValue;\n"
                                f"        value {faces};\n    }}\n}}\n")
        return tmp_path

    item = {"name": "U", "file": "0/U", "patch": "inlet", "max": 60.0}
    same = check_conformance(_spec(item), field(
        "nonuniform List<vector> 3((5 0 0) (5 0 0) (5 0 0))"), spec_sha256="x")
    assert same.items[0].found == "5"
    spread = check_conformance(_spec(item), field(
        "nonuniform List<vector> 3((0 0 0) (50 0 0) (50 0 0))"), spec_sha256="x")
    assert spread.items[0].found == "0 to 50"
