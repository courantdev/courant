"""The verdict-path audit's findings in the specification reader and drafter (4 Nov 2026,
`research/notes/verdict-path-audit/FINDINGS.md`).

Behind `--checks next` (GATE-13), each under its name in `courant.verify.checks.NEXT`:

- `spec.read_as_solver` (VPA-SPEC-1/3/4/6): a patch reached by one of its groups before any
  pattern, as v2406's `GeometricBoundaryField::readField` reaches it; an inline Function1 is not
  read as its last number; a dimension set in unit names is read in SI base exponents; a quoted
  word is the word.
- `spec.turbulence_block` (VPA-DRAFT-1): the turbulence model drafted, and compared, only from the
  block `simulationType` selects.

- `spec.draft_as_run` (VPA-DRAFT-2): a draft conforms to its own case whatever its `stopAt` or
  turbulence switch.

Open, as strict xfails: VPA-SPEC-5 (an `#include` after an entry, read under the default) and
VPA-DRAFT-3 (properties and boundary types the drafter leaves out).

Under the default mode every reading here is 0.7.0's (GATE-13): each test that changes a reading
asserts the default one too.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from courant.spec import check_conformance, deck, draft_spec
from courant.spec.model import SPEC_SCHEMA, Spec, SpecItem

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"
HOT_ROOM = ROOT / "tests" / "fixtures" / "buoyant" / "hotRoom_laminar_044"
HEATED_CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "heated_cavity" / "run"
PITZ = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__shipped"
SHA = "0" * 64

_HEAD = "FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n"


def _foam(cls: str, obj: str, body: str) -> str:
    return _HEAD.format(cls=cls, obj=obj) + body


def _write(case: Path, rel: str, text: str) -> None:
    p = case / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    return dst


def _check(case: Path, items, checks: str):
    spec = Spec(schema=SPEC_SCHEMA, items=tuple(items))
    return check_conformance(spec, case, spec_sha256=SHA, checks=checks)


def _item(block, name: str):
    return next(i for i in block.items if i.name == name)


# ---------------------------------------------------------------- VPA-SPEC-1: patch groups
_CONTROL = _foam("dictionary", "controlDict",
                 "application simpleFoam;\nstartFrom startTime;\nstartTime 0;\n"
                 "stopAt endTime;\nendTime 1;\ndeltaT 1;\n")


def _boundary(patches: str, n: int) -> str:
    return _foam("polyBoundaryMesh", "boundary", f"{n}\n(\n{patches}\n)\n")


def _grouped_case(tmp_path: Path, boundary_field: str, *, inlet_groups: str = "1(inflow)") -> Path:
    case = tmp_path / "grouped"
    _write(case, "system/controlDict", _CONTROL)
    _write(case, "constant/polyMesh/boundary", _boundary(
        f"    inlet\n    {{\n        type patch;\n        inGroups List<word> {inlet_groups};\n"
        "        nFaces 1;\n        startFace 0;\n    }\n"
        "    outlet\n    {\n        type patch;\n        nFaces 1;\n        startFace 1;\n    }\n"
        "    walls\n    {\n        type wall;\n        nFaces 1;\n        startFace 2;\n    }", 3))
    _write(case, "0/U", _foam("volVectorField", "U",
                              "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
                              f"boundaryField\n{{\n{boundary_field}\n}}\n"))
    return case


_INFLOW_THEN_PATTERN = (
    "    inflow\n    {\n        type fixedValue;\n        value uniform (1 0 0);\n    }\n"
    '    "in.*"\n    {\n        type fixedValue;\n        value uniform (10 0 0);\n    }\n'
    "    outlet\n    {\n        type zeroGradient;\n    }\n"
    "    walls\n    {\n        type noSlip;\n    }")


def test_vpa_spec_1_a_patch_group_entry_comes_before_a_pattern_under_next(tmp_path):
    """VPA-SPEC-1: v2406 applies a patch's group entry (`inflow`, the patch's `inGroups`) before
    any pattern (`"in.*"`): the inlet ran at (1 0 0), so a specification of (10 0 0) does not
    conform under next. The default reads the pattern, as 0.7.0 did."""
    case = _grouped_case(tmp_path, _INFLOW_THEN_PATTERN)
    item = SpecItem(name="U at inlet", file="0/U", patch="inlet", value=(10.0, 0.0, 0.0))
    default = _item(_check(case, [item], "default"), "U at inlet")
    assert default.conforms is True and default.found == "(10 0 0)"
    nxt = _item(_check(case, [item], "next"), "U at inlet")
    assert nxt.conforms is False
    assert nxt.found == "(1 0 0)"


def test_vpa_spec_1_the_wall_group_of_a_wall_patch(tmp_path):
    """VPA-SPEC-1: a patch of type wall is in the group `wall` whatever its inGroups
    (wallPolyPatch.C), so `wall { ... }` is applied to it before the pattern `".*"`."""
    case = _grouped_case(tmp_path, (
        "    wall\n    {\n        type fixedValue;\n        value uniform (0 0 0);\n    }\n"
        '    ".*"\n    {\n        type fixedValue;\n        value uniform (2 0 0);\n    }'))
    item = SpecItem(name="U at walls", file="0/U", patch="walls", value=(2.0, 0.0, 0.0))
    assert _item(_check(case, [item], "default"), "U at walls").conforms is True
    nxt = _item(_check(case, [item], "next"), "U at walls")
    assert nxt.conforms is False
    assert nxt.found == "(0 0 0)"


def test_vpa_spec_1_the_last_written_group_entry_wins(tmp_path):
    """VPA-SPEC-1: of two group entries naming one patch, the last written is applied
    (readField walks the dictionary in reverse); a literal patch entry beats both."""
    groups = (
        "    inflow\n    {\n        type fixedValue;\n        value uniform (1 0 0);\n    }\n"
        "    front\n    {\n        type fixedValue;\n        value uniform (3 0 0);\n    }")
    case = _grouped_case(tmp_path, groups, inlet_groups="2(inflow front)")
    item = SpecItem(name="U at inlet", file="0/U", patch="inlet", value=(3.0, 0.0, 0.0))
    assert _item(_check(case, [item], "next"), "U at inlet").conforms is True
    literal = _grouped_case(tmp_path / "lit", groups + (
        "\n    inlet\n    {\n        type fixedValue;\n        value uniform (5 0 0);\n    }"),
        inlet_groups="2(inflow front)")
    item5 = SpecItem(name="U at inlet", file="0/U", patch="inlet", value=(5.0, 0.0, 0.0))
    assert _item(_check(literal, [item5], "next"), "U at inlet").conforms is True


# ---------------------------------------------------------------- VPA-SPEC-3: Function1 values
def _flow_rate_case(tmp_path: Path, rate: str) -> Path:
    case = tmp_path / "rate"
    _write(case, "system/controlDict", _CONTROL)
    _write(case, "0/U", _foam("volVectorField", "U",
                              "internalField uniform (0 0 0);\nboundaryField\n{\n"
                              "    inlet\n    {\n        type flowRateInletVelocity;\n"
                              f"        volumetricFlowRate {rate};\n"
                              "        value uniform (0 0 0);\n    }\n}\n"))
    return case


@pytest.mark.parametrize("rate, last, word", [
    ("table ((0 0.5) (1 0.1))", 0.1, "table"),
    ("polynomial ((0.5 0) (-0.4 1))", 1.0, "polynomial"),
])
def test_vpa_spec_3_an_inline_function1_is_not_its_last_number(tmp_path, rate, last, word):
    """VPA-SPEC-3: `volumetricFlowRate table ((0 0.5) (1 0.1));` is a function of time; the
    default reads its last number (0.1, and 1 for the polynomial), next leaves it unread."""
    case = _flow_rate_case(tmp_path, rate)
    item = SpecItem(name="rate", file="0/U", patch="inlet", key="volumetricFlowRate", value=last)
    assert _item(_check(case, [item], "default"), "rate").conforms is True
    nxt = _item(_check(case, [item], "next"), "rate")
    assert nxt.conforms is None
    assert nxt.note == f"a function of time ({word}) the reader does not evaluate"


def test_vpa_spec_3_constant_and_the_pre_1_7_form_are_still_read(tmp_path):
    """VPA-SPEC-3: `constant 0.1` is 0.1, and `nu nu [0 2 -1 0 0 0 0] 1e-05;` is 1e-05, under
    next as under the default."""
    case = _flow_rate_case(tmp_path, "constant 0.1")
    _write(case, "constant/transportProperties",
           _foam("dictionary", "transportProperties", "nu nu [0 2 -1 0 0 0 0] 1e-05;\n"))
    items = [SpecItem(name="rate", file="0/U", patch="inlet", key="volumetricFlowRate", value=0.1),
             SpecItem(name="nu", file="constant/transportProperties", key="nu", value=1e-5,
                      dimensions=(0, 2, -1, 0, 0, 0, 0))]
    for checks in ("default", "next"):
        block = _check(case, items, checks)
        assert [i.conforms for i in block.items] == [True, True], checks
    assert deck.parse_value("constant 0.1", solver=True)[0] == 0.1


# ---------------------------------------------------------------- VPA-SPEC-4: unit names
@pytest.mark.parametrize("raw, dims", [
    ("[m^2/s] 1e-05", (0, 2, -1, 0, 0, 0, 0)),
    ("[kg/m^3] 1.2", (1, -3, 0, 0, 0, 0, 0)),
    ("[m^2 s^-1] 1e-05", (0, 2, -1, 0, 0, 0, 0)),
    ("[kg*m/s^2] 1", (1, 1, -2, 0, 0, 0, 0)),
    ("[Pa] 1e5", (1, -1, -2, 0, 0, 0, 0)),
    ("[N] 1", (1, 1, -2, 0, 0, 0, 0)),
    ("[J/kg/K] 1000", (0, 2, -2, -1, 0, 0, 0)),
    ("[W] 1", (1, 2, -3, 0, 0, 0, 0)),
    ("[0 2 -1 0 0 0 0] 1e-05", (0, 2, -1, 0, 0, 0, 0)),
    ("[0 2 -1 0 0] 1e-05", (0, 2, -1, 0, 0)),
    ("[furlong/fortnight] 1", None),
    ("[0 2 -1] 1", None),
])
def test_vpa_spec_4_a_dimension_set_in_unit_names_is_read_in_si_exponents(raw, dims):
    """VPA-SPEC-4: brackets are exponents only when every token is a number (5 or 7 of them);
    unit names are converted to the 7 SI base exponents; an unknown name is not stated (None).
    The default keeps 0.7.0's reading, the digits."""
    value, got, _ = deck.parse_value(raw, solver=True)
    assert got == (None if dims is None else tuple(float(x) for x in dims))
    assert value is not None
    assert deck.parse_value("[m^2/s] 1e-05")[1] == (2.0,)            # default: unchanged


def test_vpa_spec_4_a_draft_of_a_named_unit_case_conforms_to_it(tmp_path):
    """VPA-SPEC-4: a cavity with `nu [m^2/s] 0.00848;` conforms to its own next draft (no
    VER-5.11 impossibility from a misread dimension set), and to a specification of
    [0 2 -1 0 0 0 0]."""
    case = _copy(CAVITY, tmp_path)
    _write(case, "constant/transportProperties",
           _foam("dictionary", "transportProperties", "nu [m^2/s] 0.00848;\n"))
    draft = draft_spec(case, checks="next")
    nu = next(i for i in draft.items if i.name == "nu")
    assert nu.dimensions == (0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0)
    own = check_conformance(draft, case, spec_sha256=SHA, checks="next")
    assert not own.invalid and own.nonconformant == 0
    item = SpecItem(name="nu", file="constant/transportProperties", key="nu", value=0.00848,
                    dimensions=(0, 2, -1, 0, 0, 0, 0))
    assert _item(_check(case, [item], "next"), "nu").conforms is True
    assert _item(_check(case, [item], "default"), "nu").conforms is False     # 0.7.0's (2,)


# ---------------------------------------------------------------- VPA-SPEC-6: quoted words
def _quoted_hot_room(tmp_path: Path) -> Path:
    case = _copy(HOT_ROOM, tmp_path)
    tp = case / "constant" / "turbulenceProperties"
    tp.write_text(tp.read_text().replace("simulationType      laminar;",
                                         'simulationType      "laminar";'))
    t = case / "0" / "T"
    text = t.read_text()
    head, sep, rest = text.partition("    floor\n    {\n        type            fixedValue;")
    assert sep, "the fixture's floor patch moved"
    t.write_text(head + '    floor\n    {\n        type            "fixedValue";' + rest)
    return case


def test_vpa_spec_6_a_quoted_word_is_the_word(tmp_path):
    """VPA-SPEC-6: `simulationType "laminar";` and a patch's `type "fixedValue";` are the words
    OpenFOAM's word reader makes of them: they conform under next; the default keeps the quotes,
    as 0.7.0 did."""
    case = _quoted_hot_room(tmp_path)
    items = [SpecItem(name="regime", file="constant/turbulenceProperties", key="simulationType",
                      value="laminar"),
             SpecItem(name="floor type", file="0/T", patch="floor", key="type",
                      value="fixedValue")]
    default = _check(case, items, "default")
    assert [i.conforms for i in default.items] == [False, False]
    assert _item(default, "regime").found == '"laminar"'
    nxt = _check(case, items, "next")
    assert [i.conforms for i in nxt.items] == [True, True]
    names = {i.name for i in draft_spec(case, checks="next").items}
    assert "T at floor" in names
    assert "T at floor" not in {i.name for i in draft_spec(case).items}


# ---------------------------------------------------------------- VPA-DRAFT-1: the model's block
def _les_hot_room(tmp_path: Path) -> Path:
    case = _copy(HOT_ROOM, tmp_path)
    tp = case / "constant" / "turbulenceProperties"
    text = tp.read_text().replace("simulationType      laminar;", "simulationType      LES;")
    tp.write_text(text + "LES\n{\n    LESModel        Smagorinsky;\n    turbulence      on;\n"
                         "    delta           cubeRootVol;\n}\n")
    return case


def test_vpa_draft_1_the_model_is_drafted_from_the_block_simulation_type_selects(tmp_path):
    """VPA-DRAFT-1: an LES case keeping a RAS{kEpsilon} block drafts Smagorinsky under LES under
    next; the default drafts the first block listed (kEpsilon), as 0.7.0 did."""
    case = _les_hot_room(tmp_path)
    nxt = [i for i in draft_spec(case, checks="next").items if i.name == "turbulence model"]
    assert [(i.path, i.value) for i in nxt] == [(("LES",), "Smagorinsky")]
    default = [i for i in draft_spec(case).items if i.name == "turbulence model"]
    assert [(i.path, i.value) for i in default] == [(("RAS",), "kEpsilon")]


def test_vpa_draft_1_a_model_in_a_block_that_did_not_run_does_not_conform(tmp_path):
    """VPA-DRAFT-1: RAS/RASModel kEpsilon against a case whose simulationType is LES does not
    conform under next; the default compares the dead block, as 0.7.0 did."""
    case = _les_hot_room(tmp_path)
    item = SpecItem(name="turbulence model", file="constant/turbulenceProperties", path=("RAS",),
                    key="RASModel", value="kEpsilon", modelling_choice=True)
    assert _item(_check(case, [item], "default"), "turbulence model").conforms is True
    nxt = _item(_check(case, [item], "next"), "turbulence model")
    assert nxt.conforms is False
    assert nxt.note == "the RAS block is not the one the case runs (LES)"


def test_vpa_draft_1_a_laminar_case_drafts_no_model():
    """VPA-DRAFT-1: the committed laminar hotRoom drafts no turbulence model under next (the
    default drafts its dead RAS block's kEpsilon, as 0.7.0 did)."""
    assert "turbulence model" not in {i.name for i in draft_spec(HOT_ROOM, checks="next").items}
    assert "turbulence model" in {i.name for i in draft_spec(HOT_ROOM).items}


# ---------------------------------------------------------------- VPA-DRAFT-2: own case
def test_vpa_draft_2_a_stop_at_next_write_case_conforms_to_its_own_draft(tmp_path):
    """VPA-DRAFT-2 (`spec.draft_as_run`): under next a cavity with `stopAt nextWrite` conforms to
    its own draft: the endTime the run does not stop at is not drafted. The default's draft stays
    0.7.0's, the endTime in it, and does not conform (no measured basis to change it, GATE-13)."""
    case = _copy(CAVITY, tmp_path)
    cd = case / "system" / "controlDict"
    cd.write_text(cd.read_text().replace("stopAt          endTime;", "stopAt          nextWrite;"))
    draft = draft_spec(case, checks="next")
    own = check_conformance(draft, case, spec_sha256=SHA, checks="next")
    assert own.nonconformant == 0, [(i.name, i.note) for i in own.items if i.conforms is False]
    assert "endTime" not in {i.name for i in draft.items}
    default = draft_spec(case)
    assert "endTime" in {i.name for i in default.items}
    assert check_conformance(default, case, spec_sha256=SHA).nonconformant == 1


def test_vpa_draft_2_a_turbulence_off_case_conforms_to_its_own_draft(tmp_path):
    """VPA-DRAFT-2 (`spec.draft_as_run`): under next pitzDaily with `RAS { turbulence off; }`
    conforms to its own draft: the model whose equations are not solved is not drafted. The
    default's draft stays 0.7.0's, the model in it, and does not conform (GATE-13)."""
    case = _copy(PITZ, tmp_path)
    tp = case / "constant" / "turbulenceProperties"
    text = tp.read_text()
    assert "turbulence      on;" in text, "the fixture's switch moved"
    tp.write_text(text.replace("turbulence      on;", "turbulence      off;"))
    draft = draft_spec(case, checks="next")
    own = check_conformance(draft, case, spec_sha256=SHA, checks="next")
    assert own.nonconformant == 0, [(i.name, i.note) for i in own.items if i.conforms is False]
    assert "turbulence model" not in {i.name for i in draft.items}
    default = draft_spec(case)
    assert "turbulence model" in {i.name for i in default.items}
    assert check_conformance(default, case, spec_sha256=SHA).nonconformant >= 1


# ---------------------------------------------------------------- open findings
def _include_case(tmp_path: Path) -> Path:
    case = tmp_path / "inc"
    _write(case, "constant/transportProperties",
           _foam("dictionary", "transportProperties", 'nu 1e-05;\n#include "nuOverride"\n'))
    _write(case, "constant/nuOverride", "nu 2e-05;\n")
    return case


def test_vpa_spec_5_next_reads_the_include_after_the_entry(tmp_path):
    """VPA-SPEC-5: under next (`spec.directives`) the included `nu 2e-05;` is what is read."""
    case = _include_case(tmp_path)
    item = SpecItem(name="nu", file="constant/transportProperties", key="nu", value=1e-5)
    nxt = _item(_check(case, [item], "next"), "nu")
    assert nxt.conforms is False and nxt.found == "2e-05"


@pytest.mark.xfail(strict=True, reason="VPA-SPEC-5: open. Under the default an #include after "
                                       "an entry is not read (nu 1e-05 conforms where the solver "
                                       "used 2e-05); the fix is behind spec.directives (next), "
                                       "and promoting it is S's decision.")
def test_vpa_spec_5_the_default_reads_the_include_after_the_entry(tmp_path):
    """VPA-SPEC-5: `nu 1e-05;` then `#include "nuOverride"` (`nu 2e-05;`): the solver uses
    2e-05, so a specification of 1e-05 does not conform, under the default too."""
    case = _include_case(tmp_path)
    item = SpecItem(name="nu", file="constant/transportProperties", key="nu", value=1e-5)
    default = _item(_check(case, [item], "default"), "nu")
    assert default.found == "2e-05"
    assert default.conforms is False


@pytest.mark.xfail(strict=True, reason="VPA-DRAFT-3: open. The drafter leaves out beta, TRef, "
                                       "Pr, Prt, DT and the parameters of uniformFixedValue, "
                                       "surfaceNormalFixedValue, totalTemperature and fixedMean.")
def test_vpa_draft_3_a_boussinesq_draft_names_beta_tref_and_pr():
    """VPA-DRAFT-3: the heated cavity's draft names its Boussinesq properties."""
    names = {i.name for i in draft_spec(HEATED_CAVITY).items}
    assert {"beta", "TRef", "Pr"} <= names
