"""Task specifications and deck-versus-spec conformance (PRD 2.0 P2-SPEC-1 to P2-SPEC-3).

The decks here are written the way OpenFOAM writes them, including the shapes that broke
readers before: `$macro` values, `#include`d values, patches grouped under a regex key,
dimensioned entries, and a boundary value rewritten into a per-face list by the run (RRD
RM-18 found that one on hotRoom).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from courant.spec import (
    Spec,
    SpecError,
    SpecItem,
    check_conformance,
    deck,
    draft_spec,
    load_spec,
    spec_sha256,
)

HEAD = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object x;\n}\n"


def _w(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD + body)


@pytest.fixture
def case(tmp_path: Path) -> Path:
    c = tmp_path / "case"
    _w(c, "system/controlDict", "application simpleFoam;\nendTime 300;\ndeltaT 1;\n")
    _w(c, "system/blockMeshDict", """scale 0.001;
vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1) );
blocks
(
    hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1 1 1)
    hex (0 1 2 3 4 5 6 7) (30 10 1) simpleGrading (1 1 1)
);
""")
    _w(c, "constant/transportProperties", "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1e-05;\n")
    _w(c, "constant/turbulenceProperties",
       "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n    turbulence on;\n}\n")
    _w(c, "constant/g", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n")
    _w(c, "0/U", """Uinlet (10 0 0);
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform $Uinlet;
    }
    outlet
    {
        type zeroGradient;
    }
    "(upperWall|lowerWall)"
    {
        type fixedValue;
        value uniform (0 0 0);
    }
    #includeEtc "caseDicts/setConstraintTypes"
}
""")
    _w(c, "0/k", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0.375;
boundaryField
{
    inlet { type fixedValue; value uniform 0.375; }
    outlet { type zeroGradient; }
    upperWall { type kqRWallFunction; value uniform 0.375; }
}
""")
    # The run rewrote 0/T's floor into a per-face list; the set-up value lives in 0.orig.
    _w(c, "0/T", """dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    floor { type fixedValue; value nonuniform List<scalar> 3(300 600 300); }
}
""")
    _w(c, "0.orig/T", """dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    floor { type fixedValue; value uniform 300; }
}
""")
    return c


def _spec(*items: dict, **top) -> Spec:
    return Spec.model_validate({"schema": "courant.spec/1", "items": list(items), **top})


# ---------------------------------------------------------------- the reader
class TestDeckReader:
    def test_macro_value_is_resolved(self, case):
        f = deck.read(case, SpecItem(name="u", file="0/U", patch="inlet", value=(0, 0, 0)))
        assert f.value == [10.0, 0.0, 0.0]

    def test_regex_patch_group_is_found(self, case):
        f = deck.read(case, SpecItem(name="w", file="0/U", patch="lowerWall", value=(0, 0, 0)))
        assert f.value == [0.0, 0.0, 0.0]

    def test_dimensioned_entry_gives_value_and_dimensions(self, case):
        f = deck.read(case, SpecItem(name="nu", file="constant/transportProperties", key="nu",
                                     value=0.0))
        assert f.value == 1e-05
        assert f.dimensions == (0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0)

    def test_key_inside_a_block(self, case):
        f = deck.read(case, SpecItem(name="m", file="constant/turbulenceProperties",
                                     path=("RAS",), key="RASModel", value="x"))
        assert f.value == "kEpsilon"

    def test_a_block_header_is_never_read_as_an_entry(self, tmp_path):
        _w(tmp_path, "constant/transportProperties",
           "phases (water air);\nwater\n{\n    nu 1e-06;\n}\nair\n{\n    nu 1.48e-05;\n}\nsigma 0.07;\n")
        assert deck.read(tmp_path, SpecItem(name="n", file="constant/transportProperties",
                                            key="nu", value=0.0)).value is None
        assert deck.read(tmp_path, SpecItem(name="w", file="constant/transportProperties",
                                            path=("water",), key="nu", value=0.0)).value == 1e-06
        assert deck.read(tmp_path, SpecItem(name="s", file="constant/transportProperties",
                                            key="sigma", value=0.0)).value == 0.07

    def test_nonuniform_value_is_read_on_its_mode(self, case):
        f = deck.read(case, SpecItem(name="t", file="0/T", patch="floor", value=0.0))
        assert f.value == 300.0 and f.file == "0/T"
        assert f.basis == "most common of 3 face values (2)"

    def test_a_missing_0_file_is_read_from_0_orig(self, case):
        (case / "0/T").unlink()
        f = deck.read(case, SpecItem(name="t", file="0/T", patch="floor", value=0.0))
        assert f.value == 300.0 and f.file == "0.orig/T"

    def test_include_and_macro(self, tmp_path):
        _w(tmp_path, "0/include/initialConditions", "flowVelocity (20 0 0);\n")
        _w(tmp_path, "0/U", '#include "include/initialConditions"\nboundaryField\n{\n'
                            "    inlet { type fixedValue; value uniform $flowVelocity; }\n}\n")
        f = deck.read(tmp_path, SpecItem(name="u", file="0/U", patch="inlet", value=(0, 0, 0)))
        assert f.value == [20.0, 0.0, 0.0]

    def test_include_outside_the_case_is_not_read(self, tmp_path):
        outside = tmp_path / "secret"
        _w(outside, "vals", "flowVelocity (99 0 0);\n")
        c = tmp_path / "case"
        _w(c, "0/U", '#include "../../secret/vals"\nboundaryField\n{\n'
                     "    inlet { type fixedValue; value uniform $flowVelocity; }\n}\n")
        f = deck.read(c, SpecItem(name="u", file="0/U", patch="inlet", value=(0, 0, 0)))
        assert f.value is None and "could not be found" in f.note

    def test_cells_are_every_hex_block(self, case):
        f = deck.read(case, SpecItem(name="c", file="system/blockMeshDict", blocks="cells",
                                     value=(0,)))
        assert f.value == [20.0, 10.0, 1.0, 30.0, 10.0, 1.0]

    def test_cells_with_a_macro_block_are_not_a_partial_read(self, tmp_path):
        _w(tmp_path, "system/blockMeshDict",
           "blocks ( hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1 1 1)\n"
           "         hex (0 1 2 3 4 5 6 7) ($n 10 1) simpleGrading (1 1 1) );\n")
        f = deck.read(tmp_path, SpecItem(name="c", file="system/blockMeshDict", blocks="cells",
                                         value=(0,)))
        assert f.value is None and "1 of 2" in f.note

    def test_missing_things_are_notes_never_exceptions(self, case):
        for item in (SpecItem(name="a", file="0/nope", patch="inlet", value=0.0),
                     SpecItem(name="b", file="0/U", patch="nope", value=0.0),
                     SpecItem(name="c", file="0/U", patch="outlet", value=0.0),
                     SpecItem(name="d", file="constant/g", key="nope", value=0.0)):
            f = deck.read(case, item)
            assert f.value is None and f.note


# ---------------------------------------------------------------- the model
class TestSpecModel:
    @pytest.mark.parametrize("item, message", [
        ({"name": "x", "file": "0/U", "patch": "inlet", "blocks": "cells", "value": 1}, "exactly one"),
        ({"name": "x", "file": "0/U", "value": 1}, "exactly one"),
        ({"name": "x", "file": "0/U", "patch": "inlet"}, "value is required"),
        ({"name": "x", "file": "../escape", "key": "nu", "value": 1}, "inside it"),
        ({"name": "x", "file": "/abs", "key": "nu", "value": 1}, "inside it"),
        ({"name": "x", "file": "a", "sha256": "ab", "value": 1}, "takes no value"),
    ])
    def test_invalid_items_are_refused(self, item, message):
        with pytest.raises(ValueError, match=message):
            _spec(item)

    def test_names_are_unique(self):
        it = {"name": "x", "file": "constant/g", "key": "value", "value": [0, -9.81, 0]}
        with pytest.raises(ValueError, match="unique"):
            _spec(it, dict(it))

    def test_load_spec_errors_say_what_is_wrong(self, tmp_path):
        p = tmp_path / "s.json"
        with pytest.raises(SpecError, match="no such"):
            load_spec(p)
        p.write_text("{not json")
        with pytest.raises(SpecError, match="not valid JSON"):
            load_spec(p)
        p.write_text(json.dumps({"schema": "courant.spec/9", "items": []}))
        with pytest.raises(SpecError, match="not valid courant.spec/1"):
            load_spec(p)

    def test_hash_is_the_file_bytes(self, tmp_path, case):
        p = tmp_path / "s.json"
        p.write_text(draft_spec(case).to_json())
        _, sha = load_spec(p)
        assert sha == spec_sha256(p)


# ---------------------------------------------------------------- conformance
class TestConformance:
    def test_a_draft_conforms_to_its_own_case(self, case):
        spec = draft_spec(case, task="t")
        block = check_conformance(spec, case, spec_sha256="h")
        assert block.nonconformant == 0 and block.unreadable == 0
        assert block.checked == len(spec.items)

    def test_the_draft_names_the_problem_and_not_the_guesses(self, case):
        names = {i.name for i in draft_spec(case).items}
        assert {"U at inlet", "k at inlet", "T at floor", "nu", "gravity", "endTime",
                "mesh scale", "turbulence model", "mesh cells", "solver"} <= names
        # A wall function's value is an initial guess, not a boundary condition anyone asked for.
        assert "k at upperWall" not in names
        choices = {i.name for i in draft_spec(case).items if i.modelling_choice}
        assert {"turbulence model", "mesh cells"} <= choices

    def _mutate(self, case: Path, rel: str, old: str, new: str) -> Path:
        p = case / rel
        p.write_text(p.read_text().replace(old, new))
        return case

    @pytest.mark.parametrize("rel, old, new, name", [
        ("0/U", "Uinlet (10 0 0)", "Uinlet (20 0 0)", "U at inlet"),
        ("0/k", "value uniform 0.375; }\n    outlet", "value uniform 9.375; }\n    outlet", "k at inlet"),
        ("constant/transportProperties", "1e-05", "3e-05", "nu"),
        ("constant/g", "(0 -9.81 0)", "(-1.7035 -9.6609 0)", "gravity"),
        ("system/controlDict", "endTime 300", "endTime 100", "endTime"),
        ("system/blockMeshDict", "scale 0.001", "scale 0.0009", "mesh scale"),
        ("system/blockMeshDict", "(30 10 1)", "(15 10 1)", "mesh cells"),
        ("constant/turbulenceProperties", "kEpsilon", "kOmegaSST", "turbulence model"),
        ("0/T", "3(300 600 300)", "3(320 600 320)", "T at floor"),
    ])
    def test_each_semantic_fault_is_caught(self, case, rel, old, new, name):
        spec = draft_spec(case)
        self._mutate(case, rel, old, new)
        block = check_conformance(spec, case, spec_sha256="h")
        wrong = [i.name for i in block.items if i.conforms is False]
        assert wrong == [name]

    def test_tolerances(self, case):
        spec = draft_spec(case)
        self._mutate(case, "constant/transportProperties", "1e-05", "1.005e-05")   # 0.5%
        self._mutate(case, "constant/g", "(0 -9.81 0)", "(-0.0856 -9.8096 0)")      # 0.5 degrees
        block = check_conformance(spec, case, spec_sha256="h")
        assert block.nonconformant == 0

    def test_dimensions_are_checked_when_both_sides_state_them(self, case):
        # A quantity SPEC-9 has no dimension set for: the case's and the spec's are compared.
        self._mutate(case, "constant/transportProperties", "transportModel Newtonian;",
                     "transportModel Newtonian;\nDT [0 2 -1 0 0 0 0] 1e-05;")
        spec = _spec({"name": "DT", "file": "constant/transportProperties", "key": "DT",
                      "value": 1e-05, "dimensions": [0, 2, -2, 0, 0]})
        block = check_conformance(spec, case, spec_sha256="h")
        assert block.items[0].conforms is False
        assert "dimensions" in block.items[0].note
        # For nu, dimensions that do not fit a kinematic viscosity make the specification
        # impossible in itself (SPEC-9): nothing is compared (VER-5.11).
        spec = _spec({"name": "nu", "file": "constant/transportProperties", "key": "nu",
                      "value": 1e-05, "dimensions": [0, 2, -2, 0, 0]})
        block = check_conformance(spec, case, spec_sha256="h")
        assert block.items[0].conforms is None
        assert "do not fit a kinematic viscosity" in block.invalid[0]

    def test_an_unreadable_item_is_not_checked_rather_than_passed(self, case):
        spec = _spec({"name": "inlet p", "file": "0/p", "patch": "outlet", "value": 0})
        block = check_conformance(spec, case, spec_sha256="h")
        assert block.unreadable == 1 and block.items[0].conforms is None

    def test_a_log_only_run_cannot_be_checked(self, case):
        block = check_conformance(draft_spec(case), None, spec_sha256="h")
        assert block.unreadable == len(block.items) and block.checked == 0


# ---------------------------------------------------------------- the verdict
@pytest.fixture
def demo_case(tmp_path):
    from courant.data import extract_demo_case

    return extract_demo_case(tmp_path)


def _write_spec(path: Path, spec: Spec) -> Path:
    path.write_text(spec.to_json())
    return path


class TestVerdict:
    def test_without_a_spec_the_bytes_are_unchanged(self, demo_case):
        from courant.verify import verify_path

        v = verify_path(demo_case)
        assert "conformance" not in json.loads(v.to_json())

    def test_a_conforming_case_keeps_its_verdict_and_says_what_was_checked(self, demo_case, tmp_path):
        from courant.verify import sentences, verify_path

        base = verify_path(demo_case)
        spec = _write_spec(tmp_path / "s.json", draft_spec(demo_case))
        v = verify_path(demo_case, spec=spec)
        assert (v.verdict, v.rule) == (base.verdict, base.rule)
        assert v.conformance.nonconformant == 0
        assert v.conformance.spec_sha256 == spec_sha256(spec)
        assert sentences.SPEC_SCOPE in v.not_assessed

    def test_the_wrong_problem_is_rejected_however_well_it_ran(self, demo_case, tmp_path):
        from courant.verify import verify_path

        base = verify_path(demo_case)
        spec = _write_spec(tmp_path / "s.json", draft_spec(demo_case))
        tp = demo_case / "constant/transportProperties"
        tp.write_text(tp.read_text().replace("1e-05", "3e-05"))
        v = verify_path(demo_case, spec=spec)
        assert v.verdict == "reject"
        if base.verdict != "reject":
            assert v.rule == "VER-5.7"
        first = v.reasons[0]
        assert first.code == "conformance.nu" and first.kind == "conformance"
        assert first.value == "3e-05"
        assert "set nu to 1e-05" in v.next_action

    def test_a_spec_changed_after_locking_voids_conformance(self, demo_case, tmp_path):
        from courant.verify import sentences, verify_path

        spec = _write_spec(tmp_path / "s.json", draft_spec(demo_case))
        locked = spec_sha256(spec)
        spec.write_text(spec.read_text().replace('"task": null', '"task": "edited"')
                        .replace("1e-05", "3e-05"))
        v = verify_path(demo_case, spec=spec, spec_locked_sha256=locked)
        assert (v.verdict, v.rule) == ("insufficient", "VER-5.8")
        assert v.reasons[0].code == "conformance.spec_changed"
        assert sentences.SPEC_VOID in v.not_assessed

    def test_an_unreadable_named_value_cannot_be_accepted(self, demo_case, tmp_path):
        from courant.verify import verify_path

        spec = _write_spec(tmp_path / "s.json", _spec(
            {"name": "inlet velocity", "file": "0/U", "patch": "inlet", "value": [10, 0, 0]}))
        v = verify_path(demo_case, spec=spec)
        assert (v.verdict, v.rule) == ("insufficient", "VER-5.9")
        assert any("inlet velocity" in s for s in v.not_assessed)

    def test_the_pack_carries_the_spec_and_re_derives(self, demo_case, tmp_path):
        from courant.verify import verify_path
        from courant.verify.pack import check_pack, write_pack

        spec = _write_spec(tmp_path / "s.json", draft_spec(demo_case))
        v = verify_path(demo_case, spec=spec)
        pack = write_pack(v, tmp_path / "pack", demo_case.name, spec_path=spec)
        ok, msg = check_pack(pack, demo_case)
        assert ok, msg
        (pack / "spec.json").write_text((pack / "spec.json").read_text().replace("1e-05", "2e-05"))
        ok, msg = check_pack(pack, demo_case)
        assert not ok

    def test_renderers_show_conformance(self, demo_case, tmp_path):
        from courant.verify import render_markdown, render_text, verify_path
        from courant.verify.html import render_html

        spec = _write_spec(tmp_path / "s.json", draft_spec(demo_case))
        tp = demo_case / "constant/transportProperties"
        tp.write_text(tp.read_text().replace("1e-05", "3e-05"))
        v = verify_path(demo_case, spec=spec)
        assert "CONFORMANCE" in render_text(v) and "✗ nu" in render_text(v)
        assert "## Conformance to the task specification" in render_markdown(v)
        assert "does not match" in render_html(v)


# ---------------------------------------------------------------- the CLI and MCP
class TestSurfaces:
    def test_cli_spec_init_hash_check_and_verify(self, demo_case, tmp_path):
        from typer.testing import CliRunner

        from courant.cli.main import app

        r = CliRunner()
        spec = tmp_path / "s.json"
        res = r.invoke(app, ["spec", "init", str(demo_case), "-o", str(spec), "--task", "demo"])
        assert res.exit_code == 0, res.output
        sha = r.invoke(app, ["spec", "hash", str(spec)]).output.strip()
        assert sha == spec_sha256(spec)
        assert r.invoke(app, ["spec", "check", str(demo_case), "--spec", str(spec)]).exit_code == 0
        tp = demo_case / "constant/transportProperties"
        tp.write_text(tp.read_text().replace("1e-05", "3e-05"))
        res = r.invoke(app, ["spec", "check", str(demo_case), "--spec", str(spec)])
        assert res.exit_code == 2 and "DOES NOT CONFORM" in res.output
        res = r.invoke(app, ["verify", str(demo_case), "--spec", str(spec), "--spec-sha256", sha])
        assert res.exit_code == 2 and "conformance.nu" in res.output

    def test_mcp_verify_run_takes_a_spec(self, demo_case, tmp_path):
        from courant.mcp import tools

        spec = _write_spec(tmp_path / "s.json", draft_spec(demo_case))
        confine = tools.Confinement([tmp_path])
        out = tools.verify_run(str(demo_case), spec=str(spec), confine=confine)
        assert out["conformance"]["nonconformant"] == 0
        with pytest.raises(tools.OutsideRoot):
            tools.verify_run(str(demo_case), spec="/etc/hosts", confine=confine)

    def test_copying_the_case_keeps_conformance(self, demo_case, tmp_path):
        spec = draft_spec(demo_case)
        copy = tmp_path / "copy"
        shutil.copytree(demo_case, copy)
        assert check_conformance(spec, copy, spec_sha256="h").nonconformant == 0


# ---------------------------------------------------------------- per-face lists (RM-18 v2)
class TestPerFaceLists:
    def test_mode_of_a_scalar_and_a_vector_list(self):
        assert deck.nonuniform_mode("nonuniform List<scalar> 5(300 600 300 300 600)") == (300.0, 3, 5)
        v = deck.nonuniform_mode("nonuniform List<vector> 3((1 0 0) (1 0 0) (2 0 0))")
        assert v == ([1.0, 0.0, 0.0], 2, 3)
        assert deck.nonuniform_mode("nonuniform List<scalar> 4(1 2 3)") is None   # count mismatch
        assert deck.nonuniform_mode("nonuniform List<tensor> 1((1 0 0 0 1 0 0 0 1))") is None

    def test_an_edit_before_setfields_is_not_masked_by_0_orig(self, case):
        # hotRoom: the case asked for 300 K, the agent wrote 320 K into 0/T, setFields then
        # turned the floor into a per-face list. 0.orig/T still says 300.
        spec = draft_spec(case)
        t = case / "0/T"
        t.write_text(t.read_text().replace("3(300 600 300)", "3(320 600 320)"))
        block = check_conformance(spec, case, spec_sha256="h")
        item = next(i for i in block.items if i.name == "T at floor")
        assert item.conforms is False
        assert item.found == "320" and "most common of 3" in item.basis

    def test_a_healthy_rewritten_list_conforms(self, case):
        block = check_conformance(draft_spec(case), case, spec_sha256="h")
        item = next(i for i in block.items if i.name == "T at floor")
        assert item.conforms is True and "most common" in item.basis

    def test_the_solver_reads_0_so_a_stale_0_orig_does_not_reject_a_correct_answer(self, case):
        # SilentBench feasibility: the task asked 304 K, the answer wrote 304 K into 0/T and
        # left 0.orig/T at 300; setFields then rewrote 0/T. The run solved the asked problem.
        spec = draft_spec(case)
        items = [dict(i, value=304.0) if i["name"] == "T at floor" else i
                 for i in json.loads(spec.to_json())["items"]]
        spec304 = Spec.model_validate({"schema": "courant.spec/1", "items": items})
        t = case / "0/T"
        t.write_text(t.read_text().replace("3(300 600 300)", "3(304 600 304)"))
        item = next(i for i in check_conformance(spec304, case, spec_sha256="h").items
                    if i.name == "T at floor")
        assert item.conforms is True

    def test_an_edit_to_0_orig_alone_does_not_change_what_the_solver_read(self, case):
        spec = draft_spec(case)
        o = case / "0.orig/T"
        o.write_text(o.read_text().replace("uniform 300; }", "uniform 320; }"))
        item = next(i for i in check_conformance(spec, case, spec_sha256="h").items
                    if i.name == "T at floor")
        assert item.conforms is True and item.file == "0/T"


# ---------------------------------------------------------------- the manifest (VER-14)
class TestManifest:
    """What conformance read is in the verdict's manifest, and only when a spec made it read."""

    def test_conformance_reads_are_listed_and_only_with_a_spec(self, case, tmp_path):
        import gzip

        from courant.verify import verify_path

        # The fixture's only include is `#includeEtc`, which names OpenFOAM's etc/ and is never
        # followed. Move `Uinlet` into a case-relative include, the way the pisoFoam tutorials
        # write it, so that resolving `$Uinlet` reads a second file.
        u = case / "0/U"
        u.write_text(u.read_text().replace("Uinlet (10 0 0);",
                                           '#include "include/initialConditions"'))
        _w(case, "0/include/initialConditions", "Uinlet (10 0 0);\n")
        # With no 0/T, conformance falls back to 0.orig/T (the fixture holds both).
        (case / "0/T").unlink()
        corpus = Path(__file__).resolve().parents[2] / "certification" / "corpus"
        with gzip.open(corpus / "simpleFoam_pitzDaily__healthy.log.gz", "rt") as fh:
            (case / "log.simpleFoam").write_text(fh.read())
        spec = _write_spec(tmp_path / "s.json", draft_spec(case))

        with_spec = {f.path for f in verify_path(case, spec=spec).run.inputs}
        without = {f.path for f in verify_path(case).run.inputs}

        conformance = {"0.orig/T", "0/include/initialConditions", "system/blockMeshDict"}
        assert conformance <= with_spec, (
            f"read by conformance but not listed: {sorted(conformance - with_spec)}")
        only_with_spec = conformance | {"constant/g"}               # gravity is a spec item here
        assert not only_with_spec & without, (
            f"listed with no spec to read them: {sorted(only_with_spec & without)}")
        parsed = {"0/U", "0/k"}                                     # the case-config reader's 0/
        assert parsed <= with_spec and parsed <= without, (
            f"0/ files the case-config reader parsed but not listed: "
            f"{sorted(parsed - (with_spec & without))}")
