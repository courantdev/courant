"""T-FOLLOWUP-MINORS: the 0.6.0 follow-up review's eight minors (22 Oct 2026) and the seven left at
the end of T-ENERGY-BALANCE's review (3 Nov 2026). One test (or more) a minor, each seen failing
on the code before its fix; where a minor is a rename with no behaviour, the node's work order
says so.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import warnings
from pathlib import Path

import pytest

from courant.case import wordre
from courant.case.config import CaseConfig

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"
CORPUS = ROOT / "certification" / "corpus"
HEALTHY_058 = (ROOT / "tests" / "fixtures" / "residual_control"
               / "c10k.simpleFoam_pitzDaily.058.healthy.log.gz")


def _case(tmp_path: Path, name: str = "cavity") -> Path:
    case = tmp_path / name
    shutil.copytree(CAVITY, case)
    for gz in case.glob("log.*.gz"):
        with gzip.open(gz, "rb") as src, open(case / gz.name.removesuffix(".gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    return case


# ---------------------------------------------------------------- (1) a list behind plain characters
NAMES = [f"n{i:03d}" for i in range(200)]


@pytest.mark.parametrize("pattern", [
    "wing_(" + "|".join(NAMES) + ")",                      # the review's shape
    "wing_(?:" + "|".join(NAMES) + ")",                    # a non-capturing group
    "(wing_(" + "|".join(NAMES) + "))",                    # inside an enclosing group
    "w(ing_(" + "|".join(NAMES) + "))",                    # a prefix outside and inside
], ids=["prefix", "non-capturing", "enclosed", "split prefix"])
def test_a_list_of_names_behind_plain_characters_is_a_set(pattern):
    """`"wing_(n000|...|n199)"`: plain characters, then a group whose body is an alternation of
    plain names. 200 names of four characters are some 800 states, past `MAX_STATES`: matched by
    the automaton it was not evaluated, and a specification item on such a patch was
    insufficient where 0.5.0's `re` judged it. Matched as a set, with `re`'s answer."""
    for name in ("wing_n000", "wing_n150", "wing_n199", "wing_n200", "wing_", "wing_n15",
                 "xing_n150", "wing_n1500", ""):
        assert wordre.fullmatch(pattern, name) is (re.fullmatch(pattern, name) is not None), name


#: The fuzz below: seeded, its denominators counted by the test itself.
GROUP_SEED, GROUP_PATTERNS = 2611, 500


def test_lists_behind_plain_characters_say_what_re_says_on_a_seeded_fuzz():
    """500 seeded patterns of the shape: plain characters, then up to three nested groups (some
    non-capturing, some written behind more characters), the last an alternation of plain names
    sharing starts; 20 names each, half drawn from the pattern's own. Every answer is
    `re.fullmatch`'s, and every one of them is a set."""
    import random

    r = random.Random(GROUP_SEED)
    alphabet = "ab_1W"
    differ, sets, pairs = [], 0, 0
    for _ in range(GROUP_PATTERNS):
        names = sorted({"".join(r.choice(alphabet) for _ in range(r.randint(1, 4)))
                        for _ in range(r.randint(2, 30))})
        pattern = "|".join(names)
        meant = list(names)
        for _ in range(r.randint(1, 3)):
            head = "".join(r.choice(alphabet) for _ in range(r.randint(0, 3)))
            pattern = head + ("(?:" if r.random() < 0.3 else "(") + pattern + ")"
            meant = [head + n for n in meant]
        pattern = "".join(r.choice(alphabet) for _ in range(r.randint(1, 4))) + pattern
        prefix = pattern[:pattern.index("(")]
        meant = [prefix + n for n in meant]
        sets += isinstance(wordre._compile(pattern), wordre._NameSet)
        candidates = [r.choice(meant) for _ in range(10)] + [
            "".join(r.choice(alphabet) for _ in range(r.randint(0, 12))) for _ in range(10)]
        for name in candidates:
            pairs += 1
            if wordre.fullmatch(pattern, name) is not (re.fullmatch(pattern, name) is not None):
                differ.append((pattern, name))
    assert differ == []
    assert (sets, pairs) == (GROUP_PATTERNS, 20 * GROUP_PATTERNS)


def test_what_is_not_a_list_of_names_keeps_the_automaton():
    """The set is taken only where the pattern is one: a group followed by more pattern, a class
    inside, or `(?i)`, keeps the automaton's answer (here, past its bound, none)."""
    for pattern in ("wing_(" + "|".join(NAMES) + ")Wall", "wing_(" + "|".join(NAMES) + "|x.)",
                    "(?i)wing_(" + "|".join(NAMES) + ")"):
        assert not isinstance(wordre._compile(pattern), wordre._NameSet), pattern


def test_a_patch_grouped_behind_plain_characters_is_checked_by_default(tmp_path):
    """End to end, by default: the cavity's movingWall grouped under `"moving(Wall|n000|...)"`
    (200 names). The spec item was insufficient (VER-5.9); it conforms."""
    from courant.spec import deck
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    item = SpecItem(name="lid", file="0/U", patch="movingWall", value=[1.0, 0.0, 0.0])
    written = deck.read(case, item).value
    assert written is not None
    u = case / "0" / "U"
    text = u.read_text()
    assert "    movingWall\n" in text
    u.write_text(text.replace("    movingWall\n",
                              '    "moving(Wall|' + "|".join(NAMES) + ')"\n', 1))
    found = deck.read(case, item)
    assert found.value == written, found.note


# ---------------------------------------------------------------- (2), (3) residualControl's wording, one bar
def _events_058():
    from courant.watcher.parser import LogParser

    with gzip.open(HEALTHY_058, "rt") as fh:
        return list(LogParser(run_id="058").parse_text(fh.read()))


def _config_058(control: dict) -> CaseConfig:
    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"]["residual_control"] = {"SIMPLE": control}
    return CaseConfig.model_validate(meta["config"])


def test_a_tolerance_a_keyword_before_the_match_gives_is_not_called_the_cases():
    """(2) `"(?=U)U" 2e-3;` before `U 5e-3;`: U's entry certainly matches, and the strictest
    tolerance it may have, 2e-3, is the keyword's. The bar is 2e-3, and it is said to be one a
    keyword the reader does not evaluate may set, never "its residualControl 0.002"."""
    from courant.verification.convergence import judge_residuals

    config = _config_058({"p": 1e-2, '"(k|epsilon|omega|f|v2)"': 1e-2, '"(?=U)U"': 2e-3,
                          "U": 5e-3})
    judged = judge_residuals(_events_058(), config, "next")
    assert judged.field in ("Ux", "Uy") and judged.bar == 2e-3, (judged.field, judged.bar)
    assert not judged.field_declared and judged.field_unsure
    assert judged.bar_words() == ("0.002, which a residualControl keyword the reader does not "
                                  "evaluate may set")
    # the entry's own value, when it is the strictest, is the case's
    own = judge_residuals(_events_058(), _config_058(
        {"p": 1e-2, '"(k|epsilon|omega|f|v2)"': 1e-2, '"(?=U)U"': 5e-3, "U": 2e-3}), "next")
    assert own.field in ("Ux", "Uy") and own.field_declared and not own.field_unsure
    assert own.bar_words() == "its residualControl 0.002"


def _lts(control: dict) -> CaseConfig:
    return CaseConfig(solver="foamRun", ddt_scheme="localEuler",
                      residual_control={"PIMPLE": control})


@pytest.mark.parametrize(("control", "bar"), [
    ({"p": 1e-2}, 1e-2), ({"p": 1.0}, 1e-2), ({"p": 1e-8}, 1e-3), ({}, 1e-3),
    ({'"(?=p)p"': 5e-3}, 1e-3), ({'"(?=p)p"': 5e-3, "p": 2e-3}, 2e-3),
])
def test_one_definition_of_a_fields_bars(control, bar):
    """(3) `residual_bar` is the strictest bar a reading of the field holds it to, from the one
    definition the judgement reads (`convergence._bars`); `declared_tolerance`'s uncertain
    branch (no looser than the flat bar) no longer gives it. A refactor: the values are the ones
    the two gave together (pinned here, and by `test_residual_floor`'s
    `test_residual_bar_is_the_bar_the_judgement_holds_a_certain_tolerance_to`)."""
    from courant.verification import convergence

    assert convergence.residual_bar(_lts(control), "p", "next") == bar
    found = convergence._candidates(_lts(control), "p")
    if found is not None:
        assert bar == min(b for b, _, _ in convergence._bars(found, "next"))


def test_a_steady_fields_bar_is_its_tolerance():
    from courant.verification.convergence import RESIDUAL_TARGET, residual_bar

    steady = CaseConfig(residual_control={"SIMPLE": {"p": 1e-2, '"(?=U)U"': 5e-3}},
                        solver="simpleFoam", ddt_scheme="steadyState")
    assert residual_bar(steady, "p", "next") == 1e-2
    assert residual_bar(steady, "Ux", "next") == RESIDUAL_TARGET       # the keyword or none


# ---------------------------------------------------------------- (5) one name lookup
def test_the_case_readers_sibling_lookup_is_csearch(monkeypatch):
    """(5) `_foam_reader._names` was a third lookup of a name among solver keys (`first_match`,
    in declaration order). It is `wordre.csearch`, v2406's `dictionary::csearch`."""
    from courant.case import _foam_reader

    calls = []
    real = wordre.csearch
    monkeypatch.setattr(wordre, "csearch", lambda *a, **k: calls.append(a) or real(*a, **k))
    assert _foam_reader._names(['"(p|U)"', "k"], "p") is True
    assert _foam_reader._names(["p", '"(?=q)q"'], "q") is False         # not for certain
    assert _foam_reader._names(['"p"'], "p") is True                    # by its text
    assert len(calls) == 3


# ---------------------------------------------------------------- (6) a key written quoted and not
@pytest.mark.parametrize(("equations", "want"), [
    ('"U" 0.05; U 0.9;', 0.9), ('U 0.9; "U" 0.05;', 0.05),
    ('"U" 0.05; U 0.9; "U" 0.07;', 0.07), ('U 0.9; "U" 0.05; U 0.3;', 0.3),
])
def test_a_name_written_quoted_and_not_stands_as_last_written(tmp_path, equations, want):
    """(6) `"U"` and `U` are one entry in v2406's dictionary, and the value standing is the last
    written. `equation_relaxation` takes the later of the two keys in the configuration's map,
    which the review feared was the order of each key's first writing. It is the order of each
    one's last: foamlib keeps a key written again at its last place, so the map's order is the
    file's, rewritten keys too. No fix: pinned here, end to end from the file (the strictest of
    the two, the fix proposed, would read 0.05 for the first file, which v2406 relaxes by 0.9)."""
    from courant.verification.convergence import equation_relaxation

    (tmp_path / "system").mkdir()
    (tmp_path / "system" / "fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
        f"relaxationFactors\n{{\n    equations\n    {{\n        {equations}\n    }}\n}}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")             # foamlib on a key written again
        config = CaseConfig.from_foam(str(tmp_path))
    assert equation_relaxation(config, "Ux") == want


# ---------------------------------------------------------------- (7) a keyword from an #include
def test_a_keyword_from_an_include_is_noted_by_the_file_that_holds_it(tmp_path):
    """(7) Under `--checks next` the spec reader reads `0/U` with its includes inlined
    (`spec.directives`). A keyword not evaluated that the included file holds was noted as one
    in `0/U`. The note names the file that holds it."""
    from courant.spec import deck
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    u = case / "0" / "U"
    text = u.read_text()
    start = text.index("    movingWall\n")
    end = text.index("}", text.index("{", start)) + 1
    (case / "0" / "lid").write_text(
        '"(movingWall|lid)"\n{\n    type fixedValue;\n    value uniform (1 0 0);\n}\n'
        '"(a)\\1"\n{\n    type zeroGradient;\n}\n')
    u.write_text(text[:start] + '    #include "lid"\n' + text[end:])
    item = SpecItem(name="lid", file="0/U", patch="movingWall", value=[1.0, 0.0, 0.0])
    found = deck.read(case, item, macros=True)
    assert found.value is None
    assert "0/lid" in found.note and "0/U" in found.note, found.note
    assert "(a)" not in found.note                      # never the keyword
    # one in the file itself is noted by that file, as before
    (case / "0" / "lid").write_text('"(movingWall|lid)"\n{\n    type fixedValue;\n'
                                    '    value uniform (1 0 0);\n}\n')
    u.write_text(text[:start] + '    #include "lid"\n    "(a)\\1"\n    {\n        type x;\n'
                 '    }\n' + text[end:])
    found = deck.read(case, item, macros=True)
    assert found.value is None and found.note == deck.not_evaluated("0/U")


def test_a_keyword_from_an_absolute_include_under_a_relative_case_names_the_case_file(
        tmp_path, monkeypatch):
    """The review of T-FOLLOWUP-MINORS (M3): the case given by a relative path, its `0/U`
    including `0/lid` by an absolute path. `_holder` took the include's path relative to the
    case as given, which raised, and the note quoted the absolute path (the user's machine, in
    the verdict and the bundle). Both are resolved before they are compared: the note names
    `0/lid`, never a path outside the case's own words."""
    from courant.spec import deck
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    u = case / "0" / "U"
    text = u.read_text()
    start = text.index("    movingWall\n")
    end = text.index("}", text.index("{", start)) + 1
    (case / "0" / "lid").write_text(
        '"(movingWall|lid)"\n{\n    type fixedValue;\n    value uniform (1 0 0);\n}\n'
        '"(a)\\1"\n{\n    type zeroGradient;\n}\n')
    u.write_text(text[:start] + f'    #include "{(case / "0" / "lid").resolve()}"\n'
                 + text[end:])
    monkeypatch.chdir(tmp_path)
    item = SpecItem(name="lid", file="0/U", patch="movingWall", value=[1.0, 0.0, 0.0])
    found = deck.read(Path(case.name), item, macros=True)
    assert found.value is None
    assert found.note == deck.not_evaluated("0/lid", "0/U"), found.note
    assert str(tmp_path) not in found.note and str(tmp_path.resolve()) not in found.note


# ---------------------------------------------------------------- (8) a held tolerance on LTS
def test_on_lts_a_held_tolerance_never_accepts_what_no_tolerance_rejects():
    """(8) The final review of 519ab28: on a local-time-stepping foamRun run, p at 5e-4 on its
    first solve and 2e-2 on its last, under `p 1e-2`, was read at its first solve alone and
    accepted (A), where with no residualControl the flat bar reads the last solve and rejects
    it, MAJOR (past RESIDUAL_LOOSE, VER-5.3, C). The last solve is read where the flat bar reads
    it as well, for the band past RESIDUAL_LOOSE, which is MAJOR whatever the case says:
    rejected as with no tolerance. Inside it the tolerance decides, and is the case's."""
    from courant.verification.convergence import RESIDUAL_LOOSE, judge_residuals
    from tests.unit.test_residual_floor import _grade, _two_solves, _verify_lts

    events = _two_solves(5e-4, 2e-2)
    held, none = (judge_residuals(events, _lts(c), "next") for c in ({"p": 1e-2}, {}))
    assert held.worst == none.worst == pytest.approx(2e-2) and held.worst > RESIDUAL_LOOSE
    assert _grade(events, _lts({"p": 1e-2})) == _grade(events, _lts({}))
    v, without = _verify_lts(5e-4, 2e-2, {"p": 1e-2}), _verify_lts(5e-4, 2e-2, {})
    assert (v.verdict, v.rule, v.grade) == (without.verdict, without.rule, without.grade) == (
        "reject", "VER-5.3", "C"), [(r.code, r.summary) for r in v.reasons]
    # inside RESIDUAL_LOOSE the last solve is the tolerance's to loosen (S, 26 Oct 2026), and
    # the bar and its words are the case's
    for last in (4e-4, 5e-3):
        within = judge_residuals(_two_solves(5e-3, last), _lts({"p": 1e-2}), "next")
        assert within.over == () and within.field_declared and within.bar == 1e-2
        assert within.worst == pytest.approx(5e-3)


def test_on_a_steady_run_a_held_tolerance_is_read_once():
    """On a steady run under `next` both readings are the first solve (VER-16a), and the flat
    bar is not added: `p 1e-2` still loosens p's bar."""
    from courant.verification.convergence import judge_residuals
    from tests.unit.test_residual_floor import _steady_nonorth, _two_solves

    judged = judge_residuals(_two_solves(5e-3, 2e-2), _steady_nonorth({"p": 1e-2}), "next")
    assert judged.over == () and judged.field_declared and judged.bar == 1e-2


# ================================================================ T-ENERGY-BALANCE's minors
def _cht(tmp_path: Path) -> Path:
    from tests.unit.test_energy_balance import _solve

    return _solve(tmp_path, "cht_plate-converged")


def _measured(case: Path) -> set:
    from tests.unit.test_energy_balance import _key, _read_heat

    return {_key(k) for k in _read_heat(case).measures}


@pytest.mark.parametrize("rel", ["constant/fvOptions", "system/fvOptions",
                                 "constant/MRFProperties"])
def test_a_multi_region_case_is_not_refused_for_a_source_its_solver_does_not_read(tmp_path, rel):
    """(E1) A multi-region solver reads each region's sources (`constant/<region>/fvOptions`,
    `system/<region>/fvOptions`, ...) and never the case's own: `chtMultiRegionSimpleFoam`
    builds its fvOptions, MRF and radiation per region. A case-level file was taken as a source
    and the whole heat balance was not judged."""
    from tests.unit.test_energy_balance import JUDGED

    case = _cht(tmp_path)
    (case / rel).write_text("FoamFile { class dictionary; object x; }\nheater { type x; }\n")
    assert _measured(case) == JUDGED["cht_plate"]
    # a region's own file is where the solver looks: still refused
    (case / rel).unlink()
    top, name = rel.split("/", 1)
    (case / top / "plate" / name).write_text("FoamFile { class dictionary; object x; }\n")
    assert _measured(case) == set()


def test_a_multi_region_case_is_not_refused_for_the_cases_radiation_file(tmp_path):
    from tests.unit.test_energy_balance import JUDGED

    case = _cht(tmp_path)
    (case / "constant" / "radiationProperties").write_text(
        "FoamFile { class dictionary; object radiationProperties; }\nradiationModel P1;\n")
    assert _measured(case) == JUDGED["cht_plate"]


def test_every_behaviour_the_code_asks_for_is_named_once_in_next():
    """(E2) A `--checks next` name is written once: `pipeline.ENERGY_WALLHEATFLUX` repeated
    `checks.NEXT`'s literal, and a name renamed in one place and not the other is not in NEXT,
    which `enabled` takes for a behaviour promoted and turns on by default. Every first argument
    of `enabled(` and `checks.enabled(` in the package is a literal NEXT name, or a constant
    `courant.verify.checks` defines, whose value NEXT is keyed by (but `checks.py`'s own calls
    and the finding id `only_enabled` passes). From 0.7.0 a name GATE-13 promoted has left NEXT
    and is named in `checks.PROMOTED`, still asked where its behaviour is: either holds it."""
    from courant.verify import checks, pipeline

    named = {**checks.NEXT, **checks.PROMOTED}
    asked = []
    for path in (ROOT / "courant").rglob("*.py"):
        text = path.read_text()
        for m in re.finditer(r"(?<![\w.])(?:checks\.)?enabled\(\s*([^,)\s]+)", text):
            asked.append((path.relative_to(ROOT).as_posix(), m.group(1)))
    names = [(where, arg) for where, arg in asked
             if not (where == "courant/verify/checks.py" or arg in ('""', "f.id"))]
    assert names, "the scan found no call"
    for where, arg in names:
        if arg.startswith('"'):
            assert arg.strip('"') in named, (where, arg)
        else:
            assert getattr(checks, arg.removeprefix("checks."), None) in named, (where, arg)
    assert pipeline.ENERGY_WALLHEATFLUX is checks.ENERGY_WALLHEATFLUX


def test_the_closed_patch_types_are_one_list():
    """(E3) `_heat_reader._CLOSED_TYPES` repeated `reynolds._CONSTRAINT` (empty, wedge, symmetry,
    symmetryPlane): one list."""
    from courant.case import reynolds
    from courant.verification import _heat_reader

    assert _heat_reader._CLOSED_TYPES is reynolds._CONSTRAINT


def test_a_time_directory_and_a_compressed_file_are_one_rule_each(tmp_path):
    """(E4) `_heat_reader._is_time` restated `mesh.time_dirs`'s rule (a finite number) and its
    boundary's `.gz` fallback restated the field reader's (`_fluid_reader._field_file`): one
    rule each, `mesh.is_time` and `reads.plain_or_gz`."""
    from courant.case import mesh, reads
    from courant.verification import _fluid_reader, _heat_reader

    assert not hasattr(_heat_reader, "_is_time")
    for name, want in (("0", True), ("1e-3", True), ("inf", False), ("nan", False),
                       ("x", False), ("-1", True)):
        assert mesh.is_time(name) is want, name
    (tmp_path / "boundary.gz").write_bytes(gzip.compress(b"x"))
    assert reads.plain_or_gz(str(tmp_path / "boundary")) == str(tmp_path / "boundary.gz")
    (tmp_path / "boundary").write_text("x")
    assert reads.plain_or_gz(str(tmp_path / "boundary")) == str(tmp_path / "boundary")
    assert _fluid_reader._field_file(str(tmp_path), "", "boundary") == \
        reads.plain_or_gz(str(tmp_path / "boundary"))


def test_a_compressed_polymesh_file_is_read_to_the_mesh_readers_bound(tmp_path, monkeypatch):
    """(E6) `polymesh._body` decompressed a `.gz` with no bound: a small file can decompress to
    gigabytes. It reads at most the bound the mesh reader holds a mesh's files to on disk
    (`mesh.POLYMESH_MAX_BYTES`), and a file past it is refused, never parsed."""
    from courant.case import mesh, polymesh

    text = ("FoamFile { version 2.0; format ascii; class polyBoundaryMesh; object boundary; }\n"
            "1\n(\n    wall1\n    {\n        type wall;\n        nFaces 1;\n        startFace 0;"
            "\n    }\n)\n")
    path = tmp_path / "boundary.gz"
    path.write_bytes(gzip.compress(text.encode()))
    assert [p.name for p in polymesh.read_boundary(path)] == ["wall1"]
    monkeypatch.setattr(mesh, "POLYMESH_MAX_BYTES", len(text) - 1)
    with pytest.raises(ValueError, match="bound"):
        polymesh.read_boundary(path)


def test_openfoams_heat_files_are_its_own_words_in_a_support_bundle():
    """(E7) The support bundle's pseudonymiser maps every path word that is not OpenFOAM's own:
    `regionProperties`, `radiationProperties` and the table's `wallHeatFlux.dat` were mapped as
    the customer's, where a region's name and the function object's (`wallHeatFlux1`) are."""
    from courant.verify import support

    names = support._names({"run": {"inputs": [
        {"path": "constant/regionProperties"}, {"path": "constant/plate/radiationProperties"},
        {"path": "postProcessing/plate/wallHeatFlux1/0/wallHeatFlux.dat"}]}})
    for word in ("regionProperties", "radiationProperties", "wallHeatFlux.dat"):
        assert word not in names, (word, names)
    assert {"plate", "wallHeatFlux1"} <= set(names)
