"""hp/audit-robustness's tests (85c0c6c, 25f2de8), ported to the VER-3b on `main` (T-VER3B-ONE).

Two branches built VER-3b. hp/audit-outputs's is on `main`: the case reader reads a dictionary as
the solver reads it under `--checks next` (`case.directives`, `courant.spec.deck.expand`), and by
default names each value it did not read (`CaseConfig.unread_values`, `case.unread_value`).
hp/audit-robustness's tests are kept here, in `main`'s vocabulary (`directives=` for its
`macros=`, `unread_values` for its `unresolved`, `case.unread_value`'s sentence for its
`case.unresolved_value`'s), with what `main` lacked built for them:

- the spec reader reads a dictionary as the case reader does under `--checks next`
  (`spec.directives`, 0.7.0: `deck.read(..., macros=True)` through `deck.expand`, so a `$name` is
  looked up from the entry outwards and `#includeIfPresent` is followed); by default it keeps
  0.6.0's lookup, and an item whose `$name` the dictionary around the entry also defines says so
  in its basis, never quoting the macro (main's rule since the review of 22 Oct 2026: what the
  case wrote stays local);
- a residualControl or relaxation keyword the matcher did not evaluate is named under "not
  assessed" under `--checks next`, by field and dictionary, never by its text
  (`convergence.keywords_not_evaluated`, `sentences.keywords_not_evaluated`);
- the field checks take nu from the case reader (`fields.case_nu`), one reader of it.

Where the branch's assertion and `main`'s later decision differ, `main`'s decision holds and the
test says so: the default spec reader does not follow `#includeIfPresent` (a default verdict
would move, GATE-13), and no sentence quotes a macro or a keyword.

The fixture is the committed `cavity_extreme_aspect_ratio` case (icoFoam, v2406).
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from courant.case._foam_reader import read_case_config
from courant.diagnosis.checkers import UNREAD_VALUES_TITLE
from courant.verify import verify_path
from courant.verify.preflight import preflight_path

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"
CORPUS = ROOT / "certification" / "corpus"
PITZ = ROOT / "certification" / "monitored" / "simpleFoam_pitzDaily__shipped"
HEALTHY_058 = (ROOT / "tests" / "fixtures" / "residual_control"
               / "c10k.simpleFoam_pitzDaily.058.healthy.log.gz")

NOT_READ = UNREAD_VALUES_TITLE + ": "
NU_LINE = "nu              [0 2 -1 0 0 0 0] 0.00848;"
U_TOLERANCE = "tolerance       1e-05;"
U_RELTOL = "relTol          0;\n    }\n}"


def _case(tmp_path: Path, name: str = "cavity") -> Path:
    """The committed cavity, its log unzipped."""
    case = tmp_path / name
    shutil.copytree(CAVITY, case)
    for gz in case.glob("log.*.gz"):
        with gzip.open(gz, "rb") as src, open(case / gz.name.removesuffix(".gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    return case


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, (path, old)
    path.write_text(text.replace(old, new, 1))


def _notes(sentences) -> list[str]:
    return [s for s in sentences if s.startswith(NOT_READ)]


# ---------------------------------------------------------------- the faults, and their ways
#: Each injected setup fault: the file, and the finding it raises.
FAULTS = {
    "nu": ("constant/transportProperties", "physics.nu"),
    "tolerance": ("system/fvSolution", "solver.loose_tolerance"),
    "relTol": ("system/fvSolution", "solver.loose_tolerance"),
}


def _direct(case: Path, fault: str) -> None:
    if fault == "nu":
        _edit(case / "constant/transportProperties", NU_LINE, "nu [0 2 -1 0 0 0 0] 0.5;")
    elif fault == "tolerance":
        _edit(case / "system/fvSolution", U_TOLERANCE, "tolerance       0.5;")
    else:
        _edit(case / "system/fvSolution", U_RELTOL, "relTol          0.9;\n    }\n}")


def _macro(case: Path, fault: str) -> None:
    """The faulty value as a macro defined in the same file."""
    if fault == "nu":
        _edit(case / "constant/transportProperties", NU_LINE,
              "nuVal 0.5;\nnu [0 2 -1 0 0 0 0] $nuVal;")
    elif fault == "tolerance":
        _edit(case / "system/fvSolution", U_TOLERANCE, "tolerance       $tolU;")
        _edit(case / "system/fvSolution", "\nsolvers\n", "\ntolU 0.5;\n\nsolvers\n")
    else:
        _edit(case / "system/fvSolution", U_RELTOL, "relTol          $relU;\n    }\n}")
        _edit(case / "system/fvSolution", "\nsolvers\n", "\nrelU 0.9;\n\nsolvers\n")


def _include_macro(case: Path, fault: str) -> None:
    """The audit's case: the macro is defined in a file the dictionary `#include`s."""
    rel = FAULTS[fault][0]
    name = {"nu": "nuVal", "tolerance": "tolU", "relTol": "relU"}[fault]
    if fault == "nu":
        _edit(case / rel, NU_LINE, f'#include "values"\nnu [0 2 -1 0 0 0 0] ${name};')
    else:
        _edit(case / rel, "\nsolvers\n", '\n#include "values"\n\nsolvers\n')
        if fault == "tolerance":
            _edit(case / rel, U_TOLERANCE, f"tolerance       ${name};")
        else:
            _edit(case / rel, U_RELTOL, f"relTol          ${name};\n    }}\n}}")
    (case / rel).parent.joinpath("values").write_text(
        f"{name} {'0.9' if fault == 'relTol' else '0.5'};\n")


def _include_value(case: Path, fault: str) -> None:
    """The setting itself comes from the `#include`d file: the dictionary does not write it."""
    if fault == "nu":
        _edit(case / "constant/transportProperties", NU_LINE, '#include "physics"')
        (case / "constant/physics").write_text("nu [0 2 -1 0 0 0 0] 0.5;\n")
        return
    fv = case / "system/fvSolution"
    text = fv.read_text()
    start, end = text.index("\nsolvers\n"), text.index("\nPISO\n")
    solvers = text[start:end]
    _direct(case, fault)
    faulty = fv.read_text()
    block = faulty[faulty.index("\nsolvers\n"):faulty.index("\nPISO\n")]
    fv.write_text(text[:start] + '\n#include "solverSettings"\n' + text[end:])
    (case / "system/solverSettings").write_text(block)
    assert solvers != block


def _merged(case: Path, fault: str) -> None:
    """The faulty value in a dictionary merged in with `$name;` from the top level."""
    if fault == "nu":
        _edit(case / "constant/transportProperties", NU_LINE,
              "physics { nu [0 2 -1 0 0 0 0] 0.5; }\n$physics;")
        return
    fv = case / "system/fvSolution"
    value = "tolerance 0.5; relTol 0;" if fault == "tolerance" else "tolerance 1e-05; relTol 0.9;"
    _edit(fv, "\nsolvers\n", f"\nUbase {{ solver smoothSolver; smoother symGaussSeidel; {value} }}"
                             "\n\nsolvers\n")
    text = fv.read_text()
    i = text.index("    U\n    {")
    j = text.index("}", i) + 1
    fv.write_text(text[:i] + "    U\n    {\n        $Ubase;\n    }" + text[j:])


WAYS = {"macro": _macro, "include_macro": _include_macro, "include_value": _include_value,
        "merged": _merged}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_the_control_each_fault_written_directly_is_found_in_both_modes(tmp_path, fault):
    case = _case(tmp_path)
    _direct(case, fault)
    code = FAULTS[fault][1]
    for mode in ("default", "next"):
        verdict = verify_path(case, checks=mode)
        preflight = preflight_path(case, checks=mode)
        assert (verdict.verdict, verdict.rule) == ("reject", "VER-5.4"), mode
        assert code in {r.code for r in verdict.reasons}
        assert preflight.verdict == "SETUP_FAULT" and code in {r.code for r in preflight.reasons}
        assert _notes(verdict.not_assessed) == [] and _notes(preflight.not_checked) == []


@pytest.mark.usefixtures("checks_as_060")
@pytest.mark.parametrize("way", sorted(WAYS))
@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_by_default_a_fault_written_through_a_macro_or_include_is_named_not_silent(
        tmp_path, fault, way):
    """GATE-13: the default verdict is v0.5.0's (accept, A), and the setting is named, with the
    flag that reads it."""
    case = _case(tmp_path)
    WAYS[way](case, fault)
    rel = FAULTS[fault][0]
    verdict = verify_path(case)
    preflight = preflight_path(case)
    assert (verdict.verdict, verdict.rule, verdict.grade) == ("accept", "VER-5.6", "A")
    assert preflight.verdict == "NO_SETUP_FAULT_FOUND"
    [note] = _notes(verdict.not_assessed)
    assert f"{rel}: " in note and "--checks next" in note
    assert _notes(preflight.not_checked) == [note]


@pytest.mark.parametrize("way", sorted(WAYS))
@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_under_next_a_fault_written_through_a_macro_or_include_is_found(tmp_path, fault, way):
    """The reader follows the macro, the merged dictionary or the include inside the case, so
    the fault is found exactly as when it is written directly, by verify and by preflight."""
    case = _case(tmp_path)
    WAYS[way](case, fault)
    code = FAULTS[fault][1]
    verdict = verify_path(case, checks="next")
    preflight = preflight_path(case, checks="next")
    assert (verdict.verdict, verdict.rule) == ("reject", "VER-5.4")
    assert code in {r.code for r in verdict.reasons}
    assert preflight.verdict == "SETUP_FAULT" and code in {r.code for r in preflight.reasons}
    assert _notes(verdict.not_assessed) == [] and _notes(preflight.not_checked) == []


@pytest.mark.usefixtures("checks_as_060")
def test_under_next_an_included_file_is_listed_among_the_inputs(tmp_path):
    case = _case(tmp_path)
    _include_macro(case, "tolerance")
    listed = {f.path for f in verify_path(case, checks="next").run.inputs}
    assert "system/values" in listed
    assert "system/values" not in {f.path for f in verify_path(case).run.inputs}


# ---------------------------------------------------------------- what is never followed
UNFOLLOWED = {
    "calc": 'tolerance       #calc "0.5";',
    "scoped": "tolerance       $:solvers.p.tolerance;",
    "undefined": "tolerance       $nowhere;",
}


@pytest.mark.parametrize("which", sorted(UNFOLLOWED))
def test_what_the_reader_cannot_follow_is_named_in_both_modes(tmp_path, which):
    """The branch's sentence quoted `solvers/U/tolerance = <what was written>`; `main`'s names
    the file and the key path the same way."""
    case = _case(tmp_path)
    _edit(case / "system/fvSolution", U_TOLERANCE, UNFOLLOWED[which])
    for mode in ("default", "next"):
        verdict = verify_path(case, checks=mode)
        assert verdict.verdict == "accept", mode
        [note] = _notes(verdict.not_assessed)
        assert "system/fvSolution: solvers/U/tolerance " in note, note
        # The branch said "--checks next follows a $macro ..." after every default note. `main`
        # says the flag reads a value only for a kind it can read: not `#calc`. (By default a
        # scoped or undefined `$name` is still said to be expanded by the flag, which names it
        # unread too: recorded in T-VER3B-ONE's work order, open.)
        if mode == "next" or which == "calc":
            assert "--checks next" not in note, (mode, note)
        config = read_case_config(str(case), directives=mode == "next")
        assert config.linear_solvers["U"].tolerance is None


def test_an_include_outside_the_case_is_not_followed_and_is_named(tmp_path):
    case = _case(tmp_path)
    (tmp_path / "outside").write_text("nu [0 2 -1 0 0 0 0] 0.5;\n")
    _edit(case / "constant/transportProperties", NU_LINE, '#include "../../outside"')
    config = read_case_config(str(case), directives=True)
    assert "nu" not in config.transport
    assert list(config.unread_values) == ["constant/transportProperties"]
    note = config.unread_values["constant/transportProperties"]
    assert note.startswith("nu, ") and "outside the case" in note, note


def test_include_etc_is_not_followed(tmp_path):
    case = _case(tmp_path)
    _edit(case / "constant/transportProperties", NU_LINE, '#includeEtc "caseDicts/props"')
    config = read_case_config(str(case), directives=True)
    note = config.unread_values["constant/transportProperties"]
    assert note.startswith("nu, ") and "#includeEtc" in note, note


# ---------------------------------------------------------------- the reader itself
def test_a_macro_is_looked_up_outwards_from_the_entry(tmp_path):
    """OpenFOAM looks `$name` up in the dictionary around the entry first, then outwards."""
    case = _case(tmp_path)
    fv = case / "system/fvSolution"
    _edit(fv, U_TOLERANCE, "tolerance       $tolU;")
    _edit(fv, "solvers\n{\n", "solvers\n{\n    tolU 0.5;\n")
    assert read_case_config(str(case), directives=True).linear_solvers["U"].tolerance == 0.5
    default = read_case_config(str(case), directives=False)
    assert default.linear_solvers["U"].tolerance is None
    assert list(default.unread_values) == ["system/fvSolution"]
    assert "solvers/U/tolerance" in default.unread_values["system/fvSolution"]


# ---------------------------------------------------------------- one lookup for both readers
#: A specification item naming U's linear-solver tolerance, as the cavity writes it.
U_TOL_ITEM = {"name": "U tolerance", "file": "system/fvSolution", "path": ["solvers", "U"],
              "key": "tolerance", "value": 1e-06}


def _shadowed(case: Path, *, top: bool = True) -> None:
    """`tolerance $tol;` in U, with `tol 0.5;` beside it, and (with `top`) `tol 1e-06;` at the
    top level too. OpenFOAM reads $tol from the nearest dictionary: 0.5."""
    fv = case / "system/fvSolution"
    _edit(fv, U_TOLERANCE, "tol             0.5;\n        tolerance       $tol;")
    if top:
        _edit(fv, "\nsolvers\n", "\ntol 1e-06;\n\nsolvers\n")


def _spec(tmp_path: Path, *items: dict) -> Path:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": "courant.spec/1", "items": list(items)}))
    return spec


def test_a_shadowed_macro_reads_the_same_through_both_readers_under_next(tmp_path):
    """Review of 21 Oct 2026: the case reader looked `$tol` up from U outwards (0.5, as OpenFOAM
    does) and the spec reader at the top level (1e-06), so under --checks next one verdict
    rejected the tolerance as loose while its conformance item passed it on a value the solver
    never read. Under next both read the dictionary through one expansion (`deck.expand`)."""
    from courant.spec import deck
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    _shadowed(case)
    assert read_case_config(str(case), directives=True).linear_solvers["U"].tolerance == 0.5
    assert deck.read(case, SpecItem(**U_TOL_ITEM), macros=True).value == 0.5
    verdict = verify_path(case, spec=_spec(tmp_path, U_TOL_ITEM), checks="next")
    [item] = verdict.conformance.items
    assert (item.found, item.conforms) == ("0.5", False)
    codes = {r.code for r in verdict.reasons}
    assert "solver.loose_tolerance" in codes and any(c.startswith("conformance.") for c in codes)


@pytest.mark.usefixtures("checks_as_060")
def test_by_default_the_spec_reader_keeps_its_lookup_and_its_bytes(tmp_path):
    """GATE-13: by default the spec reader looks a macro up as 0.6.0 did (the file's top level,
    then its includes), so the item conforms as it did and the document is 0.6.0's byte for
    byte. The branch named the shadowing in the item's basis; the review of T-VER3B-ONE (3 Nov
    2026) found that reading of the whole file quadratic by default (M1) and its words claiming
    what OpenFOAM reads where an include may redefine the name (m1), so the basis is gone: under
    `--checks next` the item reads the nearest definition itself."""
    from courant.spec import deck
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    _shadowed(case)
    found = deck.read(case, SpecItem(**U_TOL_ITEM))
    assert (found.value, found.basis) == (1e-06, None)
    verdict = verify_path(case, spec=_spec(tmp_path, U_TOL_ITEM))
    [item] = verdict.conformance.items
    assert (verdict.verdict, item.conforms, item.basis) == ("accept", True, None)
    assert deck.read(case, SpecItem(**U_TOL_ITEM), macros=True).value == 0.5


def test_a_macro_defined_only_around_the_entry_is_found_by_both_readers_under_next(tmp_path):
    from courant.spec import deck
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    _shadowed(case, top=False)
    item = SpecItem(**U_TOL_ITEM)
    assert read_case_config(str(case), directives=True).linear_solvers["U"].tolerance == 0.5
    assert deck.read(case, item, macros=True).value == 0.5
    found = deck.read(case, item)                        # by default, 0.6.0's lookup
    assert found.value is None and "could not be found" in found.note


def test_both_readers_follow_the_same_include_directives(tmp_path):
    """`#includeIfPresent` brings entries in as `#include` does. One list of the directives
    followed, both readers; under next both follow it. The branch followed it in the spec
    reader by default too (insufficient to read); that moves a default verdict, so on `main` it
    is behind the flag (GATE-13) and the default reads it as 0.6.0 did."""
    from courant.case import _foam_reader
    from courant.spec import deck
    from courant.spec.model import SpecItem

    assert _foam_reader._FOLLOWED_INCLUDES is deck.INCLUDE_DIRECTIVES     # one list
    case = _case(tmp_path)
    fv = case / "system/fvSolution"
    _edit(fv, "\nsolvers\n", '\n#includeIfPresent "tols"\n\nsolvers\n')
    _edit(fv, U_TOLERANCE, "tolerance       $tolU;")
    (case / "system/tols").write_text("tolU 0.5;\n")
    config = read_case_config(str(case), directives=True)
    assert config.linear_solvers["U"].tolerance == 0.5 and config.unread_values == {}
    assert deck.read(case, SpecItem(**U_TOL_ITEM), macros=True).value == 0.5
    assert deck.read(case, SpecItem(**U_TOL_ITEM)).value is None


@pytest.mark.usefixtures("checks_as_060")
def test_a_macro_in_an_included_file_is_looked_up_in_the_dictionary_that_includes_it(tmp_path):
    """OpenFOAM puts an `#include`d file's entries into the dictionary that includes it, so a
    `$nuVal` there is looked up in the includer: nu is 0.5, and physics.nu rejects it."""
    case = _case(tmp_path)
    _edit(case / "constant/transportProperties", NU_LINE, 'nuVal 0.5;\n#include "physics"')
    (case / "constant/physics").write_text("nu [0 2 -1 0 0 0 0] $nuVal;\n")
    config = read_case_config(str(case), directives=True)
    assert config.transport["nu"] == 0.5 and config.unread_values == {}
    verdict = verify_path(case, checks="next")
    assert verdict.verdict == "reject" and "physics.nu" in {r.code for r in verdict.reasons}
    # by default nothing is followed, and the include is named, as before
    assert _notes(verify_path(case).not_assessed)


@pytest.mark.parametrize("way", ["direct", "macro"])
def test_the_fields_take_nu_from_the_case_reader(tmp_path, way):
    """The physics suite read nu with its own `FoamFile(...)['nu']`, so under --checks next a nu
    written as `$nuVal` was 0.5 to the setup rules and None to every field check that needs it.
    One reader of the viscosity the solver read: the case reader's, in the verify's mode."""
    from courant.case.config import CaseConfig
    from courant.verify.pipeline import _fields

    case = _case(tmp_path)
    shutil.copytree(case / "0", case / "0.5")           # a solved time, so fields are read
    (_direct if way == "direct" else _macro)(case, "nu")
    for directives in (False, True):
        config = CaseConfig.from_foam(str(case), directives=directives)
        fields = _fields(str(case), config=config)
        assert fields is not None and fields.nu == config.transport.get("nu"), directives
    assert _fields(str(case), config=CaseConfig.from_foam(str(case), directives=True)).nu == 0.5


def test_a_sibling_solver_merged_in_is_covered_and_followed(tmp_path):
    """`pFinal { $p; relTol 0; }`, in nearly every tutorial: `p` is read as its own solver, so
    nothing is named in either mode; under next pFinal reads p's tolerance and solver."""
    case = _case(tmp_path)
    default = read_case_config(str(case), directives=False)
    nxt = read_case_config(str(case), directives=True)
    assert default.unread_values == {} and nxt.unread_values == {}
    assert default.linear_solvers["pFinal"].tolerance is None
    assert (nxt.linear_solvers["pFinal"].tolerance, nxt.linear_solvers["pFinal"].solver,
            nxt.linear_solvers["pFinal"].rel_tol) == (1e-06, "PCG", 0)


def test_an_entry_written_after_a_merge_overrides_it_and_one_before_does_not(tmp_path):
    """OpenFOAM reads a dictionary in order: `$Ubase;` brings Ubase's entries in where it is."""
    case = _case(tmp_path)
    fv = case / "system/fvSolution"
    _edit(fv, "\nsolvers\n", "\nUbase { solver PCG; tolerance 0.5; relTol 0.9; }\n\nsolvers\n")
    _edit(fv, "        smoother        symGaussSeidel;\n", "        $Ubase;\n")
    u = read_case_config(str(case), directives=True).linear_solvers["U"]
    # `solver` is written before `$Ubase;`, so Ubase's wins; tolerance and relTol after it
    assert (u.solver, u.tolerance, u.rel_tol) == ("PCG", 1e-05, 0)
    default = read_case_config(str(case), directives=False)
    assert default.linear_solvers["U"].solver == "smoothSolver"          # as 0.6.0 read it
    assert list(default.unread_values) == ["system/fvSolution"]


def test_the_committed_runs_have_nothing_the_reader_does_not_follow():
    """GATE-13: no committed case directory gains a sentence in either mode."""
    dirs = [CAVITY, ROOT / "tests" / "fixtures" / "buoyant" / "hotRoom_laminar_044",
            *sorted((ROOT / "certification" / "monitored").glob("simpleFoam_*"))]
    assert len(dirs) >= 8
    for d in dirs:
        for directives in (False, True):
            assert read_case_config(str(d), directives=directives).unread_values == {}, (
                d.name, directives)


# ---------------------------------------------------------------- keyword patterns
def test_the_keyword_matcher_says_what_python_re_says():
    from courant.case import wordre

    patterns = ["(inlet|outlet)", ".*Wall", "(U|k|epsilon)Final", "(?i)INLET", r"wall\d+",
                "[^p].*", "a{2,3}", "(a|ab)*b", "proc.*Boundary[0-9]+", "x?y+z*", "^w$", ""]
    names = ["inlet", "outlet", "movingWall", "UFinal", "kFinal", "epsilonFinal", "inLet",
             "wall12", "p", "q", "aa", "aaa", "aaaa", "abab", "abb", "procBoundary3", "yy",
             "xyzz", "w", ""]
    for p in patterns:
        for n in names:
            assert wordre.fullmatch(p, n) is (re.fullmatch(p, n) is not None), (p, n)


def test_the_keyword_matcher_refuses_what_it_does_not_evaluate():
    from courant.case import wordre

    assert wordre.fullmatch(r"(a)\1", "aa") is None              # a backreference
    assert wordre.fullmatch("(?!b)a", "a") is None               # a lookaround
    assert wordre.fullmatch("a{100000}", "a") is None            # past MAX_STATES
    assert wordre.fullmatch("(unclosed", "x") is False           # not a pattern: matches nothing


def _bounded(code: str, seconds: float = 30) -> str:
    """Run `code` in a fresh interpreter and fail, not hang, past `seconds`."""
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                           text=True, timeout=seconds, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"did not return within {seconds} s: a keyword pattern stalled its matcher")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_a_pathological_keyword_returns_at_once_with_res_answer():
    """`re` doubled its time on `"(.*)*!"` with each character of the name."""
    out = _bounded(
        "from courant.case import wordre\n"
        "print(wordre.fullmatch('(.*)*!', 'x' * 40), wordre.fullmatch('(((.*)*)*)*!', 'm' * 60),"
        " wordre.fullmatch('(a|a)*b', 'a' * 60), wordre.fullmatch('(.*)*!', 'x' * 40 + '!'))")
    assert out == "False False False True"


def test_a_pathological_keyword_in_a_residual_control_does_not_stall():
    """The branch's `convergence._matches` is `main`'s `_first` (one lookup, three-valued)."""
    out = _bounded(
        "from courant.verification.convergence import _first\n"
        "print(_first(['\"(.*)*!\"'], 'U' * 40)[0], _first(['\"(U|k)\"'], 'k')[0])")
    assert out == "None 0"


def test_a_keyword_after_the_match_that_cannot_be_evaluated_is_not_guessed():
    from courant.spec import deck

    text = 'boundaryField { "(movingWall|lid)" { type fixedValue; } "(a)\\1" { type x; } }'
    bf = deck.block(text, "boundaryField")
    with pytest.raises(deck.KeywordNotEvaluated):
        deck.block(bf, "movingWall")
    before = 'boundaryField { "(a)\\1" { type x; } "(movingWall|lid)" { type fixedValue; } }'
    assert "fixedValue" in deck.block(deck.block(before, "boundaryField"), "movingWall")


def test_verify_with_a_spec_and_a_pathological_keyword_returns_and_conforms(tmp_path):
    """The audit's case, end to end: the unedited `courant spec init` draft of the cavity, and a
    `0/U` whose boundaryField holds `"(((.*)*)*)*!"` and groups movingWall under
    `"(movingWall|lid)"`. It ran for minutes; it returns, and the run conforms."""
    from typer.testing import CliRunner

    from courant.cli.main import app

    ref = _case(tmp_path, "reference")
    spec = tmp_path / "spec.json"
    assert CliRunner().invoke(app, ["spec", "init", str(ref), "-o", str(spec)]).exit_code == 0
    assert "U at movingWall" in {i["name"] for i in json.loads(spec.read_text())["items"]}
    agent = _case(tmp_path, "agent")
    _edit(agent / "0" / "U", "    movingWall\n",
          '    "(((.*)*)*)*!"\n    {\n        type zeroGradient;\n    }\n\n'
          '    "(movingWall|lid)"\n')
    out = _bounded(
        "import json\n"
        "from courant.verify import from_path, verify_inputs\n"
        "from courant.verify.ingest import with_spec\n"
        f"v = json.loads(verify_inputs(with_spec(from_path({str(agent)!r}), {str(spec)!r}))"
        ".to_json())\n"
        "items = {i['name']: i.get('conforms') for i in v['conformance']['items']}\n"
        "print(v['verdict'], items['U at movingWall'], all(items.values()))", seconds=60)
    assert out == "accept True True"


# ---------------------------------------------------------------- a keyword not evaluated, named
def _steady(control: dict, **kw):
    from courant.case.config import CaseConfig

    return CaseConfig(residual_control={"SIMPLE": control}, solver="simpleFoam",
                      ddt_scheme="steadyState", **kw)


def test_a_residual_control_keyword_not_evaluated_before_the_match_is_named():
    """25f2de8's `keywords_not_evaluated`: the keywords that may decide a field's tolerance, the
    lookup `_candidates` makes (one lookup, two askers)."""
    from courant.verification.convergence import keywords_not_evaluated

    config = _steady({'"(?=U)U"': 1e-6, '".*"': 1e-2})
    assert keywords_not_evaluated(config, "Ux") == ('residualControl "(?=U)U"',)
    # after the first match it cannot matter, and is not named
    assert keywords_not_evaluated(_steady({'".*"': 1e-2, '"(?=U)U"': 1e-6}), "U") == ()
    # nothing matches for certain: named, it may be the one
    assert keywords_not_evaluated(_steady({'"(?=U)U"': 5e-3}), "U") == (
        'residualControl "(?=U)U"',)


def test_a_relaxation_keyword_not_evaluated_after_the_last_match_is_named():
    from courant.verification.convergence import keywords_not_evaluated

    config = _steady({}, relaxation_factors={'"U.*"': 0.5, '"(?=U)U"': 0.05},
                     relaxation_scopes={'"U.*"': "equations", '"(?=U)U"': "equations"})
    assert keywords_not_evaluated(config, "Ux", relaxation=True) == (
        'relaxationFactors "(?=U)U"',)
    before = _steady({}, relaxation_factors={'"(?=U)U"': 0.05, '"U.*"': 0.5},
                     relaxation_scopes={'"(?=U)U"': "equations", '"U.*"': "equations"})
    assert keywords_not_evaluated(before, "U", relaxation=True) == ()


def _verify_058(control: dict, tmp_path: Path, checks: str):
    from courant.verify.ingest import from_log
    from courant.verify.pipeline import verify_inputs_v1

    meta = json.loads((CORPUS / "simpleFoam_pitzDaily__healthy.case.json").read_text())
    meta["config"]["residual_control"] = {"SIMPLE": control}
    side = tmp_path / "c10k.simpleFoam_pitzDaily.058.healthy.case.json"
    side.write_text(json.dumps(meta))
    return verify_inputs_v1(from_log(HEALTHY_058, side).model_copy(update={"checks": checks}))


KEYWORD_SENTENCE = "Case keyword pattern"


@pytest.mark.usefixtures("checks_as_060")
def test_the_review_case_names_the_keyword_under_next_and_never_quotes_it(tmp_path):
    """The healthy 058 run with `"(?=U)U" 1e-6;` before `U 1e-3;`: not converged under next (as
    `main` already judges it), and the keyword is named under not assessed by field and
    dictionary, a reader limitation that decides nothing itself. The branch quoted the keyword;
    on `main` a keyword is never quoted (the review of 22 Oct 2026: it is the case's text, and a
    sentence under not assessed reaches the support bundle)."""
    control = {'"(?=U)U"': 1e-6, "p": 1e-2, "U": 1e-3, '"(k|epsilon|omega|f|v2)"': 1e-3}
    baseline = {"p": 1e-2, "U": 1e-3, '"(k|epsilon|omega|f|v2)"': 1e-3}
    v = _verify_058(control, tmp_path, "next")
    assert v.verdict == "NOT_TRUSTWORTHY"
    [named] = [s for s in v.not_assessed if s.startswith(KEYWORD_SENTENCE)]
    assert "U: 1 residualControl keyword" in named, named
    assert "(?=U)U" not in named
    assert not [s for s in _verify_058(baseline, tmp_path, "next").not_assessed
                if s.startswith(KEYWORD_SENTENCE)]
    # the default mode reads no keyword, and says nothing of one: 0.6.0's document
    default = _verify_058(control, tmp_path, "default")
    assert default.verdict == _verify_058(baseline, tmp_path, "default").verdict
    assert not [s for s in default.not_assessed if s.startswith(KEYWORD_SENTENCE)]


def test_many_residual_control_keywords_are_named_once_under_next(tmp_path):
    """Two thousand five hundred keywords of 483 states each before pitzDaily's own entries: the
    verify returns, accepts as the shipped case does, and one sentence names what was not
    evaluated."""
    case = tmp_path / "pitz"
    shutil.copytree(PITZ, case)
    for gz in case.glob("log.*.gz"):
        with gzip.open(gz, "rb") as src, open(case / gz.name.removesuffix(".gz"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    fv = case / "system" / "fvSolution"
    keys = "".join(f'        "(?:.*){{160}}x{i}" 1e-2;\n' for i in range(2500))
    text = fv.read_text()
    anchor = "    residualControl\n    {\n"
    assert anchor in text
    fv.write_text(text.replace(anchor, anchor + keys, 1))
    out = _bounded(
        "from courant.verify import verify_path\n"
        f"v = verify_path({str(case)!r}, checks='next')\n"
        f"print(v.verdict, sum(s.startswith({KEYWORD_SENTENCE!r}) for s in v.not_assessed))",
        seconds=90)
    assert out == "accept 1"


# ---------------------------------------------------------------- the review of T-VER3B-ONE
PATCHES = 20_000


def _many_patches(case: Path, n: int = PATCHES) -> int:
    """`0/U` with `n` more patches, each `value uniform $U0;`, and movingWall's value too: some
    1 MB. Returns its size."""
    u = case / "0" / "U"
    text = u.read_text().replace("value           uniform (1.43 0 0);", "value uniform $U0;", 1)
    text = text.replace("\ndimensions", "\nU0 (1.43 0 0);\n\ndimensions", 1)
    end = text.rindex("}")
    extra = "".join(f"    p{i} {{ type fixedValue; value uniform $U0; }}\n" for i in range(n))
    u.write_text(text[:end] + extra + text[end:])
    return u.stat().st_size


LID = {"name": "lid", "file": "0/U", "patch": "movingWall", "value": [1.43, 0.0, 0.0]}


@pytest.mark.parametrize("macros", [False, True])
def test_a_macro_in_every_one_of_many_patches_is_read_in_linear_time(tmp_path, macros):
    """M1. A `0/U` of 20,000 patches each written `value uniform $U0;` (1 MB) and one item: the
    default's shadow reading expanded the whole file with a lookup that rescanned every scope it
    grew (quadratic: hours), and the case reader's `expand` under `case.directives` is the same
    function. Each scope keeps its names as it grows (`deck._macros`), so the reading is linear:
    a scaled time per byte, held under 20 s for 4 MB, as T-LAMINAR-REGIME's reading is."""
    case = _case(tmp_path)
    size = _many_patches(case)
    out = _bounded(
        "import time\n"
        "from courant.spec import deck\n"
        "from courant.spec.model import SpecItem\n"
        "t = time.perf_counter()\n"
        f"found = deck.read({str(case)!r}, SpecItem(**{LID!r}), macros={macros})\n"
        "print(found.value, found.basis, time.perf_counter() - t)", seconds=120)
    value, basis, elapsed = out.rsplit(" ", 2)
    assert (value, basis) == ("[1.43, 0.0, 0.0]", "None")
    assert float(elapsed) * 4_000_000 / size < 20.0, (elapsed, size)


def test_the_expansion_is_bounded_however_merges_nest():
    """M1's bound: `$name;` merges that double at each level (`a1 { $a0; $a0; }` ...) grow the
    text as 2^levels. Past `deck.MAX_EXPANDED` characters the expansion stops and the text is
    left as written, so what it would set is read as unread, never guessed, and it returns."""
    import time

    from courant.spec import deck

    text = "a0 { x 1; }\n" + "".join(f"a{i} {{ $a{i - 1}; $a{i - 1}; }}\n" for i in range(1, 40))
    text += "top { $a39; }\ny $x;\n"
    t = time.perf_counter()
    out = deck.expand(text, Path("/nowhere/system/x"), Path("/nowhere"))
    assert time.perf_counter() - t < 5.0
    assert len(out) <= deck.MAX_EXPANDED and "$a39;" in out
    assert deck.MAX_EXPANDED <= 16 * 2**20


PROTECTED = ("#inputMode {mode}\nnuVal 1e-5;\nnuVal 1e-3;\n"
             "nu              [0 2 -1 0 0 0 0] $nuVal;")


@pytest.mark.parametrize("mode", ["protect", "warn", "error"])
def test_a_macro_defined_twice_under_protect_is_not_read_by_either_reader(tmp_path, mode):
    """M2. Under `#inputMode protect` v2406 keeps the first of two definitions (1e-5), `warn` the
    first with a warning, `error` stops; `expand` took the last (1e-3), so under next a spec of
    nu 1e-3 conformed on a viscosity the solver never read: a false accept. A name defined more
    than once where such a mode is in force stays unexpanded: the spec item is not checked, and
    the case reader names nu unread. Under merge (the default) the last stands, as before."""
    from courant.case.config import CaseConfig
    from courant.spec import deck
    from courant.spec.model import SpecItem

    case = _case(tmp_path)
    _edit(case / "constant/transportProperties", NU_LINE, PROTECTED.format(mode=mode))
    item = SpecItem(name="nu", file="constant/transportProperties", key="nu", value=1e-3)
    found = deck.read(case, item, macros=True)
    assert found.value is None, found
    verdict = verify_path(case, spec=_spec(tmp_path, item.model_dump(exclude_none=True)),
                          checks="next")
    [checked] = verdict.conformance.items
    assert checked.conforms is None and verdict.verdict != "accept"
    config = CaseConfig.from_foam(str(case), directives=True)
    assert "nu" not in config.transport
    assert "constant/transportProperties" in config.unread_values
    merged = _case(tmp_path / "merge")
    _edit(merged / "constant/transportProperties", NU_LINE,
          PROTECTED.format(mode="merge"))
    assert deck.read(merged, item, macros=True).value == 1e-3
