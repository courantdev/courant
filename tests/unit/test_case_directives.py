"""A setup value written through `$macro`, `#include` or `#calc` is read, or said to be unread
(TRD VER-3b; hard-proofing audit, 21 Oct 2026).

The case reader parses the dictionaries with foamlib, which expands no macro and follows no
include: `nu [0 2 -1 0 0 0 0] $nuVal;` came back as the token `$nuVal`, `_scalar` made it None,
and `physics.nu` skipped it in silence. A viscosity of 0.5 written directly is rejected
(VER-5.4); the same 0.5 behind `$nuVal`, `#include "nuDict"` or `#calc` was accepted at grade A,
by `courant verify` and `courant preflight`, with nothing under "not checked". The deck reader of
the same codebase expands `$name` and `courant spec init` drafts 0.5 on that case: Courant read
what the solver read for conformance and ignored it for validity.

Now (GATE-13):
- by default a value a directive hides is reported, under `not_assessed` and preflight's
  `not_checked`, and the verdict is unchanged (a reader limitation never decides it, INV-3);
- under `--checks next` (`case.directives`) the dictionary is read as the solver reads it, its
  includes inside the case followed and its macros expanded (`courant.spec.deck.expand`), so the
  fault is rejected; what still cannot be read (`#calc`, a missing include) is reported.

The case is the committed 2000 x 1 cavity fixture with its own log; every variant uses that log.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAVITY = ROOT / "tests" / "fixtures" / "real_openfoam" / "cavity_extreme_aspect_ratio"

HDR = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
       "    object %s;\n}\n")
SOLVERS = ("solvers\n{\n    p { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.05; }\n"
           "    pFinal { $p; relTol 0; }\n"
           "    U { solver smoothSolver; smoother symGaussSeidel; tolerance %s; relTol 0; }\n}\n")
PISO = "PISO\n{\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n    pRefCell 0;\n    pRefValue 0;\n}\n"
FV, TP = "system/fvSolution", "constant/transportProperties"
NU = "nu [0 2 -1 0 0 0 0] %s;\n"


def _fv(body: str) -> str:
    return HDR % "fvSolution" + body


def _tp(body: str) -> str:
    return HDR % "transportProperties" + body


#: Written through a directive inside the case: reported by default, rejected under next.
FOLLOWED = {
    "tol_macro": ({FV: _fv("tolU 0.5;\n" + SOLVERS % "$tolU" + PISO)}, "solver.loose_tolerance"),
    "tol_include_macro": ({"system/tols": "tolU 0.5;\n",
                           FV: _fv('#include "tols"\n' + SOLVERS % "$tolU" + PISO)},
                          "solver.loose_tolerance"),
    "tol_include_solvers": ({"system/solverSettings": SOLVERS % "0.5",
                             FV: _fv('#include "solverSettings"\n' + PISO)},
                            "solver.loose_tolerance"),
    "tol_merge_outer": ({FV: _fv("slowU { solver smoothSolver; smoother symGaussSeidel; "
                                 "tolerance 0.5; relTol 0; }\n"
                                 + SOLVERS.replace("    U { solver smoothSolver; smoother "
                                                   "symGaussSeidel; tolerance %s; relTol 0; }",
                                                   "    U { $slowU; }") + PISO)},
                        "solver.loose_tolerance"),
    "nu_macro": ({TP: _tp("nuVal 0.5;\n" + NU % "$nuVal")}, "physics.nu"),
    "nu_include": ({"constant/nuDict": NU % "0.5", TP: _tp('#include "nuDict"\n')}, "physics.nu"),
    "nu_include_macro": ({"constant/nuDict": "nuVal 0.5;\n",
                          TP: _tp('#include "nuDict"\n' + NU % "$nuVal")}, "physics.nu"),
    "nu_include_after": ({"constant/nuDict": NU % "0.5",
                          TP: _tp(NU % "0.00848" + '#include "nuDict"\n')}, "physics.nu"),
    "nu_include_case_tag": ({"constant/nuDict": NU % "0.5",
                             TP: _tp('#include "<constant>/nuDict"\n')}, "physics.nu"),
}
#: Not followable by any mode: reported in both, the verdict unchanged.
UNFOLLOWED = {
    "nu_calc": {TP: _tp(NU % '#calc "0.25*2"')},
    "tol_calc": {FV: _fv(SOLVERS % '#calc "0.25*2"' + PISO)},
    "nu_overwrite": {TP: _tp(NU % "0.5" + "#inputMode overwrite\n" + NU % "0.00848")},
    "nu_missing_include": {TP: _tp('#include "nothere"\n')},
    "nu_include_outside": {TP: _tp('#include "../../outside/nuDict"\n')},
    "nu_include_etc": {TP: _tp('#includeEtc "caseDicts/transport"\n')},
    # An optional include of a file outside the case, through a tag: the solver read the
    # outside file's value (it exists), so the value before it is not the one it read. The
    # review of 21 Oct 2026 found it dropped as "absent" under next, and that value read silently.
    "nu_include_if_present_outside": {TP: _tp(NU % "0.00848"
                                              + '#includeIfPresent "$FOAM_CASE/../outside/nuDict"\n')},
    "nu_sinclude_outside": {TP: _tp(NU % "0.00848" + '#sinclude "<case>/../outside/nuDict"\n')},
}
#: Written directly: rejected in both modes, as before, and nothing reported unread.
DIRECT = {
    "tol_direct": ({FV: _fv(SOLVERS % "0.5" + PISO)}, "solver.loose_tolerance"),
    "tol_regex_solver": ({FV: _fv(SOLVERS.replace("    U {", '    "(U|k|epsilon)" {') % "0.5"
                                  + PISO)}, "solver.loose_tolerance"),
    "nu_direct": ({TP: _tp(NU % "0.5")}, "physics.nu"),
    "nu_nodims": ({TP: _tp("nu 0.5;\n")}, "physics.nu"),
    "nu_uniform": ({TP: _tp("nu uniform 0.5;\n")}, "physics.nu"),
    "nu_dup": ({TP: _tp(NU % "0.00848" + NU % "0.5")}, "physics.nu"),
}

UNREAD = "Case values were not read as the solver read them"
UNPARSED = "Case dictionary could not be parsed"


def _case(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    case = tmp_path / "case"
    shutil.copytree(CAVITY, case, ignore=shutil.ignore_patterns("*.gz"))
    with gzip.open(CAVITY / "log.icoFoam.gz", "rt") as fh:
        (case / "log.icoFoam").write_text(fh.read())
    for rel, text in (files or {}).items():
        (case / rel).parent.mkdir(parents=True, exist_ok=True)
        (case / rel).write_text(text)
    outside = tmp_path / "outside" / "nuDict"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text(NU % "0.5")
    return case


def _verify(case: Path, checks: str):
    from courant.verify import verify_path

    return verify_path(case, checks=checks)


def _preflight(case: Path, checks: str):
    from courant.verify.preflight import preflight_path

    return preflight_path(case, checks=checks)


def _unread(lines) -> list[str]:
    return [s for s in lines if s.startswith(UNREAD)]


def _unparsed(lines) -> list[str]:
    return [s for s in lines if s.startswith(UNPARSED)]


# ---------------------------------------------------------------- the case as committed
@pytest.mark.parametrize("checks", ["default", "next"])
def test_the_committed_cavity_reports_nothing_unread_and_is_accepted(tmp_path, checks):
    """Its `pFinal { $p; relTol 0; }` merges a solver entry the checks read anyway."""
    v = _verify(_case(tmp_path), checks)
    assert (v.verdict, v.rule) == ("accept", "VER-5.6"), (v.verdict, v.reasons)
    assert _unread(v.not_assessed) == [] and _unparsed(v.not_assessed) == []
    assert _unread(_preflight(_case(tmp_path / "p"), checks).not_checked) == []


def test_no_committed_case_directory_is_noted_in_either_mode():
    """The default notes nothing on the cases the repository holds (the monitored steady cases,
    the fixtures): their `$turbulence` schemes are expanded by the scheme check itself, and their
    `pFinal { $p; }` merges a solver entry read anyway. Under next their values are read
    expanded, and still nothing is noted."""
    from courant.case.config import CaseConfig

    tops = [d for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")]
    cases = sorted({p.parent.parent for top in tops for p in top.glob("**/system/controlDict")})
    assert len(cases) >= 10 and CAVITY in cases
    for case in cases:
        for directives in (False, True):
            config = CaseConfig.from_foam(str(case), directives=directives)
            assert config.unread_values == {}, (case, directives, config.unread_values)


# ---------------------------------------------------------------- written directly: as before
@pytest.mark.parametrize("name", sorted(DIRECT))
@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_fault_written_directly_is_rejected_as_before(tmp_path, name, checks):
    files, code = DIRECT[name]
    v = _verify(_case(tmp_path, files), checks)
    assert v.verdict == "reject" and code in [r.code for r in v.reasons], (v.verdict, v.reasons)
    assert _unread(v.not_assessed) == []
    p = _preflight(_case(tmp_path / "p", files), checks)
    assert p.verdict == "SETUP_FAULT" and code in [r.code for r in p.reasons]


# ---------------------------------------------------------------- through a directive
@pytest.mark.parametrize("name", sorted(FOLLOWED))
def test_by_default_a_value_behind_a_directive_is_reported_unread_and_the_verdict_stands(
        tmp_path, name):
    files, code = FOLLOWED[name]
    rel = next(r for r in files if r in (FV, TP))
    v = _verify(_case(tmp_path, files), "default")
    assert (v.verdict, v.rule) == ("accept", "VER-5.6"), (v.verdict, v.rule, v.reasons)
    unread = _unread(v.not_assessed)
    assert len(unread) == 1 and rel in unread[0], v.not_assessed
    assert "--checks next" in unread[0], unread
    p = _preflight(_case(tmp_path / "p", files), "default")
    assert p.verdict == "NO_SETUP_FAULT_FOUND"
    assert any(rel in s for s in _unread(p.not_checked)), p.not_checked


@pytest.mark.parametrize("name", sorted(FOLLOWED))
def test_under_next_the_directive_is_followed_and_the_fault_rejected(tmp_path, name):
    files, code = FOLLOWED[name]
    v = _verify(_case(tmp_path, files), "next")
    assert v.verdict == "reject" and code in [r.code for r in v.reasons], (v.verdict, v.reasons)
    assert _unread(v.not_assessed) == []
    p = _preflight(_case(tmp_path / "p", files), "next")
    assert p.verdict == "SETUP_FAULT" and code in [r.code for r in p.reasons], p
    assert _unread(p.not_checked) == []


@pytest.mark.parametrize("name", sorted(UNFOLLOWED))
@pytest.mark.parametrize("checks", ["default", "next"])
def test_a_value_no_mode_can_read_is_reported_in_both_and_the_verdict_stands(tmp_path, name,
                                                                            checks):
    files = UNFOLLOWED[name]
    rel = next(iter(files))
    v = _verify(_case(tmp_path, files), checks)
    assert v.verdict == "accept", (v.verdict, v.reasons)
    unread = _unread(v.not_assessed)
    assert len(unread) == 1 and rel in unread[0], v.not_assessed
    assert "--checks next does" not in unread[0], unread
    p = _preflight(_case(tmp_path / "p", files), checks)
    assert p.verdict == "NO_SETUP_FAULT_FOUND" and _unread(p.not_checked), p


# ---------------------------------------------------------------- the sentences
def test_the_default_sentence_names_the_file_the_value_and_how_it_is_written(tmp_path):
    v = _verify(_case(tmp_path, FOLLOWED["nu_macro"][0]), "default")
    assert _unread(v.not_assessed) == [
        "Case values were not read as the solver read them: constant/transportProperties: nu "
        "is written as '$nuVal', a macro the default checks do not expand (--checks next does) "
        "— findings that depend on them are unavailable, not absent"]


def test_an_include_names_the_values_it_may_set(tmp_path):
    v = _verify(_case(tmp_path, FOLLOWED["tol_include_solvers"][0]), "default")
    [line] = _unread(v.not_assessed)
    assert "system/fvSolution: relaxationFactors, SIMPLE, PIMPLE, solvers may be set by " \
           "'#include \"solverSettings\"', which the default checks do not follow " \
           "(--checks next does)" in line, line


def test_a_braced_macro_foamlib_cannot_parse_is_read_under_next(tmp_path):
    """`${nuVal}` stops foamlib parsing the file at all: by default the file is reported
    unreadable, as any file that does not parse; under next it is read expanded."""
    files = {TP: _tp("nuVal 0.5;\n" + NU % "${nuVal}")}
    v = _verify(_case(tmp_path, files), "default")
    assert v.verdict == "accept" and any(TP in s for s in _unparsed(v.not_assessed))
    v = _verify(_case(tmp_path / "n", files), "next")
    assert v.verdict == "reject" and "physics.nu" in [r.code for r in v.reasons]
    assert _unread(v.not_assessed) == [] and _unparsed(v.not_assessed) == []


def test_an_include_that_cannot_be_followed_says_why_in_both_modes(tmp_path):
    for checks in ("default", "next"):
        v = _verify(_case(tmp_path / checks, UNFOLLOWED["nu_missing_include"]), checks)
        [line] = _unread(v.not_assessed)
        assert "may be set by '#include \"nothere\"', which could not be followed: no such " \
               "file in the case" in line, line


def test_calc_is_not_evaluated_by_any_mode(tmp_path):
    v = _verify(_case(tmp_path, UNFOLLOWED["nu_calc"]), "next")
    [line] = _unread(v.not_assessed)
    assert "nu is written as '#calc \"0.25*2\"', which Courant does not evaluate" in line, line


def test_an_include_outside_the_case_is_not_read_in_any_mode(tmp_path):
    """Confined to the case, as the deck reader's includes are: the file outside is the
    fault's, and it is neither read nor listed."""
    case = _case(tmp_path, UNFOLLOWED["nu_include_outside"])
    v = _verify(case, "next")
    [line] = _unread(v.not_assessed)
    assert "which could not be followed: the file is outside the case" in line, line
    assert all("outside" not in f.path for f in v.run.inputs)


@pytest.mark.parametrize("inc", ['"nothere"', '"<constant>/nothere"', '"$FOAM_CASE/nothere"'])
def test_under_next_an_optional_include_missing_from_the_case_is_dropped_as_the_solver_drops_it(
        tmp_path, inc):
    """The solver skips an `#includeIfPresent` whose file is absent: under next the value
    before it is the one it read, and nothing is noted."""
    files = {TP: _tp(NU % "0.5" + f"#includeIfPresent {inc}\n")}
    v = _verify(_case(tmp_path, files), "next")
    assert v.verdict == "reject" and "physics.nu" in [r.code for r in v.reasons], v.reasons
    assert _unread(v.not_assessed) == []


@pytest.mark.parametrize("name", ["nu_macro", "tol_macro", "nu_include", "tol_include_solvers"])
def test_the_finding_names_files_keys_and_kinds_never_what_was_written(tmp_path, name):
    """The finding (`case.unread_value`) reaches the evidence bundle a diagnoser and the cloud
    see; what the customer wrote (a macro's name, an include's path, an expression) stays in the
    local sentence and `CaseConfig.unread_values` (the review of 21 Oct 2026)."""
    from courant.case.config import CaseConfig
    from courant.diagnosis.checkers import CaseIntegrityChecker, CheckContext

    files, _ = FOLLOWED[name]
    config = CaseConfig.from_foam(str(_case(tmp_path, files)))
    [finding] = [f for f in CaseIntegrityChecker().run(CheckContext(
        run_id="r", events=[], config=config, mesh=None, log_text="")) if f.id ==
        "case.unread_value"]
    detail = {"nu_macro": "constant/transportProperties: nu (macro)",
              "tol_macro": "system/fvSolution: solvers/U/tolerance (macro)",
              "nu_include": "constant/transportProperties: nu, rho, Pr, Prt, beta, TRef, phases "
                            "(directive)",
              "tol_include_solvers": "system/fvSolution: relaxationFactors, SIMPLE, PIMPLE, "
                                     "solvers (directive)"}[name]
    assert finding.detail == detail + " — findings that depend on them are unavailable, not absent"
    for written in ("$", "#", '"'):
        assert written not in finding.detail, finding.detail
    assert set(config.unread_keys) == set(config.unread_values)


def test_the_local_sentence_keeps_what_was_written_bounded(tmp_path):
    long = "#calc \"" + "+".join(["0.25"] * 80) + "\""
    v = _verify(_case(tmp_path, {TP: _tp(NU % long)}), "default")
    [line] = _unread(v.not_assessed)
    assert "nu is written as '#calc \"0.25+0.25" in line and len(line) < 400, line


def test_under_next_the_included_file_is_in_the_manifest(tmp_path):
    """VER-14: the file the value came from is a file that decided the verdict."""
    case = _case(tmp_path, FOLLOWED["nu_include"][0])
    listed = {f.path for f in _verify(case, "next").run.inputs}
    assert "constant/nuDict" in listed
    assert "constant/nuDict" not in {f.path for f in _verify(case, "default").run.inputs}


# ---------------------------------------------------------------- the reader itself
def test_the_reader_follows_only_when_asked(tmp_path):
    from courant.case.config import CaseConfig

    case = _case(tmp_path, FOLLOWED["nu_macro"][0])
    default = CaseConfig.from_foam(str(case))
    assert "nu" not in default.transport and list(default.unread_values) == [TP]
    assert default.unreadable == {}, "it parsed: a value was not read, not the file"
    followed = CaseConfig.from_foam(str(case), directives=True)
    assert followed.transport["nu"] == 0.5 and followed.unread_values == {}


def test_followed_merges_and_macros_give_the_values_the_solver_read(tmp_path):
    from courant.case.config import CaseConfig

    case = _case(tmp_path, FOLLOWED["tol_include_macro"][0])
    s = CaseConfig.from_foam(str(case), directives=True).linear_solvers
    assert s["U"].tolerance == 0.5
    assert (s["pFinal"].solver, s["pFinal"].tolerance, s["pFinal"].rel_tol) == ("PCG", 1e-06, 0)


def test_the_expander_follows_includes_and_scoped_macros():
    from courant.spec.deck import expand

    text = ("a 1;\nouter { b $a; inner { c $b; d $a; } }\nm { x 2; }\nn { $m; y $x; }\n"
            "later $z;\nz 3;\n")
    out = expand(text, Path("/nowhere/system/x"), Path("/nowhere"))
    from foamlib import FoamFile

    d = FoamFile.loads(out)
    assert d["outer"]["b"] == 1 and d["outer"]["inner"] == {"c": 1, "d": 1}
    assert d["n"] == {"x": 2, "y": 2}
    assert d["later"] == "$z", "a macro is expanded from the definitions before it"
