"""A RAS or LES case whose `turbulence` switch is off models no turbulence (TRD VER-17a).

OpenFOAM reads the switch as `getOrDefault<Switch>("turbulence", true)` (`RASModel.C`), and a
model's `correct()` returns at once when it is off: k and epsilon are never solved and nut stays
at its initial field. The hard-proofing audit (21 Oct 2026) ran v2406's pisoFoam/RAS/cavity and
buoyantPimpleFoam/hotRoom with only `turbulence off;` changed: 0 k and 0 epsilon solves, nut
frozen, and both accepted at grade A, where `simulationType laminar` on the same hotRoom is
rejected blind (`physics.laminar_unstable`). The case reader read `simulationType` only.

Under `case.turbulence_switch` (behind `--checks next`, GATE-13) such a case is laminar for the
checks that ask `CaseConfig.is_laminar`: `physics.laminar_unstable`, VER-17's regime check and
`regime.reynolds`. The verdicts here are on the certification corpus's own laminar twins (logs
of runs that solved no model), with their configuration saying RAS and `turbulence off`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courant.case.config import CaseConfig

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"

_HEAD = ("FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
         "    object turbulenceProperties;\n}\n")


def _read(tmp_path: Path, body: str) -> CaseConfig:
    case = tmp_path / "case"
    (case / "constant").mkdir(parents=True)
    (case / "system").mkdir()
    (case / "system" / "controlDict").write_text(
        _HEAD.replace("turbulenceProperties", "controlDict") + "application pisoFoam;\n")
    (case / "constant" / "turbulenceProperties").write_text(_HEAD + body)
    return CaseConfig.from_foam(str(case))


# ------------------------------------------------------------------ the reader
@pytest.mark.parametrize("word,switch", [("off", False), ("false", False), ("no", False),
                                         ("on", True), ("true", True), ("yes", True)])
def test_the_reader_reads_the_switch_as_openfoam_does(tmp_path, word, switch):
    cfg = _read(tmp_path, "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n"
                          f"    turbulence {word};\n    printCoeffs on;\n}}\n")
    assert (cfg.turbulence_type, cfg.turbulence_model) == ("RAS", "kEpsilon")
    assert cfg.turbulence_switch is switch


def test_an_unwritten_switch_is_none_openfoams_on(tmp_path):
    cfg = _read(tmp_path, "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n}\n")
    assert cfg.turbulence_switch is None and not cfg.is_laminar("next")


def test_an_les_switch_is_read_from_the_les_block(tmp_path):
    cfg = _read(tmp_path, "simulationType LES;\nRAS { turbulence on; }\n"
                          "LES\n{\n    LESModel Smagorinsky;\n    turbulence off;\n}\n")
    assert cfg.turbulence_switch is False and cfg.is_laminar("next")


# ------------------------------------------------------------------ is_laminar
def test_turbulence_off_is_laminar_under_next_only():
    cfg = CaseConfig(turbulence_type="RAS", turbulence_model="kEpsilon", turbulence_switch=False)
    assert cfg.is_laminar("next") and not cfg.is_laminar()


def test_laminar_and_turbulence_on_read_as_before_in_both_modes():
    lam = CaseConfig(turbulence_type="laminar")
    on = CaseConfig(turbulence_type="RAS", turbulence_switch=True)
    assert lam.is_laminar() and lam.is_laminar("next")
    assert not on.is_laminar() and not on.is_laminar("next")


# ------------------------------------------------------------------ the verdicts
def _verify(sid: str, tmp_path: Path, checks: str, **config):
    from courant.verify.ingest import from_log
    from courant.verify.pipeline import verify_inputs_v1

    meta = json.loads((CORPUS / f"{sid}.case.json").read_text())
    meta["config"].update(config)
    side = tmp_path / f"{sid}.case.json"
    side.write_text(json.dumps(meta))
    return verify_inputs_v1(from_log(CORPUS / f"{sid}.log.gz", side).model_copy(
        update={"checks": checks}))


TWIN = "buoyantPimpleFoam_hotRoom__laminar_on_turbulent"


def test_a_model_switched_off_is_rejected_under_next_as_laminar_is(tmp_path):
    v = _verify(TWIN, tmp_path, "next", turbulence_type="RAS", turbulence_switch=False)
    assert (v.verdict, v.rule) == ("NOT_TRUSTWORTHY", "VER-5.4")
    (r,) = [r for r in v.reasons if r.code == "physics.laminar_unstable"]
    assert r.summary.startswith("Laminar model with unstable residuals: RAS kEpsilon with its "
                                "turbulence switch off")
    laminar = _verify(TWIN, tmp_path, "next")
    assert (laminar.verdict, laminar.rule) == (v.verdict, v.rule)


def test_the_default_accepts_it_as_v050_did(tmp_path):
    v = _verify(TWIN, tmp_path, "default", turbulence_type="RAS", turbulence_switch=False)
    assert (v.verdict, v.rule) == ("TRUSTWORTHY", "VER-5.6")


def test_the_laminar_twins_finding_keeps_its_words(tmp_path):
    (r,) = [r for r in _verify(TWIN, tmp_path, "next").reasons
            if r.code == "physics.laminar_unstable"]
    assert "laminar turbulence type, Courant number within its stability limit" in r.summary


def test_a_settled_run_with_its_model_switched_off_is_not_rejected_on_that_alone(tmp_path):
    """Blind, the switch makes the run laminar, and it is the residuals that reject a laminar
    run: one whose history settles stays accepted. That is the blind limit WHAT-COURANT-SEES
    states; a specification naming the model catches it (SPEC-13, `test_conformance_as_run`)."""
    v = _verify("buoyantPimpleFoam_hotRoom__healthy", tmp_path, "next", turbulence_switch=False)
    assert v.verdict == "TRUSTWORTHY"
