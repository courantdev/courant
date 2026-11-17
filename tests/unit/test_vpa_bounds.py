"""Verdict-path audit: `bounds.plausibility` and round-off (VPA-CHK-2, fixed behind
`--checks next` as `courant.verify.checks.BOUNDS_ROUNDOFF`), and `scheme.relaxation` on SIMPLEC
and on `fields`/`equations` (VPA-CHK-3, open: strict xfail).

VPA-CHK-2: scalarTransportFoam's passive scalar T spans 0 to 1 (v2406's own
basic/scalarTransportFoam/pitzDaily); an undershoot to -1e-6 is round-off of that scale
(`courant.verification.bounding.ROUNDOFF`, the one definition), not a negative temperature.
"""

from __future__ import annotations

import gzip
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from courant.verify import checks

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "real_openfoam"
FIXTURES = REAL / "wedge_pipe"
DUCT = REAL / "duct"


def _gunzip_all(root: Path) -> None:
    for gz in list(root.rglob("*.gz")):
        gz.with_suffix("").write_bytes(gzip.decompress(gz.read_bytes()))
        gz.unlink()


def _copy(name: str, tmp_path: Path, root: Path = FIXTURES) -> Path:
    case = tmp_path / Path(name).name
    shutil.copytree(root / name, case)
    _gunzip_all(case)
    return case


def _rescale(text: str) -> str:
    """(T - 300) / 20 for every scalar of a T file: uniform values and the internal list."""
    text = re.sub(r"uniform\s+([-+0-9.eE]+)",
                  lambda m: f"uniform {(float(m.group(1)) - 300.0) / 20.0:.10g}", text)
    m = re.search(r"(internalField\s+nonuniform\s+List<scalar>\s*\n?\s*\d+\s*\n?\()(.*?)(\))",
                  text, re.DOTALL)
    if m is None:
        return text
    vals = [(float(v) - 300.0) / 20.0 for v in m.group(2).split()]
    body = "\n" + "\n".join(f"{v:.10g}" for v in vals) + "\n"
    return text[:m.start(2)] + body + text[m.end(2):]


def _set_first_cell(text: str, value: float) -> str:
    m = re.search(r"(internalField\s+nonuniform\s+List<scalar>\s*\n?\s*\d+\s*\n?\()(.*?)(\))",
                  text, re.DOTALL)
    assert m is not None
    vals = m.group(2).split()
    vals[0] = repr(value)
    body = "\n" + "\n".join(vals) + "\n"
    return text[:m.start(2)] + body + text[m.end(2):]


def _passive_scalar_case(tmp_path: Path, first_cell: float) -> Path:
    """The wedge-pipe energy run with T rescaled to (T - 300) / 20 (0 to 1, in 0/ and the
    latest time, boundary values included) and its first cell set to `first_cell`."""
    case = _copy("energy", tmp_path)
    for t in (case / "0" / "T", case / "1214" / "T"):
        t.write_text(_rescale(t.read_text()))
    latest = case / "1214" / "T"
    latest.write_text(_set_first_cell(latest.read_text(), first_cell))
    return case


def _bounds(case: Path, mode: str, monkeypatch):
    """The `bounds.plausibility` result of the trust report the gate built, and the verdict."""
    from courant.verification import suite
    from courant.verify.ingest import from_path
    from courant.verify.pipeline import verify_inputs

    seen = []
    real = suite.verify

    def spy(*a, **k):
        report = real(*a, **k)
        seen.append(report)
        return report

    monkeypatch.setattr(suite, "verify", spy)
    verdict = verify_inputs(from_path(case, checks=mode))
    assert seen, "the gate built no trust report"
    hits = [r for r in seen[-1].results if r.check_code == "bounds.plausibility"]
    assert len(hits) == 1
    return hits[0], verdict


def test_the_name_is_held_back_for_next():
    assert checks.NEXT[checks.BOUNDS_ROUNDOFF] == "0.8.0"
    assert not checks.enabled(checks.BOUNDS_ROUNDOFF, "default")
    assert checks.enabled(checks.BOUNDS_ROUNDOFF, "next")


def test_the_rescaled_case_spans_zero_to_one(tmp_path):
    case = _passive_scalar_case(tmp_path, 0.5)
    text = (case / "1214" / "T").read_text()
    m = re.search(r"List<scalar>\s*\n?\s*\d+\s*\n?\((.*?)\)", text, re.DOTALL)
    vals = np.array([float(v) for v in m.group(1).split()])
    assert -1e-9 <= vals.min() and vals.max() <= 1.0 + 1e-9 and vals.max() > 0.5


@pytest.mark.parametrize("undershoot", [-1e-6, -1e-12])
def test_next_holds_a_passive_scalar_undershoot_to_round_off(tmp_path, monkeypatch, undershoot):
    case = _passive_scalar_case(tmp_path, undershoot)
    result, verdict = _bounds(case, "next", monkeypatch)
    assert result.passed and result.severity.value == "ok", result.detail
    assert "bounds.plausibility" not in {r.code for r in verdict.reasons}


def test_next_still_flags_an_undershoot_beyond_round_off(tmp_path, monkeypatch):
    case = _passive_scalar_case(tmp_path, -0.5)
    result, _ = _bounds(case, "next", monkeypatch)
    assert not result.passed and result.severity.value == "major"
    assert "negative T" in result.detail


def test_default_is_unchanged_at_minus_one_micro(tmp_path, monkeypatch):
    case = _passive_scalar_case(tmp_path, -1e-6)
    result, _ = _bounds(case, "default", monkeypatch)
    assert not result.passed and result.severity.value == "major"
    assert result.detail == "negative T (min -1.00e-06)"
    assert result.value == -1e-6


@pytest.mark.parametrize("mode", ["default", "next"])
@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_a_non_finite_T_still_grades_as_before(tmp_path, monkeypatch, mode, bad):
    """Read straight from the check: a non-finite field is MAJOR in either mode."""
    from courant.verification import VerifyContext
    from courant.verification.checks import PlausibilityCheck
    from courant.verification.fields import FieldData

    t = np.linspace(0.0, 1.0, 50)
    t[3] = float(bad)
    fields = FieldData(time=1.0, internal={"T": t}, nu=None, inlet_speed=None)
    r = PlausibilityCheck().run(VerifyContext(run_id="x", events=[], fields=fields, checks=mode))
    assert not r.passed and r.severity.value == "major"
    assert r.detail == "non-finite T (1 of 50 values nan or inf)"


@pytest.mark.parametrize("name", ["k", "epsilon", "omega", "T"])
def test_the_check_reuses_the_one_round_off(name):
    """Under next, within `bounding.ROUNDOFF` of the field's range (its largest value less its
    smallest non-negative one: 2.0 - 0.25 here; the review of the fix replaced max |value|, which
    let a kelvin field's offset hide an undershoot) is not a finding, beyond it is; under the
    default any value below zero is, as in 0.7.0."""
    from courant.verification import VerifyContext, bounding
    from courant.verification.checks import PlausibilityCheck
    from courant.verification.fields import FieldData

    def run(lo, mode):
        arr = np.array([lo, 0.25, 2.0])
        fields = FieldData(time=1.0, internal={name: arr}, nu=None, inlet_speed=None)
        return PlausibilityCheck().run(
            VerifyContext(run_id="x", events=[], fields=fields, checks=mode))

    inside = -0.5 * bounding.ROUNDOFF * 1.75
    beyond = -2.0 * bounding.ROUNDOFF * 1.75
    assert run(inside, "next").passed
    assert not run(beyond, "next").passed
    assert not run(inside, "default").passed
    assert run(inside, "default").detail == f"negative {name} (min {inside:.2e})"


def test_an_all_zero_field_with_a_negative_is_still_a_finding():
    """A field whose only scale is its negative: no round-off to hold it to."""
    from courant.verification import VerifyContext
    from courant.verification.checks import PlausibilityCheck
    from courant.verification.fields import FieldData

    fields = FieldData(time=1.0, internal={"k": np.array([0.0, 0.0, -1e-6])}, nu=None,
                       inlet_speed=None)
    r = PlausibilityCheck().run(VerifyContext(run_id="x", events=[], fields=fields,
                                              checks="next"))
    assert not r.passed


# --------------------------------------------------------------------------------------------
# VPA-CHK-3, open: `scheme.relaxation` reads a flattened map and ignores SIMPLEC
# --------------------------------------------------------------------------------------------

def _relaxation_findings(case: Path) -> list:
    from courant.verify.ingest import from_path
    from courant.verify.pipeline import verify_inputs

    verdict = verify_inputs(from_path(case))
    return [r for r in verdict.reasons if r.code == "scheme.relaxation"]


def _set_relaxation(case: Path, block: str) -> None:
    fv = case / "system" / "fvSolution"
    text = fv.read_text()
    m = re.search(r"relaxationFactors\s*\{", text)
    assert m is not None
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    fv.write_text(text[:m.start()] + "relaxationFactors\n{\n" + block + "\n}\n" + text[i:])


def test_the_simplec_flow_fixture_is_accepted_as_written(tmp_path):
    case = _copy("flow", tmp_path)
    assert re.search(r"consistent\s+(yes|true|on)", (case / "system" / "fvSolution").read_text())
    assert _relaxation_findings(case) == []


def test_the_duct_fixture_has_no_relaxation_finding_as_written(tmp_path):
    assert _relaxation_findings(_copy("converged", tmp_path, DUCT)) == []


@pytest.mark.xfail(strict=True, reason="VPA-CHK-3: scheme.relaxation ignores SIMPLEC "
                   "(consistent yes): `fields { p 1; }` is judged undamped though OpenFOAM "
                   "behaves identically without it")
def test_simplec_with_p_field_factor_one_is_not_a_relaxation_finding(tmp_path):
    case = _copy("flow", tmp_path)
    _set_relaxation(case, "    fields { p 1; }\n    equations { U 0.9; T 0.9; }")
    assert _relaxation_findings(case) == []


@pytest.mark.xfail(strict=True, reason="VPA-CHK-3: scheme.relaxation flattens fields and "
                   "equations, so `equations { p 1; }` overwrites `fields { p 0.3; }`, and "
                   "pressure is never equation-relaxed")
def test_equations_p_does_not_overwrite_fields_p(tmp_path):
    case = _copy("converged", tmp_path, DUCT)
    _set_relaxation(case, "    fields { p 0.3; }\n    equations { U 0.7; h 0.7; p 1; }")
    assert _relaxation_findings(case) == []
