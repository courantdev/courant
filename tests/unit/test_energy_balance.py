"""TRD VER-22: the heat balance, from the wall heat flux a run wrote (T-ENERGY-BALANCE).

An internal evaluation of rebuilt runs (not a measurement) found its only two false accepts on a finned heat sink whose runs
closed 38.2% and 91.7% of their energy, accepted at grade A: no check read heat. Those runs are
no longer on disk, so nothing here is about them. These tests hold `conservation.energy`, behind
`courant verify --checks next`, on this repository's own v2406 solves only:

- `hot_box/{converged,early}`: buoyantSimpleFoam, one closed region, four walls;
- `cht_plate/{converged,early}`: chtMultiRegionSimpleFoam, a solid `plate` under an `air`
  channel open to flow, joined by a `mappedWall` pair.

Every case is a committed solve copied into `tmp_path` first (TST-4). The oracle is OpenFOAM's own
output in two forms, never a second Courant reading: the `wallHeatFlux.dat` rows at the log's last
time, transcribed literally below (`ROWS`) once the solves exist, and the solver log's own
`min/max/integ(<patch>) = a, b, c` lines at that step. Nothing is mocked: the reader and the check
run on the solver's rows. `_judge` only watches `TrustReport.from_results` (the real one is
called) to see the check's result where an OK result never reaches the document.

What the implementer is held to by name (one place, `_read_heat` and `_judge`):
- `courant.verification._heat_reader.read_heat(case_dir, last, config, solver, checks)`, the
  arguments the pipeline passes (work order section 5, step 7), returning a reading with
  `rows` ({region: {patch: integral}}), `files` ({region: the `.dat` path relative to the case})
  and `measures` ({a region's name, or the interface's two patch names: the measure}). The
  single region of a one-region case is OpenFOAM's `region0` (None or "" are read as it).
- the check's code `conservation.energy`; under `next` its result joins the trust report, as
  `fields.unread`'s does.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import re
import shutil
import tokenize
from decimal import Decimal
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.verification.checks import MassFluxCheck
from courant.verification.grade import VSeverity
from tests.unit.test_mass_flux import _unpack_fixture

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "tests" / "fixtures" / "real_openfoam"
CODE = "conservation.energy"
DEFAULT_REGION = "region0"           # OpenFOAM's polyMesh::defaultRegion

#: Each solve: its committed directory and the solver its log is named after (`log.<solver>`).
SOLVES = {
    "hot_box-converged": (REAL / "hot_box" / "converged", "buoyantSimpleFoam"),
    "hot_box-early": (REAL / "hot_box" / "early", "buoyantSimpleFoam"),
    "cht_plate-converged": (REAL / "cht_plate" / "converged", "chtMultiRegionSimpleFoam"),
    "cht_plate-early": (REAL / "cht_plate" / "early", "chtMultiRegionSimpleFoam"),
}
CONVERGED = {"hot_box-converged", "cht_plate-converged"}
INTERFACE = frozenset({"plate_to_air", "air_to_plate"})

#: The wall patches of each region, listed literally (work order C1), not re-derived.
WALLS = {
    "hot_box": {DEFAULT_REGION: {"hot", "cold", "top", "bottom"}},
    "cht_plate": {"plate": {"heater", "plate_left", "plate_right", "plate_to_air"},
                  "air": {"top", "air_to_plate"}},
}
#: The balances owed (VER-22 (a) and (b)): the closed regions by name, the interface as its pair.
JUDGED = {"hot_box": {DEFAULT_REGION}, "cht_plate": {"plate", INTERFACE}}

#: OpenFOAM's own rows at the log's last time, transcribed literally from each committed
#: `wallHeatFlux.dat` once the solves exist (work order section 4): {solve: (the log's last
#: `Time =` as printed, {region: {patch: the integral column as printed}})}. Not a Courant
#: reading; never computed. Empty until plan step 1 has solved and committed the four runs.
ROWS: dict[str, tuple[str, dict[str, dict[str, str]]]] = {
    "hot_box-converged": ("2243", {DEFAULT_REGION: {
        "hot": "1.5914856500e-02", "cold": "-1.5905522012e-02",
        "top": "0.0000000000e+00", "bottom": "0.0000000000e+00"}}),
    "hot_box-early": ("20", {DEFAULT_REGION: {
        "hot": "3.3116898887e-02", "cold": "-4.1469029257e-08",
        "top": "0.0000000000e+00", "bottom": "0.0000000000e+00"}}),
    "cht_plate-converged": ("3000", {
        "plate": {"heater": "1.6264992763e-01", "plate_left": "0.0000000000e+00",
                  "plate_right": "0.0000000000e+00", "plate_to_air": "-1.6264992763e-01"},
        "air": {"top": "0.0000000000e+00", "air_to_plate": "1.6264992763e-01"}}),
    "cht_plate-early": ("10", {
        "plate": {"heater": "6.1363290243e+02", "plate_left": "0.0000000000e+00",
                  "plate_right": "0.0000000000e+00", "plate_to_air": "-1.2847946579e-01"},
        "air": {"top": "0.0000000000e+00", "air_to_plate": "1.0121296230e-01"}}),
}


# ----------------------------------------------------------------------------- the fixtures
def _family(name: str) -> str:
    return name.split("-", 1)[0]


def _fixture(name: str) -> Path:
    src = SOLVES[name][0]
    assert src.is_dir(), (f"the solve {src.relative_to(ROOT)} is not committed (T-ENERGY-BALANCE "
                          f"plan step 1)")
    return src


def _solve(tmp: Path, name: str, *, packed: bool = False) -> Path:
    """The solve copied into `tmp` (TST-4): as committed (`packed`), or unzipped. The copy keeps
    the fixture's own directory name, so two copies give the same document."""
    src = _fixture(name)
    case = tmp / src.name
    if packed:
        shutil.copytree(src, case)
        return case
    return _unpack_fixture(src, case)


def _log_text(name: str) -> str:
    src, solver = SOLVES[name][0], SOLVES[name][1]
    packed, plain = src / f"log.{solver}.gz", src / f"log.{solver}"
    if packed.is_file():
        with gzip.open(packed, "rt") as fh:
            return fh.read()
    assert plain.is_file(), f"no log.{solver} is committed in {src.relative_to(ROOT)}"
    return plain.read_text()


def _last_time(log: str) -> str:
    """The log's last `Time = ` as OpenFOAM printed it."""
    times = re.findall(r"^Time = (\S+)[ \t]*$", log, flags=re.MULTILINE)
    assert times, "the log prints no Time ="
    return times[-1]


def _logged_integrals(log: str) -> dict[str, Decimal]:
    """The solver log's own `min/max/integ(<patch>) = a, b, c` lines in the last step's block
    (after the last `Time = `): v2406's wallHeatFlux executes the last step's function objects
    there, before `End`. The patch names of every fixture are distinct across its regions."""
    block = re.split(r"^Time = \S+[ \t]*$", log, flags=re.MULTILINE)[-1]
    got = {m.group(1): Decimal(m.group(4)) for m in re.finditer(
        r"min/max/integ\(([^)\s]+)\) = (\S+), (\S+), (\S+)$", block, flags=re.MULTILINE)}
    assert got, "the log's last step prints no min/max/integ(<patch>) line"
    return got


def _half_last_digit(printed: Decimal) -> float:
    """Half a unit in the last digit OpenFOAM printed."""
    return 0.5 * 10.0 ** printed.as_tuple().exponent


def _transcribed(name: str) -> tuple[str, dict[str, dict[str, str]]]:
    _fixture(name)                      # the solve first: nothing is transcribed from nothing
    assert name in ROWS, (f"the rows of {name} at the log's last time are not transcribed into "
                          f"ROWS (work order section 4: literally, from the committed .dat)")
    return ROWS[name]


def _heat_files(case: Path) -> list[str]:
    """Every `wallHeatFlux` table the solve wrote, relative to the case."""
    return sorted(p.relative_to(case).as_posix()
                  for p in (case / "postProcessing").rglob("wallHeatFlux*.dat*") if p.is_file())


def _region_of(rel: str) -> str:
    """`postProcessing/[<region>/]<name>/<startTime>/wallHeatFlux.dat` (`writeFile::baseFileDir`)."""
    parts = rel.split("/")
    return parts[1] if len(parts) == 5 else DEFAULT_REGION


def _region(key) -> str:
    return DEFAULT_REGION if key in (None, "") else key


def _key(key):
    return _region(key) if key is None or isinstance(key, str) else frozenset(key)


def _boundary_of(case: Path, region: str) -> Path:
    sub = () if region == DEFAULT_REGION else (region,)
    return case.joinpath("constant", *sub, "polyMesh", "boundary")


def _wall_patches(case: Path, region: str) -> set[str]:
    from courant.case.polymesh import read_boundary

    return {p.name for p in read_boundary(_boundary_of(case, region))
            if p.type in ("wall", "mappedWall")}


def _strip_heat_output(case: Path) -> Path:
    for rel in _heat_files(case):
        (case / rel).unlink()
    return case


# ------------------------------------------------------------------------- the oracle's sums
def _region_measure(values: list[float]) -> float | None:
    """VER-22 (a): |net| over the heat in, or over the heat out when no patch carries heat in
    above ROUND_OFF times the largest integral; None when neither does."""
    largest = max(abs(v) for v in values)
    tiny = MassFluxCheck.ROUND_OFF * largest
    net = sum(values)
    heat_in = sum(v for v in values if v > 0)
    heat_out = -sum(v for v in values if v < 0)
    if any(v > tiny for v in values):
        return abs(net) / heat_in
    if heat_out > tiny:
        return abs(net) / heat_out
    return None


def _pair_measure(a: float, b: float) -> float:
    """VER-22 (b): |q_a + q_b| over the larger of |q_a| and |q_b|."""
    return abs(a + b) / max(abs(a), abs(b))


def _expected_measures(name: str) -> dict:
    """Each owed balance's measure, from the transcribed rows alone."""
    _, rows = _transcribed(name)
    values = {r: {p: float(v) for p, v in patches.items()} for r, patches in rows.items()}
    out = {}
    for key in JUDGED[_family(name)]:
        if isinstance(key, str):
            m = _region_measure(list(values[key].values()))
            assert m is not None, f"{name}'s {key} carries no heat across its walls"
            out[key] = m
        else:
            a, b = sorted(key)
            qa = next(v[a] for v in values.values() if a in v)
            qb = next(v[b] for v in values.values() if b in v)
            out[key] = _pair_measure(qa, qb)
    return out


def _graded(measure: float) -> VSeverity:
    """MassFluxCheck's bars, read at call time (VER-18, by reference)."""
    if measure <= MassFluxCheck.TARGET:
        return VSeverity.OK
    return VSeverity.MINOR if measure <= MassFluxCheck.LOOSE else VSeverity.MAJOR


# --------------------------------------------------------------- the reader and the verdict
def _read_heat(case: Path, config_case: Path | None = None):
    """The heat reader with what the pipeline gives it (work order section 5, step 7): the log's
    last time, the case configuration and the solver that ran, read from `config_case` (an
    unpacked copy of the same solve) or the case itself."""
    from courant.verify import from_case_dir
    from courant.verify.pipeline import _solver

    assert importlib.util.find_spec("courant.verification._heat_reader") is not None, (
        "there is no heat reader (courant/verification/_heat_reader.py)")
    from courant.verification._heat_reader import read_heat

    inputs = from_case_dir(config_case or case, checks="next")
    config = CaseConfig.from_view(inputs.config_view, read_here=inputs.source == "case_dir")
    last = float(_last_time(inputs.log_text))
    return read_heat(str(case), last, config, _solver(inputs, config), "next")


def _judge(case: Path, monkeypatch, checks: str = "next"):
    """The verdict document and the `conservation.energy` result the gate graded, or None. The
    real `TrustReport.from_results` is called; it is only watched."""
    from courant.verification.grade import TrustReport
    from courant.verify import from_case_dir, verify_inputs

    seen: list[list] = []
    real = TrustReport.__dict__["from_results"].__func__

    def watched(cls, run_id, results):
        seen.append(list(results))
        return real(cls, run_id, results)

    with monkeypatch.context() as m:
        m.setattr(TrustReport, "from_results", classmethod(watched))
        doc = json.loads(verify_inputs(from_case_dir(case, checks=checks)).to_json())
    energy = [r for r in (seen[-1] if seen else []) if r.check_code == CODE]
    assert len(energy) <= 1, energy
    return doc, (energy[0] if energy else None)


def _bar(severity: VSeverity) -> float:
    return MassFluxCheck.LOOSE if severity is VSeverity.MAJOR else MassFluxCheck.TARGET


def _assert_cited(doc: dict, result, worst_key, case: Path) -> None:
    """A non-OK balance is a reason citing the value, the bar, the region or pair, the file."""
    cited = [x for x in doc["reasons"] if x["code"] == CODE]
    assert len(cited) == 1, doc["reasons"]
    reason = cited[0]
    assert reason["severity"] == result.severity.value
    assert float(reason["value"]) == result.value
    assert float(reason["threshold"]) == _bar(result.severity)
    text = f"{reason['summary']} {reason['input']}"
    if isinstance(worst_key, str):
        if worst_key != DEFAULT_REGION:
            assert worst_key in text, text
        files = [f for f in _heat_files(case) if _region_of(f) == worst_key]
    else:
        assert all(p in text for p in worst_key), text
        files = _heat_files(case)
    assert any(f in text for f in files), (files, text)


# ------------------------------------------------------------------------------------- C1
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize("name", list(SOLVES))
def test_each_region_reads_the_wallheatflux_row_of_the_logs_last_time(tmp_path, name, packed):
    """C1. Each region's rows at the log's last time: exactly its wall patches, each the literal
    transcription of the committed `.dat` and the log's own `integ(...)` to half a unit in its
    last printed digit. The committed (gzipped) solve reads as an unpacked copy does."""
    family = _family(name)
    _fixture(name)
    log = _log_text(name)
    last, rows = _transcribed(name)
    assert last == _last_time(log), (last, _last_time(log))
    logged = _logged_integrals(log)
    unpacked = _solve(tmp_path / "unpacked", name)
    # The literal lists are this solve's own wall patches, splitMeshRegions' names included.
    for region, walls in WALLS[family].items():
        assert _wall_patches(unpacked, region) == walls, region
    if family == "cht_plate":
        for here, there, patch in (("plate", "air", "air_to_plate"),
                                   ("air", "plate", "plate_to_air")):
            text = _boundary_of(unpacked, here).read_text()
            entry = re.search(rf"\b{sorted(INTERFACE - {patch})[0]}\s*\{{([^}}]*)\}}", text)
            assert entry and re.search(rf"\bsampleRegion\s+{there}\s*;", entry.group(1)) \
                and re.search(rf"\bsamplePatch\s+{patch}\s*;", entry.group(1)), here
    assert {r: set(p) for r, p in rows.items()} == WALLS[family]

    case = _solve(tmp_path / "case", name, packed=packed)
    reading = _read_heat(case, unpacked)
    got = {_region(r): dict(p) for r, p in reading.rows.items()}
    assert {r: set(p) for r, p in got.items()} == WALLS[family]
    for region, patches in got.items():
        for patch, value in patches.items():
            assert value == float(rows[region][patch]), (region, patch, value)
            ref = logged[patch]
            assert abs(value - float(ref)) <= _half_last_digit(ref), (patch, value, str(ref))
    files = _heat_files(case)
    assert {_region(r): f for r, f in reading.files.items()} == {_region_of(f): f for f in files}
    if packed:
        again = _read_heat(unpacked)
        assert {_region(r): dict(p) for r, p in again.rows.items()} == got


# ------------------------------------------------------------------------------------- C2
@pytest.mark.parametrize("name", list(SOLVES))
def test_next_grades_each_solved_balance_on_openfoams_own_rows(tmp_path, name, monkeypatch):
    """C2 (and C7's manifest). `value` is the worst measure computed here from the literal rows;
    the severity is MassFluxCheck's bars on it. A converged solve graded above TARGET is a false
    alarm to report to the orchestrator, not a bar, a setting or a stop time to change."""
    expected = _expected_measures(name)
    worst_key = max(expected, key=expected.get)
    worst = expected[worst_key]
    grade = _graded(worst)
    case = _solve(tmp_path, name)
    doc, r = _judge(case, monkeypatch)
    assert r is not None, f"no {CODE} result under --checks next"
    assert not r.skipped, r.detail
    assert r.value == pytest.approx(worst, rel=1e-9), (r.value, worst)
    assert r.severity is grade, (r.severity, grade, r.detail)
    if name in CONVERGED:
        assert grade is VSeverity.OK, (
            f"false alarm on the converged solve {name}: OpenFOAM's own rows give {worst:.3e}; "
            f"report it to the orchestrator, do not change a bar or the case")
    paths = {i["path"] for i in doc["run"]["inputs"]}
    assert set(_heat_files(case)) <= paths, sorted(paths)
    if _family(name) == "cht_plate":
        assert {"constant/regionProperties", "constant/plate/polyMesh/boundary",
                "constant/air/polyMesh/boundary"} <= paths, sorted(paths)
    else:
        assert "constant/polyMesh/boundary" in paths, sorted(paths)
    if grade is VSeverity.OK:
        assert not [x for x in doc["reasons"] if x["code"] == CODE]
    else:
        _assert_cited(doc, r, worst_key, case)


def test_a_solved_run_out_of_heat_balance_is_rejected(tmp_path, monkeypatch):
    """C2: hot_box `early` started at the cold wall's temperature and stopped at step 20, so it
    has taken in heat that has not left. If its own rows do not give a non-OK grade, report it
    (work order C2); never tune the run onto a side of a bar. MAJOR: VER-5.3 rejects, grade C."""
    name = "hot_box-early"
    worst = max(_expected_measures(name).values())
    grade = _graded(worst)
    assert grade is not VSeverity.OK, (
        f"hot_box early's own rows give {worst:.3e}, within TARGET: no solver-written non-OK "
        f"grade is shown; report it to the orchestrator")
    case = _solve(tmp_path, name)
    doc, r = _judge(case, monkeypatch)
    assert r is not None and r.severity is grade, r
    if grade is VSeverity.MAJOR:
        assert (doc["verdict"], doc["rule"], doc["grade"]) == ("reject", "VER-5.3", "C")
    else:
        assert doc["grade"] != "A"
    _assert_cited(doc, r, DEFAULT_REGION, case)


# ------------------------------------------------------------------------------------- C3
def test_a_cht_run_judges_its_solid_and_its_interface_and_says_the_open_air_is_not_judged(
        tmp_path, monkeypatch):
    """C3. On cht_plate the `plate` region and the `plate_to_air`/`air_to_plate` pair are
    judged, each measure computed; the `air` region, open through `inlet` and `outlet`, is not,
    and the sentence says the heat its flow carries is not read. On hot_box the one region is."""
    for name in ("cht_plate-converged", "cht_plate-early"):
        expected = _expected_measures(name)
        case = _solve(tmp_path / name, name)
        reading = _read_heat(case)
        measures = {_key(k): m for k, m in reading.measures.items()}
        assert set(measures) == JUDGED["cht_plate"], set(measures)
        for key, m in expected.items():
            assert measures[key] == pytest.approx(m, rel=1e-9), (key, measures[key], m)
        doc, _ = _judge(case, monkeypatch)
        air = [s for s in doc["not_assessed"]
               if "air" in s and "inlet" in s and "outlet" in s and "heat" in s.lower()]
        assert len(air) == 1, doc["not_assessed"]
        assert "flow" in air[0] and "not read" in air[0], air[0]
    case = _solve(tmp_path / "hot_box", "hot_box-converged")
    assert {_key(k) for k in _read_heat(case).measures} == JUDGED["hot_box"]


# ------------------------------------------------------------------------------------- C4
def _rows_edit(case: Path, edit) -> None:
    """`edit(fields, last)` on every data row of every heat table: a list of fields to keep the
    row (as edited), or None to drop it."""
    from courant.spec.as_run import _same_time

    log = next(case.glob("log.*Foam"))
    last = float(_last_time(log.read_text()))
    for rel in _heat_files(case):
        path = case / rel
        out = []
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                out.append(line)
                continue
            fields = line.split()
            kept = edit(fields, _same_time(float(fields[0]), last))
            if kept is not None:
                out.append("\t".join(kept))
        path.write_text("\n".join(out) + "\n")


def _at_last(patch: str, change):
    return lambda f, at_last: change(f) if (at_last and f[1] == patch) else f


def _sub(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    assert n, f"{pattern!r} is not in {path.name}"
    path.write_text(new)


def _fv(obj: str, body: str) -> str:
    return f"FoamFile{{version 2.0;format ascii;class dictionary;object {obj};}}\n{body}\n"


def _conflicting_restart(case: Path) -> None:
    """A second start-time directory beside the first, holding a different `hot` row at the
    log's last time (a restart that wrote the same time again)."""
    from courant.spec.as_run import _same_time

    (rel,) = _heat_files(case)
    first = case / rel
    last = float(_last_time(next(case.glob("log.*Foam")).read_text()))
    header = [ln for ln in first.read_text().splitlines() if ln.startswith("#")]
    row = next(ln.split() for ln in first.read_text().splitlines()
               if ln.strip() and not ln.startswith("#") and ln.split()[1] == "hot"
               and _same_time(float(ln.split()[0]), last))
    row[4] = repr(float(row[4]) * 2.0 + 1.0)
    start = next(t for t in ("1", "2") if t != first.parent.name)
    second = first.parent.parent / start / first.name
    second.parent.mkdir(parents=True)
    second.write_text("\n".join([*header, "\t".join(row)]) + "\n")


def _transient(case: Path) -> None:
    """The same rows, from a run whose configuration is buoyantPimpleFoam's, time-marching."""
    _sub(case / "system" / "controlDict", r"^(\s*application\s+)buoyantSimpleFoam", r"\1buoyantPimpleFoam")
    _sub(case / "system" / "fvSchemes", r"^(\s*default\s+)steadyState", r"\1Euler")
    _sub(next(case.glob("log.*Foam")), r"^(Exec\s*:\s*)buoyantSimpleFoam", r"\1buoyantPimpleFoam")


#: {case: (the solve, an edit of the case made to both copies, an edit of the heat output made
#: to the copy that keeps it, the key words its sentence holds literally)}.
NOT_JUDGED = {
    "missing_patch": ("hot_box-early", None,
                      lambda c: _rows_edit(c, lambda f, at_last: None if f[1] == "top" else f),
                      ["top", "no row"]),
    "cut_before_last_time": ("hot_box-early", None,
                             lambda c: _rows_edit(c, lambda f, at_last: None if at_last else f),
                             ["no row"]),
    "unparseable_row": ("hot_box-early", None,
                        lambda c: _rows_edit(c, _at_last("hot", lambda f: f[:3])),
                        ["hot", "parse"]),
    "conflicting_restart": ("hot_box-early", None, _conflicting_restart, ["hot", "different"]),
    "fvOptions": ("hot_box-early",
                  lambda c: (c / "constant" / "fvOptions").write_text(_fv("fvOptions", (
                      "source1\n{\n    type scalarSemiImplicitSource;\n    selectionMode all;\n"
                      "    volumeMode absolute;\n    sources { h (100 0); }\n}"))),
                  None, ["constant/fvOptions"]),
    "region_fvOptions": ("cht_plate-converged",
                         lambda c: (c / "constant" / "plate" / "fvOptions").write_text(_fv(
                             "fvOptions", (
                                 "source1\n{\n    type scalarSemiImplicitSource;\n"
                                 "    selectionMode all;\n    volumeMode absolute;\n"
                                 "    sources { h (100 0); }\n}"))),
                         None, ["constant/plate/fvOptions"]),
    "radiation": ("hot_box-early",
                  lambda c: (c / "constant" / "radiationProperties").write_text(_fv(
                      "radiationProperties", "radiation on;\nradiationModel P1;\n"
                      "solverFreq 1;\nabsorptionEmissionModel none;\nscatterModel none;\n"
                      "sootModel none;")),
                  None, ["radiation"]),
    "unreadable_boundary": ("cht_plate-early",
                            lambda c: (c / "constant" / "plate" / "polyMesh" / "boundary")
                            .write_bytes(b"\x00\xff not a boundary (((( {{{"),
                            None, ["constant/plate/polyMesh/boundary"]),
    "transient": ("hot_box-early", _transient, None, ["stored"]),
    "solver_mismatch": ("hot_box-early",
                        lambda c: _sub(next(c.glob("log.*Foam")),
                                       r"^(Exec\s*:\s*)buoyantSimpleFoam", r"\1buoyantPimpleFoam"),
                        None, ["buoyantPimpleFoam", "buoyantSimpleFoam", "system/controlDict"]),
}


@pytest.mark.parametrize("shape", list(NOT_JUDGED))
def test_what_is_not_judged_says_why(tmp_path, shape, monkeypatch):
    """C4. On copies of the real solves: a sentence under "not assessed", with the key words
    literally and about heat, and no exception; no `conservation.energy` reason, and verdict,
    rule and grade as on the same copy with no `wallHeatFlux` output."""
    name, case_edit, heat_edit, words = NOT_JUDGED[shape]
    case = _solve(tmp_path / "with", name)
    bare = _strip_heat_output(_solve(tmp_path / "without", name))
    if case_edit is not None:
        case_edit(case)
        case_edit(bare)
    if heat_edit is not None:
        heat_edit(case)
    doc, r = _judge(case, monkeypatch)
    without, _ = _judge(bare, monkeypatch)
    assert r is None or r.skipped, (r.severity, r.detail)
    assert not [x for x in doc["reasons"] if x["code"] == CODE], doc["reasons"]
    new = [s for s in doc["not_assessed"] if s not in without["not_assessed"]]
    if shape == "cut_before_last_time":         # the sentence names the time it has no row at
        last = _last_time(_log_text(name))
        new = [s for s in new if last in s or f"{float(last):g}" in s]
    said = [s for s in new if "heat" in s.lower() and all(w in s for w in words)]
    assert said, (words, new)
    assert (doc["verdict"], doc["rule"], doc["grade"]) == (
        without["verdict"], without["rule"], without["grade"])


def test_a_non_finite_integral_is_major(tmp_path, monkeypatch):
    """C4. A `nan` integral in a row read: MAJOR, so VER-5.3 rejects the converged run."""
    case = _solve(tmp_path, "hot_box-converged")
    _rows_edit(case, _at_last("hot", lambda f: [*f[:4], "nan"]))
    doc, r = _judge(case, monkeypatch)
    assert r is not None and r.severity is VSeverity.MAJOR and not r.passed, r
    assert (doc["verdict"], doc["rule"], doc["grade"]) == ("reject", "VER-5.3", "C")
    cited = [x for x in doc["reasons"] if x["code"] == CODE]
    assert len(cited) == 1 and cited[0]["severity"] == "major", doc["reasons"]


# ------------------------------------------------------------------------------------- C5
@pytest.mark.parametrize("name", list(SOLVES))
def test_the_default_never_opens_the_heat_output(tmp_path, name, monkeypatch):
    """C5, GATE-13, as it can be tested (see the work order's log, 2 Nov 2026): v0.6.0's own
    monitor discovery (`courant.converge.series`) already opens a `wallHeatFlux.dat` under the
    default and lists it, so "byte-equal with and without the output" cannot hold, and that
    reading is v0.6.0's, kept under the default by GATE-13 (graph node T-MONITOR-WALLHEATFLUX
    is to tell the table from a monitor behind next; TRD VER-22 says so). What this node adds must be absent from the default: the heat
    reader never runs (the document is the same with it made to fail), nothing it alone reads
    (regionProperties, a region's boundary) enters the manifest, and no reason or sentence of it
    appears."""
    from courant.verify import from_case_dir, verify_inputs

    case = _solve(tmp_path / "with", name)
    bare = _strip_heat_output(_solve(tmp_path / "without", name))
    assert importlib.util.find_spec("courant.verification._heat_reader") is not None, (
        "there is no heat reader (courant/verification/_heat_reader.py)")
    from courant.verification import _heat_reader

    def tripped(*args, **kwargs):
        raise AssertionError("the heat reader ran under the default")

    plain = verify_inputs(from_case_dir(case)).to_json()
    with monkeypatch.context() as m:
        m.setattr(_heat_reader, "read_heat", tripped)
        assert verify_inputs(from_case_dir(case)).to_json() == plain
    doc, without = json.loads(plain), json.loads(verify_inputs(from_case_dir(bare)).to_json())
    added = ({i["path"] for i in doc["run"]["inputs"]}
             - {i["path"] for i in without["run"]["inputs"]})
    assert added <= set(_heat_files(case)), sorted(added)
    assert not [x for x in doc["reasons"] if x["code"] == CODE]
    assert not [s for s in doc["not_assessed"] if s not in without["not_assessed"]
                and "heat" in s.lower()]


def test_next_adds_nothing_where_no_wallheatflux_was_written(tmp_path, monkeypatch):
    """C5. hot_box with no `postProcessing/`: under next, no `conservation.energy` result and no
    sentence of it (VER-22: a run that wrote no wallHeatFlux output gets nothing added)."""
    case = _solve(tmp_path, "hot_box-converged")
    assert (case / "postProcessing").is_dir(), "the committed hot_box wrote no postProcessing/"
    shutil.rmtree(case / "postProcessing")
    assert importlib.util.find_spec("courant.verification._heat_reader") is not None, (
        "there is no heat reader (courant/verification/_heat_reader.py)")
    doc, r = _judge(case, monkeypatch)
    assert r is None, (r.severity, r.detail)
    assert not [x for x in doc["reasons"] if x["code"] == CODE]
    assert not [s for s in doc["not_assessed"]
                if any(w in s.lower() for w in ("wallheatflux", "heat balance",
                                                "energy balance", "heat the flow"))]


# ------------------------------------------------------------------------------------- C6
def test_the_heat_balance_is_held_to_the_mass_balances_bars(tmp_path, monkeypatch):
    """C6. One fact: moving `MassFluxCheck`'s bar moves the energy grade on the same rows, and
    restoring it moves it back."""
    name = "hot_box-early"
    worst = max(_expected_measures(name).values())
    case = _solve(tmp_path, name)
    _, before = _judge(case, monkeypatch)
    assert before is not None and before.severity is _graded(worst), before
    if worst > MassFluxCheck.LOOSE:
        attr, value = "LOOSE", worst * 2.0
    elif worst > MassFluxCheck.TARGET:
        attr, value = "LOOSE", (worst + MassFluxCheck.TARGET) / 2.0
    else:
        attr, value = "TARGET", worst / 2.0
    with monkeypatch.context() as m:
        m.setattr(MassFluxCheck, attr, value)
        moved = _graded(worst)
        assert moved is not before.severity
        _, during = _judge(case, monkeypatch)
    assert during is not None and during.severity is moved, (during, moved)
    _, after = _judge(case, monkeypatch)
    assert after.severity is before.severity


def test_the_new_code_holds_no_bar_of_its_own():
    """C6. No numeric literal 1e-3, 1e-2 or 1e-9 in the heat reader or the check: the bars and
    the round-off are MassFluxCheck's, by reference."""
    import inspect

    assert importlib.util.find_spec("courant.verification._heat_reader") is not None, (
        "there is no heat reader (courant/verification/_heat_reader.py)")
    from courant.verification import _heat_reader, checks

    assert hasattr(checks, "energy_balance"), "there is no check (checks.energy_balance)"
    for where, text in (("_heat_reader", inspect.getsource(_heat_reader)),
                        ("energy_balance", inspect.getsource(checks.energy_balance))):
        numbers = [t.string for t in tokenize.generate_tokens(io.StringIO(text).readline)
                   if t.type == tokenize.NUMBER]
        assert not [n for n in numbers if float(n) in (1e-3, 1e-2, 1e-9)], (where, numbers)


# ------------------------------------------------- the boundary is the mesh the solver read
def _add_last_row(case: Path, patch: str, value: str) -> None:
    """A row for `patch` at the log's last time, in the one heat table of a one-region solve."""
    (rel,) = _heat_files(case)
    path = case / rel
    last = _last_time(next(case.glob("log.*Foam")).read_text())
    path.write_text(path.read_text().rstrip("\n") + f"\n{last}\t{patch}\t0\t0\t{value}\n")


def _with_wall(boundary_text: str, patch: str) -> str:
    """A boundary file's text with one more wall patch, of no faces, before the list closes."""
    n = re.search(r"^(\d+)\s*$", boundary_text, flags=re.MULTILINE)
    assert n, "no patch count in the boundary file"
    text = (boundary_text[:n.start(1)] + str(int(n.group(1)) + 1)
            + boundary_text[n.end(1):])
    entry = (f"    {patch}\n    {{\n        type            wall;\n        nFaces          0;\n"
             f"        startFace       0;\n    }}\n")
    close = text.rindex("\n)")
    return text[:close + 1] + entry + text[close + 1:]


def test_a_row_of_a_patch_the_boundary_read_has_no_wall_of_is_not_judged(tmp_path, monkeypatch):
    """C4, from the review of round 1: a row naming a patch the boundary read does not list as a
    wall (a mesh changed after the boundary file was written, such as a parallel snappyHexMesh
    the root mesh never saw) leaves the region unjudged, with a sentence naming the patch and the
    boundary file, never a balance summed without it."""
    name = "hot_box-early"
    case = _solve(tmp_path / "with", name)
    bare = _strip_heat_output(_solve(tmp_path / "without", name))
    _, rows = _transcribed(name)
    out = -sum(float(v) for v in rows[DEFAULT_REGION].values())
    _add_last_row(case, "ghost", repr(out))
    doc, r = _judge(case, monkeypatch)
    without, _ = _judge(bare, monkeypatch)
    assert r is None or r.skipped, (r.severity, r.detail)
    assert not [x for x in doc["reasons"] if x["code"] == CODE], doc["reasons"]
    said = [s for s in doc["not_assessed"] if s not in without["not_assessed"]
            and "heat" in s.lower() and "ghost" in s and "constant/polyMesh/boundary" in s]
    assert said, doc["not_assessed"]
    assert (doc["verdict"], doc["rule"], doc["grade"]) == (
        without["verdict"], without["rule"], without["grade"])


def test_a_parallel_run_is_judged_on_the_boundary_its_ranks_read(tmp_path, monkeypatch):
    """The review of round 1: the patches are those of the mesh the solver read
    (`courant.case.mesh.solver_mesh`), so a parallel run's `processor<N>/constant/polyMesh`, not
    the root mesh beside it. hot_box `early` as two ranks whose boundary carries a wall `ghost`
    the root boundary does not, its row closing the balance: judged on the ranks' patches, OK."""
    name = "hot_box-early"
    case = _solve(tmp_path, name)
    _, rows = _transcribed(name)
    values = [float(v) for v in rows[DEFAULT_REGION].values()]
    _add_last_row(case, "ghost", repr(-sum(values)))
    log = next(case.glob("log.*Foam"))
    _sub(log, r"^(Exec\s*:\s*buoyantSimpleFoam)[ \t]*$", r"\1 -parallel")
    _sub(log, r"^(nProcs\s*:\s*)1[ \t]*$", r"\g<1>2")
    mesh = case / "constant" / "polyMesh"
    for rank in ("processor0", "processor1"):
        shutil.copytree(mesh, case / rank / "constant" / "polyMesh")
        boundary = case / rank / "constant" / "polyMesh" / "boundary"
        boundary.write_text(_with_wall(boundary.read_text(), "ghost"))
    from courant.verification._heat_reader import read_heat
    from courant.verify import from_case_dir
    from courant.verify.pipeline import _solver

    inputs = from_case_dir(case, checks="next")
    config = CaseConfig.from_view(inputs.config_view, read_here=True)
    reading = read_heat(str(case), float(_last_time(inputs.log_text)), config,
                        _solver(inputs, config), "next", log_text=inputs.log_text)
    assert set(reading.rows[DEFAULT_REGION]) == WALLS["hot_box"][DEFAULT_REGION] | {"ghost"}
    m = _region_measure([*values, -sum(values)])
    assert {_key(k) for k in reading.measures} == {DEFAULT_REGION}, reading.sentences
    doc, r = _judge(case, monkeypatch)
    assert r is not None and not r.skipped, doc["not_assessed"]
    assert r.value == pytest.approx(m, abs=1e-12) and r.severity is _graded(m), r
    paths = {i["path"] for i in doc["run"]["inputs"]}
    assert {"processor0/constant/polyMesh/boundary",
            "processor1/constant/polyMesh/boundary"} <= paths, sorted(paths)


# ------------------------------------- the round-off scale is not the sums judged (round 2)
def _set_last(case: Path, values: dict[str, tuple[str, str, str]]) -> None:
    """The rows at the log's last time of the patches in `values`, as (min, max, integral)."""
    _rows_edit(case, lambda f, at_last: [*f[:2], *values[f[1]]]
               if at_last and f[1] in values else f)


def test_a_region_whose_heat_crosses_one_patch_both_ways_is_not_judged(tmp_path, monkeypatch):
    """The review of round 2: a region whose heat enters one part of a patch and leaves another
    (a passive solid against a fluid) has that patch's net, the solve's residual, as its only
    integral, so |net| over the heat in read is 1 whatever the solve's accuracy. hot_box
    `converged`, its last rows edited so: `hot` from -120 to 120 W/m^2 with 3e-7 W net, the other
    walls 0. Not judged, with a sentence naming `hot`; verdict, rule and grade as with no output."""
    name = "hot_box-converged"
    case = _solve(tmp_path / "with", name)
    bare = _strip_heat_output(_solve(tmp_path / "without", name))
    zero = ("0.0000000000e+00",) * 3
    _set_last(case, {"hot": ("-1.2000000000e+02", "1.2000000000e+02", "3.0000000000e-07"),
                     "cold": zero, "top": zero, "bottom": zero})
    assert _read_heat(case).measures == {}
    doc, r = _judge(case, monkeypatch)
    without, _ = _judge(bare, monkeypatch)
    assert r is None or r.skipped, (r.severity, r.detail)
    assert not [x for x in doc["reasons"] if x["code"] == CODE], doc["reasons"]
    said = [s for s in doc["not_assessed"] if s not in without["not_assessed"]
            and "heat" in s and "hot" in s and "both in and out" in s and "not judged" in s]
    assert said, doc["not_assessed"]
    assert (doc["verdict"], doc["rule"], doc["grade"]) == (
        without["verdict"], without["rule"], without["grade"])


def test_an_interface_whose_heat_crosses_both_ways_is_not_judged(tmp_path, monkeypatch):
    """The review of round 2: cht_plate `converged` with its heater off and the interface
    carrying heat both ways on each side, nets 3e-7 and -2.9e-7 W (the pair measure 0.033 as
    read). Neither the plate nor the pair is judged; each sentence names its patches."""
    name = "cht_plate-converged"
    case = _solve(tmp_path / "with", name)
    bare = _strip_heat_output(_solve(tmp_path / "without", name))
    _set_last(case, {
        "heater": ("0.0000000000e+00",) * 3,
        "plate_to_air": ("-2.4432271189e+03", "2.9773801832e+02", "3.0000000000e-07"),
        "air_to_plate": ("-2.9773801832e+02", "2.4432271189e+03", "-2.9000000000e-07")})
    reading = _read_heat(case)
    assert reading.measures == {}, reading.measures
    assert [s for s in reading.sentences if "both in and out" in s and "plate_to_air" in s
            and "air_to_plate" in s and "interface" in s], reading.sentences
    assert [s for s in reading.sentences if "both in and out" in s and "plate_to_air" in s
            and "region `plate`" in s], reading.sentences
    doc, r = _judge(case, monkeypatch)
    without, _ = _judge(bare, monkeypatch)
    assert r is None or r.skipped, (r.severity, r.detail)
    assert not [x for x in doc["reasons"] if x["code"] == CODE], doc["reasons"]
    assert (doc["verdict"], doc["rule"], doc["grade"]) == (
        without["verdict"], without["rule"], without["grade"])


def test_an_interface_whose_sides_are_round_off_is_not_judged(tmp_path):
    """The review of round 2: the pair is held to the boundary balance's round-off rule
    (`MassFluxCheck.denominator`), not an exact zero. cht_plate `converged` with the interface's
    integrals 1e-13 W next to the heater's 0.16 W: no heat crosses the pair; the plate is still
    judged."""
    name = "cht_plate-converged"
    case = _solve(tmp_path, name)
    _set_last(case, {
        "plate_to_air": ("-2.4432271189e+03", "-2.9773801832e+02", "-1.0000000000e-13"),
        "air_to_plate": ("2.9773801832e+02", "2.4432271189e+03", "1.2000000000e-13")})
    reading = _read_heat(case)
    assert {_key(k) for k in reading.measures} == {"plate"}, reading.measures
    assert [s for s in reading.sentences if "no heat crosses the interface" in s
            and "round-off" in s], reading.sentences


def test_a_two_way_patch_leaves_a_balance_graded_ok_standing(tmp_path, monkeypatch):
    """The review of round 2: a patch carrying heat both ways makes the heat in read less than
    the heat that crosses, so the measure read is the larger: one graded OK stands. hot_box
    `converged` with `hot`'s min edited below zero: judged, OK, on the transcribed rows."""
    name = "hot_box-converged"
    case = _solve(tmp_path, name)
    _, rows = _transcribed(name)
    _rows_edit(case, _at_last("hot", lambda f: [*f[:2], "-1.0000000000e+00", *f[3:]]))
    m = _region_measure([float(v) for v in rows[DEFAULT_REGION].values()])
    reading = _read_heat(case)
    assert {_key(k) for k in reading.measures} == {DEFAULT_REGION}, reading.sentences
    _, r = _judge(case, monkeypatch)
    assert r is not None and r.severity is VSeverity.OK, r
    assert r.value == pytest.approx(m, rel=1e-9)


# --------------------------------------------------------- the review of round 3 (2 Nov 2026)
def test_a_wall_patch_with_no_faces_leaves_the_two_way_guard_standing(tmp_path, monkeypatch):
    """The review of round 3 (blocking): v2406's wallHeatFlux writes a row for a wall patch with
    no faces, its gMin and gMax those of an empty field, `1e+300 -1e+300 0`. That row is never a
    scale: the round-2 two-way case with one such wall added is still not judged."""
    name = "hot_box-converged"
    case = _solve(tmp_path / "with", name)
    bare = _strip_heat_output(_solve(tmp_path / "without", name))
    for c in (case, bare):
        boundary = _boundary_of(c, DEFAULT_REGION)
        boundary.write_text(_with_wall(boundary.read_text(), "emptyWall"))
    zero = ("0.0000000000e+00",) * 3
    _set_last(case, {"hot": ("-1.2000000000e+02", "1.2000000000e+02", "3.0000000000e-07"),
                     "cold": zero, "top": zero, "bottom": zero})
    (rel,) = _heat_files(case)
    last = _last_time(next(case.glob("log.*Foam")).read_text())
    path = case / rel
    path.write_text(path.read_text().rstrip("\n")
                    + f"\n{last}\temptyWall\t1.0000000000e+300\t-1.0000000000e+300\t"
                      f"0.0000000000e+00\n")
    assert _read_heat(case).measures == {}
    doc, r = _judge(case, monkeypatch)
    without, _ = _judge(bare, monkeypatch)
    assert r is None or r.skipped, (r.severity, r.detail)
    assert [s for s in doc["not_assessed"] if "hot" in s and "both in and out" in s]
    assert (doc["verdict"], doc["rule"], doc["grade"]) == (
        without["verdict"], without["rule"], without["grade"])


def _boundary_sub(case: Path, region: str, pattern: str, repl: str) -> None:
    """`_sub` on a region's boundary file, unpacked first where it is committed packed."""
    path = _boundary_of(case, region)
    packed = Path(str(path) + ".gz")
    if not path.is_file() and packed.is_file():
        path.write_bytes(gzip.decompress(packed.read_bytes()))
        packed.unlink()
    _sub(path, pattern, repl)


def test_an_interface_named_by_a_couple_group_is_said_not_judged(tmp_path):
    """The review of round 3: v2406's mappedPatchBase may name the other side by `coupleGroup`
    alone. cht_plate `converged` with both sides written so: the pair is not judged, and a
    sentence names it; the plate is still judged."""
    case = _solve(tmp_path, "cht_plate-converged")
    for region in ("plate", "air"):
        _boundary_sub(case, region, r"^\s*sampleRegion\s+\w+;\s*\n\s*samplePatch\s+\w+;",
                      "        coupleGroup     plate_air;")
    reading = _read_heat(case)
    assert {_key(k) for k in reading.measures} == {"plate"}, reading.measures
    said = [s for s in reading.sentences if "interface" in s and "coupleGroup" in s
            and "not judged" in s]
    assert said and ("plate_to_air" in said[0] or "air_to_plate" in said[0]), reading.sentences


def _cht_transient(case: Path) -> None:
    """cht_plate as chtMultiRegionFoam ran it, Euler in each region's schemes, and the case's own
    system/fvSchemes, which that solver never reads, saying steadyState."""
    _sub(case / "system" / "controlDict", r"^(\s*application\s+)chtMultiRegionSimpleFoam",
         r"\1chtMultiRegionFoam")
    _sub(next(case.glob("log.*Foam")), r"^(Exec\s*:\s*)chtMultiRegionSimpleFoam",
         r"\1chtMultiRegionFoam")
    for region in ("plate", "air"):
        _sub(case / "system" / region / "fvSchemes", r"^(\s*default\s+)steadyState", r"\1Euler")
    _sub(case / "system" / "fvSchemes", r"^ddtSchemes \{\}", "ddtSchemes { default steadyState; }")


def test_a_multi_region_run_is_steady_by_its_regions_schemes_not_the_cases(tmp_path, monkeypatch):
    """The review of round 3: a multi-region solver reads `system/<region>/fvSchemes`, never the
    case's own. cht_plate `early` (the plate still storing heat) as a transient run whose unread
    top-level file says steadyState: not judged, the sentence about stored heat; verdict, rule
    and grade as with no output."""
    name = "cht_plate-early"
    case = _solve(tmp_path / "with", name)
    bare = _strip_heat_output(_solve(tmp_path / "without", name))
    _cht_transient(case)
    _cht_transient(bare)
    doc, r = _judge(case, monkeypatch)
    without, _ = _judge(bare, monkeypatch)
    assert r is None or r.skipped, (r.severity, r.detail)
    assert [s for s in doc["not_assessed"] if s not in without["not_assessed"]
            and "heat" in s and "stored" in s and "chtMultiRegionFoam" in s], doc["not_assessed"]
    assert (doc["verdict"], doc["rule"], doc["grade"]) == (
        without["verdict"], without["rule"], without["grade"])


def test_a_region_declaring_a_time_scheme_is_not_judged_under_a_steady_solver(tmp_path):
    """The review of round 3: chtMultiRegionSimpleFoam with `system/plate/fvSchemes` declaring
    Euler: the region's file decides, so not judged, the sentence naming that file."""
    case = _solve(tmp_path, "cht_plate-converged")
    _sub(case / "system" / "plate" / "fvSchemes", r"^(\s*default\s+)steadyState", r"\1Euler")
    reading = _read_heat(case)
    assert reading.measures == {}, reading.measures
    assert [s for s in reading.sentences if "system/plate/fvSchemes" in s and "Euler" in s
            and "not judged" in s], reading.sentences


def test_the_cases_own_schemes_do_not_decide_a_multi_region_run(tmp_path):
    """The review of round 3, the other way: the top-level file says Euler and every region
    steadyState under chtMultiRegionSimpleFoam: judged as the unedited solve is."""
    case = _solve(tmp_path, "cht_plate-converged")
    _sub(case / "system" / "fvSchemes", r"^ddtSchemes \{\}", "ddtSchemes { default Euler; }")
    assert {_key(k) for k in _read_heat(case).measures} == JUDGED["cht_plate"]


def test_an_anisotropic_solid_is_not_judged(tmp_path, monkeypatch):
    """The review of round 3: v2406's wallHeatFlux reports a solid's scalar conductivity, and a
    `constAnIso` solid is solved with a tensor one. cht_plate `converged` with the plate so: not
    judged, the sentence naming its thermophysicalProperties."""
    case = _solve(tmp_path, "cht_plate-converged")
    _sub(case / "constant" / "plate" / "thermophysicalProperties",
         r"^(\s*transport\s+)constIso;", r"\1constAnIso;")
    reading = _read_heat(case)
    assert reading.measures == {}, reading.measures
    assert [s for s in reading.sentences if "constant/plate/thermophysicalProperties" in s
            and "anisotropic" in s], reading.sentences


@pytest.mark.parametrize("where", ["postProcessing/plate/wallHeatFlux_plate/0",
                                   "constant/plate", "constant/air"])
def test_a_directory_listed_and_not_searched_is_said_never_a_traceback(tmp_path, where):
    """The review of round 3: a directory of the heat output or of a region's mesh with mode 644
    (listed, not searched) gives sentences, never a traceback, under `--checks next`."""
    import os

    from courant.verify import from_case_dir, verify_inputs

    case = _solve(tmp_path, "cht_plate-converged")
    target = case / where
    os.chmod(target, 0o644)
    try:
        doc = json.loads(verify_inputs(from_case_dir(case, checks="next")).to_json())
    finally:
        os.chmod(target, 0o755)
    assert not [x for x in doc["reasons"] if x["code"] == CODE
                and x["severity"] != "ok"], doc["reasons"]
    region = "air" if where == "constant/air" else "plate"
    said = [s for s in doc["not_assessed"] if "heat" in s and region in s]
    assert said, doc["not_assessed"]
    if where == "constant/air":                 # `_boundary`'s own sentence (the re-review)
        assert [s for s in said if "which mesh the solver read for region `air` is not known"
                in s and "could not be searched" in s], said


def test_the_heat_tables_are_read_to_one_bound(tmp_path, monkeypatch):
    """The review of round 3 (minor): the tables are read under the bound every postProcessing
    table is (`courant.converge.series.MAX_BYTES`), decompressed, one budget for the reading.
    hot_box `converged` with the bound set below its table: not judged, said, no traceback."""
    from courant.converge import series

    case = _solve(tmp_path, "hot_box-converged")
    (rel,) = _heat_files(case)
    monkeypatch.setattr(series, "MAX_BYTES", (case / rel).stat().st_size // 2)
    reading = _read_heat(case)
    assert reading.measures == {}, reading.measures
    assert [s for s in reading.sentences if "MiB" in s and "not judged" in s], reading.sentences


def test_a_one_region_anisotropic_solid_is_not_judged(tmp_path):
    """The re-review of 3 Nov 2026: a one-region case (a solidFoam run) whose
    `constant/thermophysicalProperties` names an anisotropic transport is refused as a region of
    a multi-region case is. hot_box `converged` with the cht plate's solid thermo as its own:
    judged with `constIso`; with `constAnIso`, not judged, said."""
    case = _solve(tmp_path, "hot_box-converged")
    solid = (REAL / "cht_plate" / "converged" / "constant" / "plate"
             / "thermophysicalProperties").read_text()
    target = case / "constant" / "thermophysicalProperties"
    target.write_text(solid)
    assert {_key(k) for k in _read_heat(case).measures} == {DEFAULT_REGION}
    _sub(target, r"^(\s*transport\s+)constIso;", r"\1constAnIso;")
    reading = _read_heat(case)
    assert reading.measures == {}, reading.measures
    assert [s for s in reading.sentences if "constant/thermophysicalProperties" in s
            and "anisotropic" in s], reading.sentences

