"""TRD VER-22 (c): the heat a flow carries through a region's openings (T-ENERGY-FLOW).

Under `courant verify --checks next` a steady region open to flow is judged on the heat through
its walls (the `wallHeatFlux` rows, VER-22 (a)) plus the heat carried through each opening: by
the flow, `-phi_f Cp (T_f - T_ref)`, and by conduction, `(mu/Pr) Cp dc_f (T_f - T_P) |Sf|`, as
v2406's energy equation makes them on a boundary face. Shown on this repository's own
chtMultiRegionSimpleFoam solve, `cht_plate`, whose `air` region is a channel with an `inlet` and
an `outlet`.

Every case is a committed solve copied into `tmp_path` (TST-4). The oracle is computed here from
the solver's own files (`T`, `phi`, the mesh's points, faces and owner, the `wallHeatFlux` rows)
with this file's own parsing and arithmetic, never a second Courant reading. Nothing is mocked:
the reader and the check run on the solver's output.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest

from courant.verification.checks import MassFluxCheck
from courant.verification.grade import VSeverity
from tests.unit.test_energy_balance import CODE, INTERFACE, ROWS, _judge, _key, _read_heat, _solve

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("cht_plate-converged", "cht_plate-early")
REGION = "air"
OPENINGS = ("inlet", "outlet")
#: The `air` region's thermo, as `constant/air/thermophysicalProperties` writes it (hConst,
#: perfectGas, const transport, sensibleEnthalpy), transcribed literally.
CP, MU, PR = 1005.0, 1.8e-05, 0.7


# --------------------------------------------------------------------- the oracle's parsing
def _text(path: Path) -> str:
    gz = path.with_name(path.name + ".gz")
    if path.is_file():
        return path.read_text()
    with gzip.open(gz, "rt") as fh:
        return fh.read()


def _list_after(text: str, start: int = 0) -> tuple[str, int]:
    """The contents of the first `N ( ... )` list at or after `start`, and where it ends."""
    m = re.compile(r"(\d+)\s*\(").search(text, start)
    assert m, "no list"
    depth, i = 1, m.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1], i


def _body(path: Path) -> str:
    text = re.sub(r"/\*.*?\*/", "", _text(path), flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text[text.index("}", text.index("FoamFile")) + 1:]


def _mesh(case: Path):
    d = case / "constant" / REGION / "polyMesh"
    pts = np.array([[float(x) for x in t.split()]
                    for t in re.findall(r"\(([^()]*)\)", _list_after(_body(d / "points"))[0])])
    faces = [[int(x) for x in t.split()]
             for t in re.findall(r"\d+\s*\(([^()]*)\)", _list_after(_body(d / "faces"))[0])]
    owner = [int(x) for x in _list_after(_body(d / "owner"))[0].split()]
    neigh = [int(x) for x in _list_after(_body(d / "neighbour"))[0].split()]
    patches = {}
    for name, block in re.findall(r"(\w+)\s*\{([^}]*)\}", _list_after(_body(d / "boundary"))[0]):
        start = int(re.search(r"startFace\s+(\d+)", block).group(1))
        n = int(re.search(r"nFaces\s+(\d+)", block).group(1))
        patches[name] = range(start, start + n)
    return pts, faces, owner, neigh, patches


def _cell_points(faces, owner, neigh, cell: int) -> set[int]:
    out: set[int] = set()
    for f, o in enumerate(owner):
        if o == cell or (f < len(neigh) and neigh[f] == cell):
            out.update(faces[f])
    return out


def _field(case: Path, time: str, name: str) -> tuple[str, dict[str, str]]:
    """(the internal field's text, {patch: its entry's text}) of `<time>/air/<name>`."""
    text = _body(case / time / REGION / name)
    k = text.index("boundaryField")
    internal = text[text.index("internalField"):k]
    entries = dict(re.findall(r"\n    (\w+)\n    \{(.*?)\n    \}", text[k:], flags=re.DOTALL))
    return internal, entries


def _values(raw: str, n: int) -> np.ndarray:
    m = re.search(r"uniform\s+([-+\deE.]+)\s*;", raw)
    if m and "nonuniform" not in raw:
        return np.full(n, float(m.group(1)))
    return np.array([float(x) for x in _list_after(raw, raw.index("nonuniform"))[0].split()])


def _vectors(raw: str, n: int) -> np.ndarray:
    m = re.search(r"uniform\s+\(([^()]*)\)\s*;", raw)
    if m and "nonuniform" not in raw:
        return np.tile([float(x) for x in m.group(1).split()], (n, 1))
    inner = _list_after(raw, raw.index("nonuniform"))[0]
    return np.array([[float(x) for x in v.split()] for v in re.findall(r"\(([^()]*)\)", inner)])


def _gravity(case: Path) -> np.ndarray:
    raw = re.search(r"value\s*\(([^()]*)\)", _body(case / "constant" / "g")).group(1)
    return np.array([float(x) for x in raw.split()])


def _rows(name: str) -> dict[str, float]:
    return {p: float(v) for p, v in ROWS[name][1][REGION].items()}


def _oracle(case: Path, name: str, *, conduction: bool = True) -> float:
    """The `air` region's measure from the solver's own files: |net| over the heat in, every
    face's energy (`Cp T + |U|^2/2 - g . Cf`) relative to the inflow's mass-weighted one; an opening's heat in summed face by
    face (its faces are read), a wall's from its row."""
    time = ROWS[name][0]
    pts, faces, owner, neigh, patches = _mesh(case)
    internal, t_entries = _field(case, time, "T")
    _, phi_entries = _field(case, time, "phi")
    u_internal, u_entries = _field(case, time, "U")
    n_cells = max(max(owner), max(neigh)) + 1
    t_cells = _values(internal, n_cells)
    u_cells = _vectors(u_internal, n_cells)
    g = _gravity(case)
    walls = _rows(name)
    net = sum(walls.values())
    heat_in = sum(v for v in walls.values() if v > 0)
    per_face = {}
    for patch in OPENINGS:
        idx = list(patches[patch])
        phi = _values(phi_entries[patch], len(idx))
        cells = [owner[f] for f in idx]
        # a face's own value, or (zeroGradient, which writes none) its cell's
        t_face = (_values(t_entries[patch], len(idx)) if "value" in t_entries[patch]
                  else t_cells[cells])
        u_face = (_vectors(u_entries[patch], len(idx)) if "value" in u_entries[patch]
                  else u_cells[cells])
        centre = np.array([pts[faces[f]].mean(axis=0) for f in idx])   # planar quads of a box
        e_face = CP * t_face + 0.5 * (u_face ** 2).sum(axis=1) - centre @ g
        per_face[patch] = (idx, phi, t_face, e_face)
    inflow = [(p, e) for _, phi, _, ef in per_face.values() for p, e in zip(phi, ef) if p < 0]
    e_ref = sum(p * e for p, e in inflow) / sum(p for p, _ in inflow)
    for idx, phi, t_face, e_face in per_face.values():
        for f, p, tf, ef in zip(idx, phi, t_face, e_face):
            fp = pts[faces[f]]
            centre = fp.mean(axis=0)                    # a planar quadrilateral of a box mesh
            area = 0.5 * np.cross(fp[2] - fp[0], fp[3] - fp[1])
            cell = pts[sorted(_cell_points(faces, owner, neigh, owner[f]))].mean(axis=0)
            n_hat = area / np.linalg.norm(area)
            dc = 1.0 / float(n_hat @ (centre - cell))
            q = -p * (ef - e_ref)
            if conduction:
                q += (MU / PR) * CP * dc * (tf - t_cells[owner[f]]) * np.linalg.norm(area)
            net += q
            heat_in += max(q, 0.0)
    return abs(net) / heat_in


def _air_sentences(doc_or_reading) -> list[str]:
    said = (doc_or_reading["not_assessed"] if isinstance(doc_or_reading, dict)
            else doc_or_reading.sentences)
    return [s for s in said if "`air`" in s]


def _bare(sentence: str) -> list[str]:
    """A case-authored name of this fixture written outside backticks (`_heat_reader._q`)."""
    return re.findall(r"\b(?:inlet|outlet|air|plate)\b", re.sub(r"`[^`]*`", "", sentence))


# ------------------------------------------------------------------------------- C1: judged
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize("name", NAMES)
def test_an_open_region_is_judged_on_its_walls_and_the_heat_its_flow_carries(tmp_path, name,
                                                                            packed):
    """C1. The `air` region is judged: its measure is the oracle's, from the solver's own `T`,
    `phi`, mesh and rows, packed and unpacked; on the converged solve it is within TARGET."""
    unpacked = _solve(tmp_path / "unpacked", name)
    case = _solve(tmp_path / "case", name, packed=packed)
    reading = _read_heat(case, unpacked)
    measures = {_key(k): m for k, m in reading.measures.items()}
    assert set(measures) == {"plate", REGION, INTERFACE}, (set(measures), reading.sentences)
    # abs: the converged measure is a cancellation, 2e-9 of the heat in
    assert measures[REGION] == pytest.approx(_oracle(unpacked, name), rel=1e-9, abs=1e-12)
    assert not _air_sentences(reading), reading.sentences
    if name == "cht_plate-converged":
        assert measures[REGION] <= MassFluxCheck.TARGET, measures[REGION]
        # Every term read, the converged channel closes to 2.1e-9 of its heat in; without the
        # kinetic energy and buoyancy work (or with the buoyancy work's sign flipped) 1.3e-5.
        assert measures[REGION] < 1e-7, measures[REGION]


def test_the_conduction_through_the_inlet_is_part_of_the_balance(tmp_path):
    """C2. Without the conduction through the openings the converged run would be out by twice
    TARGET or more (the planner's 2.1e-3 with the thermo's datum, 3.5e-3 against the inflow); with
    it, the reading is within TARGET. So the conductive part is read, not left out."""
    name = "cht_plate-converged"
    case = _solve(tmp_path, name)
    without = _oracle(case, name, conduction=False)
    assert without >= 2 * MassFluxCheck.TARGET, without
    m = {_key(k): v for k, v in _read_heat(case).measures.items()}[REGION]
    assert m <= MassFluxCheck.TARGET < without


def _warm_outlet(case: Path, by: float) -> None:
    path = case / ROWS["cht_plate-converged"][0] / REGION / "T"
    text = path.read_text()
    k = text.index("boundaryField")
    head, tail = text[:k], text[k:]
    m = re.search(r"(\n    outlet\n    \{.*?value\s+nonuniform List<scalar>\s*\n?\d+\s*\n?\()(.*?)(\))",
                  tail, flags=re.DOTALL)
    assert m, "the outlet entry holds no nonuniform value"
    warmer = "\n".join(repr(float(x) + by) for x in m.group(2).split())
    path.write_text(head + tail[:m.start(2)] + "\n" + warmer + "\n" + tail[m.end(2):])


def test_an_outlet_carrying_more_heat_than_the_walls_gave_is_rejected(tmp_path, monkeypatch):
    """C2. Every face of `outlet` written 1 K warmer: the flow carries out far more heat than the
    walls took in. The air's measure is the oracle's, MAJOR, and the verdict's reason cites the
    value, the bar, the region and the files it was read from, each name between backticks."""
    name = "cht_plate-converged"
    case = _solve(tmp_path, name)
    _warm_outlet(case, 1.0)
    expected = _oracle(case, name)
    assert expected > MassFluxCheck.LOOSE, expected
    m = {_key(k): v for k, v in _read_heat(case).measures.items()}[REGION]
    assert m == pytest.approx(expected, rel=1e-9)
    doc, result = _judge(case, monkeypatch)
    assert result is not None and result.severity is VSeverity.MAJOR
    assert result.value == pytest.approx(expected, rel=1e-9)
    assert doc["verdict"] == "reject", doc["verdict"]
    (reason,) = [r for r in doc["reasons"] if r["code"] == CODE]
    assert float(reason["threshold"]) == MassFluxCheck.LOOSE
    text = f"{reason['summary']} {reason['input']}"
    for word in ("`air`", "`inlet`", "`outlet`", f"{ROWS[name][0]}/air/phi",
                 f"{ROWS[name][0]}/air/T", "wallHeatFlux"):
        assert word in text, (word, text)
    assert not _bare(reason["summary"]), reason["summary"]


# ----------------------------------------------------------------- C3: not judged, and why
def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, (old, path)
    path.write_text(text.replace(old, new, 1))


def _thermo(case: Path) -> Path:
    return case / "constant" / REGION / "thermophysicalProperties"


def _edit(case: Path, what: str) -> None:
    time = ROWS["cht_plate-converged"][0]
    if what == "turbulent":
        _replace(case / "constant" / REGION / "turbulenceProperties",
                 "simulationType  laminar;", "simulationType  RAS;\nRAS { RASModel kEpsilon; }")
    elif what == "transport":
        _replace(_thermo(case), "transport       const;", "transport       sutherland;")
    elif what == "thermo":
        _replace(_thermo(case), "thermo          hConst;", "thermo          janaf;")
    elif what == "energy":
        _replace(_thermo(case), "energy          sensibleEnthalpy;",
                 "energy          sensibleInternalEnergy;")
    elif what == "eos":
        _replace(_thermo(case), "equationOfState perfectGas;",
                 "equationOfState incompressiblePerfectGas;")
    elif what == "directive":
        _replace(_thermo(case), "thermoType\n{", '#includeEtc "caseDicts/thermo"\n\nthermoType\n{')
    elif what == "no_phi":
        (case / time / REGION / "phi").unlink()
    elif what == "volume_flux":
        _replace(case / time / REGION / "phi", "[1 0 -1 0 0 0 0]", "[0 3 -1 0 0 0 0]")
    elif what == "t_no_value":
        path = case / time / REGION / "T"
        text = path.read_text()
        m = re.search(r"\n    outlet\n    \{.*?\n    \}", text, flags=re.DOTALL)
        path.write_text(text[:m.start()] + "\n    outlet\n    {\n        type            "
                        "fixedGradient;\n        gradient        uniform 0;\n    }"
                        + text[m.end():])
    elif what == "too_many_cells":
        # one owner label far past the face count: case text that would size the cell arrays
        path = case / "constant" / REGION / "polyMesh" / "owner"
        text = path.read_text()
        body, _ = _list_after(text, text.index("}") + 1)
        first = re.search(r"\d+", body)
        start = text.index(body)
        edited = text[:start + first.start()] + "999999999" + text[start + first.end():]
        path.write_text(edited.replace("nCells:", "cells:"))    # as an owner.gz is read: no note
    elif what == "geometry":
        _replace(case / "system" / REGION / "fvSchemes", "ddtSchemes",
                 "geometry\n{\n    type            averageNeighbour;\n}\n\nddtSchemes")
    elif what == "no_u":
        (case / time / REGION / "U").unlink()
    elif what == "no_g":
        (case / "constant" / "g").unlink()
    elif what == "region_g":
        (case / "constant" / REGION / "g").write_text((case / "constant" / "g").read_text())
    elif what == "faces_past_mesh":
        path = case / "constant" / REGION / "polyMesh" / "boundary"
        text = path.read_text()
        m = re.search(r"(outlet\s*\{[^}]*startFace\s+)(\d+)", text)
        path.write_text(text[:m.start(2)] + "999999" + text[m.end(2):])
    elif what == "mesh_unreadable":
        (case / "constant" / REGION / "polyMesh" / "points").write_text("not a mesh\n")
    else:
        raise AssertionError(what)


#: Each edit and the words its sentence says (literal), beside `air`.
NOT_JUDGED = {
    "turbulent": ("laminar", "`constant/air/turbulenceProperties`"),
    "transport": ("const", "`constant/air/thermophysicalProperties`"),
    "thermo": ("hConst",),
    "energy": ("sensibleEnthalpy", "sensibleInternalEnergy"),
    "eos": ("`incompressiblePerfectGas`", "perfectGas"),
    "directive": ("read whole",),
    "no_phi": ("phi",),
    "volume_flux": ("mass flux",),
    "t_no_value": ("`outlet`", "T"),
    "mesh_unreadable": ("mesh",),
    "too_many_cells": ("more cells than it has faces",),
    "geometry": ("`averageNeighbour`", "`system/air/fvSchemes`"),
    "no_u": ("`3000/air/U`",),
    "no_g": ("`constant/g`",),
    "region_g": ("`constant/air/g`", "`constant/g`"),
    "faces_past_mesh": ("`outlet`", "not boundary faces"),
}


@pytest.mark.parametrize("what", sorted(NOT_JUDGED))
def test_what_the_flow_reading_cannot_read_is_said(tmp_path, monkeypatch, what):
    """C3. On a copy of the converged solve, each edit leaves `air` not judged with a sentence
    that says why (its words literal, every case name between backticks), never a traceback;
    the `plate` region and the interface are judged as before."""
    name = "cht_plate-converged"
    clean = _solve(tmp_path / "clean", name)
    before = {_key(k): v for k, v in _read_heat(clean).measures.items()}
    case = _solve(tmp_path / "case", name)
    _edit(case, what)
    reading = _read_heat(case, clean)
    got = {_key(k): v for k, v in reading.measures.items()}
    assert set(got) == {"plate", INTERFACE}, (got, reading.sentences)
    assert got["plate"] == before["plate"] and got[INTERFACE] == before[INTERFACE]
    said = _air_sentences(reading)
    assert len(said) == 1, reading.sentences
    for word in NOT_JUDGED[what]:
        assert word in said[0], (word, said[0])
    assert "not judged" in said[0], said[0]
    assert not _bare(said[0]), said[0]
    doc, _ = _judge(case, monkeypatch)                  # the whole verdict: no traceback
    assert said[0] in doc["not_assessed"], doc["not_assessed"]


# --------------------------------------------------------------------------- C4: GATE-13
def test_with_the_flow_name_off_the_open_region_keeps_070s_sentence(tmp_path, monkeypatch):
    """C4. `conservation.energy.flow` is its own name behind the flag: with it off and the wall
    balance on, the `air` region is not judged and says v0.7.0's sentence, byte for byte."""
    from courant.verify import checks

    assert checks.NEXT.get(checks.ENERGY_FLOW) == "0.8.0"
    real = checks.enabled
    monkeypatch.setattr(checks, "enabled",
                        lambda name, mode="default": name != checks.ENERGY_FLOW
                        and real(name, mode))
    case = _solve(tmp_path, "cht_plate-converged")
    reading = _read_heat(case)
    assert {_key(k) for k in reading.measures} == {"plate", INTERFACE}
    assert _air_sentences(reading) == [(
        "Net wall heat flux ~ 0: region `air` is open to flow through `inlet` and `outlet`: the "
        "heat the flow carries through them is not read, so its heat balance is not judged (only "
        "the heat through its walls was read)")]


def test_the_default_verdict_never_reads_the_flow(tmp_path, monkeypatch):
    """C4. Under the default the flow reader is never called: made to fail, the document is the
    same."""
    import json

    from courant.verify import from_case_dir, verify_inputs

    case = _solve(tmp_path, "cht_plate-converged")
    before = verify_inputs(from_case_dir(case)).to_json()
    from courant.verification import _flow_heat

    def boom(*a, **k):
        raise AssertionError("the flow reader was called under the default")

    monkeypatch.setattr(_flow_heat, "read_openings", boom)
    assert verify_inputs(from_case_dir(case)).to_json() == before
    assert CODE not in {r["code"] for r in json.loads(before)["reasons"]}


# ------------------------------------------------------------------------- C5: one fact
def test_the_flow_reader_holds_no_bar_of_its_own():
    """C5. No numeric literal 1e-3, 1e-2 or 1e-9 in the flow reader: the bars and the round-off
    are MassFluxCheck's, by reference."""
    import inspect

    assert importlib.util.find_spec("courant.verification._flow_heat") is not None, (
        "there is no flow reader (courant/verification/_flow_heat.py)")
    from courant.verification import _flow_heat

    numbers = [t.string for t in tokenize.generate_tokens(
        io.StringIO(inspect.getsource(_flow_heat)).readline) if t.type == tokenize.NUMBER]
    assert not [n for n in numbers if float(n) in (1e-3, 1e-2, 1e-9)], numbers


def test_the_open_region_is_held_to_the_mass_balances_bars(tmp_path, monkeypatch):
    """C5. `MassFluxCheck.TARGET` moved below the converged air's measure moves its grade: the
    bar is read at call time, by reference."""
    case = _solve(tmp_path, "cht_plate-converged")
    m = {_key(k): v for k, v in _read_heat(case).measures.items()}[REGION]
    assert m > 0
    monkeypatch.setattr(MassFluxCheck, "TARGET", m / 2)
    monkeypatch.setattr(MassFluxCheck, "LOOSE", m / 2)
    _, result = _judge(case, monkeypatch)
    assert result is not None and result.severity is VSeverity.MAJOR, result
    assert result.value == pytest.approx(m, rel=1e-12)


# ------------------------------------------------------- the branches a review found untested
def test_a_zero_gradient_opening_is_read_at_its_cells_values(tmp_path):
    """An opening whose `T` is `zeroGradient` writes no value: its faces carry their cells' `T`
    (and no conduction). `outlet` written so on a copy of the converged solve: the measure is the
    oracle's, which takes the cells' values there by its own reading."""
    name = "cht_plate-converged"
    case = _solve(tmp_path, name)
    path = case / ROWS[name][0] / REGION / "T"
    text = path.read_text()
    m = re.search(r"\n    outlet\n    \{.*?\n    \}", text, flags=re.DOTALL)
    path.write_text(text[:m.start()] + "\n    outlet\n    {\n        type            "
                    "zeroGradient;\n    }" + text[m.end():])
    reading = _read_heat(case)
    got = {_key(k): v for k, v in reading.measures.items()}
    assert REGION in got, reading.sentences
    assert got[REGION] == pytest.approx(_oracle(case, name), rel=1e-9, abs=1e-12)


def test_a_decomposed_region_is_said_not_judged(tmp_path):
    """A parallel run's region mesh is several rank directories: the openings are read from one
    mesh directory only, so the `air` region is not judged, and the sentence names the ranks."""
    import shutil

    from courant.verification._heat_reader import read_heat
    from courant.verify import from_case_dir
    from courant.verify.pipeline import _solver
    from tests.unit.test_energy_balance import _last_time, _sub

    case = _solve(tmp_path, "cht_plate-converged")
    log = next(case.glob("log.*Foam"))
    _sub(log, r"^(Exec\s*:\s*chtMultiRegionSimpleFoam)[ \t]*$", r"\1 -parallel")
    _sub(log, r"^(nProcs\s*:\s*)1[ \t]*$", r"\g<1>2")
    for rank in ("processor0", "processor1"):
        for region in ("air", "plate"):
            shutil.copytree(case / "constant" / region / "polyMesh",
                            case / rank / "constant" / region / "polyMesh")
    from courant.case.config import CaseConfig

    inputs = from_case_dir(case, checks="next")
    config = CaseConfig.from_view(inputs.config_view, read_here=True)
    reading = read_heat(str(case), float(_last_time(inputs.log_text)), config,
                        _solver(inputs, config), "next", log_text=inputs.log_text)
    assert REGION not in {_key(k) for k in reading.measures}, reading.measures
    said = _air_sentences(reading)
    assert len(said) == 1 and "ranks" in said[0] and "not judged" in said[0], reading.sentences
    assert "`processor0/constant/air/polyMesh/boundary`" in said[0], said[0]
