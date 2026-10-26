"""A laminar run heated from below that its own Rayleigh number calls turbulent, on a mesh that
cannot resolve it (TRD VER-17).

RRD RM-35 version 2 (development): 3 of 29 hotRoom `laminar_on_turbulent` catalogue twins were
accepted blind (draws 044, 083, 092). The first version of this check fired on them, using the
room's height as the length. Review (B1) showed that is no lower bound: hotRoom's only held
temperatures are on its floor (the 600 K heater beside 299.3 K floor faces), so no cold wall
lies above the heater and the room's height is not a distance the heater's temperature
difference acts across. The check now acts only on a case held hotter below than above, over
the distance between those walls, and stays silent on hotRoom. The 3 draws stay accepted blind.

The fixture `tests/fixtures/buoyant/hotRoom_laminar_044` is draw 044: its dictionaries as staged
(the floor's 400-face list as setFields writes it), its checkMesh log and an excerpt of its
solver log (the first 3 and the last 30 of its 1,000 steps). Its mesh is the hotRoom fixture.

**Behind `--checks next` until 0.6.0** (TRD GATE-13, S's decision of 15 Oct 2026): the check
(`physics.laminar_regime` in `courant.verify.checks.NEXT`) and the gate's reading of
`constant/g` run only under `next`, so every checker context and verdict here asks for it. The
reader itself (`CaseConfig.from_foam`) reads gravity unless told not to.
"""

import gzip
import math
import shutil
from pathlib import Path

import pytest

from courant.case.config import CaseConfig
from courant.case.mesh import MeshQuality, polymesh_quality
from courant.case.polymesh import FILES
from courant.case.regime import DNS_SPACING, R_UNIVERSAL, RA_TRANSITION, buoyant_regime
from courant.diagnosis.checkers import CheckContext, PhysicsChecker
from courant.diagnosis.findings import FindingSeverity

FIX = Path(__file__).resolve().parents[1] / "fixtures"
CASE = FIX / "buoyant" / "hotRoom_laminar_044"
MESHES = FIX / "real_openfoam" / "polymesh"
HOTROOM_MESH = MESHES / "buoyantPimpleFoam_hotRoom"

#: The line OpenFOAM's `readGravitationalAcceleration.H` prints, as draw 044's log has it.
READ_G = "Reading g\n\nReading hRef\nCalculating field g.h\n"

AIR = {"mu": 1.8e-5, "Pr": 0.7, "molWeight": 28.9}


def _unzip_mesh(src: Path, case: Path) -> Path:
    poly = case / "constant" / "polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    for n in FILES:
        with gzip.open(src / "constant" / "polyMesh" / f"{n}.gz", "rb") as s, \
                open(poly / n, "wb") as d:
            shutil.copyfileobj(s, d)
    return case


def _case_with_mesh(tmp: Path) -> Path:
    case = tmp / "hotRoom"
    shutil.copytree(CASE, case)
    return _unzip_mesh(HOTROOM_MESH, case)


def _box(lo, hi):
    return (tuple(float(x) for x in lo), tuple(float(x) for x in hi))


def _room(lx, ly, lz, cell, dirs=(1, 1, 1)):
    n = round(lx / cell) * round(ly / cell) * max(1, round(lz / cell) if dirs[2] else 1)
    return MeshQuality(n_cells=n, bounding_box=_box((0, 0, 0), (lx, ly, lz)),
                       total_volume=lx * ly * lz, solution_directions=dirs)


def _gas(t_hot_box, t_cold_box, t_cold=290.0, t_hot=320.0, g=(0.0, -9.81, 0.0), **kw):
    return CaseConfig(turbulence_type="laminar", gravity=g, transport=dict(AIR),
                      equation_of_state="perfectGas", pressure=1e5,
                      fixed_temperature_range=(t_cold, t_hot),
                      hot_faces_box=t_hot_box, cold_faces_box=t_cold_box, **kw)


def _rb_room():
    """10 x 5 x 10 m, 0.5 m cells, floor held at 320 K, ceiling at 290 K: heated from below."""
    floor, ceiling = _box((0.25, 0, 0.25), (9.75, 0, 9.75)), _box((0.25, 5, 0.25), (9.75, 5, 9.75))
    return _gas(floor, ceiling), _room(10, 5, 10, 0.5)


def _findings(config, mesh_view, log_text=READ_G, checks="next"):
    return [f for f in PhysicsChecker().run(CheckContext(run_id="r", events=[], config=config,
                                                         checks=checks,
                                                         mesh=mesh_view, log_text=log_text))
            if f.id == "physics.laminar_regime"]


# ------------------------------------------------------------------ what the solver read
def test_the_reader_reads_gravity_the_held_wall_temperatures_the_state_and_the_pressure(tmp_path):
    cfg = CaseConfig.from_foam(str(_case_with_mesh(tmp_path)))
    assert cfg.turbulence_type == "laminar"
    assert cfg.gravity == (0.0, -9.81, 0.0)
    # setFields' per-face list on a fixedValue wall, as the solver read it; the internal field and
    # the lumped-mass ceiling are not held temperatures.
    assert cfg.fixed_temperature_range == (299.3, 600.0)
    assert cfg.hot_faces_box == _box((4.75, 0, 4.75), (5.25, 0, 5.25))      # the 4 heater faces
    assert cfg.cold_faces_box == _box((0.25, 0, 0.25), (9.75, 0, 9.75))     # the rest of the floor
    assert cfg.equation_of_state == "perfectGas"
    assert cfg.pressure == 1e5
    assert cfg.transport["mu"] == 1.8e-5 and cfg.transport["molWeight"] == 28.9


def test_without_the_mesh_files_the_wall_positions_are_unknown():
    cfg = CaseConfig.from_foam(str(CASE))
    assert cfg.fixed_temperature_range == (299.3, 600.0)
    assert cfg.hot_faces_box is None and cfg.cold_faces_box is None


def test_only_wall_patches_hold_a_temperature(tmp_path):
    case = _case_with_mesh(tmp_path)
    boundary = case / "constant" / "polyMesh" / "boundary"
    text = boundary.read_text()
    i = text.index("ceiling")
    boundary.write_text(text[:i] + text[i:].replace("wall", "patch", 1))
    t = case / "0" / "T"
    t.write_text(t.read_text().replace("lumpedMassWallTemperature", "fixedValue").replace(
        "value           uniform 300;", "value           uniform 280;"))
    cfg = CaseConfig.from_foam(str(case))
    assert cfg.fixed_temperature_range == (299.3, 600.0)              # the 280 K ceiling: a patch
    assert cfg.cold_faces_box == _box((0.25, 0, 0.25), (9.75, 0, 9.75))


def test_an_unparseable_g_is_named_unreadable_and_an_unreadable_0_file_is_absent(tmp_path):
    case = tmp_path / "hotRoom"
    shutil.copytree(CASE, case)
    (case / "constant" / "g").write_text("value ( 0 -9.81 0 ;\n{")
    (case / "0" / "T").write_text("boundaryField { floor { value ( ; } ")
    cfg = CaseConfig.from_foam(str(case))
    assert cfg.gravity is None and "constant/g" in cfg.unreadable
    assert cfg.fixed_temperature_range is None and "0/T" not in cfg.unreadable


def test_a_case_with_no_g_and_no_temperature_reads_none():
    cfg = CaseConfig.from_foam(str(FIX / "mini_case"))
    assert cfg.gravity is None and cfg.fixed_temperature_range is None
    assert cfg.hot_faces_box is None and cfg.equation_of_state is None and cfg.pressure is None


def test_the_hotroom_extent_is_checkmeshs():
    mesh = MeshQuality.from_checkmesh((CASE / "log.checkMesh").read_text())
    assert mesh.bounding_box == ((0.0, 0.0, 0.0), (10.0, 5.0, 10.0))
    assert mesh.total_volume == 500.0
    assert mesh.solution_directions == (1, 1, 1)
    assert mesh.n_cells == 4000


@pytest.mark.parametrize("name", sorted(p.name for p in MESHES.iterdir() if p.is_dir()))
def test_the_extent_from_the_mesh_files_is_checkmeshs_own(tmp_path, name):
    """The five meshes blockMesh wrote in the pinned image, against checkMesh's log of the same
    container (VER-7a's fixtures), so the computation is checked against OpenFOAM's."""
    printed = MeshQuality.from_checkmesh((MESHES / name / "log.checkMesh").read_text())
    computed = polymesh_quality(str(_unzip_mesh(MESHES / name, tmp_path / name)))
    assert printed.bounding_box is not None and printed.total_volume is not None
    assert computed.solution_directions == printed.solution_directions
    for a, b in zip(computed.bounding_box, printed.bounding_box):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-9)
    assert computed.total_volume == pytest.approx(printed.total_volume, rel=1e-5)


# ------------------------------------------------------------------------ the estimate
def test_a_room_heated_from_below_gets_the_conservative_estimate():
    cfg, mesh = _rb_room()
    r = buoyant_regime(cfg, mesh)
    nu = AIR["mu"] / (1e5 * AIR["molWeight"] / (R_UNIVERSAL * 320.0))   # the lightest gas
    alpha = nu / AIR["Pr"]
    reduced = 9.81 * (1.0 - 290.0 / 320.0)              # the density contrast, not g*beta*dT
    ra = reduced * 5.0 ** 3 / (nu * alpha)
    assert r.height == pytest.approx(5.0) and r.span == pytest.approx(5.0)
    assert r.rayleigh == pytest.approx(ra)
    assert r.kolmogorov == pytest.approx(5.0 * math.sqrt(AIR["Pr"]) * ra ** -0.25)
    assert r.cell == pytest.approx(0.5)
    assert r.turbulent and r.unresolved


def test_the_length_is_the_least_separation_for_ra_and_the_greatest_for_eta():
    # Hot faces at heights 0 to 1 m, cold at 4 to 5 m: Ra over 3 m (a lower bound), the
    # Kolmogorov bound over 5 m (it grows with the length as H^(1/4), so the greatest bounds it).
    cfg = _gas(_box((0, 0, 0), (0, 1, 0)), _box((10, 4, 10), (10, 5, 10)))
    r = buoyant_regime(cfg, _room(10, 5, 10, 0.5))
    nu = AIR["mu"] / (1e5 * AIR["molWeight"] / (R_UNIVERSAL * 320.0))
    per_m3 = 9.81 * (1.0 - 290.0 / 320.0) / (nu * nu / AIR["Pr"])
    assert r.height == pytest.approx(3.0) and r.span == pytest.approx(5.0)
    assert r.rayleigh == pytest.approx(per_m3 * 27.0)
    assert r.kolmogorov == pytest.approx(5.0 * math.sqrt(AIR["Pr"]) * (per_m3 * 125.0) ** -0.25)


def test_hotroom_is_not_heated_from_below_a_cold_wall(tmp_path):
    """B1: the heater and the cold floor are at one height, so no estimate, and draw 044 keeps
    the verdict gate version 2 gave it."""
    cfg = CaseConfig.from_foam(str(_case_with_mesh(tmp_path)))
    mesh = MeshQuality.from_checkmesh((CASE / "log.checkMesh").read_text())
    assert buoyant_regime(cfg, mesh) is None


def test_a_room_heated_from_above_is_silent():
    # The reviewer's case: ceiling 320 K, floor 290 K, 4 x 3 x 4 m, 0.1 m cells.
    ceiling, floor = _box((0.05, 3, 0.05), (3.95, 3, 3.95)), _box((0.05, 0, 0.05), (3.95, 0, 3.95))
    mesh = _room(4, 3, 4, 0.1)
    assert buoyant_regime(_gas(ceiling, floor), mesh) is None
    assert _findings(_gas(ceiling, floor), mesh) == []


def test_a_small_heater_near_the_top_of_a_tall_box_is_silent():
    heater, floor = _box((0.9, 9.5, 0.9), (1.1, 9.5, 1.1)), _box((0.05, 0, 0.05), (1.95, 0, 1.95))
    mesh = _room(2, 10, 2, 0.5)
    assert buoyant_regime(_gas(heater, floor), mesh) is None
    assert _findings(_gas(heater, floor), mesh) == []


def test_tilted_gravity_projects_every_corner():
    g = (-9.81 / math.sqrt(2), -9.81 / math.sqrt(2), 0.0)
    cfg = _gas(_box((0, 0, 0), (0, 0, 0)), _box((1, 1, 0), (1, 1, 0)), g=g)
    r = buoyant_regime(cfg, _room(10, 5, 10, 0.5))
    assert r.height == pytest.approx(math.sqrt(2)) and r.span == pytest.approx(math.sqrt(2))


def test_gravity_along_an_empty_direction_moves_nothing():
    # A 2-D slab, z empty (one cell thick), the floor hot and the ceiling cold.
    floor, ceiling = _box((0.25, 0, 0.5), (9.75, 0, 0.5)), _box((0.25, 5, 0.5), (9.75, 5, 0.5))
    slab = _room(10, 5, 1, 0.5, dirs=(1, 1, 0))
    assert buoyant_regime(_gas(floor, ceiling, g=(0.0, 0.0, -9.81)), slab) is None
    both = buoyant_regime(_gas(floor, ceiling, g=(0.0, -9.81, -9.81)), slab)
    alone = buoyant_regime(_gas(floor, ceiling, g=(0.0, -9.81, 0.0)), slab)
    assert both.rayleigh == pytest.approx(alone.rayleigh)
    assert both.cell == pytest.approx(0.5)                # the in-plane spacing, not a cube root


def test_unknown_solution_directions_give_no_estimate():
    cfg, mesh = _rb_room()
    assert buoyant_regime(cfg, mesh.model_copy(update={"solution_directions": None})) is None


def test_a_boussinesq_beta_counts_only_where_no_equation_of_state_was_read():
    cfg, mesh = _rb_room()
    stale = cfg.model_copy(update={"transport": {**AIR, "beta": 1.0, "nu": 1e-9}})
    assert buoyant_regime(stale, mesh).rayleigh == pytest.approx(buoyant_regime(cfg, mesh).rayleigh)
    # OpenFOAM's BenardCells: no thermophysicalProperties, transportProperties nu 1e-3, beta 1e-3,
    # Pr 1; floor 301 K, ceiling 300 K, 1 m apart; 90 x 10 cells over 9 x 1 x 2 m, z empty.
    benard = CaseConfig(turbulence_type="laminar", gravity=(0.0, -9.81, 0.0),
                        fixed_temperature_range=(300.0, 301.0),
                        transport={"nu": 1e-3, "beta": 1e-3, "TRef": 300.0, "Pr": 1.0},
                        hot_faces_box=_box((0.05, 0, 0), (8.95, 0, 0)),
                        cold_faces_box=_box((0.05, 1, 0), (8.95, 1, 0)))
    m = MeshQuality(n_cells=900, bounding_box=_box((0, 0, -1), (9, 1, 1)), total_volume=18.0,
                    solution_directions=(1, 1, 0))
    r = buoyant_regime(benard, m)
    assert r.rayleigh == pytest.approx(9.81 * 1e-3 * 1.0 / (1e-3 * 1e-3))
    assert r.cell == pytest.approx(0.1)
    assert not r.turbulent and _findings(benard, m) == []


@pytest.mark.parametrize("update", [
    {"gravity": None}, {"gravity": (0.0, 0.0, 0.0)}, {"fixed_temperature_range": None},
    {"fixed_temperature_range": (300.0, 300.0)}, {"equation_of_state": "rhoConst"},
    {"equation_of_state": None}, {"pressure": None}, {"transport": {"Pr": 0.7, "molWeight": 28.9}},
    {"hot_faces_box": None}, {"cold_faces_box": None},
])
def test_no_estimate_without_what_it_needs(update):
    cfg, mesh = _rb_room()
    assert buoyant_regime(cfg.model_copy(update=update), mesh) is None


def test_no_estimate_without_a_mesh_extent():
    cfg, mesh = _rb_room()
    assert buoyant_regime(cfg, None) is None
    assert buoyant_regime(cfg, mesh.model_copy(update={"bounding_box": None})) is None
    assert buoyant_regime(cfg, mesh.model_copy(update={"total_volume": None})) is None


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("update", [
    {"transport": {**AIR, "mu": 1e-160}}, {"transport": {**AIR, "mu": 1e-165}},
    {"transport": {**AIR, "mu": 0.0}}, {"transport": {**AIR, "mu": -1.8e-5}},
    {"transport": {**AIR, "mu": NAN}}, {"transport": {**AIR, "molWeight": 0.0}},
    {"transport": {**AIR, "molWeight": INF}}, {"transport": {**AIR, "Pr": NAN}},
    {"fixed_temperature_range": (290.0, 1e400)}, {"fixed_temperature_range": (0.0, 320.0)},
    {"fixed_temperature_range": (-10.0, 320.0)}, {"fixed_temperature_range": (NAN, 320.0)},
    {"fixed_temperature_range": (1e-300, 320.0)}, {"pressure": 0.0}, {"pressure": -1e5},
    {"pressure": NAN}, {"pressure": 1e308}, {"gravity": (0.0, -INF, 0.0)},
    {"gravity": (0.0, NAN, 0.0)}, {"gravity": (0.0, -1e300, 0.0)},
    {"hot_faces_box": _box((0, -1e300, 0), (0, -1e300, 0))},
    {"cold_faces_box": _box((0, INF, 0), (0, INF, 0))},
    {"cold_faces_box": _box((0, NAN, 0), (0, NAN, 0))},
])
def test_any_input_gives_a_finite_estimate_or_none_and_a_verdict(update):
    cfg, mesh = _rb_room()
    cfg = cfg.model_copy(update=update)
    r = buoyant_regime(cfg, mesh)
    if r is not None:
        for x in (r.rayleigh, r.kolmogorov, r.cell, r.height, r.span):
            assert math.isfinite(x) and x > 0
    _findings(cfg, mesh)                                  # never raises
    from courant.verify import verify_inputs
    from courant.verify.schema import RunInputs

    log = (CASE / "log.buoyantPimpleFoam").read_text()
    v = verify_inputs(RunInputs(run_id="fuzz", log_text=log, log_name="log.buoyantPimpleFoam",
                                config_view=cfg.model_dump(mode="python"),
                                mesh_view=mesh.model_dump(mode="python"),
                                checks="next")).to_v1()
    assert v.verdict in ("TRUSTWORTHY", "NOT_TRUSTWORTHY", "NOT_ASSESSABLE")


# ------------------------------------------------------------------------- the finding
def test_it_fires_on_a_laminar_room_heated_from_below_on_a_coarse_mesh():
    cfg, mesh = _rb_room()
    [f] = _findings(cfg, mesh)
    assert f.severity is FindingSeverity.WARN
    assert f.suggests == ["PHYS.LAMINAR_TURBULENT"]
    assert f.threshold == RA_TRANSITION and f.value > 1e11
    assert "Rayleigh" in f.detail and "Kolmogorov" in f.detail


def test_it_is_silent_when_the_solver_did_not_read_g():
    cfg, mesh = _rb_room()
    assert "Reading g" in (CASE / "log.buoyantPimpleFoam").read_text()
    assert _findings(cfg, mesh, log_text=None) == []
    assert _findings(cfg, mesh, log_text="Reading field U\nReading gravity elsewhere\n") == []


def test_it_is_silent_with_a_turbulence_model():
    cfg, mesh = _rb_room()
    for kind in ("RAS", "LES"):
        assert _findings(cfg.model_copy(update={"turbulence_type": kind}), mesh) == []


def test_it_is_silent_on_a_mesh_fine_enough_for_direct_simulation():
    cfg, mesh = _rb_room()
    r = buoyant_regime(cfg, mesh)
    spacing = 0.9 * DNS_SPACING * r.kolmogorov          # Groetzbach's criterion met
    fine = mesh.model_copy(update={"n_cells": int(500.0 / spacing ** 3) + 1})
    assert _findings(cfg, fine) == []


def test_it_is_silent_below_the_transition_even_on_a_coarse_mesh():
    cfg, mesh = _rb_room()
    mild = cfg.model_copy(update={"fixed_temperature_range": (300.0, 300.01)})
    assert buoyant_regime(mild, mesh).rayleigh < RA_TRANSITION
    assert _findings(mild, mesh) == []


# --------------------------------------------------------------------------- the verdict
def test_draw_044_keeps_its_blind_verdict(tmp_path):
    from courant.verify import from_case_dir, verify_inputs

    v = verify_inputs(from_case_dir(_case_with_mesh(tmp_path), checks="next")).to_v1()
    assert "physics.laminar_regime" not in [r.code for r in v.reasons]
    assert (v.verdict, v.rule) == ("TRUSTWORTHY", "VER-5.6")


def test_the_same_room_held_hot_below_and_cold_above_is_rejected(tmp_path):
    from courant.verify import from_case_dir, verify_inputs

    case = _case_with_mesh(tmp_path)
    t = case / "0" / "T"
    text = t.read_text()
    start = text.index("nonuniform List<scalar> 400")
    end = text.index(")", start) + 1
    text = text[:start] + "uniform 320" + text[end:]
    t.write_text(text.replace("lumpedMassWallTemperature", "fixedValue").replace(
        "value           uniform 300;", "value           uniform 290;"))
    v = verify_inputs(from_case_dir(case, checks="next")).to_v1()
    assert v.verdict == "NOT_TRUSTWORTHY"
    reason = next(r for r in v.reasons if r.code == "physics.laminar_regime")
    for named in ("log", "constant/g", "0/T", "0/p", "constant/polyMesh"):
        assert named in reason.input, reason.input
    # GATE-13: under the default checks the same case is judged as v0.5.0 judged it: no regime
    # check, and neither constant/g nor the mesh files are read.
    d = verify_inputs(from_case_dir(case)).to_v1()
    assert (d.verdict, d.rule) == ("TRUSTWORTHY", "VER-5.6")
    assert "physics.laminar_regime" not in [r.code for r in d.reasons]
    assert not any(f.path == "constant/g" or "polyMesh" in f.path for f in d.run.inputs)
    assert _findings(CaseConfig.from_foam(str(case)), MeshQuality.from_checkmesh(
        (case / "log.checkMesh").read_text()), checks="default") == []
