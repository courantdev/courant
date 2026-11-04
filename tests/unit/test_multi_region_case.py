"""A multi-region (CHT) case read as its solver reads it (T-MULTI-REGION; TRD VER-3d, PRE-1).

A rebuilt heat sink's preflight (an internal evaluation) read a stale top-level `0/` while `0/solid/p` was missing:
a multi-region solver (chtMultiRegionFoam, chtMultiRegionSimpleFoam) reads each region's own
`system/<region>/`, `constant/<region>/` and `0/<region>/`, and of the top level only its
`fvSolution`, for what is the whole run's (chtMultiRegionFoam's `PIMPLE/nOuterCorrectors`).
Under `--checks next` (`case.regions`, 0.7.0), for a case whose solver reads regions
(`_foam_reader.multi_region`, the one predicate; a stray `regionProperties` beside a
single-region solver changes nothing), the case reader:

- fills `CaseConfig.regions` from `constant/regionProperties`, through the one reader
  (`courant.case._foam_reader.read_region_properties`, which VER-22's heat reader uses too);
- reads each region's `system/<region>/fvSolution` (relaxation, linear solvers),
  `system/<region>/fvSchemes` (divergence schemes) and the boundary conditions of
  `0/<region>/`, each keyed `<name>@<region>` with the region named by its place in `regions`
  (`config.region_pseudonym`: `U@region1`), so no region name reaches a finding;
- does not read the top level's `fvSchemes`, turbulence, transport or thermophysical
  dictionaries or `0/` (a stale copy there is not what ran), reads of its `fvSolution` the outer
  correctors alone, and names a region dictionary the case does not hold as missing;

and the gate reads each region's checkMesh log (`log.checkMesh.<region>`), takes `postProcess`
for the utility it is (`courant.cli.tail.find_log`), and conformance finds a multi-region run's
start in its `Create fluid mesh for region <r> for time = <t>` line. The default is 0.6.0's.

The fixture is the committed two-region chtMultiRegionSimpleFoam solve, `cht_plate/converged`
(regions `air`, a fluid, and `plate`, a solid).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest

from courant.verify import verify_path
from courant.verify.preflight import preflight_path
from tests.unit.test_mass_flux import _unpack_fixture

ROOT = Path(__file__).resolve().parents[2]
CHT = ROOT / "tests" / "fixtures" / "real_openfoam" / "cht_plate" / "converged"
CHECKMESH = ROOT / "tests" / "fixtures" / "real_openfoam" / "polymesh" / \
    "buoyantPimpleFoam_hotRoom" / "log.checkMesh"
#: The default document 0.6.0 gives on the solve (de15888, before this node, 3 Nov 2026),
#: `tool.version` set to "*" and the keys sorted.
V060_DEFAULT = "1e62b4f8f3a8d294945c1c5bb6c0424d42becc09883acf70bd702ce936f457a4"


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> Path:
    """The committed solve as it ran."""
    return _unpack_fixture(CHT, tmp_path_factory.mktemp("cht") / "converged")


@pytest.fixture(scope="module")
def cht(tmp_path_factory) -> Path:
    """The solve with the air's `relaxationFactors/fields/rho` at 0.9 instead of 1, so that
    under next its verdict is decided by the change each test makes: `scheme.relaxation` flags
    a factor of 1 on any field, `rho` included, as it does on the single-region hot_box in both
    modes (`test_the_solve_as_it_ran_is_rejected_on_rho_under_next`)."""
    case = _unpack_fixture(CHT, tmp_path_factory.mktemp("cht_damped") / "converged")
    _sub(case / "system" / "air" / "fvSolution", r"^(\s*rho\s+)1;", r"\g<1>0.9;")
    return case


def _copy(src: Path, tmp_path: Path) -> Path:
    case = tmp_path / "converged"
    shutil.copytree(src, case)
    return case


def _sub(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
    assert n, f"{pattern!r} is not in {path}"
    path.write_text(new)


def _fv(obj: str, body: str) -> str:
    return f"FoamFile{{version 2.0;format ascii;class dictionary;object {obj};}}\n{body}\n"


def _codes(v) -> list[str]:
    return [r.code for r in v.reasons]


def _normalised_sha(verdict) -> str:
    d = json.loads(verdict.to_json())
    d["tool"]["version"] = "*"
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _config(case: Path, checks: str):
    from courant.case.config import CaseConfig
    from courant.verify.ingest import from_case_dir

    return CaseConfig.from_view(from_case_dir(case, checks=checks).config_view, read_here=True)


#: A stale single-region `0/` left at the top level: a patch that fixes both p and U, which
#: `bc.overconstrained` rejects wherever it is read.
_STALE_U = _fv("U", "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
                    "boundaryField { inlet { type fixedValue; value uniform (1 0 0); } }")
_STALE_P = _fv("p", "dimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 1e5;\n"
                    "boundaryField { inlet { type fixedValue; value uniform 1e5; } }")


# ---------------------------------------------------------------- the default stays 0.6.0's
@pytest.mark.usefixtures("checks_as_060")
def test_the_default_document_is_060s(raw, tmp_path):
    case = _copy(raw, tmp_path)
    v = verify_path(case)
    assert (v.verdict, v.rule) == ("accept", "VER-5.6")
    assert _normalised_sha(v) == V060_DEFAULT
    config = _config(case, "default")
    assert config.regions == [] and config.boundary == {}


def test_the_solve_as_it_ran_is_not_rejected_on_its_regions_rho(raw, tmp_path):
    """What reading the regions exposed: the air's `fields { rho 1; }` (v2406's buoyant and CHT
    tutorials write it), keyed `rho@region1`, met `scheme.relaxation`'s bar of 1 and rejected the
    solve as it ran. T-RELAX-RHO's exemption is the field's, so a region's `rho` is none either
    (the open item this test pinned, closed 3 Nov 2026)."""
    case = _copy(raw, tmp_path)
    nxt = verify_path(case, checks="next")
    assert not [r for r in nxt.reasons if r.code == "scheme.relaxation"], nxt.reasons


# ---------------------------------------------------------------- the regions, under next
def test_next_fills_regions_from_region_properties(cht, tmp_path):
    from courant.case._foam_reader import read_region_properties

    case = _copy(cht, tmp_path)
    config = _config(case, "next")
    kinds = read_region_properties(str(case))
    assert config.regions == [r for names in kinds.values() for r in names] == ["air", "plate"]
    inputs = {f.path for f in verify_path(case, checks="next").run.inputs}
    assert "constant/regionProperties" in inputs
    for rel in ("system/air/fvSolution", "system/plate/fvSolution", "system/air/fvSchemes",
                "0/air/U", "0/plate/T"):
        assert rel in inputs, rel
    # Not what the solver read: the top level's schemes are not opened; its fvSolution is, for
    # the outer correctors alone (the review, B2).
    assert "system/fvSchemes" not in inputs and "system/fvSolution" in inputs


def test_next_reads_each_regions_dictionaries_keyed_by_place(cht, tmp_path):
    from courant.case.config import region_pseudonym

    case = _copy(cht, tmp_path)
    config = _config(case, "next")
    assert region_pseudonym(config.regions, "air") == "region1"
    assert region_pseudonym(config.regions, "plate") == "region2"
    assert region_pseudonym(config.regions, "nowhere") == "a region"
    assert config.relaxation_factors == {"rho@region1": 0.9, "p_rgh@region1": 0.7,
                                         "U@region1": 0.5, "h@region1": 0.7, "h@region2": 0.7}
    assert set(config.linear_solvers) == {"rho@region1", "p_rgh@region1", '"(U|h)"@region1',
                                          "h@region2"}
    assert config.linear_solvers["h@region2"].tolerance == 1e-12
    assert {p.rsplit("@", 1)[1] for p in config.boundary} == {"region1", "region2"}
    assert config.boundary["inlet@region1"]["U"] == "fixedValue"
    assert "p_rgh" in config.boundary["inlet@region1"]
    # The region is named by its place; a patch keeps its own name (splitMeshRegions writes
    # `air_to_plate`), which the trust boundary pseudonymises whole (`redact`, `boundary`).
    assert not any("air" in k or "plate" in k
                   for k in [*config.relaxation_factors, *config.linear_solvers])
    assert "air_to_plate@region1" in config.boundary


def test_a_region_relaxation_fault_is_rejected_under_next_only(cht, tmp_path):
    """U fully undamped in the air region's own fvSolution: the default reads the top level,
    which says nothing, and accepts as 0.6.0 did; next reads the region and rejects."""
    case = _copy(cht, tmp_path)
    _sub(case / "system" / "air" / "fvSolution", r"^(\s*U\s+)0\.5;", r"\g<1>1;")
    default, nxt = verify_path(case), verify_path(case, checks="next")
    assert (default.verdict, default.rule) == ("accept", "VER-5.6")
    assert (nxt.verdict, nxt.rule) == ("reject", "VER-5.4")
    assert "scheme.relaxation" in _codes(nxt)
    pre = preflight_path(case, checks="next")
    assert pre.verdict == "SETUP_FAULT" and "scheme.relaxation" in {r.code for r in pre.reasons}
    assert preflight_path(case).verdict == "NO_SETUP_FAULT_FOUND"


def test_a_loose_tolerance_in_the_solid_region_is_rejected_under_next(cht, tmp_path):
    case = _copy(cht, tmp_path)
    _sub(case / "system" / "plate" / "fvSolution", r"^(\s*tolerance\s+)1e-12;", r"\g<1>0.1;")
    nxt = verify_path(case, checks="next")
    assert (nxt.verdict, nxt.rule) == ("reject", "VER-5.4")
    reason = next(r for r in nxt.reasons if r.code == "solver.loose_tolerance")
    assert "plate" not in json.dumps(reason.model_dump(mode="json"))
    assert verify_path(case).verdict == "accept"


def test_the_stale_top_level_is_not_read_under_next(cht, tmp_path):
    """The heat sink's shape: a top-level `0/` and `fvSolution` left from a single-region case.
    The default reads them (0.6.0: a stale over-constrained patch rejects a run that never read
    it); next reads the regions' own and accepts."""
    case = _copy(cht, tmp_path)
    (case / "0" / "U").write_text(_STALE_U)
    (case / "0" / "p").write_text(_STALE_P)
    (case / "system" / "fvSolution").write_text(
        _fv("fvSolution", "relaxationFactors { equations { U 1; } }"))
    default, nxt = verify_path(case), verify_path(case, checks="next")
    assert default.verdict == "reject" and "bc.overconstrained" in _codes(default)
    assert "U" in _config(case, "default").relaxation_factors          # read by 0.6.0
    assert (nxt.verdict, nxt.rule) == ("accept", "VER-5.6"), _codes(nxt)
    config = _config(case, "next")
    assert "inlet" not in config.boundary and "U" not in config.relaxation_factors
    assert not {"0/U", "0/p"} & {f.path for f in nxt.run.inputs}


def test_preflight_reads_the_regions_zero_not_the_stale_top_level(cht, tmp_path):
    case = _copy(cht, tmp_path)
    (case / "0" / "U").write_text(_STALE_U)
    (case / "0" / "p").write_text(_STALE_P)
    assert preflight_path(case).verdict == "SETUP_FAULT"          # 0.6.0: the stale copy
    nxt = preflight_path(case, checks="next")
    assert nxt.verdict == "NO_SETUP_FAULT_FOUND", [r.code for r in nxt.reasons]


def test_a_region_over_constrained_in_its_own_zero_is_found_under_next(cht, tmp_path):
    """The air's outlet holds p_rgh fixed already; fixing U there as well over-constrains it."""
    case = _copy(cht, tmp_path)
    assert re.search(r"outlet\s*\{\s*type\s+fixedValue;", (case / "0" / "air" / "p_rgh")
                     .read_text())
    _sub(case / "0" / "air" / "U", r"(outlet\s*\{\s*type\s+)inletOutlet;\s*inletValue[^;]*;",
         r"\1fixedValue;")
    nxt = verify_path(case, checks="next")
    assert "bc.overconstrained" in _codes(nxt)
    reason = next(r for r in nxt.reasons if r.code == "bc.overconstrained")
    assert "outlet@region1" in json.dumps(reason.model_dump(mode="json"))


def test_a_missing_region_dictionary_is_named(cht, tmp_path):
    case = _copy(cht, tmp_path)
    (case / "system" / "plate" / "fvSolution").unlink()
    config = _config(case, "next")
    assert config.missing == ["system/plate/fvSolution"]
    nxt = verify_path(case, checks="next")
    assert any("system/plate/fvSolution" in s for s in nxt.not_assessed)
    # The top level's fvSolution is one the solver reads too (createCoupledRegions.H, and
    # chtMultiRegionFoam's PIMPLE): absent, it is named (the review, B2).
    (case / "system" / "fvSolution").unlink()
    assert _config(case, "next").missing == ["system/fvSolution", "system/plate/fvSolution"]


@pytest.mark.parametrize("body", ["regions ( fluid ( $air ) );", "regions ();",
                                  "regions ( fluid () );"], ids=["macro", "empty", "no-name"])
def test_a_region_properties_naming_no_region_is_said_and_the_case_read_as_one(cht, tmp_path,
                                                                               body):
    """The review (B1): a file that cannot be read whole, or names no region, is unreadable,
    and the case is read as one region, every top-level dictionary read: nothing is skipped in
    silence."""
    case = _copy(cht, tmp_path)
    (case / "constant" / "regionProperties").write_text(_fv("regionProperties", body))
    config = _config(case, "next")
    assert config.regions == []
    assert "constant/regionProperties" in config.unreadable
    inputs = {f.path for f in verify_path(case, checks="next").run.inputs}
    assert {"system/fvSchemes", "system/fvSolution"} <= inputs


def test_the_top_level_outer_correctors_are_read_in_a_multi_region_case(cht, tmp_path):
    """The review (B2): chtMultiRegionFoam reads `PIMPLE/nOuterCorrectors` from the top level's
    fvSolution (readPIMPLEControls.H: `fvSolution solutionDict(runTime)`, read in the pinned
    image), which sets the Courant limit (VER-13). Read there, and only that."""
    case = _copy(cht, tmp_path)
    (case / "system" / "fvSolution").write_text(_fv(
        "fvSolution", "PIMPLE { nOuterCorrectors 3; }\nrelaxationFactors { equations { U 1; } }"))
    config = _config(case, "next")
    assert config.n_outer_correctors == 3
    assert "U" not in config.relaxation_factors
    assert "system/fvSolution" in {f.path for f in verify_path(case, checks="next").run.inputs}


HOT_BOX = ROOT / "tests" / "fixtures" / "real_openfoam" / "hot_box" / "converged"


def test_a_stray_region_properties_beside_a_single_region_solver_is_not_read(tmp_path):
    """The review (B1, a false accept): buoyantSimpleFoam never reads `constant/regionProperties`,
    so a stray one does not make the case multi-region: its own fvSolution is still read, and a
    loose tolerance in it is still rejected under next."""
    case = _unpack_fixture(HOT_BOX, tmp_path / "converged")
    _sub(case / "system" / "fvSolution", r"^(\s*tolerance\s+)1e-\d+;", r"\g<1>0.5;")
    (case / "constant" / "regionProperties").write_text(
        _fv("regionProperties", "regions ( fluid ( box ) );"))
    for checks in ("default", "next"):
        v = verify_path(case, checks=checks)
        assert (v.verdict, v.rule) == ("reject", "VER-5.4"), checks
        assert "solver.loose_tolerance" in _codes(v), checks
    config = _config(case, "next")
    assert config.regions == [] and "constant/regionProperties" not in config.unreadable
    from courant.case._foam_reader import multi_region

    assert not multi_region(case, "buoyantSimpleFoam")
    assert multi_region(case, "chtMultiRegionSimpleFoam")


@pytest.mark.usefixtures("checks_as_060")
def test_the_verdict_says_which_region_dictionaries_were_read(cht, tmp_path):
    from courant.verify import sentences

    case = _copy(cht, tmp_path)
    default, nxt = verify_path(case), verify_path(case, checks="next")
    assert sentences.MULTI_REGION_MERGED in default.not_assessed
    assert sentences.MULTI_REGION_KEYED_READ in nxt.not_assessed
    assert sentences.MULTI_REGION_KEYED not in nxt.not_assessed
    assert "system/<region>/" in sentences.MULTI_REGION_KEYED_READ


# ---------------------------------------------------------------- the region checkMesh logs
def _region_checkmesh(case: Path, region: str, non_ortho: float | None = None) -> None:
    text = CHECKMESH.read_text()
    if non_ortho is not None:
        text = text.replace("Mesh non-orthogonality Max: 0 average: 0",
                            f"Mesh non-orthogonality Max: {non_ortho} average: 10")
    (case / f"log.checkMesh.{region}").write_text(text)


def test_each_regions_checkmesh_log_is_read_under_next(cht, tmp_path):
    case = _copy(cht, tmp_path)
    _region_checkmesh(case, "air")
    _region_checkmesh(case, "plate", non_ortho=75.0)
    nxt = verify_path(case, checks="next")
    assert "mesh.non_orthogonality" in _codes(nxt)
    default = verify_path(case)
    assert "mesh.non_orthogonality" not in _codes(default)       # 0.6.0 read no region's log
    from courant.verify import sentences

    assert sentences.NO_MESH in default.not_assessed and sentences.NO_MESH not in nxt.not_assessed
    assert {"log.checkMesh.air", "log.checkMesh.plate"} <= {f.path for f in nxt.run.inputs}


def test_some_regions_checkmesh_logs_still_count_and_the_rest_are_said(cht, tmp_path):
    """The review (M2): a fault in the logs that are there is never thrown away for the log that
    is not; what needs every region (the cell count) is not given."""
    case = _copy(cht, tmp_path)
    _region_checkmesh(case, "plate", non_ortho=75.0)
    from courant.verify import sentences

    nxt = verify_path(case, checks="next")
    assert "mesh.non_orthogonality" in _codes(nxt)
    assert sentences.MESH_REGIONS_PARTIAL in nxt.not_assessed
    assert sentences.MESH_REGIONS_PARTIAL not in verify_path(case).not_assessed
    from courant.case.mesh import read_mesh_quality

    q = read_mesh_quality(str(case), regions=["air", "plate"])
    assert q.max_non_orthogonality == 75.0 and q.n_cells is None and q.mesh_ok is None


# ---------------------------------------------------------------- postProcess, a utility
def _post_process_log(case: Path) -> Path:
    with gzip.open(CHT / "log.chtMultiRegionSimpleFoam.gz", "rt") as fh:
        banner = "".join(fh.readlines()[:22])
    log = case / "log.postProcess"
    log.write_text(banner.replace("chtMultiRegionSimpleFoam", "postProcess -func wallHeatFlux "
                                  "-region air -latestTime")
                   + "Create time\n\nCreate mesh air for time = 3000\n\nTime = 3000\n"
                     "Reading fields:\n\nEnd\n\n")
    later = time.time() + 60
    os.utime(log, (later, later))
    return log


def test_post_process_is_a_utility_under_next(cht, tmp_path):
    """A postProcess run after the solve prints `Time =` and is newer than the solver's log:
    0.6.0 took it for the run. Under next it is the utility it is."""
    from courant.cli.tail import find_log, utility_logs

    case = _copy(cht, tmp_path)
    _post_process_log(case)
    assert find_log(case).name == "log.postProcess"                        # 0.6.0
    assert find_log(case, checks="next").name == "log.chtMultiRegionSimpleFoam"
    assert "log.postProcess" in utility_logs(case, checks="next")
    assert "log.postProcess" not in utility_logs(case)
    readable = verify_path(_copy(cht, tmp_path / "plain"), checks="next")
    nxt = verify_path(case, checks="next")
    assert (nxt.verdict, nxt.rule) == (readable.verdict, readable.rule)
    assert nxt.run.solver == "chtMultiRegionSimpleFoam"


# ---------------------------------------------------------------- the restart line
@pytest.mark.parametrize("line,start", [
    ("Create fluid mesh for region air for time = 0", "0"),
    ("Create solid mesh for region plate for time = 250", "250"),
    ("Create mesh air for time = 0.5", "0.5"),
    ("Create mesh for time = 100", "100"),
])
def test_the_run_start_is_read_from_a_region_mesh_line(line, start):
    from courant.spec.as_run import Run

    run = Run(log_text=f"Create time\n\n{line}\n\nTime = 1\n", log_name="log.x", exit_code=None)
    assert run.start_name == start


def test_the_cht_solves_start_is_found(cht):
    from courant.spec.as_run import Run

    with gzip.open(CHT / "log.chtMultiRegionSimpleFoam.gz", "rt") as fh:
        run = Run(log_text=fh.read(), log_name="log.chtMultiRegionSimpleFoam", exit_code=None)
    assert run.start_name == "0"


# ---------------------------------------------------------------- the trust boundary
def test_a_region_name_does_not_cross_in_the_configuration(cht, tmp_path):
    """The keys name a region by its place; an unreadable region dictionary, named by its
    directory locally, crosses with the region's place in its path
    (`redact.sanitize_config_view`)."""
    from courant.diagnosis.redact import find_leaks, sanitize_config_view

    case = _copy(cht, tmp_path)
    (case / "system" / "plate" / "fvSchemes").write_text("FoamFile { this does not parse")
    config = _config(case, "next")
    assert "system/plate/fvSchemes" in config.unreadable                 # local: as written
    clean = sanitize_config_view(config.model_dump(mode="json"))
    assert "system/region2/fvSchemes" in clean["unreadable"]
    assert clean["regions"] == ["region1", "region2"]
    assert find_leaks(clean, ["plate"]) == []
    # ... and the finding that reaches the Evidence Bundle names it by place (the review, B3)
    v = verify_path(case, checks="next")
    reason = [s for s in v.not_assessed if s.startswith("Case dictionary could not be parsed")]
    assert reason and "system/region2/fvSchemes" in reason[0] and "plate" not in reason[0]


# ---------------------------------------------------------------- 0/<region>/ (the review, M1)
_root_reads_everything = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 directory")


@_root_reads_everything
@pytest.mark.parametrize("rel,mode", [("0/air", 0o000), ("0/air", 0o644), ("0", 0o000)],
                         ids=["air-000", "air-644", "zero-000"])
def test_a_regions_zero_that_cannot_be_read_is_said_under_next(cht, tmp_path, rel, mode):
    from tests.unit.test_case_inputs_robust import _Mode

    case = _copy(cht, tmp_path)
    with _Mode(case / rel, mode):
        config = _config(case, "next")
        v = verify_path(case, checks="next")
    assert "0/air/" in config.unreadable
    said = " ".join(v.not_assessed)
    assert "0/region1/: PermissionError: Permission denied" in said and "0/air/" not in said
    from courant.verify import sentences

    assert sentences.MULTI_REGION_KEYED_PART in v.not_assessed
    assert sentences.MULTI_REGION_KEYED_READ not in v.not_assessed


def test_a_missing_regions_zero_is_named(cht, tmp_path):
    case = _copy(cht, tmp_path)
    shutil.rmtree(case / "0" / "plate")
    assert "0/plate/" in _config(case, "next").missing


# ---------------------------------------------------------------- the sentence (the review, m2)
def test_the_sentence_claims_the_regions_read_only_when_each_was(cht, tmp_path):
    from courant.verify import sentences

    case = _copy(cht, tmp_path)
    assert sentences.MULTI_REGION_KEYED_READ in verify_path(case, checks="next").not_assessed
    (case / "system" / "plate" / "fvSchemes").unlink()
    v = verify_path(case, checks="next")
    assert sentences.MULTI_REGION_KEYED_PART in v.not_assessed
    assert sentences.MULTI_REGION_KEYED_READ not in v.not_assessed


# ---------------------------------------------------------------- the per-field join (m3)
def test_a_regions_residual_is_relaxed_by_its_regions_own_factor(cht, tmp_path):
    """VER-20 keys a residual by the region's name (`h@air`), the case reader a factor by the
    region's place (`h@region1`): the lookup maps one to the other, so the stall detector reads
    the factor the region's own fvSolution sets."""
    from courant.verification.convergence import equation_relaxation

    case = _copy(cht, tmp_path)
    _sub(case / "system" / "plate" / "fvSolution", r"^(\s*h\s+)0\.7;", r"\g<1>0.3;")
    config = _config(case, "next")
    assert equation_relaxation(config, "h@air") == 0.7
    assert equation_relaxation(config, "h@plate") == 0.3
    assert equation_relaxation(config, "Ux@air") == 0.5
    assert equation_relaxation(config, "p_rgh@air") == 1.0           # pressure: never
    assert equation_relaxation(config, "h@nowhere") == 1.0
    assert equation_relaxation(config, "h") == 1.0                   # no region: not a region's
