"""TRD VER-20: a multi-region log's residuals are never merged across regions.

v2406's conjugate-heat-transfer solvers solve each region in turn and print a header before each
(`Solving for fluid region bottomAir`, `Solving for solid region heater`; read from
`chtMultiRegionFoam.C`, `chtMultiRegionSimpleFoam.C` and their `solveFluid.H`, `solveSolid.H`).
v0.5.0's parser ignored the headers and keyed residuals by field alone, so the solid's `h`
overwrote the fluid's in every step, and a fluid whose energy had not converged could be hidden
behind a solid that had (an internal evaluation, 20 Oct 2026).

Behind `--checks next` (`log.regions`, GATE-13) each residual is keyed `<field>@<region>`, the
step's Courant number is the largest region's (v0.5.0 read none: `Region: X Courant Number` never
matched), and a residual from a coupled energy solve, which no header names, is stored under no
region. Under the default the parse is v0.5.0's and the verdict says the residuals were merged.

OpenFOAM.org's `foamMultiRun` (v11 on) prints no header: each line of its time loop carries a
prefix instead (`regionSolvers.C`, `foamMultiRun.C` and `prefixOSstream.C` in OpenFOAM-11 and
OpenFOAM-12, read on 21 Oct 2026): the region's name padded with spaces to one more character
than the longest name, or that many spaces outside a region's solve, the `Time =` line among
them. Under `next` the prefix is read; under the default it is not, and the verdict says so.

The logs here are synthetic, in the line shapes v2406's CHT solvers and the OpenFOAM.org
foamMultiRun print.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pytest

from courant.events import EventKind
from courant.verification.convergence import final_residuals
from courant.verify import RunInputs, sentences, verify_inputs
from courant.verify.schema import InputFile
from courant.watcher.parser import LogParser, has_coupled_solve, regions_in

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "certification" / "corpus"

FLUIDS, SOLIDS = ("bottomAir", "topAir"), ("heater",)


def _cht_log(final: dict[str, dict[str, float]], n: int = 50, *, courant=None,
             coupled: dict[str, float] | None = None,
             solver: str = "chtMultiRegionSimpleFoam") -> str:
    """A CHT log: per step, each fluid region solves Ux, Uy, h, p_rgh and each solid region h,
    every residual falling geometrically from 1 to `final[region][field]`. `courant` gives each
    fluid region's Courant number (printed before `Time =`, as chtMultiRegionFoam prints it);
    `coupled` adds a coupled energy solve with those final residuals."""
    out = [f"Exec   : {solver}", "", "Create fluid mesh for region bottomAir for time = 0", ""]
    for i in range(1, n + 1):
        if courant:
            out += [f"Region: {r} Courant Number mean: {co / 5:.6g} max: {co:.6g}"
                    for r, co in courant.items()]
        out += [f"Time = {i}", ""]
        for region in FLUIDS + SOLIDS:
            out.append(f"Solving for {'fluid' if region in FLUIDS else 'solid'} region {region}")
            for name, last in final[region].items():
                r = last ** (i / n)
                out.append(f"DILUPBiCGStab:  Solving for {name}, Initial residual = {r:.6g}, "
                           f"Final residual = {r / 50:.6g}, No Iterations 2")
            if region in FLUIDS:
                out.append(f"time step continuity errors ({region}): sum local = 1e-09, "
                           f"global = 1e-11, cumulative = 1e-10")
            out.append("")
        if coupled:
            out.append("Solving energy coupled regions")
            for name, last in coupled.items():
                r = last ** (i / n)
                out.append(f"GAMG:  Solving for {name}, Initial residual = {r:.6g}, "
                           f"Final residual = {r / 50:.6g}, No Iterations 3")
            out.append("")
        out += [f"ExecutionTime = {0.05 * i:.2f} s  ClockTime = {i} s", ""]
    return "\n".join(out + ["End", ""])


#: The solid has converged; the bottom fluid's energy has not (5e-2, five times the gate's
#: MAJOR line). Merged, the solid's `h` stands for both.
HIDDEN = {"bottomAir": {"Ux": 1e-5, "Uy": 1e-5, "h": 5e-2, "p_rgh": 1e-5},
          "topAir": {"Ux": 1e-5, "Uy": 1e-5, "h": 1e-5, "p_rgh": 1e-5},
          "heater": {"h": 1e-6}}
HEALTHY = {r: {f: min(v, 1e-5) for f, v in fields.items()} for r, fields in HIDDEN.items()}


def _last_step(events):
    return [e for e in events if e.kind is EventKind.RESIDUAL and e.payload.iteration][-1].payload


def _parse(text: str, regions: bool):
    parser = LogParser(run_id="cht", regions=regions)
    return parser, list(parser.parse_text(text))


def _verify(text: str, checks: str):
    return verify_inputs(RunInputs(
        run_id="cht", log_text=text, log_name="log.chtMultiRegionSimpleFoam",
        inputs=(InputFile.of_text(text, "log.chtMultiRegionSimpleFoam"),),
        source="log", checks=checks))


# ---------------------------------------------------------------- the parser
def test_the_headers_name_the_regions():
    text = _cht_log(HIDDEN)
    assert regions_in(text) == ("bottomAir", "topAir", "heater")
    assert not has_coupled_solve(text)
    assert has_coupled_solve(_cht_log(HIDDEN, coupled={"h": 1e-4}))


def test_v050s_parse_merges_the_regions_and_the_solid_hides_the_fluid():
    """Off, the parse is v0.5.0's: one `h`, the last region's (the heater's 1e-6)."""
    _, events = _parse(_cht_log(HIDDEN), regions=False)
    last = _last_step(events)
    assert set(last.initial_residuals) == {"Ux", "Uy", "h", "p_rgh"}
    assert last.initial_residuals["h"] == pytest.approx(1e-6)
    assert max(final_residuals(events).values()) < 1e-3


def test_keyed_by_region_nothing_is_merged():
    _, events = _parse(_cht_log(HIDDEN), regions=True)
    last = _last_step(events)
    assert set(last.initial_residuals) == {
        "Ux@bottomAir", "Uy@bottomAir", "h@bottomAir", "p_rgh@bottomAir",
        "Ux@topAir", "Uy@topAir", "h@topAir", "p_rgh@topAir", "h@heater"}
    assert last.initial_residuals["h@bottomAir"] == pytest.approx(5e-2)
    assert last.initial_residuals["h@heater"] == pytest.approx(1e-6)
    assert final_residuals(events)["h@bottomAir"] == pytest.approx(5e-2)


def test_the_courant_number_is_the_largest_regions():
    """`compressibleMultiRegionCourantNo.H`: the step's CoNum is the max over fluid regions."""
    text = _cht_log(HEALTHY, courant={"bottomAir": 0.3, "topAir": 0.9},
                    solver="chtMultiRegionFoam")
    _, merged = _parse(text, regions=False)
    _, keyed = _parse(text, regions=True)
    assert {e.payload.courant_max for e in merged if e.kind is EventKind.RESIDUAL} == {None}
    # Printed before each `Time =`, so it lands on the step before, as a single-region log's
    # does; the last step has none after it.
    assert {e.payload.courant_max for e in keyed if e.kind is EventKind.RESIDUAL} == {0.9, None}
    assert sum(1 for e in keyed if e.kind is EventKind.RESIDUAL
               and e.payload.courant_max == 0.9) == 50


def test_a_coupled_solve_is_stored_under_no_region_and_a_nan_there_still_diverges():
    text = _cht_log(HEALTHY, coupled={"h": 1e-4})
    parser, events = _parse(text, regions=True)
    assert parser.unattributed == 50
    assert all("h" not in e.payload.initial_residuals for e in events
               if e.kind is EventKind.RESIDUAL)
    bad = re.sub(r"(GAMG:  Solving for h, Initial residual = )[^,]+", r"\1nan", text, count=1)
    assert bad != text
    _, events = _parse(bad, regions=True)
    assert any(e.kind is EventKind.DIVERGENCE and e.payload.field == "h" for e in events)


def test_a_phase_fraction_keyed_by_region_is_still_a_phase_fraction():
    final = {r: dict(f) for r, f in HEALTHY.items()}
    final["bottomAir"]["alpha.water"] = 3e-2
    _, events = _parse(_cht_log(final), regions=True)
    assert "alpha.water@bottomAir" in _last_step(events).initial_residuals
    assert "alpha.water@bottomAir" not in final_residuals(events)


def test_a_single_region_log_parses_the_same_either_way():
    with gzip.open(CORPUS / "pimpleFoam_planarContraction__healthy.log.gz", "rt") as fh:
        text = fh.read()
    assert regions_in(text) == ()
    def dump(events):
        return [e.model_dump(mode="json", exclude={"ts_wall"}) for e in events]

    off = dump(LogParser(run_id="r").parse_text(text))
    on = dump(LogParser(run_id="r", regions=True).parse_text(text))
    assert len(on) > 100 and on == off


# ---------------------------------------------------------------- the verdict
def test_the_hidden_fluid_is_rejected_under_next_and_the_default_says_it_merged():
    """The default's accept here is v0.5.0's verdict and a false accept: the bottom fluid's `h`
    ends at 5e-2. GATE-13 keeps a default accept unless S names the change; whether the CHT case
    of an internal evaluation (20 Oct 2026) is the measured false accept that would let the
    default abstain on a multi-region log is S's decision, open (TRD VER-20)."""
    text = _cht_log(HIDDEN)
    default, nxt = _verify(text, "default"), _verify(text, "next")
    assert (default.verdict, default.rule) == ("accept", "VER-5.6")         # v0.5.0's verdict
    assert sentences.MULTI_REGION_MERGED in default.not_assessed
    assert (nxt.verdict, nxt.rule) == ("reject", "VER-5.3")
    assert sentences.MULTI_REGION_KEYED in nxt.not_assessed
    assert sentences.MULTI_REGION_MERGED not in nxt.not_assessed


def test_a_healthy_multi_region_run_is_accepted_in_both_modes():
    text = _cht_log(HEALTHY)
    for checks in ("default", "next"):
        v = _verify(text, checks)
        assert (v.verdict, v.rule) == ("accept", "VER-5.6"), checks


def test_the_coupled_solve_is_reported_unjudged_under_next():
    v = _verify(_cht_log(HEALTHY, coupled={"h": 1e-4}), "next")
    assert sentences.MULTI_REGION_COUPLED in v.not_assessed
    doc = json.loads(v.to_json())
    assert sentences.MULTI_REGION_COUPLED in doc["not_assessed"]


def test_a_single_region_verdict_carries_no_multi_region_sentence():
    from courant.verify import verify_path

    sid = "pimpleFoam_planarContraction__healthy"
    for checks in ("default", "next"):
        v = verify_path(CORPUS / f"{sid}.log.gz", CORPUS / f"{sid}.case.json", checks=checks)
        assert not {sentences.MULTI_REGION_MERGED, sentences.MULTI_REGION_KEYED,
                    sentences.MULTI_REGION_KEYED_COURANT, sentences.MULTI_REGION_COUPLED,
                    sentences.MULTI_REGION_PREFIX_UNREAD} & set(v.not_assessed)


def test_a_steady_multi_region_verdict_names_no_courant_number_it_did_not_read():
    """chtMultiRegionSimpleFoam is steady and prints no `Region: X Courant Number` line, so the
    sentence under `next` says nothing of one; chtMultiRegionFoam's does."""
    steady = _verify(_cht_log(HEALTHY), "next")
    assert sentences.MULTI_REGION_KEYED in steady.not_assessed
    assert sentences.MULTI_REGION_KEYED_COURANT not in steady.not_assessed
    assert "Courant" not in sentences.MULTI_REGION_KEYED
    transient = _verify(_cht_log(HEALTHY, courant={"bottomAir": 0.3, "topAir": 0.9},
                                 solver="chtMultiRegionFoam"), "next")
    assert sentences.MULTI_REGION_KEYED_COURANT in transient.not_assessed
    assert sentences.MULTI_REGION_KEYED not in transient.not_assessed
    assert "largest region's" in sentences.MULTI_REGION_KEYED_COURANT


def test_the_merged_sentence_claims_only_what_merging_does():
    """A field one region alone solves (p_rgh in a case with one fluid) was judged on that
    region's own residual; only a field solved in more than one region was merged."""
    assert ("a field solved in more than one region was judged on the last region's residual "
            "only") in sentences.MULTI_REGION_MERGED
    assert "no region was judged on its own residuals" not in sentences.MULTI_REGION_MERGED


def test_the_multi_region_sentences_are_tagged_as_the_runs_not_the_forks():
    """GATE-4's `fork` is the solver fork or version (RRD RM-22 counts them); a multi-region log
    is neither. Only the OpenFOAM.org prefix the default does not read is the fork's."""
    for s in (sentences.MULTI_REGION_MERGED, sentences.MULTI_REGION_KEYED,
              sentences.MULTI_REGION_KEYED_COURANT, sentences.MULTI_REGION_COUPLED):
        assert sentences.NOT_CHECKED_CAUSE[s] == "run"
    assert sentences.NOT_CHECKED_CAUSE[sentences.MULTI_REGION_PREFIX_UNREAD] == "fork"


# ---------------------------------------------------------------- OpenFOAM.org's foamMultiRun
ORG_FLUIDS, ORG_SOLIDS = ("bottomAir", "topAir"), ("heater",)
#: `regionSolvers`: the longest region name's width plus one.
ORG_WIDTH = max(len(r) for r in ORG_FLUIDS + ORG_SOLIDS) + 1


def _org_prefix(region: str | None = None) -> str:
    """`prefixes_[i]` (the region's name, padded) or `prefix0_` (blanks), as `Sout.prefix()`."""
    return (region or "").ljust(ORG_WIDTH)


def _org_log(final: dict[str, dict[str, float]], n: int = 50, *, courant=None,
             exec_line: bool = True) -> str:
    """A foamMultiRun log in the shape OpenFOAM-11's sources print it. The banner and setup carry
    no prefix (the regions' solvers do not exist yet). Each step: every region's `preSolve`
    (a fluid's Courant number, a solid's diffusion number) behind its own prefix, then
    `setGlobalPrefix` and `Time =` behind the blank one; then one PIMPLE iteration, region by
    region within each stage (momentum, energy, pressure), as `foamMultiRun.C` loops; then
    `ExecutionTime` behind the blank prefix. Fluids solve Ux, Uy, h, p_rgh; the solid, e."""
    p = _org_prefix
    out = (["Exec   : foamMultiRun"] if exec_line else []) + [
        "", "Create time", "", "Create mesh bottomAir for time = 0", "", "Starting time loop", ""]
    for i in range(1, n + 1):
        if courant:
            out += [p(r) + f"Courant Number mean: {co / 5:.6g} max: {co:.6g}"
                    for r, co in courant.items()]
            out += [p(r) + "Diffusion Number mean: 0.001 max: 0.002" for r in ORG_SOLIDS]
        out += [p() + f"Time = {i}", p()]

        def solve(region: str, name: str) -> None:
            r = final[region][name] ** (i / n)
            out.append(p(region) + f"DILUPBiCGStab:  Solving for {name}, Initial residual = "
                       f"{r:.6g}, Final residual = {r / 50:.6g}, No Iterations 2")

        for region in ORG_FLUIDS:
            solve(region, "Ux")
            solve(region, "Uy")
        for region in ORG_FLUIDS:
            solve(region, "h")
        for region in ORG_SOLIDS:
            solve(region, "e")
        for region in ORG_FLUIDS:
            solve(region, "p_rgh")
            out.append(p(region) + "time step continuity errors : sum local = 1e-09, "
                       "global = 1e-11, cumulative = 1e-10")
        out += [p() + f"ExecutionTime = {0.05 * i:.2f} s  ClockTime = {i} s", p()]
    return "\n".join(out + [p() + "End", p(), ""])


#: The bottom fluid's energy has not converged (5e-2); the top fluid's, solved after it in each
#: step, has. Merged, the top fluid's `h` stands for both.
ORG_HIDDEN = {"bottomAir": {"Ux": 1e-5, "Uy": 1e-5, "h": 5e-2, "p_rgh": 1e-5},
              "topAir": {"Ux": 1e-5, "Uy": 1e-5, "h": 1e-5, "p_rgh": 1e-5},
              "heater": {"e": 1e-6}}
ORG_HEALTHY = {r: {f: min(v, 1e-5) for f, v in fields.items()}
               for r, fields in ORG_HIDDEN.items()}


def _verify_org(text: str, checks: str, solver_hint: str | None = None):
    return verify_inputs(RunInputs(
        run_id="org", log_text=text, log_name="log.foamMultiRun",
        inputs=(InputFile.of_text(text, "log.foamMultiRun"),), source="log",
        solver_hint=solver_hint, checks=checks))


def test_the_org_fixture_has_the_sources_shape():
    text = _org_log(ORG_HIDDEN)
    assert "bottomAir DILUPBiCGStab:  Solving for h," in text           # longest name, one blank
    assert "topAir    DILUPBiCGStab:  Solving for h," in text
    assert "\n          Time = 1\n" in text                               # behind the blank prefix
    assert regions_in(text) == ()                                       # no ESI header


def test_under_next_the_org_prefix_keys_each_residual_by_its_region():
    parser = LogParser(run_id="org", regions=True, multirun=True)
    events = list(parser.parse_text(_org_log(ORG_HIDDEN, courant={"bottomAir": 0.3,
                                                                  "topAir": 0.9})))
    steps = [e.payload for e in events if e.kind is EventKind.RESIDUAL and e.payload.iteration]
    assert [s.iteration for s in steps] == list(range(1, 51))
    assert set(steps[-1].initial_residuals) == {
        "Ux@bottomAir", "Uy@bottomAir", "h@bottomAir", "p_rgh@bottomAir",
        "Ux@topAir", "Uy@topAir", "h@topAir", "p_rgh@topAir", "e@heater"}
    assert final_residuals(events)["h@bottomAir"] == pytest.approx(5e-2)
    # Each step's Courant number is the largest region's (printed before its `Time =`, so it
    # lands on the step before, as in a single-region log; the last step has none after it).
    assert {s.courant_max for s in steps} == {0.9, None}
    assert parser.region_courant


def test_without_the_prefix_read_the_org_log_has_no_time_step():
    """v0.5.0's parse, the default's: `Time =` behind the blank prefix is not a `Time =` line,
    so no step is numbered, and every residual line, read by search, is keyed by field alone.
    `regions=True` alone does not read the prefix: only a run named foamMultiRun does."""
    def dump(events):
        return [e.model_dump(mode="json", exclude={"ts_wall"}) for e in events]

    text = _org_log(ORG_HIDDEN)
    events = list(LogParser(run_id="org").parse_text(text))
    assert all(e.payload.iteration is None for e in events if e.kind is EventKind.RESIDUAL)
    assert dump(LogParser(run_id="org", regions=True).parse_text(text)) == dump(events)


def test_the_default_says_it_did_not_read_the_org_prefix():
    v = _verify_org(_org_log(ORG_HIDDEN), "default")
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.15")          # v0.5.0's verdict
    assert sentences.MULTI_REGION_PREFIX_UNREAD in v.not_assessed
    assert sentences.MULTI_REGION_MERGED not in v.not_assessed


def test_under_next_the_hidden_org_fluid_is_rejected_and_a_healthy_one_accepted():
    nxt = _verify_org(_org_log(ORG_HIDDEN, courant={"bottomAir": 0.3, "topAir": 0.9}), "next")
    assert (nxt.verdict, nxt.rule) == ("reject", "VER-5.3")
    assert sentences.MULTI_REGION_KEYED_COURANT in nxt.not_assessed
    assert sentences.MULTI_REGION_PREFIX_UNREAD not in nxt.not_assessed
    ok = _verify_org(_org_log(ORG_HEALTHY, courant={"bottomAir": 0.3, "topAir": 0.9}), "next")
    assert (ok.verdict, ok.rule) == ("accept", "VER-5.6")


def test_a_utilitys_end_ahead_of_a_cut_org_log_is_not_the_runs_end_under_next():
    """`(blockMesh && foamMultiRun) > log.foamMultiRun`, cut mid-run: every `Time =` line sits
    behind the blank prefix, and VER-3a finds the last step there, so blockMesh's End is not the
    run's (insufficient, VER-5.16), where its 44 steps, read as a finished run, were accepted.
    The whole log is accepted, as before."""
    utility = "Exec   : blockMesh\n\nCreating block mesh topology\n\nWriting polyMesh\n\nEnd\n\n"
    text = _org_log(ORG_HEALTHY, courant={"bottomAir": 0.3, "topAir": 0.9})
    cut = text[: text.index(_org_prefix() + "Time = 45\n")]
    v = _verify_org(utility + cut, "next")
    assert (v.verdict, v.rule) == ("insufficient", "VER-5.16")
    assert sentences.TRUNCATED_LOG in v.not_assessed
    whole = _verify_org(utility + text, "next")
    assert (whole.verdict, whole.rule) == ("accept", "VER-5.6")
    # In parallel, "Finalising parallel run" follows the End behind the blank prefix: the End is
    # read there, so the finished run is accepted, not called cut.
    parallel = _verify_org(utility + text + "Finalising parallel run\n", "next")
    assert (parallel.verdict, parallel.rule) == ("accept", "VER-5.6")


def test_an_org_log_with_no_banner_is_not_recognised_as_solver_output():
    """Without its `Exec` line no line of a foamMultiRun log starts the way the recognition
    test reads solver output (`Time =` and `Solving for` lines sit behind the prefix), so the
    verdict is `insufficient` (VER-5.1) in both modes, whatever the run names as its solver: a
    gap, stated (TRD VER-20), never a judgement of merged residuals."""
    text = _org_log(ORG_HIDDEN, exec_line=False)
    for checks in ("default", "next"):
        for hint in (None, "foamMultiRun"):
            v = _verify_org(text, checks, solver_hint=hint)
            assert (v.verdict, v.rule) == ("insufficient", "VER-5.1"), (checks, hint)
