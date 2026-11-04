"""Checks behind `--checks next` (TRD GATE-13).

A new check ships here first: it is left out of every verdict unless the run is verified with
`--checks next`, and a verdict that ran it says so (`tool.checks: "next"`). After one minor
release it is promoted: its name leaves `NEXT` and it is on by default, and the release's
changelog counts the verdicts it turned on the corpus (`scripts/release.py`, GATE-13).

`NEXT` maps a name to the minor release it ships in behind the flag. A name is either a finding
id (`courant.diagnosis.checkers`), which `only_enabled` keeps out of a default verdict, or a
behaviour of a check that raises no finding of its own: the code path that has it asks
`enabled(name, mode)`, here, and nowhere decides it again. Under the default mode every name in
`NEXT` gives the released version's behaviour, byte for byte on the committed runs
(`tests/unit/test_checks_next.py`).

`PROMOTED` names what has left `NEXT`: at 0.7.0 every name 0.6.0 shipped behind the flag
(GATE-13) but `cfl.steady` (VER-13a), held for a measurement, so 0.7.0's default gives the
documents 0.6.0 gave under `--checks next` on the committed runs (`tests/unit/test_checks_next.py`,
against `187d6ae`'s documents). VER-18's flux balance from `phi` is promoted with them on RM-37's
result (the pre-registered measurement of 3 Nov 2026); `PROMOTE_MASS_FLUX_PHI`, one line, decides.
"""

from __future__ import annotations

MODES = ("default", "next")

#: NEXT names the code asks by a constant rather than its literal: defined once, here, and NEXT is
#: keyed by them (T-FOLLOWUP-MINORS: `pipeline.ENERGY_WALLHEATFLUX` repeated the literal).
LAMINAR_REGIME = "physics.laminar_unstable.regime"
ENERGY_WALLHEATFLUX = "conservation.energy.wallheatflux"
MASS_FLUX_PHI = "conservation.mass_flux.phi"

#: Whether VER-18's flux balance from `phi` (`MASS_FLUX_PHI`) is promoted with 0.6.0's other names
#: at 0.7.0: on RM-37's result (pre-registered 3 Nov 2026, every hypothesis supported)
#: it is. False keeps it behind `--checks next` for 0.7.0. The one line that decides it; the
#: tests and the preflight's sentence follow it.
PROMOTE_MASS_FLUX_PHI = True

#: {name: the minor release it is on by default from}: GATE-13's promotions. Each of these
#: shipped behind `--checks next` in 0.6.0 and is on by default from 0.7.0, the minor after
#: it (S's decision of 15 Oct 2026: the gate changes made after v0.5.0 follow GATE-13). A
#: promoted name has left `NEXT`, so `enabled` gives it in both modes; it stays named here, with
#: its TRD requirement, for the changelog and for `tests/unit/test_checks_next.py`.
PROMOTED: dict[str, str] = {
    # TRD VER-7a: checkMesh's cell aspect ratio computed from `constant/polyMesh` when the case
    # has no checkMesh log (`courant verify` and `courant preflight`). A behaviour: its finding,
    # `mesh.aspect_ratio`, is the one a checkMesh log has always raised, and stays on.
    "mesh.aspect_ratio.polymesh": "0.7.0",
    # TRD VER-16: each field held to its `residualControl` tolerance on a steady run (the
    # first-solve residual, capped at 1e-2); without it, every field against the flat 1e-3.
    "residual.control": "0.7.0",
    # TRD VER-16: the stalled-residual detector, its finding and the MINOR grade it gives
    # `convergence.residual`.
    "residual.stalled": "0.7.0",
    # TRD VER-17: a laminar run heated from below that its own Rayleigh number calls turbulent,
    # on a mesh too coarse to resolve it; with it, the case reader's `constant/g` (and the wall
    # faces it reads from the mesh files for a case with gravity).
    "physics.laminar_regime": "0.7.0",
    # TRD VER-18: the boundary flux balance (`conservation.mass_flux`), fed from the latest time's
    # `phi` (20 Oct 2026, after a false accept of a nozzle run out of balance: one rebuilt run,
    # an internal evaluation, not a measurement). A behaviour: off, the field reader does not
    # open `phi`, so the check skips as v0.5.0's did (0.6.0's default). Promoted on RM-37's
    # result (`PROMOTE_MASS_FLUX_PHI`; TRD GATE-13).
    MASS_FLUX_PHI: "0.7.0",
    # TRD VER-5.10 (ADP-1): a provisional accept on an OpenFOAM fork or version outside the set
    # the error rates were measured on (`courant.forks.MEASURED`) is `insufficient`. A behaviour:
    # the policy's rule, asked here by the pipeline. ADP-1's reading (`run.fork`,
    # `run.solver_version`) and its not-checked line are reporting, and on by default.
    "version.unsupported": "0.7.0",
    # TRD VER-3a: only an `End` after the log's last time step is the solver's, so a utility's
    # `End` ahead of a cut solver's output in one file no longer reads as the run's end (VER-5.16
    # then says insufficient where the default judges the cut run as finished). A behaviour.
    "log.end_after_last_step": "0.7.0",
    # TRD VER-21: the solver tables completed from v2406's own sources (`courant.case.solvers`:
    # `V2406_STEADY`, `V2406_ADAPTIVE`, `V2406_FIXED`, `V2406_NOT_ADAPTIVE`, `V2406_NOT_STEADY`;
    # held to the generated `tests/fixtures/solvers/v2406.json`), and an absent
    # `adjustTimeStep` read as v2406 reads it, false (`createTimeControls.H`). A behaviour: it
    # decides `is_steady`'s solver fallback, `honours_adjust_time_step` and `is_known`.
    "solvers.v2406": "0.7.0",
    # TRD VER-19: a local-time-stepping case (`ddtSchemes default localEuler`) judged on where
    # it ended up, as a steady one is (`CaseConfig.marches_to_steady_state`). A behaviour.
    "case.local_euler": "0.7.0",
    # TRD VER-20: a multi-region log's residuals keyed by the region the log names
    # (`field@region`), and its Courant number the largest region's, as the solver takes it. A
    # behaviour of the log parser (`LogParser(regions=True)`).
    "log.regions": "0.7.0",
    # TRD SPEC-13: conformance reads what ran, not only the deck: the solver from the log's
    # `Exec` line, the end time the run reached (and `stopAt`), the RAS/LES `turbulence`
    # switch and the model the log selected, a boundary value through the patch's type (an
    # entry the type does not apply is never compared), foamRun's module, and the mesh items
    # against the mesh the solver read (`courant.case.mesh.solver_mesh`), each block's split
    # counted on it and its scale with its extent; `courant spec init --checks next` drafts for
    # it. A behaviour of `courant.spec.as_run`.
    "conformance.as_run": "0.7.0",
    # TRD VER-17a: a RAS or LES case whose `turbulence` switch is off is laminar-equivalent for
    # the checks that ask `CaseConfig.is_laminar` (no model is solved). A behaviour.
    "case.turbulence_switch": "0.7.0",
    # TRD VER-16a: every field judged on its FIRST solve of the step, as residualControl reads
    # it, by the bar, the stall detector and the convergence block. A behaviour.
    "residual.first_solve": "0.7.0",
    # TRD PRE-5: `bc.overconstrained` asks the pressure the solver solves (`p_rgh` when the case
    # has it), and counts totalPressure, prghPressure and prghTotalPressure as fixing it.
    "bc.solved_pressure": "0.7.0",
    # TRD PRE-6: `physics.nu` on every viscosity the solver reads: each Newtonian phase's `nu`
    # (interFoam's `constant/transportProperties` subdictionaries), and a `mu` from
    # thermophysicalProperties through the density the case's equation of state gives
    # (`courant.case.viscosity`).
    "physics.nu.solved": "0.7.0",
    # TRD VER-3b: the case reader reads a dictionary whose value a `$macro` or an `#include`
    # writes as the solver reads it (`courant.spec.deck.expand`). A behaviour: off (0.6.0's
    # default) such a value is reported unread ("not assessed"), and no verdict moves.
    "case.directives": "0.7.0",
    # TRD VER-7b: the field checks read the fields at the latest time the log reached, not a
    # later directory another run left, and read `processor*/` when the ranks went further than
    # a partly reconstructed root. A behaviour: off (0.6.0's default) the same directory is read
    # as before, and "not assessed" says when it is not the log's.
    "fields.log_time": "0.7.0",
    # TRD VER-7b: a field the run wrote at the time the field reader judged and that was not
    # read (cut mid-write, a collated block, a file that could not be opened), or that a
    # decomposed case's ranks read only in part, caps the grade at B: a minor check,
    # `fields.unread` (`courant.verification.checks.fields_unread`). Off (0.6.0's default) "not
    # assessed" says it, and the grade is v0.5.0's.
    "fields.unread": "0.7.0",
}

#: {name: the minor release it ships in behind `--checks next`}. A name tagged "0.7.0" is on
#: by default from 0.8.0.
NEXT: dict[str, str] = {
    # TRD VER-13a: a declared steady solve's Courant number decides nothing, so its `cfl.*`
    # findings are not raised, `physics.laminar_unstable` holds it against no limit, and "not
    # assessed" says it was printed (`checkers.steady_courant`); with it, the case reader's
    # `ddtSchemes` entries and whether it read them whole (`CaseConfig.ddt_terms`,
    # `ddt_read_whole`). Behind the flag in 0.6.0 by GATE-13's rule for a new behaviour (the
    # orchestrator's decision within S's approval of 25 Oct 2026, 26 Oct 2026): every review of
    # it found a new false accept, and its one measured benefit is one false alarm, the heated
    # pipe's energy run. Held behind the flag for 0.7.0 too, re-tagged (the release preparation
    # of 0.7.0, 4 Nov 2026): it was never measured, so it waits for a measurement. A behaviour:
    # under the default every `cfl.*` check and `physics.laminar_unstable`'s Courant condition
    # are v0.5.0's, and the reader reads nothing for it.
    "cfl.steady": "0.7.0",
    # TRD VER-16c (S's decision on H-VER16-ENDTIME, 26 Oct 2026): a field's residual bar is its
    # `residualControl` tolerance clamped to [1e-3, 1e-2]. A tolerance below 1e-3 no longer
    # tightens the bar: the field is held to 1e-3, met as the flat bar is met, at the solve the
    # tolerance is read and at the solve the flat bar reads (`convergence._clamp`, `_readings`),
    # and the tolerance it did not reach is a note under "not assessed"
    # (`sentences.tolerances_unmet`), never a reason. A behaviour of
    # `residual.control` (`convergence.sets_bar`): under the default no residualControl is read
    # for the bar, and nothing changes.
    "residual.control.floor": "0.7.0",
    # TRD VER-5b (T-LAMINAR-REGIME, S's decision on H-LAMINAR-TRANSIENT, 26 Oct 2026): on a
    # time-marching laminar run `physics.laminar_unstable` defers to the case's own regime numbers
    # (`courant.case.regime.regime_numbers`: physics.laminar_regime's Rayleigh number, and a
    # Reynolds number where the case gives one unambiguously, `courant.case.reynolds`). Every
    # number below its flow's laminar bound sets the residual history aside, and the verdict says
    # why; one above its turbulent bound is cited. A behaviour of the finding: under the default
    # it is 0.6.0's, and the reader reads nothing for it.
    LAMINAR_REGIME: "0.7.0",
    # TRD VER-22 (T-ENERGY-BALANCE): a steady run's heat balance, `conservation.energy`, from the
    # per-patch integrals OpenFOAM's `wallHeatFlux` function object wrote
    # (`courant.verification._heat_reader`, `checks.energy_balance`), held to the boundary
    # balance's bars. A behaviour: under the default the reader is never called, so nothing of it
    # is read, and no result, reason or sentence of it appears.
    ENERGY_WALLHEATFLUX: "0.7.0",
    # TRD VER-3b (T-VER3B-ONE): the spec reader reads a dictionary as the solver reads it, as the
    # case reader does under `case.directives` (`courant.spec.deck.expand`, one expansion for
    # both): a `$name` is the definition nearest the entry, looked up outwards, an
    # `#includeIfPresent` is followed, a merged `$name;` read. A behaviour of `deck.read`
    # (`macros`), asked by `check_conformance` and `draft_spec`: under the default the spec
    # reader keeps 0.6.0's lookup (the file's top level, then its `#include`s), byte for byte.
    "spec.directives": "0.7.0",
    # TRD VER-3 (T-ZERO-UNLISTABLE): a `0/` the case reader cannot list or search is recorded as
    # unreadable (`CaseConfig.unreadable["0/"]`, said under "not assessed") and the rest of the
    # configuration is kept, so the setup rules that read `system/` and `constant/` run. A
    # behaviour of the case reader: under the default the listing's exception ends the reading,
    # the configuration is lost, and the verdict is 0.6.0's (an accept its fvSolution rejects).
    "case.zero_unlisted": "0.7.0",
    # TRD VER-3d (T-MULTI-REGION): a multi-region case (`constant/regionProperties`) read as its
    # solver reads it: `CaseConfig.regions` from the one reader of the file, each region's
    # `system/<region>/fvSchemes`, `fvSolution` and `0/<region>/` in place of the top level's
    # (its fvSolution read only for the outer correctors the solver reads there), for a solver
    # that reads regions (`_foam_reader.multi_region`, the one predicate)
    # (a stale copy there is not what ran), each region's checkMesh log (`log.checkMesh.<region>`),
    # and the verdict's multi-region sentence says what was read. A behaviour of the case reader
    # and the mesh reader: under the default a multi-region case is read as 0.6.0 read it.
    "case.regions": "0.7.0",
    # TRD VER-3 (T-MULTI-REGION): `postProcess` (and OpenFOAM.org's `foamPostProcess`) is a known
    # utility to `courant.cli.tail.find_log`, so a postProcess run after the solve, which prints
    # `Time =` and is newer, is not taken for the run's log. A behaviour of the log ranking.
    "log.post_process": "0.7.0",
    # T-MONITOR-WALLHEATFLUX (TRD VER-22's GATE-13 bullet, GATE-13): monitor discovery
    # (`courant.converge.series.discover`, asked there) lists a table whose rows carry a word
    # second as ignored, never as a monitor (`series.per_patch`): a header naming `patch` second
    # (wallHeatFlux's, `_heat_reader.COLUMNS`; yPlus, wallShearStress), or a word second in every
    # row (a patch name, or solverInfo's solver name). A behaviour, apart from VER-22: it moves
    # runs with no energy equation too. Under the default the discovery is v0.6.0's.
    "monitors.per_patch": "0.7.0",
    # T-EXECUTED-ONE (TRD VER-3, SB-11): SB-11's advisory `executed`
    # (`courant.verify.schema.executed_in_log`, asked there; read by the MCP `score_agent_output`
    # and SilentBench's `grade`, and through it the re-solver's `log_disagreement`) is the
    # verdict's reading of the end: `exitcode.verdict_exit_code` under the verdict's checks is 0
    # and the log parser reads no crash. A behaviour of its own, not VER-3a's 0.6.0 name: it
    # changes a 0.6.0 next output (a utility's End ahead of a solver killed by a signal: executed,
    # now not). Under the default it is v0.6.0's line rule.
    "silentbench.executed.verdict_end": "0.7.0",
}

if not PROMOTE_MASS_FLUX_PHI:                  # RM-37 refuted: VER-18 stays behind the flag
    NEXT[MASS_FLUX_PHI] = "0.7.0"
    del PROMOTED[MASS_FLUX_PHI]


def enabled(name: str, mode: str = "default") -> bool:
    """Whether a finding enters the verdict, or a behaviour runs, under this mode."""
    if mode not in MODES:
        raise ValueError(f"--checks is one of {', '.join(MODES)}, not {mode!r}")
    return mode == "next" or name not in NEXT


def only_enabled(finding_set, mode: str = "default"):
    """The findings that enter the verdict under this mode, in their order."""
    from courant.diagnosis.findings import FindingSet

    if mode == "next" or not NEXT:
        enabled("", mode)                     # still refuse an unknown mode
        return finding_set
    return FindingSet([f for f in finding_set if enabled(f.id, mode)])


__all__ = ["MODES", "NEXT", "PROMOTED", "enabled", "only_enabled"]
