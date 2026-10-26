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
(`tests/unit/test_checks_next.py`, against v0.5.0's documents).
"""

from __future__ import annotations

MODES = ("default", "next")

#: {name: the minor release it ships in behind `--checks next`}. S's decision of 15 Oct 2026:
#: the three gate changes made after v0.5.0 follow GATE-13.
NEXT: dict[str, str] = {
    # TRD VER-7a: checkMesh's cell aspect ratio computed from `constant/polyMesh` when the case
    # has no checkMesh log (`courant verify` and `courant preflight`). A behaviour: its finding,
    # `mesh.aspect_ratio`, is the one a checkMesh log has always raised, and stays on.
    "mesh.aspect_ratio.polymesh": "0.6.0",
    # TRD VER-16: each field held to its `residualControl` tolerance on a steady run (the
    # first-solve residual, capped at 1e-2); without it, every field against the flat 1e-3.
    "residual.control": "0.6.0",
    # TRD VER-16: the stalled-residual detector, its finding and the MINOR grade it gives
    # `convergence.residual`.
    "residual.stalled": "0.6.0",
    # TRD VER-17: a laminar run heated from below that its own Rayleigh number calls turbulent,
    # on a mesh too coarse to resolve it; with it, the case reader's `constant/g` (and the wall
    # faces it reads from the mesh files for a case with gravity).
    "physics.laminar_regime": "0.6.0",
    # TRD VER-18: the boundary flux balance (`conservation.mass_flux`), fed from the latest time's
    # `phi` (20 Oct 2026, after a false accept of a nozzle run out of balance: one rebuilt run,
    # an internal evaluation, not a measurement). A behaviour:
    # under the default the field reader does not open `phi`, so the check skips as v0.5.0's did.
    "conservation.mass_flux.phi": "0.6.0",
    # TRD VER-5.10 (ADP-1): a provisional accept on an OpenFOAM fork or version outside the set
    # the error rates were measured on (`courant.forks.MEASURED`) is `insufficient`. A behaviour:
    # the policy's rule, asked here by the pipeline. ADP-1's reading (`run.fork`,
    # `run.solver_version`) and its not-checked line are reporting, and on by default.
    "version.unsupported": "0.6.0",
    # TRD VER-3a: only an `End` after the log's last time step is the solver's, so a utility's
    # `End` ahead of a cut solver's output in one file no longer reads as the run's end (VER-5.16
    # then says insufficient where the default judges the cut run as finished). A behaviour.
    "log.end_after_last_step": "0.6.0",
    # TRD VER-21: the solver tables completed from v2406's own sources (`courant.case.solvers`:
    # `V2406_STEADY`, `V2406_ADAPTIVE`, `V2406_FIXED`, `V2406_NOT_ADAPTIVE`, `V2406_NOT_STEADY`;
    # held to the generated `tests/fixtures/solvers/v2406.json`), and an absent
    # `adjustTimeStep` read as v2406 reads it, false (`createTimeControls.H`). A behaviour: it
    # decides `is_steady`'s solver fallback, `honours_adjust_time_step` and `is_known`.
    "solvers.v2406": "0.6.0",
    # TRD VER-19: a local-time-stepping case (`ddtSchemes default localEuler`) judged on where
    # it ended up, as a steady one is (`CaseConfig.marches_to_steady_state`). A behaviour.
    "case.local_euler": "0.6.0",
    # TRD VER-20: a multi-region log's residuals keyed by the region the log names
    # (`field@region`), and its Courant number the largest region's, as the solver takes it. A
    # behaviour of the log parser (`LogParser(regions=True)`).
    "log.regions": "0.6.0",
    # TRD SPEC-13: conformance reads what ran, not only the deck: the solver from the log's
    # `Exec` line, the end time the run reached (and `stopAt`), the RAS/LES `turbulence`
    # switch and the model the log selected, a boundary value through the patch's type (an
    # entry the type does not apply is never compared), foamRun's module, and the mesh items
    # against the mesh the solver read (`courant.case.mesh.solver_mesh`), each block's split
    # counted on it and its scale with its extent; `courant spec init --checks next` drafts for
    # it. A behaviour of `courant.spec.as_run`.
    "conformance.as_run": "0.6.0",
    # TRD VER-17a: a RAS or LES case whose `turbulence` switch is off is laminar-equivalent for
    # the checks that ask `CaseConfig.is_laminar` (no model is solved). A behaviour.
    "case.turbulence_switch": "0.6.0",
    # TRD VER-16a: every field judged on its FIRST solve of the step, as residualControl reads
    # it, by the bar, the stall detector and the convergence block. A behaviour.
    "residual.first_solve": "0.6.0",
    # TRD PRE-5: `bc.overconstrained` asks the pressure the solver solves (`p_rgh` when the case
    # has it), and counts totalPressure, prghPressure and prghTotalPressure as fixing it.
    "bc.solved_pressure": "0.6.0",
    # TRD PRE-6: `physics.nu` on every viscosity the solver reads: each Newtonian phase's `nu`
    # (interFoam's `constant/transportProperties` subdictionaries), and a `mu` from
    # thermophysicalProperties through the density the case's equation of state gives
    # (`courant.case.viscosity`).
    "physics.nu.solved": "0.6.0",
    # TRD VER-3b: the case reader reads a dictionary whose value a `$macro` or an `#include`
    # writes as the solver reads it (`courant.spec.deck.expand`). A behaviour: by default such a
    # value is reported unread ("not assessed"), and no verdict moves.
    "case.directives": "0.6.0",
    # TRD VER-7b: the field checks read the fields at the latest time the log reached, not a
    # later directory another run left, and read `processor*/` when the ranks went further than
    # a partly reconstructed root. A behaviour: by default the same directory is read as before,
    # and "not assessed" says when it is not the log's.
    "fields.log_time": "0.6.0",
    # TRD VER-13a: a declared steady solve's Courant number decides nothing, so its `cfl.*`
    # findings are not raised, `physics.laminar_unstable` holds it against no limit, and "not
    # assessed" says it was printed (`checkers.steady_courant`); with it, the case reader's
    # `ddtSchemes` entries and whether it read them whole (`CaseConfig.ddt_terms`,
    # `ddt_read_whole`). Behind the flag by GATE-13's rule for a new behaviour (the
    # orchestrator's decision within S's approval of 25 Oct 2026, 26 Oct 2026): every review of
    # it found a new false accept, and its one measured benefit is one false alarm, the heated
    # pipe's energy run. A behaviour: under the default every `cfl.*` check and `physics.
    # laminar_unstable`'s Courant condition are v0.5.0's, and the reader reads nothing for it.
    "cfl.steady": "0.6.0",
    # TRD VER-7b: a field the run wrote at the time the field reader judged and that was not
    # read (cut mid-write, a collated block, a file that could not be opened), or that a
    # decomposed case's ranks read only in part, caps the grade at B: a minor check,
    # `fields.unread` (`courant.verification.checks.fields_unread`). By default "not assessed"
    # says it, and the grade is v0.5.0's.
    "fields.unread": "0.6.0",
}


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


__all__ = ["MODES", "NEXT", "enabled", "only_enabled"]
