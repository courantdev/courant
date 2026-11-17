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
ENERGY_FLOW = "conservation.energy.flow"
MASS_FLUX_PHI = "conservation.mass_flux.phi"
REGIONS_WHOLE = "case.regions.whole"
SPEC_READ_AS_SOLVER = "spec.read_as_solver"
SPEC_READ_WHOLE = "spec.read_whole"
CASE_MERGE_WITH_VALUE = "case.merge_with_value"
SPEC_TURBULENCE_BLOCK = "spec.turbulence_block"
RESIDUAL_NO_TIME_MARKERS = "residual.no_time_markers"
BOUNDS_ROUNDOFF = "bounds.negative_roundoff"
SPEC_LOCK_STRICT = "spec.lock_strict"
CLI_OPTIONS_STRICT = "cli.options_strict"
SPEC_DRAFT_AS_RUN = "spec.draft_as_run"
FOAM_GUARD = "case.foam_guard"
REGION_COURANT = "log.regions.courant"

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
#: by default from 0.8.0, one tagged "0.8.0" (added after 0.7.0 was released) from 0.9.0.
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
    # TRD VER-22 (c) (T-ENERGY-FLOW): a steady region open to flow is judged on the heat through
    # its walls plus the heat carried through its openings, by the flow (`phi` and the face `T`
    # the solver wrote, the thermo's `Cp`) and by conduction (v2406's boundary gradient), in
    # `courant.verification._flow_heat`. Its own name, so promoting the wall balance does not
    # promote it unmeasured. A behaviour of `_heat_reader`: off, an open region is not judged and
    # says 0.7.0's sentence, byte for byte.
    ENERGY_FLOW: "0.8.0",
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
    # Extended by the verdict-path audit (VPA-INGEST-1, 4 Nov 2026), still behind the flag: a
    # solver run with `-postProcess` is a utility's log too, and a log is ranked by every `Exec`
    # line it opens with, so `(blockMesh; simpleFoam) > log.run` is the solve's
    # (`tail._by_exec_lines`).
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
    # TRD VER-3e (T-MULTI-REGION-2): the rest of a multi-region case that `case.regions` reads as
    # regions, as its solver reads it: each region's `constant/<region>/thermophysicalProperties`
    # (`physics.nu` on a fluid region's `mu`, the region named by its place), its correctors (no
    # `residualControl`: no ESI CHT solver reads one), the fields its solver must read
    # (`missing`), a `0/<region>/` that cannot be listed said without `case.zero_unlisted`,
    # preflight's `MESH_REGIONS_PARTIAL`, and `mesh.solver_mesh` naming a single-region solver's
    # mesh beside a stray `regionProperties` (`_foam_reader.multi_region`). Its own name: 0.7.0
    # shipped `case.regions` without it, and the default set is fixed. A behaviour of the case
    # reader, the mesh reader and conformance: under the default nothing of it is read.
    REGIONS_WHOLE: "0.8.0",
    # The verdict-path audit (4 Nov 2026, research/notes/verdict-path-audit/FINDINGS.md), each
    # shipped behind the flag in the release after 0.7.0 and on by default no earlier than the
    # minor after it (GATE-13). Each changes a verdict only on an input no committed run holds.
    #
    # VPA-SPEC-1/3/4/6: the specification reader reads a value as the solver reads it
    # (`courant.spec.deck`): a patch reached by a group entry the solver applies before any
    # pattern is not read from the pattern (unread, VER-5.9); an inline Function1 (`table`,
    # `polynomial`, ...) is a function of time the reader does not evaluate (unread), never its
    # last number; a dimension set written in unit names (`[m^2/s]`) is read in SI base
    # exponents, never as its digits; a quoted word (`"kOmegaSST"`) is the word. A behaviour.
    SPEC_READ_AS_SOLVER: "0.8.0",
    # VPA-DRAFT-1: a turbulence model is drafted, and compared, only from the block the case's
    # `simulationType` selects (`courant.spec.check.draft_spec`, `courant.spec.as_run`); a laminar
    # case drafts none, and a model item whose block is not the one that ran is not compared.
    # Under it the model item and the "turbulence regime" item (`simulationType`) read the regime
    # the solver selected, from the log first when their file was read (review 0.7.0, A7-4): a
    # deck saying RAS with a log selecting laminar no longer conforms. A log line is the file's
    # only when no region's or phase's can be it (`AsRun._logged`): not beside a sibling named
    # for a phase constant/phaseProperties lists, or one that cannot be read, nor in a log naming
    # more than one model; a `.bak` beside a one-phase deck is a backup, and the log decides.
    SPEC_TURBULENCE_BLOCK: "0.8.0",
    # VPA-CHK-1: a log with no `Time =` line (VER-5.15's) is not read for a first solve by the
    # declared-tolerance bar either (`courant.verification.convergence.judge_residuals`): its one
    # step's first solve is iteration 1's, not the run's last.
    RESIDUAL_NO_TIME_MARKERS: "0.8.0",
    # VPA-CHK-2: `bounds.plausibility` holds a field that cannot go negative to round-off of its
    # own scale (`courant.verification.bounding.ROUNDOFF`, the one definition), not to a strict
    # `< 0`: a passive scalar's `T` of 0 to 1 that undershoots to -1e-6 is not a negative
    # temperature. A behaviour of the check: under the default it is 0.7.0's.
    BOUNDS_ROUNDOFF: "0.8.0",
    # VPA-ORDER-1, SEC070-18: a `--spec-sha256` that is not 64 hexadecimal digits, the empty string
    # and blanks included (an unset `$LOCK`), is refused, exit 4 and no verdict
    # (`ingest.lock_hash`, asked by `ingest.lock_of`). A behaviour of `courant verify` and
    # `courant spec check`: under the default an empty or blank lock is no lock and a malformed
    # one is VER-5.8, exit 3, as 0.7.0 gave. Constructed inputs, not a measured false accept.
    SPEC_LOCK_STRICT: "0.8.0",
    # VPA-CLI-5: `--report` other than md or html, and `--junit` without `--batch`, are refused,
    # exit 4 and no verdict. A behaviour of `courant verify`: under the default each is one warning
    # on stderr, and the verdict and its exit code are 0.7.0's.
    CLI_OPTIONS_STRICT: "0.8.0",
    # VPA-DRAFT-2: `courant spec init` leaves out an item outside `0/` whose reading as it ran
    # differs from the deck's (an endTime its `stopAt` does not stop at, a model whose `turbulence`
    # switch is off), so a draft conforms to its own case (`check.draft_spec`). A behaviour of the
    # drafter: under the default the draft is 0.7.0's, whose reference may not conform to it.
    SPEC_DRAFT_AS_RUN: "0.8.0",
    # The deck oracle (4 Nov 2026, research/notes/deck-oracle/FINDINGS.md), DO-3 and DO-4: the
    # specification reader reads a word value whole (`default CrankNicolson 0.9;` is "CrankNicolson
    # 0.9", not "0.9": `deck.parse_value`'s `whole`), and system/controlDict with its `#include`s
    # for the entries the end time is judged on (`as_run.control_text`, asked by `AsRun._end_time`
    # and `pipeline._ended_elsewhere`), so a `runTimeControl` behind an include is the case's own
    # stop. Its own name, not `spec.directives`, which 0.7.0 shipped without these and which goes
    # on by default at 0.8.0: each moves a conformance verdict, and was not in a release behind
    # the flag. Constructed inputs and a sweep of OpenFOAM's tutorials, not a measured false
    # alarm. A behaviour: under the default each reads as 0.7.0 reads it.
    SPEC_READ_WHOLE: "0.8.0",
    # The deck oracle, DO-2: a solver entry's `$name` merge written with a value (`"pcorr.*" { $p
    # tolerance 0.02; }`, which foamlib reads as one keyword, so the entry's solver and tolerance
    # are None) is noted as not read (`_Dictionary._inherited`). Reporting: `case.unread_value`
    # decides no verdict, rule or grade. Behind the flag because it changes a default verdict's
    # bytes on four stock tutorials, one of which has a committed verdict record
    # (`research/notes/tutorial-coverage/solves.json`, pimpleFoam_movingCone). Under the default it
    # is 0.7.0's: the merge is taken as read, and nothing says so.
    CASE_MERGE_WITH_VALUE: "0.8.0",
    # TRD VER-3f (the field-oracle sweep, FO-1, 4 Nov 2026): a `0/` field file written compressed
    # (`writeCompression on`: `setFields` leaves `0/T.gz` and no `0/T`) is read as OpenFOAM opens
    # it, `0/T` else `0/T.gz` (`reads.plain_or_gz`): the case reader keys its boundary types
    # under `T` and reads its held temperatures and `0/p`'s pressure from it
    # (`_foam_reader.read_case_config`'s `compressed`), and the spec reader reads `0/T.gz` where
    # it read the stale `0.orig/T` (`deck.read`'s `compressed`). A behaviour of both readers:
    # under the default a `.gz` is keyed `T.gz`, and `0/T` falls back to `0.orig/T`, as v0.7.0.
    "case.zero_compressed": "0.8.0",
    # TRD VER-3g (the field-oracle sweep, FO-2, 4 Nov 2026): `CaseConfig.boundary` keyed by the
    # mesh's patches, each `0/` field's `boundaryField` resolved as v2406's
    # `GeometricBoundaryField::readField` resolves it (`_foam_reader._by_patch`): a patch's own
    # name, then its groups (`wall`, `#includeEtc "caseDicts/setConstraintTypes"`), then `empty`,
    # then the patterns (`".*"`); a key naming no patch dropped. So `bc.overconstrained` sees a
    # patch whose p is fixed by a pattern and whose U by its name. A behaviour of the case reader:
    # under the default the keys are taken for patch names, as v0.7.0 took them.
    "case.boundary_patches": "0.8.0",
    # TRD VER-3h (the invariants at scale, INV6-F1, 5 Nov 2026): a file foamlib's parser would take
    # minutes or for ever to read is refused before foamlib sees it (`courant.case.foamguard`), so
    # a list cut mid-write gives a verdict. A behaviour of every reader that hands a file to
    # foamlib: under the default none asks, and such a file hangs as it did in 0.7.0 (review of
    # 0.7.0, A6-1). Its cost model has open gaps both ways (TRD VER-23).
    FOAM_GUARD: "0.8.0",
    # TRD VER-20 (review of 0.7.0, A5-1): under `log.regions` a `Region: <name> Courant Number`
    # line is read as that region's (`_RE_REGION_COURANT`) before the bulk pattern is tried, so a
    # region whose name does not end in a letter (`fluid1`, `fluid_2`, `fluid.`) no longer has
    # its number taken as the bulk one, which reset the step's Courant number to its own. A
    # behaviour of the log parser (`LogParser(region_courant_first=True)`), asked by the
    # pipeline: under the default such a line is read as 0.7.0 read it. It turns a false accept
    # into a reject, so it waits behind the flag (GATE-13).
    REGION_COURANT: "0.8.0",
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


__all__ = ["MODES", "NEXT", "PROMOTED", "REGIONS_WHOLE", "enabled", "only_enabled"]
