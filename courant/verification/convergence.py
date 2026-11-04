"""One definition of "this solve converged", shared by everything that asks.

There were two, and they disagreed. The verification suite asked whether the
residuals actually came down. The fix loop asked whether the solver *exited
cleanly* — `RunStatus.COMPLETED`, which means only "no NaN event, no crash,
exit code 0".

Those are not the same question, and a real OpenFOAM run showed exactly how far
apart they are. A lid-driven cavity with `deltaT` 100× too large was given the
wrong fix (a scheme change instead of a timestep reduction). On the re-run the
Courant number climbed to 4.6e+30 and the cumulative continuity error reached
-3.0e+14 — the solution was physically meaningless. But icoFoam's linear solver
still converged *within each timestep*, so no residual was ever NaN, the run
reached `endTime`, and it exited 0. `COMPLETED` ⇒ "converged" ⇒ the fix loop
reported **success**.

Verification caught it and graded the run C. The fix loop had already declared
victory. In autopilot that is the worst failure this product can have: it
applies a wrong fix, watches the solution turn to garbage, and tells the
engineer it is fixed.

So "converged" lives here, once, and both callers use it. The thresholds are
the verification suite's own — importing them rather than restating them is the
point, because two copies of a number are two numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from courant.events import Event, EventKind
from courant.events.diverged import diverged, is_residual_climb

#: Worst final initial-residual accepted. `TARGET` is clean convergence;
#: `LOOSE` is "converged enough to be worth a human's attention", and is the bar
#: a fix has to clear. Both are the verification suite's thresholds.
RESIDUAL_TARGET = 1e-3
RESIDUAL_LOOSE = 1e-2

#: A Courant number above this is not a marginal stability question, it is a
#: solution that has stopped meaning anything. Deliberately far above legitimate
#: practice — implicit transient solvers happily run at Co of 5–50 — so this
#: never argues with a working setup. The real diverged run reached 1.8e+31.
COURANT_ABSURD = 1.0e4

#: The explicit stability limit, and the bar a **fixed-timestep transient** run
#: has to clear once it has settled.
#:
#: Measured over every transient run in the certification corpus, and the
#: separation is not close. Sustained Courant over the final fifth of the run:
#:
#:     healthy    icoFoam/cavity 0.85, pisoFoam/cavity 0.24,
#:                planarContraction 0.18, rhoPimpleFoam/cavity 0.54,
#:                buoyantPimpleFoam/hotRoom 0.68
#:     faulted    pisoFoam 2.33, planarContraction 1.76, rhoPimpleFoam 3.73,
#:                icoFoam 8.92, hotRoom 1868
#:
#: Every healthy fixed-dt run is under 1; every broken one is over it. That is
#: not a fitted threshold, it is the textbook criterion the cases themselves
#: respect.
COURANT_STABILITY = 1.0

#: How far above its *own* declared `maxCo` an adaptive run may sit.
#:
#: A case with `adjustTimeStep yes` states its own stability criterion, and the
#: solver lands a little over it because it cannot subdivide a step exactly.
#: `interFoam/damBreak` declares maxCo 1 and settles at 1.083 — that is the
#: timestep controller working, not a violation, and comparing it to a fixed
#: constant would call a healthy tutorial broken.
COURANT_ADAPTIVE_SLACK = 1.35



def courant_limit(config) -> tuple[float, str]:
    """The Courant number this run must settle under, and why (TRD VER-13). One definition.

    A fixed-timestep run is held to the textbook `COURANT_STABILITY`. A run that controls its
    own step (`adjustTimeStep yes` with a declared `maxCo`) is held to the criterion its author
    declared, with the controller's slack. `solve_converged` and `CourantChecker` both ask
    here, so the convergence verdict and the `cfl.*` findings cannot disagree about it. Before
    this function they did. The checker held damBreak to 1.0 while the convergence verdict held
    it to its maxCo, and an unmodified tutorial was a false alarm (RRD RM-01 v0 to v3).

    A declared maxCo may tighten the limit. It may raise it past `COURANT_STABILITY` only when
    PIMPLE iterates outer correctors (`nOuterCorrectors` > 1), which is what lets a transient
    solve sit above Courant 1. With OpenFOAM's default of 1 the loop is PISO, and a maxCo of 10
    is the setup error itself, not the run's criterion: `interFoam_damBreak::max_co_too_high`
    settled at 1.48 and was held to 13.5, so it passed with no finding (certification, 0.5.0).
    """
    adaptive = bool(getattr(config, "adjust_timestep", False)) if config is not None else False
    declared = getattr(config, "max_co", None) if adaptive else None
    if declared:
        outer = getattr(config, "n_outer_correctors", None) or 1
        if declared > COURANT_STABILITY and outer <= 1:
            why = (f"the stability limit of {COURANT_STABILITY:g} (its maxCo of {declared:g} "
                   f"is above it, and nOuterCorrectors {outer} has no outer iteration to "
                   f"allow that)")
            return COURANT_STABILITY * COURANT_ADAPTIVE_SLACK, why
        return declared * COURANT_ADAPTIVE_SLACK, f"its own maxCo of {declared:g}"
    return COURANT_STABILITY, f"the stability limit of {COURANT_STABILITY:g}"

#: Volume-fraction fields. Their initial residual measures how far the
#: *interface moved* since the previous step, not how far the solution is from
#: satisfying its equations — and on a transient free-surface run the interface
#: never stops moving, which is the entire point of the simulation.
#:
#: Including them makes the convergence verdict permanently negative on healthy
#: multiphase work. Measured on an unmodified `interFoam/damBreak`: `p_rgh`
#: settles to 5.3e-04 while `alpha.water` sits at 1.08e-02, so the run was
#: judged not-converged on the one field that was never going to settle. Every
#: customer running VOF would have been told their correct run had failed.
#:
#: Safe to exclude, and for a physical reason rather than a convenient one:
#: MULES bounds alpha in [0, 1] every step, so it cannot run away unnoticed. The
#: same fact justifies excluding it from the divergence-slope fit in
#: `courant/watcher/anomaly.py`.
#:
#: Reachable only since the parser learned to read dotted field names — before
#: that `alpha.water` was invisible and the question never arose.
def _is_phase_fraction(field: str) -> bool:
    lowered = field.lower()
    return lowered.startswith("alpha.") or lowered == "alphas"


def _finite_items(mapping: dict[str, float | None]) -> dict[str, float]:
    return {name: v for name, v in mapping.items()
            if v is not None and not math.isnan(v) and not math.isinf(v)
            and not _is_phase_fraction(name)}


def final_residuals(events: list[Event]) -> dict[str, float]:
    """Each field's initial residual on the last step that reported any (phase fractions,
    NaN and infinity dropped, as `worst_final_residual` drops them)."""
    for event in reversed(events):
        if event.kind is not EventKind.RESIDUAL:
            continue
        values = _finite_items(event.payload.initial_residuals or {})
        if values:
            return values
    return {}


def worst_final_residual(events: list[Event]) -> float | None:
    """The largest initial residual of the last step that reported any.

    "Initial residual" rather than final: the initial residual of a step is how
    far the *solution* was from satisfying the equations before that step's
    linear solve. A final residual near zero only says the linear solver did its
    job, which it does right up to the point the physics is nonsense — as the
    run described in the module docstring demonstrates.
    """
    values = final_residuals(events)
    return max(values.values()) if values else None


# ------------------------------------------------------------------ the residual bar per field
#
# S's decision (RRD RM-35, 13 Oct 2026): the bar follows each field's declared `residualControl`
# tolerance, and only together with a detector of its own for a stalled residual history (TRD
# VER-16). Both live here. `ResidualAnalyzer` and `ConvergenceCheck` both ask `judge_residuals`,
# so with a bar per field they keep the agreement they had on the flat bar: the finding fires
# exactly when the check is not OK.

#: OpenFOAM prints a vector field's solve one component at a time (`Solving for Ux`), and
#: residualControl checks the field (`U`) against its largest solved component. Only `U` is
#: treated so: a scalar whose name happens to end in x, y or z is its own field.
_VECTOR_COMPONENTS = {"Ux": "U", "Uy": "U", "Uz": "U"}

#: Pressure fields. SIMPLE-family solvers relax them as fields (`p.relax()` in pEqn.H, from
#: `relaxationFactors/fields`), never as equations, so a catch-all `".*"` under `equations`
#: does not reach the pressure matrix or the residual printed for it.
_PRESSURE = frozenset({"p", "p_rgh"})


def _openfoam_name(field: str) -> str:
    """The field name OpenFOAM looks `field` up under (`Ux` -> `U`)."""
    return _VECTOR_COMPONENTS.get(field, field)


def _is_pattern(key: str) -> bool:
    return len(key) >= 2 and key[0] == key[-1] == '"'


def _first(keys: list[str], name: str) -> tuple[int | None, tuple[int, ...]]:
    """One dictionary lookup over `keys`, in the order given: a quoted keyword is a regular
    expression that must match the whole name (OpenFOAM's `wordRe`), anything else is the name
    itself. The keyword is the case's, so it is matched with bounded work (`courant.case.wordre`;
    Python's `re` backtracks exponentially on `"((((.*)*)*)*)*!"` against `epsilon`). (index of
    the first that names `name`, or None; indices before it of patterns not evaluated: a
    lookaround, a backreference, past the matcher's bound or its budget.)"""
    from courant.case import wordre

    return wordre.first_match(keys, name,
                              pattern=lambda key: key[1:-1] if _is_pattern(key) else None)


def _control(config) -> dict[str, float]:
    """The `residualControl` this case's solver reads as its run criterion, or {}."""
    if config is None:
        return {}
    from courant.case.solvers import residual_control_loop

    loop = residual_control_loop(getattr(config, "solver", None))
    controls = getattr(config, "residual_control", None) or {}
    return dict(controls.get(loop) or {}) if loop else {}


def _control_lookup(config, field: str
                    ) -> tuple[list[tuple[str, float]], int | None, tuple[int, ...]]:
    """The one `residualControl` lookup of `field`: (the entries with a finite tolerance, in
    declaration order; the index of the first that names the field, or None; the indices of the
    keywords before it the matcher did not evaluate). `_candidates` and `keywords_not_evaluated`
    read it."""
    name = _openfoam_name(field)
    entries = [(key, float(tol)) for key, tol in _control(config).items()
               if tol is not None and math.isfinite(tol)]
    hit, unsure = _first([key for key, _ in entries], name)
    return entries, hit, unsure


def _candidates(config, field: str) -> tuple[tuple[float, ...], bool] | None:
    """(every `residualControl` tolerance OpenFOAM may apply to `field`, the matching entry's
    last when one matches; whether one certainly applies), or None if none can.

    From the dictionary the solver's algorithm reads (`solvers.residual_control_loop`).
    `solutionControl::applyToField` returns the first entry, in declaration order, whose name
    matches the field. A tolerance that is not a finite number is no tolerance (a sidecar can
    carry NaN); zero or a negative number is a tolerance no residual can meet, as in OpenFOAM,
    whose test is `residual < tolerance`. A keyword before the match that the matcher did not
    evaluate may be the entry OpenFOAM applies, so each of them and the match (skipped,
    `"(?=U)U" 1e-6; ".*" 1e-2;` held U to 1e-2). With no entry certain to match, one applies
    only if such a keyword names the field: not certain (the review of the 0.6.0
    hard-proofing, 22 Oct 2026, found the flat bar then said to be the case's tolerance).
    """
    entries, hit, unsure = _control_lookup(config, field)
    if hit is None and not unsure:
        return None
    may_be = [entries[i][1] for i in unsure]
    if hit is not None:
        may_be.append(entries[hit][1])
    return tuple(may_be), hit is not None


def _tolerance(config, field: str) -> tuple[float, bool] | None:
    """(the strictest `residualControl` tolerance OpenFOAM may apply to `field`, whether an entry
    certainly matches the field), or None if none can (`_candidates`). The second says only
    that the flat bar is no outcome: the strictest may still be a keyword's the matcher did not
    evaluate, before that entry, so it does not say the tolerance is the case's
    (`judge_residuals` asks `_candidates` itself for that; T-FOLLOWUP-MINORS (2))."""
    found = _candidates(config, field)
    return None if found is None else (min(found[0]), found[1])


def declared_tolerance(config, field: str) -> float | None:
    """The `residualControl` tolerance OpenFOAM applies to `field`, or None if none does
    (`_tolerance`): the strictest it may apply, and, with no entry certain to match, no looser
    than the flat bar, which applies if none does."""
    found = _tolerance(config, field)
    if found is None:
        return None
    tol, certain = found
    return tol if certain else min(tol, RESIDUAL_TARGET)


def sets_bar(tolerance: float, checks: str = "default") -> bool:
    """Whether a `residualControl` tolerance sets its field's residual bar (TRD VER-16c). One
    definition, which `_clamp` asks for `residual_bar` and `judge_residuals` both.

    Under `residual.control.floor` (behind `--checks next`; S's decision on H-VER16-ENDTIME, 26
    Oct 2026) only a tolerance at or above `RESIDUAL_TARGET` does: a declared tolerance may
    loosen the bar, to `RESIDUAL_LOOSE` at most, and never tightens it. A tolerance below the
    target is the solver's stopping rule, and a run that never meets it runs on to its endTime:
    v2406's `simpleControl::loop` ends the run early only when `criteriaSatisfied()` holds, and
    otherwise returns `runTime.loop()` (`src/finiteVolume/cfdTools/general/solutionControl/
    simpleControl/simpleControl.C`, lines 136 to 161, read in the pinned image). So a tolerance
    set far below what the run reaches (1e-8, 1e-9) makes it run for its whole budget, which
    says nothing of whether the run converged; such a field is held to the flat bar's value and
    met as the flat bar is met (`_clamp`), and the tolerance it did not reach is a note
    (`UnmetTolerance`). Without the flag every tolerance sets the bar, tighter ones too (VER-16
    as 0.6.0 ships it)."""
    from courant.verify.checks import enabled

    return tolerance >= RESIDUAL_TARGET or not enabled("residual.control.floor", checks)


def _clamp(tolerance: float, checks: str = "default") -> tuple[float, bool]:
    """(the bar, whether it is met strictly) a `residualControl` tolerance holds its field to.
    One definition, applied by `judge_residuals` to every tolerance a field may have
    (`_readings`) and by `residual_bar`.

    A tolerance that sets the bar (`sets_bar`) is the bar, met strictly as OpenFOAM meets it
    (`residual < tolerance`, `_within`), and never looser than `RESIDUAL_LOOSE`, the edge of the
    band the gate has always graded MINOR: past it a residual is MAJOR whatever the case says,
    or `residualControl { p 1; }` would make any run converge on its second iteration. Under
    `residual.control.floor` (VER-16c, behind `--checks next`) one below `RESIDUAL_TARGET` holds
    the field to the target, met as the flat bar is met (`<=`): the tolerance clamped to
    [`RESIDUAL_TARGET`, `RESIDUAL_LOOSE`]. Either is read at the solve the tolerance is read,
    the field's first of the step; one below the target at the flat bar's solve as well
    (`_readings`). On a local-time-stepping run a tolerance at or above the target is read at the
    first solve, and the step's last solve, where the flat bar reads it, is MAJOR past
    `RESIDUAL_LOOSE` too (`judge_residuals`; the final review of 519ab28, T-FOLLOWUP-MINORS
    (8))."""
    if sets_bar(tolerance, checks):
        return min(tolerance, RESIDUAL_LOOSE), True
    return RESIDUAL_TARGET, False


def residual_bar(config, field: str, checks: str = "default") -> float:
    """The bar `field`'s `residualControl` tolerance holds it to on this run: `_clamp` of
    `declared_tolerance`, the clamp `judge_residuals` applies. For a field whose tolerance
    certainly applies it is the bar the judgement decides (`ResidualJudgement.bars`,
    `tests/unit/test_residual_floor.py`); otherwise (a keyword the matcher does not evaluate may
    give it a tolerance, or the flat bar reads another solve as well) the strictest bar a reading
    of it holds it to, from the one definition `judge_residuals` reads (`_bars`;
    T-FOLLOWUP-MINORS (3)), which judges each reading and keeps the one that decides. The gate
    asks `judge_residuals`.

    On a steady run whose solver reads a `residualControl` tolerance for the field, that
    tolerance, clamped (`_clamp`). Otherwise the flat `RESIDUAL_TARGET`, as before: with no
    config, on a transient run, for a solver with no run-level `residualControl`, and for a
    field the case does not declare. "Steady" is `CaseConfig.marches_to_steady_state` under
    `checks`: under `next` a local-time-stepping run too (TRD VER-19), so an OpenFOAM.org
    `foamRun` LTS case is held to its PIMPLE run criterion. The tolerance is read whichever
    mode, as `residual.control` reads it; the bar per field itself is behind `--checks next`
    (`judge_residuals`), so under the default no verdict asks for it.
    """
    if config is None or not config.marches_to_steady_state(checks):
        return RESIDUAL_TARGET
    found = _candidates(config, field)
    if found is None:
        return RESIDUAL_TARGET
    return min(bar for bar, _, _ in _bars(found, checks))


def _within(value: float, bar: float, declared: bool) -> bool:
    """A declared tolerance is met as OpenFOAM meets it, strictly (`residual < tolerance`); the
    flat bar keeps the gate's own `<=`, unchanged."""
    return value < bar if declared else value <= bar


def _equation_factors(config) -> list[tuple[str, float | None]]:
    """(key, factor) of the dictionary OpenFOAM relaxes equations from, in order
    (`CaseConfig.relaxation_equations`). A configuration captured before the reader kept it (a
    sidecar) gives it from the flattened map: the `equations` keys, or with no scoped key at
    all, the flat ones; a `fields` key written after an `equations` key of the same name has
    then overwritten it, which only a configuration read again can tell."""
    eqn = getattr(config, "relaxation_equations", None)
    if eqn is not None:
        return list(eqn.items())
    factors = getattr(config, "relaxation_factors", None) or {}
    scopes = getattr(config, "relaxation_scopes", None) or {}
    scope = "equations" if any(scopes.get(k) for k in factors) else ""
    return [(k, v) for k, v in factors.items() if scopes.get(k, "") == scope]


def _relaxation_lookup(config, field: str):
    """The one lookup of `field`'s equation relaxation, `dictionary::csearch`
    (`wordre.csearch`): (the dictionary's (key, factor) in order, `_equation_factors`; each key
    as (its text, whether it is a pattern); the indices of the entry found; the indices of the
    patterns tried before it that the matcher did not evaluate). `equation_relaxation` and
    `keywords_not_evaluated` read it."""
    from courant.case import wordre

    field, eqn = _in_region(config, field)
    keys = [(key[1:-1] if _is_pattern(key) else key, _is_pattern(key)) for key, _ in eqn]
    found, unsure = wordre.csearch(keys, _openfoam_name(field), dictionaries=False)
    return eqn, keys, found, unsure


def _in_region(config, field: str) -> tuple[str, list]:
    """(the field looked up, the equation factors it is looked up in). A multi-region residual
    (`h@air`, VER-20's key under `--checks next`) is relaxed by its region's own fvSolution, which
    the case reader keys by the region's place (`h@region1`, VER-3d): the region's entries alone,
    their suffix taken off, are the dictionary looked in (the review of T-MULTI-REGION, m3). A
    residual of no region the case names finds none."""
    eqn = _equation_factors(config)
    regions = getattr(config, "regions", None) or []
    base, at, region = field.rpartition("@")
    if regions and at:
        from courant.case.config import region_pseudonym

        tag = "@" + region_pseudonym(regions, region)
        return base, [(k[:-len(tag)], v) for k, v in eqn if k.endswith(tag)]
    return field, eqn


def keywords_not_evaluated(config, field: str, *, relaxation: bool = False) -> tuple[str, ...]:
    """The keywords the matcher did not evaluate that may decide `field`'s `residualControl`
    tolerance (`_control_lookup`), or with `relaxation` its equation relaxation factor
    (`_relaxation_lookup`): the lookups `judge_residuals` and the stall detector make, so each
    is held to the strictest value any of them may give. Each as 'residualControl "<keyword>"'
    or 'relaxationFactors "<keyword>"', the case's own text: for local use, never a sentence
    (`sentences.keywords_not_evaluated` counts them). Empty when nothing is in doubt."""
    if config is None:
        return ()
    if relaxation:
        if field in _PRESSURE:
            return ()
        eqn, _, _, unsure = _relaxation_lookup(config, field)
        keys = [eqn[i][0] for i in unsure]
        prefix = "relaxationFactors"
    else:
        entries, _, unsure = _control_lookup(config, field)
        keys = [entries[i][0] for i in unsure]
        prefix = "residualControl"
    return tuple(f"{prefix} {key}" for key in dict.fromkeys(keys))


def equation_relaxation(config, field: str) -> float:
    """The implicit (equation) relaxation factor OpenFOAM applies to `field`'s matrix; 1.0 when
    there is none.

    As v2406's `solution` reads it (`src/OpenFOAM/matrices/solution/solution.C`): `read` takes
    `relaxationFactors/equations` whole, or with neither `fields` nor `equations` the flat
    `relaxationFactors`, and nothing else (`_equation_factors`); `relaxEquation` looks the name
    up in it and, when nothing names it, takes its `default` entry (`findEntry("default",
    LITERAL)`: the key whose text is `default`). `relaxationFactors/fields` relaxes the field
    after its solve and leaves the matrix alone, so it is not this. Pressure is never
    equation-relaxed (`_PRESSURE`). The lookup is `dictionary::csearch` (`wordre.csearch`, the
    one `deck.block` makes too; a second one here took a quoted `"U"` for a pattern). The
    reader once flattened `fields` and `equations` into one map, so `fields { U 0.9; }` written
    after `equations { U 0.05; }` gave U 1.0, and never read the `default`: both hid the
    under-relaxation below 0.1 the stall detector fires on (the review of the 0.6.0 follow-ups).
    """
    if config is None or _in_region(config, field)[0] in _PRESSURE:
        return 1.0
    eqn, keys, found, unsure = _relaxation_lookup(config, field)
    # A value written last: `"U"` and `U` are one entry in v2406's dictionary, and the map holds
    # them as two keys in the order of each one's last writing (foamlib keeps a key written again
    # at its last place), so the later key is the one standing (T-FOLLOWUP-MINORS (6)).
    standing = {text: eqn[i][1] for i, (text, _) in enumerate(keys)}
    # The entry found, or with none the default (absent: no relaxation); and each pattern not
    # evaluated that may be the one found instead.
    may_be = [standing[keys[found[-1]][0]] if found else standing.get("default")]
    may_be += [standing[keys[i][0]] for i in unsure]
    # A factor outside (0, 1], or one not read, relaxes nothing (`fvMatrix::relax` returns on
    # alpha <= 0). Of several it may be, the strictest: the smallest deflates the printed
    # residual most (`Stall`).
    return min(a if a is not None and math.isfinite(a) and 0.0 < a <= 1.0 else 1.0
               for a in may_be)


#: One order of magnitude: the margin a plateaued field needs above its bar once the
#: relaxation's deflation is undone (`Stall`). Undoing it is a bound good to an order of
#: magnitude, so less than an order above the bar is inside its own uncertainty. Not tuned: it
#: is the unit in which convergence is counted (Celik et al.'s orders dropped, `converge.steady`).
STALL_ORDERS = 1


@dataclass(frozen=True)
class Stall:
    """A plateaued residual that under-relaxation hides under its bar.

    Two facts, both from what the solver read and printed, about one field:

    1. **Plateaued**, by the one definition the convergence block also uses
       (`converge.steady.residual_summary(...).plateaued`, M1-P-2): over the trailing window
       (`window_width`: 200 iterations, or a tenth of the run if longer) the residual's drift
       is under `PLATEAU_DRIFT` (0.2) of its final value. So `residual.stalled` names a field
       the convergence block reports as `CONV.RESIDUAL_PLATEAU`, never one it calls falling.
    2. **Deflated.** OpenFOAM's printed initial residual is normalised by the matrix being
       solved. `fvMatrix::relax(alpha)` first raises the diagonal to at least the sum of the
       off-diagonal magnitudes (ESI), then divides it by alpha and moves the difference to the
       source: at the current iterate the residual's numerator is unchanged, while the
       normalisation factor (`lduMatrix::solver::normFactor`, sums over |A psi - A avg| and
       |b - A avg|, both carrying the enlarged diagonal) grows, by at most about 1/alpha, as
       far as that diagonal dominates it. The printed residual of a relaxed equation can read
       up to about a factor alpha low, and `residualControl` compares that number. Dividing by
       alpha is therefore an order-of-magnitude bound, not a correction. Undone, the field sits
       at least an order of magnitude above its bar: `residual >= 10 ** STALL_ORDERS * alpha *
       bar`.

    Together they say the solver's own "converged" can be spurious: the iteration crept under
    a criterion the relaxation had loosened. It is asked only when every field is within its
    bar (`judge_residuals`), so the second fact needs `alpha <= residual / (10 * bar) < 0.1`:
    it can fire only on an equation relaxed below 0.1, never at OpenFOAM's usual 0.7 to 0.95.
    """

    window: int                 # iterations in the trailing window
    steps: int                  # residual steps of this field in the run
    field: str                  # the plateaued field the deflation puts furthest above its bar
    residual: float             # its final initial residual, as compared with its bar
    relaxation: float           # its equation relaxation factor
    bar: float                  # its bar, as the judgement decided it (`judge_residuals`)
    drift_rel: float            # its trailing-window drift over its final value (M1-P-2)


def stall_sentence(s: Stall) -> str:
    """The stall in one sentence, with the numbers an engineer can check against the log."""
    return (f"{s.field}'s residual plateaued over the last {s.window} of {s.steps} iterations "
            f"(drift {s.drift_rel:.2g} of its final value), and its final {s.residual:.3g} under "
            f"an equation relaxation of {s.relaxation:g} is about {s.residual / s.relaxation:.3g} "
            f"unrelaxed, more than an order above its bar of {s.bar:g}")


def reads_first_solve(config, checks: str = "default", *, events=None) -> bool:
    """Whether every field's residual at a step is its FIRST solve's (TRD VER-16a): on a steady
    run, as residualControl reads it, under `residual.first_solve` (behind `--checks next`).
    One definition, for the bar (`judge_residuals`), the stall detector and the convergence
    block's series (`courant.converge.series.assess_case`).

    Steady only. On a time-marching run a field solved several times a step (PISO or PIMPLE
    correctors) starts each step from the last step's solution, so its first solve measures how
    far the step moved the field, not how far the solve is from converging: the healthy
    interFoam damBreak's first `p_rgh` solve sits above 1e-3 while its last settles, and
    reading it rejected that run (`physics.laminar_unstable`, found on the committed corpus).
    With no configuration the regime is unknown, and the last solve is read, as in v0.5.0.

    Nor with no step: `events` whose residuals carry no step number, because the log has no
    `Time =` line (`courant.events.time_markers.no_time_markers`, VER-5.15), have no first solve of a
    step, so the last solve is read, as 0.6.0's default read it. Read as one step, the first
    iteration's residuals stood as the run's final ones and rejected a healthy pitzDaily log
    stripped of its `Time =` lines (VER-5.3, grade C) where VER-5.15 says insufficient (the
    release preparation of 0.7.0, which turned VER-16a on by default)."""
    from courant.events.time_markers import no_time_markers
    from courant.verify.checks import enabled

    return (enabled("residual.first_solve", checks) and config is not None
            and config.is_steady(checks)
            and not (events is not None and no_time_markers(events)))


def stalled(events: list[Event], config, final: dict[str, float],
            bars: dict[str, float], checks: str = "default") -> Stall | None:
    """The stall `Stall` describes, on a steady run, or None. `final` and `bars` are the
    judgement's own (`judge_residuals`), so the bar compared is the bar decided, and the history
    is read at the solve the bar reads (the first, under `residual.first_solve`, VER-16a).

    Steady by construction only (`CaseConfig.is_steady`), never a local-time-stepping run: the
    deflation bound is the equation relaxation `equation_relaxation` reads, and an LTS PIMPLE
    loop's final outer corrector, every solve with one corrector (every v2406 LTS tutorial),
    relaxes with the `<field>Final` factors instead."""
    if config is None or not config.is_steady(checks):
        return None
    from courant.converge.series import from_events
    from courant.converge.steady import residual_summary, window_width

    series, _, _ = from_events(                     # the convergence block's own series
        events, first_solve=reads_first_solve(config, checks, events=events))
    best: tuple[float, Stall] | None = None
    for name in sorted(final):
        values = series.get(name)
        bar = bars[name]
        if not values or bar <= 0.0:
            continue
        summary = residual_summary(name, values)
        if summary.plateaued is not True:
            continue
        alpha = equation_relaxation(config, name)
        excess = final[name] / (alpha * bar)
        if excess >= 10.0 ** STALL_ORDERS and (best is None or excess > best[0]):
            best = (excess, Stall(window=window_width(summary.n), steps=summary.n, field=name,
                                  residual=final[name], relaxation=alpha, bar=bar,
                                  drift_rel=summary.drift_rel))
    return best[1] if best else None


@dataclass(frozen=True)
class UnmetTolerance:
    """A `residualControl` tolerance below `RESIDUAL_TARGET` that a field did not reach (TRD
    VER-16c, `residual.control.floor`): reported as a note, never judged (`sets_bar`).

    Not reached as OpenFOAM tests it: v2406's `simpleControl::criteriaSatisfied` compares the
    field's first solve of the iteration, the largest component of a vector
    (`solutionControl::maxTypeResidual`, `residuals.first() = cmptMax(sp.first()
    .initialResidual())`, solutionControl.C line 232), with `residuals.first() < absTol`, strictly
    (simpleControl.C lines 70 to 71). The gate reads that first solve at the log's last step
    (`ResidualPayload.first_initial_residuals`), one component at a time."""

    field: str          # as the log names its solve: "Ux"
    residual: float     # its first solve's initial residual at the last step
    tolerance: float    # the tolerance, the strictest OpenFOAM may apply (`_tolerance`)
    #: Whether that tolerance is the value of the entry that certainly matches the field
    #: (`_candidates`). A keyword before that entry the matcher did not evaluate may pre-empt it
    #: with another value, and does not set this one (the final review of T-VER16C: under
    #: `"(?=U)U" 1e-2; U 1e-9;` U's 1e-9 was said to be the keyword's). Otherwise, with no entry
    #: certain to match or the strictest value a keyword's, the note says such a keyword may set
    #: it.
    certain: bool


@dataclass(frozen=True)
class ResidualJudgement:
    """Every field's final residual against its bar, and the stall, decided once."""

    final: dict[str, float]     # per field, the value its bar is compared with
    bars: dict[str, float]
    over: tuple[str, ...]       # fields not within their bar
    field: str                  # the deciding field: the furthest over (or nearest to) its bar
    residual: float             # its final residual
    bar: float                  # its bar
    #: The largest final residual over every reading of each field: a field only a keyword not
    #: evaluated may give a tolerance, or under VER-16c one whose tolerance is below the target,
    #: is read at its first solve and where the flat bar reads it, and past `RESIDUAL_LOOSE` on
    #: either it is MAJOR (`ConvergenceCheck`), whichever reading decides its bar (the review of
    #: the 0.6.0 follow-ups found 2e-2 then 5e-3 graded MINOR).
    worst: float
    #: Whether any field's bar came from residualControl, certainly. Under
    #: `residual.control.floor` (VER-16c) a field whose tolerance is below the target is held to
    #: the target, met as the flat bar is met, and does not count.
    declared: bool
    field_declared: bool        # whether the deciding field's did
    stall: Stall | None
    #: Whether any field's bar is a tolerance only a residualControl keyword the matcher does not
    #: evaluate may give (`_tolerance`, not certain): never said to be declared.
    unsure: bool = False
    field_unsure: bool = False  # whether the deciding field's is
    #: VER-16c: the residualControl tolerances below the target that a field did not reach, in
    #: the fields' order; judged by nothing, reported (`sentences.tolerances_unmet`).
    unmet: tuple[UnmetTolerance, ...] = ()
    #: TRD VER-3b: per field as OpenFOAM names it, in order, the residualControl and relaxation
    #: keywords the matcher did not evaluate that may decide its bar or its stall
    #: (`keywords_not_evaluated`): it was held to the strictest value they may give. Read only
    #: where the judgement reads the keywords, under `--checks next`; reported, never judged
    #: (`sentences.keywords_not_evaluated`).
    unevaluated: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def converged(self) -> bool:
        return not self.over and self.stall is None

    @property
    def per_field(self) -> bool:
        """Whether a field is held to a bar other than the flat target, so a reason names the
        deciding field and its bar rather than the worst residual and the target."""
        return self.declared or self.unsure

    def bar_words(self) -> str:
        """Where the deciding field's bar came from, for a reason's sentence."""
        if self.field_declared:
            return f"its residualControl {self.bar:g}"
        if self.field_unsure:
            return (f"{self.bar:g}, which a residualControl keyword the reader does not evaluate "
                    "may set")
        return f"the target {self.bar:.0e}"


def _last_step(events: list[Event]):
    """The payload of the last step that reported a finite residual (`final_residuals`'s)."""
    for event in reversed(events):
        if event.kind is not EventKind.RESIDUAL:
            continue
        if _finite_items(event.payload.initial_residuals or {}):
            return event.payload
    return None


def _further_over(a: tuple[float, float], b: tuple[float, float]) -> int:
    """Compare (residual, bar) pairs by residual / bar without dividing: a bar of zero or less
    (never met) ranks above any positive bar."""
    (ra, ba), (rb, bb) = a, b
    if (ba <= 0.0) != (bb <= 0.0):
        return 1 if ba <= 0.0 else -1
    if ba <= 0.0:
        return (ra > rb) - (ra < rb)
    lhs, rhs = ra * bb, rb * ba
    return (lhs > rhs) - (lhs < rhs)


#: VER-16b: a velocity component is zero to machine precision when its largest magnitude
#: anywhere is within this many units of round-off (double precision's epsilon, 2.2e-16) of the
#: field's largest |U|. Measured on v2406, on the committed healthy laminar pipe with no swirl
#: (`tests/fixtures/real_openfoam/wedge_pipe`, 2,400 cells, `tests/unit/test_wedge_steady.py`):
#: the wedge-normal Uz sits at 36.5 units (8.1e-15 of the largest |U|), while Uy, the smaller
#: component that carries flow, is 0.108 of the field, some 4.8e14 units. Not tuned: a thousand
#: roundings is 27 times the round-off measured and more than eleven orders below the flow.
ZERO_COMPONENT_ROUNDINGS = 1000
_COMPONENTS = ("Ux", "Uy", "Uz")


@dataclass(frozen=True)
class ZeroComponent:
    """A solved velocity component whose solution is zero to machine precision (VER-16b)."""

    name: str        # as the log names its solve: "Uz"
    ratio: float     # its largest magnitude over the field's largest |U|


def zero_components(fields) -> tuple[ZeroComponent, ...]:
    """The velocity components the latest U read (`FieldData`, `courant.verify.pipeline._fields`)
    shows to be zero to machine precision in every cell, when another component is not; else ().

    Such a component's normalised residual measures nothing. v2406 solves a vector one component
    at a time (`fvMatrix::solveSegregated`, src/finiteVolume/fvMatrices/fvMatrix/fvMatrixSolve.C:
    every component the mesh solves, the wedge-normal one included) and divides each initial
    residual, sum|b - A psi|, by `lduMatrix::solver::normFactor` (src/OpenFOAM/matrices/lduMatrix/
    lduMatrix/lduMatrixSolver.C): sum(|A psi - A_sum avg(psi)| + |b - A_sum avg(psi)|) + 1e-20.
    With psi zero to round-off, avg(psi), A psi and b are round-off too, so the residual is
    round-off over round-off: on the committed healthy pipe the wedge-normal Uz was above the bar
    of 1e-3 on 574 of its 1,000 iterations and ended at 3.3e-3, while Ux ended at 4.0e-11 and p
    at 8.8e-9 (`tests/fixtures/real_openfoam/wedge_pipe/flow/log.simpleFoam.gz`).

    Evidence or nothing: () when U was not read, is not a vector field, has a value that is not
    finite, was read on some ranks only, or is zero in every component (no flow, which the bar
    then judges as before)."""
    import numpy as np

    u = None if fields is None else (getattr(fields, "internal", None) or {}).get("U")
    if u is None or any(name == "U" for name, *_ in getattr(fields, "partial", ()) or ()):
        return ()
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3 or not len(u) or not np.isfinite(u).all():
        return ()
    largest = float(np.max(np.linalg.norm(u, axis=1)))
    if largest <= 0.0:
        return ()
    ratios = np.max(np.abs(u), axis=0) / largest
    bar = ZERO_COMPONENT_ROUNDINGS * float(np.finfo(float).eps)
    zero = [ZeroComponent(name, float(r)) for name, r in zip(_COMPONENTS, ratios) if r <= bar]
    return tuple(zero) if len(zero) < 3 else ()


def judged_on_the_bar(config, checks: str = "default") -> bool:
    """Whether a run's final residuals are held to the bar: with no configuration, or one that
    marches to a steady state (`CaseConfig.marches_to_steady_state`); a time-accurate run is
    judged on stability. The one test, which every asker calls: `residual.not_converged` and
    `residual.stalled` (`ResidualAnalyzer`), `ConvergenceCheck`'s steady branch, and VER-16b's
    exclusion (`courant.verify.pipeline._zero_components`), so they cannot disagree."""
    return config is None or config.marches_to_steady_state(checks)


def judge_residuals(events: list[Event], config,
                    checks: str = "default", *,
                    exclude: tuple[str, ...] | frozenset[str] = ()) -> ResidualJudgement | None:
    """The residual bar per field, and the stall detector, on one run. None if no residuals.

    Each field's final value is read from the last step that reported residuals. A field held
    to a declared tolerance is read at its first solve in that step, as residualControl reads
    it (`ResidualPayload.first_initial_residuals`). Under `residual.first_solve` (TRD VER-16a,
    behind `--checks next`) so is every other field of a steady run (`reads_first_solve`): a
    field solved more than once a step (`p`
    with non-orthogonal correctors) is otherwise judged on its last corrector, which starts from
    an almost solved field, so its initial residual is about `relTol` times the first by
    construction and the bar in effect `1/relTol` looser. Under the default a field held to
    the flat bar keeps the last solve's value it has always been judged on (v0.5.0). With no
    declared tolerance anywhere this is the flat bar exactly: every field against
    `RESIDUAL_TARGET`, the deciding field the largest.

    A field whose only candidate entries are keywords the matcher does not evaluate
    (`_tolerance`, not certain) may be held to such a keyword's tolerance or to none: both
    readings are judged, each as it would be if certain (the tolerance at the first solve, met
    strictly; the flat bar at the solve the flat bar reads), and the field is over when either
    is, and past `RESIDUAL_LOOSE` (`worst`) when either is. Its bar is the reading that decides
    (one it is over, else the one nearest its bar), and
    it is never counted as declared (the review of the 0.6.0 hard-proofing, 22 Oct 2026, found it
    judged as declared: "its residualControl" said of the flat bar, and on a local-time-stepping
    run its first solve read where the flat bar reads its last).

    **VER-16c** (`residual.control.floor`, behind `--checks next`; S's decision on
    H-VER16-ENDTIME, 26 Oct 2026): a field's bar is its tolerance clamped to
    [`RESIDUAL_TARGET`, `RESIDUAL_LOOSE`] (`_clamp`). A tolerance may loosen the bar and never
    tightens it (`sets_bar`): a field whose tolerance is below the target is held to the target,
    met as the flat bar is met (`<=`), is not counted as declared (its reasons say the flat
    bar's words), and its bar is the target in the stall detector too. It is read at both
    solves, whether or not the tolerance is certain: where the tolerance is read, its first,
    and where the flat bar reads (`_readings`), and it is over when either reading is, MAJOR
    past `RESIDUAL_LOOSE` on either (`worst`). On a steady run under `next`, where VER-16a
    holds, the two are the same solve, so it is judged exactly as on the flat bar. On a
    local-time-stepping run the flat bar reads the last solve, so a stricter tolerance is never
    judged more leniently than a looser one (read at the last alone, p at 2e-2 on its first
    solve and 5e-4 on its last was rejected under `p 1e-3` and accepted under `p 9.9e-4`: the
    review of T-VER16C) nor than no tolerance at all (read at the first alone, p at 5e-4 then
    2e-2 was accepted under `p 1e-8` and rejected with no residualControl: its final review).
    That tolerance, when the field's first solve did not reach it as OpenFOAM tests it
    (`residual < tolerance`), is in `unmet`: a note, never a reason. A field a keyword not
    evaluated may give several tolerances is judged on each (`_readings`).

    The stall is asked only of a run whose every field is within its bar. A run above its bar
    is already not converged, and is reported so; the stall is the evidence the bar cannot
    see, a "converged" that relaxation made spurious. Asked of every run, it added a finding
    to runs the bar already rejects and re-ranked their diagnosis (the certification corpus's
    `simpleFoam_pitzDaily__unbounded_scheme` put CONV.RESIDUAL_PLATEAU above the
    NUM.UNBOUNDED_SCHEME it is certified on).

    **Behind `--checks next` until 0.6.0 (TRD GATE-13).** The bar per field is
    `residual.control` and the stall `residual.stalled` in `courant.verify.checks.NEXT`; under
    the default `checks` every field is held to the flat `RESIDUAL_TARGET` on its last solve and
    no stall is asked, which is v0.5.0's judgement exactly (`worst_final_residual` against the
    flat bar).

    `exclude` (TRD VER-16b, on by default): the velocity components the latest U shows to be
    zero to machine precision (`zero_components`), whose normalised residual is round-off over
    round-off. They are judged by no bar and no stall; the verdict names them. A step whose
    every field is excluded is judged as if nothing were.
    """
    from functools import cmp_to_key

    from courant.verify.checks import enabled

    step = _last_step(events)
    if step is None:
        return None
    control = enabled("residual.control", checks)       # the bar per field (VER-16)
    last = _finite_items(step.initial_residuals or {})
    first = _finite_items(getattr(step, "first_initial_residuals", None) or {})
    if exclude and set(last) - set(exclude):
        last = {name: v for name, v in last.items() if name not in exclude}
        first = {name: v for name, v in first.items() if name not in exclude}
    steady = control and config is not None and config.marches_to_steady_state(checks)
    candidates = {name: _candidates(config, name) for name in last} if steady else {}
    floor = enabled("residual.control.floor", checks)       # VER-16c: a tighter one is a note
    every_first = reads_first_solve(config, checks, events=events)
    nearer = cmp_to_key(lambda a, b: _further_over(a[:2], b[:2]))
    final: dict[str, float] = {}
    bars: dict[str, float] = {}
    strict: dict[str, bool] = {}            # a tolerance is met strictly, the flat bar with <=
    named: set[str] = set()
    unsure: set[str] = set()
    unmet: list[UnmetTolerance] = []
    worst = -math.inf                       # the largest residual on any field's any reading
    for name, v in last.items():
        at_first = first.get(name, v)       # the solve residualControl reads
        flat = (at_first if every_first else v, RESIDUAL_TARGET, False)
        found = candidates.get(name)
        readings = [flat] if found is None else _readings(found, at_first, flat, checks)
        final[name], bars[name], strict[name] = decided = max(
            [r for r in readings if not _within(*r)] or readings, key=nearer)
        worst = max(worst, *(r[0] for r in readings))
        if found is None:
            continue
        if floor and not every_first:
            # VER-16c on a run whose flat bar reads another solve than the first (local time
            # stepping): past `RESIDUAL_LOOSE` there the field is MAJOR whatever the case says,
            # as it is with no residualControl. A held tolerance is read at the first solve alone
            # (residualControl's), so p at 5e-4 then 2e-2 under `p 1e-2` was accepted where no
            # tolerance rejects it, MAJOR (the final review of 519ab28; T-FOLLOWUP-MINORS (8)).
            # Below 1e-2 the last solve is the tolerance's to loosen, as S decided.
            worst = max(worst, flat[0])
        if decided[2]:                      # its bar is a tolerance: certainly, or it may be
            # The case's own only when it is the value of the entry that certainly matches (the
            # last candidate): a keyword before that entry may give a stricter one, which the
            # bar then is (T-FOLLOWUP-MINORS (2): "its residualControl" was said of it).
            own = found[1] and _clamp(found[0][-1], checks) == decided[1:]
            (named if own else unsure).add(name)
        tightest = min(found[0])
        if floor and not sets_bar(tightest, checks) and not at_first < tightest:
            # certain: the tightest is the value of the entry that certainly matches, the last
            # candidate (`_candidates`); a keyword before it may pre-empt it, and not set it
            unmet.append(UnmetTolerance(field=name, residual=at_first, tolerance=tightest,
                                        certain=found[1] and found[0][-1] == tightest))
    over = tuple(sorted(name for name, v in final.items()
                        if not _within(v, bars[name], strict[name])))
    field = max(sorted(final), key=cmp_to_key(
        lambda a, b: _further_over((final[a], bars[a]), (final[b], bars[b]))))
    ask_stall = not over and enabled("residual.stalled", checks)
    return ResidualJudgement(final=final, bars=bars, over=over, field=field,
                             residual=final[field], bar=bars[field],
                             worst=worst, declared=bool(named),
                             field_declared=field in named,
                             stall=stalled(events, config, final, bars, checks) if ask_stall
                             else None,
                             unsure=bool(unsure), field_unsure=field in unsure,
                             unmet=tuple(unmet),
                             unevaluated=_unevaluated(config, final, steady, ask_stall and (
                                 config is not None and config.is_steady(checks))))


def _unevaluated(config, fields, control: bool, relaxation: bool
                 ) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """`ResidualJudgement.unevaluated`: for each of `fields`, the keywords not evaluated of the
    lookups the judgement made, `residualControl` where it read the tolerances (`control`) and
    the relaxation where it asked the stall (`relaxation`, the detector's own steady test)."""
    found: dict[str, dict[str, None]] = {}            # an ordered set per field
    for name in sorted(fields):
        listed = found.setdefault(_openfoam_name(name), {})
        if control:
            listed.update(dict.fromkeys(keywords_not_evaluated(config, name)))
        if relaxation:
            listed.update(dict.fromkeys(keywords_not_evaluated(config, name, relaxation=True)))
    return tuple((name, tuple(keywords)) for name, keywords in found.items() if keywords)


def _bars(found: tuple[tuple[float, ...], bool], checks: str) -> list[tuple[float, bool, bool]]:
    """The bars a field with `residualControl` candidates (`_candidates`) is judged against, the
    one definition `judge_residuals` (`_readings`) and `residual_bar` read (T-FOLLOWUP-MINORS
    (3)): [(the bar, met strictly, read where the flat bar reads)].

    Every tolerance OpenFOAM may apply is read where OpenFOAM reads it, at the field's first solve
    of the step, against the bar `_clamp` gives it. A tolerance that sets the bar is met
    strictly, and of several the strictest decides, since each is read at the same solve and met
    the same way. One below the target (VER-16c, under `residual.control.floor`) is held to the
    target, met as the flat bar is met (`<=`), never declared, and read at both solves, certain
    or not: there, and where the flat bar reads. The flat reading is also the outcome where no
    candidate applies (none certain)."""
    tolerances, certain = found
    held = [tol for tol in tolerances if sets_bar(tol, checks)]
    bars = [(*_clamp(min(held), checks), False)] if held else []
    below = len(held) < len(tolerances)     # VER-16c: one below the target, held to it
    if below:
        bars.append((*_clamp(min(tolerances), checks), False))
    if below or not certain:
        bars.append((RESIDUAL_TARGET, False, True))
    return bars


def _readings(found: tuple[tuple[float, ...], bool], at_first: float,
              flat: tuple[float, float, bool], checks: str) -> list[tuple[float, float, bool]]:
    """The (value, bar, met strictly) readings a field with `residualControl` candidates is
    judged on: over when any is over (`judge_residuals`). The bars are `_bars`'s; each is read at
    the field's first solve (`at_first`), or where the flat bar reads (`flat`). On a steady run
    under `next`, where VER-16a holds (`residual.first_solve`), the two are the same solve. On a
    local-time-stepping run the flat bar reads the last solve (VER-16a is steady only), and a
    field read at its first solve alone was judged more leniently than with no tolerance at
    all: p at 5e-4 on its first solve and 2e-2 on its last was accepted under `p 1e-8` and
    rejected with no residualControl (the final review of T-VER16C), and read at its last alone,
    more leniently than under a looser tolerance (p at 2e-2 then 5e-4 accepted under
    `p 9.9e-4`, rejected under `p 1e-3`; the review before it). A tolerance below the target is
    kept beside a looser candidate's reading, so under `"(?=p)p" 1e-3; p 1e-8;` p is met
    strictly in case the keyword applies."""
    return [(flat[0] if at_flat else at_first, bar, strict)
            for bar, strict, at_flat in _bars(found, checks)]


def peak_courant(events: list[Event]) -> float | None:
    peak = None
    for event in events:
        if event.kind is not EventKind.RESIDUAL:
            continue
        value = event.payload.courant_max
        if value is None or math.isnan(value):
            continue
        peak = value if peak is None else max(peak, value)
    return peak


def sustained_courant(events: list[Event], fraction: float = 0.2) -> float | None:
    """The worst Courant number over the final `fraction` of the run.

    The peak is the wrong statistic for this question and the codebase has
    learned that twice. `buoyantPimpleFoam/hotRoom` — a correct tutorial —
    spikes to 25.7 on its second step as a 600 K floor patch relaxes into a
    300 K room, then decays and sits at 0.68 for the remaining 990 steps.
    Judged on the peak it is broken; judged on where it settled it is fine,
    which is also what a real blow-up fails.
    """
    values = [e.payload.courant_max for e in events
              if e.kind is EventKind.RESIDUAL
              and e.payload.courant_max is not None
              and not math.isnan(e.payload.courant_max)]
    if not values:
        return None
    return max(values[-max(1, int(len(values) * fraction)):])


def solve_converged(events: list[Event], *,
                    tolerance: float = RESIDUAL_LOOSE,
                    config=None) -> tuple[bool, str]:
    """Whether these events describe a converged solve, and why not if not.

    Returns ``(converged, reason)``. The reason is recorded on the fix attempt,
    so it has to name a number the engineer can check against the log rather
    than say "did not converge".

    Deliberately does not look at the exit code. A clean exit is necessary and
    nowhere near sufficient — that conflation is the bug this module exists to
    remove. Callers check status separately and both must hold.

    **A TRANSIENT RUN IS NOT JUDGED BY A STEADY-STATE BAR.** Without `config`
    this applies the residual threshold to everything, and that is wrong in both
    directions on a time-marching solve:

      * It failed correct repairs. On `pisoFoam/RAS/cavity` a CFL fix brought the
        Courant number to 0.8 — a stable timestep — but landed 3.4x coarser than
        the tutorial's own, and a transient run's final initial-residual is a
        function of its timestep, so it finished at 2.22e-02 against a 1e-02 bar
        and was reported as "not fixed".
      * It passed genuinely broken ones. `interFoam/damBreak` with adaptivity
        switched off ran at a sustained Courant of **6.05** and a final residual
        of 1.34e-03 — comfortably inside the bar, and physically wrong. The
        harness recorded it as a fault too weak to break the case.

    So for a transient solve the question is stability, which is what a
    time-marching scheme actually has to satisfy: did anything go non-finite,
    and did the Courant number settle inside the limit that applies to this
    case. `COURANT_STABILITY` for a fixed timestep; the case's own `maxCo` when
    it is controlling the step itself, because that is the criterion its author
    declared.

    The steady branch is unchanged. A steady solve has no timestep and its
    residual really is the convergence measure.
    """
    if diverged(events):
        return False, "residuals went NaN or infinite"
    climbs = [e for e in events if is_residual_climb(e)]
    if climbs:
        # A soft divergence is a residual *trend* from the anomaly detector, not a non-finite
        # value, so the run did not diverge (ADP-5); it is still not a converged solve. Saying
        # "NaN" for it sent RRD RM-18 looking for a NaN that was not in the log.
        p = climbs[0].payload
        slope = f", slope {p.slope:.3g} decades/step" if p.slope is not None else ""
        return False, (f"residual of {p.field or 'a field'} climbing (soft divergence at "
                       f"{p.residual:.3g}{slope})" if p.residual is not None
                       else f"residual of {p.field or 'a field'} climbing (soft divergence{slope})")

    peak = peak_courant(events)
    if peak is not None and peak > COURANT_ABSURD:
        return False, (f"Courant number reached {peak:.2e} — above {COURANT_ABSURD:.0e} "
                       f"the solution is not physical, whatever the residuals say")

    worst = worst_final_residual(events)
    if worst is None:
        return False, "the run reported no residuals"

    steady = config is not None and config.is_steady()
    if config is not None and not steady:
        settled = sustained_courant(events)
        if settled is not None:
            limit, how = courant_limit(config)
            if settled > limit:
                return False, (f"Courant number settled at {settled:.3g}, above "
                               f"{how} — a transient solve that does not respect "
                               f"its timestep criterion is not converged, "
                               f"whatever the residuals say")
        return True, (f"stable: Courant settled at {settled:.3g}"
                      if settled is not None else
                      f"stable: final worst residual {worst:.2e}")

    # **Not `judge_residuals`, on purpose (TRD VER-16).** This steady branch is the fix loop's
    # acceptance bar, `tolerance` (RESIDUAL_LOOSE unless a caller names another): did a repair
    # bring the solve out of the band the gate grades MAJOR. That line is unchanged by the
    # per-field bar, because every per-field bar is capped at RESIDUAL_LOOSE and
    # `ConvergenceCheck` still grades MAJOR exactly when the worst residual is above it, so the
    # two agree on it by construction. Asking the per-field bar or the stall here would change
    # the fix loop's success criterion and the `converged` the certification harness records
    # for its 36 committed scenarios (certification/RESULTS.json), which is a separate decision.
    if worst > tolerance:
        return False, (f"final worst residual {worst:.2e} is above the "
                       f"{tolerance:.0e} convergence tolerance")

    return True, f"final worst residual {worst:.2e}"
