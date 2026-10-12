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


def _finite_values(mapping: dict[str, float | None]) -> list[float]:
    return [v for name, v in mapping.items()
            if v is not None and not math.isnan(v) and not math.isinf(v)
            and not _is_phase_fraction(name)]


def worst_final_residual(events: list[Event]) -> float | None:
    """The largest initial residual of the last step that reported any.

    "Initial residual" rather than final: the initial residual of a step is how
    far the *solution* was from satisfying the equations before that step's
    linear solve. A final residual near zero only says the linear solver did its
    job, which it does right up to the point the physics is nonsense — as the
    run described in the module docstring demonstrates.
    """
    for event in reversed(events):
        if event.kind is not EventKind.RESIDUAL:
            continue
        values = _finite_values(event.payload.initial_residuals or {})
        if values:
            return max(values)
    return None


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

    if worst > tolerance:
        return False, (f"final worst residual {worst:.2e} is above the "
                       f"{tolerance:.0e} convergence tolerance")

    return True, f"final worst residual {worst:.2e}"
