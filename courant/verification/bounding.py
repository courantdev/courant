"""Was this field clipped, and does it matter? THE ONE PLACE THIS IS ANSWERED.

`BoundingChecker`, `_unbounded_scheme_symptom` and the grade's `TurbulenceBoundsCheck` all
need it. The checker and the scheme symptom were unified in August; the grade still counted
every bounding event on its own. On `simpleFoam/T3A`, the first fresh tutorial run after v0.3,
that made an unmodified case grade C, for 1,001 events in which k's minimum was exactly 0.

Two facts decide it:
- **An excursion is a field that went negative.** OpenFOAM prints a bounding line when a
  field reaches its floor. A minimum of exactly 0 is the floor touched: T3A's k in its laminar
  region before transition. A minimum a few parts in a million below 0 is round-off: T3A's
  gammaInt, at -3.7e-05 against a maximum of 1.0. Neither is unbounded transport. The
  injected unbounded-scheme fault on pitzDaily clips epsilon to -2,741 against a maximum of
  80,809, which is 3.4%. pitzDaily's own startup reaches -205 against 33,174, which is 0.6%.
  `ROUNDOFF` (0.1%) sits clear of both: 30 times below the fault and 6 times below the startup.
- **Severity comes from the tail.** A field still being clipped in the final fifth of the run
  is persistent, and anything earlier is the solution relaxing. That is the convention
  `CourantChecker` uses for the Courant number. The rule it replaces asked whether every
  excursion fell in the first 10% of the run. That boundary was fitted to pitzDaily (1% against
  100%), and T3A's transition model stops clipping gammaInt at 10.4% of its 1,000 iterations,
  then converges cleanly. A boundary that a healthy case crosses by 0.4% is not a physical
  criterion; "is it still happening at the end" is.

An event without magnitudes (written before the parser kept them) counts as an excursion,
which is the behaviour it had before.
"""

from __future__ import annotations

from courant.events.core import EventKind

ROUNDOFF = 1.0e-3

#: Clipping in the final fifth of the run is persistent; before it, the solution is relaxing.
#: The same tail `CourantChecker` judges the Courant number on. Measured: pitzDaily (healthy)
#: last clips on step 3 of 282, T3A (healthy) on step 104 of 1,000, and the unbounded-scheme
#: fault on step 300 of 300.
TAIL_FRACTION = 0.20
#: Kept for callers that name the old constants.
SETTLE_FRACTION = 1.0 - TAIL_FRACTION
SETTLE_MIN_STEPS = 3


def is_excursion(payload) -> bool:
    """A bounding event in which the field genuinely went negative."""
    lo = getattr(payload, "minimum", None)
    if lo is None:
        return True
    if lo >= 0.0:
        return False
    scale = max(abs(getattr(payload, "maximum", None) or 0.0),
                abs(getattr(payload, "average", None) or 0.0))
    return scale == 0.0 or abs(lo) > ROUNDOFF * scale


def bounding_by_field(events) -> dict[str, tuple[int, bool]]:
    """Per field: how many excursions, and whether they all happened during startup."""
    steps = 0
    counts: dict[str, int] = {}
    last_step: dict[str, int] = {}
    for e in events:
        if e.kind == EventKind.RESIDUAL:
            steps = steps + 1
        elif (e.kind == EventKind.WARN and getattr(e.payload, "category", None) == "bounding"
              and is_excursion(e.payload)):
            f = e.payload.field or "?"
            counts[f] = counts.get(f, 0) + e.payload.count
            last_step[f] = steps
    if not counts:
        return {}
    # With no residuals there is no run to be early or late in: report, never silence.
    window = max(SETTLE_MIN_STEPS, steps * (1.0 - TAIL_FRACTION)) if steps else float("-inf")
    return {f: (c, last_step.get(f, 0) <= window) for f, c in counts.items()}


def floor_clips_by_field(events) -> dict[str, int]:
    """Bounding events that touched the floor without going negative: reported, never judged."""
    out: dict[str, int] = {}
    for e in events:
        if (e.kind == EventKind.WARN and getattr(e.payload, "category", None) == "bounding"
                and not is_excursion(e.payload)):
            f = e.payload.field or "?"
            out[f] = out.get(f, 0) + e.payload.count
    return out
