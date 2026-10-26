# What Courant can actually name

The taxonomy lists **25 failure modes**. That number is a vocabulary, not a
coverage claim, and the two were being reported as if they were the same thing.
This file separates them, and `tests/unit/test_coverage_claims.py` keeps it honest: a
code cannot be added to the taxonomy without either a route to emitting it or a
written reason why there is none.

Three tiers, narrowest first.

## 1. Named on real solver output — 13 of 25

Courant put these in the top two of a diagnosis for a log **OpenFOAM actually
produced**, replayed through the shipping parser, checkers and knowledge base.
This is the only tier that is evidence rather than intent.

Measured over the 31-log corpus in `certification/corpus`: **top-1 19/24
(79%), top-2 24/24 (100%)**, and no ranked cause on any of the 7
unmodified tutorials.

- `BC.OVERCONSTRAINED_PU` — Pressure-velocity coupling over/under specified at boundaries.
- `CONV.RESIDUAL_PLATEAU` — Residuals flatten above the convergence criterion; not converged.
- `ENV.MISSING_DICT` — A required field file or dictionary entry is absent.
- `MESH.ASPECT_RATIO` — Aspect-ratio extremes degrade conditioning and stability.
- `MESH.SKEWNESS` — Skewness or non-orthogonality above solver limits without correctors.
- `NUM.BAD_INITIAL_FIELD` — Uninitialized/poor initial field; first steps diverge without potentialFoam.
- `NUM.CFL_VIOLATION` — Timestep too large for the mesh/velocity; Co exceeds stability limit.
- `NUM.MISSING_ADJUSTABLE_DT` — No adaptive timestep control; Co spikes on transients.
- `NUM.OVER_RELAXATION` — Under-relaxation factors too high; SIMPLE loop destabilizes.
- `NUM.SOLVER_TOLERANCE` — Linear solver tolerance too loose/tight for the field coupling.
- `NUM.UNBOUNDED_SCHEME` — Non-bounded div scheme on a tet-dominant mesh causes oscillation.
- `PHYS.LAMINAR_TURBULENT` — Laminar setup for a turbulent regime (or vice versa).
- `PHYS.TRANSPORT_UNITS` — Viscosity/density/units wrong, shifting the effective regime.

## 2. Wired, but no scenario in this corpus — 9 of 25

A route exists — a finding some checker emits, a log pattern, a config
predicate, or a checker suggestion — and nothing in the committed corpus
exercises it. These are **not** measured on real solver output. Some are covered by the full Docker
matrix, which runs scenarios the replay corpus has no log for; the rest need a
fault nobody has written yet.

Listing them here rather than in the first tier is the point: a route
that has never fired is a hypothesis about our own code.

"Certified" here means measured on output a real solver produced, with the result
published with each release (`courant verify --about`). It is not a regulator's approval.

- `BC.REVERSED_OUTFLOW` — via findings `crash.reversed_outflow`, `flow.reversed_outflow`; checker suggestion
- `BC.TURB_INLET` — via checker suggestion
- `BC.WALL_FUNCTION` — via 1 log pattern(s)
- `CONV.OSCILLATION` — via findings `residual.growth`
- `ENV.DECOMP_MISMATCH` — via findings `crash.decomp_mismatch`; 1 log pattern(s); checker suggestion
- `MESH.LEAKY_SNAPPY` — via 1 log pattern(s)
- `MESH.NEGATIVE_VOLUME` — via findings `mesh.negative_volume`, `crash.negative_volume`; 1 log pattern(s); checker suggestion
- `MESH.YPLUS_RESOLUTION` — via findings `bounding.nut`; 1 log pattern(s)
- `PHYS.CHT_THERMO` — via 3 log pattern(s)

## 3. In the vocabulary, not in the claim — 3 of 25

These stay in the taxonomy because they are real failure modes an engineer will
recognise and a report may need to discuss. They are excluded from the coverage
number because Courant cannot emit them, and counting them would be claiming a
capability that does not exist.

An audit over the knowledge base, the checkers and the 31 real logs found each
of these had no route of any kind — no finding, no pattern, no predicate, no
suggestion — while being counted in "25 failure modes".

### `PHYS.TURB_REGIME` — Turbulence model unsuited to regime

Whether kEpsilon or kOmegaSST suits a flow is a modelling judgement about a regime the log does not report. It needs a Reynolds number, which needs a length scale, which is a property of the geometry rather than of anything the solver prints. PHYS.LAMINAR_TURBULENT covers what the case and the log can show: a laminar setup on a run that will not settle (physics.laminar_unstable), and, under --checks next, a buoyant laminar case whose own numbers make it turbulent (TRD VER-17). A laminar run of a turbulent flow that converges is not judged on its regime by default (the Reynolds-number check is not implemented, TRD VER-7c), and this entry is what is left over.

### `CONV.MASS_IMBALANCE` — Mass/energy imbalance

A symptom, not a cause, and the number is dimensional. OpenFOAM prints `sum local` and `cumulative` continuity errors in units of volumetric flow, so any threshold is case-specific: measured on this corpus a healthy pitzDaily ends at 3.0e-03 and a healthy planarContraction at 2.1e-11, eight orders apart with nothing wrong in either. The fault that produces an imbalance is diagnosed by its cause — the loose pressure tolerance — which is both correct and actionable, where 'mass does not balance' is neither. The same reasoning applies to grading: `ContinuityCheck` in courant/verification/checks.py reports this number and deliberately does not deduct on it, after an absolute threshold gave two unmodified OpenFOAM tutorials a C.

### `CONV.MESH_DEPENDENCE` — Not mesh-independent

Not a property of a run. Mesh independence is a property of a *sequence* of runs on successively refined meshes, and Courant watches one solve. Nothing in a single log, however carefully read, can establish it. Detecting this means running the case again on a finer mesh, which is a different product behaviour and not a checker.

## How the number moved

The previous report said "16 of 25 taxonomy codes are reachable ... and
certified against a real solver here". That counted intent. The audit that
produced this file found three codes with no route at all, two signatures
written from what a fault *ought* to print rather than from a log, and four
signatures that fired on unmodified tutorials. In this report "certified" means
measured on output a real solver produced, with the result published.

The measured number is the one above. It is smaller, and it is true.

*Generated by `scripts/coverage_report.py`. Do not edit by hand.*
