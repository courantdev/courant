# Verification: icoFoam_cavity::cfl_blowup

**Verdict:** `NOT_TRUSTWORTHY` (rule VER-5.3)  
**Grade:** C  
**Solver:** icoFoam  
**Verifier:** courant {VERSION}, schema `courant.verdict/1`

## Why

| # | Code | Severity | Finding | Read | Method | Value | Threshold |
|---|---|---|---|---|---|---|---|
| 1 | `cfl.max` | critical | Courant number exceeds stability limit: max Co=9.66 at t=0.15 exceeds limit 1.0; still 8.92 over the final fifth of the run | icoFoam_cavity__cfl_blowup.log.gz (t = 0.15) | checker cfl | 9.66086 | 1.0 |
| 2 | `convergence.residual` | major | transient solve judged on stability rather than a residual threshold — Courant number settled at 8.92, above the stability limit of 1 — a transient solve that does not respect its timestep criterion is not converged, whatever the residuals say (final worst residual 3.61e-01) | icoFoam_cavity__cfl_blowup.log.gz | check convergence.residual (Residuals converged below tolerance) | 0.361172 | — |
| 3 | `cfl.climbing` | warn | Courant number climbing on a fixed timestep: Co rose 0 -> 9.66; icoFoam has no adaptive timestep | icoFoam_cavity__cfl_blowup.log.gz | checker cfl | 0.0 | 1.0 |

## Probable causes

1. `NUM.CFL_VIOLATION` — Courant number too high
2. `NUM.MISSING_ADJUSTABLE_DT` — Fixed dt without adjustableRunTime

## Not checked

- Solution fields were not read: no time directory with fields was found.
- Discretisation error was not quantified: a single mesh was supplied.
- Accuracy against a reference was not assessed: no reference data was supplied.
- Field values physically plausible: no internal fields
- Flow regime matches turbulence modeling: not implemented (Courant derives no length scale for a Reynolds number from the case)
- Near-wall y+ compatible with wall treatment: not implemented (Courant reads no y+, neither a yPlus field nor the yPlus function object's output)
- Net boundary mass flux ~ 0: no field flux data

## Next action

Suggested, not applied: Reduce deltaT in controlDict

## Inputs

| File | Bytes | SHA-256 |
|---|---|---|
| `icoFoam_cavity__cfl_blowup.case.json` | 1276 | `f7df306c6d60ec9cf5e3c5dbf61acb03430faeb2f1c4ffd27315d526b447d89a` |
| `icoFoam_cavity__cfl_blowup.log.gz` | 1887 | `2e851daab9151dd9aa29cdf76163afcb11e38b5ab1f5feb398d2f4f2ef3737ed` |

Parsed 10 events, 10 residual steps, 0 crash events, 0 divergence events.
