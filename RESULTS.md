# Courant benchmarks

Measured on real OpenFOAM v2406 runs (courant 0.5.0). Every number has its denominator, and the ranges in brackets are Wilson confidence intervals. Each study was pre-registered before its first run, and its result is published whatever it showed.

## Frontier models as judges (2026-10-15)

The same 420 runs (140 right, 280 wrong), the same files and the task's specification for every judge.

| Judge | Wrong runs accepted | Of them, the wrong problem | Right runs rejected |
|---|---|---|---|
| **Courant** | **0 of 264 (0–1%)** | **0 of 140 (0–3%)** | 10 of 140 (4–13%) |
| Claude Haiku 5.5 | 52 of 249 (16–26%) | 46 of 129 (28–44%) | 17 of 122 (9–21%) |
| Claude Opus 5.5 | 69 of 264 (21–32%) | 61 of 140 (36–52%) | 5 of 140 (2–8%) |
| Claude Sonnet 5.5 | 76 of 263 (24–35%) | 59 of 139 (35–51%) | 0 of 138 (0–3%) |

Each model accepted more wrong runs than Courant (exact one-sided McNemar; Claude Haiku 5.5 p = 2e-16, Claude Opus 5.5 p = 2e-21, Claude Sonnet 5.5 p = 1e-23). Abstentions are left out of the denominators. These models, this prompt and these dates only.

## At scale: Courant-10K (1,827 runs)

Each draw run right, with a catalogue fault, and as the wrong problem.

| With the task's specification | Result | Interval | Pre-registered bar |
|---|---|---|---|
| Faulted runs accepted (≤ 1%) | 0 of 609 | 0.00–0.63% | supported |
| Wrong-problem runs accepted (≤ 1%) | 0 of 609 | 0.00–0.63% | supported |
| Right runs left insufficient (≤ 5%) | 0 of 609 | 0.00–0.63% | supported |
| Verifier exceptions (0) | 0 exceptions in 3654 verifications | — | supported |
| Right runs rejected (≤ 2%) | 34 of 609 | 4.02–7.70% | refuted |

The right runs rejected are 30 pitzDaily, 4 hotRoom references that never settle under the tutorial's own settings: the gate's reason holds by OpenFOAM's semantics. The next corpus version restricts those draws.

## The Courant Test (31 runs)

Judged correctly 31 of 31; wrong results accepted 0 of 23; good results rejected 0 of 8.

Tutorial-sized runs and injected faults; no customer data. "Certified" means measured against a real solver with the error rate published, not a regulator's approval.
