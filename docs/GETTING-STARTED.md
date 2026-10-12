# Getting started with Courant

Courant checks a finished or running OpenFOAM simulation and tells you three things:
- whether the result can be accepted, and why not;
- where the engineering answer stopped changing, and how much of the run came after;
- what it did not check.

It runs on your machine and reads files. It never edits your case and never runs a solver. Its
checks never use the network; the few commands whose purpose is a network call (`courant
update`, and a language model you name) say what they contact first. [NETWORK.md](NETWORK.md)
lists them.

Version 0.4.0. Measured on OpenFOAM v2406. Every figure quoted below carries its denominator;
`courant verify --about` prints the error rates measured for this version.

<!-- Every bash block below is run by tests/integration/test_docs_commands.py on the demo data
that ships inside Courant. A command exits 0 unless a "# exit N" note after it says otherwise. A
block marked docs-skip cannot run in CI, and says why. -->

## 1. Install

Either of these works. Neither needs root.

<!-- docs-skip: installs from the package index, which CI does not reach -->
```bash
pip install 'courant-dev[mcp]'      # Python 3.11 or later; the command is `courant`
```

The single-file binary needs no Python. It is attached, signed, to every release of
github.com/courantdev/courant:

<!-- docs-skip: downloads the installer from GitHub, which CI does not reach -->
```bash
curl -fsSL https://github.com/courantdev/courant/releases/latest/download/install.sh | sh
```

It checks the binary's SHA-256 against the release's `SHA256SUMS`, and the Ed25519 signature of
`SHA256SUMS` against the release key it carries when python3 with the `cryptography` package or
OpenSSL 3 is available. With neither, it says that only the checksum was checked and how to check
the signature by hand; set `COURANT_REQUIRE_SIGNATURE=1` to make it refuse instead.

Check the install. This needs no account and no network:

```bash
courant demo
```

It shows four real OpenFOAM runs that ship inside Courant: a healthy one, one with a
timestep ten times too large, one with its viscosity in the wrong units (caught by preflight
before the solver would have run), and a steady run with a fixed iteration budget. It writes an
HTML report you can open.

## 2. Verify a run

```bash
courant verify ./my_case                   # a case directory: finds the solver log itself
courant verify log.simpleFoam              # or a log on its own
```

You get one of three verdicts (`courant.verdict/2`):

| Verdict | Exit code | Meaning |
|---|---|---|
| `accept` | 0 | No check failed, and the run produced enough output to be judged |
| `reject` | 2 | At least one check failed. Each reason gives the file it read, the method, the threshold and the value |
| `insufficient` | 3 | There was not enough to judge. The verdict carries one reason, `insufficient.<name>`, from a closed list, and the report says what was missing |

An unreadable input exits 4, and so does an unexpected internal error, with one line that names
the command and the version. `--debug` prints the traceback. Every report also lists **what was
not checked**, in fixed words. For example: "a single mesh was supplied, so discretisation error
was not quantified." It states its **basis** too: `validity` when no task specification was
given, `validity+conformance` when one was given and locked before the run, and
`validity+unlocked-spec` when it was given without a lock.

The reasons an `insufficient` verdict can carry are a closed list, each with one sentence and
one rule:

| Code | Rule | Cause tag |
|---|---|---|
| `insufficient.no_output` | VER-5.1 | run |
| `insufficient.unrecognised_log` | VER-5.1 | fork |
| `insufficient.no_residuals` | VER-5.5 | run |
| `insufficient.no_time_markers` | VER-5.15 | run |
| `insufficient.spec_hash_mismatch` | VER-5.8 | spec |
| `insufficient.spec_value_unreadable` | VER-5.9 | spec |
| `insufficient.unsupported_version` | VER-5.10 | fork |
| `insufficient.spec_invalid` | VER-5.11 | spec |
| `insufficient.spec_extracted_mismatch` | VER-5.12 | spec |
| `insufficient.spec_role_unresolved` | VER-5.13 | fork |
| `insufficient.time_scheme_unreadable` | VER-5.14 | fork |

The cause tag says what to change for the verdict to move: `run` the run's own output, `spec` the
task specification, `fork` the solver fork or version.

The previous verdict names (`TRUSTWORTHY`, `NOT_TRUSTWORTHY`, `NOT_ASSESSABLE`,
`courant.verdict/1`) are available for one more minor release with `--schema 1`; the decision
is the same.

Other outputs:

```bash
courant verify ./my_case --html report.html     # one self-contained file: open it, mail it
courant verify ./my_case --report md            # Markdown
courant verify ./my_case --json                 # canonical JSON, byte-identical on a re-run
courant verify ./my_case --schema 1 --json      # the previous verdict names
courant verify ./my_case --pack evidence/       # verdict + report + SHA-256 manifest of inputs
courant verify --check-pack evidence/ --root ./my_case   # anyone can re-derive it
courant verify --about                          # the error rates published for this version
```

`--about` prints the false accepts and false alarms measured for this version, each with its
denominator and confidence interval, the abstentions apart, and says so when a rate is not measured.
The HTML report's footer and the evidence pack's manifest carry the same rates. They are read
from the results that ship with Courant and are not part of the verdict, so a verdict's bytes do
not change when a results file does.

## 2b. Is it the problem that was asked? Verify against a task specification

A run can converge beautifully and still answer the wrong question: an inlet doubled, a
viscosity tripled, gravity rotated, the geometry scaled. Nothing in its log says so. When the
task is on record — which it always is when an AI agent set the case up — Courant checks
every value the task names against the case.

```bash
courant spec init ./reference_case -o spec.json --task "pitzDaily, inlet 10 m/s, k-epsilon"
courant spec hash spec.json              # lock it before the run; keep the hash
courant verify ./agent_case --spec spec.json --spec-sha256 <hash>
```

`courant spec init` drafts the specification from the case the task is written against:
boundary-condition parameters, material properties, gravity, end time, mesh scale, and, marked
as modelling choices, the solver, the turbulence model and the mesh cell counts. Review it,
delete what the task leaves open, and lock it.

The verdict then adds a conformance block, and:
- a named value that does not match makes the verdict `reject` (rule VER-5.7), with the file,
  the value found and the value asked for; so does a named patch that is certainly not in the
  case: absent from `constant/polyMesh/boundary` (groups included) and from the field file;
- a named value that cannot be read makes an otherwise clean run `insufficient`
  (`insufficient.spec_value_unreadable`, VER-5.9);
- a specification that no longer matches its locked hash makes it `insufficient`
  (`insufficient.spec_hash_mismatch`, VER-5.8), and the check is void.

`courant spec check ./case --spec spec.json` runs the conformance check alone. It needs no
log, so it works before the solver spends anything. Measured on real OpenFOAM runs (RRD
RM-18): of 20 runs with the right physics and the wrong input, a verdict without a
specification rejected 8; against the specification, 20. What a specification does not name
is not checked, and the report says so.

## 3. Before the run: preflight

```bash
courant preflight ./my_case
```

This checks what no timestep can change before the solver spends anything:
- the viscosity's units;
- linear-solver tolerances;
- relaxation on a steady loop;
- boundaries that fix both pressure and velocity;
- mesh quality, when `checkMesh` output exists.

`SETUP_FAULT` exits 2, with the file and value for each fault. On the corpus it caught 8 of 23
broken setups and flagged 0 of 8 good ones. Everything that needs the run is listed as not
checked.

## 3b. Where did the answer stop changing?

Add a function object that writes the quantity you care about. For example, forces or force
coefficients, a surface average, or probes. Then:

```bash
courant convergence ./my_case
```

```
  1  Answer       forceCoeffs1/coefficient:Cd: value 0.0128 at step 1,011, 0.0128 at the end (step 2,000)
  2  Change       0.03 counts after the stopping point (limit 1 count)
  3  Iterations   989 of 1,999 past convergence (49.5%)
  4  Wall clock   60 s of 120 s past convergence (solver ClockTime)
  5  Core-hours   0.0667 past convergence (4 ranks, nProcs in the log, x wall clock)
```

(Illustrative output; your numbers come from your run.)

What it reads and how it judges:
- **What it reads.** Monitors under `postProcessing/`: `forceCoeffs`, `forces`,
  `surfaceFieldValue`, `volFieldValue` and `probes`. Restarts across start-time folders are
  merged.
- **The default monitors.** The totals each function object writes, not their pressure/viscous
  or front/rear splits. Choose your own with `--monitor '*:Cd'`.
- **Tolerances.** Coefficients are judged to 1 count (1e-4) and everything else to 1% of its
  scale. Change this with `--tol Cd=5e-5` or `--tol-rel 'areaAverage(p)=0.002'`.
  <!-- claims-ok: 1% is a tolerance setting, not a measured result -->
- **Steady runs** are judged on whether each monitor reached a fixed point.
- **A limit cycle under a steady solver** is named as one, and no saving is claimed for it.
- **Transient runs** are reported for reference. Add `--transient-average` when your answer is
  a time-average; Courant then judges the mean's confidence interval.
- **Prices.** `--cost-per-core-hour 0.05` prices the core-hours, and `--ccf-ghz` adds FIA
  allocation unit hours.

`courant verify` on a case directory shows the same result in a CONVERGENCE block, beside the
verdict. In v0.3 it does not change the verdict.

## 4. While it runs

```bash
courant watch ./my_case
```

This prints a line when the state changes: RUNNING, FAILING (a crash, a non-finite residual,
or residuals three orders above their best), or CONVERGED (on a steady run with monitors). When
the solver prints `End`, it prints the full verdict and exits with its code. It is read-only:
it never stops your run. To act on it, use your own job script:

<!-- docs-skip: needs a solver and a Slurm scheduler to act on -->
```bash
simpleFoam > log.simpleFoam 2>&1 &
courant watch . --exit-on-fail || scancel "$SLURM_JOB_ID"
```

On a case directory with a setup fault, `watch` is FAILING before the first timestep. On
Courant's corpus no healthy run was ever flagged (0 of 8). Courant does not claim to predict a
crash: without a setup fault, the alarm fires when the run fails.

## 4b. A folder of runs: the compute audit

```bash
courant audit ./runs --cost-per-core-hour 0.05 --html audit.html   # exit 2: runs/ holds rejected runs
```

For every run, the audit gives the verdict, the convergence and the preflight result; its exit
code is the worst run's, as for `verify --batch`. Then it totals, from the logs alone:
- the core-hours spent on results that cannot be trusted;
- the core-hours after the answer stopped changing;
- the core-hours on setups that were broken before they started.

Each total has its denominator, and the cost is shown only at your rate.

## 5. In CI

```bash
courant verify --batch runs/ --junit courant.xml     # exit 2: runs/ holds rejected runs
```

`reject` is a failed test, `insufficient` is a skipped test, and an unreadable input is an
error. Any CI system that reads JUnit (GitHub Actions, GitLab, Jenkins, Azure) shows each run as
a test. The batch exit code is 2 if any run is rejected, otherwise 3 if any is insufficient,
otherwise 4 if any is unreadable.

## 6. For AI agents (MCP)

<!-- docs-skip: starts a server that waits for an agent on stdio; tests/unit/test_mcp_e2e.py drives it end to end -->
```bash
courant mcp --root ~/runs
```

The server offers six read-only tools, confined to the directories you name:
- `verify_run`;
- `score_agent_output`;
- `check_convergence`;
- `explain_code`;
- `list_failure_modes`;
- `verifier_certification`.

`verify_run` and `score_agent_output` return `courant.verdict/2`, or the previous major with
`schema=1`. `score_agent_output` is advice to the agent, not a certification: given the task
specification and whether the agent is about to report success, it returns the verdict, the
task class, and whether the run is a *silent error* (it executed, the agent would report
success, and the gate rejects it) or an *unverified success* (the same, with `insufficient`).

To configure your client, print the settings for the directories you want it to read:

```bash
courant mcp --print-config --root ~/runs
```

For Claude Code that is a single command:

<!-- docs-skip: needs the claude CLI, which CI does not have -->
```bash
claude mcp add courant -- courant mcp --root ~/runs
```

An agent that runs a simulation can then check its own result before it reports success.

## 7. What Courant does not do (v0.3)

- It does not make a solver faster. It shows where compute was spent after the answer was
  known, or after the run had already failed.
- It does not quantify discretisation error from a single mesh, or compare against
  experiments. Both appear under "not checked".
- It does not yet verify Fluent, STAR-CCM+ or OpenFOAM Foundation v10. They are planned, not
  measured.
- It does not edit a case. Anything it suggests is labelled "Suggested, not applied".

## 8. When something looks wrong

Write a support bundle and send us that. It holds the versions, the platform, the command line
and, if you name one, a verdict. It contains file names, hashes and check values. It does not
contain your mesh, geometry or field data, and Courant never sends anything by itself. If the
run id, patch names or file names are your own words, add `--pseudonymise`: they become `run1`,
`patch1`, `file1` and the like, and paths and hostnames are stripped.

```bash
courant doctor --bundle support.json --verdict evidence/verdict.json --pseudonymise
```
