# Courant

**Physics acceptance testing for machine-produced simulation.**

Courant is a read-only, deterministic gate between a simulation (an AI agent's or an
engineer's) and the person who signs it. Without a task specification it decides whether an
OpenFOAM run is *valid*. With the task the run was asked to solve, it decides whether the run is
*the problem that was asked*.

Every verdict is `accept`, `reject` or `insufficient`, and every reason cites the file it read,
the method, the threshold and the value. Deterministic code decides every verdict; no language
model is in the verdict's path. Courant never edits a case, never runs a solver, and its checks
never use the network.

**[courant.dev](https://courant.dev)** &middot;
[Results](https://github.com/courantdev/courant/blob/main/RESULTS.md) &middot;
[Getting started](https://github.com/courantdev/courant/blob/main/docs/GETTING-STARTED.md) &middot;
[PyPI: courant-dev](https://pypi.org/project/courant-dev/) &middot;
[Releases](https://github.com/courantdev/courant/releases)

![Two real OpenFOAM runs of the pitzDaily tutorial, their residuals arriving iteration by iteration. Both end with exit 0. Courant accepts run A and rejects run B, with its rule and its first reason.](https://raw.githubusercontent.com/courantdev/courant/main/docs/media/verdicts.gif)

<sub>Two real OpenFOAM v2406 runs of the pitzDaily tutorial, from `certification/corpus/`. Run A
is the tutorial as shipped. Run B is the same case with its pressure solver's tolerance loosened
from 1e-06 to 0.01. The solver ends both with `End`, exit 0. Their residuals are read by the
parser the verdict uses, and the verdicts are `courant verify --json` on the same two files:
A is accepted (rule VER-5.6); B is rejected (rule VER-5.4), first reason
`residual.not_converged`.</sub>

## What it catches

Every figure is drawn from real solver output and Courant's own verdicts, and its numbers are
the published ones, under [Benchmarks](#benchmarks) and in
[RESULTS.md](https://github.com/courantdev/courant/blob/main/RESULTS.md). Click one to see it
full size.

<table>
<tr>
<td valign="top" align="center"><a href="https://github.com/courantdev/courant/blob/main/docs/media/courant-test.svg"><img src="https://raw.githubusercontent.com/courantdev/courant/main/docs/media/courant-test.svg" width="390" alt="The Courant Test's real runs as tiles, one per run: its residuals, whether its log ends, and Courant's verdict. Every right run is accepted and every wrong run rejected."></a></td>
<td valign="top" align="center"><a href="https://github.com/courantdev/courant/blob/main/docs/media/judges.svg"><img src="https://raw.githubusercontent.com/courantdev/courant/main/docs/media/judges.svg" width="390" alt="Wrong runs accepted on the same 420 real runs, as bars with Wilson intervals: Courant beside Claude Haiku 5.5, Claude Opus 5.5 and Claude Sonnet 5.5."></a></td>
</tr>
<tr>
<td valign="top"><b>Exit 0 is not a verdict.</b><br><sub>On the 31 real runs of the Courant Test, 31 of 31 are judged correctly: 0 of 23 wrong results accepted, 0 of 8 good results rejected. The two runs above are ringed.</sub></td>
<td valign="top"><b>Beside frontier models.</b><br><sub>The same 420 runs, the same files and the task's specification for every judge. Courant accepted 0 of 264 wrong runs; Claude Haiku 5.5 accepted 52 of 249, Claude Opus 5.5 69 of 264 and Claude Sonnet 5.5 76 of 263.</sub></td>
</tr>
<tr>
<td valign="top" align="center"><a href="https://github.com/courantdev/courant/blob/main/docs/media/wrong-problem.svg"><img src="https://raw.githubusercontent.com/courantdev/courant/main/docs/media/wrong-problem.svg" width="390" alt="Runs with the right physics and one wrong input, one square each, grouped by tutorial: rejected without the task and with it."></a></td>
<td valign="top" align="center"><a href="https://github.com/courantdev/courant/blob/main/docs/media/scale.svg"><img src="https://raw.githubusercontent.com/courantdev/courant/main/docs/media/scale.svg" width="390" alt="Courant-10K on 1,827 real runs: each pre-registered hypothesis, its observed rate, its Wilson interval and its bar, including the one that was refuted."></a></td>
</tr>
<tr>
<td valign="top"><b>The right physics, the wrong problem.</b><br><sub>On 20 real runs that converged with one input wrong, the verdict without a specification rejected 8 of 20; against the task specification, 20 of 20.</sub></td>
<td valign="top"><b>At scale.</b><br><sub>On 1,827 real runs, with the task's specification, 0 of 609 faulted and 0 of 609 wrong-problem runs were accepted. Right runs rejected: 34 of 609, against a pre-registered bar of 2%; refuted, and published.</sub></td>
</tr>
</table>

## Install

<!-- docs-skip: installs from the package index, which CI does not reach -->
```bash
pip install 'courant-dev[mcp]'      # Python 3.11 or later; the command is `courant`
```

The distribution is named `courant-dev` (PyPI's `courant` is an unrelated project); the import
package and the commands are `courant` and `courant-mcp`.

A single-file binary for Linux and macOS, which needs no Python, is attached to every
[release](https://github.com/courantdev/courant/releases), with `SHA256SUMS` and its Ed25519
signature:

<!-- docs-skip: downloads the installer from GitHub, which CI does not reach -->
```bash
curl -fsSL https://github.com/courantdev/courant/releases/latest/download/install.sh | sh
```

The installer checks the binary's SHA-256 against the release's `SHA256SUMS`, and the signature
of `SHA256SUMS` against the release key it carries when python3 with the `cryptography` package
or OpenSSL 3 is on the machine. With neither, it says that only the checksum was checked and how
to check the signature by hand; `COURANT_REQUIRE_SIGNATURE=1` makes it refuse instead.
`courant update` installs nothing unless the signature verifies against the key built into the
binary.

## Try it offline

These run on the real OpenFOAM runs that ship inside the package, with no network:

```bash
courant demo                     # four real runs and an HTML report
courant verify --about           # the error rates published for this version
```

## What a verdict looks like

`courant demo --keep DIR` writes the demo's runs to a folder you keep. One of them is the icoFoam
cavity with a timestep ten times too large:

```bash
courant demo --keep ./demo
courant verify demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.log.gz --case-json demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.case.json   # exit 2
```

```text
$ courant verify demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.log.gz --case-json demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.case.json
VERDICT   reject   (rule VER-5.3)
BASIS     validity
GRADE     C
RUN       icoFoam_cavity::cfl_blowup · icoFoam

WHY
  1  cfl.max — Courant number exceeds stability limit: max Co=9.66 at t=0.15 exceeds limit 1.0; still 8.92 over the final fifth of the run
     read    icoFoam_cavity__cfl_blowup.log.gz (t = 0.15)
     method  checker cfl
     value   9.66086   threshold 1.0
  2  convergence.residual — transient solve judged on stability rather than a residual threshold — Courant number settled at 8.92, above the stability limit of 1 — a transient solve that does not respect its timestep criterion is not converged, whatever the residuals say (final worst residual 3.61e-01)
     read    icoFoam_cavity__cfl_blowup.log.gz
     method  check convergence.residual (Residuals converged below tolerance)
     value   0.361172   threshold —
  3  cfl.climbing — Courant number climbing on a fixed timestep: Co rose 0 -> 9.66; icoFoam has no adaptive timestep
     read    icoFoam_cavity__cfl_blowup.log.gz
     method  checker cfl
     value   0.0   threshold 1.0

PROBABLE CAUSES
  1  NUM.CFL_VIOLATION — Courant number too high
  2  NUM.MISSING_ADJUSTABLE_DT — Fixed dt without adjustableRunTime

NOT CHECKED
  - Solution fields were not read: no time directory with fields was found.
  - Discretisation error was not quantified: a single mesh was supplied.
  - Accuracy against a reference was not assessed: no reference data was supplied.
  - Field values physically plausible: no internal fields
  - Flow regime matches turbulence modeling: not implemented (Courant derives no length scale for a Reynolds number from the case)
  - Near-wall y+ compatible with wall treatment: not implemented (Courant reads no y+, neither a yPlus field nor the yPlus function object's output)
  - Net boundary mass flux ~ 0: no field flux data
  - No specification given: conformance to the task was not assessed.

NEXT  Suggested, not applied: Reduce deltaT in controlDict
```

The solver exited normally on this run. The verdict says why it cannot be accepted, and what was
not checked. The exit code is the verdict's: `accept` 0, `reject` 2, `insufficient` 3, and 4 for
an input that cannot be read. `--json` prints the same verdict as canonical JSON
(`courant.verdict/2`), `--html` and `--report md` as a report, and `--pack` writes an evidence
pack that `--check-pack` re-derives from its inputs. `--spec` with `--spec-sha256` adds the task:
every value the specification names is checked against the case
([Getting started](https://github.com/courantdev/courant/blob/main/docs/GETTING-STARTED.md),
section 2b).

## The rule this repository keeps

Deterministic code decides every verdict, number and grade. A language model may draft prose,
and may extract a task specification that is shown, hashed and can only lower a verdict to
`insufficient`; nothing else, and nothing a model writes is in the verdict's path. Courant never
edits a case, never runs a solver, and its checks never use the network. A pull request that
lets a model, a threshold nobody recorded or the network into the verdict's path is not merged
([CONTRIBUTING.md](https://github.com/courantdev/courant/blob/main/CONTRIBUTING.md)).

These tests in this repository hold the model-free, read-only, offline and reproducible parts of
it:

- `tests/unit/test_public_verdict_path.py`: no module the verdict is made of (the gate, the
  checks it calls, and the commands and tools that read a run) imports a language-model client
  or a network library, names a model provider's endpoint or starts a process (but for the
  workers `--jobs` asks for), and importing all of it loads no model client. The lint reads the
  code's syntax, and is first run on code that breaks each of its rules.
- `tests/unit/test_public_network.py`: `courant verify` on the demo runs, in each of its
  outputs, opens no socket and starts no process; every visible command and every tool of the
  MCP server runs on the demo data with network connections refused; and each command whose
  purpose is a network call names its host first
  ([NETWORK.md](https://github.com/courantdev/courant/blob/main/docs/NETWORK.md)).
- `tests/unit/test_mcp_tools.py` and `tests/unit/test_mcp_e2e.py`: the agent tools refuse a path
  outside the directories named, change no file and open no network connection.
- `tests/unit/test_action.py`: the CI gate opens no network connection and writes only where it
  is told.
- `tests/unit/test_preflight.py`, `tests/unit/test_live.py` and `tests/unit/test_audit.py`:
  preflight, `watch` and the audit read a case and leave every byte of it as it was.
- `tests/unit/test_verify_pack.py`: an evidence pack is the same bytes when written twice, an
  edit to a file the gate read changes that file's entry in the manifest, and `--check-pack`
  re-verifies only the run under `--root`, as the root holds it: a pack whose manifest names a
  file outside it, lists other files than its verdict read, leaves out the sidecar the root holds
  for its log, judges a case directory by its log alone, or carries another specification than
  the one `--spec-sha256` names, is refused.
- `tests/integration/test_docs_commands.py`: every `bash` block in this README and in Getting
  started runs, offline, on the demo data, with the exit code it documents.

## Use it from an agent

`courant mcp` serves the gate to any AI agent over MCP (stdio), read-only and confined to the
directories you name. Print the settings for your client:

```bash
courant mcp --print-config --root ./runs
```

For Claude Code that is one command:

<!-- docs-skip: needs the claude CLI, which CI does not have -->
```bash
claude mcp add courant -- courant mcp --root ~/runs
```

The server offers six read-only tools: `verify_run`, `score_agent_output`, `check_convergence`,
`explain_code`, `list_failure_modes` and `verifier_certification`. `score_agent_output` is
advice to the agent before it reports, not a certification: given the task specification and
whether the agent is about to report success, it returns the verdict and whether the run is a
*silent error* (it executed, the agent would report success, and the gate rejects it).

## In CI: the GitHub Action

[`courant-gate`](https://github.com/courantdev/courant/tree/main/.github/actions/courant-gate)
verifies the runs a repository names, writes one summary row per run and one evidence pack per
run, and fails the check unless every run is accepted. Its logic is `courant gate`, which runs
the same way on any CI system.

```yaml
- uses: courantdev/courant/.github/actions/courant-gate@v0.6.0
  with:
    paths: runs/*.log.gz
    spec: task/spec.json
    spec-sha256: ${{ vars.TASK_SPEC_SHA256 }}
```

Without the Action, `courant verify --batch runs/ --junit courant.xml` writes one JUnit test per
run, which any CI system shows.

## Commands

| Command | What it does |
|---|---|
| `courant verify <case or log>` | `accept`, `reject` or `insufficient` (exit 0, 2 or 3; `courant.verdict/2`), with the basis it was judged on and what was not checked. `--spec` with `--spec-sha256` checks every value a task specification names against the case. `--html`, `--report md`, `--json`, `--pack` and `--check-pack` (an evidence pack anyone can re-derive), `--batch` with `--junit`, `--record`, `--about` |
| `courant spec init / hash / check / lock / confirm / extract` | Draft a task specification from a reference case, lock it by hash, and check a case against it before or after its run |
| `courant gate` | The CI check, and the [GitHub Action](https://github.com/courantdev/courant/tree/main/.github/actions/courant-gate): verify the runs a repository names, one summary row per run, one evidence pack per run |
| `courant preflight <case>` | A broken setup, caught before the solver spends anything |
| `courant convergence <case>` | Where the monitored answer stopped changing, and the iterations, wall clock and core-hours after it |
| `courant watch <case>` | A read-only live advisory while the solver runs, with `--exit-on-fail` for job scripts |
| `courant audit <dir>` | What a folder of runs cost, and how much of it bought nothing |
| `courant test run --judge …` | The Courant Test: score any judge, including an AI model, on real OpenFOAM runs |
| `courant mcp` | The same tools for any AI agent over MCP, read-only and confined to the directories named; `score_agent_output` tells an agent, before it reports, whether its run is a silent error |
| `courant doctor`, `courant update`, `courant version` | Diagnostics and a support bundle you send yourself; a signed update of the binary |

Start with
[GETTING-STARTED.md](https://github.com/courantdev/courant/blob/main/docs/GETTING-STARTED.md).

## Configuration and the network

Courant has no account, no telemetry and no service of its own, and needs no configuration to
verify a run. What it reads when it is there:

| Setting | What it does |
|---|---|
| `courant.toml` | For `courant gate` and `courant verify --batch`: `fail-on`, `pack-store`, `courant-version`, and `[[run]]` tables of `paths`, `outputs`, `spec` and `spec-sha256`. An unknown key is an error that names it |
| `COURANT_GATE_*` | Every `courant gate` option as an environment variable; the Action sets them from its inputs |
| `COURANT_CACHE_DIR` | Where `--batch` keeps its verdict cache (else `~/.cache/courant/verdicts`); off by default where `CI` is set |
| `COURANT_DEBUG=1` | Print the traceback of an internal error, as `--debug` does |
| `COURANT_REQUIRE_SIGNATURE=1` | The installer refuses a binary whose signature it cannot check |
| `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` | Used only by the commands whose purpose is a model call you name, and `courant doctor` says whether one is set; a paid model is refused unless `COURANT_ALLOW_PAID_LLM=1` is set. A key may also sit in a `.env` file; `COURANT_NO_DOTENV=1` ignores it |

A fixture repository's `courant.toml`, whose check fails on purpose:

```toml
fail-on = "not-accept"
pack-store = "evidence"

[[run]]
paths = ["runs/*.log.gz"]
```

The checks never use the network. A command contacts something only when that is its purpose
and you ran it, and it names the host first: `courant update` (github.com), `courant doctor
--models`, and a language model you name
([NETWORK.md](https://github.com/courantdev/courant/blob/main/docs/NETWORK.md)).

## From Python

```python
from courant.verify import verify_path

v = verify_path("demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.log.gz",
                "demo/icoFoam_cavity__cfl_blowup/icoFoam_cavity__cfl_blowup.case.json")
v.verdict, v.rule, v.reasons[0].code     # ('reject', 'VER-5.3', 'cfl.max')
v.exit_code                              # 2
print(v.to_json())                       # the canonical courant.verdict/2
```

`verify_path` takes a case directory or a solver log, and `spec=` with `spec_locked_sha256=` for
the task.

## Benchmarks

On OpenFOAM v2406, with every figure carrying its denominator (`courant verify --about` prints
the rates that ship with each version):

- **Detection:** 31 of 31 real runs judged correctly. 0 of 23 wrong results accepted and 0 of 8
  good results rejected.
- **The wrong problem:** on 20 real runs with the right physics and a wrong input (inlet
  doubled, viscosity tripled, gravity rotated, geometry scaled, end time cut), the verdict
  without a specification rejected 8 of 20; against the task specification, 20 of 20.
- **Before the run:** preflight caught 8 of 23 broken setups, and flagged 0 of 8 good ones.
- **Fixed budgets:** where the answer settled, 1,023 of 1,999 iterations (51.2%, pitzDaily)
  and 547 of 999 (54.8%, T3A) came after it.
- **Reproducible:** evidence packs re-derive byte for byte, 31 of 31.
- **Beside frontier models:** on the same 420 real runs, given the task's specification, Courant
  accepted 0 of 264 wrong runs; Claude Opus 5.5 accepted 69 of 264, Claude Sonnet 5.5 76 of 263
  and Claude Haiku 5.5 52 of 249. Courant rejected 10 of 140 right runs.
- **At scale:** on 1,827 real runs, 0 of 609 faulted and 0 of 609 wrong-problem runs were
  accepted with the task's specification.

Every figure, its interval and its pre-registered bar, including the one that missed, are in
[RESULTS.md](https://github.com/courantdev/courant/blob/main/RESULTS.md).

"Certified" here means measured against a real solver, with the error rate published. It is not
a regulator's approval. The runs are tutorial-sized, with injected faults; no customer data.

## Layout

```text
courant/
  verify/          the gate: ingest, checks, policy, the verdict, reports, evidence packs, batch
  spec/            task specifications (courant.spec/1): draft, hash, lock, check against a case
  gate/            courant gate, behind the GitHub Action
  case/            an OpenFOAM case as the solver read it: dictionaries, mesh, regime
  watcher/         the log parser: solver output to events
  verification/    the physics checks and the grade
  diagnosis/       findings and probable causes
  converge/        where the answer stopped changing
  live/            courant watch
  record/          verified records (courant.record/1)
  ctest/           the Courant Test and its judges
  llm/             the model door: the one way to a language model, for the commands that call one
  mcp/             the MCP server and its read-only tools
  cli/             the courant command
  data/            shipped with the package: the demo runs, the published error rates, the schemas
certification/     the real OpenFOAM v2406 runs the gate is measured on
tests/             unit, integration and robustness tests, all offline
packaging/         the single-file binary, the installer, the hashed release locks
.github/           the courant-gate Action and the CI and release workflows
docs/              Getting started, NETWORK.md, COVERAGE.md, and these figures (docs/media/)
```

## Development

<!-- docs-skip: builds a virtual environment -->
```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,mcp]"
.venv/bin/python -m ruff check --select F,E9 courant tests
.venv/bin/python -m pytest -q          # offline: no Docker, no network, no solver
```

A change comes with a test that fails without it, and real solver output beats a synthetic log.
See [CONTRIBUTING.md](https://github.com/courantdev/courant/blob/main/CONTRIBUTING.md).

## Security

Report a vulnerability privately, through GitHub's private vulnerability reporting on this
repository or by email to hello@courant.dev. A verdict a crafted input can turn to `accept`, a
command that reads outside the directories it was given or opens a connection that is not its
purpose, and a release whose signature check can be bypassed all count. See
[SECURITY.md](https://github.com/courantdev/courant/blob/main/SECURITY.md).

## License

Licensed under the [Apache License 2.0](https://github.com/courantdev/courant/blob/main/LICENSE).

Copyright 2026 Shivanshu Dwivedi (Courant, https://courant.dev)
