# Courant

**Physics acceptance testing for machine-produced simulation.** Courant is a read-only,
deterministic gate between a simulation (an AI agent's or an engineer's) and the person who
signs it. Without a task specification it decides whether an OpenFOAM run is *valid*; with the
task the run was asked to solve, it decides whether the run is *the problem that was asked*.

Every verdict is `accept`, `reject` or `insufficient`, and every reason cites the file it read,
the method, the threshold and the value. Deterministic code decides every verdict; no language
model is in the verdict's path. Courant never edits a case, never runs a solver, and its checks
never use the network.

Licensed under the [Apache License 2.0](https://github.com/courantdev/courant/blob/main/LICENSE).

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

## What it does

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

The checks never use the network. A command contacts something only when that is its purpose
and you ran it, and it names the host first: `courant update` (github.com), `courant doctor
--models`, and a language model you name
([NETWORK.md](https://github.com/courantdev/courant/blob/main/docs/NETWORK.md)). Start with
[GETTING-STARTED.md](https://github.com/courantdev/courant/blob/main/docs/GETTING-STARTED.md).

## What is measured

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
a regulator's approval. Courant has no account, no telemetry and no service of its own.

## Contributing and security

See [CONTRIBUTING.md](https://github.com/courantdev/courant/blob/main/CONTRIBUTING.md) and
[SECURITY.md](https://github.com/courantdev/courant/blob/main/SECURITY.md).

Copyright 2026 Shivanshu Dwivedi (Courant, https://courant.dev)
