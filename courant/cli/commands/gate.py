"""`courant gate`: the check a CI job runs (TRD GATE-7, GATE-8).

Verifies the runs a repository names, writes a Markdown job summary and an evidence pack per
run, and exits 0 when the check passes. It is what the `courant-gate` GitHub Action calls: every
Action input arrives here as an environment variable (`COURANT_GATE_*`), so the Action has no
logic of its own and the same check runs on a laptop.

Exit codes:
- 0 the check passes
- 1 a run was judged and fails the check
- 4 something could not be judged: a configuration error, a pattern that matches no run, a run
     with no output, an unreadable input

Settings come from `courant.toml` in --root (or --config), then these flags, which add one
more [[run]] group and override the top-level keys. An unknown key in `courant.toml` is an
error. Read-only apart from --packs-dir, --pack-store and --summary. No network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer


def _split(values: list[str] | None) -> list[str]:
    """A list option from the Action arrives as one newline- or comma-separated string."""
    out: list[str] = []
    for v in values or []:
        out += [p.strip() for chunk in v.splitlines() for p in chunk.split(",") if p.strip()]
    return out


def gate(
    root: str = typer.Option(".", "--root", envvar="COURANT_GATE_ROOT", help="The repository root; globs and paths are relative to it."),
    config: str = typer.Option(None, "--config", envvar="COURANT_GATE_CONFIG", help="The settings file (default: courant.toml in --root, if it exists)."),
    paths: list[str] = typer.Option(None, "--paths", envvar="COURANT_GATE_PATHS", help="Case globs: case directories and solver logs. Newline- or comma-separated from the Action."),
    outputs: list[str] = typer.Option(None, "--outputs", envvar="COURANT_GATE_OUTPUTS", help="Where each run's log and results are: a directory with one entry per run name."),
    spec: str = typer.Option(None, "--spec", envvar="COURANT_GATE_SPEC", help="The task specification (courant.spec/1) the runs were asked to meet."),
    spec_sha256: str = typer.Option(None, "--spec-sha256", envvar="COURANT_GATE_SPEC_SHA256", help="The hash the specification was locked at. Required with --spec."),
    fail_on: str = typer.Option(None, "--fail-on", envvar="COURANT_GATE_FAIL_ON", help="not-accept (default): anything but accept fails. reject: an insufficient run passes."),
    pack_store: str = typer.Option(None, "--pack-store", envvar="COURANT_GATE_PACK_STORE", help="Also keep each run's evidence pack at this path in the repository."),
    courant_version: str = typer.Option(None, "--courant-version", envvar="COURANT_GATE_COURANT_VERSION", help="Refuse to run under another Courant version ('latest' pins nothing)."),
    packs_dir: str = typer.Option(None, "--packs-dir", envvar="COURANT_GATE_PACKS_DIR", help="Write each run's evidence pack here (the Action uploads it as an artefact)."),
    summary: str = typer.Option(None, "--summary", envvar="COURANT_GATE_SUMMARY", help="Append the Markdown summary to this file ($GITHUB_STEP_SUMMARY in a workflow)."),
    jobs: int = typer.Option(1, "--jobs", min=1, help="Verify this many runs at once."),
    cache_on: bool = typer.Option(False, "--cache", help="Use the verdict cache ($COURANT_CACHE_DIR). Off by default: a CI runner is a shared, throw-away machine."),
    debug: bool = typer.Option(False, "--debug", help="Print the traceback of an unexpected error."),
) -> None:
    """The CI check: verify the named runs, write the summary and the packs, pass or fail."""
    from courant.gate.config import ConfigError, GateConfig, load_config, make_group, merged
    from courant.gate.run import render_summary, run_gate
    from courant.verify.cache import VerdictCache

    root_path = Path(root)
    try:
        cfg_path = Path(config) if config else root_path / "courant.toml"
        if config and not cfg_path.is_file():
            raise ConfigError(f"no such settings file: {config}")
        base = load_config(cfg_path) if cfg_path.is_file() else GateConfig()
        group = None
        if _split(paths):
            group = make_group(_split(paths), _split(outputs), spec or None, spec_sha256 or None,
                               where="the Action's inputs")
        elif outputs or spec or spec_sha256:
            raise ConfigError("`outputs`, `spec` and `spec-sha256` apply to `paths`, which is empty")
        cfg = merged(base, group=group, fail_on=fail_on or None, pack_store=pack_store or None,
                     courant_version=courant_version or None)
        result = run_gate(root_path, cfg, packs_dir=packs_dir or None, jobs=jobs,
                          cache=VerdictCache() if cache_on else None)
    except ConfigError as exc:
        print(f"courant gate: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    except Exception as exc:                            # GATE-10: no traceback
        if debug:
            raise
        from courant import __version__

        print(f"courant gate: internal error (courant {__version__}): "
              f"{type(exc).__name__}: {exc}. Rerun with --debug for the traceback.", file=sys.stderr)
        raise typer.Exit(4) from exc
    text = render_summary(result)
    sys.stdout.write(text)
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    raise typer.Exit(result.exit_code)
