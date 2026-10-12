"""`courant preflight`: catch a broken setup before the solver runs (TRD PRE-1).

Exit codes:
- 0: NO_SETUP_FAULT_FOUND;
- 2: SETUP_FAULT;
- 3: NOT_ASSESSABLE (no readable configuration);
- 4: the input could not be read.

Read-only and offline. No log is needed, because nothing has run yet.
"""

from __future__ import annotations

import sys

import typer

EXIT = {"NO_SETUP_FAULT_FOUND": 0, "SETUP_FAULT": 2, "NOT_ASSESSABLE": 3}


def preflight(
    case: str = typer.Argument(..., help="A case directory, before (or after) it runs."),
    case_json: str = typer.Option(None, "--case-json", help="A captured configuration sidecar (*.case.json) instead of a case directory."),
    as_json: bool = typer.Option(False, "--json", help="Print the canonical preflight JSON."),
) -> None:
    """Check a case's setup before the solver spends anything: tolerances, relaxation, boundaries, units, mesh."""
    from courant.verify.ingest import InputError
    from courant.verify.preflight import preflight_path, render_text

    try:
        p = preflight_path(case, case_json)
    except (InputError, OSError, ValueError) as exc:
        print(f"courant preflight: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    sys.stdout.write(p.to_json() if as_json else render_text(p))
    raise typer.Exit(EXIT[p.verdict])
