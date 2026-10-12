"""`courant audit`: what a folder of runs cost, and how much of it bought nothing (TRD AUDIT-1).

Exit codes follow `courant verify --batch`:
- 2 if any run is NOT_TRUSTWORTHY;
- 3 if any is NOT_ASSESSABLE;
- 4 if any could not be read;
- 0 otherwise.

Read-only and offline. Writes only the files you name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer


def audit(
    root: str = typer.Argument(..., help="A directory of runs: case directories with a log, or *.log / *.log.gz files."),
    cost: float = typer.Option(None, "--cost-per-core-hour", help="Your rate, to price the core-hours. Without it no money is shown."),
    as_json: bool = typer.Option(False, "--json", help="Print the canonical audit JSON."),
    html: str = typer.Option(None, "--html", help="Write a self-contained HTML audit to this file."),
) -> None:
    """The compute audit: core-hours on untrusted results, after convergence, and on broken setups."""
    from courant.verify.audit import audit as run_audit
    from courant.verify.audit import render_html, render_text
    from courant.verify.schema import canonical_json

    if not Path(root).is_dir():
        print(f"courant audit: not a directory: {root}", file=sys.stderr)
        raise typer.Exit(4)
    doc = run_audit(root, cost)
    if html:
        Path(html).write_text(render_html(doc))
        print(f"audit written: {html}", file=sys.stderr)
    sys.stdout.write(canonical_json(doc) if as_json else render_text(doc))
    t = doc["totals"]
    raise typer.Exit(2 if t["not_trustworthy"] else 3 if t["not_assessable"] else 4 if t["unreadable"] else 0)
