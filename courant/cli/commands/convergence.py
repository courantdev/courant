"""`courant convergence`: where the answer stopped changing, and what the rest cost (TRD CONV-13).

Exit codes:
- 0 PASS or WARN
- 2 FAIL
- 3 NOT_ASSESSABLE
- 4 the inputs could not be read

Read-only. Named `convergence`, not `stop`: a command called `stop` that does not stop the
solver is a trap for the person typing it.
"""

from __future__ import annotations

import sys

import typer

EXIT = {"PASS": 0, "WARN": 0, "FAIL": 2, "NOT_ASSESSABLE": 3}


def _pairs(items: list[str], flag: str) -> dict[str, float]:
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"{flag} expects NAME=VALUE, got {item!r}")
        out[name] = float(value)
    return out


def convergence(
    path: str = typer.Argument(..., help="A case directory (monitors are read from postProcessing/), or a solver log."),
    case_json: str = typer.Option(None, "--case-json", help="Captured configuration sidecar (*.case.json)."),
    as_json: bool = typer.Option(False, "--json", help="Print the canonical convergence JSON."),
    tol: list[str] = typer.Option(None, "--tol", help="Absolute tolerance for a monitor or column: NAME=VALUE. Repeatable."),
    tol_rel: list[str] = typer.Option(None, "--tol-rel", help="Relative tolerance: NAME=VALUE (0.01 is one part in a hundred). Repeatable."),
    ref_scale: list[str] = typer.Option(None, "--ref-scale", help="Reference scale for a relative tolerance: NAME=VALUE."),
    transient_average: bool = typer.Option(False, "--transient-average", help="Judge transient monitors as time-averages."),
    monitor: list[str] = typer.Option(None, "--monitor", help="Judge only monitors matching this glob (id or column), e.g. '*:Cd'. Repeatable. Default: the totals each function object writes."),
    cost: float = typer.Option(None, "--cost-per-core-hour", help="Your rate, to price the core-hours past convergence."),
    ccf: float = typer.Option(None, "--ccf-ghz", help="FIA CCF in GHz: also report allocation unit hours (Art. F4.4.4)."),
) -> None:
    """Where the answer stopped changing, and the iterations, wall clock and core-hours after it."""
    from pathlib import Path

    from courant.converge import Options, from_run_inputs, render_text, to_document
    from courant.converge.series import solver_from_log
    from courant.verify.ingest import InputError, from_path
    from courant.verify.schema import canonical_json

    try:
        options = Options(tol_abs=_pairs(tol, "--tol"), tol_rel=_pairs(tol_rel, "--tol-rel"),
                          ref_scale=_pairs(ref_scale, "--ref-scale"),
                          transient_average=transient_average, monitors=tuple(monitor or ()),
                          cost_per_core_hour=cost, ccf_ghz=ccf)
        inputs = from_path(path, case_json)
        conv = from_run_inputs(inputs, options)
    except (InputError, ValueError) as exc:
        print(f"courant convergence: {exc}", file=sys.stderr)
        raise typer.Exit(4) from exc
    name = Path(path).name
    solver = inputs.solver_hint or solver_from_log(inputs.log_text)
    if as_json:
        sys.stdout.write(canonical_json(to_document(conv, name, solver)))
    else:
        sys.stdout.write(render_text(conv, name, solver))
    raise typer.Exit(EXIT[conv.status])
