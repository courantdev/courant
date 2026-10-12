"""Courant convergence: where the engineering answer stopped changing (TRD §18).

    from courant.converge import convergence_path
    conv = convergence_path("./airfoil")
    conv.status                       # PASS | WARN | FAIL | NOT_ASSESSABLE
    conv.data["converged_at_step"]    # the last monitor to settle
    conv.data["waste"]                # iterations, wall clock and core-hours after it

Read-only (INV-5). The arithmetic is pure and exact (`exact.py`). Only `series.py` reads files.
"""

from __future__ import annotations

from pathlib import Path

from courant.converge.assess import Convergence, Options, assess
from courant.converge.render import render_markdown, render_text, summary
from courant.converge.series import assess_case

SCHEMA = "courant.convergence/1"


def from_run_inputs(inputs, options: Options = Options()) -> Convergence:
    """Assess a run the verifier has already ingested (one parse of the log)."""
    from courant.case.config import CaseConfig
    from courant.watcher.exitcode import infer_exit_code
    from courant.watcher.parser import LogParser

    exit_code = inputs.exit_code if inputs.exit_code is not None else infer_exit_code(inputs.log_text)
    events = list(LogParser(run_id=inputs.run_id).parse_text(inputs.log_text, return_code=exit_code))
    config = None
    if inputs.config_view:
        try:
            config = CaseConfig.model_validate(inputs.config_view)
        except Exception:                                   # noqa: BLE001
            config = None
    return assess_case(inputs.case_dir, log_text=inputs.log_text, events=events, config=config,
                       solver=inputs.solver_hint, options=options, read=inputs.inputs)


def convergence_path(path: str | Path, case_json: str | Path | None = None,
                     options: Options = Options()) -> Convergence:
    from courant.verify.ingest import from_path

    return from_run_inputs(from_path(path, case_json), options)


def to_document(conv: Convergence, run_id: str, solver: str | None) -> dict:
    """The `--json` document: canonical, versioned, and carrying its own inputs."""
    from courant import __version__

    return {"schema": SCHEMA, "tool": {"name": "courant", "version": __version__},
            "run": {"id": run_id, "solver": solver}, "convergence": conv.as_dict()}


__all__ = ["SCHEMA", "Convergence", "Options", "assess", "assess_case", "convergence_path",
           "from_run_inputs", "render_markdown", "render_text", "summary", "to_document"]
