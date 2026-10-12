"""Courant Verify: a verdict on a completed simulation, with its evidence.

    from courant.verify import verify_path
    verdict = verify_path("./pitzDaily")
    verdict.verdict      # "accept" | "reject" | "insufficient"   (courant.verdict/2)
    verdict.basis        # "validity" | "validity+conformance" | ...
    verdict.to_v1()      # the same decision as courant.verdict/1 (TRUSTWORTHY, ...)

Read-only: no case is edited, no solver is run, no socket is opened (INV-5).
"""

from __future__ import annotations

from pathlib import Path

from courant.verify.ingest import InputError, from_case_dir, from_log, from_path, with_spec
from courant.verify.pipeline import verify_inputs
from courant.verify.render import render_json, render_markdown, render_text
from courant.verify.schema import Reason, RunInputs, Verdict
from courant.verify.verdict2 import VerdictV2, load_verdict


def verify_path(path: str | Path, case_json: str | Path | None = None, *,
                spec: str | Path | None = None,
                spec_locked_sha256: str | None = None) -> VerdictV2:
    inputs = from_path(path, case_json)
    if spec is not None:
        inputs = with_spec(inputs, spec, spec_locked_sha256)
    return verify_inputs(inputs)


__all__ = [
    "InputError", "Reason", "RunInputs", "Verdict", "VerdictV2",
    "from_case_dir", "from_log", "from_path", "load_verdict",
    "render_json", "render_markdown", "render_text",
    "verify_inputs", "verify_path", "with_spec",
]
