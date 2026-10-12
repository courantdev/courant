"""Task specifications and deck-versus-spec conformance (PRD 2.0 P2-SPEC).

Without a specification, Courant decides whether a run is *valid*. With one, it decides
whether the run is *the problem that was asked*: every value the specification names is
compared to the case, deterministically, and a run that solved a different problem is
rejected however well it converged (RRD RM-18).

    from courant.spec import load_spec, check_conformance, draft_spec
    spec, sha = load_spec("spec.json")
    block = check_conformance(spec, "./case", spec_sha256=sha)

Read-only, deterministic, no network (INV-5).
"""

from __future__ import annotations

from courant.spec.check import check_conformance, conforms, draft_spec
from courant.spec.model import (
    SPEC_SCHEMA,
    Provenance,
    Spec,
    SpecError,
    SpecItem,
    load_spec,
    spec_sha256,
)

__all__ = [
    "SPEC_SCHEMA", "Provenance", "Spec", "SpecError", "SpecItem",
    "check_conformance", "conforms", "draft_spec", "load_spec", "spec_sha256",
]
