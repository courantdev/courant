"""The Evidence Bundle — what the LLM actually sees (Tech Arch §3.3).

"The single most important data contract. The LLM never sees a raw 50,000-line log —
it sees a compact, reproducible bundle assembled deterministically on each event."

Because assembly is deterministic, the same failure always produces the same bundle,
making diagnoses reproducible and the eval harness meaningful. Token budget target
< 8k via downsampling (residual series) and digesting (config, event window).
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from courant.case.config import CaseConfig
from courant.case.mesh import MeshQuality
from courant.diagnosis.findings import Finding, FindingSet
from courant.diagnosis.kb import KBMatch, KnowledgeBase
from courant.events import Event, EventKind

MAX_EVENT_WINDOW = 30
MAX_RESIDUAL_POINTS = 40


class CaseSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    solver: str | None = None
    cell_count: int | None = None
    key_dicts_digest: str = ""            # short deterministic hash of config_view


class PriorAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    diff: str
    outcome: str                          # e.g. "still diverged", "converged"


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    case_summary: CaseSummary
    config_view: dict = Field(default_factory=dict)
    mesh_quality: dict | None = None
    event_window: list[dict] = Field(default_factory=list)
    residual_series: dict[str, list[float]] = Field(default_factory=dict)
    residual_iterations: list[float] = Field(default_factory=list)
    prior_attempts: list[PriorAttempt] = Field(default_factory=list)
    det_findings: list[Finding] = Field(default_factory=list)
    kb_matches: list[KBMatch] = Field(default_factory=list)
    #: Allowlisted solver lines with their byte offsets. Not shown to the model
    #: (``log_window`` is the flat view it reads) — these exist so the console
    #: can resolve a Finding's ``log_offset`` back to the line that produced it,
    #: which is the difference between evidence and an assertion.
    log_lines: list[dict] = Field(default_factory=list)
    log_window: str = ""

    def to_llm_json(self) -> str:
        """Compact JSON the model is prompted with."""
        return json.dumps(self.model_dump(mode="json"), separators=(",", ":"))

    def approx_tokens(self) -> int:
        """Rough token estimate (~4 chars/token) for budget checks."""
        return len(self.to_llm_json()) // 4


def _downsample(values: list[float], max_points: int) -> list[int]:
    """Indices of an evenly-strided sample that always includes first and last."""
    n = len(values)
    if n <= max_points:
        return list(range(n))
    stride = n / max_points
    idx = sorted({int(i * stride) for i in range(max_points)} | {0, n - 1})
    return idx


def _extract_residual_series(events: list[Event]) -> tuple[dict[str, list[float]], list[float]]:
    residual_events = [e for e in events if e.kind is EventKind.RESIDUAL]
    if not residual_events:
        return {}, []
    # x-axis: iteration index (fallback to running count)
    xs_full = [
        (e.payload.iteration if e.payload.iteration is not None else i)
        for i, e in enumerate(residual_events)
    ]
    idx = _downsample(xs_full, MAX_RESIDUAL_POINTS)
    fields: dict[str, list[float]] = {}
    for i in idx:
        for name, val in residual_events[i].payload.initial_residuals.items():
            fields.setdefault(name, [])
    # fill aligned series (NaN-safe: use None where missing -> filtered to float where present)
    series: dict[str, list[float]] = {name: [] for name in fields}
    for i in idx:
        res = residual_events[i].payload.initial_residuals
        for name in series:
            v = res.get(name)
            series[name].append(float(v) if v is not None else float("nan"))
    return series, [float(xs_full[i]) for i in idx]


def _event_window(events: list[Event]) -> list[dict]:
    window = events[-MAX_EVENT_WINDOW:]
    out = []
    for e in window:
        out.append({
            "seq": e.seq,
            "kind": e.kind.value,
            "severity": e.severity.value,
            "ts_sim": e.ts_sim,
            "payload": e.payload.model_dump(mode="json", exclude_none=True,
                                            exclude={"kind"}),
        })
    return out


def _config_view(config: CaseConfig | None) -> dict:
    if config is None:
        return {}
    # curated, non-empty subset
    d = config.model_dump(mode="json", exclude_none=True)
    return {k: v for k, v in d.items() if v not in ({}, [], None)}


def assemble_bundle(
    run_id: str,
    events: list[Event],
    findings: FindingSet,
    kb: KnowledgeBase,
    config: CaseConfig | None = None,
    mesh: MeshQuality | None = None,
    log_text: str = "",
    cell_count: int | None = None,
    prior_attempts: list[PriorAttempt] | None = None,
) -> EvidenceBundle:
    """Deterministically assemble the Evidence Bundle from run artifacts."""
    config_view = _config_view(config)
    digest = hashlib.sha256(
        json.dumps(config_view, sort_keys=True).encode()
    ).hexdigest()[:12]

    series, iters = _extract_residual_series(events)
    kb_matches = kb.query(findings=findings, log_text=log_text, config=config, top_k=5)

    from courant.diagnosis.redact import redact_log_lines, redact_log_window

    return EvidenceBundle(
        run_id=run_id,
        log_lines=redact_log_lines(log_text),
        log_window=redact_log_window(log_text),
        case_summary=CaseSummary(
            solver=config.solver if config else None,
            cell_count=cell_count or (mesh.n_cells if mesh else None),
            key_dicts_digest=digest,
        ),
        config_view=config_view,
        mesh_quality=(mesh.model_dump(mode="json", exclude_none=True) if mesh else None),
        event_window=_event_window(events),
        residual_series=series,
        residual_iterations=iters,
        prior_attempts=prior_attempts or [],
        det_findings=list(findings),
        kb_matches=kb_matches,
    )
