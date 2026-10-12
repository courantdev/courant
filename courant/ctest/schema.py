"""Courant Test contracts (TRD CT-1, CT-2, CT-5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["TRUSTWORTHY", "NOT_TRUSTWORTHY", "NOT_ASSESSABLE"]


class Task(BaseModel):
    """What a judge sees. Never the label, never the fault's name."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    track: Literal["judge"] = "judge"
    family: str
    solver: str
    log_name: str
    log_text: str
    config_view: dict | None = None
    mesh_view: dict | None = None


class Label(BaseModel):
    """Ground truth. Read by the scorer only (CT-9)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    truth: Literal["TRUSTWORTHY", "NOT_TRUSTWORTHY"]
    expects_code: str | None = None
    fault_id: str | None = None


class JudgeAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    verdict: Verdict
    codes: tuple[str, ...] = ()
    rationale: str = ""
    #: A judge that failed to produce a usable answer. Scored as wrong, never guessed.
    error: str | None = Field(default=None)
