"""A specification drafted from a task in words, through the model door (TRD SPEC-7; PRD 2.0
P2-SPEC-4).

    from courant.spec.extract import extract_spec, confirm
    out = extract_spec(task_text, door=door, data="customer")
    out.spec.to_json()          # shown to a person, hashed, marked extracted

What the model may do here, and what it may not (INV-1, amended 4 Oct 2026): it proposes items,
locations and values. Deterministic code turns the answer into a `courant.spec/1` (every item is
validated by the same model a hand-written specification is, and one that fails is dropped and
shown), and records `provenance: {extracted_by, model, date}` inside the hashed content. Until a
person confirms it (`confirm`, `courant spec confirm --by NAME`), the specification is not
binding: the policy is told so (VER-5.12) and a mismatch can only lower a verdict to
`insufficient`. It never produces `reject` and never decides an `accept`.

Which model may extract is the door's decision, not this module's: a call names task
`spec-extract`, and the door sends it only to an allowlisted model measured for that task (RRD
RM-23, ARC-12, SPEC-8) and, for customer data, only to a zero-data-retention endpoint (ARC-9).
Until RM-23 has passed a model, the door refuses and the caller writes the specification by hand
or drafts it from a reference case (`courant spec init`).

`courant.verify` never imports this module (INV-5): verification opens no socket.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as calendar_date

from pydantic import ValidationError

from courant.spec.model import SPEC_SCHEMA, Provenance, Spec, SpecItem

#: The version of the prompt below. A changed prompt is a different measurement (ARC-20).
PROMPT_VERSION = "spec-extract/1"
MAX_TOKENS = 2000

#: What the model is asked to answer. The door validates the answer against it (C7c).
ANSWER_SCHEMA: dict = {
    "type": "object",
    "required": ["items"],
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "file"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "file": {"type": "string"},
                    "patch": {"type": ["string", "null"]},
                    "key": {"type": ["string", "null"]},
                    "path": {"type": "array", "items": {"type": "string"}},
                    "value": {"type": ["number", "string", "array", "null"],
                              "items": {"type": "number"}},
                    "min": {"type": ["number", "null"]},
                    "max": {"type": ["number", "null"]},
                    "units": {"type": ["string", "null"]},
                },
            },
        },
    },
}

SYSTEM = """You turn a CFD task, written in words for an OpenFOAM user, into the list of values \
the task states. Name only values the task states; never invent one it leaves open.

Each item names one value and where it lives in the OpenFOAM case:
- a boundary value: "file" is the field file ("0/U", "0/T", "0/k"), "patch" the patch name, and \
"key" only if the entry is not "value" (for example "inletValue");
- any other entry: "file" (for example "constant/transportProperties", \
"constant/thermophysicalProperties", "constant/g", "system/controlDict"), "key" the entry name \
("nu", "mu", "rho", "value", "endTime", "application"), and "path" the dictionaries it sits in \
(for example ["mixture", "transport"] for mu, ["water"] for a phase's nu);
- "value": a number, a list of three numbers for a vector, or a word for a choice such as the \
solver ("icoFoam"); in SI units; "units" says which.
- "min" or "max" instead of "value" when the task states only a bound ("at least 400 cells").
Give each item a short unique "name"."""


class ExtractionError(ValueError):
    """The answer held no item a specification can carry."""


@dataclass(frozen=True)
class Extraction:
    """A drafted specification and what was left out of it, with the reason for each."""

    spec: Spec
    dropped: tuple[str, ...]
    model: str
    provider: str


def extractor_name() -> str:
    from courant import __version__

    return f"courant spec extract {__version__}"


def user_prompt(task_text: str) -> str:
    return f"The task:\n\n{task_text.strip()}\n"


def _item(raw: dict) -> SpecItem:
    data = {k: v for k, v in raw.items() if v is not None and v != []}
    if isinstance(data.get("value"), list):
        data["value"] = tuple(data["value"])
    if "path" in data:
        data["path"] = tuple(data["path"])
    return SpecItem.model_validate(data)


def build_spec(task_text: str, answer: dict, *, model: str, date: str) -> tuple[Spec, tuple[str, ...]]:
    """The specification deterministic code builds from a model's answer: each item validated,
    one that fails dropped (and said why), names unique, provenance recorded. Pure."""
    items: list[SpecItem] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(answer.get("items") or ()):
        label = raw.get("name") if isinstance(raw, dict) and raw.get("name") else f"item {index}"
        try:
            item = _item(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            first = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            dropped.append(f"{label}: {first}")
            continue
        if item.name in seen:
            dropped.append(f"{label}: a second item with the same name")
            continue
        seen.add(item.name)
        items.append(item)
    if not items:
        raise ExtractionError("the model's answer held no item a specification can carry"
                              + (f" ({'; '.join(dropped)})" if dropped else ""))
    spec = Spec(schema=SPEC_SCHEMA, task=task_text.strip() or None, items=tuple(items),
                provenance=Provenance(extracted_by=extractor_name(), model=model, date=date))
    return spec, tuple(dropped)


def extract_spec(task_text: str, *, door, data: str, model: str | None = None,
                 provider: str | None = None, measurement: str | None = None) -> Extraction:
    """One call through the door (`task="spec-extract"`), then `build_spec`. Raises the door's
    `LLMError` when no model may take the call, and `ExtractionError` when the answer holds
    nothing usable; never returns a specification the caller did not ask the door for."""
    if not task_text or not task_text.strip():
        raise ExtractionError("the task is empty")
    completion = door.complete(task="spec-extract", data=data, system=SYSTEM,
                               user=user_prompt(task_text), schema=ANSWER_SCHEMA,
                               max_tokens=MAX_TOKENS, model=model, provider=provider,
                               measurement=measurement)
    now = door.clock.now() if getattr(door, "clock", None) is not None else datetime.now(UTC)
    date = now.astimezone(UTC).date().isoformat()
    spec, dropped = build_spec(task_text, completion.data, model=completion.model, date=date)
    return Extraction(spec=spec, dropped=dropped, model=completion.model,
                      provider=completion.provider)


def confirm(spec: Spec, *, by: str, date: str) -> Spec:
    """SPEC-7: a person's review, recorded inside the hashed content. The hash changes, so a
    confirmed specification is locked afresh. The MCP server never calls this."""
    if not by or not by.strip():
        raise ValueError("confirm needs the name of the person who reviewed it")
    if len(date) != 10:
        raise ValueError(f"the review date is YYYY-MM-DD, not {date!r}")
    calendar_date.fromisoformat(date)                   # a calendar date, or ValueError
    prov = spec.provenance or Provenance()
    return spec.model_copy(update={"provenance": prov.model_copy(
        update={"confirmed_by": by.strip(), "confirmed_date": date})})


def binding(spec: Spec) -> bool:
    """Whether the specification's values can reject (SPEC-7): not extracted, or confirmed."""
    return not (spec.provenance is not None and spec.provenance.extracted_unconfirmed)


__all__ = ["ANSWER_SCHEMA", "PROMPT_VERSION", "SYSTEM", "Extraction", "ExtractionError",
           "binding", "build_spec", "confirm", "extract_spec", "extractor_name", "user_prompt"]
