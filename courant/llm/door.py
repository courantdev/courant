"""The model door (ARC-5): every language-model call in the product goes through ``Door``.

A call is let through only when all of these hold, and otherwise the caller takes its
deterministic path (ARC-11):

  * the model is on the allowlist and so free, or a caller pinned it with S's
    ``COURANT_ALLOW_PAID_LLM=1`` (ARC-6); a router or an ``:online`` variant never (ARC-7);
  * the data may go there: customer-derived data only to a zero-data-retention entry, with
    that policy attached to the request (ARC-9);
  * the model is measured on the task, or the call names the pre-registered measurement it
    belongs to (ARC-12);
  * the provider's quota allows it: under 20 requests a minute, not after a daily-cap 429,
    and a batch only if it can finish today (ARC-8). Whether a request draws on the free
    daily allowance is one rule, ``draws_on_free_allowance``, for the request and for its
    batch: a paid pin never does (ARC-20);
  * the payload carries only named keys (ARC-7), and the answer passes local validation
    against the caller's schema (ARC-11).

Every request sent leaves a ``CallRecord`` in its provider's ledger, success or failure
(ARC-10), and nothing is sent that cannot be recorded (ARC-18): each request is reserved in the
ledger under a file lock before it leaves, counted against the window and the day's allowance
in the same locked step, and refused with ``LedgerError`` when the ledger cannot be written.
The safety latches (a paid answer halts the run, a 404 drops the id, a 401 skips the provider)
are set before an outcome is written, so a failed write cannot disarm them. A call that no
model answered is the model's failure only when a model was asked and failed; when the quota,
a rejected key or a model that is not served stopped it, the stop is raised instead, so a
batch never scores it (ARC-19). A rate limit that outlasts its retries is such a stop
(``RateLimited``), and so is an account out of credit (402), as a rejected key is (ARC-20).
So is any other 4xx that is not a rate limit (a 400 ``invalid_request_error``, a 403): the
provider rejected the request, no model was asked, and the door raises ``Rejected``, never a
failed exchange a batch would score as the model's abstention.
Every rule applies per provider (ARC-17), and no provider is a fallback that lets a paid id
through: automatic selection walks the allowlists only.

Every paid request records what it cost (ARC-21): the provider's reported cost, or, for a
provider whose answers carry none (Anthropic), its usage priced by ``prices``; a paid Anthropic
model with no price is refused before anything is sent. A door built with a ``spend_cap``
reserves each paid request's upper-bound cost in the ledger, under the same lock as the request
itself, and refuses one that would pass the cap (``SpendCapReached``, a stop).

A batch (``Door.batch``) is keyed by what it is (ARC-20): its name, its identity (the
measurement, the model or judge, the prompt's version) and each item's content, so a changed
prompt or item is run again, and a finished batch run again returns its kept answers with the
times they were finished, never as a new result.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from courant.llm import budget, policy, prices
from courant.llm.errors import (
    REJECTION_REASONS,
    BudgetRefused,
    Halted,
    InvalidResponse,
    LedgerError,
    LLMError,
    PaidResponse,
    QuotaExhausted,
    RateLimited,
    Refused,
    Rejected,
    SpendCapReached,
    TransportError,
    provider_name,
)
from courant.llm.transport import RawResponse, Transport

__all__ = [
    "REGISTRY", "BatchResults", "BudgetRefused", "CallRecord", "Candidate", "Completion",
    "Door", "Halted", "InvalidResponse", "LLMError", "LedgerError", "PaidResponse",
    "QuotaExhausted", "RateLimited", "RawResponse", "Refused", "Rejected", "SpendCapReached",
    "SystemClock",
    "Transport", "TransportError", "check_schema", "default_door", "draws_on_free_allowance", "parse_json",
    "provider_name", "request_of", "validate",
]

log = logging.getLogger("courant.llm")

#: The real providers, in registry order, each with the environment variable holding its key.
#: ``courant.config.Settings.providers`` reports configured keys in this order. The order is
#: not a preference between vendors: automatic selection walks the allowlists, and Anthropic's
#: is empty.
REGISTRY = {"openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

#: The host each provider's transport contacts. A command that may call a model names it to
#: the user before the first request (docs/NETWORK.md); this is the one place it is written.
HOSTS = {"openrouter": "openrouter.ai", "anthropic": "api.anthropic.com"}

#: Where `courant doctor --models` reads the public model listing, without a key.
LISTING_HOST = HOSTS["openrouter"]


def contact_notice(provider: str) -> str:
    """One line saying what a model call contacts and with which key, for a command to print
    before it calls (docs/NETWORK.md). The key's value is never part of it."""
    host = HOSTS.get(provider, provider)
    key = REGISTRY.get(provider)
    return (f"contacting {host} ({provider}) with your {key}" if key
            else f"contacting {host} ({provider})")

#: A pin for every call a default door makes, checked like any other pin (C3d).
PIN_ENV = "COURANT_OPENROUTER_MODEL"

#: Statuses worth another attempt at the same model; 5xx are too (C4d).
_RETRYABLE = frozenset({408, 425, 429})

#: Why a candidate that did not answer moved the walk on, by its ledger status, when that is
#: not the model's failure (ARC-19): the day's quota stopped it, the provider rejected the key
#: or the account has no credit (402, ARC-20; either way the door skips the provider from now
#: on), or the id is not served (the door drops it). A rate limit that outlasted its retries
#: is ``"quota"`` too (``_send``). Any other 4xx that is not retried is ``"rejected"`` (the
#: provider rejected the request: ``Rejected``), and a transport that could not send at all
#: (no SDK, no key) is ``"unsent"``; neither is the model's. Only a 5xx or a dropped connection
#: that outlasted its retries is the exchange failing: ``"failed"``, the model's failure on the
#: call.
_WHY = {budget.DAILY_CAP: "quota", "http-401": "key", "http-402": "key", "http-404": "gone"}

#: An error type a provider names in its error body (Anthropic: ``invalid_request_error``), kept
#: only when it has this shape, so no request text can ride along in it (SEC-1).
_ERROR_TYPE = re.compile(r"[a-z][a-z0-9_]{0,63}")

#: What a run has learned, kept for the rest of the process: the paid response that halted it
#: (C2d), and the ids that answered 404 (C6c). Keyed by state directory, so a fresh door in the
#: same run inherits both, and two runs with their own state directories share nothing.
_HALTED: dict[str, str] = {}
_DROPPED: dict[str, set[tuple[str, str]]] = {}


# --------------------------------------------------------------------------- what comes back


@dataclass(frozen=True)
class CallRecord:
    """One request sent to a provider (C6d), exactly as its outcome line in the ledger holds
    it, less the ``rid`` that line shares with the request's reservation (ARC-18 C1).

    No prompt, no answer, no key: a ledger of customer calls is as safe to keep as a ledger of
    public ones (C5d, SEC-1).
    """

    ts: str
    provider: str
    model: str
    task: str
    data: str
    measurement: str | None
    status: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    cost: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Completion:
    """A validated answer: its text, the parsed object, and the record of the request that
    produced it. ``records`` is every request this call sent, failed attempts included."""

    text: str
    data: Any
    provider: str
    model: str
    record: CallRecord
    records: tuple[CallRecord, ...] = ()

    @property
    def usage(self) -> dict:
        return {"input_tokens": self.record.input_tokens or 0,
                "output_tokens": self.record.output_tokens or 0,
                "cost": self.record.cost}


@dataclass(frozen=True)
class Candidate:
    """A model the door may send a call to. ``entry`` is its allowlist entry, or None for a
    paid model a caller pinned with S's flag."""

    provider: str
    id: str
    entry: Mapping | None = field(default=None, compare=False, hash=False, repr=False)

    @property
    def paid(self) -> bool:
        return self.entry is None


def draws_on_free_allowance(candidate: Candidate) -> bool:
    """Whether a request to ``candidate`` counts against its provider's free daily allowance
    (ARC-8, ARC-20): an allowlisted free model's does, a paid model S pinned never does. The
    one rule for the per-request reservation (``Door._reserve``, and the ``paid`` flag its
    ledger line carries) and for a batch's up-front check (``Door.batch``)."""
    return not candidate.paid


def request_of(*, system: str, user: str, schema: Mapping) -> dict:
    """What a call asks the model, as ``Door.complete`` takes it: its system prompt, its user
    prompt and its schema. The one definition a batch item's content is built from (ARC-20),
    so an item is keyed by the request it sends: the judge's rendered task and the
    diagnoser's rendered Evidence Bundle alike."""
    return {"system": system, "user": user, "schema": dict(schema)}


class BatchResults(dict):
    """What ``Door.batch`` returns: ``{id: result}``, and with it ``finished``, ``{id: ledger
    stamp}``, when each returned item was finished, by this run or an earlier one, and
    ``ran``, the ids this run ran (ARC-20). A finished batch run again has ``ran`` empty and
    the original stamps."""

    def __init__(self, results: Mapping, *, finished: Mapping[str, str], ran) -> None:
        super().__init__(results)
        self.finished = {item_id: finished[item_id] for item_id in self if item_id in finished}
        self.ran = tuple(ran)


class SystemClock:
    """The real clock. Tests inject one whose minutes pass in microseconds."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, float(seconds)))


# ------------------------------------------------------------------------------- answers


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """An object out of a model's answer, tolerantly.

    Free models fence their JSON, prefix it with "Here is the diagnosis:", or append a note.
    Tolerant about packaging, never about content: what comes out is validated against the
    caller's schema next, and then by the caller's own control plane.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fenced = _FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except ValueError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    raise InvalidResponse("the model did not return parseable JSON")


#: The JSON Schema keywords the door's validator implements (C7c): those the product's two
#: schemas use. A schema with any other keyword is refused when the call is made, so validation
#: never passes an answer because it ignored a keyword.
SCHEMA_KEYWORDS = frozenset({"type", "properties", "required", "enum", "items",
                             "additionalProperties"})
_JSON_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})


def check_schema(schema, path: str = "$") -> None:
    """Raise ValueError unless every keyword in ``schema`` is one the validator implements
    (TypeError when a schema is not an object at all)."""
    if not isinstance(schema, Mapping):
        raise TypeError(f"{path}: a schema is a JSON object")
    unknown = sorted(set(schema) - SCHEMA_KEYWORDS)
    if unknown:
        raise ValueError(f"{path}: the door's validator does not implement {unknown}; a schema "
                         f"may use only {sorted(SCHEMA_KEYWORDS)} (C7c)")
    if "type" in schema:
        kinds = [schema["type"]] if isinstance(schema["type"], str) else schema["type"]
        if not isinstance(kinds, list) or not kinds or any(k not in _JSON_TYPES for k in kinds):
            raise ValueError(f"{path}: type must name JSON types")
    if "properties" in schema:
        if not isinstance(schema["properties"], Mapping):
            raise ValueError(f"{path}: properties maps names to schemas")
        for name, sub in schema["properties"].items():
            check_schema(sub, f"{path}.{name}")
    if "required" in schema and (not isinstance(schema["required"], list)
                                 or not all(isinstance(n, str) for n in schema["required"])):
        raise ValueError(f"{path}: required is a list of property names")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{path}: enum is a non-empty list")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")
    extra = schema.get("additionalProperties", True)
    if not isinstance(extra, bool):
        check_schema(extra, f"{path}.*")


def validate(instance, schema: Mapping, path: str = "$") -> list[str]:
    """Where ``instance`` breaks ``schema``, as paths. Empty when it conforms.

    The messages name schema paths, never the offending values: an answer about a customer's
    case may echo it, and these messages are logged.
    """
    if "type" in schema:
        kinds = [schema["type"]] if isinstance(schema["type"], str) else schema["type"]
        if not any(_is_type(instance, kind) for kind in kinds):
            return [f"{path}: not {' or '.join(kinds)}"]
    problems = []
    if "enum" in schema and not any(_same(instance, allowed) for allowed in schema["enum"]):
        problems.append(f"{path}: not one of the allowed values")
    if isinstance(instance, dict):
        properties = schema.get("properties") or {}
        problems += [f"{path}: missing {name!r}" for name in schema.get("required") or ()
                     if name not in instance]
        extra = schema.get("additionalProperties", True)
        for name, value in instance.items():
            if name in properties:
                problems += validate(value, properties[name], f"{path}.{name}")
            elif extra is False:
                problems.append(f"{path}: a property the schema does not allow")
            elif isinstance(extra, Mapping):
                problems += validate(value, extra, f"{path}.*")
    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            problems += validate(item, schema["items"], f"{path}[{index}]")
    return problems


def _is_type(value, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False                                    # NaN and Infinity are not JSON numbers
    return kind == "number" or isinstance(value, int) or value.is_integer()


def _same(a, b) -> bool:
    """JSON equality: true is not 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def schema_instruction(schema: Mapping) -> str:
    """The schema, restated in the prompt. Free models accept ``response_format`` and then
    ignore it; a few hundred tokens is the difference between the free tier working and not."""
    return ("Respond with JSON only — no prose, no Markdown fences — matching exactly this "
            f"schema:\n{json.dumps(schema, separators=(',', ':'))}")


def _answer_text(body) -> str:
    if not isinstance(body, Mapping):
        raise InvalidResponse("the answer is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise InvalidResponse("the answer has an unexpected shape (no choices)")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise InvalidResponse("the answer has an unexpected shape (no message)")
    content = message.get("content")
    if isinstance(content, list):                       # content parts
        content = "".join(part.get("text") or "" for part in content
                          if isinstance(part, Mapping) and part.get("type") in (None, "text"))
    text = content.strip() if isinstance(content, str) else ""
    if not text:
        if message.get("refusal"):
            raise InvalidResponse("the model declined to answer")
        raise InvalidResponse("the model returned no text")
    return text


def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 \
        else None


def _cost(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) else None


def _names_daily_cap(body) -> bool:
    """A 429 naming the free daily limit (ARC-8), wherever in the error body it is named."""
    try:
        return "free-models-per-day" in json.dumps(body).lower()
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class _Outcome:
    """What one HTTP answer means, and what the door does next."""

    status: str                         # the ledger's status for this request
    scope: str                          # ok | retry | candidate | provider | call
    error: LLMError | None = None
    text: str = ""
    data: Any = None
    retry_after: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None


class _MoveOn(Exception):
    """A candidate failed in a way the next candidate may not share. ``why`` is ``"quota"``,
    ``"key"``, ``"gone"`` (``_WHY``) or ``"failed"``, the model's failure: what it means for
    the call if no candidate answers (``_no_answer``)."""

    def __init__(self, error: LLMError, *, whole_provider: bool, why: str = "failed") -> None:
        super().__init__(error.message)
        self.error, self.whole_provider, self.why = error, whole_provider, why


# ------------------------------------------------------------------------------------ door


class Door:
    """The one way to a model. See the module docstring for what it lets through.

    ``allowlist`` defaults to the committed one; ``state_dir`` to ``COURANT_LLM_STATE`` or
    ``~/.courant/llm``; ``transports`` maps provider names to transports (a provider with no
    transport is simply not called); ``clock`` provides ``now()`` and ``sleep()``; ``pin`` is a
    ``(provider, model id)`` every unpinned call uses instead of automatic selection.
    ``spend_cap`` (``budget.SpendCap``) holds every paid request this door sends to at most that
    many US dollars under its key, counted in the provider's ledger (ARC-21).
    """

    def __init__(self, *, allowlist: Mapping | None = None, state_dir: Path | str | None = None,
                 transports: Mapping[str, Transport] | None = None, clock=None,
                 pin: tuple[str, str] | None = None, retry: budget.RetryPolicy | None = None,
                 per_minute: int = budget.PER_MINUTE,
                 spend_cap: budget.SpendCap | None = None) -> None:
        self.allowlist = policy.load_allowlist(
            policy.committed_allowlist() if allowlist is None else allowlist)
        self.state_dir = (Path(state_dir).expanduser() if state_dir is not None
                          else budget.default_state_dir())
        self.transports = dict(transports or {})
        self.clock = clock if clock is not None else SystemClock()
        if pin is not None and not (isinstance(pin, tuple) and len(pin) == 2
                                    and all(isinstance(p, str) and p for p in pin)):
            raise ValueError("pin is (provider, model id)")
        if isinstance(per_minute, bool) or not isinstance(per_minute, int) or per_minute < 1:
            raise ValueError("per_minute is a positive integer")
        if spend_cap is not None and not isinstance(spend_cap, budget.SpendCap):
            raise TypeError("spend_cap is a budget.SpendCap")
        self.pin = pin
        self.spend_cap = spend_cap
        self.retry = retry or budget.RetryPolicy()
        self.per_minute = per_minute
        self._run = str(self.state_dir.resolve())       # the key of this run's latches
        self._dropped = _DROPPED.setdefault(self._run, set())   # answered 404 (C6c)
        self._rejected: dict[str, Rejected] = {}        # provider -> why its key is unusable
        #: provider -> (UTC day, free requests allowed that day), as this door's last batch
        #: read it from the provider's key endpoint (ARC-18 C4).
        self._allowance: dict[str, tuple[date, int]] = {}

    # ------------------------------------------------------------------ asking the door

    def complete(self, *, task: str, data: str, system: str, user: str, schema: Mapping,
                 max_tokens: int, model: str | None = None, measurement: str | None = None,
                 provider: str | None = None) -> Completion:
        """One validated answer to one prompt, or an ``LLMError``.

        ``data`` has no default (C5a): "public" for anything generated from public tutorials,
        "customer" for anything derived from a customer's case. ``model`` pins a model, which
        is then the only one tried; without it the door walks the allowlist (C7b).
        ``measurement`` names the pre-registered run the call belongs to (ARC-12).

        When no candidate answers, the error says whose failure that is (ARC-19): a
        ``TransportError`` only when a model was asked and failed and nothing else stopped the
        walk, otherwise the stop (``_no_answer``).
        """
        policy.check_call(task, data)
        check_schema(schema)
        if not isinstance(system, str) or not isinstance(user, str):
            raise TypeError("system and user are the prompt's text")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens is a positive integer")
        self._check_not_halted()
        candidates = self._candidates(task=task, data=data, capability="structured",
                                      measurement=measurement, model=model, provider=provider)

        sent: list[CallRecord] = []
        tried: list[tuple[str, str]] = []           # (why, what happened), in walk order
        exhausted: list[QuotaExhausted] = []
        rejected: list[Rejected] = []
        skipped: set[str] = set()
        for candidate in candidates:
            if candidate.provider in skipped:
                continue
            if (candidate.provider, candidate.id) in self._dropped:   # by a call since
                gone = f"{candidate.id}: answered 404 earlier in this run (ARC-10)"
                tried.append(("gone", gone))
                continue
            blocked = self._blocked(candidate.provider)
            if blocked is not None:
                if isinstance(blocked, QuotaExhausted):
                    exhausted.append(blocked)
                    tried.append(("quota", blocked.message))
                else:                                       # its key was rejected
                    tried.append(("key", blocked.message))
                    if isinstance(blocked, Rejected):
                        rejected.append(blocked)
                skipped.add(candidate.provider)
                continue
            try:
                payload = self.build_payload(model=candidate.id, data=data, system=system,
                                             user=user, schema=schema, max_tokens=max_tokens)
                policy.check_payload(payload, model=candidate.id, data=data)
                return self._send(candidate, payload, schema=schema, task=task, data=data,
                                  measurement=measurement, sent=sent)
            except _MoveOn as move:
                if move.why == "quota":
                    exhausted.append(move.error)
                    tried.append(("quota", move.error.message))
                else:
                    tried.append((move.why, f"{candidate.id}: {move.error.message}"))
                if isinstance(move.error, Rejected):
                    rejected.append(move.error)
                if move.whole_provider:
                    skipped.add(candidate.provider)
                log.warning("%s (%s): %s", candidate.id, candidate.provider, move.error.message)
            except LLMError as exc:
                exc.records = tuple(sent)
                raise
        raise _no_answer(tried, exhausted, sent, rejected)

    def batch(self, name: str, items: Mapping, fn: Callable, *, requests_per_item: int = 1,
              provider: str | None = None, data: str | None = None, model: str | None = None,
              identity: Mapping | None = None,
              content: Callable[[Any], Any] | None = None, task: str | None = None,
              capability: str | None = None,
              measurement: str | None = None, limit: int | None = None) -> BatchResults:
        """Run ``fn(item)`` for every item not already done, if today's quota can finish them.

        A batch is what it says it is (ARC-20). Each item's key (``budget.item_key``) is
        ``identity`` (what the batch measures: the measurement, the model or judge, the
        prompt's version; JSON), the item's id and ``content(item)`` (default: the item
        itself, which must then be JSON, a pydantic model or a dataclass). An item is done
        only when the checkpoint holds a line with its current key, so a changed prompt,
        model or item under the same name is run again. Every key is computed before anything
        is sent; an item that cannot be keyed raises ``TypeError``.

        Before anything is sent, a batch whose requests draw on the free allowance
        (``draws_on_free_allowance``: every batch, unless ``model``, or the door's pin, is a
        paid model S allowed) reads the day's allowance from the provider's key endpoint (the
        lowest documented cap when it does not say) and subtracts today's free requests; a
        batch that needs more than is left is refused whole (C4b). What was read is also the
        allowance this door holds each of its free requests to for the rest of that UTC day
        (ARC-18 C4). A paid batch is held to no free allowance. Each finished item is
        checkpointed, so a batch stopped by ``QuotaExhausted`` (or anything else) resumes
        where it stopped when re-run under the same ``name`` (C4e).

        ``data`` is what the items are (ARC-9). A public batch keeps each item's result, which
        must be JSON (``TypeError`` otherwise, and the item is not marked), on its checkpoint
        line, and returns ``{id: result}`` for every id of ``items`` that is done, whether
        this run or an earlier one finished it, in ``items`` order, each as it was kept
        (ARC-19 C8). A customer batch, or one that does not say, keeps ids, times and keys
        only and returns ``{id: fn(item)}`` for the items run this time; ``fn`` keeps what it
        needs to. Either way the result is a ``BatchResults``: its ``finished`` says when each
        returned item was finished, so a finished batch run again keeps its original times.

        ``task`` (with ``capability`` and ``measurement``) is the call each item makes. When it
        is given, a batch with items to run is refused (``Refused``) unless a model may take
        that call, before the allowance read or any item, so a call the door would refuse
        contacts nothing (docs/NETWORK.md). Every product batch names it
        (tests/unit/test_network.py).

        ``limit`` runs at most that many of the pending items, in ``items`` order, and the
        allowance is checked for those only: a batch larger than any one day's allowance is
        worked through a slice at a time under the same name, by a scheduled runner.
        What is returned is the same: every done item, this run's and earlier ones.
        """
        if data is not None and data not in policy.DATA:
            raise ValueError(f"data must be one of {policy.DATA}, or None, not {data!r}")
        if isinstance(requests_per_item, bool) or not isinstance(requests_per_item, int) \
                or requests_per_item < 1:
            raise ValueError("requests_per_item is a positive integer")
        identity = dict(identity or {})
        describe = content if content is not None else (lambda item: item)
        keys = {str(item_id): budget.item_key(identity, str(item_id), describe(item))
                for item_id, item in items.items()}
        provider = provider or self._batch_provider()
        if provider not in self.transports:
            raise Refused(f"batch {name!r}: no {provider} key is configured")
        self._check_not_halted()
        draws = self._batch_draws(provider, model)
        ledger = budget.Ledger(self.state_dir, provider)
        now = budget.utc(self.clock.now())
        until = _from_ledger(ledger, lambda: ledger.exhausted_until(now))
        if until is not None:
            raise QuotaExhausted(f"{provider}'s free daily quota is spent until "
                                 f"{until:%Y-%m-%d %H:%M} UTC; batch {name!r} resumes then (ARC-8)",
                                 provider=provider, until=until)

        keep = data == "public"
        checkpoint = budget.Checkpoint(self.state_dir, name)
        held = checkpoint.held(keys)
        finished = checkpoint.finished(keys) if keep else {}
        ids = list(dict.fromkeys(str(item_id) for item_id in items))
        unkept = [i for i in ids if held.get(i) is budget.NO_RESULT] if keep else []
        if unkept:
            raise ValueError(f"batch {name!r} keeps its results, but its checkpoint holds "
                             f"{len(unkept)} item(s) done without one, {unkept[0]!r} first: one "
                             f"batch name is one class of data (ARC-19 C8)")
        pending = [(str(item_id), item) for item_id, item in items.items()
                   if str(item_id) not in held]
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("limit is a positive integer, or None")
            pending = pending[:limit]
        if pending and task is not None:
            # ARC-12 before ARC-18: eligibility is decided locally, before the allowance read
            # (GET /api/v1/key) or any item reaches the network.
            self._candidates(task=task, data=data or "customer", capability=capability,
                             measurement=measurement, model=model, provider=provider)
        if pending and draws:
            limit, used = self._day_budget(provider, ledger, now)
            self._allowance[provider] = (now.date(), limit)
            need, left = len(pending) * requests_per_item, max(0, limit - used)
            if need > left:
                raise BudgetRefused(
                    f"batch {name!r} needs {need} requests ({len(pending)} items × "
                    f"{requests_per_item}), but {provider} allows {limit} free requests a day "
                    f"and {used} are used today (UTC), so {left} remain. Nothing was sent "
                    f"(ARC-8)")

        results = {}
        for item_id, item in pending:
            result = fn(item)
            when = self.clock.now()
            if keep:
                held[item_id] = checkpoint.mark(item_id, when, result=result,
                                                key=keys[item_id])
            else:
                checkpoint.mark(item_id, when, key=keys[item_id])
                results[item_id] = result
            finished[item_id] = budget.stamp(when)
        ran = [item_id for item_id, _ in pending]
        if keep:
            return BatchResults({item_id: held[item_id] for item_id in ids if item_id in held},
                                finished=finished, ran=ran)
        return BatchResults(results, finished=finished, ran=ran)

    def eligible(self, *, task: str, data: str, capability: str | None = None,
                 measurement: str | None = None, model: str | None = None,
                 provider: str | None = None) -> list[Candidate]:
        """The models this door would try for such a call, in order. Empty when none may."""
        try:
            self._check_not_halted()
            return self._candidates(task=task, data=data, capability=capability,
                                    measurement=measurement, model=model, provider=provider)
        except Refused:
            return []

    def why_not(self, *, task: str, data: str, capability: str | None = None,
                measurement: str | None = None, model: str | None = None,
                provider: str | None = None) -> str | None:
        """Why no model may take such a call, or None when one may: the reason a caller shows
        when it takes the deterministic path instead."""
        try:
            self._check_not_halted()
            self._candidates(task=task, data=data, capability=capability,
                             measurement=measurement, model=model, provider=provider)
        except Refused as exc:
            return exc.message
        return None

    def build_payload(self, *, model: str, data: str, system: str, user: str, schema: Mapping,
                      max_tokens: int) -> dict:
        """The one OpenAI-compatible payload every provider is sent. Whatever this returns is
        checked by ``policy.check_payload`` before any transport sees it (C3b)."""
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{user}\n\n{schema_instruction(schema)}"},
            ],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "answer", "strict": True,
                                                "schema": schema}},
            "usage": {"include": True},                 # so a cost is reported (ARC-6)
        }
        if data == "customer":
            payload["provider"] = dict(policy.CUSTOMER_POLICY)
        return payload

    # ------------------------------------------------------------------ choosing a model

    def _candidates(self, *, task: str, data: str, capability: str | None,
                    measurement: str | None, model: str | None,
                    provider: str | None) -> list[Candidate]:
        today = budget.utc(self.clock.now()).date()
        if model is None and self.pin is not None and provider in (None, self.pin[0]):
            provider, model = self.pin
        if model is not None:
            candidate = self._resolve(model, provider)
            why = policy.why_not(candidate.entry, task=task, data=data,
                                 capability=None if candidate.paid else capability,
                                 measurement=measurement, today=today)
            if why:
                raise Refused(f"{model} may not take this {task} call: {why}")
            if candidate.paid:
                unpriced = self._unpriced(candidate)
                if unpriced:
                    raise Refused(f"{model} may not take this {task} call: {unpriced}")
            if candidate.provider not in self.transports:
                raise Refused(f"{model}: no {candidate.provider} key is configured")
            if (candidate.provider, candidate.id) in self._dropped:
                raise Refused(f"{model} answered 404 earlier in this run and is dropped (ARC-10)")
            return [candidate]

        policy.check_call(task, data)
        reasons: Counter[str] = Counter()
        found: list[Candidate] = []
        for name, entries in self.allowlist.items():
            if provider is not None and name != provider:
                continue
            for entry in entries:
                if name not in self.transports:
                    why = f"no {name} key is configured"
                elif (name, entry["id"]) in self._dropped:
                    why = "answered 404 earlier in this run (ARC-10)"
                else:
                    why = policy.why_not(entry, task=task, data=data, capability=capability,
                                         measurement=measurement, today=today)
                if why:
                    reasons[why] += 1
                else:
                    found.append(Candidate(name, entry["id"], entry))
        if found:
            return found
        if not reasons:
            raise Refused(f"no model is allowlisted{' on ' + provider if provider else ''} "
                          f"(ARC-6)")
        detail = "; ".join(f"{why} [{n} {'model' if n == 1 else 'models'}]"
                           for why, n in reasons.most_common())
        raise Refused(f"no allowlisted model may take a {task} call on {data} data: {detail}")

    def _resolve(self, model: str, provider: str | None) -> Candidate:
        """A pinned id: its allowlist entry, or a paid model when S allows paid models."""
        if not isinstance(model, str):
            raise TypeError("model is a model id")
        if not model.strip():
            raise ValueError("model is a model id, not blank")
        if policy.is_side_door(model):
            raise Refused(f"{model} is a router or an :online variant, refused even when paid "
                          f"models are allowed (ARC-7)")
        for name in ([provider] if provider is not None else list(self.allowlist)):
            for entry in self.allowlist.get(name, ()):
                if entry["id"] == model:
                    return Candidate(name, model, entry)
        if not policy.paid_allowed():
            where = f"the {provider} allowlist" if provider else "the allowlist"
            raise Refused(f"{model} is not on {where}, and a model that is not is paid: it "
                          f"needs S's {policy.PAID_FLAG}=1 (ARC-6)")
        if provider is None:
            if len(self.transports) != 1:
                raise Refused(f"{model} is not allowlisted; a paid model must name its provider "
                              f"({', '.join(sorted(self.transports)) or 'none configured'})")
            provider = next(iter(self.transports))
        return Candidate(provider, model, None)

    def _unpriced(self, candidate: Candidate) -> str | None:
        """Why a paid candidate's cost could not be recorded or bounded, or None (ARC-21)."""
        price = prices.price_of(candidate.provider, candidate.id)
        if price is None and prices.needs_table(candidate.provider):
            return (f"{candidate.provider} reports no cost and {candidate.id} has no price in "
                    f"courant/llm/prices.py, so what it costs could not be recorded (ARC-21)")
        if price is None and self.spend_cap is not None:
            return (f"{candidate.id} has no price in courant/llm/prices.py, so the spend cap "
                    f"cannot bound its requests before they are sent (ARC-21)")
        return None

    def _batch_draws(self, provider: str, model: str | None) -> bool:
        """Whether a batch's requests draw on ``provider``'s free allowance: the same rule a
        request's reservation applies, to the model the batch pins, or the door's pin
        (ARC-20). An unpinned batch walks the allowlists, which hold free models only."""
        if model is None and self.pin is not None and provider == self.pin[0]:
            model = self.pin[1]
        if model is None:
            return True
        return draws_on_free_allowance(self._resolve(model, provider))

    def _blocked(self, provider: str) -> LLMError | None:
        """Why a provider takes no request right now: its daily quota, or a rejected key.

        A fast path only: the reservation decides again under the ledger lock, and a ledger
        this cannot read is left to it, where it refuses with ``LedgerError`` (ARC-18).
        """
        if provider in self._rejected:
            was = self._rejected[provider]
            return Rejected(was.message, provider=was.provider, status=was.status,
                            error_type=was.error_type, reason=was.reason)
        try:
            until = budget.Ledger(self.state_dir, provider).exhausted_until(self.clock.now())
        except OSError:
            return None
        if until is not None:
            return budget.quota_spent(provider, until)
        return None

    # ------------------------------------------------------------------ sending

    def _send(self, candidate: Candidate, payload: dict, *, schema: Mapping, task: str,
              data: str, measurement: str | None, sent: list[CallRecord]) -> Completion:
        """One model, up to ``retry.attempts`` requests. Every request is reserved in the
        ledger before it leaves and its outcome appended when it answers (ARC-18 C1)."""
        transport = self.transports[candidate.provider]
        ledger = budget.Ledger(self.state_dir, candidate.provider)
        draws = draws_on_free_allowance(candidate)
        fields = {"model": candidate.id, "task": task, "data": data, "measurement": measurement,
                  "paid": not draws}
        bound = None
        if self.spend_cap is not None and candidate.paid:
            price = prices.price_of(candidate.provider, candidate.id)
            if price is None:                   # _candidates refuses it first; never unbounded
                raise Refused(self._unpriced(candidate) or f"{candidate.id} has no price")
            bound = prices.request_bound(price, payload)
        attempt = 0
        while True:
            attempt += 1
            try:
                reservation = self._reserve(ledger, fields, draws=draws, cost_bound=bound)
            except QuotaExhausted as exc:   # provider-scoped, as a daily-cap 429 is (C4)
                raise _MoveOn(exc, whole_provider=True, why="quota") from None
            started = reservation.at
            raw, failure = None, None
            try:
                raw = transport.send(payload)
            except LLMError as exc:
                failure = exc
            except Exception as exc:  # noqa: BLE001 — a dropped connection, whatever raised it
                failure = TransportError(f"no answer from {candidate.provider} "
                                         f"({type(exc).__name__})")
            latency = max(0.0, (budget.utc(self.clock.now()) - started).total_seconds())
            outcome = self._read(candidate, raw, failure, schema, started)
            record = CallRecord(
                ts=reservation.ts, provider=candidate.provider, model=candidate.id,
                task=task, data=data, measurement=measurement, status=outcome.status,
                latency_s=latency, input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens, cost=outcome.cost)
            sent.append(record)
            self._arm(candidate, outcome)               # before the write, which may fail (C6)
            self._write(ledger, record, reservation.rid, outcome)

            if outcome.scope == "ok":
                return Completion(text=outcome.text, data=outcome.data,
                                  provider=candidate.provider, model=candidate.id,
                                  record=record, records=tuple(sent))
            error = outcome.error
            if outcome.scope == "retry":
                wait = self.retry.delay(attempt, outcome.retry_after)
                if wait is not None:
                    log.info("%s (%s): %s; retrying in %.0f s", candidate.id,
                             candidate.provider, error.message, wait)
                    self.clock.sleep(wait)
                    continue
                if outcome.status == "http-429":    # the rate limit outlasted the retries
                    raise _MoveOn(self._rate_limited(candidate, attempt, outcome),
                                  whole_provider=False, why="quota")
                raise _MoveOn(error, whole_provider=False)
            if outcome.scope in ("candidate", "provider"):
                raise _MoveOn(error, whole_provider=outcome.scope == "provider",
                              why=_why(outcome))
            raise error

    def _rate_limited(self, candidate: Candidate, attempt: int,
                      outcome: _Outcome) -> RateLimited:
        """The stop a 429 that outlasted the retries is (ARC-20): resumable after its
        ``Retry-After``, or a rolling window when it gave none."""
        wait = outcome.retry_after if outcome.retry_after is not None \
            else budget.WINDOW.total_seconds()
        until = budget.utc(self.clock.now()) + timedelta(seconds=wait)
        return RateLimited(
            f"{candidate.provider} rate-limited {candidate.id} through {attempt} attempt(s); "
            f"it is not the model's answer, and a batch resumes after "
            f"{until:%Y-%m-%d %H:%M:%S} UTC (ARC-20)", provider=candidate.provider, until=until)

    def _reserve(self, ledger: budget.Ledger, fields: Mapping, *, draws: bool,
                 cost_bound: float | None = None) -> budget.Reservation:
        """The request's reservation (ARC-18 C1-C4). A full window is waited out on the
        injected clock with the ledger lock released, and the reservation asked for again.
        A request that draws on the free allowance (``draws_on_free_allowance``) is held to
        the day's allowance; a paid one S allowed is not (ARC-20). A paid request under the
        door's spend cap reserves its ``cost_bound`` in the same locked step (ARC-21)."""
        def day_limit(day: date) -> int:
            return self._free_allowance(ledger.provider, day)

        while True:
            capped = cost_bound is not None
            got = ledger.reserve(fields, clock=self.clock, per_minute=self.per_minute,
                                 day_limit=day_limit if draws else None,
                                 spend_cap=self.spend_cap if capped else None,
                                 cost_bound=cost_bound)
            if isinstance(got, budget.Reservation):
                return got
            self.clock.sleep(got)

    def _free_allowance(self, provider: str, day: date) -> int:
        """Free requests ``provider`` allows on the UTC ``day``: what this door's last batch
        read from its key endpoint, if that was on ``day``, else the lowest documented cap
        (ARC-18 C4). Never a second read of the key endpoint."""
        read = self._allowance.get(provider)
        if read is not None and read[0] == day:
            return read[1]
        return budget.LOWEST_DOCUMENTED_CAP

    def _arm(self, candidate: Candidate, outcome: _Outcome) -> None:
        """Set the latch an answer calls for: a paid answer halts the run (C2d), a 404 drops
        the id (C6c), a 401 or a 402 skips the provider (ARC-20). Armed before the outcome
        is written, so a failed write cannot disarm it (ARC-18 C6)."""
        if outcome.status == "paid":
            _HALTED[self._run] = outcome.error.message
        elif outcome.status == "http-404":
            self._dropped.add((candidate.provider, candidate.id))
        elif outcome.status in ("http-401", "http-402"):
            self._rejected[candidate.provider] = outcome.error

    def _read(self, candidate: Candidate, raw: RawResponse | None, failure: LLMError | None,
              schema: Mapping, started: datetime) -> _Outcome:
        """What one answer means: its ledger status, and whether to retry, move on or stop."""
        if failure is not None:
            if not isinstance(failure, TransportError):
                failure = TransportError(failure.message)
            return _Outcome("transport-error", "retry" if failure.retryable else "candidate",
                            error=failure)

        status, body = raw.status, raw.body
        if 200 <= status < 300 and isinstance(body, Mapping) and "choices" not in body \
                and isinstance(body.get("error"), Mapping):
            code = body["error"].get("code")            # an error delivered with a 200
            status = code if isinstance(code, int) and 400 <= code <= 599 else 502
        if 200 <= status < 300:
            return self._read_answer(candidate, body, schema)

        retry_after = budget.retry_after_seconds(raw.header("Retry-After"), started)
        said = f"{candidate.provider} answered {status}"
        kind, reason = _error_fields(body)
        if status == 429 and _names_daily_cap(body):
            until = budget.next_utc_midnight(started)
            return _Outcome(budget.DAILY_CAP, "provider", error=QuotaExhausted(
                f"{candidate.provider}'s free daily quota is spent; nothing more is sent to it "
                f"until {until:%Y-%m-%d %H:%M} UTC (ARC-8)",
                provider=candidate.provider, until=until))
        if status == 404:
            return _Outcome("http-404", "candidate", error=TransportError(
                f"{said}: {candidate.id} is not served, and is dropped for this run (ARC-10)",
                status=status, retryable=False))
        if status in (401, 402):        # the key or the account, not the model (ARC-20)
            return _Outcome(f"http-{status}", "provider", error=Rejected(
                provider=candidate.provider, status=status, error_type=kind, reason=reason))
        retryable = status in _RETRYABLE or status >= 500
        if 400 <= status < 500 and not retryable:       # rejected: no model was asked (ARC-19)
            return _Outcome(f"http-{status}", "candidate", error=Rejected(
                provider=candidate.provider, status=status, error_type=kind, reason=reason))
        return _Outcome(f"http-{status}", "retry" if retryable else "candidate",
                        error=TransportError(said, status=status, retryable=retryable),
                        retry_after=retry_after)

    def _read_answer(self, candidate: Candidate, body, schema: Mapping) -> _Outcome:
        usage = body.get("usage") if isinstance(body, Mapping) else None
        usage = usage if isinstance(usage, Mapping) else {}
        spent = {"input_tokens": _count(usage.get("prompt_tokens")),
                 "output_tokens": _count(usage.get("completion_tokens")),
                 "cost": _cost(usage.get("cost"))}
        if candidate.paid:                  # what a paid request cost, from the table (ARC-21)
            price = prices.price_of(candidate.provider, candidate.id)
            if price is not None and (spent["cost"] is None
                                      or prices.needs_table(candidate.provider)):
                spent["cost"] = prices.cost(price, usage)
        if not candidate.paid and spent["cost"] is not None and spent["cost"] > 0:
            return _Outcome("paid", "call", error=PaidResponse(
                f"{candidate.id} is on the free allowlist and reported a cost of "
                f"{spent['cost']}; the run halts (ARC-6)"), **spent)
        try:
            text = _answer_text(body)
            answer = parse_json(text)
            problems = validate(answer, schema)
            if problems:
                raise InvalidResponse("the answer does not match the schema: "
                                      + "; ".join(problems[:5]) + " (ARC-11)")
        except InvalidResponse as exc:
            return _Outcome("invalid", "call", error=exc, **spent)
        return _Outcome("ok", "ok", text=text, data=answer, **spent)

    # ------------------------------------------------------------------ budget and state

    def _day_budget(self, provider: str, ledger: budget.Ledger,
                    now: datetime) -> tuple[int, int]:
        """(today's allowance, requests already used today), for a batch (C4b).

        Used is this machine's ledger count, or what the provider reports when that is more:
        another machine may share the key.
        """
        body = None
        reader = getattr(self.transports.get(provider), "key_info", None)
        if callable(reader):
            try:
                body = reader()
            except LLMError as exc:
                log.warning("could not read %s's daily allowance (%s); assuming the lowest "
                            "documented cap", provider, exc.message)
        counted = _from_ledger(ledger, lambda: ledger.count_on(now.date()))
        used = max(counted, budget.reported_used(body) or 0)
        return budget.allowance(body), used

    def _batch_provider(self) -> str:
        for name in self.allowlist:
            if name in self.transports and self.allowlist[name]:
                return name
        raise Refused("no provider with an allowlisted model is configured")

    def _write(self, ledger: budget.Ledger, record: CallRecord, rid: str,
               outcome: _Outcome) -> None:
        """Append a request's outcome line: its eleven fields and its reservation's ``rid``.

        When it cannot be written, the answer is discarded. The latch this answer armed holds
        either way (C6): a paid answer raises its ``PaidResponse``, anything else
        ``LedgerError(sent=True)``.
        """
        try:
            ledger.append({**record.to_dict(), "rid": rid})
        except OSError as exc:
            log.error("cannot write the %s ledger (%s): the outcome of a request already sent "
                      "is not recorded, and its answer is discarded", ledger.provider,
                      type(exc).__name__)
            if outcome.status == "paid":
                raise outcome.error from None
            raise LedgerError(f"cannot write the {ledger.provider} ledger "
                              f"({type(exc).__name__}): a request was sent and its outcome "
                              f"cannot be recorded, so its answer is discarded (ARC-18)",
                              sent=True) from None

    def _check_not_halted(self) -> None:
        reason = _HALTED.get(self._run)
        if reason is not None:
            raise Halted(f"this run is halted, and the door sends nothing more: {reason}")


def _why(outcome: _Outcome) -> str:
    """What a candidate's unanswered outcome means for the call (``_WHY``)."""
    if outcome.status in _WHY:
        return _WHY[outcome.status]
    if isinstance(outcome.error, Rejected):
        return "rejected"
    if outcome.status == "transport-error":             # not retryable: nothing could be sent
        return "unsent"
    return "failed"


def _error_fields(body) -> tuple[str | None, str | None]:
    """The provider's error type and the transport's recognised reason, from an error body,
    each only when it is one of the fixed shapes a message may carry (SEC-1)."""
    error = body.get("error") if isinstance(body, Mapping) else None
    if not isinstance(error, Mapping):
        return None, None
    kind, reason = error.get("type"), error.get("courant_reason")
    kind = kind if isinstance(kind, str) and _ERROR_TYPE.fullmatch(kind) else None
    reason = reason if isinstance(reason, str) and reason in REJECTION_REASONS else None
    return kind, reason


def _no_answer(tried: list[tuple[str, str]], exhausted: list[QuotaExhausted],
               sent: list[CallRecord], rejected: list[Rejected] = ()) -> LLMError:
    """What a call raises when no candidate answered it, from why each did not (ARC-19).

    Only a model that is served, asked and failing (``"failed"``: a 400, a 5xx after its
    retries, no connection) is the model's failure on the call: a ``TransportError``, which a
    batch scores as that item's abstention. Anything else in the walk makes the call a stop,
    which ``stops_batch`` says a batch stops on and never scores:

      * the day's quota, or a rate limit that outlasted its retries (``RateLimited``,
        ARC-20), stopped a candidate that could have answered: ``QuotaExhausted``, until the
        earliest reset, and naming the rest of the walk when there was any;
      * a provider rejected the key, or the account has no credit (402, ARC-20), so a
        candidate that could have answered was not asked, and the door skips that provider
        for the rest of the run: ``Refused``;
      * every candidate answered 404: no model is served, and the door refuses every later
        call the same way, so this one is refused too: ``Refused``;
      * a provider rejected the request with any other 4xx that is not a rate limit (a 400,
        a 403), so no model was asked: ``Rejected``, naming the status and the provider's
        error type; and a transport that could not send at all (no SDK, no key): ``Refused``.

    ``records`` holds every request the call sent, whichever it raises.
    """
    whys = {why for why, _ in tried}
    said = "; ".join(text for _, text in tried)
    if exhausted:
        first = min(exhausted, key=lambda exc: exc.until or datetime.max.replace(tzinfo=UTC))
        if whys != {"quota"}:
            first = type(first)(f"no allowlisted model answered: {said}",
                                provider=first.provider, until=first.until)
        first.records = tuple(sent)
        return first
    if whys & {"key", "rejected", "unsent"} or "failed" not in whys:
        if rejected:
            first = rejected[0]
            if len(tried) == 1:
                return Rejected(first.message, provider=first.provider, status=first.status,
                                error_type=first.error_type, reason=first.reason,
                                records=sent)
            return Rejected(f"{first.line}. No allowlisted model could answer: {said}",
                            provider=first.provider, status=first.status,
                            error_type=first.error_type, reason=first.reason, records=sent)
        return Refused(f"no allowlisted model could answer: {said}", records=sent)
    return TransportError(f"no allowlisted model answered: {said}", retryable=False,
                          records=sent)


def _from_ledger(ledger: budget.Ledger, read: Callable[[], Any]) -> Any:
    """What ``read()`` counts in ``ledger``. A ledger that cannot be read is a quota that
    cannot be counted: ``LedgerError``, and nothing is sent (ARC-18)."""
    try:
        return read()
    except OSError as exc:
        raise LedgerError(f"cannot read the {ledger.provider} ledger ({type(exc).__name__}), "
                          f"so its quota cannot be counted; nothing was sent (ARC-18)",
                          sent=False) from None


def default_door(*, clock=None, state_dir: Path | str | None = None,
                 spend_cap: budget.SpendCap | None = None,
                 per_minute: int = budget.PER_MINUTE) -> Door:
    """The door this installation uses: the committed allowlist, read now; a transport for
    every provider whose key is configured (``courant.config.real_key``, the one definition of
    configured); ``COURANT_OPENROUTER_MODEL`` as a pin, checked like any other (C3d). A batch
    runner passes its ``spend_cap`` and its rate (``per_minute``) (ARC-21)."""
    from courant.config import load_dotenv, real_key
    from courant.llm import allowlist

    load_dotenv()
    transports = {}
    for name, variable in REGISTRY.items():
        key = real_key(variable)
        if key:
            transports[name] = _transport(name, key)
    pinned = (os.environ.get(PIN_ENV) or "").strip()
    return Door(allowlist=allowlist.ALLOWLIST, state_dir=state_dir, transports=transports,
                clock=clock, pin=("openrouter", pinned) if pinned else None,
                spend_cap=spend_cap, per_minute=per_minute)


def _transport(name: str, key: str) -> Transport:
    if name == "openrouter":
        from courant.llm.openrouter import OpenRouterTransport

        return OpenRouterTransport(api_key=key)
    from courant.config import anthropic_workspace_id
    from courant.llm.anthropic import AnthropicTransport

    return AnthropicTransport(api_key=key, workspace_id=anthropic_workspace_id())
