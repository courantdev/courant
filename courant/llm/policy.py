"""The model door's rules, each decided here and nowhere else (ARC-6, ARC-7, ARC-9, ARC-10, ARC-12).

Pure: no network, no clock, no file. The door asks these functions; the CLI, `courant doctor`
and an agent harness pinning its side models (``allowlisted``, C3e) ask the same ones, so no
two places can disagree about what is free, what may leave, or where customer data may go.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from types import MappingProxyType

from courant.llm.errors import Refused

#: The only marker of a free OpenRouter model, matched exactly and case-sensitively (ARC-6).
FREE_SUFFIX = ":free"

#: OpenRouter's routers: they choose the model, so the allowlist would not (ARC-7). Refused even
#: when S allows paid models, as is any `:online` variant, whose web search bills even on a
#: free model.
SIDE_DOORS = frozenset({"openrouter/auto", "openrouter/free"})
ONLINE = ":online"

#: Providers with no free tier. Their ids are paid, so their allowlist is empty by definition.
NO_FREE_TIER = frozenset({"anthropic"})

#: S's switch for paid models. Only the exact value "1" sets it (C2b).
PAID_FLAG = "COURANT_ALLOW_PAID_LLM"

#: What a call may be for, and what its data may be (C5a, C8).
TASKS = ("diagnosis", "judge", "spec-extract", "agent")
DATA = ("public", "customer")
CAPABILITIES = frozenset({"tools", "structured"})

#: The top-level keys an outgoing payload may carry (C3b). It names what may cross, the same
#: design as the trust boundary: `models` (a fallback list), `plugins`, `tools`, `transforms`
#: and `route` are refused because they are not named, not because each was foreseen.
PAYLOAD_KEYS = frozenset({"model", "messages", "max_tokens", "response_format", "provider",
                          "usage"})

#: Attached to every call carrying customer-derived data: zero-data-retention endpoints only,
#: and no training on inputs (ARC-9). Read-only: it is the policy, not a default to adjust.
CUSTOMER_POLICY = MappingProxyType({"zdr": True, "data_collection": "deny"})

_ENTRY_KEYS = ("id", "capabilities", "zdr", "expires", "measured")
_MEASURED = re.compile(r"RM-\d+ \d{4}-\d{2}-\d{2}")
_PROVIDER = re.compile(r"[a-z0-9][a-z0-9-]*")


def paid_allowed() -> bool:
    """Whether S has allowed paid models for this process: exactly ``COURANT_ALLOW_PAID_LLM=1``."""
    return os.environ.get(PAID_FLAG) == "1"


def is_side_door(model_id: str) -> bool:
    """A router, or an `:online` variant (ARC-7)."""
    return model_id.split(":", 1)[0] in SIDE_DOORS or ONLINE in model_id


def check_call(task: str, data: str) -> None:
    """``task`` and ``data`` are each one of a closed set. Anything else is a caller's bug."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, not {task!r}")
    if data not in DATA:
        raise ValueError(f"data must be one of {DATA}, not {data!r}")


# ------------------------------------------------------------------------------- the allowlist


def load_allowlist(allowlist: Mapping) -> dict[str, list[dict]]:
    """A checked copy of an allowlist. Raises ValueError on any entry that breaks a rule (C2a).

    Every listed id ends in exactly ``:free``, is no side door, and appears once in its
    provider's section; a provider with no free tier lists nothing. A section or an entry of
    the wrong shape raises TypeError.
    """
    if not isinstance(allowlist, Mapping):
        raise TypeError("an allowlist maps provider names to lists of entries")
    checked: dict[str, list[dict]] = {}
    for provider, entries in allowlist.items():
        if not isinstance(provider, str) or not _PROVIDER.fullmatch(provider):
            raise ValueError(f"not a provider name: {provider!r}")
        if isinstance(entries, (str, bytes, Mapping)) or not isinstance(entries, Iterable):
            raise TypeError(f"{provider}: the allowlist section must be a list of entries")
        entries = list(entries)
        if provider in NO_FREE_TIER and entries:
            raise ValueError(f"{provider} has no free tier: its ids are paid and are never "
                             f"allowlisted (ARC-6)")
        seen: set[str] = set()
        section = []
        for entry in entries:
            clean = _checked_entry(provider, entry)
            if clean["id"] in seen:
                raise ValueError(f"{provider}: {clean['id']!r} is listed twice")
            seen.add(clean["id"])
            section.append(clean)
        checked[provider] = section
    return checked


def _checked_entry(provider: str, entry) -> dict:
    if not isinstance(entry, Mapping):
        raise TypeError(f"{provider}: an allowlist entry is a mapping, not {entry!r}")
    missing = [key for key in _ENTRY_KEYS if key not in entry]
    if missing:
        raise ValueError(f"{provider}: entry {entry.get('id')!r} lacks {missing}")

    model_id = entry["id"]
    if (not isinstance(model_id, str) or not model_id.endswith(FREE_SUFFIX)
            or model_id == FREE_SUFFIX or any(ch.isspace() for ch in model_id)):
        raise ValueError(f"{provider}: {model_id!r} does not end in exactly {FREE_SUFFIX!r}, "
                         f"so it is not a free model (ARC-6)")
    if is_side_door(model_id):
        raise ValueError(f"{provider}: {model_id!r} is a router or an :online variant (ARC-7)")

    capabilities = entry["capabilities"]
    if isinstance(capabilities, str) or not isinstance(capabilities, Iterable):
        raise TypeError(f"{model_id}: capabilities must be a list")
    capabilities = list(capabilities)
    unknown = [c for c in capabilities if c not in CAPABILITIES]
    if unknown:
        raise ValueError(f"{model_id}: unknown capabilities {unknown}")

    if not (entry["zdr"] is None or isinstance(entry["zdr"], bool)):
        raise ValueError(f"{model_id}: zdr is True, False or None (unknown)")

    expires = entry["expires"]
    if expires is not None and not (isinstance(expires, str) and len(expires) == 10
                                    and _as_date(expires) is not None):
        raise ValueError(f"{model_id}: expires is None or a YYYY-MM-DD date, not {expires!r}")

    measured = entry["measured"]
    if not isinstance(measured, Mapping):
        raise TypeError(f"{model_id}: measured maps a task to its measurement")
    for task, run in measured.items():
        if task not in TASKS or not isinstance(run, str) or not _MEASURED.fullmatch(run):
            raise ValueError(f"{model_id}: measured[{task!r}] = {run!r}; expected a task in "
                             f"{TASKS} and 'RM-nn YYYY-MM-DD' (ARC-12)")

    return {**entry, "capabilities": capabilities, "measured": dict(measured)}


def committed_allowlist() -> Mapping:
    """The repository's allowlist, read when asked, so a change to it is seen at once."""
    from courant.llm import allowlist

    return allowlist.ALLOWLIST


def allowlisted(provider: str, model_id: str, allowlist: Mapping | None = None) -> bool:
    """Whether ``model_id`` is on ``provider``'s allowlist: the one check an agent harness pins
    its side models with (C3e). S's paid flag does not change it: allowed is not listed."""
    if not isinstance(model_id, str) or is_side_door(model_id):
        return False
    lists = load_allowlist(committed_allowlist() if allowlist is None else allowlist)
    return any(entry["id"] == model_id for entry in lists.get(provider, ()))


# --------------------------------------------------------------------------------- eligibility


def expired(entry: Mapping, today: date) -> bool:
    """Past its ``expires`` date: not before ``today``, so the last day is still usable."""
    end = _as_date(entry.get("expires"))
    return end is not None and end < _as_date(today)


def why_not(entry: Mapping | None, *, task: str, data: str, capability: str | None = None,
            measurement: str | None = None, today: date | None = None) -> str | None:
    """Why this entry may not take this call, or None when it may.

    ``entry`` None is a paid model a caller pinned with S's flag: nothing is known about it,
    so its capabilities are the caller's claim and its data-retention class is unknown, which
    ARC-9 treats as training on inputs. The checks run in a fixed order and the first that
    fails is the reason, so the same call always gets the same answer.
    """
    check_call(task, data)
    if entry is None:
        if data == "customer":
            return ("data-retention class unknown, and customer data goes only to "
                    "zero-data-retention endpoints (ARC-9)")
        if not measurement:
            return (f"not measured for {task}, and the call names no pre-registered "
                    f"measurement (ARC-12)")
        return None
    if today is not None and expired(entry, today):
        return f"expired on {entry.get('expires')} (ARC-10)"
    if capability is not None and capability not in entry.get("capabilities", ()):
        return f"no {capability} output"
    if data == "customer" and entry.get("zdr") is not True:
        return "not zero-data-retention, and the data is a customer's (ARC-9)"
    if not measurement and task not in (entry.get("measured") or {}):
        return f"not measured for {task} (ARC-12)"
    return None


def eligible(provider: str, *, task: str, data: str, capability: str | None = None,
             measurement: str | None = None, allowlist: Mapping | None = None,
             today: date | None = None) -> list[dict]:
    """The entries of ``provider``'s allowlist that may take this call, in allowlist order."""
    lists = load_allowlist(committed_allowlist() if allowlist is None else allowlist)
    return [entry for entry in lists.get(provider, ())
            if why_not(entry, task=task, data=data, capability=capability,
                       measurement=measurement, today=today) is None]


# ------------------------------------------------------------------------------- the payload


def check_payload(payload: Mapping, *, model: str, data: str) -> None:
    """Refuse a payload that may not leave (C3b), before any transport sees it.

    Beyond the key allowlist: the model is the one the door chose, and a customer payload
    still carries the zero-data-retention policy. A hook that swapped either would otherwise
    reach a paid model or a training endpoint with nothing else in the way.
    """
    extra = sorted(str(key) for key in payload if key not in PAYLOAD_KEYS)
    if extra:
        raise Refused(f"payload keys {extra} may not leave the door; only "
                      f"{sorted(PAYLOAD_KEYS)} may (ARC-7)")
    if payload.get("model") != model or is_side_door(model):
        raise Refused(f"the payload names a model other than the one chosen ({model}) (ARC-6)")
    if data == "customer" and payload.get("provider") != CUSTOMER_POLICY:
        raise Refused("a customer payload must carry the zero-data-retention policy (ARC-9)")


# ---------------------------------------------------------------------------------- churn


def refresh(entries: Iterable[Mapping], listing, today: date) -> tuple[list[dict], list[tuple]]:
    """Entries the live listing still serves, and the dropped ones with a reason (C6b).

    ``listing`` is the body of GET /api/v1/models (or its ``data`` list). An id absent from it
    is "not listed", the listing's equivalent of a 404; an id whose expiration date (the
    listing's or the entry's) is before ``today`` is "expired".
    """
    models = listing.get("data", []) if isinstance(listing, Mapping) else listing
    by_id = {m.get("id"): m for m in models or () if isinstance(m, Mapping)}
    kept: list[dict] = []
    dropped: list[tuple] = []
    for entry in entries:
        model = by_id.get(entry["id"])
        if model is None:
            dropped.append((dict(entry), "not listed"))
            continue
        ends = [_as_date(model.get("expiration_date")), _as_date(entry.get("expires"))]
        if any(end is not None and end < _as_date(today) for end in ends):
            dropped.append((dict(entry), "expired"))
            continue
        kept.append(dict(entry))
    return kept, dropped


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10]) if len(value) >= 10 else None
        except ValueError:
            return None
    return None
