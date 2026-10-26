"""The quota is a budget (ARC-8), kept per provider (ARC-17), and nothing is sent unrecorded
(ARC-18).

One append-only JSONL ledger per provider, ``<state>/ledger-<provider>.jsonl``, is the single
source for every count the door makes: requests in the last minute, requests today (a batch's
budget, and the day's allowance per request), and whether today's free quota is already spent.

A request is two lines. Its reservation (``status: "reserved"``, a fresh ``rid``) is appended
before it is sent, under an exclusive lock on ``<state>/ledger-<provider>.lock`` taken for the
counting and the append together (``Ledger.reserve``), so a second door or a second process on
the same state directory counts the first one's requests, those still in flight included, and
two of them never take the last place in a window or a day. Its outcome (the eleven fields of
a ``CallRecord`` and the same ``rid``) is appended when the answer arrives. Counts count
requests: reservations, and the lines of ledgers written before reservations existed, which
have no ``rid`` and are one request each. Failed requests count, because the provider counts
them. A reservation says whether its request draws on the free daily allowance (``paid``,
decided by ``door.draws_on_free_allowance``, ARC-20): the day's allowance counts the requests
that do, and a line that does not say (an older ledger) is counted as one that does. The
rolling window counts every request, paid or free.

A paid request under a spend cap (ARC-21) also carries, on its reservation, the cap's ``spend``
key and its ``cost_bound``, the most it can cost (``prices.request_bound``); its outcome line
carries its ``cost``. What a key has spent is counted from the whole ledger in the same locked
step: each of its reservations at its outcome's recorded cost, at nothing when the provider
answered with an HTTP error (not billed), and at its bound while its answer has not arrived or
when no answer did (a dropped connection may still be billed). A request whose bound would take
the key past its cap is refused with ``SpendCapReached`` and nothing is appended, so two doors or
processes sharing the state directory can never together send past the cap.

The ledger holds no prompt, no answer and no key (C5d, SEC-1). The state directory is
``COURANT_LLM_STATE`` or ``~/.courant/llm/``, never a case directory (TST-4), and it and the
lock file are created on the first reservation, so constructing a door writes nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from courant.llm.errors import LedgerError, QuotaExhausted, SpendCapReached

#: OpenRouter allows 20 requests a minute on free models; the door stays one under (C4a).
PER_MINUTE = 19
WINDOW = timedelta(seconds=60)

#: The lowest daily free-request cap OpenRouter documents (50 until $10 of credits has ever
#: been bought, 1,000 after). Used when the key endpoint does not state the allowance (C4b),
#: and per request by a door whose batch has not read it today (ARC-18).
LOWEST_DOCUMENTED_CAP = 50

#: The ledger status of a 429 naming ``free-models-per-day``: terminal until 00:00 UTC (C4c).
DAILY_CAP = "daily-cap"

#: The ledger status of a request's reservation, written before it is sent (ARC-18).
RESERVED = "reserved"

#: How far past the oldest timestamp a backwards read keeps going before it stops. A
#: reservation is stamped when it is appended, under the lock, but clocks of two processes may
#: disagree a little, and ledgers written before reservations existed appended each line when
#: its answer arrived. An outcome line repeats its reservation's stamp however late it is
#: appended, so it never stops a read.
_SCAN_SLACK = timedelta(hours=1)
_BLOCK = 64 * 1024
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_state_dir() -> Path:
    """``COURANT_LLM_STATE``, else ``~/.courant/llm``. Not created here."""
    configured = os.environ.get("COURANT_LLM_STATE")
    return Path(configured).expanduser() if configured else Path.home() / ".courant" / "llm"


def utc(when: datetime) -> datetime:
    """An aware UTC datetime. A naive one is taken to be UTC already."""
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)


def start_of_utc_day(when: datetime) -> datetime:
    return datetime.combine(utc(when).date(), time(0), tzinfo=UTC)


def next_utc_midnight(when: datetime) -> datetime:
    """The 00:00 UTC that ends ``when``'s UTC day: when a daily cap hit at ``when`` resets."""
    return start_of_utc_day(when) + timedelta(days=1)


def stamp(when: datetime) -> str:
    """A ledger timestamp: ISO 8601, UTC, microseconds, "Z"."""
    return utc(when).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_stamp(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return utc(datetime.fromisoformat(value))
    except ValueError:
        return None


# ---------------------------------------------------------------------------------- ledger


@dataclass(frozen=True)
class SpendCap:
    """At most ``usd`` US dollars on paid requests reserved under ``key`` (ARC-21)."""

    key: str
    usd: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("a spend cap names its key (the batch it holds)")
        if isinstance(self.usd, bool) or not isinstance(self.usd, (int, float)) \
                or not math.isfinite(self.usd) or self.usd <= 0:
            raise ValueError("a spend cap is a positive number of US dollars")


@dataclass(frozen=True)
class Reservation:
    """A request's place in its provider's ledger, appended before it is sent (ARC-18 C1)."""

    rid: str                    # the request's id; its outcome line repeats it
    ts: str                     # the ledger stamp; its outcome line repeats it
    at: datetime                # the same instant, when the reservation was appended


class Ledger:
    """One provider's append-only record of every request it was sent."""

    def __init__(self, state_dir: Path | str, provider: str) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", provider or ""):
            raise ValueError(f"not a provider name: {provider!r}")
        self.provider = provider
        self.path = Path(state_dir) / f"ledger-{provider}.jsonl"
        #: Held exclusively for each reservation's counting and append (ARC-18 C2).
        self.lock_path = Path(state_dir) / f"ledger-{provider}.lock"

    def append(self, record: Mapping) -> None:
        """One line, written with a single append so concurrent writers do not interleave."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(record), separators=(",", ":"), ensure_ascii=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)

    def since(self, start: datetime) -> list[tuple[datetime, dict]]:
        """``(timestamp, record)`` for every line stamped at or after ``start``, newest first.

        Read backwards from the end, so the cost is the recent lines, not the file's history.
        """
        start = utc(start)
        found: list[tuple[datetime, dict]] = []
        for raw in _lines_backwards(self.path):
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            when = parse_stamp(record.get("ts")) if isinstance(record, dict) else None
            if when is None:
                continue
            if when >= start:
                found.append((when, record))
            elif when < start - _SCAN_SLACK and _is_request(record):
                break
        return found

    def sent_in_window(self, now: datetime, window: timedelta = WINDOW) -> list[datetime]:
        """Requests in the rolling window ending now, ``now - window < ts <= now``, ascending:
        each counted once, by its reservation, whether or not its answer has arrived."""
        now = utc(now)
        return _in_window(self.since(now - window), now, window)

    def count_on(self, day: date) -> int:
        """Requests on a UTC calendar day that drew on the free allowance, failed ones
        included (C4b, ARC-20). A paid request S allowed is not one of them."""
        return _count_on(self.since(datetime.combine(day, time(0), tzinfo=UTC)), day)

    def exhausted_until(self, now: datetime) -> datetime | None:
        """When today's spent free quota resets, if a daily-cap 429 was recorded today (C4c)."""
        now = utc(now)
        return _cap_lifts(self.since(start_of_utc_day(now)), now)

    def spent(self, key: str) -> float:
        """US dollars spent under spend key ``key``, as a reservation counts it (module
        docstring): recorded costs, in-flight bounds included."""
        return _spent(self._all(), key)

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for raw in self.path.read_bytes().splitlines():
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if isinstance(record, dict):
                out.append(record)
        return out

    def reserve(self, fields: Mapping, *, clock, per_minute: int = PER_MINUTE,
                window: timedelta = WINDOW,
                day_limit: int | Callable[[date], int | None] | None = None,
                spend_cap: SpendCap | None = None, cost_bound: float | None = None,
                ) -> Reservation | float:
        """Append one request's reservation, or say how long to wait for room (ARC-18 C1-C4).

        ``fields`` names the request: ``model``, ``task``, ``data``, ``measurement`` and
        ``paid`` (whether it does not draw on the free allowance, ARC-20); the line takes
        those, this ledger's provider, the time, ``status: "reserved"`` and a fresh ``rid``,
        and nothing else. Everything below happens while this process holds an
        exclusive lock on ``lock_path``, so every door and process sharing the state directory
        counts the others' requests, those still in flight included, and no two of them take
        the last place in a window or a day:

          * a daily-cap 429 recorded today refuses with ``QuotaExhausted`` (C4c);
          * so does today's (UTC) count of free requests at or above ``day_limit``: an int,
            or a callable giving it for the UTC date. None, a paid model's, is held to no
            allowance;
          * with ``spend_cap``, a request whose ``cost_bound`` would take what the cap's key
            has spent past the cap refuses with ``SpendCapReached`` (ARC-21);
          * a full rolling window returns the seconds until a place frees, and appends
            nothing. The caller sleeps with the lock released and asks again: nothing here
            sleeps;
          * otherwise the reservation is appended and returned.

        Any ``OSError``, and a platform without ``fcntl``, raise ``LedgerError``: the request
        cannot be recorded, so it is not sent (C5).
        """
        if isinstance(per_minute, bool) or not isinstance(per_minute, int) or per_minute < 1:
            raise ValueError("per_minute is a positive integer")
        line = {"provider": self.provider, "model": fields["model"], "task": fields["task"],
                "data": fields["data"], "measurement": fields.get("measurement"),
                "paid": bool(fields.get("paid", False))}
        if fields.get("note"):           # what a reservation that is not a request stands for
            line["note"] = str(fields["note"])
        if spend_cap is not None:
            if isinstance(cost_bound, bool) or not isinstance(cost_bound, (int, float)) \
                    or not math.isfinite(cost_bound) or cost_bound < 0:
                raise ValueError("a request under a spend cap states its cost bound")
            line.update(spend=spend_cap.key, cost_bound=float(cost_bound))
        try:  # lazily, so courant.llm imports on a platform without fcntl
            import fcntl
        except ImportError:
            raise LedgerError(f"cannot lock the {self.provider} ledger on this platform (no "
                              f"fcntl), so its quota cannot be counted; nothing was sent "
                              f"(ARC-18)", sent=False) from None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_path, "a", encoding="utf-8") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)     # released when the file closes
                return self._reserve_locked(line, clock, per_minute, window, day_limit,
                                            spend_cap)
        except OSError as exc:
            raise LedgerError(f"cannot write the {self.provider} ledger ({type(exc).__name__}), "
                              f"so its quota cannot be counted; nothing was sent (ARC-18)",
                              sent=False) from None

    def _reserve_locked(self, line: dict, clock, per_minute: int, window: timedelta,
                        day_limit, spend_cap: SpendCap | None = None) -> Reservation | float:
        """``reserve``'s one locked step. The clock is read here, under the lock, so a
        reservation is stamped after every one appended before it and is counted by every
        one appended after it."""
        now = utc(clock.now())
        seen = self.since(min(start_of_utc_day(now), now - window))

        until = _cap_lifts(seen, now)
        if until is not None:
            raise quota_spent(self.provider, until)
        limit = day_limit(now.date()) if callable(day_limit) else day_limit
        if limit is not None:
            used = _count_on(seen, now.date())
            if used >= limit:
                until = next_utc_midnight(now)
                raise QuotaExhausted(
                    f"{self.provider} allows {limit} free requests a day and {used} were sent "
                    f"today (UTC) by every door on this state directory; nothing more is sent "
                    f"to it until {until:%Y-%m-%d %H:%M} UTC (ARC-8, ARC-18)",
                    provider=self.provider, until=until)

        if spend_cap is not None:
            spent = self.spent(spend_cap.key)
            if spent + line["cost_bound"] > spend_cap.usd + 1e-12:
                raise SpendCapReached(
                    f"{spend_cap.key}: ${spent:.6f} of its ${spend_cap.usd:.2f} cap is spent or "
                    f"reserved on {self.provider}, and the next request may cost up to "
                    f"${line['cost_bound']:.6f}; it is not sent, and the batch resumes only "
                    f"under a higher cap (ARC-21)", key=spend_cap.key, cap=spend_cap.usd,
                    spent=spent, bound=line["cost_bound"])

        recent = _in_window(seen, now, window)
        if len(recent) >= per_minute:
            # The window must lose its oldest len - (per_minute - 1) requests before one fits.
            release = recent[len(recent) - per_minute] + window
            return (release - now).total_seconds()

        reservation = Reservation(rid=uuid.uuid4().hex, ts=stamp(now), at=now)
        self.append({"ts": reservation.ts, **line, "status": RESERVED, "rid": reservation.rid})
        return reservation


def _is_request(record: Mapping) -> bool:
    """Whether a ledger line is one request: its reservation, or a line written before
    reservations existed (no ``rid``). An outcome line repeats its reservation's ``rid`` and
    is never counted, so a request counts once whether or not its answer has arrived."""
    return record.get("status") == RESERVED or not record.get("rid")


def _in_window(seen, now: datetime, window: timedelta) -> list[datetime]:
    return sorted(when for when, record in seen
                  if now - window < when <= now and _is_request(record))


def _draws_on_allowance(record: Mapping) -> bool:
    """Whether a ledger line is a request counted against the free daily allowance: a request
    whose reservation does not say it is paid (ARC-20). An older line that does not say is
    counted, so the count errs towards refusing."""
    return _is_request(record) and record.get("paid") is not True


def _count_on(seen, day: date) -> int:
    return sum(1 for when, record in seen
               if when.date() == day and _draws_on_allowance(record))


def _spent(records, key: str) -> float:
    """What spend key ``key`` has spent over ``records`` (a whole ledger), in US dollars."""
    bounds: dict[str, float] = {}
    for record in records:
        if record.get("status") == RESERVED and record.get("spend") == key \
                and isinstance(record.get("rid"), str):
            bound = record.get("cost_bound")
            ok = isinstance(bound, (int, float)) and not isinstance(bound, bool) \
                and math.isfinite(bound)
            bounds[record["rid"]] = float(bound) if ok else math.inf
    outcomes = {record["rid"]: record for record in records
                if record.get("status") != RESERVED and record.get("rid") in bounds}
    total = 0.0
    for rid, bound in bounds.items():
        outcome = outcomes.get(rid)
        cost = outcome.get("cost") if outcome is not None else None
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost):
            total += max(0.0, float(cost))
        elif outcome is not None and str(outcome.get("status", "")).startswith("http-"):
            continue                        # the provider answered with an error: not billed
        else:
            total += bound                  # in flight, or no answer arrived
    return total


def _cap_lifts(seen, now: datetime) -> datetime | None:
    """The earliest reset still ahead of ``now`` of a daily-cap 429 among ``seen``."""
    ends = [next_utc_midnight(when) for when, record in seen
            if record.get("status") == DAILY_CAP]
    ends = [end for end in ends if end > now]
    return min(ends) if ends else None


def quota_spent(provider: str, until: datetime) -> QuotaExhausted:
    """The refusal for a provider whose daily-cap 429 was recorded today (C4c)."""
    return QuotaExhausted(f"{provider}'s free daily quota is spent until "
                          f"{until:%Y-%m-%d %H:%M} UTC (ARC-8)", provider=provider, until=until)


def _lines_backwards(path: Path) -> Iterator[bytes]:
    if not path.exists():
        return
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        partial = b""
        while position > 0:
            step = min(_BLOCK, position)
            position -= step
            handle.seek(position)
            lines = (handle.read(step) + partial).split(b"\n")
            partial = lines.pop(0)
            for line in reversed(lines):
                if line.strip():
                    yield line
        if partial.strip():
            yield partial


# --------------------------------------------------------------------------------- allowance


def allowance(key_body) -> int:
    """The day's free-request allowance from a GET /api/v1/key body (C4b).

    The documented shape is ``free_model_daily_requests: {limit, remaining, used}``; a bare
    number is read as the limit too. Absent or unreadable, it is the lowest documented cap.
    """
    field = _key_data(key_body).get("free_model_daily_requests")
    if isinstance(field, Mapping):
        field = field.get("limit")
    if isinstance(field, int) and not isinstance(field, bool) and field >= 0:
        return field
    return LOWEST_DOCUMENTED_CAP


def reported_used(key_body) -> int | None:
    """Free requests the provider says this key has used today, when it says so."""
    field = _key_data(key_body).get("free_model_daily_requests")
    used = field.get("used") if isinstance(field, Mapping) else None
    if isinstance(used, int) and not isinstance(used, bool) and used >= 0:
        return used
    return None


def _key_data(key_body) -> Mapping:
    if not isinstance(key_body, Mapping):
        return {}
    data = key_body.get("data", key_body)
    return data if isinstance(data, Mapping) else {}


# ----------------------------------------------------------------------------------- retries


@dataclass(frozen=True)
class RetryPolicy:
    """At most ``attempts`` sends of one request, waiting 1 s then 2 s (doubling), or the
    provider's ``Retry-After`` when that is longer (C4d). A wait past ``max_wait`` is not
    taken: the request moves on rather than block a run for longer."""

    attempts: int = 3
    first_wait: float = 1.0
    max_wait: float = 60.0

    def delay(self, attempt: int, retry_after: float | None = None) -> float | None:
        """Seconds to wait after failed attempt ``attempt`` (1-based), or None to stop."""
        if attempt >= self.attempts:
            return None
        wait = max(self.first_wait * 2 ** (attempt - 1), retry_after or 0.0)
        return wait if wait <= self.max_wait else None


def retry_after_seconds(value: str | None, now: datetime) -> float | None:
    """A ``Retry-After`` header in seconds: delta-seconds or an HTTP date."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            seconds = (utc(parsedate_to_datetime(text)) - utc(now)).total_seconds()
        except (TypeError, ValueError, IndexError):
            return None
    return max(0.0, seconds) if not math.isnan(seconds) else None


# -------------------------------------------------------------------------------- checkpoint


class _NoResult:
    def __repr__(self) -> str:
        return "NO_RESULT"


#: What ``Checkpoint.held`` gives an item whose line keeps no result, and ``mark``'s default.
NO_RESULT: Any = _NoResult()


def item_key(identity: Mapping, item_id: str, content: Any) -> str:
    """The key of one item of a batch: SHA-256 over the batch's identity (the measurement,
    the model or judge, the prompt's version), the item's id and its content, as canonical
    JSON (ARC-20). Two runs share a finished item only when all of these are the same.

    ``content`` is JSON, or a pydantic model or dataclass made of JSON. Anything else raises
    ``TypeError``: an item whose content cannot be stated cannot be told apart from another.
    """
    record = {"identity": _as_json(identity), "id": str(item_id), "content": _as_json(content)}
    try:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"the content of item {str(item_id)!r} is not JSON ({exc}), so the "
                        f"batch cannot key it (ARC-20)") from None
    return hashlib.sha256(line.encode("ascii")).hexdigest()


def _as_json(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):                                  # a pydantic model
        return dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


class Checkpoint:
    """Which items of a named batch are done, as JSONL under the state directory (C4e).

    Each line holds an item's id, when it finished, and its key (``item_key``: the batch's
    identity and the item's content, ARC-20). A batch over public data also keeps each item's
    result on its line (ARC-19 C8), so a resumed batch returns what an earlier run finished; a
    batch over customer data keeps ids, times and keys only, no prompt and no answer. The ids
    are the caller's, written verbatim, so a batch names its items by something that is not
    customer data (C5d). Asked with ``keys``, an item is done only by a line carrying its
    current key: a line written for other content, another prompt or another model, or before
    keys existed, is not that item finished.
    """

    def __init__(self, state_dir: Path | str, name: str) -> None:
        safe = _SAFE.sub("_", str(name)).strip("._")
        if not safe:
            raise ValueError(f"not a usable batch name: {name!r}")
        self.name = name
        self.path = Path(state_dir) / "batches" / f"{safe}.jsonl"

    def _lines(self, keys: Mapping[str, str] | None) -> dict[str, dict]:
        """``{id: line}`` for every item done, the last matching line of each, in the order
        they finished."""
        if not self.path.exists():
            return {}
        found: dict[str, dict] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                continue
            if keys is not None and (record["id"] not in keys
                                     or record.get("key") != keys[record["id"]]):
                continue
            found.pop(record["id"], None)
            found[record["id"]] = record
        return found

    def held(self, keys: Mapping[str, str] | None = None) -> dict[str, Any]:
        """``{id: result}`` for every item done, in the order they finished. An item whose
        line keeps no result maps to ``NO_RESULT``. With ``keys`` (``{id: item_key}``), only
        lines carrying the item's current key count."""
        return {item_id: record.get("result", NO_RESULT)
                for item_id, record in self._lines(keys).items()}

    def finished(self, keys: Mapping[str, str] | None = None) -> dict[str, str]:
        """``{id: ledger stamp}``: when each item done was finished, as ``held`` reads them."""
        return {item_id: record.get("ts") for item_id, record in self._lines(keys).items()}

    def done(self, keys: Mapping[str, str] | None = None) -> set[str]:
        return set(self._lines(keys))

    def mark(self, item_id: str, when: datetime, result: Any = NO_RESULT,
             key: str | None = None) -> Any:
        """Record an item as done, with its key and its result when they are given, and
        return the result as kept: what a later ``held()`` reads back.

        The line is serialised before anything is written. A result that is not JSON raises
        ``TypeError``, and the item is not marked.
        """
        record: dict[str, Any] = {"id": str(item_id), "ts": stamp(when)}
        if key is not None:
            record["key"] = key
        if result is not NO_RESULT:
            record["result"] = result
        try:
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"batch {self.name!r}: the result of {str(item_id)!r} is not JSON "
                            f"({exc}), so it is not kept and the item is not marked done") from None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return json.loads(line)["result"] if result is not NO_RESULT else NO_RESULT
