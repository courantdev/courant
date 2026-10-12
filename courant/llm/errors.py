"""What the model door raises (ARC-5 to ARC-12, ARC-18, ARC-19).

Every refusal and every failure is an ``LLMError``, so a caller has exactly one thing to catch
on its way to the deterministic path (ARC-11). The subclasses say why, for the caller that
needs to know: a batch stops on ``QuotaExhausted`` and resumes after ``until``; a run stops on
``PaidResponse`` and everything after it is ``Halted``; nothing is sent while the ledger cannot
be written (``LedgerError``). ``stops_batch`` is the one place that sorts them into a stop and
a model's failure on one item (ARC-19).

Messages name model ids, providers, HTTP statuses and rules. They never carry a prompt, a
response, or a key (SEC-1), because they are logged and shown.
"""

from __future__ import annotations

from datetime import datetime


class LLMError(Exception):
    """A language-model call did not produce a usable answer.

    ``records`` holds the ``CallRecord`` of every request the failed call sent, so a run can
    account for what a failure spent as well as what a success did (ARC-10).
    """

    def __init__(self, message: str = "", *, records=()) -> None:
        super().__init__(message)
        self.message = message
        self.records = tuple(records)


class Refused(LLMError):
    """The door refused the call. Nothing was sent, except when ``records`` holds the requests
    that showed the door it must refuse: a provider rejecting the key (401) or a model that is
    not served (404), after which it sends nothing more to that provider or that id in the run.
    """


class BudgetRefused(Refused):
    """A batch cannot finish inside the day's allowance (ARC-8). Nothing was sent."""


class Halted(Refused):
    """A free call reported a cost earlier in this run, so the door sends nothing more (ARC-6)."""


class QuotaExhausted(LLMError):
    """A provider's daily free quota is spent. Terminal until ``until``, the next 00:00 UTC."""

    def __init__(self, message: str = "", *, provider: str = "",
                 until: datetime | None = None, records=()) -> None:
        super().__init__(message, records=records)
        self.provider = provider
        self.until = until


class RateLimited(QuotaExhausted):
    """A provider's rate limit (a 429 that does not name the daily cap) outlasted the door's
    retries. It is not the model's answer: a batch stops on it and resumes after ``until``,
    as after the daily cap (ARC-20)."""


class PaidResponse(LLMError):
    """A call to a free model reported a non-zero cost (ARC-6). The run halts."""


class InvalidResponse(LLMError):
    """The answer is not usable: no text, a refusal, no JSON, or it fails the schema (ARC-11)."""


class TransportError(LLMError):
    """The exchange with the provider failed: an HTTP error status or no connection."""

    def __init__(self, message: str = "", *, status: int | None = None,
                 retryable: bool = True, records=()) -> None:
        super().__init__(message, records=records)
        self.status = status
        self.retryable = retryable


class LedgerError(LLMError):
    """The provider's ledger cannot be written, so the door does not send (ARC-18).

    ``sent`` is False when the reservation failed: nothing left. It is True when the request
    had left and its outcome could not be recorded: the answer, if any, is discarded.
    """

    def __init__(self, message: str = "", *, sent: bool = False, records=()) -> None:
        super().__init__(message, records=records)
        self.sent = sent


def stops_batch(exc: BaseException) -> bool:
    """Whether ``exc`` stops a batch of model calls rather than failing one item of it (ARC-19).

    A quota stop (a rate limit that outlasted its retries included, ARC-20), a paid response,
    an unwritable ledger and any refusal by the door (a halt, a batch's budget, a rejected key
    or an account out of credit, and a model that is not served included) are not a model's
    answer: the batch stops, leaves its checkpoint, and nothing is scored. An invalid answer, a
    failed exchange and anything that is not an ``LLMError`` are the model's failure on that
    item. The door raises a failed exchange only when nothing else stopped the call
    (``door._no_answer``). This is the only place the line is drawn; the judge, the diagnoser
    and the batch runners all ask it.
    """
    return isinstance(exc, (QuotaExhausted, PaidResponse, LedgerError, Refused))
