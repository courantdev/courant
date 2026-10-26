"""The model door (TRD §29): every language-model call in the product goes through here (ARC-5).

    door = default_door()
    completion = door.complete(task="diagnosis", data="customer", system=..., user=...,
                               schema=..., max_tokens=...)

A call goes through only to an allowlisted free model (or one S allowed as paid), only to an
endpoint the data may reach, only for a task the model was measured on, and only within the
provider's quota. Anything else raises an ``LLMError``, and the caller takes its deterministic
path (ARC-11). ``courant/verify/`` and ``courant/mcp/`` never import this package: verification
opens no socket (INV-5).

Modules: ``policy`` (the rules, pure), ``allowlist`` (the ids), ``budget`` (the ledger and its
locked reservation, allowance, retries, checkpoints), ``door`` (the door), and one transport
per provider: ``openrouter`` and ``anthropic``. Only modules in this package may import a model client or
name a model endpoint (TST-6); everything else imports from here.
"""

from courant.llm.budget import SpendCap
from courant.llm.door import (
    HOSTS,
    LISTING_HOST,
    REGISTRY,
    BatchResults,
    CallRecord,
    Candidate,
    Completion,
    Door,
    check_schema,
    contact_notice,
    default_door,
    draws_on_free_allowance,
    parse_json,
    request_of,
    validate,
)
from courant.llm.errors import (
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
    stops_batch,
)
from courant.llm.openrouter import public_listing
from courant.llm.policy import allowlisted
from courant.llm.transport import RawResponse, Transport

__all__ = [
    "HOSTS", "LISTING_HOST", "REGISTRY", "BatchResults", "BudgetRefused", "CallRecord", "Candidate", "Completion",
    "Door", "Halted", "InvalidResponse", "LLMError", "LedgerError", "PaidResponse",
    "QuotaExhausted", "RateLimited", "RawResponse", "Refused", "Rejected", "SpendCap", "SpendCapReached", "Transport", "TransportError",
    "allowlisted", "check_schema", "contact_notice", "default_door", "draws_on_free_allowance", "parse_json",
    "public_listing", "request_of", "stops_batch", "validate",
]
